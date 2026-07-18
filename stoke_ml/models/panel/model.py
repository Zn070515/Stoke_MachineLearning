import torch
import torch.nn as nn
import torch.nn.functional as F
from stoke_ml.models.panel.config import PanelConfig
from stoke_ml.models.panel.components import GRN, GateAddNorm
from stoke_ml.models.panel.vsn import VariableSelectionNetwork
from stoke_ml.models.panel.xlstm import xLSTMBackbone
from stoke_ml.models.panel.heads import DirectionHead, ReturnHead, VolatilityHead


class PanelModel(nn.Module):
    """VSN + xLSTM panel stock prediction model.

    Architecture:
      1. VSN selects informative features per stock (scalar path, memory-efficient)
      2. Static Encoder produces 4 context vectors from stock metadata
      3. xLSTM backbone (sLSTM + mLSTM) extracts temporal features
         — no temporal attention → no gradient collapse
      4. Static enrichment modulates temporal features per stock
      5. Multi-head output: direction (3-class), return %, volatility

    Input: static (B, S), past_known (B, T, PK), past_observed (B, T, PO)
    Output: direction logits (B, 3), return % (B, 1), volatility (B, 1)
    """

    def __init__(self, config: PanelConfig):
        super().__init__()
        self.config = config
        h = config.hidden_dim

        # ── Variable Selection Networks (scalar path, memory-efficient) ──
        self.vsn_past = VariableSelectionNetwork(
            input_dim=1, hidden_dim=h,
            num_features=config.past_known_dim, dropout=config.dropout,
            context_dim=h if config.static_dim > 0 else None,
        ) if config.past_known_dim > 0 else None

        self.vsn_obs = VariableSelectionNetwork(
            input_dim=1, hidden_dim=h,
            num_features=config.past_observed_dim, dropout=config.dropout,
            context_dim=h if config.static_dim > 0 else None,
        ) if config.past_observed_dim > 0 else None

        # ── Static Encoder: 4 context vectors from stock metadata ──
        #  ζ = StaticProj(static) →
        #    c_e  = GRN(ζ)  → static enrichment (post-xLSTM)
        #    c_h  = GRN(ζ)  → xLSTM initial state (sLSTM hidden)
        #    c_vs = GRN(ζ)  → VSN context (per-stock feature selection)
        if config.static_dim > 0:
            self.static_proj = nn.Linear(config.static_dim, h)
            self.static_enrich_context = GRN(
                input_dim=h, hidden_dim=h, output_dim=h, dropout=config.dropout,
            )
            self.static_hidden_context = GRN(
                input_dim=h, hidden_dim=h, output_dim=h, dropout=config.dropout,
            )
            self.static_vs_context = GRN(
                input_dim=h, hidden_dim=h, output_dim=h, dropout=config.dropout,
            )
            # Project c_h to sLSTM initial state shape (B, num_heads, head_dim)
            self.c_h_proj = nn.Linear(h, h)
        else:
            self.static_proj = None
            self.static_enrich_context = None
            self.static_hidden_context = None
            self.static_vs_context = None
            self.c_h_proj = None

        # ── xLSTM backbone (replaces LSTM + Attention + post_attn_GRN) ──
        self.xlstm = xLSTMBackbone(
            hidden_dim=h,
            num_blocks=config.xlstm_num_blocks,
            slstm_ratio=config.xlstm_slstm_ratio,
            num_heads=config.xlstm_num_heads,
            dropout=config.dropout,
        )

        # VSN output → xLSTM input projection
        n_vsn = (1 if config.past_known_dim > 0 else 0) + (1 if config.past_observed_dim > 0 else 0)
        if n_vsn == 2:
            self.feat_proj = nn.Linear(2 * h, h)
        elif n_vsn == 1:
            self.feat_proj = nn.Linear(h, h)
        else:
            self.feat_proj = None

        # ── Post-xLSTM processing ──
        # GateAddNorm: GLU gating + residual + LayerNorm (TFT canonical pattern)
        self.post_xlstm_gate = GateAddNorm(h, dropout=config.dropout)

        # Static enrichment: modulate temporal features with c_e per stock
        if config.static_dim > 0:
            self.static_enrich = GRN(
                input_dim=h, hidden_dim=h, output_dim=h,
                dropout=config.dropout, context_dim=h,
            )
        else:
            self.static_enrich = None

        # Decoder GRN stack (preserves feature transformation depth)
        self.decoder_grns = nn.ModuleList([
            GRN(input_dim=h, hidden_dim=h, output_dim=h, dropout=config.dropout)
            for _ in range(config.grn_layers)
        ])

        # ── Output heads (bottleneck + ELU + head-specific dropout) ──
        self.direction_head = DirectionHead(
            h, config.num_direction_classes, config.head_dropout,
        )
        self.return_head = ReturnHead(h, config.head_dropout)
        self.volatility_head = VolatilityHead(h, config.head_dropout)

    def forward(
        self,
        static_features: torch.Tensor,
        past_known: torch.Tensor,
        past_observed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = past_known.shape
        device = past_known.device
        h = self.config.hidden_dim

        # ── 1. Static Encoder (before VSN — c_vs feeds into feature selection) ──
        if self.static_proj is not None and static_features is not None:
            zeta = F.elu(self.static_proj(static_features))               # (B, h)
            c_e = self.static_enrich_context(zeta)                         # (B, h)
            c_h = self.static_hidden_context(zeta)                         # (B, h)
            c_vs = self.static_vs_context(zeta)                            # (B, h)
        else:
            c_e = None
            c_h = None
            c_vs = None

        # ── 2. Variable Selection (per-stock context from c_vs) ──
        c_vs_ctx = (
            c_vs.unsqueeze(1).expand(-1, T, -1).reshape(B * T, h)
            if c_vs is not None else None
        )
        vsn_outputs = []
        if self.vsn_past is not None:
            pk_vars = past_known.unsqueeze(-1)  # (B, T, P, 1)
            pk_feat, _ = self.vsn_past(pk_vars, context=c_vs_ctx)  # (B, T, h)
            vsn_outputs.append(pk_feat)
        if self.vsn_obs is not None:
            po_vars = past_observed.unsqueeze(-1)  # (B, T, O, 1)
            po_feat, _ = self.vsn_obs(po_vars, context=c_vs_ctx)  # (B, T, h)
            vsn_outputs.append(po_feat)

        if len(vsn_outputs) == 2:
            feat = self.feat_proj(torch.cat(vsn_outputs, dim=-1))
        elif len(vsn_outputs) == 1:
            feat = self.feat_proj(vsn_outputs[0])
        else:
            feat = torch.zeros(B, T, h, device=device)

        # ── 3. xLSTM backbone ──
        # Build initial states from c_h for each sLSTM block
        if c_h is not None:
            h_init = self.c_h_proj(c_h).reshape(
                B, self.config.xlstm_num_heads, -1,
            )
            zero_state = torch.zeros_like(h_init)
            num_slstm = int(
                self.config.xlstm_num_blocks * self.config.xlstm_slstm_ratio,
            )
            states = [
                (h_init, zero_state.clone(), zero_state.clone(), zero_state.clone())
                for _ in range(num_slstm)
            ]
        else:
            states = None
        xlstm_out, _ = self.xlstm(feat, states=states)  # (B, T, h)

        # GateAddNorm: GLU(xlstm_out) + skip(feat) + LayerNorm
        xlstm_out = self.post_xlstm_gate(xlstm_out, feat)

        # ── 4. Static Enrichment ──
        if self.static_enrich is not None and c_e is not None:
            c_e_tiled = c_e.unsqueeze(1).expand(-1, T, -1)  # (B, T, h)
            xlstm_out = self.static_enrich(xlstm_out, context=c_e_tiled)

        # ── 5. Decoder GRN stack ──
        decoder_out = xlstm_out
        for grn in self.decoder_grns:
            decoder_out = grn(decoder_out)

        # ── 6. Output heads ──
        direction = self.direction_head(decoder_out)
        return_pct = self.return_head(decoder_out)
        volatility = self.volatility_head(decoder_out)

        return direction, return_pct, volatility
