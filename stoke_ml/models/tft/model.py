import torch
import torch.nn as nn
import torch.nn.functional as F
from stoke_ml.models.tft.config import TFTConfig
from stoke_ml.models.tft.components import GRN
from stoke_ml.models.tft.vsn import VariableSelectionNetwork
from stoke_ml.models.tft.attention import InterpretableMultiHeadAttention
from stoke_ml.models.tft.heads import DirectionHead, ReturnHead, VolatilityHead


class TFTModel(nn.Module):
    """Temporal Fusion Transformer for panel stock prediction.

    Input: static features (B, S), past_known (B, T, P), past_observed (B, T, O)
    Output: direction logits (B, 3), return % (B, 1), volatility (B, 1)
    """

    def __init__(self, config: TFTConfig):
        super().__init__()
        self.config = config
        h = config.hidden_dim

        # Variable Selection Networks (x3) — skip when dim=0
        self.vsn_past = VariableSelectionNetwork(
            input_dim=1, hidden_dim=h,
            num_features=config.past_known_dim, dropout=config.dropout,
        ) if config.past_known_dim > 0 else None

        self.vsn_obs = VariableSelectionNetwork(
            input_dim=1, hidden_dim=h,
            num_features=config.past_observed_dim, dropout=config.dropout,
        ) if config.past_observed_dim > 0 else None

        # Static features are a flat vector (no per-feature selection needed).
        # Simple Linear + GRN is equivalent and cheaper than VSN(num_features=1).
        if config.static_dim > 0:
            self.static_proj = nn.Linear(config.static_dim, h)
            self.static_grn = GRN(
                input_dim=h, hidden_dim=h, output_dim=h,
                dropout=config.dropout,
            )
        else:
            self.static_proj = None
            self.static_grn = None

        # LSTM input projection: concat past+observed VSN outputs, then project
        if config.past_known_dim > 0 and config.past_observed_dim > 0:
            self.lstm_input_proj = nn.Linear(2 * h, h)
        elif config.past_known_dim > 0 or config.past_observed_dim > 0:
            self.lstm_input_proj = nn.Linear(h, h)
        else:
            self.lstm_input_proj = None

        # LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=h,
            hidden_size=h,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0,
        )

        # Static enrichment GRN
        if config.static_dim > 0:
            self.static_enrich_grn = GRN(
                input_dim=h, hidden_dim=h, output_dim=h,
                dropout=config.dropout, context_dim=h,
            )
        else:
            self.static_enrich_grn = None

        # Multi-Head Attention
        self.attention = InterpretableMultiHeadAttention(
            d_model=h, n_heads=config.attention_heads, dropout=config.dropout,
        )

        # Post-attention GRN
        self.post_attn_grn = GRN(
            input_dim=h, hidden_dim=h, output_dim=h, dropout=config.dropout,
        )

        # Decoder GRN stack
        self.decoder_grns = nn.ModuleList([
            GRN(input_dim=h, hidden_dim=h, output_dim=h, dropout=config.dropout)
            for _ in range(config.grn_layers)
        ])

        # Output heads
        self.direction_head = DirectionHead(h, config.num_direction_classes, config.dropout)
        self.return_head = ReturnHead(h, config.dropout)
        self.volatility_head = VolatilityHead(h, config.dropout)

    def forward(
        self,
        static_features: torch.Tensor,
        past_known: torch.Tensor,
        past_observed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = past_known.shape

        # VSN: treat each scalar feature as a separate variable
        if self.vsn_past is not None:
            pk_vars = past_known.unsqueeze(-1)  # (B, T, P, 1)
            past_selected, _ = self.vsn_past(pk_vars)  # (B, T, h)
        else:
            past_selected = torch.zeros(B, T, self.config.hidden_dim, device=past_known.device, dtype=torch.float32)

        if self.vsn_obs is not None:
            po_vars = past_observed.unsqueeze(-1)  # (B, T, O, 1)
            obs_selected, _ = self.vsn_obs(po_vars)  # (B, T, h)
        else:
            obs_selected = torch.zeros(B, T, self.config.hidden_dim, device=past_observed.device, dtype=torch.float32)

        # Combine past + observed into LSTM input (concat then project)
        if self.vsn_past is not None and self.vsn_obs is not None:
            lstm_input = self.lstm_input_proj(
                torch.cat([past_selected, obs_selected], dim=-1)
            )
        elif self.vsn_past is not None:
            lstm_input = self.lstm_input_proj(past_selected)
        elif self.vsn_obs is not None:
            lstm_input = self.lstm_input_proj(obs_selected)
        else:
            lstm_input = torch.zeros(B, T, self.config.hidden_dim, device=past_known.device, dtype=torch.float32)

        # LSTM Encoder
        lstm_out, _ = self.lstm(lstm_input)  # (B, T, h)

        # Static enrichment
        if self.static_enrich_grn is not None and static_features is not None:
            static_enc = self.static_grn(F.elu(self.static_proj(static_features)))  # (B, h)
            static_tiled = static_enc.unsqueeze(1).expand(-1, T, -1)  # (B, T, h)
            lstm_out = self.static_enrich_grn(lstm_out, context=static_tiled)

        # Multi-Head Attention (self-attention across time)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Post-attention GRN
        attn_out = self.post_attn_grn(attn_out)

        # Decoder GRN stack
        decoder_out = attn_out
        for grn in self.decoder_grns:
            decoder_out = grn(decoder_out)

        # Output heads (all use last timestep)
        direction = self.direction_head(decoder_out)
        return_pct = self.return_head(decoder_out)
        volatility = self.volatility_head(decoder_out)

        return direction, return_pct, volatility
