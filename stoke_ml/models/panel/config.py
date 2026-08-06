from dataclasses import dataclass


@dataclass
class PanelConfig:
    """VSN + xLSTM model hyperparameters.

    Architecture: VSN (variable selection) → xLSTM backbone → Static Enrichment
                 → Multi-Head Outputs (direction / return / volatility)

    Designed for RTX 4090 24GB training on 488 A-share stocks with daily data.

    Key differences from the old TFT config:
    - No temporal attention → no gradient collapse risk
    - xLSTM backbone instead of LSTM + MHA + GRN stack
    - Full Static Encoder with 4 context vectors
    - Per-layer gradient clipping values
    - Cosine LR schedule with warmup (transformer-training standard)
    """

    # Input dimensions (overridden at runtime from actual data).  Defaults track
    # the current prebuilt panel: S=9 PIT static (2 continuous + 6 board one-hot;
    # price_60d_q removed §五 P0 → fresh builds emit S=8 until the panel is
    # rebuilt), PK=255, PO=1418 — the docs guard (check_docs_consistency.py)
    # enforces README / CONTEXT against the frozen defaults, so keep the three
    # in sync when the prebuilt panel is regenerated.
    static_dim: int = 9
    past_known_dim: int = 255
    past_observed_dim: int = 1418

    # Core model
    hidden_dim: int = 128
    dropout: float = 0.25       # backbone dropout
    head_dropout: float = 0.35  # output-head dropout (higher → anti-collapse)

    # xLSTM backbone
    xlstm_num_blocks: int = 2
    xlstm_slstm_ratio: float = 0.67  # 2 sLSTM : 1 mLSTM
    xlstm_num_heads: int = 2
    grn_layers: int = 2  # decoder GRN stack after xLSTM

    # Training
    batch_size: int = 128
    grad_accum_steps: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3       # 3e-4 → 1e-3 (stronger L2 for financial noise)
    early_stop_patience: int = 8
    max_epochs: int = 200

    # Gradient clipping (per-layer: backbone loose, heads loose for anti-collapse)
    backbone_grad_clip: float = 1.0
    head_grad_clip: float = 5.0

    # LR scheduler (CosineAnnealing with LinearWarmup)
    lr_warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Reproducibility
    seed: int | None = 42
    # Strict bitwise CUDA determinism: torch.use_deterministic_algorithms(True).
    # Opt-in because a single non-deterministic op (e.g. a CUDA atomic
    # reduction) then RAISES mid-training instead of silently diverging.
    # Default False = the safe knobs (cudnn deterministic, no TF32) + a
    # declared "statistical-only" reproducibility guarantee.
    deterministic_algorithms: bool = False

    # Sequence
    seq_len: int = 60

    # Sample eligibility: a window is trainable only if its
    # input has >= min_history real observations (new listings with mostly
    # zero-padded history are excluded) AND the target day is entry-eligible
    # AND at least one target mask is set.
    min_history: int = 50

    # Date-centric sampling (§七/§十六): max stocks per date in one batch.
    # When a date has more valid stocks, a random subset is sampled each epoch.
    # None = no cap (use all valid stocks — val default).  The DataLoader
    # batch_size is always 1 in date-centric mode (one date == one batch);
    # gradient-accumulation steps control how many dates to average over.
    max_stocks_per_date: int | None = 512

    # Minimum number of eligible stocks a cross-section needs before its
    # per-day RankIC is kept.  Shared by checkpoint
    # selection (train._compute_val_loss) and the formal clean-IC evaluator
    # (evaluate._compute_daily_ic) so the two are the SAME quantity — the old
    # training-side threshold of 2 let statistically weak days select a
    # checkpoint while the report required >= 10.
    min_stocks_per_day: int = 20

    # Output
    num_direction_classes: int = 3  # down / flat / up
    horizon: int = 5

    # Hardware
    use_amp: bool = True
    compile_model: bool = True
    num_workers: int = 0  # 0 = main-process loading (avoids Windows shared-memory error 1455)

    # Ranking loss weight (0 = disabled, 0.1–0.5 recommended)
    rank_loss_weight: float = 0.1

    # ── Architecture ablation (§十一.3) ─────────────────────────────────
    # Answers "does performance come from xLSTM, VSN, multi-task, ranking,
    # or more history input?" by switching each component off in isolation.
    # train_panel.py --ablation NAME maps to these overrides.  All default to
    # the production architecture so the formal baseline is unchanged.
    backbone: str = "xlstm"          # "xlstm" | "lstm" (plain nn.LSTM, zero-init)
    use_vsn: bool = True             # False → per-group linear projection instead
    use_dir_head: bool = True        # False → model emits zero direction logits
    use_vol_head: bool = True        # False → model emits zero volatility
    use_ranking_loss: bool = True    # False → train without the ranking term
    fixed_task_weights: bool = False # True → equal fixed weights (no learned log-vars)
    use_pit_static: bool = True      # False → model ignores PIT static features

    # One-way transaction cost (fraction of notional) applied per fill in the
    # sleeve-account evaluation.  0.0005 = 5 bps/side.
    txn_cost: float = 0.0005

    # Diagnostics (expensive — enable for debugging gradient collapse)
    log_gradient_flow: bool = False
