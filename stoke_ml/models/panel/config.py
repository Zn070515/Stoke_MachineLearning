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

    # Input dimensions
    static_dim: int = 30
    past_known_dim: int = 250
    past_observed_dim: int = 120

    # Core model
    hidden_dim: int = 128
    dropout: float = 0.25       # backbone dropout
    head_dropout: float = 0.35  # output-head dropout (higher → anti-collapse)

    # xLSTM backbone
    xlstm_num_blocks: int = 3
    xlstm_slstm_ratio: float = 0.67  # 2 sLSTM : 1 mLSTM
    xlstm_num_heads: int = 2
    grn_layers: int = 2  # decoder GRN stack after xLSTM

    # Training
    batch_size: int = 64
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

    # Sequence
    seq_len: int = 60

    # Output
    num_direction_classes: int = 3  # down / flat / up
    horizon: int = 1

    # Hardware
    use_amp: bool = False
    compile_model: bool = True
    num_workers: int = 8

    # Diagnostics (expensive — enable for debugging gradient collapse)
    log_gradient_flow: bool = False
