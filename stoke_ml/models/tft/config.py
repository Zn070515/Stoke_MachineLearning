from dataclasses import dataclass


@dataclass
class TFTConfig:
    """TFT model hyperparameters. ~20M params with defaults below.

    Training defaults:
    - ReduceLROnPlateau on val_loss (not Sharpe — too few samples)
    - Early stopping on val_loss plateau with IC as secondary metric
    - Moderate regularization for financial data (dropout=0.20, wd=3e-4)
    """

    # Input dimensions
    static_dim: int = 30
    past_known_dim: int = 250
    past_observed_dim: int = 120

    # Core
    hidden_dim: int = 128
    lstm_layers: int = 2
    attention_heads: int = 4
    grn_layers: int = 2
    dropout: float = 0.20

    # Training
    batch_size: int = 64
    grad_accum_steps: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 3e-4
    early_stop_patience: int = 10
    max_grad_norm: float = 0.1
    max_epochs: int = 200

    # LR scheduler (ReduceLROnPlateau — same as Qlib + pytorch-forecasting)
    lr_reduce_factor: float = 0.5
    lr_reduce_patience: int = 10
    lr_reduce_threshold: float = 1e-4
    min_lr: float = 1e-5

    # Reproducibility
    seed: int | None = 42

    # Sequence (research: 60 steps optimal for daily stock data)
    seq_len: int = 60

    # Output
    num_direction_classes: int = 3  # down / flat / up
    horizon: int = 1  # forward-return horizon (trading days)

    # Hardware
    use_amp: bool = False
    compile_model: bool = True
    num_workers: int = 8
