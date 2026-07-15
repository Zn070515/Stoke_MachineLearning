from dataclasses import dataclass


@dataclass
class TFTConfig:
    """TFT model hyperparameters. ~20M params with defaults below."""

    # Input dimensions
    static_dim: int = 30
    past_known_dim: int = 250
    past_observed_dim: int = 120

    # Core
    hidden_dim: int = 256
    lstm_layers: int = 2
    attention_heads: int = 4
    grn_layers: int = 3
    dropout: float = 0.15

    # Training
    batch_size: int = 512
    grad_accum_steps: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    max_epochs: int = 100

    # Sequence
    seq_len: int = 252

    # Output
    num_direction_classes: int = 2

    # Hardware
    use_amp: bool = True
    compile_model: bool = True
    num_workers: int = 8
