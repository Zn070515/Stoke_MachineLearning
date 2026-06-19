"""Global configuration loader using Hydra/OmegaConf."""
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

_PROJECT_ROOT = Path(__file__).parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config.yaml"


def load_config(config_path: Path | None = None) -> DictConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. Defaults to project config.yaml.

    Returns:
        OmegaConf DictConfig object with all settings.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    cfg = OmegaConf.load(path)
    cfg.project.data_dir = str(_PROJECT_ROOT / cfg.project.data_dir)
    cfg.project.model_dir = str(_PROJECT_ROOT / cfg.project.model_dir)
    return cfg


def get_project_root() -> Path:
    """Return the project root directory."""
    return _PROJECT_ROOT
