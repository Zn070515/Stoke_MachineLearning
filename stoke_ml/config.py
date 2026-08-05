"""Global configuration loader using Hydra/OmegaConf."""
import os
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

_PROJECT_ROOT = Path(__file__).parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config.yaml"


def _resolve_path(value: object) -> str | None:
    """Anchor a config path value to the project root (§十一).

    Absolute paths, ``$VAR`` env-expanded paths, and ``~`` home-expanded paths
    pass through unchanged; a relative path is anchored to the project root so
    consumers never depend on the process CWD.
    """
    if value is None:
        return None
    s = os.path.expandvars(os.path.expanduser(str(value)))
    if os.path.isabs(s):
        return s
    return str(_PROJECT_ROOT / s)


def _resolve_config_paths(cfg: DictConfig) -> None:
    """Resolve every path-typed config key to an absolute path (§十一).

    Only keys known to hold filesystem paths are rewritten — project
    data_dir/model_dir plus the preprocessing registry path and the topic-model
    cache dir.  Non-path keys are untouched.  ``output_dir`` / ``monitor.log_dir``
    are dead keys in the checked-in config.yaml (no consumers anywhere in the
    repo); if a consumer ever re-appears they are still resolved here, but the
    keys were removed from config.yaml to avoid implying they are wired.
    """
    proj = cfg.get("project")
    if proj is not None:
        proj.data_dir = _resolve_path(proj.get("data_dir", "./data"))
        proj.model_dir = _resolve_path(proj.get("model_dir", "./models/checkpoints"))
    pp = cfg.get("preprocessing")
    if pp is not None:
        if "registry_path" in pp:
            pp.registry_path = _resolve_path(pp.registry_path)
        if "output_dir" in pp:
            pp.output_dir = _resolve_path(pp.output_dir)
        text_cfg = pp.get("text", {})
        if hasattr(text_cfg, "get"):
            tm = text_cfg.get("topic_model", {})
            if "model_cache_dir" in tm:
                tm.model_cache_dir = _resolve_path(tm.model_cache_dir)
        mon = pp.get("monitor", {})
        if hasattr(mon, "get") and "log_dir" in mon:
            mon.log_dir = _resolve_path(mon.log_dir)


def load_config(config_path: Path | None = None) -> DictConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. Defaults to project config.yaml.

    Returns:
        OmegaConf DictConfig object with all settings.  Path-typed settings are
        anchored to the project root as absolute paths, so callers are
        independent of the process CWD (§十一).
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    cfg = OmegaConf.load(path)
    _resolve_config_paths(cfg)
    return cfg


def get_project_root() -> Path:
    """Return the project root directory."""
    return _PROJECT_ROOT
