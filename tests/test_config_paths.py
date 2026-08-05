"""§十一: path-typed config keys resolve to absolute project-root paths.

`load_config()` must be CWD-independent — the same absolute path whether the
process runs from the project root or any other directory.  Covers project
data_dir/model_dir plus the preprocessing registry path and the topic-model
cache dir, and asserts the dead `output_dir` / `monitor.log_dir` keys were
removed from the checked-in config.yaml.
"""

import os

import pytest

from stoke_ml.config import _resolve_path, load_config

PATH_KEYS = [
    ("project", "data_dir"),
    ("project", "model_dir"),
    ("preprocessing", "registry_path"),
    ("preprocessing", "text", "topic_model", "model_cache_dir"),
]


def _cfg_get(cfg, keys):
    node = cfg
    for k in keys:
        node = node[k]
    return node


def _assert_absolute_paths(cfg):
    for keys in PATH_KEYS:
        value = _cfg_get(cfg, keys)
        assert value is not None and str(value), f"{keys} resolved to {value!r}"
        assert os.path.isabs(value), f"{keys} not absolute: {value!r}"


def test_path_keys_resolve_absolute_from_project_root():
    cfg = load_config()
    _assert_absolute_paths(cfg)


def test_path_keys_resolve_to_same_absolute_from_other_cwd(tmp_path, monkeypatch):
    """CWD must not change the resolved absolute paths (§十一)."""
    baseline = load_config()
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    _assert_absolute_paths(cfg)
    for keys in PATH_KEYS:
        assert _cfg_get(cfg, keys) == _cfg_get(baseline, keys)


def test_resolve_path_passthrough_absolute_env_home(monkeypatch):
    monkeypatch.setenv("STOKE_TEST_HOME", "/env/stoke")
    assert _resolve_path("/abs/dir") == "/abs/dir"
    assert _resolve_path("$STOKE_TEST_HOME") == "/env/stoke"
    assert _resolve_path(None) is None


def test_resolve_path_anchors_relative_to_project_root():
    from stoke_ml.config import get_project_root

    out = _resolve_path("./data")
    assert out == str(get_project_root() / "data")
    assert os.path.isabs(out)


def test_dead_config_keys_removed_from_config_yaml():
    """output_dir / monitor.log_dir had no consumers anywhere — removed so
    config.yaml does not imply they are wired (§十一)."""
    cfg = load_config()
    assert "output_dir" not in cfg.preprocessing
    assert "log_dir" not in cfg.preprocessing.monitor
