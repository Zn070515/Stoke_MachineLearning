"""requirements.txt must never drift from pyproject.toml (§十八-3)."""
import tomllib
from pathlib import Path

import pytest

from scripts.maintenance.current.gen_requirements import REQUIREMENTS, render

ROOT = Path(__file__).resolve().parents[2]


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_render_flattens_core_and_extras():
    text = render({"project": {
        "dependencies": ["pandas>=2.0"],
        "optional-dependencies": {
            "ml": ["torch==2.11.0", "scikit-learn>=1.3"],
        },
    }})
    assert "pandas>=2.0" in text
    assert "# [ml]" in text
    assert "torch==2.11.0" in text
    assert "scikit-learn>=1.3" in text
    # torch keeps its CUDA index-url guidance, not silently dropped.
    assert "--index-url https://download.pytorch.org/whl/cu128" in text


def test_requirements_txt_in_sync_with_pyproject():
    """Regenerating from pyproject must reproduce the checked-in file."""
    data = _pyproject()
    on_disk = REQUIREMENTS.read_text(encoding="utf-8")
    assert render(data) == on_disk, (
        "requirements.txt is out of sync with pyproject.toml — re-run "
        "scripts/maintenance/current/gen_requirements.py"
    )


def test_pyproject_has_build_system():
    data = _pyproject()
    assert data["build-system"]["build-backend"] == "setuptools.build_meta"
    assert data["build-system"]["requires"] == ["setuptools>=68", "wheel"]
