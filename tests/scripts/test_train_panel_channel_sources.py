"""§T2 registry-derivation pin for the ``train_panel_panel`` consumer.

Lives in the scripts slice (ml CI job, torch installed): importing the script
pulls ``stoke_ml.models.panel`` (→ torch) at module load, but ``tests/data``
runs in the light storage-parquet CI slice without torch.  Moved here so the
consumer derivation still executes on CI rather than erroring/skipping.
"""
import importlib.util
from pathlib import Path

from stoke_ml.data.channel_sources import CHANNEL_SOURCE, live_data_type

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "production"


def _load_script(name: str):
    path = _SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_train_panel_panel_live_data_type():
    # The MarketWideStorage loop reads the LIVE dir, not the *_processed one —
    # live_data_type must resolve capital_flow → "capital_flow".
    train_panel_panel_mod = _load_script("train_panel_panel")
    spec = CHANNEL_SOURCE["capital_flow"]
    assert live_data_type(spec) == "capital_flow"
    assert train_panel_panel_mod._MARKET_WIDE_CHANNELS == (
        "margin", "northbound", "dragon_tiger", "capital_flow",
        "block_trade", "shareholder", "lockup", "dividend", "valuation",
    )
    for ch in train_panel_panel_mod._MARKET_WIDE_CHANNELS:
        assert live_data_type(CHANNEL_SOURCE[ch]) in \
            {d for d in __import__(
                "stoke_ml.data.market_wide_storage",
                fromlist=["MARKET_DATA_TYPES"]).MARKET_DATA_TYPES}
