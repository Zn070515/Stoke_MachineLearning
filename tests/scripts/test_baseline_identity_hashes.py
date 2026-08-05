"""Tests for the §十七 baseline identity hashes in train_baselines_panel.py.

Four identity hashes bind a baseline fold's tape / ledger / registry signature
to the exact configuration that produced it:

- ``_baseline_input_recipe_hash``     — --with-seq-features + seq_len + construction version
- ``_baseline_hyperparameter_hash``   — the exact hyperparameters make_model builds
- ``_training_sample_policy_hash``    — --max-train-rows + sampling strategy
- ``_scaler_hash``                    — the feature-scaling recipe (v14 §十七)

The v14 requirement adds ``_scaler_hash`` as a DISTINCT identity and asserts
that flipping --with-seq-features changes the input-recipe hash.

The module is loaded the same way test_train_panel_universe.py loads
train_panel.py (spec_from_file_location).  train_baselines_panel.py does its
own ``sys.path.insert(0, ...)`` + ``from train_panel import (...)`` at the top
of the module, which works under that loader.  No side effects: main() is
guarded by ``if __name__ == "__main__"`` and lightgbm is imported lazily inside
_LGBMWrapper.fit, so module import needs no network / data / training.
"""

import importlib.util
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "scripts", "production", "train_baselines_panel.py")

# The [:16] hexdigest convention every §十七 baseline identity hash shares.
HEX16 = re.compile(r"^[0-9a-f]{16}$")


@pytest.fixture(scope="module")
def blp():
    spec = importlib.util.spec_from_file_location("train_baselines_panel_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── input recipe ───────────────────────────────────────────────────────

def test_input_recipe_hash_changes_with_seq_features_flag(blp):
    # v14 §十七 explicit case: with vs without --with-seq-features is a NEW
    # input identity (a different feature vector the baseline is fit on).
    assert (
        blp._baseline_input_recipe_hash(True, 60)
        != blp._baseline_input_recipe_hash(False, 60)
    )


def test_input_recipe_hash_changes_with_seq_len(blp):
    # Boundary sanity: the sequence-window length is part of the recipe.
    assert (
        blp._baseline_input_recipe_hash(True, 60)
        != blp._baseline_input_recipe_hash(True, 120)
    )


def test_input_recipe_hash_is_stable_16_hex(blp):
    h = blp._baseline_input_recipe_hash(False, 60)
    assert HEX16.match(h)
    assert h == blp._baseline_input_recipe_hash(False, 60)


# ── scaler recipe (v14 §十七) ──────────────────────────────────────────

def test_scaler_hash_is_deterministic_16_hex(blp):
    h = blp._scaler_hash()
    assert HEX16.match(h)
    assert h == blp._scaler_hash()


def test_scaler_hash_is_distinct_from_other_identities(blp):
    scaler = blp._scaler_hash()
    assert scaler != blp._baseline_input_recipe_hash(False, 60)
    assert scaler != blp._baseline_hyperparameter_hash("ridge")
    assert scaler != blp._training_sample_policy_hash(10_000)


def test_scaler_recipe_version_tag_pins_standard_scaler_fit_train(blp):
    # The version tag is the honest fingerprint of the fit-basis recipe (the
    # scaler itself is not a versioned artifact); it must not drift silently.
    assert blp._SCALER_RECIPE_VERSION == "standard-scaler-fit-train+v1"


# ── training-sample policy ─────────────────────────────────────────────

def test_sample_policy_hash_binds_max_train_rows(blp):
    assert (
        blp._training_sample_policy_hash(5_000)
        != blp._training_sample_policy_hash(10_000)
    )


# ── hyperparameters ────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "a,b", [("ridge", "mlp"), ("ridge", "lgbm"), ("mlp", "lgbm")]
)
def test_hyperparameter_hash_differs_across_models(blp, a, b):
    assert blp._baseline_hyperparameter_hash(a) != blp._baseline_hyperparameter_hash(b)


def test_hyperparameter_hash_stable_for_same_model(blp):
    assert (
        blp._baseline_hyperparameter_hash("ridge")
        == blp._baseline_hyperparameter_hash("ridge")
    )
