"""Panel baselines benchmark.

The VSN+xLSTM panel model is complex; without simple reference points there is
no way to tell whether the complexity earns its keep.  This script trains
Linear Ridge / LightGBM / MLP / naive-momentum on the SAME prebuilt panel
features and the SAME non-overlapping walk-forward folds as train_panel.py, then
scores each with the SAME ``evaluate_portfolio`` sleeve-account evaluator.

- Fold boundaries are imported from train_panel.py (same module), so the two
  runs are aligned by construction, not by copy.
- Feature contract is point-in-time: a sample entering at day ``e`` uses the
  feature cross-section at ``e-1`` (the decision column the xLSTM also sees).
- The naive "past return mean" baseline is the floor to beat: if the
  complex model cannot beat it, the complexity adds no value.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/train_baselines_panel.py \
      --prebuilt data/features_panel --stocks 200 --max-folds 2 \
      --models ridge,lgbm,mlp,momentum --outdir reports/experiments/baselines
"""
import argparse
import hashlib
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Import fold helpers from train_panel.py so boundaries stay identical.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stoke_ml.config import get_project_root, load_config
from stoke_ml.features.pipeline import FeaturePipeline
from stoke_ml.models.baseline.panel_baselines import (
    FittedScoreAdapter,
    PrecomputedScoreAdapter,
    ScaledPredictor,
    build_flat_samples,
    build_momentum_grid,
    fit_mlp_with_early_stopping,
)
from stoke_ml.models.panel import PanelConfig
from stoke_ml.models.panel.evaluate import EVALUATOR_VERSION, evaluate_portfolio
from train_panel import (  # noqa: E402  (same-module fold logic)
    _append_experiment_registry,
    _apply_candidate_gates,
    _check_verified_until_scope,
    _cross_sectional_normalize,
    _discover_stocks,
    _distinct_trial_count,
    _EXPERIMENT_REGISTRY_PATH,
    _experiment_signature,
    _experiment_version,
    _fmt_date,
    _fold_delist_day,
    _fold_eligible_stocks,
    _fold_universe_gates,
    _gate_enforced,
    _load_experiment_registry,
    _mask_stocks,
    _objective_desc,
    _prebuilt_channel_coverage,
    _predict_outer,
    _replay_continuous_oos,
    _require_quality_gate,
    _resolve_universe,
    _slice_panel,
    _universe_artifact_hashes,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def _file_sha256(path: str) -> str:
    """Content SHA-256 of a file's bytes — the baseline tape's weight fingerprint.

    §十七: the pickle's real content hash, so a re-fit that changes the weights
    is detectable.  Full-length digest (not truncated) so two pickles collide
    only by cryptographic accident.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _baseline_source_hash() -> str:
    """§十六: SHA-256 over the baseline implementation + training-script source
    files — the model_source_hash a baseline tape carries (its analogue of the
    deep model's architecture-source hash).  Shared by every baseline model and
    fold of a run, so a baseline's folds only ever blend under the same source.
    """
    h = hashlib.sha256()
    root = str(get_project_root())
    for rel in ("stoke_ml/models/baseline/panel_baselines.py",
                "scripts/production/train_baselines_panel.py"):
        p = os.path.join(root, rel)
        fp = _file_sha256(p) if os.path.isfile(p) else "absent"
        h.update(rel.encode("utf-8"))
        h.update(b"=")
        h.update(fp.encode("utf-8"))
        h.update(b";")
    return h.hexdigest()[:16]


def _baseline_hyperparameter_dict(name: str) -> dict:
    """The exact hyperparameters ``make_model`` constructs for ``name`` (§十七).

    Kept in sync with ``make_model`` — a hyperparameter edit that changes what a
    baseline trains with must change this digest (and thus the tape / ledger /
    experiment signature).  The random seed is deliberately excluded: it is a
    separate research lever that already enters the experiment signature via the
    version's ``random_seed``.
    """
    if name == "ridge":
        return {"alpha": 10.0, "solver": "lsqr"}
    if name == "lgbm":
        return dict(_LGBMWrapper._PARAMS)
    if name == "mlp":
        return {
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "solver": "adam",
            "alpha": 1e-3,
            "batch_size": 256,
            "max_iter": 25,
            "early_stopping": False,
        }
    raise ValueError(f"unknown baseline model: {name}")


def _baseline_hyperparameter_hash(name: str) -> str:
    """Stable SHA-256 of a baseline family's hyperparameters (§十七)."""
    h = hashlib.sha256()
    h.update(json.dumps(_baseline_hyperparameter_dict(name),
                        sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()[:16]


# §十七: version tag of the baseline INPUT construction (build_flat_samples /
# entry_column_features / sequence_summary).  Bump when the feature vector a
# baseline is fit on changes shape or meaning.
_BASELINE_INPUT_RECIPE_VERSION = "flat-decision-cols+v1"


def _baseline_input_recipe_hash(with_seq_features: bool, seq_len: int) -> str:
    """SHA-256 over what defines a baseline's input vector (§十七): the
    --with-seq-features flag, the sequence-window length, and the construction
    version of the flat sample builder."""
    h = hashlib.sha256()
    h.update(f"recipe={_BASELINE_INPUT_RECIPE_VERSION};"
             f"with_seq={bool(with_seq_features)};seq_len={int(seq_len)};"
             .encode("utf-8"))
    return h.hexdigest()[:16]


# §十七: version tag of the baseline TRAINING-SAMPLE policy (date-stratified
# quota sampling inside build_flat_samples).  Bump when the sampling strategy
# changes.
_TRAINING_SAMPLE_POLICY_VERSION = "date-stratified-quotas+v1"


def _training_sample_policy_hash(max_train_rows: int) -> str:
    """SHA-256 over the training-sample policy (§十七): the --max-train-rows cap
    and the sampling strategy that turns the budget into a per-date sample."""
    h = hashlib.sha256()
    h.update(f"policy={_TRAINING_SAMPLE_POLICY_VERSION};"
             f"max_train_rows={int(max_train_rows)};".encode("utf-8"))
    return h.hexdigest()[:16]


# §十七: version tag of the baseline INPUT feature-scaling recipe — the
# StandardScaler applied to the flat training sample before fitting
# (``scaler = StandardScaler(); scaler.fit_transform(Xtr)``).  Bump when the
# transform a baseline is fit on changes (a different scaler class or a
# different fit basis).  The FITTED values are already fingerprinted per-fold
# by ``weight_hash`` — the real content hash of the pickled ``ScaledPredictor``
# — so this is the configuration-level identity.  The scaler itself is not a
# versioned artifact, so the version tag is the honest way to track it (the
# same convention as ``_BASELINE_INPUT_RECIPE_VERSION`` /
# ``_TRAINING_SAMPLE_POLICY_VERSION``).
_SCALER_RECIPE_VERSION = "standard-scaler-fit-train+v1"


def _scaler_hash() -> str:
    """SHA-256 over the baseline feature-scaling recipe (§十七): the scaler
    class + fit basis, pinned by the construction version tag.  Distinct from
    the input-recipe / hyperparameter / sample-policy hashes because it
    fingerprints the transform, not the feature vector or the model."""
    h = hashlib.sha256()
    h.update(f"scaler={_SCALER_RECIPE_VERSION};".encode("utf-8"))
    return h.hexdigest()[:16]


class _LGBMWrapper:
    """LightGBM regression booster with a sklearn-like fit/predict surface."""

    _PARAMS = {
        "objective": "regression",
        "metric": "l2",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.3,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 100,
        "verbosity": -1,
        "n_jobs": -1,
    }

    def __init__(self, seed: int):
        self._params = dict(self._PARAMS)
        self._params["seed"] = seed
        self._bst = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None):
        import lightgbm as lgb

        if X_val is not None and len(y_val) >= 20:
            # Early-stop on the same chronological inner_val
            # region the deep model uses for checkpoint selection, instead of a
            # fixed 150 rounds.
            self._bst = lgb.train(
                self._params, lgb.Dataset(X, label=y),
                num_boost_round=1000,
                valid_sets=[lgb.Dataset(X_val, label=y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        else:
            self._bst = lgb.train(
                self._params, lgb.Dataset(X, label=y), num_boost_round=150
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._bst is None:
            raise RuntimeError("not fitted")
        it = self._bst.best_iteration or 150
        return self._bst.predict(X, num_iteration=it)


def make_model(name: str, seed: int):
    """Construct the sklearn-style regressor for a baseline name.

    Built from ``_baseline_hyperparameter_dict`` so the actual model and its
    §十七 hyperparameter fingerprint cannot drift apart (the seed is added
    separately and stays out of the hyperparameter hash).
    """
    params = _baseline_hyperparameter_dict(name)
    if name == "ridge":
        return Ridge(**params)
    if name == "lgbm":
        return _LGBMWrapper(seed)
    if name == "mlp":
        return MLPRegressor(**params, random_state=seed)
    raise ValueError(f"unknown baseline model: {name}")


def main():
    parser = argparse.ArgumentParser(description="Panel baselines benchmark")
    parser.add_argument("--prebuilt", type=str, default="data/features_panel",
                        help="Panel-mode prebuilt features dir (same as train_panel.py)")
    parser.add_argument("--require-feature-manifest",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Require every prebuilt feature parquet to carry a "
                             "matching sidecar manifest (missing / stale / "
                             "schema-drift / different-git-commit FAILS the run "
                             "instead of warning). Default: on. Use "
                             "--no-require-feature-manifest for legacy prebuilt "
                             "dirs built without manifests (§十一-1)")
    parser.add_argument("--stocks", type=int, default=200,
                        help="Universe size / cap (default: 200)")
    parser.add_argument("--universe", type=str, default="random",
                        choices=["first", "random", "stratified", "all",
                                 "csi300", "csi500", "csi800"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=str, default="2000-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--max-folds", type=int, default=2,
                        help="Limit number of walk-forward folds (default: 2)")
    parser.add_argument("--lockbox-months", type=int, default=12)
    parser.add_argument("--models", type=str, default="ridge,lgbm,mlp,momentum",
                        help="Comma-separated baselines to run")
    parser.add_argument("--max-train-rows", type=int, default=100000,
                        help="Cap on baseline training rows per fold (benchmark "
                             "memory guard; sampled date-stratified)")
    parser.add_argument("--with-seq-features", action="store_true",
                        help="Append the shared sequence summary (trailing mean/std/"
                             "slope + lags of the seq_len window) to baseline inputs "
                             "so baselines see the same history as the xLSTM")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Override seq_len (default: 60)")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-require-quality-gate", action="store_true",
                        help="Skip the required quality-gate report check "
                             "(dev smoke only; §六-2 wants a matching report "
                             "before any real training run)")
    parser.add_argument("--no-formal", action="store_true",
                        help="Exploratory mode: allow degraded universe gates "
                             "when a required PIT artifact is missing, with a "
                             "prominent warning, instead of refusing to start "
                             "(§P0-7; formal is the default)")
    parser.add_argument("--quality-gate-report", type=str, default=None,
                        help="Path to the quality-gate report to verify "
                             "(default: <repo>/reports/data_quality_gate.json)")
    args = parser.parse_args()

    if args.end is None:
        args.end = datetime.now().strftime("%Y-%m-%d")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    cfg = load_config()
    data_dir = cfg.project.data_dir

    # §六-2: the SAME formal quality-gate contract as train_panel.py — a
    # baseline result on data the gate has not validated (or that changed since
    # the gate PASS) is not a research result (§P0-5).
    if not args.no_require_quality_gate:
        _report_path = args.quality_gate_report or str(
            get_project_root() / "reports" / "data_quality_gate.json"
        )
        _require_quality_gate(
            data_dir, args.prebuilt, _report_path,
            universe_name=args.universe, requested=False,
        )
        logger.info("Quality-gate report verified: %s", _report_path)

    # §七-P0: same full-market guard as train_panel.py — `--universe all` must
    # consume prebuilt panel features, never live-engineer the whole market.
    if args.universe == "all" and not args.prebuilt:
        raise SystemExit(
            "--universe all requires --prebuilt: the full market cannot be "
            "feature-engineered in memory (§七-P0).  Run "
            "scripts/production/build_features.py --panel-mode first, then "
            "re-run with --prebuilt data/features_panel."
        )

    all_stocks = _discover_stocks(data_dir, None)
    stock_list, universe_desc = _resolve_universe(
        all_stocks, args.universe, args.stocks, args.seed, data_dir,
        # §八-2: _resolve_universe's `formal` is the gate-ENFORCEMENT predicate
        # (_gate_enforced = not --no-require-quality-gate), NOT --no-formal — so
        # baselines' csi* missing-member refusal matches train_panel.  --no-formal
        # still governs the §P0-7 universe-artifact degradation via
        # _fold_universe_gates below, not this §八-2 drop refusal.
        formal=_gate_enforced(args),
    )
    if not stock_list:
        logger.error("No stocks found")
        sys.exit(1)
    universe_resolved = list(stock_list)
    universe_used = list(stock_list)

    logger.info("Universe: %s", universe_desc)
    logger.info("Loading K-line data for %d stocks from %s to %s...",
                len(stock_list), args.start, args.end)

    from stoke_ml.data.storage import DataStorage
    ds = DataStorage(data_dir)
    frames = []
    for code in stock_list:
        df = ds.load_daily(code, args.start, args.end,
                           require_valid_manifest=True)
        if df is not None and not df.empty:
            df["stock_code"] = code
            frames.append(df)
    if not frames:
        logger.error("No data loaded for any stock")
        sys.exit(1)

    panel = pd.concat(frames, ignore_index=True)

    seq_len = args.seq_len or 60
    fp = FeaturePipeline(
        seq_len=seq_len,
        use_sentiment=True, use_announcements=True,
        use_guba=True, use_comment=True, use_margin=True,
        use_northbound=True, use_dragon_tiger=True,
        use_fundamental=True, use_etf_flow=True,
        use_capital_flow=True, use_block_trade=True,
        use_shareholder=True, use_lockup=True, use_dividend=True,
        use_valuation=True,
        use_board=False, use_sector=False, use_concept=False,
    )
    panel_data = fp.build_panel_features(
        panel, aux_data=None, horizon=args.horizon, prebuilt_dir=args.prebuilt,
    )
    # §v12-P0: same row-identity invariant as train_panel.py — stock_codes
    # comes from the pipeline's valid_codes, never re-derived from the raw
    # panel (a cleaned-out stock would shift every subsequent row's label).
    panel_stocks = list(panel_data["stock_codes"])
    assert len(panel_stocks) == panel_data["past_observed"].shape[0], (
        "panel stock_codes length != past_observed rows (row identity broken)")
    assert len(panel_stocks) == panel_data["static_features"].shape[0], (
        "panel stock_codes length != static_features rows (row identity broken)")
    assert len(set(panel_stocks)) == len(panel_stocks), (
        "duplicate stock codes in panel (row identity broken)")
    channel_manifest = _prebuilt_channel_coverage(panel_data)

    global_dates = panel_data.get("global_dates")

    # §九-3: same strict formal-run scope as train_panel.py — a baseline
    # spanning forward-estimate calendar days must be refused, not silently run
    # on guessed holidays (§P0-5).
    refusal = _check_verified_until_scope(
        global_dates, enforce=not args.no_require_quality_gate)
    if refusal:
        raise SystemExit(refusal)

    # §七-1/§七-3: the SAME whole-run universe gates the deep model consumes —
    # 未退市 (nd_mask), per-day index membership (mem_mask) and the delisting
    # days feeding each fold's sleeve force-sell (delist_global).  A baseline
    # that ranked delisted / non-member stocks as tradable candidates would not
    # be measuring the same task (§P0-5).
    nd_mask, mem_mask, delist_global, universe_status = _fold_universe_gates(
        global_dates, panel_stocks, args.universe, data_dir,
        formal=not args.no_formal,
    )
    universe_hashes = _universe_artifact_hashes(
        universe_status, data_dir, args.universe,
    )

    # §十五-1 / §十二.6: same DSR multiplicity basis as the deep model — the
    # registry counts DISTINCT experiments.  Each baseline model in this run is
    # its own trial, so N = prior distinct trials + len(models).
    experiment_registry = _load_experiment_registry(_EXPERIMENT_REGISTRY_PATH)

    n_stocks = panel_data["static_features"].shape[0]
    n_timesteps = panel_data["past_known"].shape[1]
    static_dim = panel_data["static_features"].shape[2]
    pk_dim = panel_data["past_known"].shape[2]
    po_dim = panel_data["past_observed"].shape[2]
    logger.info("Panel data: %d stocks × %d timesteps  dims S=%d PK=%d PO=%d "
                "horizon=%d", n_stocks, n_timesteps, static_dim, pk_dim, po_dim,
                args.horizon)

    config = PanelConfig(
        seq_len=seq_len,
        static_dim=static_dim,
        past_known_dim=pk_dim,
        past_observed_dim=po_dim,
        horizon=args.horizon,
        seed=args.seed,
    )

    version_info = _experiment_version(
        data_dir, universe_used, args.prebuilt,
        static_dim, pk_dim, po_dim, config, args.start, args.end, args.seed,
    )
    logger.info("Version freeze: commit=%s data=%s feat=%s uni=%s cal=%s",
                version_info["git_commit"][:10],
                version_info["data_manifest_hash"],
                version_info["feature_schema_hash"],
                version_info["universe_hash"],
                version_info["calendar_version"])

    # §十六: baseline model-identity hashes for the OOS tape — the source hash
    # fingerprints the baseline implementation files (shared by every model and
    # fold), while the per-model config hash below binds a tape to its family.
    baseline_source_hash = _baseline_source_hash()

    val_len = 126
    step = val_len
    purge = config.seq_len
    lockbox_len = int(args.lockbox_months * 21)
    lockbox_start = max(0, n_timesteps - lockbox_len)
    if lockbox_start <= 0:
        logger.error("Lockbox (%d steps) leaves no trainable panel (n=%d)",
                     lockbox_len, n_timesteps)
        sys.exit(1)

    outdir = args.outdir or os.path.join(
        "reports", "experiments", "baselines_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(outdir, exist_ok=True)
    # §P0-5: per-fold OOS prediction tapes (same layout/contract as the deep
    # model's fold_XXX.npz) so a baseline number is offline-replayable.
    oos_dir = os.path.join(outdir, "oos_preds")
    os.makedirs(oos_dir, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    # §十二.6: distinct-experiment DSR multiplicity basis — each baseline model
    # in this run is its own trial.
    n_trials = _distinct_trial_count(experiment_registry) + len(models)
    summary_rows: list[dict] = []
    fold_records: list[dict] = []
    rng = np.random.RandomState(args.seed)

    # Reserve `horizon` steps before the lockbox as a
    # settlement buffer so the last fold's exits stop before the lockbox opens.
    last_val_start = n_timesteps - config.horizon - val_len - lockbox_len
    val_start = last_val_start
    fold = 0
    while val_start >= 0:
        if args.max_folds and fold >= args.max_folds:
            break
        train_end = val_start - purge
        if train_end < config.seq_len + 1:
            break
        fold += 1
        val_end = min(val_start + val_len, n_timesteps)
        val_context_start = val_start - config.seq_len

        fold_eligible = _fold_eligible_stocks(panel_data, train_end)
        fold_stocks = [panel_stocks[i] for i in np.where(fold_eligible)[0]]
        if len(fold_stocks) < 20:
            logger.warning("Fold %d: only %d stocks eligible PIT (need >= 20) — "
                           "skipping fold", fold, len(fold_stocks))
            val_start -= step
            continue

        # Mirror train_panel.py's inner_val carve-out.  The
        # deep model fits [0, inner_train_end) and selects its checkpoint on the
        # last ~15% of the trainable span; a baseline that trained on the whole
        # [0, train_end) would win by "seeing more labels", not by modeling the
        # data better.  inner_val here is used ONLY for early stopping /
        # checkpoint selection — never appended to the fit.
        n_train_targets = train_end - config.seq_len
        inner_val_len = max(1, int(round(0.15 * n_train_targets)))
        inner_val_context_start = train_end - inner_val_len - config.seq_len
        if inner_val_context_start < config.seq_len + 1:
            break
        inner_train_end = inner_val_context_start

        inner_train_data = _mask_stocks(
            _slice_panel(panel_data, slice(0, inner_train_end),
                         price_pad=config.horizon),
            fold_eligible,
        )
        inner_val_data = _mask_stocks(
            _slice_panel(panel_data, slice(inner_val_context_start, train_end),
                         price_pad=config.horizon),
            fold_eligible,
        )
        outer_test_data = _mask_stocks(
            _slice_panel(panel_data, slice(val_context_start, val_end),
                         price_pad=config.horizon),
            fold_eligible,
        )

        # §七-3: the SAME universe candidate gates the deep model applies to its
        # inner_val + outer_test pools (§P0-5) — a baseline must not rank
        # delisted / non-member stocks as tradable when the xLSTM could not.
        rows = np.where(fold_eligible)[0]
        for name, tslice, dd in (
            ("inner_val", slice(inner_val_context_start, train_end), inner_val_data),
            ("outer_test", slice(val_context_start, val_end), outer_test_data),
        ):
            _apply_candidate_gates(dd, tslice, rows, nd_mask, mem_mask)

        # Same target normalization as train_panel.py: cross-sectional z-score
        # per date (clean open-to-open returns), then clip to [-5, 5].
        for dd in (inner_train_data, inner_val_data, outer_test_data):
            dd["y_return"] = _cross_sectional_normalize(
                dd["y_return"], dd["return_target_mask"])
            np.clip(dd["y_return"], -5.0, 5.0, out=dd["y_return"])

        inner_train_T = inner_train_data["past_known"].shape[1]
        inner_val_T = inner_val_data["past_known"].shape[1]
        n_windows = outer_test_data["past_known"].shape[1] - config.seq_len

        # Early-stopping set for the deep-style baselines: the inner_val region,
        # with entry columns capped so every label is realized inside
        # [inner_val_context_start, train_end) — never into the outer test.
        inner_val_samples = None
        if any(m in ("lgbm", "mlp") for m in models):
            inner_val_samples = build_flat_samples(
                inner_val_data, config.seq_len, inner_val_T - config.horizon,
                seq_features=args.with_seq_features, seq_len=config.seq_len,
                sample_rng=rng)
            if len(inner_val_samples[1]) < 20:
                inner_val_samples = None

        logger.info("Fold %d/%d: inner_train [0:%d] inner_val [%d:%d] "
                    "outer_test [%d:%d] (%d eligible stocks, %d entry days)",
                    fold, args.max_folds or "∞",
                    0, inner_train_end,
                    inner_val_context_start, train_end,
                    val_context_start, val_end, len(fold_stocks), n_windows)

        for model_name in models:
            t0 = time.time()
            if model_name == "momentum":
                grid = build_momentum_grid(
                    outer_test_data, config.seq_len, config.seq_len + n_windows,
                    config.seq_len)
                adapter = PrecomputedScoreAdapter(grid)
                n_train = 0
            else:
                # PIT label gate: y_return[:, e] is realized at open[e+horizon],
                # so a training sample is usable only when its label is fully
                # known inside the fit span — entry_end = inner_train_T - horizon
                # for the carved train slice.  Without this
                # gate the baseline would train on labels unfolding into the
                # inner_val region or the purge gap before the test window.
                Xtr, ytr = build_flat_samples(
                    inner_train_data, config.seq_len,
                    inner_train_T - config.horizon,
                    max_rows=args.max_train_rows,
                    seq_features=args.with_seq_features,
                    seq_len=config.seq_len,
                    sample_rng=rng)
                n_train = len(ytr)
                if n_train < 100:
                    logger.warning("  fold %d %s: only %d training rows — skipping",
                                   fold, model_name, n_train)
                    continue
                scaler = StandardScaler()
                Xtr_s = scaler.fit_transform(Xtr)
                Xval_s = None
                yval = None
                if model_name in ("lgbm", "mlp") and inner_val_samples is not None:
                    Xval_s = scaler.transform(inner_val_samples[0])
                    yval = inner_val_samples[1]
                if model_name == "mlp" and Xval_s is not None:
                    # Chronological early stopping on the same
                    # inner_val region the deep model uses for checkpoint
                    # selection, instead of a fixed 25 epochs.
                    model = fit_mlp_with_early_stopping(
                        Xtr_s, ytr, Xval_s, yval, seed=args.seed)
                elif model_name == "lgbm" and Xval_s is not None:
                    model = _LGBMWrapper(args.seed)
                    model.fit(Xtr_s, ytr, X_val=Xval_s, y_val=yval)
                else:
                    model = make_model(model_name, args.seed)
                    model.fit(Xtr_s, ytr)
                adapter = FittedScoreAdapter(
                    ScaledPredictor(model, scaler),
                    with_seq=args.with_seq_features)
            adapter.reset()
            # §P0-6: force-sell delist-day grid in THIS fold's sim column space
            # (same mapping as train_panel.py) so a baseline tape replays the
            # delisting force-sell exactly as the sleeve account executed it.
            Wp = outer_test_data["close_price"].shape[1] - config.seq_len
            delist_day = _fold_delist_day(delist_global, fold_eligible, val_start, Wp)
            outer_m = evaluate_portfolio(
                adapter, outer_test_data, config, device,
                horizon=config.horizon,
                top_fraction=0.1,
                raw_returns=outer_test_data["realized_return"],
                require_price_path=True,
                n_boot=500,
                return_ledger=True,
                delist_day=delist_day,
                n_trials=n_trials,
            )
            elapsed = time.time() - t0
            # §P0-5: persist the fitted baseline (ridge/lgbm/mlp) alongside the
            # tape so fold_XXX_<model>.npz maps to exact fitted weights — the
            # analogue of the deep model's fold_XXX_model.pt.  §P1-3: the tape's
            # model_hash is the REAL content hash of this artifact (not a static
            # "baseline-<name>" string), so a re-fit that changes the weights is
            # detectable.  A pickle is best effort in non-formal mode: a score
            # tape without weights is still replayable from the saved preds grid,
            # so a serialization hiccup must not kill a fold — but in formal
            # mode an unhashable tape aborts the run instead of silently saving
            # a weight whose provenance is unverifiable.
            weight_hash = None
            try:
                model_path = os.path.join(
                    oos_dir, f"fold_{fold:03d}_{model_name}.pkl")
                with open(model_path, "wb") as f:
                    pickle.dump(adapter, f)
                weight_hash = _file_sha256(model_path)
            except Exception as exc:  # noqa: BLE001
                if not args.no_formal:
                    logger.error(
                        "  Fold %d %s: model pickle FAILED (%s) — formal mode "
                        "aborts rather than save a tape whose weight hash is "
                        "unverifiable", fold, model_name, exc)
                    raise
                logger.warning("  Fold %d %s: model pickle failed (%s) — "
                               "tape saved without weights", fold, model_name, exc)
            # §P0-5: per-fold OOS prediction tape, same layout/contract as the
            # deep model's fold_XXX.npz — a baseline number must be
            # offline-replayable through the same sleeve account.  The adapter's
            # replay grid was consumed by evaluate_portfolio, so reset before
            # re-running for the tape.
            adapter.reset()
            oos_preds = _predict_outer(adapter, outer_test_data, config, device)
            if oos_preds is not None:
                n_w = oos_preds.shape[1]
                p0 = config.seq_len
                entry_dates = [_fmt_date(global_dates, val_start + d)
                               for d in range(n_w)]
                dec = outer_test_data["decision_eligible_mask"][:, p0:p0 + n_w]
                hist = outer_test_data["history_eligible_mask"][:, p0:p0 + n_w]
                pool = dec & hist
                elig = outer_test_data["entry_eligible_mask"][:, p0:p0 + n_w]
                rt_mask = outer_test_data["return_target_mask"][:, p0:p0 + n_w]
                rt = outer_test_data["y_return_raw"][:, p0:p0 + n_w]
                price_grid = outer_test_data["close_price"][:, p0:p0 + n_w + config.horizon]
                open_grid = outer_test_data["open_price"][:, p0:p0 + n_w + config.horizon]
                price_dates = [_fmt_date(global_dates, val_start + d)
                               for d in range(n_w + config.horizon)]
                np.savez(
                    os.path.join(oos_dir, f"fold_{fold:03d}_{model_name}.npz"),
                    preds=oos_preds,
                    dates=np.array(entry_dates),
                    stocks=np.array(fold_stocks),
                    decision_eligible=dec,
                    history_eligible=hist,
                    pool=pool,
                    entry_eligible=elig,
                    return_target_mask=rt_mask,
                    return_target=rt,
                    close_price=price_grid,
                    open_price=open_grid,
                    price_dates=np.array(price_dates),
                    horizon=config.horizon,
                    seq_len=config.seq_len,
                    top_fraction=0.1,
                    cost=config.txn_cost,
                    delist_day=delist_day,
                    universe_status_hash=universe_hashes["universe_status_hash"],
                    membership_hash=universe_hashes["membership_hash"],
                    data_version=version_info["data_manifest_hash"],
                    # §P1-3: real content hash of the fitted pickle (not a
                    # static "baseline-<name>" string); falls back to the
                    # legacy string only when the non-formal pickle failed.
                    model_hash=weight_hash or ("baseline-" + model_name),
                    seed=args.seed,
                    evaluator_version=EVALUATOR_VERSION,
                    # §十八-1: omit the weight_hash key when the non-formal
                    # pickle failed — a stored None becomes an object-dtype
                    # array that crashes the replay reader (allow_pickle=False)
                    # before the §十八-1 gate / legacy tolerance can give the
                    # clean "unverifiable weights" handling.
                    **({"weight_hash": weight_hash}
                       if weight_hash is not None else {}),
                    calendar_hash=version_info["calendar_artifact_hash"],
                    # §十六: the split model-identity hashes the formal replay
                    # requires to be identical across a model's folds.  The
                    # config hash binds the tape to (model family, hyperparams,
                    # panel config); feature_schema_hash comes from the shared
                    # version freeze.  Only `weight_hash` may differ per fold.
                    model_source_hash=baseline_source_hash,
                    model_config_hash=hashlib.sha256(
                        f"baseline:{model_name}:{repr(config)}".encode("utf-8")
                    ).hexdigest()[:16],
                    feature_schema_hash=version_info["feature_schema_hash"],
                    # §十七: baseline identity hashes — input recipe (with_seq +
                    # seq_len + construction version), model hyperparameters,
                    # the training-sample policy, and the feature-scaling recipe
                    # — so the tape is bound to exactly what produced it, not
                    # just the model family.
                    baseline_input_recipe_hash=_baseline_input_recipe_hash(
                        args.with_seq_features, config.seq_len),
                    baseline_hyperparameter_hash=_baseline_hyperparameter_hash(
                        model_name),
                    training_sample_policy_hash=_training_sample_policy_hash(
                        args.max_train_rows),
                    scaler_hash=_scaler_hash(),
                    # §十五-3: identical policy metadata as the deep tapes, so a
                    # mixed oos_dir (deep + baseline, or two baselines) is
                    # rejected by the continuous replay instead of blended.
                    price_convention="open_to_open",
                    exit_policy="scheduled_horizon_delayed_delist_force_sell",
                    strategy_mode="long_top_fraction",
                )
                # Per-position ledger, same contract as train_panel.py — the
                # tape is self-contained, so each realized number is traceable.
                ledger_rows = outer_m.get("long_ledger")
                if ledger_rows:
                    ldf = pd.DataFrame(ledger_rows)
                    si = ldf["stock"].to_numpy(dtype=int)
                    di = ldf["entry_day"].to_numpy(dtype=int)
                    ldf["entry_date"] = [entry_dates[c] for c in di]
                    ldf["stock_code"] = [fold_stocks[i] for i in si]
                    ldf["prediction"] = oos_preds[si, di]
                    ldf["candidate_eligible"] = pool[si, di]
                    ldf["entry_eligible"] = elig[si, di]
                    ldf["fold"] = fold
                    ldf["data_version"] = version_info["data_manifest_hash"]
                    # §十七: per-fold REAL file hash (matches the tape's
                    # weight_hash / model_hash), falling back to the legacy
                    # label only when the non-formal pickle produced no hash.
                    ldf["model_hash"] = weight_hash or ("baseline-" + model_name)
                    # §十七: baseline identity hashes on every ledger row.
                    ldf["baseline_input_recipe_hash"] = _baseline_input_recipe_hash(
                        args.with_seq_features, config.seq_len)
                    ldf["baseline_hyperparameter_hash"] = _baseline_hyperparameter_hash(
                        model_name)
                    ldf["training_sample_policy_hash"] = _training_sample_policy_hash(
                        args.max_train_rows)
                    ldf["scaler_hash"] = _scaler_hash()
                    ldf = ldf[["fold", "entry_day", "entry_date", "stock",
                               "stock_code", "mode", "prediction",
                               "candidate_eligible", "entry_eligible",
                               "entry_price", "entry_notional", "target_weight",
                               "executed_weight", "entry_nav",
                               "shares", "scheduled_exit_day", "actual_exit_day",
                               "exit_status", "exit_price", "realized_return",
                               "mark_day", "mark_price", "gross_pnl",
                               "entry_cost", "exit_cost", "net_pnl",
                               "unrealized_pnl",
                               "baseline_input_recipe_hash",
                               "baseline_hyperparameter_hash",
                               "training_sample_policy_hash",
                               "scaler_hash"]]
                    ledger_path = os.path.join(
                        oos_dir, f"fold_{fold:03d}_{model_name}_ledger.parquet")
                    ldf.to_parquet(ledger_path)
                    logger.info("  Fold %d %s: ledger %d filled positions -> %s",
                                fold, model_name, len(ldf), ledger_path)
            rec = {
                "model": model_name,
                "fold": fold,
                "train_rows": n_train,
                "inner_val_start": _fmt_date(global_dates, inner_val_context_start),
                "inner_val_end": _fmt_date(global_dates, train_end - 1),
                "ls_sharpe": outer_m.get("ls_sharpe", 0.0),
                "ls_gross_sharpe": outer_m.get("ls_gross_sharpe", 0.0),
                "ic_mean": outer_m.get("ic_mean", 0.0),
                "ic_ir": outer_m.get("ic_ir", 0.0),
                "long_sharpe": outer_m.get("long_sharpe", 0.0),
                "q5mq1_ret": outer_m.get("q5mq1_ret", 0.0),
                "eligible_ew_sharpe": outer_m.get("eligible_ew_sharpe", 0.0),
                "selected_universe_ew_sharpe": outer_m.get("selected_universe_ew_sharpe", 0.0),
                "entry_start": _fmt_date(global_dates, val_start),
                "entry_end": _fmt_date(global_dates, val_start + val_len - 1),
                "seconds": round(elapsed, 1),
                "dsr_n_trials": outer_m.get("dsr_n_trials", 0),
                # §P1-3: real content hash of the persisted pickle — re-fit
                # changes it, so the registry can aggregate a real fingerprint.
                "weight_hash": weight_hash,
                # §十七: input-recipe / hyperparameter / sample-policy / scaler
                # hashes bind each fold's tape + ledger to the exact
                # configuration that produced it.
                "baseline_hyperparameter_hash": _baseline_hyperparameter_hash(model_name),
                "baseline_input_recipe_hash": _baseline_input_recipe_hash(
                    args.with_seq_features, config.seq_len),
                "training_sample_policy_hash": _training_sample_policy_hash(
                    args.max_train_rows),
                "scaler_hash": _scaler_hash(),
            }
            fold_records.append(rec)
            logger.info(
                "  %-8s fold %d | LS_Sharpe=%.2f IC=%.4f Long=%.2f Q5-Q1=%.1fbp "
                "ElgEW=%.2f SelUniEW=%.2f (%.1fs)",
                model_name, fold, rec["ls_sharpe"], rec["ic_mean"],
                rec["long_sharpe"], rec["q5mq1_ret"] * 10000,
                rec["eligible_ew_sharpe"], rec["selected_universe_ew_sharpe"], elapsed,
            )
        val_start -= step

    if not fold_records:
        logger.error("No baseline fold completed")
        sys.exit(1)

    rows = pd.DataFrame(fold_records)
    rows.to_csv(os.path.join(outdir, "baseline_table.csv"), index=False)

    summary = {
        "version": version_info,
        "universe": universe_desc,
        "models": models,
        "n_folds": int(rows["fold"].nunique()),
        "eval_note": (
            "Same prebuilt features + same non-overlapping folds + same "
            "evaluate_portfolio sleeve account as train_panel.py.  Baselines "
            "now mirror the deep model's inner_val carve-out: "
            "fit on [0, inner_train_end) only and early-stop / select on the "
            "same chronological inner_val region, with entry_end capped so no "
            "training label unfolds past the fit span.  Training rows are "
            "sampled date-stratified under --max-train-rows; "
            "LGBM/MLP tune on inner_val with chronological early stopping; "
            "--with-seq-features appends the shared "
            "sequence summary (trailing mean/std/slope + lags) so baselines see "
            "the same history as the xLSTM."
        ),
        "models_result": {},
    }
    for model_name in models:
        sub = rows[rows["model"] == model_name]
        if sub.empty:
            continue
        def _agg(col):
            vals = sub[col].dropna().astype(float)
            return {
                "mean": float(vals.mean()) if len(vals) else None,
                "std": float(vals.std()) if len(vals) > 1 else None,
                "n": int(len(vals)),
            }
        summary["models_result"][model_name] = {
            "ls_sharpe": _agg("ls_sharpe"),
            "ic_mean": _agg("ic_mean"),
            "long_sharpe": _agg("long_sharpe"),
            "q5mq1_ret": _agg("q5mq1_ret"),
            "eligible_ew_sharpe": _agg("eligible_ew_sharpe"),
            "selected_universe_ew_sharpe": _agg("selected_universe_ew_sharpe"),
        }

    # §十五-3: baseline continuous long-only OOS account — the SAME replay the
    # deep model produces.  Each baseline model's fold tapes
    # (fold_000_<model>.npz) are replayed as ONE account through
    # _run_sleeve_sim, so its headline Sharpe is comparable to the deep
    # model's oos_continuous_sharpe on the same LONG basis (§十二.3).  The
    # account's daily NAV + ledger are persisted next to the summary so the
    # number is traceable to positions, not just a scalar.  Formal replay is
    # used here too: a baseline tape missing required metadata must not be
    # silently blended into the account.
    cont_dir = os.path.join(outdir, "continuous")
    os.makedirs(cont_dir, exist_ok=True)
    for model_name in models:
        sub = rows[rows["model"] == model_name]
        if sub.empty:
            continue
        trial_sharpes = [float(v) for v in sub["long_sharpe"].dropna().astype(float)]
        try:
            cont = _replay_continuous_oos(
                oos_dir,
                model_name=model_name,
                formal=True,
                n_trials=n_trials,
                trial_sharpes=trial_sharpes,
            )
        except ValueError as exc:
            if not args.no_formal:
                raise
            logger.warning("  %s continuous OOS failed (%s) — skipped",
                           model_name, exc)
            cont = None
        if cont is None:
            logger.warning("  %s: no continuous OOS tapes found", model_name)
            continue
        daily = np.asarray(cont["account"]["daily"], dtype=np.float64)
        # Same NAV identity as the deep side: account starts at 1.0 on the
        # close before day 0, so NAV after day c = cumulative product of daily
        # returns through c (final row == final_nav).
        nav = (1.0 + daily).cumprod()
        pd.DataFrame({
            "price_date": cont["price_dates"],
            "nav": nav,
            "daily_return": daily,
        }).to_parquet(
            os.path.join(cont_dir, f"oos_continuous_{model_name}.parquet"),
            index=False)
        if cont["ledger"] is not None:
            cont["ledger"].to_parquet(
                os.path.join(cont_dir, f"oos_continuous_{model_name}_ledger.parquet"))
        mets = cont["metrics"]
        summary["models_result"][model_name]["oos_continuous"] = {
            "sharpe": mets["sharpe"],
            "maxdd": mets["maxdd"],
            "cagr": mets["cagr"],
            "final_nav": mets["final_nav"],
            "n_days": mets["n_days"],
            "n_stocks": mets["n_stocks"],
            "n_eff": mets.get("n_eff"),
            "psr": mets.get("psr"),
            "dsr": mets.get("dsr"),
            "dsr_trial_sharpes_n": mets.get("dsr_trial_sharpes_n"),
            "dsr_trial_variance_source": mets.get("dsr_trial_variance_source"),
            "file": f"continuous/oos_continuous_{model_name}.parquet",
            "ledger": (f"continuous/oos_continuous_{model_name}_ledger.parquet"
                       if cont["ledger"] is not None else None),
        }
        logger.info(
            "  %-8s continuous OOS: Sharpe=%.2f MaxDD=%.2f CAGR=%.4f "
            "final_nav=%.3f (%d days, %d stocks)",
            model_name, mets["sharpe"], mets["maxdd"],
            mets["cagr"] if mets["cagr"] is not None else float("nan"),
            mets["final_nav"] if mets["final_nav"] is not None else float("nan"),
            mets["n_days"], mets["n_stocks"])

    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(outdir, "universe_resolved.txt"), "w", encoding="utf-8") as f:
        f.write(f"# {universe_desc}\n# n={len(universe_resolved)}\n")
        f.write("\n".join(universe_resolved))
        f.write("\n")

    logger.info("=== Baseline Summary (%d folds, disjoint signal windows) ===",
                summary["n_folds"])
    header = (f"{'model':<10} {'LS_sharpe':>10} {'IC':>8} {'long_sharpe':>12} "
              f"{'q5mq1(bp)':>10} {'OOS_cont':>10}")
    logger.info(header)
    for model_name in models:
        r = summary["models_result"].get(model_name)
        if not r:
            continue
        cont = r.get("oos_continuous")
        cont_s = (f"{cont['sharpe']:.2f}"
                  if cont and cont.get("sharpe") is not None else "-")
        logger.info("%-10s %10.2f %8.4f %12.2f %10.1f %10s",
                    model_name,
                    r["ls_sharpe"]["mean"] or 0.0,
                    r["ic_mean"]["mean"] or 0.0,
                    r["long_sharpe"]["mean"] or 0.0,
                    (r["q5mq1_ret"]["mean"] or 0.0) * 10000,
                    cont_s)
    logger.info("Artifacts saved to %s", outdir)

    # §十二.6: register each baseline model as a distinct research trial in the
    # project-wide experiment ledger, so future DSR deflations count baseline
    # trials too (and this run's N).  One entry per model, deduped by its own
    # baseline signature — a re-run of the same baseline into a new outdir
    # replaces the old row instead of double-counting.
    for model_name in models:
        sub = rows[rows["model"] == model_name]
        if sub.empty:
            continue
        long_vals = sub["long_sharpe"].dropna().astype(float)
        # §十二.3: trial Sharpe on the LONG sleeve basis — the deep model's
        # oos_continuous_sharpe is a long account, so baseline trials use the
        # same basis to keep the historical DSR variance pool comparable.
        # §十五-3: when the continuous replay produced a REAL account, its
        # Sharpe is the headline (mirrors the deep model exactly); only fall
        # back to the mean fold long-Sharpe when no continuous account exists.
        cont_res = summary["models_result"].get(model_name, {}).get("oos_continuous")
        if cont_res and cont_res.get("sharpe") is not None:
            sharpe = float(cont_res["sharpe"])
        else:
            sharpe = float(long_vals.mean()) if len(long_vals) else None
        # §P1-3: the registry fingerprint is a REAL content hash over the fold
        # weight artifacts (SHA-256 of the sorted per-fold weight hashes), not a
        # static "baseline-<name>" label — a re-fit that changes any fold's
        # weights changes the fingerprint.  Falls back to the label only when
        # no fold produced a hashable pickle (non-formal failure path).
        weight_hashes = [str(v) for v in sub["weight_hash"].dropna()]
        if weight_hashes:
            model_hash = hashlib.sha256(
                "|".join(sorted(weight_hashes)).encode("utf-8")).hexdigest()
        else:
            model_hash = f"baseline-{model_name}"
        baseline_entry = {
            "experiment_signature": _experiment_signature(
                version_info, config, model_key=f"baseline-{model_name}",
                seq_features=args.with_seq_features,
                baseline_hyperparameter_hash=_baseline_hyperparameter_hash(model_name),
                baseline_input_recipe_hash=_baseline_input_recipe_hash(
                    args.with_seq_features, config.seq_len),
                training_sample_policy_hash=_training_sample_policy_hash(
                    args.max_train_rows),
                scaler_hash=_scaler_hash()),
            "outdir": outdir,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "git_commit": version_info.get("git_commit"),
            "data_manifest_hash": version_info.get("data_manifest_hash"),
            "feature_schema_hash": version_info.get("feature_schema_hash"),
            "model_hash": model_hash,
            "universe_hash": version_info.get("universe_hash"),
            "horizon": config.horizon,
            "objective": _objective_desc(config),
            "n_folds": int(sub["fold"].nunique()),
            "aborted": sharpe is None,
            "ls_sharpe_mean": float(sub["ls_sharpe"].mean()) if len(sub) else None,
            "oos_continuous_sharpe": sharpe,
            "psr": None,
            "dsr": None,
            "dsr_n_trials": n_trials,
            "model_family": "baseline",
        }
        _append_experiment_registry(_EXPERIMENT_REGISTRY_PATH, baseline_entry)
        logger.info("Baseline registered: %-8s -> trial signature %s",
                    model_name, baseline_entry["experiment_signature"])


if __name__ == "__main__":
    main()
