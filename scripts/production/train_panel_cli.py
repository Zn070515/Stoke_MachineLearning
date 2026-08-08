"""CLI argument construction for panel training (§二十一).

Extracted from ``scripts.production.train_panel`` — the architecture-ablation
switchboard (``_ABLATIONS``) and the ``build_parser`` ArgumentParser
construction.  ``train_panel`` re-exports these names for backward
compatibility.
"""
import argparse


# §十一.3 architecture-ablation switchboard.  Each entry maps a human name to
# PanelConfig field overrides that switch OFF one component of the production
# architecture, isolating where the model's edge comes from.  All default to
# the production config, so a run WITHOUT --ablation is the formal baseline.
_ABLATIONS: dict[str, dict] = {
    "plain_lstm": {"backbone": "lstm"},
    "vsn_lstm": {"backbone": "lstm", "use_vsn": True},
    "xlstm_no_vsn": {"use_vsn": False},
    "return_only": {"use_dir_head": False, "use_vol_head": False},
    "no_vol_head": {"use_vol_head": False},
    "no_dir_head": {"use_dir_head": False},
    "fixed_task_weights": {"fixed_task_weights": True},
    "no_ranking": {"use_ranking_loss": False},
    "no_pit_static": {"use_pit_static": False},
}


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser (exposed for tests)."""
    parser = argparse.ArgumentParser(description="Train VSN+xLSTM panel model")
    parser.add_argument("--stocks", type=int, default=500,
                        help="Universe size / cap (default: 500; with "
                             "--universe first: first N sorted; random/stratified: "
                             "N sampled; csi*: N cap)")
    parser.add_argument("--universe", type=str, default="random",
                        choices=["first", "random", "stratified", "all",
                                 "csi300", "csi500", "csi800"],
                        help="Stock universe selection (default: random; "
                             "csi* = index constituents, PIT ever-held union)")
    parser.add_argument("--allow-high-risk-universe", action="store_true",
                        help="§七-P0 escape hatch: an explicit override for the "
                             "universe memory guard — a high-memory universe "
                             "(all / large csi800) proceeds with a prominent "
                             "warning instead of being refused.  Default: "
                             "refused.")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for universe sampling and data "
                             "augmentation (default: 42)")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Experiment artifact dir (default: "
                             "reports/experiments/<timestamp>)")
    parser.add_argument("--stock-list", type=str, default=None,
                        help="Comma-separated stock codes")
    parser.add_argument("--start", type=str, default="2000-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-folds", type=int, default=3,
                        help="Limit number of walk-forward folds (default: 3)")
    parser.add_argument("--lockbox-months", type=int, default=0,
                        help="Reserve the last N months as an untouched lockbox "
                             "— no fold trains on or evaluates it; kept for a "
                             "single final run once the design freezes.  The "
                             "lockbox is single-use: the first FORMAL run that "
                             "opens it records the marker and a later formal run "
                             "is refused.  Default 0 = lockbox OFF (opt in for "
                             "the one final run with --lockbox-months 12).")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--horizon", type=int, default=5,
                        help="Forward return horizon in days (1/5/20)")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Model hidden dimension (default: 128)")
    parser.add_argument("--xlstm-blocks", type=int, default=2,
                        help="Number of xLSTM blocks (default: 2)")
    parser.add_argument("--rank-weight", type=float, default=0.1,
                        help="Ranking loss weight (0=disable, default: 0.1)")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=sorted(_ABLATIONS),
                        help="§十一.3: switch OFF ONE architecture component to "
                             "isolate where performance comes from.  Choices: "
                             + ", ".join(sorted(_ABLATIONS))
                             + ".  Default: full production architecture "
                             "(the formal baseline).")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="§十一.1: apply the fixed per-fold corruption pass "
                             "(Gaussian noise + one global time mask + one global "
                             "feature dropout, generated once and reused across "
                             "all epochs).  OFF by default — this is a fixed "
                             "data-corruption, not online per-sample augmentation, "
                             "so it is opt-in ablation only.")
    parser.add_argument("--log-gradient-flow", action="store_true",
                        help="Log per-parameter-group gradient norms each epoch "
                             "(after optimizer.step, before zero_grad)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--no-aux", action="store_true",
                        help="Skip auxiliary data loading (faster startup)")
    parser.add_argument("--require-aux-channels", type=str, default="",
                        help="Comma-separated aux channels that must have "
                             "loaded_stocks>0; experiment "
                             "FAILS otherwise. Default: none required")
    parser.add_argument("--feature-profile", type=str, default="headline_v1",
                        help="Frozen feature profile (stoke_ml/config/"
                             "feature_profile.py).  A FORMAL, gate-enforced run "
                             "adds the profile's required_channels to "
                             "--require-aux-channels and enforces its "
                             "minimum-coverage thresholds on probeable "
                             "channels.  'none' disables the required-channel "
                             "coverage gate (§十四). Default: headline_v1")
    parser.add_argument("--prebuilt", type=str, default=None,
                        help="Load panel-mode prebuilt features from this dir "
                             "(built via build_features.py --panel-mode). "
                             "Skips aux data loading and live feature "
                             "engineering — the panel is built from the "
                             "prebuilt parquets")
    parser.add_argument("--require-feature-manifest",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Require every prebuilt feature parquet to carry a "
                             "matching sidecar manifest (missing / stale / "
                             "schema-drift / different-git-commit FAILS the run "
                             "instead of warning). Default: on. Use "
                             "--no-require-feature-manifest for legacy prebuilt "
                             "dirs built without manifests (§十一-1)")
    parser.add_argument("--panel-store", type=str, default=None,
                        help="§十六 memmap lazy-storage dir for the built panel. "
                             "When DIR already holds a complete store it is loaded "
                             "instead of loading K-line + re-stacking the panel, so "
                             "a large-universe re-run never materializes the whole "
                             "dense (N,T,D) feature grid in RAM (arrays are mmap'd "
                             "and read lazily by the panel dataset / _slice_panel).  A "
                             "store's meta.json config fingerprint is checked on "
                             "load — a mismatch (horizon/seq_len/start/end/"
                             "universe/feature switches) REFUSES the run so a stale "
                             "store can't silently train on wrong targets.  "
                             "Otherwise the panel built this run is persisted there "
                             "for future runs.  Default: off — build in memory as "
                             "always.")
    parser.add_argument("--scratch-dir", type=str, default=None,
                        help="§T7 scratch dir for the STREAMING panel build's "
                             "per-stock Pass-1 pickles.  Default: derived as "
                             "<panel-store>/scratch/<run_id>/; with no "
                             "--panel-store the build is not streaming (dense "
                             "in-memory) so this is unused.  An explicit "
                             "location is never swept by the startup stale "
                             "cleanup (only tool-owned dirs are).")
    parser.add_argument("--no-require-quality-gate", action="store_true",
                        help="Skip the required quality-gate report check "
                             "(dev smoke only; §六-2 wants a matching report "
                             "before any real training run)")
    parser.add_argument("--no-formal", action="store_true",
                        help="Exploratory mode: allow degraded universe gates "
                             "when a required PIT artifact is missing, with a "
                             "prominent warning, instead of refusing to start "
                             "(§P0-7; formal is the default)")
    parser.add_argument("--strict-index-training",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="§八.3 + §T6 decision 2: gate the inner-TRAIN loss "
                             "by per-day index membership for csi300/csi500/"
                             "csi800.  Default: None = decide from the universe "
                             "— ON for the strict-CSI universes "
                             "(csi300/csi500/csi800), OFF otherwise.  An "
                             "explicit --strict-index-training / "
                             "--no-strict-index-training forces the value "
                             "regardless of universe.  When OFF, inner_train "
                             "learns from the broad historical-member union and "
                             "only the RANKED candidate pools "
                             "(inner_val/outer_test) are membership-gated.")
    parser.add_argument("--vintage-policy", type=str, default="revision-safe",
                        choices=["revision-safe", "allow-revised",
                                 "headline-strict"],
                        help="§T2/§T7/§T3: vintage-admission policy for the "
                             "feature set.  revision-safe (default) admits "
                             "channels whose source_vintage is "
                             "immutable_snapshot and DENIES "
                             "latest_revised-sourced ones (fundamental/macro/"
                             "earnings/valuation/pledge/shareholder/"
                             "index_membership/market_env_refine/sector/"
                             "concept); allow-revised additionally admits "
                             "latest_revised-sourced channels (legacy / "
                             "ablation use); headline-strict (new) is the "
                             "strictest tier — it additionally requires "
                             "pit_alignment == \"verified\" (with an explicit "
                             "scale-invariant waiver for daily_qfq/market_env), "
                             "so proxy-aligned channels are denied unless "
                             "waived.  The legacy name \"safe-only\" is the "
                             "pre-T3 alias for revision-safe.")
    parser.add_argument("--allow-fundamental-ablation", action="store_true",
                        help="T3 research decision #1: ABLATION ONLY — force the "
                             "fundamental channel ON even under revision-safe.  "
                             "This is the ONLY way fundamental enters a "
                             "revision-safe run; never use it for formal "
                             "headline/lockbox runs.  Under allow-revised "
                             "fundamental is already on, so this is a no-op "
                             "there.  Only the fundamental channel is affected "
                             "— the other policy-denied channels stay off.")
    parser.add_argument("--quality-gate-report", type=str, default=None,
                        help="Path to the quality-gate report to verify "
                             "(default: <repo>/reports/data_quality_gate.json)")
    parser.add_argument("--allow-missing-universe", action="store_true",
                        help="§八-2 escape hatch: proceed when the gate's "
                             "universe reconciliation reports requested stocks "
                             "missing from disk.  The missing list is still "
                             "recorded (universe_missing.txt in the outdir) — "
                             "the gap is surfaced, never silent.")
    parser.add_argument("--minute", action="store_true",
                        help="Use minute-frequency K-line data instead of daily")
    parser.add_argument("--minute-frequency", type=str, default="60",
                        choices=["5", "15", "30", "60"],
                        help="Bar frequency for minute mode (default: 60)")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Override seq_len (default: 60 daily, 64 minute)")
    parser.add_argument("--device", type=str, default="cuda")
    return parser
