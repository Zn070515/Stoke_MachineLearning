"""Vintage-based channel admission policy (§T2/T7 / v15 §六/§十, v16 §十二).

Decides which channels a training run may consume based on their declared 3-dim
vintage classification (see ``stoke_ml.data.channel_vintage``).

- ``SAFE_ONLY`` (``"safe-only"``) admits ``immutable_snapshot``-sourced
  channels and DENIES ``latest_revised``-sourced ones (fundamental/macro/
  earnings/valuation/pledge/shareholder/index_membership/market_env_refine/
  sector/concept).  The default for formal headline/lockbox runs — a
  research-correctness guard against revision leakage.  ``daily_qfq`` is
  ``immutable_snapshot``-sourced, so the price channel stays admissible (a
  model cannot train without it).
- ``ALLOW_REVISED`` (``"allow-revised"``) additionally admits the
  ``latest_revised``-sourced channels (legacy / ablation use).

Admission checks BOTH the source and transform layers: a channel whose
``source_vintage`` or ``transform`` is the reserved ``"unknown"`` fallback (or
an undeclared channel) is denied under BOTH policies — the mandatory
deny-by-default.  ``pit_alignment`` is a RECORDING dimension and does not gate
admission.

This module imports ``channel_vintage`` one-way; ``channel_vintage`` never
imports this module (no circular import).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from stoke_ml.data import channel_vintage as _cv


class VintagePolicy(Enum):
    SAFE_ONLY = "safe-only"
    ALLOW_REVISED = "allow-revised"


def channel_allowed(
    channel: str,
    policy: VintagePolicy,
    *,
    vintage_by_name: dict | None = None,
) -> bool:
    """Whether ``channel`` may be consumed under ``policy``.

    SOURCE-based admission with a BOTH-LAYERS check: an undeclared channel (no
    declaration) OR a channel whose ``source_vintage``/``transform`` is the
    reserved ``"unknown"`` fallback is False under BOTH policies — the
    mandatory deny-by-default.  ``immutable_snapshot``-sourced channels are
    always allowed; ``latest_revised``-sourced channels only under
    ``ALLOW_REVISED``.
    """
    if not isinstance(policy, VintagePolicy):
        policy = VintagePolicy(policy)
    entry = _cv.declaration_of(channel, vintage_by_name=vintage_by_name)
    if entry is None:
        return False
    if entry.source_vintage == "unknown" or entry.transform == "unknown":
        return False
    if entry.source_vintage == "latest_revised":
        return policy is VintagePolicy.ALLOW_REVISED
    return True


def allowed_channels(
    policy: VintagePolicy,
    *,
    declaration=None,
) -> frozenset[str]:
    """Channels the policy admits, over ``declaration``.

    ``declaration`` defaults to ``_cv.CHANNEL_VINTAGE``; a caller may pass a
    crafted declaration (test injection point) without touching module globals.
    """
    if declaration is None:
        declaration = _cv.CHANNEL_VINTAGE
    by_name = {e.channel: e for e in declaration}
    return frozenset(
        e.channel
        for e in declaration
        if channel_allowed(e.channel, policy, vintage_by_name=by_name)
    )


def denied_channels(
    policy: VintagePolicy,
    *,
    declaration=None,
) -> frozenset[str]:
    """Declared channels the policy denies — the complement of
    ``allowed_channels`` over the declaration's declared channels."""
    if declaration is None:
        declaration = _cv.CHANNEL_VINTAGE
    declared = {e.channel for e in declaration}
    return declared - allowed_channels(policy, declaration=declaration)


def vintage_report(
    policy: VintagePolicy,
    *,
    declaration=None,
    documented_dims=None,
) -> dict:
    """The run's vintage-admission report.

    Both ``declaration`` and ``documented_dims`` are TEST INJECTION POINTS — a
    caller can pass a crafted partial declaration or a hypothetical declaration
    to prove enforcement without touching module globals.

    Returns ``{"vintage_policy", "channels": [{channel,source_vintage,
    transform,pit_alignment,rationale,allowed}...], "missing_channels",
    "daily_qfq_allowed", "declaration_complete"}``.  ``declaration_complete``
    is True iff ``missing_channels`` is empty AND every declared channel has
    all three dims set to DECLARED (non-``"unknown"``) values.
    """
    if not isinstance(policy, VintagePolicy):
        policy = VintagePolicy(policy)
    if declaration is None:
        declaration = _cv.CHANNEL_VINTAGE
    if documented_dims is None:
        documented_dims = _cv.DOCUMENTED_USE_DIMS
    by_name = {e.channel: e for e in declaration}
    channels = [
        {
            "channel": e.channel,
            "source_vintage": e.source_vintage,
            "transform": e.transform,
            "pit_alignment": e.pit_alignment,
            "rationale": e.rationale,
            "allowed": channel_allowed(e.channel, policy, vintage_by_name=by_name),
        }
        for e in declaration  # declaration order preserved → deterministic
    ]
    missing_channels = sorted(
        set(documented_dims) - {e.channel for e in declaration}
    )
    declaration_complete = (
        not missing_channels
        and all(
            e.source_vintage != "unknown"
            and e.transform != "unknown"
            and e.pit_alignment != "unknown"
            for e in declaration
        )
    )
    return {
        "vintage_policy": policy.value,
        "channels": channels,
        "missing_channels": missing_channels,
        "daily_qfq_allowed": channel_allowed(
            "daily_qfq", policy, vintage_by_name=by_name
        ),
        "declaration_complete": declaration_complete,
    }


@dataclass(frozen=True)
class UniverseVintagePolicy:
    """§十四: declared provenance of the UNIVERSE-membership gate, SEPARATE
    from the feature ``VintagePolicy``.

    The feature policy governs which CHANNELS a run may consume; this governs
    the CSI universe gate's membership data.  A CSI universe (csi300/csi500/
    csi800) consumes ``membership.parquet``, which is Baostock-MONTHLY-
    RECONSTRUCTED (NOT official effective-date data), so feature-vintage
    ``safe-only`` does NOT mean the research avoided latest-reconstructed
    data — the universe gate itself reads latest-reconstructed membership.
    That must be declared EXPLICITLY (in store meta / experiment summary /
    signature), never implied-bypassed.
    """

    source: str
    vintage: str
    resolution: str

    def provenance(self) -> dict:
        """The provenance dict recorded in store meta / experiment summary."""
        return {
            "source": self.source,
            "vintage": self.vintage,
            "resolution": self.resolution,
        }


# §T6/§十四: the CSI membership provenance — Baostock monthly reconstruction,
# latest-reconstructed, consumed by every csi300/csi500/csi800 universe gate.
CSI_MONTHLY_RECONSTRUCTED = UniverseVintagePolicy(
    source="Baostock monthly reconstruction",
    vintage="latest-reconstructed",
    resolution="monthly",
)

# Mirrors the CSI set in train_panel_universe._is_csi_universe and
# train_panel_folds._universe_artifact_hashes (existing duplication — the data
# layer must not import from scripts; not consolidated here).
CSI_UNIVERSE_NAMES = frozenset({"csi300", "csi500", "csi800"})


def universe_membership_provenance(universe_name: str | None) -> dict | None:
    """The membership provenance for ``universe_name``, or None when the
    universe does not consume membership.parquet (any non-CSI universe)."""
    if universe_name in CSI_UNIVERSE_NAMES:
        return CSI_MONTHLY_RECONSTRUCTED.provenance()
    return None
