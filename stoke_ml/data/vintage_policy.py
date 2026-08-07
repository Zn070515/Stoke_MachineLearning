"""Vintage-based channel admission policy (§T2/T7 / v15 §六/§十, v16 §十二,
§T3/v17 §九).

Decides which channels a training run may consume based on their declared 3-dim
vintage classification (see ``stoke_ml.data.channel_vintage``).

- ``REVISION_SAFE`` (``"revision-safe"``; the pre-T3 ``"safe-only"`` tier,
  renamed) admits ``immutable_snapshot``-sourced channels and DENIES
  ``latest_revised``-sourced ones (fundamental/macro/earnings/valuation/pledge/
  shareholder/index_membership/market_env_refine/sector/concept).  The default
  for formal headline/lockbox runs — a research-correctness guard against
  revision leakage.  ``daily_qfq`` is ``immutable_snapshot``-sourced, so the
  price channel stays admissible (a model cannot train without it).
- ``ALLOW_REVISED`` (``"allow-revised"``) additionally admits the
  ``latest_revised``-sourced channels (legacy / ablation use).
- ``HEADLINE_STRICT`` (``"headline-strict"``, NEW) is the strictest tier: on
  top of the source+transform check it ALSO gates on ``pit_alignment ==
  "verified"``.  A ``proxy``-aligned channel is denied UNLESS it is on the
  explicit scale-invariant waiver whitelist
  (``HEADLINE_STRICT_WAIVER_CHANNELS``) — see that constant for the per-channel
  rationale.

Admission checks the source and transform layers under EVERY policy: a channel
whose ``source_vintage`` or ``transform`` is the reserved ``"unknown"``
fallback (or an undeclared channel) is denied under all tiers — the mandatory
deny-by-default.  ``pit_alignment`` is a RECORDING dimension under
revision-safe / allow-revised and becomes an admission gate under
headline-strict (plus its waivers).

Backward compatibility: the legacy serialized string ``"safe-only"`` (the
pre-T3 name of the revision-safe tier) still parses to ``REVISION_SAFE`` via a
``VintagePolicy`` alias lookup, so old store/experiment records are not all
rejected by strict reads.

This module imports ``channel_vintage`` one-way; ``channel_vintage`` never
imports this module (no circular import).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from stoke_ml.data import channel_vintage as _cv


class VintagePolicy(Enum):
    REVISION_SAFE = "revision-safe"
    ALLOW_REVISED = "allow-revised"
    HEADLINE_STRICT = "headline-strict"

    @classmethod
    def _missing_(cls, value):
        # Legacy alias (§T3): pre-T3 stored "safe-only" strings parse to the
        # renamed REVISION_SAFE tier — the source-layer admission semantics are
        # UNCHANGED, so old store/experiment records keep parsing on strict reads.
        if value == "safe-only":
            return cls.REVISION_SAFE
        return None


# §T3: the scale-invariant waiver whitelist for HEADLINE_STRICT.  A channel
# here passes headline-strict EVEN IF its declared pit_alignment is "proxy".
# The waiver is justified ONLY by scale-invariance to the proxy-day alignment:
# the feature a model actually consumes is unchanged by the proxy re-anchoring.
# A non-scale-invariant proxy channel (e.g. industry, whose membership ALSO
# derives from the historically-restructured sector classification) is NOT
# waived and stays alignment-gated.
HEADLINE_STRICT_WAIVER_CHANNELS: frozenset[str] = frozenset({
    # daily_qfq: the qfq re-anchoring is a per-date UNIFORM scale factor over
    # all history; adjusted-price RETURNS (and price ratios) cancel that factor,
    # so the proxy alignment does not bias the price signal a model trains on.
    "daily_qfq",
    # market_env: §T5 split — the broadcast file carries a PRICE part
    # (high/low breadth, market turnover, industry advance — scale-invariant
    # ratios/ranks from realized prices, where qfq re-scaling cancels) and an
    # ACCOUNT part (absolute monthly counts: investor_new_num etc., NOT
    # scale-invariant).  HEADLINE_STRICT consumes ONLY the verified PRICE part
    # (the account part is PROXY-PIT and ablation-only via use_market_env_account
    # under every tier), so the scale-invariance justification applies only to
    # the part actually consumed — the waiver is accurate.
    "market_env",
})


def channel_allowed(
    channel: str,
    policy: VintagePolicy,
    *,
    vintage_by_name: dict | None = None,
) -> bool:
    """Whether ``channel`` may be consumed under ``policy``.

    SOURCE-based admission with a BOTH-LAYERS check: an undeclared channel (no
    declaration), a channel whose ``source_vintage``/``transform`` is OUTSIDE
    the KNOWN_* vocabularies, OR one set to the reserved ``"unknown"`` fallback
    is False under EVERY policy — the mandatory deny-by-default.
    ``immutable_snapshot``-sourced channels are always allowed (for
    revision-safe / allow-revised); ``latest_revised``-sourced channels only
    under ``ALLOW_REVISED``.

    ``HEADLINE_STRICT`` additionally gates on ``pit_alignment == "verified"``:
    a proxy-aligned channel is False unless it is on
    ``HEADLINE_STRICT_WAIVER_CHANNELS``.  Under revision-safe / allow-revised
    ``pit_alignment`` remains a RECORDING dimension and does not gate.
    """
    if not isinstance(policy, VintagePolicy):
        policy = VintagePolicy(policy)
    entry = _cv.declaration_of(channel, vintage_by_name=vintage_by_name)
    if entry is None:
        return False
    if entry.source_vintage not in _cv.KNOWN_SOURCE_VINTAGES:
        return False  # out-of-vocabulary source → treat as undeclared
    if entry.transform not in _cv.KNOWN_TRANSFORMS:
        return False  # out-of-vocabulary transform → treat as undeclared
    if entry.source_vintage == "unknown" or entry.transform == "unknown":
        return False
    if entry.source_vintage == "latest_revised":
        return policy is VintagePolicy.ALLOW_REVISED
    if policy is VintagePolicy.HEADLINE_STRICT:
        if entry.pit_alignment != "verified":
            return channel in HEADLINE_STRICT_WAIVER_CHANNELS
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
    all three dims set to DECLARED values — a member of its KNOWN_* vocabulary
    AND not the reserved ``"unknown"`` fallback.  ``pit_alignment`` is surfaced
    per channel so a headline-strict denial (or waiver) is auditable.
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
            e.source_vintage in _cv.KNOWN_SOURCE_VINTAGES
            and e.source_vintage != "unknown"
            and e.transform in _cv.KNOWN_TRANSFORMS
            and e.transform != "unknown"
            and e.pit_alignment in _cv.KNOWN_PIT_ALIGNMENTS
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
    ``revision-safe`` does NOT mean the research avoided latest-reconstructed
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
