"""§T6 decision 2: per-day index-member cross-section normalization.

``build_panel_features(..., daily_membership=...)`` restricts the per-date
cross-sectional STATISTICAL SET to that day's index members (half-open
``in_date <= date < out_date``); non-member stocks are still z-scored but do NOT
contribute to the mean/std.  ``daily_membership=None`` is the EXACT current
all-stock behavior.

These tests exercise the real ``build_panel_features`` output arrays (the z-norm
is never mocked).  The sentinel feature is a ``sentiment_mean`` column injected
through ``aux_data`` so an "extreme" value lands on a precise (stock, date)
without touching the OHLCV-engineered features.
"""

import numpy as np
import pandas as pd
import pytest

from stoke_ml.features.panel_builder import (
    _daily_member_flag, build_panel_features,
)
from stoke_ml.features.pipeline import FeaturePipeline

from features.test_static_feature_pit import _make_synthetic_panel

SEQ_LEN = 20
HORIZON = 5


def _pipeline(**kw):
    base = dict(
        seq_len=SEQ_LEN, minute_mode=False,
        use_board=False, use_sector=False, use_concept=False,
    )
    base.update(kw)
    return FeaturePipeline(**base)


def _col_index(cols, name):
    return list(cols).index(name)


def _date_index(global_dates, date):
    return int(np.where(global_dates == np.datetime64(date))[0][0])


def _member_frame(member_codes, in_date, out_date):
    """Long-form membership frame matching load_index_membership's schema."""
    return pd.DataFrame({
        "stock_code": member_codes,
        "index_code": ["000300"] * len(member_codes),
        "in_date": pd.to_datetime([in_date] * len(member_codes)),
        "out_date": pd.to_datetime([out_date] * len(member_codes)),
    })


def _sentiment_aux(panel, dates, values_by_code):
    """aux_data: {code: {"sentiment": df[date, sentiment_mean]}}."""
    aux = {}
    for code in panel["stock_code"].unique():
        aux[code] = {"sentiment": pd.DataFrame({
            "date": dates,
            "sentiment_mean": [float(values_by_code[code])] * len(dates),
        })}
    return aux


class TestDailyMemberFlag:
    """Unit tests for the row-level membership helper (the heart of §T6)."""

    def test_half_open_interval(self):
        feat = pd.DataFrame({
            "date": pd.to_datetime(["2022-01-04", "2022-01-05", "2022-01-06"]),
            "stock_code": ["600000", "600000", "600000"],
        })
        mem = pd.DataFrame({
            "stock_code": ["600000"],
            "in_date": [pd.Timestamp("2022-01-05")],
            "out_date": [pd.Timestamp("2022-01-07")],
        })
        flag = _daily_member_flag(feat, mem)
        # member on [2022-01-05, 2022-01-07) — in_date inclusive, out_date exclusive.
        assert flag.tolist() == [False, True, True]

    def test_na_t_out_date_member_forever(self):
        feat = pd.DataFrame({
            "date": pd.to_datetime(["2022-01-04", "2022-12-30", "2023-06-01"]),
            "stock_code": ["600000"] * 3,
        })
        mem = pd.DataFrame({
            "stock_code": ["600000"],
            "in_date": [pd.Timestamp("2022-01-05")],
            "out_date": [pd.NaT],
        })
        flag = _daily_member_flag(feat, mem)
        assert flag.tolist() == [False, True, True]

    def test_code_normalization_int_vs_str(self):
        """'600001' (str), 600001 (int), 600001.0 (float) must all match."""
        feat = pd.DataFrame({
            "date": pd.to_datetime(["2022-01-05", "2022-01-05", "2022-01-05"]),
            "stock_code": ["600001", 600001, 600001.0],
        })
        mem = pd.DataFrame({
            "stock_code": ["600001"],
            "in_date": [pd.Timestamp("2022-01-01")],
            "out_date": [pd.Timestamp("2022-02-01")],
        })
        flag = _daily_member_flag(feat, mem)
        assert flag.tolist() == [True, True, True]

    def test_non_members_false(self):
        feat = pd.DataFrame({
            "date": pd.to_datetime(["2022-01-05", "2022-01-05"]),
            "stock_code": ["600000", "600999"],
        })
        mem = pd.DataFrame({
            "stock_code": ["600000"],
            "in_date": [pd.Timestamp("2022-01-01")],
            "out_date": [pd.NaT],
        })
        flag = _daily_member_flag(feat, mem)
        assert flag.tolist() == [True, False]

    def test_overlapping_intervals_merge(self):
        """A stock in both indices of a csi800 universe has OVERLAPPING windows
        with different out_dates — the merged coverage must answer "any interval
        covers" exactly (searchsorted on the last-starting interval alone would
        miss the earlier interval's longer tail)."""
        feat = pd.DataFrame({
            "date": pd.to_datetime([
                "2022-07-01",   # inside BOTH (000300 out 2023, 000905 out 2022)
                "2022-12-01",   # inside 000300 only (000905 ended 2022-06-30)
                "2024-01-01",   # inside neither
            ]),
            "stock_code": ["600000"] * 3,
        })
        mem = pd.DataFrame({
            "stock_code": ["600000", "600000"],
            "in_date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2021-06-30")],
            "out_date": [pd.Timestamp("2023-12-31"), pd.Timestamp("2022-06-30")],
        })
        flag = _daily_member_flag(feat, mem)
        assert flag.tolist() == [True, True, False]

    def test_empty_membership_all_false(self):
        feat = pd.DataFrame({
            "date": pd.to_datetime(["2022-01-05"]),
            "stock_code": ["600000"],
        })
        empty = pd.DataFrame(columns=["stock_code", "in_date", "out_date"])
        assert _daily_member_flag(feat, empty).tolist() == [False]
        assert _daily_member_flag(feat, None).tolist() == [False]


class TestMemberLimitedNormalization:
    """§T6 decision 2 assertions on real build_panel_features output."""

    @pytest.fixture()
    def panel_dates(self):
        panel = _make_synthetic_panel(n_stocks=8, n_days=200)
        return panel, sorted(panel["date"].unique())

    def test_non_member_excluded_from_stats(self, panel_dates):
        """The core decision-2 assertion: an extreme NON-member value must not
        move the cross-section stats used to z-score a member stock; under the
        all-stock path it does."""
        panel, dates = panel_dates
        member_codes = [f"{600000 + i:06d}" for i in range(6)]
        non_member_codes = [f"{600006:06d}", f"{600007:06d}"]
        D = dates[100]
        membership = _member_frame(member_codes, dates[0], dates[-1])

        # Member M extreme +100 on D; non-member N extreme -100 on D; all else 0.
        aux = {}
        for code in panel["stock_code"].unique():
            rows = []
            for d in dates:
                v = 0.0
                if d == D:
                    if code == member_codes[0]:
                        v = 100.0
                    elif code == non_member_codes[0]:
                        v = -100.0
                rows.append({"date": d, "sentiment_mean": v})
            aux[code] = {"sentiment": pd.DataFrame(rows)}

        data_a = _pipeline().build_panel_features(
            panel, aux_data=aux, horizon=HORIZON)
        data_b = _pipeline().build_panel_features(
            panel, aux_data=aux, horizon=HORIZON,
            daily_membership=membership)

        col = _col_index(data_a["past_observed_cols"], "sentiment_mean")
        sidx = list(data_a["stock_codes"]).index(member_codes[0])
        dcol = _date_index(data_a["global_dates"], D)
        za = data_a["past_observed"][sidx, dcol, col]
        zb = data_b["past_observed"][sidx, dcol, col]

        # (a) all-stock: N's -100 drags the mean DOWN + inflates std → M's +100
        # z-score is diluted.  (b) member-only: N is excluded → M's z-score is
        # computed from member stats alone and is higher.  The two must differ,
        # in the direction proving N's value is out of (b)'s statistical set.
        assert abs(za - zb) > 0.05, f"member z differs: all={za:.4f} member={zb:.4f}"
        assert zb > za, f"member-only z={zb:.4f} should exceed all-stock z={za:.4f}"

    def test_zero_member_date_falls_back_to_all_stock(self, panel_dates):
        """A date with ZERO member rows must still be z-scored via the all-stock
        fallback — otherwise .map → NaN → nan_to_num zeroes the WHOLE date."""
        panel, dates = panel_dates
        member_codes = [f"{600000 + i:06d}" for i in range(6)]
        cut = dates[100]
        # Members cover only dates[0..99]; dates[100..] have ZERO members.
        membership = _member_frame(member_codes, dates[0], cut)
        # Distinct per-stock sentiment so the fallback cross-section varies.
        aux = _sentiment_aux(panel, dates, {c: i + 1 for i, c in enumerate(
            sorted(panel["stock_code"].unique()))})

        data = _pipeline().build_panel_features(
            panel, aux_data=aux, horizon=HORIZON,
            daily_membership=membership)

        assert np.isfinite(data["past_known"]).all()
        assert np.isfinite(data["past_observed"]).all()
        dcol = _date_index(data["global_dates"], cut)
        # The zero-member date must NOT be an all-zero-from-NaN column — the §T6
        # fallback keeps its features finite and cross-sectionally meaningful.
        assert np.count_nonzero(data["past_observed"][:, dcol, :]) > 0
        assert np.count_nonzero(data["past_known"][:, dcol, :]) > 0

    def test_member_only_stats_used(self, panel_dates):
        """On a fully-member date the stats equal the member-only mean/std —
        hand-compute from member rows and compare against the z-scored output."""
        panel, dates = panel_dates
        member_codes = [f"{600000 + i:06d}" for i in range(6)]
        non_member_codes = [f"{600006:06d}", f"{600007:06d}"]
        membership = _member_frame(member_codes, dates[0], dates[-1])

        # Members hold values [1..6], non-members hold -50 (never in the stats).
        values = {c: i + 1 for i, c in enumerate(member_codes)}
        values.update({c: -50.0 for c in non_member_codes})
        aux = _sentiment_aux(panel, dates, values)

        data = _pipeline().build_panel_features(
            panel, aux_data=aux, horizon=HORIZON,
            daily_membership=membership)

        col = _col_index(data["past_observed_cols"], "sentiment_mean")
        member_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        mean = float(np.mean(member_vals))
        std = float(np.std(member_vals, ddof=1))  # groupby.agg('std') = ddof=1
        D = dates[50]
        dcol = _date_index(data["global_dates"], D)
        stocks = list(data["stock_codes"])
        for i, code in enumerate(member_codes):
            z = data["past_observed"][stocks.index(code), dcol, col]
            expected = (member_vals[i] - mean) / std
            assert z == pytest.approx(expected, abs=1e-3), (
                f"member {code} z={z:.4f} != member-only z={expected:.4f}")

    def test_zero_member_sparse_date_pooled_fallback(self):
        """A ZERO-member date whose all-stock cross-section is sparse (<5 rows)
        must route the missing-date fallback through the sparse
        expanding-moments path too.  Pre-fix the fallback ran a RAW groupby: on
        a degenerate cross-section (e.g. [0, 0, 0]) the std clips to the 1e-8
        floor and that date's z-scores collapse to exactly 0.  The pooled
        expanding-moments fallback yields a real cross-sectional std instead.
        """
        panel = _make_synthetic_panel(n_stocks=3, n_days=200)
        dates = sorted(panel["date"].unique())
        member_codes = [f"{600000 + i:06d}" for i in range(2)]
        non_member = f"{600002:06d}"
        cut = dates[100]
        # Members cover only dates[0..99]; dates[100..] have ZERO member rows.
        membership = _member_frame(member_codes, dates[0], cut)
        spike = dates[100]   # a zero-member date — non-member spikes to 100
        target = dates[105]  # later zero-member date — non-member is back to 0

        # Members stay constant 0.0; the non-member is 0.0 except a one-day
        # spike on ``spike``.  On ``target`` the all-stock cross-section is
        # [0, 0, 0] → the raw groupby std is 0 (clipped to 1e-8), but the
        # expanding-moments fallback pools the earlier spike and yields a real
        # pooled std, so ``target`` is z-scored with genuine cross-sectional
        # scale rather than a degenerate clip.
        aux = {}
        for code in panel["stock_code"].unique():
            rows = []
            for d in dates:
                v = 0.0
                if code == non_member and d == spike:
                    v = 100.0
                rows.append({"date": d, "sentiment_mean": v})
            aux[code] = {"sentiment": pd.DataFrame(rows)}

        data = _pipeline().build_panel_features(
            panel, aux_data=aux, horizon=HORIZON,
            daily_membership=membership)

        assert np.isfinite(data["past_observed"]).all()
        col = _col_index(data["past_observed_cols"], "sentiment_mean")
        sidx = list(data["stock_codes"]).index(non_member)
        dcol = _date_index(data["global_dates"], target)
        z = data["past_observed"][sidx, dcol, col]
        # Pre-fix the fallback's degenerate clip makes this exactly 0; the
        # pooled fallback must produce a real, non-zero statistic instead.
        assert abs(z) > 1e-3, (
            f"degenerate clip: zero-member sparse-date z={z:.6f} is ~0")
        assert abs(z) < 20, (
            f"z not bounded after pooled fallback: {z:.6f}")
