# DIAGNOSTIC (diagnostics): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""Probe historical depth of capital-flow and margin APIs."""
import sys

from stoke_ml.data.sources.a_shares.capital_flow_source import CapitalFlowSource

print("=== capital flow (Sina) 000001 days=6000 ===", flush=True)
try:
    src = CapitalFlowSource()
    df = src.fetch_daily("000001", days=6000)
    print(f"rows={len(df)} range={df['date'].min()}~{df['date'].max()}", flush=True)
except Exception as e:
    print(f"ERR: {e}", flush=True)

print("\n=== margin (AKShare SSE) historical dates ===", flush=True)
import akshare as ak
for d in ["20180102", "20190102", "20200102", "20220104"]:
    try:
        sse = ak.stock_margin_detail_sse(date=d)
        print(f"SSE {d}: {len(sse)} rows  cols={list(sse.columns)[:4]}", flush=True)
    except Exception as e:
        print(f"SSE {d}: ERR {type(e).__name__} {e}", flush=True)
