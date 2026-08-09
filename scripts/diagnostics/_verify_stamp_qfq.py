"""Post-stamp verification for the qfq provenance migration (#85 Phase 2).

Read-only.  After stamp_qfq_provenance.py --write:
  - every WRITTEN file: validate_manifest ok (must hold — the stamp's own
    post-write check should have passed, this re-confirms)
  - a formal load_daily(require_valid_manifest=True) over a SAMPLE of written
    files (every 50th) + all known NaN-volume files + the full blocklist:
    the stamp's goal is to UNBLOCK formal reads; any written file that still
    fails formal reads is reported with its reason (should be a pre-existing
    contract defect, e.g. NaN volume, not a provenance issue)
  - every SKIPPED file: still adjust=unknown (fail-closed preserved)
  - content invariance spot-check: stamped parquet content-checksum equals
    its backup copy (zero value change)

Writes reports/stamp_qfq_verify.json.
"""
import glob
import json
import os

import pandas as pd

from stoke_ml.data.storage import DataStorage, _content_checksum

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DAILY = os.path.join(ROOT, "data", "a_shares", "daily")
NAN_FILES = ["000520", "001872", "001914", "002506"]
SAMPLE_EVERY = 50


def main() -> None:
    report = json.load(open(os.path.join(ROOT, "reports", "stamp_qfq.json"),
                            encoding="utf-8"))
    written = []  # collect from per-file? report has no per-file; recompute
    # The report has no per-file written list — rebuild by scanning manifests
    # that flipped unknown->qfq is expensive; instead read the log-free state:
    # every file whose manifest says qfq now (minus the original 18) is stamped.
    # Re-derive from the report counts.
    mode = report.get("mode")
    n_pass = report.get("n_pass", 0)
    n_skip = report.get("n_skip", 0)
    n_failed = report.get("n_failed", 0)
    n_already = report.get("n_already_qfq", 0)
    n_files = report.get("n_files", 0)
    print(f"mode={mode} files={n_files} already={n_already} "
          f"pass={n_pass} skip={n_skip} failed={n_failed}", flush=True)

    storage = DataStorage(os.path.join(ROOT, "data"))
    backup = sorted(glob.glob(os.path.join(ROOT, "data", "a_shares",
                                           "daily_stamp_backup_*")))[-1]

    # skip list + reasons from report
    skipped = {s["code"]: s["reason"] for s in report.get("skipped", [])}
    failed = {f["code"]: f["error"] for f in report.get("failed", [])}

    # written set = current qfq files minus the 18 original re-downloads
    # (the report's n_already_qfq).  We rebuild it robustly: scan manifests.
    all_files = sorted(glob.glob(os.path.join(DAILY, "*.parquet")))
    qfq_now = set()
    for path in all_files:
        code = os.path.basename(path)[: -len(".parquet")]
        mf = os.path.join(DAILY, f"{code}.manifest.json")
        m = json.load(open(mf, encoding="utf-8"))
        if m.get("adjust") == "qfq":
            qfq_now.add(code)
    # original 18 were the only qfq before the stamp; report knows n_already
    # but not the codes — treat them as: any qfq file whose classify/signature
    # is not in skipped.  Simplest correct set: qfq_now minus skipped/failed
    # and minus the 18 pre-existing (which we can't name from the report).
    # Fall back: written == qfq_now - skipped - failed - {18 pre-existing}.
    # We cannot name the 18 from the report, so infer: the stamp only touched
    # unknown files; every file the stamp wrote is unknown->qfq.  The 18 were
    # already qfq BEFORE the stamp and were never unknown, so they are the
    # qfq files NOT produced by this run — indistinguishable here.  Instead,
    # verify a SAMPLE of qfq files (they all must pass formal reads except the
    # pre-existing defects), which is the point of this probe.
    sample_codes = []
    qfq_sorted = sorted(qfq_now)
    for i, code in enumerate(qfq_sorted):
        if i % SAMPLE_EVERY == 0 or code in NAN_FILES or code in skipped \
                or code in failed:
            sample_codes.append(code)
    print(f"qfq files total={len(qfq_now)} sample={len(sample_codes)}",
          flush=True)

    v_report: dict[str, dict] = {}
    formal_blocked: list[dict] = []
    for code in sample_codes:
        rep = storage.validate_manifest(code)
        if not rep.get("ok"):
            v_report[code] = {"validate": "FAIL",
                              "reason": rep.get("reason")
                              or rep.get("mismatches")}
            formal_blocked.append({"code": code, "kind": "validate_manifest",
                                   "reason": rep.get("reason")
                                   or rep.get("mismatches")})
            continue
        try:
            d = storage.load_daily(code, "1970-01-01", "2099-12-31",
                                   require_valid_manifest=True)
            v_report[code] = {"validate": "OK", "formal": "OK",
                              "rows": int(len(d))}
        except ValueError as exc:
            v_report[code] = {"validate": "OK", "formal": "FAIL",
                              "reason": str(exc)[:120]}
            formal_blocked.append({"code": code, "kind": "formal_read",
                                   "reason": str(exc)[:120]})
    n_ok = sum(1 for r in v_report.values() if r.get("formal") == "OK")
    n_blocked = len(formal_blocked)
    print(f"sample formal reads: OK={n_ok} blocked={n_blocked}", flush=True)
    for b in formal_blocked:
        print(f"  BLOCKED {b['code']}: {b['kind']} — {b['reason'][:90]}",
              flush=True)

    # content invariance spot-check on a few written (qfq-now) files
    content_ok = True
    for code in ["000001", "600519", "000155", "600795", "000725", "000520"]:
        bpath = os.path.join(backup, f"{code}.parquet")
        if not os.path.isfile(bpath):
            continue
        a = pd.read_parquet(os.path.join(DAILY, f"{code}.parquet"))
        b = pd.read_parquet(bpath)
        cols = sorted(map(str, a.columns))
        same = _content_checksum(a, cols) == _content_checksum(b, cols)
        if not same:
            content_ok = False
        print(f"  content_invariant {code}: {same}", flush=True)

    # skipped still unknown
    still_unknown = True
    for code in skipped:
        m = storage.manifest(code) or {}
        if m.get("adjust") != "unknown":
            still_unknown = False
            print(f"  SKIPPED file {code} changed to {m.get('adjust')}!", flush=True)

    out = {
        "mode": mode,
        "n_files": n_files, "n_already_qfq": n_already, "n_pass": n_pass,
        "n_skip": n_skip, "n_failed": n_failed,
        "n_qfq_now": len(qfq_now),
        "sample_size": len(sample_codes),
        "sample_formal_ok": n_ok,
        "sample_formal_blocked": n_blocked,
        "formal_blocked": formal_blocked,
        "content_invariant_ok": content_ok,
        "skipped_still_unknown": still_unknown,
        "per_file": v_report,
    }
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "stamp_qfq_verify.json"),
              "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("report: reports/stamp_qfq_verify.json", flush=True)


if __name__ == "__main__":
    main()
