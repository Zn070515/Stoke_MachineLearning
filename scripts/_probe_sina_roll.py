"""Probe Sina roll keyword news API for historical depth on a stock code."""
import requests
import time


def roll(kw: str, page: int, num: int = 50):
    url = (
        "https://feed.mix.sina.com.cn/api/roll/get"
        f"?pageid=153&lid=2516&k={kw}&num={num}&page={page}"
        f"&r={time.time()}"
    )
    h = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://finance.sina.com.cn/",
    }
    try:
        r = requests.get(url, headers=h, timeout=15)
        j = r.json()
        items = j.get("result", {}).get("data", []) or []
        dates = sorted(i.get("ctime", "")[:10] for i in items)
        return f"{len(items)} items {dates[0] if dates else '-'}~{dates[-1] if dates else '-'}"
    except Exception as e:
        return f"ERR {type(e).__name__}: {e}"


for kw in ["000001", "平安银行"]:
    print(f"=== Sina roll keyword={kw} ===")
    for page in [1, 3, 10, 30, 100]:
        print(f"  page {page}: {roll(kw, page)}")
