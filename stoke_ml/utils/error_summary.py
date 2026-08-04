"""Error classification + aggregation for broad-exception sites.

The repo has many ``except Exception:`` sites, and some bare ``pass`` handlers.
Not all are wrong, but in the critical paths — download / storage / manifest /
preprocessing / feature build — a bare "log warning and continue" hides the
failure picture.  This module gives those loops:

  * ``classify_error`` — map an arbitrary exception onto a small typed category,
    so a broad catch can say *what kind* of error it just swallowed.  No optional
    dependency (requests / pandas / pyarrow / urllib3 ...) is imported at module
    load; third-party types are recognised by module prefix at call time.
  * ``ErrorSummary`` — a per-loop accumulator counting ``(category, source)``
    pairs, with ``record_exc`` auto-classification, ``merge`` for parallel
    workers, and ``report_lines`` / ``log_summary`` to emit the aggregate.

The intended use is the pattern

    summary = ErrorSummary()
    ...
    except Exception as exc:
        summary.record_exc(exc, "channel_load")
        # continue, but the run now reports what actually failed
    ...
    log_summary(summary, logger, "build_features")
"""

from __future__ import annotations

import errno
import enum
import json

__all__ = [
    "ErrorCategory",
    "ErrorSummary",
    "classify_error",
    "log_summary",
]

# Categories are stable ASCII keys — they appear in logs and reports verbatim.
class ErrorCategory(str, enum.Enum):
    NETWORK = "NETWORK"            # connectivity / timeout / TLS / retryable HTTP
    RESOURCE = "RESOURCE"          # provider-side: quota, rate-limit, 5xx/429
    IO = "IO"                      # generic disk / os-level failure
    PERMISSION = "PERMISSION"      # permissions / locked / disk-full
    NOT_FOUND = "NOT_FOUND"        # file / stock / record absent (benign skip)
    DATA_INTEGRITY = "DATA_INTEGRITY"  # parse/decode/corrupt-frame / bad shape
    SCHEMA = "SCHEMA"              # column/type drift (assigned explicitly)
    CONTRACT = "CONTRACT"          # manifest / contract mismatch (explicit)
    UNKNOWN = "UNKNOWN"            # anything not otherwise classified


# errno values that mean "network went away" — some are absent on Windows.
_NET_ERRNOS = {
    e for e in (
        getattr(errno, "ECONNRESET", None),
        getattr(errno, "ECONNABORTED", None),
        getattr(errno, "ECONNREFUSED", None),
        getattr(errno, "EHOSTDOWN", None),
        getattr(errno, "EHOSTUNREACH", None),
        getattr(errno, "ENETDOWN", None),
        getattr(errno, "ENETUNREACH", None),
        getattr(errno, "ENETRESET", None),
        getattr(errno, "ETIMEDOUT", None),
        getattr(errno, "EPIPE", None),
    ) if e is not None
}

_HTTP_MODULES = ("requests", "urllib3", "httpx", "aiohttp", "urllib.error")
_DATA_MODULES = ("pandas", "pyarrow", "fastparquet")


def _http_status(exc: BaseException) -> int | None:
    """Extract an HTTP status from a requests / urllib / httpx error, if any."""
    for attr in ("status_code", "status", "code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    resp = getattr(exc, "response", None)
    if resp is not None:
        st = getattr(resp, "status_code", None)
        if isinstance(st, int):
            return st
    return None


def classify_error(exc: BaseException) -> ErrorCategory:
    """Map an arbitrary exception onto an :class:`ErrorCategory`.

    Specific stdlib types are matched first (PermissionError before OSError,
    FileNotFoundError before IO).  Third-party HTTP/data libraries are matched
    by module prefix so they never need to be installed for classification to
    work; their status codes refine the category (404 -> NOT_FOUND, 429/5xx ->
    RESOURCE).
    """
    if isinstance(exc, PermissionError):
        return ErrorCategory.PERMISSION
    if isinstance(exc, (FileNotFoundError, FileExistsError,
                        NotADirectoryError, IsADirectoryError)):
        return ErrorCategory.NOT_FOUND
    if isinstance(exc, (ConnectionError, TimeoutError, BrokenPipeError)):
        return ErrorCategory.NETWORK
    if isinstance(exc, OSError):
        if exc.errno in _NET_ERRNOS:
            return ErrorCategory.NETWORK
        return ErrorCategory.IO

    mod = type(exc).__module__ or ""
    if mod.startswith(_HTTP_MODULES):
        status = _http_status(exc)
        if status in (404, 410):
            return ErrorCategory.NOT_FOUND
        if status is not None and (status == 429 or status >= 500):
            return ErrorCategory.RESOURCE
        return ErrorCategory.NETWORK
    if mod.startswith(_DATA_MODULES):
        return ErrorCategory.DATA_INTEGRITY
    if isinstance(exc, (ValueError, TypeError, KeyError, IndexError, LookupError,
                        UnicodeError)):
        return ErrorCategory.DATA_INTEGRITY
    return ErrorCategory.UNKNOWN


def _short_desc(exc: BaseException, limit: int = 200) -> str:
    msg = str(exc) or type(exc).__name__
    return msg if len(msg) <= limit else msg[: limit - 1] + "…"


class ErrorSummary:
    """Per-loop accumulator of (category, source) error counts.

    Not thread-safe by itself; parallel workers each build their own and the
    parent calls :meth:`merge`.
    """

    def __init__(self) -> None:
        self._counts: dict[tuple[str, str], int] = {}
        self._examples: dict[tuple[str, str], str] = {}

    # -- recording ------------------------------------------------------

    def record(self, category, source: str, *, detail: str | None = None) -> str:
        """Count one error under ``(category, source)``; keep the first detail."""
        cat = category.value if isinstance(category, ErrorCategory) else str(category)
        key = (cat, source)
        self._counts[key] = self._counts.get(key, 0) + 1
        if detail and key not in self._examples:
            self._examples[key] = detail
        return cat

    def record_exc(
        self, exc: BaseException, source: str, *, detail: str | None = None
    ) -> str:
        """Classify ``exc``, count it, and return the category string."""
        return self.record(classify_error(exc), source,
                           detail=detail if detail else _short_desc(exc))

    def merge(self, other: "ErrorSummary") -> "ErrorSummary":
        """Fold another summary into this one (e.g. a worker's per-stock tally)."""
        for key, n in other._counts.items():
            self._counts[key] = self._counts.get(key, 0) + n
        for key, ex in other._examples.items():
            self._examples.setdefault(key, ex)
        return self

    # -- reporting ------------------------------------------------------

    def total(self) -> int:
        return sum(self._counts.values())

    def as_dict(self) -> dict[str, dict[str, int]]:
        """``{category: {source: count}}``, keys sorted."""
        out: dict[str, dict[str, int]] = {}
        for (cat, src), n in sorted(self._counts.items()):
            out.setdefault(cat, {})[src] = n
        return out

    def rows(self) -> list[dict]:
        """Flattened rows sorted by count desc, then category."""
        return [
            {"category": cat, "source": src, "count": n}
            for (cat, src), n in sorted(
                self._counts.items(), key=lambda kv: (-kv[1], kv[0])
            )
        ]

    def report_lines(self) -> list[str]:
        """Human-readable lines; empty when nothing was recorded."""
        if not self._counts:
            return []
        lines = [f"Error summary ({self.total()} total)"]
        for row in self.rows():
            cat, src, n = row["category"], row["source"], row["count"]
            item = f"  {cat:<14} {src:<22} {n}"
            ex = self._examples.get((cat, src))
            if ex:
                item += f"  e.g. {ex}"
            lines.append(item)
        return lines

    def __bool__(self) -> bool:
        return bool(self._counts)

    def __len__(self) -> int:
        return self.total()


def log_summary(summary: ErrorSummary, logger, label: str) -> None:
    """Emit every report line of ``summary`` through ``logger`` (warning level)."""
    for line in summary.report_lines():
        logger.warning("%s: %s", label, line)
