"""Unit tests for stoke_ml.utils.error_summary."""
import errno
import json

from stoke_ml.utils.error_summary import (
    ErrorCategory,
    ErrorSummary,
    classify_error,
    log_summary,
)


def _make(module: str, name: str = "Error", **attrs) -> BaseException:
    """Build an exception that *claims* a third-party module (no import needed)."""
    cls = type(name, (Exception,), {"__module__": module})
    obj = cls("boom")
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


class TestClassifyError:
    def test_stdlib_not_found(self):
        assert classify_error(FileNotFoundError("x")) is ErrorCategory.NOT_FOUND
        assert classify_error(NotADirectoryError()) is ErrorCategory.NOT_FOUND

    def test_stdlib_permission(self):
        assert classify_error(PermissionError()) is ErrorCategory.PERMISSION

    def test_stdlib_network(self):
        assert classify_error(ConnectionResetError()) is ErrorCategory.NETWORK
        assert classify_error(TimeoutError()) is ErrorCategory.NETWORK
        assert classify_error(BrokenPipeError()) is ErrorCategory.NETWORK

    def test_oserror_network_errno(self):
        err = OSError(getattr(errno, "ECONNREFUSED", 111), "refused")
        assert classify_error(err) is ErrorCategory.NETWORK

    def test_oserror_io(self):
        err = OSError(errno.EIO, "disk failure")
        assert classify_error(err) is ErrorCategory.IO

    def test_stdlib_data_integrity(self):
        assert classify_error(ValueError("bad")) is ErrorCategory.DATA_INTEGRITY
        assert classify_error(KeyError("k")) is ErrorCategory.DATA_INTEGRITY
        assert classify_error(json.JSONDecodeError("x", "doc", 0)) is ErrorCategory.DATA_INTEGRITY
        assert classify_error(UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")) is ErrorCategory.DATA_INTEGRITY

    def test_unknown(self):
        assert classify_error(RuntimeError("mystery")) is ErrorCategory.UNKNOWN
        assert classify_error(Exception("plain")) is ErrorCategory.UNKNOWN

    def test_requests_rate_limit(self):
        exc = _make("requests.exceptions", "HTTPError", status_code=429)
        assert classify_error(exc) is ErrorCategory.RESOURCE

    def test_requests_server_error(self):
        exc = _make("requests.exceptions", "HTTPError", status_code=503)
        assert classify_error(exc) is ErrorCategory.RESOURCE

    def test_requests_not_found(self):
        exc = _make("requests.exceptions", "HTTPError", status_code=404)
        assert classify_error(exc) is ErrorCategory.NOT_FOUND

    def test_requests_other_status_is_network(self):
        exc = _make("requests.exceptions", "HTTPError", status_code=200)
        assert classify_error(exc) is ErrorCategory.NETWORK

    def test_urllib_error(self):
        exc = _make("urllib.error", "HTTPError", code=500)
        assert classify_error(exc) is ErrorCategory.RESOURCE

    def test_pandas_module_prefix(self):
        exc = _make("pandas.errors", "ParserError")
        assert classify_error(exc) is ErrorCategory.DATA_INTEGRITY

    def test_pyarrow_module_prefix(self):
        exc = _make("pyarrow.lib", "ArrowInvalid")
        assert classify_error(exc) is ErrorCategory.DATA_INTEGRITY


class TestErrorSummary:
    def test_record_counts_by_category_source(self):
        s = ErrorSummary()
        s.record("NETWORK", "download")
        s.record("NETWORK", "download")
        s.record("DATA_INTEGRITY", "feature")
        assert s.total() == 3
        assert s.as_dict() == {"NETWORK": {"download": 2},
                               "DATA_INTEGRITY": {"feature": 1}}
        assert len(s) == 3

    def test_accepts_category_enum(self):
        s = ErrorSummary()
        s.record(ErrorCategory.PERMISSION, "storage")
        assert s.as_dict() == {"PERMISSION": {"storage": 1}}

    def test_record_exc_classifies(self):
        s = ErrorSummary()
        cat = s.record_exc(ConnectionError("refused"), "download")
        assert cat == "NETWORK"
        assert s.as_dict() == {"NETWORK": {"download": 1}}

    def test_merge(self):
        a, b = ErrorSummary(), ErrorSummary()
        a.record("NETWORK", "download")
        b.record("NETWORK", "download")
        b.record("IO", "storage")
        a.merge(b)
        assert a.as_dict() == {"NETWORK": {"download": 2}, "IO": {"storage": 1}}

    def test_bool_empty(self):
        assert not ErrorSummary()
        s = ErrorSummary()
        s.record("UNKNOWN", "x")
        assert s

    def test_report_lines_only_when_nonempty(self):
        assert ErrorSummary().report_lines() == []
        s = ErrorSummary()
        s.record_exc(ValueError("bad frame"), "feature")
        lines = s.report_lines()
        assert lines and "DATA_INTEGRITY" in "".join(lines)
        assert "feature" in "".join(lines)

    def test_example_captured_once(self):
        s = ErrorSummary()
        s.record_exc(ValueError("first"), "src")
        s.record_exc(ValueError("second"), "src")
        joined = "".join(s.report_lines())
        assert "first" in joined
        assert "second" not in joined


class TestLogSummary:
    def test_logs_via_logger(self, caplog):
        import logging
        logger = logging.getLogger("test_error_summary")
        s = ErrorSummary()
        s.record_exc(FileNotFoundError("nope"), "manifest")
        log_summary(s, logger, "build_features")
        text = caplog.text
        assert "build_features" in text
        assert "NOT_FOUND" in text
