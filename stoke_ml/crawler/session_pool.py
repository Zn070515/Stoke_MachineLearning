"""Session pool with error scoring and automatic retirement.

Each session simulates a distinct user with its own cookie jar,
header set, and usage history. The pool manages lifecycle:
creation, random selection, error- or usage-based retirement,
and automatic replenishment.
"""
import time
import random
from typing import List
from curl_cffi import requests

BLOCKED_STATUS_CODES = {401, 403, 429}


class Session:
    """A single user session with usage and error tracking."""

    def __init__(
        self,
        max_age_seconds: float = 1800,
        max_usage: int = 30,
        max_error_score: float = 3.0,
        error_score_decrement: float = 0.5,
        impersonate: str = "chrome120",
    ):
        self._created_at = time.time()
        self._max_age = max_age_seconds
        self._max_usage = max_usage
        self._max_error_score = max_error_score
        self._error_score_decrement = error_score_decrement
        self._usage_count = 0
        self._error_score = 0.0
        self._http = requests.Session(impersonate=impersonate)

    @property
    def usage_count(self) -> int:
        return self._usage_count

    @property
    def error_score(self) -> float:
        return self._error_score

    @property
    def http(self):
        return self._http

    def is_usable(self) -> bool:
        if time.time() - self._created_at > self._max_age:
            return False
        if self._usage_count >= self._max_usage:
            return False
        if self._error_score >= self._max_error_score:
            return False
        return True

    def mark_used(self):
        self._usage_count += 1

    def mark_good(self):
        self._error_score = max(0.0, self._error_score - self._error_score_decrement)

    def mark_bad(self):
        self._error_score += 1.0

    def retire(self):
        self._usage_count = self._max_usage
        self._http.close()

    def close(self):
        self._http.close()


class SessionPool:
    """Pool of user sessions with lifecycle management."""

    def __init__(
        self,
        max_sessions: int = 50,
        max_age_seconds: float = 1800,
        max_usage: int = 30,
        max_error_score: float = 3.0,
        error_score_decrement: float = 0.5,
        impersonate: str = "chrome120",
    ):
        self._max_sessions = max_sessions
        self._session_params = {
            "max_age_seconds": max_age_seconds,
            "max_usage": max_usage,
            "max_error_score": max_error_score,
            "error_score_decrement": error_score_decrement,
            "impersonate": impersonate,
        }
        self._sessions: List[Session] = []
        self._fill_initial()

    def _fill_initial(self):
        for _ in range(self._max_sessions):
            self._sessions.append(Session(**self._session_params))

    def get_session(self) -> Session:
        usable = [s for s in self._sessions if s.is_usable()]
        if not usable:
            self.replenish()
            usable = [s for s in self._sessions if s.is_usable()]
            if not usable:
                s = Session(**self._session_params)
                self._sessions.append(s)
                return s
        return random.choice(usable)

    def replenish(self):
        kept, retired = [], []
        for s in self._sessions:
            if s.is_usable():
                kept.append(s)
            else:
                retired.append(s)
        for s in retired:
            s.close()
        self._sessions = kept
        while len(self._sessions) < self._max_sessions:
            self._sessions.append(Session(**self._session_params))

    def active_count(self) -> int:
        return len([s for s in self._sessions if s.is_usable()])

    def close_all(self):
        for s in self._sessions:
            s.close()
        self._sessions.clear()
