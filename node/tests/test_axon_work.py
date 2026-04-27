"""Unit tests for axon_quic.work.AxonWork."""
from __future__ import annotations

import pytest

from axon_quic.work import AxonWork


class TestAxonWork:
    def test_is_completed(self):
        w = AxonWork()
        assert w.is_completed() is True

    def test_is_success(self):
        assert AxonWork().is_success() is True
        assert AxonWork(exc=ValueError("boom")).is_success() is False

    def test_wait_no_error(self):
        w = AxonWork()
        assert w.wait() is True

    def test_wait_reraises(self):
        exc = ConnectionError("peer timeout")
        w = AxonWork(exc=exc)
        with pytest.raises(ConnectionError, match="peer timeout"):
            w.wait()

    def test_exception_accessor(self):
        exc = RuntimeError("test")
        w = AxonWork(exc=exc)
        assert w.exception() is exc

    def test_get_future_resolved(self):
        import concurrent.futures
        fut = AxonWork().get_future()
        assert fut.done()
        assert fut.result() is None

    def test_get_future_with_exception(self):
        exc = ConnectionError("gone")
        fut = AxonWork(exc=exc).get_future()
        assert fut.done()
        with pytest.raises(ConnectionError):
            fut.result()

    def test_synchronize_no_error(self):
        AxonWork().synchronize()  # should not raise

    def test_synchronize_reraises(self):
        with pytest.raises(ValueError):
            AxonWork(exc=ValueError("x")).synchronize()
