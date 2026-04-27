"""Unit tests for AxonCoordinatorStore using httpx mock transport."""
from __future__ import annotations

import base64
import json
import sys
import types
from unittest.mock import MagicMock

import pytest

# ── Minimal httpx stub if not installed ────────────────────────────────────

if "httpx" not in sys.modules:
    httpx_stub = types.ModuleType("httpx")

    class _FakeResponse:
        def __init__(self, status_code=200, body=None):
            self.status_code = status_code
            self._body = body or {}

        def json(self):
            return self._body

        def raise_for_status(self):
            if self.status_code >= 400:
                raise Exception(f"HTTP {self.status_code}")

    class _FakeClient:
        def __init__(self, *a, **kw):
            self._responses: list[_FakeResponse] = []

        def _next_response(self):
            if self._responses:
                return self._responses.pop(0)
            return _FakeResponse()

        def put(self, url, **kw):
            return self._next_response()

        def get(self, url, **kw):
            return self._next_response()

        def post(self, url, **kw):
            return self._next_response()

        def delete(self, url, **kw):
            return self._next_response()

        def close(self):
            pass

    httpx_stub.Client = _FakeClient
    sys.modules["httpx"] = httpx_stub

import httpx
from axon_quic.coordinator_store import AxonCoordinatorStore


class TestAxonCoordinatorStore:
    def _store(self) -> AxonCoordinatorStore:
        return AxonCoordinatorStore(
            coordinator_url="http://coordinator:8080",
            cluster_id="cluster-test",
        )

    def test_set_encodes_base64(self, monkeypatch):
        calls = []

        def fake_put(url, json=None, **kw):
            calls.append((url, json))
            r = MagicMock()
            r.raise_for_status = MagicMock()
            return r

        store = self._store()
        monkeypatch.setattr(store._client, "put", fake_put)
        store.set("rank", b"\x00\x01\x02")
        assert len(calls) == 1
        url, body = calls[0]
        assert "/store/cluster-test/rank" in url
        assert body["value"] == base64.b64encode(b"\x00\x01\x02").decode()

    def test_get_decodes_base64(self, monkeypatch):
        encoded = base64.b64encode(b"hello").decode()

        def fake_get(url, **kw):
            r = MagicMock()
            r.status_code = 200
            r.raise_for_status = MagicMock()
            r.json = MagicMock(return_value={"value": encoded})
            return r

        store = self._store()
        monkeypatch.setattr(store._client, "get", fake_get)
        result = store.get("mykey")
        assert result == b"hello"

    def test_get_timeout_raises(self, monkeypatch):
        def fake_get(url, **kw):
            r = MagicMock()
            r.status_code = 408
            r.raise_for_status = MagicMock()
            return r

        store = self._store()
        monkeypatch.setattr(store._client, "get", fake_get)
        with pytest.raises(TimeoutError, match="mykey"):
            store.get("mykey")

    def test_add_returns_value(self, monkeypatch):
        def fake_post(url, json=None, **kw):
            r = MagicMock()
            r.status_code = 200
            r.raise_for_status = MagicMock()
            r.json = MagicMock(return_value={"value": 3})
            return r

        store = self._store()
        monkeypatch.setattr(store._client, "post", fake_post)
        result = store.add("counter", 1)
        assert result == 3

    def test_wait_calls_get_for_each_key(self, monkeypatch):
        calls = []

        def fake_get(url, **kw):
            calls.append(url)
            r = MagicMock()
            r.status_code = 200
            r.raise_for_status = MagicMock()
            r.json = MagicMock(return_value={"value": base64.b64encode(b"x").decode()})
            return r

        store = self._store()
        monkeypatch.setattr(store._client, "get", fake_get)
        store.wait(["k1", "k2"])
        assert len(calls) == 2
        assert any("k1" in u for u in calls)
        assert any("k2" in u for u in calls)

    def test_set_timeout(self):
        store = self._store()
        store.set_timeout(120.0)
        assert store._timeout == 120.0
