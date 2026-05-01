"""Spawn an isolated Python subprocess that runs gloo init via AxonCoordinatorStore.

Mirrors the vLLM EngineCore pattern (fresh interpreter, PYTHONPATH to ``node/src``)
without loading vLLM or CUDA models.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

pytest.importorskip("torch")

_NODE_SRC = Path(__file__).resolve().parents[1] / "src"


class _MiniStore:
    """In-memory store matching coordinator semantics (per cluster_id)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._clusters: dict[str, dict[str, bytes]] = {}

    def _cluster(self, cid: str) -> dict[str, bytes]:
        if cid not in self._clusters:
            self._clusters[cid] = {}
        return self._clusters[cid]

    def set(self, cid: str, key: str, raw: bytes) -> None:
        with self._cond:
            self._cluster(cid)[key] = raw
            self._cond.notify_all()

    def get(self, cid: str, key: str, timeout_ms: int) -> bytes | None:
        deadline = __import__("time").time() + timeout_ms / 1000.0
        with self._cond:
            while True:
                d = self._cluster(cid)
                if key in d:
                    return d[key]
                if __import__("time").time() >= deadline:
                    return None
                self._cond.wait(timeout=min(0.1, max(0.001, deadline - __import__("time").time())))

    def add(self, cid: str, key: str, amount: int) -> int:
        with self._cond:
            d = self._cluster(cid)
            cur = int.from_bytes(d.get(key, b"\x00" * 8), "little")
            newv = cur + amount
            d[key] = newv.to_bytes(8, "little")
            self._cond.notify_all()
            return newv

    def compare_set(self, cid: str, key: str, expected: bytes, desired: bytes) -> bytes:
        with self._cond:
            d = self._cluster(cid)
            cur = d.get(key)
            if cur is None:
                if len(expected) == 0:
                    d[key] = desired
                    self._cond.notify_all()
                    return desired
                return b""
            if len(expected) == 0 and len(cur) == 0:
                d[key] = desired
                self._cond.notify_all()
                return desired
            if cur == expected:
                d[key] = desired
                self._cond.notify_all()
                return desired
            return cur


_STORE = _MiniStore()


class _Handler(BaseHTTPRequestHandler):
    log_message = lambda self, *_args, **_kw: None  # noqa: E731

    def _send_json(self, code: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_PUT(self) -> None:
        parsed = urlparse(self.path)
        parts = parsed.path.removeprefix("/store/").split("/", 1)
        if len(parts) != 2:
            self.send_error(400)
            return
        cid, key = parts[0], parts[1]
        n = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(n) if n else b"{}"
        body = json.loads(raw_body.decode())
        val = base64.b64decode(body["value"])
        _STORE.set(cid, key, val)
        self._send_json(200, {"ok": True})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        q = parse_qs(parsed.query)
        timeout_ms = 30000
        if q.get("timeout_ms") and q["timeout_ms"][0].isdigit():
            timeout_ms = max(1, int(q["timeout_ms"][0]))
        parts = parsed.path.removeprefix("/store/").split("/", 1)
        if len(parts) != 2:
            self.send_error(400)
            return
        cid, key = parts[0], parts[1]
        val = _STORE.get(cid, key, timeout_ms)
        if val is None:
            self.send_error(408, "timeout")
            return
        self._send_json(200, {"value": base64.b64encode(val).decode()})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        tail = parsed.path.removeprefix("/store/")
        n = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(n) if n else b"{}"
        body = json.loads(raw_body.decode())
        if tail.endswith("/add"):
            base = tail[: -len("/add")]
            cid, key = base.split("/", 1)
            amt = int(body["amount"])
            v = _STORE.add(cid, key, amt)
            self._send_json(200, {"value": v})
            return
        if tail.endswith("/compare_set"):
            base = tail[: -len("/compare_set")]
            cid, key = base.split("/", 1)
            exp = base64.b64decode(body["expected"])
            des = base64.b64decode(body["desired"])
            out = _STORE.compare_set(cid, key, exp, des)
            self._send_json(200, {"value": base64.b64encode(out).decode()})
            return
        self.send_error(400)


@pytest.fixture
def coordinator_url() -> str:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()


def test_subprocess_init_process_group(coordinator_url: str) -> None:
    """Fresh subprocess: AxonCoordinatorStore + gloo init (rank 0, world 1)."""
    child = r"""
import os
import torch.distributed as dist
from axon_quic.coordinator_store import AxonCoordinatorStore

store = AxonCoordinatorStore(
    coordinator_url=os.environ["AXON_COORDINATOR_URL"],
    cluster_id=os.environ["AXON_CLUSTER_ID"],
)
dist.init_process_group(
    backend="gloo",
    store=store,
    rank=int(os.environ["AXON_PP_RANK"]),
    world_size=int(os.environ["AXON_PP_SIZE"]),
)
dist.destroy_process_group()
print("axon_subprocess_init_ok")
"""
    env = {
        **os.environ,
        "PYTHONPATH": str(_NODE_SRC) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "AXON_COORDINATOR_URL": coordinator_url,
        "AXON_CLUSTER_ID": "pytest-pp",
        "AXON_PP_RANK": "0",
        "AXON_PP_SIZE": "1",
    }
    proc = subprocess.run(
        [sys.executable, "-c", child],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert "axon_subprocess_init_ok" in proc.stdout
