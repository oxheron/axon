"""AxonCoordinatorStore — torch.distributed.Store backed by coordinator HTTP.

Used to initialize torch.distributed.init_process_group() across two nodes
without requiring direct TCP connectivity between them. Both nodes reach the
coordinator via the existing TCP connection established during registration.

The coordinator must be running with the /store/{cluster_id}/{key} endpoint
(coordinator/internal/server/store.go).
"""
from __future__ import annotations

import base64
import logging
from datetime import timedelta
from typing import Union

import httpx

LOGGER = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 60.0  # seconds for get/wait operations


class AxonCoordinatorStore:
    """torch.distributed.Store-compatible KV store over coordinator HTTP."""

    def __init__(
        self,
        coordinator_url: str,
        cluster_id: str,
        default_timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._base = coordinator_url.rstrip("/")
        self._cluster_id = cluster_id
        self._timeout = default_timeout
        self._client = httpx.Client(timeout=default_timeout + 5.0)

    def _url(self, key: str) -> str:
        return f"{self._base}/store/{self._cluster_id}/{key}"

    # ── torch.distributed.Store interface ──────────────────────────────

    def set(self, key: str, value: Union[bytes, str]) -> None:
        if isinstance(value, str):
            value = value.encode()
        encoded = base64.b64encode(value).decode()
        resp = self._client.put(self._url(key), json={"value": encoded})
        resp.raise_for_status()
        LOGGER.debug("[store] set key=%s", key)

    def get(self, key: str) -> bytes:
        timeout_ms = int(self._timeout * 1000)
        resp = self._client.get(
            self._url(key),
            params={"timeout_ms": timeout_ms},
            timeout=self._timeout + 5.0,
        )
        if resp.status_code == 408:
            raise TimeoutError(f"AxonCoordinatorStore: key '{key}' not found within {self._timeout}s")
        resp.raise_for_status()
        encoded = resp.json()["value"]
        return base64.b64decode(encoded)

    def wait(
        self,
        keys: list[str],
        timeout: Union[timedelta, float, None] = None,
    ) -> None:
        timeout_secs = self._timeout
        if isinstance(timeout, timedelta):
            timeout_secs = timeout.total_seconds()
        elif isinstance(timeout, (int, float)):
            timeout_secs = float(timeout)
        for key in keys:
            timeout_ms = int(timeout_secs * 1000)
            resp = self._client.get(
                self._url(key),
                params={"timeout_ms": timeout_ms},
                timeout=timeout_secs + 5.0,
            )
            if resp.status_code == 408:
                raise TimeoutError(f"AxonCoordinatorStore: timeout waiting for key '{key}'")
            resp.raise_for_status()

    def add(self, key: str, amount: int) -> int:
        resp = self._client.post(
            f"{self._url(key)}/add",
            json={"amount": amount},
        )
        resp.raise_for_status()
        return int(resp.json()["value"])

    def delete_key(self, key: str) -> bool:
        try:
            resp = self._client.delete(self._url(key))
            return resp.status_code == 200
        except Exception:
            return False

    def num_keys(self) -> int:
        # No direct endpoint; return 0 as a safe stub.
        return 0

    def set_timeout(self, timeout: Union[timedelta, float]) -> None:
        if isinstance(timeout, timedelta):
            self._timeout = timeout.total_seconds()
        else:
            self._timeout = float(timeout)
        self._client.timeout = self._timeout + 5.0

    def close(self) -> None:
        self._client.close()

    def __del__(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
