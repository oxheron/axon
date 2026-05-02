"""Helpers for Gloo TCP device selection in multi-node pipeline parallelism.

When ``GLOO_SOCKET_IFNAME`` is unset, PyTorch builds Gloo devices from the
hostname via ``getaddrinfo(AF_UNSPEC)``. Different hosts can then publish IPv4 vs
IPv6 first, and Gloo fails during mesh setup with::

    RuntimeError: ... ss1.ss_family == ss2.ss_family. 2 vs 10

Binding to the default-route interface avoids that mismatch on typical setups.
"""
from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
from typing import Optional

LOGGER = logging.getLogger(__name__)


def detect_default_route_interface() -> Optional[str]:
    """Best-effort outbound interface for Gloo (Linux / macOS)."""
    system = platform.system()
    if system == "Linux":
        try:
            with open("/proc/net/route", encoding="utf-8") as f:
                next(f)
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2 and parts[1] == "00000000":
                        return parts[0]
        except OSError:
            pass
    elif system == "Darwin":
        try:
            proc = subprocess.run(
                ["route", "-n", "get", "default"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            for line in (proc.stdout or "").splitlines():
                m = re.match(r"\s*interface:\s*(\S+)", line)
                if m:
                    return m.group(1)
        except (OSError, subprocess.TimeoutExpired):
            pass
    return None


def apply_default_gloo_socket_ifname(
    env: Optional[dict[str, str]] = None,
    *,
    pp_size: Optional[int] = None,
) -> None:
    """If unset, set ``GLOO_SOCKET_IFNAME`` to the default-route interface.

    No-op when ``GLOO_SOCKET_IFNAME`` is already set or ``pp_size <= 1``.
    """
    target = env if env is not None else os.environ
    if target.get("GLOO_SOCKET_IFNAME"):
        return

    size = pp_size
    if size is None:
        raw = target.get("AXON_PP_SIZE", os.environ.get("AXON_PP_SIZE", "1"))
        try:
            size = int(raw)
        except ValueError:
            size = 1
    if size <= 1:
        return

    iface = detect_default_route_interface()
    if iface:
        target["GLOO_SOCKET_IFNAME"] = iface
        LOGGER.info(
            "[axon_quic] set GLOO_SOCKET_IFNAME=%s (avoid Gloo IPv4/IPv6 mismatch across nodes)",
            iface,
        )
    else:
        LOGGER.warning(
            "[axon_quic] could not detect default network interface for Gloo; "
            "if init fails with ss_family 2 vs 10, set GLOO_SOCKET_IFNAME to your "
            "outbound interface (e.g. eth0 on Linux, en0 on macOS)"
        )
