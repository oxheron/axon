"""Tests for default GLOO_SOCKET_IFNAME helper."""

from __future__ import annotations

import pytest

from axon_quic.gloo_socket_env import apply_default_gloo_socket_ifname, detect_default_route_interface


def test_apply_skips_when_pp_size_one() -> None:
    env: dict[str, str] = {}
    apply_default_gloo_socket_ifname(env, pp_size=1)
    assert "GLOO_SOCKET_IFNAME" not in env


def test_apply_respects_existing_gloo_socket_ifname() -> None:
    env = {"GLOO_SOCKET_IFNAME": "eth99"}
    apply_default_gloo_socket_ifname(env, pp_size=2)
    assert env["GLOO_SOCKET_IFNAME"] == "eth99"


@pytest.mark.skipif(
    detect_default_route_interface() is None,
    reason="no default route parser for this platform / environment",
)
def test_apply_sets_interface_when_pp_gt_one() -> None:
    env: dict[str, str] = {}
    apply_default_gloo_socket_ifname(env, pp_size=2)
    assert env.get("GLOO_SOCKET_IFNAME")
