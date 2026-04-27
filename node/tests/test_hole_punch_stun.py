"""Tests for STUN discovery and symmetric NAT detection in hole_punch.py."""
from __future__ import annotations

import asyncio
from unittest.mock import patch, MagicMock

import pytest

from transport.hole_punch import discover_external_addr


@pytest.mark.asyncio
async def test_same_port_not_symmetric():
    """Both STUN servers return the same mapped port → not symmetric."""
    with patch("transport.hole_punch._run_stun_probe", return_value=("Full Cone NAT", "5.5.5.5", 54321)):
        result = await discover_external_addr(bind_port=9000)
    assert result.is_symmetric is False
    assert result.external_ip == "5.5.5.5"
    assert result.external_port == 54321


@pytest.mark.asyncio
async def test_different_ports_is_symmetric():
    """STUN servers return different mapped ports → symmetric NAT."""
    call_count = 0

    def _probe(host, port, source_port):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ("Symmetric NAT", "5.5.5.5", 54321)
        return ("Symmetric NAT", "5.5.5.5", 54678)

    with patch("transport.hole_punch._run_stun_probe", side_effect=_probe):
        result = await discover_external_addr(bind_port=9000)
    assert result.is_symmetric is True


@pytest.mark.asyncio
async def test_stun_exception_returns_symmetric():
    """If STUN throws, discover_external_addr returns is_symmetric=True (safe fallback)."""
    with patch(
        "transport.hole_punch._run_stun_probe",
        side_effect=OSError("network unreachable"),
    ):
        result = await discover_external_addr(bind_port=9000)
    assert result.is_symmetric is True


@pytest.mark.asyncio
async def test_stun_partial_failure_is_symmetric():
    """First STUN probe succeeds, second raises → treated as symmetric."""
    call_count = 0

    def _probe(host, port, source_port):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ("Full Cone NAT", "5.5.5.5", 54321)
        raise OSError("timeout")

    with patch("transport.hole_punch._run_stun_probe", side_effect=_probe):
        result = await discover_external_addr(bind_port=9000)
    assert result.is_symmetric is True
