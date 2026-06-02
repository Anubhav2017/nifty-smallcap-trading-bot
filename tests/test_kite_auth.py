"""Tests for Kite auth env-var checks (no live API)."""

from __future__ import annotations

import pytest

from trading_bot.data.kite_auth import (
    check_kite_auth,
    check_kite_connect,
    check_kite_env_vars,
    normalize_mcp_session_id,
)


def test_normalize_mcp_session_id_strips_signature() -> None:
    full = "kitemcp-abc-123|deadbeef"
    assert normalize_mcp_session_id(full) == "kitemcp-abc-123"
    assert normalize_mcp_session_id("  kitemcp-abc-123  ") == "kitemcp-abc-123"


def test_env_vars_both_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KITE_API_KEY", "test_key")
    monkeypatch.setenv("KITE_ACCESS_TOKEN", "test_token")
    result = check_kite_env_vars()
    assert result["ok"] is True
    assert result["api_key_set"] is True
    assert result["access_token_set"] is True


def test_env_vars_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KITE_API_KEY", raising=False)
    monkeypatch.setenv("KITE_ACCESS_TOKEN", "test_token")
    result = check_kite_env_vars()
    assert result["ok"] is False
    assert "KITE_API_KEY" in result["message"]
    assert result["api_key_set"] is False
    assert result["access_token_set"] is True


def test_env_vars_missing_access_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KITE_API_KEY", "test_key")
    monkeypatch.delenv("KITE_ACCESS_TOKEN", raising=False)
    result = check_kite_env_vars()
    assert result["ok"] is False
    assert "KITE_ACCESS_TOKEN" in result["message"]


def test_env_vars_empty_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KITE_API_KEY", "   ")
    monkeypatch.setenv("KITE_ACCESS_TOKEN", "")
    result = check_kite_env_vars()
    assert result["ok"] is False


def test_connect_skips_api_when_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KITE_API_KEY", raising=False)
    monkeypatch.delenv("KITE_ACCESS_TOKEN", raising=False)
    result = check_kite_connect(validate_api=True)
    assert result["ok"] is False
    assert "Missing env var" in result["message"]


def test_check_kite_auth_mcp_skipped_without_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KITE_API_KEY", raising=False)
    monkeypatch.delenv("KITE_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("KITE_MCP_SESSION_ID", raising=False)
    result = check_kite_auth(validate_connect=False, check_mcp=True)
    assert result["ok"] is False
    assert result["env"]["ok"] is False
    assert result["connect"]["ok"] is False
    assert result["mcp"]["ok"] is False
    assert result["mcp"].get("skipped") is True
