"""Kite authentication checks for Connect env vars and Cursor MCP."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

MCP_URL = os.environ.get("KITE_MCP_URL", "https://mcp.kite.trade/mcp")

AuthStatus = dict[str, Any]


def normalize_mcp_session_id(session_id: str) -> str:
    """Return the MCP header session id (UUID only).

    The Kite login URL embeds ``kitemcp-<uuid>|<signature>`` in redirect_params.
    That full string is for OAuth only — HTTP requests must send just ``kitemcp-<uuid>``.
    """
    sid = session_id.strip()
    if "|" in sid:
        sid = sid.split("|", 1)[0]
    return sid


def _status(ok: bool, message: str, **extra: Any) -> AuthStatus:
    return {"ok": ok, "message": message, **extra}


def check_kite_env_vars() -> AuthStatus:
    """Return whether KITE_API_KEY and KITE_ACCESS_TOKEN are set."""
    api_key = os.environ.get("KITE_API_KEY", "").strip()
    access_token = os.environ.get("KITE_ACCESS_TOKEN", "").strip()
    api_key_set = bool(api_key)
    access_token_set = bool(access_token)

    if api_key_set and access_token_set:
        return _status(
            True,
            "KITE_API_KEY and KITE_ACCESS_TOKEN are set.",
            api_key_set=True,
            access_token_set=True,
        )

    missing = []
    if not api_key_set:
        missing.append("KITE_API_KEY")
    if not access_token_set:
        missing.append("KITE_ACCESS_TOKEN")
    return _status(
        False,
        f"Missing env var(s): {', '.join(missing)}.",
        api_key_set=api_key_set,
        access_token_set=access_token_set,
    )


def check_kite_connect(*, validate_api: bool = True) -> AuthStatus:
    """Check Kite Connect credentials; optionally validate with profile API."""
    env = check_kite_env_vars()
    if not env["ok"]:
        return _status(
            False,
            env["message"],
            api_key_set=env.get("api_key_set", False),
            access_token_set=env.get("access_token_set", False),
        )

    try:
        from kiteconnect import KiteConnect
    except ImportError:
        return _status(
            False,
            "kiteconnect is not installed.",
            api_key_set=True,
            access_token_set=True,
        )

    api_key = os.environ["KITE_API_KEY"].strip()
    access_token = os.environ["KITE_ACCESS_TOKEN"].strip()
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    if not validate_api:
        return _status(
            True,
            "Env vars present (API not validated).",
            api_key_set=True,
            access_token_set=True,
        )

    try:
        profile = kite.profile()
    except Exception as exc:  # noqa: BLE001
        return _status(
            False,
            f"Kite Connect token invalid or expired: {exc}",
            api_key_set=True,
            access_token_set=True,
        )

    user_id = profile.get("user_id")
    user_name = profile.get("user_name")
    return _status(
        True,
        f"Logged in as {user_name} ({user_id}).",
        api_key_set=True,
        access_token_set=True,
        user_id=user_id,
        user_name=user_name,
    )


def _mcp_call(session_id: str, method: str, params: dict) -> dict:
    sid = normalize_mcp_session_id(session_id)
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = urllib.request.Request(
        MCP_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "mcp-session-id": sid,
            "User-Agent": "nifty-smallcap-trading-bot/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read().decode()
    if raw.startswith("event:") or "data:" in raw[:200]:
        for line in raw.splitlines():
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload:
                    return json.loads(payload)
        raise RuntimeError(f"No JSON in SSE response: {raw[:500]}")
    return json.loads(raw)


def _parse_mcp_tool_result(msg: dict) -> dict | list | str:
    if "error" in msg:
        err = msg["error"]
        if isinstance(err, dict):
            raise RuntimeError(err.get("message", str(err)))
        raise RuntimeError(str(err))
    result = msg.get("result", msg)
    if result.get("isError"):
        content = result.get("content", [])
        detail = content[0].get("text", "MCP tool error") if content else "MCP tool error"
        raise RuntimeError(detail)
    content = result.get("content", [])
    if content and isinstance(content[0], dict) and "text" in content[0]:
        text = content[0]["text"]
        if isinstance(text, str):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        return text
    return result


def check_kite_mcp_logged_in(session_id: str | None = None) -> AuthStatus:
    """Check Kite MCP login via get_profile (Cursor agent or KITE_MCP_SESSION_ID)."""
    sid = (session_id or os.environ.get("KITE_MCP_SESSION_ID") or "").strip()
    if not sid:
        return _status(
            False,
            "KITE_MCP_SESSION_ID not set. In Cursor, call user-kite get_profile to verify MCP login.",
            skipped=True,
        )

    try:
        msg = _mcp_call(
            sid,
            "tools/call",
            {"name": "get_profile", "arguments": {}},
        )
        profile = _parse_mcp_tool_result(msg)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace").strip()
        hint = ""
        if exc.code == 400 and "Invalid session ID" in detail:
            hint = " Use the UUID-only MCP session id (not the uuid|signature from the login URL)."
        elif exc.code in (400, 403):
            hint = " Re-authenticate via Kite MCP in Cursor (get_profile) or refresh KITE_MCP_SESSION_ID."
        return _status(
            False,
            f"MCP HTTP {exc.code}: session missing or expired.{hint}",
            skipped=False,
        )
    except Exception as exc:  # noqa: BLE001
        return _status(False, f"MCP get_profile failed: {exc}", skipped=False)

    if not isinstance(profile, dict):
        return _status(False, f"Unexpected MCP profile response: {profile!r}", skipped=False)

    user_id = profile.get("user_id")
    user_name = profile.get("user_name")
    if user_id or user_name:
        label = user_name or user_id
        suffix = f" ({user_id})" if user_name and user_id else ""
        return _status(
            True,
            f"MCP logged in as {label}{suffix}.",
            skipped=False,
            user_id=user_id,
            user_name=user_name,
        )

    return _status(False, "MCP get_profile returned no user info.", skipped=False)


def check_kite_auth(*, validate_connect: bool = True, check_mcp: bool = True) -> AuthStatus:
    """Aggregate Kite auth status for env, Connect API, and MCP."""
    env = check_kite_env_vars()
    connect = check_kite_connect(validate_api=validate_connect and env["ok"])
    mcp: AuthStatus = (
        check_kite_mcp_logged_in()
        if check_mcp
        else _status(False, "MCP check skipped.", skipped=True)
    )

    ok = connect["ok"] or mcp["ok"]
    if connect["ok"] and mcp["ok"]:
        summary = "Kite Connect and MCP are both authenticated."
    elif connect["ok"]:
        summary = "Kite Connect is authenticated."
    elif mcp["ok"]:
        summary = "Kite MCP is authenticated (Connect env vars not usable)."
    else:
        summary = "Not logged in via Kite Connect or MCP."

    return {
        "ok": ok,
        "message": summary,
        "env": env,
        "connect": connect,
        "mcp": mcp,
    }
