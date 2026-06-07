#!/usr/bin/env python3
"""Open Kite login in browser, capture request_token, and save access_token."""

import argparse
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException

from env_utils import load_env_file

DEFAULT_PORT = 8765
DEFAULT_ENV_FILE = ".env"
LOGIN_TIMEOUT_SECONDS = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automate Kite Connect login and save access token."
    )
    parser.add_argument("--api-key", default=os.getenv("KITE_API_KEY"), help="Kite API key")
    parser.add_argument("--api-secret", default=os.getenv("KITE_API_SECRET"), help="Kite API secret")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("KITE_REDIRECT_PORT", DEFAULT_PORT)),
        help=f"Local callback port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--env-file",
        default=DEFAULT_ENV_FILE,
        help=f"File to update with KITE_ACCESS_TOKEN (default: {DEFAULT_ENV_FILE})",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Print login URL instead of opening browser automatically",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run browser login even if the current access token is still valid",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify existing session; do not open browser login",
    )
    return parser.parse_args()


def check_existing_session(api_key: str, access_token: str) -> tuple[bool, dict | None, str | None]:
    """Return (is_logged_in, profile_dict, error_message) via Kite profile API."""
    if not access_token or not access_token.strip():
        return False, None, "No access token in environment."

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token.strip())
    try:
        profile = kite.profile()
        return True, profile, None
    except KiteException as exc:
        return False, None, str(exc)
    except Exception as exc:
        return False, None, str(exc)


def _format_user(profile: dict) -> str:
    return profile.get("user_name") or profile.get("user_id") or "unknown"


def set_env_value(env_path: Path, key: str, value: str) -> None:
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    updated = False
    new_lines: list[str] = []
    for line in lines:
        if line.startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        new_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


class LoginCallbackHandler(BaseHTTPRequestHandler):
    request_token: Optional[str] = None
    login_status: Optional[str] = None
    done_event: threading.Event

    def log_message(self, _format: str, *_args) -> None:
        return

    def do_GET(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        LoginCallbackHandler.request_token = query.get("request_token", [None])[0]
        LoginCallbackHandler.login_status = query.get("status", [None])[0]

        if LoginCallbackHandler.login_status == "success" and LoginCallbackHandler.request_token:
            body = (
                "<html><body style='font-family:sans-serif;padding:2rem'>"
                "<h2>Login successful</h2>"
                "<p>Access token is being generated. You can close this tab.</p>"
                "</body></html>"
            ).encode("utf-8")
        else:
            body = (
                "<html><body style='font-family:sans-serif;padding:2rem'>"
                "<h2>Login failed</h2>"
                "<p>Check the terminal for details and try again.</p>"
                "</body></html>"
            ).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        LoginCallbackHandler.done_event.set()


def wait_for_request_token(port: int) -> str:
    done_event = threading.Event()
    LoginCallbackHandler.done_event = done_event
    LoginCallbackHandler.request_token = None
    LoginCallbackHandler.login_status = None

    server = HTTPServer(("127.0.0.1", port), LoginCallbackHandler)
    server.timeout = 1

    def serve() -> None:
        while not done_event.is_set():
            server.handle_request()
        server.server_close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    if not done_event.wait(timeout=LOGIN_TIMEOUT_SECONDS):
        server.server_close()
        raise TimeoutError(
            f"No login callback received within {LOGIN_TIMEOUT_SECONDS} seconds."
        )

    thread.join(timeout=2)

    if LoginCallbackHandler.login_status != "success":
        raise RuntimeError(f"Kite login failed with status={LoginCallbackHandler.login_status!r}")
    if not LoginCallbackHandler.request_token:
        raise RuntimeError("Login callback did not include request_token.")

    return LoginCallbackHandler.request_token


def main() -> None:
    args = parse_args()
    env_path = Path(args.env_file)
    load_env_file(env_path)

    api_key = args.api_key or os.getenv("KITE_API_KEY")
    api_secret = args.api_secret or os.getenv("KITE_API_SECRET")
    if not api_key:
        raise ValueError(
            "Missing API key. Set KITE_API_KEY in .env or pass --api-key."
        )

    access_token = os.getenv("KITE_ACCESS_TOKEN", "")

    if not args.force:
        ok, profile, err = check_existing_session(api_key, access_token)
        if ok and profile is not None:
            print(f"Already logged in as {_format_user(profile)} (access token is valid).")
            if args.check_only:
                return
            print("Skipping browser login. Use --force to log in again.")
            print("You can run: python scripts/build_equity_dataset.py --config config/dataset.smallcap250.json")
            return

        if args.check_only:
            print(f"Not logged in: {err or 'invalid or missing token'}")
            raise SystemExit(1)

        if access_token.strip():
            print(f"Existing token is invalid or expired: {err}")
            print("Starting fresh login...\n")
    elif args.check_only:
        raise ValueError("--check-only cannot be used with --force.")

    if not api_secret:
        raise ValueError(
            "Missing API secret. Set KITE_API_SECRET in .env or pass --api-secret."
        )

    redirect_url = f"http://127.0.0.1:{args.port}"
    login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"

    print(f"Redirect URL (must match Kite app settings): {redirect_url}")
    print("Starting local callback server...")

    result: dict[str, Optional[str]] = {"request_token": None, "error": None}

    def run_flow() -> None:
        try:
            result["request_token"] = wait_for_request_token(args.port)
        except Exception as exc:
            result["error"] = str(exc)

    flow_thread = threading.Thread(target=run_flow, daemon=True)
    flow_thread.start()

    print(f"Open this URL to login:\n{login_url}\n")
    if args.no_browser:
        print("Browser auto-open disabled (--no-browser).")
    else:
        webbrowser.open(login_url)

    print("Waiting for login callback...")
    flow_thread.join(timeout=LOGIN_TIMEOUT_SECONDS + 5)

    if result["error"]:
        raise RuntimeError(result["error"])
    if not result["request_token"]:
        raise TimeoutError(
            f"No login callback received within {LOGIN_TIMEOUT_SECONDS} seconds."
        )

    request_token = result["request_token"]
    print("Received request_token. Generating access_token...")

    kite = KiteConnect(api_key=api_key)
    session = kite.generate_session(request_token, api_secret=api_secret)
    access_token = session["access_token"]

    set_env_value(env_path, "KITE_API_KEY", api_key)
    set_env_value(env_path, "KITE_ACCESS_TOKEN", access_token)

    print(f"Saved KITE_ACCESS_TOKEN to {env_path}")
    print(f"User: {session.get('user_name', session.get('user_id', 'unknown'))}")
    print("Login complete. You can now run: python download_kite_ohlcv.py")


if __name__ == "__main__":
    main()
