"""OAuth 2.0 + PKCE login flow for OpenAI Codex (ChatGPT subscription)."""

import asyncio
import base64
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse

import aiohttp

from omniagent.auth.pkce import generate_pkce
from omniagent.auth.token_store import OAuthCredentials

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
SCOPE = "openid profile email offline_access"
_JWT_CLAIM_PATH = "https://api.openai.com/auth"
_CALLBACK_PORT = 1455


async def build_authorization_url() -> Tuple[str, str, str]:
    """Build the OAuth authorization URL with PKCE.

    Returns:
        (verifier, url, state) tuple.
    """
    verifier, challenge = generate_pkce()
    state = os.urandom(16).hex()

    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    url = f"{AUTHORIZE_URL}?{urlencode(params)}"
    return verifier, url, state


async def exchange_code(code: str, verifier: str) -> OAuthCredentials:
    """Exchange an authorization code for access + refresh tokens."""
    payload = {
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "code": code,
        "code_verifier": verifier,
        "redirect_uri": REDIRECT_URI,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            TOKEN_URL,
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Token exchange failed ({resp.status}): {text}")
            data = await resp.json()

    return OAuthCredentials(
        provider="openai-codex",
        access=data["access_token"],
        refresh=data["refresh_token"],
        expires=int(time.time() * 1000) + data.get("expires_in", 3600) * 1000,
        account_id=get_account_id(data["access_token"]) or "",
    )


async def refresh_openai_codex_token(refresh_token: str) -> OAuthCredentials:
    """Obtain a new access token using a refresh token."""
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": CLIENT_ID,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            TOKEN_URL,
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Token refresh failed ({resp.status}): {text}")
            data = await resp.json()

    return OAuthCredentials(
        provider="openai-codex",
        access=data["access_token"],
        refresh=data["refresh_token"],
        expires=int(time.time() * 1000) + data.get("expires_in", 3600) * 1000,
        account_id=get_account_id(data["access_token"]) or "",
    )


def get_account_id(access_token: str) -> Optional[str]:
    """Decode the JWT access_token and extract the ChatGPT account ID."""
    try:
        parts = access_token.split(".")
        if len(parts) < 2:
            return None
        padding = 4 - len(parts[1]) % 4
        padded = parts[1] + "=" * (padding % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode())
        return payload.get(_JWT_CLAIM_PATH, {}).get("chatgpt_account_id")
    except Exception:
        return None


async def login_openai_codex(on_open_url: Optional[Callable[[str], None]] = None) -> OAuthCredentials:
    """Run the full OAuth login flow and return credentials.

    Starts a local HTTP server on port 1455, opens the authorization URL,
    waits for the callback, and exchanges the code for tokens.
    """
    verifier, url, state = await build_authorization_url()

    if on_open_url:
        on_open_url(url)

    code = await _wait_for_callback(state)
    if not code:
        raise RuntimeError(
            "Did not receive authorization code within 60 seconds. "
            "Make sure you completed the login in the browser."
        )

    return await exchange_code(code, verifier)


async def _wait_for_callback(expected_state: str, timeout: int = 60) -> Optional[str]:
    """Start a local HTTP server and wait for the OAuth callback.

    Returns the authorization code, or None on timeout.
    Raises RuntimeError if port 1455 is already in use.
    """
    received_code: Optional[str] = None

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            nonlocal received_code
            parsed = urlparse(self.path)
            if parsed.path != "/auth/callback":
                self.send_response(404)
                self.end_headers()
                return
            params = parse_qs(parsed.query)
            state = params.get("state", [None])[0]
            if state != expected_state:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"State mismatch")
                return
            code = params.get("code", [None])[0]
            if not code:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing code")
                return
            received_code = code
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"<p>Authorization successful. You can close this tab.</p>"
            )

        def log_message(self, format, *args):
            pass

    loop = asyncio.get_running_loop()

    def _serve():
        try:
            server = HTTPServer(("127.0.0.1", _CALLBACK_PORT), Handler)
        except OSError as e:
            raise RuntimeError(
                f"Cannot bind to port {_CALLBACK_PORT}: {e}. "
                "Free the port and try again."
            ) from e
        server.timeout = 1
        deadline = time.time() + timeout
        while received_code is None and time.time() < deadline:
            server.handle_request()
        server.server_close()

    await loop.run_in_executor(None, _serve)
    return received_code
