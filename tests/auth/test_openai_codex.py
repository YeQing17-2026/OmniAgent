import asyncio
import base64
import json
import time
from unittest.mock import AsyncMock, patch
from omniagent.auth.openai_codex import (
    build_authorization_url,
    exchange_code,
    get_account_id,
    CLIENT_ID,
)


def test_build_authorization_url_contains_required_params():
    _, url, state = asyncio.run(
        build_authorization_url()
    )
    assert "code_challenge" in url
    assert "code_challenge_method=S256" in url
    assert f"client_id={CLIENT_ID}" in url
    assert f"state={state}" in url
    assert "redirect_uri" in url


def test_get_account_id_extracts_from_jwt():
    claim_path = "https://api.openai.com/auth"
    payload = {claim_path: {"chatgpt_account_id": "org-test123"}}
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload).encode()
    ).decode().rstrip("=")
    fake_jwt = f"header.{encoded}.sig"
    assert get_account_id(fake_jwt) == "org-test123"


def test_get_account_id_returns_none_on_invalid_jwt():
    assert get_account_id("not.a.jwt") is None
    assert get_account_id("") is None


def test_exchange_code_makes_post_request():
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        "access_token": "access-abc",
        "refresh_token": "refresh-xyz",
        "expires_in": 3600,
    })

    async def _run():
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aexit__ = AsyncMock(return_value=False)
            return await exchange_code("auth-code-123", "verifier-abc")

    creds = asyncio.run(_run())
    assert creds.access == "access-abc"
    assert creds.refresh == "refresh-xyz"
    assert creds.expires > int(time.time() * 1000)
