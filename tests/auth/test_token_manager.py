import asyncio
import time
import pytest
from omniagent.auth.token_store import TokenStore, OAuthCredentials
from omniagent.auth.token_manager import TokenManager

PROFILE_ID = "openai-codex:default"


def _make_creds(expires_in_ms: int, access="tok-old", refresh="ref-old") -> OAuthCredentials:
    return OAuthCredentials(
        provider="openai-codex",
        access=access,
        refresh=refresh,
        expires=int(time.time() * 1000) + expires_in_ms,
        account_id="org-test",
    )


@pytest.fixture
def store(tmp_path):
    return TokenStore(tmp_path / "auth-profiles.json")


@pytest.fixture
def valid_manager(store):
    store.save(PROFILE_ID, _make_creds(expires_in_ms=3600_000))  # 1 hour from now
    return TokenManager(store, PROFILE_ID)


@pytest.fixture
def expiring_manager(store):
    store.save(PROFILE_ID, _make_creds(expires_in_ms=60_000))  # 1 min (< 5 min buffer)
    return TokenManager(store, PROFILE_ID)


def test_returns_valid_token_without_refresh(valid_manager):
    token = asyncio.run(valid_manager.get_access_token())
    assert token == "tok-old"


def test_refreshes_expiring_token(expiring_manager, store):
    new_creds = _make_creds(expires_in_ms=3600_000, access="tok-new", refresh="ref-new")

    async def fake_refresh(refresh_token: str) -> OAuthCredentials:
        return new_creds

    token = asyncio.run(expiring_manager.get_access_token(refresh_fn=fake_refresh))
    assert token == "tok-new"
    saved = store.load(PROFILE_ID)
    assert saved.access == "tok-new"


def test_raises_when_no_credentials(tmp_path):
    store = TokenStore(tmp_path / "auth-profiles.json")
    manager = TokenManager(store, PROFILE_ID)
    with pytest.raises(RuntimeError, match="No credentials found"):
        asyncio.run(manager.get_access_token())


def test_concurrent_refresh_triggers_once(expiring_manager, store):
    refresh_count = 0

    async def counting_refresh(refresh_token: str) -> OAuthCredentials:
        nonlocal refresh_count
        refresh_count += 1
        await asyncio.sleep(0.01)
        return _make_creds(expires_in_ms=3600_000, access="tok-new")

    async def run():
        tasks = [expiring_manager.get_access_token(refresh_fn=counting_refresh) for _ in range(5)]
        return await asyncio.gather(*tasks)

    asyncio.run(run())
    assert refresh_count == 1
