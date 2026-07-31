import json
import time
import pytest
from pathlib import Path
from omniagent.auth.token_store import TokenStore, OAuthCredentials


@pytest.fixture
def tmp_store(tmp_path):
    return TokenStore(tmp_path / "auth-profiles.json")


def test_save_and_load_credentials(tmp_store):
    creds = OAuthCredentials(
        provider="openai-codex",
        access="access-token-123",
        refresh="refresh-token-456",
        expires=int(time.time() * 1000) + 3600_000,
        account_id="org-abc",
    )
    tmp_store.save("openai-codex:default", creds)
    loaded = tmp_store.load("openai-codex:default")
    assert loaded is not None
    assert loaded.access == "access-token-123"
    assert loaded.refresh == "refresh-token-456"
    assert loaded.account_id == "org-abc"


def test_load_nonexistent_returns_none(tmp_store):
    result = tmp_store.load("openai-codex:default")
    assert result is None


def test_delete_credentials(tmp_store):
    creds = OAuthCredentials(
        provider="openai-codex",
        access="tok",
        refresh="ref",
        expires=9999999999999,
        account_id="org-x",
    )
    tmp_store.save("openai-codex:default", creds)
    tmp_store.delete("openai-codex:default")
    assert tmp_store.load("openai-codex:default") is None


def test_json_file_structure(tmp_store, tmp_path):
    creds = OAuthCredentials(
        provider="openai-codex",
        access="a",
        refresh="r",
        expires=1000,
        account_id="org-y",
    )
    tmp_store.save("openai-codex:default", creds)
    raw = json.loads((tmp_path / "auth-profiles.json").read_text())
    assert "profiles" in raw
    assert "openai-codex:default" in raw["profiles"]
