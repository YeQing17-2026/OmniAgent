import asyncio
import time
from typing import Awaitable, Callable, Optional

from omniagent.auth.token_store import OAuthCredentials, TokenStore

_REFRESH_BUFFER_MS = 5 * 60 * 1000  # refresh 5 minutes before expiry


class TokenManager:
    """Provides a valid access token, refreshing it when it is about to expire.

    Uses asyncio.Lock so that concurrent callers trigger only one refresh.
    """

    def __init__(self, store: TokenStore, profile_id: str):
        self._store = store
        self._profile_id = profile_id
        self._lock = asyncio.Lock()

    async def get_access_token(
        self,
        refresh_fn: Optional[Callable[[str], Awaitable[OAuthCredentials]]] = None,
    ) -> str:
        creds = self._store.load(self._profile_id)
        if creds is None:
            raise RuntimeError(
                f"No credentials found for profile '{self._profile_id}'. "
                "Run: omniagent auth login --provider openai-codex"
            )

        if self._is_fresh(creds):
            return creds.access

        async with self._lock:
            # Re-check after acquiring lock — another coroutine may have refreshed already
            creds = self._store.load(self._profile_id)
            if creds is None:
                raise RuntimeError(
                    f"No credentials found for profile '{self._profile_id}'. "
                    "Run: omniagent auth login --provider openai-codex"
                )
            if self._is_fresh(creds):
                return creds.access

            if refresh_fn is None:
                from omniagent.auth.openai_codex import refresh_openai_codex_token
                refresh_fn = refresh_openai_codex_token

            assert refresh_fn is not None
            new_creds = await refresh_fn(creds.refresh)
            # Preserve account_id if the refresh token response omitted it
            if not new_creds.account_id and creds.account_id:
                new_creds.account_id = creds.account_id
            self._store.save(self._profile_id, new_creds)
            return new_creds.access

    @staticmethod
    def _is_fresh(creds: OAuthCredentials) -> bool:
        return int(time.time() * 1000) < creds.expires - _REFRESH_BUFFER_MS
