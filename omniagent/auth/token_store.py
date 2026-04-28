import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class OAuthCredentials:
    """OAuth credentials for a single profile."""

    provider: str
    access: str
    refresh: str
    expires: int      # Unix timestamp in milliseconds
    account_id: str


class TokenStore:
    """Reads and writes OAuth credentials to a JSON file.

    File schema:
    {
      "profiles": {
        "<profile_id>": { "type": "oauth", "provider": ..., "access": ..., ... }
      }
    }
    """

    def __init__(self, path: Path):
        self._path = path

    def load(self, profile_id: str) -> Optional[OAuthCredentials]:
        """Return credentials for profile_id, or None if not found."""
        data = self._read()
        raw = data.get("profiles", {}).get(profile_id)
        if raw is None:
            return None
        return OAuthCredentials(
            provider=raw["provider"],
            access=raw["access"],
            refresh=raw["refresh"],
            expires=raw["expires"],
            account_id=raw.get("account_id", ""),
        )

    def save(self, profile_id: str, creds: OAuthCredentials) -> None:
        """Persist credentials under profile_id."""
        data = self._read()
        if "profiles" not in data:
            data["profiles"] = {}
        data["profiles"][profile_id] = {
            "type": "oauth",
            **asdict(creds),
        }
        self._write(data)

    def delete(self, profile_id: str) -> None:
        """Remove credentials for profile_id."""
        data = self._read()
        data.get("profiles", {}).pop(profile_id, None)
        self._write(data)

    def _read(self) -> dict:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _write(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
