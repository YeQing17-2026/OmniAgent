import base64
import hashlib
import os
from typing import Tuple


def generate_pkce() -> Tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256).

    Returns:
        (verifier, challenge) tuple — both are base64url-encoded strings without padding.
    """
    verifier_bytes = os.urandom(32)
    verifier = _base64url_encode(verifier_bytes)

    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = _base64url_encode(digest)

    return verifier, challenge


def _base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()
