import base64
import hashlib
from omniagent.auth.pkce import generate_pkce


def test_generate_pkce_returns_verifier_and_challenge():
    verifier, challenge = generate_pkce()
    assert isinstance(verifier, str)
    assert isinstance(challenge, str)


def test_verifier_is_base64url_encoded():
    verifier, _ = generate_pkce()
    assert "+" not in verifier
    assert "/" not in verifier
    assert "=" not in verifier


def test_challenge_is_sha256_of_verifier():
    verifier, challenge = generate_pkce()
    digest = hashlib.sha256(verifier.encode()).digest()
    expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    assert challenge == expected


def test_verifier_length_is_43_chars():
    verifier, _ = generate_pkce()
    assert len(verifier) == 43
