"""Public error shaping for gateway responses."""

import json
import re
from typing import Any

from aiohttp import web

PUBLIC_ERROR_FALLBACK = "Something went wrong. Please try again."

_AUTHORIZATION_HEADER_PATTERN = re.compile(
    r"\bAuthorization\s*[:=]\s*(Bearer\s+)?[^\s,}]+",
    re.IGNORECASE,
)
_BEARER_TOKEN_PATTERN = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_OPENAI_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b")
_PRIVATE_KEY_VALUE_PATTERN = re.compile(
    r"\b(api[_-]?key|token|secret|password)\s*[:=]\s*([\"']?)[^\"'\s,}]+",
    re.IGNORECASE,
)


def _extract_serialized_error(message: str) -> str | None:
    trimmed = message.strip()
    if not trimmed.startswith(("{", "[")):
        return None

    try:
        parsed = json.loads(trimmed)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, str) and parsed.strip():
        return parsed

    if not isinstance(parsed, dict):
        return PUBLIC_ERROR_FALLBACK

    for key in ("error", "message"):
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            return value

    return PUBLIC_ERROR_FALLBACK


def _replace_product_name(message: str) -> str:
    return re.sub(r"\bOmniAgent\b|\bOmniagent\b", "Vercel Agent", message)


def _redact_private_values(message: str) -> str:
    redacted = message
    redacted = _AUTHORIZATION_HEADER_PATTERN.sub(
        lambda m: "Authorization: Bearer ****" if m.group(1) else "Authorization: ****",
        redacted,
    )
    redacted = _BEARER_TOKEN_PATTERN.sub("Bearer ****", redacted)
    redacted = _OPENAI_KEY_PATTERN.sub("sk-****", redacted)
    redacted = _PRIVATE_KEY_VALUE_PATTERN.sub(
        lambda m: f"{m.group(1)}={m.group(2)}****",
        redacted,
    )
    return redacted


def public_error_message(
    error: Any,
    *,
    fallback: str = PUBLIC_ERROR_FALLBACK,
    prefix: str | None = None,
) -> str:
    """Return a stable, user-safe error message for external interfaces."""
    if error is None:
        message = fallback
    elif isinstance(error, str):
        message = error
    else:
        message = str(error)

    message = _extract_serialized_error(message) or message
    message = message.strip() or fallback

    if "Type validation failed" in message or "invalid_union" in message:
        message = fallback

    message = _replace_product_name(message)
    message = _redact_private_values(message)

    if len(message) > 240:
        message = f"{message[:237].rstrip()}\u2026"

    if prefix:
        return f"{prefix}: {message}"
    return message


def public_json_error(
    error: Any,
    *,
    status: int = 500,
    fallback: str = PUBLIC_ERROR_FALLBACK,
    prefix: str | None = None,
) -> web.Response:
    """Return a JSON error response without leaking raw exception details."""
    return web.json_response(
        {"error": public_error_message(error, fallback=fallback, prefix=prefix)},
        status=status,
    )
