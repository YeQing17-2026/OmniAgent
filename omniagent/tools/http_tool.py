"""HTTP tool for making arbitrary HTTP requests."""

import ipaddress
import socket
import aiohttp
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from .base import Tool, ToolResult


# Private/internal IP ranges that should be blocked
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def _is_private_url(url: str) -> bool:
    """Check if a URL points to a private/internal network (SSRF prevention)."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return False

        # Resolve hostname to IP
        try:
            ip = socket.gethostbyname(hostname)
            ip_obj = ipaddress.ip_address(ip)
            return any(ip_obj in net for net in _BLOCKED_NETWORKS)
        except (socket.gaierror, ValueError):
            return False
    except Exception:
        return False


class HttpTool(Tool):
    """Make HTTP requests."""

    def __init__(self, work_dir=None):
        super().__init__(
            name="http",
            description="Make HTTP requests (GET, POST, PUT, DELETE). Basic SSRF protection included.",
        )
        self.work_dir = work_dir

    def _get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to send the request to",
                },
                "method": {
                    "type": "string",
                    "description": "HTTP method (default: GET)",
                    "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
                },
                "headers": {
                    "type": "object",
                    "description": "Request headers as key-value pairs",
                },
                "body": {
                    "type": "string",
                    "description": "Request body (for POST/PUT/PATCH)",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Request timeout in seconds (default: 30)",
                },
            },
            "required": ["url"],
        }

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        url = params.get("url", "")
        method = params.get("method", "GET").upper()
        headers = params.get("headers", {})
        body = params.get("body", "")
        timeout = params.get("timeout", 30)

        if not url:
            return ToolResult(success=False, output="", error="Missing required parameter: url")

        # Basic URL validation
        if not url.startswith(("http://", "https://")):
            return ToolResult(success=False, output="", error="URL must start with http:// or https://")

        # SSRF check
        if _is_private_url(url):
            return ToolResult(
                success=False,
                output="",
                error=f"URL points to a private/internal network (SSRF protection): {url}",
            )

        try:
            timeout = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                kwargs: Dict[str, Any] = {"headers": headers}
                if body and method in ("POST", "PUT", "PATCH"):
                    kwargs["data"] = body

                async with session.request(method, url, **kwargs) as resp:
                    response_body = await resp.text()
                    status = resp.status

                    # Truncate large responses
                    max_chars = 50000
                    if len(response_body) > max_chars:
                        response_body = response_body[:max_chars] + "\n... [truncated]"

                    output_parts = [
                        f"HTTP {method} {url}",
                        f"Status: {status} {resp.reason}",
                        f"Headers: {dict(resp.headers)}",
                    ]
                    if response_body:
                        output_parts.append(f"\nBody:\n{response_body}")

                    return ToolResult(
                        success=status < 400,
                        output="\n".join(output_parts),
                        metadata={
                            "status": status,
                            "content_type": resp.headers.get("Content-Type", ""),
                        },
                    )

        except aiohttp.ClientError as e:
            return ToolResult(success=False, output="", error=f"HTTP client error: {e}")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"HTTP request failed: {e}")
