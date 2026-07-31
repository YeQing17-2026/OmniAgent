"""Auth CLI commands and shared OAuth helpers."""

import asyncio
import webbrowser
from pathlib import Path

import click

from omniagent.auth.openai_codex import login_openai_codex
from omniagent.auth.token_store import OAuthCredentials, TokenStore

_DEFAULT_AUTH_PATH = Path.home() / ".omniagent" / "auth-profiles.json"
_PROFILE_KEY = "openai-codex:default"


def run_openai_codex_login(auth_path: Path = _DEFAULT_AUTH_PATH) -> OAuthCredentials:
    """Run the OAuth PKCE login flow and persist credentials. Returns OAuthCredentials."""

    def _open_url(url: str) -> None:
        click.echo(f"\nOpening browser for authorization...")
        click.echo(f"  URL: {url}\n")
        try:
            webbrowser.open(url)
        except Exception:
            click.echo("Could not open browser automatically. Please open the URL above manually.")

    creds = asyncio.run(login_openai_codex(on_open_url=_open_url))
    TokenStore(auth_path).save(_PROFILE_KEY, creds)
    return creds


@click.group()
def auth() -> None:
    """Manage OAuth credentials for providers."""
    pass


@auth.command("login")
@click.option("--provider", default="openai-codex", show_default=True,
              help="OAuth provider to log in to")
def auth_login(provider: str) -> None:
    """Log in to an OAuth provider and save credentials."""
    if provider != "openai-codex":
        click.echo(f"Unsupported provider: {provider}", err=True)
        raise SystemExit(1)

    click.echo(f"Starting OAuth login for {provider}...")
    try:
        creds = run_openai_codex_login()
        click.echo(f"\n✓ Login successful!")
        click.echo(f"  Account : {creds.account_id}")
    except Exception as e:
        click.echo(f"\n✗ Login failed: {e}", err=True)
        raise SystemExit(1)


@auth.command("status")
@click.option("--provider", default="openai-codex", show_default=True,
              help="OAuth provider to check")
def auth_status(provider: str) -> None:
    """Show current OAuth token status."""
    import time
    from datetime import datetime

    if provider != "openai-codex":
        click.echo(f"Unsupported provider: {provider}", err=True)
        raise SystemExit(1)

    creds = TokenStore(_DEFAULT_AUTH_PATH).load(_PROFILE_KEY)
    if creds is None:
        click.echo(f"No credentials found for {provider}.")
        click.echo(f"Run: omniagent auth login --provider {provider}")
        return

    expires_dt = datetime.fromtimestamp(creds.expires / 1000)
    now_ms = int(time.time() * 1000)
    remaining_ms = creds.expires - now_ms
    remaining_min = remaining_ms // 60_000
    expires_str = expires_dt.strftime('%Y-%m-%d %H:%M:%S')

    if remaining_ms > 0:
        status = f"valid (expires in {remaining_min} min)"
        expires_line = f"{expires_str} ({remaining_min} min remaining)"
    else:
        status = "EXPIRED — run: omniagent auth login"
        expires_line = f"{expires_str} (expired)"

    click.echo(f"Provider : {provider}")
    click.echo(f"Account  : {creds.account_id}")
    click.echo(f"Expires  : {expires_line}")
    click.echo(f"Status   : {status}")


@auth.command("logout")
@click.option("--provider", default="openai-codex", show_default=True,
              help="OAuth provider to log out of")
@click.confirmation_option(prompt="Remove saved credentials?")
def auth_logout(provider: str) -> None:
    """Remove saved OAuth credentials."""
    if provider != "openai-codex":
        click.echo(f"Unsupported provider: {provider}", err=True)
        raise SystemExit(1)

    TokenStore(_DEFAULT_AUTH_PATH).delete(_PROFILE_KEY)
    click.echo(f"✓ Credentials removed for {provider}.")
