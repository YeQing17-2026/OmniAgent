"""Main CLI entry point."""

import click
from pathlib import Path

from omniagent import __version__
from omniagent.config import load_config, save_config, get_default_config
from .chat import chat


@click.group()
@click.version_option(version=__version__)
@click.option(
    "--config",
    type=click.Path(path_type=Path),
    help="Path to config file"
)
@click.pass_context
def main(ctx: click.Context, config: Path) -> None:
    """
    OmniAgent - Next-generation AI Agent framework

    OmniAgent — three core innovations:
    1. Intelligence & Planning: ReAct, Reflexion, Dynamic Agent System
    2. Harness: Progressive Loading, Dependency Graph, Security Guard
    3. Self-Evolution: RLAIF, Self-Improving, Skill Management
    """
    ctx.ensure_object(dict)

    # Load config
    try:
        ctx.obj["config"] = load_config(config)
        ctx.obj["config_path"] = str(config) if config else None
    except Exception as e:
        click.echo(f"Error loading config: {e}", err=True)
        ctx.obj["config"] = get_default_config()


@main.command()
@click.option(
    "--host",
    default=None,
    help="Gateway host (overrides config)"
)
@click.option(
    "--port",
    default=None,
    type=int,
    help="Gateway port (overrides config, default: 18790)"
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Verbose output"
)
@click.pass_context
def serve(ctx: click.Context, host: str, port: int, verbose: bool) -> None:
    """Start the OmniAgent gateway server."""
    from omniagent.infra import setup_logging
    from omniagent.gateway import GatewayServer
    from omniagent.agents import ReflexionAgent

    # Setup logging
    setup_logging(verbose)

    # Get config
    config = ctx.obj["config"]

    # Override config if specified
    if host:
        config.gateway.host = host
    if port:
        config.gateway.port = port

    # Check API key (top-level or in providers)
    has_key = bool(config.api_key or config.openai_api_key or config.anthropic_api_key)
    if not has_key and hasattr(config, "providers"):
        for p in config.providers.values():
            if getattr(p, "api_key", None):
                has_key = True
                break
    if not has_key:
        click.echo("Warning: No API key configured. Agent will not work.", err=True)
        click.echo("Set api_key in config or use OMNIAGENT_API_KEY environment variable.", err=True)

    click.echo(f"Starting OmniAgent gateway at {config.gateway.host}:{config.gateway.port}...")
    click.echo(f"Using {config.agent.model_provider}/{config.agent.model_id}")

    if verbose:
        click.echo("Verbose mode enabled")

    # Create agent
    agent = ReflexionAgent(config)

    # Create and run server
    server = GatewayServer(config, agent=agent)

    # Set agent handler
    async def agent_handler(message):
        return await agent.handle_message(message)

    server.router.set_agent_handler(agent_handler)

    click.echo("Gateway server ready!")
    click.echo(f"Web UI: http://{config.gateway.host}:{config.gateway.port}/")
    click.echo(f"HTTP API: http://{config.gateway.host}:{config.gateway.port}/message")
    click.echo(f"Health: http://{config.gateway.host}:{config.gateway.port}/health")

    # Show channel info
    try:
        from omniagent.channels.registry import discover_all
        available = list(discover_all().keys())
        if available:
            click.echo(f"Available channels: {', '.join(available)}")
    except Exception:
        pass

    server.run()


@main.command()
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    help="Output path for config file"
)
@click.pass_context
def config(ctx: click.Context, output: Path) -> None:
    """Show or initialize configuration."""
    config_obj = ctx.obj["config"]

    if output:
        # Save config to specified path
        save_config(config_obj, output)
        click.echo(f"Config saved to: {output}")
    else:
        # Show current config
        click.echo("Current configuration:")
        click.echo(f"  Work directory: {config_obj.work_dir}")
        click.echo(f"  Agent model: {config_obj.agent.model_provider}/{config_obj.agent.model_id}")
        click.echo(f"  Gateway: {config_obj.gateway.host}:{config_obj.gateway.port}")
        click.echo(f"  Tool profile: {config_obj.tools.profile}")
        click.echo(f"  Progressive loading: {config_obj.enable_progressive_loading}")
        click.echo(f"  Parallel execution: {config_obj.enable_parallel_execution}")
        click.echo(f"  Security guard: {config_obj.enable_security_guard}")
        click.echo(f"  Self-improving: {config_obj.enable_self_improving}")


@main.command()
@click.pass_context
def onboard(ctx: click.Context) -> None:
    """Interactive setup wizard — configure API key, provider, and preferences."""
    import os
    import yaml
    from pathlib import Path

    click.echo("╔══════════════════════════════════════════╗")
    click.echo("║         OmniAgent Setup Wizard            ║")
    click.echo("╚══════════════════════════════════════════╝")
    click.echo()

    config_obj = ctx.obj["config"]

    # --- Step 1: Choose LLM provider ---
    providers = [
        ("deepseek", "DeepSeek (e.g., deepseek-chat)"),
        ("openai", "OpenAI (e.g., gpt-4o)"),
        ("anthropic", "Anthropic (e.g., claude-sonnet-4-20250514)"),
        ("ollama", "Ollama (e.g., llama3)"),
        ("gemini", "Google Gemini (e.g., gemini-2.0-flash)"),
        ("openrouter", "OpenRouter (e.g., openai/gpt-4o)"),
        ("custom", "Custom OpenAI-compatible endpoint"),
    ]

    click.echo("Step 1/5: Choose LLM Provider")
    click.echo("-" * 40)
    for i, (key, desc) in enumerate(providers, 1):
        default = " (current)" if config_obj.agent.model_provider == key else ""
        click.echo(f"  {i}. {desc}{default}")

    choice = click.prompt(
        "\n  Select provider",
        type=click.IntRange(1, len(providers)),
        default=next((i for i, (k, _) in enumerate(providers, 1) if k == config_obj.agent.model_provider), 1),
    )
    selected_provider, selected_desc = providers[choice - 1]
    click.echo(f"  → {selected_desc}")

    # --- Step 2: Provider-specific config ---
    provider_api_url = None

    if selected_provider == "custom":
        click.echo(f"\nStep 2/5: Custom Provider Settings")
        click.echo("-" * 40)
        provider_api_url = click.prompt("  API Base URL", type=str)
        click.echo(f"  → {provider_api_url}")
    else:
        click.echo(f"\nStep 2/5: (skipped — using {selected_desc} defaults)")

    # --- Step 3: Enter API key ---
    click.echo(f"\nStep 3/5: API Key")
    click.echo("-" * 40)

    env_var_map = {
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "ollama": None,
        "gemini": "GOOGLE_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "custom": None,
    }
    env_var = env_var_map.get(selected_provider)
    existing_key = config_obj.api_key or (os.getenv(env_var) if env_var else None)

    if existing_key:
        click.echo(f"  Current key: {'*' * min(8, len(existing_key))}... ({len(existing_key)} chars)")
        use_existing = click.confirm("  Use existing key?", default=True)
        api_key = existing_key if use_existing else click.prompt("  Enter API key")
    else:
        api_key = click.prompt("  Enter API key" if selected_provider != "ollama" else "  API key (usually empty for Ollama)", default="", show_default=False)

    # --- Step 4: Choose model ---
    click.echo(f"\nStep 4/5: Model ID")
    click.echo("-" * 40)

    default_models = {
        "deepseek": "deepseek-chat",
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4-20250514",
        "ollama": "llama3",
        "gemini": "gemini-2.0-flash",
        "openrouter": "openai/gpt-4o",
    }

    if selected_provider == "custom":
        model_id = click.prompt("  Model ID")
    elif selected_provider == "ollama":
        model_id = click.prompt("  Model name", default=default_models.get("ollama", "llama3"))
    else:
        default_model = default_models.get(selected_provider, "")
        use_default = click.confirm(f"  Use default model '{default_model}'?", default=True)
        model_id = default_model if use_default else click.prompt("  Model name")

    click.echo(f"  → {model_id}")

    # --- Step 5: Save ---
    click.echo(f"\nStep 5/5: Save Configuration")
    click.echo("-" * 40)

    config_path = ctx.obj.get("config_path") or os.path.expanduser("~/.omniagent/config.yaml")
    click.echo(f"  Config will be saved to: {config_path}")

    do_save = click.confirm("  Save?", default=True)
    if not do_save:
        click.echo("  Cancelled. No changes made.")
        return

    # Build provider config dict (providers block for extensibility)
    provider_data = {"api_key": api_key, "model_id": model_id}
    if provider_api_url:
        provider_data["api_url"] = provider_api_url

    config_data = {
        "providers": {
            selected_provider: provider_data,
        },
        "agent": {
            "model_provider": selected_provider,
            "temperature": config_obj.agent.temperature,
            "max_iterations": config_obj.agent.max_iterations,
            "reflexion_enabled": config_obj.agent.reflexion_enabled,
            "reflexion_max_attempts": config_obj.agent.reflexion_max_attempts,
        },
        "gateway": {
            "host": config_obj.gateway.host,
            "port": config_obj.gateway.port,
            "session_timeout": config_obj.gateway.session_timeout,
        },
        "tools": {
            "profile": config_obj.tools.profile,
        },
        "enable_progressive_loading": config_obj.enable_progressive_loading,
        "enable_parallel_execution": config_obj.enable_parallel_execution,
        "enable_security_guard": config_obj.enable_security_guard,
        "enable_self_improving": config_obj.enable_self_improving,
    }

    # Merge with existing file if present
    if Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}
        if isinstance(existing, dict):
            def _merge(base, override):
                for k, v in override.items():
                    if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                        _merge(base[k], v)
                    else:
                        base[k] = v
            _merge(existing, config_data)
            config_data = existing

    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # Generate bootstrap files (AGENTS.md, SOUL.md, CUSTOM.md)
    bootstrap_dir = config_path.parent
    try:
        from omniagent.agents.bootstrap import AGENTS_TEMPLATE, SOUL_TEMPLATE, CUSTOM_TEMPLATE
        for filename, content in [("AGENTS.md", AGENTS_TEMPLATE), ("SOUL.md", SOUL_TEMPLATE), ("CUSTOM.md", CUSTOM_TEMPLATE)]:
            filepath = bootstrap_dir / filename
            if not filepath.exists():
                filepath.write_text(content, encoding="utf-8")
        click.echo("  ✓ Bootstrap files created (AGENTS.md, SOUL.md, CUSTOM.md)")
    except Exception:
        click.echo("  ! Could not create bootstrap files (non-critical)")

    click.echo()
    click.echo("  ✓ Configuration saved!")
    click.echo(f"  ✓ Provider: {selected_provider}")
    click.echo(f"  ✓ Model: {model_id}")
    if api_key:
        click.echo(f"  ✓ API key: {'*' * 8}... (hidden)")
    click.echo()
    click.echo("You can now run:")
    click.echo(f"  omniagent chat              # Start interactive chat")
    click.echo(f"  omniagent serve             # Start gateway + Web UI")
    click.echo()
    click.echo("To change settings later, edit:")
    click.echo(f"  {config_path}")


@main.command()
@click.pass_context
def doctor(ctx: click.Context) -> None:
    """Diagnose OmniAgent installation and configuration."""
    click.echo("OmniAgent Doctor")
    click.echo("=" * 50)

    # Check Python version
    import sys
    click.echo(f"Python version: {sys.version}")

    # Check config
    config_obj = ctx.obj["config"]
    click.echo(f"Config loaded: ✓")

    # Check API keys
    if config_obj.openai_api_key:
        click.echo("OpenAI API key: ✓")
    else:
        click.echo("OpenAI API key: ✗ (not configured)")

    if config_obj.anthropic_api_key:
        click.echo("Anthropic API key: ✓")
    else:
        click.echo("Anthropic API key: ✗ (not configured)")

    # Check work directory
    if config_obj.work_dir.exists():
        click.echo(f"Work directory: ✓ ({config_obj.work_dir})")
    else:
        click.echo(f"Work directory: ✗ ({config_obj.work_dir} does not exist)")

    click.echo("=" * 50)
    click.echo("Diagnosis complete")


# Register chat command
main.add_command(chat)


if __name__ == "__main__":
    main()
