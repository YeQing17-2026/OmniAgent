"""Interactive chat CLI for OmniAgent."""

import asyncio
import sys
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory

from omniagent.config import load_config
from omniagent.agents import ReflexionAgent
from omniagent.agents.llm import LLMMessage
from omniagent.gateway.session import SessionManager
from omniagent.infra.logging import setup_logging

console = Console()
_history = InMemoryHistory()
_prompt_executor = ThreadPoolExecutor(max_workers=1)


class ChatSession:
    """Interactive chat session."""

    def __init__(self, work_dir: Optional[Path] = None, verbose: bool = False, provider: Optional[str] = None):
        """Initialize chat session."""
        self.work_dir = work_dir or Path.cwd()
        self.config = load_config()  # Load from file or defaults with env vars
        self.agent: Optional[ReflexionAgent] = None
        self.verbose = verbose

        # Override provider if specified via CLI
        if provider and provider in (self.config.providers or {}):
            self.config.agent.model_provider = provider
            provider_cfg = self.config.providers[provider]
            if provider_cfg.model_id:
                self.config.agent.model_id = provider_cfg.model_id
            if provider_cfg.api_url:
                self.config.agent.api_url = provider_cfg.api_url
        self.session_id: Optional[str] = None

        # Logging: file always enabled, console controlled by --verbose
        from datetime import date
        log_dir = Path.home() / ".omniagent" / "logs"
        log_file = log_dir / f"chat_{date.today().isoformat()}.log"
        setup_logging(verbose=verbose, log_file=log_file)

        # Session persistence
        sessions_dir = Path.home() / ".omniagent" / "sessions"
        self.session_manager = SessionManager(
            storage_dir=sessions_dir,
            session_timeout=self.config.gateway.session_timeout,
        )

    async def _ask_approval(self, tool_name: str, params: dict) -> bool:
        """Ask user for approval of a tool call in the TUI."""
        console.print(
            f"\n[yellow]⚠[/yellow] Tool [cyan]{tool_name}[/cyan] requires approval."
        )
        # Show command for bash, show params for others
        if tool_name == "bash":
            console.print(f"  Command: [bold]{params.get('command', '')}[/bold]")
        else:
            for k, v in params.items():
                console.print(f"  {k}: {v}")

        loop = asyncio.get_event_loop()
        while True:
            answer = await loop.run_in_executor(
                _prompt_executor, lambda: prompt("  Allow? [Y]/n/a ", default="Y")
            )
            if answer.lower() in ("y", "yes", ""):
                return True
            elif answer.lower() in ("n", "no"):
                return False
            elif answer.lower() == "a":
                # Add allow rule for this tool so it won't ask again
                if self.agent and self.agent.policy:
                    from omniagent.security.policy import PolicyRule, PolicyDecision
                    self.agent.policy.add_rule(
                        PolicyRule(
                            name=f"user_allow_{tool_name}",
                            decision=PolicyDecision.ALLOW,
                            tools=[tool_name],
                            priority=30,  # Higher than approval rule (20)
                        )
                    )
                console.print(f"  [green]✓[/green] {tool_name} auto-approved for this session.")
                return True

    async def start(self):
        """Start chat session."""
        console.print(
            Panel.fit(
                "[bold cyan]OmniAgent Interactive Chat[/bold cyan]\n"
                f"Working Directory: {self.work_dir}\n"
                f"Log: ~/.omniagent/logs/chat_YYYY-MM-DD.log"
                + (" [dim](verbose on)[/dim]" if self.verbose else "")
                + "\n"
                "Type 'exit' or 'quit' to end the session\n"
                "Type 'help' for available commands",
                border_style="cyan",
            )
        )

        # Initialize agent with interactive approval callback
        try:
            self.agent = ReflexionAgent(
                config=self.config,
                work_dir=self.work_dir,
                enable_security=True,
                approval_callback=self._ask_approval,
            )
            console.print("[green]✓[/green] Agent initialized\n")
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to initialize agent: {e}")
            return

        # Create or resume session
        self.session_id = str(__import__("uuid").uuid4())[:8]
        session = self.session_manager.create_session(
            user_id="cli_user",
            channel_id="chat",
            session_id=self.session_id,
        )

        # Check for recent session to resume
        recent = self._find_recent_session()
        if recent:
            loaded = self._restore_session(recent)
            if loaded:
                self.session_id = recent.id
                session = recent
                console.print(
                    f"[dim]Resumed session {recent.id} "
                    f"({len(self.agent.conversation_history)} messages)[/dim]\n"
                )

        try:
            # Chat loop
            while True:
                try:
                    # Get user input (run in executor to avoid event loop conflict)
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        _prompt_executor, lambda: prompt("\nYou> ", history=_history)
                    )

                    # Record user query timestamp
                    query_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    if not user_input.strip():
                        continue

                    # Handle commands
                    if user_input.lower() in ["exit", "quit"]:
                        self._save_current_session(session)
                        console.print(
                            f"\n[yellow]Goodbye![/yellow] "
                            f"[dim]Session: {self.session_id}[/dim]"
                        )
                        break

                    if user_input.lower() == "help":
                        self._show_help()
                        continue

                    if user_input.lower() == "clear":
                        if self.agent:
                            self.agent.clear_history()
                        console.clear()
                        console.print("[yellow]Conversation history cleared.[/yellow]")
                        continue

                    if user_input.lower().startswith("/tools"):
                        self._show_tools()
                        continue

                    if user_input.lower().startswith("/status"):
                        self._show_status()
                        continue

                    if user_input.lower().startswith("/history"):
                        self._show_history()
                        continue

                    if user_input.lower().startswith("/sessions"):
                        self._show_sessions()
                        continue

                    # Send to agent
                    console.print(f"[dim]{query_time}[/dim] [dim]Thinking...[/dim]")
                    result = await self.agent.execute(user_input)

                    # Sync to session
                    self._save_current_session(session)

                    # === Feature activity: plain text output ===
                    # Order: guard blocked -> response -> reflection/infra -> evolution
                    if self.agent:
                        from datetime import datetime as _dt
                        now = _dt.now().strftime("%Y-%m-%d %H:%M")

                        # [1] Guardian blocked events (before response)
                        if getattr(self.agent, '_guardian_blocked_events', []):
                            for ev in self.agent._guardian_blocked_events:
                                params = ev.get("params_summary", "")
                                findings = ev.get("findings", [])
                                desc = "; ".join(findings) if findings else ev.get("risk_level", "high")
                                detail = f'（{params}）' if params else ""
                                ev_time = ev.get("time", now)
                                console.print(f"{desc}安全问题已被 Guardian 拦截{detail}---HyperHarness，{ev_time}")

                    # [2] Display response
                    response_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if result.success:
                        console.print(f"[bold green]OmniAgent[/bold green] [dim]{response_time}[/dim] ", end="")
                        if any(marker in result.response for marker in ["#", "```", "*", "-"]):
                            console.print(Markdown(result.response))
                        else:
                            console.print(result.response)
                    else:
                        console.print(f"[red]Error:[/red] {result.error}")

                    # Show metadata if verbose
                    if result.metadata:
                        iterations = result.metadata.get("iterations")
                        if iterations:
                            console.print(
                                f"\n[dim]Completed in {iterations} iteration(s)[/dim]"
                            )

                    # [3] Post-response features
                    if self.agent:
                        from datetime import datetime as _dt
                        now = _dt.now().strftime("%Y-%m-%d %H:%M")

                        # Reflexion (enriched)
                        if getattr(self.agent, '_reflections', []):
                            count = len(self.agent._reflections)
                            last = self.agent._reflections[-1]
                            preview = last[:100] + "..." if len(last) > 100 else last
                            console.print(f"OmniAgent已进行{count}次反思，本次反思：{preview}---Deep Reflection，{now}")

                        # Sentinel (brief)
                        if getattr(self.agent, '_sentinel_activated', False):
                            console.print(f"哨兵agent已激活，任务已分解为里程碑---哨兵agent，{now}")

                        # Parallel execution (brief)
                        if getattr(self.agent, '_parallel_exec_count', 0) > 0:
                            batch = self.agent._parallel_exec_count
                            tools = self.agent._parallel_tools_count
                            console.print(f"并行执行引擎启动：{batch}批{tools}个工具并发执行---HyperHarness，{now}")

                        # Context compaction (brief)
                        if getattr(self.agent, '_compaction_count', 0) > 0:
                            count = self.agent._compaction_count
                            console.print(f"上下文已压缩{count}次---HyperHarness，{now}")

                        # Stuck/loop detection (brief)
                        if getattr(self.agent, '_stuck_detected_count', 0) > 0:
                            count = self.agent._stuck_detected_count
                            console.print(f"卡死检测触发{count}次，已自动纠正---深度反思，{now}")

                        # Skill evolution (enriched, renamed)
                        if getattr(self.agent, '_skill_evolution', None) and self.agent._skill_evolution.last_session_results:
                            sr = self.agent._skill_evolution.last_session_results
                            ev_time = sr.get("time", now)
                            if sr.get("skill_compiled"):
                                name = sr.get("skill_name", "未知")
                                desc = sr.get("skill_description", "")
                                patterns = sr.get("skill_source_patterns", 0)
                                detail = f"「{name}」{desc}（基于{patterns}次重复模式）" if desc else f"「{name}」（基于{patterns}次重复模式）"
                                console.print(f"技能自进化：新技能编译完成，{detail}---技能自进化，{ev_time}")
                            if sr.get("patches_written"):
                                name = sr.get("patch_skill_name", "未知")
                                desc = sr.get("patch_description", "")
                                detail = f"「{name}」" + (f"（{desc}）" if desc else "")
                                console.print(f"技能自进化：修复补丁已写入，{detail}---技能自进化，{ev_time}")

                        # Context evolution (enriched, renamed -> 主动式记忆)
                        if getattr(self.agent, '_context_evolution', None) and self.agent._context_evolution.last_session_results:
                            cr = self.agent._context_evolution.last_session_results
                            ev_time = cr.get("time", now)
                            if cr.get("rules_promoted"):
                                count = cr['rules_promoted']
                                details = cr.get("rule_details", [])
                                if details:
                                    rules_text = "；".join(details[:3])
                                    console.print(f"主动式记忆：{count}条经验已写入AGENTS.md —— {rules_text}---主动式记忆，{ev_time}")
                                else:
                                    console.print(f"主动式记忆：{count}条经验已写入AGENTS.md---主动式记忆，{ev_time}")
                            if cr.get("lessons_extracted"):
                                count = cr['lessons_extracted']
                                details = cr.get("lesson_details", [])
                                if details:
                                    lessons_text = "；".join(details[:3])
                                    console.print(f"主动式记忆：提取了{count}条教训 —— {lessons_text}---主动式记忆，{ev_time}")
                                else:
                                    console.print(f"主动式记忆：提取了{count}条教训---主动式记忆，{ev_time}")

                except KeyboardInterrupt:
                    console.print("\n\n[yellow]Use 'exit' or 'quit' to end the session[/yellow]")
                    continue
                except Exception as e:
                    console.print(f"\n[red]Error:[/red] {e}")
        finally:
            # Always save on exit
            self._save_current_session(session)
            self.session_manager.close_session(self.session_id)

    def _show_help(self):
        """Show help message."""
        help_text = """
[bold]Available Commands:[/bold]

  [cyan]help[/cyan]      - Show this help message
  [cyan]exit/quit[/cyan] - Exit the chat session
  [cyan]clear[/cyan]     - Clear the screen and conversation history
  [cyan]/tools[/cyan]    - Show available tools
  [cyan]/status[/cyan]   - Show agent status
  [cyan]/history[/cyan]  - Show conversation history
  [cyan]/sessions[/cyan] - List recent chat sessions

[bold]Examples:[/bold]

  • Read file: "Read the contents of README.md"
  • Write file: "Create a file called test.txt with 'Hello World'"
  • Execute command: "List all Python files in the current directory"
  • Ask question: "What is 2 + 2?"
"""
        console.print(Panel(help_text, border_style="cyan"))

    def _save_current_session(self, session) -> None:
        """Sync agent conversation history into the session and save."""
        if not self.agent:
            return
        # Sync conversation_history to session history
        session.history = []
        for msg in self.agent.conversation_history:
            session.add_message(
                role=msg.role,
                content=msg.content,
            )
        self.session_manager._save_session(session)

    def _find_recent_session(self):
        """Find the most recent active session from this working directory."""
        sessions = self.session_manager.list_sessions(
            user_id="cli_user",
            channel_id="chat",
        )
        # Filter to recent (within timeout) active sessions
        active = [
            s for s in sessions
            if not s.is_expired(self.session_manager.session_timeout)
        ]
        if not active:
            return None
        # Pick the most recent
        active.sort(key=lambda s: s.last_active_at, reverse=True)
        return active[0]

    def _restore_session(self, session) -> bool:
        """Restore agent conversation history from a session."""
        if not self.agent or not session.history:
            return False
        self.agent.conversation_history = [
            LLMMessage(role=msg.role, content=msg.content)
            for msg in session.history
        ]
        self.session_manager.resume_session(session.id)
        return True

    def _show_sessions(self):
        """Show recent chat sessions."""
        sessions = self.session_manager.list_sessions(
            user_id="cli_user",
            channel_id="chat",
        )
        if not sessions:
            console.print("[dim]No sessions found.[/dim]")
            return

        # Show most recent first, up to 10
        sessions.sort(key=lambda s: s.last_active_at, reverse=True)
        sessions = sessions[:10]

        text = "[bold]Recent Sessions:[/bold]\n\n"
        for s in sessions:
            state_tag = {
                "active": "[green]active[/green]",
                "paused": "[yellow]paused[/yellow]",
                "closed": "[dim]closed[/dim]",
            }.get(s.state.value, s.state.value)

            current = " [bold](current)[/bold]" if s.id == self.session_id else ""
            elapsed = (datetime.now() - s.last_active_at).total_seconds()
            if elapsed < 60:
                time_ago = f"{int(elapsed)}s ago"
            elif elapsed < 3600:
                time_ago = f"{int(elapsed / 60)}m ago"
            else:
                time_ago = f"{int(elapsed / 3600)}h ago"

            text += (
                f"  [cyan]{s.id}[/cyan] {state_tag}{current} "
                f"[dim]{time_ago} · {len(s.history)} messages[/dim]\n"
            )

        console.print(Panel(text, border_style="cyan"))

    def _show_tools(self):
        """Show available tools."""
        if not self.agent:
            console.print("[red]Agent not initialized[/red]")
            return

        tools_text = "[bold]Available Tools:[/bold]\n\n"
        for tool_name, tool in self.agent.tools.items():
            tools_text += f"  [cyan]{tool_name}[/cyan] - {tool.description}\n"

        console.print(Panel(tools_text, border_style="cyan"))

    def _show_status(self):
        """Show agent status."""
        if not self.agent:
            console.print("[red]Agent not initialized[/red]")
            return

        status_text = f"""
[bold]Agent Status:[/bold]

  Session: {self.session_id or 'N/A'}
  Model: {self.config.agent.model_id}
  Provider: {self.config.agent.model_provider}
  Max Iterations: {self.agent.max_iterations}
  Security: {"Enabled" if self.agent.enable_security else "Disabled"}
  Tools: {len(self.agent.tools)}
  Working Directory: {self.work_dir}
"""
        console.print(Panel(status_text, border_style="cyan"))

    def _show_history(self):
        """Show conversation history summary."""
        if not self.agent:
            console.print("[red]Agent not initialized[/red]")
            return
        history = self.agent.get_history()
        if not history:
            console.print("[dim]No conversation history.[/dim]")
            return
        console.print(f"[bold]History: {len(history)} messages[/bold]\n")
        for i, msg in enumerate(history):
            role_color = "blue" if msg.role == "user" else "green"
            preview = msg.content[:80].replace("\n", " ")
            console.print(f"  [{i}] [{role_color}]{msg.role}[/{role_color}]: {preview}...")


@click.command()
@click.option(
    "--work-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=None,
    help="Working directory (default: current directory)",
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose logging (show debug output in terminal)",
)
@click.option(
    "--provider",
    type=str,
    default=None,
    help="LLM provider override (e.g. deepseek, openai)",
)
def chat(work_dir: Optional[str], verbose: bool, provider: Optional[str]):
    """Start an interactive chat session with OmniAgent."""
    work_dir_path = Path(work_dir) if work_dir else Path.cwd()
    session = ChatSession(work_dir=work_dir_path, verbose=verbose, provider=provider)

    try:
        asyncio.run(session.start())
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Session interrupted[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    chat()
