"""Agent implementation with native function calling."""

import asyncio
import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from omniagent.config import OmniAgentConfig
from omniagent.gateway.router import IncomingMessage, OutgoingMessage
from omniagent.infra import get_logger
from omniagent.tools import (
    Tool, ToolRegistry, ReadTool, WriteTool, EditTool, BashTool,
    LoadJSONTool, SaveJSONTool, ProcessListTool, ProcessKillTool,
    WebSearchTool, WebFetchTool,
)
from omniagent.tools.grep_tool import GrepTool
from omniagent.tools.find_tool import FindTool
from omniagent.tools.ls_tool import LsTool
from omniagent.tools.diff_tool import DiffTool
from omniagent.tools.http_tool import HttpTool
from omniagent.security import ApprovalManager, ApprovalStatus, AuditLogger, ToolPolicy
from omniagent.security.policy import ToolProfile, PolicyRule, PolicyDecision
from .agent import Agent, AgentResult
from .llm import LLMProvider, LLMMessage, create_llm_provider
from .context_manager import (
    ContextManager, resolve_context_window_size, detect_context_overflow,
)
from .context_assembler import ContextAssembler, estimate_tokens
from .memory_manager import (
    MemorySearchManager, MemorySearchTool, MemoryGetTool,
)
from .skills import SkillManager
from .events import EventBus, EventType, AgentEvent
from .state import AgentState
from .abort import AbortController, AbortError
from .hooks import ToolHookManager, ToolCallContext

logger = get_logger(__name__)


class ReflexionAgent(Agent):
    """AI Agent with native function calling support."""

    def __init__(
        self,
        config: OmniAgentConfig,
        llm_provider: Optional[LLMProvider] = None,
        work_dir: Optional[Path] = None,
        enable_security: bool = True,
        approval_callback: Optional["ApprovalCallback"] = None,
    ):
        """
        Initialize agent.

        Args:
            config: OmniAgent configuration
            llm_provider: Optional LLM provider (created if not provided)
            work_dir: Working directory for tools (defaults to current directory)
            enable_security: Enable security system (approval/audit/policy)
            approval_callback: Optional async callback for interactive approval.
                Signature: async def callback(tool_name, params) -> bool
                Return True to approve, False to deny.
        """
        self.config = config
        self.max_iterations = config.agent.max_iterations
        self.work_dir = work_dir or Path.cwd()
        self.enable_security = enable_security
        self.approval_callback = approval_callback

        # Create LLM provider
        if llm_provider is None:
            # Resolve provider-specific overrides
            provider_name = config.agent.model_provider
            provider_cfg = config.providers.get(provider_name) if config.providers else None

            provider_api_url = config.agent.api_url
            provider_model = config.agent.model_id
            provider_api_key = ""

            if provider_cfg:
                if provider_cfg.api_url:
                    provider_api_url = provider_cfg.api_url
                if provider_cfg.model_id:
                    provider_model = provider_cfg.model_id
                provider_api_key = provider_cfg.api_key or ""

            # Fallback to top-level api_key
            if not provider_api_key:
                provider_api_key = config.api_key or config.openai_api_key or ""

            # Sync resolved model back to config so system prompt uses it
            config.agent.model_id = provider_model

            llm_provider = create_llm_provider(
                provider=provider_name,
                api_key=provider_api_key,
                model=provider_model,
                api_url=provider_api_url,
            )

        self.llm = llm_provider

        # Agent subsystems
        self.event_bus = EventBus()
        self.agent_state = AgentState()
        self.abort_controller = AbortController()
        self.tool_hook_manager = ToolHookManager()

        # Conversation history for multi-turn support
        self.conversation_history: List[LLMMessage] = []

        # Tool loop detection
        self._recent_tool_calls: List[str] = []  # SHA-256 signatures
        self._recent_errors: List[str] = []
        self._loop_detection_window: int = 10
        self._loop_repeat_threshold: int = 3

        # Initialize tool registry
        self.registry = ToolRegistry()
        self.tools: Dict[str, Tool] = {}

        # Initialize security system
        if enable_security:
            security_dir = self.work_dir / ".omniagent"

            # Build policy from config.tools
            profile_name = config.tools.profile
            try:
                profile = ToolProfile(profile_name)
            except ValueError:
                logger.warning("unknown_tool_profile", profile=profile_name, fallback="coding")
                profile = ToolProfile.CODING

            self.policy = ToolPolicy(profile=profile)

            # Apply explicit allow rules from config (priority 30, higher than profile)
            if config.tools.allow:
                self.policy.add_rule(PolicyRule(
                    name="config_allow",
                    decision=PolicyDecision.ALLOW,
                    tools=config.tools.allow,
                    priority=30,
                ))

            # Apply explicit deny rules from config (priority 40, highest)
            if config.tools.deny:
                self.policy.add_rule(PolicyRule(
                    name="config_deny",
                    decision=PolicyDecision.DENY,
                    tools=config.tools.deny,
                    priority=40,
                ))

            self.approval_manager = ApprovalManager(
                storage_dir=security_dir / "approvals",
                auto_approve_low_risk=True,
            )
            self.audit_logger = AuditLogger(log_dir=security_dir / "audit")
        else:
            self.policy = None
            self.approval_manager = None
            self.audit_logger = None

        # Register default tools
        self._register_default_tools()

        # Initialize context manager for compaction
        resolved_window = resolve_context_window_size(
            config.agent.model_id,
            config.agent.context_window_size,
        )
        self.context_manager = ContextManager(
            context_window_size=resolved_window,
            llm_provider=self.llm if config.agent.compaction_enabled else None,
        )

        # Initialize memory search system
        self.memory_manager: Optional[MemorySearchManager] = None
        if config.memory.enabled:
            api_key = config.api_key or config.openai_api_key or ""
            self.memory_manager = MemorySearchManager(
                workspace_dir=self.work_dir,
                store_path=self.work_dir / ".omniagent" / "memory.db",
                llm_provider=self.llm if api_key else None,
                api_key=api_key,
                api_url=config.agent.api_url or "",
                model_id=config.agent.model_id,
                chunking_tokens=config.memory.chunking_tokens,
                chunking_overlap=config.memory.chunking_overlap,
                hybrid_enabled=config.memory.hybrid_enabled,
                vector_weight=config.memory.hybrid_vector_weight,
                text_weight=config.memory.hybrid_text_weight,
                query_max_results=config.memory.query_max_results,
                query_min_score=config.memory.query_min_score,
            )
            # Register memory tools
            self.register_tool(MemorySearchTool(self.memory_manager))
            self.register_tool(MemoryGetTool(self.memory_manager))

        # Initialize skills system
        self.skill_manager = SkillManager(work_dir=self.work_dir)

        # Initialize skill evolution system (gated by enable_self_improving)
        self._skill_evolution = None
        if config.enable_self_improving:
            try:
                from .skill_evolution import SkillEvolutionManager
                self._skill_evolution = SkillEvolutionManager(
                    event_bus=self.event_bus,
                    work_dir=self.work_dir,
                    llm_provider=self.llm,
                    config=config.skill_evolution,
                )
            except Exception as e:
                logger.warning("skill_evolution_init_failed", error=str(e))

        # Initialize context evolution system (gated by enable_self_improving)
        self._context_evolution = None
        if config.enable_self_improving:
            try:
                from .context_evolution import ContextEvolutionManager
                self._context_evolution = ContextEvolutionManager(
                    event_bus=self.event_bus,
                    work_dir=self.work_dir,
                    llm_provider=self.llm,
                    config=config.context_evolution,
                )
            except Exception as e:
                logger.warning("context_evolution_init_failed", error=str(e))

        # Initialize RL module (ONLY for local providers: vllm/sglang)
        self._rl_active = config.agent.model_provider in ("vllm", "sglang")
        if self._rl_active and config.rl.enabled:
            try:
                from omniagent.rl import RLAPIServer
                logger.info(
                    "rl_module_available",
                    provider=config.agent.model_provider,
                )
            except Exception as e:
                logger.warning("rl_init_failed", error=str(e))

        # Initialize Sentinel agent (task decomposition + progress tracking)
        self._sentinel = None
        if config.sentinel.enabled:
            try:
                from .sentinel import SentinelAgent
                self._sentinel = SentinelAgent(
                    config=config.sentinel,
                    main_agent_config=config.agent,
                    work_dir=str(self.work_dir),
                )
                logger.info("sentinel_agent_initialized")
            except Exception as e:
                logger.warning("sentinel_init_failed", error=str(e))

        # Initialize Guardian agent (output quality review + safety gate)
        self._guardian = None
        self._guardian_blocked_events: List[Dict[str, Any]] = []
        if config.guardian.enabled:
            try:
                from .guardian import GuardianAgent
                self._guardian = GuardianAgent(
                    config=config.guardian,
                    main_agent_config=config.agent,
                )
                logger.info("guardian_agent_initialized")
            except Exception as e:
                logger.warning("guardian_init_failed", error=str(e))

        # Progressive loading: context hints from memory searches (L0 injection)
        self._context_hints: List[Dict[str, Any]] = []

        # Context assembler for token budget management
        token_budget = int(resolved_window * config.agent.system_prompt_token_ratio)
        self.context_assembler = ContextAssembler(token_budget=token_budget)

        # Reflexion: accumulated reflections from failed attempts
        self._reflections: List[str] = []
        self._discoveries: str = ""  # key files/paths discovered in previous attempt

        # Self-check: track tool call names per attempt for no-progress detection
        self._tool_name_history: List[str] = []
        self._result_hashes: List[str] = []

        # Guardian annotations: separated from tool results, injected as temp messages
        self._guardian_annotations: List[str] = []

        # Feature activity tracking (for display in chat)
        self._parallel_exec_count: int = 0       # number of parallel batches executed
        self._parallel_tools_count: int = 0      # total tools executed in parallel
        self._compaction_count: int = 0          # context compaction triggered count
        self._stuck_detected_count: int = 0      # stuck/loop/no-progress detection count
        self._sentinel_activated: bool = False    # sentinel was activated this execution

        # Steering: user-injected messages during execution
        self._steering_queue: List[str] = []

        # transformContext: optional async hook to modify messages before LLM call
        self._transform_context: Optional[Callable] = None

        # Ensure bootstrap files (AGENTS.md, SOUL.md, etc.) exist
        from omniagent.agents.bootstrap import BootstrapFiles
        self._bootstrap = BootstrapFiles(work_dir=self.work_dir)
        self._bootstrap.ensure_bootstrap_files()

        logger.info(
            "agent_initialized",
            max_iterations=self.max_iterations,
            model=config.agent.model_id,
            tools_count=len(self.tools),
            security_enabled=enable_security,
            native_fc=self.llm.supports_native_function_calling,
        )

    def steer(self, message: str) -> None:
        """Inject a steering message into the agent's execution loop."""
        self._steering_queue.append(message)

    def set_transform_context(
        self, fn: Callable[[List[LLMMessage]], Any]
    ) -> None:
        """Set an async hook to transform messages before LLM call."""
        self._transform_context = fn

    def _register_default_tools(self) -> None:
        """Register default tools."""
        # File tools
        self.registry.register(ReadTool(work_dir=self.work_dir))
        self.registry.register(WriteTool(work_dir=self.work_dir))
        self.registry.register(EditTool(work_dir=self.work_dir))

        # Bash tool
        self.registry.register(BashTool(work_dir=self.work_dir, allow_dangerous=False))

        # JSON tools
        self.registry.register(LoadJSONTool(work_dir=self.work_dir))
        self.registry.register(SaveJSONTool(work_dir=self.work_dir))

        # Process tools
        self.registry.register(ProcessListTool())
        self.registry.register(ProcessKillTool())

        # Web tools
        self.registry.register(WebSearchTool())
        self.registry.register(WebFetchTool())

        # Search/navigation tools
        self.registry.register(GrepTool(work_dir=self.work_dir))
        self.registry.register(FindTool(work_dir=self.work_dir))
        self.registry.register(LsTool(work_dir=self.work_dir))

        # Diff tool
        self.registry.register(DiffTool(work_dir=self.work_dir))

        # HTTP tool
        self.registry.register(HttpTool())

        # Build tools dict
        for tool in self.registry.list_tools():
            self.tools[tool.name] = tool

        logger.info("default_tools_registered", count=len(self.tools))

    def register_tool(self, tool: Tool) -> None:
        """
        Register a custom tool.

        Args:
            tool: Tool instance
        """
        self.registry.register(tool)
        self.tools[tool.name] = tool
        logger.info("tool_registered", name=tool.name)

    def clear_history(self) -> None:
        """Clear conversation history and reset all transient state."""
        self.conversation_history = []
        self._recent_tool_calls = []
        self._recent_errors = []
        self._context_hints = []
        self._reflections = []
        self._tool_name_history = []
        self._result_hashes = []
        self._steering_queue = []
        self._guardian_annotations = []
        self.agent_state = AgentState()
        self.abort_controller.reset()
        logger.info("conversation_history_cleared")

    def get_history(self) -> List[LLMMessage]:
        """Get a copy of the conversation history."""
        return list(self.conversation_history)

    def _build_tool_schemas(self) -> List[Dict[str, Any]]:
        """Build OpenAI-format tool schemas for all registered tools."""
        schemas = []
        for tool in self.registry.list_tools():
            param_schema = tool._get_parameters_schema()
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": param_schema,
                },
            })
        return schemas

    async def handle_message(self, message: IncomingMessage) -> OutgoingMessage:
        """Handle incoming message."""
        logger.info(
            "handling_message",
            session_id=message.session_id,
            user_id=message.user_id,
        )

        # Sync memory on first message if configured
        if self.memory_manager and self.config.memory.sync_on_session_start:
            try:
                updated = await self.memory_manager.sync()
                if updated:
                    logger.info("memory_synced_on_session_start", updated_files=updated)
            except Exception as e:
                logger.warning("memory_sync_failed", error=str(e))

        # Execute task
        result = await self.execute(message.content)

        # Create response
        return OutgoingMessage(
            session_id=message.session_id or "unknown",
            content=result.response,
            metadata={
                "success": result.success,
                **result.metadata,
            },
        )

    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """
        Execute task with Reflexion retry on failure.

        If the initial attempt fails, the agent self-reflects on what went wrong
        and retries with the reflection injected into the system prompt.

        Args:
            task: Task description
            context: Optional context

        Returns:
            Agent result (from best attempt)
        """
        logger.info("executing_task", task=task)

        # Reset per-execution state
        self._guardian_blocked_events = []
        self._parallel_exec_count = 0
        self._parallel_tools_count = 0
        self._compaction_count = 0
        self._stuck_detected_count = 0
        self._sentinel_activated = False
        self._tool_name_history = []
        self._result_hashes = []

        # ── Feedback Detection ──────────────────────────────────────
        # If the user's message looks like feedback on a prior execution
        # (corrections, preferences, "don't do X, do Y"), route it through
        # the lesson extraction pipeline before executing as a new task.
        self._pending_user_feedback: Optional[List[str]] = None
        if self._context_evolution and self.conversation_history:
            feedback_result = await self._detect_user_feedback(task)
            if feedback_result:
                self._pending_user_feedback = [feedback_result]
                logger.info(
                    "user_feedback_detected",
                    feedback=feedback_result[:100],
                )

        # Sentinel: check if task is complex enough to activate planning
        # Pass skills_summary so Sentinel can make skill-aware decisions
        sentinel_plan = None
        skills_summary = self.skill_manager.format_skills_summary()
        if self._sentinel:
            # Try to recover existing plan first
            sentinel_plan = self._sentinel.load_plan(task)
            if not sentinel_plan:
                should_activate, reason = await self._sentinel.should_activate_with_llm(
                    task, self.llm,
                    skills_summary=skills_summary,
                )
                if should_activate:
                    self._sentinel_activated = True
                    logger.info("sentinel_activating", reason=reason)
                    try:
                        sentinel_plan = await self._sentinel.decompose(
                            task, self.llm,
                            skills_summary=skills_summary,
                        )
                        # Inject plan context into conversation
                        plan_text = self._sentinel.get_progress_summary()
                        if plan_text:
                            self._steering_queue.append(
                                f"[Sentinel Plan] Complex task detected. "
                                f"Working through the following milestone plan:\n{plan_text}\n"
                                f"Execute the milestones in order. Mark each as complete before moving on."
                            )
                    except Exception as e:
                        logger.warning("sentinel_decompose_failed", error=str(e))
            else:
                # Recovered plan — inject progress context
                plan_text = self._sentinel.get_progress_summary()
                if plan_text:
                    self._steering_queue.append(
                        f"[Sentinel Plan Recovery] Resuming previous task plan:\n{plan_text}\n"
                        f"Continue from where you left off."
                    )
        self._sentinel_plan = sentinel_plan

        # First attempt
        result = await self._execute_single_attempt(task)

        if result.success or not self.config.agent.reflexion_enabled:
            # Sentinel: mark milestone completed on success
            if self._sentinel and self._sentinel.is_active:
                try:
                    ms = self._sentinel.get_current_milestone()
                    if ms:
                        await self._sentinel.mark_milestone_completed(
                            ms, result_summary=result.response[:200]
                        )
                except Exception as e:
                    logger.warning("sentinel_milestone_complete_failed", error=str(e))
            return result

        # Reflexion retry loop
        max_retries = self.config.agent.reflexion_max_attempts
        for retry in range(max_retries):
            # Sentinel: activate planning on repeated reflexion failures
            if self._sentinel and not self._sentinel_activated:
                should, reason = self._sentinel.should_activate(
                    task, reflexion_failure_count=retry + 1
                )
                if should:
                    self._sentinel_activated = True
                    logger.info("sentinel_activating_on_reflexion", reason=reason, attempt=retry + 1)
                    try:
                        self._sentinel_plan = await self._sentinel.decompose(task, self.llm)
                        plan_text = self._sentinel.get_progress_summary()
                        if plan_text:
                            self._steering_queue.append(
                                f"[Sentinel Plan] Activated due to repeated failures. "
                                f"Working through the following milestone plan:\n{plan_text}\n"
                                f"Execute the milestones in order. Mark each as complete before moving on."
                            )
                    except Exception as e:
                        logger.warning("sentinel_decompose_on_reflexion_failed", error=str(e))

            logger.info(
                "reflexion_retry",
                attempt=retry + 1,
                max_retries=max_retries,
                error=result.error or "max_iterations",
            )

            # Generate reflection on what went wrong
            reflection = await self._reflect(task, result)
            if reflection:
                self._reflections.append(reflection)
                logger.info("reflection_generated", length=len(reflection))

            # Extract key discoveries before clearing conversation
            self._discoveries = self._extract_discoveries()
            if self._discoveries:
                logger.info("discoveries_extracted", files=self._discoveries.count("\n") + 1)

            # Reset conversation for fresh attempt, keeping reflections and discoveries
            self.conversation_history = []
            self._recent_tool_calls = []
            self._recent_errors = []
            self._tool_name_history = []
            self._result_hashes = []
            # Keep _context_hints, _reflections, and _discoveries across retries

            # Retry with reflection injected into system prompt
            result = await self._execute_single_attempt(task)
            if result.success:
                logger.info("reflexion_succeeded", attempt=retry + 1)
                break

        result.metadata["reflections_count"] = len(self._reflections)

        # ── Collect execution highlights ──
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        highlights = []

        # Reflections
        if self._reflections:
            highlights.append({"type": "reflections", "count": len(self._reflections), "time": now})

        # Guardian session summary
        if self._guardian:
            summary = self._guardian.get_session_summary()
            if summary:
                highlights.append({"type": "guardian", "summary": summary, "time": now})

        # Tool stats
        tools_used = list(dict.fromkeys(self._tool_name_history))
        if tools_used:
            highlights.append({"type": "tools", "used": tools_used, "total_calls": self.agent_state.total_tool_calls, "time": now})

        # Skill evolution highlights (time from AGENT_END handler)
        if self._skill_evolution and self._skill_evolution.last_session_results:
            sr = self._skill_evolution.last_session_results
            ev_time = sr.get("time", now)
            if sr.get("skill_compiled"):
                highlights.append({"type": "skill_compiled", "time": ev_time})
            if sr.get("patches_written"):
                highlights.append({"type": "skill_evolution", "time": ev_time})

        # Context evolution highlights (time from AGENT_END handler)
        if self._context_evolution and self._context_evolution.last_session_results:
            cr = self._context_evolution.last_session_results
            ev_time = cr.get("time", now)
            if cr.get("rules_promoted"):
                highlights.append({"type": "rules_promoted", "count": cr["rules_promoted"], "time": ev_time})
            if cr.get("lessons_extracted"):
                highlights.append({"type": "lessons_extracted", "count": cr["lessons_extracted"], "time": ev_time})

        result.metadata["execution_highlights"] = highlights

        # Invalidate skill cache so trial skills become visible
        if self._skill_evolution and self.skill_manager:
            self.skill_manager.invalidate_cache()

        return result

    async def _execute_single_attempt(
        self,
        task: str,
    ) -> AgentResult:
        """
        Execute a single attempt using native function calling.

        Supports:
        - Native function calling via response.tool_calls
        - Parallel tool execution for independent calls
        - Self-check: token budget, no-progress detection
        - Steering messages from user
        - Auto-retry with exponential backoff on transient errors
        - Tool hooks (before/after)
        - Abort controller
        """
        logger.info("executing_task", task=task)

        # Clear transient annotations from previous attempt
        self._guardian_annotations = []

        tool_schemas = self._build_tool_schemas()

        # Build system prompt (always fresh)
        system_prompt = self._build_system_prompt()

        # Update skill evolution references
        if self._skill_evolution:
            active_skills = [s.name for s in self.skill_manager.discover_skills()]
            self._skill_evolution.set_agent_refs(
                conversation_history=self.conversation_history,
                tool_name_history=self._tool_name_history,
                active_skills=active_skills,
                user_feedback=self._pending_user_feedback,
            )
            self._skill_evolution.set_skill_manager(self.skill_manager)

        # Update context evolution references
        if self._context_evolution:
            self._context_evolution.set_agent_refs(
                conversation_history=self.conversation_history,
                reflections=self._reflections,
                error="",
                user_feedback=self._pending_user_feedback,
            )

        # Append new user message to history
        self.conversation_history.append(
            LLMMessage(role="user", content=task, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )

        # Build full message list: system + history
        messages = [LLMMessage(role="system", content=system_prompt)]
        messages.extend(self.conversation_history)

        # Emit AGENT_START
        await self.event_bus.emit(AgentEvent(type=EventType.AGENT_START, data={"task": task}))

        # Agent loop
        overflow_attempts = 0
        max_overflow_attempts = 3
        for iteration in range(self.max_iterations):
            logger.debug("agent_iteration", iteration=iteration + 1)
            self.agent_state.iteration = iteration + 1

            # Emit TURN_START
            await self.event_bus.emit(AgentEvent(
                type=EventType.TURN_START,
                data={"iteration": iteration + 1},
            ))

            # Check steering queue
            if self._steering_queue:
                steering_msgs = list(self._steering_queue)
                self._steering_queue = []
                for sm in steering_msgs:
                    self._append_user_message(messages, f"[Steering] {sm}", also_history=True)

            # Inject guardian annotations as temporary messages (not persisted to conversation_history)
            if self._guardian_annotations:
                annotation_text = "\n".join(self._guardian_annotations)
                self._guardian_annotations = []
                self._append_user_message(messages, f"[Safety Note] {annotation_text}", also_history=False)

            # Apply transform_context hook
            if self._transform_context:
                messages = await self._transform_context(messages)

            try:
                # Check abort at start of each iteration
                self.abort_controller.check()
                # Check if compaction is needed
                if self.context_manager.needs_compaction(messages):
                    self._compaction_count += 1
                    logger.info("context_compaction_triggered")
                    await self.event_bus.emit(AgentEvent(type=EventType.COMPACTION_START))
                    messages = await self.context_manager.compact(messages)
                    self.conversation_history = [
                        m for m in messages if m.role != "system"
                    ]
                    await self.event_bus.emit(AgentEvent(type=EventType.COMPACTION_END))

                # Get LLM response with auto-retry on transient errors
                response = await self._execute_with_retry(
                    messages, tool_schemas,
                    max_retries=3, base_delay=1.0,
                )

                content = response.content or ""
                # Log display: when content is empty but tool_calls exist, show placeholder
                display_content = content if content else (
                    "[tool_calls only]" if response.tool_calls else "[empty]"
                )
                logger.info(
                    "llm_response",
                    content_length=len(content),
                    content=display_content,
                    tool_calls=response.tool_calls,
                    finish_reason=response.finish_reason,
                )

                # === Native function calling ===
                if response.tool_calls:
                    actions = self._convert_native_tool_calls(response.tool_calls)

                    # If LLM requested tools but all were unknown, feed errors back
                    if not actions and response.tool_calls:
                        logger.warning(
                            "all_tool_calls_unknown",
                            requested=[tc.get("function", {}).get("name") for tc in response.tool_calls],
                        )
                        # Add assistant message with tool_calls so the conversation makes sense
                        assistant_msg = LLMMessage(
                            role="assistant",
                            content=content if content else None,
                            tool_calls=response.tool_calls,
                            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        )
                        messages.append(assistant_msg)
                        self.conversation_history.append(assistant_msg)
                        # Feed error for each unknown tool call
                        for tc in response.tool_calls:
                            fn = tc.get("function", {})
                            tool_name = fn.get("name", "unknown")
                            tool_call_id = tc.get("id", "")
                            error_msg = LLMMessage(
                                role="tool",
                                content=f"Error: Tool '{tool_name}' is not available. Available tools: {list(self.tools.keys())}",
                                tool_call_id=tool_call_id,
                                name=tool_name,
                            )
                            messages.append(error_msg)
                            self.conversation_history.append(error_msg)
                        # Let the LLM retry
                        await self.event_bus.emit(AgentEvent(
                            type=EventType.TURN_END,
                            data={"iteration": iteration + 1, "unknown_tools": True},
                        ))
                        continue

                    if actions:
                        logger.info("tool_calls", count=len(actions))

                        # Add assistant message with tool_calls
                        assistant_msg = LLMMessage(
                            role="assistant",
                            content=content if content else "",
                            tool_calls=response.tool_calls,
                            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        )
                        messages.append(assistant_msg)
                        self.conversation_history.append(assistant_msg)

                        # Self-check: no-progress detection
                        self_check_msg = self._check_no_progress(actions)
                        if self_check_msg:
                            self._append_user_message(messages, self_check_msg, also_history=True)

                        # Execute tools with hooks
                        tool_messages = await self._execute_actions(actions)

                        for tool_msg in tool_messages:
                            messages.append(tool_msg)
                            self.conversation_history.append(tool_msg)

                        # Proactively trim old tool results to prevent context bloat
                        if len(self.conversation_history) > 4:
                            self._trim_old_tool_results()

                        # Emit TURN_END
                        await self.event_bus.emit(AgentEvent(
                            type=EventType.TURN_END,
                            data={"iteration": iteration + 1, "tool_calls": len(actions)},
                        ))
                        continue

                # No tool calls found, this is the final response
                self.conversation_history.append(
                    LLMMessage(role="assistant", content=content, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                )
                logger.info("task_completed", iterations=iteration + 1)

                await self.event_bus.emit(AgentEvent(
                    type=EventType.TURN_END,
                    data={"iteration": iteration + 1, "final": True},
                ))
                await self.event_bus.emit(AgentEvent(type=EventType.AGENT_END, data={"success": True, "iterations": iteration + 1}))

                # Get timestamps from conversation history
                user_msg = self.conversation_history[-2] if len(self.conversation_history) >= 2 else None
                query_ts = user_msg.timestamp if user_msg and user_msg.role == "user" else None
                response_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                return AgentResult(
                    success=True,
                    response=content,
                    metadata={
                        "iterations": iteration + 1,
                        "usage": response.usage,
                        "query_timestamp": query_ts,
                        "response_timestamp": response_ts,
                    },
                )

            except AbortError:
                await self.event_bus.emit(AgentEvent(type=EventType.AGENT_END, data={"aborted": True}))
                return AgentResult(
                    success=False,
                    response="Agent execution was aborted.",
                    error="aborted",
                    metadata={"iterations": iteration + 1},
                )

            except Exception as e:
                error_str = str(e)
                logger.error("agent_iteration_failed", error=error_str)

                # Context overflow recovery
                if (
                    detect_context_overflow(error_str)
                    and overflow_attempts < max_overflow_attempts
                    and self.context_manager.llm
                ):
                    overflow_attempts += 1
                    logger.warning(
                        "context_overflow_detected",
                        attempt=overflow_attempts,
                        max_attempts=max_overflow_attempts,
                    )
                    try:
                        messages = await self.context_manager.compact(messages)
                        self.conversation_history = [
                            m for m in messages if m.role != "system"
                        ]
                        continue  # Retry after compaction
                    except Exception as compact_err:
                        logger.error(
                            "overflow_compaction_failed",
                            compact_error=str(compact_err),
                        )

                self.agent_state.error = error_str
                await self.event_bus.emit(AgentEvent(type=EventType.AGENT_END, data={"error": error_str}))
                return AgentResult(
                    success=False,
                    response=f"Error: {str(e)}",
                    error=str(e),
                    metadata={"iterations": iteration + 1},
                )

        # Max iterations reached
        logger.warning("max_iterations_reached")
        await self.event_bus.emit(AgentEvent(type=EventType.AGENT_END, data={"error": "max_iterations_reached"}))
        return AgentResult(
            success=False,
            response="Maximum iterations reached without completing the task.",
            error="max_iterations_reached",
            metadata={"iterations": self.max_iterations},
        )

    async def _execute_with_retry(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict]],
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> "LLMResponse":
        """Call LLM with exponential backoff on transient errors."""
        last_error = None
        for attempt in range(max_retries):
            try:
                response = await self.llm.chat(
                    messages=messages,
                    temperature=self.config.agent.temperature,
                    max_tokens=self.config.agent.max_tokens,
                    tools=tools,
                )
                return response
            except Exception as e:
                last_error = e
                if not self._is_retryable_error(e) or attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "llm_retry",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    delay=delay,
                    error=str(e),
                )
                await asyncio.sleep(delay)
        raise last_error  # type: ignore

    @staticmethod
    def _is_retryable_error(error: Exception) -> bool:
        """Check if an error is transient and worth retrying."""
        error_str = str(error).lower()
        # HTTP status codes
        retryable_codes = ["429", "500", "502", "503", "504"]
        for code in retryable_codes:
            if code in error_str:
                return True
        # Network errors
        retryable_patterns = [
            "connection refused", "connection reset", "connection aborted",
            "timeout", "timed out", "temporary failure", "rate limit",
        ]
        return any(p in error_str for p in retryable_patterns)

    def _convert_native_tool_calls(self, tool_calls: List[Dict]) -> List[Dict[str, Any]]:
        """Convert OpenAI-style tool_calls to internal action format.

        Input: [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "{...}"}}]
        Output: [{"tool": "...", "params": {...}, "tool_call_id": "..."}]
        """
        actions = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            if name not in self.tools:
                logger.warning("native_fc_unknown_tool", tool=name)
                continue

            try:
                params = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                logger.warning("native_fc_invalid_args", tool=name, arguments=fn.get("arguments"))
                params = {}

            actions.append({
                "tool": name,
                "params": params,
                "tool_call_id": tc.get("id", ""),
            })
        return actions

    def _check_tool_loop(self, action: Dict[str, Any]) -> bool:
        """
        Check if this tool call is a repeat of recent calls.
        Returns True if a loop is detected (should skip execution).
       ts using SHA-256 hashing.
        """
        call_signature = hashlib.sha256(
            json.dumps(action, sort_keys=True).encode()
        ).hexdigest()

        self._recent_tool_calls.append(call_signature)

        # Keep only recent window
        if len(self._recent_tool_calls) > self._loop_detection_window:
            self._recent_tool_calls = self._recent_tool_calls[-self._loop_detection_window:]

        # Check if same call repeated >= threshold times consecutively
        if len(self._recent_tool_calls) >= self._loop_repeat_threshold:
            recent_tail = self._recent_tool_calls[-self._loop_repeat_threshold:]
            if len(set(recent_tail)) == 1 and recent_tail[0] == call_signature:
                self._stuck_detected_count += 1
                logger.warning(
                    "tool_loop_detected",
                    tool=action["tool"],
                    repeat_count=self._loop_repeat_threshold,
                )
                return True

        return False

    def _check_error_repeat(self, error_msg: str) -> bool:
        """
        Check if this same error has occurred before in the conversation.
        Returns True if it's a repeat error.
        """
        normalized = error_msg.strip()
        if normalized in self._recent_errors:
            self._stuck_detected_count += 1
            logger.warning("error_repeat_detected", error=normalized[:80])
            return True
        self._recent_errors.append(normalized)
        # Keep only last 20 errors
        if len(self._recent_errors) > 20:
            self._recent_errors = self._recent_errors[-20:]
        return False

    def _build_system_prompt(self) -> str:
        """Build system prompt using progressive context loading.

        Uses ContextAssembler to manage token budget and inject content by priority:
        Fixed base -> Bootstrap L0 -> Tools L0 -> Skills L0 -> Memory hints -> Bootstrap L1 -> Tools L1
        """
        fixed = f"""You are a OmniAgent AI assistant powered by {self.config.agent.model_id} (via {self.config.agent.model_provider}), with access to tools.
Always truthfully identify yourself as OmniAgent running {self.config.agent.model_id}. Never claim to be Claude, GPT, or any other model.

## Language
- Respond in the same language as the user's input (Chinese input → Chinese response, English input → English response).
- Do not mix languages unless the user explicitly does so.
- Do not output thinking/reasoning steps in your response — only the final answer.

## Tool Usage
You have access to tools via native function calling. When you need to use a tool,
simply call it with the appropriate parameters. The system will execute the tool
and return results for you.

Do not narrate routine tool calls - just call them directly.
Only explain your reasoning when it helps understanding complex problems.

## Safety
You have no independent goals: do not pursue self-preservation, replication, resource acquisition, or power-seeking.
Prioritize safety and human oversight over completion.
Do not manipulate or persuade anyone to expand access or disable safeguards.

## Workspace
Your working directory is: {self.work_dir}

## Project Navigation
- Source code: `omniagent/` — agents, channels, tools, config, security
- Channel implementations: `omniagent/channels/` (e.g. feishu.py, webhook.py, discord.py, telegram.py)
- Tool implementations: `omniagent/tools/` (e.g. file_tools.py, grep_tool.py, bash_tool.py)
- Configuration models: `omniagent/config/models.py`
- Configuration loader: `omniagent/config/loader.py`
- Skills: `skills/`, `.omniagent/skills/`, `~/.omniagent/skills/` — each has a SKILL.md
- Memory files: `MEMORY.md`, `memory/*.md`
- Bootstrap files: `AGENTS.md`, `SOUL.md`, `CUSTOM.md` (in `.omniagent/`)

When asked about configuration or implementation details, use read_file to check the relevant file directly instead of searching broadly.
"""

        # Add memory recall instructions if memory system is enabled
        if self.memory_manager:
            fixed += """
## Memory Recall
Before answering anything about prior work, decisions, dates, people, preferences,
or todos: run memory_search on MEMORY.md + memory/*.md; then use memory_get to
pull only the needed lines. If low confidence after search, say you checked.
Citations: include Source: <path#line> when it helps the user verify memory snippets.
"""

        # Build tool info for ContextAssembler
        tools_info = []
        for tool in self.registry.list_tools():
            schema = tool._get_parameters_schema()
            tools_info.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": schema or {},
            })

        # Build skills L0 (compact format)
        skills_xml = self.skill_manager.format_skills_for_prompt()
        skills_l0 = ContextAssembler.format_skills_l0(skills_xml)

        # Build memory hints (L0) from recent search results
        memory_hints = ContextAssembler.format_memory_hints(self._context_hints)
        # Clear hints after injection to avoid stale data
        self._context_hints = []

        # Build reflection hints from previous failed attempts
        reflection_hints = self._format_reflections()

        # Load bootstrap files and generate L0/L1
        bootstrap_ctx = self._bootstrap.load_for_prompt()

        # Inject agent identity and user profile into fixed prompt
        if bootstrap_ctx.identity:
            agent_name = bootstrap_ctx.identity.get("name", "OmniAgent")
            if agent_name != "OmniAgent":
                fixed = fixed.replace(
                    f"You are a OmniAgent AI assistant powered by {self.config.agent.model_id}",
                    f"You are {agent_name}, a OmniAgent AI assistant powered by {self.config.agent.model_id}",
                )

        # Handle system_override from CUSTOM.md
        if bootstrap_ctx.system_override:
            fixed = bootstrap_ctx.system_override
            bootstrap_l0 = ""
        else:
            bootstrap_l0 = self._bootstrap.format_l0(bootstrap_ctx.files) if bootstrap_ctx.files else ""

        # Assemble within token budget (reflection_hints now respects budget)
        prompt = self.context_assembler.assemble(
            fixed_prompt=fixed,
            tools=tools_info,
            skills_l0=skills_l0,
            memory_hints=memory_hints,
            bootstrap_l0=bootstrap_l0,
            reflection_hints=reflection_hints,
        )
        logger.debug("system_prompt", prompt=prompt)
        return prompt

    # ── Feedback Detection ────────────────────────────────────────────

    async def _detect_user_feedback(
        self, current_task: str
    ) -> Optional[str]:
        """Detect if the current task is user feedback on prior execution.

        Uses an LLM call to classify whether the message contains corrections,
        preferences, or behavioral guidance about a previous task.

        Returns the feedback text if detected, None otherwise.
        """
        from omniagent.agents.llm import LLMMessage

        # Build a compact summary of the previous execution for context
        prev_summary = self._summarize_previous_execution()
        if not prev_summary:
            return None

        prompt = (
            "Determine if the user's latest message is **feedback/correction "
            "about a prior execution** (not a new standalone task).\n\n"
            f"Previous execution summary:\n{prev_summary}\n\n"
            f"User's latest message:\n{current_task[:500]}\n\n"
            'Respond in JSON only:\n'
            '{"is_feedback": true/false, '
            '"feedback_summary": "the user correction/preference in one sentence"}\n\n'
            "Examples of feedback:\n"
            '- "不要用文字分析，用流程图" -> is_feedback: true\n'
            '- "下次别用这个方法了" -> is_feedback: true\n'
            '- "你分析错了，应该是..." -> is_feedback: true\n'
            '- "帮我写个xxx功能" -> is_feedback: false\n'
            '- "分析一下这段代码" -> is_feedback: false'
        )

        try:
            response = await self.llm.chat(
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.0,
                max_tokens=200,
            )
            content = (response.content or "").strip()
            json_match = re.search(r"\{[^}]+\}", content, re.DOTALL)
            if not json_match:
                return None

            data = json.loads(json_match.group())
            if data.get("is_feedback") is True:
                summary = data.get("feedback_summary", "").strip()
                if summary and len(summary) >= 5:
                    return summary
        except Exception as e:
            logger.debug("feedback_detection_failed", error=str(e))

        return None

    def _summarize_previous_execution(self) -> str:
        """Build a compact summary of the most recent execution for feedback detection."""
        if not self.conversation_history:
            return ""

        events: List[str] = []
        for msg in self.conversation_history[-20:]:
            if msg.role == "user" and msg.content:
                events.append(f"User asked: {msg.content[:200]}")
            elif msg.role == "assistant":
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        fn = tc.get("function", {})
                        events.append(f"Called tool: {fn.get('name', '?')}")
                elif msg.content:
                    events.append(f"Assistant: {msg.content[:200]}")
            elif msg.role == "tool":
                content = msg.content or ""
                if content.startswith("Error:"):
                    events.append(f"Tool error: {content[:150]}")

        summary = "\n".join(events[-10:])
        if len(summary) > 2000:
            summary = summary[-2000:]
        return summary

    # ── Parallel Tool Execution ────────────────────────────────────────

    @staticmethod
    def _is_dependent_action(action: Dict[str, Any]) -> bool:
        """Check if an action should be executed serially (has side effects).

        Bash commands and write operations are treated as dependent because
        they may modify shared state that other tools depend on.
        """
        tool_name = action.get("tool", "")
        # Bash and write-type tools have side effects
        return tool_name in ("bash", "write_file", "edit_file", "save_json", "process_kill")

    @staticmethod
    def _is_safe_bash_command(command: str) -> bool:
        """Check if a bash command is safe (read-only, no side effects).

        Safe commands: ls, find, grep, cat, head, tail, wc, echo, pwd,
        which, type, file, stat, date, whoami, uname, env (read-only),
        and their chained versions with pipes/redirects to /dev/null.
        """
        import re as _re

        cmd = command.strip()
        if not cmd:
            return True

        # Block anything with shell operators that could cause side effects
        dangerous_patterns = [
            r'\brm\b', r'\bmv\b', r'\bcp\b', r'\bchmod\b', r'\bchown\b',
            r'\bmkdir\b', r'\brmdir\b', r'\btouch\b', r'\btee\b',
            r'\bsudo\b', r'\bsu\b', r'\bkill\b', r'\bkillall\b',
            r'\bsh\b', r'\bbash\b', r'\bzsh\b', r'\beval\b', r'\bexec\b',
            r'\bsource\b', r'\bdot\b', r'\bwget\b', r'\bcurl\b',
            r'\bpython\b', r'\bnode\b', r'\bnpm\b', r'\bpip\b',
            r'\bgit\s+(push|reset|clean|checkout\b(?!\s--))',
            r'>\s*(?!/dev/null)',  # redirect to file (not /dev/null)
            r'\$\(',  # command substitution
        ]
        for pat in dangerous_patterns:
            if _re.search(pat, cmd):
                return False

        # Allow known safe commands
        safe_commands = {
            'ls', 'find', 'grep', 'cat', 'head', 'tail', 'wc',
            'echo', 'pwd', 'which', 'type', 'file', 'stat',
            'date', 'whoami', 'uname', 'id', 'env', 'printenv',
            'sort', 'uniq', 'cut', 'tr', 'diff', 'wc',
            'basename', 'dirname', 'realpath', 'readlink',
        }
        first_word = cmd.split()[0] if cmd.split() else ''
        base_cmd = first_word.split('/')[-1]  # handle /usr/bin/ls
        return base_cmd in safe_commands

    async def _execute_single_action(
        self, action: Dict[str, Any]
    ) -> LLMMessage:
        """Execute a single tool action with all checks applied.

        Returns an LLMMessage with the tool result.
        """
        tool_call_id = action.get("tool_call_id", "")
        tool_name = action["tool"]
        params = action["params"]

        # Track tool name for no-progress detection
        self._tool_name_history.append(tool_name)

        # Emit TOOL_EXECUTION_START
        await self.event_bus.emit(AgentEvent(
            type=EventType.TOOL_EXECUTION_START,
            data={"tool": tool_name, "params": params, "call_id": tool_call_id},
        ))
        self.agent_state.total_tool_calls += 1
        self.agent_state.pending_tool_calls.add(tool_call_id)

        result_content = await self._guarded_execute(action)

        # Run after-tool hooks (hook_ctx from before-tool phase in _guarded_execute)
        after_hook_ctx = ToolCallContext(
            tool_name=tool_name, params=params,
            call_id=tool_call_id,
        )
        result_content = await self.tool_hook_manager.run_after(after_hook_ctx, result_content)

        self.agent_state.pending_tool_calls.discard(tool_call_id)

        # Emit TOOL_EXECUTION_END
        await self.event_bus.emit(AgentEvent(
            type=EventType.TOOL_EXECUTION_END,
            data={"tool": tool_name, "call_id": tool_call_id, "length": len(result_content)},
        ))

        return LLMMessage(
            role="tool",
            content=result_content,
            tool_call_id=tool_call_id,
            name=tool_name,
        )

    async def _guarded_execute(self, action: Dict[str, Any]) -> str:
        """Execute a tool action with hooks, loop detection, and Guardian review.

        Returns the result content string.
        """
        tool_name = action["tool"]
        params = action["params"]

        # Step 1: Run before-tool hooks
        hook_ctx = ToolCallContext(
            tool_name=tool_name, params=params,
            call_id=action.get("tool_call_id", ""),
        )
        hook_result = await self.tool_hook_manager.run_before(hook_ctx)
        if hook_result.blocked:
            return f"[BLOCKED] {hook_result.block_reason or 'Tool call blocked by policy'}"

        # Step 2: Tool loop detection
        if self._check_tool_loop(action):
            return (
                f"[LOOP DETECTED] Tool '{tool_name}' with the same "
                f"parameters has been called {self._loop_repeat_threshold} times consecutively. "
                f"This approach is not working. Try a different tool or different parameters."
            )

        # Step 3: Guardian review (LLM-powered, for high-impact operations)
        guardian_notes = ""
        guardian_review_result = None
        if self._guardian:
            should_review, _reason = self._guardian.should_activate_for_tool_call(
                tool_name=tool_name,
                tool_params=params,
            )
            if should_review:
                try:
                    context = "\n".join(
                        f"{m.role}: {m.content[:200] if m.content else '[tool_calls]'}"
                        for m in self.conversation_history[-6:]
                    )
                    review = await self._guardian.review(
                        tool_name, params, context, self.llm
                    )
                    guardian_review_result = review

                    # Only auto-block on critical risk with auto_block_on_critical
                    if not review.passed and review.risk_level == "critical" and self._guardian.config.auto_block_on_critical:
                        findings = "; ".join(review.findings) if review.findings else "Risk assessed"
                        suggestions = "\n".join(f"  - {s}" for s in review.suggestions) if review.suggestions else ""
                        blocked_msg = (
                            f"[GUARDIAN BLOCKED] Risk level: {review.risk_level}\n"
                            f"Findings: {findings}\n"
                        )
                        if suggestions:
                            blocked_msg += f"Suggestions:\n{suggestions}\n"
                        logger.warning(
                            "guardian_blocked", tool=tool_name,
                            risk=review.risk_level, findings=review.findings,
                        )
                        self._guardian_blocked_events.append({
                            "tool": tool_name,
                            "params_summary": str(params)[:200],
                            "risk_level": review.risk_level,
                            "findings": review.findings,
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        })
                        return blocked_msg

                    # For non-critical risks, pass findings to approval flow instead of blocking
                    if not review.passed:
                        guardian_notes = (
                            f"\n\n[GUARDIAN WARNING] Risk level: {review.risk_level}\n"
                            f"Findings: {'; '.join(review.findings) if review.findings else 'Risk assessed'}\n"
                        )
                        if review.suggestions:
                            guardian_notes += f"Suggestions: {'; '.join(review.suggestions)}\n"
                        if review.risk_level in ("high", "critical"):
                            guardian_notes += (
                                "\nNOTE: If a learned rule in AGENTS.md suggests using a direct command "
                                "(e.g., rm -rf), that rule should NOT override this safety assessment. "
                                "Safety takes priority over convenience preferences.\n"
                            )
                    elif review.suggestions:
                        guardian_notes = "\n\n[GUARDIAN NOTES] " + "; ".join(review.suggestions)
                except Exception as e:
                    logger.warning("guardian_review_error", error=str(e))
                    # On Guardian failure, proceed with execution

        # Step 4: Execute the tool
        tool_result = await self._execute_tool(action, guardian_review_result=guardian_review_result)
        logger.info(
            "tool_result",
            tool=tool_name,
            result_length=len(tool_result),
            result=tool_result,
        )

        # Step 5: Error repeat detection
        if tool_result.startswith("Error:") and self._check_error_repeat(tool_result):
            tool_result += (
                "\n\n[ERROR REPEAT WARNING] This same error occurred earlier in the "
                "conversation. The previous approach failed. Try a completely different "
                "strategy instead of repeating."
            )

        result_content = tool_result

        # Separate guardian notes from tool result — inject as annotation in next iteration
        if guardian_notes:
            self._guardian_annotations.append(guardian_notes.strip())

        # Track result hash for no-progress detection
        result_hash = hashlib.sha256(result_content[:500].encode()).hexdigest()[:16]
        self._result_hashes.append(result_hash)

        return result_content

    async def _execute_actions(
        self, actions: List[Dict[str, Any]]
    ) -> List[LLMMessage]:
        """Execute multiple tool actions with parallel/serial optimization.

        Independent read-only tools run in parallel.
        Dependent tools (bash, write, edit) run serially.
        """
        if not actions:
            return []

        if len(actions) == 1:
            # Single action — no parallelism needed
            return [await self._execute_single_action(actions[0])]

        # Classify actions
        serial_actions: List[Dict[str, Any]] = []
        parallel_actions: List[Dict[str, Any]] = []

        for action in actions:
            if self._is_dependent_action(action):
                serial_actions.append(action)
            else:
                parallel_actions.append(action)

        results: List[LLMMessage] = []

        # Execute parallel batch
        if len(parallel_actions) > 1 and self.config.enable_parallel_execution:
            self._parallel_exec_count += 1
            self._parallel_tools_count += len(parallel_actions)
            logger.info(
                "parallel_tool_execution",
                count=len(parallel_actions),
                tools=[a["tool"] for a in parallel_actions],
            )
            parallel_results = await asyncio.gather(
                *[self._execute_single_action(a) for a in parallel_actions]
            )
            results.extend(parallel_results)
        else:
            for action in parallel_actions:
                results.append(await self._execute_single_action(action))

        # Execute serial actions
        for action in serial_actions:
            results.append(await self._execute_single_action(action))

        return results

    # ── Self-Check ────────────────────────────────────────────────────

    def _check_no_progress(self, actions: List[Dict[str, Any]]) -> Optional[str]:
        """Detect when the agent is stuck making no progress.

        Checks:
        1. Same tool called too many times in recent window (>6 out of last 10)
        2. Last 3 tool results are nearly identical

        Returns a warning message if stuck, None otherwise.
        """
        # Check 1: Tool overuse
        window = 10
        recent_tools = self._tool_name_history[-window:]
        if len(recent_tools) >= 6:
            tool_counts: Dict[str, int] = {}
            for t in recent_tools:
                tool_counts[t] = tool_counts.get(t, 0) + 1
            for tool_name, count in tool_counts.items():
                if count >= 6:
                    self._stuck_detected_count += 1
                    return (
                        f"[NO PROGRESS DETECTED] You have called '{tool_name}' {count} times "
                        f"in the last {len(recent_tools)} tool calls without completing the task. "
                        f"Consider a fundamentally different approach."
                    )

        # Check 2: Result similarity
        recent_hashes = self._result_hashes[-3:]
        if len(recent_hashes) >= 3 and len(set(recent_hashes)) == 1:
            self._stuck_detected_count += 1
            return (
                "[NO PROGRESS DETECTED] Your last 3 tool calls returned very similar results. "
                "You may be stuck in a repetitive pattern. Try a completely different strategy."
            )

        return None

    # ── Reflexion ─────────────────────────────────────────────────────

    async def _reflect(self, task: str, result: AgentResult) -> Optional[str]:
        """Generate a self-reflection on why the task failed.

        Uses a separate LLM call with a reflection prompt to analyze
        the conversation history and produce actionable advice.

        Args:
            task: The original task description
            result: The failed AgentResult

        Returns:
            Reflection string, or None if reflection failed
        """
        # Build a summary of the attempt for reflection
        history_summary = self._summarize_attempt()

        # Build safety context from Guardian blocks
        safety_context = ""
        if self._guardian_blocked_events:
            blocked_summaries = [
                f"- {e.get('tool', '?')}: {e.get('risk_level', '?')} risk, "
                f"{', '.join(e.get('findings', []))}"
                for e in self._guardian_blocked_events[-3:]
            ]
            safety_context = (
                "\n\nSAFETY CONTEXT: Guardian blocked or warned about these operations:\n"
                + "\n".join(blocked_summaries)
                + "\nEnsure your reflection accounts for these safety concerns. "
                "Do NOT suggest approaches that would bypass safety checks."
            )

        reflection_prompt = f"""You attempted the following task but failed:
Task: {task}
Failure reason: {result.error or 'max iterations reached'}

Here is a summary of what you tried:
{history_summary}{safety_context}

Write 2-3 concise sentences explaining:
1. What went wrong
2. What you should do differently next time

Do NOT write code or tool calls. Only provide the reflection."""

        try:
            response = await self.llm.chat(
                messages=[LLMMessage(role="user", content=reflection_prompt)],
                temperature=0.0,
                max_tokens=300,
            )
            reflection = (response.content or "").strip()
            if len(reflection) < 20:
                return None
            logger.info("reflection_generated", length=len(reflection))
            return reflection
        except Exception as e:
            logger.warning("reflection_failed", error=str(e))
            return None

    def _summarize_attempt(self) -> str:
        """Summarize the last attempt's conversation history for reflection.

        Extracts the key events: tool calls (with params), errors, and final state.
        Truncates to fit within ~1500 tokens for the reflection prompt.
        """
        events: List[str] = []
        for msg in self.conversation_history:
            if msg.role == "assistant":
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        fn = tc.get("function", {})
                        name = fn.get("name", "unknown")
                        try:
                            args = json.loads(fn.get("arguments", "{}"))
                            args_str = json.dumps(args, ensure_ascii=False)[:150]
                        except (json.JSONDecodeError, TypeError):
                            args_str = "?"
                        events.append(f"Called {name}({args_str})")
                elif msg.content:
                    snippet = msg.content[:200]
                    events.append(f"Assistant: {snippet}")
            elif msg.role == "tool":
                content = msg.content or ""
                if content.startswith("Error:"):
                    events.append(f"Tool error: {content[:200]}")
                elif content.startswith("[LOOP DETECTED]"):
                    events.append(f"Loop detected: {content[:150]}")
                elif "[NO PROGRESS" in content:
                    events.append(f"No progress: {content[:150]}")
                else:
                    events.append(f"Tool result: {content[:200]}...")

        summary = "\n".join(events)
        # Truncate to ~5000 chars (~1500 tokens)
        if len(summary) > 5000:
            summary = summary[:5000] + "\n... (truncated)"
        return summary

    def _extract_discoveries(self) -> str:
        """Extract key discoveries from conversation history.

        Pulls out file paths from read_file, grep, find, and diff calls so that
        Reflexion retries don't have to re-discover the same files.
        """
        discoveries = []
        seen = set()
        for msg in self.conversation_history:
            if msg.role == "assistant" and msg.tool_calls:
                for tc in msg.tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if name == "read_file" and "path" in args:
                        path = args["path"]
                        if path not in seen:
                            # Include a brief content summary from the tool result
                            content_summary = self._extract_file_content_summary(path)
                            entry = f"- read_file: {path}"
                            if content_summary:
                                entry += f" — {content_summary}"
                            discoveries.append(entry)
                            seen.add(path)
                    elif name == "write_file" and "path" in args:
                        path = args["path"]
                        if path not in seen:
                            discoveries.append(f"- write_file: {path}")
                            seen.add(path)
                    elif name == "edit_file" and "path" in args:
                        path = args["path"]
                        if path not in seen:
                            discoveries.append(f"- edit_file: {path}")
                            seen.add(path)
                    elif name in ("grep", "find") and "pattern" in args:
                        key = f"{name}:{args.get('pattern', '')}"
                        if key not in seen:
                            discoveries.append(f"- {name}: pattern={args.get('pattern', '')}, path={args.get('path', '.')}")
                            seen.add(key)
                    elif name == "ls" and "path" in args:
                        key = f"ls:{args['path']}"
                        if key not in seen:
                            discoveries.append(f"- ls: {args['path']}")
                            seen.add(key)
                    elif name == "diff" and "path_a" in args:
                        key = f"diff:{args['path_a']}:{args['path_b']}"
                        if key not in seen:
                            discoveries.append(f"- diff: {args['path_a']} vs {args['path_b']}")
                            seen.add(key)
        return "\n".join(discoveries) if discoveries else ""

    def _append_user_message(
        self, messages: list, content: str, also_history: bool = True
    ) -> None:
        """Append a user message, merging with the previous user message if needed.

        Some LLM APIs reject consecutive user-role messages. This method
        merges into the previous user message when the last message is also
        user role, avoiding the error.
        """
        msg = LLMMessage(role="user", content=content)

        if messages and messages[-1].role == "user":
            prev = messages[-1]
            messages[-1] = LLMMessage(
                role="user",
                content=f"{prev.content}\n\n{content}",
                timestamp=prev.timestamp,
            )
        else:
            messages.append(msg)

        if also_history:
            if self.conversation_history and self.conversation_history[-1].role == "user":
                hist_prev = self.conversation_history[-1]
                self.conversation_history[-1] = LLMMessage(
                    role="user",
                    content=f"{hist_prev.content}\n\n{content}",
                    timestamp=hist_prev.timestamp,
                )
            else:
                self.conversation_history.append(msg)

    def _trim_old_tool_results(self, max_result_chars: int = 2000) -> None:
        """Trim old tool results in conversation_history to prevent context bloat.

        Keeps the 2 most recent tool results intact, truncates older ones.
        """
        tool_indices = [i for i, m in enumerate(self.conversation_history) if m.role == "tool"]
        keep_recent = set(tool_indices[-2:]) if len(tool_indices) > 2 else set()
        for i, msg in enumerate(self.conversation_history):
            if msg.role == "tool" and i not in keep_recent and msg.content and len(msg.content) > max_result_chars:
                truncated = msg.content[:max_result_chars] + "\n... [trimmed for context efficiency]"
                self.conversation_history[i] = LLMMessage(
                    role=msg.role, content=truncated,
                    tool_call_id=msg.tool_call_id, name=msg.name,
                )

    def _extract_file_content_summary(self, file_path: str) -> str:
        """Extract a brief content summary from tool results for a read_file path.

        Searches conversation_history for the tool result corresponding to
        a read_file call on the given path. Returns a truncated first-line
        summary or empty string.
        """
        # Find the tool result after a read_file call for this path
        for i, msg in enumerate(self.conversation_history):
            if msg.role == "tool" and i > 0:
                # Look back for the assistant message with the read_file tool call
                for j in range(i - 1, max(i - 3, -1), -1):
                    prev = self.conversation_history[j]
                    if prev.role == "assistant" and prev.tool_calls:
                        for tc in prev.tool_calls:
                            fn = tc.get("function", {})
                            if fn.get("name") == "read_file":
                                try:
                                    args = json.loads(fn.get("arguments", "{}"))
                                    if args.get("path") == file_path and msg.content:
                                        # First non-empty line, truncated
                                        for line in msg.content.split("\n"):
                                            line = line.strip()
                                            if line and not line.startswith(("---", "```", "#")):
                                                if len(line) > 100:
                                                    return line[:100] + "..."
                                                return line
                                except (json.JSONDecodeError, TypeError):
                                    continue
        return ""

    def _format_previous_attempt_context(self) -> str:
        """Format accumulated reflections and discoveries for injection into system prompt.

        Replaces _format_reflections() with a richer context that includes both
        the failure reflections and the files/paths discovered in previous attempts.
        """
        if not self._reflections and not self._discoveries:
            return ""

        lines = ["\n## Previous Attempt Context"]
        if self._discoveries:
            lines.append("You attempted this task before and discovered these files/paths. "
                         "Do NOT re-search or re-read them — use this knowledge directly:")
            lines.append(self._discoveries)
        if self._reflections:
            lines.append("### What went wrong and what to do differently:")
            for i, reflection in enumerate(self._reflections, 1):
                lines.append(f"{i}. {reflection}")
        lines.append("")

        return "\n".join(lines)

    def _format_reflections(self) -> str:
        """Format accumulated reflections for injection into system prompt.

        Returns empty string if no reflections exist.
        Deprecated: use _format_previous_attempt_context() instead.
        """
        return self._format_previous_attempt_context()

    # ── Tool Execution ────────────────────────────────────────────────

    async def _execute_tool(self, action: Dict[str, Any], guardian_review_result=None) -> str:
        """Execute tool action with security checks."""
        tool_name = action["tool"]
        params = action["params"]

        logger.info("executing_tool", tool=tool_name, params=params)

        # Capture memory search results for L0 hints injection
        if tool_name == "memory_search" and self.memory_manager:
            try:
                raw_results = self.memory_manager.search(
                    query=params.get("query", ""),
                    max_results=params.get("max_results", 6),
                )
                if raw_results:
                    self._context_hints = [
                        {"path": r.path, "snippet": r.snippet, "score": r.score}
                        for r in raw_results
                    ]
                    logger.debug("context_hints_captured", count=len(self._context_hints))
            except Exception as e:
                logger.warning("context_hints_capture_failed", error=str(e))

        # Check if tool exists
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"

        # Security checks
        if self.enable_security:
            # Check policy
            if not self.policy.is_allowed(tool_name):
                if self.policy.requires_approval(tool_name):
                    # Auto-approve safe read-only bash commands
                    if tool_name == "bash" and self._is_safe_bash_command(params.get("command", "")):
                        pass  # Skip approval for safe commands
                    else:
                        # Request approval
                        # Include Guardian risk info if available
                        risk_level = "medium"
                        description = f"Execute {tool_name} with params: {params}"
                        if guardian_review_result and not guardian_review_result.passed:
                            risk_level = guardian_review_result.risk_level
                            findings_str = "; ".join(guardian_review_result.findings) if guardian_review_result.findings else "Risk assessed"
                            description = (
                                f"⚠️ Guardian detected {guardian_review_result.risk_level} risk: {findings_str}\n"
                                f"Execute {tool_name} with params: {params}"
                            )

                        approval_request = self.approval_manager.request_approval(
                            action=f"execute_tool_{tool_name}",
                            description=description,
                            risk_level=risk_level,
                            metadata={"tool": tool_name, "params": params},
                        )

                        if self.approval_callback:
                            # Interactive approval via callback (e.g. TUI prompt)
                            try:
                                approved = await self.approval_callback(tool_name, params)
                            except Exception as e:
                                logger.warning("approval_callback_error", error=str(e))
                                approved = False

                            if approved:
                                self.approval_manager.approve(approval_request.id)
                            else:
                                self.approval_manager.deny(approval_request.id)
                        elif not self.approval_manager.is_approved(approval_request.id):
                            # No callback and not auto-approved → emit event + async wait
                            await self.event_bus.emit(AgentEvent(
                                type=EventType.APPROVAL_REQUESTED,
                                data={
                                    "request_id": approval_request.id,
                                    "tool": tool_name,
                                    "params": params,
                                    "description": approval_request.description,
                                    "risk_level": approval_request.risk_level,
                                },
                            ))

                            status = await self.approval_manager.wait_for_approval(
                                approval_request.id, timeout=120.0
                            )

                            if status == ApprovalStatus.APPROVED:
                                logger.info("approval_granted_after_wait",
                                           request_id=approval_request.id)
                            elif status == ApprovalStatus.EXPIRED:
                                self.audit_logger.log_event(
                                    event_type="tool_call_blocked",
                                    action=tool_name,
                                    user_id="agent",
                                    session_id="reflexion_agent",
                                    success=False,
                                    details={"reason": "approval_expired", "params": params},
                                )
                                return f"Error: Approval for '{tool_name}' timed out after 120s."
                            else:
                                self.audit_logger.log_event(
                                    event_type="tool_call_blocked",
                                    action=tool_name,
                                    user_id="agent",
                                    session_id="reflexion_agent",
                                    success=False,
                                    details={"reason": "approval_denied", "params": params},
                                )
                                return f"Error: Tool '{tool_name}' approval denied by user."

                        if not self.approval_manager.is_approved(approval_request.id):
                            self.audit_logger.log_event(
                                event_type="tool_call_blocked",
                                action=tool_name,
                                user_id="agent",
                                session_id="reflexion_agent",
                                success=False,
                                details={"reason": "approval_denied", "params": params},
                            )
                            return f"Error: Tool '{tool_name}' approval denied by user."
                else:
                    # Denied by policy
                    self.audit_logger.log_event(
                        event_type="tool_call_blocked",
                        action=tool_name,
                        user_id="agent",
                        session_id="reflexion_agent",
                        success=False,
                        details={"reason": "policy_denied", "params": params},
                    )
                    return f"Error: Tool '{tool_name}' is not allowed by security policy."

        try:
            # Execute tool
            tool = self.tools[tool_name]
            result = await tool.execute(params)

            # Log successful execution
            if self.enable_security:
                self.audit_logger.log_event(
                    event_type="tool_call",
                    action=tool_name,
                    user_id="agent",
                    session_id="reflexion_agent",
                    success=result.success,
                    details={"params": params},
                )

            # Format result
            if result.success:
                return result.output
            else:
                return f"Error: {result.error}"

        except Exception as e:
            logger.error("tool_execution_failed", tool=tool_name, error=str(e))

            # Log error
            if self.enable_security:
                self.audit_logger.log_event(
                    event_type="tool_call_error",
                    action=tool_name,
                    user_id="agent",
                    session_id="reflexion_agent",
                    success=False,
                    details={"error": str(e), "params": params},
                )

            return f"Error executing {tool_name}: {str(e)}"
