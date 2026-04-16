"""REST API handlers for OmniAgent gateway."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from aiohttp import web

from omniagent.config.loader import DEFAULT_CONFIG_PATH, deep_merge, load_config, save_config
from omniagent.infra import get_logger
from omniagent.config.models import OmniAgentConfig

if TYPE_CHECKING:
    from omniagent.agents.reflexion import ReflexionAgent
    from omniagent.gateway.session import SessionManager
    from omniagent.channels.manager import ChannelManager

logger = get_logger(__name__)

# ── Sensitive field masking ────────────────────────────────────

_SENSITIVE_FIELDS = {"api_key", "openai_api_key", "anthropic_api_key"}


def _mask_sensitive_fields(data: dict) -> None:
    """Mask sensitive fields in a config dict (in-place)."""
    for name in _SENSITIVE_FIELDS:
        if name in data and data[name]:
            val = data[name]
            if isinstance(val, str) and len(val) > 4:
                data[name] = val[:4] + "****"
            else:
                data[name] = "****"
    # Mask provider-level api_keys
    providers = data.get("providers")
    if isinstance(providers, dict):
        for pval in providers.values():
            if isinstance(pval, dict) and pval.get("api_key"):
                key = pval["api_key"]
                pval["api_key"] = key[:4] + "****" if len(key) > 4 else "****"


def _contains_sensitive(updates: dict) -> List[str]:
    """Return list of sensitive field paths found in updates dict."""
    found = []
    for name in _SENSITIVE_FIELDS:
        if name in updates:
            found.append(name)
    providers = updates.get("providers")
    if isinstance(providers, dict):
        for pname, pval in providers.items():
            if isinstance(pval, dict) and "api_key" in pval:
                found.append(f"providers.{pname}.api_key")
    return found


# ── API Context ────────────────────────────────────────────────


@dataclass
class APIContext:
    """Shared context for all API handlers."""

    agent: Optional["ReflexionAgent"] = None
    session_manager: Optional["SessionManager"] = None
    config: Optional["OmniAgentConfig"] = None
    channel_manager: Optional["ChannelManager"] = None
    start_time: float = field(default_factory=time.time)


# ── Config Handlers ───────────────────────────────────────────


async def get_config(request: web.Request) -> web.Response:
    """GET /api/config -- return current config with masked sensitive fields."""
    ctx: APIContext = request.app["api_ctx"]
    if ctx.config is None:
        return web.json_response({"error": "Config not loaded"}, status=503)

    data = ctx.config.model_dump(mode="json")
    _mask_sensitive_fields(data)
    return web.json_response(data)


async def update_config(request: web.Request) -> web.Response:
    """PATCH /api/config -- partial update of config fields."""
    ctx: APIContext = request.app["api_ctx"]
    if ctx.config is None:
        return web.json_response({"error": "Config not loaded"}, status=503)

    try:
        updates = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    if not isinstance(updates, dict):
        return web.json_response({"error": "Body must be a JSON object"}, status=400)

    # Reject sensitive fields
    sensitive = _contains_sensitive(updates)
    if sensitive:
        return web.json_response(
            {"error": f"Cannot update sensitive fields via API: {', '.join(sensitive)}"},
            status=400,
        )

    # Deep merge
    current = ctx.config.model_dump(mode="json")
    merged = deep_merge(current, updates)

    try:
        new_config = OmniAgentConfig(**merged)
    except Exception as e:
        return web.json_response({"error": f"Invalid config: {e}"}, status=400)

    # Save to disk
    try:
        save_config(new_config)
    except Exception as e:
        logger.error("config_save_error", error=str(e))
        return web.json_response({"error": f"Failed to save config: {e}"}, status=500)

    # Update in-memory
    ctx.config = new_config
    if ctx.agent is not None:
        ctx.agent.config = new_config

    result = new_config.model_dump(mode="json")
    _mask_sensitive_fields(result)
    return web.json_response({"status": "updated", "config": result})


async def reload_config_handler(request: web.Request) -> web.Response:
    """POST /api/config/reload -- reload config from disk."""
    ctx: APIContext = request.app["api_ctx"]
    if ctx.config is None:
        return web.json_response({"error": "Config not loaded"}, status=503)

    try:
        reloaded = load_config(DEFAULT_CONFIG_PATH)
    except Exception as e:
        return web.json_response({"error": f"Failed to reload: {e}"}, status=500)

    ctx.config = reloaded
    if ctx.agent is not None:
        ctx.agent.config = reloaded

    return web.json_response({"status": "reloaded"})


# ── Session Handlers ──────────────────────────────────────────


async def list_sessions(request: web.Request) -> web.Response:
    """GET /api/sessions -- list sessions with optional filters."""
    ctx: APIContext = request.app["api_ctx"]
    if ctx.session_manager is None:
        return web.json_response({"sessions": []})

    from omniagent.gateway.session import SessionState

    state_str = request.query.get("state")
    user_id = request.query.get("user_id")
    channel_id = request.query.get("channel_id")

    state_enum = None
    if state_str:
        try:
            state_enum = SessionState(state_str)
        except ValueError:
            return web.json_response({"error": f"Invalid state: {state_str}"}, status=400)

    sessions = ctx.session_manager.list_sessions(
        user_id=user_id,
        channel_id=channel_id,
        state=state_enum,
    )

    result = []
    for s in sessions:
        title = "Untitled"
        if s.history:
            first_user = next((m for m in s.history if m.role == "user"), None)
            if first_user:
                title = first_user.content[:50].replace("\n", " ")

        result.append(
            {
                "id": s.id,
                "user_id": s.user_id,
                "channel_id": s.channel_id,
                "state": s.state.value,
                "message_count": len(s.history),
                "created_at": s.created_at.isoformat(),
                "last_active_at": s.last_active_at.isoformat(),
                "title": title,
            }
        )

    return web.json_response({"sessions": result})


async def get_session(request: web.Request) -> web.Response:
    """GET /api/sessions/{session_id} -- get session with full history."""
    ctx: APIContext = request.app["api_ctx"]
    if ctx.session_manager is None:
        return web.json_response({"error": "Session manager not available"}, status=503)

    session_id = request.match_info["session_id"]
    session = ctx.session_manager.get_session(session_id)
    if session is None:
        return web.json_response({"error": "Session not found"}, status=404)

    return web.json_response(session.to_dict())


async def create_session(request: web.Request) -> web.Response:
    """POST /api/sessions -- create a new session."""
    ctx: APIContext = request.app["api_ctx"]
    if ctx.session_manager is None:
        return web.json_response({"error": "Session manager not available"}, status=503)

    try:
        body = await request.json()
    except Exception:
        body = {}

    session = ctx.session_manager.create_session(
        user_id=body.get("user_id", "anonymous"),
        channel_id=body.get("channel_id", "web"),
    )
    return web.json_response(
        {
            "id": session.id,
            "state": session.state.value,
            "created_at": session.created_at.isoformat(),
        },
        status=201,
    )


async def update_session(request: web.Request) -> web.Response:
    """PATCH /api/sessions/{session_id} -- pause/resume/close a session."""
    ctx: APIContext = request.app["api_ctx"]
    if ctx.session_manager is None:
        return web.json_response({"error": "Session manager not available"}, status=503)

    session_id = request.match_info["session_id"]
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    action = body.get("action")
    sm = ctx.session_manager

    if action == "pause":
        ok = sm.pause_session(session_id)
    elif action == "resume":
        ok = sm.resume_session(session_id)
    elif action == "close":
        ok = sm.close_session(session_id)
    else:
        return web.json_response({"error": f"Invalid action: {action}. Use pause/resume/close"}, status=400)

    if not ok:
        return web.json_response({"error": "Session not found"}, status=404)

    return web.json_response({"status": action + "d"})


async def delete_session(request: web.Request) -> web.Response:
    """DELETE /api/sessions/{session_id} -- delete a session."""
    ctx: APIContext = request.app["api_ctx"]
    if ctx.session_manager is None:
        return web.json_response({"error": "Session manager not available"}, status=503)

    session_id = request.match_info["session_id"]
    session = ctx.session_manager.get_session(session_id)
    if session is None:
        return web.json_response({"error": "Session not found"}, status=404)

    # Remove from memory and disk
    ctx.session_manager.sessions.pop(session_id, None)
    ctx.session_manager._delete_session(session_id)

    return web.json_response({"status": "deleted"})


# ── Health Handler ────────────────────────────────────────────


async def get_health(request: web.Request) -> web.Response:
    """GET /api/health -- enhanced health dashboard data."""
    ctx: APIContext = request.app["api_ctx"]

    status = {
        "status": "healthy",
        "uptime_seconds": round(time.time() - ctx.start_time),
    }

    # Agent state
    if ctx.agent is not None:
        agent_state = ctx.agent.agent_state
        status["agent"] = {
            "model_provider": ctx.agent.config.agent.model_provider if ctx.agent.config else "unknown",
            "model_id": ctx.agent.config.agent.model_id if ctx.agent.config else "unknown",
            "is_streaming": agent_state.is_streaming,
            "current_iteration": agent_state.iteration,
            "pending_tool_calls": agent_state.pending_tool_calls,
            "total_tool_calls": agent_state.total_tool_calls,
            "error": agent_state.error,
        }
    else:
        status["agent"] = None

    # Session counts
    if ctx.session_manager is not None:
        from omniagent.gateway.session import SessionState

        all_sessions = list(ctx.session_manager.sessions.values())
        status["sessions"] = {
            "total": len(all_sessions),
            "active": sum(1 for s in all_sessions if s.state == SessionState.ACTIVE),
            "paused": sum(1 for s in all_sessions if s.state == SessionState.PAUSED),
            "closed": sum(1 for s in all_sessions if s.state == SessionState.CLOSED),
        }
    else:
        status["sessions"] = {"total": 0, "active": 0, "paused": 0, "closed": 0}

    # Channels
    if ctx.channel_manager is not None:
        status["channels"] = ctx.channel_manager.get_status()

    # Memory
    if ctx.agent is not None and ctx.agent.memory_manager is not None:
        mm = ctx.agent.memory_manager
        status["memory"] = {
            "enabled": True,
            "hybrid_enabled": getattr(mm, "hybrid_enabled", False),
        }
    else:
        status["memory"] = {"enabled": False}

    # Security
    if ctx.agent is not None:
        if ctx.agent.policy is not None:
            status["security"] = {
                "policy_profile": ctx.agent.policy.current_profile if hasattr(ctx.agent.policy, "current_profile") else "unknown",
                "allowed_tools": len(ctx.agent.policy.get_allowed_tools()) if hasattr(ctx.agent.policy, "get_allowed_tools") else 0,
                "pending_approvals": len(ctx.agent.approval_manager.get_pending_requests()) if ctx.agent.approval_manager else 0,
            }
        else:
            status["security"] = {"policy_profile": "none", "allowed_tools": 0, "pending_approvals": 0}

    # Feature flags
    if ctx.config is not None:
        status["features"] = {
            "progressive_loading": ctx.config.enable_progressive_loading,
            "parallel_execution": ctx.config.enable_parallel_execution,
            "security_guard": ctx.config.enable_security_guard,
            "self_improving": ctx.config.enable_self_improving,
        }
    else:
        status["features"] = {}

    return web.json_response(status)


# ── Skills Handlers ───────────────────────────────────────────


async def list_skills(request: web.Request) -> web.Response:
    """GET /api/skills -- list discovered skills."""
    ctx: APIContext = request.app["api_ctx"]
    if ctx.agent is None or ctx.agent.skill_manager is None:
        return web.json_response({"skills": [], "count": 0})

    skills = ctx.agent.skill_manager.discover_skills()
    result = [
        {
            "name": s.name,
            "description": s.description,
            "root_dir": str(s.root_dir),
        }
        for s in skills
    ]
    return web.json_response({"skills": result, "count": len(result)})


async def get_skill(request: web.Request) -> web.Response:
    """GET /api/skills/{name} -- get skill content."""
    ctx: APIContext = request.app["api_ctx"]
    if ctx.agent is None or ctx.agent.skill_manager is None:
        return web.json_response({"error": "Skill manager not available"}, status=503)

    name = request.match_info["skill_name"]
    # Validate name to prevent path traversal
    if ".." in name or "/" in name:
        return web.json_response({"error": "Invalid skill name"}, status=400)

    skills = ctx.agent.skill_manager.discover_skills()
    skill = next((s for s in skills if s.name == name), None)
    if skill is None:
        return web.json_response({"error": f"Skill '{name}' not found"}, status=404)

    content = ""
    patches = ""
    if skill.path.exists():
        content = skill.path.read_text(encoding="utf-8")

    if ctx.agent._skill_evolution:
        tracker = ctx.agent._skill_evolution.tracker
        if tracker:
            patch_content = tracker.get_patches_for_skill(name)
            if patch_content:
                patches = patch_content

    return web.json_response({"name": name, "content": content, "patches": patches})


async def refresh_skills(request: web.Request) -> web.Response:
    """POST /api/skills/refresh -- re-discover skills."""
    ctx: APIContext = request.app["api_ctx"]
    if ctx.agent is None or ctx.agent.skill_manager is None:
        return web.json_response({"error": "Skill manager not available"}, status=503)

    ctx.agent.skill_manager.invalidate_cache()
    skills = ctx.agent.skill_manager.discover_skills()
    return web.json_response({"status": "refreshed", "count": len(skills)})


# ── Tool Handlers ─────────────────────────────────────────────


async def list_tools(request: web.Request) -> web.Response:
    """GET /api/tools -- list tools with policy decisions."""
    ctx: APIContext = request.app["api_ctx"]
    if ctx.agent is None:
        return web.json_response({"tools": [], "profile": "none"})

    tools = []
    if ctx.agent.registry:
        for tool in ctx.agent.registry.list_tools():
            decision = "allow"
            if ctx.agent.policy:
                policy_result = ctx.agent.policy.check_tool(tool.name)
                decision = policy_result.value if hasattr(policy_result, "value") else str(policy_result)
            tools.append(
                {
                    "name": tool.name,
                    "description": tool.description if hasattr(tool, "description") else "",
                    "decision": decision,
                }
            )

    profile = "none"
    if ctx.agent.policy and hasattr(ctx.agent.policy, "current_profile"):
        profile = ctx.agent.policy.current_profile

    return web.json_response({"tools": tools, "profile": profile})


# ── Approval Handlers ─────────────────────────────────────────


async def list_pending_approvals(request: web.Request) -> web.Response:
    """GET /api/approvals -- list pending approval requests."""
    ctx: APIContext = request.app["api_ctx"]
    if ctx.agent is None or ctx.agent.approval_manager is None:
        return web.json_response({"pending": [], "count": 0})

    pending = ctx.agent.approval_manager.get_pending_requests()
    result = [
        {
            "id": r.id,
            "action": r.action,
            "description": r.description,
            "risk_level": r.risk_level,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in pending
    ]
    return web.json_response({"pending": result, "count": len(result)})


async def approve_request(request: web.Request) -> web.Response:
    """POST /api/approvals/{id}/approve -- approve a pending request."""
    ctx: APIContext = request.app["api_ctx"]
    if ctx.agent is None or ctx.agent.approval_manager is None:
        return web.json_response({"error": "Approval manager not available"}, status=503)

    req_id = request.match_info["request_id"]
    ctx.agent.approval_manager.approve(req_id)
    return web.json_response({"status": "approved"})


async def deny_request(request: web.Request) -> web.Response:
    """POST /api/approvals/{id}/deny -- deny a pending request."""
    ctx: APIContext = request.app["api_ctx"]
    if ctx.agent is None or ctx.agent.approval_manager is None:
        return web.json_response({"error": "Approval manager not available"}, status=503)

    req_id = request.match_info["request_id"]
    ctx.agent.approval_manager.deny(req_id)
    return web.json_response({"status": "denied"})


# ── Router Factory ────────────────────────────────────────────


def create_api_router(ctx: APIContext) -> List[web.RouteDef]:
    """Create all API route definitions."""
    return [
        # Config
        web.get("/api/config", get_config),
        web.patch("/api/config", update_config),
        web.post("/api/config/reload", reload_config_handler),
        # Sessions
        web.get("/api/sessions", list_sessions),
        web.post("/api/sessions", create_session),
        web.get("/api/sessions/{session_id}", get_session),
        web.patch("/api/sessions/{session_id}", update_session),
        web.delete("/api/sessions/{session_id}", delete_session),
        # Health
        web.get("/api/health", get_health),
        # Skills
        web.get("/api/skills", list_skills),
        web.get("/api/skills/{skill_name}", get_skill),
        web.post("/api/skills/refresh", refresh_skills),
        # Tools
        web.get("/api/tools", list_tools),
        # Approvals
        web.get("/api/approvals", list_pending_approvals),
        web.post("/api/approvals/{request_id}/approve", approve_request),
        web.post("/api/approvals/{request_id}/deny", deny_request),
    ]
