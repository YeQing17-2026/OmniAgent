"""Configuration models using Pydantic."""

import os
from typing import Any, Dict, List, Literal, Optional
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_ALLOWED_PRICING_OVERRIDE_RATES = {"input", "output", "cache_read"}
_PROVIDER_API_TYPES = {
    "deepseek",
    "openai",
    "anthropic",
    "ollama",
    "gemini",
    "openrouter",
    "vllm",
    "sglang",
    "openai-compatible",
}
_API_TYPE_ALIASES = {
    "custom": "openai-compatible",
    "openai_compatible": "openai-compatible",
}


def _normalize_api_type(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    return _API_TYPE_ALIASES.get(normalized, normalized)


def _ensure_supported_api_type(api_type: Optional[str]) -> Optional[str]:
    if api_type is None:
        return None
    if api_type not in _PROVIDER_API_TYPES:
        supported = ", ".join(sorted(_PROVIDER_API_TYPES))
        raise ValueError(f"Unsupported provider api_type {api_type!r}; expected one of: {supported}")
    return api_type


def _default_api_type_for_provider_key(provider_key: str) -> Optional[str]:
    normalized = _normalize_api_type(provider_key)
    return normalized if normalized in _PROVIDER_API_TYPES else None


class ToolsConfig(BaseModel):
    """Tools configuration."""

    model_config = ConfigDict(extra="allow")

    profile: Literal["minimal", "coding", "messaging", "full"] = Field(
        default="coding",
        description="Tool profile preset"
    )
    allow: List[str] = Field(
        default_factory=list,
        description="Explicitly allowed tools"
    )
    deny: List[str] = Field(
        default_factory=list,
        description="Explicitly denied tools"
    )


class ProviderConfig(BaseModel):
    """Per-provider configuration (api_key, api_url, default model).

    The providers map key is the provider identity used for selection, usage
    accounting, and pricing. api_type selects the implementation/parser.
    """

    model_config = ConfigDict(extra="allow")

    api_type: Optional[str] = Field(
        default=None,
        description="Provider implementation type (deepseek, openai, anthropic, ollama, gemini, openrouter, vllm, sglang, openai-compatible)"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for this provider"
    )
    api_url: Optional[str] = Field(
        default=None,
        description="API base URL for this provider"
    )
    model_id: Optional[str] = Field(
        default=None,
        description="Default model ID for this provider"
    )

    @field_validator("api_type", mode="before")
    @classmethod
    def _normalize_provider_api_type(cls, value: Any) -> Optional[str]:
        return _ensure_supported_api_type(_normalize_api_type(value))

    @model_validator(mode="before")
    @classmethod
    def _strip_legacy_provider_name(cls, value: Any) -> Any:
        if isinstance(value, dict):
            cleaned = dict(value)
            cleaned.pop("provider_name", None)
            return cleaned
        return value


class AgentConfig(BaseModel):
    """Agent configuration."""

    model_config = ConfigDict(extra="allow")

    model_provider: str = Field(
        default="deepseek",
        min_length=1,
        description="Active provider key from the providers map"
    )
    model_id: str = Field(
        default="deepseek-chat",
        description="Model identifier"
    )
    api_url: Optional[str] = Field(
        default=None,
        description="Custom API URL (overrides provider default)"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    max_tokens: int = Field(
        default=4096,
        gt=0,
        description="Maximum tokens per response"
    )
    max_iterations: int = Field(
        default=10,
        gt=0,
        description="Maximum ReAct loop iterations"
    )
    context_window_size: int = Field(
        default=32768,
        ge=0,
        description="Context window size in tokens (0 = auto-detect from model)"
    )
    compaction_enabled: bool = Field(
        default=True,
        description="Enable automatic context compaction"
    )
    system_prompt_token_ratio: float = Field(
        default=0.15,
        gt=0.0,
        le=0.5,
        description="Max fraction of context window used for system prompt"
    )
    reflexion_enabled: bool = Field(
        default=True,
        description="Enable reflexion self-reflection on failure"
    )
    reflexion_max_attempts: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Max additional attempts after reflection (0 = no retry)"
    )


class MemorySearchConfig(BaseModel):
    """Memory search configuration.

   ts defaults.
    """

    model_config = ConfigDict(extra="allow")

    enabled: bool = Field(
        default=True,
        description="Enable memory search system"
    )
    provider: str = Field(
        default="auto",
        description="Embedding provider: 'auto', or explicit provider name"
    )
    chunking_tokens: int = Field(
        default=400,
        gt=0,
        description="Max tokens per memory chunk"
    )
    chunking_overlap: int = Field(
        default=80,
        ge=0,
        description="Overlap tokens between chunks"
    )
    sync_on_session_start: bool = Field(
        default=True,
        description="Sync memory files on session start"
    )
    sync_on_search: bool = Field(
        default=True,
        description="Sync memory files before each search"
    )
    query_max_results: int = Field(
        default=6,
        gt=0,
        description="Max search results returned"
    )
    query_min_score: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score for results"
    )
    hybrid_enabled: bool = Field(
        default=True,
        description="Enable hybrid search (BM25 + vector)"
    )
    hybrid_vector_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for vector search in hybrid scoring"
    )
    hybrid_text_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for text search in hybrid scoring"
    )
    embedding_provider: str = Field(
        default="openai",
        description="Embedding provider: 'openai', 'local', or custom"
    )
    local_embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model name for local embeddings"
    )
    mmr_lambda: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Lambda parameter for Maximal Marginal Relevance diversity"
    )
    temporal_decay_hours: float = Field(
        default=168.0,
        ge=0.0,
        description="Hours over which memory relevance decays (default 7 days)"
    )


class GatewayConfig(BaseModel):
    """Gateway configuration."""

    model_config = ConfigDict(extra="allow")

    host: str = Field(
        default="127.0.0.1",
        description="Gateway host"
    )
    port: int = Field(
        default=18790,
        gt=0,
        lt=65536,
        description="Gateway port"
    )
    session_timeout: int = Field(
        default=3600,
        gt=0,
        description="Session timeout in seconds"
    )
    usage_summary_max_days: int = Field(
        default=365,
        ge=1,
        description="Maximum days accepted by /api/usage/summary"
    )


class UsageConfig(BaseModel):
    """Usage accounting configuration."""

    model_config = ConfigDict(extra="allow")

    pricing_catalog_enabled: bool = Field(
        default=True,
        description="Use the default model pricing catalog for usage cost estimates"
    )
    pricing_catalog_url: str = Field(
        default="https://models.dev/api.json",
        description="Default model pricing catalog URL"
    )
    pricing_cache_ttl_seconds: int = Field(
        default=900,
        ge=0,
        description="Seconds to cache fetched model pricing data"
    )
    pricing_request_timeout_seconds: float = Field(
        default=2.0,
        gt=0,
        description="Timeout for fetching default model pricing data"
    )
    pricing_overrides: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Manual pricing overrides keyed by 'provider/model_id'. Supported rates are input, output, and cache_read, in USD per million tokens"
    )

    @field_validator("pricing_overrides", mode="before")
    @classmethod
    def _sanitize_pricing_overrides(cls, value: Any) -> Any:
        """Drop unsupported pricing keys before config is exposed or saved."""
        if value is None or not isinstance(value, dict):
            return value

        sanitized: Dict[str, Any] = {}
        for rule_key, rule in value.items():
            if not isinstance(rule, dict):
                sanitized[str(rule_key)] = rule
                continue
            filtered = {
                rate_key: rate
                for rate_key, rate in rule.items()
                if rate_key in _ALLOWED_PRICING_OVERRIDE_RATES
            }
            if filtered:
                sanitized[str(rule_key)] = filtered
        return sanitized


class ChannelsConfig(BaseModel):
    """Chat channels configuration.

    Each channel's config is stored as an extra field (e.g. .feishu).
    Each channel implementation parses its own config section.
    """

    model_config = ConfigDict(extra="allow")


class SkillEvolutionConfig(BaseModel):
    """Configuration for skill self-evolution.

    Controls skill creation (from repeated patterns) and skill evolution
    (from error-recovery pairs).
    """

    model_config = ConfigDict(extra="allow")

    # Pattern recording
    pattern_min_occurrences: int = Field(
        default=3,
        ge=2,
        le=20,
        description="Minimum times a tool sequence must appear before compilation is triggered",
    )
    pattern_max_file_size_mb: float = Field(
        default=50.0,
        ge=1.0,
        le=500.0,
        description="Maximum size of patterns.jsonl before oldest entries are pruned",
    )

    # Skill evolution (patching)
    evolution_enabled: bool = Field(
        default=True,
        description="Enable skill evolution (write patches on error-recovery)",
    )
    patch_max_per_skill: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum patches per skill before oldest is archived",
    )

    # Compilation
    compile_max_tokens: int = Field(
        default=3072,
        ge=256,
        le=8192,
        description="Max tokens for LLM skill compilation prompt (SKILL.md + script)",
    )
    feedback_evolution_enabled: bool = Field(
        default=True,
        description="Enable skill evolution from user feedback (only for skill-targeted feedback)",
    )


class ContextEvolutionConfig(BaseModel):
    """Configuration for context self-evolution.

    Controls automatic extraction of lessons from failures, reflections,
    and user feedback. Validated lessons are promoted to AGENTS.md.
    """

    model_config = ConfigDict(extra="allow")

    evolution_enabled: bool = Field(
        default=True,
        description="Enable context evolution (lesson recording and promotion)",
    )
    lesson_min_evidence: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Minimum times a lesson must be confirmed before promotion",
    )
    max_learnings: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Maximum lessons before oldest are pruned",
    )
    max_agents_rules: int = Field(
        default=30,
        ge=5,
        le=100,
        description="Maximum promoted rules in AGENTS.md before oldest are archived",
    )
    compile_max_tokens: int = Field(
        default=1024,
        ge=256,
        le=4096,
        description="Max tokens for LLM lesson extraction prompt",
    )
    promotion_enabled: bool = Field(
        default=True,
        description="Enable auto-promotion of validated lessons to AGENTS.md",
    )
    user_feedback_min_evidence: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Minimum evidence for user_feedback lessons (lower than auto-learned)",
    )


class SentinelConfig(BaseModel):
    """Configuration for Sentinel Agent — task planning and progress tracking.

    Sentinel is a lightweight planning agent that activates for complex,
    multi-step tasks. It decomposes tasks into milestones, tracks progress,
    and persists plans for cross-session recovery.
    """

    model_config = ConfigDict(extra="allow")

    enabled: bool = Field(
        default=True,
        description="Enable Sentinel agent",
    )
    # Activation thresholds
    max_reflexion_failures_before_activate: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Consecutive reflexion failures before Sentinel activates",
    )
    multi_step_keyword_threshold: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Minimum multi-step keyword occurrences (e.g. 然后/接着/then) to activate",
    )
    bash_dir_threshold: int = Field(
        default=3,
        ge=2,
        le=10,
        description="Minimum distinct bash directories before Sentinel activates",
    )
    llm_complexity_enabled: bool = Field(
        default=True,
        description="Use LLM to estimate task complexity when rule-based checks don't trigger",
    )
    # Behavior
    max_milestones: int = Field(
        default=10,
        ge=2,
        le=50,
        description="Maximum milestones per plan",
    )
    milestone_iteration_ratio: float = Field(
        default=3.0,
        ge=1.0,
        le=10.0,
        description="Iterations per milestone (used to auto-cap max_milestones = max_iterations / ratio)",
    )
    plan_persist_path: str = Field(
        default=".omniagent/sentinel",
        description="Directory for persisted milestone plans",
    )
    compile_max_tokens: int = Field(
        default=1024,
        ge=256,
        le=4096,
        description="Max tokens for LLM decomposition prompt",
    )
    # LLM (None = inherit from main agent)
    model_provider: Optional[str] = Field(
        default=None,
        description="Override LLM provider (None = use main agent's provider)",
    )
    model_id: Optional[str] = Field(
        default=None,
        description="Override model ID (None = use main agent's model)",
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Sampling temperature (lower for deterministic planning)",
    )


class GuardianConfig(BaseModel):
    """Configuration for Guardian Agent — output quality review and safety gate.

    Guardian is a lightweight review agent that activates before high-impact
    operations. It provides LLM-powered review on top of static ToolPolicy.
    """

    model_config = ConfigDict(extra="allow")

    enabled: bool = Field(
        default=True,
        description="Enable Guardian agent",
    )
    # Activation thresholds
    max_file_writes_before_activate: int = Field(
        default=3,
        ge=2,
        le=10,
        description="File write/edit operations before Guardian activates",
    )
    high_risk_bash_always_activate: bool = Field(
        default=True,
        description="Always review bash commands with high risk level",
    )
    review_final_on_risky: bool = Field(
        default=True,
        description="Review final response after risky operations",
    )
    # Behavior
    max_review_tokens: int = Field(
        default=2000,
        ge=512,
        le=8192,
        description="Max tokens for LLM review prompt",
    )
    auto_block_on_critical: bool = Field(
        default=True,
        description="Automatically block operations with critical findings",
    )
    # LLM (None = inherit from main agent)
    model_provider: Optional[str] = Field(
        default=None,
        description="Override LLM provider (None = use main agent's provider)",
    )
    model_id: Optional[str] = Field(
        default=None,
        description="Override model ID (None = use main agent's model)",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description="Sampling temperature (0.0 = deterministic for safety)",
    )


class RLConfig(BaseModel):
    """Configuration for Reinforcement Learning (GRPO) training.

    Only activates when api_type is "vllm" or "sglang".
    Controls the FastAPI proxy, PRM scorer, and rollout worker.
    """

    model_config = ConfigDict(extra="allow")

    enabled: bool = Field(
        default=False,
        description="Enable RL system (proxy, PRM, rollout)",
    )
    # SGLang/vLLM serving
    sglang_router_ip: str = Field(
        default="localhost",
        description="SGLang router IP address",
    )
    sglang_router_port: int = Field(
        default=30000,
        description="SGLang router port",
    )
    hf_checkpoint: str = Field(
        default="",
        description="HuggingFace checkpoint path for the policy model",
    )
    served_model_name: str = Field(
        default="default",
        description="Model name exposed by the OpenAI-compatible API",
    )
    # PRM (Process Reward Model)
    prm_enabled: bool = Field(
        default=True,
        description="Enable PRM scoring",
    )
    prm_num_gpus: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Number of GPUs for PRM inference",
    )
    prm_model_path: str = Field(
        default="",
        description="PRM model path (empty = use same as policy model)",
    )
    prm_m: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of independent PRM evaluations for majority vote",
    )
    prm_temperature: float = Field(
        default=0.6,
        ge=0.0,
        le=2.0,
        description="PRM sampling temperature",
    )
    prm_max_tokens: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Max tokens for PRM response",
    )
    # Rollout
    rollout_batch_size: int = Field(
        default=16,
        ge=1,
        le=256,
        description="Number of trajectories per training batch",
    )
    rollout_max_response_len: int = Field(
        default=8192,
        ge=256,
        le=32768,
        description="Max response tokens per rollout",
    )
    rollout_temperature: float = Field(
        default=0.6,
        ge=0.0,
        le=2.0,
        description="Rollout sampling temperature",
    )
    # Training (GRPO)
    train_backend: str = Field(
        default="fsdp",
        description="Training backend: 'fsdp' or 'megatron'",
    )
    lr: float = Field(
        default=1e-5,
        ge=1e-8,
        le=1e-2,
        description="Learning rate",
    )
    eps_clip_low: float = Field(
        default=0.2,
        description="Lower PPO clipping epsilon",
    )
    eps_clip_high: float = Field(
        default=0.28,
        description="Upper PPO clipping epsilon",
    )
    kl_coef: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="KL divergence penalty coefficient",
    )
    # LoRA (optional)
    lora_enabled: bool = Field(
        default=True,
        description="Use LoRA for training (vs full fine-tuning)",
    )
    lora_rank: int = Field(
        default=16,
        ge=1,
        le=128,
        description="LoRA rank",
    )
    lora_alpha: int = Field(
        default=32,
        ge=1,
        le=256,
        description="LoRA alpha",
    )
    # Data
    record_dir: str = Field(
        default=".omniagent/rl/records",
        description="Directory for trajectory JSONL records",
    )
    context_length: int = Field(
        default=32768,
        ge=1024,
        le=131072,
        description="Maximum context length for SGLang serving",
    )


class OmniAgentConfig(BaseModel):
    """Main OmniAgent configuration."""

    model_config = ConfigDict(extra="allow")

    version: str = Field(
        default="0.1.0",
        description="Config schema version"
    )
    includes: List[str] = Field(
        default_factory=list,
        description="List of extra config file paths to deep merge"
    )
    agents: Dict[str, AgentConfig] = Field(
        default_factory=dict,
        description="Multi-agent named configurations"
    )

    # Named provider presets
    providers: Dict[str, ProviderConfig] = Field(
        default_factory=dict,
        description="Named LLM provider configs keyed by provider identity"
    )

    # Working directory
    work_dir: Path = Field(
        default_factory=lambda: Path.cwd(),
        description="Working directory for agent operations"
    )

    # API keys (with environment variable fallbacks)
    api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OMNIAGENT_API_KEY") or os.getenv("DEEPSEEK_API_KEY"),
        description="API key for the configured provider"
    )
    openai_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY"),
        description="OpenAI API key (deprecated, use api_key)"
    )
    anthropic_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"),
        description="Anthropic API key (deprecated, use api_key)"
    )

    # Component configs
    tools: ToolsConfig = Field(
        default_factory=ToolsConfig,
        description="Tools configuration"
    )
    agent: AgentConfig = Field(
        default_factory=AgentConfig,
        description="Agent configuration"
    )
    gateway: GatewayConfig = Field(
        default_factory=GatewayConfig,
        description="Gateway configuration"
    )

    usage: UsageConfig = Field(
        default_factory=UsageConfig,
        description="Usage accounting configuration"
    )

    # Feature flags
    enable_progressive_loading: bool = Field(
        default=True,
        description="Enable progressive context loading"
    )
    enable_parallel_execution: bool = Field(
        default=True,
        description="Enable parallel tool execution"
    )
    enable_security_guard: bool = Field(
        default=True,
        description="Enable security guard"
    )
    enable_self_improving: bool = Field(
        default=True,
        description="Enable self-improving system"
    )

    # Skill evolution
    skill_evolution: SkillEvolutionConfig = Field(
        default_factory=SkillEvolutionConfig,
        description="Skill self-evolution settings",
    )

    # Context evolution
    context_evolution: ContextEvolutionConfig = Field(
        default_factory=ContextEvolutionConfig,
        description="Context self-evolution settings",
    )

    # Reinforcement Learning (only for local providers: vllm/sglang)
    rl: RLConfig = Field(
        default_factory=RLConfig,
        description="RL training settings (GRPO, PRM, rollout)",
    )

    # Sentinel agent (task planning and progress tracking)
    sentinel: SentinelConfig = Field(
        default_factory=SentinelConfig,
        description="Sentinel agent settings",
    )

    # Guardian agent (output quality review and safety gate)
    guardian: GuardianConfig = Field(
        default_factory=GuardianConfig,
        description="Guardian agent settings",
    )

    # Memory search
    memory: MemorySearchConfig = Field(
        default_factory=MemorySearchConfig,
        description="Memory search configuration"
    )

    # Chat channels
    channels: ChannelsConfig = Field(
        default_factory=ChannelsConfig,
        description="Chat channels configuration"
    )


    @model_validator(mode="before")
    @classmethod
    def _normalize_provider_layout(cls, value: Any) -> Any:
        """Migrate legacy provider aliases into provider-key identities."""
        if not isinstance(value, dict):
            return value

        data = dict(value)
        providers = data.get("providers")
        if providers is None:
            providers = {}
        elif not isinstance(providers, dict):
            return data

        agent = data.get("agent")
        agent_data = dict(agent) if isinstance(agent, dict) else {}
        active_provider = agent_data.get("model_provider")
        normalized_providers: Dict[str, Any] = {}
        provider_renames: List[tuple[str, str, Optional[str], Optional[str]]] = []

        for raw_key, raw_provider in providers.items():
            provider_key = str(raw_key)
            if not isinstance(raw_provider, dict):
                normalized_providers[provider_key] = raw_provider
                continue

            provider_data = dict(raw_provider)
            legacy_name_raw = provider_data.pop("provider_name", None)
            legacy_name = str(legacy_name_raw).strip() if legacy_name_raw else None
            api_type = _ensure_supported_api_type(_normalize_api_type(provider_data.get("api_type")))
            provider_key_api_type = _default_api_type_for_provider_key(provider_key)
            legacy_name_api_type = _default_api_type_for_provider_key(legacy_name) if legacy_name else None
            if api_type is not None:
                provider_data["api_type"] = api_type

            target_key = provider_key
            if legacy_name and legacy_name != provider_key:
                conflict = legacy_name in providers or legacy_name in normalized_providers
                if provider_key_api_type and not conflict:
                    target_key = legacy_name
                    if not provider_data.get("api_type"):
                        provider_data["api_type"] = provider_key_api_type
                elif not provider_data.get("api_type"):
                    fallback_api_type = provider_key_api_type or legacy_name_api_type
                    if fallback_api_type:
                        provider_data["api_type"] = fallback_api_type
                provider_renames.append((provider_key, target_key, provider_data.get("model_id"), legacy_name))

            if not provider_data.get("api_type") and not _default_api_type_for_provider_key(target_key):
                raise ValueError(f"providers.{target_key}.api_type is required for named provider keys")

            normalized_providers[target_key] = provider_data
            if active_provider == provider_key and target_key != provider_key:
                agent_data["model_provider"] = target_key

        if agent_data:
            active_key = str(agent_data.get("model_provider") or "deepseek").strip()
            active_provider = normalized_providers.get(active_key)
            has_agent_model = str(agent_data.get("model_id") or "").strip()
            if not has_agent_model and isinstance(active_provider, dict) and active_provider.get("model_id"):
                agent_data["model_id"] = active_provider["model_id"]
            data["agent"] = agent_data
        data["providers"] = normalized_providers
        cls._validate_active_provider_key(data, normalized_providers)
        cls._migrate_pricing_override_keys(data, provider_renames)
        return data

    @staticmethod
    def _validate_active_provider_key(data: Dict[str, Any], providers: Dict[str, Any]) -> None:
        agent = data.get("agent")
        active_provider = "deepseek"
        if isinstance(agent, dict):
            raw_active = agent.get("model_provider")
            if raw_active is not None:
                active_provider = str(raw_active).strip()

        if not active_provider:
            return
        if active_provider in providers:
            provider_data = providers[active_provider]
            if isinstance(provider_data, dict):
                api_type = _ensure_supported_api_type(_normalize_api_type(provider_data.get("api_type")))
                if api_type is not None:
                    provider_data["api_type"] = api_type
                elif not _default_api_type_for_provider_key(active_provider):
                    raise ValueError(
                        f"providers.{active_provider}.api_type is required for active named provider key"
                    )
            return

        if _default_api_type_for_provider_key(active_provider):
            return

        raise ValueError(
            f"agent.model_provider={active_provider!r} requires providers.{active_provider}.api_type; "
            "use a built-in provider key or add a provider entry with api_type"
        )

    @staticmethod
    def _migrate_pricing_override_keys(
        data: Dict[str, Any],
        provider_renames: List[tuple[str, str, Optional[str], Optional[str]]],
    ) -> None:
        if not provider_renames:
            return
        usage = data.get("usage")
        if not isinstance(usage, dict):
            return
        overrides = usage.get("pricing_overrides")
        if not isinstance(overrides, dict):
            return

        migrated = dict(overrides)
        for old_key, new_key, model_id, legacy_name in provider_renames:
            if not model_id:
                continue
            model_key = str(model_id)
            destination = f"{new_key}/{model_key}"
            candidates = [destination, f"{old_key}/{model_key}"]
            if legacy_name:
                candidates.insert(0, f"{legacy_name}/{model_key}")
            for candidate in candidates:
                if candidate in migrated:
                    migrated.setdefault(destination, migrated[candidate])
                    break
            for candidate in candidates:
                if candidate != destination:
                    migrated.pop(candidate, None)
        usage["pricing_overrides"] = migrated
