"""Reinforcement Learning (GRPO) training system for OmniAgent.

RL training pipeline for OmniAgent:
- FastAPI proxy server (api_server.py) for logprob collection and PRM scoring
- Async rollout worker (rollout.py) bridging proxy and SLIME trainer
- Config adapter (config.py) converting OmniAgent RLConfig to args namespace

**Only activates when model_provider is "vllm" or "sglang".**
When using remote LLM providers (DeepSeek, OpenAI, etc.), RL is completely
disabled and zero overhead.

Launch script references (for SLIME integration):
    --rollout-function-path omniagent.rl.rollout.generate_rollout
    --custom-generate-function-path omniagent.rl.api_server.generate
    --custom-rm-path omniagent.rl.api_server.reward_func
"""

from .config import RLArgs, make_rl_args, make_rl_args_from_config
from .api_server import (
    RLAPIServer,
    reward_func,
    generate,
    _build_prm_judge_prompt,
    _parse_prm_score,
    _majority_vote,
)
from .rollout import (
    AsyncRolloutWorker,
    generate_rollout,
    get_global_worker,
    stop_global_worker,
)

__all__ = [
    # Config
    "RLArgs",
    "make_rl_args",
    "make_rl_args_from_config",
    # API Server
    "RLAPIServer",
    "reward_func",
    "generate",
    "_build_prm_judge_prompt",
    "_parse_prm_score",
    "_majority_vote",
    # Rollout
    "AsyncRolloutWorker",
    "generate_rollout",
    "get_global_worker",
    "stop_global_worker",
]
