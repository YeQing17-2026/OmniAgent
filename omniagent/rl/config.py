"""RL Config Adapter — bridges OmniAgent RLConfig to RL args namespace.

Converts OmniAgent's Pydantic RLConfig into the argparse-like namespace
expected by the api_server and rollout modules.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RLArgs:
    """Args namespace compatible with the RL API server and rollout.

    Created from OmniAgent's RLConfig. Additional fields can be set
    directly on this object before passing to RLAPIServer.
    """

    # SGLang serving
    sglang_router_ip: str = "localhost"
    sglang_router_port: int = 30000
    hf_checkpoint: str = ""
    served_model_name: str = "default"

    # PRM
    prm_enable: bool = True
    prm_m: int = 3
    prm_temperature: float = 0.6
    prm_max_new_tokens: int = 50
    prm_model_path: str = ""
    prm_num_gpus: int = 1
    prm_router_ip: Optional[str] = None
    prm_router_port: Optional[int] = None

    # Rollout
    rollout_batch_size: int = 16
    rollout_max_response_len: int = 8192
    rollout_temperature: float = 0.6

    # Training
    train_backend: str = "fsdp"
    lr: float = 1e-5
    eps_clip_low: float = 0.2
    eps_clip_high: float = 0.28
    kl_coef: float = 0.02
    disable_rewards_normalization: bool = True
    advantage_estimator: str = "grpo"

    # LoRA
    lora_enabled: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32

    # Context
    context_length: int = 32768

    # Extra (SLIME may expect these)
    extra: dict = field(default_factory=dict)

    def __getattr__(self, name: str):
        """Fallback to extra dict for fields not explicitly defined."""
        try:
            return self.extra[name]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute '{name}'"
            )

    def __setattr__(self, name: str, value):
        if name in ("extra",):
            super().__setattr__(name, value)
        elif hasattr(self.__class__, name):
            super().__setattr__(name, value)
        else:
            self.extra[name] = value


def make_rl_args(
    sglang_router_ip: str = "localhost",
    sglang_router_port: int = 30000,
    hf_checkpoint: str = "",
    served_model_name: str = "default",
    prm_enable: bool = True,
    prm_m: int = 3,
    prm_temperature: float = 0.6,
    prm_max_new_tokens: int = 50,
    prm_model_path: str = "",
    prm_num_gpus: int = 1,
    prm_router_ip: Optional[str] = None,
    prm_router_port: Optional[int] = None,
    rollout_batch_size: int = 16,
    rollout_max_response_len: int = 8192,
    rollout_temperature: float = 0.6,
    train_backend: str = "fsdp",
    lr: float = 1e-5,
    eps_clip_low: float = 0.2,
    eps_clip_high: float = 0.28,
    kl_coef: float = 0.02,
    lora_enabled: bool = True,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    context_length: int = 32768,
) -> RLArgs:
    """Create RLArgs from individual parameters.

    Can be called with OmniAgentConfig fields directly.
    """
    return RLArgs(
        sglang_router_ip=sglang_router_ip,
        sglang_router_port=sglang_router_port,
        hf_checkpoint=hf_checkpoint,
        served_model_name=served_model_name,
        prm_enable=prm_enable,
        prm_m=prm_m,
        prm_temperature=prm_temperature,
        prm_max_new_tokens=prm_max_new_tokens,
        prm_model_path=prm_model_path,
        prm_num_gpus=prm_num_gpus,
        prm_router_ip=prm_router_ip,
        prm_router_port=prm_router_port,
        rollout_batch_size=rollout_batch_size,
        rollout_max_response_len=rollout_max_response_len,
        rollout_temperature=rollout_temperature,
        train_backend=train_backend,
        lr=lr,
        eps_clip_low=eps_clip_low,
        eps_clip_high=eps_clip_high,
        kl_coef=kl_coef,
        lora_enabled=lora_enabled,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        context_length=context_length,
    )


def make_rl_args_from_config(config) -> RLArgs:
    """Create RLArgs from a OmniAgentConfig object."""
    rl = config.rl
    return make_rl_args(
        sglang_router_ip=rl.sglang_router_ip,
        sglang_router_port=rl.sglang_router_port,
        hf_checkpoint=rl.hf_checkpoint,
        served_model_name=rl.served_model_name,
        prm_enable=rl.prm_enabled,
        prm_m=rl.prm_m,
        prm_temperature=rl.prm_temperature,
        prm_max_new_tokens=rl.prm_max_tokens,
        prm_model_path=rl.prm_model_path,
        prm_num_gpus=rl.prm_num_gpus,
        rollout_batch_size=rl.rollout_batch_size,
        rollout_max_response_len=rl.rollout_max_response_len,
        rollout_temperature=rl.rollout_temperature,
        train_backend=rl.train_backend,
        lr=rl.lr,
        eps_clip_low=rl.eps_clip_low,
        eps_clip_high=rl.eps_clip_high,
        kl_coef=rl.kl_coef,
        lora_enabled=rl.lora_enabled,
        lora_rank=rl.lora_rank,
        lora_alpha=rl.lora_alpha,
        context_length=rl.context_length,
    )
