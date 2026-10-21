"""Exercise model selection through agent configuration and SDK serialization."""

import json

import anthropic
import httpx
import openai
import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from omniagent.agents.llm import LLMMessage
from omniagent.agents.reflexion import ReflexionAgent
from omniagent.config.models import AgentConfig, OmniAgentConfig


@pytest.mark.parametrize("provider", ["minimax", "minimax-anthropic"])
@pytest.mark.parametrize("region", ["minimax.io", "minimaxi.com"])
@pytest.mark.parametrize(
    ("agent_model", "provider_model", "expected"),
    [
        (None, None, "MiniMax-M3.1-Flash"),
        ("MiniMax-M3", None, "MiniMax-M3"),
        ("MiniMax-M2.7", None, "MiniMax-M2.7"),
        ("deepseek-chat", None, "deepseek-chat"),
        ("saved-custom", None, "saved-custom"),
        (None, "MiniMax-M2.7", "MiniMax-M2.7"),
        ("MiniMax-M3", "provider-custom", "provider-custom"),
    ],
)
@pytest.mark.asyncio
async def test_config_to_sdk_payload(
    monkeypatch, tmp_path, provider, region, agent_model, provider_model, expected
):
    use_anthropic = provider == "minimax-anthropic"
    base_url = f"https://api.{region}/" + ("anthropic" if use_anthropic else "v1")
    captured = []

    async def handler(request):
        captured.append((str(request.url), json.loads(request.content)))
        if use_anthropic:
            response = {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": expected,
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        else:
            response = {
                "id": "chatcmpl_test",
                "object": "chat.completion",
                "created": 0,
                "model": expected,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        return httpx.Response(200, json=response)

    def client_factory(**kwargs):
        kwargs["http_client"] = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return (AsyncAnthropic if use_anthropic else AsyncOpenAI)(**kwargs)

    monkeypatch.setattr(
        anthropic if use_anthropic else openai,
        "AsyncAnthropic" if use_anthropic else "AsyncOpenAI",
        client_factory,
    )
    agent_data = {"model_provider": provider, "context_window_size": 0}
    if agent_model is not None:
        agent_data["model_id"] = agent_model
    provider_data = {"api_key": "test-key", "api_url": base_url}
    if provider_model is not None:
        provider_data["model_id"] = provider_model
    config = OmniAgentConfig(
        agent=agent_data,
        providers={provider: provider_data},
        memory={"enabled": False},
        sentinel={"enabled": False},
        guardian={"enabled": False},
        enable_self_improving=False,
    )
    agent = ReflexionAgent(config, work_dir=tmp_path, enable_security=False)
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"query":"hello"}'},
    }
    messages = [
        LLMMessage(role="system", content="Keep this system instruction."),
        LLMMessage(role="user", content="hello"),
        LLMMessage(role="assistant", content="Checking.", tool_calls=[tool_call]),
        LLMMessage(role="tool", content="saved result", tool_call_id="call_1"),
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a value",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }
    ]
    response = await agent.llm.chat(messages, tools=tools)
    assert response.content == "ok"
    assert config.agent.model_id == expected
    assert len(captured) == 1
    url, payload = captured[0]
    assert url == base_url + ("/v1/messages" if use_anthropic else "/chat/completions")
    assert payload["model"] == expected
    assert payload["max_tokens"] == 4096
    # Neither adapter injects unsupported thinking or effort overrides.
    assert (
        not {"thinking", "reasoning_effort", "output_config", "service_tier", "reasoning_split"}
        & payload.keys()
    )
    if use_anthropic:
        assert payload["system"] == "Keep this system instruction."
        assert payload["messages"][-2]["content"][1] == {
            "type": "tool_use",
            "id": "call_1",
            "name": "lookup",
            "input": {"query": "hello"},
        }
        assert payload["messages"][-1]["content"] == [
            {"type": "tool_result", "tool_use_id": "call_1", "content": "saved result"}
        ]
        assert payload["tools"][0]["input_schema"] == tools[0]["function"]["parameters"]
    else:
        assert payload["messages"][0] == {
            "role": "system",
            "content": "Keep this system instruction.",
        }
        assert payload["messages"][-2]["tool_calls"] == [tool_call]
        assert payload["messages"][-1] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "saved result",
        }
        assert payload["tools"] == tools


@pytest.mark.parametrize(("choice", "provider"), [(4, "minimax"), (5, "minimax-anthropic")])
def test_onboarding_saves_flash_default(tmp_path, choice, provider):
    import yaml
    from click.testing import CliRunner
    from omniagent.cli.main import onboard

    config_path = tmp_path / "config.yaml"
    result = CliRunner().invoke(
        onboard,
        input=f"{choice}\ntest-key\ny\ny\n",
        obj={"config": OmniAgentConfig(), "config_path": str(config_path)},
    )
    assert result.exit_code == 0, result.output
    saved = yaml.safe_load(config_path.read_text())
    assert saved["providers"][provider]["model_id"] == "MiniMax-M3.1-Flash"
    assert saved["agent"]["model_provider"] == provider


def test_other_provider_defaults_and_explicit_config_are_unchanged():
    assert AgentConfig().model_id == "deepseek-chat"
    for provider in ("deepseek", "openai", "anthropic", "ollama", "gemini", "openrouter", "custom"):
        assert AgentConfig(model_provider=provider).model_id == "deepseek-chat"
        assert (
            AgentConfig(model_provider=provider, model_id="saved-model").model_id == "saved-model"
        )
    data = {"model_provider": "minimax"}
    AgentConfig.model_validate(data)
    assert data == {"model_provider": "minimax"}
