"""Reasoning must survive SDK responses, agent tool loops, and session reloads."""

import json

import anthropic
import httpx
import openai
import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from omniagent.agents.reflexion import ReflexionAgent
from omniagent.cli.chat import ChatSession
from omniagent.config.models import OmniAgentConfig
from omniagent.gateway.session import SessionManager


@pytest.mark.parametrize("provider", ["minimax", "minimax-anthropic"])
@pytest.mark.parametrize("region", ["minimax.io", "minimaxi.com"])
@pytest.mark.parametrize("known_tool", [True, False])
@pytest.mark.parametrize("model", [None, "MiniMax-M3", "MiniMax-M2.7", "saved-custom"])
@pytest.mark.asyncio
async def test_reasoning_response_to_next_request_and_session(
    monkeypatch, tmp_path, provider, region, known_tool, model
):
    use_anthropic = provider == "minimax-anthropic"
    expected_model = model or "MiniMax-M3.1-Flash"
    tool_name = "read_file" if known_tool else "unavailable_tool"
    tool_input = {"path": str(tmp_path / "fixture.txt")}
    (tmp_path / "fixture.txt").write_text("offline tool result")
    tool_call = {
        "id": "call_reasoning",
        "type": "function",
        "function": {"name": tool_name, "arguments": json.dumps(tool_input)},
    }
    first_blocks = [
        {
            "type": "thinking",
            "thinking": "Keep this reasoning exactly.",
            "signature": "signature-1",
        },
        {"type": "text", "text": "Checking."},
        {"type": "redacted_thinking", "data": "opaque-redacted-data"},
        {"type": "tool_use", "id": "call_reasoning", "name": tool_name, "input": tool_input},
    ]
    final_blocks = [
        {"type": "thinking", "thinking": "Final reasoning.", "signature": "signature-2"},
        {"type": "text", "text": "Done."},
    ]
    base_url = f"https://api.{region}/" + ("anthropic" if use_anthropic else "v1")
    captured = []

    async def handler(request):
        payload = json.loads(request.content)
        captured.append(payload)
        assert str(request.url) == base_url + (
            "/v1/messages" if use_anthropic else "/chat/completions"
        )
        assert payload["model"] == expected_model
        first = len(captured) == 1
        if use_anthropic:
            response = {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": expected_model,
                "content": first_blocks if first else final_blocks,
                "stop_reason": "tool_use" if first else "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        else:
            message = {
                "role": "assistant",
                "content": None if first else "Done.",
                "reasoning_content": (
                    "Keep this reasoning exactly." if first else "Final reasoning."
                ),
            }
            if first:
                message["tool_calls"] = [tool_call]
            response = {
                "id": "chatcmpl_test",
                "object": "chat.completion",
                "created": 0,
                "model": expected_model,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if first else "stop",
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
    agent_data = {"model_provider": provider, "context_window_size": 0, "compaction_enabled": False}
    if model is not None:
        agent_data["model_id"] = model
    config = OmniAgentConfig(
        agent=agent_data,
        providers={provider: {"api_key": "test-key", "api_url": base_url}},
        memory={"enabled": False},
        sentinel={"enabled": False},
        guardian={"enabled": False},
        enable_self_improving=False,
    )
    agent = ReflexionAgent(config, work_dir=tmp_path, enable_security=False)
    result = await agent._execute_single_attempt("Read the fixture.")
    assert result.success and result.response == "Done."
    assert len(captured) == 2

    def assert_tool_history(payload):
        messages = payload["messages"]
        assistant = next(m for m in messages if m["role"] == "assistant")
        if use_anthropic:
            assert assistant["content"] == first_blocks
            tool_result = next(
                b
                for m in messages
                if m["role"] == "user" and isinstance(m["content"], list)
                for b in m["content"]
                if b["type"] == "tool_result"
            )
            assert tool_result["tool_use_id"] == "call_reasoning"
        else:
            assert assistant["reasoning_content"] == "Keep this reasoning exactly."
            assert assistant["tool_calls"] == [tool_call]
            tool_result = next(m for m in messages if m["role"] == "tool")
            assert tool_result["tool_call_id"] == "call_reasoning"
        assert ("offline tool result" if known_tool else "not available") in tool_result["content"]

    assert_tool_history(captured[1])

    # Exercise the real CLI persistence bridge and fresh disk deserialization.
    chat = ChatSession.__new__(ChatSession)
    chat.agent = agent
    chat.config = config
    chat.work_dir = tmp_path
    chat._last_result_metadata = {}
    storage = tmp_path / "sessions"
    chat.session_manager = SessionManager(storage)
    session = chat.session_manager.create_session("test", "chat", "reasoning")
    chat._save_current_session(session)
    chat.session_manager = SessionManager(storage)
    restored = chat.session_manager.get_session("reasoning")
    agent.conversation_history = []
    assert chat._restore_session(restored)
    result = await agent._execute_single_attempt("Continue after reload.")
    assert result.success and len(captured) == 3
    assert_tool_history(captured[2])
    final_assistant = [m for m in captured[2]["messages"] if m["role"] == "assistant"][-1]
    if use_anthropic:
        assert final_assistant["content"] == final_blocks
    else:
        assert final_assistant["reasoning_content"] == "Final reasoning."
