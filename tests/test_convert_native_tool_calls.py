"""Regression tests for DeepSeek-style tool_calls arguments (issue #52).

Some OpenAI-compatible providers (DeepSeek among them) return the
``function.arguments`` field of a native tool call already parsed as a JSON
object instead of a JSON string. ``_convert_native_tool_calls`` must accept
both shapes instead of raising ``TypeError``.
"""

from pathlib import Path

from omniagent.agents.reflexion import ReflexionAgent
from omniagent.config import OmniAgentConfig

from test_memory_recall_priority import FakeLLMProvider


def _make_agent(tmp_path: Path) -> ReflexionAgent:
    config = OmniAgentConfig(api_key="", enable_self_improving=False)
    return ReflexionAgent(
        config=config,
        llm_provider=FakeLLMProvider(),
        work_dir=tmp_path,
        enable_security=False,
    )


def test_convert_native_tool_calls_accepts_parsed_dict_arguments(tmp_path):
    agent = _make_agent(tmp_path)
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "read_file", "arguments": {"path": "/tmp/x.py"}},
        }
    ]

    actions = agent._convert_native_tool_calls(tool_calls)

    assert actions == [
        {"tool": "read_file", "params": {"path": "/tmp/x.py"}, "tool_call_id": "call_1"}
    ]


def test_convert_native_tool_calls_still_accepts_json_string_arguments(tmp_path):
    agent = _make_agent(tmp_path)
    tool_calls = [
        {
            "id": "call_2",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": '{"path": "/tmp/y.py"}',
            },
        }
    ]

    actions = agent._convert_native_tool_calls(tool_calls)

    assert actions == [
        {"tool": "read_file", "params": {"path": "/tmp/y.py"}, "tool_call_id": "call_2"}
    ]


def test_convert_native_tool_calls_recovers_from_invalid_arguments(tmp_path):
    agent = _make_agent(tmp_path)
    tool_calls = [
        {
            "id": "call_3",
            "type": "function",
            "function": {"name": "read_file", "arguments": "not json"},
        }
    ]

    actions = agent._convert_native_tool_calls(tool_calls)

    assert actions == [{"tool": "read_file", "params": {}, "tool_call_id": "call_3"}]
