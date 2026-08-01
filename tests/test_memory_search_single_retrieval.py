"""Regression tests for single-retrieval memory_search (issue #39).

A single ``memory_search`` tool call must retrieve from memory exactly once.
Previously ``ReflexionAgent._execute_tool`` pre-ran ``memory_manager.search()``
to capture L0 context hints and then ``MemorySearchTool.execute`` ran the same
search again, doubling retrieval latency and log noise.
"""

import asyncio

from test_memory_recall_priority import EmptyMemoryManager, FakeLLMProvider

from omniagent.agents.memory_manager import (
    MemorySearchResult,
    MemorySearchTool,
)
from omniagent.agents.reflexion import ReflexionAgent
from omniagent.config import OmniAgentConfig


class CountingMemoryManager(EmptyMemoryManager):
    """Memory manager that records how many times search() is invoked."""

    def __init__(self):
        self.search_calls = 0

    def search(self, query, max_results=6, min_score=0.35):
        self.search_calls += 1
        return [
            MemorySearchResult(
                chunk_id="chunk-1",
                path="MEMORY.md",
                source="memory",
                start_line=1,
                end_line=3,
                text="user prefers Python",
                score=0.9,
                snippet="user prefers Python",
            )
        ]


def test_memory_search_tool_exposes_context_hints_in_metadata():
    manager = CountingMemoryManager()
    tool = MemorySearchTool(manager)

    result = asyncio.run(tool.execute({"query": "Python"}))

    assert result.success
    assert manager.search_calls == 1
    hints = result.metadata.get("context_hints")
    assert hints is not None
    assert hints == [{"path": "MEMORY.md", "snippet": "user prefers Python", "score": 0.9}]


def test_memory_search_tool_metadata_hints_empty_when_no_results():
    tool = MemorySearchTool(EmptyMemoryManager())

    result = asyncio.run(tool.execute({"query": "nothing found"}))

    assert result.success
    # No results -> no context_hints metadata, and no duplicate search
    assert result.metadata.get("context_hints") is None


def test_execute_tool_derives_hints_from_metadata_without_research(tmp_path):
    manager = CountingMemoryManager()
    config = OmniAgentConfig(api_key="", enable_self_improving=False)
    config.memory.enabled = True

    agent = ReflexionAgent(
        config=config,
        llm_provider=FakeLLMProvider(),
        work_dir=tmp_path,
        enable_security=False,
    )
    agent.memory_manager = manager
    agent.register_tool(MemorySearchTool(manager))

    result = asyncio.run(
        agent._execute_tool({"tool": "memory_search", "params": {"query": "Python"}})
    )

    # Exactly one retrieval for one tool call
    assert manager.search_calls == 1
    assert "user prefers Python" in result
    assert agent._context_hints == [
        {"path": "MEMORY.md", "snippet": "user prefers Python", "score": 0.9}
    ]
