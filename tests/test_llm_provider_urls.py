"""Tests for LLM provider URL handling."""

import anthropic
import httpx
import openai
import pytest
from anthropic import AsyncAnthropic as SDKAsyncAnthropic
from openai import AsyncOpenAI as SDKAsyncOpenAI

from omniagent.agents.llm import (
    DeepSeekLLMProvider,
    LLMMessage,
    MiniMaxAnthropicLLMProvider,
    MiniMaxLLMProvider,
    OpenRouterLLMProvider,
    create_llm_provider,
    normalize_openai_chat_completions_url,
)
from omniagent.agents.context_manager import resolve_context_window_size
from omniagent.config.models import AgentConfig


@pytest.mark.parametrize(
    ("api_url", "expected"),
    [
        ("https://api.deepseek.com", "https://api.deepseek.com/chat/completions"),
        ("https://api.deepseek.com/", "https://api.deepseek.com/chat/completions"),
        ("https://api.deepseek.com/v1", "https://api.deepseek.com/v1/chat/completions"),
        (
            "https://api.deepseek.com/chat/completions",
            "https://api.deepseek.com/chat/completions",
        ),
        ("https://api.openai.com/v1", "https://api.openai.com/v1/chat/completions"),
        (
            "https://ark.cn-beijing.volces.com/api/coding/v3",
            "https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions",
        ),
    ],
)
def test_normalize_openai_chat_completions_url(api_url, expected):
    assert normalize_openai_chat_completions_url(api_url) == expected


def test_normalize_openai_chat_completions_url_rejects_empty_url():
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_openai_chat_completions_url("  ")


def test_deepseek_default_uses_official_base_url_and_chat_endpoint():
    provider = create_llm_provider(provider="deepseek", api_key="test-key")

    assert isinstance(provider, DeepSeekLLMProvider)
    assert provider.base_url == "https://api.deepseek.com"
    assert provider.chat_completions_url == "https://api.deepseek.com/chat/completions"
    assert provider.api_url == provider.chat_completions_url


def test_deepseek_accepts_openai_compatible_base_url():
    provider = DeepSeekLLMProvider(
        api_key="test-key",
        api_url="https://ark.cn-beijing.volces.com/api/coding/v3",
        model="minimax-latest",
    )

    assert provider.base_url == "https://ark.cn-beijing.volces.com/api/coding/v3"
    assert (
        provider.chat_completions_url
        == "https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions"
    )


def test_openrouter_preserves_custom_api_url():
    provider = OpenRouterLLMProvider(
        api_key="test-key",
        model="openai/gpt-4o",
        api_url="https://gateway.example.test/api/v1",
    )

    assert provider.base_url == "https://gateway.example.test/api/v1"


def test_minimax_provider_defaults_cover_supported_models():
    openai_provider = create_llm_provider(provider="minimax", api_key="test-key")
    anthropic_provider = create_llm_provider(
        provider="minimax-anthropic",
        api_key="test-key",
        model="MiniMax-M2.7",
    )

    assert isinstance(openai_provider, MiniMaxLLMProvider)
    assert openai_provider.base_url == "https://api.minimax.io/v1"
    assert openai_provider.model == "MiniMax-M3"
    assert openai_provider.SUPPORTED_MODELS == ("MiniMax-M3", "MiniMax-M2.7")

    assert isinstance(anthropic_provider, MiniMaxAnthropicLLMProvider)
    assert anthropic_provider.api_url == "https://api.minimax.io/anthropic"
    assert anthropic_provider.model == "MiniMax-M2.7"
    assert anthropic_provider.SUPPORTED_MODELS == ("MiniMax-M3", "MiniMax-M2.7")


@pytest.mark.parametrize("provider", ["minimax", "minimax-anthropic"])
def test_minimax_provider_values_are_valid_agent_config_options(provider):
    assert AgentConfig(model_provider=provider).model_provider == provider


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("MiniMax-M3", 1_000_000),
        ("MiniMax-M2.7", 204_800),
    ],
)
def test_minimax_context_window_resolution(model_id, expected):
    assert resolve_context_window_size(model_id) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("api_url", "expected_url"),
    [
        (
            "https://api.minimax.io/v1",
            "https://api.minimax.io/v1/chat/completions",
        ),
        (
            "https://api.minimaxi.com/v1",
            "https://api.minimaxi.com/v1/chat/completions",
        ),
    ],
)
async def test_minimax_openai_sdk_request_urls(monkeypatch, api_url, expected_url):
    seen_urls = []

    async def handler(request):
        seen_urls.append(str(request.url))
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "chatcmpl_test",
                "object": "chat.completion",
                "created": 0,
                "model": "MiniMax-M3",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    def client_factory(**kwargs):
        kwargs["http_client"] = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return SDKAsyncOpenAI(**kwargs)

    monkeypatch.setattr(openai, "AsyncOpenAI", client_factory)
    provider = create_llm_provider(
        provider="minimax",
        api_key="test-key",
        api_url=api_url,
    )

    await provider.chat([LLMMessage(role="user", content="hello")])

    assert seen_urls == [expected_url]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("api_url", "expected_url"),
    [
        (
            "https://api.minimax.io/anthropic",
            "https://api.minimax.io/anthropic/v1/messages",
        ),
        (
            "https://api.minimaxi.com/anthropic",
            "https://api.minimaxi.com/anthropic/v1/messages",
        ),
    ],
)
async def test_minimax_anthropic_sdk_request_urls(monkeypatch, api_url, expected_url):
    seen_urls = []

    async def handler(request):
        seen_urls.append(str(request.url))
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "MiniMax-M3",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    def client_factory(**kwargs):
        kwargs["http_client"] = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return SDKAsyncAnthropic(**kwargs)

    monkeypatch.setattr(anthropic, "AsyncAnthropic", client_factory)
    provider = create_llm_provider(
        provider="minimax-anthropic",
        api_key="test-key",
        api_url=api_url,
    )

    await provider.chat([LLMMessage(role="user", content="hello")])

    assert seen_urls == [expected_url]
