"""Tests for LLM provider URL handling."""

import pytest

from omniagent.agents.llm import (
    DeepSeekLLMProvider,
    MiniMaxLLMProvider,
    OpenRouterLLMProvider,
    create_llm_provider,
    normalize_openai_chat_completions_url,
)


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


def test_minimax_default_uses_official_openai_compatible_base_url():
    provider = create_llm_provider(provider="minimax", api_key="test-key")

    assert isinstance(provider, MiniMaxLLMProvider)
    assert provider.base_url == "https://api.minimax.io/v1"
    assert provider.model == "MiniMax-M3"


def test_minimax_accepts_cn_openai_compatible_base_url():
    provider = create_llm_provider(
        provider="minimax",
        api_key="test-key",
        api_url="https://api.minimaxi.com/v1",
    )

    assert isinstance(provider, MiniMaxLLMProvider)
    assert provider.base_url == "https://api.minimaxi.com/v1"
