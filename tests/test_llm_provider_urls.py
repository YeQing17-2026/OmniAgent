"""Tests for LLM provider URL handling."""

import pytest

from omniagent.agents.llm import (
    CHEAPER_INFERENCE_API_URL,
    CHEAPER_INFERENCE_DEFAULT_MODEL,
    DeepSeekLLMProvider,
    OpenAILLMProvider,
    OpenRouterLLMProvider,
    create_llm_provider,
    normalize_openai_chat_completions_url,
)
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


def test_cheaperinference_uses_openai_compatible_defaults():
    provider = create_llm_provider(provider="cheaperinference", api_key="test-key")

    assert isinstance(provider, OpenAILLMProvider)
    assert provider.base_url == CHEAPER_INFERENCE_API_URL
    assert provider.model == CHEAPER_INFERENCE_DEFAULT_MODEL


def test_cheaperinference_preserves_custom_model_and_api_url():
    provider = create_llm_provider(
        provider="cheaperinference",
        api_key="test-key",
        model="claude-sonnet-5",
        api_url="https://gateway.example.test/v1",
    )

    assert isinstance(provider, OpenAILLMProvider)
    assert provider.base_url == "https://gateway.example.test/v1"
    assert provider.model == "claude-sonnet-5"


def test_agent_config_accepts_cheaperinference_provider():
    config = AgentConfig(model_provider="cheaperinference")

    assert config.model_provider == "cheaperinference"


def test_openrouter_preserves_custom_api_url():
    provider = OpenRouterLLMProvider(
        api_key="test-key",
        model="openai/gpt-4o",
        api_url="https://gateway.example.test/api/v1",
    )

    assert provider.base_url == "https://gateway.example.test/api/v1"
