"""LLM provider abstraction."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, AsyncIterator

import aiohttp

from omniagent.infra import get_logger

logger = get_logger(__name__)


@dataclass
class LLMMessage:
    """LLM message."""

    role: str  # "system", "user", "assistant", "tool"
    content: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None  # OpenAI-style tool_calls
    tool_call_id: Optional[str] = None  # For role="tool" messages
    name: Optional[str] = None  # Tool name for tool messages
    timestamp: Optional[str] = None  # ISO format timestamp for user/assistant messages


@dataclass
class LLMResponse:
    """LLM response."""

    content: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_calls: Optional[List[Dict]] = None  # OpenAI-style tool_calls
    logprobs: Optional[List[Dict[str, Any]]] = None  # Per-token logprobs (vLLM/SGLang)


class LLMProvider(ABC):
    """Abstract LLM provider."""

    @property
    def supports_native_function_calling(self) -> bool:
        """Whether this provider supports native function calling."""
        return False

    @abstractmethod
    async def chat(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        tools: Optional[List[Dict]] = None,
    ) -> LLMResponse:
        """
        Send chat request to LLM.

        Args:
            messages: Conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response
            tools: Optional tool schemas for native function calling

        Returns:
            LLM response
        """
        pass

    @abstractmethod
    async def chat_stream(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """
        Stream chat response from LLM.

        Args:
            messages: Conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Yields:
            Response chunks
        """
        pass

    @staticmethod
    def strip_thinking(content: str) -> str:
        """Strip thinking/reasoning blocks from model output.

        Some providers (DeepSeek, MiniMax, Ollama) may include thinking tokens
        in the content field. This method removes common patterns:
        - <think...</think] or <thinking...</thinking> blocks
        - Unclosed <think... tags (model didn't close the tag)
        """
        import re as _re

        if not content:
            return content

        # Strip thinking/reasoning blocks from model output.
        # Handles: <think\n...\n</think\n>, <thinking>...</thinking>, <think attrs>...</think >

        # Step 1: Remove all closed thinking blocks
        # Try <thinking>...</thinking> first (longest tag name, avoids partial match)
        for _ in range(3):  # iterate to handle nested or multiple blocks
            prev = content
            content = _re.sub(
                r"<thinking\b[^>]*>.*?</thinking\s*>",
                "", content, flags=_re.DOTALL | _re.IGNORECASE,
            )
            content = _re.sub(
                r"<reasoning\b[^>]*>.*?</reasoning\s*>",
                "", content, flags=_re.DOTALL | _re.IGNORECASE,
            )
            # <think...>...</think...> with optional > on both ends
            content = _re.sub(
                r"<think\b[^>]*>.*?</think\s*>",
                "", content, flags=_re.DOTALL | _re.IGNORECASE,
            )
            # DeepSeek: <think\n...\n</think\n (no > in tags)
            content = _re.sub(
                r"<think\b[^\n>]*\n.*?</think\b[^\n>]*\n?",
                "", content, flags=_re.DOTALL | _re.IGNORECASE,
            )
            if content == prev:
                break

        # Step 2: Remove unclosed thinking tags (opening tag but no closing tag)
        if _re.search(r"<(?:thinking|reasoning|think)\b", content, _re.IGNORECASE):
            # Check if there's a closing tag at all
            if not _re.search(r"</(?:thinking|reasoning|think)\b", content, _re.IGNORECASE):
                # No closing tag found — discard from opening tag to end
                content = _re.sub(
                    r"<(?:thinking|reasoning|think)\b.*",
                    "", content, flags=_re.DOTALL | _re.IGNORECASE,
                )

        # Pattern 3: Strip leading blank lines left after removal
        content = content.strip()
        return content


class DeepSeekLLMProvider(LLMProvider):
    """DeepSeek LLM provider with native function calling support (OpenAI-compatible API)."""

    @property
    def supports_native_function_calling(self) -> bool:
        return True

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str = "deepseek-chat",
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model

        logger.info(
            "deepseek_llm_initialized",
            api_url=api_url,
            model=model,
        )

    def _convert_messages(self, messages: List[LLMMessage]) -> List[Dict]:
        """Convert LLMMessage list to OpenAI-compatible API format with tool support."""
        api_messages = []
        for msg in messages:
            if msg.role == "tool":
                api_messages.append({
                    "role": "tool",
                    "content": msg.content or "",
                    "tool_call_id": msg.tool_call_id,
                })
            elif msg.role == "assistant" and msg.tool_calls:
                api_messages.append({
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": msg.tool_calls,
                })
            elif msg.role == "compaction_summary":
                api_messages.append({
                    "role": "user",
                    "content": f"The conversation history before this point was compacted into the following summary:\n<summary>\n{msg.content}\n</summary>",
                })
            else:
                api_messages.append({
                    "role": msg.role,
                    "content": msg.content or "",
                })
        return api_messages

    async def chat(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        tools: Optional[List[Dict]] = None,
    ) -> LLMResponse:
        """Send chat request with optional native function calling."""
        api_messages = self._convert_messages(messages)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        logger.debug(
            "llm_request",
            model=self.model,
            messages_count=len(messages),
            temperature=temperature,
            has_tools=tools is not None,
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                json=payload,
                headers=headers,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(
                        "llm_request_failed",
                        status=resp.status,
                        error=error_text,
                    )
                    raise RuntimeError(f"LLM request failed: {error_text}")

                data = await resp.json()

                choice = data["choices"][0]
                message = choice["message"]
                content = message.get("content") or ""
                # Discard reasoning_content (DeepSeek thinking tokens)
                # Some providers include this; it should not be exposed to the user
                finish_reason = choice.get("finish_reason")
                usage = data.get("usage", {})

                # Extract tool calls
                tool_calls = None
                if "tool_calls" in message:
                    tool_calls = message["tool_calls"]

                logger.info(
                    "llm_response_received",
                    content_length=len(content),
                    finish_reason=finish_reason,
                    has_tool_calls=tool_calls is not None,
                    usage=usage,
                )

                return LLMResponse(
                    content=self.strip_thinking(content),
                    finish_reason=finish_reason,
                    usage=usage,
                    metadata={"model": self.model},
                    tool_calls=tool_calls,
                )

    async def chat_stream(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Stream chat response, filtering out thinking/reasoning blocks."""
        api_messages = self._convert_messages(messages)

        payload = {
            "model": self.model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        logger.debug(
            "llm_stream_request",
            model=self.model,
            messages_count=len(messages),
        )

        import re as _re
        think_open_re = _re.compile(r"<(?:thinking|reasoning|think)\b[^\n>]*\n?", _re.IGNORECASE)
        think_close_re = _re.compile(r"</(?:thinking|reasoning|think)\b[^\n>]*\n?", _re.IGNORECASE)

        in_think_block = False
        tag_buffer = ""

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                json=payload,
                headers=headers,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(
                        "llm_stream_failed",
                        status=resp.status,
                        error=error_text,
                    )
                    raise RuntimeError(f"LLM stream failed: {error_text}")

                async for line in resp.content:
                    line = line.decode("utf-8").strip()

                    if not line:
                        continue

                    if line.startswith("data: "):
                        line = line[6:]

                    if line == "[DONE]":
                        if tag_buffer and not in_think_block:
                            yield tag_buffer
                        break

                    try:
                        chunk = json.loads(line)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")

                        if not content:
                            continue

                        tag_buffer += content

                        while tag_buffer:
                            if in_think_block:
                                close_match = think_close_re.search(tag_buffer)
                                if close_match:
                                    tag_buffer = tag_buffer[close_match.end():]
                                    in_think_block = False
                                else:
                                    tag_buffer = ""
                                    break
                            else:
                                open_match = think_open_re.search(tag_buffer)
                                if open_match:
                                    before = tag_buffer[:open_match.start()]
                                    if before.strip():
                                        yield before
                                    tag_buffer = tag_buffer[open_match.end():]
                                    in_think_block = True
                                else:
                                    last_angle = tag_buffer.rfind("<")
                                    if last_angle >= 0 and len(tag_buffer) - last_angle < 20:
                                        safe_part = tag_buffer[:last_angle]
                                        tag_buffer = tag_buffer[last_angle:]
                                        if safe_part.strip():
                                            yield safe_part
                                    else:
                                        yield tag_buffer
                                        tag_buffer = ""
                                    break

                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(
                            "stream_parse_error",
                            line=line,
                            error=str(e),
                        )
                        continue


class OpenAILLMProvider(LLMProvider):
    """OpenAI LLM provider with native function calling support."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        api_url: Optional[str] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = api_url or "https://api.openai.com/v1"

        logger.info("openai_llm_initialized", model=model, base_url=self.base_url)

    @property
    def supports_native_function_calling(self) -> bool:
        return True

    def _convert_messages(self, messages: List[LLMMessage]) -> List[Dict]:
        """Convert LLMMessage list to OpenAI API format."""
        result = []
        for msg in messages:
            m: Dict[str, Any] = {"role": msg.role}

            if msg.role == "tool":
                m["tool_call_id"] = msg.tool_call_id
                m["content"] = msg.content or ""
                if msg.name:
                    m["name"] = msg.name
            else:
                m["content"] = msg.content or ""

            if msg.tool_calls:
                m["tool_calls"] = msg.tool_calls

            result.append(m)
        return result

    async def chat(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        tools: Optional[List[Dict]] = None,
    ) -> LLMResponse:
        """Send chat request to OpenAI with optional native function calling."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        api_messages = self._convert_messages(messages)

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        logger.debug(
            "openai_request",
            model=self.model,
            messages_count=len(messages),
            has_tools=tools is not None,
        )

        try:
            response = await client.chat.completions.create(**kwargs)
            choice = response.choices[0]

            content = choice.message.content
            tool_calls = None
            if choice.message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.message.tool_calls
                ]

            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens or 0,
                    "completion_tokens": response.usage.completion_tokens or 0,
                    "total_tokens": response.usage.total_tokens or 0,
                }

            logger.info(
                "openai_response_received",
                content_length=len(content) if content else 0,
                finish_reason=choice.finish_reason,
                tool_calls_count=len(tool_calls) if tool_calls else 0,
                usage=usage,
            )

            return LLMResponse(
                content=content,
                finish_reason=choice.finish_reason,
                usage=usage,
                tool_calls=tool_calls,
                metadata={"model": self.model, "provider": "openai"},
            )
        finally:
            await client.close()

    async def chat_stream(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Stream chat response from OpenAI."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        api_messages = self._convert_messages(messages)

        logger.debug(
            "openai_stream_request",
            model=self.model,
            messages_count=len(messages),
        )

        try:
            stream = await client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        finally:
            await client.close()


class AnthropicLLMProvider(LLMProvider):
    """Anthropic LLM provider with native function calling support."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        api_url: Optional[str] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.api_url = api_url

        logger.info("anthropic_llm_initialized", model=model)

    @property
    def supports_native_function_calling(self) -> bool:
        return True

    def _convert_messages(self, messages: List[LLMMessage]) -> tuple:
        """Convert LLMMessage list to Anthropic API format.

        Returns (system_prompt, anthropic_messages).
        Anthropic requires system as a top-level parameter, and tool results
        use content blocks instead of separate role="tool" messages.
        """
        system_parts: List[str] = []
        anthropic_messages: List[Dict] = []

        for msg in messages:
            if msg.role == "system":
                if msg.content:
                    system_parts.append(msg.content)
                continue

            if msg.role == "tool":
                # Tool result → user message with tool_result content block
                result_content = msg.content or ""
                tool_result_block = {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": result_content,
                }
                # Check if last anthropic message is user (append to it)
                if anthropic_messages and anthropic_messages[-1]["role"] == "user":
                    if not isinstance(anthropic_messages[-1]["content"], list):
                        anthropic_messages[-1]["content"] = [
                            {"type": "text", "text": anthropic_messages[-1]["content"]}
                        ]
                    anthropic_messages[-1]["content"].append(tool_result_block)
                else:
                    anthropic_messages.append({
                        "role": "user",
                        "content": [tool_result_block],
                    })
                continue

            if msg.role == "assistant":
                content_blocks: List[Dict] = []

                # Add text content if present
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})

                # Add tool_use blocks
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        fn = tc.get("function", {})
                        try:
                            arguments = json.loads(fn.get("arguments", "{}"))
                        except json.JSONDecodeError:
                            arguments = {}
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": fn.get("name", ""),
                            "input": arguments,
                        })

                anthropic_messages.append({
                    "role": "assistant",
                    "content": content_blocks if content_blocks else [{"type": "text", "text": ""}],
                })
                continue

            # role == "user"
            anthropic_messages.append({
                "role": "user",
                "content": msg.content or "",
            })

        system_prompt = "\n\n".join(system_parts) if system_parts else None
        return system_prompt, anthropic_messages

    def _convert_tools(self, tools: List[Dict]) -> List[Dict]:
        """Convert OpenAI tool format to Anthropic tool format."""
        anthropic_tools = []
        for tool in tools:
            fn = tool.get("function", {})
            anthropic_tools.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return anthropic_tools

    async def chat(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        tools: Optional[List[Dict]] = None,
    ) -> LLMResponse:
        """Send chat request to Anthropic with optional native function calling."""
        import anthropic

        client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.api_url:
            client_kwargs["base_url"] = self.api_url
        client = anthropic.AsyncAnthropic(**client_kwargs)

        system_prompt, anthropic_messages = self._convert_messages(messages)

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        logger.debug(
            "anthropic_request",
            model=self.model,
            messages_count=len(messages),
            has_tools=tools is not None,
        )

        try:
            response = await client.messages.create(**kwargs)

            content = None
            tool_calls = None

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]

            if text_blocks:
                content = "\n".join(b.text for b in text_blocks)
            if tool_use_blocks:
                tool_calls = [
                    {
                        "id": b.id,
                        "type": "function",
                        "function": {
                            "name": b.name,
                            "arguments": json.dumps(b.input),
                        },
                    }
                    for b in tool_use_blocks
                ]

            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.input_tokens or 0,
                    "completion_tokens": response.usage.output_tokens or 0,
                    "total_tokens": (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0),
                }

            logger.info(
                "anthropic_response_received",
                content_length=len(content) if content else 0,
                stop_reason=response.stop_reason,
                tool_calls_count=len(tool_calls) if tool_calls else 0,
                usage=usage,
            )

            return LLMResponse(
                content=content,
                finish_reason=response.stop_reason,
                usage=usage,
                tool_calls=tool_calls,
                metadata={"model": self.model, "provider": "anthropic"},
            )
        finally:
            await client.close()

    async def chat_stream(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Stream chat response from Anthropic."""
        import anthropic

        client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.api_url:
            client_kwargs["base_url"] = self.api_url
        client = anthropic.AsyncAnthropic(**client_kwargs)

        system_prompt, anthropic_messages = self._convert_messages(messages)

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        logger.debug(
            "anthropic_stream_request",
            model=self.model,
            messages_count=len(messages),
        )

        try:
            async with client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        finally:
            await client.close()


class OllamaLLMProvider(LLMProvider):
    """Ollama LLM provider (local models) with native function calling."""

    DEFAULT_API_URL = "http://localhost:11434/api/chat"

    @property
    def supports_native_function_calling(self) -> bool:
        return True

    def __init__(self, api_url: str = None, model: str = "llama3"):
        self.api_url = api_url or self.DEFAULT_API_URL
        self.model = model
        logger.info("ollama_llm_initialized", api_url=self.api_url, model=self.model)

    def _convert_messages(self, messages: List[LLMMessage]) -> List[Dict]:
        api_messages = []
        for msg in messages:
            if msg.role == "tool":
                api_messages.append({
                    "role": "tool",
                    "content": msg.content or "",
                    "tool_call_id": msg.tool_call_id,
                })
            elif msg.role == "assistant" and msg.tool_calls:
                api_messages.append({
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": msg.tool_calls,
                })
            elif msg.role == "compaction_summary":
                api_messages.append({
                    "role": "user",
                    "content": f"The conversation history before this point was compacted into the following summary:\n<summary>\n{msg.content}\n</summary>",
                })
            else:
                api_messages.append({
                    "role": msg.role,
                    "content": msg.content or "",
                })
        return api_messages

    async def chat(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        tools: Optional[List[Dict]] = None,
    ) -> LLMResponse:
        api_messages = self._convert_messages(messages)
        payload = {
            "model": self.model,
            "messages": api_messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        logger.debug("ollama_request", model=self.model, messages_count=len(messages), has_tools=tools is not None)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error("ollama_request_failed", status=resp.status, error=error_text)
                        raise RuntimeError(f"Ollama request failed: {error_text}")
                    data = await resp.json()
                    choice = data["choices"][0]
                    message = choice["message"]
                    content = message.get("content") or ""
                    finish_reason = choice.get("finish_reason")
                    usage = data.get("usage", {})
                    tool_calls = message.get("tool_calls")
                    logger.info("ollama_response", content_length=len(content), finish_reason=finish_reason, has_tool_calls=tool_calls is not None, usage=usage)
                    return LLMResponse(content=content, finish_reason=finish_reason, usage=usage, metadata={"model": self.model}, tool_calls=tool_calls)
        except aiohttp.ClientConnectorError:
            raise RuntimeError(f"Ollama not reachable at {self.api_url}. Make sure Ollama is running.")

    async def chat_stream(self, messages: List[LLMMessage], temperature: float = 0.7, max_tokens: int = 4096) -> AsyncIterator[str]:
        raise NotImplementedError("Ollama streaming not implemented in this provider. Use chat() instead.")


class GoogleGeminiLLMProvider(LLMProvider):
    """Google Gemini LLM provider with native function calling."""

    @property
    def supports_native_function_calling(self) -> bool:
        return True

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash", api_url: str = None):
        self.api_key = api_key
        self.model = model
        self._api_url = api_url
        self._client = None
        logger.info("gemini_llm_initialized", model=model)

    def _ensure_client(self):
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model)
                logger.info("gemini_client_initialized", model=self.model)
            except ImportError:
                raise RuntimeError(
                    "google-generativeai package is required for Gemini provider. "
                    "Install it with: pip install google-generativeai"
                )

    def _convert_messages(self, messages: List[LLMMessage]) -> tuple:
        """Convert LLMMessage list to Gemini format. Returns (system_text, gemini_contents)."""
        system_parts = []
        contents = []
        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content or "")
            elif msg.role == "compaction_summary":
                contents.append({
                    "role": "user",
                    "parts": [{"text": f"The conversation history before this point was compacted into the following summary:\n<summary>\n{msg.content}\n</summary>"}],
                })
            elif msg.role == "user":
                contents.append({"role": "user", "parts": [{"text": msg.content or ""}]})
            elif msg.role == "assistant":
                contents.append({"role": "model", "parts": [{"text": msg.content or ""}]})
            elif msg.role == "tool":
                contents.append({"role": "function", "parts": [{"function_response": {"name": msg.name or "tool", "response": msg.content or ""}}]})
        return "\n".join(system_parts), contents

    async def chat(self, messages: List[LLMMessage], temperature: float = 0.7, max_tokens: int = 4096, stream: bool = False, tools: Optional[List[Dict]] = None) -> LLMResponse:
        self._ensure_client()
        system_text, gemini_contents = self._convert_messages(messages)
        from google.generativeai.types import ContentTypes, GenerateContentConfig

        config_kwargs = {}
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        config_kwargs["max_output_tokens"] = max_tokens

        gen_config = GenerateContentConfig(**config_kwargs) if config_kwargs else None
        tool_configs = None
        if tools:
            from google.generativeai.types import FunctionDeclaration, Tool as GeminiTool
            gemini_tools = []
            for t in tools:
                func = t.get("function", {})
                params = func.get("parameters", {})
                gemini_tools.append(FunctionDeclaration(name=func.get("name", "unknown"), description=func.get("description", ""), parameters=params))
            tool_configs = [GeminiTool(function_declarations=gemini_tools)]

        try:
            response = await self._client.generate_content(
                system_instruction=system_text or None,
                contents=gemini_contents,
                generation_config=gen_config,
                tools=tool_configs,
            )
            content = ""
            tool_calls = None
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        content += part.text
                    if hasattr(part, "function_call") and part.function_call:
                        tool_calls = [{
                            "id": f"call_{hash(part.function_call.name + str(part.function_call.args))[:16]}",
                            "type": "function",
                            "function": {
                                "name": part.function_call.name,
                                "arguments": json.dumps(part.function_call.args) if part.function_call.args else "{}",
                            },
                        }]
            usage = {}
            if response.usage_metadata:
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                    "completion_tokens": response.usage_metadata.candidates_token_count or 0,
                    "total_tokens": response.usage_metadata.total_token_count or 0,
                }
            return LLMResponse(content=content, finish_reason="stop", usage=usage, metadata={"model": self.model}, tool_calls=tool_calls)
        except Exception as e:
            logger.error("gemini_error", error=str(e))
            raise RuntimeError(f"Gemini request failed: {e}")

    async def chat_stream(self, messages, temperature=0.7, max_tokens=4096):
        raise NotImplementedError("Gemini streaming not implemented. Use chat() instead.")


class OpenRouterLLMProvider(OpenAILLMProvider):
    """OpenRouter LLM provider (aggregated model access)."""

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str, model: str = "openai/gpt-4o", api_url: str = None):
        super().__init__(
            api_key=api_key,
            model=model,
            api_url=api_url or self.DEFAULT_BASE_URL,
        )
        # Override the base URL for OpenRouter
        self.base_url = self.DEFAULT_BASE_URL
        logger.info("openrouter_llm_initialized", model=model, base_url=self.base_url)


class LocalInferenceLLMProvider(OpenAILLMProvider):
    """Local inference provider (vLLM/SGLang) with logprobs support for RL.

    vLLM and SGLang both expose OpenAI-compatible APIs.
    This provider extends OpenAILLMProvider to collect per-token logprobs,
    which are essential for GRPO/PPO importance sampling.

    Only used when model_provider is "vllm" or "sglang".
    """

    def __init__(
        self,
        api_url: str,
        api_key: str = "not-needed",
        model: str = "default",
        logprobs_enabled: bool = True,
        top_logprobs: int = 0,
    ):
        super().__init__(api_key=api_key, model=model, api_url=api_url)
        self._logprobs_enabled = logprobs_enabled
        self._top_logprobs = top_logprobs

    @property
    def supports_logprobs(self) -> bool:
        """Whether this provider can return per-token logprobs."""
        return True

    async def chat(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        tools: Optional[List[Dict]] = None,
    ) -> LLMResponse:
        """Send chat request with logprobs collection."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        api_messages = self._convert_messages(messages)

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        # Inject logprobs parameters for RL
        if self._logprobs_enabled:
            kwargs["logprobs"] = True
            if self._top_logprobs > 0:
                kwargs["top_logprobs"] = self._top_logprobs

        try:
            response = await client.chat.completions.create(**kwargs)
            choice = response.choices[0]

            content = choice.message.content
            tool_calls = None
            if choice.message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.message.tool_calls
                ]

            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens or 0,
                    "completion_tokens": response.usage.completion_tokens or 0,
                    "total_tokens": response.usage.total_tokens or 0,
                }

            # Extract per-token logprobs
            logprobs = None
            if self._logprobs_enabled and hasattr(choice, 'logprobs') and choice.logprobs and choice.logprobs.content:
                logprobs = []
                for token_lp in choice.logprobs.content:
                    entry: Dict[str, Any] = {
                        "token": token_lp.token,
                        "logprob": token_lp.logprob,
                    }
                    if token_lp.top_logprobs:
                        entry["top_logprobs"] = [
                            {"token": tlp.token, "logprob": tlp.logprob}
                            for tlp in token_lp.top_logprobs
                        ]
                    logprobs.append(entry)

            provider_name = "vllm"
            if self.base_url and "sglang" in self.base_url.lower():
                provider_name = "sglang"

            return LLMResponse(
                content=content,
                finish_reason=choice.finish_reason,
                usage=usage,
                tool_calls=tool_calls,
                metadata={"model": self.model, "provider": provider_name},
                logprobs=logprobs,
            )
        finally:
            await client.close()


def create_llm_provider(
    provider: str = "deepseek",
    api_key: str = "",
    model: str = "",
    api_url: str = None,
) -> LLMProvider:
    """Factory function to create an LLM provider instance."""
    provider = provider.lower()

    if provider == "deepseek":
        return DeepSeekLLMProvider(api_key=api_key, model=model or "deepseek-chat", api_url=api_url or "https://api.deepseek.com/v1")
    elif provider == "openai":
        return OpenAILLMProvider(api_key=api_key, model=model or "gpt-4o", api_url=api_url)
    elif provider == "anthropic":
        return AnthropicLLMProvider(api_key=api_key, model=model or "claude-sonnet-4-20250514", api_url=api_url)
    elif provider == "ollama":
        return OllamaLLMProvider(api_url=api_url, model=model or "llama3")
    elif provider == "gemini":
        return GoogleGeminiLLMProvider(api_key=api_key, model=model or "gemini-2.0-flash", api_url=api_url)
    elif provider == "openrouter":
        return OpenRouterLLMProvider(api_key=api_key, model=model or "openai/gpt-4o", api_url=api_url)
    elif provider in ("vllm", "sglang"):
        default_url = "http://localhost:8000/v1" if provider == "vllm" else "http://localhost:30000/v1"
        return LocalInferenceLLMProvider(
            api_url=api_url or default_url,
            api_key=api_key or "not-needed",
            model=model or "default",
        )
    elif provider == "custom":
        if not api_url:
            raise ValueError("Custom provider requires api_url to be set (e.g. in config: providers.custom.api_url)")
        return OpenAILLMProvider(api_key=api_key, model=model or "default", api_url=api_url)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
