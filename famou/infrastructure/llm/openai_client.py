"""
OpenAI LLM client implementation.

Provides production-ready OpenAI API integration with:
- Automatic retry logic for transient failures
- Rate limit handling
- Timeout management
- Token usage tracking
"""

from typing import Any, Dict, Optional

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from famou.infrastructure.llm.base import LLMResponse


class OpenAIClient:
    """
    OpenAI API client with robust error handling.

    Features:
    - Automatic retry with exponential backoff
    - Rate limit handling
    - Configurable timeouts
    - Support for all OpenAI models (GPT-4, GPT-4o, o1, etc.)
    - Token usage tracking

    Example:
        >>> client = OpenAIClient(
        ...     api_key="sk-...",
        ...     model="gpt-4o",
        ...     temperature=0.7,
        ...     timeout=60
        ... )
        >>> response = client.generate(
        ...     prompt="Write a Python function to sort a list",
        ...     system="You are an expert Python programmer"
        ... )
        >>> print(response.text)
        >>> print(f"Tokens used: {response.usage['total_tokens']}")
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: int = 60,
        max_retries: int = 3,
        **default_kwargs,
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model identifier (gpt-4, gpt-4o, gpt-4o-mini, o1, etc.)
            api_base: Optional API base URL (for proxies or compatible APIs)
            temperature: Default sampling temperature
            max_tokens: Default maximum tokens to generate
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            **default_kwargs: Additional default parameters for API calls
        """
        # Initialize OpenAI client
        client_kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if api_base:
            client_kwargs["base_url"] = api_base

        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_kwargs = default_kwargs

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate text using OpenAI API.

        Automatically retries on:
        - Rate limit errors (429)
        - Server errors (500, 502, 503, 504)
        - Timeout errors

        Args:
            prompt: User prompt/instruction
            system: System prompt to set context
            temperature: Sampling temperature (overrides default)
            max_tokens: Maximum tokens (overrides default)
            **kwargs: Additional OpenAI API parameters
                - top_p: Nucleus sampling
                - frequency_penalty: Penalize frequent tokens
                - presence_penalty: Penalize present tokens
                - stop: Stop sequences
                - seed: Random seed (for reproducibility)
                - etc.

        Returns:
            LLMResponse with generated text and metadata

        Raises:
            OpenAI API errors after retries exhausted
        """
        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Merge parameters (kwargs > method args > instance defaults)
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            **self.default_kwargs,
            **kwargs,
        }

        # Call API with retry logic
        response = self._call_with_retry(**params)

        # Extract response data
        choice = response.choices[0]
        message = choice.message

        # Handle thinking/reasoning content from different models
        # - OpenAI o1: message.reasoning
        # - DeepSeek: message.reasoning_content or similar
        # - Other providers may use different field names
        thinking = None
        for field in ["reasoning", "reasoning_content", "thinking", "deepseek_reasoning"]:
            if hasattr(message, field):
                thinking = getattr(message, field)
                if thinking:
                    break

        # Also check in raw response dict for nested thinking fields
        if not thinking and isinstance(message, dict):
            for key in ["reasoning", "reasoning_content", "thinking", "deepseek_reasoning"]:
                if key in message:
                    thinking = message[key]
                    if thinking:
                        break

        return LLMResponse(
            text=message.content or "",
            thinking=thinking,
            raw_response=response.model_dump(),
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            model=response.model,
            finish_reason=choice.finish_reason,
        )

    @retry(
        retry=retry_if_exception_type(
            (
                Exception,  # Catch-all for OpenAI errors
            )
        ),
        stop=stop_after_attempt(3),  # Default 3 attempts
        wait=wait_exponential(multiplier=1, min=2, max=10),  # 2s, 4s, 8s
        reraise=True,
    )
    def _call_with_retry(self, **params):
        """
        Call OpenAI API with automatic retry.

        Uses exponential backoff: 2s → 4s → 8s between retries.

        Args:
            **params: Parameters to pass to chat.completions.create()

        Returns:
            OpenAI ChatCompletion response

        Raises:
            OpenAI API errors after max retries
        """
        return self.client.chat.completions.create(**params)

    def __repr__(self) -> str:
        """Concise representation."""
        return (
            f"OpenAIClient(model={self.model}, "
            f"temperature={self.temperature}, "
            f"max_tokens={self.max_tokens})"
        )


class AsyncOpenAIClient:
    """
    Async OpenAI client for parallel generation.

    Similar to OpenAIClient but uses async/await for concurrent requests.
    Useful when generating multiple programs in parallel.

    Example:
        >>> import asyncio
        >>> client = AsyncOpenAIClient(api_key="sk-...")
        >>>
        >>> async def generate_many():
        ...     prompts = ["Write sort", "Write search", "Write reverse"]
        ...     tasks = [client.generate(p, system="Code expert") for p in prompts]
        ...     responses = await asyncio.gather(*tasks)
        ...     return responses
        >>>
        >>> responses = asyncio.run(generate_many())
        >>> len(responses)
        3
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: int = 60,
        max_retries: int = 3,
        **default_kwargs,
    ):
        """Initialize async OpenAI client."""
        from openai import AsyncOpenAI

        client_kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if api_base:
            client_kwargs["base_url"] = api_base

        self.client = AsyncOpenAI(**client_kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_kwargs = default_kwargs

    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate text asynchronously.

        Args:
            prompt: User prompt
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated text
        """
        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Merge parameters
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            **self.default_kwargs,
            **kwargs,
        }

        # Call API (async)
        response = await self._call_with_retry(**params)

        # Extract response
        choice = response.choices[0]
        message = choice.message

        # Handle thinking/reasoning content from different models
        # - OpenAI o1: message.reasoning
        # - DeepSeek: message.reasoning_content or similar
        # - Other providers may use different field names
        thinking = None
        for field in ["reasoning", "reasoning_content", "thinking", "deepseek_reasoning"]:
            if hasattr(message, field):
                thinking = getattr(message, field)
                if thinking:
                    break

        # Also check in raw response dict for nested thinking fields
        if not thinking and isinstance(message, dict):
            for key in ["reasoning", "reasoning_content", "thinking", "deepseek_reasoning"]:
                if key in message:
                    thinking = message[key]
                    if thinking:
                        break

        return LLMResponse(
            text=message.content or "",
            thinking=thinking,
            raw_response=response.model_dump(),
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            model=response.model,
            finish_reason=choice.finish_reason,
        )

    @retry(
        retry=retry_if_exception_type((Exception,)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _call_with_retry(self, **params):
        """Call OpenAI API with retry (async)."""
        return await self.client.chat.completions.create(**params)

    def __repr__(self) -> str:
        """Concise representation."""
        return (
            f"AsyncOpenAIClient(model={self.model}, "
            f"temperature={self.temperature}, "
            f"max_tokens={self.max_tokens})"
        )
