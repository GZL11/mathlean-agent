"""
Base LLM client protocol and data models.

Defines the interface that all LLM clients must implement, along with
standard response/request models.
"""

from typing import Any, Dict, Optional, Protocol

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """
    Standardized response from LLM generation.

    This model encapsulates all data returned from an LLM call,
    providing a uniform interface regardless of the underlying provider.

    Fields:
        text: The main generated text content
        thinking: Chain-of-thought reasoning (for models like o1)
        raw_response: Original response dict from the provider
        usage: Token usage statistics
        model: Model identifier that was used
        finish_reason: Why generation stopped (length, stop, etc.)
    """

    text: str = Field(description="Generated text content")
    thinking: Optional[str] = Field(
        default=None, description="Chain-of-thought reasoning (o1 models)"
    )
    raw_response: Dict[str, Any] = Field(
        default_factory=dict, description="Raw provider response"
    )
    usage: Dict[str, int] = Field(
        default_factory=dict,
        description="Token usage (prompt_tokens, completion_tokens, total_tokens)",
    )
    model: str = Field(description="Model identifier used")
    finish_reason: Optional[str] = Field(
        default=None, description="Why generation stopped"
    )

    model_config = {"arbitrary_types_allowed": True}

    def __repr__(self) -> str:
        """Concise representation."""
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"LLMResponse(model={self.model}, text='{text_preview}')"


class LLMClient(Protocol):
    """
    Protocol for LLM client implementations.

    All LLM clients (OpenAI, Anthropic, Mock, etc.) must implement this interface.
    This ensures consistent API across different providers.

    The interface is designed to be:
    - Simple: Single generate() method for most use cases
    - Flexible: Accepts **kwargs for provider-specific options
    - Type-safe: Returns structured LLMResponse objects
    - Observable: Includes usage statistics for cost tracking

    Example Implementation:
        >>> class MyLLMClient:
        ...     def generate(
        ...         self,
        ...         prompt: str,
        ...         system: str = "",
        ...         temperature: float = 0.7,
        ...         max_tokens: int = 2000,
        ...         **kwargs
        ...     ) -> LLMResponse:
        ...         # Call provider API
        ...         response = my_provider.call(prompt, system, ...)
        ...         return LLMResponse(
        ...             text=response.content,
        ...             model=response.model,
        ...             usage={...}
        ...         )

    Example Usage:
        >>> client: LLMClient = OpenAIClient(api_key="...")
        >>> response = client.generate(
        ...     prompt="Write a sorting function",
        ...     system="You are a Python expert",
        ...     temperature=0.7
        ... )
        >>> print(response.text)
        >>> print(f"Cost: {response.usage['total_tokens']} tokens")
    """

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate text from a prompt.

        This is the primary method for interacting with LLMs.
        All implementations must support these core parameters,
        though they may accept additional provider-specific options via **kwargs.

        Args:
            prompt: The user prompt/instruction
            system: System prompt to set context/role
            temperature: Sampling temperature (0.0 = deterministic, 2.0 = very random)
                - 0.0-0.3: Focused, deterministic (good for code generation)
                - 0.4-0.7: Balanced (general use)
                - 0.8-1.0: Creative (brainstorming)
                - 1.0+: Very random (experimental)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific options
                - top_p: Nucleus sampling parameter
                - frequency_penalty: Penalize token frequency
                - presence_penalty: Penalize token presence
                - stop: Stop sequences
                - seed: Random seed (for reproducibility if supported)
                - model: Override default model
                - etc.

        Returns:
            LLMResponse with generated text and metadata

        Raises:
            Exception: Implementation-specific errors (network, quota, etc.)

        Example:
            >>> response = client.generate(
            ...     prompt="Explain bubble sort",
            ...     system="You are a CS teacher",
            ...     temperature=0.5,
            ...     max_tokens=500
            ... )
            >>> print(response.text)
        """
        ...
