"""Selección por entorno y fachada independiente del proveedor."""
import os
from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing

from pydantic import SecretStr, ValidationError

from clients import AnthropicClient, BaseLLMClient, OpenAIClient
from schemas import ChatMessage, ModelConfig, ModelResponse, Provider, StreamEvent


class ConfigurationError(ValueError):
    pass


class AsyncLLMManager:
    def __init__(self, provider: Provider, config: ModelConfig, api_key: str):
        if provider not in ("openai", "anthropic"):
            raise ConfigurationError("LLM_PROVIDER debe ser openai o anthropic.")
        if not api_key.strip():
            raise ConfigurationError(f"Falta {provider.upper()}_API_KEY.")
        self.api_key = SecretStr(api_key)
        client_type = OpenAIClient if provider == "openai" else AnthropicClient
        try:
            self.client: BaseLLMClient = client_type(config, self.api_key.get_secret_value())
        except ValueError as exc:
            raise ConfigurationError("Configuración incompatible con el proveedor (Anthropic: temperatura 0 a 1).") from exc

    @classmethod
    def from_env(cls, provider: Provider | None = None):
        selected = provider or os.getenv("LLM_PROVIDER", "openai")
        if selected not in ("openai", "anthropic"):
            raise ConfigurationError("LLM_PROVIDER debe ser openai o anthropic.")
        try:
            config = ModelConfig(
                model=os.getenv(f"{selected.upper()}_MODEL", ""),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "256")),
                timeout=float(os.getenv("LLM_TIMEOUT", "30")),
                max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
            )
        except (ValueError, ValidationError) as exc:
            raise ConfigurationError("Revisá el modelo y los parámetros numéricos del .env.") from exc
        return cls(selected, config, os.getenv(f"{selected.upper()}_API_KEY", ""))

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        return await self.client.generate(messages)

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamEvent]:
        async with aclosing(self.client.stream(messages)) as events:
            async for event in events:
                yield event

    async def aclose(self) -> None:
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.aclose()
