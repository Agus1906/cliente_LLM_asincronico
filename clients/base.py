"""Interfaz común y traducción segura de errores del SDK."""
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence

import anthropic
import httpx
import openai
from pydantic import ValidationError

from schemas import ChatMessage, GenerationRequest, LLMError, ModelConfig, ModelResponse, StreamEvent

logger = logging.getLogger(__name__)
SDK_ERRORS = (openai.APIError, anthropic.APIError)
CONTROLLED_ERRORS = (*SDK_ERRORS, httpx.TransportError, ValidationError, ValueError)


def controlled_error(exc: Exception) -> LLMError:
    """Nunca incluir cuerpos HTTP, prompts ni credenciales en errores/logs."""
    status = getattr(exc, "status_code", None)
    if isinstance(exc, (ValidationError, ValueError)):
        code, message, retryable = "validation", "Entrada o configuración inválida.", False
    elif status in (401, 403):
        code, message, retryable = "authentication", "Revisá la API key y los permisos.", False
    elif status == 429:
        code, message, retryable = "rate_limit", "Límite de tasa o cuota alcanzado; revisá tu cuenta.", True
    elif isinstance(exc, (openai.APIConnectionError, anthropic.APIConnectionError, httpx.TransportError)):
        code, message, retryable = "network", "No se pudo conectar o se agotó el tiempo de espera.", True
    else:
        code, message, retryable = "provider", "El proveedor rechazó o no pudo completar la solicitud.", status is not None and status >= 500
    logger.warning("llm_error code=%s status=%s", code, status)
    return LLMError(code=code, message=message, retryable=retryable)


class BaseLLMClient(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config

    def validate(self, messages: Sequence[ChatMessage]) -> tuple[ChatMessage, ...]:
        return GenerationRequest(messages=tuple(messages)).messages

    @abstractmethod
    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        """Devuelve una respuesta completa o un error controlado."""

    @abstractmethod
    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamEvent]:
        """Devuelve fragmentos y un evento terminal done/error."""

    @abstractmethod
    async def aclose(self) -> None:
        """Libera conexiones HTTP."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.aclose()
