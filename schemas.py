"""Contratos validados e inmutables compartidos por ambos proveedores."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Provider = Literal["openai", "anthropic"]


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChatMessage(Schema):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("El contenido no puede estar vacío.")
        return value


class ModelConfig(Schema):
    model: str = Field(min_length=1, pattern=r"\S")
    temperature: float = Field(default=0.7, ge=0, le=2, allow_inf_nan=False)
    max_tokens: int = Field(default=256, gt=0, strict=True)
    timeout: float = Field(default=30, gt=0, le=300, allow_inf_nan=False)
    max_retries: int = Field(default=2, ge=0, le=5, strict=True)


class GenerationRequest(Schema):
    messages: tuple[ChatMessage, ...] = Field(min_length=1)


class LLMError(Schema):
    code: str
    message: str
    retryable: bool = False


class ModelResponse(Schema):
    provider: Provider
    model: str
    content: str = ""
    finish_reason: str | None = None
    error: LLMError | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class StreamEvent(Schema):
    type: Literal["text", "done", "error"]
    text: str = ""
    error: LLMError | None = None
    finish_reason: str | None = None
