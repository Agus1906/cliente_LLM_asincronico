"""Adaptadores de los SDK oficiales; no hay llamadas síncronas a modelos."""
from collections.abc import AsyncIterator, Sequence

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from schemas import ChatMessage, ModelConfig, ModelResponse, StreamEvent
from .base import BaseLLMClient, CONTROLLED_ERRORS, controlled_error


class OpenAIClient(BaseLLMClient):
    provider = "openai"

    def __init__(self, config: ModelConfig, api_key: str):
        super().__init__(config)
        self.sdk = AsyncOpenAI(api_key=api_key, timeout=config.timeout, max_retries=config.max_retries)

    def parameters(self, messages):
        return dict(model=self.config.model, temperature=self.config.temperature,
                    max_completion_tokens=self.config.max_tokens,
                    messages=[m.model_dump() for m in self.validate(messages)])

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        try:
            result = await self.sdk.chat.completions.create(**self.parameters(messages))
            choice = result.choices[0]
            return ModelResponse(provider=self.provider, model=result.model,
                                 content=choice.message.content or choice.message.refusal or "",
                                 finish_reason=choice.finish_reason)
        except CONTROLLED_ERRORS as exc:
            return ModelResponse(provider=self.provider, model=self.config.model, error=controlled_error(exc))

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamEvent]:
        try:
            stream = await self.sdk.chat.completions.create(**self.parameters(messages), stream=True)
            reason = None
            async with stream:
                async for chunk in stream:
                    if chunk.choices:
                        choice = chunk.choices[0]
                        text = choice.delta.content or choice.delta.refusal
                        if text:
                            yield StreamEvent(type="text", text=text)
                        reason = choice.finish_reason or reason
            yield StreamEvent(type="done", finish_reason=reason)
        except CONTROLLED_ERRORS as exc:
            yield StreamEvent(type="error", error=controlled_error(exc))

    async def aclose(self) -> None:
        await self.sdk.close()


class AnthropicClient(BaseLLMClient):
    provider = "anthropic"

    def __init__(self, config: ModelConfig, api_key: str):
        super().__init__(config)
        if config.temperature > 1:
            raise ValueError("Anthropic admite temperatura de 0 a 1.")
        self.sdk = AsyncAnthropic(api_key=api_key, timeout=config.timeout, max_retries=config.max_retries)

    def parameters(self, messages):
        validated = self.validate(messages)
        system, conversation = [], []
        for message in validated:
            if message.role == "system":
                if conversation:
                    raise ValueError("Los mensajes system deben preceder la conversación.")
                system.append(message.content)
            else:
                conversation.append(message.model_dump())
        if not conversation or conversation[0]["role"] != "user":
            raise ValueError("Anthropic requiere una conversación que comience con user.")
        params = dict(model=self.config.model, temperature=self.config.temperature,
                      max_tokens=self.config.max_tokens, messages=conversation)
        if system:
            params["system"] = "\n\n".join(system)
        return params

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        try:
            result = await self.sdk.messages.create(**self.parameters(messages))
            return ModelResponse(provider=self.provider, model=result.model,
                                 content="".join(b.text for b in result.content if b.type == "text"),
                                 finish_reason=result.stop_reason)
        except CONTROLLED_ERRORS as exc:
            return ModelResponse(provider=self.provider, model=self.config.model, error=controlled_error(exc))

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamEvent]:
        try:
            async with self.sdk.messages.stream(**self.parameters(messages)) as stream:
                async for text in stream.text_stream:
                    yield StreamEvent(type="text", text=text)
                result = await stream.get_final_message()
            yield StreamEvent(type="done", finish_reason=result.stop_reason)
        except CONTROLLED_ERRORS as exc:
            yield StreamEvent(type="error", error=controlled_error(exc))

    async def aclose(self) -> None:
        await self.sdk.close()
