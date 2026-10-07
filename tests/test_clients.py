"""Pruebas contra los SDK reales con transporte HTTP simulado: sin API keys reales."""
import asyncio
import json
from contextlib import aclosing

import httpx
import pytest
from pydantic import ValidationError

from clients import AnthropicClient, OpenAIClient
from manager import AsyncLLMManager, ConfigurationError
from schemas import ChatMessage, ModelConfig

MESSAGES = [ChatMessage(role="system", content="Sé breve."),
            ChatMessage(role="user", content="¿Qué es la entropía?")]


def payload(provider):
    if provider == "openai":
        return {"id": "chat-test", "object": "chat.completion", "created": 1,
                "model": "test-model", "choices": [{"index": 0, "message": {
                    "role": "assistant", "content": "Incertidumbre."}, "finish_reason": "stop"}]}
    return {"id": "msg-test", "type": "message", "role": "assistant", "model": "test-model",
            "content": [{"type": "text", "text": "Incertidumbre."}], "stop_reason": "end_turn",
            "stop_sequence": None, "usage": {"input_tokens": 5, "output_tokens": 3}}


def sse(provider):
    if provider == "openai":
        events = [{"id": "chat-test", "object": "chat.completion.chunk", "created": 1,
                   "model": "test-model", "choices": [{"index": 0, "delta": {"content": text},
                   "finish_reason": reason}]} for text, reason in [("Incert", None), ("idumbre.", None), (None, "stop")]]
        return "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    message = payload(provider) | {"content": [], "stop_reason": None}
    events = [
        {"type": "message_start", "message": message},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        *[{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}
          for text in ["Incert", "idumbre."]],
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
         "usage": {"output_tokens": 3}},
        {"type": "message_stop"},
    ]
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)


def client(provider, handler, retries=0):
    client_type = OpenAIClient if provider == "openai" else AnthropicClient
    instance = client_type(ModelConfig(model="test-model", max_retries=retries), "test-key")
    # Reutilizamos los SDK reales, cambiando únicamente su transporte de red.
    original = instance.sdk
    instance.sdk = original.with_options(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return instance, original


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_normal_and_stream(provider):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if body.get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=sse(provider))
        return httpx.Response(200, json=payload(provider))

    instance, original = client(provider, handler)
    async with original, instance:
        response = await instance.generate(MESSAGES)
        events = [event async for event in instance.stream(MESSAGES)]
    assert response.ok and response.content == "Incertidumbre."
    assert "".join(e.text for e in events) == response.content
    assert [e.type for e in events] == ["text", "text", "done"]
    if provider == "anthropic":
        assert requests[0]["system"] == "Sé breve."
        assert requests[0]["messages"] == [{"role": "user", "content": "¿Qué es la entropía?"}]
    else:
        assert requests[0]["messages"][0]["role"] == "system"
        assert requests[0]["max_completion_tokens"] == 256


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("status,code", [(401, "authentication"), (403, "authentication"),
                                        (429, "rate_limit"), (500, "provider"), (400, "provider")])
async def test_http_errors(provider, status, code):
    instance, original = client(provider, lambda _: httpx.Response(
        status, json={"error": {"message": "secret response", "type": "api_error"}}))
    async with original, instance:
        response = await instance.generate(MESSAGES)
        events = [event async for event in instance.stream(MESSAGES)]
    assert response.error.code == code
    assert [e.type for e in events] == ["error"]
    assert events[0].error.code == code
    assert "secret" not in response.model_dump_json()


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_network_and_retry(provider):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("sensitive URL", request=request)

    instance, original = client(provider, handler, retries=1)
    async with original, instance:
        response = await instance.generate(MESSAGES)
    assert calls == 2
    assert response.error.code == "network"


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_rate_limit_retry_recovers(provider):
    calls = 0

    def handler(_):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"retry-after-ms": "1"}, json={"error": {"message": "limited"}})
        return httpx.Response(200, json=payload(provider))

    instance, original = client(provider, handler, retries=1)
    async with original, instance:
        assert (await instance.generate(MESSAGES)).ok
    assert calls == 2


class InterruptedStream(httpx.AsyncByteStream):
    def __init__(self, prefix):
        self.prefix = prefix
        self.closed = False

    async def __aiter__(self):
        yield self.prefix.encode()
        raise httpx.ReadError("stream disconnected")

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_partial_stream_does_not_retry(provider):
    data = sse(provider)
    marker = "data: [DONE]" if provider == "openai" else "event: content_block_stop"
    data = data.split(marker)[0]
    stream = InterruptedStream(data)
    calls = 0

    def handler(_):
        nonlocal calls
        calls += 1
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    instance, original = client(provider, handler, retries=2)
    async with original, instance:
        events = [event async for event in instance.stream(MESSAGES)]
    assert calls == 1
    assert stream.closed
    assert any(e.type == "text" for e in events)
    assert events[-1].type == "error"


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_early_close_releases_stream(provider):
    stream = InterruptedStream(sse(provider))
    instance, original = client(provider, lambda _: httpx.Response(
        200, headers={"content-type": "text/event-stream"}, stream=stream))
    async with original, instance:
        async with aclosing(instance.stream(MESSAGES)) as events:
            async for event in events:
                assert event.type == "text"
                break
        assert stream.closed


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_invalid_messages_controlled(provider):
    def handler(_):
        pytest.fail("No debe llamar a la API con mensajes vacíos")

    instance, original = client(provider, handler)
    async with original, instance:
        assert (await instance.generate([])).error.code == "validation"
        events = [event async for event in instance.stream([])]
    assert events[0].error.code == "validation"


@pytest.mark.parametrize("kwargs", [{"temperature": -1}, {"temperature": 2.1},
    {"temperature": float("nan")}, {"max_tokens": 0}, {"max_tokens": True},
    {"timeout": 0}, {"max_retries": -1}, {"model": " "}])
def test_config_validation(kwargs):
    with pytest.raises(ValidationError):
        ModelConfig(**({"model": "test-model"} | kwargs))


def test_message_validation():
    for value in [{"role": "tool", "content": "x"}, {"role": "user", "content": " "},
                  {"role": "user", "content": {"nested": "wrong"}}]:
        with pytest.raises(ValidationError):
            ChatMessage(**value)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_manager_env(monkeypatch, provider):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv(f"{provider.upper()}_API_KEY", "test-key")
    monkeypatch.setenv(f"{provider.upper()}_MODEL", "test-model")
    async with AsyncLLMManager.from_env() as manager:
        assert manager.client.provider == provider
        assert "test-key" not in repr(manager.api_key)


def test_configuration_errors(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unknown")
    with pytest.raises(ConfigurationError):
        AsyncLLMManager.from_env()
    with pytest.raises(ConfigurationError):
        AsyncLLMManager("openai", ModelConfig(model="test-model"), "")
    with pytest.raises(ConfigurationError):
        AsyncLLMManager("anthropic", ModelConfig(model="test-model", temperature=1.5), "test-key")


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_event_loop_and_cancellation(provider):
    entered, release = asyncio.Event(), asyncio.Event()

    async def handler(_):
        entered.set()
        await release.wait()
        return httpx.Response(200, json=payload(provider))

    instance, original = client(provider, handler)
    async with original, instance:
        task = asyncio.create_task(instance.generate(MESSAGES))
        await asyncio.wait_for(entered.wait(), timeout=2)
        # El loop pudo ejecutar otra tarea mientras la petición esperaba red.
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_timeout_controlled(provider):
    def handler(request):
        raise httpx.ReadTimeout("timeout", request=request)

    instance, original = client(provider, handler)
    async with original, instance:
        assert (await instance.generate(MESSAGES)).error.code == "network"
        events = [event async for event in instance.stream(MESSAGES)]
        assert events[-1].error.code == "network"


async def test_anthropic_invalid_system_position():
    def handler(_):
        pytest.fail("No debe llamar a la API con system intercalado")

    instance, original = client("anthropic", handler)
    messages = [MESSAGES[1], MESSAGES[0]]
    async with original, instance:
        assert (await instance.generate(messages)).error.code == "validation"


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_manager_delegates(provider):
    def handler(request):
        if json.loads(request.content).get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=sse(provider))
        return httpx.Response(200, json=payload(provider))

    instance, original = client(provider, handler)
    manager = AsyncLLMManager(provider, instance.config, "test-key")
    await manager.client.aclose()
    manager.client = instance
    async with original, manager:
        assert (await manager.generate(MESSAGES)).content == "Incertidumbre."
        events = [event async for event in manager.stream(MESSAGES)]
        assert events[-1].type == "done"


def test_environment_invalid_numeric(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MAX_TOKENS", "invalid")
    with pytest.raises(ConfigurationError, match="parámetros numéricos"):
        AsyncLLMManager.from_env()
