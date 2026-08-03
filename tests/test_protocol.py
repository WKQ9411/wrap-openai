import asyncio
import json

import httpx
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from wrap_openai import app, register_generate


client = TestClient(app)


def _stream_data(response):
    return [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def test_standard_fields_are_accepted_and_only_registered_fields_are_forwarded():
    engine = object()
    captured = {}

    def generate(messages, engine, temperature, top_k):
        captured.update(
            messages=messages,
            engine=engine,
            temperature=temperature,
            top_k=top_k,
        )
        return "answer"

    register_generate(
        generate_func=generate,
        support_stream=False,
        model_id="research-model",
        fixed_kwargs={"engine": engine},
        openai_kwargs={"temperature": 0.7},
        custom_kwargs={"top_k": 50},
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "research-model",
            "messages": [
                {"role": "developer", "content": "Be brief", "name": "policy"},
                {"role": "user", "content": "hello"},
            ],
            "temperature": 0.2,
            "top_k": 20,
            "stream_options": {"include_usage": True},
            "logprobs": False,
            "moderation": False,
            "prompt_cache_options": {"ttl": "24h"},
            "tools": [{"type": "function", "function": {"name": "unused"}}],
            "response_format": {"type": "text"},
        },
    )

    assert response.status_code == 200
    assert captured["engine"] is engine
    assert captured["temperature"] == 0.2
    assert captured["top_k"] == 20
    assert captured["messages"][0] == {
        "role": "developer",
        "content": "Be brief",
        "name": "policy",
    }

    body = response.json()
    assert body["choices"][0]["message"]["content"] == "answer"
    assert body["usage"] == {
        "prompt_tokens": 13,
        "completion_tokens": 6,
        "total_tokens": 19,
        "prompt_tokens_details": None,
        "completion_tokens_details": None,
    }


def test_extended_openai_field_can_be_registered_and_forwarded():
    captured = {}

    def generate(messages, logprobs):
        captured["logprobs"] = logprobs
        return "ok"

    register_generate(
        generate_func=generate,
        support_stream=False,
        model_id="research-model",
        openai_kwargs={"logprobs": False},
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "research-model",
            "messages": [{"role": "user", "content": "hello"}],
            "logprobs": True,
        },
    )

    assert response.status_code == 200
    assert captured["logprobs"] is True


def test_stream_options_include_usage_returns_standard_usage_chunk():
    def generate(messages):
        yield "hel"
        yield "lo"

    register_generate(
        generate_func=generate,
        support_stream=True,
        model_id="research-model",
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "research-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "logprobs": False,
        },
    )

    assert response.status_code == 200
    data = _stream_data(response)
    assert data[-1] == "[DONE]"
    chunks = [json.loads(item) for item in data[:-1]]

    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert "".join(
        chunk["choices"][0]["delta"].get("content", "")
        for chunk in chunks
        if chunk["choices"]
    ) == "hello"
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert all(chunk["usage"] is None for chunk in chunks[:-1])
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["prompt_tokens"] == 2
    assert chunks[-1]["usage"]["completion_tokens"] == 5
    assert chunks[-1]["usage"]["total_tokens"] == 7


def test_non_streaming_generate_can_return_one_standard_stream_chunk():
    def generate(messages):
        return "complete"

    register_generate(
        generate_func=generate,
        support_stream=False,
        model_id="research-model",
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "research-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    chunks = [json.loads(item) for item in _stream_data(response)[:-1]]
    content = "".join(
        chunk["choices"][0]["delta"].get("content", "")
        for chunk in chunks
        if chunk["choices"]
    )
    assert content == "complete"
    assert "Warning" not in response.text
    assert all("usage" not in chunk for chunk in chunks)


def test_openai_sdk_can_parse_stream_and_usage_chunk():
    def generate(messages):
        yield "sdk"
        yield " response"

    register_generate(
        generate_func=generate,
        support_stream=True,
        model_id="research-model",
    )

    async def consume_stream():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as http_client:
            sdk = AsyncOpenAI(
                base_url="http://testserver/v1",
                api_key="test-key",
                http_client=http_client,
            )
            stream = await sdk.chat.completions.create(
                model="research-model",
                messages=[{"role": "user", "content": "hello"}],
                stream=True,
                stream_options={"include_usage": True},
                logprobs=False,
            )
            return [chunk async for chunk in stream]

    chunks = asyncio.run(consume_stream())
    content = "".join(
        choice.delta.content or ""
        for chunk in chunks
        for choice in chunk.choices
    )
    usage = next(chunk.usage for chunk in chunks if chunk.usage is not None)

    assert content == "sdk response"
    assert usage.prompt_tokens == 5
    assert usage.completion_tokens == 12
    assert usage.total_tokens == 17


def test_stream_request_validation_finishes_before_response_starts():
    called = False

    def generate(messages, engine):
        nonlocal called
        called = True
        return "unused"

    register_generate(
        generate_func=generate,
        support_stream=False,
        model_id="research-model",
        fixed_kwargs={"engine": object()},
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "research-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "engine": "client override",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"].startswith("Fixed parameters cannot be overridden")
    assert called is False


def test_unknown_nonstandard_field_is_rejected():
    def generate(messages):
        return "unused"

    register_generate(
        generate_func=generate,
        support_stream=False,
        model_id="research-model",
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "research-model",
            "messages": [{"role": "user", "content": "hello"}],
            "unknown_experiment_option": True,
        },
    )

    assert response.status_code == 422
    assert "Unknown custom parameters" in response.json()["detail"]
