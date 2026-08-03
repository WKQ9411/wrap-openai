"""Smoke test for an installed wrap-openai distribution.

Run this script with Python isolated mode from a clean environment so the
installed package is tested instead of the repository checkout:

    python -I tests/published_smoke.py
"""

import asyncio

import httpx
from openai import AsyncOpenAI

from wrap_openai import __version__, app, register_generate


captured = {}


def generate(messages, temperature):
    captured["messages"] = messages
    captured["temperature"] = temperature
    yield "published"
    yield " package"


register_generate(
    generate_func=generate,
    support_stream=True,
    model_id="smoke-model",
    openai_kwargs={"temperature": 0.7},
)


async def run_smoke_test():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as http_client:
        client = AsyncOpenAI(
            base_url="http://testserver/v1",
            api_key="smoke-key",
            http_client=http_client,
        )
        stream = await client.chat.completions.create(
            model="smoke-model",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.2,
            stream=True,
            stream_options={"include_usage": True},
            logprobs=False,
        )
        return [chunk async for chunk in stream]


chunks = asyncio.run(run_smoke_test())
content = "".join(
    choice.delta.content or ""
    for chunk in chunks
    for choice in chunk.choices
)
usage = next(chunk.usage for chunk in chunks if chunk.usage is not None)

assert captured["temperature"] == 0.2
assert content == "published package"
assert usage.completion_tokens == len("published package")

print(f"wrap-openai {__version__}: published package smoke test passed")
