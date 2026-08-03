# Wrap OpenAI

Wrap a research-oriented custom generate function as an OpenAI Chat Completions compatible API service.

> **Experimental Package**: This package is designed for model research and prototype validation, not production inference serving.

## 1. Features

- OpenAI Chat Completions request and response format
- Streaming and non-streaming generation
- Raw `messages` input, including structured multimodal content
- Standard Chat Completions fields are accepted even when the generate function does not use them
- Explicit fixed, OpenAI, and custom parameter groups
- Custom client parameters through OpenAI SDK `extra_body`
- API Key management, CORS, and health check endpoints

## 2. Installation

```bash
pip install wrap-openai
```

Install from source:

```bash
git clone https://github.com/WKQ9411/wrap-openai.git
cd wrap-openai
uv sync
```

Install the Qwen demo dependencies:

```bash
uv sync --extra qwen
```

## 3. Generate Function Contract

The registered generate function must accept the OpenAI `messages` list as its first positional argument. The parameter name is not enforced, although `messages` is recommended.

```python
def generate(messages, model, tokenizer, temperature=0.7):
    ...
```

The messages structure is preserved:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        ],
    },
]
```

The generate function is responsible for applying the model-specific chat template:

```python
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

Return a string when `support_stream=False`, or yield string chunks when `support_stream=True`.

## 4. Register a Generate Function

```python
from wrap_openai import register_generate, run_server


def generate(
    messages,
    model,
    tokenizer,
    temperature=0.7,
    max_tokens=512,
    top_p=0.9,
    top_k=50,
    draft_steps=4,
):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        temperature=temperature,
        max_new_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
        draft_steps=draft_steps,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


register_generate(
    generate_func=generate,
    support_stream=False,
    model_id="research-model-v1",
    fixed_kwargs={
        "model": model,
        "tokenizer": tokenizer,
    },
    openai_kwargs={
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 0.9,
    },
    custom_kwargs={
        "top_k": 50,
        "draft_steps": 4,
    },
)

run_server(host="0.0.0.0", port=8000)
```

`register_generate` accepts three flat keyword dictionaries:

- `fixed_kwargs`: server-only objects and values. Clients cannot override them.
- `openai_kwargs`: OpenAI standard parameters that are forwarded to the generate function, with server defaults. Standard request fields override them.
- `custom_kwargs`: custom parameters and server defaults. `extra_body` fields override them.

All three dictionaries are flattened when calling the function:

```python
generate_func(
    messages,
    **fixed_kwargs,
    **effective_openai_kwargs,
    **effective_custom_kwargs,
)
```

`wrap-openai` accepts the standard Chat Completions request fields declared by
the current protocol adapter. Wrapper-owned fields such as `model`, `messages`,
`stream`, and `stream_options` are handled internally. Any other standard field
can be registered through `openai_kwargs` when the generate function supports
it. Standard fields that are not registered are accepted but ignored.

Non-OpenAI parameters such as `top_k` and experimental decoding controls belong in `custom_kwargs`.

Registration validates that:

- the function accepts messages as its first positional argument;
- registered keyword names are accepted by the function or `**kwargs`;
- the three keyword groups do not overlap;
- OpenAI parameters are placed in `openai_kwargs`;
- reserved fields are not exposed as custom parameters.

## 5. OpenAI SDK Client

```python
from openai import OpenAI


client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-dummy",
)

response = client.chat.completions.create(
    model="research-model-v1",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.2,
    max_tokens=256,
    extra_body={
        "top_k": 20,
        "draft_steps": 8,
    },
)

print(response.choices[0].message.content)
```

`extra_body` values are merged into the JSON request body by the OpenAI SDK. Only non-standard fields declared in `custom_kwargs` are accepted. Unknown custom fields and attempts to override fixed values return HTTP 422. Unregistered standard OpenAI fields do not produce an error and are not forwarded to the generate function.

The request `model` must match the `model_id` passed to `register_generate`.

## 6. Streaming

A streaming generate function yields string chunks:

```python
def stream_generate(messages, model, tokenizer, temperature=0.7):
    for text_chunk in custom_model_stream(messages, model, tokenizer, temperature):
        yield text_chunk


register_generate(
    generate_func=stream_generate,
    support_stream=True,
    model_id="research-model-v1",
    fixed_kwargs={
        "model": model,
        "tokenizer": tokenizer,
    },
    openai_kwargs={
        "temperature": 0.7,
    },
)
```

When the client requests `stream=False`, wrap-openai collects the chunks into one response. When a non-streaming function receives a `stream=True` request, the complete result is returned in one standard SSE content chunk.

The protocol adapter handles `stream_options` itself. When the client sends
`stream_options={"include_usage": true}`, the final event before `[DONE]` is a
standard usage chunk with an empty `choices` list. Token counts are currently
estimated from character counts; they are not tokenizer-accurate measurements.

Client example:

```python
stream = client.chat.completions.create(
    model="research-model-v1",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
```

## 7. Server Configuration

```python
run_server(
    host="0.0.0.0",
    port=8000,
    require_api_key=False,
    allow_remote_api_key_management=False,
    enable_cors=True,
    cors_origins="*",
)
```

Health check:

```text
GET /health
```

## 8. API Key Management

```bash
wrap-openai --generate --name "my-key"
wrap-openai --list
wrap-openai --revoke <api_key>
```

Configure the storage path from Python:

```python
from wrap_openai import set_api_keys_path

set_api_keys_path("/custom/path/to/keys")
```

## 9. Protocol Tests

Run the repeatable protocol test suite before publishing:

```bash
uv run --extra dev pytest
```

After installing a published build into a clean environment, run the isolated
smoke test from the repository checkout:

```bash
python -I tests/published_smoke.py
```

Python isolated mode prevents the repository root from shadowing the installed
distribution.

## 10. Examples

- `demo/run_server.py`: lightweight messages, streaming, and custom parameter example
- `demo/server_demo.py`: Qwen model deployment example
- `demo/run_client.py`: OpenAI SDK client examples
- `demo/chat_demo.py`: CLI chat application
- `demo/manage_api_keys.py`: API Key management over HTTP

## 11. License

MIT License
