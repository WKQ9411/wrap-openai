# Wrap OpenAI

Wrap custom generate function as an OpenAI SDK compatible API service.

> **Experimental Package**: This package is intended for experimental purposes and may not be suitable for production deployment.

## 1. Features

- **OpenAI Compatible API**: Compatible with OpenAI Chat Completions API format
- **Streaming & Non-streaming**: Support both generation modes with smart mode selection
- **Async Support**: Support OpenAI SDK's async client
- **Dynamic Parameters**: Support dynamic passing of common parameters (temperature, max_tokens, etc.)
- **API Key Management**: Built-in API Key authentication and management
- **CORS Support**: Built-in CORS middleware for cross-origin requests

## 2. Installation

```bash
# pip
pip install wrap-openai
# uv
uv pip install wrap-openai
```

Or install from source:

```bash
git clone https://github.com/WKQ9411/wrap-openai.git
cd wrap-openai
# pip
pip install -e .
# uv
uv sync
# if you want to use the demo
uv sync --extra qwen
```

## 3. Quick Start

### (1) Deploy Service

#### Custom Generate Function

The first parameter of the generate function should be `prompt: str` or `messages: List[Dict]`. The return type should be `str` if the function does not support streaming, or `Generator[str, None, None]` if the function supports streaming.

You can also pass additional parameters to the function, which will be used as server defaults and can be overridden by OpenAI client requests.

If your generate function supports multimodal content, you can use `messages: List[Dict]` as the first parameter. The content of the message can be a string or a list with type field, for example:

```python
# Simple text messages
messages = [
    {
        "role": "user",
        "content": "Hello"
    }
]

# Multimodal messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Hello"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
        ]
    }
]
```

You should parse the messages in your own generate function.

#### Basic Usage

You can simply use the `register_funcs` function to register your functions and use the `run_server` function to start the server.

```python
from wrap_openai import register_funcs, run_server

# step 1: define your generate function
# Non-streaming function
# the first parameter is prompt: str or messages: List[Dict], the return type is str
def my_generate(prompt: str) -> str:
    return f"Response to '{prompt}'"

# Streaming function
# the first parameter is prompt: str or messages: List[Dict], the return type is Generator[str, None, None]
def my_stream_generate(prompt: str):
    response = f"Streaming response to '{prompt}'"
    for char in response:
        yield char

# step 2: register your functions
# Register functions
register_funcs(my_generate, support_stream=False)
register_funcs(my_stream_generate, support_stream=True)

# step 3: start the server
run_server(host="0.0.0.0", port=8000)
```

#### With Additional Parameters

You can pass additional parameters to the generate function, which will be used as server defaults and can be overridden by OpenAI client requests.

These parameters can be passed as server defaults and overridden by client requests:

- `temperature` - Controls randomness
- `max_tokens` - Maximum tokens to generate
- `max_new_tokens` - Mapped to `max_tokens` (HuggingFace compatibility)
- `top_p` - Nucleus sampling
- `top_k` - Extended support
- `presence_penalty` - Encourages new topics
- `frequency_penalty` - Reduces repetition
- `n` - Number of completion choices
- `stop` - Stop sequences
- `seed` - Random seed for reproducibility

```python
def my_generate(prompt: str, model, tokenizer, temperature=0.7, max_tokens=512) -> str:
    # Your generation logic
    input_ids = tokenizer.encode(prompt)
    output_ids = model.generate(input_ids, temperature, max_tokens)
    output = tokenizer.decode(output_ids)
    return output

# Register with parameters
register_funcs(
    my_generate,
    support_stream=False,
    model=model,              # Fixed parameter (bound permanently)
    tokenizer=tokenizer,      # Fixed parameter (bound permanently)
    temperature=0.8,          # Dynamic parameter (server default, can be overridden by client)
    max_new_tokens=1024       # Dynamic parameter (server default, can be overridden by client)
)
```

#### Server Configuration

You can require API Key verification and remote API Key management via HTTP API. CORS is enabled by default to allow cross-origin requests.

```python
from wrap_openai import run_server, set_api_keys_path, set_cors

# Configure API Keys storage path (optional, default: .api_keys/keys.json)
set_api_keys_path("/custom/path/to/keys")  # Directory or file path

# Start server with options
run_server(
    host="0.0.0.0",                       # Server bind address (default: "0.0.0.0")
    port=8000,                            # Server port (default: 8000)
    require_api_key=False,                # Enable API Key verification (default: False)
    allow_remote_api_key_management=True, # Allow remote API Key management via HTTP API (default: True)
    enable_cors=True,                    # Enable CORS (default: True)
    cors_origins="*",                    # Allowed origins: "*" for all, or list of specific origins (default: "*")
    cors_allow_credentials=False,        # Allow credentials in CORS requests (default: False)
    cors_allow_methods="*",              # Allowed HTTP methods (default: "*")
    cors_allow_headers="*"               # Allowed headers (default: "*")
)
```

**CORS Configuration Options:**

You can also configure CORS separately using the `set_cors` function:

```python
from wrap_openai import set_cors, run_server

# Configure CORS with specific origins
set_cors(
    enabled=True,
    origins=["http://localhost:3000", "https://example.com"],  # Specific origins
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"]
)

# Start server
run_server(host="0.0.0.0", port=8000)
```

**Note**: By default, CORS is enabled with `origins="*"` to allow all origins, which is convenient for development and testing.

### (2) API Key Management

#### Using CLI Tool

CLI tool is provided for API Key management. You can use the `wrap-openai` command to generate, list and revoke API Keys. This command can be used on the server side to manage API Keys.

```bash
# Generate API Key
wrap-openai --generate [--name "my-key"] [--api-keys-path /path/to/keys]

# List all API Keys
wrap-openai --list [--api-keys-path /path/to/keys]

# Revoke API Key
wrap-openai --revoke <api_key> [--api-keys-path /path/to/keys]
```

#### Using HTTP API

You can use the HTTP API to manage API Keys on the client side. The API endpoints are:

```bash
# Generate API Key
python demo/manage_api_keys.py generate [--name "my-key"] --base-url http://localhost:8000

# List all API Keys
python demo/manage_api_keys.py list --base-url http://localhost:8000

# Revoke API Key
python demo/manage_api_keys.py revoke <api_key> --base-url http://localhost:8000
```

If the `allow_remote_api_key_management` is ser to `False`, the API Key management can only be done on the server side.

#### Using Python Code

```python
from wrap_openai import get_api_key_manager

api_key_manager = get_api_key_manager()
api_key = api_key_manager.generate_key(name="my-key")
print(f"Generated API Key: {api_key}")
```

### (3) Use OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-dummy"  # Required if API Key verification is enabled
)

# Non-streaming
response = client.chat.completions.create(
    model="custom-model",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,      # Override server default
    max_tokens=512        # Override server default
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="custom-model",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## 4. Examples

See the `demo/` directory for complete examples:

- **`demo/run_server.py`** - Simple echo server with all features
- **`demo/server_demo.py`** - Qwen2.5-0.5B model deployment example
- **`demo/chat_demo.py`** - Simple CLI chat application
- **`demo/manage_api_keys.py`** - API Key management via HTTP API
- **`demo/run_client.py`** - Client usage examples

## 5. Parameter Handling

### (1) Fixed vs Dynamic Parameters

When registering functions with `**kwargs`:

- **Fixed Parameters** (e.g., `model`, `tokenizer`): Bound permanently using `functools.partial`, cannot be overridden by client
- **Dynamic Parameters** (e.g., `temperature`, `max_tokens`): Server defaults, can be overridden by client requests

### (2) Stream Parameter

The `stream` parameter has special handling:

- If `support_stream=True`: Function supports both modes, client's `stream` parameter is respected
- If `support_stream=False`: Only non-streaming mode, client's `stream` parameter is ignored

## 6. License

MIT License
