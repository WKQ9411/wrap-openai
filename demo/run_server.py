import time
from typing import Dict, List

from wrap_openai import (
    register_generate,
    run_server,
    set_api_keys_path,
)


def messages_generate(
    messages: List[Dict],
    response_prefix: str,
    temperature: float = 0.7,
    top_k: int = 50,
) -> str:
    """Non-streaming generate function. The first argument receives messages."""
    return (
        f"{response_prefix}: {messages}\n"
        f"(Temperature: {temperature}, Top-K: {top_k})"
    )


def messages_stream_generate(
    messages: List[Dict],
    response_prefix: str,
    temperature: float = 0.7,
    top_k: int = 50,
):
    """Streaming generate function yielding string chunks."""
    response = messages_generate(
        messages,
        response_prefix=response_prefix,
        temperature=temperature,
        top_k=top_k,
    )
    for char in response:
        time.sleep(0.01)
        yield char


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start Wrap OpenAI API server")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming generation")
    parser.add_argument("--model-id", default="echo-model", help="Model ID exposed by the API")
    parser.add_argument("--host", default="127.0.0.1", help="Server bind address")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--require-api-key", action="store_true", help="Enable API Key verification")
    parser.add_argument("--disable-remote-key-manage", action="store_true", help="Disable remote API Key management")
    parser.add_argument("--api-keys-path", default=None, help="Custom API Key storage path")
    args = parser.parse_args()

    if args.api_keys_path:
        set_api_keys_path(args.api_keys_path)

    generate_func = messages_generate if args.no_stream else messages_stream_generate
    register_generate(
        generate_func=generate_func,
        support_stream=not args.no_stream,
        model_id=args.model_id,
        fixed_kwargs={
            "response_prefix": "Echo to Messages",
        },
        openai_kwargs={
            "temperature": 0.7,
        },
        custom_kwargs={
            "top_k": 50,
        },
    )

    print(f"Server starting at http://{args.host}:{args.port}")
    print(f"Model ID: {args.model_id}")
    print(f"API endpoint: http://{args.host}:{args.port}/v1/chat/completions")
    print("Custom request example: extra_body={'top_k': 20}")

    run_server(
        host=args.host,
        port=args.port,
        require_api_key=args.require_api_key,
        allow_remote_api_key_management=not args.disable_remote_key_manage,
    )
