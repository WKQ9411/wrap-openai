from wrap_openai import (
    register_funcs,
    run_server,
    set_api_key_required,
    set_allow_remote_api_key_management,
    set_api_keys_path,
)
import time

def simple_generate(prompt: str, temperature: float = 0.7) -> str:
    """
    Simple non-streaming generate function
    
    Args:
        prompt: Input text prompt
        temperature: Generation temperature (dynamic parameter, can be overridden by client)
    """
    return f"Echo: {prompt}\n(Temperature: {temperature})"


def simple_stream_generate(prompt: str, temperature: float = 0.7):
    """
    Simple streaming generate function (returns Generator)
    
    Args:
        prompt: Input text prompt
        temperature: Generation temperature (dynamic parameter)
    """
    response = f"Streaming echo: {prompt}\n(Temperature: {temperature})"
    for char in response:
        time.sleep(0.01)  # Simulate latency
        yield char


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start Wrap OpenAI API server")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming mode (default: streaming)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server bind address")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--require-api-key", action="store_true", help="Enable API Key verification")
    parser.add_argument("--disable-remote-key-manage", action="store_true", help="Disable remote API Key management (keys can only be managed on server side)")
    parser.add_argument("--api-keys-path", type=str, default=None, help="Custom path for API Keys storage (directory or file path)")
    
    args = parser.parse_args()
    
    # Configure API Keys storage path (optional)
    if args.api_keys_path:
        set_api_keys_path(args.api_keys_path)
        print(f"✅ API Keys will be stored at: {args.api_keys_path}")
    
    # Register functions
    if args.no_stream:
        # Non-streaming function (support_stream=False)
        register_funcs(
            simple_generate,
            support_stream=False,
            temperature=0.7  # Server default, can be overridden by client
        )
    else:
        # Streaming function (support_stream=True - supports both streaming and non-streaming)
        register_funcs(
            simple_stream_generate,
            support_stream=True,
            temperature=0.7  # Server default
        )
    
    print("\n" + "=" * 60)
    print("🚀 Wrap OpenAI API Server")
    print("=" * 60)
    print(f"Server starting at http://{args.host}:{args.port}")
    print(f"API endpoint: http://{args.host}:{args.port}/v1/chat/completions")
    print(f"Health check: http://{args.host}:{args.port}/health")
    
    if args.require_api_key:
        print(f"API Key management: http://{args.host}:{args.port}/api/keys")
        print("\n📝 To manage API Keys:")
        if args.disable_remote_key_manage:
            print(f"  1. Use CLI at server side: wrap-openai --generate (or --list, --revoke)")
        else:
            print(f"  1. Remote management: python demo/manage_api_keys.py generate/list/revoke --base-url http://localhost:{args.port}")
            print(f"  2. Use CLI at server side: wrap-openai --generate (or --list, --revoke)")
    
    print("=" * 60 + "\n")
    
    # Start server
    run_server(
        host=args.host,
        port=args.port,
        require_api_key=args.require_api_key,
        allow_remote_api_key_management=not args.disable_remote_key_manage
    )
