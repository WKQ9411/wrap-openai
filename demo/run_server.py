from wrap_openai import (
    register_funcs,
    run_server,
    set_api_key_required,
    set_allow_remote_api_key_management,
    set_api_keys_path,
)
import time
from typing import List, Dict, Union, Generator

# ------------------------------------- Helper Functions -------------------------------------
def _normalize_messages(messages: List[Dict]) -> List[Dict]:
    """
    Extract text content from message content
    """
    for message in messages:
        if message["role"] == "user":
            content = message["content"]
            if isinstance(content, list):
                parts = []
                for item in content:
                    if hasattr(item, "type") and item.type == "text":  # for TextContent
                        parts.append(item.text)
                    elif isinstance(item, dict) and item.get("type") == "text":  # for dict type
                        parts.append(item.get("text", ""))
                message["content"] = "\n".join(parts)
            elif isinstance(content, str):
                message["content"] = content
            else:
                message["content"] = str(content)
    return messages


# ------------------------------------- Prompt + Non-streaming -------------------------------------
def simple_generate(prompt: str, temperature: float = 0.7) -> str:
    """
    Simple non-streaming generate function
    
    Args:
        prompt: Input text prompt
        temperature: Generation temperature (dynamic parameter, can be overridden by client)
    """
    return f"Echo to Prompt: {prompt}\n(Temperature: {temperature})"


# ------------------------------------- Prompt + Streaming -------------------------------------
def simple_stream_generate(prompt: str, temperature: float = 0.7):
    """
    Simple streaming generate function (returns Generator)
    
    Args:
        prompt: Input text prompt
        temperature: Generation temperature (dynamic parameter)
    """
    response = f"Streaming echo to Prompt: {prompt}\n(Temperature: {temperature})"
    for char in response:
        time.sleep(0.01)  # Simulate latency
        yield char


# ------------------------------------- Messages + Non-streaming -------------------------------------
def messages_generate(messages: List[Dict], temperature: float = 0.7) -> str:
    """
    Non-streaming generate function that accepts messages format
    
    Args:
        messages: List of message dicts with role and content
                  Example: [{"role": "user", "content": "Hello"}]
        temperature: Generation temperature (dynamic parameter, can be overridden by client)
    
    Returns:
        Generated response string
    """
    formatted_messages = _normalize_messages(messages)
    return f"Echo to Messages: {formatted_messages}\n(Temperature: {temperature})"


# ------------------------------------- Messages + Streaming -------------------------------------
def messages_stream_generate(messages: List[Dict], temperature: float = 0.7):
    """
    Streaming generate function that accepts messages format (returns Generator)
    
    Args:
        messages: List of message dicts with role and content
                  Example: [{"role": "user", "content": "Hello"}]
        temperature: Generation temperature (dynamic parameter)
    
    Yields:
        Character chunks of the response
    """
    formatted_messages = _normalize_messages(messages)
    response = f"Streaming echo to Messages: {formatted_messages}\n(Temperature: {temperature})"
    
    for char in response:
        time.sleep(0.01)  # Simulate latency
        yield char


# ------------------------------------- Messages + Unified Generate -------------------------------------
def unified_generate(
    messages: List[Dict], 
    stream: bool = False, 
    temperature: float = 0.7
    ) -> Union[str, Generator[str, None, None]]:
    """
    Unified generate function that supports both streaming and non-streaming modes.
    
    Args:
        messages: List of message dicts
        stream: Whether to return a generator for streaming mode
        temperature: Generation temperature (dynamic parameter)
    
    Returns:
        Generated response string or generator for streaming mode
    """
    # The first parameter of the generate function is recommended to use messages: List[Dict] format,
    # so that you can process the content of the message in your own generate function.
    messages = _normalize_messages(messages)

    # And the generate function is recommended to return a generator,
    # because it supports both streaming and non-streaming requests.
    if stream:
        return messages_stream_generate(messages, temperature)
    else:
        return messages_generate(messages, temperature)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start Wrap OpenAI API server")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming mode (default: streaming)")
    parser.add_argument("--use-messages", action="store_true", help="Use messages format instead of prompt format (for testing messages support)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server bind address")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--require-api-key", action="store_true", help="Enable API Key verification")
    parser.add_argument("--disable-remote-key-manage", action="store_true", help="Disable remote API Key management (keys can only be managed on server side)")
    parser.add_argument("--api-keys-path", type=str, default=None, help="Custom path for API Keys storage (directory or file path)")
    parser.add_argument("--use-unified-generate", action="store_true", help="Use unified generate function instead of separate functions")
    
    args = parser.parse_args()
    
    # Configure API Keys storage path (optional)
    if args.api_keys_path:
        set_api_keys_path(args.api_keys_path)
        print(f"✅ API Keys will be stored at: {args.api_keys_path}")
    
    # Register functions
    if args.use_unified_generate:  # Use unified generate function
        
        register_funcs(
            unified_generate,
            support_stream= True if not args.no_stream else False,  # the paramter of wrap_openai.register_funcs
            stream=True if not args.no_stream else False,  # the paramter of unified_generate
            temperature=0.7  # Server default
        )
        print("📝 Using unified generate function")

    else:  # Use separate functions

        if args.use_messages:
            # Use messages format functions
            if args.no_stream:
                # Non-streaming function with messages format (support_stream=False)
                register_funcs(
                    messages_generate,
                    support_stream=False,
                    temperature=0.7  # Server default, can be overridden by client
                )
                print("📝 Using messages format (non-streaming)")
            else:
                # Streaming function with messages format (support_stream=True - supports both streaming and non-streaming)
                register_funcs(
                    messages_stream_generate,
                    support_stream=True,
                    temperature=0.7  # Server default
                )
                print("📝 Using messages format (streaming)")
        else:
            # Use prompt format functions (default)
            if args.no_stream:
                # Non-streaming function (support_stream=False)
                register_funcs(
                    simple_generate,
                    support_stream=False,
                    temperature=0.7  # Server default, can be overridden by client
                )
                print("📝 Using prompt format (non-streaming)")
            else:
                # Streaming function (support_stream=True - supports both streaming and non-streaming)
                register_funcs(
                    simple_stream_generate,
                    support_stream=True,
                    temperature=0.7  # Server default
                )
                print("📝 Using prompt format (streaming)")
    
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
