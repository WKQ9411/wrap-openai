import asyncio
import time
import uuid
import threading
import inspect
from typing import Callable, AsyncGenerator, Generator, Any, Optional, Union
from fastapi import FastAPI, HTTPException, Depends, Header, Security
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .api_keys import get_api_key_manager
from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    Message,
    Choice,
    Usage,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    GenerateKeyRequest,
)


app = FastAPI(
    title="Wrap OpenAI API",
    description="Wrap any custom generate function as an OpenAI SDK compatible API service",
    version="0.2.1",
)

# CORS configuration
_cors_enabled = False
_cors_origins = ["*"]
_cors_allow_credentials = False
_cors_allow_methods = ["*"]
_cors_allow_headers = ["*"]
_cors_middleware_added = False

# API Key authentication
security = HTTPBearer(auto_error=False)

# Store registered generate functions (supports streaming and non-streaming)
# First parameter can be either:
#   - prompt: str (for text-only, backward compatible)
#   - messages: list[dict] (original messages list, so that the function can process multimodal content if needed)
_registered_funcs = {
    'generate': {
        'func': None,  # Callable[[Union[str, list[dict]], ...], str] | None
        'default_params': {}  # dict[str, Any]
    },
    'stream': {
        'func': None,  # Callable[[Union[str, list[dict]], ...], Generator[str, None, None]] | None
        'default_params': {}  # dict[str, Any]
    }
}

# API Key verification switch
_api_key_required = False

# API Key management switch (controls whether remote API Key management is allowed)
_allow_remote_api_key_management = True

# Dynamic parameter list (these parameters can be dynamically overridden by client)
# Note: 
#   - OpenAI API officially supports: temperature, max_tokens, top_p, presence_penalty, frequency_penalty, n, stop, seed
#   - max_new_tokens and top_k are included for compatibility with other frameworks (e.g., HuggingFace Transformers)
_DYNAMIC_PARAMS = {
    'temperature',       # OpenAI API supported
    'max_tokens',        # OpenAI API supported
    'max_new_tokens',    # Compatible with Hugging Face (mapped to max_tokens in requests)
    'top_p',             # OpenAI API supported
    'top_k',             # Extended support (some frameworks use this instead of top_p)
    'presence_penalty',  # OpenAI API supported
    'frequency_penalty', # OpenAI API supported
    'n',                 # OpenAI API supported
    'stop',              # OpenAI API supported
    'seed'               # OpenAI API supported
}


def set_allow_remote_api_key_management(allow: bool):
    """
    Set whether remote API Key management is allowed
    
    Args:
        allow: If True, allows API Key management (generate/list/revoke) via HTTP API.
              If False, API Key management can only be done on the server side.
    """
    global _allow_remote_api_key_management
    _allow_remote_api_key_management = allow


def set_api_key_required(required: bool):
    """Set whether API Key verification is required"""
    global _api_key_required
    _api_key_required = required


def set_cors(
    enabled: bool = True,
    origins: Union[list[str], str] = "*",
    allow_credentials: bool = False,
    allow_methods: Union[list[str], str] = "*",
    allow_headers: Union[list[str], str] = "*",
):
    """
    Configure CORS settings
    
    Args:
        enabled: Whether to enable CORS (default: True)
        origins: List of allowed origins or "*" for all origins (default: "*")
        allow_credentials: Whether to allow credentials (default: False)
        allow_methods: Allowed HTTP methods or "*" for all methods (default: "*")
        allow_headers: Allowed headers or "*" for all headers (default: "*")
    """
    global _cors_enabled, _cors_origins, _cors_allow_credentials, _cors_allow_methods, _cors_allow_headers, _cors_middleware_added
    
    _cors_enabled = enabled
    
    # Convert string to list if needed
    if isinstance(origins, str):
        _cors_origins = [origins] if origins != "*" else ["*"]
    else:
        _cors_origins = origins
    
    _cors_allow_credentials = allow_credentials
    
    if isinstance(allow_methods, str):
        _cors_allow_methods = [allow_methods] if allow_methods != "*" else ["*"]
    else:
        _cors_allow_methods = allow_methods
    
    if isinstance(allow_headers, str):
        _cors_allow_headers = [allow_headers] if allow_headers != "*" else ["*"]
    else:
        _cors_allow_headers = allow_headers
    
    # Apply CORS middleware to app (only add once)
    if _cors_enabled and not _cors_middleware_added:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=_cors_origins if _cors_origins != ["*"] else ["*"],
            allow_credentials=_cors_allow_credentials,
            allow_methods=_cors_allow_methods if _cors_allow_methods != ["*"] else ["*"],
            allow_headers=_cors_allow_headers if _cors_allow_headers != ["*"] else ["*"],
        )
        _cors_middleware_added = True


async def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
    authorization: Optional[str] = Header(None),
):
    """
    Verify API Key
    
    Args:
        credentials: HTTP Bearer authentication credentials
        authorization: Authorization header
        
    Returns:
        Returns API Key if verification passes, otherwise raises HTTPException
    """
    if not _api_key_required:
        # If API Key verification is not enabled, allow all requests
        return None
    
    api_key = None
    
    # Try to get from Bearer token
    if credentials:
        api_key = credentials.credentials
    # Try to get from Authorization header (compatible with OpenAI format)
    elif authorization:
        if authorization.startswith("Bearer "):
            api_key = authorization.replace("Bearer ", "")
        else:
            api_key = authorization
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API Key not provided. Please add 'Authorization: Bearer <your-api-key>' to the request header"
        )
    
    # Verify API Key
    api_key_manager = get_api_key_manager()
    if not api_key_manager.validate_key(api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )
    
    return api_key


def _register_func_internal(func_type: str, func: Callable, **kwargs):
    """
    Internal unified registration function for both generate and stream functions
    
    Args:
        func_type: 'generate' or 'stream'
        func: Function to register
        **kwargs: Additional parameters
    """
    global _registered_funcs
    
    if kwargs:
        # Separate fixed and dynamic parameters
        from functools import partial
        fixed_params = {}
        dynamic_params = {}
        
        for key, value in kwargs.items():
            if key in _DYNAMIC_PARAMS:
                # Dynamic parameter, save as server default
                dynamic_params[key] = value
            else:
                # Fixed parameter (e.g., model, tokenizer), bind using partial
                fixed_params[key] = value
        
        # Save dynamic parameters as server defaults
        _registered_funcs[func_type]['default_params'] = dynamic_params
        
        # Only bind fixed parameters
        if fixed_params:
            _registered_funcs[func_type]['func'] = partial(func, **fixed_params)
        else:
            _registered_funcs[func_type]['func'] = func
    else:
        # Direct registration
        _registered_funcs[func_type]['func'] = func
        _registered_funcs[func_type]['default_params'] = {}


def register_funcs(generate_func: Callable, support_stream: bool, **kwargs):
    """
    Register generate function
    
    Args:
        generate_func: Generate function to register
                      - First parameter: prompt: str (text-only) OR messages: list[dict]
                      - Return type: str (support_stream=False) OR Generator[str, None, None] (support_stream=True)
        support_stream: Whether function supports streaming
                       - False: Returns str, only non-streaming mode (client's stream parameter ignored)
                       - True: Returns Generator, supports both modes (respects client's stream parameter)
        **kwargs: The parameters of the generate function
                 - Fixed params (e.g., model, tokenizer): bound permanently
                 - Dynamic params (e.g., temperature): server defaults, can be overridden by client
    
    Examples:
        # Non-streaming function (returns str)
        register_funcs(my_generate, support_stream=False)
        
        # Streaming function (returns Generator)
        register_funcs(my_stream_generate, support_stream=True)
        
        # With parameters
        register_funcs(my_generate, support_stream=False, model=model, tokenizer=tokenizer, temperature=0.7)
    """
    if generate_func is None:
        raise ValueError("generate_func cannot be None")
    
    if not isinstance(support_stream, bool):
        raise ValueError(f"support_stream must be a boolean, got {type(support_stream).__name__}")
    
    if support_stream:
        # Function returns Generator: supports both streaming and non-streaming modes
        # Register generator function for streaming mode (client requests stream=True)
        _register_func_internal('stream', generate_func, **kwargs)
        
        # Create wrapper that collects chunks for non-streaming mode (client requests stream=False)
        registered_stream_func = _registered_funcs['stream']['func']
        def non_stream_wrapper(*args, **func_kwargs):
            result = ""
            for chunk in registered_stream_func(*args, **func_kwargs):
                result += chunk
            return result
        _registered_funcs['generate']['func'] = non_stream_wrapper
        _registered_funcs['generate']['default_params'] = _registered_funcs['stream']['default_params'].copy()
    else:
        # Function returns str: only supports non-streaming mode
        # Client's stream parameter will be ignored (handled in chat_completions endpoint)
        _register_func_internal('generate', generate_func, **kwargs)


def _format_messages(messages: list[Message]) -> str:
    """
    Format message list into text
    
    Supports multimodal content:
    - If content is a string, use directly
    - If content is an array, convert multimodal content to text description
    """
    formatted = []
    for msg in messages:
        role = msg.role
        content = msg.content
        
        # Handle multimodal content
        if isinstance(content, list):
            # Multimodal content: convert to text description
            content_parts = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type == "text":
                        content_parts.append(item.get("text", ""))
                    elif item_type == "image_url":
                        image_url = item.get("image_url", {})
                        url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
                        content_parts.append(f"[Image: {url}]")
                    else:
                        content_parts.append(f"[{item_type}: {item}]")
                else:
                    content_parts.append(str(item))
            content_text = " ".join(content_parts)
        else:
            # String content
            content_text = str(content)
        
        if role == "system":
            formatted.append(f"System: {content_text}")
        elif role == "user":
            formatted.append(f"User: {content_text}")
        elif role == "assistant":
            formatted.append(f"Assistant: {content_text}")
    return "\n".join(formatted)


def _convert_messages_to_dict(messages: list[Message]) -> list[dict]:
    """
    Convert Message object list to dictionary list (preserves multimodal structure)
    
    Returns:
        Message dictionary list in format [{"role": "...", "content": "..."}, ...]
    """
    result = []
    for msg in messages:
        msg_dict = {
            "role": msg.role,
            "content": msg.content
        }
        result.append(msg_dict)
    return result


def _get_function_signature(func: Callable) -> dict[str, Any]:
    """
    Get function signature information, including parameter names and default values
    Supports functools.partial bound functions
    
    Returns:
        Dictionary containing parameter information, keys are parameter names, values are default values (if any)
    """
    try:
        # If it's a partial function, get original function signature
        from functools import partial
        if isinstance(func, partial):
            # Get original function
            original_func = func.func
            # Get bound parameters
            bound_args = func.keywords or {}
            # Get original function signature
            sig = inspect.signature(original_func)
            params = {}
            param_list = list(sig.parameters.items())
            # Skip first positional parameter (usually prompt)
            for i, (name, param) in enumerate(param_list):
                # Skip first positional parameter (usually prompt)
                if i == 0:
                    continue
                # Skip bound parameters
                if name in bound_args:
                    continue
                # Record parameter (regardless of whether it has a default value)
                params[name] = param.default if param.default != inspect.Parameter.empty else inspect.Parameter.empty
            return params
        else:
            # Regular function
            sig = inspect.signature(func)
            params = {}
            param_list = list(sig.parameters.items())
            # Skip first positional parameter (usually prompt)
            for i, (name, param) in enumerate(param_list):
                # Skip first positional parameter (usually prompt)
                if i == 0:
                    continue
                # Record parameter (regardless of whether it has a default value)
                params[name] = param.default if param.default != inspect.Parameter.empty else inspect.Parameter.empty
            return params
    except Exception:
        # If unable to get signature, return empty dictionary
        return {}


def _call_with_dynamic_params(func: Callable, prompt: str, request: ChatCompletionRequest, 
                             server_defaults: dict[str, Any] = None) -> Any:
    """
    Dynamically call function based on function signature, passing parameters from request
    
    Supports two modes:
    1. If function signature first parameter is `messages`, pass original messages list (supports multimodal)
    2. If function signature first parameter is `prompt`, pass formatted string (backward compatible)
    
    Parameter priority: client request parameters > server default parameters > function default parameters
    
    Args:
        func: Function to call (may be partial function)
        prompt: Formatted prompt text (for backward compatibility)
        request: Request object containing all possible parameters
        server_defaults: Server default parameters (fallback values)
        
    Returns:
        Function call result
    """
    if server_defaults is None:
        server_defaults = {}
    
    # Get function signature (including first parameter)
    from functools import partial
    if isinstance(func, partial):
        original_func = func.func
        sig = inspect.signature(original_func)
        param_list = list(sig.parameters.items())
        # Check if first parameter is bound
        bound_params = func.keywords or {}
    else:
        sig = inspect.signature(func)
        param_list = list(sig.parameters.items())
        bound_params = {}
    
    # Check first parameter (excluding bound ones)
    first_param_name = None
    first_param = None
    for name, param in param_list:
        if name not in bound_params:
            first_param_name = name
            first_param = param
            break
    
    # Determine what to pass as first argument
    # Priority: parameter name > type annotation > default (prompt)
    use_messages = False
    
    if first_param_name == "messages":
        # Parameter name is "messages", use messages list
        use_messages = True
    elif first_param is not None and first_param.annotation != inspect.Parameter.empty:
        # Check type annotation if parameter name is not "messages"
        annotation = first_param.annotation
        
        # Handle string annotations (e.g., "list[dict]" from __future__ annotations)
        if isinstance(annotation, str):
            if "list" in annotation.lower() and "dict" in annotation.lower():
                use_messages = True
        # Handle actual type annotations
        else:
            import typing
            origin = getattr(annotation, '__origin__', None)
            if origin is list or (hasattr(typing, 'List') and origin is typing.List):
                # It's a list type, likely messages
                use_messages = True
            elif origin is not None and 'list' in str(origin).lower() and 'dict' in str(annotation).lower():
                # Generic list type (e.g., list[dict])
                use_messages = True
    
    if use_messages:
        # Function accepts messages parameter, pass original messages list (supports multimodal)
        first_arg = _convert_messages_to_dict(request.messages)
    else:
        # Function accepts prompt parameter, pass formatted string (backward compatible)
        first_arg = prompt
    
    # Get function signature (excluding bound parameters and first parameter)
    sig_excluding_first = _get_function_signature(func)
    
    # Build parameter dictionary
    kwargs = {}
    
    # Check and pass common parameters
    # Note: max_tokens may be named max_tokens in request, but user function may use max_new_tokens
    param_mapping = {
        'temperature': 'temperature',
        'max_tokens': 'max_tokens',
        'max_new_tokens': 'max_tokens',  # Compatible with max_new_tokens parameter name
        'top_p': 'top_p',
        'top_k': 'top_k',
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
        'n': 'n',
        'stop': 'stop',
        'seed': 'seed',
    }
    
    # Get parameter values from request
    request_params = {
        'temperature': request.temperature,
        'max_tokens': request.max_tokens,
        'top_p': getattr(request, 'top_p', None),
        'top_k': getattr(request, 'top_k', None),
        'presence_penalty': getattr(request, 'presence_penalty', None),
        'frequency_penalty': getattr(request, 'frequency_penalty', None),
        'n': getattr(request, 'n', None),
        'stop': getattr(request, 'stop', None),
        'seed': getattr(request, 'seed', None),
    }
    
    # Check function signature, if function accepts a parameter, pass it
    # Priority: client parameters > server default parameters
    for func_param_name, request_param_name in param_mapping.items():
        if func_param_name in sig_excluding_first:
            # Prefer client-provided parameters
            client_value = request_params.get(request_param_name)
            if client_value is not None:
                kwargs[func_param_name] = client_value
            # If client didn't provide, use server default
            elif func_param_name in server_defaults:
                kwargs[func_param_name] = server_defaults[func_param_name]
            # If neither, use function's own default (don't pass parameter)
    
    # Call function (if partial function, parameters will be merged automatically)
    return func(first_arg, **kwargs)


async def _async_generator_wrapper(sync_generator: Generator[str, None, None]):
    """Wrap synchronous generator as async generator to avoid blocking event loop"""
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue()
    finished = threading.Event()
    
    def run_generator():
        """Run synchronous generator in thread"""
        try:
            for item in sync_generator:
                # Put item into queue
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, {"error": str(e)})
        finally:
            finished.set()
            loop.call_soon_threadsafe(queue.put_nowait, {"done": True})
    
    # Run generator in background thread
    thread = threading.Thread(target=run_generator, daemon=True)
    thread.start()
    
    # Asynchronously get data from queue
    while True:
        try:
            # Wait for data in queue or generator completion
            if finished.is_set() and queue.empty():
                break
            
            # Use timeout to periodically check if generator is done
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                if finished.is_set():
                    break
                continue
            
            if isinstance(item, dict):
                if "error" in item:
                    raise Exception(item["error"])
                if "done" in item:
                    break
            else:
                yield item
        except Exception as e:
            raise


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    api_key: Optional[str] = Depends(verify_api_key)
):
    """
    OpenAI-compatible chat completion endpoint
    
    Supports both streaming and non-streaming modes.
    
    Stream mode selection logic:
    - If support_stream=True: Both generate_func and stream_func are registered (stream_func is the generator, 
      generate_func is a wrapper that collects chunks). Use client's stream parameter to decide which mode.
    - If support_stream=False: Only generate_func is registered (returns str). Client's stream parameter is ignored,
      always use non-streaming mode.
    """
    # Check which functions are available
    has_generate = _registered_funcs['generate']['func'] is not None
    has_stream = _registered_funcs['stream']['func'] is not None
    
    # Determine which mode to use
    client_requested_stream = bool(request.stream)
    
    if has_generate and has_stream:
        # Both available (support_stream=True): use client's stream parameter to decide
        use_stream = client_requested_stream
    elif has_stream:
        # Only streaming available: use streaming mode (ignore client's stream parameter)
        use_stream = True
    elif has_generate:
        # Only non-streaming available (support_stream=False)
        # If client requests streaming, return streaming response with complete content in one chunk
        # Otherwise, return non-streaming response
        use_stream = client_requested_stream
    else:
        # Neither available: error
        raise HTTPException(
            status_code=500,
            detail="No generate function registered. Please call register_funcs(generate_func, ...) first"
        )
    
    if use_stream:
        # Streaming endpoint
        
        async def generate_stream():
            """Async generator"""
            prompt = _format_messages(request.messages)
            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            created = int(time.time())
            model_name = request.model or "custom-model"  # Use default if model not provided
            
            try:
                # Check if server only supports non-streaming but client requested streaming
                if not has_stream and has_generate:
                    # Server only supports non-streaming, but client requested streaming
                    # Generate complete response first, then yield as single chunk
                    response_text = _call_with_dynamic_params(
                        _registered_funcs['generate']['func'], prompt, request, _registered_funcs['generate']['default_params']
                    )
                    
                    # Send warning chunk first
                    warning_chunk = ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=model_name,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(
                                    content="[Warning: Server does not support streaming. Returning complete response in one chunk.]\n\n"
                                ),
                                finish_reason=None
                            )
                        ]
                    )
                    yield f"data: {warning_chunk.model_dump_json()}\n\n"
                    
                    # Yield complete content as single chunk
                    if response_text:
                        content_chunk = ChatCompletionChunk(
                            id=chunk_id,
                            created=created,
                            model=model_name,
                            choices=[
                                ChatCompletionChunkChoice(
                                    index=0,
                                    delta=ChatCompletionChunkDelta(content=response_text),
                                    finish_reason=None
                                )
                            ]
                        )
                        yield f"data: {content_chunk.model_dump_json()}\n\n"
                    
                    # Send completion marker
                    final_chunk = ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=model_name,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(),
                                finish_reason="stop"
                            )
                        ]
                    )
                    yield f"data: {final_chunk.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"
                else:
                    # Normal streaming: server supports streaming
                    # Call synchronous generator, passing dynamic parameters (using streaming function server defaults)
                    sync_generator = _call_with_dynamic_params(
                        _registered_funcs['stream']['func'], prompt, request, _registered_funcs['stream']['default_params']
                    )
                    
                    # Convert synchronous generator to async generator to avoid blocking event loop
                    async for chunk_text in _async_generator_wrapper(sync_generator):
                        if chunk_text:
                            chunk = ChatCompletionChunk(
                                id=chunk_id,
                                created=created,
                                model=model_name,
                                choices=[
                                    ChatCompletionChunkChoice(
                                        index=0,
                                        delta=ChatCompletionChunkDelta(content=chunk_text),
                                        finish_reason=None
                                    )
                                ]
                            )
                            yield f"data: {chunk.model_dump_json()}\n\n"
                    
                    # Send completion marker
                    final_chunk = ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=model_name,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(),
                                finish_reason="stop"
                            )
                        ]
                    )
                    yield f"data: {final_chunk.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error during generation: {str(e)}")
        
        return StreamingResponse(
            generate_stream(), 
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            }
        )
    else:
        # Non-streaming endpoint (client requested non-streaming)
        
        prompt = _format_messages(request.messages)
        model_name = request.model or "custom-model"  # Use default if model not provided
        
        try:
            if has_stream and not client_requested_stream:
                # Server supports streaming but client requested non-streaming
                # Collect all chunks from streaming function and return complete response
                sync_generator = _call_with_dynamic_params(
                    _registered_funcs['stream']['func'], prompt, request, _registered_funcs['stream']['default_params']
                )
                response_text = ""
                for chunk_text in sync_generator:
                    response_text += chunk_text
            else:
                # Server only supports non-streaming, or both available and client requested non-streaming
                # Call generate function, passing dynamic parameters (using non-streaming function server defaults)
                response_text = _call_with_dynamic_params(
                    _registered_funcs['generate']['func'], prompt, request, _registered_funcs['generate']['default_params']
                )
            
            # Build response
            response = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                created=int(time.time()),
                model=model_name,
                choices=[
                    Choice(
                        index=0,
                        message=Message(role="assistant", content=response_text),
                        finish_reason="stop"
                    )
                ],
                usage=Usage(
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=len(response_text.split()),
                    total_tokens=len(prompt.split()) + len(response_text.split())
                )
            )
            
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during generation: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "generate_func_registered": _registered_funcs['generate']['func'] is not None,
        "stream_generate_func_registered": _registered_funcs['stream']['func'] is not None,
        "api_key_required": _api_key_required,
        "allow_remote_api_key_management": _allow_remote_api_key_management,
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Wrap OpenAI API Service",
        "version": "0.2.1",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "health": "/health",
            "api_keys": "/api/keys",
        }
    }


# API Key management endpoints
@app.post("/api/keys/generate")
async def generate_api_key(request: GenerateKeyRequest):
    """
    Generate new API Key
    
    Args:
        request: Request body containing optional name
    
    Returns:
        Generated API Key information
    
    Raises:
        HTTPException: If remote API Key management is disabled
    """
    if not _allow_remote_api_key_management:
        raise HTTPException(
            status_code=403,
            detail="Remote API Key management is disabled. Please manage API Keys on the server side."
        )
    
    api_key_manager = get_api_key_manager()
    api_key = api_key_manager.generate_key(name=request.name)
    key_info = api_key_manager.get_key_info(api_key)
    
    return {
        "api_key": api_key,  # Full key only returned here
        "key_preview": key_info["key_preview"],
        "name": key_info["name"],
        "created_at": key_info["created_at"],
        "message": "Please save this API Key securely, it will only be shown once!"
    }


@app.get("/api/keys")
async def list_api_keys():
    """
    List all API Keys
    
    Returns:
        API Keys list
    
    Raises:
        HTTPException: If remote API Key management is disabled
    """
    if not _allow_remote_api_key_management:
        raise HTTPException(
            status_code=403,
            detail="Remote API Key management is disabled. Please manage API Keys on the server side."
        )
    
    api_key_manager = get_api_key_manager()
    return {
        "keys": api_key_manager.list_keys(),
        "total": len(api_key_manager.keys)
    }


@app.delete("/api/keys/{api_key}")
async def revoke_api_key(api_key: str):
    """
    Revoke API Key
    
    Args:
        api_key: API Key to revoke
    
    Returns:
        Operation result
    
    Raises:
        HTTPException: If remote API Key management is disabled
    """
    if not _allow_remote_api_key_management:
        raise HTTPException(
            status_code=403,
            detail="Remote API Key management is disabled. Please manage API Keys on the server side."
        )
    
    api_key_manager = get_api_key_manager()
    if api_key_manager.revoke_key(api_key):
        return {
            "success": True,
            "message": f"API Key revoked"
        }
    else:
        raise HTTPException(
            status_code=404,
            detail="API Key does not exist"
        )


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    require_api_key: bool = False,
    allow_remote_api_key_management: bool = True,
    enable_cors: bool = True,
    cors_origins: Union[list[str], str] = "*",
    cors_allow_credentials: bool = False,
    cors_allow_methods: Union[list[str], str] = "*",
    cors_allow_headers: Union[list[str], str] = "*",
):
    """
    Run server
    
    Args:
        host: Host address to bind
        port: Port number
        require_api_key: Whether API Key verification is required
        allow_remote_api_key_management: Whether to allow API Key management (generate/list/revoke) via HTTP API.
                                        If False, API Key management can only be done on the server side.
        enable_cors: Whether to enable CORS (default: True)
        cors_origins: List of allowed origins or "*" for all origins (default: "*")
        cors_allow_credentials: Whether to allow credentials in CORS (default: False)
        cors_allow_methods: Allowed HTTP methods or "*" for all methods (default: "*")
        cors_allow_headers: Allowed headers or "*" for all headers (default: "*")
    """
    if require_api_key:
        set_api_key_required(True)
        print("✅  API Key verification enabled")
    else:
        print("⚠️  API Key verification disabled, all requests are allowed")
    
    set_allow_remote_api_key_management(allow_remote_api_key_management)
    if allow_remote_api_key_management:
        print("✅  Remote API Key management enabled")
    else:
        print("🔒  Remote API Key management disabled (API Keys can only be managed on server side)")
    
    # Configure CORS
    if enable_cors:
        set_cors(
            enabled=True,
            origins=cors_origins,
            allow_credentials=cors_allow_credentials,
            allow_methods=cors_allow_methods,
            allow_headers=cors_allow_headers,
        )
        print("✅  CORS enabled")
    else:
        print("⚠️  CORS disabled")
    
    # Check streaming support
    has_generate = _registered_funcs['generate']['func'] is not None
    has_stream = _registered_funcs['stream']['func'] is not None
    
    if has_stream:
        print("✅  Streaming mode supported")
    elif has_generate:
        print("⚠️  Streaming mode not supported (only non-streaming mode available)")
    else:
        print("❌  No generate function registered!")
    
    uvicorn.run(app, host=host, port=port)

