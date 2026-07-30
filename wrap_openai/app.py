import asyncio
import time
import uuid
import threading
import inspect
from typing import Callable, Generator, Any, Optional, Union
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
    version="0.3.0",
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

# A registered generate function always receives OpenAI messages as its first
# positional argument. All remaining arguments are passed as keyword arguments.
_registered_generate = {
    "func": None,
    "support_stream": False,
    "model_id": None,
    "fixed_kwargs": {},
    "openai_kwargs": {},
    "custom_kwargs": {},
}

# API Key verification switch
_api_key_required = False

# API Key management switch (controls whether remote API Key management is allowed)
_allow_remote_api_key_management = True

# OpenAI request fields that can be explicitly enabled through openai_kwargs.
_OPENAI_GENERATION_KWARGS = {
    "temperature",
    "max_tokens",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "n",
    "stop",
    "seed",
}

_RESERVED_GENERATE_KWARGS = {"messages", "model_id", "stream"}
_RESERVED_CUSTOM_KWARGS = _RESERVED_GENERATE_KWARGS | {"model"}


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


def _copy_kwargs(name: str, value: Optional[dict[str, Any]]) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dict or None")
    if any(not isinstance(key, str) or not key for key in value):
        raise ValueError(f"All keys in {name} must be non-empty strings")
    return dict(value)


def _validate_registered_kwargs(
    generate_func: Callable,
    fixed_kwargs: dict[str, Any],
    openai_kwargs: dict[str, Any],
    custom_kwargs: dict[str, Any],
) -> None:
    groups = {
        "fixed_kwargs": set(fixed_kwargs),
        "openai_kwargs": set(openai_kwargs),
        "custom_kwargs": set(custom_kwargs),
    }

    group_names = list(groups)
    for index, first_name in enumerate(group_names):
        for second_name in group_names[index + 1:]:
            overlap = groups[first_name] & groups[second_name]
            if overlap:
                joined = ", ".join(sorted(overlap))
                raise ValueError(
                    f"Parameters cannot appear in both {first_name} and {second_name}: {joined}"
                )

    invalid_openai = set(openai_kwargs) - _OPENAI_GENERATION_KWARGS
    if invalid_openai:
        joined = ", ".join(sorted(invalid_openai))
        raise ValueError(
            f"Unsupported parameters in openai_kwargs: {joined}. "
            "Move non-OpenAI parameters to custom_kwargs."
        )

    misplaced_openai = (set(fixed_kwargs) | set(custom_kwargs)) & _OPENAI_GENERATION_KWARGS
    if misplaced_openai:
        joined = ", ".join(sorted(misplaced_openai))
        raise ValueError(f"OpenAI parameters must be declared in openai_kwargs: {joined}")

    reserved = (set(fixed_kwargs) | set(openai_kwargs)) & _RESERVED_GENERATE_KWARGS
    reserved |= set(custom_kwargs) & _RESERVED_CUSTOM_KWARGS
    if reserved:
        joined = ", ".join(sorted(reserved))
        raise ValueError(f"Reserved parameters cannot be registered as kwargs: {joined}")

    all_kwargs = {**fixed_kwargs, **openai_kwargs, **custom_kwargs}
    try:
        signature = inspect.signature(generate_func)
        signature.bind(object(), **all_kwargs)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "generate_func must accept messages as its first positional argument "
            f"and all registered keyword arguments: {exc}"
        ) from exc


def register_generate(
    generate_func: Callable,
    *,
    support_stream: bool,
    model_id: str,
    fixed_kwargs: Optional[dict[str, Any]] = None,
    openai_kwargs: Optional[dict[str, Any]] = None,
    custom_kwargs: Optional[dict[str, Any]] = None,
) -> None:
    """Register a custom generate function as an OpenAI-compatible API.

    The generate function receives the OpenAI messages list as its first
    positional argument. Registered values are flattened and passed as keyword
    arguments.

    Args:
        generate_func: Custom function returning ``str`` when support_stream is
            False, or an iterable of string chunks when support_stream is True.
        support_stream: Whether generate_func returns streaming text chunks.
        model_id: Model identifier exposed through the OpenAI-compatible API.
        fixed_kwargs: Server-only values that clients cannot override.
        openai_kwargs: Defaults for enabled OpenAI generation parameters.
            Clients override them through standard request fields.
        custom_kwargs: Defaults for custom generation parameters. Clients
            override them through OpenAI SDK ``extra_body`` fields.
    """
    global _registered_generate

    if not callable(generate_func):
        raise TypeError("generate_func must be callable")
    if not isinstance(support_stream, bool):
        raise TypeError("support_stream must be a bool")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")

    fixed = _copy_kwargs("fixed_kwargs", fixed_kwargs)
    openai = _copy_kwargs("openai_kwargs", openai_kwargs)
    custom = _copy_kwargs("custom_kwargs", custom_kwargs)
    _validate_registered_kwargs(generate_func, fixed, openai, custom)

    _registered_generate = {
        "func": generate_func,
        "support_stream": support_stream,
        "model_id": model_id,
        "fixed_kwargs": fixed,
        "openai_kwargs": openai,
        "custom_kwargs": custom,
    }


def _convert_messages_to_dict(messages: list[Message]) -> list[dict]:
    return [message.model_dump(exclude_none=True) for message in messages]


def _resolve_request_kwargs(request: ChatCompletionRequest) -> dict[str, Any]:
    fixed_kwargs = _registered_generate["fixed_kwargs"]
    openai_defaults = _registered_generate["openai_kwargs"]
    custom_defaults = _registered_generate["custom_kwargs"]

    provided_openai = request.model_fields_set & _OPENAI_GENERATION_KWARGS
    unsupported_openai = provided_openai - set(openai_defaults)
    if unsupported_openai:
        joined = ", ".join(sorted(unsupported_openai))
        raise HTTPException(
            status_code=422,
            detail=f"OpenAI parameters not enabled for this generate function: {joined}",
        )

    effective_openai = dict(openai_defaults)
    for name in provided_openai:
        effective_openai[name] = getattr(request, name)

    extra_fields = dict(request.model_extra or {})
    fixed_conflicts = set(extra_fields) & set(fixed_kwargs)
    if fixed_conflicts:
        joined = ", ".join(sorted(fixed_conflicts))
        raise HTTPException(
            status_code=422,
            detail=f"Fixed parameters cannot be overridden by clients: {joined}",
        )

    unknown_custom = set(extra_fields) - set(custom_defaults)
    if unknown_custom:
        joined = ", ".join(sorted(unknown_custom))
        allowed = ", ".join(sorted(custom_defaults)) or "none"
        raise HTTPException(
            status_code=422,
            detail=f"Unknown custom parameters: {joined}. Allowed custom parameters: {allowed}",
        )

    effective_custom = {**custom_defaults, **extra_fields}
    return {**fixed_kwargs, **effective_openai, **effective_custom}


def _call_registered_generate(request: ChatCompletionRequest) -> Any:
    messages = _convert_messages_to_dict(request.messages)
    kwargs = _resolve_request_kwargs(request)
    return _registered_generate["func"](messages, **kwargs)


def _iter_text_chunks(result: Any):
    if isinstance(result, str) or not hasattr(result, "__iter__"):
        raise TypeError(
            "support_stream=True requires generate_func to return an iterable of strings"
        )
    for chunk in result:
        if not isinstance(chunk, str):
            raise TypeError("Streaming generate_func must yield strings")
        yield chunk


def _estimate_content_tokens(value: Any) -> int:
    if isinstance(value, str):
        return len(value.split())
    if isinstance(value, list):
        return sum(_estimate_content_tokens(item) for item in value)
    if isinstance(value, dict):
        return sum(_estimate_content_tokens(item) for item in value.values())
    return 0


def _estimate_message_tokens(messages: list[dict]) -> int:
    return sum(_estimate_content_tokens(message.get("content")) for message in messages)


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
    """OpenAI-compatible chat completion endpoint."""
    if _registered_generate["func"] is None:
        raise HTTPException(
            status_code=500,
            detail="No generate function registered. Call register_generate(...) first.",
        )

    registered_model_id = _registered_generate["model_id"]
    if request.model != registered_model_id:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model}' is not registered. Available model: {registered_model_id}",
        )

    if request.stream:
        async def generate_stream():
            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            created = int(time.time())

            try:
                result = _call_registered_generate(request)

                if _registered_generate["support_stream"]:
                    chunks = _iter_text_chunks(result)
                    async for chunk_text in _async_generator_wrapper(chunks):
                        if not chunk_text:
                            continue
                        chunk = ChatCompletionChunk(
                            id=chunk_id,
                            created=created,
                            model=registered_model_id,
                            choices=[
                                ChatCompletionChunkChoice(
                                    index=0,
                                    delta=ChatCompletionChunkDelta(content=chunk_text),
                                    finish_reason=None,
                                )
                            ],
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"
                else:
                    if not isinstance(result, str):
                        raise TypeError(
                            "support_stream=False requires generate_func to return a string"
                        )

                    warning_chunk = ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=registered_model_id,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(
                                    content="[Warning: Server does not support streaming. Returning complete response in one chunk.]\n\n"
                                ),
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {warning_chunk.model_dump_json()}\n\n"

                    if result:
                        content_chunk = ChatCompletionChunk(
                            id=chunk_id,
                            created=created,
                            model=registered_model_id,
                            choices=[
                                ChatCompletionChunkChoice(
                                    index=0,
                                    delta=ChatCompletionChunkDelta(content=result),
                                    finish_reason=None,
                                )
                            ],
                        )
                        yield f"data: {content_chunk.model_dump_json()}\n\n"

                final_chunk = ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=registered_model_id,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(),
                            finish_reason="stop",
                        )
                    ],
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error during generation: {str(e)}")

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        result = _call_registered_generate(request)
        if _registered_generate["support_stream"]:
            response_text = "".join(_iter_text_chunks(result))
        else:
            if not isinstance(result, str):
                raise TypeError("support_stream=False requires generate_func to return a string")
            response_text = result

        messages = _convert_messages_to_dict(request.messages)
        prompt_tokens = _estimate_message_tokens(messages)
        completion_tokens = len(response_text.split())
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=registered_model_id,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during generation: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "generate_registered": _registered_generate["func"] is not None,
        "model_id": _registered_generate["model_id"],
        "support_stream": _registered_generate["support_stream"],
        "api_key_required": _api_key_required,
        "allow_remote_api_key_management": _allow_remote_api_key_management,
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Wrap OpenAI API Service",
        "version": "0.3.0",
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
    
    if _registered_generate["func"] is None:
        print("❌  No generate function registered!")
    elif _registered_generate["support_stream"]:
        print("✅  Streaming mode supported")
        print(f"✅  Registered model: {_registered_generate['model_id']}")
    else:
        print("⚠️  Streaming mode not supported (only non-streaming mode available)")
        print(f"✅  Registered model: {_registered_generate['model_id']}")
    
    uvicorn.run(app, host=host, port=port)

