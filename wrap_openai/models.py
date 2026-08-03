from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class TextContent(BaseModel):
    """Text content."""

    model_config = ConfigDict(extra="allow")

    type: Literal["text"] = "text"
    text: str


class ImageURL(BaseModel):
    """Image URL."""

    model_config = ConfigDict(extra="allow")

    url: str
    detail: Optional[str] = None


class ImageContent(BaseModel):
    """Image content."""

    model_config = ConfigDict(extra="allow")

    type: Literal["image_url"] = "image_url"
    image_url: ImageURL


class Message(BaseModel):
    """OpenAI-compatible chat message.

    Extra fields are preserved so developer, tool, multimodal, and future
    message variants reach the registered generate function unchanged.
    """

    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = None


class ChatCompletionStreamOptions(BaseModel):
    """Options that control the streamed protocol response."""

    model_config = ConfigDict(extra="allow")

    include_usage: bool = False
    include_obfuscation: Optional[bool] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI Chat Completions request.

    Standard fields are declared explicitly so they can be accepted even when
    the registered generate function does not use them. Only fields registered
    through ``openai_kwargs`` are forwarded to that function.
    """

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="Registered model identifier")
    messages: List[Message] = Field(..., description="List of conversation messages")
    stream: bool = False
    stream_options: Optional[ChatCompletionStreamOptions] = None

    audio: Any = None
    frequency_penalty: Any = None
    function_call: Any = None
    functions: Any = None
    logit_bias: Any = None
    logprobs: Any = None
    max_completion_tokens: Any = None
    max_tokens: Any = None
    metadata: Any = None
    modalities: Any = None
    moderation: Any = None
    n: Any = None
    parallel_tool_calls: Any = None
    prediction: Any = None
    presence_penalty: Any = None
    prompt_cache_key: Any = None
    prompt_cache_options: Any = None
    prompt_cache_retention: Any = None
    reasoning_effort: Any = None
    response_format: Any = None
    safety_identifier: Any = None
    seed: Any = None
    service_tier: Any = None
    stop: Any = None
    store: Any = None
    temperature: Any = None
    tool_choice: Any = None
    tools: Any = None
    top_logprobs: Any = None
    top_p: Any = None
    user: Any = None
    verbosity: Any = None
    web_search_options: Any = None


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: Optional[str] = "stop"
    logprobs: Any = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: Any = None
    completion_tokens_details: Any = None


class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl-default"
    object: str = "chat.completion"
    created: int = 0
    model: str = "custom-model"
    choices: List[Choice]
    usage: Usage
    service_tier: Any = None
    system_fingerprint: Optional[str] = None


class ChatCompletionChunkDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    refusal: Optional[str] = None
    tool_calls: Any = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None
    logprobs: Any = None


class ChatCompletionChunk(BaseModel):
    id: str = "chatcmpl-default"
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = "custom-model"
    choices: List[ChatCompletionChunkChoice]
    usage: Optional[Usage] = None
    service_tier: Any = None
    system_fingerprint: Optional[str] = None


class GenerateKeyRequest(BaseModel):
    """Request model for API Key generation."""

    name: Optional[str] = Field(default=None, description="API Key name (optional)")
