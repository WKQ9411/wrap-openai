from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field


class TextContent(BaseModel):
    """Text content"""
    type: Literal["text"] = "text"
    text: str


class ImageURL(BaseModel):
    """Image URL"""
    url: str
    detail: Optional[str] = None  # "low", "high", "auto"


class ImageContent(BaseModel):
    """Image content"""
    type: Literal["image_url"] = "image_url"
    image_url: ImageURL


class Message(BaseModel):
    """
    Message model
    
    Supports multimodal input:
    - content can be a string (simple text)
    - content can be an array of objects (multimodal: text + images, etc.)
    
    Examples:
    # Simple text
    Message(role="user", content="Hello")
    
    # Multimodal (text + image)
    Message(role="user", content=[
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
    ])
    """
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[Union[TextContent, ImageContent, dict]]] = Field(
        ...,
        description="Message content, can be a string or an array of objects (supports multimodal)"
    )


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = Field(default="custom-model", description="Model name (optional, defaults to 'custom-model')")
    messages: List[Message] = Field(..., description="List of conversation messages")
    temperature: Optional[float] = Field(default=1.0, ge=0, le=2, description="Temperature parameter, controls randomness")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum number of tokens to generate")
    top_p: Optional[float] = Field(default=None, ge=0, le=1, description="Nucleus sampling parameter")
    top_k: Optional[int] = Field(default=None, ge=1, description="Top-K sampling parameter")
    presence_penalty: Optional[float] = Field(default=None, ge=-2, le=2, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(default=None, ge=-2, le=2, description="Frequency penalty")
    n: Optional[int] = Field(default=1, ge=1, description="Number of completion choices to generate")
    stop: Optional[List[str]] = Field(default=None, description="List of stop sequences")
    seed: Optional[int] = Field(default=None, description="Random seed")
    stream: Optional[bool] = Field(default=False, description="Whether to stream the response")


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: Optional[str] = Field(default="stop")


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default="chatcmpl-default")
    object: str = Field(default="chat.completion")
    created: int = Field(default=0)
    model: str = Field(default="custom-model")
    choices: List[Choice]
    usage: Usage


class ChatCompletionChunkDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str = Field(default="chatcmpl-default")
    object: str = Field(default="chat.completion.chunk")
    created: int = Field(default=0)
    model: str = Field(default="custom-model")
    choices: List[ChatCompletionChunkChoice]


class GenerateKeyRequest(BaseModel):
    """Request model for API Key generation"""
    name: Optional[str] = Field(default=None, description="API Key name (optional)")
