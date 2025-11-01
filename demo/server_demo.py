from typing import Generator, Optional, List, Dict, Union
from wrap_openai import register_funcs, run_server
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading


def load_model(model_path: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Load Qwen model and tokenizer"""
    print(f"🔄 Loading model: {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Fix pad_token to avoid attention mask warning
    # If pad_token is None or same as eos_token, use unk_token or eos_token
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        else:
            # Fallback to eos_token if unk_token is not available
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto"
    )
    print("✅ Model loaded successfully!")
    return model, tokenizer


def extract_text_from_content(content: Union[str, List[Dict]]) -> str:
    """
    Extract text content from message content
    
    Supports:
    - String content: returns directly
    - List content with type field: extracts text from "text" type items
      For other types (e.g., "image_url"), converts to text description
    
    Args:
        content: Message content (str or list of content objects)
        
    Returns:
        Extracted text string
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Multimodal content: extract text parts
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type", "")
                if item_type == "text":
                    # Extract text from text type
                    text_parts.append(item.get("text", ""))
                elif item_type == "image_url":
                    # Convert image to text description
                    image_url = item.get("image_url", {})
                    url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
                    text_parts.append(f"[Image: {url}]")
                else:
                    # Other types: convert to string representation
                    text_parts.append(f"[{item_type}: {item}]")
            else:
                # Fallback: convert to string
                text_parts.append(str(item))
        return " ".join(text_parts)
    else:
        # Fallback: convert to string
        return str(content)


def stream_generate(
    messages: List[Dict],
    model,
    tokenizer,
    max_new_tokens: int = 512,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> Generator[str, None, None]:
    """
    Streaming generation function
    
    Supports:
    - messages: List of message dicts with role and content
    - content can be string or list with type field (multimodal support)
    
    Examples:
    # Simple text messages
    messages = [
        {"role": "user", "content": "Hello"}
    ]
    
    # Multimodal messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
            ]
        }
    ]
    
    Supports dynamic parameters that can be overridden by client requests.
    """
    # Convert messages to tokenizer format (extract text from content)
    # For multimodal content, extract text parts
    tokenizer_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        # Extract text from content (handles both str and list with type field)
        text_content = extract_text_from_content(content)
        
        tokenizer_messages.append({
            "role": role,
            "content": text_content
        })
    
    try:
        text = tokenizer.apply_chat_template(tokenizer_messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        # Fallback: combine all messages as plain text
        text = "\n".join([
            f"{msg.get('role', 'user')}: {extract_text_from_content(msg.get('content', ''))}"
            for msg in messages
        ])
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    actual_max_new_tokens = max_tokens if max_tokens is not None else max_new_tokens
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = {
        "input_ids": model_inputs.input_ids,
        "attention_mask": model_inputs.attention_mask,  # Explicitly pass attention_mask
        "max_new_tokens": actual_max_new_tokens,
        "do_sample": True,
        "streamer": streamer,
        "temperature": temperature,
    }
    
    if top_p is not None:
        generation_kwargs["top_p"] = top_p
    if top_k is not None:
        generation_kwargs["top_k"] = top_k
    
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    for token in streamer:
        yield token


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen2.5-0.5B Server Demo")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                       help="Model path or HuggingFace model ID")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server bind address")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--require-api-key", action="store_true",
                       help="Enable API Key verification")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model)
    
    # Register streaming function (supports both streaming and non-streaming modes)
    register_funcs(
        stream_generate,
        support_stream=True,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
    )
    
    print("\n" + "=" * 60)
    print("🚀 Qwen2.5-0.5B Server Started")
    print("=" * 60)
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Endpoint: http://{args.host}:{args.port}/v1/chat/completions")
    print("=" * 60 + "\n")
    
    # Start server
    run_server(host=args.host, port=args.port, require_api_key=args.require_api_key)

