from openai import OpenAI
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wrap OpenAI API Client")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000/v1", help="API service URL")
    parser.add_argument("--api-key", type=str, default="sk-dummy", help="API Key (use actual key if verification is enabled)")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming mode (default: streaming)")
    
    args = parser.parse_args()
    
    # Determine streaming mode: default is True, --no-stream disables it
    use_stream = not args.no_stream
    
    # Initialize client
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    
    mode = "streaming" if use_stream else "non-streaming"
    print("=" * 60)
    print(f"Wrap OpenAI API Client ({mode} mode)")
    print("=" * 60)
    print("Type your message and press Enter.")
    print("Type 'quit' or 'exit' to exit.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            # Non-streaming request
            if not use_stream:
                response = client.chat.completions.create(
                    model="custom-model",
                    messages=[{"role": "user", "content": user_input}]
                )
                print(f"Assistant: {response.choices[0].message.content}\n")
            
            # Streaming request
            else:
                print("Assistant: ", end="", flush=True)
                stream = client.chat.completions.create(
                    model="custom-model",
                    messages=[{"role": "user", "content": user_input}],
                    stream=True
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        print(chunk.choices[0].delta.content, end="", flush=True)
                print("\n")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")
