from openai import OpenAI


class ChatDemo:
    """Simple CLI Chat Application"""
    
    def __init__(self, base_url: str = "http://localhost:8000/v1", 
                 api_key: str = "sk-dummy", stream: bool = True):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.messages = []
        self.stream = stream
    
    def chat(self, user_input: str) -> str:
        """Send message and get response"""
        self.messages.append({"role": "user", "content": user_input})
        
        try:
            response = self.client.chat.completions.create(
                model="custom-model",  # Model name doesn't matter for custom server
                messages=self.messages,
                stream=self.stream
            )
            
            if self.stream:
                full_response = ""
                print("\nAssistant: ", end="", flush=True)
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        full_response += content
                print("\n")
                self.messages.append({"role": "assistant", "content": full_response})
                return full_response
            else:
                content = response.choices[0].message.content
                print(f"\nAssistant: {content}\n")
                self.messages.append({"role": "assistant", "content": content})
                return content
        except Exception as e:
            print(f"❌ Error: {e}\n")
            return ""
    
    def run(self):
        """Run interactive chat loop"""
        print("=" * 60)
        print("💬 Chat Demo")
        print("=" * 60)
        print("Type your message and press Enter.")
        print("Type 'quit', 'exit', or 'bye' to exit.")
        print("Type 'clear' to clear conversation history.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    self.messages = []
                    print("✅ Conversation history cleared.\n")
                    continue
                
                self.chat(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple CLI Chat Demo")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1",
                       help="API service URL")
    parser.add_argument("--api-key", type=str, default="sk-dummy",
                       help="API Key (use 'sk-dummy' if API Key verification is disabled)")
    parser.add_argument("--no-stream", action="store_true",
                       help="Disable streaming mode")
    
    args = parser.parse_args()
    
    demo = ChatDemo(
        base_url=args.base_url,
        api_key=args.api_key,
        stream=not args.no_stream
    )
    
    demo.run()

