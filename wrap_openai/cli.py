import argparse
from typing import Optional
from .api_keys import get_api_key_manager, get_api_keys_dir, set_api_keys_path


def generate_key(name: Optional[str] = None):
    """Generate new API Key"""
    api_key_manager = get_api_key_manager()
    api_key = api_key_manager.generate_key(name=name)
    key_info = api_key_manager.get_key_info(api_key)
    
    keys_dir = get_api_keys_dir()
    
    print("\n" + "=" * 60)
    print("✅ API Key generated successfully!")
    print("=" * 60)
    print(f"API Key: {api_key}")
    print(f"Name: {key_info['name']}")
    print(f"Created at: {key_info['created_at']}")
    print(f"\nStorage location: {keys_dir / 'keys.json'}")
    print("\nℹ️  Please save this API Key securely, it will only be shown once!")
    print("=" * 60 + "\n")
    
    return api_key


def list_keys():
    """List all API Keys"""
    api_key_manager = get_api_key_manager()
    keys = api_key_manager.list_keys()
    keys_dir = get_api_keys_dir()
    
    print("\n" + "=" * 60)
    print(f"API Keys list (Total: {len(keys)})")
    print("=" * 60)
    print(f"Storage location: {keys_dir / 'keys.json'}")
    print("=" * 60)
    
    if len(keys) == 0:
        print("No API Keys found")
    else:
        for key_info in keys:
            print(f"\nAPI Key: {key_info['api_key']}")
            print(f"Name: {key_info['name']}")
            print(f"Created at: {key_info['created_at']}")
            if key_info.get('last_used_at'):
                print(f"Last used: {key_info['last_used_at']}")
            print("-" * 60)
    
    print()
    return keys


def revoke_key(api_key: str):
    """Revoke API Key"""
    api_key_manager = get_api_key_manager()
    keys_dir = get_api_keys_dir()
    
    if api_key_manager.revoke_key(api_key):
        print("\n" + "=" * 60)
        print("✅ API Key revoked successfully!")
        print("=" * 60)
        print(f"Revoked API Key: {api_key}")
        print(f"Storage location: {keys_dir / 'keys.json'}")
        print("=" * 60 + "\n")
        return True
    else:
        print("\n❌ Error: API Key not found or already revoked")
        print(f"Storage location: {keys_dir / 'keys.json'}\n")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Wrap OpenAI API Key management tool",
        prog="wrap-openai"
    )
    
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate new API Key"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all API Keys"
    )
    parser.add_argument(
        "--revoke",
        type=str,
        metavar="API_KEY",
        help="Revoke API Key"
    )
    parser.add_argument(
        "--name",
        type=str,
        help="API Key name (used with --generate)"
    )
    parser.add_argument(
        "--api-keys-path",
        type=str,
        help="Custom path for API Keys storage (directory or file path)"
    )
    
    args = parser.parse_args()
    
    # Set API keys path if provided (must be done before any API key operations)
    # Use silent=True to keep output consistent with default path behavior
    if args.api_keys_path:
        set_api_keys_path(args.api_keys_path, silent=True)
    
    if args.generate:
        generate_key(args.name)
    elif args.list:
        list_keys()
    elif args.revoke:
        revoke_key(args.revoke)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

