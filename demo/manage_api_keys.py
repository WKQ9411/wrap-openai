import argparse
import requests
import sys
from datetime import datetime


def format_datetime(dt_str: str) -> str:
    """Format datetime string: replace T with space and remove microseconds"""
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, AttributeError):
        # If parsing fails, just replace T with space and remove microseconds
        if 'T' in dt_str:
            result = dt_str.replace('T', ' ')
            if '.' in result:
                result = result.split('.')[0]
            return result
        return dt_str


def generate_key(base_url: str, name: str = None):
    """Generate new API Key"""
    url = f"{base_url}/api/keys/generate"
    data = {}
    if name:
        data["name"] = name
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        
        print(f"\n✅ API Key generated: {result['api_key']}")
        print(f"   Name: {result['name']}")
        print(f"   Created: {format_datetime(result['created_at'])}")
        print(f"\n⚠️  {result['message']}\n")
        return result['api_key']
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   {e.response.text}")
        return None


def list_keys(base_url: str):
    """List all API Keys"""
    url = f"{base_url}/api/keys"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        result = response.json()
        
        print(f"\n📋 API Keys (Total: {result['total']})")
        print("=" * 60)
        
        if result['total'] == 0:
            print("No API Keys found")
        else:
            for key_info in result['keys']:
                print(f"\nKey: {key_info['api_key']}")
                print(f"Name: {key_info['name']}")
                print(f"Created: {format_datetime(key_info['created_at'])}")
                if key_info.get('last_used_at'):
                    print(f"Last used: {format_datetime(key_info['last_used_at'])}")
        print("\n")
        return result['keys']
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   {e.response.text}")
        return None


def revoke_key(base_url: str, api_key: str):
    """Revoke API Key"""
    url = f"{base_url}/api/keys/{api_key}"
    
    try:
        response = requests.delete(url)
        response.raise_for_status()
        result = response.json()
        print(f"\n✅ {result['message']}\n")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   {e.response.text}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wrap OpenAI API Key management")
    subparsers = parser.add_subparsers(dest="command", help="Command", required=True)
    
    # Generate
    gen_parser = subparsers.add_parser("generate", help="Generate new API Key")
    gen_parser.add_argument("--name", type=str, help="API Key name")
    gen_parser.add_argument("--base-url", type=str, default="http://localhost:8000", help="API service base URL")
    
    # List
    list_parser = subparsers.add_parser("list", help="List all API Keys")
    list_parser.add_argument("--base-url", type=str, default="http://localhost:8000", help="API service base URL")
    
    # Revoke
    revoke_parser = subparsers.add_parser("revoke", help="Revoke API Key")
    revoke_parser.add_argument("api_key", type=str, help="API Key to revoke (specify directly after revoke)")
    revoke_parser.add_argument("--base-url", type=str, default="http://localhost:8000", help="API service base URL")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        generate_key(args.base_url, args.name)
    elif args.command == "list":
        list_keys(args.base_url)
    elif args.command == "revoke":
        revoke_key(args.base_url, args.api_key)
    else:
        parser.print_help()
        sys.exit(1)

