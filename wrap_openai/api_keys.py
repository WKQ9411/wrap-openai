import secrets
import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

# Custom API Keys storage path (if set, overrides default project root/.api_keys)
# Can be either:
# - A directory path: keys.json will be stored in this directory
# - A file path: keys will be stored at this exact path
_custom_api_keys_path: Optional[Path] = None
_custom_api_keys_file: Optional[Path] = None  # If set, use this exact file path


def _get_project_root() -> Path:
    """
    Get project root directory (directory containing pyproject.toml)
    
    Search strategy (by priority):
    1. Check environment variable WRAP_OPENAI_PROJECT_ROOT (if set)
    2. Search upward from current file (__file__) for directory containing pyproject.toml (most reliable)
    3. Check if current working directory contains pyproject.toml
    4. If all fail, use current working directory as fallback and print warning
    
    Returns:
        Project root directory path
    """
    import os
    
    # Strategy 1: Check environment variable (allows manual specification)
    env_root = os.getenv("WRAP_OPENAI_PROJECT_ROOT")
    if env_root:
        env_path = Path(env_root).resolve()
        if env_path.exists() and (env_path / "pyproject.toml").exists():
            return env_path
    
    # Strategy 2: Search upward from current file (most reliable)
    # This way even if package is installed, can find correct project root
    current_file = Path(__file__).resolve()
    for parent in [current_file.parent] + list(current_file.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    
    # Strategy 3: Check current working directory
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").exists():
        return cwd
    
    # Strategy 4: If all fail, use current working directory as fallback and print warning
    print(f"⚠️  Warning: Could not find project root directory containing pyproject.toml")
    print(f"   Will use current working directory as API Keys storage location: {cwd}")
    print(f"   Tip: Run command from project root directory, or set environment variable WRAP_OPENAI_PROJECT_ROOT")
    return cwd


def _get_api_keys_dir() -> Path:
    """
    Get API Keys storage directory
    
    Priority:
    1. Custom path set by set_api_keys_path() (if set)
    2. Project root / .api_keys (default)
    
    Returns:
        API Keys storage directory path
    """
    global _custom_api_keys_path
    
    if _custom_api_keys_path is not None:
        # Use custom path
        api_keys_dir = Path(_custom_api_keys_path).resolve()
        api_keys_dir.mkdir(parents=True, exist_ok=True)
        return api_keys_dir
    else:
        # Use default: project root / .api_keys
        project_root = _get_project_root()
        api_keys_dir = project_root / ".api_keys"
        api_keys_dir.mkdir(parents=True, exist_ok=True)
        return api_keys_dir


def get_api_keys_dir() -> Path:
    """
    Get API Keys storage directory (public interface, for debugging and viewing)
    
    Returns:
        API Keys storage directory path
    """
    return _get_api_keys_dir()


def _get_api_keys_file() -> Path:
    """
    Get API Keys storage file path
    
    Priority:
    1. Custom file path set by set_api_keys_path() (if set)
    2. Custom directory / keys.json (if custom directory is set)
    3. Default: project root / .api_keys / keys.json
    
    Returns:
        API Keys storage file path
    """
    global _custom_api_keys_file
    
    if _custom_api_keys_file is not None:
        return _custom_api_keys_file
    else:
        return _get_api_keys_dir() / "keys.json"


class APIKeyManager:
    """API Key manager (supports persistent storage)"""
    
    def __init__(self):
        """Initialize API Key manager"""
        self.keys: Dict[str, Dict] = {}  # {api_key: {name, created_at, last_used_at}}
        self._load_keys()
    
    def _load_keys(self):
        """Load API Keys from file"""
        keys_file = _get_api_keys_file()
        if keys_file.exists():
            try:
                with open(keys_file, 'r', encoding='utf-8') as f:
                    self.keys = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️  Failed to load API Keys file: {e}, will use empty list")
                self.keys = {}
    
    def _save_keys(self):
        """Save API Keys to file"""
        keys_file = _get_api_keys_file()
        try:
            with open(keys_file, 'w', encoding='utf-8') as f:
                json.dump(self.keys, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"⚠️  Failed to save API Keys file: {e}")
    
    def generate_key(self, name: Optional[str] = None, prefix: str = "sk-") -> str:
        """
        Generate new API Key
        
        Args:
            name: API Key name (optional)
            prefix: API Key prefix
            
        Returns:
            Generated API Key
        """
        # Use hex encoding to generate secure random key
        key_suffix = secrets.token_hex(16)
        api_key = f"{prefix}{key_suffix}"
        
        # Store key information
        self.keys[api_key] = {
            "name": name or f"Key-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "created_at": datetime.now().isoformat(),
            "last_used_at": None,
        }
        self._save_keys()
        
        return api_key
    
    def validate_key(self, api_key: str) -> bool:
        """
        Validate if API Key is valid
        
        Args:
            api_key: API Key to validate
            
        Returns:
            True if valid, False otherwise
        """
        if api_key in self.keys:
            # Update last used time
            self.keys[api_key]["last_used_at"] = datetime.now().isoformat()
            self._save_keys()
            return True
        return False
    
    def revoke_key(self, api_key: str) -> bool:
        """
        Revoke API Key
        
        Args:
            api_key: API Key to revoke
            
        Returns:
            True if successfully revoked, False otherwise
        """
        if api_key in self.keys:
            del self.keys[api_key]
            self._save_keys()
            return True
        return False
    
    def list_keys(self) -> list:
        """
        List all API Keys (shows full keys)
        
        Returns:
            API Keys list
        """
        result = []
        for key, info in self.keys.items():
            result.append({
                "api_key": key,  # Show full key
                "name": info["name"],
                "created_at": info["created_at"],
                "last_used_at": info.get("last_used_at"),
            })
        return result
    
    def get_key_info(self, api_key: str) -> Optional[Dict]:
        """
        Get API Key detailed information
        
        Args:
            api_key: API Key
            
        Returns:
            API Key information dictionary, None if not exists
        """
        if api_key in self.keys:
            info = self.keys[api_key].copy()
            # For compatibility, keep key_preview field (shows full key)
            info["key_preview"] = api_key
            info["api_key"] = api_key
            return info
        return None


# Global API Key manager instance (supports persistent storage)
_api_key_manager: Optional[APIKeyManager] = None


def _reload_api_key_manager():
    """
    Reload API Key manager (useful when storage path changes)
    This will reload keys from the new storage location
    """
    global _api_key_manager
    _api_key_manager = APIKeyManager()


def get_api_key_manager() -> APIKeyManager:
    """Get global API Key manager instance"""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager


def set_api_keys_path(path: str | Path, silent: bool = False):
    """
    Set custom API Keys storage path
    
    This will override the default project root/.api_keys location.
    The path can be either a directory (keys.json will be created inside) or a file path.
    
    Args:
        path: Custom storage path (directory or file path)
              - If directory: keys.json will be stored in this directory
              - If file: keys will be stored at this exact path
        silent: If True, suppress output messages (default: False)
    
    Raises:
        ValueError: If path is invalid or cannot be created
    
    Examples:
        # Set to a directory (keys.json will be created inside)
        set_api_keys_path("/path/to/my_keys")
        
        # Set to a specific file path
        set_api_keys_path("/path/to/my_keys.json")
        
        # Set silently (for CLI usage)
        set_api_keys_path("/path/to/my_keys", silent=True)
    """
    global _custom_api_keys_path, _custom_api_keys_file
    
    path_obj = Path(path).resolve()
    
    # Check if path has .json extension or exists as a file
    # This helps distinguish between directory and file intentions
    if path_obj.suffix.lower() == '.json' or (path_obj.exists() and path_obj.is_file()):
        # Treat as file path
        _custom_api_keys_file = path_obj
        _custom_api_keys_path = None  # Clear directory path
        # Ensure parent directory exists
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        if not silent:
            print(f"✅ API Keys storage file set to: {_custom_api_keys_file}")
    else:
        # Treat as directory (or create new directory)
        _custom_api_keys_file = None  # Clear file path
        try:
            path_obj.mkdir(parents=True, exist_ok=True)
            _custom_api_keys_path = path_obj
            if not silent:
                print(f"✅ API Keys storage directory set to: {_custom_api_keys_path}")
                print(f"   File: keys.json")
        except Exception as e:
            raise ValueError(f"Failed to create API Keys storage directory at {path_obj}: {e}")
    
    # Reload API Key manager to use new path
    _reload_api_key_manager()
    if not silent:
        print(f"✅ API Key manager reloaded with new storage path")


def get_api_keys_path() -> Path:
    """
    Get current API Keys storage file path
    
    Returns:
        Current API Keys storage file path
    """
    return _get_api_keys_file()


