"""
Environment variable loading utilities.
"""

import os
from typing import Optional


def load_env_variables(env_file: str = ".env", verbose: bool = False) -> bool:
    """
    Load environment variables from a .env file.
    
    Args:
        env_file: Path to the .env file (default: ".env")
        verbose: Whether to print debug information
        
    Returns:
        True if environment variables were loaded successfully, False otherwise
    """
    try:
        # Try using python-dotenv if available
        from dotenv import load_dotenv
        if verbose:
            print(f"Loading environment variables from {env_file} using python-dotenv")
        load_dotenv(env_file)
        return True
    except ImportError:
        # Fallback: manual parsing if python-dotenv is not available
        if verbose:
            print(f"python-dotenv not available, manually parsing {env_file}")
        return _load_env_manually(env_file, verbose)


def _load_env_manually(env_file: str, verbose: bool = False) -> bool:
    """
    Manually parse and load environment variables from a .env file.
    
    Args:
        env_file: Path to the .env file
        verbose: Whether to print debug information
        
    Returns:
        True if file was loaded successfully, False otherwise
    """
    if not os.path.exists(env_file):
        if verbose:
            print(f"Environment file {env_file} not found")
        return False
    
    try:
        with open(env_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse key=value pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    # Set environment variable
                    os.environ[key] = value
                    
                    if verbose:
                        print(f"Loaded {key}={'*' * (len(value) - 4) + value[-4:] if len(value) > 4 else '***'}")
                else:
                    if verbose:
                        print(f"Warning: Invalid line format at line {line_num}: {line}")
        
        if verbose:
            print(f"Successfully loaded environment variables from {env_file}")
        return True
        
    except Exception as e:
        if verbose:
            print(f"Error loading environment file {env_file}: {e}")
        return False


def get_env_var(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """
    Get an environment variable with optional default and required validation.
    
    Args:
        key: Environment variable name
        default: Default value if variable is not set
        required: Whether the variable is required (raises ValueError if missing)
        
    Returns:
        Environment variable value or default
        
    Raises:
        ValueError: If required variable is not set
    """
    value = os.getenv(key, default)
    
    if required and value is None:
        raise ValueError(f"Required environment variable {key} is not set")
    
    return value


def check_required_env_vars(required_vars: list[str], verbose: bool = False) -> bool:
    """
    Check if all required environment variables are set.
    
    Args:
        required_vars: List of required environment variable names
        verbose: Whether to print debug information
        
    Returns:
        True if all required variables are set, False otherwise
    """
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        if verbose:
            print(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    if verbose:
        print(f"All required environment variables are set: {', '.join(required_vars)}")
    return True
