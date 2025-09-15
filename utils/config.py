"""
Configuration Management for QuantAgent
"""
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class Config:
    """
    Configuration manager for QuantAgent application.
    Handles environment variables, API keys, and application settings.
    """
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            env_file: Path to .env file (optional)
        """
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()  # Load from default .env file
        
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        return {
            # API Keys
            'groq_api_key': os.getenv('GROQ_API_KEY'),
            'openbb_pat': os.getenv('OPENBB_PAT'),
            
            # Application Settings
            'app_title': os.getenv('APP_TITLE', 'QuantAgent - AI Trading Assistant'),
            'default_symbol': os.getenv('DEFAULT_SYMBOL', 'AAPL'),
            'default_timeframe': os.getenv('DEFAULT_TIMEFRAME', '1d'),
            'default_period': os.getenv('DEFAULT_PERIOD', '1y'),
            
            # Chart Settings
            'chart_width': int(os.getenv('CHART_WIDTH', '12')),
            'chart_height': int(os.getenv('CHART_HEIGHT', '8')),
            'chart_dpi': int(os.getenv('CHART_DPI', '300')),
            
            # LLM Settings
            'llm_model': os.getenv('LLM_MODEL', 'llama3-8b-8192'),
            'llm_temperature': float(os.getenv('LLM_TEMPERATURE', '0.1')),
            'llm_max_tokens': int(os.getenv('LLM_MAX_TOKENS', '1000')),
            
            # Data Settings
            'data_cache_enabled': os.getenv('DATA_CACHE_ENABLED', 'true').lower() == 'true',
            'data_cache_duration': int(os.getenv('DATA_CACHE_DURATION', '300')),  # 5 minutes
            
            # Risk Management
            'max_position_size': float(os.getenv('MAX_POSITION_SIZE', '1.0')),
            'default_stop_loss': float(os.getenv('DEFAULT_STOP_LOSS', '0.02')),  # 2%
            'default_take_profit': float(os.getenv('DEFAULT_TAKE_PROFIT', '0.04')),  # 4%
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self._config[key] = value
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """
        Validate that required API keys are present.
        
        Returns:
            Dict with validation status for each API key
        """
        return {
            'groq_api_key': bool(self._config.get('groq_api_key')),
            'openbb_pat': bool(self._config.get('openbb_pat'))
        }
    
    def get_missing_keys(self) -> list:
        """
        Get list of missing required API keys.
        
        Returns:
            List of missing API key names
        """
        validation = self.validate_api_keys()
        return [key for key, valid in validation.items() if not valid]
    
    def is_configured(self) -> bool:
        """
        Check if all required configuration is present.
        
        Returns:
            True if fully configured, False otherwise
        """
        return len(self.get_missing_keys()) == 0
    
    def get_chart_config(self) -> Dict[str, Any]:
        """Get chart-specific configuration."""
        return {
            'width': self.get('chart_width'),
            'height': self.get('chart_height'),
            'dpi': self.get('chart_dpi')
        }
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM-specific configuration."""
        return {
            'model': self.get('llm_model'),
            'temperature': self.get('llm_temperature'),
            'max_tokens': self.get('llm_max_tokens'),
            'api_key': self.get('groq_api_key')
        }
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data-specific configuration."""
        return {
            'cache_enabled': self.get('data_cache_enabled'),
            'cache_duration': self.get('data_cache_duration'),
            'openbb_pat': self.get('openbb_pat')
        }
    
    def __str__(self) -> str:
        """String representation of configuration (without sensitive data)."""
        safe_config = {k: v for k, v in self._config.items() 
                      if 'key' not in k.lower() and 'token' not in k.lower()}
        return f"Config({safe_config})"

