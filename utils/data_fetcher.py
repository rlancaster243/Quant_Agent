"""
Data Fetcher for QuantAgent using OpenBB and yfinance as fallback
"""
import pandas as pd
import yfinance as yf
from typing import Optional, Dict, Any
import warnings
from datetime import datetime, timedelta
import time


class DataFetcher:
    """
    Data fetcher that uses OpenBB as primary source and yfinance as fallback.
    Handles data caching and error recovery.
    """
    
    def __init__(self, cache_enabled: bool = True, cache_duration: int = 300):
        """
        Initialize the data fetcher.
        
        Args:
            cache_enabled: Whether to enable data caching
            cache_duration: Cache duration in seconds
        """
        self.cache_enabled = cache_enabled
        self.cache_duration = cache_duration
        self._cache = {}
        
        # Try to import OpenBB
        self.openbb_available = False
        try:
            import openbb
            self.obb = openbb
            self.openbb_available = True
        except ImportError:
            warnings.warn("OpenBB not available, using yfinance only")
    
    def fetch_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a given symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        # Check cache first
        cache_key = f"{symbol}_{period}_{interval}"
        if self.cache_enabled and self._is_cache_valid(cache_key):
            return self._cache[cache_key]['data']
        
        # Try OpenBB first
        if self.openbb_available:
            data = self._fetch_with_openbb(symbol, period, interval)
            if data is not None:
                self._update_cache(cache_key, data)
                return data
        
        # Fallback to yfinance
        data = self._fetch_with_yfinance(symbol, period, interval)
        if data is not None:
            self._update_cache(cache_key, data)
            return data
        
        return None
    
    def _fetch_with_openbb(self, symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch data using OpenBB."""
        try:
            # Convert period to start_date and end_date for OpenBB
            end_date = datetime.now()
            
            period_mapping = {
                '1d': timedelta(days=1),
                '5d': timedelta(days=5),
                '1mo': timedelta(days=30),
                '3mo': timedelta(days=90),
                '6mo': timedelta(days=180),
                '1y': timedelta(days=365),
                '2y': timedelta(days=730),
                '5y': timedelta(days=1825),
                '10y': timedelta(days=3650),
                'ytd': timedelta(days=(datetime.now() - datetime(datetime.now().year, 1, 1)).days),
                'max': timedelta(days=7300)  # ~20 years
            }
            
            start_date = end_date - period_mapping.get(period, timedelta(days=365))
            
            # Fetch data using OpenBB
            data = self.obb.equity.price.historical(
                symbol=symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                interval=interval
            )
            
            if data is not None and not data.empty:
                # Standardize column names
                data = self._standardize_columns(data)
                return data
            
        except Exception as e:
            warnings.warn(f"OpenBB fetch failed for {symbol}: {str(e)}")
        
        return None
    
    def _fetch_with_yfinance(self, symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch data using yfinance as fallback."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data is not None and not data.empty:
                # Standardize column names
                data = self._standardize_columns(data)
                return data
            
        except Exception as e:
            warnings.warn(f"yfinance fetch failed for {symbol}: {str(e)}")
        
        return None
    
    def _standardize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to OHLCV format."""
        # Common column name mappings
        column_mapping = {
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'adj close': 'Adj Close',
            'adjusted_close': 'Adj Close'
        }
        
        # Rename columns to standard format
        data.columns = [column_mapping.get(col.lower(), col) for col in data.columns]
        
        # Ensure we have the required OHLCV columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                if col == 'Volume' and 'volume' in data.columns.str.lower():
                    # Handle case-sensitive volume column
                    volume_col = [c for c in data.columns if c.lower() == 'volume'][0]
                    data[col] = data[volume_col]
                else:
                    warnings.warn(f"Missing required column: {col}")
                    return None
        
        # Remove any duplicate columns and keep only OHLCV + Adj Close
        keep_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if 'Adj Close' in data.columns:
            keep_columns.append('Adj Close')
        
        data = data[keep_columns]
        
        # Ensure numeric data types
        for col in keep_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove rows with NaN values
        data = data.dropna()
        
        return data
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache:
            return False
        
        cache_time = self._cache[cache_key]['timestamp']
        return (time.time() - cache_time) < self.cache_duration
    
    def _update_cache(self, cache_key: str, data: pd.DataFrame) -> None:
        """Update cache with new data."""
        if self.cache_enabled:
            self._cache[cache_key] = {
                'data': data.copy(),
                'timestamp': time.time()
            }
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached data."""
        return {
            'enabled': self.cache_enabled,
            'duration': self.cache_duration,
            'cached_items': len(self._cache),
            'cache_keys': list(self._cache.keys())
        }
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol exists and can be fetched.
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            True if symbol is valid, False otherwise
        """
        try:
            data = self.fetch_data(symbol, period="5d", interval="1d")
            return data is not None and len(data) > 0
        except:
            return False
    
    def get_available_intervals(self) -> list:
        """Get list of available data intervals."""
        return ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
    
    def get_available_periods(self) -> list:
        """Get list of available data periods."""
        return ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']

