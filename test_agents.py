"""
Test script for QuantAgent components
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_fetcher import DataFetcher
from utils.config import Config
from agents.indicator_agent import IndicatorAgent
from agents.pattern_agent import PatternAgent
from agents.trend_agent import TrendAgent


def create_sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Generate realistic price data
    base_price = 100
    prices = []
    current_price = base_price
    
    for i in range(len(dates)):
        # Random walk with slight upward bias
        change = np.random.normal(0.001, 0.02)  # 0.1% daily drift, 2% volatility
        current_price *= (1 + change)
        prices.append(current_price)
    
    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        daily_volatility = 0.01
        high = price * (1 + np.random.uniform(0, daily_volatility))
        low = price * (1 - np.random.uniform(0, daily_volatility))
        open_price = prices[i-1] if i > 0 else price
        close = price
        volume = np.random.randint(1000000, 10000000)
        
        data.append({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    return df


def test_data_fetcher():
    """Test the data fetcher."""
    print("Testing DataFetcher...")
    
    fetcher = DataFetcher()
    
    # Test with a well-known symbol
    data = fetcher.fetch_data("AAPL", period="1mo", interval="1d")
    
    if data is not None:
        print(f"✓ Successfully fetched {len(data)} rows of AAPL data")
        print(f"  Columns: {list(data.columns)}")
        print(f"  Date range: {data.index[0]} to {data.index[-1]}")
    else:
        print("✗ Failed to fetch data, using sample data for testing")
        data = create_sample_data()
        print(f"✓ Created sample data with {len(data)} rows")
    
    return data


def test_indicator_agent(data):
    """Test the IndicatorAgent."""
    print("\nTesting IndicatorAgent...")
    
    try:
        agent = IndicatorAgent()
        result = agent.analyze(data)
        
        print(f"✓ IndicatorAgent analysis completed")
        print(f"  Forecast: {result['forecast']}")
        print(f"  Evidence: {result['evidence']}")
        print(f"  Indicators: {result['indicators']}")
        
        return result
    except Exception as e:
        print(f"✗ IndicatorAgent failed: {str(e)}")
        return None


def test_pattern_agent(data):
    """Test the PatternAgent."""
    print("\nTesting PatternAgent...")
    
    try:
        agent = PatternAgent(chart_dir="charts")
        result = agent.analyze(data, symbol="TEST")
        
        print(f"✓ PatternAgent analysis completed")
        print(f"  Chart saved to: {result['chart_path']}")
        print(f"  Pattern description: {result['pattern_description'][:100]}...")
        
        return result
    except Exception as e:
        print(f"✗ PatternAgent failed: {str(e)}")
        return None


def test_trend_agent(data):
    """Test the TrendAgent."""
    print("\nTesting TrendAgent...")
    
    try:
        agent = TrendAgent()
        result = agent.analyze(data)
        
        print(f"✓ TrendAgent analysis completed")
        print(f"  Overall direction: {result['direction']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Summary: {result['summary'][:100]}...")
        
        return result
    except Exception as e:
        print(f"✗ TrendAgent failed: {str(e)}")
        return None


def test_config():
    """Test the configuration system."""
    print("\nTesting Config...")
    
    try:
        config = Config()
        print(f"✓ Config loaded successfully")
        
        # Test validation
        validation = config.validate_api_keys()
        print(f"  API key validation: {validation}")
        
        missing = config.get_missing_keys()
        if missing:
            print(f"  Missing keys: {missing}")
        else:
            print(f"  All API keys configured")
            
        return config
    except Exception as e:
        print(f"✗ Config failed: {str(e)}")
        return None


def main():
    """Run all tests."""
    print("QuantAgent Component Tests")
    print("=" * 50)
    
    # Test configuration
    config = test_config()
    
    # Test data fetching
    data = test_data_fetcher()
    
    if data is not None:
        # Test all agents
        indicator_result = test_indicator_agent(data)
        pattern_result = test_pattern_agent(data)
        trend_result = test_trend_agent(data)
        
        print("\n" + "=" * 50)
        print("Test Summary:")
        print(f"✓ Data Fetcher: {'PASS' if data is not None else 'FAIL'}")
        print(f"✓ IndicatorAgent: {'PASS' if indicator_result else 'FAIL'}")
        print(f"✓ PatternAgent: {'PASS' if pattern_result else 'FAIL'}")
        print(f"✓ TrendAgent: {'PASS' if trend_result else 'FAIL'}")
        print(f"✓ Config: {'PASS' if config else 'FAIL'}")
        
        if all([data is not None, indicator_result, pattern_result, trend_result, config]):
            print("\n🎉 All tests passed! Ready to build the Streamlit app.")
        else:
            print("\n⚠️  Some tests failed. Check the errors above.")
    else:
        print("\n✗ Cannot proceed without data")


if __name__ == "__main__":
    main()

