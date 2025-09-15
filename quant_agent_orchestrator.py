"""
QuantAgent Orchestrator - Main coordination system for all agents
"""
import os
import sys
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import warnings

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import Config
from utils.data_fetcher import DataFetcher
from agents.indicator_agent import IndicatorAgent
from agents.pattern_agent import PatternAgent
from agents.trend_agent import TrendAgent
from agents.decision_agent import DecisionAgent


class QuantAgentOrchestrator:
    """
    Main orchestrator for the QuantAgent system.
    
    Coordinates data fetching and analysis across all specialized agents:
    - IndicatorAgent: Technical indicators
    - PatternAgent: Chart patterns and visualization
    - TrendAgent: Trend analysis
    - DecisionAgent: Final trading decision synthesis
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the QuantAgent orchestrator.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        # Load configuration
        self.config = Config(config_path)
        
        # Initialize data fetcher
        data_config = self.config.get_data_config()
        self.data_fetcher = DataFetcher(
            cache_enabled=data_config['cache_enabled'],
            cache_duration=data_config['cache_duration']
        )
        
        # Initialize agents
        self.indicator_agent = IndicatorAgent()
        self.pattern_agent = PatternAgent(chart_dir="charts")
        self.trend_agent = TrendAgent()
        
        # Initialize decision agent only if Groq API key is available
        self.decision_agent = None
        if self.config.get('groq_api_key'):
            try:
                llm_config = self.config.get_llm_config()
                self.decision_agent = DecisionAgent(
                    api_key=llm_config['api_key'],
                    model=llm_config['model']
                )
            except Exception as e:
                warnings.warn(f"Failed to initialize DecisionAgent: {e}")
        
        # Analysis cache
        self.last_analysis = None
    
    def analyze_symbol(self, symbol: str, period: str = "1y", interval: str = "1d") -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a trading symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'MSFT')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            
        Returns:
            Comprehensive analysis results from all agents
        """
        try:
            # Step 1: Fetch market data
            print(f"Fetching data for {symbol}...")
            data = self.data_fetcher.fetch_data(symbol, period, interval)
            
            if data is None or len(data) < 10:
                return {
                    'success': False,
                    'error': f'Insufficient data for {symbol}. Please check the symbol and try again.',
                    'symbol': symbol
                }
            
            print(f"Retrieved {len(data)} data points for {symbol}")
            
            # Step 2: Run technical indicator analysis
            print("Running technical indicator analysis...")
            indicator_analysis = self.indicator_agent.analyze(data)
            
            # Step 3: Run trend analysis
            print("Running trend analysis...")
            trend_analysis = self.trend_agent.analyze(data)
            
            # Step 4: Run pattern analysis (with indicator data for enhanced charts)
            print("Running pattern analysis and generating charts...")
            pattern_analysis = self.pattern_agent.analyze(
                data, 
                symbol=symbol, 
                indicators=indicator_analysis.get('indicators', {})
            )
            
            # Step 5: Run decision analysis (if available)
            decision_analysis = None
            if self.decision_agent:
                print("Running decision analysis...")
                try:
                    decision_analysis = self.decision_agent.analyze(
                        indicator_analysis=indicator_analysis,
                        pattern_analysis=pattern_analysis,
                        trend_analysis=trend_analysis,
                        symbol=symbol
                    )
                except Exception as e:
                    warnings.warn(f"Decision analysis failed: {e}")
                    decision_analysis = {
                        'decision': 'HOLD',
                        'confidence': 0.0,
                        'justification': f'Decision analysis unavailable: {str(e)}',
                        'risk_level': 'HIGH'
                    }
            else:
                decision_analysis = {
                    'decision': 'HOLD',
                    'confidence': 0.0,
                    'justification': 'Groq API key not configured. Please set GROQ_API_KEY environment variable.',
                    'risk_level': 'HIGH'
                }
            
            # Step 6: Compile comprehensive results
            analysis_result = {
                'success': True,
                'symbol': symbol,
                'period': period,
                'interval': interval,
                'data_points': len(data),
                'current_price': float(data['Close'].iloc[-1]),
                'price_change': float(((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100) if len(data) > 1 else 0,
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                
                # Agent results
                'indicator_analysis': indicator_analysis,
                'pattern_analysis': pattern_analysis,
                'trend_analysis': trend_analysis,
                'decision_analysis': decision_analysis,
                
                # Summary
                'summary': self._generate_summary(
                    indicator_analysis, pattern_analysis, trend_analysis, decision_analysis
                )
            }
            
            # Cache the analysis
            self.last_analysis = analysis_result
            
            print("✓ Analysis completed successfully")
            return analysis_result
            
        except Exception as e:
            error_result = {
                'success': False,
                'error': f'Analysis failed: {str(e)}',
                'symbol': symbol
            }
            print(f"✗ Analysis failed: {str(e)}")
            return error_result
    
    def _generate_summary(self, indicator_analysis: Dict, pattern_analysis: Dict,
                         trend_analysis: Dict, decision_analysis: Dict) -> Dict[str, Any]:
        """Generate a comprehensive summary of all analyses."""
        
        # Extract key insights
        indicator_forecast = indicator_analysis.get('forecast', 'Unknown')
        trend_direction = trend_analysis.get('direction', 'Unknown')
        trend_confidence = trend_analysis.get('confidence', 0)
        decision = decision_analysis.get('decision', 'HOLD')
        decision_confidence = decision_analysis.get('confidence', 0)
        
        # Determine overall sentiment
        bullish_signals = 0
        bearish_signals = 0
        
        if indicator_forecast == 'Bullish':
            bullish_signals += 1
        elif indicator_forecast == 'Bearish':
            bearish_signals += 1
            
        if trend_direction == 'Bullish':
            bullish_signals += 1
        elif trend_direction == 'Bearish':
            bearish_signals += 1
            
        if decision == 'LONG':
            bullish_signals += 1
        elif decision == 'SHORT':
            bearish_signals += 1
        
        if bullish_signals > bearish_signals:
            overall_sentiment = 'Bullish'
        elif bearish_signals > bullish_signals:
            overall_sentiment = 'Bearish'
        else:
            overall_sentiment = 'Neutral'
        
        # Calculate overall confidence
        overall_confidence = (trend_confidence + decision_confidence) / 2
        
        return {
            'overall_sentiment': overall_sentiment,
            'overall_confidence': overall_confidence,
            'indicator_forecast': indicator_forecast,
            'trend_direction': trend_direction,
            'final_decision': decision,
            'key_insights': [
                f"Technical indicators suggest {indicator_forecast.lower()} momentum",
                f"Trend analysis shows {trend_direction.lower()} direction with {trend_confidence:.1%} confidence",
                f"Final recommendation: {decision} with {decision_confidence:.1%} confidence"
            ],
            'risk_assessment': decision_analysis.get('risk_level', 'UNKNOWN')
        }
    
    def get_last_analysis(self) -> Optional[Dict[str, Any]]:
        """Get the results of the last analysis."""
        return self.last_analysis
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol can be analyzed.
        
        Args:
            symbol: Trading symbol to validate
            
        Returns:
            True if symbol is valid, False otherwise
        """
        return self.data_fetcher.validate_symbol(symbol)
    
    def get_available_periods(self) -> list:
        """Get list of available data periods."""
        return self.data_fetcher.get_available_periods()
    
    def get_available_intervals(self) -> list:
        """Get list of available data intervals."""
        return self.data_fetcher.get_available_intervals()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get the status of all system components."""
        return {
            'data_fetcher': {
                'available': True,
                'cache_info': self.data_fetcher.get_cache_info()
            },
            'indicator_agent': {
                'available': True,
                'last_analysis': self.indicator_agent.get_last_analysis() is not None
            },
            'pattern_agent': {
                'available': True,
                'last_analysis': self.pattern_agent.get_last_analysis() is not None
            },
            'trend_agent': {
                'available': True,
                'last_analysis': self.trend_agent.get_last_analysis() is not None
            },
            'decision_agent': {
                'available': self.decision_agent is not None,
                'last_analysis': self.decision_agent.get_last_analysis() is not None if self.decision_agent else False
            },
            'config': {
                'api_keys_configured': self.config.validate_api_keys(),
                'missing_keys': self.config.get_missing_keys()
            }
        }
    
    def clear_cache(self):
        """Clear all cached data."""
        self.data_fetcher.clear_cache()
        self.last_analysis = None
        print("Cache cleared successfully")

