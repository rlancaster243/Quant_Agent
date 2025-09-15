"""
PatternAgent - Chart Pattern Recognition
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np
from typing import Dict, Any, List, Tuple
import os
import sys
from .base_agent import BaseAgent

# Add utils to path for importing chart enhancer
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from chart_enhancer import ChartEnhancer


class PatternAgent(BaseAgent):
    """
    PatternAgent is responsible for generating candlestick charts and identifying visual patterns.
    
    This agent creates visual representations of price data and identifies:
    - Support and resistance levels
    - Chart patterns (triangles, channels, etc.)
    - Candlestick patterns
    """
    
    def __init__(self, chart_dir: str = "charts"):
        super().__init__("PatternAgent")
        self.chart_dir = chart_dir
        os.makedirs(chart_dir, exist_ok=True)
        self.chart_enhancer = ChartEnhancer()
        
    def analyze(self, data: pd.DataFrame, symbol: str = "UNKNOWN", 
                indicators: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate candlestick chart and analyze visual patterns.
        
        Args:
            data (pd.DataFrame): Market data with OHLCV columns
            symbol (str): Symbol name for chart title
            indicators (Dict): Technical indicators data for enhanced charting
            
        Returns:
            Dict[str, Any]: Analysis results including chart path and pattern description
        """
        if not self.validate_data(data):
            raise ValueError("Invalid data format. Required columns: Open, High, Low, Close, Volume")
        
        # Analyze patterns
        patterns = self._analyze_patterns(data)
        
        # Calculate support and resistance
        support_resistance = self._calculate_support_resistance(data)
        patterns['support_resistance'] = support_resistance
        
        # Generate enhanced candlestick chart
        chart_path = self._generate_enhanced_chart(data, symbol, indicators or {}, patterns)
        
        # Generate pattern description
        pattern_description = self._generate_pattern_description(data, patterns, support_resistance)
        
        analysis_result = {
            'agent': self.name,
            'chart_path': chart_path,
            'patterns': patterns,
            'support_resistance': support_resistance,
            'pattern_description': pattern_description,
            'visual_summary': self._get_visual_summary(data, patterns),
            'chart_analysis': self._get_chart_analysis(data, patterns)
        }
        
        self.last_analysis = analysis_result
        return analysis_result
    
    def _generate_enhanced_chart(self, data: pd.DataFrame, symbol: str,
                               indicators: Dict[str, Any], patterns: Dict[str, Any]) -> str:
        """Generate an enhanced candlestick chart using ChartEnhancer."""
        chart_filename = f"{symbol.replace('/', '_')}_enhanced_chart.png"
        chart_path = os.path.join(self.chart_dir, chart_filename)
        
        try:
            # Use enhanced charting if indicators are available
            if indicators and any(indicators.values()):
                return self.chart_enhancer.create_enhanced_candlestick_chart(
                    data, symbol, indicators, patterns, chart_path
                )
            else:
                # Fallback to basic chart
                return self._generate_basic_candlestick_chart(data, symbol, chart_path)
        except Exception as e:
            # Fallback to basic chart if enhanced fails
            print(f"Enhanced chart generation failed: {e}, falling back to basic chart")
            return self._generate_basic_candlestick_chart(data, symbol, chart_path)
    
    def _generate_basic_candlestick_chart(self, data: pd.DataFrame, symbol: str, chart_path: str) -> str:
        """Generate a basic candlestick chart (fallback method)."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Prepare data
        data_copy = data.copy()
        if not isinstance(data_copy.index, pd.DatetimeIndex):
            data_copy.index = pd.to_datetime(data_copy.index)
        
        # Plot candlesticks
        self._plot_candlesticks(ax1, data_copy)
        
        # Add moving averages
        self._add_moving_averages(ax1, data_copy)
        
        # Add support/resistance lines
        support_resistance = self._calculate_support_resistance(data_copy)
        self._add_support_resistance_lines(ax1, data_copy, support_resistance)
        
        # Plot volume
        self._plot_volume(ax2, data_copy)
        
        # Formatting
        ax1.set_title(f'{symbol} - Candlestick Chart with Technical Analysis', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    def _plot_candlesticks(self, ax, data: pd.DataFrame):
        """Plot candlestick chart."""
        up = data[data['Close'] >= data['Open']]
        down = data[data['Close'] < data['Open']]
        
        # Plot up candles (green)
        ax.bar(up.index, up['Close'] - up['Open'], bottom=up['Open'], 
               color='green', alpha=0.8, width=0.8)
        ax.bar(up.index, up['High'] - up['Close'], bottom=up['Close'], 
               color='green', alpha=0.4, width=0.1)
        ax.bar(up.index, up['Open'] - up['Low'], bottom=up['Low'], 
               color='green', alpha=0.4, width=0.1)
        
        # Plot down candles (red)
        ax.bar(down.index, down['Open'] - down['Close'], bottom=down['Close'], 
               color='red', alpha=0.8, width=0.8)
        ax.bar(down.index, down['High'] - down['Open'], bottom=down['Open'], 
               color='red', alpha=0.4, width=0.1)
        ax.bar(down.index, down['Close'] - down['Low'], bottom=down['Low'], 
               color='red', alpha=0.4, width=0.1)
    
    def _add_moving_averages(self, ax, data: pd.DataFrame):
        """Add moving averages to the chart."""
        # 20-period and 50-period moving averages
        ma20 = data['Close'].rolling(window=20).mean()
        ma50 = data['Close'].rolling(window=50).mean()
        
        ax.plot(data.index, ma20, label='MA20', color='blue', linewidth=1)
        ax.plot(data.index, ma50, label='MA50', color='orange', linewidth=1)
    
    def _add_support_resistance_lines(self, ax, data: pd.DataFrame, support_resistance: Dict):
        """Add support and resistance lines to the chart."""
        if support_resistance['support']:
            ax.axhline(y=support_resistance['support'], color='green', 
                      linestyle='--', alpha=0.7, label=f"Support: {support_resistance['support']:.2f}")
        
        if support_resistance['resistance']:
            ax.axhline(y=support_resistance['resistance'], color='red', 
                      linestyle='--', alpha=0.7, label=f"Resistance: {support_resistance['resistance']:.2f}")
    
    def _plot_volume(self, ax, data: pd.DataFrame):
        """Plot volume bars."""
        colors = ['green' if close >= open_price else 'red' 
                 for close, open_price in zip(data['Close'], data['Open'])]
        ax.bar(data.index, data['Volume'], color=colors, alpha=0.6)
    
    def _analyze_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze chart patterns."""
        patterns = {
            'trend': self._identify_trend(data),
            'volatility': self._calculate_volatility(data),
            'price_action': self._analyze_price_action(data)
        }
        return patterns
    
    def _identify_trend(self, data: pd.DataFrame) -> str:
        """Identify the overall trend."""
        if len(data) < 20:
            return "Insufficient data"
        
        recent_high = data['High'].tail(10).max()
        recent_low = data['Low'].tail(10).min()
        earlier_high = data['High'].head(10).max()
        earlier_low = data['Low'].head(10).min()
        
        if recent_high > earlier_high and recent_low > earlier_low:
            return "Uptrend"
        elif recent_high < earlier_high and recent_low < earlier_low:
            return "Downtrend"
        else:
            return "Sideways"
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate price volatility."""
        returns = data['Close'].pct_change().dropna()
        return returns.std() * 100  # Convert to percentage
    
    def _analyze_price_action(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze recent price action."""
        if len(data) < 5:
            return {"pattern": "Insufficient data"}
        
        recent_data = data.tail(5)
        closes = recent_data['Close'].values
        
        # Simple pattern recognition
        if all(closes[i] > closes[i-1] for i in range(1, len(closes))):
            pattern = "Strong Bullish"
        elif all(closes[i] < closes[i-1] for i in range(1, len(closes))):
            pattern = "Strong Bearish"
        elif closes[-1] > closes[0]:
            pattern = "Bullish"
        elif closes[-1] < closes[0]:
            pattern = "Bearish"
        else:
            pattern = "Neutral"
        
        return {
            "pattern": pattern,
            "price_change": ((closes[-1] - closes[0]) / closes[0]) * 100
        }
    
    def _calculate_support_resistance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate support and resistance levels."""
        if len(data) < 20:
            return {"support": None, "resistance": None}
        
        # Simple support/resistance calculation using recent highs and lows
        recent_data = data.tail(20)
        
        # Support: lowest low in recent period
        support = recent_data['Low'].min()
        
        # Resistance: highest high in recent period
        resistance = recent_data['High'].max()
        
        return {"support": support, "resistance": resistance}
    
    def _generate_pattern_description(self, data: pd.DataFrame, patterns: Dict, 
                                    support_resistance: Dict) -> str:
        """Generate a textual description of the chart patterns."""
        description = f"Chart Pattern Analysis:\\n"
        description += f"- Overall Trend: {patterns['trend']}\\n"
        description += f"- Volatility: {patterns['volatility']:.2f}%\\n"
        description += f"- Recent Price Action: {patterns['price_action']['pattern']}\\n"
        
        if support_resistance['support']:
            description += f"- Support Level: {support_resistance['support']:.2f}\\n"
        if support_resistance['resistance']:
            description += f"- Resistance Level: {support_resistance['resistance']:.2f}\\n"
        
        current_price = data['Close'].iloc[-1]
        if support_resistance['support'] and support_resistance['resistance']:
            range_position = ((current_price - support_resistance['support']) / 
                            (support_resistance['resistance'] - support_resistance['support'])) * 100
            description += f"- Current price is {range_position:.1f}% within the support-resistance range"
        
        return description
    
    def _get_visual_summary(self, data: pd.DataFrame, patterns: Dict) -> str:
        """Generate a visual summary for LLM consumption."""
        summary = f"Visual Chart Summary: "
        summary += f"The chart shows a {patterns['trend'].lower()} pattern with "
        summary += f"{patterns['volatility']:.1f}% volatility. "
        summary += f"Recent price action indicates {patterns['price_action']['pattern'].lower()} momentum."
        
        return summary


    
    def _get_chart_analysis(self, data: pd.DataFrame, patterns: Dict[str, Any]) -> str:
        """Generate detailed chart analysis for LLM consumption."""
        analysis = "Detailed Chart Analysis: "
        
        # Price action analysis
        price_action = patterns.get('price_action', {})
        if price_action:
            pattern_type = price_action.get('pattern', 'Neutral')
            price_change = price_action.get('price_change', 0)
            analysis += f"Recent price action shows {pattern_type.lower()} momentum with {price_change:.2f}% change. "
        
        # Trend analysis
        trend = patterns.get('trend', 'Unknown')
        analysis += f"Overall trend direction is {trend.lower()}. "
        
        # Volatility analysis
        volatility = patterns.get('volatility', 0)
        if volatility > 3:
            vol_desc = "high"
        elif volatility > 1.5:
            vol_desc = "moderate"
        else:
            vol_desc = "low"
        analysis += f"Market volatility is {vol_desc} at {volatility:.2f}%. "
        
        # Support/resistance analysis
        sr = patterns.get('support_resistance', {})
        if sr.get('support') and sr.get('resistance'):
            current_price = data['Close'].iloc[-1]
            support_dist = ((current_price - sr['support']) / sr['support']) * 100
            resistance_dist = ((sr['resistance'] - current_price) / current_price) * 100
            analysis += f"Current price is {support_dist:.1f}% above support and {resistance_dist:.1f}% below resistance."
        
        return analysis

