"""
Chart Enhancement Utilities for QuantAgent
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import seaborn as sns


class ChartEnhancer:
    """
    Enhanced charting utilities for better visualization and pattern recognition.
    """
    
    def __init__(self):
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Chart styling
        self.colors = {
            'bullish': '#2E8B57',  # Sea Green
            'bearish': '#DC143C',  # Crimson
            'neutral': '#708090',  # Slate Gray
            'support': '#32CD32',  # Lime Green
            'resistance': '#FF6347',  # Tomato
            'ma_short': '#4169E1',  # Royal Blue
            'ma_long': '#FF8C00',   # Dark Orange
            'volume_up': '#90EE90', # Light Green
            'volume_down': '#FFB6C1' # Light Pink
        }
        
    def create_enhanced_candlestick_chart(self, data: pd.DataFrame, symbol: str,
                                        indicators: Dict[str, Any],
                                        patterns: Dict[str, Any],
                                        save_path: str) -> str:
        """
        Create an enhanced candlestick chart with technical analysis overlays.
        
        Args:
            data: OHLCV data
            symbol: Trading symbol
            indicators: Technical indicators data
            patterns: Pattern analysis data
            save_path: Path to save the chart
            
        Returns:
            Path to the saved chart
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Create subplots
        gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
        ax_price = fig.add_subplot(gs[0])
        ax_volume = fig.add_subplot(gs[1])
        ax_rsi = fig.add_subplot(gs[2])
        ax_macd = fig.add_subplot(gs[3])
        
        # Plot main candlestick chart
        self._plot_enhanced_candlesticks(ax_price, data, symbol)
        
        # Add technical overlays
        self._add_moving_averages(ax_price, data)
        self._add_bollinger_bands(ax_price, data)
        self._add_support_resistance(ax_price, data, patterns.get('support_resistance', {}))
        
        # Plot volume with color coding
        self._plot_enhanced_volume(ax_volume, data)
        
        # Plot RSI
        self._plot_rsi(ax_rsi, data, indicators)
        
        # Plot MACD
        self._plot_macd(ax_macd, data, indicators)
        
        # Add annotations
        self._add_pattern_annotations(ax_price, data, patterns)
        
        # Final formatting
        self._format_chart(fig, ax_price, ax_volume, ax_rsi, ax_macd, symbol)
        
        # Save chart
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path
    
    def _plot_enhanced_candlesticks(self, ax, data: pd.DataFrame, symbol: str):
        """Plot enhanced candlestick chart."""
        # Separate up and down candles
        up = data[data['Close'] >= data['Open']]
        down = data[data['Close'] < data['Open']]
        
        # Calculate candle widths
        width = 0.8
        width2 = 0.1
        
        # Plot up candles (green)
        ax.bar(up.index, up['Close'] - up['Open'], bottom=up['Open'], 
               color=self.colors['bullish'], alpha=0.8, width=width)
        ax.bar(up.index, up['High'] - up['Close'], bottom=up['Close'], 
               color=self.colors['bullish'], alpha=0.6, width=width2)
        ax.bar(up.index, up['Open'] - up['Low'], bottom=up['Low'], 
               color=self.colors['bullish'], alpha=0.6, width=width2)
        
        # Plot down candles (red)
        ax.bar(down.index, down['Open'] - down['Close'], bottom=down['Close'], 
               color=self.colors['bearish'], alpha=0.8, width=width)
        ax.bar(down.index, down['High'] - down['Open'], bottom=down['Open'], 
               color=self.colors['bearish'], alpha=0.6, width=width2)
        ax.bar(down.index, down['Close'] - down['Low'], bottom=down['Low'], 
               color=self.colors['bearish'], alpha=0.6, width=width2)
        
        ax.set_title(f'{symbol} - Enhanced Technical Analysis Chart', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    def _add_moving_averages(self, ax, data: pd.DataFrame):
        """Add multiple moving averages."""
        # Calculate moving averages
        ma10 = data['Close'].rolling(window=10).mean()
        ma20 = data['Close'].rolling(window=20).mean()
        ma50 = data['Close'].rolling(window=50).mean()
        
        # Plot moving averages
        ax.plot(data.index, ma10, label='MA10', color='purple', linewidth=1, alpha=0.8)
        ax.plot(data.index, ma20, label='MA20', color=self.colors['ma_short'], linewidth=1.5)
        ax.plot(data.index, ma50, label='MA50', color=self.colors['ma_long'], linewidth=2)
    
    def _add_bollinger_bands(self, ax, data: pd.DataFrame):
        """Add Bollinger Bands."""
        # Calculate Bollinger Bands
        ma20 = data['Close'].rolling(window=20).mean()
        std20 = data['Close'].rolling(window=20).std()
        upper_band = ma20 + (std20 * 2)
        lower_band = ma20 - (std20 * 2)
        
        # Plot Bollinger Bands
        ax.plot(data.index, upper_band, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        ax.plot(data.index, lower_band, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        ax.fill_between(data.index, upper_band, lower_band, alpha=0.1, color='gray', label='Bollinger Bands')
    
    def _add_support_resistance(self, ax, data: pd.DataFrame, sr_data: Dict):
        """Add support and resistance lines."""
        if sr_data.get('support'):
            ax.axhline(y=sr_data['support'], color=self.colors['support'], 
                      linestyle='--', alpha=0.8, linewidth=2, 
                      label=f"Support: ${sr_data['support']:.2f}")
        
        if sr_data.get('resistance'):
            ax.axhline(y=sr_data['resistance'], color=self.colors['resistance'], 
                      linestyle='--', alpha=0.8, linewidth=2, 
                      label=f"Resistance: ${sr_data['resistance']:.2f}")
    
    def _plot_enhanced_volume(self, ax, data: pd.DataFrame):
        """Plot volume with color coding."""
        # Color volume bars based on price movement
        colors = [self.colors['volume_up'] if close >= open_price else self.colors['volume_down']
                 for close, open_price in zip(data['Close'], data['Open'])]
        
        ax.bar(data.index, data['Volume'], color=colors, alpha=0.7)
        ax.set_ylabel('Volume', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add volume moving average
        vol_ma = data['Volume'].rolling(window=20).mean()
        ax.plot(data.index, vol_ma, color='black', linewidth=1, alpha=0.8, label='Volume MA20')
        ax.legend(loc='upper left')
    
    def _plot_rsi(self, ax, data: pd.DataFrame, indicators: Dict):
        """Plot RSI indicator."""
        # Calculate RSI if not provided
        if 'rsi_series' not in indicators:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))
        else:
            rsi_series = indicators['rsi_series']
        
        # Plot RSI
        ax.plot(data.index, rsi_series, color='purple', linewidth=2)
        ax.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
        ax.fill_between(data.index, 30, 70, alpha=0.1, color='gray')
        
        ax.set_ylabel('RSI', fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
    
    def _plot_macd(self, ax, data: pd.DataFrame, indicators: Dict):
        """Plot MACD indicator."""
        # Calculate MACD if not provided
        if 'macd_series' not in indicators:
            exp1 = data['Close'].ewm(span=12).mean()
            exp2 = data['Close'].ewm(span=26).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
        else:
            macd_line = indicators['macd_series']
            signal_line = indicators['signal_series']
            histogram = indicators['histogram_series']
        
        # Plot MACD
        ax.plot(data.index, macd_line, color='blue', linewidth=2, label='MACD')
        ax.plot(data.index, signal_line, color='red', linewidth=2, label='Signal')
        
        # Plot histogram
        colors = ['green' if h >= 0 else 'red' for h in histogram]
        ax.bar(data.index, histogram, color=colors, alpha=0.6, label='Histogram')
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.set_ylabel('MACD', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
    
    def _add_pattern_annotations(self, ax, data: pd.DataFrame, patterns: Dict):
        """Add pattern annotations to the chart."""
        # Add trend annotations
        trend = patterns.get('trend', 'Unknown')
        if trend != 'Unknown':
            ax.text(0.02, 0.98, f'Trend: {trend}', transform=ax.transAxes,
                   fontsize=12, fontweight='bold', verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Add volatility info
        volatility = patterns.get('volatility', 0)
        if volatility > 0:
            ax.text(0.02, 0.90, f'Volatility: {volatility:.2f}%', transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    def _format_chart(self, fig, ax_price, ax_volume, ax_rsi, ax_macd, symbol: str):
        """Apply final formatting to the chart."""
        # Format x-axis for all subplots
        for ax in [ax_price, ax_volume, ax_rsi, ax_macd]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add legends
        ax_price.legend(loc='upper left', bbox_to_anchor=(0, 1))
        
        # Set x-label only on bottom subplot
        ax_macd.set_xlabel('Date', fontsize=12)
        
        # Remove x-tick labels from upper subplots
        for ax in [ax_price, ax_volume, ax_rsi]:
            ax.set_xticklabels([])
        
        # Add overall title
        fig.suptitle(f'{symbol} - Comprehensive Technical Analysis', 
                    fontsize=18, fontweight='bold', y=0.98)
    
    def create_pattern_recognition_chart(self, data: pd.DataFrame, symbol: str,
                                       patterns: Dict[str, Any], save_path: str) -> str:
        """
        Create a specialized chart for pattern recognition.
        
        Args:
            data: OHLCV data
            symbol: Trading symbol
            patterns: Detected patterns
            save_path: Path to save the chart
            
        Returns:
            Path to the saved chart
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot candlesticks
        self._plot_enhanced_candlesticks(ax, data, symbol)
        
        # Highlight patterns
        self._highlight_chart_patterns(ax, data, patterns)
        
        # Add pattern descriptions
        self._add_pattern_descriptions(ax, patterns)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path
    
    def _highlight_chart_patterns(self, ax, data: pd.DataFrame, patterns: Dict):
        """Highlight specific chart patterns."""
        # This is a simplified pattern highlighting
        # In a real implementation, you would have more sophisticated pattern detection
        
        price_action = patterns.get('price_action', {})
        pattern_type = price_action.get('pattern', 'Neutral')
        
        if 'Bullish' in pattern_type:
            # Highlight bullish pattern area
            recent_data = data.tail(5)
            ax.fill_between(recent_data.index, recent_data['Low'], recent_data['High'],
                           alpha=0.2, color='green', label='Bullish Pattern')
        elif 'Bearish' in pattern_type:
            # Highlight bearish pattern area
            recent_data = data.tail(5)
            ax.fill_between(recent_data.index, recent_data['Low'], recent_data['High'],
                           alpha=0.2, color='red', label='Bearish Pattern')
    
    def _add_pattern_descriptions(self, ax, patterns: Dict):
        """Add pattern descriptions to the chart."""
        y_pos = 0.95
        
        for key, value in patterns.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    text = f'{sub_key}: {sub_value}'
                    ax.text(0.02, y_pos, text, transform=ax.transAxes,
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    y_pos -= 0.05
            else:
                text = f'{key}: {value}'
                ax.text(0.02, y_pos, text, transform=ax.transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                y_pos -= 0.05

