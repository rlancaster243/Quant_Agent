"""
IndicatorAgent - Technical Indicators Analysis
"""
import pandas as pd
import ta
from typing import Dict, Any, List
from .base_agent import BaseAgent


class IndicatorAgent(BaseAgent):
    """
    IndicatorAgent is responsible for calculating and interpreting technical indicators.
    
    Based on the QuantAgent paper, this agent computes:
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Rate of Change (RoC)
    - Stochastic Oscillator
    - Williams %R
    """
    
    def __init__(self):
        super().__init__("IndicatorAgent")
        
        # Indicator thresholds for classification
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.willr_overbought = -20
        self.willr_oversold = -80
        
    def analyze(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Calculate technical indicators and classify signals.
        
        Args:
            data (pd.DataFrame): Market data with OHLCV columns
            
        Returns:
            Dict[str, Any]: Structured analysis with indicators and signals
        """
        if not self.validate_data(data):
            raise ValueError("Invalid data format. Required columns: Open, High, Low, Close, Volume")
        
        # Calculate technical indicators
        indicators = self._calculate_indicators(data)
        
        # Classify signals
        signals = self._classify_signals(indicators)
        
        # Generate summary
        summary = self._generate_summary(indicators, signals)
        
        analysis_result = {
            'agent': self.name,
            'indicators': indicators,
            'signals': signals,
            'summary': summary,
            'forecast': self._get_forecast(signals),
            'evidence': self._get_evidence(indicators, signals),
            'trigger': self._get_trigger(signals)
        }
        
        self.last_analysis = analysis_result
        return analysis_result
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate all technical indicators using ta library."""
        
        # RSI (14-period)
        rsi = ta.momentum.RSIIndicator(data['Close'], window=14)
        
        # MACD
        macd = ta.trend.MACD(data['Close'])
        
        # Rate of Change (10-period)
        roc = ta.momentum.ROCIndicator(data['Close'], window=10)
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'])
        
        # Williams %R
        willr = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close'])
        
        return {
            'rsi': rsi.rsi().iloc[-1] if len(rsi.rsi()) > 0 else 50.0,
            'macd_line': macd.macd().iloc[-1] if len(macd.macd()) > 0 else 0.0,
            'macd_signal': macd.macd_signal().iloc[-1] if len(macd.macd_signal()) > 0 else 0.0,
            'macd_histogram': macd.macd_diff().iloc[-1] if len(macd.macd_diff()) > 0 else 0.0,
            'roc': roc.roc().iloc[-1] if len(roc.roc()) > 0 else 0.0,
            'stoch_k': stoch.stoch().iloc[-1] if len(stoch.stoch()) > 0 else 50.0,
            'stoch_d': stoch.stoch_signal().iloc[-1] if len(stoch.stoch_signal()) > 0 else 50.0,
            'willr': willr.williams_r().iloc[-1] if len(willr.williams_r()) > 0 else -50.0
        }
    
    def _classify_signals(self, indicators: Dict[str, float]) -> Dict[str, str]:
        """Classify each indicator as Bullish, Bearish, or Neutral."""
        signals = {}
        
        # RSI classification
        if indicators['rsi'] > self.rsi_overbought:
            signals['rsi'] = 'Bearish'
        elif indicators['rsi'] < self.rsi_oversold:
            signals['rsi'] = 'Bullish'
        else:
            signals['rsi'] = 'Neutral'
        
        # MACD classification
        if indicators['macd_line'] > indicators['macd_signal']:
            signals['macd'] = 'Bullish'
        elif indicators['macd_line'] < indicators['macd_signal']:
            signals['macd'] = 'Bearish'
        else:
            signals['macd'] = 'Neutral'
        
        # Rate of Change classification
        if indicators['roc'] > 2:
            signals['roc'] = 'Bullish'
        elif indicators['roc'] < -2:
            signals['roc'] = 'Bearish'
        else:
            signals['roc'] = 'Neutral'
        
        # Stochastic classification
        if indicators['stoch_k'] > 80:
            signals['stoch'] = 'Bearish'
        elif indicators['stoch_k'] < 20:
            signals['stoch'] = 'Bullish'
        else:
            signals['stoch'] = 'Neutral'
        
        # Williams %R classification
        if indicators['willr'] > self.willr_overbought:
            signals['willr'] = 'Bearish'
        elif indicators['willr'] < self.willr_oversold:
            signals['willr'] = 'Bullish'
        else:
            signals['willr'] = 'Neutral'
        
        return signals
    
    def _generate_summary(self, indicators: Dict[str, float], signals: Dict[str, str]) -> str:
        """Generate a human-readable summary of the technical analysis."""
        bullish_count = sum(1 for signal in signals.values() if signal == 'Bullish')
        bearish_count = sum(1 for signal in signals.values() if signal == 'Bearish')
        neutral_count = sum(1 for signal in signals.values() if signal == 'Neutral')
        
        summary = f"Technical Analysis Summary:\\n"
        summary += f"- RSI ({indicators['rsi']:.2f}): {signals['rsi']}\\n"
        summary += f"- MACD ({indicators['macd_line']:.4f}): {signals['macd']}\\n"
        summary += f"- Rate of Change ({indicators['roc']:.2f}%): {signals['roc']}\\n"
        summary += f"- Stochastic ({indicators['stoch_k']:.2f}): {signals['stoch']}\\n"
        summary += f"- Williams %R ({indicators['willr']:.2f}): {signals['willr']}\\n\\n"
        summary += f"Signal Distribution: {bullish_count} Bullish, {bearish_count} Bearish, {neutral_count} Neutral"
        
        return summary
    
    def _get_forecast(self, signals: Dict[str, str]) -> str:
        """Generate a forecast based on signal consensus."""
        bullish_count = sum(1 for signal in signals.values() if signal == 'Bullish')
        bearish_count = sum(1 for signal in signals.values() if signal == 'Bearish')
        
        if bullish_count > bearish_count:
            return "Bullish"
        elif bearish_count > bullish_count:
            return "Bearish"
        else:
            return "Neutral"
    
    def _get_evidence(self, indicators: Dict[str, float], signals: Dict[str, str]) -> str:
        """Provide evidence for the forecast."""
        evidence_points = []
        
        for indicator, signal in signals.items():
            if signal != 'Neutral':
                value = indicators.get(indicator, indicators.get(f"{indicator}_line", 0))
                evidence_points.append(f"{indicator.upper()}: {signal} ({value:.2f})")
        
        return "; ".join(evidence_points) if evidence_points else "Mixed signals with no clear direction"
    
    def _get_trigger(self, signals: Dict[str, str]) -> str:
        """Identify potential trigger conditions."""
        triggers = []
        
        if signals['rsi'] in ['Bullish', 'Bearish']:
            triggers.append(f"RSI {signals['rsi'].lower()} condition")
        
        if signals['macd'] in ['Bullish', 'Bearish']:
            triggers.append(f"MACD {signals['macd'].lower()} crossover")
        
        return "; ".join(triggers) if triggers else "No clear trigger identified"

