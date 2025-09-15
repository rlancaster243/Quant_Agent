"""
TrendAgent - Trend Analysis and Momentum Detection
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from scipy import stats
from .base_agent import BaseAgent


class TrendAgent(BaseAgent):
    """
    TrendAgent is responsible for analyzing trends and momentum across multiple timeframes.
    
    This agent performs:
    - Trend line analysis
    - Momentum calculation
    - Multi-timeframe trend assessment
    - Breakout detection
    """
    
    def __init__(self):
        super().__init__("TrendAgent")
        
    def analyze(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Analyze trends and momentum in the market data.
        
        Args:
            data (pd.DataFrame): Market data with OHLCV columns
            
        Returns:
            Dict[str, Any]: Comprehensive trend analysis
        """
        if not self.validate_data(data):
            raise ValueError("Invalid data format. Required columns: Open, High, Low, Close, Volume")
        
        # Multi-timeframe trend analysis
        trend_analysis = self._analyze_trends(data)
        
        # Momentum analysis
        momentum_analysis = self._analyze_momentum(data)
        
        # Trend strength calculation
        trend_strength = self._calculate_trend_strength(data)
        
        # Breakout analysis
        breakout_analysis = self._analyze_breakouts(data)
        
        # Generate comprehensive summary
        summary = self._generate_trend_summary(trend_analysis, momentum_analysis, 
                                             trend_strength, breakout_analysis)
        
        analysis_result = {
            'agent': self.name,
            'trend_analysis': trend_analysis,
            'momentum_analysis': momentum_analysis,
            'trend_strength': trend_strength,
            'breakout_analysis': breakout_analysis,
            'summary': summary,
            'direction': self._get_overall_direction(trend_analysis, momentum_analysis),
            'confidence': self._calculate_confidence(trend_analysis, momentum_analysis, trend_strength)
        }
        
        self.last_analysis = analysis_result
        return analysis_result
    
    def _analyze_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends across multiple timeframes."""
        trends = {}
        
        # Short-term trend (last 10 periods)
        trends['short_term'] = self._calculate_trend(data.tail(10))
        
        # Medium-term trend (last 20 periods)
        trends['medium_term'] = self._calculate_trend(data.tail(20))
        
        # Long-term trend (all available data)
        trends['long_term'] = self._calculate_trend(data)
        
        return trends
    
    def _calculate_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trend for a specific timeframe."""
        if len(data) < 3:
            return {"direction": "Insufficient data", "slope": 0, "r_squared": 0}
        
        # Use closing prices for trend calculation
        prices = data['Close'].values
        x = np.arange(len(prices))
        
        # Linear regression to find trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
        
        # Determine trend direction
        if slope > 0.01:
            direction = "Bullish"
        elif slope < -0.01:
            direction = "Bearish"
        else:
            direction = "Neutral"
        
        return {
            "direction": direction,
            "slope": slope,
            "r_squared": r_value ** 2,
            "strength": self._classify_trend_strength(abs(slope), r_value ** 2)
        }
    
    def _classify_trend_strength(self, slope: float, r_squared: float) -> str:
        """Classify the strength of a trend."""
        if r_squared < 0.3:
            return "Weak"
        elif r_squared < 0.6:
            return "Moderate"
        else:
            return "Strong"
    
    def _analyze_momentum(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price momentum."""
        momentum = {}
        
        # Price momentum (rate of change)
        momentum['price_momentum'] = self._calculate_price_momentum(data)
        
        # Volume momentum
        momentum['volume_momentum'] = self._calculate_volume_momentum(data)
        
        # Acceleration (second derivative of price)
        momentum['acceleration'] = self._calculate_acceleration(data)
        
        return momentum
    
    def _calculate_price_momentum(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate price momentum over different periods."""
        close_prices = data['Close']
        
        momentum_1 = ((close_prices.iloc[-1] - close_prices.iloc[-2]) / close_prices.iloc[-2]) * 100 if len(close_prices) >= 2 else 0
        momentum_5 = ((close_prices.iloc[-1] - close_prices.iloc[-6]) / close_prices.iloc[-6]) * 100 if len(close_prices) >= 6 else 0
        momentum_10 = ((close_prices.iloc[-1] - close_prices.iloc[-11]) / close_prices.iloc[-11]) * 100 if len(close_prices) >= 11 else 0
        
        return {
            "1_period": momentum_1,
            "5_period": momentum_5,
            "10_period": momentum_10,
            "overall": self._classify_momentum(momentum_5)
        }
    
    def _classify_momentum(self, momentum: float) -> str:
        """Classify momentum as strong, moderate, or weak."""
        if abs(momentum) > 5:
            return "Strong " + ("Bullish" if momentum > 0 else "Bearish")
        elif abs(momentum) > 2:
            return "Moderate " + ("Bullish" if momentum > 0 else "Bearish")
        else:
            return "Weak"
    
    def _calculate_volume_momentum(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume momentum."""
        if len(data) < 10:
            return {"trend": "Insufficient data", "ratio": 1.0}
        
        recent_volume = data['Volume'].tail(5).mean()
        historical_volume = data['Volume'].head(-5).mean() if len(data) > 10 else recent_volume
        
        volume_ratio = recent_volume / historical_volume if historical_volume > 0 else 1.0
        
        if volume_ratio > 1.5:
            trend = "Increasing"
        elif volume_ratio < 0.7:
            trend = "Decreasing"
        else:
            trend = "Stable"
        
        return {
            "trend": trend,
            "ratio": volume_ratio
        }
    
    def _calculate_acceleration(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate price acceleration (second derivative)."""
        if len(data) < 3:
            return {"value": 0, "direction": "Insufficient data"}
        
        prices = data['Close'].values
        
        # Calculate first derivative (velocity)
        velocity = np.diff(prices)
        
        # Calculate second derivative (acceleration)
        acceleration = np.diff(velocity)
        
        recent_acceleration = acceleration[-1] if len(acceleration) > 0 else 0
        
        if recent_acceleration > 0:
            direction = "Accelerating Up"
        elif recent_acceleration < 0:
            direction = "Accelerating Down"
        else:
            direction = "Constant Velocity"
        
        return {
            "value": recent_acceleration,
            "direction": direction
        }
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall trend strength using multiple indicators."""
        if len(data) < 20:
            return {"score": 0, "classification": "Insufficient data"}
        
        # ADX-like calculation (simplified)
        high_low = data['High'] - data['Low']
        high_close = abs(data['High'] - data['Close'].shift(1))
        low_close = abs(data['Low'] - data['Close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=14).mean()
        
        # Directional movement
        up_move = data['High'] - data['High'].shift(1)
        down_move = data['Low'].shift(1) - data['Low']
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=14).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=14).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=14).mean()
        
        current_adx = adx.iloc[-1] if not adx.empty else 0
        
        if current_adx > 50:
            classification = "Very Strong"
        elif current_adx > 25:
            classification = "Strong"
        elif current_adx > 15:
            classification = "Moderate"
        else:
            classification = "Weak"
        
        return {
            "score": current_adx,
            "classification": classification
        }
    
    def _analyze_breakouts(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze potential breakouts from support/resistance levels."""
        if len(data) < 20:
            return {"status": "Insufficient data"}
        
        # Calculate recent support and resistance
        recent_data = data.tail(20)
        support = recent_data['Low'].min()
        resistance = recent_data['High'].max()
        current_price = data['Close'].iloc[-1]
        
        # Check for breakouts
        breakout_status = "None"
        if current_price > resistance * 1.001:  # 0.1% buffer
            breakout_status = "Resistance Breakout"
        elif current_price < support * 0.999:  # 0.1% buffer
            breakout_status = "Support Breakdown"
        
        # Calculate breakout strength
        if breakout_status != "None":
            if breakout_status == "Resistance Breakout":
                strength = ((current_price - resistance) / resistance) * 100
            else:
                strength = ((support - current_price) / support) * 100
        else:
            strength = 0
        
        return {
            "status": breakout_status,
            "strength": strength,
            "support_level": support,
            "resistance_level": resistance,
            "current_price": current_price
        }
    
    def _generate_trend_summary(self, trend_analysis: Dict, momentum_analysis: Dict,
                               trend_strength: Dict, breakout_analysis: Dict) -> str:
        """Generate a comprehensive trend summary."""
        summary = "Trend Analysis Summary:\\n"
        
        # Trend directions
        summary += f"- Short-term trend: {trend_analysis['short_term']['direction']} "
        summary += f"({trend_analysis['short_term']['strength']})\\n"
        summary += f"- Medium-term trend: {trend_analysis['medium_term']['direction']} "
        summary += f"({trend_analysis['medium_term']['strength']})\\n"
        summary += f"- Long-term trend: {trend_analysis['long_term']['direction']} "
        summary += f"({trend_analysis['long_term']['strength']})\\n"
        
        # Momentum
        summary += f"- Price momentum: {momentum_analysis['price_momentum']['overall']}\\n"
        summary += f"- Volume momentum: {momentum_analysis['volume_momentum']['trend']}\\n"
        summary += f"- Price acceleration: {momentum_analysis['acceleration']['direction']}\\n"
        
        # Trend strength
        summary += f"- Trend strength: {trend_strength['classification']} "
        summary += f"(Score: {trend_strength['score']:.1f})\\n"
        
        # Breakouts
        summary += f"- Breakout status: {breakout_analysis['status']}"
        if breakout_analysis['status'] != "None" and breakout_analysis['status'] != "Insufficient data":
            summary += f" (Strength: {breakout_analysis['strength']:.2f}%)"
        
        return summary
    
    def _get_overall_direction(self, trend_analysis: Dict, momentum_analysis: Dict) -> str:
        """Determine overall trend direction."""
        # Weight different timeframes
        short_weight = 0.5
        medium_weight = 0.3
        long_weight = 0.2
        
        bullish_score = 0
        bearish_score = 0
        
        # Score trends
        for timeframe, weight in [('short_term', short_weight), 
                                 ('medium_term', medium_weight), 
                                 ('long_term', long_weight)]:
            direction = trend_analysis[timeframe]['direction']
            if direction == "Bullish":
                bullish_score += weight
            elif direction == "Bearish":
                bearish_score += weight
        
        # Add momentum score
        momentum = momentum_analysis['price_momentum']['overall']
        if "Bullish" in momentum:
            bullish_score += 0.2
        elif "Bearish" in momentum:
            bearish_score += 0.2
        
        if bullish_score > bearish_score:
            return "Bullish"
        elif bearish_score > bullish_score:
            return "Bearish"
        else:
            return "Neutral"
    
    def _calculate_confidence(self, trend_analysis: Dict, momentum_analysis: Dict,
                            trend_strength: Dict) -> float:
        """Calculate confidence in the trend analysis."""
        confidence_factors = []
        
        # Trend alignment across timeframes
        directions = [trend_analysis[tf]['direction'] for tf in ['short_term', 'medium_term', 'long_term']]
        alignment = len(set(directions))
        if alignment == 1:  # All same direction
            confidence_factors.append(0.4)
        elif alignment == 2:  # Two same, one different
            confidence_factors.append(0.2)
        else:  # All different
            confidence_factors.append(0.0)
        
        # Trend strength
        strength_score = trend_strength['score'] / 100  # Normalize to 0-1
        confidence_factors.append(strength_score * 0.3)
        
        # Momentum consistency
        momentum = momentum_analysis['price_momentum']
        momentum_consistency = 1 - (abs(momentum['1_period'] - momentum['5_period']) / 10)
        confidence_factors.append(max(0, momentum_consistency) * 0.3)
        
        return min(1.0, sum(confidence_factors))

