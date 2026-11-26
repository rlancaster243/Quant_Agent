"""
DecisionAgent - Final Trading Decision Synthesis using Groq LLM
"""
import json
import os
import re
from typing import Dict, Any, Optional
from groq import Groq
from .base_agent import BaseAgent

DEFAULT_MODEL = "moonshotai/kimi-k2-instruct-0905"


def _to_float(value: Any, default: float = 0.0) -> float:
    """Coerce LLM-provided numbers (which may arrive as strings or null) to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class DecisionAgent(BaseAgent):
    """
    DecisionAgent synthesizes information from all other agents to make final trading decisions.
    
    This agent uses the Groq API to process structured inputs from:
    - IndicatorAgent: Technical indicators analysis
    - PatternAgent: Chart patterns and visual analysis
    - TrendAgent: Trend and momentum analysis
    
    Output: Structured trading decision (LONG/SHORT) with justification
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__("DecisionAgent")
        
        # Initialize Groq client
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable.")
        
        self.client = Groq(api_key=self.api_key)
        self.model = model or DEFAULT_MODEL
        
    def analyze(self, indicator_analysis: Dict[str, Any], 
                pattern_analysis: Dict[str, Any], 
                trend_analysis: Dict[str, Any],
                symbol: str = "UNKNOWN",
                **kwargs) -> Dict[str, Any]:
        """
        Make final trading decision based on all agent analyses.
        
        Args:
            indicator_analysis: Results from IndicatorAgent
            pattern_analysis: Results from PatternAgent  
            trend_analysis: Results from TrendAgent
            symbol: Trading symbol
            
        Returns:
            Dict[str, Any]: Final trading decision with justification
        """
        
        # Construct comprehensive prompt
        prompt = self._construct_decision_prompt(
            indicator_analysis, pattern_analysis, trend_analysis, symbol
        )
        
        # Get LLM decision
        llm_response = self._query_llm(prompt)
        
        # Parse and validate response
        decision_result = self._parse_llm_response(llm_response)
        
        # Add metadata
        decision_result.update({
            'agent': self.name,
            'symbol': symbol,
            'input_analyses': {
                'indicator': indicator_analysis,
                'pattern': pattern_analysis,
                'trend': trend_analysis
            }
        })
        
        self.last_analysis = decision_result
        return decision_result
    
    def _construct_decision_prompt(self, indicator_analysis: Dict, pattern_analysis: Dict,
                                 trend_analysis: Dict, symbol: str) -> str:
        """Construct a comprehensive prompt for the LLM."""
        
        prompt = f"""You are a professional quantitative trading analyst making high-frequency trading decisions. 
Analyze the following comprehensive market data for {symbol} and provide a structured trading decision.

=== TECHNICAL INDICATORS ANALYSIS ===
{indicator_analysis.get('summary', 'No indicator analysis available')}

Indicator Forecast: {indicator_analysis.get('forecast', 'Unknown')}
Evidence: {indicator_analysis.get('evidence', 'No evidence')}
Trigger: {indicator_analysis.get('trigger', 'No trigger')}

=== CHART PATTERN ANALYSIS ===
{pattern_analysis.get('pattern_description', 'No pattern analysis available')}

Visual Summary: {pattern_analysis.get('visual_summary', 'No visual summary')}

=== TREND ANALYSIS ===
{trend_analysis.get('summary', 'No trend analysis available')}

Overall Direction: {trend_analysis.get('direction', 'Unknown')}
Confidence: {trend_analysis.get('confidence', 0):.2f}

=== DECISION REQUIREMENTS ===
Based on the above analysis, provide a trading decision following these guidelines:

1. Consider all three analyses with appropriate weights:
   - Technical Indicators: 30%
   - Chart Patterns: 25% 
   - Trend Analysis: 45%

2. Account for risk management:
   - Only recommend LONG/SHORT if confidence is reasonable
   - Consider conflicting signals
   - Evaluate market volatility

3. Provide clear justification for your decision

Respond ONLY with a valid JSON object in this exact format:
{{
    "decision": "LONG" | "SHORT" | "HOLD",
    "confidence": 0.0-1.0,
    "justification": "Clear explanation of the decision reasoning",
    "risk_level": "LOW" | "MEDIUM" | "HIGH",
    "key_factors": ["factor1", "factor2", "factor3"],
    "stop_loss_suggestion": number,
    "take_profit_suggestion": number
}}

Ensure the JSON is valid and complete."""

        return prompt
    
    def _query_llm(self, prompt: str) -> str:
        """Query the Groq LLM with the constructed prompt."""
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional quantitative trading analyst. Always respond with valid JSON only."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.1,  # Low temperature for consistent decisions
                max_tokens=1000
            )
            
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            # Fallback response in case of API error
            return json.dumps({
                "decision": "HOLD",
                "confidence": 0.0,
                "justification": f"LLM API error: {str(e)}",
                "risk_level": "HIGH",
                "key_factors": ["API_ERROR"],
                "stop_loss_suggestion": 0,
                "take_profit_suggestion": 0
            })
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate the LLM response."""
        try:
            # Models often wrap JSON in markdown fences or add prose around it;
            # extract the outermost {...} block instead of assuming a clean payload.
            match = re.search(r"\{.*\}", response or "", re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in response")
            decision_data = json.loads(match.group(0))
            if not isinstance(decision_data, dict):
                raise ValueError("Response JSON is not an object")
            
            # Validate required fields
            required_fields = ["decision", "confidence", "justification", "risk_level"]
            for field in required_fields:
                if field not in decision_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate decision values (case-insensitive)
            decision = str(decision_data["decision"]).strip().upper()
            decision_data["decision"] = decision if decision in ("LONG", "SHORT", "HOLD") else "HOLD"
            
            # Validate confidence range
            decision_data["confidence"] = max(0.0, min(1.0, _to_float(decision_data["confidence"])))
            
            # Validate risk level
            risk = str(decision_data["risk_level"]).strip().upper()
            decision_data["risk_level"] = risk if risk in ("LOW", "MEDIUM", "HIGH") else "MEDIUM"
            
            # Normalize optional fields
            factors = decision_data.get("key_factors") or []
            decision_data["key_factors"] = [str(f) for f in factors] if isinstance(factors, list) else [str(factors)]
            decision_data["stop_loss_suggestion"] = _to_float(decision_data.get("stop_loss_suggestion"))
            decision_data["take_profit_suggestion"] = _to_float(decision_data.get("take_profit_suggestion"))
            
            return decision_data
            
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            # Return safe fallback decision
            return {
                "decision": "HOLD",
                "confidence": 0.0,
                "justification": f"Failed to parse LLM response: {str(e)}. Response: {response[:200]}...",
                "risk_level": "HIGH",
                "key_factors": ["PARSING_ERROR"],
                "stop_loss_suggestion": 0,
                "take_profit_suggestion": 0
            }
    
    def get_decision_summary(self) -> str:
        """Get a human-readable summary of the last decision."""
        if not self.last_analysis:
            return "No decision made yet."
        
        decision = self.last_analysis
        summary = f"Trading Decision: {decision['decision']}\n"
        summary += f"Confidence: {decision['confidence']:.1%}\n"
        summary += f"Risk Level: {decision['risk_level']}\n"
        summary += f"Justification: {decision['justification']}\n"
        
        if decision['key_factors']:
            summary += f"Key Factors: {', '.join(decision['key_factors'])}\n"
        
        if decision['stop_loss_suggestion'] > 0:
            summary += f"Suggested Stop Loss: {decision['stop_loss_suggestion']:.2f}\n"
        
        if decision['take_profit_suggestion'] > 0:
            summary += f"Suggested Take Profit: {decision['take_profit_suggestion']:.2f}"
        
        return summary

