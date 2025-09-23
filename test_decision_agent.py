"""
Test script for DecisionAgent with mock Groq API
"""
import os
import sys
import json
from unittest.mock import Mock, patch

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.decision_agent import DecisionAgent
from utils.config import DEFAULT_GROQ_MODEL


def create_mock_groq_response():
    """Create a mock response from Groq API."""
    return Mock(
        choices=[
            Mock(
                message=Mock(
                    content=json.dumps({
                        "decision": "LONG",
                        "confidence": 0.75,
                        "justification": "Technical indicators show bullish momentum with RSI in neutral territory and MACD showing positive divergence. Chart patterns indicate an uptrend with strong support levels.",
                        "risk_level": "MEDIUM",
                        "key_factors": ["RSI_NEUTRAL", "MACD_BULLISH", "UPTREND_PATTERN"],
                        "stop_loss_suggestion": 95.50,
                        "take_profit_suggestion": 105.20
                    })
                )
            )
        ]
    )


def test_decision_agent_with_mock():
    """Test DecisionAgent with mocked Groq API."""
    print("Testing DecisionAgent with Mock Groq API...")
    
    # Create sample analysis data
    indicator_analysis = {
        'agent': 'IndicatorAgent',
        'forecast': 'Bullish',
        'evidence': 'RSI: Neutral (52.41); MACD: Bullish (0.15)',
        'trigger': 'MACD bullish crossover',
        'summary': 'Technical indicators show mixed signals with slight bullish bias'
    }
    
    pattern_analysis = {
        'agent': 'PatternAgent',
        'pattern_description': 'Chart shows uptrend pattern with moderate volatility',
        'visual_summary': 'The chart shows an uptrend pattern with 1.46% volatility',
        'chart_analysis': 'Recent price action shows bullish momentum with support at $95'
    }
    
    trend_analysis = {
        'agent': 'TrendAgent',
        'direction': 'Bullish',
        'confidence': 0.80,
        'summary': 'Short-term: Bullish, Medium-term: Bullish, Long-term: Neutral'
    }
    
    # Mock the Groq client
    with patch('groq.Groq') as mock_groq_class:
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = create_mock_groq_response()
        mock_groq_class.return_value = mock_client
        
        try:
            # Create DecisionAgent with mock API key
            agent = DecisionAgent(api_key="mock_api_key")
            
            # Run analysis
            result = agent.analyze(
                indicator_analysis=indicator_analysis,
                pattern_analysis=pattern_analysis,
                trend_analysis=trend_analysis,
                symbol="AAPL"
            )
            
            print("✓ DecisionAgent analysis completed successfully")
            print(f"  Decision: {result['decision']}")
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Risk Level: {result['risk_level']}")
            print(f"  Justification: {result['justification'][:100]}...")
            print(f"  Key Factors: {result['key_factors']}")
            
            # Test decision summary
            summary = agent.get_decision_summary()
            print(f"\\nDecision Summary:")
            print(summary)
            
            return True
            
        except Exception as e:
            print(f"✗ DecisionAgent test failed: {str(e)}")
            return False


def test_decision_agent_error_handling():
    """Test DecisionAgent error handling."""
    print("\\nTesting DecisionAgent error handling...")
    
    # Mock the Groq client to raise an exception
    with patch('groq.Groq') as mock_groq_class:
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_groq_class.return_value = mock_client
        
        try:
            agent = DecisionAgent(api_key="mock_api_key")
            
            result = agent.analyze(
                indicator_analysis={'agent': 'IndicatorAgent'},
                pattern_analysis={'agent': 'PatternAgent'},
                trend_analysis={'agent': 'TrendAgent'},
                symbol="TEST"
            )
            
            # Should return a safe fallback decision
            if result['decision'] == 'HOLD' and 'API error' in result['justification']:
                print("✓ Error handling works correctly - returned safe fallback")
                return True
            else:
                print("✗ Error handling failed - unexpected result")
                return False
                
        except Exception as e:
            print(f"✗ Error handling test failed: {str(e)}")
            return False


def test_decision_agent_json_parsing():
    """Test DecisionAgent JSON parsing with malformed response."""
    print("\\nTesting DecisionAgent JSON parsing...")
    
    # Mock malformed JSON response
    malformed_response = Mock(
        choices=[
            Mock(
                message=Mock(
                    content="This is not valid JSON {malformed"
                )
            )
        ]
    )
    
    with patch('groq.Groq') as mock_groq_class:
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = malformed_response
        mock_groq_class.return_value = mock_client
        
        try:
            agent = DecisionAgent(api_key="mock_api_key")
            
            result = agent.analyze(
                indicator_analysis={'agent': 'IndicatorAgent'},
                pattern_analysis={'agent': 'PatternAgent'},
                trend_analysis={'agent': 'TrendAgent'},
                symbol="TEST"
            )
            
            # Should return a safe fallback decision
            if result['decision'] == 'HOLD' and 'Failed to parse' in result['justification']:
                print("✓ JSON parsing error handling works correctly")
                return True
            else:
                print("✗ JSON parsing error handling failed")
                return False
                
        except Exception as e:
            print(f"✗ JSON parsing test failed: {str(e)}")
            return False


def test_decision_agent_remaps_decommissioned_model():
    """Ensure decommissioned models are automatically remapped."""

    with patch('groq.Groq') as mock_groq_class:
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = create_mock_groq_response()
        mock_groq_class.return_value = mock_client

        agent = DecisionAgent(api_key="mock_api_key", model="llama3-8b-8192")
        assert agent.model == DEFAULT_GROQ_MODEL


def main():
    """Run all DecisionAgent tests."""
    print("DecisionAgent Tests")
    print("=" * 50)
    
    test1 = test_decision_agent_with_mock()
    test2 = test_decision_agent_error_handling()
    test3 = test_decision_agent_json_parsing()
    
    print("\\n" + "=" * 50)
    print("Test Summary:")
    print(f"✓ Basic functionality: {'PASS' if test1 else 'FAIL'}")
    print(f"✓ Error handling: {'PASS' if test2 else 'FAIL'}")
    print(f"✓ JSON parsing: {'PASS' if test3 else 'FAIL'}")
    
    if all([test1, test2, test3]):
        print("\\n🎉 All DecisionAgent tests passed!")
    else:
        print("\\n⚠️  Some DecisionAgent tests failed.")


if __name__ == "__main__":
    main()

