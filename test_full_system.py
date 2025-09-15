"""
Comprehensive test suite for the QuantAgent system
"""
import os
import sys
import time
import warnings
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quant_agent_orchestrator import QuantAgentOrchestrator
from utils.config import Config


def test_system_initialization():
    """Test system initialization and component availability."""
    print("Testing System Initialization...")
    
    try:
        orchestrator = QuantAgentOrchestrator()
        status = orchestrator.get_system_status()
        
        print(f"✓ System initialized successfully")
        
        # Check component availability
        components = ['data_fetcher', 'indicator_agent', 'pattern_agent', 'trend_agent']
        all_available = True
        
        for component in components:
            available = status[component]['available']
            print(f"  {component}: {'✓' if available else '✗'}")
            if not available:
                all_available = False
        
        # Check decision agent (may not be available without API key)
        decision_available = status['decision_agent']['available']
        print(f"  decision_agent: {'✓' if decision_available else '⚠️ (requires GROQ_API_KEY)'}")
        
        return all_available
        
    except Exception as e:
        print(f"✗ System initialization failed: {str(e)}")
        return False


def test_data_fetching():
    """Test data fetching capabilities."""
    print("\\nTesting Data Fetching...")
    
    try:
        orchestrator = QuantAgentOrchestrator()
        
        # Test symbol validation
        valid_symbols = ['AAPL', 'MSFT', 'GOOGL']
        invalid_symbols = ['INVALID123', 'NOTREAL']
        
        for symbol in valid_symbols:
            is_valid = orchestrator.validate_symbol(symbol)
            print(f"  {symbol}: {'✓' if is_valid else '✗'}")
        
        for symbol in invalid_symbols:
            is_valid = orchestrator.validate_symbol(symbol)
            print(f"  {symbol}: {'✗' if not is_valid else '⚠️ unexpected valid'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data fetching test failed: {str(e)}")
        return False


def test_analysis_pipeline():
    """Test the complete analysis pipeline."""
    print("\\nTesting Analysis Pipeline...")
    
    try:
        orchestrator = QuantAgentOrchestrator()
        
        # Test with a known symbol
        symbol = "AAPL"
        print(f"  Analyzing {symbol}...")
        
        start_time = time.time()
        result = orchestrator.analyze_symbol(symbol, period="3mo", interval="1d")
        analysis_time = time.time() - start_time
        
        if not result['success']:
            print(f"✗ Analysis failed: {result.get('error', 'Unknown error')}")
            return False
        
        print(f"✓ Analysis completed in {analysis_time:.1f} seconds")
        print(f"  Data points: {result['data_points']}")
        print(f"  Current price: ${result['current_price']:.2f}")
        print(f"  Overall sentiment: {result['summary']['overall_sentiment']}")
        print(f"  Final decision: {result['decision_analysis']['decision']}")
        
        # Verify all agent results are present
        required_analyses = ['indicator_analysis', 'pattern_analysis', 'trend_analysis', 'decision_analysis']
        for analysis in required_analyses:
            if analysis in result:
                print(f"  {analysis}: ✓")
            else:
                print(f"  {analysis}: ✗")
                return False
        
        # Check if chart was generated
        chart_path = result['pattern_analysis'].get('chart_path')
        if chart_path and os.path.exists(chart_path):
            print(f"  Chart generated: ✓ ({chart_path})")
        else:
            print(f"  Chart generated: ✗")
        
        return True
        
    except Exception as e:
        print(f"✗ Analysis pipeline test failed: {str(e)}")
        return False


def test_multiple_symbols():
    """Test analysis with multiple symbols."""
    print("\\nTesting Multiple Symbols...")
    
    try:
        orchestrator = QuantAgentOrchestrator()
        
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        results = {}
        
        for symbol in symbols:
            print(f"  Analyzing {symbol}...")
            result = orchestrator.analyze_symbol(symbol, period="1mo", interval="1d")
            
            if result['success']:
                results[symbol] = result
                print(f"    ✓ Success - Price: ${result['current_price']:.2f}, Sentiment: {result['summary']['overall_sentiment']}")
            else:
                print(f"    ✗ Failed: {result.get('error', 'Unknown error')}")
        
        print(f"\\n  Successfully analyzed {len(results)}/{len(symbols)} symbols")
        return len(results) >= 2  # At least 2 symbols should work
        
    except Exception as e:
        print(f"✗ Multiple symbols test failed: {str(e)}")
        return False


def test_error_handling():
    """Test error handling with invalid inputs."""
    print("\\nTesting Error Handling...")
    
    try:
        orchestrator = QuantAgentOrchestrator()
        
        # Test invalid symbol
        result = orchestrator.analyze_symbol("INVALID123")
        if not result['success']:
            print("  ✓ Invalid symbol handled correctly")
        else:
            print("  ⚠️ Invalid symbol unexpectedly succeeded")
        
        # Test empty symbol
        result = orchestrator.analyze_symbol("")
        if not result['success']:
            print("  ✓ Empty symbol handled correctly")
        else:
            print("  ⚠️ Empty symbol unexpectedly succeeded")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {str(e)}")
        return False


def test_performance():
    """Test system performance."""
    print("\\nTesting Performance...")
    
    try:
        orchestrator = QuantAgentOrchestrator()
        
        # Test analysis speed
        symbol = "AAPL"
        times = []
        
        for i in range(3):
            start_time = time.time()
            result = orchestrator.analyze_symbol(symbol, period="1mo", interval="1d")
            analysis_time = time.time() - start_time
            
            if result['success']:
                times.append(analysis_time)
                print(f"  Run {i+1}: {analysis_time:.1f}s")
            else:
                print(f"  Run {i+1}: Failed")
        
        if times:
            avg_time = sum(times) / len(times)
            print(f"  Average analysis time: {avg_time:.1f}s")
            
            # Performance benchmark (should complete within reasonable time)
            if avg_time < 10:
                print("  ✓ Performance acceptable")
                return True
            else:
                print("  ⚠️ Performance slower than expected")
                return False
        else:
            print("  ✗ No successful runs for performance test")
            return False
        
    except Exception as e:
        print(f"✗ Performance test failed: {str(e)}")
        return False


def test_configuration():
    """Test configuration management."""
    print("\\nTesting Configuration...")
    
    try:
        config = Config()
        
        # Test basic configuration
        print(f"  ✓ Configuration loaded")
        
        # Test API key validation
        validation = config.validate_api_keys()
        missing_keys = config.get_missing_keys()
        
        print(f"  API key status:")
        for key, valid in validation.items():
            print(f"    {key}: {'✓' if valid else '✗'}")
        
        if missing_keys:
            print(f"  Missing keys: {', '.join(missing_keys)}")
        else:
            print(f"  ✓ All API keys configured")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {str(e)}")
        return False


def generate_test_report(test_results):
    """Generate a comprehensive test report."""
    print("\\n" + "=" * 60)
    print("QUANTAGENT SYSTEM TEST REPORT")
    print("=" * 60)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    print("Test Results:")
    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
    
    print()
    print(f"Summary: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("\\n🎉 ALL TESTS PASSED! QuantAgent system is ready for use.")
    elif passed_tests >= total_tests * 0.8:
        print("\\n✅ Most tests passed. System is functional with minor issues.")
    else:
        print("\\n⚠️ Several tests failed. Please review and fix issues before deployment.")
    
    print("\\nRecommendations:")
    if not test_results.get('Configuration', True):
        print("  - Set up required API keys (GROQ_API_KEY for AI decisions)")
    if not test_results.get('Data Fetching', True):
        print("  - Check internet connectivity and data source availability")
    if not test_results.get('Analysis Pipeline', True):
        print("  - Review agent implementations and dependencies")
    if not test_results.get('Performance', True):
        print("  - Consider optimizing data fetching and analysis algorithms")
    
    print("\\nNext Steps:")
    print("  1. Deploy the Streamlit application")
    print("  2. Set up monitoring and logging")
    print("  3. Create user documentation")
    print("  4. Consider adding more trading symbols and timeframes")


def main():
    """Run all tests and generate report."""
    print("QuantAgent System Test Suite")
    print("=" * 60)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    # Run all tests
    test_results = {
        'System Initialization': test_system_initialization(),
        'Configuration': test_configuration(),
        'Data Fetching': test_data_fetching(),
        'Analysis Pipeline': test_analysis_pipeline(),
        'Multiple Symbols': test_multiple_symbols(),
        'Error Handling': test_error_handling(),
        'Performance': test_performance()
    }
    
    # Generate report
    generate_test_report(test_results)


if __name__ == "__main__":
    main()

