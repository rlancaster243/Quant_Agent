"""
QuantAgent Streamlit Application
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quant_agent_orchestrator import QuantAgentOrchestrator
from utils.config import Config


# Page configuration
st.set_page_config(
    page_title="QuantAgent - AI Trading Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    
    .bullish {
        color: #2e8b57;
        font-weight: bold;
    }
    
    .bearish {
        color: #dc143c;
        font-weight: bold;
    }
    
    .neutral {
        color: #708090;
        font-weight: bold;
    }
    
    .decision-long {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    
    .decision-short {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    
    .decision-hold {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_orchestrator():
    """Initialize the QuantAgent orchestrator (cached)."""
    return QuantAgentOrchestrator()


def format_sentiment_color(sentiment):
    """Format sentiment with appropriate color."""
    if sentiment == 'Bullish':
        return f'<span class="bullish">{sentiment}</span>'
    elif sentiment == 'Bearish':
        return f'<span class="bearish">{sentiment}</span>'
    else:
        return f'<span class="neutral">{sentiment}</span>'


def format_decision_color(decision):
    """Format decision with appropriate color and background."""
    if decision == 'LONG':
        return f'<div class="decision-long">🚀 {decision}</div>'
    elif decision == 'SHORT':
        return f'<div class="decision-short">📉 {decision}</div>'
    else:
        return f'<div class="decision-hold">⏸️ {decision}</div>'


def create_price_chart(data, symbol):
    """Create an interactive price chart using Plotly."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{symbol} Price Chart', 'Volume'),
        row_width=[0.7, 0.3]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Moving averages
    ma20 = data['Close'].rolling(window=20).mean()
    ma50 = data['Close'].rolling(window=50).mean()
    
    fig.add_trace(
        go.Scatter(x=data.index, y=ma20, name='MA20', line=dict(color='blue', width=1)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=data.index, y=ma50, name='MA50', line=dict(color='orange', width=1)),
        row=1, col=1
    )
    
    # Volume chart
    colors = ['green' if close >= open_price else 'red' 
              for close, open_price in zip(data['Close'], data['Open'])]
    
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=colors),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} Technical Analysis',
        xaxis_rangeslider_visible=False,
        height=600,
        showlegend=True
    )
    
    return fig


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">QuantAgent</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Trading Analysis Assistant</p>', unsafe_allow_html=True)
    
    # Initialize orchestrator
    try:
        orchestrator = initialize_orchestrator()
    except Exception as e:
        st.error(f"Failed to initialize QuantAgent: {str(e)}")
        st.stop()
    
    # Sidebar configuration
    st.sidebar.header("📊 Analysis Configuration")
    
    # Symbol input
    symbol = st.sidebar.text_input(
        "Trading Symbol",
        value="AAPL",
        help="Enter a stock symbol (e.g., AAPL, MSFT, GOOGL)"
    ).upper()
    
    # Period and interval selection
    periods = orchestrator.get_available_periods()
    intervals = orchestrator.get_available_intervals()
    
    period = st.sidebar.selectbox(
        "Time Period",
        periods,
        index=periods.index("1y") if "1y" in periods else 0,
        help="Select the time period for analysis"
    )
    
    interval = st.sidebar.selectbox(
        "Data Interval",
        intervals,
        index=intervals.index("1d") if "1d" in intervals else 0,
        help="Select the data interval"
    )
    
    # Analysis button
    analyze_button = st.sidebar.button("🔍 Analyze", type="primary", use_container_width=True)
    
    # System status
    with st.sidebar.expander("🔧 System Status"):
        status = orchestrator.get_system_status()
        
        st.write("**Components:**")
        for component, info in status.items():
            if component != 'config':
                icon = "✅" if info['available'] else "❌"
                st.write(f"{icon} {component.replace('_', ' ').title()}")
        
        st.write("**API Configuration:**")
        api_status = status['config']['api_keys_configured']
        for key, configured in api_status.items():
            icon = "✅" if configured else "❌"
            st.write(f"{icon} {key.replace('_', ' ').title()}")
        
        if status['config']['missing_keys']:
            st.warning(f"Missing API keys: {', '.join(status['config']['missing_keys'])}")
    
    # Main content area
    if analyze_button:
        if not symbol:
            st.error("Please enter a trading symbol")
            return
        
        # Validate symbol
        with st.spinner(f"Validating symbol {symbol}..."):
            if not orchestrator.validate_symbol(symbol):
                st.error(f"Invalid or unavailable symbol: {symbol}")
                return
        
        # Run analysis
        with st.spinner(f"Analyzing {symbol}... This may take a few moments."):
            start_time = time.time()
            result = orchestrator.analyze_symbol(symbol, period, interval)
            analysis_time = time.time() - start_time
        
        if not result['success']:
            st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
            return
        
        # Display results
        st.success(f"✅ Analysis completed in {analysis_time:.1f} seconds")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Price",
                f"${result['current_price']:.2f}",
                f"{result['price_change']:+.2f}%"
            )
        
        with col2:
            sentiment = result['summary']['overall_sentiment']
            st.markdown(f"**Overall Sentiment**")
            st.markdown(format_sentiment_color(sentiment), unsafe_allow_html=True)
        
        with col3:
            confidence = result['summary']['overall_confidence']
            st.metric("Confidence", f"{confidence:.1%}")
        
        with col4:
            decision = result['decision_analysis']['decision']
            st.markdown("**Recommendation**")
            st.markdown(format_decision_color(decision), unsafe_allow_html=True)
        
        # Tabs for detailed analysis
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Overview", "📊 Technical Indicators", "🎯 Patterns", "📉 Trends", "🤖 AI Decision"
        ])
        
        with tab1:
            st.subheader("Market Overview")
            
            # Interactive chart
            if 'data' in locals():
                fig = create_price_chart(data, symbol)
                st.plotly_chart(fig, use_container_width=True)
            
            # Display static chart from PatternAgent
            chart_path = result['pattern_analysis'].get('chart_path')
            if chart_path and os.path.exists(chart_path):
                st.subheader("Technical Analysis Chart")
                st.image(chart_path, caption=f"{symbol} Technical Analysis", use_column_width=True)
            
            # Summary insights
            st.subheader("Key Insights")
            for insight in result['summary']['key_insights']:
                st.write(f"• {insight}")
        
        with tab2:
            st.subheader("Technical Indicators Analysis")
            
            indicator_data = result['indicator_analysis']
            
            # Indicators table
            indicators = indicator_data['indicators']
            signals = indicator_data['signals']
            
            indicator_df = pd.DataFrame([
                {"Indicator": "RSI", "Value": f"{indicators['rsi']:.2f}", "Signal": signals['rsi']},
                {"Indicator": "MACD", "Value": f"{indicators['macd_line']:.4f}", "Signal": signals['macd']},
                {"Indicator": "Rate of Change", "Value": f"{indicators['roc']:.2f}%", "Signal": signals['roc']},
                {"Indicator": "Stochastic", "Value": f"{indicators['stoch_k']:.2f}", "Signal": signals['stoch']},
                {"Indicator": "Williams %R", "Value": f"{indicators['willr']:.2f}", "Signal": signals['willr']},
            ])
            
            st.dataframe(indicator_df, use_container_width=True)
            
            # Indicator summary
            st.subheader("Summary")
            st.write(indicator_data['summary'])
            
            # Forecast and evidence
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Forecast:** {indicator_data['forecast']}")
            with col2:
                st.write(f"**Evidence:** {indicator_data['evidence']}")
        
        with tab3:
            st.subheader("Chart Patterns Analysis")
            
            pattern_data = result['pattern_analysis']
            
            # Pattern description
            st.write(pattern_data['pattern_description'])
            
            # Support and resistance
            sr = pattern_data['support_resistance']
            if sr['support'] and sr['resistance']:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Support Level", f"${sr['support']:.2f}")
                with col2:
                    st.metric("Resistance Level", f"${sr['resistance']:.2f}")
            
            # Visual summary
            st.subheader("Visual Analysis")
            st.write(pattern_data['visual_summary'])
        
        with tab4:
            st.subheader("Trend Analysis")
            
            trend_data = result['trend_analysis']
            
            # Trend summary
            st.write(trend_data['summary'])
            
            # Trend metrics
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Direction:** {trend_data['direction']}")
            with col2:
                st.write(f"**Confidence:** {trend_data['confidence']:.1%}")
            
            # Detailed trend analysis
            trends = trend_data['trend_analysis']
            trend_df = pd.DataFrame([
                {"Timeframe": "Short-term", "Direction": trends['short_term']['direction'], "Strength": trends['short_term']['strength']},
                {"Timeframe": "Medium-term", "Direction": trends['medium_term']['direction'], "Strength": trends['medium_term']['strength']},
                {"Timeframe": "Long-term", "Direction": trends['long_term']['direction'], "Strength": trends['long_term']['strength']},
            ])
            
            st.dataframe(trend_df, use_container_width=True)
        
        with tab5:
            st.subheader("AI Trading Decision")
            
            decision_data = result['decision_analysis']
            
            # Decision summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(format_decision_color(decision_data['decision']), unsafe_allow_html=True)
            with col2:
                st.metric("Confidence", f"{decision_data['confidence']:.1%}")
            with col3:
                st.write(f"**Risk Level:** {decision_data['risk_level']}")
            
            # Justification
            st.subheader("Justification")
            st.write(decision_data['justification'])
            
            # Key factors
            if decision_data.get('key_factors'):
                st.subheader("Key Factors")
                for factor in decision_data['key_factors']:
                    st.write(f"• {factor}")
            
            # Risk management suggestions
            if decision_data.get('stop_loss_suggestion', 0) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Suggested Stop Loss", f"${decision_data['stop_loss_suggestion']:.2f}")
                with col2:
                    if decision_data.get('take_profit_suggestion', 0) > 0:
                        st.metric("Suggested Take Profit", f"${decision_data['take_profit_suggestion']:.2f}")
    
    else:
        # Welcome message
        st.markdown("""
        ## Welcome to QuantAgent! 🚀
        
        QuantAgent is an AI-powered trading analysis assistant that combines multiple specialized agents to provide comprehensive market analysis:
        
        ### 🔍 **What QuantAgent Does:**
        - **Technical Indicators**: RSI, MACD, Stochastic, Williams %R, and Rate of Change analysis
        - **Chart Patterns**: Visual pattern recognition and support/resistance identification
        - **Trend Analysis**: Multi-timeframe trend assessment with confidence scoring
        - **AI Decision Making**: LLM-powered synthesis of all analyses for trading recommendations
        
        ### 📊 **How to Use:**
        1. Enter a stock symbol in the sidebar (e.g., AAPL, MSFT, GOOGL)
        2. Select your preferred time period and data interval
        3. Click "Analyze" to get comprehensive market analysis
        4. Review the results across different tabs for detailed insights
        
        ### ⚙️ **Configuration:**
        - Set your `GROQ_API_KEY` environment variable for AI-powered decisions
        - The system uses yfinance for market data (no API key required)
        
        **Ready to start? Enter a symbol in the sidebar and click Analyze!**
        """)
        
        # Sample symbols
        st.subheader("Popular Symbols to Try:")
        sample_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
        
        cols = st.columns(4)
        for i, sym in enumerate(sample_symbols):
            with cols[i % 4]:
                if st.button(sym, key=f"sample_{sym}"):
                    st.session_state.symbol = sym
                    st.rerun()


if __name__ == "__main__":
    main()

