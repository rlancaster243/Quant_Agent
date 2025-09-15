# QuantAgent Project Design

## 1. System Architecture

The QuantAgent system will be a Streamlit web application that leverages a multi-agent architecture for financial market analysis and trading decisions. The system will be composed of the following key components:

- **Streamlit UI:** A web-based interface for user interaction, allowing users to select financial instruments and view analysis results.
- **Data Acquisition:** Utilizes the OpenBB SDK to fetch historical price data (OHLCV).
- **Agent Framework:** A set of specialized Python modules (agents) that perform specific analytical tasks:
    - **IndicatorAgent:** Calculates technical indicators (e.g., RSI, MACD) using the `pandas_ta` library.
    - **PatternAgent:** Generates candlestick charts using Matplotlib to identify visual patterns.
    - **TrendAgent:** Analyzes trend lines and support/resistance levels.
    - **DecisionAgent:** A Groq-powered LLM that synthesizes information from other agents to make a final trading decision.
- **LLM Integration:** Interfaces with the Groq API for fast and efficient language model reasoning.

## 2. Data Flow

1. The user selects a ticker and timeframe in the Streamlit UI and clicks "Analyze."
2. The Streamlit app calls the data acquisition module to fetch OHLCV data from OpenBB.
3. The data (as a Pandas DataFrame) is passed to the `IndicatorAgent`, `PatternAgent`, and `TrendAgent`.
4. The `IndicatorAgent` calculates technical indicators and returns a structured summary.
5. The `PatternAgent` generates a candlestick chart and saves it as an image.
6. The `TrendAgent` analyzes trend lines and provides its analysis.
7. The outputs from all agents are compiled into a comprehensive prompt for the `DecisionAgent` (Groq LLM).
8. The `DecisionAgent` processes the information and returns a JSON object containing the trading decision (LONG/SHORT) and justification.
9. The Streamlit UI displays the chart, agent analyses, and the final decision to the user.

## 3. Technology Stack

- **Language:** Python 3.11
- **Web Framework:** Streamlit
- **Data Source:** OpenBB SDK
- **Technical Analysis:** pandas_ta
- **Charting:** Matplotlib
- **LLM:** Groq API


