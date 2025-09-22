# QuantAgent - AI-Powered Trading Analysis Assistant

QuantAgent is a multi-agent large language model (LLM) framework designed for high-frequency trading analysis. The framework was introduced in the research paper QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading by Fei Xiong (Stony Brook University, Carnegie Mellon University), Xiang Zhang (University of British Columbia), Aosong Feng (Yale University), Siqi Sun (Fudan University), and Chenyu You (Stony Brook University).

This project implements the core concepts of QuantAgent using a modern Python stack, providing a comprehensive and interactive tool for financial market analysis.

This application leverages a modular, agent-based architecture to analyze market data from various perspectives, including technical indicators, chart patterns, and trend analysis. The insights from these specialized agents are then synthesized by a decision agent, powered by a large language model (LLM), to provide a final trading recommendation.




## Key Features

- **Multi-Agent Architecture:** The system is composed of specialized agents, each responsible for a specific analytical task:
    - **IndicatorAgent:** Calculates and interprets a wide range of technical indicators.
    - **PatternAgent:** Generates advanced candlestick charts and identifies visual patterns.
    - **TrendAgent:** Analyzes market trends across multiple timeframes.
    - **DecisionAgent:** Utilizes the Groq LLM to synthesize agent analyses and provide trading recommendations.
- **Interactive Web Interface:** A user-friendly Streamlit application for easy interaction and visualization of analysis results.
- **Flexible Data Acquisition:** Fetches market data using the OpenBB SDK with yfinance as a fallback, ensuring data availability.
- **Comprehensive Analysis:** Provides a holistic view of the market by combining technical, pattern, and trend analysis.
- **AI-Powered Decisions:** Leverages the speed and power of the Groq LLM for near real-time decision-making.
- **Extensible and Modular:** The agent-based design allows for easy extension and customization.




## System Architecture

The QuantAgent system is designed with a modular, multi-agent architecture that separates concerns and allows for specialized analysis. The orchestrator coordinates the workflow, from data fetching to the final AI-powered decision.


<img width="773" height="1546" alt="pako_eNp1kl1vmzAUhv-K5YuJSiRLAwkNlSYloUlpwtqp7c0gFx6cBGtgmDHTMsR_n3EgkqWOK57D-yGb0-C4SAC7-MRJmaI3L2JIPsvwvQKOfFbWwkWv5_xHkZnoBTgtElOOBfDfJDug0egLWhnfasLE8gRMoGcep1AJTkTBb-4vYSslWzceEQRtQEgFRyP0XAJbrT6fj5QRFkPbq9dK7" src="https://github.com/user-attachments/assets/6cd2e6c8-dcfd-4381-8505-789b27898c50" />




## Getting Started

### Prerequisites

- Python 3.8 or higher
- Pip (Python package installer)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rlancaster243/quant-agent.git
   cd quant-agent
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```
   This project now explicitly depends on **Seaborn** (version 0.12 or newer) for enhanced statistical charting, so ensure your
   environment can install the package successfully.

3. **Set up your API keys:**
   - Rename the `.env.example` file to `.env`.
   - Add your Groq API key to the `.env` file:
     ```
     GROQ_API_KEY="your-groq-api-key"
     ```

### Usage

To run the Streamlit application, execute the following command in your terminal:

```bash
streamlit run streamlit_app.py
```

This will start the web server and open the application in your default browser. From there, you can enter a stock symbol, select the analysis parameters, and get a comprehensive trading analysis.




## Project Structure

```
quant-agent/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py
│   ├── indicator_agent.py
│   ├── pattern_agent.py
│   ├── trend_agent.py
│   └── decision_agent.py
├── charts/
│   └── (generated charts)
├── data/
│   └── (cached data)
├── utils/
│   ├── __init__.py
│   ├── config.py
│   ├── data_fetcher.py
│   └── chart_enhancer.py
├── .env
├── .env.example
├── requirements.txt
├── streamlit_app.py
├── quant_agent_orchestrator.py
├── test_agents.py
├── test_decision_agent.py
├── test_full_system.py
└── README.md
```




## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.


