# QuantAgent — Multi-Agent LLM Trading Analysis

An implementation of the architecture in *QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading* (Xiong, Zhang, Feng, Sun, You). Three deterministic analysis agents (indicators, chart patterns, trends) feed a Groq-hosted LLM that synthesizes a structured LONG / SHORT / HOLD decision, served through a Streamlit UI.

> **Scope note.** The paper targets high-frequency (intraday) trading. This project runs on daily bars from Yahoo Finance by default, and it implements the paper's *architecture*, not its experimental results. See [Evaluation](#evaluation) and [Limitations](#limitations).

## Architecture

```
                 ┌──────────────┐
   yfinance ───► │ DataFetcher  │  (OpenBB used only if the optional SDK is installed)
                 └──────┬───────┘
         ┌──────────────┼───────────────┐
         ▼              ▼               ▼
 IndicatorAgent    PatternAgent     TrendAgent        deterministic (pandas / ta / scipy)
 RSI, MACD, ROC,   candlestick      multi-horizon
 Stoch, Williams%R chart, S/R       regression + momentum
         └──────────────┼───────────────┘
                        ▼
                 DecisionAgent  ──►  Groq LLM (default: moonshotai/kimi-k2-instruct-0905)
                        │            strict-JSON prompt, validated + clamped,
                        ▼            falls back to HOLD on API/parse errors
              LONG / SHORT / HOLD + confidence, risk, stop/target
```

| Component | File |
|---|---|
| Orchestrator | `quant_agent_orchestrator.py` |
| Agents | `agents/{indicator,pattern,trend,decision}_agent.py` |
| Data, config, charts | `utils/` |
| Walk-forward evaluation | `evaluation/backtest.py` |
| Tests (offline, mocked LLM) | `tests/` |
| UI | `streamlit_app.py` |

## Getting started

Requires **Python 3.10–3.12**.

```bash
git clone https://github.com/rlancaster243/Quant_Agent.git
cd Quant_Agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then add your GROQ_API_KEY; .env is git-ignored
streamlit run streamlit_app.py
```

Optional settings (`.env`): `LLM_MODEL` overrides the default Groq model. `OPENBB_PAT` is only used if you install the OpenBB SDK yourself.

Without a Groq key, the three analysis agents still run and the decision defaults to `HOLD`.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

There are 34 tests, and all of them run offline. Market data is synthetic and the Groq client is mocked, so CI needs no secrets. They cover:
- the agent output contracts
- trend detection on synthetic up- and downtrends
- LLM response parsing: markdown fences, surrounding prose, string-typed numbers, invalid values, API failures
- the full orchestrator pipeline, with and without the LLM
- the backtest engine: a no-look-ahead check, non-overlapping steps, and exact metrics on deterministic series

CI runs the suite on every push and PR (`.github/workflows/ci.yml`).

## Evaluation

`evaluation/backtest.py` runs a **walk-forward** evaluation:
- At each step, the decider sees only bars up to and including *t*.
- It holds a position for `horizon` bars, scored on the forward return.
- Steps don't overlap.

It reports these metrics:
- hit rate, with a binomial p-value against 50%
- total return compared with buy-and-hold
- annualized Sharpe ratio
- max drawdown
- coverage (the share of steps that are not HOLD)

```bash
# Rule-based + naive baselines (no API key needed)
python -m evaluation.backtest --symbols AAPL MSFT SPY --period 5y --horizon 5

# Full LLM pipeline (one Groq call per step; cap steps to bound cost)
python -m evaluation.backtest --symbols SPY --deciders llm always_long --max-steps 60
```

### Baseline results

These use daily bars, a 5-day horizon and a 60-bar warm-up, run 2026-11-25. Decision dates run from 2022-02 to 2026-11, with 238 steps per symbol. Full output is in `evaluation/results/`.

| Symbol | Decider | Trades | Hit rate (p) | Strategy return | Buy & hold | Sharpe | Max DD |
|---|---|---|---|---|---|---|---|
| AAPL | consensus | 81 | 0.543 (0.51) | −8.3% | +123.2% | −0.04 | −41.0% |
| AAPL | random | 238 | 0.492 (0.85) | −15.8% | +123.2% | −0.01 | −47.1% |
| MSFT | consensus | 77 | 0.416 (0.17) | −50.7% | +84.9% | −0.89 | −60.7% |
| MSFT | random | 238 | 0.492 (0.85) | −26.8% | +84.9% | −0.09 | −49.8% |
| SPY | consensus | 52 | 0.500 (1.00) | −2.3% | +80.8% | −0.03 | −22.0% |
| SPY | random | 238 | 0.504 (0.95) | −3.1% | +80.8% | +0.03 | −28.6% |

**Interpretation:**
- The rule-based consensus of the Indicator and Trend agents has **no statistically significant edge**: every p-value is above 0.17.
- It underperforms buy-and-hold on all three symbols.
- One reason is structural, and `tests/test_analysis_agents.py` documents it. The oscillators (RSI, Stochastic, Williams %R) read strong rallies as *overbought* (bearish) while the TrendAgent reads them as bullish. So the two agents disagree, and the consensus HOLDs, exactly when trends are strongest.

**The LLM decider has not been benchmarked yet.** Whether the LLM synthesis adds value over these baselines is the project's open question, and the `llm` decider is in place to measure it.

## Limitations

- **No transaction costs, slippage or borrow costs** are modeled. Short returns are idealized.
- **Daily bars only.** The paper's high-frequency setting is not reproduced.
- **Three large-cap symbols over one ~5-year window** is a small sample, and results may not generalize.
- **The LLM decider is non-deterministic.** Even at temperature 0.1, model changes on Groq can alter results.
- PatternAgent's "visual" analysis is computed from prices. No vision model reads the chart image.
- This is an educational project, **not investment advice**.

## License

MIT. See `LICENSE`.
