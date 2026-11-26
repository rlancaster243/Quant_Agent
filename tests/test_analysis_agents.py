import os

import pandas as pd
import pytest

from agents.indicator_agent import IndicatorAgent
from agents.pattern_agent import PatternAgent
from agents.trend_agent import TrendAgent

SIGNALS = {"Bullish", "Bearish", "Neutral"}


def test_indicator_agent_output_contract(ohlcv):
    result = IndicatorAgent().analyze(ohlcv)
    assert {"indicators", "signals", "summary", "forecast", "evidence", "trigger"} <= result.keys()
    assert result["forecast"] in SIGNALS
    assert 0 <= result["indicators"]["rsi"] <= 100
    assert "\\n" not in result["summary"], "summary must contain real newlines, not escaped ones"


@pytest.mark.parametrize("agent", [IndicatorAgent(), TrendAgent()])
def test_agents_reject_missing_columns(agent, ohlcv):
    with pytest.raises(ValueError):
        agent.analyze(ohlcv.drop(columns=["Volume"]))


def test_trend_agent_detects_up_and_down_trends(uptrend, downtrend):
    agent = TrendAgent()
    up, down = agent.analyze(uptrend), agent.analyze(downtrend)
    assert up["direction"] == "Bullish"
    assert down["direction"] == "Bearish"
    assert 0.0 <= up["confidence"] <= 1.0


def test_oscillators_are_contrarian_in_strong_trends(uptrend):
    """Documents a design property: RSI/Stochastic/Williams %R read an extended rally as
    overbought (Bearish) while TrendAgent reads it as Bullish, so the two agents disagree
    exactly when trends are strongest. The backtest's consensus decider HOLDs in that case."""
    result = IndicatorAgent().analyze(uptrend)
    assert result["signals"]["rsi"] == "Bearish"
    assert result["signals"]["macd"] == "Bullish"
    assert TrendAgent().analyze(uptrend)["direction"] == "Bullish"


def test_pattern_agent_writes_chart(ohlcv, tmp_path):
    agent = PatternAgent(chart_dir=str(tmp_path / "charts"))
    indicators = IndicatorAgent().analyze(ohlcv)["indicators"]
    result = agent.analyze(ohlcv, symbol="TEST", indicators=indicators)
    assert os.path.isfile(result["chart_path"])
    sr = result["support_resistance"]
    assert sr["support"] <= sr["resistance"]
    assert "\\n" not in result["pattern_description"]


def test_agents_do_not_mutate_input(ohlcv):
    before = ohlcv.copy()
    IndicatorAgent().analyze(ohlcv)
    TrendAgent().analyze(ohlcv)
    pd.testing.assert_frame_equal(ohlcv, before)
