"""DecisionAgent tests with the Groq client mocked where it is looked up (agents.decision_agent.Groq)."""
import json
from unittest.mock import MagicMock

import pytest

from agents import decision_agent as da

INPUTS = dict(
    indicator_analysis={"summary": "RSI 55", "forecast": "Bullish", "evidence": "e", "trigger": "t"},
    pattern_analysis={"pattern_description": "uptrend", "visual_summary": "higher highs"},
    trend_analysis={"summary": "up", "direction": "Bullish", "confidence": 0.7},
    symbol="AAPL",
)


def _reply(content: str):
    msg = MagicMock()
    msg.choices = [MagicMock(message=MagicMock(content=content))]
    return msg


@pytest.fixture
def groq(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(da, "Groq", MagicMock(return_value=client))
    return client


def make_agent(**kw):
    return da.DecisionAgent(api_key="test-key", **kw)


def test_requires_api_key():
    with pytest.raises(ValueError):
        da.DecisionAgent()


def test_default_model_used_when_none(groq):
    assert make_agent().model == da.DEFAULT_MODEL
    assert make_agent(model=None).model == da.DEFAULT_MODEL
    assert make_agent(model="custom").model == "custom"


def test_valid_json_decision(groq):
    payload = {"decision": "LONG", "confidence": 0.8, "justification": "aligned",
               "risk_level": "LOW", "key_factors": ["trend"], "stop_loss_suggestion": 95,
               "take_profit_suggestion": 110}
    groq.chat.completions.create.return_value = _reply(json.dumps(payload))
    result = make_agent().analyze(**INPUTS)
    assert result["decision"] == "LONG"
    assert result["confidence"] == 0.8
    assert result["symbol"] == "AAPL"
    prompt = groq.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "AAPL" in prompt and "uptrend" in prompt


@pytest.mark.parametrize("content", [
    '```json\n{"decision":"SHORT","confidence":0.6,"justification":"x","risk_level":"HIGH"}\n```',
    'Here is my analysis:\n{"decision":"SHORT","confidence":0.6,"justification":"x","risk_level":"HIGH"}\nThanks.',
    '```\n{"decision":"short","confidence":"0.6","justification":"x","risk_level":"high"}\n```',
])
def test_tolerates_fences_prose_and_loose_types(groq, content):
    groq.chat.completions.create.return_value = _reply(content)
    result = make_agent().analyze(**INPUTS)
    assert result["decision"] == "SHORT"
    assert result["confidence"] == pytest.approx(0.6)
    assert result["risk_level"] == "HIGH"
    assert result["stop_loss_suggestion"] == 0.0


@pytest.mark.parametrize("content", ["not json", '{"decision": "LONG"}', "[1, 2]", ""])
def test_unparseable_or_incomplete_falls_back_to_hold(groq, content):
    groq.chat.completions.create.return_value = _reply(content)
    result = make_agent().analyze(**INPUTS)
    assert result["decision"] == "HOLD"
    assert result["confidence"] == 0.0
    assert "PARSING_ERROR" in result["key_factors"]


def test_out_of_range_values_are_clamped(groq):
    groq.chat.completions.create.return_value = _reply(
        '{"decision":"BUY","confidence":7,"justification":"x","risk_level":"EXTREME","stop_loss_suggestion":"n/a"}')
    result = make_agent().analyze(**INPUTS)
    assert result["decision"] == "HOLD"
    assert result["confidence"] == 1.0
    assert result["risk_level"] == "MEDIUM"
    assert result["stop_loss_suggestion"] == 0.0


def test_api_error_falls_back_to_hold(groq):
    groq.chat.completions.create.side_effect = RuntimeError("rate limited")
    result = make_agent().analyze(**INPUTS)
    assert result["decision"] == "HOLD"
    assert "API_ERROR" in result["key_factors"]


def test_summary_uses_real_newlines(groq):
    groq.chat.completions.create.return_value = _reply(
        '{"decision":"LONG","confidence":0.5,"justification":"x","risk_level":"LOW","stop_loss_suggestion":"99.5"}')
    agent = make_agent()
    agent.analyze(**INPUTS)
    summary = agent.get_decision_summary()
    assert "\\n" not in summary
    assert "Suggested Stop Loss: 99.50" in summary.splitlines()
