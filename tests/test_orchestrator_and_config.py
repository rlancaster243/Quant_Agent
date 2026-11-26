from unittest.mock import MagicMock

from agents import decision_agent as da
from quant_agent_orchestrator import QuantAgentOrchestrator
from utils.config import Config
from utils.data_fetcher import DataFetcher


def test_config_model_default_defers_to_decision_agent(monkeypatch):
    assert Config().get("llm_model") is None
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-20b")
    assert Config().get("llm_model") == "openai/gpt-oss-20b"


def test_only_groq_key_is_required(monkeypatch):
    assert Config().get_missing_keys() == ["groq_api_key"]
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret-value")
    monkeypatch.setenv("OPENBB_PAT", "openbb-secret-value")
    cfg = Config()
    assert cfg.get_missing_keys() == [] and cfg.is_configured()
    assert "secret-value" not in str(cfg), "config repr must not leak secrets"


def test_orchestrator_uses_decision_agent_default_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setattr(da, "Groq", MagicMock())
    orch = QuantAgentOrchestrator()
    assert orch.decision_agent.model == da.DEFAULT_MODEL


def test_pipeline_without_api_key_returns_hold(monkeypatch, ohlcv):
    monkeypatch.setattr(DataFetcher, "fetch_data", lambda self, *a, **k: ohlcv)
    result = QuantAgentOrchestrator().analyze_symbol("TEST")
    assert result["success"] is True
    assert result["decision_analysis"]["decision"] == "HOLD"
    assert result["data_points"] == len(ohlcv)
    assert result["summary"]["overall_sentiment"] in {"Bullish", "Bearish", "Neutral"}


def test_pipeline_with_mocked_llm(monkeypatch, ohlcv):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    client = MagicMock()
    client.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(
        content='{"decision":"LONG","confidence":0.9,"justification":"x","risk_level":"LOW"}'))]
    monkeypatch.setattr(da, "Groq", MagicMock(return_value=client))
    monkeypatch.setattr(DataFetcher, "fetch_data", lambda self, *a, **k: ohlcv)
    result = QuantAgentOrchestrator().analyze_symbol("TEST")
    assert result["decision_analysis"]["decision"] == "LONG"
    assert result["summary"]["final_decision"] == "LONG"


def test_insufficient_data_is_reported(monkeypatch, ohlcv):
    monkeypatch.setattr(DataFetcher, "fetch_data", lambda self, *a, **k: ohlcv.head(5))
    result = QuantAgentOrchestrator().analyze_symbol("TEST")
    assert result["success"] is False and "Insufficient" in result["error"]
