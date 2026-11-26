import numpy as np
import pandas as pd
import pytest

from evaluation.backtest import consensus_decider, summarize, walk_forward
from tests.conftest import make_ohlcv


def linear_series(n=200, step=1.0):
    close = 100 + step * np.arange(n)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1e6}, index=idx)


def test_no_look_ahead():
    data = make_ohlcv(200)
    seen = []

    def spy(window):
        seen.append(window.index[-1])
        return "LONG"

    trades = walk_forward(data, spy, horizon=5, warmup=60)
    # The decider's last visible bar is exactly the decision date: never a future bar.
    assert list(trades["date"]) == seen
    assert all(data.index.get_loc(d) + 5 < len(data) for d in seen)


def test_non_overlapping_steps_and_forward_return():
    data = linear_series(100)
    trades = walk_forward(data, lambda w: "LONG", horizon=10, warmup=30)
    positions = [data.index.get_loc(d) for d in trades["date"]]
    assert np.all(np.diff(positions) == 10)
    t = positions[0]
    assert trades["forward_return"].iloc[0] == pytest.approx(data.Close.iloc[t + 10] / data.Close.iloc[t] - 1)


def test_metrics_on_deterministic_uptrend():
    data = linear_series(200)
    long_res = summarize(walk_forward(data, lambda w: "LONG"), "long", "X", 5)
    short_res = summarize(walk_forward(data, lambda w: "SHORT"), "short", "X", 5)
    hold_res = summarize(walk_forward(data, lambda w: "HOLD"), "hold", "X", 5)

    assert long_res.hit_rate == 1.0 and short_res.hit_rate == 0.0
    assert long_res.strategy_total_return == pytest.approx(long_res.buy_hold_total_return)
    assert long_res.max_drawdown == 0.0
    assert short_res.strategy_total_return < 0
    assert hold_res.trades == 0 and hold_res.coverage == 0 and hold_res.hit_rate is None
    assert hold_res.strategy_total_return == 0.0


def test_hit_rate_p_value_is_insignificant_for_coin_flip():
    data = make_ohlcv(600, drift=0.0)
    rng = np.random.default_rng(0)
    res = summarize(walk_forward(data, lambda w: rng.choice(["LONG", "SHORT"])), "random", "X", 5)
    assert res.hit_rate_p_value > 0.01


def test_consensus_decider_runs_end_to_end():
    data = make_ohlcv(200)
    trades = walk_forward(data, consensus_decider(), horizon=5, warmup=60)
    assert set(trades["decision"]) <= {"LONG", "SHORT", "HOLD"}
    assert len(trades) == len(range(60, len(data) - 5, 5))


@pytest.mark.parametrize("kw", [{"horizon": 0}, {"warmup": 10}])
def test_invalid_parameters(kw):
    with pytest.raises(ValueError):
        walk_forward(make_ohlcv(100), lambda w: "LONG", **kw)


def test_too_short_series_raises():
    with pytest.raises(ValueError):
        summarize(walk_forward(make_ohlcv(40), lambda w: "LONG", warmup=35, horizon=10), "x", "X", 10)
