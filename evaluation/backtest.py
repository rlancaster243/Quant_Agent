"""
Walk-forward evaluation for QuantAgent decisions.

At each evaluation point t the decider sees ONLY bars [0, t] (no look-ahead) and returns
LONG / SHORT / HOLD. The position is held for `horizon` bars and scored on the
forward return close[t + horizon] / close[t] - 1. Evaluation points are spaced
`horizon` bars apart so trades never overlap and returns compound cleanly.

Deciders:
    consensus  - rule-based: LONG/SHORT only when IndicatorAgent and TrendAgent agree (no LLM)
    llm        - full DecisionAgent (Groq); requires GROQ_API_KEY, one API call per step
    always_long, random - naive baselines every result must be compared against

Usage:
    python -m evaluation.backtest --symbols AAPL MSFT SPY --period 5y --horizon 5
    python -m evaluation.backtest --symbols AAPL --deciders consensus llm --max-steps 40
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.indicator_agent import IndicatorAgent  # noqa: E402
from agents.trend_agent import TrendAgent  # noqa: E402

Decider = Callable[[pd.DataFrame], str]
DIRECTION = {"LONG": 1, "SHORT": -1, "HOLD": 0}


# --------------------------------------------------------------------------- deciders
def consensus_decider() -> Decider:
    indicator, trend = IndicatorAgent(), TrendAgent()

    def decide(window: pd.DataFrame) -> str:
        forecast = indicator.analyze(window)["forecast"]
        direction = trend.analyze(window)["direction"]
        if forecast == direction == "Bullish":
            return "LONG"
        if forecast == direction == "Bearish":
            return "SHORT"
        return "HOLD"

    return decide


def always_long_decider() -> Decider:
    return lambda window: "LONG"


def random_decider(seed: int = 0) -> Decider:
    rng = np.random.default_rng(seed)
    return lambda window: str(rng.choice(["LONG", "SHORT"]))


def llm_decider(symbol: str, model: Optional[str] = None) -> Decider:
    """Full pipeline incl. the Groq DecisionAgent. Pattern analysis skips chart rendering."""
    from agents.decision_agent import DecisionAgent
    from agents.pattern_agent import PatternAgent

    indicator, trend, pattern = IndicatorAgent(), TrendAgent(), PatternAgent()
    decision_agent = DecisionAgent(model=model)

    def decide(window: pd.DataFrame) -> str:
        patterns = pattern._analyze_patterns(window)
        sr = pattern._calculate_support_resistance(window)
        patterns["support_resistance"] = sr
        pattern_analysis = {
            "pattern_description": pattern._generate_pattern_description(window, patterns, sr),
            "visual_summary": pattern._get_visual_summary(window, patterns),
        }
        result = decision_agent.analyze(
            indicator_analysis=indicator.analyze(window),
            pattern_analysis=pattern_analysis,
            trend_analysis=trend.analyze(window),
            symbol=symbol,
        )
        return result["decision"]

    return decide


# --------------------------------------------------------------------------- engine
@dataclass
class BacktestResult:
    decider: str
    symbol: str
    steps: int
    trades: int
    coverage: float
    hit_rate: Optional[float]
    hit_rate_p_value: Optional[float]
    mean_trade_return: Optional[float]
    strategy_total_return: float
    buy_hold_total_return: float
    excess_vs_buy_hold: float
    sharpe_annualized: Optional[float]
    max_drawdown: float


def walk_forward(data: pd.DataFrame, decide: Decider, horizon: int = 5, warmup: int = 60,
                 max_steps: Optional[int] = None) -> pd.DataFrame:
    """Return one row per evaluation point: decision, forward return, strategy return."""
    if horizon < 1 or warmup < 30:
        raise ValueError("horizon must be >= 1 and warmup >= 30 (indicators need history)")
    close = data["Close"].to_numpy(dtype=float)
    rows = []
    for t in range(warmup, len(data) - horizon, horizon):
        if max_steps is not None and len(rows) >= max_steps:
            break
        decision = decide(data.iloc[: t + 1])  # bars 0..t inclusive, nothing after t
        fwd = close[t + horizon] / close[t] - 1.0
        rows.append({
            "date": data.index[t],
            "decision": decision,
            "forward_return": fwd,
            "strategy_return": DIRECTION.get(decision, 0) * fwd,
        })
    return pd.DataFrame(rows)


def summarize(trades: pd.DataFrame, decider: str, symbol: str, horizon: int,
              bars_per_year: int = 252) -> BacktestResult:
    if trades.empty:
        raise ValueError("No evaluation points: series too short for warmup + horizon")
    active = trades[trades["decision"] != "HOLD"]
    n_active = len(active)
    hits = int((active["strategy_return"] > 0).sum())
    hit_rate = hits / n_active if n_active else None
    p_value = float(stats.binomtest(hits, n_active, 0.5).pvalue) if n_active else None

    equity = (1 + trades["strategy_return"]).cumprod()
    buy_hold = float((1 + trades["forward_return"]).prod() - 1)
    strat_total = float(equity.iloc[-1] - 1)
    drawdown = float((equity / equity.cummax() - 1).min())

    rets = trades["strategy_return"]
    sharpe = None
    if rets.std(ddof=1) > 0:
        sharpe = float(rets.mean() / rets.std(ddof=1) * math.sqrt(bars_per_year / horizon))

    return BacktestResult(
        decider=decider, symbol=symbol, steps=len(trades), trades=n_active,
        coverage=n_active / len(trades), hit_rate=hit_rate, hit_rate_p_value=p_value,
        mean_trade_return=float(active["strategy_return"].mean()) if n_active else None,
        strategy_total_return=strat_total, buy_hold_total_return=buy_hold,
        excess_vs_buy_hold=strat_total - buy_hold, sharpe_annualized=sharpe, max_drawdown=drawdown,
    )


def build_decider(name: str, symbol: str, seed: int) -> Decider:
    if name == "consensus":
        return consensus_decider()
    if name == "always_long":
        return always_long_decider()
    if name == "random":
        return random_decider(seed)
    if name == "llm":
        return llm_decider(symbol)
    raise ValueError(f"Unknown decider: {name}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "SPY"])
    parser.add_argument("--period", default="5y")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=None, help="cap steps (use with llm to bound API cost)")
    parser.add_argument("--deciders", nargs="+", default=["consensus", "always_long", "random"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="evaluation/results")
    args = parser.parse_args(argv)

    from utils.data_fetcher import DataFetcher
    fetcher = DataFetcher(cache_enabled=False)
    os.makedirs(args.out, exist_ok=True)

    results: List[Dict] = []
    for symbol in args.symbols:
        data = fetcher.fetch_data(symbol, args.period, "1d")
        if data is None or len(data) < args.warmup + args.horizon + 1:
            print(f"[skip] {symbol}: insufficient data")
            continue
        for name in args.deciders:
            trades = walk_forward(data, build_decider(name, symbol, args.seed),
                                  args.horizon, args.warmup, args.max_steps)
            res = summarize(trades, name, symbol, args.horizon)
            results.append(asdict(res))
            trades.to_csv(os.path.join(args.out, f"trades_{symbol}_{name}.csv"), index=False)
            hr = f"{res.hit_rate:.3f} (p={res.hit_rate_p_value:.3f})" if res.hit_rate is not None else "n/a"
            print(f"{symbol:6s} {name:12s} trades={res.trades:4d} hit={hr:20s} "
                  f"strat={res.strategy_total_return:+.3f} buy&hold={res.buy_hold_total_return:+.3f}")

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
