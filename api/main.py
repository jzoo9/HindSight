
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Backtest API")
api = APIRouter()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BENCHMARK_TICKER = "SPY"

@app.get("/")
def root():
    return {
        "ok": True,
        "message": "Backtest API is running",
        "endpoints": {
            "run_backtest": "POST /run_backtest (recommended on Vercel), POST /api/run_backtest",
        },
    }


class BacktestRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    commission_bps: float = 0.0
    slippage_bps: float = 0.0


def data_loader(ticker: str, start: str = "2020-01-01", end: Optional[str] = None):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty or len(df) < 50:
        raise ValueError(f"Insufficient data for {ticker} (need at least 50 rows)")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close", "Open"]].copy()
    df.rename(columns={"Close": "price", "Open": "open"}, inplace=True)
    return df


def trade_signal(df: pd.DataFrame) -> pd.DataFrame:
    sim_df = df.copy()
    sim_df["ma50"] = sim_df["price"].rolling(50).mean()
    sim_df["signal"] = 0
    sim_df.loc[sim_df["price"] > sim_df["ma50"], "signal"] = 1
    sim_df["trade"] = sim_df["signal"].diff().abs().fillna(0)
    return sim_df


def backtest_engine(df: pd.DataFrame, commission_bps: float = 0.0, slippage_bps: float = 0.0) -> pd.DataFrame:
    sim_df = df.copy()
    sim_df["returns"] = sim_df["price"].pct_change()

    # Signal is known at today's close, so the position it implies can only be
    # acted on from tomorrow's open. `position` is what's held today, decided
    # by yesterday's close; `entry`/`exit` flag the day that position actually
    # changes, so those days earn a partial-day return from the open instead
    # of the full close-to-close move (avoids same-bar-close lookahead).
    position = sim_df["signal"].shift(1)
    signal_change = sim_df["signal"].diff()
    entry = signal_change.shift(1) == 1
    exit_ = signal_change.shift(1) == -1

    held_returns = position * sim_df["returns"]
    entry_returns = (sim_df["price"] - sim_df["open"]) / sim_df["open"]
    exit_returns = (sim_df["open"] - sim_df["price"].shift(1)) / sim_df["price"].shift(1)

    sim_df["strategy_returns"] = np.select(
        [entry, exit_],
        [entry_returns, exit_returns],
        default=held_returns,
    )

    cost_per_trade = (commission_bps + slippage_bps) / 10000
    sim_df["strategy_returns"] -= sim_df["trade"].shift(1).fillna(0) * cost_per_trade

    sim_df["equity"] = (1 + sim_df["strategy_returns"]).cumprod()
    sim_df["buy_hold"] = (1 + sim_df["returns"]).cumprod()
    return sim_df.dropna()


def key_performance_metrics(df: pd.DataFrame, benchmark_df: pd.DataFrame):
    strategy_returns = df["strategy_returns"]
    market_returns = benchmark_df["returns"].reindex(df.index).fillna(0)

    total_return = df["equity"].iloc[-1] - 1
    n_days = len(df)
    annualized_return = (1 + total_return) ** (252 / n_days) - 1 if n_days else 0

    sr_std = strategy_returns.std()
    sharpe = float(np.sqrt(252) * (strategy_returns.mean() / sr_std)) if sr_std and sr_std != 0 else 0.0

    cov = np.cov(strategy_returns, market_returns)
    var_market = cov[1, 1]
    beta = float(cov[0, 1] / var_market) if var_market and var_market != 0 else 0.0

    market_annualized = (1 + market_returns.mean()) ** 252 - 1
    alpha = annualized_return - (beta * market_annualized)

    rolling_max = df["equity"].cummax()
    drawdown = df["equity"] / rolling_max - 1
    max_drawdown = float(drawdown.min())

    return {
        "Total Return (%)": round(total_return * 100, 2),
        "Annualized Return (%)": round(annualized_return * 100, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Alpha (%)": round(alpha * 100, 2),
        "Beta": round(beta, 2),
        "Max Drawdown (%)": round(max_drawdown * 100, 2),
    }


def run_backtest_result(ticker: str, start_date: str, end_date: str, commission_bps: float = 0.0, slippage_bps: float = 0.0):
    """Shared backtest logic; used by FastAPI and Vercel serverless."""
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker is required")

    df = data_loader(ticker, start=start_date, end=end_date)
    df = trade_signal(df)
    df = backtest_engine(df, commission_bps=commission_bps, slippage_bps=slippage_bps)

    try:
        benchmark_df = data_loader(BENCHMARK_TICKER, start=start_date, end=end_date)
        benchmark_df["returns"] = benchmark_df["price"].pct_change().fillna(0)
    except Exception:
        benchmark_df = pd.DataFrame(index=df.index)
        benchmark_df["returns"] = 0.0

    metrics = key_performance_metrics(df, benchmark_df)
    bench_equity = (1 + benchmark_df["returns"].reindex(df.index).fillna(0)).cumprod()

    chart_data = []
    for ts in df.index:
        chart_data.append({
            "date": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "strategy_equity": round(float(df.loc[ts, "equity"]), 4),
            "buy_hold": round(float(df.loc[ts, "buy_hold"]), 4),
            "benchmark_equity": round(float(bench_equity.loc[ts]), 4),
        })

    return {
        "ticker": ticker,
        "benchmark_ticker": BENCHMARK_TICKER,
        "metrics": metrics,
        "chart_data": chart_data,
    }


@api.post("/run_backtest")
def run_backtest(req: BacktestRequest):
    try:
        return run_backtest_result(
            req.ticker,
            req.start_date,
            req.end_date,
            commission_bps=req.commission_bps,
            slippage_bps=req.slippage_bps,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Surface a useful error message to the frontend (and logs in Vercel).
        print("run_backtest failed:", repr(e))
        raise HTTPException(status_code=500, detail=f"Backtest failed: {e}")


app.include_router(api, prefix="/api")
