"""
FastAPI app for backtesting - POST /api/run_backtest endpoint.
Integrates: data_loader, trade_signal, backtest_engine, key_performance_metrics.
"""
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


class BacktestRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str


def data_loader(ticker: str, start: str = "2020-01-01", end: Optional[str] = None) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty or len(df) < 50:
        raise ValueError(f"Insufficient data for {ticker} (need at least 50 rows)")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].copy()
    df.rename(columns={"Close": "price"}, inplace=True)
    return df


def trade_signal(df: pd.DataFrame) -> pd.DataFrame:
    sim_df = df.copy()
    sim_df["ma50"] = sim_df["price"].rolling(50).mean()
    sim_df["signal"] = 0
    sim_df.loc[sim_df["price"] > sim_df["ma50"], "signal"] = 1
    return sim_df


def backtest_engine(df: pd.DataFrame) -> pd.DataFrame:
    sim_df = df.copy()
    sim_df["returns"] = sim_df["price"].pct_change()
    sim_df["strategy_returns"] = sim_df["signal"].shift(1) * sim_df["returns"]
    sim_df["equity"] = (1 + sim_df["strategy_returns"]).cumprod()
    sim_df["buy_hold"] = (1 + sim_df["returns"]).cumprod()
    return sim_df.dropna()


def key_performance_metrics(df: pd.DataFrame, benchmark_df: pd.DataFrame) -> dict:
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


def run_backtest_result(ticker: str, start_date: str, end_date: str) -> dict:
    """Shared backtest logic; used by FastAPI and Vercel serverless."""
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker is required")

    df = data_loader(ticker, start=start_date, end=end_date)
    df = trade_signal(df)
    df = backtest_engine(df)

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
        return run_backtest_result(req.ticker, req.start_date, req.end_date)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


app.include_router(api, prefix="/api")
