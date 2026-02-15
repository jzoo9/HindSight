import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import os

def data_loader(ticker, start='2020-01-01'):
    df = yf.download(ticker, start=start, auto_adjust = True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Close']].copy()
    df.rename(columns={'Close': 'price'}, inplace=True)

    return df

def trade_signal(df):
    sim_df = df.copy()

    sim_df['ma50'] = sim_df['price'].rolling(50).mean()
    sim_df['signal'] = 0
    sim_df.loc[sim_df['price'] > sim_df ['ma50'], 'signal'] = 1
    return sim_df

def backtest_engine(df):
    sim_df = df.copy()
    sim_df['returns'] = sim_df['price'].pct_change()
    sim_df['strategy_returns'] = sim_df['signal'].shift(1) * sim_df ['returns']

    sim_df['equity'] = (1 + sim_df['strategy_returns']).cumprod()
    sim_df['buy_hold'] = (1 + sim_df['returns']).cumprod()


    return sim_df.dropna()


def key_performance_metrics(df, benchmark_df):
    # Align the benchmark returns with our strategy returns
    strategy_returns = df['strategy_returns']
    market_returns = benchmark_df['returns'].reindex(df.index)

    # 1. Basic Returns
    total_return = df['equity'].iloc[-1] - 1
    annualized_return = (1 + total_return) ** (252 / len(df)) - 1

    # 2. Sharpe Ratio
    sharpe = np.sqrt(252) * (strategy_returns.mean() / strategy_returns.std())

    # 3. Beta Calculation
    # Covariance of strategy and market divided by variance of market
    covariance_matrix = np.cov(strategy_returns, market_returns)
    beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]

    # 4. Alpha Calculation (Annualized)
    # Alpha = Strategy Return - (Beta * Market Return)
    # We use annualized figures for a standard 'Alpha'
    market_annualized = (1 + market_returns.mean())**252 - 1
    alpha = annualized_return - (beta * market_annualized)

    # 5. Max Drawdown
    rolling_max = df['equity'].cummax()
    drawdown = df['equity'] / rolling_max - 1
    max_drawdown = drawdown.min()

    return {
        "Total Return (%)": round(total_return * 100, 2),
        "Annualized Return (%)": round(annualized_return * 100, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Beta": round(beta, 2),
        "Alpha (%)": round(alpha * 100, 2),
        "Max Drawdown (%)": round(max_drawdown * 100, 2),
    }


def plot_results(df, bench_df, ticker, benchmark_ticker):
    plt.figure(figsize=(12, 6))
    plt.title(f"Strategy Comparison: {ticker} vs {benchmark_ticker}")
    
    # Normalize Benchmark for plotting (start at 1.0)
    bench_equity = (1 + bench_df['returns'].reindex(df.index).fillna(0)).cumprod()

    plt.plot(df.index, df['equity'], label=f"Strategy ({ticker} MA50)", linewidth=2)
    plt.plot(df.index, df['buy_hold'], label=f"Buy & Hold ({ticker})", alpha=0.6)
    plt.plot(df.index, bench_equity, label=f"Market Benchmark ({benchmark_ticker})", linestyle='--')

    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (Starting at 1.0)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.show()


def main():
    ticker = "NVDA"
    benchmark_ticker = "SPY"
    
    # Loading Primary Data
    df = data_loader(ticker)
    df = trade_signal(df)
    df = backtest_engine(df)

    # Load Benchmark Data & Calc Returns
    benchmark_df = data_loader(benchmark_ticker)
    benchmark_df['returns'] = benchmark_df['price'].pct_change()

    metrics = key_performance_metrics(df, benchmark_df)
    print(f"--- Performance Metrics: {ticker} vs {benchmark_ticker} ---")
    for k, v in metrics.items(): 
        print(f"{k}: {v}")


    plot_results(df, ticker)

if __name__ == "__main__":
            main()

