import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
import os
from datetime import datetime, timedelta


tickers = ['AAPL', 'GOOGL', 'TSLA', 'BND', 'BTC-USD']
start_date = datetime(2015, 1, 1)
end_date = datetime.today()


data = yf.download(tickers, start=start_date, end=end_date)
data = data.stack(level=1, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
data.columns = data.columns.str.lower()


data = data.sort_values(["ticker", "date"])
data[["close", "high", "low", "open", "volume"]] = data.groupby("ticker")[["close", "high", "low", "open", "volume"]].ffill()


data['returns'] = data.groupby('ticker')['close'].pct_change(fill_method=None)
data = data.dropna()


# Check returns scale (should be small, e.g., around 0.01)
print("Sample returns:\n", data.groupby('ticker')['returns'].mean())


# Estimate GARCH volatility for each asset
data['garch_vol'] = 0.0  # Initialize column
for tic in tickers:
    tic_returns = data[data['ticker'] == tic]['returns'].dropna()
    if len(tic_returns) > 10:  # Need sufficient data
        try:
            # Use unscaled returns (decimal form) and disable auto-rescaling
            garch = arch_model(tic_returns, vol='Garch', p=1, q=1, dist='Normal', rescale=False)
            garch_fit = garch.fit(disp='off')
            vol = garch_fit.conditional_volatility
            # Assign volatility to matching indices
            data.loc[data['ticker'] == tic, 'garch_vol'] = vol.reindex(data[data['ticker'] == tic].index, fill_value=0)
        except Exception as e:
            print(f"GARCH failed for {tic}: {e}")
            data.loc[data['ticker'] == tic, 'garch_vol'] = 0.0  # Fallback
    else:
        print(f"Insufficient data for GARCH on {tic}")


# Verify volatility scale (should be small, e.g., 0.01 to 0.1)
print("Sample GARCH volatilities:\n", data.groupby('ticker')['garch_vol'].mean())


# Split datasets
delta = end_date - start_date
train_end = start_date + timedelta(days=delta.days * 0.75)
val_end = start_date + timedelta(days=delta.days * 0.85)
print(train_end, val_end)

train_data = data[data['date'] <= train_end]
val_data = data[(data['date'] > train_end) & (data['date'] <= val_end)]
test_data = data[data['date'] > val_end]

os.makedirs('data', exist_ok=True)
train_data.to_csv(os.path.join("data", "train_data.csv"), index=False)
val_data.to_csv(os.path.join("data", "val_data.csv"), index=False)
test_data.to_csv(os.path.join("data", "test_data.csv"), index=False)

print("\nData prepared:\nTrain shape: {}\nVal shape: {}\nTest shape: {}".format(train_data.shape, val_data.shape, test_data.shape))
