import os
from datetime import timedelta

import config

import yfinance as yf
import numpy as np
import pandas as pd
from arch import arch_model


def get_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end)['Close']
    log_returns = np.log(df / df.shift(1))
    log_returns.dropna(inplace=True)
    return df, log_returns


def fit_garch_multi(returns):
    volatilities = pd.DataFrame(index=returns.index, columns=returns.columns)
    models = {}
    for ticker in returns.columns:
        model = arch_model(returns[ticker] * 100, vol='Garch', p=2, q=2)
        res = model.fit(disp='off')
        models[ticker] = res
        volatilities[ticker] = res.conditional_volatility / 100
    return models, volatilities


if __name__ == "__main__":
    ### RUN CODE TO GENERATE DATA ###
    TICKERS = config.TICKERS
    start_date = config.DATA_START_DATE
    end_date = config.DATA_END_DATE
    prices, returns = get_data(TICKERS, start_date, end_date)
    _, volatilities = fit_garch_multi(returns)

    delta = end_date - start_date
    train_end = start_date + timedelta(days=delta.days * 0.85)
    test_start = start_date + timedelta(days=delta.days * 0.85)

    train_returns = returns[returns.index <= train_end]
    # val_returns = returns[(returns.index > train_end) & (returns.index <= test_start)]
    test_returns = returns[returns.index > test_start]

    train_volatilities = volatilities[volatilities.index <= train_end]
    # val_volatilities = volatilities[(volatilities.index > train_end) & (volatilities.index <= test_start)]
    test_volatilities = volatilities[volatilities.index > test_start]

    train_returns.to_csv(os.path.join("data", "train_returns.csv"), index=False)
    train_volatilities.to_csv(os.path.join("data", "train_vols.csv"), index=False)
    # val_returns.to_csv(os.path.join("data", "val_returns.csv"), index=False)
    # val_volatilities.to_csv(os.path.join("data", "val_vols.csv"), index=False)
    test_returns.to_csv(os.path.join("data", "test_returns.csv"), index=False)
    test_volatilities.to_csv(os.path.join("data", "test_vols.csv"), index=False)

    print("\nData Download and Processing Complete:")
    print("Train Data Size: {}, {}".format(train_returns.shape, train_volatilities.shape))
    # print("Validation Data Size: {}, {}".format(val_returns.shape, val_volatilities.shape))
    print("Test Data Size: {}, {}".format(test_returns.shape, test_volatilities.shape))
