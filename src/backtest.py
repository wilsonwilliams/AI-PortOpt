import os

import config
from src.data_processing import get_data, fit_garch_multi
from src.environment import PortfolioEnv

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import backtrader as bt


class MultiVolFeed(bt.feeds.PandasData):
    lines = ('vol',)
    params = (('vol', -1),)


class RLPortfolioStrategy(bt.Strategy):
    def __init__(self, model, tickers):
        self.model = model
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.obs = None
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"BUY EXECUTED: {order.data._name}, Price={order.executed.price}, Size={order.executed.size}")
            else:
                print(f"SELL EXECUTED: {order.data._name}, Price={order.executed.price}, Size={order.executed.size}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print(f"Order problem: {order.data._name}, Status={order.Status[order.status]}")


    def next(self):
        current_prices = np.array([self.datas[i].close[0] for i in range(self.n_assets)])
        current_vols = np.array([self.datas[i].vol[0] for i in range(self.n_assets)])
        current_returns = np.log(current_prices / np.array([self.datas[i].close[-1] for i in range(self.n_assets)]))
        obs = np.concatenate([current_returns, current_vols])
        action, _ = self.model.predict(obs, deterministic=True)
        weights = action / np.sum(action) if np.sum(action) > 0 else np.ones(self.n_assets) / self.n_assets

        value = self.broker.getvalue()
        buy_orders = []
        sell_orders = []
        for i, ticker in enumerate(self.tickers):
            target_value = value * weights[i]
            current_position = self.getpositionbyname(ticker).size * current_prices[i]
            if target_value > current_position:
                qty = int((target_value - current_position) / current_prices[i])
                if qty > 0:
                    buy_orders.append((self.getdatabyname(ticker), qty))
            elif target_value < current_position:
                qty = (current_position - target_value) / current_prices[i]
                if qty > 0:
                    sell_orders.append((self.getdatabyname(ticker), qty))
        for order in sell_orders:
            self.sell(data=order[0], size=order[1])
        for order in buy_orders:
            self.buy(data=order[0], size=order[1])


def monte_carlo_simulation(garch_models, num_simulations=1_000, steps=252):
    sim_returns = { ticker: [] for ticker in garch_models }
    sim_vols = { ticker: [] for ticker in garch_models }
    for ticker, res in garch_models.items():
        for _ in range(num_simulations):
            forecast = res.forecast(horizon=steps, method="simulation", simulations=1)
            sim = forecast.simulations.values[0][0] / 100
            sim_returns[ticker].append(sim)
            sim_vols[ticker].append(forecast.variance.values[0] / 100)
    return { "returns": { ticker: np.array(sim_returns[ticker]) for ticker in garch_models }, "vols": { ticker: np.array(sim_vols[ticker]) for ticker in garch_models } }



if __name__ == "__main__":
    ### RUN CODE TO BACKTEST AND FORWARD TEST ###
    tickers = config.TICKERS

    prices, returns = get_data(tickers, config.TEST_START_DATE, config.DATA_END_DATE)
    _, vols = fit_garch_multi(returns)

    for i, ticker in enumerate(tickers):
        prices[f'vol{i}'] = vols[ticker]

    print('=' * 20 + "  Backtesting with Backtrader  " + '=' * 20)    
    cerebro = bt.Cerebro()

    for i, ticker in enumerate(tickers):
        df = pd.DataFrame({
            'open': prices[ticker],
            'close': prices[ticker],
            'vol': prices[f'vol{i}'],
        }).dropna()

        data = MultiVolFeed(dataname=df)
        cerebro.adddata(data, name=ticker)

    model = PPO.load(os.path.join("models", config.CURR_MODEL))

    cerebro.addstrategy(RLPortfolioStrategy, model=model, tickers=tickers)
    cerebro.broker.setcash(10_000.0)
    cerebro.broker.setcommission(commission=1e-3)
    cerebro.run()
    print("\nFinal Portfolio Value: ${:,.2f}".format(cerebro.broker.getvalue()))

    print('=' * 20 + "  Backtesting Complete  " + '=' * 20 + "\n")
    print('=' * 20 + "  Forward Testing with Monte Carlo Simulations  " + '=' * 20)

    prices_train, returns_train = get_data(tickers, config.TEST_START_DATE, config.DATA_END_DATE)
    garch_models, _ = fit_garch_multi(returns_train)

    print("Generating Monte Carlo paths...")
    sim_data = monte_carlo_simulation(garch_models)

    last_prices = prices_train.iloc[-1].values
    sim_prices = {}
    for i, ticker in enumerate(tickers):
        sim_prices[ticker] = last_prices[i] * np.exp(np.cumsum(sim_data["returns"][ticker], axis=1))

    sim_returns_array = np.stack([sim_data['returns'][t] for t in tickers], axis=2).transpose(1, 0, 2)
    sim_vols_array = np.stack([sim_data['vols'][t] for t in tickers], axis=2).transpose(1, 0, 2)

    port_values = []
    for i in range(len(sim_returns_array)):
        sim_returns_df = pd.DataFrame(sim_returns_array[i], columns=tickers)
        sim_vols_df = pd.DataFrame(sim_vols_array[i], columns=tickers)
        env = PortfolioEnv(sim_returns_df, sim_vols_df)
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
        port_values.append(env.portfolio_value)
        print("Simulated Portfolio Value (Path {}): ${:,.2f}".format(i+1, env.portfolio_value))

    port_values = np.array(port_values)
    print("\nAverage Simulated Portfolio Value: ${:,.2f}".format(np.mean(port_values)))
    print("\n" + '=' * 20 + "  Forward Testing Complete  " + '=' * 20 + "\n")
