from datetime import datetime, timedelta

TICKERS = ["AAPL", "MSFT", "GOOGL", "BND", "TLT", "META", "NVDA", "AVGO", "LLY", "MGK", "MGC", "VOO", "SPY", "QQQ", "IYW"]

DATA_START_DATE = datetime(2015, 1, 1)
DATA_END_DATE = datetime.today()

_DELTA = DATA_END_DATE - DATA_START_DATE
TRAIN_END_DATE = DATA_START_DATE + timedelta(days=_DELTA.days * 0.85)
TEST_START_DATE = DATA_START_DATE + timedelta(days=_DELTA.days * 0.85)

CURR_MODEL = "PPO_100K_MODEL"