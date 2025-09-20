"""Additional regression coverage for risk management and simulator."""

import asyncio
from datetime import datetime, timedelta

import pytest

from safla_trading.simulator.risk_manager import RiskManager, RiskCheck
from safla_trading.strategies.sma_strategy import TradingSignal
from safla_trading.simulator.trading_simulator import TradingSimulator
from safla_trading.data_feed.binance_feed import OHLCV


def _make_signal(symbol: str, side: str, price: float, quantity: float) -> TradingSignal:
    signal = TradingSignal.__new__(TradingSignal)
    signal.symbol = symbol
    signal.signal = side
    signal.price = price
    signal.quantity = quantity
    signal.confidence = 1.0
    signal.reason = {"source": "test"}
    return signal


def test_risk_manager_allows_exit_when_exposure_reduces():
    manager = RiskManager(initial_balance=100_000.0, logger=None)

    buy_signal = _make_signal("BTC/USDT", "buy", price=10_000.0, quantity=1.0)
    check_buy = manager.check_trade_risk(buy_signal, balance=manager.current_balance)
    assert isinstance(check_buy, RiskCheck)
    assert check_buy.allowed

    manager.open_position(buy_signal.symbol, buy_signal.quantity, buy_signal.price)

    sell_signal = _make_signal("BTC/USDT", "sell", price=10_500.0, quantity=1.0)
    check_sell = manager.check_trade_risk(sell_signal, balance=manager.current_balance)

    assert check_sell.allowed, "Risk manager should allow closing trades that reduce exposure"
    assert pytest.approx(check_sell.risk_factors["new_exposure_pct"], abs=1e-9) == 0.0

    pnl = manager.close_position(sell_signal.symbol, sell_signal.price)
    assert pnl is not None
    assert manager.positions.get(sell_signal.symbol) is None


@pytest.mark.asyncio
async def test_trading_simulator_runs_on_stubbed_feed(monkeypatch):
    prices = [100.0] * 40 + [110.0] * 20 + [90.0] * 20
    start_time = datetime.now()

    candles = [
        OHLCV(
            symbol="BTC/USDT",
            timestamp=int((start_time + timedelta(minutes=i)).timestamp() * 1000),
            open=price,
            high=price + 1,
            low=price - 1,
            close=price,
            volume=5.0,
        )
        for i, price in enumerate(prices)
    ]

    class DummyLogger:
        def __init__(self, *_args, **_kwargs):
            self.session_id = "dummy"

        def log_system_event(self, *args, **kwargs):
            pass

        def log_trade(self, *args, **kwargs):
            pass

        def log_decision(self, *args, **kwargs):
            pass

        def log_performance(self, *args, **kwargs):
            pass

        def log_market_data(self, *args, **kwargs):
            pass

        def log_error(self, *args, **kwargs):
            pass

        def close(self):
            pass

    class DummyDataFeed:
        def __init__(self, _logger):
            self._candles = candles

        async def stream_historical_as_live(
            self, _symbol, _timeframe, _start_date, _end_date, _speed_multiplier
        ):
            for candle in self._candles:
                yield candle

        async def close(self):
            pass

    monkeypatch.setattr("safla_trading.simulator.trading_simulator.TradeLogger", lambda session_id: DummyLogger())
    monkeypatch.setattr("safla_trading.simulator.trading_simulator.BinanceDataFeed", lambda logger: DummyDataFeed(logger))

    simulator = TradingSimulator(symbol="BTC/USDT")

    performance = await simulator.run_backtest(
        start_date=start_time,
        end_date=start_time + timedelta(minutes=len(prices)),
        speed_multiplier=float("inf"),
    )

    assert performance.total_trades > 0

    summary = simulator.get_simulation_summary()
    assert summary["portfolio"]["open_positions"] == 0
    assert summary["performance"]["total_trades"] == performance.total_trades

    await simulator.close()
