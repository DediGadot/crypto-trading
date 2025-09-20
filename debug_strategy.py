#!/usr/bin/env python3
"""Debug strategy signal generation"""

import sys
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from safla_trading.strategies import SMAStrategy
from safla_trading.data_feed.binance_feed import OHLCV
from safla_trading.logging_system import TradeLogger

def debug_strategy():
    """Debug why strategy isn't generating signals"""

    logger = TradeLogger("debug_strategy")
    strategy = SMAStrategy("BTC/USDT", logger)

    print(f"Strategy config:")
    print(f"  Fast period: {strategy.fast_period}")
    print(f"  Slow period: {strategy.slow_period}")
    print(f"  Entry threshold: {strategy.entry_threshold}")
    print(f"  Exit threshold: {strategy.exit_threshold}")

    # Create test price data with clear trend
    prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 115, 120]

    print(f"\nProcessing {len(prices)} price points:")

    signals = []
    for i, price in enumerate(prices):
        candle = OHLCV(
            symbol="BTC/USDT",
            timestamp=int(datetime.now().timestamp() * 1000) + i * 60000,
            open=price,
            high=price + 1,
            low=price - 1,
            close=price,
            volume=1000
        )

        signal = strategy.process_candle(candle)
        signals.append(signal)

        # Get strategy state
        state = strategy.get_strategy_state()

        print(f"  Candle {i+1}: price={price}, signal={signal.signal}")
        print(f"    Fast MA: {state['current_fast_ma']}")
        print(f"    Slow MA: {state['current_slow_ma']}")
        print(f"    MA diff: {state['ma_diff_pct']}%")
        print(f"    Ready: {state['ready_to_trade']}")
        print(f"    Reason: {signal.reason}")
        print()

    # Count signals
    buy_signals = [s for s in signals if s.signal == 'buy']
    sell_signals = [s for s in signals if s.signal == 'sell']
    hold_signals = [s for s in signals if s.signal == 'hold']

    print(f"Signal summary:")
    print(f"  Buy signals: {len(buy_signals)}")
    print(f"  Sell signals: {len(sell_signals)}")
    print(f"  Hold signals: {len(hold_signals)}")

    logger.close()

if __name__ == "__main__":
    debug_strategy()