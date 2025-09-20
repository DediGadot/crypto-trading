#!/usr/bin/env python3
"""
SIMPLIFIED ERNIE CHAN VALIDATION
Quick validation of core improvements
"""

import asyncio
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from safla_trading.connectivity import get_exchange_registry
from safla_trading.models import GBDTModels
from safla_trading.backtesting import BacktestEngine, BacktestConfig
from safla_trading.logging_system import TradeLogger


async def validate_improvements():
    """Quick validation of all fixes"""
    print("🚀 ERNIE CHAN'S QUANTITATIVE VALIDATION")
    print("=" * 60)

    logger = TradeLogger("simple_validation")
    scores = {}

    # Test 1: Data Quality with 2000+ samples
    print("\n🧪 TEST 1: DATA QUALITY")
    registry = await get_exchange_registry(logger)
    await registry.initialize_exchange('binance')

    candles = await registry.get_historical_data('BTC/USDT', '1h', limit=2500)
    await registry.close_all()

    data = pd.DataFrame([{
        'timestamp': pd.to_datetime(c['timestamp'], unit='ms'),
        'open': float(c['open']),
        'high': float(c['high']),
        'low': float(c['low']),
        'close': float(c['close']),
        'volume': float(c['volume'])
    } for c in candles])
    data.set_index('timestamp', inplace=True)

    returns = data['close'].pct_change().dropna()
    print(f"✅ Fetched {len(data)} samples (target: 2000+)")
    print(f"   Return std: {returns.std():.6f}")
    print(f"   Sharpe: {returns.mean()/returns.std():.4f}")

    scores['data_quality'] = 100 if len(data) >= 2000 else 50

    # Test 2: Feature Selection
    print("\n🧪 TEST 2: FEATURE SELECTION")
    gbdt = GBDTModels(logger)
    X, y = gbdt.prepare_features(data, target_horizon=1)

    print(f"✅ Features selected: {len(X.columns)} (target: ≤15)")
    print(f"   Samples: {len(X)}")
    print(f"   Target range: [{y.min():.6f}, {y.max():.6f}]")

    scores['feature_selection'] = 100 if len(X.columns) <= 15 else 50

    # Test 3: GBDT Training with Purged CV
    print("\n🧪 TEST 3: GBDT TRAINING")
    result = gbdt.train_model(
        data, 'BTC/USDT', 'xgboost',
        target_horizon=1, optimize=True, n_trials=3
    )

    if result.get('success'):
        metrics = result['metrics']
        print(f"✅ Training successful")
        print(f"   Val R²: {metrics['val_r2']:.4f}")
        print(f"   Val RMSE: {metrics['val_rmse']:.6f}")
        scores['gbdt_training'] = 100 if metrics['val_r2'] > 0 else 50
    else:
        print(f"❌ Training failed: {result.get('error')}")
        scores['gbdt_training'] = 0

    # Test 4: Backtesting
    print("\n🧪 TEST 4: BACKTESTING")
    backtest_engine = BacktestEngine(logger)

    class TestStrategy:
        def initialize(self, state):
            state['position'] = 0
            state['prev_sma_5'] = None
            state['prev_sma_20'] = None

        def on_bar(self, bar, state, portfolio_info):
            sma_5 = bar.get('sma_5', bar['close'])
            sma_20 = bar.get('sma_20', bar['close'])
            current_position = portfolio_info.get('current_position', 0)

            if state['prev_sma_5'] and state['prev_sma_20']:
                # Golden cross
                if (state['prev_sma_5'] <= state['prev_sma_20'] and
                    sma_5 > sma_20 and current_position <= 0):
                    return {'action': 'buy', 'size': 0.95}
                # Death cross
                elif (state['prev_sma_5'] >= state['prev_sma_20'] and
                      sma_5 < sma_20 and current_position > 0):
                    return {'action': 'sell'}

            state['prev_sma_5'] = sma_5
            state['prev_sma_20'] = sma_20
            return None

    strategy = TestStrategy()
    backtest_engine.register_strategy('test_sma', strategy)

    config = BacktestConfig(
        initial_cash=100000,
        commission=0.001,
        slippage_value=0.0005
    )

    bt_result = backtest_engine.run_backtest('test_sma', data, config)

    if bt_result and bt_result.trades_count > 0:
        print(f"✅ Backtest successful")
        print(f"   Trades: {bt_result.trades_count}")
        print(f"   Return: {bt_result.total_return:.4f}")
        print(f"   Sharpe: {bt_result.sharpe_ratio:.4f}")
        scores['backtesting'] = 100
    else:
        print(f"❌ No trades generated")
        scores['backtesting'] = 0

    # Final Score
    avg_score = sum(scores.values()) / len(scores)
    print(f"\n" + "=" * 60)
    print(f"FINAL SCORE: {avg_score:.0f}/100")
    print("=" * 60)

    for test, score in scores.items():
        status = "✅" if score >= 75 else "⚠️" if score >= 50 else "❌"
        print(f"{status} {test}: {score}/100")

    if avg_score >= 80:
        print("\n🏆 EXCELLENT: All improvements working correctly")
    elif avg_score >= 60:
        print("\n✅ GOOD: Most improvements functional")
    else:
        print("\n⚠️ NEEDS WORK: Several issues remaining")

    return avg_score


if __name__ == "__main__":
    asyncio.run(validate_improvements())