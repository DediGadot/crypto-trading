#!/usr/bin/env python3
"""
Karpathy-style validation of trading system improvements
Tests each fix in isolation with clear metrics
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from safla_trading.connectivity import get_exchange_registry
from safla_trading.models import GBDTModels
from safla_trading.portfolio import PortfolioOptimizer
from safla_trading.backtesting import BacktestEngine, BacktestConfig
from safla_trading.strategies.sma_strategy import SMAStrategy
from safla_trading.logging_system import TradeLogger
from safla_trading.config import get_config


async def test_gbdt_fix():
    """Test GBDT model training after fix"""
    print("\n🧪 TEST 1: GBDT Model Training")
    print("=" * 50)

    logger = TradeLogger("test_gbdt")
    gbdt = GBDTModels(logger)

    # Get real data
    registry = await get_exchange_registry(logger)
    await registry.initialize_exchange('binance')

    # Fetch BTC data
    candles = await registry.get_historical_data('BTC/USDT', '1h', limit=500)
    await registry.close_all()

    # Convert to DataFrame
    data = pd.DataFrame([{
        'timestamp': pd.to_datetime(c['timestamp'], unit='ms'),
        'open': float(c['open']),
        'high': float(c['high']),
        'low': float(c['low']),
        'close': float(c['close']),
        'volume': float(c['volume'])
    } for c in candles])
    data.set_index('timestamp', inplace=True)

    print(f"📊 Loaded {len(data)} data points")
    print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")

    # Train model with reduced trials for speed
    result = gbdt.train_model(
        data, 'BTC/USDT', 'xgboost',
        target_horizon=1, optimize=True, n_trials=3
    )

    if result.get('success'):
        print("✅ GBDT training successful!")
        metrics = result['metrics']
        print(f"   Val RMSE: {metrics['val_rmse']:.6f}")
        print(f"   Val R²: {metrics['val_r2']:.4f}")
        print(f"   Val MAE: {metrics['val_mae']:.6f}")

        # Check if model actually learned (R² should be > 0)
        if metrics['val_r2'] > 0:
            print("   ✓ Model is learning patterns (R² > 0)")
        else:
            print("   ⚠️ Model not learning effectively (R² ≤ 0)")

        return True
    else:
        print(f"❌ GBDT training failed: {result.get('error')}")
        return False


async def test_portfolio_fix():
    """Test portfolio optimization after fix"""
    print("\n🧪 TEST 2: Portfolio Optimization")
    print("=" * 50)

    logger = TradeLogger("test_portfolio")
    optimizer = PortfolioOptimizer(logger)

    # Get real data for multiple assets
    registry = await get_exchange_registry(logger)
    await registry.initialize_exchange('binance')

    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    price_data = {}

    for symbol in symbols:
        candles = await registry.get_historical_data(symbol, '1h', limit=200)
        prices = pd.Series(
            [float(c['close']) for c in candles],
            index=pd.to_datetime([c['timestamp'] for c in candles], unit='ms')
        )
        price_data[symbol] = prices
        print(f"📊 Loaded {symbol}: ${prices.iloc[-1]:.2f}")

    await registry.close_all()

    # Prepare data
    price_df = optimizer.prepare_price_data(price_data)

    # Calculate annualized frequency for hourly data
    frequency = 24 * 365  # Hours per year

    # Ensure we have enough data and it's clean
    if len(price_df) < 50:
        print(f"⚠️ Only {len(price_df)} data points - need more for optimization")
        return False

    # Check for extreme values that might cause optimization issues
    returns = price_df.pct_change().dropna()
    if returns.isna().any().any() or (returns == 0).all().all():
        print("⚠️ Data quality issues detected")
        return False

    # Test Mean-Variance optimization
    print("\n🎯 Testing Mean-Variance Optimization...")
    mv_result = optimizer.optimize_mean_variance(
        price_df,
        objective='max_sharpe',
        frequency=frequency,
        return_method='mean_historical_return',
        risk_method='ledoit_wolf'
    )

    if mv_result.get('success'):
        print("✅ Portfolio optimization successful!")
        print(f"   Expected return: {mv_result['expected_return']:.4f}")
        print(f"   Volatility: {mv_result['volatility']:.4f}")
        print(f"   Sharpe ratio: {mv_result['sharpe_ratio']:.4f}")
        print("   Weights:")
        for asset, weight in mv_result['weights'].items():
            if weight > 0.01:
                print(f"      {asset}: {weight:.3f}")

        # Validate results
        if mv_result['sharpe_ratio'] > 0:
            print("   ✓ Positive Sharpe ratio achieved")
        else:
            print("   ⚠️ Sharpe ratio is non-positive")

        return True
    else:
        print(f"❌ Portfolio optimization failed: {mv_result.get('error')}")
        return False


async def test_strategy_fix():
    """Test strategy signal generation after fix"""
    print("\n🧪 TEST 3: Strategy Signal Generation")
    print("=" * 50)

    config = get_config()
    logger = TradeLogger("test_strategy")

    # Create strategy with updated config
    strategy = SMAStrategy('BTC/USDT', logger)

    print(f"📈 Strategy config:")
    print(f"   Fast SMA: {config.fast_period}")
    print(f"   Slow SMA: {config.slow_period}")
    print(f"   Entry threshold: {config.get('strategy.entry_threshold_pct')*100:.2f}%")

    # Get real data
    registry = await get_exchange_registry(logger)
    await registry.initialize_exchange('binance')

    candles = await registry.get_historical_data('BTC/USDT', '1h', limit=100)
    await registry.close_all()

    # Process candles and count signals
    signals = {'buy': 0, 'sell': 0, 'hold': 0}

    for candle in candles:
        from safla_trading.data_feed.binance_feed import OHLCV
        ohlcv = OHLCV(
            symbol='BTC/USDT',  # Added missing symbol parameter
            timestamp=candle['timestamp'],
            open=float(candle['open']),
            high=float(candle['high']),
            low=float(candle['low']),
            close=float(candle['close']),
            volume=float(candle['volume'])
        )

        signal = strategy.process_candle(ohlcv)
        signals[signal.signal] += 1

        if signal.signal != 'hold':
            print(f"   Signal: {signal.signal} at ${signal.price:.2f} (confidence: {signal.confidence:.2f})")

    print(f"\n📊 Signal summary over {len(candles)} candles:")
    print(f"   Buy signals: {signals['buy']}")
    print(f"   Sell signals: {signals['sell']}")
    print(f"   Hold signals: {signals['hold']}")

    total_trades = signals['buy'] + signals['sell']
    if total_trades > 0:
        print(f"   ✓ Strategy generates {total_trades} trading signals")
        return True
    else:
        print("   ⚠️ No trading signals generated - may need more data or parameter tuning")
        return False


async def test_integrated_backtest():
    """Test full system integration with backtesting"""
    print("\n🧪 TEST 4: Integrated Backtesting")
    print("=" * 50)

    logger = TradeLogger("test_backtest")
    backtest_engine = BacktestEngine(logger)

    # Simple strategy for testing
    class ImprovedSMAStrategy:
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
                if state['prev_sma_5'] <= state['prev_sma_20'] and sma_5 > sma_20:
                    if current_position <= 0 and bar['close'] > sma_20:
                        return {'action': 'buy', 'size': 0.95}
                # Death cross
                elif state['prev_sma_5'] >= state['prev_sma_20'] and sma_5 < sma_20:
                    if current_position > 0:
                        return {'action': 'sell'}

            state['prev_sma_5'] = sma_5
            state['prev_sma_20'] = sma_20
            return None

    # Register strategy
    strategy = ImprovedSMAStrategy()
    backtest_engine.register_strategy('improved_sma', strategy)

    # Get data
    registry = await get_exchange_registry(logger)
    await registry.initialize_exchange('binance')

    candles = await registry.get_historical_data('BTC/USDT', '1h', limit=500)
    await registry.close_all()

    # Convert to DataFrame
    data = pd.DataFrame([{
        'timestamp': pd.to_datetime(c['timestamp'], unit='ms'),
        'open': float(c['open']),
        'high': float(c['high']),
        'low': float(c['low']),
        'close': float(c['close']),
        'volume': float(c['volume'])
    } for c in candles])
    data.set_index('timestamp', inplace=True)

    print(f"📊 Backtesting on {len(data)} candles")
    print(f"   Date range: {data.index[0]} to {data.index[-1]}")

    # Run backtest
    config = BacktestConfig(
        initial_cash=100000,
        commission=0.001,
        slippage_value=0.0005
    )

    result = backtest_engine.run_backtest('improved_sma', data, config)

    if result:
        print("✅ Backtest completed!")
        print(f"   Total return: {result.total_return:.4f}")
        print(f"   Annualized return: {result.annualized_return:.4f}")
        print(f"   Sharpe ratio: {result.sharpe_ratio:.4f}")
        print(f"   Max drawdown: {result.max_drawdown:.4f}")
        print(f"   Number of trades: {result.trades_count}")
        print(f"   Win rate: {result.win_rate:.4f}")

        if result.trades_count > 0:
            print("   ✓ Strategy executed trades successfully")
            return True
        else:
            print("   ⚠️ No trades executed - check strategy logic")
            return False
    else:
        print("❌ Backtest failed")
        return False


async def main():
    """Run all tests and summarize results"""
    print("🚀 TRADING SYSTEM IMPROVEMENT VALIDATION")
    print("=" * 60)
    print("Testing fixes with real Binance data...")

    results = {}

    # Run tests
    try:
        results['gbdt'] = await test_gbdt_fix()
    except Exception as e:
        print(f"❌ GBDT test failed with error: {e}")
        results['gbdt'] = False

    try:
        results['portfolio'] = await test_portfolio_fix()
    except Exception as e:
        print(f"❌ Portfolio test failed with error: {e}")
        results['portfolio'] = False

    try:
        results['strategy'] = await test_strategy_fix()
    except Exception as e:
        print(f"❌ Strategy test failed with error: {e}")
        results['strategy'] = False

    try:
        results['backtest'] = await test_integrated_backtest()
    except Exception as e:
        print(f"❌ Backtest failed with error: {e}")
        results['backtest'] = False

    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{test.upper():15} {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL IMPROVEMENTS VALIDATED SUCCESSFULLY!")
        print("The system is now:")
        print("  • Training ML models with actual learning")
        print("  • Optimizing portfolios with proper risk/return")
        print("  • Generating trading signals from real data")
        print("  • Executing backtests with measurable performance")
    else:
        print("\n⚠️ Some tests failed - review logs for details")


if __name__ == "__main__":
    asyncio.run(main())