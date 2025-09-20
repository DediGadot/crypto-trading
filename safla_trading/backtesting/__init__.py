"""
BACKTESTING MODULE
Event-driven backtesting with proper slippage and fees
"""

from .backtest_engine import BacktestEngine, BacktestConfig, BacktestResult, StrategyAdapter

__all__ = [
    'BacktestEngine',
    'BacktestConfig',
    'BacktestResult',
    'StrategyAdapter'
]