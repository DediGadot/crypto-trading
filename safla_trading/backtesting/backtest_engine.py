"""
BACKTESTING ENGINE
Event-driven backtesting using Backtesting.py with proper slippage and fees
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field

try:
    from backtesting import Backtest, Strategy as BTStrategy
    from backtesting.lib import crossover
    BACKTESTING_AVAILABLE = True
except ImportError:
    BACKTESTING_AVAILABLE = False
    # Fallback classes
    class Backtest:
        pass
    class BTStrategy:
        pass

from ..config.config_loader import get_config
from ..logging_system import TradeLogger


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_cash: float = 100000.0
    commission: float = 0.002  # 0.2%
    margin: float = 1.0  # No leverage by default
    trade_on_close: bool = False
    hedging: bool = False
    exclusive_orders: bool = False
    # Slippage configuration
    slippage_type: str = 'percentage'  # 'percentage' or 'fixed'
    slippage_value: float = 0.001  # 0.1% or fixed amount
    # Risk management
    max_drawdown: float = 0.20  # 20%
    position_sizing: str = 'fixed'  # 'fixed', 'percentage', 'kelly'
    position_size: float = 0.10  # 10% of capital


@dataclass
class BacktestResult:
    """Backtesting results"""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_cash: float
    final_value: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    trades_count: int
    win_rate: float
    profit_factor: float
    avg_trade_duration: timedelta
    detailed_stats: Dict[str, Any] = field(default_factory=dict)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)


class StrategyAdapter(BTStrategy):
    """Adapter to bridge SAFLA strategies with Backtesting.py"""

    def init(self):
        """Initialize strategy"""
        # Get strategy implementation from parameters
        self.strategy_impl = self._strategy_impl
        self.config = self._config
        self.logger = self._logger

        # Initialize strategy state
        self.strategy_state = {}

        # Position sizing
        self.position_size = self.config.position_size

        # Initialize the strategy implementation
        if hasattr(self.strategy_impl, 'initialize'):
            self.strategy_impl.initialize(self.strategy_state)

    def next(self):
        """Process next bar"""
        try:
            # Current market data
            current_bar = {
                'timestamp': self.data.index[-1],
                'open': float(self.data.Open[-1]),
                'high': float(self.data.High[-1]),
                'low': float(self.data.Low[-1]),
                'close': float(self.data.Close[-1]),
                'volume': float(self.data.Volume[-1]) if hasattr(self.data, 'Volume') else 0
            }

            # Add technical indicators to bar if available
            if hasattr(self.data, 'SMA_20'):
                current_bar['sma_20'] = float(self.data.SMA_20[-1])
            if hasattr(self.data, 'SMA_50'):
                current_bar['sma_50'] = float(self.data.SMA_50[-1])

            # Get portfolio information
            portfolio_info = {
                'cash': self.broker.cash,
                'equity': self.broker.equity,
                'position_size': len(self.trades),
                'current_position': self.position.size if self.position else 0
            }

            # Call strategy implementation
            signals = self.strategy_impl.on_bar(current_bar, self.strategy_state, portfolio_info)

            # Process signals
            if signals:
                self._process_signals(signals, current_bar)

        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    'backtest_engine', 'strategy_error',
                    f"Strategy error: {e}",
                    exception=e
                )

    def _process_signals(self, signals: Dict[str, Any], current_bar: Dict[str, Any]):
        """Process trading signals

        Args:
            signals: Trading signals from strategy
            current_bar: Current market data
        """
        action = signals.get('action')
        size = signals.get('size', self.position_size)

        if action == 'buy' and not self.position:
            # Calculate position size
            if self.config.position_sizing == 'percentage':
                cash_to_use = self.broker.equity * size
                shares = int(cash_to_use / current_bar['close'])
            elif self.config.position_sizing == 'fixed':
                shares = int(size)
            else:
                shares = int(self.broker.equity * size / current_bar['close'])

            if shares > 0:
                self.buy(size=shares)

        elif action == 'sell' and self.position:
            self.sell()

        elif action == 'close' and self.position:
            self.position.close()


class BacktestEngine:
    """Advanced backtesting engine with SAFLA integration"""

    def __init__(self, logger: Optional[TradeLogger] = None):
        """Initialize backtest engine

        Args:
            logger: Trade logger instance
        """
        if not BACKTESTING_AVAILABLE:
            raise ImportError("backtesting is required. Install with: pip install backtesting")

        self.config = get_config()
        self.logger = logger

        # Strategy registry
        self.strategies: Dict[str, Any] = {}

        # Results storage
        self.results: List[BacktestResult] = []

    def register_strategy(self, name: str, strategy_impl: Any):
        """Register a strategy for backtesting

        Args:
            name: Strategy name
            strategy_impl: Strategy implementation
        """
        self.strategies[name] = strategy_impl

        if self.logger:
            self.logger.log_system_event(
                'backtest_engine', 'strategy_registered',
                {'strategy_name': name}
            )

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for backtesting

        Args:
            data: Raw OHLCV data

        Returns:
            Prepared data with technical indicators
        """
        # Ensure proper column names for Backtesting.py
        required_columns = ['Open', 'High', 'Low', 'Close']

        # Rename columns if needed
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }

        for old_col, new_col in column_mapping.items():
            if old_col in data.columns and new_col not in data.columns:
                data = data.rename(columns={old_col: new_col})

        # Check for required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Add technical indicators
        data = self._add_technical_indicators(data)

        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data = data.set_index('timestamp')
            else:
                data.index = pd.to_datetime(data.index)

        return data

    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to data

        Args:
            data: OHLCV data

        Returns:
            Data with technical indicators
        """
        # Simple moving averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()

        # Exponential moving averages
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()

        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()

        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)

        # Volume indicators (if volume available)
        if 'Volume' in data.columns:
            data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()

        return data

    def run_backtest(self, strategy_name: str, data: pd.DataFrame,
                    config: Optional[BacktestConfig] = None) -> BacktestResult:
        """Run backtest for a strategy

        Args:
            strategy_name: Name of registered strategy
            data: Historical data
            config: Backtest configuration

        Returns:
            Backtest results
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not registered")

        # Use default config if not provided
        if config is None:
            config = BacktestConfig()

        # Prepare data
        prepared_data = self.prepare_data(data)

        # Get strategy implementation
        strategy_impl = self.strategies[strategy_name]

        # Create adapter class with injected dependencies
        class ConfiguredAdapter(StrategyAdapter):
            _strategy_impl = strategy_impl
            _config = config
            _logger = self.logger

        try:
            # Run backtest
            bt = Backtest(
                prepared_data,
                ConfiguredAdapter,
                cash=config.initial_cash,
                commission=config.commission,
                margin=config.margin,
                trade_on_close=config.trade_on_close,
                hedging=config.hedging,
                exclusive_orders=config.exclusive_orders
            )

            # Run the backtest
            bt_result = bt.run()

            # Convert to our result format
            result = self._convert_result(
                strategy_name,
                bt_result,
                prepared_data,
                config
            )

            # Store result
            self.results.append(result)

            if self.logger:
                self.logger.log_system_event(
                    'backtest_engine', 'backtest_completed',
                    {
                        'strategy_name': strategy_name,
                        'total_return': result.total_return,
                        'sharpe_ratio': result.sharpe_ratio,
                        'max_drawdown': result.max_drawdown,
                        'trades_count': result.trades_count
                    }
                )

            return result

        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    'backtest_engine', 'backtest_failed',
                    f"Backtest failed for {strategy_name}: {e}",
                    exception=e
                )
            raise

    def _convert_result(self, strategy_name: str, bt_result: Any,
                       data: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
        """Convert backtesting.py result to our format

        Args:
            strategy_name: Strategy name
            bt_result: Backtesting.py result
            data: Historical data
            config: Backtest configuration

        Returns:
            Converted result
        """
        # Extract basic metrics
        start_date = data.index[0]
        end_date = data.index[-1]
        duration = end_date - start_date

        # Calculate metrics
        total_return = bt_result['Return [%]'] / 100
        annualized_return = (1 + total_return) ** (365.25 / duration.days) - 1

        # Get trades if available
        trades_df = pd.DataFrame()
        if hasattr(bt_result, '_trades') and bt_result._trades is not None:
            trades_df = bt_result._trades

        # Calculate additional metrics
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade_duration = timedelta()

        if not trades_df.empty and 'PnL' in trades_df.columns:
            winning_trades = trades_df[trades_df['PnL'] > 0]
            losing_trades = trades_df[trades_df['PnL'] < 0]

            win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0

            gross_profit = winning_trades['PnL'].sum() if not winning_trades.empty else 0
            gross_loss = abs(losing_trades['PnL'].sum()) if not losing_trades.empty else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            if 'EntryTime' in trades_df.columns and 'ExitTime' in trades_df.columns:
                durations = pd.to_datetime(trades_df['ExitTime']) - pd.to_datetime(trades_df['EntryTime'])
                avg_trade_duration = durations.mean()

        return BacktestResult(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            initial_cash=config.initial_cash,
            final_value=bt_result['Equity Final [$]'],
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=bt_result.get('Volatility [%]', 0) / 100,
            sharpe_ratio=bt_result.get('Sharpe Ratio', 0),
            max_drawdown=bt_result.get('Max. Drawdown [%]', 0) / 100,
            trades_count=len(trades_df),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_duration=avg_trade_duration,
            detailed_stats=dict(bt_result),
            trades=trades_df
        )

    def run_optimization(self, strategy_name: str, data: pd.DataFrame,
                        param_grid: Dict[str, List[Any]],
                        metric: str = 'sharpe_ratio',
                        config: Optional[BacktestConfig] = None) -> Dict[str, Any]:
        """Run parameter optimization

        Args:
            strategy_name: Strategy name
            data: Historical data
            param_grid: Parameter grid to optimize
            metric: Optimization metric
            config: Backtest configuration

        Returns:
            Optimization results
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not registered")

        # Use default config if not provided
        if config is None:
            config = BacktestConfig()

        # Prepare data
        prepared_data = self.prepare_data(data)

        # Get strategy implementation
        strategy_impl = self.strategies[strategy_name]

        # Create optimization results
        optimization_results = []

        # Generate parameter combinations
        import itertools
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        for param_combination in itertools.product(*param_values):
            params = dict(zip(param_names, param_combination))

            try:
                # Create strategy with parameters
                class ParameterizedAdapter(StrategyAdapter):
                    _strategy_impl = strategy_impl
                    _config = config
                    _logger = self.logger
                    _params = params

                    def init(self):
                        super().init()
                        # Set parameters in strategy state
                        self.strategy_state.update(self._params)

                # Run backtest
                bt = Backtest(
                    prepared_data,
                    ParameterizedAdapter,
                    cash=config.initial_cash,
                    commission=config.commission,
                    margin=config.margin,
                    trade_on_close=config.trade_on_close,
                    hedging=config.hedging,
                    exclusive_orders=config.exclusive_orders
                )

                bt_result = bt.run()

                # Extract metric value
                if metric == 'sharpe_ratio':
                    metric_value = bt_result.get('Sharpe Ratio', 0)
                elif metric == 'total_return':
                    metric_value = bt_result['Return [%]'] / 100
                elif metric == 'max_drawdown':
                    metric_value = -bt_result.get('Max. Drawdown [%]', 0) / 100  # Negative for minimization
                else:
                    metric_value = bt_result.get(metric, 0)

                optimization_results.append({
                    'parameters': params,
                    'metric_value': metric_value,
                    'result': bt_result
                })

            except Exception as e:
                if self.logger:
                    self.logger.log_error(
                        'backtest_engine', 'optimization_error',
                        f"Optimization error for params {params}: {e}",
                        context={'strategy': strategy_name, 'params': params}
                    )

        # Find best parameters
        if optimization_results:
            best_result = max(optimization_results, key=lambda x: x['metric_value'])

            if self.logger:
                self.logger.log_system_event(
                    'backtest_engine', 'optimization_completed',
                    {
                        'strategy_name': strategy_name,
                        'best_parameters': best_result['parameters'],
                        'best_metric_value': best_result['metric_value'],
                        'total_combinations': len(optimization_results)
                    }
                )

            return {
                'best_parameters': best_result['parameters'],
                'best_metric_value': best_result['metric_value'],
                'best_result': best_result['result'],
                'all_results': optimization_results
            }

        return {'error': 'No valid optimization results'}

    def get_results_summary(self) -> pd.DataFrame:
        """Get summary of all backtest results

        Returns:
            Summary DataFrame
        """
        if not self.results:
            return pd.DataFrame()

        summary_data = []
        for result in self.results:
            summary_data.append({
                'Strategy': result.strategy_name,
                'Start Date': result.start_date,
                'End Date': result.end_date,
                'Total Return': f"{result.total_return:.2%}",
                'Annualized Return': f"{result.annualized_return:.2%}",
                'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
                'Max Drawdown': f"{result.max_drawdown:.2%}",
                'Trades': result.trades_count,
                'Win Rate': f"{result.win_rate:.2%}",
                'Profit Factor': f"{result.profit_factor:.2f}"
            })

        return pd.DataFrame(summary_data)

    def clear_results(self):
        """Clear all stored results"""
        self.results.clear()

        if self.logger:
            self.logger.log_system_event(
                'backtest_engine', 'results_cleared',
                {'action': 'all results cleared'}
            )