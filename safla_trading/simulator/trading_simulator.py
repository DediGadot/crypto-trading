"""
TRADING SIMULATOR THAT ACTUALLY SIMULATES TRADING
Not a philosophical exploration of AI consciousness
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import time

from ..config.config_loader import get_config
from ..logging_system import TradeLogger
from ..data_feed import BinanceDataFeed
from ..strategies import SMAStrategy
from .risk_manager import RiskManager, Position


@dataclass
class SimulatedTrade:
    """Simulated trade execution"""
    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    slippage: float = 0.0


@dataclass
class PerformanceMetrics:
    """Trading performance metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float


class TradingSimulator:
    """
    TRADING SIMULATOR THAT ACTUALLY WORKS
    Simulates real trading with real market data and real slippage
    No "emotional valence" or "self-awareness" - just P&L
    """

    def __init__(self, symbol: str = None):
        """Initialize trading simulator

        Args:
            symbol: Trading symbol (defaults to config)
        """
        self.config = get_config()
        self.symbol = symbol or self.config.test_symbol

        # Initialize components
        session_id = f"sim_{int(time.time())}"
        self.logger = TradeLogger(session_id)
        self.data_feed = BinanceDataFeed(self.logger)
        self.strategy = SMAStrategy(self.symbol, self.logger)
        self.risk_manager = RiskManager(self.config.initial_balance, self.logger)

        # Simulation state
        self.balance = self.config.initial_balance
        self.trades: List[SimulatedTrade] = []
        self.trade_counter = 0

        # Performance tracking
        self.start_time = datetime.now()
        self.total_commission_paid = 0.0
        self.pnl_history: List[float] = []

        # Simulation parameters
        self.commission_rate = self.config.commission_rate
        self.slippage_config = self.config.get_section('simulation')['slippage']

        self.logger.log_system_event(
            'simulator', 'initialized',
            {
                'symbol': self.symbol,
                'initial_balance': self.balance,
                'commission_rate': self.commission_rate,
                'session_id': session_id
            }
        )

    async def run_backtest(self, start_date: datetime = None,
                          end_date: datetime = None,
                          speed_multiplier: float = float('inf')) -> PerformanceMetrics:
        """Run backtest simulation

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            speed_multiplier: Simulation speed (inf = fastest)

        Returns:
            Performance metrics
        """
        # Default to config dates if not provided
        if start_date is None:
            start_date = datetime.strptime(self.config.get('testing.start_date'), '%Y-%m-%d')
        if end_date is None:
            end_date = datetime.strptime(self.config.get('testing.end_date'), '%Y-%m-%d')

        self.logger.log_system_event(
            'simulator', 'backtest_started',
            {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'speed_multiplier': speed_multiplier
            }
        )

        try:
            # Stream historical data as live simulation
            candle_count = 0
            async for candle in self.data_feed.stream_historical_as_live(
                self.symbol,
                self.config.get('market_data.timeframe'),
                start_date,
                end_date,
                speed_multiplier
            ):
                await self._process_candle(candle)
                candle_count += 1

                # Log progress periodically
                if candle_count % 1000 == 0:
                    self.logger.log_system_event(
                        'simulator', 'backtest_progress',
                        {
                            'candles_processed': candle_count,
                            'current_balance': self.balance,
                            'total_trades': len(self.trades)
                        }
                    )

            # Calculate final performance
            performance = self._calculate_performance_metrics()

            self.logger.log_system_event(
                'simulator', 'backtest_completed',
                {
                    'candles_processed': candle_count,
                    'final_balance': self.balance,
                    'total_trades': len(self.trades),
                    'performance': performance.__dict__
                }
            )

            return performance

        except Exception as e:
            self.logger.log_error(
                'simulator', 'backtest_error',
                f"Backtest failed: {e}",
                exception=e
            )
            raise

        finally:
            await self.data_feed.close()

    async def _process_candle(self, candle):
        """Process single market candle

        Args:
            candle: OHLCV candle data
        """
        # Update strategy with current position
        current_position = self._get_current_position()
        self.strategy.update_position(current_position)

        # Generate trading signal
        signal = self.strategy.process_candle(candle)

        # Update risk manager with current prices
        self.risk_manager.update_positions({self.symbol: candle.close})

        # Check for automatic exits (stop loss / take profit)
        await self._check_automatic_exits(candle.close)

        # Process signal if not hold
        if signal.signal != 'hold' and signal.quantity > 0:
            await self._execute_signal(signal, candle.close)

        # Log performance periodically
        if len(self.trades) > 0 and len(self.trades) % self.config.get('metrics.report_interval_trades') == 0:
            self._log_performance_update()

    async def _execute_signal(self, signal, current_price: float):
        """Execute trading signal

        Args:
            signal: Trading signal from strategy
            current_price: Current market price
        """
        # Check risk limits
        risk_check = self.risk_manager.check_trade_risk(signal, self.balance)

        if not risk_check.allowed:
            self.logger.log_decision(
                'risk_manager', 'rejected',
                signal.symbol,
                {'rejection_reason': risk_check.reason},
                {'price': current_price},
                'trade_rejected'
            )
            return

        # Use adjusted quantity if risk manager modified it
        quantity = risk_check.adjusted_quantity

        # Simulate trade execution
        executed_trade = await self._simulate_trade_execution(
            signal.symbol, signal.signal, quantity, current_price
        )

        if executed_trade:
            # Update positions
            if signal.signal == 'buy':
                self.risk_manager.open_position(
                    signal.symbol, quantity, executed_trade.price
                )
                self.balance -= executed_trade.price * quantity + executed_trade.commission

            elif signal.signal == 'sell':
                realized_pnl = self.risk_manager.close_position(
                    signal.symbol, executed_trade.price
                )
                self.balance += executed_trade.price * quantity - executed_trade.commission

                if realized_pnl is not None:
                    self.pnl_history.append(realized_pnl)

            # Log the trade
            self.logger.log_trade(
                'market', signal.symbol, signal.signal,
                executed_trade.price, quantity, executed_trade.commission,
                executed_trade.trade_id, 0  # No execution latency in simulation
            )

    async def _simulate_trade_execution(self, symbol: str, side: str,
                                      quantity: float, market_price: float) -> Optional[SimulatedTrade]:
        """Simulate realistic trade execution with slippage

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Trade quantity
            market_price: Current market price

        Returns:
            Simulated trade or None if execution failed
        """
        # Calculate slippage
        trade_value = quantity * market_price
        slippage = self._calculate_slippage(trade_value)

        # Apply slippage
        if side == 'buy':
            execution_price = market_price * (1 + slippage)
        else:
            execution_price = market_price * (1 - slippage)

        # Calculate commission
        commission = trade_value * self.commission_rate
        self.total_commission_paid += commission

        # Create trade record
        self.trade_counter += 1
        trade = SimulatedTrade(
            trade_id=f"SIM_{self.trade_counter:06d}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=execution_price,
            commission=commission,
            timestamp=datetime.now(),
            slippage=slippage
        )

        self.trades.append(trade)
        return trade

    def _calculate_slippage(self, trade_value_usd: float) -> float:
        """Calculate realistic slippage based on trade size

        Args:
            trade_value_usd: Trade value in USD

        Returns:
            Slippage as decimal (e.g., 0.001 = 0.1%)
        """
        base_slippage = self.slippage_config['base_bps'] / 10000  # Convert bps to decimal
        impact_coefficient = self.slippage_config['impact_coefficient']
        max_slippage = self.slippage_config['max_slippage_bps'] / 10000

        # Market impact based on trade size
        market_impact = (trade_value_usd / 1000) * impact_coefficient / 10000

        total_slippage = base_slippage + market_impact
        return min(total_slippage, max_slippage)

    async def _check_automatic_exits(self, current_price: float):
        """Check for automatic position exits

        Args:
            current_price: Current market price
        """
        stop_loss_triggers, take_profit_triggers = self.risk_manager.update_positions(
            {self.symbol: current_price}
        )

        # Execute stop loss exits
        for symbol in stop_loss_triggers:
            if symbol in self.risk_manager.positions:
                position = self.risk_manager.positions[symbol]
                await self._execute_exit(position, current_price, 'stop_loss')

        # Execute take profit exits
        for symbol in take_profit_triggers:
            if symbol in self.risk_manager.positions:
                position = self.risk_manager.positions[symbol]
                await self._execute_exit(position, current_price, 'take_profit')

    async def _execute_exit(self, position: Position, current_price: float, exit_type: str):
        """Execute position exit

        Args:
            position: Position to exit
            current_price: Current market price
            exit_type: 'stop_loss' or 'take_profit'
        """
        # Create exit signal
        from ..strategies.sma_strategy import TradingSignal

        signal = TradingSignal(
            symbol=position.symbol,
            signal='sell',
            price=current_price,
            quantity=abs(position.quantity),
            confidence=1.0,  # Automatic exit
            reason={'exit_type': exit_type}
        )

        await self._execute_signal(signal, current_price)

    def _get_current_position(self) -> float:
        """Get current position size for symbol

        Returns:
            Position quantity (positive for long, negative for short)
        """
        if self.symbol in self.risk_manager.positions:
            return self.risk_manager.positions[self.symbol].quantity
        return 0.0

    def _log_performance_update(self):
        """Log current performance metrics"""
        portfolio_summary = self.risk_manager.get_portfolio_summary()

        self.logger.log_performance(
            balance=self.balance,
            unrealized_pnl=portfolio_summary['unrealized_pnl'],
            realized_pnl=portfolio_summary['realized_pnl'],
            positions={k: v.__dict__ for k, v in self.risk_manager.positions.items()},
            metrics=self._get_current_metrics()
        )

    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics

        Returns:
            Current metrics dictionary
        """
        if not self.pnl_history:
            return {}

        winning_trades = sum(1 for pnl in self.pnl_history if pnl > 0)
        total_trades = len(self.pnl_history)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        total_pnl = sum(self.pnl_history)
        portfolio_summary = self.risk_manager.get_portfolio_summary()

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'current_drawdown': portfolio_summary['current_drawdown_pct'],
            'commission_paid': self.total_commission_paid
        }

    def _calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate final performance metrics

        Returns:
            Performance metrics
        """
        if not self.pnl_history:
            return PerformanceMetrics(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0.0, total_pnl=0.0, max_drawdown=0.0,
                sharpe_ratio=0.0, profit_factor=0.0
            )

        total_trades = len(self.pnl_history)
        winning_trades = sum(1 for pnl in self.pnl_history if pnl > 0)
        losing_trades = sum(1 for pnl in self.pnl_history if pnl < 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        total_pnl = sum(self.pnl_history)

        # Calculate max drawdown
        running_max = 0
        max_drawdown = 0
        cumulative_pnl = 0

        for pnl in self.pnl_history:
            cumulative_pnl += pnl
            running_max = max(running_max, cumulative_pnl)
            drawdown = running_max - cumulative_pnl
            max_drawdown = max(max_drawdown, drawdown)

        # Calculate Sharpe ratio (simplified)
        if len(self.pnl_history) > 1:
            returns = np.array(self.pnl_history)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0

        # Calculate profit factor
        gross_profit = sum(pnl for pnl in self.pnl_history if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in self.pnl_history if pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor
        )

    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get comprehensive simulation summary

        Returns:
            Simulation summary
        """
        performance = self._calculate_performance_metrics()
        portfolio = self.risk_manager.get_portfolio_summary()
        runtime = datetime.now() - self.start_time

        return {
            'simulation_info': {
                'symbol': self.symbol,
                'start_time': self.start_time.isoformat(),
                'runtime_seconds': runtime.total_seconds(),
                'initial_balance': self.config.initial_balance,
                'final_balance': self.balance
            },
            'performance': performance.__dict__,
            'portfolio': portfolio,
            'commission_paid': self.total_commission_paid,
            'strategy_state': self.strategy.get_strategy_state()
        }

    async def close(self):
        """Close simulator and cleanup resources"""
        self.logger.close()
        if hasattr(self.data_feed, 'close'):
            await self.data_feed.close()