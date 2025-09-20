#!/usr/bin/env python3
"""
DYNAMIC PORTFOLIO MANAGER WITH KELLY CRITERION
Optimal position sizing for maximum long-term growth

"Fortune favors the bold, but statistics favor the smart." - Larry Williams
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from ultra_alpha_strategy import UltraSignal
from linus_trading_system import TradingSignal


@dataclass
class PositionInfo:
    """Information about an active position"""
    symbol: str
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    expected_return: float
    risk_score: float
    current_pnl: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class PortfolioMetrics:
    """Portfolio-level risk metrics"""
    total_value: float
    total_exposure: float
    leverage_ratio: float
    var_95: float  # Value at Risk 95%
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    kelly_fraction: float


class KellyCriterionCalculator:
    """
    Calculate optimal position sizes using Kelly Criterion
    Maximizes long-term growth while controlling risk
    """

    def __init__(self, lookback_periods: int = 100):
        self.lookback_periods = lookback_periods
        self.trade_history = []
        self.win_rate_cache = {}
        self.avg_win_cache = {}
        self.avg_loss_cache = {}

    def calculate_kelly_fraction(self,
                                win_rate: float,
                                avg_win: float,
                                avg_loss: float,
                                confidence: float = 1.0) -> float:
        """
        Calculate Kelly fraction for optimal position sizing

        Kelly% = (bp - q) / b
        where:
        b = avg_win / avg_loss (odds received)
        p = win_rate (probability of winning)
        q = 1 - p (probability of losing)

        Args:
            win_rate: Historical win rate (0 to 1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive value)
            confidence: Signal confidence multiplier

        Returns:
            Kelly fraction (0 to 1)
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.05  # Conservative fallback

        # Calculate odds ratio
        b = avg_win / avg_loss

        # Kelly formula
        kelly_fraction = (b * win_rate - (1 - win_rate)) / b

        # Apply confidence multiplier
        kelly_fraction *= confidence

        # Conservative bounds (never risk more than 25% on single trade)
        kelly_fraction = np.clip(kelly_fraction, 0.0, 0.25)

        # Fractional Kelly for safety (use 25% of full Kelly)
        return kelly_fraction * 0.25

    def update_trade_history(self, pnl: float, trade_type: str = 'general'):
        """Update trade history for Kelly calculation"""
        self.trade_history.append({
            'timestamp': datetime.now(),
            'pnl': pnl,
            'type': trade_type
        })

        # Keep only recent history
        if len(self.trade_history) > self.lookback_periods:
            self.trade_history = self.trade_history[-self.lookback_periods:]

        # Update cached metrics
        self._update_cached_metrics(trade_type)

    def _update_cached_metrics(self, trade_type: str):
        """Update cached win rate and average win/loss metrics"""
        if not self.trade_history:
            return

        # Filter by trade type if specified
        if trade_type != 'general':
            trades = [t for t in self.trade_history if t.get('type') == trade_type]
        else:
            trades = self.trade_history

        if not trades:
            return

        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p < 0]

        # Calculate metrics
        win_rate = len(wins) / len(pnls) if pnls else 0
        avg_win = np.mean(wins) if wins else 0.01
        avg_loss = np.mean(losses) if losses else 0.01

        # Cache results
        self.win_rate_cache[trade_type] = win_rate
        self.avg_win_cache[trade_type] = avg_win
        self.avg_loss_cache[trade_type] = avg_loss

    def get_cached_metrics(self, trade_type: str = 'general') -> Tuple[float, float, float]:
        """Get cached win rate and average win/loss"""
        win_rate = self.win_rate_cache.get(trade_type, 0.5)  # Default 50%
        avg_win = self.avg_win_cache.get(trade_type, 0.02)   # Default 2%
        avg_loss = self.avg_loss_cache.get(trade_type, 0.015) # Default 1.5%

        return win_rate, avg_win, avg_loss


class DynamicPortfolioManager:
    """
    Dynamic portfolio manager with Kelly-optimal position sizing
    Manages multiple positions with risk overlay
    """

    def __init__(self,
                 initial_capital: float = 100000,
                 max_portfolio_risk: float = 0.20,  # 20% portfolio risk
                 max_single_position: float = 0.25,  # 25% max in single position
                 max_leverage: float = 3.0,          # 3x max leverage
                 stop_loss_pct: float = 0.03,        # 3% stop loss
                 take_profit_pct: float = 0.08):     # 8% take profit

        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_portfolio_risk = max_portfolio_risk
        self.max_single_position = max_single_position
        self.max_leverage = max_leverage
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # Portfolio state
        self.active_positions: Dict[str, PositionInfo] = {}
        self.cash = initial_capital
        self.total_equity = initial_capital

        # Risk management
        self.kelly_calculator = KellyCriterionCalculator()
        self.risk_metrics = PortfolioMetrics(
            total_value=initial_capital,
            total_exposure=0.0,
            leverage_ratio=1.0,
            var_95=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            win_rate=0.5,
            profit_factor=1.0,
            kelly_fraction=0.05
        )

        # Performance tracking
        self.equity_curve = [initial_capital]
        self.trade_log = []
        self.daily_returns = []

    def calculate_position_size(self,
                              signal: UltraSignal,
                              current_price: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion

        Args:
            signal: Trading signal with confidence and expected return
            current_price: Current market price

        Returns:
            Position size in base currency units
        """
        # Get historical performance metrics
        win_rate, avg_win, avg_loss = self.kelly_calculator.get_cached_metrics(signal.timeframe)

        # Calculate Kelly fraction
        kelly_fraction = self.kelly_calculator.calculate_kelly_fraction(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            confidence=signal.confidence
        )

        # Adjust for signal strength and expected return
        signal_multiplier = min(signal.expected_return * 10, 2.0)  # Cap at 2x
        kelly_fraction *= signal_multiplier

        # Apply portfolio constraints
        max_position_value = self.current_capital * self.max_single_position
        kelly_position_value = self.current_capital * kelly_fraction

        # Use the more conservative of Kelly and portfolio limit
        target_position_value = min(kelly_position_value, max_position_value)

        # Apply leverage if signal confidence is high
        if signal.confidence > 0.8:
            leverage_multiplier = min(signal.size_multiplier, self.max_leverage)
            target_position_value *= leverage_multiplier

        # Convert to position size in units
        position_size = target_position_value / current_price

        # Final risk checks
        if self._check_portfolio_risk_limits(target_position_value):
            return position_size
        else:
            # Reduce position size to meet risk limits
            return position_size * 0.5

    def open_position(self,
                     signal: UltraSignal,
                     current_price: float) -> bool:
        """
        Open new position based on signal

        Args:
            signal: Trading signal
            current_price: Current market price

        Returns:
            True if position opened successfully
        """
        if signal.symbol in self.active_positions:
            return False  # Already have position in this symbol

        # Calculate position size
        position_size = self.calculate_position_size(signal, current_price)

        if position_size <= 0:
            return False

        # Calculate position value
        position_value = position_size * current_price

        # Check if we have enough cash (considering leverage)
        required_cash = position_value / self.max_leverage
        if required_cash > self.cash:
            return False

        # Calculate stop loss and take profit
        if signal.action == 'buy':
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)
        else:  # sell/short
            stop_loss = current_price * (1 + self.stop_loss_pct)
            take_profit = current_price * (1 - self.take_profit_pct)
            position_size = -position_size  # Negative for short

        # Create position
        position = PositionInfo(
            symbol=signal.symbol,
            size=position_size,
            entry_price=current_price,
            entry_time=signal.timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
            expected_return=signal.expected_return,
            risk_score=signal.risk_score
        )

        # Update portfolio
        self.active_positions[signal.symbol] = position
        self.cash -= required_cash

        # Log trade
        self.trade_log.append({
            'timestamp': signal.timestamp,
            'symbol': signal.symbol,
            'action': 'OPEN_' + signal.action.upper(),
            'size': position_size,
            'price': current_price,
            'confidence': signal.confidence,
            'expected_return': signal.expected_return
        })

        return True

    def update_positions(self, current_prices: Dict[str, float]) -> List[Dict]:
        """
        Update all positions with current prices and check for exits

        Args:
            current_prices: Dictionary of symbol -> current price

        Returns:
            List of exit signals generated
        """
        exit_signals = []

        for symbol, position in list(self.active_positions.items()):
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]

            # Update unrealized P&L
            if position.size > 0:  # Long position
                position.unrealized_pnl = (current_price - position.entry_price) * position.size
            else:  # Short position
                position.unrealized_pnl = (position.entry_price - current_price) * abs(position.size)

            # Check exit conditions
            should_exit, exit_reason = self._check_exit_conditions(position, current_price)

            if should_exit:
                # Close position
                exit_signal = self._close_position(symbol, current_price, exit_reason)
                if exit_signal:
                    exit_signals.append(exit_signal)

        # Update portfolio metrics
        self._update_portfolio_metrics(current_prices)

        return exit_signals

    def _check_exit_conditions(self, position: PositionInfo, current_price: float) -> Tuple[bool, str]:
        """Check if position should be exited"""

        # Stop loss check
        if position.size > 0:  # Long position
            if current_price <= position.stop_loss:
                return True, "STOP_LOSS"
            if current_price >= position.take_profit:
                return True, "TAKE_PROFIT"
        else:  # Short position
            if current_price >= position.stop_loss:
                return True, "STOP_LOSS"
            if current_price <= position.take_profit:
                return True, "TAKE_PROFIT"

        # Time-based exit (close after 24 hours if no profit)
        time_elapsed = datetime.now() - position.entry_time
        if time_elapsed > timedelta(hours=24) and position.unrealized_pnl <= 0:
            return True, "TIME_EXIT"

        # Risk-based exit (if position risk too high)
        position_risk = abs(position.unrealized_pnl) / self.current_capital
        if position_risk > self.max_single_position:
            return True, "RISK_EXIT"

        return False, ""

    def _close_position(self, symbol: str, current_price: float, reason: str) -> Optional[Dict]:
        """Close position and update portfolio"""
        if symbol not in self.active_positions:
            return None

        position = self.active_positions[symbol]

        # Calculate realized P&L
        realized_pnl = position.unrealized_pnl

        # Update cash
        position_value = abs(position.size) * current_price
        self.cash += position_value + realized_pnl

        # Update capital
        self.current_capital += realized_pnl

        # Update Kelly calculator
        return_pct = realized_pnl / (abs(position.size) * position.entry_price)
        self.kelly_calculator.update_trade_history(return_pct, 'ultra_alpha')

        # Log trade
        trade_log = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': f'CLOSE_{reason}',
            'size': position.size,
            'entry_price': position.entry_price,
            'exit_price': current_price,
            'pnl': realized_pnl,
            'return_pct': return_pct,
            'hold_time': datetime.now() - position.entry_time
        }

        self.trade_log.append(trade_log)

        # Remove from active positions
        del self.active_positions[symbol]

        return trade_log

    def _check_portfolio_risk_limits(self, new_position_value: float) -> bool:
        """Check if new position would exceed portfolio risk limits"""

        # Calculate total exposure including new position
        current_exposure = sum(abs(pos.size * pos.entry_price) for pos in self.active_positions.values())
        total_exposure = current_exposure + new_position_value

        # Check portfolio risk limit
        portfolio_risk = total_exposure / self.current_capital
        if portfolio_risk > self.max_portfolio_risk * self.max_leverage:
            return False

        # Check leverage limit
        leverage_ratio = total_exposure / self.current_capital
        if leverage_ratio > self.max_leverage:
            return False

        return True

    def _update_portfolio_metrics(self, current_prices: Dict[str, float]):
        """Update portfolio-level risk metrics"""

        # Calculate total portfolio value
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.active_positions.values())
        self.total_equity = self.cash + total_unrealized_pnl

        # Update equity curve
        self.equity_curve.append(self.total_equity)

        # Calculate total exposure
        total_exposure = sum(
            abs(pos.size * current_prices.get(pos.symbol, pos.entry_price))
            for pos in self.active_positions.values()
        )

        # Update risk metrics
        self.risk_metrics.total_value = self.total_equity
        self.risk_metrics.total_exposure = total_exposure
        self.risk_metrics.leverage_ratio = total_exposure / self.total_equity if self.total_equity > 0 else 1.0

        # Calculate performance metrics
        if len(self.equity_curve) > 1:
            returns = pd.Series(self.equity_curve).pct_change().dropna()

            if len(returns) > 0:
                self.risk_metrics.sharpe_ratio = returns.mean() / returns.std() * np.sqrt(365*24) if returns.std() > 0 else 0
                self.risk_metrics.max_drawdown = self._calculate_max_drawdown()
                self.risk_metrics.var_95 = returns.quantile(0.05) * self.total_equity

        # Calculate trade-based metrics
        if self.trade_log:
            closed_trades = [t for t in self.trade_log if 'CLOSE' in t.get('action', '')]
            if closed_trades:
                winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
                losing_trades = [t for t in closed_trades if t.get('pnl', 0) < 0]

                self.risk_metrics.win_rate = len(winning_trades) / len(closed_trades)

                if winning_trades and losing_trades:
                    avg_win = np.mean([t['pnl'] for t in winning_trades])
                    avg_loss = abs(np.mean([t['pnl'] for t in losing_trades]))
                    self.risk_metrics.profit_factor = avg_win / avg_loss if avg_loss > 0 else 1.0

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve"""
        if len(self.equity_curve) < 2:
            return 0.0

        equity_series = pd.Series(self.equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max

        return drawdown.min()

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        return {
            'capital': {
                'initial': self.initial_capital,
                'current': self.current_capital,
                'total_equity': self.total_equity,
                'cash': self.cash,
                'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.active_positions.values())
            },
            'positions': {
                'active_count': len(self.active_positions),
                'total_exposure': self.risk_metrics.total_exposure,
                'leverage_ratio': self.risk_metrics.leverage_ratio,
                'positions': [
                    {
                        'symbol': pos.symbol,
                        'size': pos.size,
                        'entry_price': pos.entry_price,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'return_pct': pos.unrealized_pnl / (abs(pos.size) * pos.entry_price) * 100
                    }
                    for pos in self.active_positions.values()
                ]
            },
            'performance': {
                'total_return': (self.total_equity - self.initial_capital) / self.initial_capital,
                'sharpe_ratio': self.risk_metrics.sharpe_ratio,
                'max_drawdown': self.risk_metrics.max_drawdown,
                'win_rate': self.risk_metrics.win_rate,
                'profit_factor': self.risk_metrics.profit_factor,
                'total_trades': len([t for t in self.trade_log if 'CLOSE' in t.get('action', '')])
            },
            'risk': {
                'var_95': self.risk_metrics.var_95,
                'portfolio_risk': self.risk_metrics.total_exposure / self.total_equity,
                'kelly_fraction': self.risk_metrics.kelly_fraction
            }
        }

    def convert_ultra_signal_to_trading_signal(self, ultra_signal: UltraSignal) -> TradingSignal:
        """Convert UltraSignal to standard TradingSignal for backtesting"""

        # Calculate position size
        position_size = self.calculate_position_size(ultra_signal, ultra_signal.price)

        return TradingSignal(
            timestamp=ultra_signal.timestamp,
            symbol=ultra_signal.symbol,
            action=ultra_signal.action,
            size=position_size,
            confidence=ultra_signal.confidence,
            reason=ultra_signal.reason,
            price=ultra_signal.price
        )


def create_dynamic_portfolio_manager(initial_capital: float = 100000,
                                   max_portfolio_risk: float = 0.20,
                                   max_leverage: float = 3.0) -> DynamicPortfolioManager:
    """
    Factory function for dynamic portfolio manager

    Args:
        initial_capital: Starting capital
        max_portfolio_risk: Maximum portfolio risk (20% = 0.20)
        max_leverage: Maximum leverage multiplier

    Returns:
        Configured DynamicPortfolioManager instance
    """
    return DynamicPortfolioManager(
        initial_capital=initial_capital,
        max_portfolio_risk=max_portfolio_risk,
        max_leverage=max_leverage
    )


if __name__ == "__main__":
    print("💰 DYNAMIC PORTFOLIO MANAGER INITIALIZED")
    print("Kelly Criterion optimal position sizing")
    print("=" * 50)

    # Test Kelly calculation
    kelly_calc = KellyCriterionCalculator()

    # Simulate some trade history
    for i in range(50):
        pnl = np.random.normal(0.02, 0.05)  # 2% mean return, 5% volatility
        kelly_calc.update_trade_history(pnl)

    win_rate, avg_win, avg_loss = kelly_calc.get_cached_metrics()
    kelly_fraction = kelly_calc.calculate_kelly_fraction(win_rate, avg_win, avg_loss, confidence=0.8)

    print(f"📊 Test Kelly Metrics:")
    print(f"   Win Rate: {win_rate:.1%}")
    print(f"   Avg Win: {avg_win:.1%}")
    print(f"   Avg Loss: {avg_loss:.1%}")
    print(f"   Kelly Fraction: {kelly_fraction:.1%}")

    # Test portfolio manager
    portfolio = create_dynamic_portfolio_manager()
    summary = portfolio.get_portfolio_summary()

    print(f"\n💼 Portfolio Initialized:")
    print(f"   Capital: ${summary['capital']['current']:,.2f}")
    print(f"   Max Risk: {portfolio.max_portfolio_risk:.1%}")
    print(f"   Max Leverage: {portfolio.max_leverage:.1f}x")

    print("✅ Dynamic portfolio manager ready for ultra-alpha trading")