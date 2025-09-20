"""
RISK MANAGEMENT SYSTEM
Real risk controls that prevent you from losing all your money
Not "confidence scores" and "emotional valence"
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import pytz

from ..config.config_loader import get_config
from ..logging_system import TradeLogger
from ..strategies.sma_strategy import TradingSignal


@dataclass
class Position:
    """Trading position"""
    symbol: str
    quantity: float  # Positive for long, negative for short
    entry_price: float
    entry_time: datetime
    current_price: float
    unrealized_pnl: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class RiskCheck:
    """Risk check result"""
    allowed: bool
    reason: str
    adjusted_quantity: float = 0.0
    risk_factors: Dict[str, Any] = None


class RiskManager:
    """
    RISK MANAGEMENT THAT ACTUALLY MANAGES RISK
    Hard limits to prevent catastrophic losses
    """

    def __init__(self, initial_balance: float, logger: Optional[TradeLogger] = None):
        """Initialize risk manager

        Args:
            initial_balance: Initial trading balance
            logger: Trade logger
        """
        self.config = get_config()
        self.logger = logger

        # Account state
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance

        # Positions
        self.positions: Dict[str, Position] = {}

        # Daily tracking
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.daily_reset_time = datetime.now(pytz.UTC).replace(hour=0, minute=0, second=0, microsecond=0)

        # Risk limits from config
        self.max_position_size_usd = self.config.get('strategy.max_position_size_usd')
        self.max_open_positions = self.config.max_open_positions
        self.max_portfolio_exposure = self.config.get('risk.max_portfolio_exposure_pct')
        self.stop_loss_pct = self.config.stop_loss_pct
        self.take_profit_pct = self.config.take_profit_pct
        self.max_daily_loss_pct = self.config.get('risk.max_daily_loss_pct')
        self.max_daily_trades = self.config.max_daily_trades
        self.max_drawdown_pct = self.config.get('risk.max_drawdown_pct')

        # Risk tracking
        self.total_realized_pnl = 0.0
        self.max_concurrent_positions = 0

    def check_trade_risk(self, signal: TradingSignal, balance: float) -> RiskCheck:
        """Check if trade passes risk controls

        Args:
            signal: Trading signal to check
            balance: Current account balance

        Returns:
            RiskCheck result
        """
        self._update_daily_reset()
        self.current_balance = balance

        risk_factors = {}

        # Check if trading is allowed
        if not self._is_trading_allowed():
            return RiskCheck(
                allowed=False,
                reason='trading_suspended',
                risk_factors={'daily_loss_exceeded': True}
            )

        # Position size check
        if signal.signal in ['buy', 'sell'] and signal.quantity > 0:
            is_position_reduction = (
                signal.signal == 'sell'
                and signal.symbol in self.positions
                and self.positions[signal.symbol].quantity > 0
                and signal.quantity <= abs(self.positions[signal.symbol].quantity) + 1e-9
            )
            position_value = signal.price * signal.quantity
            risk_factors['position_value_usd'] = position_value

            # Max position size check
            if position_value > self.max_position_size_usd and not is_position_reduction:
                adjusted_quantity = self.max_position_size_usd / signal.price
                risk_factors['position_size_exceeded'] = True

                if self.logger:
                    self.logger.log_system_event(
                        'risk_manager', 'position_size_adjusted',
                        {
                            'symbol': signal.symbol,
                            'original_quantity': signal.quantity,
                            'adjusted_quantity': adjusted_quantity,
                            'max_position_usd': self.max_position_size_usd
                        }
                    )

                return RiskCheck(
                    allowed=True,
                    reason='position_size_adjusted',
                    adjusted_quantity=adjusted_quantity,
                    risk_factors=risk_factors
                )

        # Daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            risk_factors['daily_trades_exceeded'] = True
            return RiskCheck(
                allowed=False,
                reason='daily_trade_limit_exceeded',
                risk_factors=risk_factors
            )

        # Portfolio exposure check (accounting for position reductions)
        total_exposure = self._calculate_portfolio_exposure()
        current_symbol_exposure = 0.0
        current_quantity = 0.0
        current_price = signal.price

        if signal.symbol in self.positions:
            current_position = self.positions[signal.symbol]
            current_quantity = current_position.quantity
            current_price = current_position.current_price
            current_symbol_exposure = abs(current_quantity * current_price)

        if signal.signal == 'buy':
            projected_quantity = current_quantity + signal.quantity
        else:  # sell reduces or closes the position
            projected_quantity = current_quantity - signal.quantity
            # Clamp projections that would over-close due to rounding noise
            if current_quantity >= 0 and projected_quantity < 0:
                projected_quantity = 0.0

        new_symbol_exposure = abs(projected_quantity * signal.price)
        projected_exposure = total_exposure - current_symbol_exposure + new_symbol_exposure
        exposure_pct = projected_exposure / max(balance, 1e-9)

        risk_factors['current_exposure_pct'] = total_exposure / max(balance, 1e-9)
        risk_factors['new_exposure_pct'] = exposure_pct

        if exposure_pct > self.max_portfolio_exposure:
            return RiskCheck(
                allowed=False,
                reason='portfolio_exposure_exceeded',
                risk_factors=risk_factors
            )

        # Maximum open positions check
        if signal.signal == 'buy' and signal.symbol not in self.positions:
            if len(self.positions) >= self.max_open_positions:
                risk_factors['max_positions_exceeded'] = True
                return RiskCheck(
                    allowed=False,
                    reason='max_open_positions_exceeded',
                    risk_factors=risk_factors
                )

        # Drawdown check
        current_drawdown = self._calculate_drawdown()
        risk_factors['current_drawdown_pct'] = current_drawdown

        if current_drawdown > self.max_drawdown_pct:
            return RiskCheck(
                allowed=False,
                reason='max_drawdown_exceeded',
                risk_factors=risk_factors
            )

        # All checks passed
        return RiskCheck(
            allowed=True,
            reason='risk_checks_passed',
            adjusted_quantity=signal.quantity,
            risk_factors=risk_factors
        )

    def open_position(self, symbol: str, quantity: float, price: float) -> Position:
        """Open new position with risk controls

        Args:
            symbol: Trading symbol
            quantity: Position quantity
            price: Entry price

        Returns:
            Position object
        """
        # Calculate stop loss and take profit
        if quantity > 0:  # Long position
            stop_loss = price * (1 - self.stop_loss_pct)
            take_profit = price * (1 + self.take_profit_pct)
        else:  # Short position
            stop_loss = price * (1 + self.stop_loss_pct)
            take_profit = price * (1 - self.take_profit_pct)

        position = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            entry_time=datetime.now(pytz.UTC),
            current_price=price,
            unrealized_pnl=0.0,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        self.positions[symbol] = position
        self.daily_trades += 1

        # Track statistics
        self.max_concurrent_positions = max(
            self.max_concurrent_positions, len(self.positions)
        )

        if self.logger:
            self.logger.log_system_event(
                'risk_manager', 'position_opened',
                {
                    'symbol': symbol,
                    'quantity': quantity,
                    'entry_price': price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'position_value_usd': abs(quantity * price)
                }
            )

        return position

    def close_position(self, symbol: str, price: float) -> Optional[float]:
        """Close position and calculate realized P&L

        Args:
            symbol: Trading symbol
            price: Exit price

        Returns:
            Realized P&L or None if position doesn't exist
        """
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]

        # Calculate realized P&L
        if position.quantity > 0:  # Long position
            realized_pnl = position.quantity * (price - position.entry_price)
        else:  # Short position (quantity is negative)
            # For shorts: profit when price goes down, loss when price goes up
            # Use absolute quantity for correct P&L sign
            realized_pnl = abs(position.quantity) * (position.entry_price - price)

        # Update tracking
        self.total_realized_pnl += realized_pnl
        self.daily_pnl += realized_pnl

        # Remove position
        del self.positions[symbol]

        if self.logger:
            self.logger.log_system_event(
                'risk_manager', 'position_closed',
                {
                    'symbol': symbol,
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'exit_price': price,
                    'realized_pnl': realized_pnl,
                    'hold_time_minutes': (datetime.now(pytz.UTC) - position.entry_time).total_seconds() / 60
                }
            )

        return realized_pnl

    def update_positions(self, market_prices: Dict[str, float]):
        """Update all positions with current market prices

        Args:
            market_prices: Dictionary of symbol -> current price
        """
        stop_loss_triggers = []
        take_profit_triggers = []

        for symbol, position in self.positions.items():
            if symbol in market_prices:
                position.current_price = market_prices[symbol]

                # Calculate unrealized P&L
                if position.quantity > 0:  # Long
                    position.unrealized_pnl = position.quantity * (
                        position.current_price - position.entry_price
                    )
                else:  # Short (quantity is negative)
                    # Use absolute quantity for correct P&L sign
                    position.unrealized_pnl = abs(position.quantity) * (
                        position.entry_price - position.current_price
                    )

                # Check stop loss
                if position.stop_loss:
                    if (position.quantity > 0 and position.current_price <= position.stop_loss) or \
                       (position.quantity < 0 and position.current_price >= position.stop_loss):
                        stop_loss_triggers.append(symbol)

                # Check take profit
                if position.take_profit:
                    if (position.quantity > 0 and position.current_price >= position.take_profit) or \
                       (position.quantity < 0 and position.current_price <= position.take_profit):
                        take_profit_triggers.append(symbol)

        # Log risk events
        if stop_loss_triggers and self.logger:
            self.logger.log_system_event(
                'risk_manager', 'stop_loss_triggered',
                {'symbols': stop_loss_triggers}
            )

        if take_profit_triggers and self.logger:
            self.logger.log_system_event(
                'risk_manager', 'take_profit_triggered',
                {'symbols': take_profit_triggers}
            )

        return stop_loss_triggers, take_profit_triggers

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio risk summary

        Returns:
            Portfolio summary with risk metrics
        """
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_value = self.current_balance + total_unrealized_pnl
        portfolio_exposure = self._calculate_portfolio_exposure()
        drawdown = self._calculate_drawdown()

        return {
            'balance': self.current_balance,
            'unrealized_pnl': total_unrealized_pnl,
            'realized_pnl': self.total_realized_pnl,
            'total_value': total_value,
            'open_positions': len(self.positions),
            'portfolio_exposure_usd': portfolio_exposure,
            'portfolio_exposure_pct': portfolio_exposure / self.current_balance if self.current_balance > 0 else 0,
            'current_drawdown_pct': drawdown,
            'daily_trades': self.daily_trades,
            'daily_pnl': self.daily_pnl,
            'max_concurrent_positions': self.max_concurrent_positions,
            'risk_limits': {
                'max_position_size_usd': self.max_position_size_usd,
                'max_open_positions': self.max_open_positions,
                'max_portfolio_exposure_pct': self.max_portfolio_exposure,
                'max_daily_trades': self.max_daily_trades,
                'max_drawdown_pct': self.max_drawdown_pct
            }
        }

    def _is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed

        Returns:
            True if trading allowed
        """
        # Check daily loss limit
        daily_loss_limit = self.initial_balance * self.max_daily_loss_pct
        if self.daily_pnl < -daily_loss_limit:
            return False

        # Check drawdown limit
        if self._calculate_drawdown() > self.max_drawdown_pct:
            return False

        return True

    def _calculate_portfolio_exposure(self) -> float:
        """Calculate total portfolio exposure in USD

        Returns:
            Total exposure in USD
        """
        total_exposure = 0.0
        for position in self.positions.values():
            exposure = abs(position.quantity * position.current_price)
            total_exposure += exposure

        return total_exposure

    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown percentage

        Returns:
            Drawdown percentage (0.0 to 1.0)
        """
        total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        current_value = self.current_balance + total_unrealized

        if self.peak_balance > 0:
            # Update peak if current value is higher
            if current_value > self.peak_balance:
                self.peak_balance = current_value

            drawdown = (self.peak_balance - current_value) / self.peak_balance
            return max(0.0, drawdown)

        return 0.0

    def _update_daily_reset(self):
        """Reset daily counters if new day"""
        now = datetime.now(pytz.UTC)
        if now.date() > self.daily_reset_time.date():
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.daily_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0)

            if self.logger:
                self.logger.log_system_event(
                    'risk_manager', 'daily_reset',
                    {'date': now.date().isoformat()}
                )
