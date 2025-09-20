"""
SIMPLE MOVING AVERAGE STRATEGY
That actually generates trading signals, not consciousness
"""

import numpy as np
from typing import Dict, List, Optional, Any
from collections import deque
from dataclasses import dataclass

from ..config.config_loader import get_config
from ..data_feed.binance_feed import OHLCV
from ..logging_system import TradeLogger


@dataclass
class TradingSignal:
    """Trading signal from strategy"""
    symbol: str
    signal: str  # 'buy', 'sell', 'hold'
    price: float
    quantity: float
    confidence: float  # 0.0 to 1.0
    reason: Dict[str, Any]


class SMAStrategy:
    """
    SIMPLE MOVING AVERAGE CROSSOVER STRATEGY
    When fast MA crosses above slow MA: BUY
    When fast MA crosses below slow MA: SELL
    No neural networks, no self-awareness, just math that works
    """

    def __init__(self, symbol: str, logger: Optional[TradeLogger] = None):
        """Initialize SMA strategy

        Args:
            symbol: Trading symbol
            logger: Trade logger
        """
        self.symbol = symbol
        self.logger = logger
        self.config = get_config()

        # Strategy parameters from config
        self.fast_period = self.config.fast_period
        self.slow_period = self.config.slow_period
        self.entry_threshold = self.config.get('strategy.entry_threshold_pct')
        self.exit_threshold = self.config.get('strategy.exit_threshold_pct')

        # Price history for moving averages
        self.prices = deque(maxlen=self.slow_period + 10)  # Extra buffer

        # Moving averages
        self.fast_ma = deque(maxlen=100)  # Store MA history
        self.slow_ma = deque(maxlen=100)

        # Signal tracking
        self.last_signal = 'hold'
        self.signal_count = 0

        # Position tracking (from external system)
        self.current_position = 0.0  # Will be updated by simulator

    def update_position(self, position: float):
        """Update current position from external system

        Args:
            position: Current position size
        """
        self.current_position = position

    def process_candle(self, candle: OHLCV) -> TradingSignal:
        """Process new OHLCV candle and generate signal

        Args:
            candle: New OHLCV candle

        Returns:
            Trading signal
        """
        # Add new price
        self.prices.append(candle.close)

        # Calculate moving averages
        fast_ma, slow_ma = self._calculate_moving_averages()

        if fast_ma is None or slow_ma is None:
            # Not enough data yet
            return TradingSignal(
                symbol=self.symbol,
                signal='hold',
                price=candle.close,
                quantity=0.0,
                confidence=0.0,
                reason={'status': 'insufficient_data', 'prices_count': len(self.prices)}
            )

        # Store MA values
        self.fast_ma.append(fast_ma)
        self.slow_ma.append(slow_ma)

        # Generate signal
        signal = self._generate_signal(candle, fast_ma, slow_ma)

        # Log decision
        if self.logger:
            self.logger.log_decision(
                strategy='sma_crossover',
                signal=signal.signal,
                symbol=self.symbol,
                reason=signal.reason,
                market_data={
                    'price': candle.close,
                    'fast_ma': fast_ma,
                    'slow_ma': slow_ma,
                    'timestamp': candle.timestamp
                },
                action_taken=signal.signal if signal.quantity > 0 else 'hold'
            )

        return signal

    def _calculate_moving_averages(self) -> tuple[Optional[float], Optional[float]]:
        """Calculate fast and slow moving averages

        Returns:
            Tuple of (fast_ma, slow_ma) or (None, None) if insufficient data
        """
        if len(self.prices) < self.slow_period:
            return None, None

        # Convert to numpy for efficient calculation
        prices_array = np.array(list(self.prices))

        # Calculate moving averages
        fast_ma = np.mean(prices_array[-self.fast_period:])
        slow_ma = np.mean(prices_array[-self.slow_period:])

        return float(fast_ma), float(slow_ma)

    def _generate_signal(self, candle: OHLCV, fast_ma: float, slow_ma: float) -> TradingSignal:
        """Generate trading signal based on moving averages

        Args:
            candle: Current candle
            fast_ma: Fast moving average
            slow_ma: Slow moving average

        Returns:
            Trading signal
        """
        # Calculate crossover
        # Prevent division by zero
        if slow_ma != 0:
            ma_diff = (fast_ma - slow_ma) / slow_ma  # Percentage difference
        else:
            ma_diff = 0.0  # Safe default when slow_ma is zero
        ma_direction = 'up' if fast_ma > slow_ma else 'down'

        # Check for crossover if we have previous MAs
        crossover = None
        if len(self.fast_ma) >= 2 and len(self.slow_ma) >= 2:
            prev_fast = self.fast_ma[-2]
            prev_slow = self.slow_ma[-2]

            if prev_fast <= prev_slow and fast_ma > slow_ma:
                crossover = 'golden_cross'  # Bullish crossover
            elif prev_fast >= prev_slow and fast_ma < slow_ma:
                crossover = 'death_cross'  # Bearish crossover

        # Generate signal based on strategy logic
        signal_data = self._apply_strategy_logic(
            candle, fast_ma, slow_ma, ma_diff, crossover
        )

        self.signal_count += 1
        return signal_data

    def _apply_strategy_logic(self, candle: OHLCV, fast_ma: float, slow_ma: float,
                            ma_diff: float, crossover: Optional[str]) -> TradingSignal:
        """Apply strategy logic to generate signal

        Args:
            candle: Current candle
            fast_ma: Fast MA value
            slow_ma: Slow MA value
            ma_diff: MA difference percentage
            crossover: Crossover type if any

        Returns:
            Trading signal
        """
        signal = 'hold'
        quantity = 0.0
        confidence = 0.0
        reason = {
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'ma_diff_pct': ma_diff * 100,
            'crossover': crossover,
            'current_position': self.current_position
        }

        # Entry signals
        if crossover == 'golden_cross' and abs(ma_diff) >= self.entry_threshold:
            if self.current_position <= 0:  # Not already long
                signal = 'buy'
                quantity = self._calculate_position_size(candle.close)
                confidence = min(0.9, abs(ma_diff) / self.entry_threshold)
                reason['entry_reason'] = 'golden_cross_entry'

        elif crossover == 'death_cross' and abs(ma_diff) >= self.entry_threshold:
            if self.current_position > 0:  # Currently long
                signal = 'sell'
                quantity = abs(self.current_position)  # Close entire position
                confidence = min(0.9, abs(ma_diff) / self.entry_threshold)
                reason['exit_reason'] = 'death_cross_exit'

        # Alternative: Strong trend without crossover (for testing)
        elif crossover is None and abs(ma_diff) >= self.entry_threshold * 2:
            if ma_diff > 0 and self.current_position <= 0:
                signal = 'buy'
                quantity = self._calculate_position_size(candle.close)
                confidence = min(0.8, abs(ma_diff) / (self.entry_threshold * 2))
                reason['entry_reason'] = 'strong_uptrend'

        # Exit signals (take profit / stop loss conditions)
        elif self.current_position > 0 and ma_diff < -self.exit_threshold:
            signal = 'sell'
            quantity = abs(self.current_position)
            confidence = 0.7
            reason['exit_reason'] = 'trend_reversal'

        # Update last signal
        if signal != 'hold':
            self.last_signal = signal

        return TradingSignal(
            symbol=self.symbol,
            signal=signal,
            price=candle.close,
            quantity=quantity,
            confidence=confidence,
            reason=reason
        )

    def _calculate_position_size(self, price: float) -> float:
        """Calculate position size based on config

        Args:
            price: Current price

        Returns:
            Position size in base currency
        """
        # This will be called by simulator with actual balance
        # For now, return a fixed size based on config
        max_usd = self.config.max_position_size_usd
        position_pct = self.config.position_size_pct

        # Calculate based on percentage of max position
        target_usd = max_usd * position_pct
        quantity = target_usd / price

        return quantity

    def get_strategy_state(self) -> Dict[str, Any]:
        """Get current strategy state

        Returns:
            Strategy state dictionary
        """
        fast_ma = self.fast_ma[-1] if self.fast_ma else None
        slow_ma = self.slow_ma[-1] if self.slow_ma else None

        return {
            'symbol': self.symbol,
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'current_fast_ma': fast_ma,
            'current_slow_ma': slow_ma,
            'ma_diff_pct': ((fast_ma - slow_ma) / slow_ma * 100) if (fast_ma and slow_ma and slow_ma != 0) else None,
            'last_signal': self.last_signal,
            'signal_count': self.signal_count,
            'current_position': self.current_position,
            'prices_in_buffer': len(self.prices),
            'ready_to_trade': len(self.prices) >= self.slow_period
        }

    def reset(self):
        """Reset strategy state"""
        self.prices.clear()
        self.fast_ma.clear()
        self.slow_ma.clear()
        self.last_signal = 'hold'
        self.signal_count = 0
        self.current_position = 0.0


# Strategy factory function
def create_strategy(symbol: str, strategy_type: str = 'sma_crossover',
                   logger: Optional[TradeLogger] = None) -> SMAStrategy:
    """Create strategy instance

    Args:
        symbol: Trading symbol
        strategy_type: Strategy type (only 'sma_crossover' for now)
        logger: Trade logger

    Returns:
        Strategy instance
    """
    if strategy_type != 'sma_crossover':
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    return SMAStrategy(symbol, logger)