"""
TRADING SIGNALS MODULE
Ernie Chan + Linus Torvalds: Proper signal generation without look-ahead bias

This module implements trading signals with strict temporal alignment.
All signals are computed at time t but can only be used for decisions at t+1.

"The difference between research and production is proper signal alignment" - Ernie Chan
"Code that looks right but behaves wrong is the worst kind of bug" - Linus Torvalds
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from .alignment import shift_for_decision, warmup_mask


def sma_crossover(close: pd.Series,
                 fast: int,
                 slow: int,
                 apply_shift: bool = True) -> Tuple[pd.Series, pd.Series]:
    """
    Generate SMA crossover signals with proper temporal alignment

    Args:
        close: Close price series
        fast: Fast SMA period
        slow: Slow SMA period
        apply_shift: Whether to apply shift(1) for decision alignment

    Returns:
        Tuple of (buy_signal, sell_signal) properly aligned for backtesting

    Example:
        buy_signal, sell_signal = sma_crossover(data['close'], 5, 20)
        # Signals are automatically shifted for next-period decisions
    """
    # Calculate SMAs
    sma_fast = close.rolling(window=fast).mean()
    sma_slow = close.rolling(window=slow).mean()

    # Generate raw signals (at time t)
    cross_above = (sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))
    cross_below = (sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1))

    if apply_shift:
        # Apply shift for decision alignment (decision at t+1)
        buy_signal = shift_for_decision(cross_above)
        sell_signal = shift_for_decision(cross_below)

        # Apply warmup mask
        lookbacks = {'sma_fast': fast, 'sma_slow': slow}
        df_dummy = pd.DataFrame(index=close.index)
        warmup = warmup_mask(df_dummy, lookbacks)

        buy_signal = buy_signal & warmup
        sell_signal = sell_signal & warmup
    else:
        buy_signal = cross_above
        sell_signal = cross_below

    return buy_signal, sell_signal


def rsi_signals(close: pd.Series,
               period: int = 14,
               oversold: float = 30,
               overbought: float = 70,
               apply_shift: bool = True) -> Tuple[pd.Series, pd.Series]:
    """
    Generate RSI signals with proper temporal alignment

    Args:
        close: Close price series
        period: RSI calculation period
        oversold: Oversold threshold (buy signal)
        overbought: Overbought threshold (sell signal)
        apply_shift: Whether to apply shift(1) for decision alignment

    Returns:
        Tuple of (buy_signal, sell_signal)
    """
    # Calculate RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Generate signals
    buy_signal = rsi < oversold
    sell_signal = rsi > overbought

    if apply_shift:
        buy_signal = shift_for_decision(buy_signal)
        sell_signal = shift_for_decision(sell_signal)

        # Apply warmup mask
        lookbacks = {'rsi': period}
        df_dummy = pd.DataFrame(index=close.index)
        warmup = warmup_mask(df_dummy, lookbacks)

        buy_signal = buy_signal & warmup
        sell_signal = sell_signal & warmup

    return buy_signal, sell_signal


def macd_signals(close: pd.Series,
                fast_period: int = 12,
                slow_period: int = 26,
                signal_period: int = 9,
                apply_shift: bool = True) -> Tuple[pd.Series, pd.Series]:
    """
    Generate MACD signals with proper temporal alignment

    Args:
        close: Close price series
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line EMA period
        apply_shift: Whether to apply shift(1) for decision alignment

    Returns:
        Tuple of (buy_signal, sell_signal)
    """
    # Calculate MACD
    ema_fast = close.ewm(span=fast_period).mean()
    ema_slow = close.ewm(span=slow_period).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal_period).mean()

    # Generate signals
    buy_signal = (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))
    sell_signal = (macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))

    if apply_shift:
        buy_signal = shift_for_decision(buy_signal)
        sell_signal = shift_for_decision(sell_signal)

        # Apply warmup mask
        lookbacks = {'macd': slow_period + signal_period}
        df_dummy = pd.DataFrame(index=close.index)
        warmup = warmup_mask(df_dummy, lookbacks)

        buy_signal = buy_signal & warmup
        sell_signal = sell_signal & warmup

    return buy_signal, sell_signal


def bollinger_signals(close: pd.Series,
                     period: int = 20,
                     std_dev: float = 2.0,
                     apply_shift: bool = True) -> Tuple[pd.Series, pd.Series]:
    """
    Generate Bollinger Band signals with proper temporal alignment

    Args:
        close: Close price series
        period: Moving average period
        std_dev: Standard deviation multiplier
        apply_shift: Whether to apply shift(1) for decision alignment

    Returns:
        Tuple of (buy_signal, sell_signal)
    """
    # Calculate Bollinger Bands
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)

    # Generate signals
    buy_signal = close < lower_band  # Price below lower band
    sell_signal = close > upper_band  # Price above upper band

    if apply_shift:
        buy_signal = shift_for_decision(buy_signal)
        sell_signal = shift_for_decision(sell_signal)

        # Apply warmup mask
        lookbacks = {'bollinger': period}
        df_dummy = pd.DataFrame(index=close.index)
        warmup = warmup_mask(df_dummy, lookbacks)

        buy_signal = buy_signal & warmup
        sell_signal = sell_signal & warmup

    return buy_signal, sell_signal


class StrategySignalGenerator:
    """
    Signal generator with built-in alignment validation

    This class ensures all signals follow proper temporal alignment rules
    and provides debugging tools to catch look-ahead bias.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize signal generator

        Args:
            data: OHLCV dataframe with datetime index
        """
        self.data = data
        self.signals = {}
        self.lookbacks = {}
        self.debug_mode = False

    def enable_debug(self):
        """Enable debug mode for signal validation"""
        self.debug_mode = True

    def add_sma_crossover(self, name: str, fast: int, slow: int):
        """Add SMA crossover signals"""
        buy_signal, sell_signal = sma_crossover(
            self.data['close'], fast, slow, apply_shift=True
        )

        self.signals[f"{name}_buy"] = buy_signal
        self.signals[f"{name}_sell"] = sell_signal
        self.lookbacks[name] = max(fast, slow)

        if self.debug_mode:
            self._debug_signal(f"{name}_buy", buy_signal)

    def add_rsi_signals(self, name: str, period: int = 14,
                       oversold: float = 30, overbought: float = 70):
        """Add RSI signals"""
        buy_signal, sell_signal = rsi_signals(
            self.data['close'], period, oversold, overbought, apply_shift=True
        )

        self.signals[f"{name}_buy"] = buy_signal
        self.signals[f"{name}_sell"] = sell_signal
        self.lookbacks[name] = period

        if self.debug_mode:
            self._debug_signal(f"{name}_buy", buy_signal)

    def add_macd_signals(self, name: str, fast: int = 12, slow: int = 26, signal: int = 9):
        """Add MACD signals"""
        buy_signal, sell_signal = macd_signals(
            self.data['close'], fast, slow, signal, apply_shift=True
        )

        self.signals[f"{name}_buy"] = buy_signal
        self.signals[f"{name}_sell"] = sell_signal
        self.lookbacks[name] = slow + signal

        if self.debug_mode:
            self._debug_signal(f"{name}_buy", buy_signal)

    def get_signals(self) -> Dict[str, pd.Series]:
        """Get all generated signals"""
        return self.signals.copy()

    def get_lookbacks(self) -> Dict[str, int]:
        """Get lookback periods for warmup calculation"""
        return self.lookbacks.copy()

    def validate_alignment(self) -> bool:
        """Validate that all signals are properly aligned"""
        max_lookback = max(self.lookbacks.values()) if self.lookbacks else 0

        for name, signal in self.signals.items():
            # Check warmup period
            warmup_signals = signal.iloc[:max_lookback]
            if warmup_signals.any():
                print(f"❌ {name}: Signal active during warmup period")
                return False

            # Check for NaN handling
            if signal.isna().any():
                nan_count = signal.isna().sum()
                print(f"⚠️ {name}: {nan_count} NaN values found")

        print("✅ All signals properly aligned")
        return True

    def _debug_signal(self, name: str, signal: pd.Series):
        """Debug helper for signal validation"""
        print(f"\n🔍 DEBUG: {name}")
        print(f"  First True at: {signal.idxmax() if signal.any() else 'Never'}")
        print(f"  Total signals: {signal.sum()}")
        print(f"  NaN count: {signal.isna().sum()}")

    def generate_strategy_signals(self, strategy_config: Dict[str, Any]) -> Dict[str, pd.Series]:
        """
        Generate signals based on strategy configuration

        Args:
            strategy_config: Configuration dict with strategy parameters

        Returns:
            Dictionary of aligned signals ready for backtesting

        Example:
            config = {
                'sma_cross': {'fast': 5, 'slow': 20},
                'rsi': {'period': 14, 'oversold': 30, 'overbought': 70}
            }
            signals = generator.generate_strategy_signals(config)
        """
        # Clear existing signals
        self.signals = {}
        self.lookbacks = {}

        # Add signals based on configuration
        if 'sma_cross' in strategy_config:
            params = strategy_config['sma_cross']
            self.add_sma_crossover('sma', params['fast'], params['slow'])

        if 'rsi' in strategy_config:
            params = strategy_config['rsi']
            self.add_rsi_signals('rsi', params['period'],
                               params.get('oversold', 30),
                               params.get('overbought', 70))

        if 'macd' in strategy_config:
            params = strategy_config['macd']
            self.add_macd_signals('macd',
                                params.get('fast', 12),
                                params.get('slow', 26),
                                params.get('signal', 9))

        if 'bollinger' in strategy_config:
            params = strategy_config['bollinger']
            buy_signal, sell_signal = bollinger_signals(
                self.data['close'],
                params.get('period', 20),
                params.get('std_dev', 2.0),
                apply_shift=True
            )
            self.signals['bollinger_buy'] = buy_signal
            self.signals['bollinger_sell'] = sell_signal
            self.lookbacks['bollinger'] = params.get('period', 20)

        return self.get_signals()


def create_test_data() -> pd.DataFrame:
    """Create test data for signal validation"""
    dates = pd.date_range('2024-01-01', periods=100, freq='H')

    # Create synthetic price data with clear trends for testing
    base_price = 100
    trend = np.linspace(0, 20, 100)  # Upward trend
    noise = np.random.RandomState(42).normal(0, 2, 100)  # Deterministic noise
    prices = base_price + trend + noise

    data = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.RandomState(42).randint(1000, 10000, 100)
    }, index=dates)

    return data


def test_sma_crossover_alignment():
    """Test SMA crossover signal alignment"""
    print("\n🧪 Testing SMA Crossover Alignment")

    # Create test data
    data = create_test_data()

    # Generate signals
    buy_signal, sell_signal = sma_crossover(data['close'], 5, 20, apply_shift=True)

    # Test 1: No signals during warmup period (first 20 periods)
    warmup_period = 20
    assert not buy_signal.iloc[:warmup_period].any(), "Buy signals during warmup"
    assert not sell_signal.iloc[:warmup_period].any(), "Sell signals during warmup"

    # Test 2: Signals are boolean
    assert buy_signal.dtype == bool, "Buy signal not boolean"
    assert sell_signal.dtype == bool, "Sell signal not boolean"

    # Test 3: First signal after warmup
    first_buy = buy_signal.idxmax() if buy_signal.any() else None
    if first_buy:
        signal_index = data.index.get_loc(first_buy)
        assert signal_index >= warmup_period, f"First buy signal at index {signal_index}, should be >= {warmup_period}"

    print("✅ SMA crossover alignment test passed")


def test_strategy_signal_generator():
    """Test the strategy signal generator"""
    print("\n🧪 Testing Strategy Signal Generator")

    data = create_test_data()
    generator = StrategySignalGenerator(data)
    generator.enable_debug()

    # Generate multiple signals
    config = {
        'sma_cross': {'fast': 5, 'slow': 20},
        'rsi': {'period': 14, 'oversold': 30, 'overbought': 70}
    }

    signals = generator.generate_strategy_signals(config)

    # Validate alignment
    is_valid = generator.validate_alignment()
    assert is_valid, "Signal alignment validation failed"

    # Check signal counts
    expected_signals = ['sma_buy', 'sma_sell', 'rsi_buy', 'rsi_sell']
    for signal_name in expected_signals:
        assert signal_name in signals, f"Missing signal: {signal_name}"

    print("✅ Strategy signal generator test passed")


if __name__ == "__main__":
    print("🧪 Running Signal Generation Tests")
    test_sma_crossover_alignment()
    test_strategy_signal_generator()
    print("✅ All signal tests passed!")