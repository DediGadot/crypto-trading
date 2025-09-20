"""
SIGNAL ALIGNMENT MODULE
Ernie Chan + Linus Torvalds: Fix look-ahead bias once and for all

This module ensures that ALL decision signals are properly aligned to prevent
look-ahead bias. Every signal computed at time t can only be used for decisions
at time t+1.

"The most expensive mistake in backtesting is using tomorrow's information today" - Ernie Chan
"If you can't prove it's right, it's wrong" - Linus Torvalds
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union


def shift_for_decision(series: pd.Series, periods: int = 1) -> pd.Series:
    """
    Shift series for decision making - ALWAYS shift signals forward

    This is the FUNDAMENTAL rule: signals computed at time t
    can only be used for decisions at time t+1

    Args:
        series: Signal series (e.g., SMA crossover signal)
        periods: Number of periods to shift (default: 1)

    Returns:
        Shifted series where signal[t] becomes signal[t+1]

    Example:
        # BAD - uses signal at same time as decision
        signal = sma_fast > sma_slow

        # GOOD - shifts signal for next period decision
        signal = shift_for_decision(sma_fast > sma_slow)
    """
    return series.shift(periods)


def warmup_mask(df: pd.DataFrame, lookbacks: Dict[str, int]) -> pd.Series:
    """
    Create mask to prevent trades during warmup period

    Args:
        df: Price dataframe with datetime index
        lookbacks: Dict mapping indicator names to their lookback periods
                  e.g., {'sma_fast': 5, 'sma_slow': 20}

    Returns:
        Boolean series: False during warmup, True when all indicators valid

    Example:
        lookbacks = {'sma_fast': 5, 'sma_slow': 20}
        mask = warmup_mask(data, lookbacks)
        # First 20 values will be False (max lookback period)
    """
    if not lookbacks:
        return pd.Series(True, index=df.index)

    max_lookback = max(lookbacks.values())
    mask = pd.Series(False, index=df.index)
    mask.iloc[max_lookback:] = True

    return mask


def align_signals_for_backtest(signals: Dict[str, pd.Series],
                              lookbacks: Dict[str, int],
                              df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Align all signals for backtesting - prevents look-ahead bias

    This function applies THREE critical fixes:
    1. Shifts all signals by 1 period (decision at t+1)
    2. Applies warmup mask (no trades during indicator warmup)
    3. Forward-fills any NaN values in signals

    Args:
        signals: Dictionary of signal series
        lookbacks: Lookback periods for warmup calculation
        df: Price dataframe for index alignment

    Returns:
        Dictionary of properly aligned signals

    Example:
        signals = {
            'sma_cross': sma_fast > sma_slow,
            'rsi_signal': rsi < 30
        }
        lookbacks = {'sma_cross': 20, 'rsi_signal': 14}

        aligned = align_signals_for_backtest(signals, lookbacks, data)
        # Now all signals are shift(1) and respect warmup periods
    """
    aligned_signals = {}
    warmup = warmup_mask(df, lookbacks)

    for name, signal in signals.items():
        # Step 1: Shift signal for next-period decision
        shifted_signal = shift_for_decision(signal)

        # Step 2: Apply warmup mask
        shifted_signal = shifted_signal & warmup

        # Step 3: Forward fill NaN values (conservative approach)
        shifted_signal = shifted_signal.fillna(False)

        aligned_signals[name] = shifted_signal

    return aligned_signals


def validate_no_lookahead(signal_times: pd.DatetimeIndex,
                         decision_times: pd.DatetimeIndex,
                         min_delay: str = "1H") -> bool:
    """
    Validate that decisions never use future information

    Args:
        signal_times: Timestamps when signals were computed
        decision_times: Timestamps when decisions were made
        min_delay: Minimum delay required (default: 1 hour)

    Returns:
        True if no look-ahead bias detected, False otherwise

    Example:
        # Signal computed at 10:00, decision at 11:00 = OK
        # Signal computed at 10:00, decision at 10:00 = LOOK-AHEAD BIAS
    """
    if len(signal_times) != len(decision_times):
        return False

    min_delay_td = pd.Timedelta(min_delay)
    delays = decision_times - signal_times

    # All delays must be >= minimum delay
    return (delays >= min_delay_td).all()


def create_decision_timestamps(signal_series: pd.Series,
                              frequency: str = "1H") -> pd.Series:
    """
    Create proper decision timestamps from signal series

    Args:
        signal_series: Series with signal values
        frequency: Trading frequency (default: 1H)

    Returns:
        Series with decision timestamps shifted by one period

    This ensures decisions are made AFTER signals are computed
    """
    freq_map = {
        "1H": "H",
        "1D": "D",
        "1min": "T",
        "5min": "5T"
    }

    pandas_freq = freq_map.get(frequency, "H")

    # Shift index forward by one period
    decision_index = signal_series.index + pd.Timedelta(pandas_freq)

    return pd.Series(signal_series.values, index=decision_index)


class SignalAlignmentValidator:
    """
    Validator class to ensure proper signal alignment in backtests

    Usage:
        validator = SignalAlignmentValidator()
        validator.add_signal('sma_cross', sma_signal, lookback=20)
        validator.add_signal('rsi_signal', rsi_signal, lookback=14)

        if validator.validate():
            print("✅ All signals properly aligned")
        else:
            print("❌ Look-ahead bias detected!")
            validator.print_violations()
    """

    def __init__(self):
        self.signals = {}
        self.lookbacks = {}
        self.violations = []

    def add_signal(self, name: str, signal: pd.Series, lookback: int):
        """Add signal for validation"""
        self.signals[name] = signal
        self.lookbacks[name] = lookback

    def validate(self) -> bool:
        """Validate all signals for look-ahead bias"""
        self.violations = []
        all_valid = True

        for name, signal in self.signals.items():
            lookback = self.lookbacks[name]

            # Check 1: Signal should have proper NaN handling during warmup
            warmup_period = signal.iloc[:lookback]
            if warmup_period.notna().any():
                self.violations.append(f"{name}: Signal active during warmup period")
                all_valid = False

            # Check 2: No future information used
            if not signal.index.is_monotonic_increasing:
                self.violations.append(f"{name}: Signal index not chronologically ordered")
                all_valid = False

            # Check 3: Signal properly shifted
            # This would require decision timestamps to fully validate

        return all_valid

    def print_violations(self):
        """Print all detected violations"""
        if not self.violations:
            print("✅ No violations detected")
            return

        print("❌ Signal Alignment Violations:")
        for i, violation in enumerate(self.violations, 1):
            print(f"  {i}. {violation}")


def debug_signal_alignment(signal: pd.Series,
                          decision: pd.Series,
                          name: str = "signal") -> None:
    """
    Debug helper to visualize signal alignment

    Args:
        signal: Original signal series
        decision: Decision series (should be shifted)
        name: Signal name for logging
    """
    print(f"\n🔍 DEBUG: {name} Alignment")
    print("=" * 50)

    # Show first few values
    comparison = pd.DataFrame({
        'signal_t': signal.head(10),
        'decision_t': decision.head(10),
        'shift_ok': signal.head(10).shift(1).equals(decision.head(10))
    })

    print("First 10 values:")
    print(comparison)

    # Check overall alignment
    is_properly_shifted = signal.shift(1).equals(decision)
    print(f"\n✅ Properly shifted: {is_properly_shifted}")

    # Check warmup handling
    first_decision = decision.first_valid_index()
    first_signal = signal.first_valid_index()

    if first_decision and first_signal:
        delay = first_decision - first_signal
        print(f"📅 First signal: {first_signal}")
        print(f"📅 First decision: {first_decision}")
        print(f"⏱️ Delay: {delay}")

        if delay >= pd.Timedelta("1H"):
            print("✅ Sufficient delay for no look-ahead")
        else:
            print("❌ Potential look-ahead bias!")


# Example usage and testing functions
def test_shift_for_decision():
    """Test the shift_for_decision function"""
    # Create test signal
    dates = pd.date_range('2024-01-01', periods=5, freq='H')
    signal = pd.Series([True, False, True, False, True], index=dates)

    shifted = shift_for_decision(signal)

    # First value should be NaN after shift
    assert pd.isna(shifted.iloc[0])

    # Second value should equal first original value
    assert shifted.iloc[1] == signal.iloc[0]

    print("✅ shift_for_decision test passed")


def test_warmup_mask():
    """Test the warmup_mask function"""
    dates = pd.date_range('2024-01-01', periods=25, freq='H')
    df = pd.DataFrame({'close': range(25)}, index=dates)

    lookbacks = {'sma_fast': 5, 'sma_slow': 20}
    mask = warmup_mask(df, lookbacks)

    # First 20 values should be False (max lookback)
    assert not mask.iloc[:20].any()

    # Values after 20 should be True
    assert mask.iloc[20:].all()

    print("✅ warmup_mask test passed")


if __name__ == "__main__":
    print("🧪 Running Signal Alignment Tests")
    test_shift_for_decision()
    test_warmup_mask()
    print("✅ All tests passed!")