"""
TRIPLE-BARRIER LABELING SYSTEM
Ernie Chan + Linus Torvalds: Labels that actually reflect trading P&L

This module implements Marcos López de Prado's triple-barrier method
for creating realistic trading labels that account for stop-losses,
profit targets, and time decay.

"Fixed-horizon labels are academic fantasy" - Ernie Chan
"If your labels don't match your P&L, your model is garbage" - Linus Torvalds
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, Tuple, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings


@dataclass
class BarrierConfig:
    """
    Configuration for triple-barrier labeling

    Attributes:
        profit_target: Profit target multiplier (e.g., 2.0 = 2x volatility)
        stop_loss: Stop loss multiplier (e.g., 1.0 = 1x volatility)
        vertical_barrier: Maximum holding period (e.g., '24H', '3D')
        vol_lookback: Periods for volatility estimation
        min_return: Minimum return threshold to generate label
    """
    profit_target: float = 2.0
    stop_loss: float = 1.0
    vertical_barrier: str = '24H'
    vol_lookback: int = 20
    min_return: float = 0.001  # 0.1% minimum return


def get_daily_volatility(close: pd.Series, span: int = 20) -> pd.Series:
    """
    Calculate daily volatility using exponential weighted moving average

    Args:
        close: Close price series
        span: Span for EWMA calculation

    Returns:
        Daily volatility series

    Example:
        vol = get_daily_volatility(data['close'], span=20)
    """
    # Calculate returns
    returns = close.pct_change().dropna()

    # Calculate exponentially weighted volatility
    volatility = returns.ewm(span=span).std()

    return volatility


def get_vertical_barriers(timestamps: pd.DatetimeIndex,
                         barrier_period: str) -> pd.Series:
    """
    Create vertical barriers (time-based exits)

    Args:
        timestamps: Entry timestamps
        barrier_period: Barrier period (e.g., '24H', '3D', '1W')

    Returns:
        Series with exit timestamps for each entry

    Example:
        entry_times = pd.DatetimeIndex(['2024-01-01', '2024-01-02'])
        exit_times = get_vertical_barriers(entry_times, '24H')
    """
    barrier_timedelta = pd.Timedelta(barrier_period)
    vertical_barriers = timestamps + barrier_timedelta

    return pd.Series(vertical_barriers, index=timestamps)


def apply_triple_barrier(close: pd.Series,
                        events: pd.DataFrame,
                        profit_target: Optional[pd.Series] = None,
                        stop_loss: Optional[pd.Series] = None,
                        vertical_barriers: Optional[pd.Series] = None,
                        min_return: float = 0.0) -> pd.DataFrame:
    """
    Apply triple-barrier method to generate labels

    Args:
        close: Close price series
        events: DataFrame with entry points and side (1=long, -1=short)
        profit_target: Profit target thresholds (if None, uses events)
        stop_loss: Stop loss thresholds (if None, uses events)
        vertical_barriers: Time-based exit points
        min_return: Minimum return to generate non-zero label

    Returns:
        DataFrame with barrier-based labels

    Events DataFrame should have columns:
        - 't1': entry timestamp (index)
        - 'side': position side (1=long, -1=short)
        - 'pt': profit target (optional)
        - 'sl': stop loss (optional)

    Example:
        events = pd.DataFrame({
            'side': [1, -1, 1],
            'pt': [0.02, 0.02, 0.02],
            'sl': [0.01, 0.01, 0.01]
        }, index=pd.DatetimeIndex(['2024-01-01', '2024-01-02', '2024-01-03']))

        labels = apply_triple_barrier(close, events)
    """
    if events.empty:
        return pd.DataFrame()

    # Set up barriers
    if profit_target is None:
        profit_target = events.get('pt', pd.Series(0.02, index=events.index))
    if stop_loss is None:
        stop_loss = events.get('sl', pd.Series(0.01, index=events.index))
    if vertical_barriers is None:
        vertical_barriers = get_vertical_barriers(events.index, '24H')

    # Ensure all series have the same index
    events = events.reindex(events.index)
    profit_target = profit_target.reindex(events.index)
    stop_loss = stop_loss.reindex(events.index)
    vertical_barriers = vertical_barriers.reindex(events.index)

    labels = []

    for idx, (t0, event) in enumerate(events.iterrows()):
        # Get position side and barriers for this event
        side = event.get('side', 1)
        pt = profit_target.iloc[idx] if not np.isnan(profit_target.iloc[idx]) else 0.02
        sl = stop_loss.iloc[idx] if not np.isnan(stop_loss.iloc[idx]) else 0.01
        t1 = vertical_barriers.iloc[idx]

        # Get price path from entry to vertical barrier
        price_path = close.loc[t0:t1]

        if len(price_path) < 2:
            # Not enough data
            labels.append({
                'entry_time': t0,
                'exit_time': t0,
                'label': 0,
                'return': 0.0,
                'barrier_touched': 'insufficient_data',
                'holding_period': timedelta(0)
            })
            continue

        entry_price = price_path.iloc[0]
        returns = (price_path / entry_price - 1) * side

        # Initialize result
        result = {
            'entry_time': t0,
            'exit_time': t1,  # Default to vertical barrier
            'label': 0,
            'return': 0.0,
            'barrier_touched': 'vertical',
            'holding_period': t1 - t0
        }

        # Check for barrier touches
        for timestamp, ret in returns.items():
            if timestamp == t0:  # Skip entry point
                continue

            # Check profit target
            if ret >= pt:
                result.update({
                    'exit_time': timestamp,
                    'label': 1,
                    'return': ret,
                    'barrier_touched': 'profit_target',
                    'holding_period': timestamp - t0
                })
                break

            # Check stop loss
            elif ret <= -sl:
                result.update({
                    'exit_time': timestamp,
                    'label': -1,
                    'return': ret,
                    'barrier_touched': 'stop_loss',
                    'holding_period': timestamp - t0
                })
                break

        # If no barrier was touched, use final return
        if result['barrier_touched'] == 'vertical':
            final_return = returns.iloc[-1]
            result['return'] = final_return

            # Label based on final return and minimum threshold
            if abs(final_return) >= min_return:
                result['label'] = 1 if final_return > 0 else -1
            else:
                result['label'] = 0

        labels.append(result)

    # Convert to DataFrame
    labels_df = pd.DataFrame(labels)
    labels_df.set_index('entry_time', inplace=True)

    return labels_df


def triple_barrier_labels(prices: pd.Series,
                         events: Optional[pd.DataFrame] = None,
                         config: Optional[BarrierConfig] = None,
                         vol_adjust: bool = True) -> pd.DataFrame:
    """
    Main interface for triple-barrier labeling

    Args:
        prices: Price series (typically close prices)
        events: DataFrame with entry points (if None, creates daily entries)
        config: Barrier configuration
        vol_adjust: Whether to adjust barriers by volatility

    Returns:
        DataFrame with triple-barrier labels

    Example:
        # Simple usage with default parameters
        labels = triple_barrier_labels(data['close'])

        # Advanced usage with custom config
        config = BarrierConfig(
            profit_target=1.5,
            stop_loss=0.8,
            vertical_barrier='48H',
            vol_lookback=30
        )
        labels = triple_barrier_labels(data['close'], config=config)
    """
    if config is None:
        config = BarrierConfig()

    # Create events if not provided
    if events is None:
        # Create daily events (simple approach)
        daily_timestamps = prices.resample('D').first().dropna().index
        events = pd.DataFrame({
            'side': 1  # Default to long positions
        }, index=daily_timestamps)

    # Calculate volatility if vol_adjust is True
    if vol_adjust:
        volatility = get_daily_volatility(prices, span=config.vol_lookback)

        # Align volatility with events
        vol_at_events = volatility.reindex(events.index, method='ffill')

        # Scale barriers by volatility
        profit_target = config.profit_target * vol_at_events
        stop_loss = config.stop_loss * vol_at_events
    else:
        # Use fixed barriers
        profit_target = pd.Series(config.profit_target, index=events.index)
        stop_loss = pd.Series(config.stop_loss, index=events.index)

    # Create vertical barriers
    vertical_barriers = get_vertical_barriers(events.index, config.vertical_barrier)

    # Apply triple-barrier method
    labels = apply_triple_barrier(
        close=prices,
        events=events,
        profit_target=profit_target,
        stop_loss=stop_loss,
        vertical_barriers=vertical_barriers,
        min_return=config.min_return
    )

    return labels


class MetaLabeler:
    """
    Meta-labeling implementation for filtering trading signals

    Meta-labeling uses a secondary model to predict the probability
    that a primary signal will be profitable. This improves precision
    by filtering out low-quality signals.
    """

    def __init__(self,
                 primary_threshold: float = 0.55,
                 meta_threshold: float = 0.5):
        """
        Initialize meta-labeler

        Args:
            primary_threshold: Threshold for primary model predictions
            meta_threshold: Threshold for meta model predictions
        """
        self.primary_threshold = primary_threshold
        self.meta_threshold = meta_threshold

    def create_meta_labels(self,
                          primary_predictions: pd.Series,
                          barrier_labels: pd.DataFrame) -> pd.DataFrame:
        """
        Create meta-labels for training meta-labeling model

        Args:
            primary_predictions: Predictions from primary model
            barrier_labels: Triple-barrier labels

        Returns:
            DataFrame with meta-labels (binary: trade/no-trade)

        The meta-label is 1 if we should trade on the primary signal,
        0 if we should not trade.
        """
        # Align predictions with barrier labels
        aligned_data = pd.DataFrame({
            'primary_pred': primary_predictions,
            'barrier_label': barrier_labels['label'],
            'barrier_return': barrier_labels['return']
        }).dropna()

        # Create meta-labels
        # Meta-label = 1 if primary prediction and barrier label agree
        meta_labels = (
            (aligned_data['primary_pred'] > self.primary_threshold) &
            (aligned_data['barrier_label'] > 0)
        ) | (
            (aligned_data['primary_pred'] < -self.primary_threshold) &
            (aligned_data['barrier_label'] < 0)
        )

        meta_labels_df = pd.DataFrame({
            'meta_label': meta_labels.astype(int),
            'primary_pred': aligned_data['primary_pred'],
            'barrier_label': aligned_data['barrier_label'],
            'barrier_return': aligned_data['barrier_return']
        }, index=aligned_data.index)

        return meta_labels_df

    def apply_meta_filter(self,
                         primary_signals: pd.Series,
                         meta_predictions: pd.Series) -> pd.Series:
        """
        Apply meta-labeling filter to primary signals

        Args:
            primary_signals: Raw signals from primary model
            meta_predictions: Predictions from meta-labeling model

        Returns:
            Filtered signals

        Only signals where meta-model predicts >meta_threshold are kept
        """
        # Apply both primary and meta thresholds
        filtered_signals = primary_signals.copy()

        # Zero out signals where meta-model says don't trade
        mask = meta_predictions < self.meta_threshold
        filtered_signals[mask] = 0

        return filtered_signals


def analyze_barrier_stats(labels_df: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze statistics of triple-barrier labels

    Args:
        labels_df: DataFrame from triple_barrier_labels()

    Returns:
        Dictionary with barrier touch statistics

    Example:
        stats = analyze_barrier_stats(labels)
        print(f"Profit target hit rate: {stats['pt_hit_rate']:.2%}")
    """
    if labels_df.empty:
        return {}

    total_labels = len(labels_df)

    # Count barrier touches
    barrier_counts = labels_df['barrier_touched'].value_counts()

    stats = {
        'total_labels': total_labels,
        'pt_hit_rate': barrier_counts.get('profit_target', 0) / total_labels,
        'sl_hit_rate': barrier_counts.get('stop_loss', 0) / total_labels,
        'vertical_hit_rate': barrier_counts.get('vertical', 0) / total_labels,
        'avg_holding_period': labels_df['holding_period'].mean(),
        'avg_return': labels_df['return'].mean(),
        'return_std': labels_df['return'].std(),
        'label_distribution': labels_df['label'].value_counts(normalize=True).to_dict()
    }

    return stats


def optimize_barrier_parameters(prices: pd.Series,
                               pt_range: Tuple[float, float] = (1.0, 3.0),
                               sl_range: Tuple[float, float] = (0.5, 2.0),
                               n_trials: int = 20) -> Dict[str, float]:
    """
    Optimize barrier parameters using grid search

    Args:
        prices: Price series
        pt_range: Range for profit target multipliers
        sl_range: Range for stop loss multipliers
        n_trials: Number of trials for each parameter

    Returns:
        Dictionary with optimal parameters and performance metrics

    This function finds the barrier parameters that maximize
    the information content of the labels.
    """
    pt_values = np.linspace(pt_range[0], pt_range[1], n_trials)
    sl_values = np.linspace(sl_range[0], sl_range[1], n_trials)

    best_score = -np.inf
    best_params = {}
    results = []

    for pt in pt_values:
        for sl in sl_values:
            config = BarrierConfig(
                profit_target=pt,
                stop_loss=sl,
                vertical_barrier='24H'
            )

            try:
                labels = triple_barrier_labels(prices, config=config)

                if len(labels) > 10:  # Minimum samples
                    stats = analyze_barrier_stats(labels)

                    # Score based on balance and return quality
                    pt_rate = stats.get('pt_hit_rate', 0)
                    sl_rate = stats.get('sl_hit_rate', 0)
                    avg_return = abs(stats.get('avg_return', 0))

                    # Prefer balanced barriers with good returns
                    balance_score = 1 - abs(pt_rate - sl_rate)
                    return_score = avg_return * 100  # Scale up

                    score = balance_score * return_score

                    results.append({
                        'pt': pt,
                        'sl': sl,
                        'score': score,
                        'pt_rate': pt_rate,
                        'sl_rate': sl_rate,
                        'avg_return': avg_return
                    })

                    if score > best_score:
                        best_score = score
                        best_params = {
                            'profit_target': pt,
                            'stop_loss': sl,
                            'score': score,
                            'stats': stats
                        }

            except Exception as e:
                # Skip invalid parameter combinations
                continue

    if best_params:
        best_params['optimization_results'] = results
    else:
        # Return default if no optimization succeeded
        best_params = {
            'profit_target': 2.0,
            'stop_loss': 1.0,
            'score': 0.0,
            'optimization_results': results
        }

    return best_params


# Testing functions
def test_triple_barrier_basic():
    """Test basic triple-barrier functionality"""
    print("\n🧪 Testing Triple-Barrier Labels")

    # Create test data with clear trend
    dates = pd.date_range('2024-01-01', periods=100, freq='h')
    trend = np.linspace(100, 120, 100)  # Upward trend
    noise = np.random.RandomState(42).normal(0, 1, 100)
    prices = pd.Series(trend + noise, index=dates)

    # Create simple events
    entry_times = dates[::24]  # Every 24 hours
    events = pd.DataFrame({
        'side': 1  # Long positions
    }, index=entry_times)

    # Apply triple-barrier
    config = BarrierConfig(
        profit_target=2.0,
        stop_loss=1.0,
        vertical_barrier='12H'
    )

    labels = triple_barrier_labels(prices, events, config)

    # Verify results
    assert not labels.empty, "Labels should not be empty"
    assert 'label' in labels.columns, "Missing label column"
    assert 'return' in labels.columns, "Missing return column"
    assert 'barrier_touched' in labels.columns, "Missing barrier_touched column"

    # Check label values
    unique_labels = labels['label'].unique()
    valid_labels = [-1, 0, 1]
    for label in unique_labels:
        assert label in valid_labels, f"Invalid label: {label}"

    print(f"  Generated {len(labels)} labels")
    print(f"  Label distribution: {labels['label'].value_counts().to_dict()}")
    print("✅ Triple-barrier basic test passed")


def test_meta_labeling():
    """Test meta-labeling functionality"""
    print("\n🧪 Testing Meta-Labeling")

    # Create test data
    dates = pd.date_range('2024-01-01', periods=50, freq='h')

    # Create barrier labels
    barrier_labels = pd.DataFrame({
        'label': np.random.RandomState(42).choice([-1, 0, 1], 50),
        'return': np.random.RandomState(42).normal(0, 0.02, 50)
    }, index=dates)

    # Create primary predictions
    primary_predictions = pd.Series(
        np.random.RandomState(42).normal(0, 0.8, 50),
        index=dates
    )

    # Test meta-labeling
    meta_labeler = MetaLabeler()
    meta_labels = meta_labeler.create_meta_labels(primary_predictions, barrier_labels)

    # Verify results
    assert not meta_labels.empty, "Meta-labels should not be empty"
    assert 'meta_label' in meta_labels.columns, "Missing meta_label column"

    # Check meta-label values
    unique_meta = meta_labels['meta_label'].unique()
    assert all(label in [0, 1] for label in unique_meta), "Meta-labels should be binary"

    print(f"  Generated {len(meta_labels)} meta-labels")
    print(f"  Meta-label rate: {meta_labels['meta_label'].mean():.2%}")
    print("✅ Meta-labeling test passed")


def test_barrier_optimization():
    """Test barrier parameter optimization"""
    print("\n🧪 Testing Barrier Optimization")

    # Create test data
    dates = pd.date_range('2024-01-01', periods=200, freq='h')
    trend = np.linspace(100, 110, 200)
    volatility = np.random.RandomState(42).normal(0, 2, 200)
    prices = pd.Series(trend + volatility, index=dates)

    # Optimize barriers (small grid for testing)
    optimal_params = optimize_barrier_parameters(
        prices,
        pt_range=(1.0, 2.0),
        sl_range=(0.5, 1.5),
        n_trials=3
    )

    # Verify results
    assert 'profit_target' in optimal_params, "Missing profit_target in optimization"
    assert 'stop_loss' in optimal_params, "Missing stop_loss in optimization"
    assert 'score' in optimal_params, "Missing score in optimization"

    print(f"  Optimal profit target: {optimal_params['profit_target']:.2f}")
    print(f"  Optimal stop loss: {optimal_params['stop_loss']:.2f}")
    print(f"  Optimization score: {optimal_params['score']:.4f}")
    print("✅ Barrier optimization test passed")


if __name__ == "__main__":
    print("🧪 Running Triple-Barrier Tests")
    test_triple_barrier_basic()
    test_meta_labeling()
    test_barrier_optimization()
    print("✅ All triple-barrier tests passed!")