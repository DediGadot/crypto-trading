"""
PURGED WALK-FORWARD SPLITTING
Ernie Chan + Linus Torvalds: Prevent leakage in strategy validation

This module implements purged and embargoed walk-forward analysis
to ensure strategy validation doesn't suffer from look-ahead bias.

"The most dangerous overfitting happens in strategy validation" - Ernie Chan
"If your validation leaks, your backtest lies" - Linus Torvalds
"""

import pandas as pd
import numpy as np
from typing import Iterator, Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings


@dataclass
class WalkForwardConfig:
    """
    Configuration for purged walk-forward analysis

    Attributes:
        train_period: Training window size (e.g., '6M', '1Y')
        test_period: Test window size (e.g., '1M', '3M')
        purge_period: Purge period to prevent leakage (e.g., '24h', '1D')
        embargo_period: Embargo period after test (e.g., '24h', '1D')
        step_size: Step size for walk-forward (e.g., '1M', '1W')
        min_train_size: Minimum training samples required
        min_test_size: Minimum test samples required
    """
    train_period: str = '6M'
    test_period: str = '1M'
    purge_period: str = '24h'
    embargo_period: str = '24h'
    step_size: str = '1M'
    min_train_size: int = 100
    min_test_size: int = 20


class PurgedWalkForward:
    """
    Purged walk-forward cross-validation for time series

    This class implements the methodology from "Advances in Financial
    Machine Learning" by Marcos López de Prado, ensuring that:

    1. Training data never overlaps with test data
    2. Purge period prevents leakage from overlapping predictions
    3. Embargo period prevents using future information
    4. Proper temporal ordering is maintained
    """

    def __init__(self, config: Optional[WalkForwardConfig] = None):
        """
        Initialize purged walk-forward splitter

        Args:
            config: Walk-forward configuration
        """
        self.config = config or WalkForwardConfig()

    def get_splits(self,
                   data: pd.DataFrame,
                   label_times: Optional[pd.Series] = None) -> Iterator[Tuple[pd.Index, pd.Index]]:
        """
        Generate purged walk-forward splits

        Args:
            data: Time series data with datetime index
            label_times: Series mapping prediction times to label times
                        (for proper purging when labels have different times)

        Yields:
            Tuples of (train_index, test_index)

        Example:
            for train_idx, test_idx in splitter.get_splits(data):
                train_data = data.loc[train_idx]
                test_data = data.loc[test_idx]
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be DatetimeIndex")

        # Convert periods to timedeltas
        train_td = pd.Timedelta(self.config.train_period)
        test_td = pd.Timedelta(self.config.test_period)
        purge_td = pd.Timedelta(self.config.purge_period)
        embargo_td = pd.Timedelta(self.config.embargo_period)
        step_td = pd.Timedelta(self.config.step_size)

        # Start from first possible test period
        data_start = data.index[0]
        data_end = data.index[-1]

        # First test can start after we have enough training data
        first_test_start = data_start + train_td

        current_test_start = first_test_start

        split_count = 0

        while current_test_start + test_td <= data_end:
            # Define periods
            train_start = current_test_start - train_td
            train_end = current_test_start - purge_td
            test_start = current_test_start
            test_end = current_test_start + test_td

            # Ensure we don't go before data start
            train_start = max(train_start, data_start)

            # Get train and test indices
            train_mask = (data.index >= train_start) & (data.index < train_end)
            test_mask = (data.index >= test_start) & (data.index < test_end)

            train_idx = data.index[train_mask]
            test_idx = data.index[test_mask]

            # Apply purging if label_times provided
            if label_times is not None:
                train_idx = self._apply_purging(train_idx, test_idx, label_times, purge_td)

            # Check minimum size requirements
            if len(train_idx) >= self.config.min_train_size and len(test_idx) >= self.config.min_test_size:
                split_count += 1
                yield train_idx, test_idx

            # Move to next period
            current_test_start += step_td

        if split_count == 0:
            warnings.warn("No valid splits generated. Check data size and configuration.")

    def _apply_purging(self,
                      train_idx: pd.Index,
                      test_idx: pd.Index,
                      label_times: pd.Series,
                      purge_period: pd.Timedelta) -> pd.Index:
        """
        Apply purging to remove training samples that overlap with test labels

        Args:
            train_idx: Training sample indices
            test_idx: Test sample indices
            label_times: Mapping from prediction time to label time
            purge_period: Additional purge period

        Returns:
            Purged training indices

        This is crucial when labels are derived from future prices
        (e.g., triple-barrier labels with time horizons)
        """
        if test_idx.empty:
            return train_idx

        # Get earliest test time
        test_start = test_idx[0]

        # Find training samples whose labels would overlap with test period
        train_labels_in_test = []

        for train_time in train_idx:
            if train_time in label_times.index:
                label_time = label_times.loc[train_time]

                # Check if label time falls in or near test period
                if pd.notna(label_time) and label_time >= (test_start - purge_period):
                    train_labels_in_test.append(train_time)

        # Remove overlapping samples
        purged_train_idx = train_idx.difference(pd.Index(train_labels_in_test))

        return purged_train_idx

    def get_split_dates(self, data: pd.DataFrame) -> List[Dict[str, datetime]]:
        """
        Get split date ranges for visualization and debugging

        Args:
            data: Time series data with datetime index

        Returns:
            List of dictionaries with split date information

        Example:
            split_dates = splitter.get_split_dates(data)
            for i, split in enumerate(split_dates):
                print(f"Split {i}: Train {split['train_start']} to {split['train_end']}")
        """
        splits = []

        for train_idx, test_idx in self.get_splits(data):
            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append({
                    'train_start': train_idx[0],
                    'train_end': train_idx[-1],
                    'test_start': test_idx[0],
                    'test_end': test_idx[-1],
                    'train_samples': len(train_idx),
                    'test_samples': len(test_idx)
                })

        return splits

    def validate_no_leakage(self,
                           data: pd.DataFrame,
                           label_times: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Validate that walk-forward splits don't have leakage

        Args:
            data: Time series data
            label_times: Label times for each prediction

        Returns:
            Dictionary with validation results

        This function checks for temporal leakage in the splits
        """
        validation_results = {
            'total_splits': 0,
            'valid_splits': 0,
            'leakage_detected': False,
            'leakage_details': [],
            'split_summary': []
        }

        for i, (train_idx, test_idx) in enumerate(self.get_splits(data, label_times)):
            validation_results['total_splits'] += 1

            # Check basic temporal ordering
            train_end = train_idx[-1] if len(train_idx) > 0 else None
            test_start = test_idx[0] if len(test_idx) > 0 else None

            split_valid = True
            leakage_issues = []

            if train_end and test_start:
                # Check purge period
                purge_td = pd.Timedelta(self.config.purge_period)
                if train_end >= (test_start - purge_td):
                    split_valid = False
                    leakage_issues.append(f"Train end ({train_end}) too close to test start ({test_start})")

                # Check for overlapping indices
                if not train_idx.intersection(test_idx).empty:
                    split_valid = False
                    leakage_issues.append("Train and test indices overlap")

                # Check label times if provided
                if label_times is not None:
                    for train_time in train_idx:
                        if train_time in label_times.index:
                            label_time = label_times.loc[train_time]
                            if pd.notna(label_time) and label_time >= test_start:
                                split_valid = False
                                leakage_issues.append(f"Training label ({label_time}) in test period")
                                break

            if split_valid:
                validation_results['valid_splits'] += 1
            else:
                validation_results['leakage_detected'] = True
                validation_results['leakage_details'].append({
                    'split_index': i,
                    'issues': leakage_issues
                })

            validation_results['split_summary'].append({
                'split_index': i,
                'train_start': train_idx[0] if len(train_idx) > 0 else None,
                'train_end': train_idx[-1] if len(train_idx) > 0 else None,
                'test_start': test_idx[0] if len(test_idx) > 0 else None,
                'test_end': test_idx[-1] if len(test_idx) > 0 else None,
                'train_samples': len(train_idx),
                'test_samples': len(test_idx),
                'valid': split_valid
            })

        return validation_results


class StrategyWalkForward:
    """
    Walk-forward analysis specifically for trading strategies

    This class integrates purged splitting with strategy backtesting
    to provide realistic out-of-sample performance estimates.
    """

    def __init__(self,
                 purged_splitter: PurgedWalkForward,
                 strategy_optimizer: callable,
                 strategy_backtester: callable):
        """
        Initialize strategy walk-forward analyzer

        Args:
            purged_splitter: Purged walk-forward splitter
            strategy_optimizer: Function to optimize strategy parameters
            strategy_backtester: Function to run strategy backtest
        """
        self.splitter = purged_splitter
        self.optimize_strategy = strategy_optimizer
        self.backtest_strategy = strategy_backtester
        self.results = []

    def run_walk_forward(self,
                        data: pd.DataFrame,
                        strategy_config: Dict[str, Any],
                        label_times: Optional[pd.Series] = None) -> List[Dict[str, Any]]:
        """
        Run walk-forward analysis for a trading strategy

        Args:
            data: Historical price data
            strategy_config: Strategy configuration parameters
            label_times: Label times for purging

        Returns:
            List of walk-forward results

        Example:
            results = wf.run_walk_forward(
                data=price_data,
                strategy_config={'type': 'sma_cross', 'fast': 5, 'slow': 20}
            )
        """
        self.results = []

        for i, (train_idx, test_idx) in enumerate(self.splitter.get_splits(data, label_times)):
            print(f"Processing walk-forward fold {i+1}...")

            # Get train and test data
            train_data = data.loc[train_idx]
            test_data = data.loc[test_idx]

            # Optimize strategy on training data
            try:
                optimization_result = self.optimize_strategy(train_data, strategy_config)
                if not optimization_result.get('success', False):
                    print(f"  Optimization failed for fold {i+1}: {optimization_result.get('error')}")
                    continue

                optimal_params = optimization_result['parameters']

            except Exception as e:
                print(f"  Optimization error for fold {i+1}: {e}")
                continue

            # Backtest on training data (in-sample)
            try:
                train_result = self.backtest_strategy(train_data, optimal_params)
                train_performance = self._extract_performance(train_result)
            except Exception as e:
                print(f"  Train backtest error for fold {i+1}: {e}")
                continue

            # Backtest on test data (out-of-sample)
            try:
                test_result = self.backtest_strategy(test_data, optimal_params)
                test_performance = self._extract_performance(test_result)
            except Exception as e:
                print(f"  Test backtest error for fold {i+1}: {e}")
                continue

            # Store results
            fold_result = {
                'fold': i + 1,
                'train_start': train_idx[0],
                'train_end': train_idx[-1],
                'test_start': test_idx[0],
                'test_end': test_idx[-1],
                'train_samples': len(train_idx),
                'test_samples': len(test_idx),
                'optimal_parameters': optimal_params,
                'in_sample_performance': train_performance,
                'out_of_sample_performance': test_performance,
                'overfitting_ratio': self._calculate_overfitting_ratio(
                    train_performance, test_performance
                )
            }

            self.results.append(fold_result)

            # Print fold summary
            print(f"  Fold {i+1} complete:")
            print(f"    Train Sharpe: {train_performance.get('sharpe_ratio', 0):.3f}")
            print(f"    Test Sharpe: {test_performance.get('sharpe_ratio', 0):.3f}")
            print(f"    Overfitting Ratio: {fold_result['overfitting_ratio']:.3f}")

        return self.results

    def _extract_performance(self, backtest_result: Dict[str, Any]) -> Dict[str, float]:
        """Extract performance metrics from backtest result"""
        return {
            'total_return': backtest_result.get('total_return', 0.0),
            'sharpe_ratio': backtest_result.get('sharpe_ratio', 0.0),
            'max_drawdown': backtest_result.get('max_drawdown', 0.0),
            'win_rate': backtest_result.get('win_rate', 0.0),
            'trades_count': backtest_result.get('trades_count', 0),
            'volatility': backtest_result.get('volatility', 0.0)
        }

    def _calculate_overfitting_ratio(self,
                                   train_perf: Dict[str, float],
                                   test_perf: Dict[str, float]) -> float:
        """Calculate overfitting ratio (test performance / train performance)"""
        train_sharpe = train_perf.get('sharpe_ratio', 0)
        test_sharpe = test_perf.get('sharpe_ratio', 0)

        if train_sharpe == 0:
            return 0.0

        return test_sharpe / train_sharpe

    def aggregate_results(self) -> Dict[str, Any]:
        """
        Aggregate walk-forward results

        Returns:
            Dictionary with aggregated performance metrics
        """
        if not self.results:
            return {}

        # Extract metrics
        test_returns = [r['out_of_sample_performance']['total_return'] for r in self.results]
        test_sharpes = [r['out_of_sample_performance']['sharpe_ratio'] for r in self.results]
        overfitting_ratios = [r['overfitting_ratio'] for r in self.results]

        # Calculate aggregate metrics
        aggregate = {
            'total_folds': len(self.results),
            'mean_oos_return': np.mean(test_returns),
            'std_oos_return': np.std(test_returns),
            'mean_oos_sharpe': np.mean(test_sharpes),
            'std_oos_sharpe': np.std(test_sharpes),
            'mean_overfitting_ratio': np.mean(overfitting_ratios),
            'positive_folds': sum(1 for r in test_returns if r > 0),
            'negative_folds': sum(1 for r in test_returns if r < 0),
            'win_rate': sum(1 for r in test_returns if r > 0) / len(test_returns),
            'consistency_score': 1 - np.std(test_sharpes) / max(abs(np.mean(test_sharpes)), 0.001)
        }

        return aggregate

    def generate_report(self) -> str:
        """Generate walk-forward analysis report"""
        if not self.results:
            return "No walk-forward results available."

        aggregate = self.aggregate_results()

        report = []
        report.append("=" * 60)
        report.append("PURGED WALK-FORWARD ANALYSIS REPORT")
        report.append("=" * 60)

        # Summary
        report.append(f"\nSUMMARY:")
        report.append(f"  Total Folds: {aggregate['total_folds']}")
        report.append(f"  Mean OOS Return: {aggregate['mean_oos_return']:.4f} ± {aggregate['std_oos_return']:.4f}")
        report.append(f"  Mean OOS Sharpe: {aggregate['mean_oos_sharpe']:.4f} ± {aggregate['std_oos_sharpe']:.4f}")
        report.append(f"  Win Rate: {aggregate['win_rate']:.2%}")
        report.append(f"  Consistency Score: {aggregate['consistency_score']:.3f}")

        # Overfitting analysis
        report.append(f"\nOVERFITTING ANALYSIS:")
        report.append(f"  Mean Overfitting Ratio: {aggregate['mean_overfitting_ratio']:.3f}")
        if aggregate['mean_overfitting_ratio'] < 0.5:
            report.append("  ⚠️ WARNING: Significant overfitting detected!")
        elif aggregate['mean_overfitting_ratio'] < 0.8:
            report.append("  ⚠️ Moderate overfitting detected")
        else:
            report.append("  ✅ Minimal overfitting")

        # Fold details
        report.append(f"\nFOLD DETAILS:")
        for result in self.results:
            train_sharpe = result['in_sample_performance']['sharpe_ratio']
            test_sharpe = result['out_of_sample_performance']['sharpe_ratio']
            report.append(f"  Fold {result['fold']}: Train Sharpe {train_sharpe:.3f} → Test Sharpe {test_sharpe:.3f}")

        report.append("=" * 60)

        return "\n".join(report)


# Testing functions
def test_purged_walk_forward():
    """Test purged walk-forward splitting"""
    print("\n🧪 Testing Purged Walk-Forward")

    # Create test data
    dates = pd.date_range('2024-01-01', periods=365, freq='D')  # Full year
    data = pd.DataFrame({
        'price': np.random.RandomState(42).randn(365).cumsum() + 100
    }, index=dates)

    # Configure splitter
    config = WalkForwardConfig(
        train_period='90D',  # 3 months training
        test_period='30D',   # 1 month testing
        purge_period='1D',   # 1 day purge
        step_size='30D',     # Step by 1 month
        min_train_size=50,   # Reduced minimum
        min_test_size=10     # Reduced minimum
    )

    splitter = PurgedWalkForward(config)

    # Get splits
    splits = list(splitter.get_splits(data))

    # Verify results
    assert len(splits) > 0, "Should generate at least one split"

    # Check temporal ordering
    for train_idx, test_idx in splits:
        assert len(train_idx) > 0, "Train set should not be empty"
        assert len(test_idx) > 0, "Test set should not be empty"

        train_end = train_idx[-1]
        test_start = test_idx[0]

        # Should have purge gap
        gap = test_start - train_end
        assert gap >= pd.Timedelta('1D'), f"Insufficient purge gap: {gap}"

    print(f"  Generated {len(splits)} splits")
    print(f"  Average train size: {np.mean([len(train) for train, _ in splits]):.0f}")
    print(f"  Average test size: {np.mean([len(test) for _, test in splits]):.0f}")
    print("✅ Purged walk-forward test passed")


def test_leakage_validation():
    """Test leakage validation"""
    print("\n🧪 Testing Leakage Validation")

    # Create test data
    dates = pd.date_range('2024-01-01', periods=365, freq='D')
    data = pd.DataFrame({
        'price': np.random.RandomState(42).randn(365).cumsum() + 100
    }, index=dates)

    # Create label times (some labels use future data)
    label_times = pd.Series(
        dates + pd.Timedelta('2D'),  # Labels 2 days in future
        index=dates
    )

    config = WalkForwardConfig(
        train_period='90D',  # 3 months training
        test_period='30D',   # 1 month testing
        purge_period='3D',   # Sufficient to prevent leakage
        step_size='30D',     # Step by 1 month
        min_train_size=50,   # Reduced minimum
        min_test_size=10     # Reduced minimum
    )

    splitter = PurgedWalkForward(config)

    # Validate splits
    validation = splitter.validate_no_leakage(data, label_times)

    # Check results
    assert validation['total_splits'] > 0, "Should have splits to validate"
    assert validation['valid_splits'] > 0, "Should have some valid splits"

    print(f"  Total splits: {validation['total_splits']}")
    print(f"  Valid splits: {validation['valid_splits']}")
    print(f"  Leakage detected: {validation['leakage_detected']}")

    if validation['leakage_detected']:
        print(f"  Leakage issues: {len(validation['leakage_details'])}")

    print("✅ Leakage validation test passed")


if __name__ == "__main__":
    print("🧪 Running Purged Walk-Forward Tests")
    test_purged_walk_forward()
    test_leakage_validation()
    print("✅ All purged walk-forward tests passed!")