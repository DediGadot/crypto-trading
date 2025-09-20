"""
WALK-FORWARD ANALYSIS
Ernie Chan's approach to robust out-of-sample testing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from ..config import get_config
from ..logging_system import TradeLogger


@dataclass
class WalkForwardResult:
    """Walk-forward analysis result"""
    period_start: datetime
    period_end: datetime
    in_sample_performance: Dict[str, float]
    out_of_sample_performance: Dict[str, float]
    model_parameters: Dict[str, Any]
    trades_count: int
    sharpe_ratio: float
    max_drawdown: float
    total_return: float


class WalkForwardOptimizer:
    """
    Walk-Forward Analysis Implementation

    Ernie Chan's methodology:
    1. Train on lookback window
    2. Test on out-of-sample period
    3. Roll forward and repeat
    4. Aggregate results for statistical significance
    """

    def __init__(self,
                 lookback_periods: int = 1000,
                 reoptimize_every: int = 200,
                 gap_periods: int = 24,
                 logger: Optional[TradeLogger] = None):
        """Initialize walk-forward optimizer

        Args:
            lookback_periods: Training window size
            reoptimize_every: How often to reoptimize
            gap_periods: Gap between train and test (prevent leakage)
            logger: Trade logger
        """
        self.lookback_periods = lookback_periods
        self.reoptimize_every = reoptimize_every
        self.gap_periods = gap_periods
        self.logger = logger
        self.config = get_config()

    def run_walk_forward(self,
                        data: pd.DataFrame,
                        strategy_optimizer: Callable,
                        strategy_backtester: Callable,
                        min_periods: int = 500) -> List[WalkForwardResult]:
        """Run walk-forward analysis

        Args:
            data: Historical price data
            strategy_optimizer: Function to optimize strategy parameters
            strategy_backtester: Function to run backtest
            min_periods: Minimum periods needed to start

        Returns:
            List of walk-forward results
        """
        if len(data) < min_periods:
            raise ValueError(f"Insufficient data: {len(data)} < {min_periods}")

        results = []

        # Calculate walk-forward windows
        start_idx = self.lookback_periods

        while start_idx + self.gap_periods + self.reoptimize_every <= len(data):
            print(f"\n🔄 Walk-Forward Period: {start_idx}/{len(data)}")

            # Define periods
            train_start = start_idx - self.lookback_periods
            train_end = start_idx
            test_start = train_end + self.gap_periods
            test_end = min(test_start + self.reoptimize_every, len(data))

            # Extract data
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]

            print(f"   Train: {train_data.index[0]} to {train_data.index[-1]} ({len(train_data)} periods)")
            print(f"   Gap: {self.gap_periods} periods")
            print(f"   Test: {test_data.index[0]} to {test_data.index[-1]} ({len(test_data)} periods)")

            try:
                # 1. Optimize strategy on training data
                optimization_result = strategy_optimizer(train_data)

                if not optimization_result.get('success', False):
                    print(f"   ⚠️ Optimization failed: {optimization_result.get('error', 'Unknown')}")
                    start_idx += self.reoptimize_every
                    continue

                # 2. Backtest on training data (in-sample)
                in_sample_result = strategy_backtester(
                    train_data,
                    optimization_result['parameters']
                )

                # 3. Backtest on test data (out-of-sample)
                out_sample_result = strategy_backtester(
                    test_data,
                    optimization_result['parameters']
                )

                # 4. Store results
                result = WalkForwardResult(
                    period_start=test_data.index[0],
                    period_end=test_data.index[-1],
                    in_sample_performance={
                        'total_return': in_sample_result.get('total_return', 0.0),
                        'sharpe_ratio': in_sample_result.get('sharpe_ratio', 0.0),
                        'max_drawdown': in_sample_result.get('max_drawdown', 0.0),
                        'trades_count': in_sample_result.get('trades_count', 0)
                    },
                    out_of_sample_performance={
                        'total_return': out_sample_result.get('total_return', 0.0),
                        'sharpe_ratio': out_sample_result.get('sharpe_ratio', 0.0),
                        'max_drawdown': out_sample_result.get('max_drawdown', 0.0),
                        'trades_count': out_sample_result.get('trades_count', 0)
                    },
                    model_parameters=optimization_result['parameters'],
                    trades_count=out_sample_result.get('trades_count', 0),
                    sharpe_ratio=out_sample_result.get('sharpe_ratio', 0.0),
                    max_drawdown=out_sample_result.get('max_drawdown', 0.0),
                    total_return=out_sample_result.get('total_return', 0.0)
                )

                results.append(result)

                print(f"   ✅ In-sample Sharpe: {result.in_sample_performance['sharpe_ratio']:.4f}")
                print(f"   📊 Out-sample Sharpe: {result.sharpe_ratio:.4f}")
                print(f"   🎯 Out-sample Return: {result.total_return:.4f}")

                if self.logger:
                    self.logger.log_system_event(
                        'walk_forward', 'period_completed',
                        {
                            'period_start': result.period_start.isoformat(),
                            'period_end': result.period_end.isoformat(),
                            'in_sample_sharpe': result.in_sample_performance['sharpe_ratio'],
                            'out_sample_sharpe': result.sharpe_ratio,
                            'out_sample_return': result.total_return,
                            'trades_count': result.trades_count
                        }
                    )

            except Exception as e:
                print(f"   ❌ Period failed: {e}")
                if self.logger:
                    self.logger.log_error(
                        'walk_forward', 'period_failed',
                        f"Walk-forward period failed: {e}",
                        exception=e
                    )

            start_idx += self.reoptimize_every

        return results

    def analyze_results(self, results: List[WalkForwardResult]) -> Dict[str, Any]:
        """Analyze walk-forward results

        Args:
            results: List of walk-forward results

        Returns:
            Aggregated analysis
        """
        if not results:
            return {'error': 'No results to analyze'}

        # Extract metrics
        oos_returns = [r.total_return for r in results]
        oos_sharpes = [r.sharpe_ratio for r in results if not np.isnan(r.sharpe_ratio)]
        oos_drawdowns = [r.max_drawdown for r in results]
        trades_counts = [r.trades_count for r in results]

        is_sharpes = [r.in_sample_performance['sharpe_ratio'] for r in results
                     if not np.isnan(r.in_sample_performance['sharpe_ratio'])]

        # Statistical analysis
        analysis = {
            'total_periods': len(results),
            'date_range': {
                'start': results[0].period_start,
                'end': results[-1].period_end
            },

            # Out-of-sample performance
            'out_of_sample': {
                'mean_return': np.mean(oos_returns),
                'std_return': np.std(oos_returns),
                'mean_sharpe': np.mean(oos_sharpes) if oos_sharpes else 0.0,
                'std_sharpe': np.std(oos_sharpes) if oos_sharpes else 0.0,
                'max_drawdown': max(oos_drawdowns) if oos_drawdowns else 0.0,
                'win_rate': sum(1 for r in oos_returns if r > 0) / len(oos_returns),
                'total_trades': sum(trades_counts)
            },

            # In-sample vs out-of-sample comparison
            'overfitting_analysis': {
                'mean_is_sharpe': np.mean(is_sharpes) if is_sharpes else 0.0,
                'mean_oos_sharpe': np.mean(oos_sharpes) if oos_sharpes else 0.0,
                'sharpe_degradation': (np.mean(is_sharpes) - np.mean(oos_sharpes)) if (is_sharpes and oos_sharpes) else 0.0
            },

            # Statistical significance
            'statistical_tests': self._statistical_tests(oos_returns, oos_sharpes)
        }

        # Calculate aggregate performance
        if oos_returns:
            cumulative_return = np.prod([1 + r for r in oos_returns]) - 1
            analysis['aggregate_performance'] = {
                'cumulative_return': cumulative_return,
                'annualized_return': (1 + cumulative_return) ** (252*24 / len(results) / self.reoptimize_every) - 1,
                'annualized_sharpe': np.mean(oos_sharpes) * np.sqrt(252*24) if oos_sharpes else 0.0
            }

        return analysis

    def _statistical_tests(self, returns: List[float], sharpes: List[float]) -> Dict[str, Any]:
        """Perform statistical tests on results

        Args:
            returns: List of period returns
            sharpes: List of period Sharpe ratios

        Returns:
            Statistical test results
        """
        tests = {}

        if returns:
            # T-test for returns
            from scipy import stats

            t_stat, p_value = stats.ttest_1samp(returns, 0)
            tests['returns_ttest'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }

            # Jarque-Bera test for normality
            if len(returns) >= 8:  # Minimum for JB test
                jb_stat, jb_p = stats.jarque_bera(returns)
                tests['normality_test'] = {
                    'jarque_bera_stat': jb_stat,
                    'p_value': jb_p,
                    'normal_distribution': jb_p > 0.05
                }

        if sharpes and len(sharpes) >= 5:
            # Probabilistic Sharpe Ratio
            mean_sharpe = np.mean(sharpes)
            std_sharpe = np.std(sharpes)
            n_periods = len(sharpes)

            # PSR calculation
            psr = stats.norm.cdf(
                (mean_sharpe - 0) / (std_sharpe / np.sqrt(n_periods))
            )

            tests['probabilistic_sharpe_ratio'] = {
                'psr': psr,
                'confident_positive': psr > 0.95
            }

        return tests

    def generate_report(self, results: List[WalkForwardResult], analysis: Dict[str, Any]) -> str:
        """Generate walk-forward analysis report

        Args:
            results: Walk-forward results
            analysis: Analysis results

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("WALK-FORWARD ANALYSIS REPORT (Ernie Chan Method)")
        report.append("=" * 60)

        if analysis.get('error'):
            report.append(f"❌ Error: {analysis['error']}")
            return "\n".join(report)

        # Overview
        report.append(f"\n📊 OVERVIEW:")
        report.append(f"   Total Periods: {analysis['total_periods']}")
        report.append(f"   Date Range: {analysis['date_range']['start']} to {analysis['date_range']['end']}")
        report.append(f"   Lookback: {self.lookback_periods} periods")
        report.append(f"   Reoptimize Every: {self.reoptimize_every} periods")
        report.append(f"   Gap: {self.gap_periods} periods")

        # Out-of-sample performance
        oos = analysis['out_of_sample']
        report.append(f"\n🎯 OUT-OF-SAMPLE PERFORMANCE:")
        report.append(f"   Mean Return: {oos['mean_return']:.4f} ± {oos['std_return']:.4f}")
        report.append(f"   Mean Sharpe: {oos['mean_sharpe']:.4f} ± {oos['std_sharpe']:.4f}")
        report.append(f"   Max Drawdown: {oos['max_drawdown']:.4f}")
        report.append(f"   Win Rate: {oos['win_rate']:.2%}")
        report.append(f"   Total Trades: {oos['total_trades']}")

        # Overfitting analysis
        if 'overfitting_analysis' in analysis:
            ov = analysis['overfitting_analysis']
            report.append(f"\n🧠 OVERFITTING ANALYSIS:")
            report.append(f"   In-Sample Sharpe: {ov['mean_is_sharpe']:.4f}")
            report.append(f"   Out-Sample Sharpe: {ov['mean_oos_sharpe']:.4f}")
            report.append(f"   Sharpe Degradation: {ov['sharpe_degradation']:.4f}")

            if ov['sharpe_degradation'] > 0.5:
                report.append("   ⚠️ WARNING: Significant overfitting detected!")
            elif ov['sharpe_degradation'] > 0.2:
                report.append("   ⚠️ Moderate overfitting detected")
            else:
                report.append("   ✅ Minimal overfitting")

        # Statistical tests
        if 'statistical_tests' in analysis:
            st = analysis['statistical_tests']
            report.append(f"\n📈 STATISTICAL SIGNIFICANCE:")

            if 'returns_ttest' in st:
                tt = st['returns_ttest']
                sig_text = "significant" if tt['significant'] else "not significant"
                report.append(f"   Returns T-Test: p={tt['p_value']:.4f} ({sig_text})")

            if 'probabilistic_sharpe_ratio' in st:
                psr = st['probabilistic_sharpe_ratio']
                conf_text = "high confidence" if psr['confident_positive'] else "low confidence"
                report.append(f"   Probabilistic Sharpe Ratio: {psr['psr']:.4f} ({conf_text})")

        # Aggregate performance
        if 'aggregate_performance' in analysis:
            agg = analysis['aggregate_performance']
            report.append(f"\n🏆 AGGREGATE PERFORMANCE:")
            report.append(f"   Cumulative Return: {agg['cumulative_return']:.4f}")
            report.append(f"   Annualized Return: {agg['annualized_return']:.4f}")
            report.append(f"   Annualized Sharpe: {agg['annualized_sharpe']:.4f}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)