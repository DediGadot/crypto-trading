"""
PERFORMANCE METRICS MODULE
Ernie Chan + Linus Torvalds: Honest assessment beyond raw Sharpe

This module implements sophisticated performance metrics that reveal
the truth about trading strategy performance, including statistical
significance and capacity constraints.

"A high Sharpe ratio means nothing without statistical significance" - Ernie Chan
"If you can't measure it properly, you can't improve it" - Linus Torvalds
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
from scipy import stats
import math


@dataclass
class PerformanceMetrics:
    """
    Container for comprehensive performance metrics

    Attributes:
        total_return: Total return over period
        annualized_return: Annualized return
        volatility: Return volatility (annualized)
        sharpe_ratio: Sharpe ratio
        max_drawdown: Maximum drawdown
        win_rate: Percentage of winning trades
        profit_factor: Ratio of gross profit to gross loss
        trades_count: Total number of trades
        avg_trade_return: Average return per trade
        best_trade: Best single trade return
        worst_trade: Worst single trade return
    """
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    trades_count: int = 0
    avg_trade_return: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0


def probabilistic_sharpe_ratio(returns: pd.Series,
                              benchmark_sr: float = 0.0,
                              skewness: Optional[float] = None,
                              kurtosis: Optional[float] = None) -> Dict[str, float]:
    """
    Calculate Probabilistic Sharpe Ratio (PSR)

    The PSR answers: "What is the probability that the Sharpe ratio
    is greater than a benchmark?"

    Args:
        returns: Return series
        benchmark_sr: Benchmark Sharpe ratio to compare against
        skewness: Return skewness (calculated if None)
        kurtosis: Return kurtosis (calculated if None)

    Returns:
        Dictionary with PSR and related statistics

    Reference: Bailey & López de Prado (2012)
    "The Sharpe Ratio Efficient Frontier"
    """
    if len(returns) < 2:
        return {'psr': 0.0, 'sharpe_ratio': 0.0, 'confidence': 0.0}

    # Calculate observed Sharpe ratio
    mean_return = returns.mean()
    std_return = returns.std()
    observed_sr = mean_return / std_return if std_return > 0 else 0.0

    # Calculate skewness and kurtosis if not provided
    if skewness is None:
        skewness = returns.skew()
    if kurtosis is None:
        kurtosis = returns.kurtosis()

    # Number of observations
    n = len(returns)

    # Standard error of Sharpe ratio
    sr_std = math.sqrt((1 + 0.5 * observed_sr**2 - skewness * observed_sr +
                       (kurtosis - 3) / 4 * observed_sr**2) / n)

    # PSR calculation
    if sr_std > 0:
        psr = stats.norm.cdf((observed_sr - benchmark_sr) / sr_std)
    else:
        psr = 0.5  # Undefined case

    # Confidence interval (95%)
    confidence_95 = observed_sr - 1.96 * sr_std, observed_sr + 1.96 * sr_std

    return {
        'psr': psr,
        'sharpe_ratio': observed_sr,
        'sharpe_std_error': sr_std,
        'confidence_lower': confidence_95[0],
        'confidence_upper': confidence_95[1],
        'confidence': psr,
        'significant_95': psr > 0.95
    }


def deflated_sharpe_ratio(returns: pd.Series,
                         num_trials: int,
                         skewness: Optional[float] = None,
                         kurtosis: Optional[float] = None) -> Dict[str, float]:
    """
    Calculate Deflated Sharpe Ratio (DSR)

    The DSR adjusts the Sharpe ratio for multiple testing bias.
    Critical when you've tested many strategies.

    Args:
        returns: Return series
        num_trials: Number of strategies tested
        skewness: Return skewness
        kurtosis: Return kurtosis

    Returns:
        Dictionary with DSR and related statistics

    Reference: Bailey & López de Prado (2014)
    "The Deflated Sharpe Ratio: Correcting for Selection Bias"
    """
    if len(returns) < 2:
        return {'dsr': 0.0, 'threshold': 0.0}

    # Calculate skewness and kurtosis if not provided
    if skewness is None:
        skewness = returns.skew()
    if kurtosis is None:
        kurtosis = returns.kurtosis()

    # Observed Sharpe ratio
    mean_return = returns.mean()
    std_return = returns.std()
    observed_sr = mean_return / std_return if std_return > 0 else 0.0

    # Number of observations
    n = len(returns)

    # Expected maximum Sharpe ratio under null hypothesis
    gamma = 0.5772156649015329  # Euler-Mascheroni constant
    expected_max_sr = (1 - gamma) * stats.norm.ppf(1 - 1.0/num_trials) + \
                     gamma * stats.norm.ppf(1 - 1.0/(num_trials * math.e))

    # Variance of maximum Sharpe ratio
    var_max_sr = (1 - gamma) * stats.norm.ppf(1 - 1.0/num_trials)**2 + \
                 gamma * stats.norm.ppf(1 - 1.0/(num_trials * math.e))**2 - \
                 expected_max_sr**2

    # Standard error adjustment for higher moments
    sr_std = math.sqrt((1 + 0.5 * observed_sr**2 - skewness * observed_sr +
                       (kurtosis - 3) / 4 * observed_sr**2) / n)

    # Threshold Sharpe ratio
    threshold_sr = expected_max_sr + math.sqrt(var_max_sr) * sr_std

    # Deflated Sharpe ratio
    if sr_std > 0:
        dsr = stats.norm.cdf((observed_sr - threshold_sr) / sr_std)
    else:
        dsr = 0.5

    return {
        'dsr': dsr,
        'threshold_sr': threshold_sr,
        'observed_sr': observed_sr,
        'expected_max_sr': expected_max_sr,
        'significant': dsr > 0.95
    }


def minimum_track_record_length(target_sr: float,
                               prob_sr: float = 0.95,
                               skewness: float = 0.0,
                               kurtosis: float = 3.0) -> int:
    """
    Calculate minimum track record length for statistical significance

    Args:
        target_sr: Target Sharpe ratio to achieve
        prob_sr: Desired probability (default: 95%)
        skewness: Expected return skewness
        kurtosis: Expected return kurtosis

    Returns:
        Minimum number of observations needed

    Reference: Bailey & López de Prado (2012)
    """
    # Standard normal quantile
    z_alpha = stats.norm.ppf(prob_sr)

    # Variance adjustment for higher moments
    var_adjustment = 1 + 0.5 * target_sr**2 - skewness * target_sr + \
                    (kurtosis - 3) / 4 * target_sr**2

    # Minimum track record length
    min_trl = (z_alpha**2 * var_adjustment) / target_sr**2

    return max(1, int(math.ceil(min_trl)))


def calculate_capacity(returns: pd.Series,
                      position_sizes: pd.Series,
                      market_impact_coeff: float = 0.1) -> Dict[str, float]:
    """
    Calculate strategy capacity based on market impact

    Args:
        returns: Strategy returns
        position_sizes: Position sizes (as fraction of daily volume)
        market_impact_coeff: Market impact coefficient

    Returns:
        Dictionary with capacity metrics

    Capacity is the maximum capital that can be deployed
    before market impact destroys returns.
    """
    if len(returns) != len(position_sizes):
        raise ValueError("Returns and position sizes must have same length")

    # Calculate impact-adjusted returns
    market_impact = market_impact_coeff * np.sqrt(position_sizes)
    adjusted_returns = returns - market_impact

    # Find capacity where net return approaches zero
    capacity_levels = np.linspace(0.1, 2.0, 20)  # 10% to 200% capacity
    net_returns = []

    for capacity in capacity_levels:
        scaled_impact = market_impact * capacity
        net_return = (returns - scaled_impact).mean()
        net_returns.append(net_return)

    net_returns = np.array(net_returns)

    # Find capacity where net return is still positive
    positive_mask = net_returns > 0
    if positive_mask.any():
        max_capacity = capacity_levels[positive_mask][-1]
    else:
        max_capacity = 0.1  # Minimum capacity

    # Calculate capacity curve slope
    capacity_slope = np.polyfit(capacity_levels, net_returns, 1)[0]

    return {
        'max_capacity': max_capacity,
        'capacity_slope': capacity_slope,
        'impact_coefficient': market_impact_coeff,
        'gross_return': returns.mean(),
        'net_return_at_max': net_returns[capacity_levels <= max_capacity][-1] if max_capacity > 0 else 0
    }


def hit_rate_analysis(returns: pd.Series,
                     trades_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """
    Analyze hit rate and return distribution

    Args:
        returns: Return series
        trades_df: Optional trades dataframe with individual trade returns

    Returns:
        Dictionary with hit rate metrics
    """
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]

    # Basic hit rate
    hit_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0

    # Average win/loss
    avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
    avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0

    # Win/loss ratio
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

    # Profit factor
    gross_profit = positive_returns.sum()
    gross_loss = abs(negative_returns.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Kelly criterion
    if avg_loss != 0:
        kelly_fraction = hit_rate - (1 - hit_rate) / win_loss_ratio
    else:
        kelly_fraction = hit_rate

    kelly_fraction = max(0, min(kelly_fraction, 1))  # Bound between 0 and 1

    return {
        'hit_rate': hit_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': win_loss_ratio,
        'profit_factor': profit_factor,
        'kelly_fraction': kelly_fraction,
        'total_trades': len(returns),
        'winning_trades': len(positive_returns),
        'losing_trades': len(negative_returns)
    }


def drawdown_analysis(equity_curve: pd.Series) -> Dict[str, Any]:
    """
    Comprehensive drawdown analysis

    Args:
        equity_curve: Cumulative equity curve

    Returns:
        Dictionary with drawdown metrics
    """
    # Calculate running maximum (peak)
    running_max = equity_curve.expanding().max()

    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max

    # Maximum drawdown
    max_drawdown = drawdown.min()

    # Drawdown duration analysis
    is_drawdown = drawdown < 0
    drawdown_periods = []

    in_drawdown = False
    start_idx = None

    for i, is_dd in enumerate(is_drawdown):
        if is_dd and not in_drawdown:
            # Start of drawdown
            in_drawdown = True
            start_idx = i
        elif not is_dd and in_drawdown:
            # End of drawdown
            in_drawdown = False
            if start_idx is not None:
                duration = i - start_idx
                drawdown_periods.append(duration)

    # Handle case where we end in drawdown
    if in_drawdown and start_idx is not None:
        duration = len(equity_curve) - start_idx
        drawdown_periods.append(duration)

    # Calculate statistics
    avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
    max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0

    # Recovery time (time to new high after max drawdown)
    max_dd_idx = drawdown.idxmin()
    subsequent_equity = equity_curve.loc[max_dd_idx:]
    peak_at_max_dd = running_max.loc[max_dd_idx]

    recovery_idx = subsequent_equity[subsequent_equity >= peak_at_max_dd].index
    if len(recovery_idx) > 0:
        recovery_time = len(subsequent_equity.loc[:recovery_idx[0]])
    else:
        recovery_time = len(subsequent_equity)  # Still recovering

    return {
        'max_drawdown': max_drawdown,
        'avg_drawdown_duration': avg_drawdown_duration,
        'max_drawdown_duration': max_drawdown_duration,
        'recovery_time': recovery_time,
        'current_drawdown': drawdown.iloc[-1],
        'drawdown_periods_count': len(drawdown_periods),
        'time_in_drawdown': sum(drawdown_periods) / len(equity_curve)
    }


def exposure_analysis(positions: pd.Series,
                     benchmark_positions: Optional[pd.Series] = None) -> Dict[str, float]:
    """
    Analyze portfolio exposure patterns

    Args:
        positions: Position series (1=long, 0=flat, -1=short)
        benchmark_positions: Optional benchmark positions for comparison

    Returns:
        Dictionary with exposure metrics
    """
    # Calculate exposure statistics
    long_exposure = (positions > 0).mean()
    short_exposure = (positions < 0).mean()
    flat_exposure = (positions == 0).mean()

    # Gross exposure (absolute position)
    gross_exposure = positions.abs().mean()

    # Net exposure (directional bias)
    net_exposure = positions.mean()

    # Turnover (position changes)
    position_changes = positions.diff().abs()
    turnover = position_changes.sum() / 2  # Divide by 2 for one-way turnover

    # Average holding period
    position_changes_binary = (positions.diff() != 0)
    if position_changes_binary.sum() > 0:
        avg_holding_period = len(positions) / position_changes_binary.sum()
    else:
        avg_holding_period = len(positions)

    result = {
        'long_exposure': long_exposure,
        'short_exposure': short_exposure,
        'flat_exposure': flat_exposure,
        'gross_exposure': gross_exposure,
        'net_exposure': net_exposure,
        'turnover': turnover,
        'avg_holding_period': avg_holding_period
    }

    # Beta analysis if benchmark provided
    if benchmark_positions is not None:
        aligned_data = pd.DataFrame({
            'strategy': positions,
            'benchmark': benchmark_positions
        }).dropna()

        if len(aligned_data) > 1:
            correlation = aligned_data['strategy'].corr(aligned_data['benchmark'])
            result['beta_to_benchmark'] = correlation
        else:
            result['beta_to_benchmark'] = 0.0

    return result


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis class

    This class provides a unified interface for calculating
    all performance metrics with proper statistical rigor.
    """

    def __init__(self,
                 risk_free_rate: float = 0.02,
                 periods_per_year: int = 252):
        """
        Initialize performance analyzer

        Args:
            risk_free_rate: Annual risk-free rate
            periods_per_year: Periods per year for annualization
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def analyze_returns(self,
                       returns: pd.Series,
                       positions: Optional[pd.Series] = None,
                       trades_df: Optional[pd.DataFrame] = None,
                       num_trials: int = 1) -> Dict[str, Any]:
        """
        Comprehensive return analysis

        Args:
            returns: Return series
            positions: Position series (optional)
            trades_df: Individual trades (optional)
            num_trials: Number of strategies tested (for DSR)

        Returns:
            Dictionary with all performance metrics
        """
        if len(returns) == 0:
            return {'error': 'No returns provided'}

        # Basic metrics
        basic_metrics = self._calculate_basic_metrics(returns)

        # Statistical significance
        psr_result = probabilistic_sharpe_ratio(returns)
        dsr_result = deflated_sharpe_ratio(returns, num_trials)

        # Track record length
        min_trl = minimum_track_record_length(
            basic_metrics.sharpe_ratio,
            prob_sr=0.95
        )

        # Hit rate analysis
        hit_rate_result = hit_rate_analysis(returns, trades_df)

        # Drawdown analysis
        equity_curve = (1 + returns).cumprod()
        drawdown_result = drawdown_analysis(equity_curve)

        # Exposure analysis (if positions provided)
        exposure_result = {}
        if positions is not None:
            exposure_result = exposure_analysis(positions)

        # Compile results
        results = {
            'basic_metrics': basic_metrics.__dict__,
            'statistical_significance': {
                'psr': psr_result,
                'dsr': dsr_result,
                'min_track_record_length': min_trl,
                'current_length': len(returns),
                'sufficient_data': len(returns) >= min_trl
            },
            'hit_rate_analysis': hit_rate_result,
            'drawdown_analysis': drawdown_result,
            'exposure_analysis': exposure_result
        }

        return results

    def _calculate_basic_metrics(self, returns: pd.Series) -> PerformanceMetrics:
        """Calculate basic performance metrics"""
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (self.periods_per_year / len(returns)) - 1
        volatility = returns.std() * np.sqrt(self.periods_per_year)

        excess_return = annualized_return - self.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0

        # Equity curve for drawdown
        equity_curve = (1 + returns).cumprod()
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        # Trade-level metrics
        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]

        win_rate = len(winning_returns) / len(returns) if len(returns) > 0 else 0

        gross_profit = winning_returns.sum() if len(winning_returns) > 0 else 0
        gross_loss = abs(losing_returns.sum()) if len(losing_returns) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            trades_count=len(returns),
            avg_trade_return=returns.mean(),
            best_trade=returns.max() if len(returns) > 0 else 0,
            worst_trade=returns.min() if len(returns) > 0 else 0
        )

    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate comprehensive performance report"""
        if 'error' in analysis_results:
            return f"Error: {analysis_results['error']}"

        basic = analysis_results['basic_metrics']
        stat_sig = analysis_results['statistical_significance']
        hit_rate = analysis_results['hit_rate_analysis']
        drawdown = analysis_results['drawdown_analysis']

        report = []
        report.append("=" * 60)
        report.append("COMPREHENSIVE PERFORMANCE ANALYSIS")
        report.append("=" * 60)

        # Basic metrics
        report.append(f"\nBASIC METRICS:")
        report.append(f"  Total Return: {basic['total_return']:.2%}")
        report.append(f"  Annualized Return: {basic['annualized_return']:.2%}")
        report.append(f"  Volatility: {basic['volatility']:.2%}")
        report.append(f"  Sharpe Ratio: {basic['sharpe_ratio']:.3f}")
        report.append(f"  Max Drawdown: {basic['max_drawdown']:.2%}")

        # Statistical significance
        psr = stat_sig['psr']['psr']
        dsr = stat_sig['dsr']['dsr']
        report.append(f"\nSTATISTICAL SIGNIFICANCE:")
        report.append(f"  Probabilistic Sharpe Ratio: {psr:.3f}")
        report.append(f"  Deflated Sharpe Ratio: {dsr:.3f}")
        report.append(f"  Minimum Track Record: {stat_sig['min_track_record_length']} periods")
        report.append(f"  Current Track Record: {stat_sig['current_length']} periods")

        if stat_sig['sufficient_data']:
            report.append("  ✅ Sufficient data for statistical significance")
        else:
            report.append("  ⚠️ Insufficient data for statistical significance")

        if psr > 0.95:
            report.append("  ✅ Statistically significant (PSR > 95%)")
        else:
            report.append("  ❌ Not statistically significant (PSR < 95%)")

        # Hit rate analysis
        report.append(f"\nHIT RATE ANALYSIS:")
        report.append(f"  Win Rate: {hit_rate['hit_rate']:.2%}")
        report.append(f"  Average Win: {hit_rate['avg_win']:.3%}")
        report.append(f"  Average Loss: {hit_rate['avg_loss']:.3%}")
        report.append(f"  Win/Loss Ratio: {hit_rate['win_loss_ratio']:.2f}")
        report.append(f"  Profit Factor: {hit_rate['profit_factor']:.2f}")
        report.append(f"  Kelly Fraction: {hit_rate['kelly_fraction']:.2%}")

        # Drawdown analysis
        report.append(f"\nDRAWDOWN ANALYSIS:")
        report.append(f"  Max Drawdown: {drawdown['max_drawdown']:.2%}")
        report.append(f"  Average DD Duration: {drawdown['avg_drawdown_duration']:.1f} periods")
        report.append(f"  Max DD Duration: {drawdown['max_drawdown_duration']} periods")
        report.append(f"  Recovery Time: {drawdown['recovery_time']} periods")
        report.append(f"  Time in Drawdown: {drawdown['time_in_drawdown']:.2%}")

        report.append("=" * 60)

        return "\n".join(report)


# Testing functions
def test_probabilistic_sharpe_ratio():
    """Test PSR calculation"""
    print("\n🧪 Testing Probabilistic Sharpe Ratio")

    # Create test returns with known Sharpe ratio
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 1000))  # ~0.16 Sharpe

    psr_result = probabilistic_sharpe_ratio(returns)

    assert 'psr' in psr_result, "Missing PSR value"
    assert 0 <= psr_result['psr'] <= 1, "PSR should be between 0 and 1"
    assert 'sharpe_ratio' in psr_result, "Missing Sharpe ratio"

    print(f"  Sharpe Ratio: {psr_result['sharpe_ratio']:.3f}")
    print(f"  PSR: {psr_result['psr']:.3f}")
    print(f"  Statistically Significant: {psr_result['significant_95']}")
    print("✅ PSR test passed")


def test_performance_analyzer():
    """Test comprehensive performance analyzer"""
    print("\n🧪 Testing Performance Analyzer")

    # Create test data
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0005, 0.01, 252))  # Daily returns
    positions = pd.Series(np.random.choice([-1, 0, 1], 252))

    analyzer = PerformanceAnalyzer()
    results = analyzer.analyze_returns(returns, positions, num_trials=10)

    # Verify results structure
    assert 'basic_metrics' in results, "Missing basic metrics"
    assert 'statistical_significance' in results, "Missing statistical significance"
    assert 'hit_rate_analysis' in results, "Missing hit rate analysis"
    assert 'drawdown_analysis' in results, "Missing drawdown analysis"

    basic = results['basic_metrics']
    assert 'sharpe_ratio' in basic, "Missing Sharpe ratio in basic metrics"
    assert 'max_drawdown' in basic, "Missing max drawdown"

    print(f"  Sharpe Ratio: {basic['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {basic['max_drawdown']:.2%}")
    print(f"  Win Rate: {results['hit_rate_analysis']['hit_rate']:.2%}")
    print("✅ Performance analyzer test passed")


def test_capacity_analysis():
    """Test capacity analysis"""
    print("\n🧪 Testing Capacity Analysis")

    # Create test data
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.002, 0.01, 100))
    position_sizes = pd.Series(np.random.uniform(0.01, 0.05, 100))

    capacity_result = calculate_capacity(returns, position_sizes)

    assert 'max_capacity' in capacity_result, "Missing max capacity"
    assert 'capacity_slope' in capacity_result, "Missing capacity slope"
    assert capacity_result['max_capacity'] > 0, "Max capacity should be positive"

    print(f"  Max Capacity: {capacity_result['max_capacity']:.2f}x")
    print(f"  Capacity Slope: {capacity_result['capacity_slope']:.4f}")
    print(f"  Gross Return: {capacity_result['gross_return']:.3%}")
    print("✅ Capacity analysis test passed")


if __name__ == "__main__":
    print("🧪 Running Performance Metrics Tests")
    test_probabilistic_sharpe_ratio()
    test_performance_analyzer()
    test_capacity_analysis()
    print("✅ All performance metrics tests passed!")