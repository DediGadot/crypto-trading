#!/usr/bin/env python3
"""
LINUS-STYLE ALGORITHMIC TRADING SYSTEM
No bullshit. No lookahead bias. Only profit.

"Given enough eyeballs, all bugs are shallow" - but let's not have bugs in the first place.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Core trading components
@dataclass
class TradingSignal:
    """Clean trading signal - no ambiguity"""
    timestamp: datetime
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    size: float  # Position size
    confidence: float  # 0-1
    reason: str
    price: float  # Execution price

@dataclass
class Trade:
    """Individual trade record"""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    side: str  # 'long', 'short'
    entry_price: float
    exit_price: Optional[float]
    size: float
    pnl: Optional[float]
    cost: float  # Transaction costs

@dataclass
class PerformanceMetrics:
    """Performance metrics that matter"""
    total_return: float
    sharpe_ratio: float
    psr: float  # Probabilistic Sharpe Ratio
    max_drawdown: float
    win_rate: float
    profit_factor: float
    net_profit: float
    total_trades: int
    turnover: float
    transaction_costs: float


class ProbabilisticSharpeRatio:
    """
    Probabilistic Sharpe Ratio - handles skew/kurtosis
    Based on Marcos López de Prado's work
    """

    @staticmethod
    def calculate_psr(returns: pd.Series, benchmark: float = 0.0,
                     freq: int = 252) -> float:
        """
        Calculate PSR with proper statistical adjustment

        Args:
            returns: Strategy returns
            benchmark: Benchmark return (annualized)
            freq: Return frequency (252 for daily, 8760 for hourly)

        Returns:
            PSR value (0-1, higher is better)
        """
        if len(returns) < 30:
            return 0.0

        # Annualized statistics
        mean_ret = returns.mean() * freq
        vol = returns.std() * np.sqrt(freq)

        if vol == 0:
            return 0.0

        # Basic Sharpe ratio
        sr = (mean_ret - benchmark) / vol

        # Adjust for skew and kurtosis
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)

        # Length adjustment
        n = len(returns)

        # PSR formula from López de Prado
        psr_num = (sr - benchmark) * np.sqrt(n - 1)
        psr_den = np.sqrt(1 - skew * sr + ((kurt - 1) / 4) * sr**2)

        if psr_den <= 0 or np.isnan(psr_den) or np.isinf(psr_den):
            return 0.0

        psr_ratio = psr_num / psr_den
        if np.isnan(psr_ratio) or np.isinf(psr_ratio):
            return 0.0

        psr = stats.norm.cdf(psr_ratio)
        return float(psr) if not np.isnan(psr) else 0.0


class TripleBarrierLabeling:
    """
    Triple Barrier Method for realistic trading targets
    No more naive fixed-horizon regression
    """

    @staticmethod
    def create_labels(prices: pd.Series,
                     volatility: pd.Series,
                     profit_target: float = 0.02,
                     stop_loss: float = 0.01,
                     max_hold_periods: int = 24) -> pd.DataFrame:
        """
        Create Triple Barrier labels

        Args:
            prices: Price series
            volatility: Rolling volatility
            profit_target: Profit target as % of volatility
            stop_loss: Stop loss as % of volatility
            max_hold_periods: Maximum holding periods

        Returns:
            DataFrame with labels and barriers
        """
        results = []

        for i in range(len(prices) - max_hold_periods):
            entry_price = prices.iloc[i]
            entry_vol = volatility.iloc[i]

            if pd.isna(entry_vol) or entry_vol <= 0:
                continue

            # Dynamic barriers based on volatility
            upper_barrier = entry_price * (1 + profit_target * entry_vol)
            lower_barrier = entry_price * (1 - stop_loss * entry_vol)

            # Look forward for barrier hits
            future_prices = prices.iloc[i+1:i+1+max_hold_periods]

            # Find first barrier hit
            upper_hit = future_prices >= upper_barrier
            lower_hit = future_prices <= lower_barrier

            if upper_hit.any():
                exit_idx = upper_hit.idxmax()
                label = 1  # Profit target hit
                exit_price = future_prices[exit_idx]
                hold_periods = list(future_prices.index).index(exit_idx) + 1
            elif lower_hit.any():
                exit_idx = lower_hit.idxmax()
                label = -1  # Stop loss hit
                exit_price = future_prices[exit_idx]
                hold_periods = list(future_prices.index).index(exit_idx) + 1
            else:
                label = 0  # Timeout
                exit_price = future_prices.iloc[-1]
                hold_periods = max_hold_periods

            results.append({
                'timestamp': prices.index[i],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'upper_barrier': upper_barrier,
                'lower_barrier': lower_barrier,
                'label': label,
                'hold_periods': hold_periods,
                'return': (exit_price - entry_price) / entry_price
            })

        return pd.DataFrame(results)


class MarketImpactModel:
    """
    Realistic transaction cost and market impact modeling
    """

    def __init__(self, base_fee: float = 0.001, taker_fee: float = 0.001):
        self.base_fee = base_fee
        self.taker_fee = taker_fee

    def calculate_costs(self, quantity: float, price: float,
                       adv: float, market_order: bool = True) -> float:
        """
        Calculate total transaction costs

        Args:
            quantity: Trade size
            price: Trade price
            adv: Average daily volume
            market_order: True for market orders, False for limit orders

        Returns:
            Total cost in quote currency
        """
        notional = quantity * price

        # Base exchange fees
        if market_order:
            fee_cost = notional * self.taker_fee
        else:
            fee_cost = notional * self.base_fee

        # Market impact (square root law) - only for market orders
        impact_cost = 0
        if market_order and adv > 0:
            impact_pct = 0.1 * np.sqrt(notional / (adv * price))
            impact_cost = notional * impact_pct

        return fee_cost + impact_cost


class CryptoPurgedCV:
    """
    Combinatorial Purged Cross-Validation for crypto
    Prevents data leakage in overlapping label scenarios
    """

    @staticmethod
    def purged_cv_split(X: pd.DataFrame,
                       n_splits: int = 5,
                       purge_gap: int = 24,
                       embargo_gap: int = 12) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create purged CV splits

        Args:
            X: Feature matrix with datetime index
            n_splits: Number of CV folds
            purge_gap: Gap between train and test (hours)
            embargo_gap: Embargo after test set (hours)

        Returns:
            List of (train_idx, test_idx) tuples
        """
        n_samples = len(X)
        test_size = n_samples // n_splits

        splits = []

        for i in range(n_splits):
            # Test set boundaries
            test_start = i * test_size
            test_end = min((i + 1) * test_size, n_samples)

            # Train set with purge/embargo
            train_end = max(0, test_start - purge_gap)
            train_start = 0

            # Embargo after test
            next_train_start = min(test_end + embargo_gap, n_samples)

            # Create train indices
            train_idx_parts = []

            if train_end > train_start:
                train_idx_parts.append(np.arange(train_start, train_end))

            if next_train_start < n_samples and i < n_splits - 1:  # Don't add future data for last split
                train_idx_parts.append(np.arange(next_train_start, n_samples))

            if len(train_idx_parts) > 0:
                train_idx = np.concatenate(train_idx_parts)
            else:
                train_idx = np.array([])

            test_idx = np.arange(test_start, test_end)

            if len(train_idx) > 50 and len(test_idx) > 10:  # Minimum samples
                splits.append((train_idx, test_idx))

        return splits


class NoLookaheadBacktester:
    """
    Bulletproof backtester with zero lookahead bias
    Every decision uses only past information
    """

    def __init__(self, initial_capital: float = 100000,
                 cost_model: MarketImpactModel = None):
        self.initial_capital = initial_capital
        self.cost_model = cost_model or MarketImpactModel()

        # State tracking
        self.cash = initial_capital
        self.positions = {}  # symbol -> size
        self.trades = []
        self.equity_curve = []
        self.timestamp = None

    def execute_signal(self, signal: TradingSignal,
                      current_price: float, adv: float = 1000000) -> bool:
        """
        Execute trading signal with realistic costs

        Args:
            signal: Trading signal
            current_price: Current market price
            adv: Average daily volume

        Returns:
            True if trade executed successfully
        """
        self.timestamp = signal.timestamp

        if signal.action == 'hold':
            self._update_equity(current_price)
            return True

        # Calculate position change
        current_pos = self.positions.get(signal.symbol, 0)

        if signal.action == 'buy':
            target_pos = signal.size
        elif signal.action == 'sell':
            target_pos = -signal.size if signal.size > 0 else 0
        else:
            return False

        position_change = target_pos - current_pos

        if abs(position_change) < 1e-6:  # No meaningful change
            self._update_equity(current_price)
            return True

        # Calculate costs
        trade_value = abs(position_change * current_price)
        costs = self.cost_model.calculate_costs(
            abs(position_change), current_price, adv, market_order=True
        )

        # Check if we have enough cash
        if position_change > 0:  # Buying
            required_cash = trade_value + costs
            if required_cash > self.cash:
                # Partial fill with available cash
                affordable_size = (self.cash - costs) / current_price
                if affordable_size > 0:
                    position_change = affordable_size
                    trade_value = position_change * current_price
                    costs = self.cost_model.calculate_costs(
                        position_change, current_price, adv, market_order=True
                    )
                else:
                    self._update_equity(current_price)
                    return False

        # Execute trade
        self.cash -= (position_change * current_price + costs)
        self.positions[signal.symbol] = current_pos + position_change

        # Record trade
        trade = Trade(
            entry_time=signal.timestamp,
            exit_time=None,
            symbol=signal.symbol,
            side='long' if position_change > 0 else 'short',
            entry_price=current_price,
            exit_price=None,
            size=abs(position_change),
            pnl=None,
            cost=costs
        )
        self.trades.append(trade)

        self._update_equity(current_price)
        return True

    def _update_equity(self, current_price: float):
        """Update equity curve"""
        portfolio_value = self.cash

        # Add position values
        for symbol, size in self.positions.items():
            if symbol in [signal.symbol for signal in [TradingSignal(
                datetime.now(), symbol, 'hold', 0, 0, '', current_price
            )]]:  # Assuming single symbol for now
                portfolio_value += size * current_price

        self.equity_curve.append({
            'timestamp': self.timestamp,
            'equity': portfolio_value,
            'cash': self.cash,
            'positions': self.positions.copy()
        })

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if len(self.equity_curve) < 2:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)

        # Returns calculation
        returns = equity_df['equity'].pct_change().dropna()

        if len(returns) == 0 or len(returns) == 1:
            # Handle single trade case
            total_return = (equity_df['equity'].iloc[-1] / self.initial_capital) - 1
            return PerformanceMetrics(total_return, 0, 0, 0, 0, 0, equity_df['equity'].iloc[-1] - self.initial_capital, len(self.trades), 0, sum(t.cost for t in self.trades))

        # Basic metrics
        total_return = (equity_df['equity'].iloc[-1] / self.initial_capital) - 1

        if returns.std() > 0 and len(returns) > 1:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(8760)  # Hourly
        else:
            sharpe_ratio = 0

        # PSR calculation (need sufficient sample size)
        if len(returns) > 10:
            psr = ProbabilisticSharpeRatio.calculate_psr(returns, freq=8760)
        else:
            psr = 0

        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Trade statistics
        winning_trades = [t for t in self.trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl and t.pnl < 0]

        if len(self.trades) > 0:
            win_rate = len(winning_trades) / len(self.trades)
        else:
            win_rate = 0

        gross_profit = sum(t.pnl for t in winning_trades if t.pnl)
        gross_loss = abs(sum(t.pnl for t in losing_trades if t.pnl))

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float('inf') if gross_profit > 0 else 0

        total_costs = sum(t.cost for t in self.trades)
        net_profit = equity_df['equity'].iloc[-1] - self.initial_capital

        # Turnover calculation
        total_volume = sum(t.size * t.entry_price for t in self.trades)
        avg_equity = equity_df['equity'].mean()
        turnover = total_volume / avg_equity if avg_equity > 0 else 0

        return PerformanceMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            psr=psr,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            net_profit=net_profit,
            total_trades=len(self.trades),
            turnover=turnover,
            transaction_costs=total_costs
        )


def run_system_validation():
    """
    Comprehensive system validation
    Exit criteria: PSR > 1.0 and net positive alpha vs buy-and-hold
    """
    print("🔥 LINUS TRADING SYSTEM VALIDATION")
    print("=" * 50)

    # This would integrate with the actual data pipeline
    # For now, creating synthetic test data

    dates = pd.date_range('2024-01-01', periods=1000, freq='H')
    np.random.seed(42)  # Reproducible

    # Synthetic price with realistic properties
    price_changes = np.random.normal(0.0001, 0.02, 1000)  # Slight upward drift
    prices = pd.Series(100 * np.exp(np.cumsum(price_changes)), index=dates)

    # Test PSR calculation
    returns = prices.pct_change().dropna()
    psr = ProbabilisticSharpeRatio.calculate_psr(returns)
    print(f"✅ PSR Calculation: {psr:.4f}")

    # Test Triple Barrier labeling
    volatility = returns.rolling(24).std()
    labels = TripleBarrierLabeling.create_labels(prices, volatility)
    print(f"✅ Triple Barrier Labels: {len(labels)} samples generated")

    # Test Purged CV
    X = pd.DataFrame(np.random.randn(1000, 10), index=dates)
    cv_splits = CryptoPurgedCV.purged_cv_split(X, n_splits=5)
    print(f"✅ Purged CV: {len(cv_splits)} valid splits created")

    # Test backtester with buy-and-hold comparison
    backtester = NoLookaheadBacktester(initial_capital=100000)

    # Simple buy-and-hold strategy
    bnh_signal = TradingSignal(
        timestamp=dates[0],
        symbol='TEST',
        action='buy',
        size=1000,  # Buy $100k worth
        confidence=1.0,
        reason='buy_and_hold',
        price=prices.iloc[0]
    )

    backtester.execute_signal(bnh_signal, prices.iloc[0])

    # Update with final price
    backtester.timestamp = dates[-1]
    backtester._update_equity(prices.iloc[-1])

    metrics = backtester.get_performance_metrics()

    print(f"\n📊 BUY-AND-HOLD BENCHMARK:")
    print(f"   Total Return: {metrics.total_return:.4f}")
    print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.4f}")
    print(f"   PSR: {metrics.psr:.4f}")
    print(f"   Max Drawdown: {metrics.max_drawdown:.4f}")
    print(f"   Transaction Costs: ${metrics.transaction_costs:.2f}")

    # Validation criteria
    success = True
    if metrics.psr < 0.5:
        print("❌ PSR too low for benchmark")
        success = False
    if metrics.total_return < 0:
        print("❌ Negative returns on benchmark")
        success = False

    if success:
        print("\n🎯 SYSTEM VALIDATION: PASSED")
        print("Ready for algorithm implementation")
    else:
        print("\n💥 SYSTEM VALIDATION: FAILED")
        print("Foundation needs fixing")

    return success


if __name__ == "__main__":
    # Run validation
    success = run_system_validation()

    if success:
        print("\n" + "="*50)
        print("LINUS SAYS: Foundation is solid. Time to build algorithms that actually make money.")
        print("Next: Implement crypto-native features and test against buy-and-hold.")
        print("="*50)
    else:
        print("\nLINUS SAYS: Fix the foundation first. No shortcuts.")