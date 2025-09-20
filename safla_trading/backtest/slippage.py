"""
SLIPPAGE AND MARKET IMPACT MODULE
Ernie Chan + Linus Torvalds: Realistic execution modeling

This module models the real cost of moving markets when trading.
Without slippage modeling, your backtest is living in fantasy land.

"Slippage is the difference between paper trading and poverty" - Ernie Chan
"Model the pain or feel the pain" - Linus Torvalds
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import math


class ExecutionModel(Enum):
    """Execution models for slippage calculation"""
    LINEAR = "linear"          # Linear impact
    SQUARE_ROOT = "sqrt"       # Square root impact
    LOGARITHMIC = "log"        # Logarithmic impact
    ALMGREN_CHRISS = "ac"      # Almgren-Chriss model


@dataclass
class SlippageParams:
    """
    Parameters for slippage calculation

    Attributes:
        impact_coefficient: Market impact coefficient
        adv_lookback: Days for calculating average daily volume
        participation_rate: Maximum participation rate in daily volume
        noise_factor: Random noise factor (0 = no noise, 1 = high noise)
        temporary_impact_decay: How quickly temporary impact decays
    """
    impact_coefficient: float = 0.1
    adv_lookback: int = 30
    participation_rate: float = 0.1  # 10% of daily volume
    noise_factor: float = 0.1
    temporary_impact_decay: float = 0.5


def calculate_adv(volume_series: pd.Series,
                 lookback_days: int = 30) -> float:
    """
    Calculate Average Daily Volume (ADV)

    Args:
        volume_series: Series of trading volumes
        lookback_days: Number of days to look back

    Returns:
        Average daily volume

    Example:
        adv = calculate_adv(data['volume'], lookback_days=30)
    """
    if len(volume_series) < lookback_days:
        return volume_series.mean()

    return volume_series.tail(lookback_days).mean()


def linear_impact(trade_size: float,
                 adv: float,
                 impact_coefficient: float) -> float:
    """
    Calculate linear market impact

    Args:
        trade_size: Size of trade in shares/contracts
        adv: Average daily volume
        impact_coefficient: Impact coefficient (calibrated parameter)

    Returns:
        Market impact as fraction of price

    Formula: impact = k * (trade_size / adv)
    """
    if adv <= 0:
        return 0.0

    participation_rate = abs(trade_size) / adv
    impact = impact_coefficient * participation_rate

    return impact


def sqrt_impact(trade_size: float,
               adv: float,
               impact_coefficient: float) -> float:
    """
    Calculate square-root market impact (more realistic for large trades)

    Args:
        trade_size: Size of trade
        adv: Average daily volume
        impact_coefficient: Impact coefficient

    Returns:
        Market impact as fraction of price

    Formula: impact = k * sqrt(trade_size / adv)
    """
    if adv <= 0:
        return 0.0

    participation_rate = abs(trade_size) / adv
    impact = impact_coefficient * math.sqrt(participation_rate)

    return impact


def logarithmic_impact(trade_size: float,
                      adv: float,
                      impact_coefficient: float) -> float:
    """
    Calculate logarithmic market impact (for very large trades)

    Args:
        trade_size: Size of trade
        adv: Average daily volume
        impact_coefficient: Impact coefficient

    Returns:
        Market impact as fraction of price

    Formula: impact = k * log(1 + trade_size / adv)
    """
    if adv <= 0:
        return 0.0

    participation_rate = abs(trade_size) / adv
    impact = impact_coefficient * math.log(1 + participation_rate)

    return impact


def almgren_chriss_impact(trade_size: float,
                         volatility: float,
                         time_horizon: float,
                         adv: float,
                         gamma: float = 0.1) -> Dict[str, float]:
    """
    Almgren-Chriss optimal execution model

    Args:
        trade_size: Total trade size
        volatility: Price volatility (daily)
        time_horizon: Execution time horizon (days)
        adv: Average daily volume
        gamma: Risk aversion parameter

    Returns:
        Dictionary with optimal execution trajectory and costs

    This is the academic gold standard for execution modeling
    """
    if adv <= 0 or time_horizon <= 0:
        return {
            'permanent_impact': 0.0,
            'temporary_impact': 0.0,
            'total_impact': 0.0,
            'optimal_rate': 0.0
        }

    # Market impact parameters (simplified)
    eta = 0.1  # Temporary impact coefficient
    tau = 0.05  # Permanent impact coefficient

    # Participation rate
    participation = abs(trade_size) / (adv * time_horizon)

    # Permanent impact (affects price permanently)
    permanent_impact = tau * participation

    # Temporary impact (affects only during execution)
    temporary_impact = eta * participation / math.sqrt(time_horizon)

    # Total expected impact
    total_impact = permanent_impact + temporary_impact

    # Optimal execution rate (constant rate for simplicity)
    optimal_rate = trade_size / time_horizon

    return {
        'permanent_impact': permanent_impact,
        'temporary_impact': temporary_impact,
        'total_impact': total_impact,
        'optimal_rate': optimal_rate,
        'participation_rate': participation
    }


class SlippageModel:
    """
    Comprehensive slippage model for backtesting

    This class provides realistic slippage estimation based on:
    - Trade size relative to average daily volume
    - Market volatility
    - Execution style (aggressive vs passive)
    - Random noise to simulate real-world uncertainty
    """

    def __init__(self,
                 slippage_params: SlippageParams,
                 execution_model: ExecutionModel = ExecutionModel.SQUARE_ROOT,
                 random_seed: Optional[int] = None):
        """
        Initialize slippage model

        Args:
            slippage_params: Slippage calculation parameters
            execution_model: Model for calculating market impact
            random_seed: Seed for random noise (None = no seed)
        """
        self.params = slippage_params
        self.execution_model = execution_model
        self.rng = np.random.RandomState(random_seed) if random_seed else np.random

    def calculate_slippage(self,
                          trade_size: float,
                          price: float,
                          volume_series: pd.Series,
                          volatility: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate total slippage for a trade

        Args:
            trade_size: Size of trade (positive = buy, negative = sell)
            price: Current market price
            volume_series: Historical volume data
            volatility: Price volatility (optional, estimated if not provided)

        Returns:
            Dictionary with slippage breakdown

        Example:
            slippage = model.calculate_slippage(
                trade_size=1000,
                price=50000,
                volume_series=btc_volume,
                volatility=0.03
            )
        """
        # Calculate average daily volume
        adv = calculate_adv(volume_series, self.params.adv_lookback)

        if adv <= 0:
            return self._zero_slippage_result()

        # Calculate market impact based on model
        if self.execution_model == ExecutionModel.LINEAR:
            market_impact = linear_impact(trade_size, adv, self.params.impact_coefficient)
        elif self.execution_model == ExecutionModel.SQUARE_ROOT:
            market_impact = sqrt_impact(trade_size, adv, self.params.impact_coefficient)
        elif self.execution_model == ExecutionModel.LOGARITHMIC:
            market_impact = logarithmic_impact(trade_size, adv, self.params.impact_coefficient)
        elif self.execution_model == ExecutionModel.ALMGREN_CHRISS:
            if volatility is None:
                volatility = volume_series.pct_change().std()
            ac_result = almgren_chriss_impact(trade_size, volatility, 1.0, adv)
            market_impact = ac_result['total_impact']
        else:
            market_impact = sqrt_impact(trade_size, adv, self.params.impact_coefficient)

        # Add random noise
        if self.params.noise_factor > 0:
            noise = self.rng.normal(0, market_impact * self.params.noise_factor)
            market_impact += noise

        # Ensure impact is always positive and reasonable
        market_impact = abs(market_impact)
        market_impact = min(market_impact, 0.1)  # Cap at 10%

        # Calculate slippage in price terms
        slippage_amount = price * market_impact

        # Apply direction (buy = positive slippage, sell = negative)
        direction = 1 if trade_size > 0 else -1
        slippage_amount *= direction

        # Calculate participation rate
        participation_rate = abs(trade_size) / adv

        return {
            'slippage_bps': market_impact * 10000,
            'slippage_amount': slippage_amount,
            'slippage_percentage': market_impact,
            'participation_rate': participation_rate,
            'adv': adv,
            'impact_coefficient_used': self.params.impact_coefficient
        }

    def _zero_slippage_result(self) -> Dict[str, float]:
        """Return zero slippage result for edge cases"""
        return {
            'slippage_bps': 0.0,
            'slippage_amount': 0.0,
            'slippage_percentage': 0.0,
            'participation_rate': 0.0,
            'adv': 0.0,
            'impact_coefficient_used': 0.0
        }

    def apply_slippage_to_trades(self,
                               trades_df: pd.DataFrame,
                               price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply slippage to a dataframe of trades

        Args:
            trades_df: DataFrame with columns ['timestamp', 'size', 'price']
            price_data: DataFrame with OHLCV data

        Returns:
            DataFrame with slippage-adjusted prices and costs

        Example:
            trades = pd.DataFrame({
                'timestamp': [...],
                'size': [1000, -500, 2000],
                'price': [50000, 50100, 50200]
            })
            adjusted_trades = model.apply_slippage_to_trades(trades, price_data)
        """
        result_trades = trades_df.copy()

        slippage_amounts = []
        adjusted_prices = []

        for _, trade in trades_df.iterrows():
            # Get volume series up to trade date
            trade_date = trade['timestamp']
            historical_volume = price_data[price_data.index <= trade_date]['volume']

            if len(historical_volume) == 0:
                slippage_amounts.append(0.0)
                adjusted_prices.append(trade['price'])
                continue

            # Calculate slippage
            slippage_result = self.calculate_slippage(
                trade_size=trade['size'],
                price=trade['price'],
                volume_series=historical_volume
            )

            slippage_amount = slippage_result['slippage_amount']
            adjusted_price = trade['price'] + slippage_amount

            slippage_amounts.append(slippage_amount)
            adjusted_prices.append(adjusted_price)

        result_trades['slippage_amount'] = slippage_amounts
        result_trades['adjusted_price'] = adjusted_prices
        result_trades['slippage_bps'] = [
            abs(slip / price) * 10000 if price != 0 else 0
            for slip, price in zip(slippage_amounts, trades_df['price'])
        ]

        return result_trades

    def calibrate_impact_coefficient(self,
                                   realized_trades: pd.DataFrame,
                                   price_data: pd.DataFrame) -> float:
        """
        Calibrate impact coefficient from realized trading data

        Args:
            realized_trades: DataFrame with actual trade data
            price_data: Historical price and volume data

        Returns:
            Calibrated impact coefficient

        This is crucial for backtesting accuracy - use real execution data
        """
        if len(realized_trades) == 0:
            return self.params.impact_coefficient

        total_impact = 0.0
        valid_trades = 0

        for _, trade in realized_trades.iterrows():
            # Calculate theoretical impact without coefficient
            trade_date = trade['timestamp']
            historical_volume = price_data[price_data.index <= trade_date]['volume']
            adv = calculate_adv(historical_volume, self.params.adv_lookback)

            if adv <= 0:
                continue

            # Actual slippage from trade
            actual_slippage = trade.get('actual_slippage', 0)

            # Theoretical participation rate
            participation_rate = abs(trade['size']) / adv

            if participation_rate > 0:
                # Back out the impact coefficient
                if self.execution_model == ExecutionModel.SQUARE_ROOT:
                    implied_coefficient = actual_slippage / math.sqrt(participation_rate)
                else:  # Linear
                    implied_coefficient = actual_slippage / participation_rate

                total_impact += implied_coefficient
                valid_trades += 1

        if valid_trades > 0:
            calibrated_coefficient = total_impact / valid_trades
            return max(0.001, min(calibrated_coefficient, 1.0))  # Reasonable bounds

        return self.params.impact_coefficient


def create_twap_execution(total_size: float,
                         time_horizon_minutes: int,
                         interval_minutes: int = 5) -> pd.DataFrame:
    """
    Create TWAP (Time-Weighted Average Price) execution schedule

    Args:
        total_size: Total trade size to execute
        time_horizon_minutes: Total execution time in minutes
        interval_minutes: Interval between trades

    Returns:
        DataFrame with execution schedule

    Example:
        # Execute 1000 shares over 60 minutes in 5-minute intervals
        schedule = create_twap_execution(1000, 60, 5)
    """
    num_intervals = time_horizon_minutes // interval_minutes
    size_per_interval = total_size / num_intervals

    times = pd.date_range(
        start='2024-01-01 09:30:00',
        periods=num_intervals,
        freq=f'{interval_minutes}min'
    )

    schedule = pd.DataFrame({
        'timestamp': times,
        'size': size_per_interval,
        'cumulative_size': np.arange(1, num_intervals + 1) * size_per_interval
    })

    return schedule


def create_vwap_execution(total_size: float,
                         volume_profile: pd.Series) -> pd.DataFrame:
    """
    Create VWAP (Volume-Weighted Average Price) execution schedule

    Args:
        total_size: Total trade size to execute
        volume_profile: Historical volume profile by time of day

    Returns:
        DataFrame with execution schedule weighted by volume

    Example:
        # Use historical volume pattern to weight execution
        volume_profile = historical_data.groupby(data.index.time)['volume'].mean()
        schedule = create_vwap_execution(1000, volume_profile)
    """
    # Normalize volume profile to weights
    weights = volume_profile / volume_profile.sum()
    sizes = weights * total_size

    schedule = pd.DataFrame({
        'time': volume_profile.index,
        'size': sizes,
        'weight': weights,
        'cumulative_size': sizes.cumsum()
    })

    return schedule


# Testing functions
def test_impact_models():
    """Test different impact models"""
    print("\n🧪 Testing Impact Models")

    trade_size = 1000
    adv = 100000  # 100k daily volume
    coefficient = 0.1

    # Test linear impact
    linear = linear_impact(trade_size, adv, coefficient)
    expected_linear = coefficient * (trade_size / adv)
    assert abs(linear - expected_linear) < 0.001, f"Linear impact: {linear} != {expected_linear}"

    # Test sqrt impact
    sqrt = sqrt_impact(trade_size, adv, coefficient)
    expected_sqrt = coefficient * math.sqrt(trade_size / adv)
    assert abs(sqrt - expected_sqrt) < 0.001, f"Sqrt impact: {sqrt} != {expected_sqrt}"

    # Test log impact
    log = logarithmic_impact(trade_size, adv, coefficient)
    expected_log = coefficient * math.log(1 + trade_size / adv)
    assert abs(log - expected_log) < 0.001, f"Log impact: {log} != {expected_log}"

    print(f"  Linear impact: {linear:.4f}")
    print(f"  Sqrt impact: {sqrt:.4f}")
    print(f"  Log impact: {log:.4f}")
    print("✅ Impact model tests passed")


def test_slippage_model():
    """Test comprehensive slippage model"""
    print("\n🧪 Testing Slippage Model")

    # Create test data
    dates = pd.date_range('2024-01-01', periods=100, freq='h')
    volume = pd.Series(
        np.random.RandomState(42).randint(10000, 100000, 100),
        index=dates
    )

    # Initialize model
    params = SlippageParams(
        impact_coefficient=0.1,
        noise_factor=0.0  # No noise for testing
    )
    model = SlippageModel(params, ExecutionModel.SQUARE_ROOT, random_seed=42)

    # Test slippage calculation
    slippage = model.calculate_slippage(
        trade_size=1000,
        price=50000,
        volume_series=volume
    )

    # Verify result structure
    required_keys = ['slippage_bps', 'slippage_amount', 'participation_rate']
    for key in required_keys:
        assert key in slippage, f"Missing slippage key: {key}"

    # Verify reasonable values
    assert slippage['slippage_bps'] >= 0, "Slippage should be positive"
    assert slippage['participation_rate'] >= 0, "Participation rate should be positive"

    print(f"  Slippage: {slippage['slippage_bps']:.1f} bps")
    print(f"  Participation rate: {slippage['participation_rate']:.4f}")
    print("✅ Slippage model test passed")


def test_execution_schedules():
    """Test TWAP and VWAP execution schedules"""
    print("\n🧪 Testing Execution Schedules")

    # Test TWAP
    twap_schedule = create_twap_execution(1000, 60, 5)
    assert len(twap_schedule) == 12, f"TWAP schedule length: {len(twap_schedule)} != 12"
    assert abs(twap_schedule['size'].sum() - 1000) < 0.001, "TWAP total size mismatch"

    # Test VWAP
    volume_profile = pd.Series([100, 200, 300, 200, 100],
                              index=pd.date_range('09:30', '13:30', freq='h').time)
    vwap_schedule = create_vwap_execution(1000, volume_profile)
    assert abs(vwap_schedule['size'].sum() - 1000) < 0.001, "VWAP total size mismatch"

    print(f"  TWAP intervals: {len(twap_schedule)}")
    print(f"  VWAP weighted distribution: {vwap_schedule['weight'].tolist()}")
    print("✅ Execution schedule tests passed")


if __name__ == "__main__":
    print("🧪 Running Slippage Model Tests")
    test_impact_models()
    test_slippage_model()
    test_execution_schedules()
    print("✅ All slippage tests passed!")