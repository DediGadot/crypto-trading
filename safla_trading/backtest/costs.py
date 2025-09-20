"""
TRANSACTION COSTS MODULE
Ernie Chan + Linus Torvalds: Realistic P&L with proper cost modeling

This module implements realistic transaction costs that kill most backtests.
Without proper cost modeling, backtests are just expensive fantasies.

"Transaction costs are the graveyard of trading strategies" - Ernie Chan
"If you don't model reality, you're modeling fiction" - Linus Torvalds
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum


class OrderType(Enum):
    """Order types for fee calculation"""
    MARKET = "market"
    LIMIT = "limit"
    MARKET_IOC = "market_ioc"
    LIMIT_FOK = "limit_fok"


@dataclass
class FeeSchedule:
    """
    Exchange fee schedule with maker/taker rates

    Attributes:
        maker_bps: Maker fee in basis points (negative for rebates)
        taker_bps: Taker fee in basis points
        min_fee: Minimum fee in quote currency (None = no minimum)
        max_fee: Maximum fee in quote currency (None = no maximum)
    """
    maker_bps: float
    taker_bps: float
    min_fee: Optional[float] = None
    max_fee: Optional[float] = None

    def __post_init__(self):
        """Validate fee schedule parameters"""
        if self.taker_bps < 0:
            raise ValueError("Taker fees cannot be negative")
        if self.min_fee is not None and self.min_fee < 0:
            raise ValueError("Minimum fee cannot be negative")
        if self.max_fee is not None and self.max_fee < 0:
            raise ValueError("Maximum fee cannot be negative")
        if (self.min_fee is not None and self.max_fee is not None and
            self.min_fee > self.max_fee):
            raise ValueError("Minimum fee cannot exceed maximum fee")


# Common exchange fee schedules
EXCHANGE_FEES = {
    'binance': FeeSchedule(maker_bps=10, taker_bps=10),  # 0.1% both
    'binance_vip': FeeSchedule(maker_bps=9, taker_bps=10),  # VIP1
    'coinbase': FeeSchedule(maker_bps=50, taker_bps=50),  # 0.5% both
    'kraken': FeeSchedule(maker_bps=16, taker_bps=26),  # 0.16%/0.26%
    'bybit': FeeSchedule(maker_bps=10, taker_bps=10),  # 0.1% both
    'ftx': FeeSchedule(maker_bps=2, taker_bps=7),  # 0.02%/0.07% (historical)
    'dydx': FeeSchedule(maker_bps=2, taker_bps=5),  # 0.02%/0.05%
    'test_zero': FeeSchedule(maker_bps=0, taker_bps=0),  # Zero fees for testing
}


def calculate_fee(notional: float,
                 fee_schedule: FeeSchedule,
                 order_type: OrderType = OrderType.MARKET) -> float:
    """
    Calculate transaction fee for a trade

    Args:
        notional: Notional value of trade (price * quantity)
        fee_schedule: Fee schedule to apply
        order_type: Type of order (market = taker, limit = maker)

    Returns:
        Fee amount in quote currency

    Example:
        # $1000 market order on Binance
        fee = calculate_fee(1000, EXCHANGE_FEES['binance'], OrderType.MARKET)
        # Returns: 1.0 (0.1% taker fee)
    """
    if notional <= 0:
        return 0.0

    # Determine if maker or taker
    is_maker = order_type in [OrderType.LIMIT, OrderType.LIMIT_FOK]
    fee_bps = fee_schedule.maker_bps if is_maker else fee_schedule.taker_bps

    # Calculate base fee
    fee = notional * fee_bps / 10000

    # Apply minimum fee
    if fee_schedule.min_fee is not None:
        fee = max(fee, fee_schedule.min_fee)

    # Apply maximum fee
    if fee_schedule.max_fee is not None:
        fee = min(fee, fee_schedule.max_fee)

    return fee


def apply_fees(trades_df: pd.DataFrame,
              fee_schedule: FeeSchedule,
              order_type_col: str = 'order_type',
              notional_col: str = 'notional') -> pd.Series:
    """
    Apply fees to a dataframe of trades

    Args:
        trades_df: Dataframe with trade information
        fee_schedule: Fee schedule to apply
        order_type_col: Column name for order type
        notional_col: Column name for notional value

    Returns:
        Series with fee amounts for each trade

    Example:
        trades = pd.DataFrame({
            'notional': [1000, 2000, 500],
            'order_type': ['market', 'limit', 'market']
        })
        fees = apply_fees(trades, EXCHANGE_FEES['binance'])
    """
    fees = []

    for _, trade in trades_df.iterrows():
        notional = trade[notional_col]
        order_type_str = trade.get(order_type_col, 'market')

        # Convert string to enum
        order_type = OrderType(order_type_str)

        fee = calculate_fee(notional, fee_schedule, order_type)
        fees.append(fee)

    return pd.Series(fees, index=trades_df.index)


class TradingCostModel:
    """
    Comprehensive trading cost model

    This class calculates total trading costs including:
    - Exchange fees (maker/taker)
    - Bid-ask spread costs
    - Market impact (slippage)
    - Funding costs (for leveraged positions)
    """

    def __init__(self,
                 fee_schedule: FeeSchedule,
                 spread_bps: float = 5.0,
                 impact_coefficient: float = 0.1,
                 funding_rate: float = 0.0):
        """
        Initialize trading cost model

        Args:
            fee_schedule: Exchange fee schedule
            spread_bps: Bid-ask spread in basis points
            impact_coefficient: Market impact coefficient
            funding_rate: Funding rate for leveraged positions (annual)
        """
        self.fee_schedule = fee_schedule
        self.spread_bps = spread_bps
        self.impact_coefficient = impact_coefficient
        self.funding_rate = funding_rate

    def calculate_total_cost(self,
                           notional: float,
                           order_type: OrderType = OrderType.MARKET,
                           position_size_relative: float = 0.01,
                           holding_hours: float = 0.0) -> Dict[str, float]:
        """
        Calculate total trading costs breakdown

        Args:
            notional: Notional value of trade
            order_type: Type of order
            position_size_relative: Position size relative to average daily volume
            holding_hours: Hours position is held (for funding costs)

        Returns:
            Dictionary with cost breakdown

        Example:
            costs = model.calculate_total_cost(
                notional=10000,
                order_type=OrderType.MARKET,
                position_size_relative=0.005,  # 0.5% of daily volume
                holding_hours=24
            )
        """
        # Exchange fees
        exchange_fee = calculate_fee(notional, self.fee_schedule, order_type)

        # Bid-ask spread cost (half spread for market orders)
        spread_cost = notional * self.spread_bps / 10000
        if order_type in [OrderType.LIMIT, OrderType.LIMIT_FOK]:
            spread_cost *= 0.5  # Limit orders pay less spread

        # Market impact (slippage)
        impact_cost = notional * position_size_relative * self.impact_coefficient

        # Funding costs (for leveraged positions)
        funding_cost = 0.0
        if holding_hours > 0 and self.funding_rate != 0:
            funding_cost = notional * self.funding_rate * (holding_hours / (365 * 24))

        total_cost = exchange_fee + spread_cost + impact_cost + funding_cost

        return {
            'exchange_fee': exchange_fee,
            'spread_cost': spread_cost,
            'impact_cost': impact_cost,
            'funding_cost': funding_cost,
            'total_cost': total_cost,
            'cost_bps': (total_cost / notional) * 10000 if notional > 0 else 0
        }

    def break_even_return(self,
                         notional: float,
                         order_type: OrderType = OrderType.MARKET,
                         position_size_relative: float = 0.01) -> float:
        """
        Calculate minimum return needed to break even after costs

        Args:
            notional: Notional value of trade
            order_type: Type of order
            position_size_relative: Position size relative to daily volume

        Returns:
            Break-even return in decimal (e.g., 0.001 = 0.1%)

        This is critical for setting decision thresholds in strategies
        """
        entry_costs = self.calculate_total_cost(
            notional, order_type, position_size_relative
        )
        exit_costs = self.calculate_total_cost(
            notional, order_type, position_size_relative
        )

        total_costs = entry_costs['total_cost'] + exit_costs['total_cost']
        break_even = total_costs / notional if notional > 0 else 0

        return break_even


def calculate_turnover_cost(positions: pd.Series,
                          cost_per_turnover: float) -> pd.Series:
    """
    Calculate cost based on portfolio turnover

    Args:
        positions: Series of position weights over time
        cost_per_turnover: Cost per unit of turnover (e.g., 0.001 = 0.1%)

    Returns:
        Series of turnover costs

    Example:
        # Position weights changing from 0.5 to 0.3 = 0.2 turnover
        positions = pd.Series([0.5, 0.3, 0.4])
        costs = calculate_turnover_cost(positions, 0.001)
    """
    position_changes = positions.diff().abs()
    turnover_costs = position_changes * cost_per_turnover
    return turnover_costs.fillna(0)


def analyze_cost_impact(returns: pd.Series,
                       costs: pd.Series) -> Dict[str, float]:
    """
    Analyze the impact of costs on strategy performance

    Args:
        returns: Gross returns series
        costs: Trading costs series

    Returns:
        Dictionary with performance metrics

    Example:
        impact = analyze_cost_impact(gross_returns, trading_costs)
        print(f"Cost drag: {impact['cost_drag_annual']:.2%}")
    """
    if len(returns) != len(costs):
        raise ValueError("Returns and costs series must have same length")

    gross_cumret = (1 + returns).cumprod().iloc[-1] - 1
    net_returns = returns - costs
    net_cumret = (1 + net_returns).cumprod().iloc[-1] - 1

    total_costs = costs.sum()
    cost_drag = gross_cumret - net_cumret

    # Annualize metrics (assuming daily data)
    periods_per_year = 252
    if len(returns) > 0:
        years = len(returns) / periods_per_year
        cost_drag_annual = cost_drag / years if years > 0 else cost_drag
    else:
        cost_drag_annual = 0

    return {
        'gross_return': gross_cumret,
        'net_return': net_cumret,
        'total_costs': total_costs,
        'cost_drag': cost_drag,
        'cost_drag_annual': cost_drag_annual,
        'cost_ratio': total_costs / abs(gross_cumret) if gross_cumret != 0 else np.inf
    }


def optimize_order_type(expected_return: float,
                       volatility: float,
                       cost_model: TradingCostModel,
                       notional: float,
                       time_horizon_hours: float = 1.0) -> Dict[str, Union[OrderType, float]]:
    """
    Optimize order type based on expected return and costs

    Args:
        expected_return: Expected return (decimal)
        volatility: Price volatility (decimal)
        cost_model: Trading cost model
        notional: Notional value of trade
        time_horizon_hours: Expected time to fill limit order

    Returns:
        Dictionary with optimal order type and cost analysis

    This helps decide between market orders (immediate fill, higher cost)
    vs limit orders (potential better price, execution risk)
    """
    # Market order costs (immediate execution)
    market_costs = cost_model.calculate_total_cost(
        notional, OrderType.MARKET
    )

    # Limit order costs (assuming fill)
    limit_costs = cost_model.calculate_total_cost(
        notional, OrderType.LIMIT
    )

    # Execution risk for limit orders
    # Simplified model: risk = volatility * time
    execution_risk = volatility * np.sqrt(time_horizon_hours / 24)

    # Net expected return after costs
    market_net_return = expected_return - (market_costs['cost_bps'] / 10000)
    limit_net_return = expected_return - (limit_costs['cost_bps'] / 10000)

    # Adjust limit order return for execution risk
    limit_net_return_adjusted = limit_net_return * (1 - execution_risk)

    optimal_type = OrderType.MARKET
    if limit_net_return_adjusted > market_net_return:
        optimal_type = OrderType.LIMIT

    return {
        'optimal_order_type': optimal_type,
        'market_net_return': market_net_return,
        'limit_net_return': limit_net_return,
        'limit_net_return_adjusted': limit_net_return_adjusted,
        'execution_risk': execution_risk,
        'cost_savings': market_costs['cost_bps'] - limit_costs['cost_bps']
    }


# Testing and validation functions
def test_fee_calculation():
    """Test fee calculation functionality"""
    print("\n🧪 Testing Fee Calculation")

    # Test basic fee calculation
    binance_fees = EXCHANGE_FEES['binance']

    # Market order (taker)
    market_fee = calculate_fee(1000, binance_fees, OrderType.MARKET)
    expected_market = 1000 * 10 / 10000  # 0.1%
    assert abs(market_fee - expected_market) < 0.001, f"Market fee: {market_fee} != {expected_market}"

    # Limit order (maker)
    limit_fee = calculate_fee(1000, binance_fees, OrderType.LIMIT)
    expected_limit = 1000 * 10 / 10000  # 0.1%
    assert abs(limit_fee - expected_limit) < 0.001, f"Limit fee: {limit_fee} != {expected_limit}"

    print("✅ Fee calculation test passed")


def test_cost_model():
    """Test comprehensive cost model"""
    print("\n🧪 Testing Cost Model")

    cost_model = TradingCostModel(
        fee_schedule=EXCHANGE_FEES['binance'],
        spread_bps=5.0,
        impact_coefficient=0.1
    )

    costs = cost_model.calculate_total_cost(
        notional=10000,
        order_type=OrderType.MARKET,
        position_size_relative=0.005
    )

    # Verify all cost components are present
    required_keys = ['exchange_fee', 'spread_cost', 'impact_cost', 'total_cost', 'cost_bps']
    for key in required_keys:
        assert key in costs, f"Missing cost component: {key}"

    # Verify costs are reasonable
    assert costs['total_cost'] > 0, "Total cost should be positive"
    assert costs['exchange_fee'] > 0, "Exchange fee should be positive"
    assert costs['cost_bps'] > 0, "Cost in bps should be positive"

    print(f"  Total cost for $10,000 trade: ${costs['total_cost']:.2f}")
    print(f"  Cost in basis points: {costs['cost_bps']:.1f} bps")
    print("✅ Cost model test passed")


def test_break_even_analysis():
    """Test break-even return calculation"""
    print("\n🧪 Testing Break-Even Analysis")

    cost_model = TradingCostModel(
        fee_schedule=EXCHANGE_FEES['binance'],
        spread_bps=5.0,
        impact_coefficient=0.1
    )

    break_even = cost_model.break_even_return(
        notional=10000,
        order_type=OrderType.MARKET,
        position_size_relative=0.005
    )

    assert break_even > 0, "Break-even return should be positive"
    assert break_even < 0.1, "Break-even return should be reasonable (<10%)"

    print(f"  Break-even return: {break_even:.4f} ({break_even*100:.2f}%)")
    print("✅ Break-even analysis test passed")


if __name__ == "__main__":
    print("🧪 Running Transaction Cost Tests")
    test_fee_calculation()
    test_cost_model()
    test_break_even_analysis()
    print("✅ All transaction cost tests passed!")