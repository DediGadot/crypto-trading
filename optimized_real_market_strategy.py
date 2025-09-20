#!/usr/bin/env python3
"""
OPTIMIZED STRATEGY FOR REAL MARKET CONDITIONS
Adapting the algorithm to work with actual market microstructure

"First make it work, then make it right, then make it fast." - Kent Beck
Our system correctly identified the strategy failure. Now we fix it.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging

from crypto_strategy_implementation import CryptoFeatureEngine
from linus_trading_system import TradingSignal


class RealMarketStatisticalArbitrage:
    """
    Statistical arbitrage strategy optimized for real market conditions
    Based on analysis of actual BTC price behavior
    """

    def __init__(self,
                 lookback_hours: int = 72,  # Shorter lookback for real markets
                 entry_threshold: float = 1.5,  # Lower threshold for real data
                 exit_threshold: float = 0.3,   # Quicker exits
                 min_volume_ratio: float = 0.8):  # Volume filter

        self.lookback_hours = lookback_hours
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.min_volume_ratio = min_volume_ratio

        # Strategy state
        self.position = 0.0
        self.entry_price = None
        self.entry_time = None
        self.feature_engine = CryptoFeatureEngine()

        # Real market adaptations
        self.volatility_lookback = 24  # 1 day volatility
        self.trend_lookback = 12       # 12 hour trend
        self.momentum_lookback = 6     # 6 hour momentum

    def calculate_real_market_features(self, prices: pd.Series, volumes: pd.Series) -> pd.DataFrame:
        """
        Calculate features optimized for real market behavior
        """
        features = pd.DataFrame(index=prices.index)
        returns = prices.pct_change()

        # Short-term mean reversion signals (works better in crypto)
        short_ma = prices.rolling(self.momentum_lookback).mean()
        features['short_mean_revert'] = (prices - short_ma) / short_ma

        # Volatility regime detection
        vol_short = returns.rolling(self.volatility_lookback).std()
        vol_long = returns.rolling(self.volatility_lookback * 3).std()
        features['vol_regime'] = vol_short / vol_long

        # Momentum confirmation
        momentum_1h = prices / prices.shift(1) - 1
        momentum_4h = prices / prices.shift(4) - 1
        features['momentum_divergence'] = momentum_1h - momentum_4h

        # Volume confirmation
        vol_ma = volumes.rolling(self.volatility_lookback).mean()
        features['volume_ratio'] = volumes / vol_ma

        # Price velocity (acceleration/deceleration)
        price_change = prices.diff()
        features['price_acceleration'] = price_change.diff()

        # Bollinger Band position (normalized)
        bb_period = 20
        bb_ma = prices.rolling(bb_period).mean()
        bb_std = prices.rolling(bb_period).std()
        features['bb_position'] = (prices - bb_ma) / (2 * bb_std)

        return features.fillna(0)

    def generate_signals(self,
                        prices: pd.Series,
                        volumes: pd.Series,
                        funding_rates: pd.Series = None) -> List[TradingSignal]:
        """
        Generate signals optimized for real market conditions
        """
        if len(prices) < max(self.lookback_hours, 50):  # Need minimum data
            return []

        # Calculate real market features
        features = self.calculate_real_market_features(prices, volumes)

        # Enhanced mean reversion with multiple confirmations
        returns = prices.pct_change()

        signals = []

        for i in range(max(self.lookback_hours, 50), len(prices)):
            timestamp = prices.index[i]
            current_price = prices.iloc[i]

            # Current feature values
            mean_revert = features['short_mean_revert'].iloc[i]
            vol_regime = features['vol_regime'].iloc[i]
            volume_ratio = features['volume_ratio'].iloc[i]
            bb_position = features['bb_position'].iloc[i]
            price_accel = features['price_acceleration'].iloc[i]

            # Calculate z-score for mean reversion
            recent_returns = returns.iloc[i-self.lookback_hours:i]
            if len(recent_returns) < self.lookback_hours:
                continue

            z_score = (recent_returns.iloc[-1] - recent_returns.mean()) / recent_returns.std()

            # Signal generation with multiple filters
            signal_action = 'hold'
            confidence = 0.0
            size = 0.0
            reason = 'no_signal'

            # Entry conditions - much more restrictive for real markets
            if self.position == 0.0:  # No current position

                # Long entry: Multiple confirmations required
                if (mean_revert < -0.02 and  # Significant deviation
                    z_score < -self.entry_threshold and  # Statistical significance
                    vol_regime < 1.2 and  # Not in high vol regime
                    volume_ratio > self.min_volume_ratio and  # Volume confirmation
                    bb_position < -0.8 and  # Oversold
                    abs(price_accel) < current_price * 0.001):  # Not accelerating away

                    signal_action = 'buy'
                    size = min(5000, 10000 / current_price)  # Smaller positions
                    confidence = min(0.8, abs(z_score) / (self.entry_threshold * 2))
                    reason = f'mean_revert_long_z={z_score:.2f}_bb={bb_position:.2f}'

                # Short entry: Very strict conditions
                elif (mean_revert > 0.02 and
                      z_score > self.entry_threshold and
                      vol_regime < 1.2 and
                      volume_ratio > self.min_volume_ratio and
                      bb_position > 0.8 and
                      abs(price_accel) < current_price * 0.001):

                    signal_action = 'sell'
                    size = min(5000, 10000 / current_price)
                    confidence = min(0.8, abs(z_score) / (self.entry_threshold * 2))
                    reason = f'mean_revert_short_z={z_score:.2f}_bb={bb_position:.2f}'

            # Exit conditions - quick exits for real markets
            elif self.position != 0.0:  # Have position

                # Calculate unrealized P&L
                if self.entry_price:
                    pnl_pct = (current_price - self.entry_price) / self.entry_price
                    if self.position < 0:  # Short position
                        pnl_pct = -pnl_pct

                    # Exit conditions
                    should_exit = (
                        abs(z_score) < self.exit_threshold or  # Mean reversion complete
                        vol_regime > 1.5 or  # High volatility regime
                        pnl_pct < -0.015 or  # 1.5% stop loss
                        pnl_pct > 0.025 or   # 2.5% take profit
                        volume_ratio < 0.5   # Volume drying up
                    )

                    if should_exit:
                        signal_action = 'sell' if self.position > 0 else 'buy'
                        size = abs(self.position)
                        confidence = 0.9
                        reason = f'exit_z={z_score:.2f}_pnl={pnl_pct:.3f}'

            # Create signal if action required
            if signal_action != 'hold':
                signal = TradingSignal(
                    timestamp=timestamp,
                    symbol='BTC/USDT',
                    action=signal_action,
                    size=size,
                    confidence=confidence,
                    reason=reason,
                    price=current_price
                )
                signals.append(signal)

                # Update strategy state
                if signal_action in ['buy', 'sell']:
                    if self.position == 0.0:  # Opening position
                        self.position = float(size) if signal_action == 'buy' else float(-size)
                        self.entry_price = current_price
                        self.entry_time = timestamp
                    else:  # Closing position
                        self.position = 0.0
                        self.entry_price = None
                        self.entry_time = None

        return signals


class MarketRegimeStrategy:
    """
    Regime-aware strategy that adapts to market conditions
    """

    def __init__(self):
        self.base_strategy = RealMarketStatisticalArbitrage()
        self.regime_lookback = 168  # 1 week for regime detection

    def detect_market_regime(self, prices: pd.Series, volumes: pd.Series) -> str:
        """
        Detect current market regime
        """
        if len(prices) < self.regime_lookback:
            return 'unknown'

        recent_prices = prices.iloc[-self.regime_lookback:]
        returns = recent_prices.pct_change().dropna()

        # Regime indicators
        volatility = returns.std() * np.sqrt(24)  # Annualized hourly vol
        trend_strength = (recent_prices.iloc[-1] / recent_prices.iloc[0]) - 1
        autocorr = returns.autocorr(1) if len(returns) > 1 else 0

        # Volume trend
        recent_volumes = volumes.iloc[-self.regime_lookback:]
        volume_trend = (recent_volumes.rolling(24).mean().iloc[-1] /
                       recent_volumes.rolling(24).mean().iloc[0]) - 1

        # Classify regime
        if volatility > 0.15:  # High volatility
            return 'high_vol'
        elif abs(trend_strength) > 0.20:  # Strong trend
            return 'trending'
        elif autocorr < -0.1:  # Mean reverting
            return 'mean_reverting'
        else:
            return 'choppy'

    def generate_regime_aware_signals(self,
                                    prices: pd.Series,
                                    volumes: pd.Series,
                                    funding_rates: pd.Series = None) -> List[TradingSignal]:
        """
        Generate signals adapted to current market regime
        """
        regime = self.detect_market_regime(prices, volumes)

        # Adjust strategy parameters based on regime
        if regime == 'high_vol':
            # More conservative in high volatility
            self.base_strategy.entry_threshold = 2.0
            self.base_strategy.exit_threshold = 0.2
            self.base_strategy.min_volume_ratio = 1.2

        elif regime == 'trending':
            # Don't fight strong trends
            self.base_strategy.entry_threshold = 2.5
            self.base_strategy.exit_threshold = 0.1

        elif regime == 'mean_reverting':
            # More aggressive in mean reverting markets
            self.base_strategy.entry_threshold = 1.2
            self.base_strategy.exit_threshold = 0.4
            self.base_strategy.min_volume_ratio = 0.6

        else:  # choppy
            # Conservative approach
            self.base_strategy.entry_threshold = 1.8
            self.base_strategy.exit_threshold = 0.3

        # Generate signals with adapted parameters
        signals = self.base_strategy.generate_signals(prices, volumes, funding_rates)

        # Add regime information to signal reasons
        for signal in signals:
            signal.reason += f'_regime={regime}'

        return signals


def create_optimized_strategy() -> MarketRegimeStrategy:
    """
    Factory function for optimized real market strategy
    """
    return MarketRegimeStrategy()


if __name__ == "__main__":
    print("🔧 REAL MARKET STRATEGY OPTIMIZATION")
    print("Adapting algorithm for actual market conditions")
    print("Key improvements:")
    print("  - Shorter lookback periods")
    print("  - Lower entry thresholds")
    print("  - Multiple confirmation filters")
    print("  - Regime-aware parameter adaptation")
    print("  - Quick exit conditions")
    print("  - Volume-based filters")
    print("Ready for production testing...")