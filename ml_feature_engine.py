#!/usr/bin/env python3
"""
ML-POWERED FEATURE ENGINEERING FOR ULTRA-ALPHA
Advanced machine learning features for crypto market prediction

"In God we trust. All others must bring data." - W. Edwards Deming
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.ensemble import IsolationForest
    HAS_ML_LIBS = True
except ImportError:
    print("⚠️ ML libraries not available - using simplified features")
    HAS_ML_LIBS = False


class MLFeatureEngine:
    """
    Advanced ML-driven feature engineering for crypto markets
    Extracts non-linear patterns and complex interactions
    """

    def __init__(self, feature_selection_top_k: int = 50):
        self.feature_selection_top_k = feature_selection_top_k
        self.scaler = RobustScaler() if HAS_ML_LIBS else None
        self.feature_importance_cache = {}
        self.model_cache = {}

    def engineer_ml_features(self,
                           prices: pd.Series,
                           volumes: pd.Series,
                           funding_rates: pd.Series = None) -> pd.DataFrame:
        """
        Engineer comprehensive ML features for crypto trading

        Returns:
            DataFrame with engineered features optimized for ML models
        """
        features = pd.DataFrame(index=prices.index)

        # Base technical indicators
        tech_features = self._calculate_technical_features(prices, volumes)
        features = pd.concat([features, tech_features], axis=1)

        # Statistical features
        stat_features = self._calculate_statistical_features(prices, volumes)
        features = pd.concat([features, stat_features], axis=1)

        # Fractal and complexity features
        fractal_features = self._calculate_fractal_features(prices)
        features = pd.concat([features, fractal_features], axis=1)

        # Microstructure features
        micro_features = self._calculate_microstructure_features(prices, volumes)
        features = pd.concat([features, micro_features], axis=1)

        # Behavioral features
        behavioral_features = self._calculate_behavioral_features(prices, volumes)
        features = pd.concat([features, behavioral_features], axis=1)

        # Cross-asset features (proxy using price patterns)
        cross_features = self._calculate_cross_asset_features(prices)
        features = pd.concat([features, cross_features], axis=1)

        # Funding rate features if available
        if funding_rates is not None:
            funding_features = self._calculate_funding_features(prices, funding_rates)
            features = pd.concat([features, funding_features], axis=1)

        # Feature interactions
        interaction_features = self._calculate_feature_interactions(features)
        features = pd.concat([features, interaction_features], axis=1)

        # Clean and normalize features
        features = self._clean_features(features)

        # Feature selection
        if HAS_ML_LIBS and len(features.columns) > self.feature_selection_top_k:
            features = self._select_best_features(features, prices)

        return features

    def _calculate_technical_features(self, prices: pd.Series, volumes: pd.Series) -> pd.DataFrame:
        """Advanced technical indicators"""
        features = pd.DataFrame(index=prices.index)

        # Multi-timeframe moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            ma = prices.rolling(window).mean()
            features[f'ma_{window}'] = ma
            features[f'ma_ratio_{window}'] = prices / ma
            features[f'ma_distance_{window}'] = (prices - ma) / ma

        # Exponential moving averages
        for span in [12, 26, 50, 200]:
            ema = prices.ewm(span=span).mean()
            features[f'ema_{span}'] = ema
            features[f'ema_ratio_{span}'] = prices / ema

        # MACD variations
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = macd - signal

        # Bollinger Bands
        for window in [20, 50]:
            ma = prices.rolling(window).mean()
            std = prices.rolling(window).std()
            features[f'bb_upper_{window}'] = ma + (2 * std)
            features[f'bb_lower_{window}'] = ma - (2 * std)
            features[f'bb_position_{window}'] = (prices - ma) / (2 * std)
            features[f'bb_width_{window}'] = (4 * std) / ma

        # RSI variations
        for periods in [14, 21, 28]:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(periods).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(periods).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            features[f'rsi_{periods}'] = rsi
            features[f'rsi_momentum_{periods}'] = rsi.diff()

        # Stochastic oscillator
        for k_periods in [14, 21]:
            low_min = prices.rolling(k_periods).min()
            high_max = prices.rolling(k_periods).max()
            k_percent = 100 * (prices - low_min) / (high_max - low_min)
            d_percent = k_percent.rolling(3).mean()
            features[f'stoch_k_{k_periods}'] = k_percent
            features[f'stoch_d_{k_periods}'] = d_percent

        # Volume indicators
        features['volume_sma_10'] = volumes.rolling(10).mean()
        features['volume_ratio'] = volumes / features['volume_sma_10']
        features['volume_rsi'] = self._calculate_rsi(volumes, 14)

        return features

    def _calculate_statistical_features(self, prices: pd.Series, volumes: pd.Series) -> pd.DataFrame:
        """Statistical and distributional features"""
        features = pd.DataFrame(index=prices.index)
        returns = prices.pct_change()

        # Rolling statistics
        for window in [12, 24, 72, 168]:
            features[f'return_mean_{window}'] = returns.rolling(window).mean()
            features[f'return_std_{window}'] = returns.rolling(window).std()
            features[f'return_skew_{window}'] = returns.rolling(window).skew()
            features[f'return_kurt_{window}'] = returns.rolling(window).kurt()

            # Higher moments
            features[f'return_var_{window}'] = returns.rolling(window).var()
            features[f'return_sem_{window}'] = returns.rolling(window).sem()

        # Autocorrelation features
        for lag in [1, 6, 12, 24]:
            features[f'autocorr_{lag}'] = returns.rolling(168).apply(
                lambda x: x.autocorr(lag) if len(x) > lag else np.nan
            )

        # Jarque-Bera normality test proxy
        for window in [72, 168]:
            skew = returns.rolling(window).skew()
            kurt = returns.rolling(window).kurt()
            features[f'jb_stat_{window}'] = (skew**2 + (kurt**2)/4) * window / 6

        # Hurst exponent (simplified)
        for window in [100, 200]:
            features[f'hurst_{window}'] = returns.rolling(window).apply(
                lambda x: self._calculate_hurst_exponent(x) if len(x) >= window else np.nan
            )

        return features

    def _calculate_fractal_features(self, prices: pd.Series) -> pd.DataFrame:
        """Fractal and complexity features"""
        features = pd.DataFrame(index=prices.index)

        # Highs and lows
        for window in [5, 10, 20]:
            features[f'high_{window}'] = prices.rolling(window).max()
            features[f'low_{window}'] = prices.rolling(window).min()
            features[f'range_{window}'] = features[f'high_{window}'] - features[f'low_{window}']

        # Support and resistance levels
        for window in [50, 100]:
            features[f'resistance_{window}'] = prices.rolling(window).quantile(0.95)
            features[f'support_{window}'] = prices.rolling(window).quantile(0.05)
            features[f'sr_position_{window}'] = (prices - features[f'support_{window}']) / (
                features[f'resistance_{window}'] - features[f'support_{window}']
            )

        # Pivot points
        features['pivot_point'] = (prices.shift(1) + prices.rolling(3).max().shift(1) +
                                 prices.rolling(3).min().shift(1)) / 3

        # Fractal dimension (simplified)
        for window in [50, 100]:
            features[f'fractal_dim_{window}'] = prices.rolling(window).apply(
                lambda x: self._calculate_fractal_dimension(x) if len(x) >= 10 else np.nan
            )

        return features

    def _calculate_microstructure_features(self, prices: pd.Series, volumes: pd.Series) -> pd.DataFrame:
        """Market microstructure features"""
        features = pd.DataFrame(index=prices.index)

        # Price impact
        returns = prices.pct_change()
        features['price_impact'] = returns / (volumes + 1e-8)

        # Volume-weighted average price (VWAP)
        for window in [20, 50]:
            vwap = (prices * volumes).rolling(window).sum() / volumes.rolling(window).sum()
            features[f'vwap_{window}'] = vwap
            features[f'vwap_distance_{window}'] = (prices - vwap) / vwap

        # Trade intensity
        features['trade_intensity'] = volumes / volumes.rolling(24).mean()

        # Bid-ask spread proxy (using price volatility)
        features['spread_proxy'] = returns.rolling(12).std()

        # Market depth proxy
        features['depth_proxy'] = volumes.rolling(12).std() / volumes.rolling(12).mean()

        # Tick rule (price direction)
        price_diff = prices.diff()
        features['tick_rule'] = np.sign(price_diff)
        features['tick_imbalance'] = features['tick_rule'].rolling(20).mean()

        return features

    def _calculate_behavioral_features(self, prices: pd.Series, volumes: pd.Series) -> pd.DataFrame:
        """Behavioral and sentiment features"""
        features = pd.DataFrame(index=prices.index)
        returns = prices.pct_change()

        # Fear and greed proxy
        for window in [24, 72]:
            vol = returns.rolling(window).std()
            features[f'fear_greed_{window}'] = vol / vol.rolling(window*4).mean()

        # Herding behavior (momentum clustering)
        momentum = returns.rolling(12).mean()
        features['herding'] = momentum / momentum.rolling(72).std()

        # Overreaction indicator
        large_moves = (returns.abs() > returns.rolling(168).quantile(0.95)).astype(float)
        features['overreaction'] = large_moves.rolling(24).mean()

        # Weekend effect proxy
        features['hour_of_day'] = prices.index.hour
        features['day_of_week'] = prices.index.dayofweek

        # News impact proxy (large volume spikes)
        vol_spike = volumes > volumes.rolling(24).quantile(0.99)
        features['news_impact'] = vol_spike.astype(float)

        return features

    def _calculate_cross_asset_features(self, prices: pd.Series) -> pd.DataFrame:
        """Cross-asset correlation features (simplified)"""
        features = pd.DataFrame(index=prices.index)
        returns = prices.pct_change()

        # Simulated cross-asset correlations
        # In production, these would be real BTC/ETH, BTC/stocks correlations
        synthetic_asset = prices * (1 + np.random.normal(0, 0.1, len(prices)))
        synthetic_returns = synthetic_asset.pct_change()

        for window in [24, 72, 168]:
            features[f'cross_corr_{window}'] = returns.rolling(window).corr(synthetic_returns)

        # Risk-on/risk-off proxy
        features['risk_appetite'] = returns.rolling(24).mean() / returns.rolling(24).std()

        return features

    def _calculate_funding_features(self, prices: pd.Series, funding_rates: pd.Series) -> pd.DataFrame:
        """Advanced funding rate features"""
        features = pd.DataFrame(index=prices.index)

        # Funding momentum
        features['funding_momentum'] = funding_rates.rolling(3).mean()
        features['funding_acceleration'] = funding_rates.diff().diff()

        # Funding extremes
        funding_zscore = (funding_rates - funding_rates.rolling(168).mean()) / funding_rates.rolling(168).std()
        features['funding_zscore'] = funding_zscore
        features['funding_extreme'] = (funding_zscore.abs() > 2).astype(float)

        # Carry trade signal
        features['carry_signal'] = -funding_rates  # Negative funding = bullish

        # Funding-price correlation
        returns = prices.pct_change()
        features['funding_price_corr'] = returns.rolling(72).corr(funding_rates)

        return features

    def _calculate_feature_interactions(self, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate feature interactions for non-linear patterns"""
        interaction_features = pd.DataFrame(index=features.index)

        # Select key features for interactions
        key_features = [col for col in features.columns if any(x in col.lower() for x in
                       ['rsi', 'ma_ratio', 'bb_position', 'volume_ratio', 'momentum'])][:10]

        # Polynomial interactions
        for i, feat1 in enumerate(key_features):
            if feat1 in features.columns:
                # Squared terms
                interaction_features[f'{feat1}_squared'] = features[feat1] ** 2

                # Cross-products with other features
                for feat2 in key_features[i+1:]:
                    if feat2 in features.columns:
                        interaction_features[f'{feat1}_x_{feat2}'] = features[feat1] * features[feat2]

        return interaction_features

    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize features"""
        # Replace infinite values
        features = features.replace([np.inf, -np.inf], np.nan)

        # Forward fill then backward fill
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # Remove features with too many NaN values
        nan_threshold = len(features) * 0.8  # Keep features with <80% NaN
        features = features.loc[:, features.isnull().sum() < nan_threshold]

        # Cap extreme outliers
        for col in features.columns:
            if features[col].dtype in ['float64', 'int64']:
                q99 = features[col].quantile(0.99)
                q01 = features[col].quantile(0.01)
                features[col] = features[col].clip(lower=q01, upper=q99)

        return features

    def _select_best_features(self, features: pd.DataFrame, target_prices: pd.Series) -> pd.DataFrame:
        """Select best features using mutual information"""
        if not HAS_ML_LIBS:
            return features.iloc[:, :self.feature_selection_top_k]

        try:
            # Calculate forward returns as target
            returns = target_prices.pct_change(1).shift(-1).fillna(0)

            # Align features and target
            aligned_features = features.loc[returns.index]
            aligned_target = returns.loc[aligned_features.index]

            # Remove features with zero variance
            feature_vars = aligned_features.var()
            non_zero_var_features = feature_vars[feature_vars > 1e-8].index
            aligned_features = aligned_features[non_zero_var_features]

            if len(aligned_features.columns) <= self.feature_selection_top_k:
                return aligned_features

            # Calculate mutual information
            mi_scores = mutual_info_regression(
                aligned_features.fillna(0),
                aligned_target,
                random_state=42
            )

            # Select top features
            feature_scores = pd.Series(mi_scores, index=aligned_features.columns)
            top_features = feature_scores.nlargest(self.feature_selection_top_k).index

            return aligned_features[top_features]

        except Exception as e:
            print(f"⚠️ Feature selection failed: {e}")
            return features.iloc[:, :self.feature_selection_top_k]

    def _calculate_rsi(self, series: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate RSI for any series"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(periods).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_hurst_exponent(self, series: pd.Series) -> float:
        """Calculate Hurst exponent (simplified)"""
        try:
            series = series.dropna()
            if len(series) < 10:
                return 0.5

            # Calculate cumulative sum
            cumsum = series.cumsum()

            # Calculate R/S statistic
            n = len(series)
            mean_adjusted = series - series.mean()
            cumsum_mean_adj = mean_adjusted.cumsum()

            R = cumsum_mean_adj.max() - cumsum_mean_adj.min()
            S = series.std()

            if S == 0:
                return 0.5

            rs = R / S
            hurst = np.log(rs) / np.log(n)

            return np.clip(hurst, 0, 1)
        except:
            return 0.5

    def _calculate_fractal_dimension(self, series: pd.Series) -> float:
        """Calculate fractal dimension (simplified box-counting)"""
        try:
            series = series.dropna()
            if len(series) < 10:
                return 1.5

            # Normalize series
            normalized = (series - series.min()) / (series.max() - series.min() + 1e-8)

            # Simple fractal dimension approximation
            diff = normalized.diff().abs().sum()
            length = len(normalized)

            # Fractal dimension approximation
            fractal_dim = 1 + (np.log(diff) / np.log(length))

            return np.clip(fractal_dim, 1, 2)
        except:
            return 1.5


class MLModelTrainer:
    """
    ML model trainer for feature importance and signal prediction
    """

    def __init__(self):
        self.models = {}
        self.feature_importance = {}

    def train_feature_importance_model(self,
                                     features: pd.DataFrame,
                                     target_returns: pd.Series) -> Dict[str, float]:
        """Train model to get feature importance"""
        if not HAS_ML_LIBS:
            # Fallback: random importance
            return {col: np.random.random() for col in features.columns}

        try:
            # Align data
            aligned_features = features.loc[target_returns.index].fillna(0)
            aligned_target = target_returns.loc[aligned_features.index]

            # Train XGBoost model
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )

            model.fit(aligned_features, aligned_target)

            # Get feature importance
            importance_dict = dict(zip(aligned_features.columns, model.feature_importances_))

            # Store model
            self.models['feature_importance'] = model
            self.feature_importance = importance_dict

            return importance_dict

        except Exception as e:
            print(f"⚠️ Model training failed: {e}")
            return {col: 1.0/len(features.columns) for col in features.columns}


def create_ml_feature_engine(feature_selection_top_k: int = 50) -> MLFeatureEngine:
    """
    Factory function for ML feature engine

    Args:
        feature_selection_top_k: Number of top features to select

    Returns:
        Configured MLFeatureEngine instance
    """
    return MLFeatureEngine(feature_selection_top_k=feature_selection_top_k)


if __name__ == "__main__":
    print("🧠 ML FEATURE ENGINE INITIALIZED")
    print("Advanced feature engineering for crypto markets")
    print("=" * 50)

    # Test with synthetic data
    from crypto_strategy_implementation import create_synthetic_crypto_data

    prices, volumes, funding_rates = create_synthetic_crypto_data(1000)
    engine = create_ml_feature_engine()

    print(f"📊 Test data: {len(prices)} periods")

    features = engine.engineer_ml_features(prices, volumes, funding_rates)
    print(f"🔧 Generated {len(features.columns)} ML features")
    print(f"📏 Feature matrix shape: {features.shape}")

    if len(features.columns) > 0:
        print("✅ ML feature engine ready for ultra-alpha generation")
        print(f"🏆 Top features: {list(features.columns[:5])}")
    else:
        print("⚠️ No features generated - check implementation")