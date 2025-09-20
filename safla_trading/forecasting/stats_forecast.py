"""
STATISTICAL FORECASTING
High-performance statistical models using StatsForecast
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging

try:
    from statsforecast import StatsForecast
    from statsforecast.models import (
        AutoARIMA, AutoETS, AutoTheta, AutoCES,
        SeasonalNaive, Naive, RandomWalkWithDrift,
        CrostonClassic, ADIDA, IMAPA
    )
    STATSFORECAST_AVAILABLE = True
except ImportError:
    STATSFORECAST_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

from ..config.config_loader import get_config
from ..logging_system import TradeLogger


class StatisticalForecaster:
    """High-performance statistical forecasting using StatsForecast"""

    def __init__(self, logger: Optional[TradeLogger] = None):
        """Initialize statistical forecaster

        Args:
            logger: Trade logger instance
        """
        if not STATSFORECAST_AVAILABLE:
            raise ImportError("statsforecast is required. Install with: pip install statsforecast")

        self.config = get_config()
        self.logger = logger

        # Available models
        self.available_models = {
            'auto_arima': AutoARIMA,
            'auto_ets': AutoETS,
            'auto_theta': AutoTheta,
            'auto_ces': AutoCES,
            'seasonal_naive': SeasonalNaive,
            'naive': Naive,
            'random_walk': RandomWalkWithDrift,
            'croston': CrostonClassic,
            'adida': ADIDA,
            'imapa': IMAPA
        }

        # Default model configuration
        self.default_models = [
            AutoARIMA(season_length=24),  # Hourly seasonality
            AutoETS(season_length=24),
            AutoTheta(season_length=24),
            SeasonalNaive(season_length=24),
            Naive()
        ]

        # Fitted models storage
        self.fitted_models: Dict[str, StatsForecast] = {}

        # Performance tracking
        self.model_performance: Dict[str, Dict[str, float]] = {}

    def prepare_data(self, data: pd.DataFrame, symbol: str,
                    price_column: str = 'close') -> pd.DataFrame:
        """Prepare data for StatsForecast

        Args:
            data: Raw price data
            symbol: Trading symbol
            price_column: Column containing prices

        Returns:
            Prepared data in StatsForecast format
        """
        # StatsForecast requires specific column names
        prepared_data = data.copy()

        # Ensure datetime index
        if not isinstance(prepared_data.index, pd.DatetimeIndex):
            if 'timestamp' in prepared_data.columns:
                prepared_data['timestamp'] = pd.to_datetime(prepared_data['timestamp'])
                prepared_data = prepared_data.set_index('timestamp')
            else:
                prepared_data.index = pd.to_datetime(prepared_data.index)

        # Create StatsForecast format
        forecast_data = pd.DataFrame({
            'unique_id': symbol,
            'ds': prepared_data.index,
            'y': prepared_data[price_column]
        })

        # Remove any NaN values
        forecast_data = forecast_data.dropna()

        # Ensure proper sorting
        forecast_data = forecast_data.sort_values('ds')

        return forecast_data

    def fit_models(self, data: pd.DataFrame, symbol: str,
                  models: Optional[List[str]] = None,
                  season_length: int = 24) -> Dict[str, Any]:
        """Fit statistical models to data

        Args:
            data: Prepared data
            symbol: Trading symbol
            models: List of model names (None for default)
            season_length: Seasonal period

        Returns:
            Fitting results
        """
        if models is None:
            model_instances = self.default_models
        else:
            model_instances = []
            for model_name in models:
                if model_name in self.available_models:
                    model_class = self.available_models[model_name]
                    if model_name in ['auto_arima', 'auto_ets', 'auto_theta', 'auto_ces', 'seasonal_naive']:
                        model_instances.append(model_class(season_length=season_length))
                    else:
                        model_instances.append(model_class())

        try:
            # Create StatsForecast instance
            sf = StatsForecast(
                models=model_instances,
                freq='H',  # Hourly frequency by default
                n_jobs=-1  # Use all available cores
            )

            # Fit models
            sf.fit(data)

            # Store fitted models
            self.fitted_models[symbol] = sf

            if self.logger:
                self.logger.log_system_event(
                    'stats_forecast', 'models_fitted',
                    {
                        'symbol': symbol,
                        'models_count': len(model_instances),
                        'data_points': len(data)
                    }
                )

            return {
                'success': True,
                'models_fitted': len(model_instances),
                'data_points': len(data)
            }

        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    'stats_forecast', 'fitting_failed',
                    f"Failed to fit models for {symbol}: {e}",
                    exception=e
                )
            return {'success': False, 'error': str(e)}

    def forecast(self, symbol: str, horizon: int = 24,
                confidence_intervals: List[int] = [80, 95]) -> Dict[str, Any]:
        """Generate forecasts

        Args:
            symbol: Trading symbol
            horizon: Forecast horizon
            confidence_intervals: Confidence interval levels

        Returns:
            Forecast results
        """
        if symbol not in self.fitted_models:
            return {'error': f'No fitted models for {symbol}'}

        try:
            sf = self.fitted_models[symbol]

            # Generate forecasts
            forecasts = sf.predict(h=horizon, level=confidence_intervals)

            # Convert to more usable format
            forecast_dict = {
                'symbol': symbol,
                'horizon': horizon,
                'timestamp': datetime.now(),
                'forecasts': {}
            }

            # Extract forecasts for each model
            for col in forecasts.columns:
                if col not in ['unique_id', 'ds']:
                    model_name = col.split('/')[0] if '/' in col else col
                    if model_name not in forecast_dict['forecasts']:
                        forecast_dict['forecasts'][model_name] = {}

                    if col.endswith(('-lo', '-hi')):
                        # Confidence intervals
                        level = col.split('-')[-2]
                        bound = 'lower' if col.endswith('-lo') else 'upper'
                        if 'confidence_intervals' not in forecast_dict['forecasts'][model_name]:
                            forecast_dict['forecasts'][model_name]['confidence_intervals'] = {}
                        if level not in forecast_dict['forecasts'][model_name]['confidence_intervals']:
                            forecast_dict['forecasts'][model_name]['confidence_intervals'][level] = {}
                        forecast_dict['forecasts'][model_name]['confidence_intervals'][level][bound] = forecasts[col].tolist()
                    else:
                        # Point forecasts
                        forecast_dict['forecasts'][model_name]['values'] = forecasts[col].tolist()

            # Add timestamps
            forecast_dict['forecast_dates'] = forecasts['ds'].tolist()

            if self.logger:
                self.logger.log_system_event(
                    'stats_forecast', 'forecast_generated',
                    {
                        'symbol': symbol,
                        'horizon': horizon,
                        'models_count': len(forecast_dict['forecasts'])
                    }
                )

            return forecast_dict

        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    'stats_forecast', 'forecast_failed',
                    f"Failed to generate forecast for {symbol}: {e}",
                    exception=e
                )
            return {'error': str(e)}

    def cross_validate(self, data: pd.DataFrame, symbol: str,
                      n_windows: int = 3, h: int = 24) -> Dict[str, Any]:
        """Perform cross-validation

        Args:
            data: Prepared data
            symbol: Trading symbol
            n_windows: Number of validation windows
            h: Forecast horizon

        Returns:
            Cross-validation results
        """
        if symbol not in self.fitted_models:
            return {'error': f'No fitted models for {symbol}'}

        try:
            sf = self.fitted_models[symbol]

            # Perform cross-validation
            cv_results = sf.cross_validation(
                df=data,
                h=h,
                step_size=h,
                n_windows=n_windows
            )

            # Calculate accuracy metrics
            from statsforecast.utils import evaluate

            accuracy_metrics = evaluate(
                cv_results,
                metrics=['mae', 'mse', 'rmse', 'mape', 'smape']
            )

            # Store performance metrics
            self.model_performance[symbol] = {}
            for model in accuracy_metrics['unique_id'].unique():
                model_metrics = accuracy_metrics[accuracy_metrics['unique_id'] == model]
                self.model_performance[symbol][model] = {
                    'mae': float(model_metrics['mae'].iloc[0]),
                    'mse': float(model_metrics['mse'].iloc[0]),
                    'rmse': float(model_metrics['rmse'].iloc[0]),
                    'mape': float(model_metrics['mape'].iloc[0]),
                    'smape': float(model_metrics['smape'].iloc[0])
                }

            if self.logger:
                self.logger.log_system_event(
                    'stats_forecast', 'cross_validation_completed',
                    {
                        'symbol': symbol,
                        'n_windows': n_windows,
                        'horizon': h,
                        'models_evaluated': len(accuracy_metrics['unique_id'].unique())
                    }
                )

            return {
                'cross_validation_results': cv_results,
                'accuracy_metrics': accuracy_metrics,
                'performance_summary': self.model_performance[symbol]
            }

        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    'stats_forecast', 'cross_validation_failed',
                    f"Cross-validation failed for {symbol}: {e}",
                    exception=e
                )
            return {'error': str(e)}

    def get_best_model(self, symbol: str, metric: str = 'rmse') -> Optional[str]:
        """Get best performing model for symbol

        Args:
            symbol: Trading symbol
            metric: Performance metric

        Returns:
            Best model name or None
        """
        if symbol not in self.model_performance:
            return None

        performance = self.model_performance[symbol]
        if not performance:
            return None

        # Find model with lowest error metric
        best_model = min(performance.keys(),
                        key=lambda x: performance[x].get(metric, float('inf')))

        return best_model

    def ensemble_forecast(self, symbol: str, horizon: int = 24,
                         weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Create ensemble forecast

        Args:
            symbol: Trading symbol
            horizon: Forecast horizon
            weights: Model weights (None for equal weights)

        Returns:
            Ensemble forecast
        """
        forecast_results = self.forecast(symbol, horizon)

        if 'error' in forecast_results:
            return forecast_results

        forecasts = forecast_results['forecasts']
        model_names = list(forecasts.keys())

        if not model_names:
            return {'error': 'No forecast results available'}

        # Use equal weights if not provided
        if weights is None:
            weights = {model: 1.0 / len(model_names) for model in model_names}

        # Normalize weights
        total_weight = sum(weights.values())
        weights = {model: weight / total_weight for model, weight in weights.items()}

        # Calculate weighted ensemble
        ensemble_values = None
        for model_name, weight in weights.items():
            if model_name in forecasts and 'values' in forecasts[model_name]:
                model_forecast = np.array(forecasts[model_name]['values'])
                if ensemble_values is None:
                    ensemble_values = weight * model_forecast
                else:
                    ensemble_values += weight * model_forecast

        if ensemble_values is None:
            return {'error': 'Could not create ensemble forecast'}

        return {
            'symbol': symbol,
            'horizon': horizon,
            'ensemble_forecast': ensemble_values.tolist(),
            'forecast_dates': forecast_results['forecast_dates'],
            'weights': weights,
            'individual_forecasts': forecasts
        }

    def get_forecast_signals(self, symbol: str, current_price: float,
                           horizon: int = 24, threshold: float = 0.02) -> Dict[str, Any]:
        """Generate trading signals from forecasts

        Args:
            symbol: Trading symbol
            current_price: Current market price
            horizon: Forecast horizon
            threshold: Signal threshold (percentage)

        Returns:
            Trading signals
        """
        ensemble_result = self.ensemble_forecast(symbol, horizon)

        if 'error' in ensemble_result:
            return ensemble_result

        ensemble_forecast = ensemble_result['ensemble_forecast']

        if not ensemble_forecast:
            return {'error': 'Empty ensemble forecast'}

        # Calculate expected return
        future_price = ensemble_forecast[min(len(ensemble_forecast) - 1, horizon - 1)]
        expected_return = (future_price - current_price) / current_price

        # Generate signals
        signal = 'hold'
        confidence = abs(expected_return) / threshold

        if expected_return > threshold:
            signal = 'buy'
        elif expected_return < -threshold:
            signal = 'sell'

        return {
            'symbol': symbol,
            'signal': signal,
            'expected_return': expected_return,
            'confidence': min(confidence, 1.0),
            'current_price': current_price,
            'forecasted_price': future_price,
            'threshold': threshold,
            'forecast_horizon': horizon
        }

    def get_model_performance(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get model performance metrics

        Args:
            symbol: Specific symbol (None for all)

        Returns:
            Performance metrics
        """
        if symbol:
            return self.model_performance.get(symbol, {})
        return self.model_performance

    def clear_models(self, symbol: Optional[str] = None):
        """Clear fitted models

        Args:
            symbol: Specific symbol (None for all)
        """
        if symbol:
            self.fitted_models.pop(symbol, None)
            self.model_performance.pop(symbol, None)
        else:
            self.fitted_models.clear()
            self.model_performance.clear()

        if self.logger:
            self.logger.log_system_event(
                'stats_forecast', 'models_cleared',
                {'symbol': symbol or 'all'}
            )


class ProphetForecaster:
    """Facebook Prophet forecasting integration"""

    def __init__(self, logger: Optional[TradeLogger] = None):
        """Initialize Prophet forecaster

        Args:
            logger: Trade logger instance
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("prophet is required. Install with: pip install prophet")

        self.logger = logger
        self.fitted_models: Dict[str, Prophet] = {}

    def prepare_data(self, data: pd.DataFrame, price_column: str = 'close') -> pd.DataFrame:
        """Prepare data for Prophet

        Args:
            data: Raw price data
            price_column: Column containing prices

        Returns:
            Prepared data in Prophet format
        """
        prepared_data = data.copy()

        # Ensure datetime index
        if not isinstance(prepared_data.index, pd.DatetimeIndex):
            if 'timestamp' in prepared_data.columns:
                prepared_data['timestamp'] = pd.to_datetime(prepared_data['timestamp'])
                prepared_data = prepared_data.set_index('timestamp')
            else:
                prepared_data.index = pd.to_datetime(prepared_data.index)

        # Create Prophet format
        prophet_data = pd.DataFrame({
            'ds': prepared_data.index,
            'y': prepared_data[price_column]
        })

        return prophet_data.dropna()

    def fit_model(self, data: pd.DataFrame, symbol: str,
                 **prophet_kwargs) -> Dict[str, Any]:
        """Fit Prophet model

        Args:
            data: Prepared data
            symbol: Trading symbol
            **prophet_kwargs: Prophet model parameters

        Returns:
            Fitting results
        """
        try:
            # Create Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                **prophet_kwargs
            )

            # Fit model
            model.fit(data)

            # Store fitted model
            self.fitted_models[symbol] = model

            if self.logger:
                self.logger.log_system_event(
                    'prophet_forecast', 'model_fitted',
                    {
                        'symbol': symbol,
                        'data_points': len(data)
                    }
                )

            return {
                'success': True,
                'data_points': len(data)
            }

        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    'prophet_forecast', 'fitting_failed',
                    f"Failed to fit Prophet model for {symbol}: {e}",
                    exception=e
                )
            return {'success': False, 'error': str(e)}

    def forecast(self, symbol: str, horizon: int = 24) -> Dict[str, Any]:
        """Generate Prophet forecast

        Args:
            symbol: Trading symbol
            horizon: Forecast horizon

        Returns:
            Forecast results
        """
        if symbol not in self.fitted_models:
            return {'error': f'No fitted model for {symbol}'}

        try:
            model = self.fitted_models[symbol]

            # Create future dataframe
            future = model.make_future_dataframe(periods=horizon, freq='H')

            # Generate forecast
            forecast = model.predict(future)

            # Extract relevant columns
            result = {
                'symbol': symbol,
                'horizon': horizon,
                'forecast_dates': forecast['ds'].tail(horizon).tolist(),
                'forecasted_values': forecast['yhat'].tail(horizon).tolist(),
                'lower_bound': forecast['yhat_lower'].tail(horizon).tolist(),
                'upper_bound': forecast['yhat_upper'].tail(horizon).tolist(),
                'trend': forecast['trend'].tail(horizon).tolist()
            }

            return result

        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    'prophet_forecast', 'forecast_failed',
                    f"Failed to generate Prophet forecast for {symbol}: {e}",
                    exception=e
                )
            return {'error': str(e)}