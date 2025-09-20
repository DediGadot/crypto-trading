"""
PORTFOLIO OPTIMIZATION
Advanced portfolio construction using PyPortfolioOpt
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging

try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt.hierarchical_portfolio import HRPOpt
    from pypfopt.black_litterman import BlackLittermanModel
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False

try:
    import cvxpy
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

from ..config.config_loader import get_config
from ..logging_system import TradeLogger


logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Advanced portfolio optimization using modern portfolio theory"""

    def __init__(self, logger: Optional[TradeLogger] = None):
        """Initialize portfolio optimizer

        Args:
            logger: Trade logger instance
        """
        if not PYPFOPT_AVAILABLE:
            raise ImportError("pypfopt is required. Install with: pip install pypfopt")

        self.config = get_config()
        self.logger = logger

        # Risk model settings
        self.default_risk_model = 'ledoit_wolf'
        self.default_return_model = 'mean_historical_return'

        # Optimization constraints
        self.default_constraints = {
            'min_weight': 0.01,  # Minimum 1% allocation
            'max_weight': 0.30,  # Maximum 30% allocation
            'max_turnover': 0.20  # Maximum 20% turnover
        }

        # Portfolio storage
        self.portfolios: Dict[str, Dict[str, Any]] = {}

    def prepare_price_data(self, price_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """Prepare price data for optimization

        Args:
            price_data: Dictionary of symbol -> price series

        Returns:
            Prepared price DataFrame
        """
        # Convert to DataFrame
        df = pd.DataFrame(price_data)

        # Remove NaN values
        df = df.dropna()

        # Ensure we have enough data
        if len(df) < 30:
            raise ValueError("Insufficient price data for optimization (minimum 30 periods)")

        return df

    def calculate_expected_returns(self, prices: pd.DataFrame,
                                 method: str = 'mean_historical_return',
                                 **kwargs) -> pd.Series:
        """Calculate expected returns

        Args:
            prices: Price data
            method: Return calculation method
            **kwargs: Additional parameters

        Returns:
            Expected returns series
        """
        if method == 'mean_historical_return':
            return expected_returns.mean_historical_return(
                prices,
                frequency=kwargs.get('frequency', 252)
            )
        elif method == 'ema_historical_return':
            return expected_returns.ema_historical_return(
                prices,
                frequency=kwargs.get('frequency', 252),
                span=kwargs.get('span', 500)
            )
        elif method == 'capm_return':
            return expected_returns.capm_return(
                prices,
                market_prices=kwargs.get('market_prices'),
                frequency=kwargs.get('frequency', 252)
            )
        else:
            raise ValueError(f"Unknown return method: {method}")

    def calculate_risk_model(self, prices: pd.DataFrame,
                           method: str = 'ledoit_wolf',
                           **kwargs) -> pd.DataFrame:
        """Calculate risk model (covariance matrix)

        Args:
            prices: Price data
            method: Risk model method
            **kwargs: Additional parameters

        Returns:
            Covariance matrix
        """
        if method == 'sample_cov':
            return risk_models.sample_cov(
                prices,
                frequency=kwargs.get('frequency', 252)
            )
        elif method == 'ledoit_wolf':
            return risk_models.CovarianceShrinkage(
                prices,
                frequency=kwargs.get('frequency', 252)
            ).ledoit_wolf()
        elif method == 'oracle_approximating':
            return risk_models.CovarianceShrinkage(
                prices,
                frequency=kwargs.get('frequency', 252)
            ).oracle_approximating()
        elif method == 'exp_cov':
            return risk_models.exp_cov(
                prices,
                frequency=kwargs.get('frequency', 252),
                span=kwargs.get('span', 180)
            )
        else:
            raise ValueError(f"Unknown risk model: {method}")

    def optimize_mean_variance(
        self,
        prices: pd.DataFrame,
        objective: str = 'max_sharpe',
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Optimize portfolio using mean-variance optimization

        Args:
            prices: Price data
            objective: Optimization objective
            constraints: Portfolio constraints
            **kwargs: Additional parameters

        Returns:
            Optimization results
        """
        try:
            # Ernie Chan: Adaptive constraint relaxation for robustness
            constraint_sets = [
                {'min_weight': 0.0, 'max_weight': 1.0},     # Most relaxed
                {'min_weight': 0.01, 'max_weight': 0.60},   # Moderate
                {'min_weight': 0.05, 'max_weight': 0.40},   # Conservative
                {'min_weight': 0.10, 'max_weight': 0.30},   # Most strict
            ]

            user_constraints = constraints or {}
            last_error: Optional[Exception] = None

            for attempt_index, constraint_set in enumerate(constraint_sets, start=1):
                try:
                    logger.info(
                        "Optimizing with constraint set %d (min=%.2f, max=%.2f)",
                        attempt_index,
                        constraint_set['min_weight'],
                        constraint_set['max_weight'],
                    )

                    mu = self.calculate_expected_returns(
                        prices,
                        kwargs.get('return_method', self.default_return_model),
                        **kwargs,
                    )
                    S = self.calculate_risk_model(
                        prices,
                        kwargs.get('risk_method', self.default_risk_model),
                        **kwargs,
                    )

                    if mu.isna().any() or np.isclose(mu.values, 0.0).all():
                        logger.warning("Expected returns are degenerate; skipping constraint set %d", attempt_index)
                        continue

                    condition_number = float(np.linalg.cond(S))
                    if condition_number > 1e12:
                        logger.warning(
                            "Ill-conditioned covariance matrix detected (cond=%.2e); applying diagonal shrinkage",
                            condition_number,
                        )
                        S = S + np.eye(len(S)) * 1e-6

                    ef = EfficientFrontier(mu, S)

                    if attempt_index > 1:
                        ef.add_constraint(lambda w: w >= constraint_set['min_weight'])
                        ef.add_constraint(lambda w: w <= constraint_set['max_weight'])

                    if 'min_weight' in user_constraints:
                        ef.add_constraint(lambda w: w >= user_constraints['min_weight'])
                    if 'max_weight' in user_constraints:
                        ef.add_constraint(lambda w: w <= user_constraints['max_weight'])

                    if objective == 'max_sharpe':
                        ef.max_sharpe()
                    elif objective == 'min_volatility':
                        ef.min_volatility()
                    elif objective == 'efficient_return':
                        target_return = kwargs.get('target_return', 0.10)
                        ef.efficient_return(target_return)
                    elif objective == 'efficient_risk':
                        target_volatility = kwargs.get('target_volatility', 0.15)
                        ef.efficient_risk(target_volatility)
                    else:
                        raise ValueError(f"Unknown objective: {objective}")

                    performance = ef.portfolio_performance(verbose=False)
                    cleaned_weights = ef.clean_weights()

                    return {
                        'success': True,
                        'weights': cleaned_weights,
                        'expected_return': performance[0],
                        'volatility': performance[1],
                        'sharpe_ratio': performance[2],
                        'method': 'mean_variance',
                        'objective': objective,
                        'assets': list(prices.columns),
                        'constraint_set_used': attempt_index,
                        'optimization_date': datetime.now(),
                    }

                except Exception as constraint_error:  # noqa: BLE001
                    last_error = constraint_error
                    logger.warning(
                        "Constraint set %d failed: %s",
                        attempt_index,
                        str(constraint_error)[:200],
                        exc_info=logger.isEnabledFor(logging.DEBUG),
                    )

            if last_error:
                raise last_error
            raise RuntimeError("Mean-variance optimization failed: no viable constraint set")

        except Exception as exc:  # noqa: BLE001
            if self.logger:
                self.logger.log_error(
                    'portfolio_optimizer',
                    'mv_optimization_failed',
                    f"Mean-variance optimization failed: {exc}",
                    exception=exc,
                )
            logger.error("Mean-variance optimization failed: %s", exc)
            return {'success': False, 'error': str(exc)}

    def optimize_hierarchical_risk_parity(self, prices: pd.DataFrame,
                                        **kwargs) -> Dict[str, Any]:
        """Optimize portfolio using Hierarchical Risk Parity (HRP)

        Args:
            prices: Price data
            **kwargs: Additional parameters

        Returns:
            Optimization results
        """
        try:
            # Calculate returns
            returns = prices.pct_change().dropna()

            # Create HRP optimizer
            hrp = HRPOpt(returns)

            # Optimize
            weights = hrp.optimize()

            # Calculate portfolio performance with proper frequency
            frequency = kwargs.get('frequency', 252)  # Use passed frequency
            portfolio_return = np.sum(weights * returns.mean() * frequency)
            portfolio_vol = np.sqrt(
                np.dot(weights, np.dot(returns.cov() * frequency, weights))
            )
            sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
            print(f"   HRP Sharpe: {sharpe_ratio:.4f}, Return: {portfolio_return:.4f}")

            return {
                'success': True,
                'weights': weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'method': 'hierarchical_risk_parity',
                'assets': list(prices.columns),
                'optimization_date': datetime.now()
            }

        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    'portfolio_optimizer', 'hrp_optimization_failed',
                    f"HRP optimization failed: {e}",
                    exception=e
                )
            return {'success': False, 'error': str(e)}

    def optimize_black_litterman(self, prices: pd.DataFrame,
                               views: Dict[str, float],
                               confidences: Optional[Dict[str, float]] = None,
                               **kwargs) -> Dict[str, Any]:
        """Optimize portfolio using Black-Litterman model

        Args:
            prices: Price data
            views: Investor views on expected returns
            confidences: Confidence in views (0-1)
            **kwargs: Additional parameters

        Returns:
            Optimization results
        """
        try:
            # Calculate market-implied returns
            S = self.calculate_risk_model(prices, **kwargs)
            market_caps = kwargs.get('market_caps')

            if market_caps is None:
                # Use equal market caps if not provided
                market_caps = pd.Series(1.0, index=prices.columns)

            # Create Black-Litterman model
            bl = BlackLittermanModel(S, pi=market_caps)

            # Add views
            if confidences is None:
                confidences = {asset: 0.5 for asset in views.keys()}

            for asset, view in views.items():
                confidence = confidences.get(asset, 0.5)
                bl.add_view(asset, view, confidence)

            # Get Black-Litterman returns
            mu_bl = bl.bl_returns()

            # Optimize portfolio
            ef = EfficientFrontier(mu_bl, S)
            weights = ef.max_sharpe()
            performance = ef.portfolio_performance(verbose=False)
            cleaned_weights = ef.clean_weights()

            return {
                'success': True,
                'weights': cleaned_weights,
                'expected_return': performance[0],
                'volatility': performance[1],
                'sharpe_ratio': performance[2],
                'method': 'black_litterman',
                'views': views,
                'assets': list(prices.columns),
                'optimization_date': datetime.now()
            }

        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    'portfolio_optimizer', 'bl_optimization_failed',
                    f"Black-Litterman optimization failed: {e}",
                    exception=e
                )
            return {'success': False, 'error': str(e)}

    def discrete_allocation(self, weights: Dict[str, float],
                          latest_prices: Dict[str, float],
                          total_portfolio_value: float) -> Dict[str, Any]:
        """Calculate discrete allocation of assets

        Args:
            weights: Target portfolio weights
            latest_prices: Latest asset prices
            total_portfolio_value: Total portfolio value

        Returns:
            Discrete allocation results
        """
        try:
            # Convert to pandas Series
            weights_series = pd.Series(weights)
            prices_series = pd.Series(latest_prices)

            # Create discrete allocation
            da = DiscreteAllocation(weights_series, prices_series, total_portfolio_value)

            # Get allocation
            allocation, leftover = da.lp_portfolio()

            # Calculate actual weights
            actual_weights = {}
            for asset, shares in allocation.items():
                value = shares * latest_prices[asset]
                actual_weights[asset] = value / total_portfolio_value

            return {
                'success': True,
                'allocation': allocation,
                'leftover_cash': leftover,
                'actual_weights': actual_weights,
                'total_value': total_portfolio_value - leftover
            }

        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    'portfolio_optimizer', 'discrete_allocation_failed',
                    f"Discrete allocation failed: {e}",
                    exception=e
                )
            return {'success': False, 'error': str(e)}

    def rebalance_portfolio(self, current_weights: Dict[str, float],
                          target_weights: Dict[str, float],
                          current_prices: Dict[str, float],
                          portfolio_value: float,
                          max_turnover: float = 0.20) -> Dict[str, Any]:
        """Calculate portfolio rebalancing trades

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            current_prices: Current asset prices
            portfolio_value: Total portfolio value
            max_turnover: Maximum allowed turnover

        Returns:
            Rebalancing instructions
        """
        try:
            # Calculate weight differences
            weight_changes = {}
            total_turnover = 0

            for asset in set(list(current_weights.keys()) + list(target_weights.keys())):
                current_weight = current_weights.get(asset, 0)
                target_weight = target_weights.get(asset, 0)
                weight_change = target_weight - current_weight
                weight_changes[asset] = weight_change
                total_turnover += abs(weight_change)

            # Check turnover constraint
            if total_turnover > max_turnover:
                # Scale down changes to meet turnover constraint
                scale_factor = max_turnover / total_turnover
                weight_changes = {asset: change * scale_factor
                                for asset, change in weight_changes.items()}

            # Calculate trades
            trades = {}
            for asset, weight_change in weight_changes.items():
                if abs(weight_change) > 0.001:  # Minimum trade threshold
                    value_change = weight_change * portfolio_value
                    price = current_prices.get(asset, 0)
                    if price > 0:
                        shares = value_change / price
                        trades[asset] = {
                            'shares': round(shares, 6),
                            'value': value_change,
                            'action': 'buy' if shares > 0 else 'sell'
                        }

            return {
                'success': True,
                'trades': trades,
                'total_turnover': sum(abs(change) for change in weight_changes.values()),
                'weight_changes': weight_changes
            }

        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    'portfolio_optimizer', 'rebalancing_failed',
                    f"Portfolio rebalancing failed: {e}",
                    exception=e
                )
            return {'success': False, 'error': str(e)}

    def risk_budget_optimization(self, prices: pd.DataFrame,
                               risk_budgets: Dict[str, float]) -> Dict[str, Any]:
        """Optimize portfolio using risk budgeting

        Args:
            prices: Price data
            risk_budgets: Risk budget for each asset

        Returns:
            Optimization results
        """
        try:
            # Calculate returns and covariance
            returns = prices.pct_change().dropna()
            cov_matrix = returns.cov() * 252

            # Normalize risk budgets
            total_budget = sum(risk_budgets.values())
            normalized_budgets = {asset: budget / total_budget
                                for asset, budget in risk_budgets.items()}

            # Risk parity optimization (equal risk contribution)
            n_assets = len(prices.columns)
            weights = np.array([1.0 / n_assets] * n_assets)

            # Iterative optimization to match risk budgets
            for _ in range(100):  # Max iterations
                # Calculate marginal risk contributions
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                marginal_contribs = np.dot(cov_matrix, weights) / portfolio_vol
                risk_contribs = weights * marginal_contribs / portfolio_vol

                # Adjust weights based on risk budget targets
                for i, asset in enumerate(prices.columns):
                    target_contrib = normalized_budgets.get(asset, 1.0 / n_assets)
                    current_contrib = risk_contribs[i]
                    if current_contrib > 0:
                        weights[i] *= target_contrib / current_contrib

                # Normalize weights
                weights = weights / np.sum(weights)

            # Convert to dictionary
            weight_dict = dict(zip(prices.columns, weights))

            # Calculate portfolio performance
            portfolio_return = np.sum(weights * returns.mean() * 252)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0

            return {
                'success': True,
                'weights': weight_dict,
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'method': 'risk_budgeting',
                'risk_budgets': normalized_budgets,
                'optimization_date': datetime.now()
            }

        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    'portfolio_optimizer', 'risk_budget_failed',
                    f"Risk budget optimization failed: {e}",
                    exception=e
                )
            return {'success': False, 'error': str(e)}

    def save_portfolio(self, portfolio_name: str, portfolio_data: Dict[str, Any]):
        """Save portfolio configuration

        Args:
            portfolio_name: Portfolio identifier
            portfolio_data: Portfolio data
        """
        self.portfolios[portfolio_name] = portfolio_data

        if self.logger:
            self.logger.log_system_event(
                'portfolio_optimizer', 'portfolio_saved',
                {
                    'portfolio_name': portfolio_name,
                    'method': portfolio_data.get('method'),
                    'assets_count': len(portfolio_data.get('assets', []))
                }
            )

    def get_portfolio(self, portfolio_name: str) -> Optional[Dict[str, Any]]:
        """Get saved portfolio

        Args:
            portfolio_name: Portfolio identifier

        Returns:
            Portfolio data or None
        """
        return self.portfolios.get(portfolio_name)

    def list_portfolios(self) -> List[str]:
        """List all saved portfolios

        Returns:
            List of portfolio names
        """
        return list(self.portfolios.keys())

    def analyze_portfolio_performance(self, weights: Dict[str, float],
                                    prices: pd.DataFrame,
                                    benchmark_prices: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Analyze portfolio performance

        Args:
            weights: Portfolio weights
            prices: Price data
            benchmark_prices: Benchmark prices for comparison

        Returns:
            Performance analysis
        """
        try:
            # Calculate portfolio returns
            returns = prices.pct_change().dropna()
            portfolio_returns = returns.dot(pd.Series(weights))

            # Performance metrics
            total_return = (1 + portfolio_returns).prod() - 1
            annualized_return = (1 + portfolio_returns.mean()) ** 252 - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

            # Drawdown analysis
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()

            # Risk metrics
            var_95 = portfolio_returns.quantile(0.05)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()

            analysis = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'portfolio_returns': portfolio_returns.tolist()
            }

            # Benchmark comparison if provided
            if benchmark_prices is not None:
                benchmark_returns = benchmark_prices.pct_change().dropna()
                benchmark_total_return = (1 + benchmark_returns).prod() - 1
                benchmark_volatility = benchmark_returns.std() * np.sqrt(252)

                analysis['benchmark_total_return'] = benchmark_total_return
                analysis['benchmark_volatility'] = benchmark_volatility
                analysis['excess_return'] = total_return - benchmark_total_return
                analysis['tracking_error'] = (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
                analysis['information_ratio'] = (analysis['excess_return'] / analysis['tracking_error']
                                               if analysis['tracking_error'] > 0 else 0)

            return analysis

        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    'portfolio_optimizer', 'performance_analysis_failed',
                    f"Portfolio performance analysis failed: {e}",
                    exception=e
                )
            return {'error': str(e)}
