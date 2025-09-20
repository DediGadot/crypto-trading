"""
EXCHANGE REGISTRY
Unified multi-exchange connector using CCXT
"""

import ccxt.async_support as ccxt
import asyncio
import logging
import os
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import yaml
from datetime import datetime, timedelta

from ..config.config_loader import get_config
from ..logging_system import TradeLogger
from ..utils.circuit_breaker import circuit_manager, CircuitConfig


class ExchangeRegistry:
    """Unified registry for multiple cryptocurrency exchanges"""

    def __init__(self, logger: Optional[TradeLogger] = None):
        """Initialize exchange registry

        Args:
            logger: Trade logger instance
        """
        self.config = get_config()
        self.logger = logger

        # Supported exchanges with CCXT
        self.supported_exchanges = {
            'binance': ccxt.binance,
            'kraken': ccxt.kraken,
            'bybit': ccxt.bybit,
            'okx': ccxt.okx,
            'kucoin': ccxt.kucoin,
            'huobi': ccxt.huobi,
            'bitfinex': ccxt.bitfinex
        }

        # Active exchange connections
        self.exchanges: Dict[str, ccxt.Exchange] = {}

        # Circuit breakers for each exchange
        self.circuit_breakers = {}

        # Exchange capabilities cache
        self.capabilities_cache = {}

    async def initialize_exchange(self, exchange_name: str,
                                credentials: Optional[Dict[str, str]] = None) -> bool:
        """Initialize a specific exchange connection

        Args:
            exchange_name: Name of the exchange
            credentials: Optional API credentials

        Returns:
            True if initialized successfully
        """
        if exchange_name not in self.supported_exchanges:
            if self.logger:
                self.logger.log_error(
                    'exchange_registry', 'unsupported_exchange',
                    f"Exchange {exchange_name} not supported",
                    context={'supported': list(self.supported_exchanges.keys())}
                )
            return False

        try:
            # Get exchange class
            exchange_class = self.supported_exchanges[exchange_name]

            # Setup configuration - PRODUCTION MODE
            config = {
                'sandbox': False,  # ALWAYS use production for real data
                'rateLimit': 1200,  # Conservative rate limiting
                'timeout': 30000,
                'enableRateLimit': True,
                'verbose': False
            }

            # Add credentials if provided
            if credentials:
                config.update(credentials)
            else:
                # Try to get from environment variables first (most secure)
                env_prefix = exchange_name.upper()
                api_key = os.getenv(f'{env_prefix}_API_KEY')
                secret = os.getenv(f'{env_prefix}_SECRET')
                passphrase = os.getenv(f'{env_prefix}_PASSPHRASE')

                # Fall back to config file if no environment variables
                if not api_key:
                    try:
                        api_key = self.config.get(f'exchanges.{exchange_name}.api_key')
                    except KeyError:
                        api_key = None
                if not secret:
                    try:
                        secret = self.config.get(f'exchanges.{exchange_name}.secret')
                    except KeyError:
                        secret = None
                if not passphrase:
                    try:
                        passphrase = self.config.get(f'exchanges.{exchange_name}.passphrase')
                    except KeyError:
                        passphrase = None

                if api_key and secret:
                    config['apiKey'] = api_key
                    config['secret'] = secret
                    if passphrase:
                        config['password'] = passphrase

                    if self.logger:
                        self.logger.log_system_event(
                            'exchange_registry', 'credentials_loaded',
                            {
                                'exchange': exchange_name,
                                'source': 'environment' if os.getenv(f'{env_prefix}_API_KEY') else 'config',
                                'has_passphrase': bool(passphrase)
                            }
                        )
                else:
                    if self.logger:
                        self.logger.log_system_event(
                            'exchange_registry', 'no_credentials',
                            {
                                'exchange': exchange_name,
                                'message': 'No API credentials found - using public endpoints only'
                            }
                        )

            # Create exchange instance
            exchange = exchange_class(config)

            # Test connection
            await exchange.load_markets()

            # Store exchange
            self.exchanges[exchange_name] = exchange

            # Setup circuit breaker
            circuit_config = CircuitConfig(
                failure_threshold=self.config.get(f'exchanges.{exchange_name}.circuit_breaker.failure_threshold', 5),
                recovery_timeout=self.config.get(f'exchanges.{exchange_name}.circuit_breaker.recovery_timeout', 30.0),
                success_threshold=self.config.get(f'exchanges.{exchange_name}.circuit_breaker.success_threshold', 3),
                timeout=self.config.get(f'exchanges.{exchange_name}.timeout_seconds', 30)
            )
            self.circuit_breakers[exchange_name] = circuit_manager.get_circuit(
                f'{exchange_name}_api', circuit_config
            )

            # Cache capabilities
            self.capabilities_cache[exchange_name] = {
                'markets': len(exchange.markets),
                'symbols': list(exchange.markets.keys())[:10],  # First 10 for logging
                'has_websocket': getattr(exchange, 'has', {}).get('ws', False),
                'has_ohlcv': exchange.has.get('fetchOHLCV', False),
                'has_trades': exchange.has.get('fetchTrades', False),
                'has_order_book': exchange.has.get('fetchOrderBook', False)
            }

            if self.logger:
                self.logger.log_system_event(
                    'exchange_registry', 'exchange_initialized',
                    {
                        'exchange': exchange_name,
                        'markets_count': len(exchange.markets),
                        'capabilities': self.capabilities_cache[exchange_name]
                    }
                )

            return True

        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    'exchange_registry', 'initialization_failed',
                    f"Failed to initialize {exchange_name}: {e}",
                    exception=e,
                    context={'exchange': exchange_name}
                )
            return False

    async def get_exchange(self, exchange_name: str) -> Optional[ccxt.Exchange]:
        """Get exchange instance

        Args:
            exchange_name: Name of the exchange

        Returns:
            Exchange instance or None
        """
        if exchange_name not in self.exchanges:
            # Try to initialize
            success = await self.initialize_exchange(exchange_name)
            if not success:
                return None

        return self.exchanges.get(exchange_name)

    async def get_unified_ticker(self, symbol: str,
                               exchanges: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get ticker from multiple exchanges for best price discovery

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            exchanges: List of exchanges to query (None for all)

        Returns:
            Unified ticker data with best bid/ask
        """
        if exchanges is None:
            exchanges = list(self.exchanges.keys())

        tickers = {}
        best_bid = 0.0
        best_ask = float('inf')
        best_bid_exchange = None
        best_ask_exchange = None

        for exchange_name in exchanges:
            exchange = await self.get_exchange(exchange_name)
            if not exchange:
                continue

            try:
                # Use circuit breaker
                circuit_breaker = self.circuit_breakers.get(exchange_name)
                if circuit_breaker:
                    ticker = await circuit_breaker.call(
                        exchange.fetch_ticker, symbol
                    )
                else:
                    ticker = await exchange.fetch_ticker(symbol)

                tickers[exchange_name] = ticker

                # Update best prices
                if ticker.get('bid') and ticker['bid'] > best_bid:
                    best_bid = ticker['bid']
                    best_bid_exchange = exchange_name

                if ticker.get('ask') and ticker['ask'] < best_ask:
                    best_ask = ticker['ask']
                    best_ask_exchange = exchange_name

            except Exception as e:
                if self.logger:
                    self.logger.log_error(
                        'exchange_registry', 'ticker_fetch_failed',
                        f"Failed to fetch ticker from {exchange_name}: {e}",
                        context={'exchange': exchange_name, 'symbol': symbol}
                    )

        # Create unified ticker
        unified_ticker = {
            'symbol': symbol,
            'timestamp': datetime.now().timestamp() * 1000,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'best_bid_exchange': best_bid_exchange,
            'best_ask_exchange': best_ask_exchange,
            'spread': best_ask - best_bid if best_ask != float('inf') else 0,
            'spread_pct': ((best_ask - best_bid) / best_bid * 100) if best_bid > 0 and best_ask != float('inf') else 0,
            'exchanges': tickers
        }

        return unified_ticker

    async def get_unified_order_book(self, symbol: str,
                                   exchanges: Optional[List[str]] = None,
                                   limit: int = 100) -> Dict[str, Any]:
        """Get aggregated order book from multiple exchanges

        Args:
            symbol: Trading symbol
            exchanges: List of exchanges to query
            limit: Depth limit per exchange

        Returns:
            Aggregated order book
        """
        if exchanges is None:
            exchanges = list(self.exchanges.keys())

        all_bids = []
        all_asks = []
        exchange_books = {}

        for exchange_name in exchanges:
            exchange = await self.get_exchange(exchange_name)
            if not exchange:
                continue

            try:
                circuit_breaker = self.circuit_breakers.get(exchange_name)
                if circuit_breaker:
                    order_book = await circuit_breaker.call(
                        exchange.fetch_order_book, symbol, limit
                    )
                else:
                    order_book = await exchange.fetch_order_book(symbol, limit)

                exchange_books[exchange_name] = order_book

                # Add exchange info to orders
                for bid in order_book['bids']:
                    all_bids.append([bid[0], bid[1], exchange_name])

                for ask in order_book['asks']:
                    all_asks.append([ask[0], ask[1], exchange_name])

            except Exception as e:
                if self.logger:
                    self.logger.log_error(
                        'exchange_registry', 'orderbook_fetch_failed',
                        f"Failed to fetch order book from {exchange_name}: {e}",
                        context={'exchange': exchange_name, 'symbol': symbol}
                    )

        # Sort and aggregate
        all_bids.sort(key=lambda x: x[0], reverse=True)  # Highest bids first
        all_asks.sort(key=lambda x: x[0])  # Lowest asks first

        return {
            'symbol': symbol,
            'timestamp': datetime.now().timestamp() * 1000,
            'bids': all_bids[:limit],
            'asks': all_asks[:limit],
            'best_bid': all_bids[0][0] if all_bids else 0,
            'best_ask': all_asks[0][0] if all_asks else 0,
            'exchanges': exchange_books
        }

    async def get_historical_data(self, symbol: str, timeframe: str,
                                since: Optional[datetime] = None,
                                limit: int = 1000,
                                exchange_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get historical OHLCV data

        Args:
            symbol: Trading symbol
            timeframe: Timeframe ('1m', '5m', '1h', etc.)
            since: Start date
            limit: Maximum candles
            exchange_name: Specific exchange (None for first available)

        Returns:
            Historical OHLCV data
        """
        # Select exchange
        if exchange_name:
            exchange = await self.get_exchange(exchange_name)
            if not exchange:
                return []
        else:
            # Use first available exchange
            for name in self.exchanges:
                exchange = await self.get_exchange(name)
                if exchange and exchange.has.get('fetchOHLCV'):
                    exchange_name = name
                    break
            else:
                return []

        try:
            since_ms = None
            if since:
                since_ms = int(since.timestamp() * 1000)

            circuit_breaker = self.circuit_breakers.get(exchange_name)
            if circuit_breaker:
                ohlcv = await circuit_breaker.call(
                    exchange.fetch_ohlcv, symbol, timeframe, since_ms, limit
                )
            else:
                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since_ms, limit)

            # Convert to standard format
            candles = []
            for candle in ohlcv:
                candles.append({
                    'timestamp': candle[0],
                    'open': candle[1],
                    'high': candle[2],
                    'low': candle[3],
                    'close': candle[4],
                    'volume': candle[5],
                    'exchange': exchange_name,
                    'symbol': symbol,
                    'timeframe': timeframe
                })

            return candles

        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    'exchange_registry', 'historical_data_failed',
                    f"Failed to fetch historical data: {e}",
                    context={
                        'exchange': exchange_name,
                        'symbol': symbol,
                        'timeframe': timeframe
                    }
                )
            return []

    def get_supported_symbols(self, exchange_name: Optional[str] = None) -> List[str]:
        """Get supported trading symbols

        Args:
            exchange_name: Specific exchange (None for all)

        Returns:
            List of supported symbols
        """
        if exchange_name:
            exchange = self.exchanges.get(exchange_name)
            if exchange and hasattr(exchange, 'markets'):
                return list(exchange.markets.keys())
            return []

        # Combine all symbols
        all_symbols = set()
        for exchange in self.exchanges.values():
            if hasattr(exchange, 'markets'):
                all_symbols.update(exchange.markets.keys())

        return sorted(list(all_symbols))

    def get_exchange_capabilities(self, exchange_name: str) -> Dict[str, Any]:
        """Get exchange capabilities

        Args:
            exchange_name: Exchange name

        Returns:
            Capabilities dictionary
        """
        return self.capabilities_cache.get(exchange_name, {})

    def get_status(self) -> Dict[str, Any]:
        """Get registry status"""
        status = {
            'connected_exchanges': list(self.exchanges.keys()),
            'total_supported': len(self.supported_exchanges),
            'circuit_breaker_stats': {}
        }

        # Add circuit breaker statistics
        for name, breaker in self.circuit_breakers.items():
            status['circuit_breaker_stats'][name] = breaker.get_stats()

        return status

    async def close_all(self):
        """Close all exchange connections"""
        for exchange_name, exchange in self.exchanges.items():
            try:
                await exchange.close()
                if self.logger:
                    self.logger.log_system_event(
                        'exchange_registry', 'exchange_closed',
                        {'exchange': exchange_name}
                    )
            except Exception as e:
                if self.logger:
                    self.logger.log_error(
                        'exchange_registry', 'close_failed',
                        f"Failed to close {exchange_name}: {e}",
                        context={'exchange': exchange_name}
                    )

        self.exchanges.clear()
        self.circuit_breakers.clear()


# Global registry instance
_registry_instance = None


async def get_exchange_registry(logger: Optional[TradeLogger] = None) -> ExchangeRegistry:
    """Get global exchange registry instance"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ExchangeRegistry(logger)
    return _registry_instance