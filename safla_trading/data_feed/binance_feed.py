"""
BINANCE DATA FEED
Real market data from real exchange, not imaginary vectors
"""

import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
import asyncio
import time
from typing import Dict, List, Optional, AsyncGenerator, Any
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
import pytz

from ..config.config_loader import get_config
from ..logging_system import TradeLogger
from ..utils.circuit_breaker import circuit_manager, CircuitConfig
from ..monitoring import get_performance_monitor


@dataclass
class MarketTick:
    """Market data tick"""
    symbol: str
    timestamp: int  # milliseconds
    price: float
    volume: float
    bid: float
    ask: float


@dataclass
class OHLCV:
    """OHLCV candle data"""
    symbol: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class BinanceDataFeed:
    """
    REAL BINANCE DATA FEED
    Historical data for backtesting, not fairy tale embeddings
    """

    def __init__(self, logger: Optional[TradeLogger] = None):
        """Initialize Binance data feed

        Args:
            logger: Trade logger instance
        """
        self.config = get_config()
        self.logger = logger

        # Initialize CCXT exchange
        self.exchange = ccxt.binance({
            'sandbox': self.config.get('exchange.sandbox', True),
            'rateLimit': 60000 // self.config.get('exchange.rate_limit_per_minute', 1200),
            'timeout': self.config.get('exchange.timeout_seconds', 30) * 1000,
            'enableRateLimit': True
        })

        # Cache settings
        self.cache_dir = Path(self.config.get('storage.cache_directory', 'data/cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Data validation
        self.last_prices = {}  # For price validation
        self.price_change_threshold = 0.10  # 10% max price change per minute

        # Circuit breaker for resilience
        circuit_config = CircuitConfig(
            failure_threshold=self.config.get('exchange.circuit_breaker.failure_threshold', 5),
            recovery_timeout=self.config.get('exchange.circuit_breaker.recovery_timeout', 30.0),
            success_threshold=self.config.get('exchange.circuit_breaker.success_threshold', 3),
            timeout=self.config.get('exchange.timeout_seconds', 30)
        )
        self.circuit_breaker = circuit_manager.get_circuit('binance_api', circuit_config)

        # Performance monitoring
        self.performance_monitor = get_performance_monitor(logger)

    async def fetch_historical_ohlcv(self, symbol: str, timeframe: str = '1m',
                                    since: Optional[datetime] = None,
                                    limit: int = 1000) -> List[OHLCV]:
        """Fetch historical OHLCV data

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe ('1m', '5m', '1h', etc.)
            since: Start date (None for latest)
            limit: Maximum number of candles

        Returns:
            List of OHLCV candles
        """
        try:
            # Convert datetime to timestamp if provided
            since_ms = None
            if since:
                since_ms = int(since.timestamp() * 1000)

            # Check cache first
            cache_key = f"{symbol.replace('/', '_')}_{timeframe}_{since_ms}_{limit}"
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            csv_cache_file = cache_file.with_suffix('.csv')

            if cache_file.exists():
                df = pd.read_parquet(cache_file)
                cache_backend = 'parquet'
                if self.logger:
                    self.logger.log_system_event(
                        'data_feed', 'cache_hit',
                        {
                            'symbol': symbol,
                            'cache_file': str(cache_file),
                            'cache_backend': cache_backend,
                        }
                    )
            elif csv_cache_file.exists():
                df = pd.read_csv(csv_cache_file)
                cache_backend = 'csv'
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                if self.logger:
                    self.logger.log_system_event(
                        'data_feed', 'cache_hit',
                        {
                            'symbol': symbol,
                            'cache_file': str(csv_cache_file),
                            'cache_backend': cache_backend,
                        }
                    )
            else:
                # Fetch from exchange
                raw_data = await self._fetch_ohlcv_with_retry(
                    symbol, timeframe, since_ms, limit
                )

                if not raw_data:
                    return []

                # Convert to DataFrame
                df = pd.DataFrame(raw_data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume'
                ])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                cache_backend = 'parquet'
                cache_path_used = cache_file
                try:
                    df.to_parquet(cache_file)
                except (ImportError, ValueError, AttributeError) as parquet_error:
                    cache_backend = 'csv'
                    cache_path_used = csv_cache_file
                    df.to_csv(cache_path_used, index=False)
                    if self.logger:
                        self.logger.log_system_event(
                            'data_feed', 'cache_fallback',
                            {
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'reason': str(parquet_error)[:200],
                                'cache_backend': cache_backend,
                                'cache_file': str(cache_path_used),
                            }
                        )

                if self.logger:
                    self.logger.log_system_event(
                        'data_feed', 'data_fetched',
                        {
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'candles': len(df),
                            'cache_backend': cache_backend,
                            'cache_file': str(cache_path_used),
                        }
                    )

            # Convert to OHLCV objects
            ohlcv_data = []
            for _, row in df.iterrows():
                ohlcv_data.append(OHLCV(
                    symbol=symbol,
                    timestamp=int(row['timestamp'].timestamp() * 1000),
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=float(row['volume'])
                ))

            return ohlcv_data

        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    'data_feed', 'fetch_error',
                    f"Failed to fetch OHLCV for {symbol}: {e}",
                    exception=e,
                    context={'symbol': symbol, 'timeframe': timeframe}
                )
            raise

    async def stream_historical_as_live(self, symbol: str, timeframe: str = '1m',
                                       start_date: datetime = None,
                                       end_date: datetime = None,
                                       speed_multiplier: float = 1.0) -> AsyncGenerator[OHLCV, None]:
        """Stream historical data as if it's live data

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date for replay
            end_date: End date for replay
            speed_multiplier: Playback speed (1.0 = real-time, 10.0 = 10x faster)

        Yields:
            OHLCV candles in chronological order
        """
        # Default to last 30 days if no dates provided (UTC)
        if start_date is None:
            start_date = datetime.now(pytz.UTC) - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now(pytz.UTC)

        # Fetch historical data in chunks
        current_date = start_date
        chunk_size = 1000  # Binance limit

        while current_date < end_date:
            try:
                # Fetch chunk
                chunk_data = await self.fetch_historical_ohlcv(
                    symbol, timeframe, current_date, chunk_size
                )

                if not chunk_data:
                    break

                # Stream chunk data
                for candle in chunk_data:
                    if candle.timestamp > int(end_date.timestamp() * 1000):
                        return

                    yield candle

                    # Simulate real-time delay
                    if speed_multiplier < float('inf'):
                        delay = self._get_timeframe_seconds(timeframe) / speed_multiplier
                        await asyncio.sleep(delay)

                # Move to next chunk
                last_timestamp = chunk_data[-1].timestamp
                current_date = datetime.fromtimestamp(last_timestamp / 1000, tz=pytz.UTC) + timedelta(minutes=1)

            except Exception as e:
                if self.logger:
                    self.logger.log_error(
                        'data_feed', 'stream_error',
                        f"Error streaming data for {symbol}: {e}",
                        exception=e
                    )
                break

    async def get_current_price(self, symbol: str) -> Optional[MarketTick]:
        """Get current market price

        Args:
            symbol: Trading symbol

        Returns:
            Current market tick or None
        """
        try:
            ticker = await self._fetch_ticker_with_retry(symbol)
            if not ticker:
                return None

            # Validate price change
            if symbol in self.last_prices:
                last_price = self.last_prices[symbol]
                current_price = ticker['last']
                # Prevent division by zero
                if last_price != 0:
                    change_pct = abs(current_price - last_price) / last_price
                else:
                    change_pct = float('inf') if current_price != 0 else 0.0

                if change_pct > self.price_change_threshold:
                    if self.logger:
                        self.logger.log_error(
                            'data_feed', 'price_validation_error',
                            f"Suspicious price change for {symbol}: {change_pct:.2%}",
                            context={
                                'last_price': last_price,
                                'current_price': current_price,
                                'change_pct': change_pct
                            }
                        )

            self.last_prices[symbol] = ticker['last']

            return MarketTick(
                symbol=symbol,
                timestamp=int(ticker['timestamp']),
                price=ticker['last'],
                volume=ticker['baseVolume'] or 0,
                bid=ticker['bid'] or ticker['last'],
                ask=ticker['ask'] or ticker['last']
            )

        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    'data_feed', 'price_fetch_error',
                    f"Failed to get price for {symbol}: {e}",
                    exception=e
                )
            return None

    async def _fetch_ohlcv_with_retry(self, symbol: str, timeframe: str,
                                     since: Optional[int], limit: int) -> Optional[List]:
        """Fetch OHLCV with retry logic

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            since: Since timestamp
            limit: Limit

        Returns:
            Raw OHLCV data or None
        """
        max_retries = self.config.get('exchange.retry_attempts', 3)
        retry_delay = self.config.get('exchange.retry_delay_seconds', 1)

        # Use circuit breaker for resilience
        try:
            # Track API call for performance monitoring
            self.performance_monitor.record_api_call()
            return await self.circuit_breaker.call(
                self.exchange.fetch_ohlcv, symbol, timeframe, since, limit
            )
        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    'data_feed', 'fetch_ohlcv_error',
                    f"Failed to fetch OHLCV for {symbol}: {e}",
                    exception=e,
                    context={
                        'circuit_stats': self.circuit_breaker.get_stats(),
                        'symbol': symbol,
                        'timeframe': timeframe
                    }
                )
            raise

    async def _fetch_ticker_with_retry(self, symbol: str) -> Optional[Dict]:
        """Fetch ticker with retry logic

        Args:
            symbol: Trading symbol

        Returns:
            Ticker data or None
        """
        max_retries = self.config.get('exchange.retry_attempts', 3)
        retry_delay = self.config.get('exchange.retry_delay_seconds', 1)

        # Use circuit breaker for resilience
        try:
            # Track API call for performance monitoring
            self.performance_monitor.record_api_call()
            return await self.circuit_breaker.call(
                self.exchange.fetch_ticker, symbol
            )
        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    'data_feed', 'fetch_ticker_error',
                    f"Failed to fetch ticker for {symbol}: {e}",
                    exception=e,
                    context={
                        'circuit_stats': self.circuit_breaker.get_stats(),
                        'symbol': symbol
                    }
                )
            return None

    def _get_timeframe_seconds(self, timeframe: str) -> float:
        """Convert timeframe to seconds

        Args:
            timeframe: Timeframe string

        Returns:
            Seconds in timeframe
        """
        timeframe_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }
        return timeframe_map.get(timeframe, 60)

    async def close(self):
        """Close exchange connection"""
        if hasattr(self.exchange, 'close'):
            await self.exchange.close()


# Convenience function
async def get_test_data(symbol: str = 'BTC/USDT',
                       days: int = 7) -> List[OHLCV]:
    """Get test data for development

    Args:
        symbol: Trading symbol
        days: Number of days of data

    Returns:
        Historical OHLCV data
    """
    feed = BinanceDataFeed()
    start_date = datetime.now(pytz.UTC) - timedelta(days=days)

    try:
        return await feed.fetch_historical_ohlcv(symbol, '1m', start_date, 1000)
    finally:
        await feed.close()
