"""
CRYPTOFEED STREAMING CONNECTOR
Real-time multi-exchange data streams using cryptofeed
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime
from dataclasses import dataclass
import threading

try:
    from cryptofeed import FeedHandler
    from cryptofeed.exchanges import Binance, Coinbase, Kraken, Bybit, OKX
    from cryptofeed.defines import TRADES, L2_BOOK, TICKER, CANDLES
    from cryptofeed.types import Trade, OrderBook, Ticker, Candle
    CRYPTOFEED_AVAILABLE = True
except ImportError:
    CRYPTOFEED_AVAILABLE = False
    # Fallback types
    class Trade:
        pass
    class OrderBook:
        pass
    class Ticker:
        pass
    class Candle:
        pass

from ..logging_system import TradeLogger
from ..config.config_loader import get_config


@dataclass
class StreamConfig:
    """Configuration for streaming data"""
    exchanges: List[str]
    symbols: List[str]
    channels: List[str]  # TRADES, L2_BOOK, TICKER, CANDLES
    buffer_size: int = 1000
    enable_snapshots: bool = True


class CryptofeedStreams:
    """Real-time cryptocurrency data streams using cryptofeed"""

    def __init__(self, logger: Optional[TradeLogger] = None):
        """Initialize cryptofeed streams

        Args:
            logger: Trade logger instance
        """
        if not CRYPTOFEED_AVAILABLE:
            raise ImportError("cryptofeed is required for streaming functionality. Install with: pip install cryptofeed")

        self.config = get_config()
        self.logger = logger

        # Supported exchanges mapping
        self.exchange_map = {
            'binance': Binance,
            'coinbase': Coinbase,
            'kraken': Kraken,
            'bybit': Bybit,
            'okx': OKX
        }

        # Feed handler
        self.feed_handler = None

        # Data buffers
        self.trade_buffer: Dict[str, List[Trade]] = {}
        self.orderbook_buffer: Dict[str, OrderBook] = {}
        self.ticker_buffer: Dict[str, Ticker] = {}
        self.candle_buffer: Dict[str, List[Candle]] = {}

        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            'trade': [],
            'orderbook': [],
            'ticker': [],
            'candle': []
        }

        # Thread safety
        self._lock = threading.RLock()

        # Stream status
        self.is_running = False
        self.connected_exchanges: Set[str] = set()

    def add_callback(self, data_type: str, callback: Callable):
        """Add callback for specific data type

        Args:
            data_type: Type of data ('trade', 'orderbook', 'ticker', 'candle')
            callback: Callback function
        """
        with self._lock:
            if data_type in self.callbacks:
                self.callbacks[data_type].append(callback)

    def remove_callback(self, data_type: str, callback: Callable):
        """Remove callback for specific data type

        Args:
            data_type: Type of data
            callback: Callback function to remove
        """
        with self._lock:
            if data_type in self.callbacks and callback in self.callbacks[data_type]:
                self.callbacks[data_type].remove(callback)

    async def start_streams(self, stream_config: StreamConfig):
        """Start streaming data

        Args:
            stream_config: Stream configuration
        """
        if self.is_running:
            if self.logger:
                self.logger.log_system_event(
                    'cryptofeed_streams', 'already_running',
                    {'status': 'streams already running'}
                )
            return

        try:
            # Initialize feed handler
            self.feed_handler = FeedHandler()

            # Setup exchanges and subscriptions
            for exchange_name in stream_config.exchanges:
                if exchange_name not in self.exchange_map:
                    if self.logger:
                        self.logger.log_error(
                            'cryptofeed_streams', 'unsupported_exchange',
                            f"Exchange {exchange_name} not supported",
                            context={'supported': list(self.exchange_map.keys())}
                        )
                    continue

                exchange_class = self.exchange_map[exchange_name]

                # Create exchange config
                exchange_config = {}

                # Add callbacks for each channel
                if TRADES in stream_config.channels:
                    exchange_config[TRADES] = stream_config.symbols

                if L2_BOOK in stream_config.channels:
                    exchange_config[L2_BOOK] = stream_config.symbols

                if TICKER in stream_config.channels:
                    exchange_config[TICKER] = stream_config.symbols

                if CANDLES in stream_config.channels:
                    exchange_config[CANDLES] = stream_config.symbols

                # Add exchange to feed handler
                self.feed_handler.add_feed(
                    exchange_class(
                        subscription=exchange_config,
                        callbacks={
                            TRADES: self._handle_trade,
                            L2_BOOK: self._handle_orderbook,
                            TICKER: self._handle_ticker,
                            CANDLES: self._handle_candle
                        }
                    )
                )

                self.connected_exchanges.add(exchange_name)

            # Start the feed handler
            if self.logger:
                self.logger.log_system_event(
                    'cryptofeed_streams', 'starting_streams',
                    {
                        'exchanges': list(self.connected_exchanges),
                        'symbols': stream_config.symbols,
                        'channels': stream_config.channels
                    }
                )

            # Run in background task
            self.is_running = True
            await self.feed_handler.run()

        except Exception as e:
            self.is_running = False
            if self.logger:
                self.logger.log_error(
                    'cryptofeed_streams', 'start_failed',
                    f"Failed to start streams: {e}",
                    exception=e
                )
            raise

    def stop_streams(self):
        """Stop streaming data"""
        if not self.is_running:
            return

        try:
            if self.feed_handler:
                # Stop the feed handler
                self.feed_handler.stop()

            self.is_running = False
            self.connected_exchanges.clear()

            if self.logger:
                self.logger.log_system_event(
                    'cryptofeed_streams', 'streams_stopped',
                    {'status': 'all streams stopped'}
                )

        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    'cryptofeed_streams', 'stop_failed',
                    f"Failed to stop streams: {e}",
                    exception=e
                )

    async def _handle_trade(self, trade: Trade, receipt_timestamp: float):
        """Handle incoming trade data

        Args:
            trade: Trade data
            receipt_timestamp: Receipt timestamp
        """
        try:
            with self._lock:
                # Buffer trade
                symbol = trade.symbol
                if symbol not in self.trade_buffer:
                    self.trade_buffer[symbol] = []

                self.trade_buffer[symbol].append(trade)

                # Keep buffer size manageable
                if len(self.trade_buffer[symbol]) > 1000:
                    self.trade_buffer[symbol] = self.trade_buffer[symbol][-500:]

            # Call user callbacks
            for callback in self.callbacks['trade']:
                try:
                    await callback(trade, receipt_timestamp)
                except Exception as e:
                    if self.logger:
                        self.logger.log_error(
                            'cryptofeed_streams', 'callback_error',
                            f"Trade callback error: {e}",
                            context={'symbol': trade.symbol}
                        )

        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    'cryptofeed_streams', 'trade_handler_error',
                    f"Trade handler error: {e}",
                    exception=e
                )

    async def _handle_orderbook(self, orderbook: OrderBook, receipt_timestamp: float):
        """Handle incoming order book data

        Args:
            orderbook: Order book data
            receipt_timestamp: Receipt timestamp
        """
        try:
            with self._lock:
                # Store latest order book
                self.orderbook_buffer[orderbook.symbol] = orderbook

            # Call user callbacks
            for callback in self.callbacks['orderbook']:
                try:
                    await callback(orderbook, receipt_timestamp)
                except Exception as e:
                    if self.logger:
                        self.logger.log_error(
                            'cryptofeed_streams', 'callback_error',
                            f"Orderbook callback error: {e}",
                            context={'symbol': orderbook.symbol}
                        )

        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    'cryptofeed_streams', 'orderbook_handler_error',
                    f"Orderbook handler error: {e}",
                    exception=e
                )

    async def _handle_ticker(self, ticker: Ticker, receipt_timestamp: float):
        """Handle incoming ticker data

        Args:
            ticker: Ticker data
            receipt_timestamp: Receipt timestamp
        """
        try:
            with self._lock:
                # Store latest ticker
                self.ticker_buffer[ticker.symbol] = ticker

            # Call user callbacks
            for callback in self.callbacks['ticker']:
                try:
                    await callback(ticker, receipt_timestamp)
                except Exception as e:
                    if self.logger:
                        self.logger.log_error(
                            'cryptofeed_streams', 'callback_error',
                            f"Ticker callback error: {e}",
                            context={'symbol': ticker.symbol}
                        )

        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    'cryptofeed_streams', 'ticker_handler_error',
                    f"Ticker handler error: {e}",
                    exception=e
                )

    async def _handle_candle(self, candle: Candle, receipt_timestamp: float):
        """Handle incoming candle data

        Args:
            candle: Candle data
            receipt_timestamp: Receipt timestamp
        """
        try:
            with self._lock:
                # Buffer candle
                symbol = candle.symbol
                if symbol not in self.candle_buffer:
                    self.candle_buffer[symbol] = []

                self.candle_buffer[symbol].append(candle)

                # Keep buffer size manageable
                if len(self.candle_buffer[symbol]) > 1000:
                    self.candle_buffer[symbol] = self.candle_buffer[symbol][-500:]

            # Call user callbacks
            for callback in self.callbacks['candle']:
                try:
                    await callback(candle, receipt_timestamp)
                except Exception as e:
                    if self.logger:
                        self.logger.log_error(
                            'cryptofeed_streams', 'callback_error',
                            f"Candle callback error: {e}",
                            context={'symbol': candle.symbol}
                        )

        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    'cryptofeed_streams', 'candle_handler_error',
                    f"Candle handler error: {e}",
                    exception=e
                )

    def get_latest_trade(self, symbol: str) -> Optional[Trade]:
        """Get latest trade for symbol

        Args:
            symbol: Trading symbol

        Returns:
            Latest trade or None
        """
        with self._lock:
            trades = self.trade_buffer.get(symbol, [])
            return trades[-1] if trades else None

    def get_latest_orderbook(self, symbol: str) -> Optional[OrderBook]:
        """Get latest order book for symbol

        Args:
            symbol: Trading symbol

        Returns:
            Latest order book or None
        """
        with self._lock:
            return self.orderbook_buffer.get(symbol)

    def get_latest_ticker(self, symbol: str) -> Optional[Ticker]:
        """Get latest ticker for symbol

        Args:
            symbol: Trading symbol

        Returns:
            Latest ticker or None
        """
        with self._lock:
            return self.ticker_buffer.get(symbol)

    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        """Get recent trades for symbol

        Args:
            symbol: Trading symbol
            limit: Maximum number of trades

        Returns:
            List of recent trades
        """
        with self._lock:
            trades = self.trade_buffer.get(symbol, [])
            return trades[-limit:] if trades else []

    def get_recent_candles(self, symbol: str, limit: int = 100) -> List[Candle]:
        """Get recent candles for symbol

        Args:
            symbol: Trading symbol
            limit: Maximum number of candles

        Returns:
            List of recent candles
        """
        with self._lock:
            candles = self.candle_buffer.get(symbol, [])
            return candles[-limit:] if candles else []

    def get_status(self) -> Dict[str, Any]:
        """Get stream status"""
        with self._lock:
            return {
                'is_running': self.is_running,
                'connected_exchanges': list(self.connected_exchanges),
                'symbols_with_trades': list(self.trade_buffer.keys()),
                'symbols_with_orderbooks': list(self.orderbook_buffer.keys()),
                'symbols_with_tickers': list(self.ticker_buffer.keys()),
                'symbols_with_candles': list(self.candle_buffer.keys()),
                'total_callbacks': sum(len(callbacks) for callbacks in self.callbacks.values())
            }

    def clear_buffers(self):
        """Clear all data buffers"""
        with self._lock:
            self.trade_buffer.clear()
            self.orderbook_buffer.clear()
            self.ticker_buffer.clear()
            self.candle_buffer.clear()


# Convenience function for easy stream setup
async def create_crypto_streams(exchanges: List[str],
                              symbols: List[str],
                              channels: List[str],
                              logger: Optional[TradeLogger] = None) -> CryptofeedStreams:
    """Create and configure cryptofeed streams

    Args:
        exchanges: List of exchange names
        symbols: List of trading symbols
        channels: List of data channels
        logger: Trade logger

    Returns:
        Configured CryptofeedStreams instance
    """
    streams = CryptofeedStreams(logger)

    config = StreamConfig(
        exchanges=exchanges,
        symbols=symbols,
        channels=channels
    )

    return streams, config