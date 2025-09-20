"""
PROPER LOGGING SYSTEM
Structured logs for actual debugging, not academic feelings
"""

import json
import time
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import threading
from collections import deque

from ..config.config_loader import get_config


@dataclass
class LogEntry:
    """Standard log entry structure"""
    timestamp: float
    level: str
    component: str
    event: str
    data: Dict[str, Any]
    session_id: str
    latency_us: Optional[int] = None
    trace_id: Optional[str] = None


class TradeLogger:
    """
    STRUCTURED LOGGING FOR TRADING SYSTEMS
    Every decision, every trade, every error - logged properly
    """

    def __init__(self, session_id: str, log_dir: str = "logs"):
        """Initialize logger

        Args:
            session_id: Unique session identifier
            log_dir: Directory for log files
        """
        self.session_id = session_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self.config = get_config()

        # Thread-safe file handles
        self._lock = threading.Lock()
        self._files = {}

        # Initialize log files
        log_files = ['trades', 'market_data', 'decisions', 'performance', 'errors']
        for log_type in log_files:
            file_path = self.log_dir / f"{log_type}.jsonl"
            self._files[log_type] = open(file_path, 'a', buffering=1)  # Line buffered

        # In-memory buffers for analysis
        self.trade_buffer = deque(maxlen=1000)
        self.performance_buffer = deque(maxlen=1000)

        # Performance tracking
        self.start_time = time.time()
        self.log_count = 0

    def log_trade(self, trade_type: str, symbol: str, side: str,
                  price: float, quantity: float, commission: float,
                  order_id: str, execution_time_us: int = 0, **kwargs):
        """Log trade execution

        Args:
            trade_type: 'market', 'limit', 'stop'
            symbol: Trading symbol
            side: 'buy' or 'sell'
            price: Execution price
            quantity: Trade quantity
            commission: Commission paid
            order_id: Order identifier
            execution_time_us: Execution latency in microseconds
            **kwargs: Additional trade data
        """
        entry = LogEntry(
            timestamp=time.time(),
            level='TRADE',
            component='simulator',
            event='trade_executed',
            data={
                'trade_type': trade_type,
                'symbol': symbol,
                'side': side,
                'price': price,
                'quantity': quantity,
                'commission': commission,
                'order_id': order_id,
                'value_usd': price * quantity,
                **kwargs
            },
            session_id=self.session_id,
            latency_us=execution_time_us
        )

        self._write_log('trades', entry)
        self.trade_buffer.append(entry)

    def log_decision(self, strategy: str, signal: str, symbol: str,
                     reason: Dict[str, Any], market_data: Dict[str, Any],
                     action_taken: str = None):
        """Log trading decision

        Args:
            strategy: Strategy name
            signal: 'buy', 'sell', 'hold'
            symbol: Trading symbol
            reason: Why this decision was made
            market_data: Current market state
            action_taken: What actually happened
        """
        entry = LogEntry(
            timestamp=time.time(),
            level='DECISION',
            component='strategy',
            event='trading_decision',
            data={
                'strategy': strategy,
                'signal': signal,
                'symbol': symbol,
                'reason': reason,
                'market_data': market_data,
                'action_taken': action_taken
            },
            session_id=self.session_id
        )

        self._write_log('decisions', entry)

    def log_performance(self, balance: float, unrealized_pnl: float,
                       realized_pnl: float, positions: Dict[str, Any],
                       metrics: Dict[str, float] = None):
        """Log performance metrics

        Args:
            balance: Current cash balance
            unrealized_pnl: Unrealized P&L
            realized_pnl: Realized P&L
            positions: Current positions
            metrics: Additional performance metrics
        """
        total_value = balance + unrealized_pnl

        entry = LogEntry(
            timestamp=time.time(),
            level='PERFORMANCE',
            component='portfolio',
            event='performance_update',
            data={
                'balance': balance,
                'unrealized_pnl': unrealized_pnl,
                'realized_pnl': realized_pnl,
                'total_value': total_value,
                'position_count': len(positions),
                'positions': positions,
                'metrics': metrics or {}
            },
            session_id=self.session_id
        )

        self._write_log('performance', entry)
        self.performance_buffer.append(entry)

    def log_market_data(self, symbol: str, timestamp: int, price: float,
                       volume: float, bid: float = None, ask: float = None):
        """Log market data tick

        Args:
            symbol: Trading symbol
            timestamp: Market timestamp
            price: Current price
            volume: Trading volume
            bid: Best bid price
            ask: Best ask price
        """
        entry = LogEntry(
            timestamp=time.time(),
            level='MARKET',
            component='data_feed',
            event='market_tick',
            data={
                'symbol': symbol,
                'market_timestamp': timestamp,
                'price': price,
                'volume': volume,
                'bid': bid,
                'ask': ask,
                'spread': (ask - bid) if (bid and ask) else None
            },
            session_id=self.session_id
        )

        self._write_log('market_data', entry)

    def log_error(self, component: str, error_type: str, message: str,
                  exception: Exception = None, context: Dict[str, Any] = None):
        """Log error with context

        Args:
            component: Component where error occurred
            error_type: Type of error
            message: Error message
            exception: Exception object if available
            context: Additional context
        """
        entry = LogEntry(
            timestamp=time.time(),
            level='ERROR',
            component=component,
            event='error_occurred',
            data={
                'error_type': error_type,
                'message': message,
                'exception': str(exception) if exception else None,
                'context': context or {}
            },
            session_id=self.session_id
        )

        self._write_log('errors', entry)

    def log_system_event(self, component: str, event: str, data: Dict[str, Any]):
        """Log general system event

        Args:
            component: Component name
            event: Event name
            data: Event data
        """
        entry = LogEntry(
            timestamp=time.time(),
            level='INFO',
            component=component,
            event=event,
            data=data,
            session_id=self.session_id
        )

        # Log to appropriate file based on component
        if component == 'simulator':
            self._write_log('trades', entry)
        elif component == 'data_feed':
            self._write_log('market_data', entry)
        else:
            self._write_log('performance', entry)

    def _write_log(self, log_type: str, entry: LogEntry):
        """Write log entry to file

        Args:
            log_type: Type of log file
            entry: Log entry to write
        """
        with self._lock:
            try:
                log_line = json.dumps(asdict(entry), separators=(',', ':'))
                self._files[log_type].write(log_line + '\n')
                self.log_count += 1
            except Exception as e:
                # Fallback logging to stderr
                print(f"LOGGING ERROR: {e}", file=sys.stderr)

    def get_recent_trades(self, count: int = 10) -> list:
        """Get recent trades from buffer

        Args:
            count: Number of recent trades

        Returns:
            List of recent trade entries
        """
        if count >= len(self.trade_buffer):
            return list(self.trade_buffer)
        else:
            # Use itertools.islice to avoid creating full list
            from itertools import islice
            return list(islice(self.trade_buffer, max(0, len(self.trade_buffer) - count), None))

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from buffer

        Returns:
            Performance summary
        """
        if not self.performance_buffer:
            return {}

        latest = self.performance_buffer[-1]
        return {
            'latest_balance': latest.data['balance'],
            'latest_total_value': latest.data['total_value'],
            'latest_unrealized_pnl': latest.data['unrealized_pnl'],
            'latest_realized_pnl': latest.data['realized_pnl'],
            'position_count': latest.data['position_count'],
            'uptime_seconds': time.time() - self.start_time,
            'log_count': self.log_count
        }

    def close(self):
        """Close all log files"""
        with self._lock:
            for f in self._files.values():
                f.close()

    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Cleanup old log files to prevent disk space issues

        Args:
            days_to_keep: Number of days of logs to keep
        """
        import os
        from pathlib import Path

        cutoff_time = time.time() - (days_to_keep * 24 * 3600)

        try:
            # Clean up logs directory
            logs_dir = Path(self.config.get('storage.logs_directory', 'logs'))
            if logs_dir.exists():
                for log_file in logs_dir.glob('*.json'):
                    if log_file.stat().st_mtime < cutoff_time:
                        log_file.unlink()

                        # Log the cleanup action
                        self.log_system_event(
                            'logger', 'cleanup',
                            {'deleted_file': str(log_file), 'age_days': days_to_keep}
                        )
        except Exception as e:
            self.log_error(
                'logger', 'cleanup_error',
                f"Failed to cleanup old logs: {e}",
                exception=e
            )


# Module-level convenience functions
def create_session_logger(session_id: str = None) -> TradeLogger:
    """Create logger with session ID

    Args:
        session_id: Session ID (auto-generated if None)

    Returns:
        TradeLogger instance
    """
    if session_id is None:
        session_id = f"session_{int(time.time())}"

    return TradeLogger(session_id)


# Global logger instance for convenience
_logger = None

def get_logger() -> TradeLogger:
    """Get global logger instance"""
    global _logger
    if _logger is None:
        _logger = create_session_logger()
    return _logger