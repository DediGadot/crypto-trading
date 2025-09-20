"""Enhanced error handling framework for robust system operation."""

import functools
import logging
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from collections import defaultdict, deque


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    NETWORK = "network"
    DATABASE = "database"
    COMPUTATION = "computation"
    MEMORY = "memory"
    CONFIGURATION = "configuration"
    EXTERNAL_API = "external_api"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


@dataclass
class ErrorEvent:
    """Represents an error event with context."""
    timestamp: datetime
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"    # Normal operation
    OPEN = "open"        # Failing, blocking calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    success_threshold: int = 3
    timeout: float = 10.0


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.config.recovery_timeout

    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = 0

    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
            elif (self.state == CircuitBreakerState.CLOSED and
                  self.failure_count >= self.config.failure_threshold):
                self.state = CircuitBreakerState.OPEN


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class RetryConfig:
    """Configuration for retry logic."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter


class ErrorHandler:
    """Centralized error handling system."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_history: deque = deque(maxlen=1000)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = defaultdict(list)
        self._lock = threading.Lock()

        # Setup default recovery strategies
        self._setup_default_recovery_strategies()

    def _setup_default_recovery_strategies(self):
        """Setup default recovery strategies for different error categories."""
        self.recovery_strategies[ErrorCategory.NETWORK].extend([
            self._retry_with_backoff,
            self._switch_to_fallback_endpoint
        ])

        self.recovery_strategies[ErrorCategory.DATABASE].extend([
            self._retry_database_operation,
            self._clear_connection_pool
        ])

        self.recovery_strategies[ErrorCategory.MEMORY].extend([
            self._garbage_collect,
            self._reduce_memory_usage
        ])

    def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        attempt_recovery: bool = True
    ) -> ErrorEvent:
        """Handle an error with logging, tracking, and optional recovery."""
        context = context or {}

        # Create error event
        error_event = ErrorEvent(
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            category=category,
            context=context,
            stack_trace=traceback.format_exc()
        )

        # Log error
        self._log_error(error_event)

        # Track error
        with self._lock:
            self.error_history.append(error_event)
            self.error_counts[error_event.error_type] += 1

        # Attempt recovery if requested
        if attempt_recovery:
            error_event.recovery_attempted = True
            error_event.recovery_successful = self._attempt_recovery(error_event)

        return error_event

    def _log_error(self, error_event: ErrorEvent):
        """Log error with appropriate level."""
        log_msg = f"[{error_event.category.value}] {error_event.error_type}: {error_event.error_message}"

        if error_event.context:
            log_msg += f" | Context: {error_event.context}"

        if error_event.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_msg)
        elif error_event.severity == ErrorSeverity.HIGH:
            self.logger.error(log_msg)
        elif error_event.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_msg)
        else:
            self.logger.info(log_msg)

    def _attempt_recovery(self, error_event: ErrorEvent) -> bool:
        """Attempt to recover from error using registered strategies."""
        recovery_functions = self.recovery_strategies.get(error_event.category, [])

        for recovery_func in recovery_functions:
            try:
                if recovery_func(error_event):
                    self.logger.info(f"Recovery successful using {recovery_func.__name__}")
                    return True
            except Exception as e:
                self.logger.warning(f"Recovery attempt failed: {recovery_func.__name__}: {e}")

        return False

    def _retry_with_backoff(self, error_event: ErrorEvent) -> bool:
        """Generic retry with exponential backoff."""
        # This is a placeholder - actual implementation would depend on context
        return False

    def _switch_to_fallback_endpoint(self, error_event: ErrorEvent) -> bool:
        """Switch to fallback endpoint for network errors."""
        # Placeholder for fallback endpoint switching
        return False

    def _retry_database_operation(self, error_event: ErrorEvent) -> bool:
        """Retry database operations."""
        # Placeholder for database retry logic
        return False

    def _clear_connection_pool(self, error_event: ErrorEvent) -> bool:
        """Clear database connection pool."""
        # Placeholder for connection pool clearing
        return False

    def _garbage_collect(self, error_event: ErrorEvent) -> bool:
        """Force garbage collection for memory errors."""
        import gc
        gc.collect()
        return True

    def _reduce_memory_usage(self, error_event: ErrorEvent) -> bool:
        """Reduce memory usage by clearing caches."""
        # Placeholder for memory reduction strategies
        return False

    def create_circuit_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Create and register a circuit breaker."""
        config = config or CircuitBreakerConfig()
        circuit_breaker = CircuitBreaker(name, config)
        self.circuit_breakers[name] = circuit_breaker
        return circuit_breaker

    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self.circuit_breakers.get(name)

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        with self._lock:
            total_errors = len(self.error_history)
            recent_errors = [
                e for e in self.error_history
                if e.timestamp > datetime.now() - timedelta(hours=1)
            ]

            return {
                "total_errors": total_errors,
                "recent_errors_1h": len(recent_errors),
                "error_counts": dict(self.error_counts),
                "circuit_breaker_states": {
                    name: cb.state.value
                    for name, cb in self.circuit_breakers.items()
                },
                "most_common_errors": sorted(
                    self.error_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            }


# Global error handler instance
_error_handler = None
_error_handler_lock = threading.Lock()


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    global _error_handler
    if _error_handler is None:
        with _error_handler_lock:
            if _error_handler is None:
                _error_handler = ErrorHandler()
    return _error_handler


def handle_errors(
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    context: Dict[str, Any] = None,
    reraise: bool = True
):
    """Decorator for automatic error handling."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler = get_error_handler()
                func_context = context or {}
                func_context.update({
                    'function': func.__name__,
                    'module': func.__module__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                })

                error_handler.handle_error(
                    error=e,
                    context=func_context,
                    severity=severity,
                    category=category
                )

                if reraise:
                    raise
                return None
        return wrapper
    return decorator


def with_circuit_breaker(breaker_name: str, config: CircuitBreakerConfig = None):
    """Decorator for circuit breaker protection."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = get_error_handler()
            circuit_breaker = error_handler.get_circuit_breaker(breaker_name)

            if circuit_breaker is None:
                circuit_breaker = error_handler.create_circuit_breaker(breaker_name, config)

            return circuit_breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def error_context(
    category: ErrorCategory,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    context: Dict[str, Any] = None
):
    """Context manager for error handling."""
    try:
        yield
    except Exception as e:
        error_handler = get_error_handler()
        error_handler.handle_error(
            error=e,
            context=context,
            severity=severity,
            category=category
        )
        raise


def retry_on_error(
    config: RetryConfig = None,
    exceptions: Union[Type[Exception], tuple] = Exception
):
    """Decorator for retrying operations on error."""
    config = config or RetryConfig()

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt < config.max_attempts - 1:
                        # Calculate delay
                        delay = min(
                            config.base_delay * (config.exponential_base ** attempt),
                            config.max_delay
                        )

                        if config.jitter:
                            import random
                            delay *= (0.5 + random.random() * 0.5)

                        time.sleep(delay)

            # If we get here, all attempts failed
            raise last_exception
        return wrapper
    return decorator


__all__ = [
    'ErrorSeverity',
    'ErrorCategory',
    'ErrorEvent',
    'CircuitBreaker',
    'CircuitBreakerConfig',
    'CircuitBreakerOpenError',
    'RetryConfig',
    'ErrorHandler',
    'get_error_handler',
    'handle_errors',
    'with_circuit_breaker',
    'error_context',
    'retry_on_error'
]