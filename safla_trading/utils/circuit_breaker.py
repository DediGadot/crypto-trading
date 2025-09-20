"""
CIRCUIT BREAKER PATTERN
Prevents cascading failures by detecting problems and stopping requests
"""

import asyncio
import time
import threading
from typing import Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5      # Failures before opening
    recovery_timeout: float = 30.0  # Seconds before trying half-open
    success_threshold: int = 3      # Successes to close from half-open
    timeout: float = 10.0          # Request timeout seconds


class CircuitBreaker:
    """
    CIRCUIT BREAKER FOR API RESILIENCE
    Fails fast when external services are down
    """

    def __init__(self, name: str, config: CircuitConfig = None):
        """Initialize circuit breaker

        Args:
            name: Circuit breaker name for logging
            config: Configuration parameters
        """
        self.name = name
        self.config = config or CircuitConfig()

        # State tracking - make thread-safe
        self._lock = threading.Lock()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0

        # Statistics
        self.total_requests = 0
        self.total_failures = 0
        self.total_timeouts = 0

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: When circuit is open
            TimeoutError: When request times out
        """
        # Thread-safe state check and update
        with self._lock:
            self.total_requests += 1

            # Check if circuit should remain open
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time < self.config.recovery_timeout:
                    raise CircuitBreakerOpenError(f"Circuit {self.name} is OPEN")
                else:
                    # Try half-open
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )

            # Handle success
            await self._on_success()
            return result

        except asyncio.TimeoutError:
            self.total_timeouts += 1
            await self._on_failure()
            raise TimeoutError(f"Circuit {self.name} request timed out")

        except Exception as e:
            await self._on_failure()
            raise

    async def _on_success(self):
        """Handle successful request"""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    # Recovered - close circuit
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = max(0, self.failure_count - 1)

    async def _on_failure(self):
        """Handle failed request"""
        with self._lock:
            self.total_failures += 1
            self.failure_count += 1
            self.last_failure_time = time.time()

            if (self.state == CircuitState.CLOSED and
                self.failure_count >= self.config.failure_threshold):
                # Open circuit
                self.state = CircuitState.OPEN
            elif self.state == CircuitState.HALF_OPEN:
                # Failed during recovery - back to open
                self.state = CircuitState.OPEN
                self.success_count = 0

    def get_stats(self) -> dict:
        """Get circuit breaker statistics

        Returns:
            Statistics dictionary
        """
        return {
            'name': self.name,
            'state': self.state.value,
            'total_requests': self.total_requests,
            'total_failures': self.total_failures,
            'total_timeouts': self.total_timeouts,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'failure_rate': self.total_failures / max(1, self.total_requests),
            'last_failure_time': self.last_failure_time,
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'recovery_timeout': self.config.recovery_timeout,
                'success_threshold': self.config.success_threshold,
                'timeout': self.config.timeout
            }
        }

    def reset(self):
        """Reset circuit breaker to closed state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreakerManager:
    """
    MANAGES MULTIPLE CIRCUIT BREAKERS
    Central registry for all circuit breakers in the system
    """

    def __init__(self):
        """Initialize circuit breaker manager"""
        self.circuits = {}

    def get_circuit(self, name: str, config: CircuitConfig = None) -> CircuitBreaker:
        """Get or create circuit breaker

        Args:
            name: Circuit name
            config: Circuit configuration

        Returns:
            Circuit breaker instance
        """
        if name not in self.circuits:
            self.circuits[name] = CircuitBreaker(name, config)
        return self.circuits[name]

    def get_all_stats(self) -> dict:
        """Get statistics for all circuits

        Returns:
            Dictionary of all circuit statistics
        """
        return {name: circuit.get_stats()
                for name, circuit in self.circuits.items()}

    def reset_all(self):
        """Reset all circuit breakers"""
        for circuit in self.circuits.values():
            circuit.reset()


# Global circuit breaker manager
circuit_manager = CircuitBreakerManager()