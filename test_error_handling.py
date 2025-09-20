#!/usr/bin/env python3
"""Test script to verify enhanced error handling functionality."""

import time
from unittest.mock import Mock, patch

from safla_trading.core.error_handling import (
    ErrorHandler, ErrorSeverity, ErrorCategory,
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpenError,
    handle_errors, with_circuit_breaker, retry_on_error,
    RetryConfig
)

def test_error_handler():
    """Test basic error handler functionality."""
    print("Testing error handler...")

    error_handler = ErrorHandler()

    # Test error handling
    try:
        raise ValueError("Test error")
    except Exception as e:
        error_event = error_handler.handle_error(
            e,
            context={'test': 'context'},
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.COMPUTATION
        )

    # Verify error was recorded
    assert error_event.error_type == "ValueError"
    assert error_event.error_message == "Test error"
    assert error_event.severity == ErrorSeverity.HIGH
    assert error_event.category == ErrorCategory.COMPUTATION

    # Test error stats
    stats = error_handler.get_error_stats()
    assert stats['total_errors'] == 1
    assert 'ValueError' in stats['error_counts']

    print("✅ Error handler working correctly")

def test_circuit_breaker():
    """Test circuit breaker functionality."""
    print("\nTesting circuit breaker...")

    config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1.0)
    circuit_breaker = CircuitBreaker("test_breaker", config)

    # Test successful calls
    result = circuit_breaker.call(lambda: "success")
    assert result == "success"

    # Test failures
    failure_count = 0
    for i in range(2):  # Only need 2 failures to trip the circuit breaker
        try:
            def failing_func():
                raise ValueError("test failure")
            circuit_breaker.call(failing_func)
        except ValueError:
            failure_count += 1

    # Circuit breaker should be open now
    assert circuit_breaker.state.name == "OPEN"

    # Test that calls are blocked
    try:
        circuit_breaker.call(lambda: "should not execute")
        assert False, "Should have raised CircuitBreakerOpenError"
    except CircuitBreakerOpenError:
        pass

    print("✅ Circuit breaker working correctly")

def test_decorators():
    """Test error handling decorators."""
    print("\nTesting error handling decorators...")

    call_count = [0]

    @handle_errors(severity=ErrorSeverity.MEDIUM, category=ErrorCategory.COMPUTATION, reraise=False)
    def failing_function():
        call_count[0] += 1
        raise RuntimeError("Decorator test error")

    # Call should handle error and return None
    result = failing_function()
    assert result is None
    assert call_count[0] == 1

    print("✅ Error handling decorators working correctly")

def test_retry_mechanism():
    """Test retry mechanism."""
    print("\nTesting retry mechanism...")

    attempt_count = [0]

    @retry_on_error(
        config=RetryConfig(max_attempts=3, base_delay=0.1),
        exceptions=ValueError
    )
    def flaky_function():
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            raise ValueError("Temporary failure")
        return "success"

    # Should succeed on third attempt
    result = flaky_function()
    assert result == "success"
    assert attempt_count[0] == 3

    print("✅ Retry mechanism working correctly")

def test_circuit_breaker_decorator():
    """Test circuit breaker decorator."""
    print("\nTesting circuit breaker decorator...")

    error_handler = ErrorHandler()

    @with_circuit_breaker("test_decorator_breaker", CircuitBreakerConfig(failure_threshold=1))
    def decorated_function(should_fail=False):
        if should_fail:
            raise RuntimeError("Decorated function failure")
        return "success"

    # Test successful call
    result = decorated_function(False)
    assert result == "success"

    # Test failure
    try:
        decorated_function(True)
    except RuntimeError:
        pass

    # Circuit breaker should be open now, next call should fail immediately
    try:
        decorated_function(False)
        assert False, "Should have raised CircuitBreakerOpenError"
    except CircuitBreakerOpenError:
        pass

    print("✅ Circuit breaker decorator working correctly")

def test_memory_error_handling():
    """Test memory system error handling."""
    print("\nTesting memory system error handling...")

    from safla_trading.memory import EpisodicMemory, EpisodicMemoryEntry

    # Test with an intentionally bad database operation
    memory = EpisodicMemory()  # Start with in-memory only
    memory._use_persistence = True
    memory._db_path = "/tmp/nonexistent_dir/test.db"  # This will fail

    # Should still work in-memory even with invalid path
    entry = EpisodicMemoryEntry(
        context={"test": "data"},
        action="test_action",
        outcome={"success": False}
    )

    entry_id = memory.store(entry)
    retrieved = memory.retrieve(entry_id)

    assert retrieved is not None
    assert retrieved.action == "test_action"

    print("✅ Memory error handling working correctly")

if __name__ == "__main__":
    print("Testing enhanced error handling system...")

    # Run all tests
    test_error_handler()
    test_circuit_breaker()
    test_decorators()
    test_retry_mechanism()
    test_circuit_breaker_decorator()
    test_memory_error_handling()

    print("\n🎯 Summary:")
    print("   ✅ Error handler: Recording and tracking errors")
    print("   ✅ Circuit breaker: Protecting against cascading failures")
    print("   ✅ Decorators: Automatic error handling in functions")
    print("   ✅ Retry mechanism: Automatic retry with backoff")
    print("   ✅ Memory resilience: Graceful degradation on storage failures")

    print("\n✅ All error handling functionality working correctly!")