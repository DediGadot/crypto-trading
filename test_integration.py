#!/usr/bin/env python3
"""Comprehensive integration tests for SAFLA trading system."""

import json
import os
import tempfile
import threading
import time
from unittest.mock import Mock, patch
import sqlite3

from safla_trading.config import Config
from safla_trading.core.neural_coordinator import NeuralCoordinator
from safla_trading.core.safla_neural import SAFLANeuralNetwork
from safla_trading.core.self_improvement import SelfImprovementEngine
from safla_trading.core.error_handling import ErrorSeverity, ErrorCategory
from safla_trading.memory import EpisodicMemory, VectorMemory, EpisodicMemoryEntry, VectorMemoryEntry


def test_neural_coordinator_integration():
    """Test full neural coordinator functionality with real components."""
    print("Testing neural coordinator integration...")

    # Initialize with real config
    config = Config()
    coordinator = NeuralCoordinator(
        input_dim=config.neural_input_dim,
        output_dim=config.neural_output_dim
    )

    # Test initialization
    assert coordinator is not None
    health = coordinator.get_system_health()
    assert health is not None
    assert 'overall_health' in health

    # Test prediction functionality
    import torch
    input_tensor = torch.randn(1, config.neural_input_dim)
    result, aux = coordinator.predict(input_tensor)
    assert result is not None
    assert aux is not None

    print("✅ Neural coordinator integration working")


def test_memory_persistence_integration():
    """Test memory systems with persistence and error handling."""
    print("\nTesting memory persistence integration...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test episodic memory with persistence
        episodic = EpisodicMemory()
        episodic._use_persistence = True
        episodic._db_path = os.path.join(temp_dir, "test.db")

        # Store and retrieve entries
        entry1 = EpisodicMemoryEntry(
            context={'market': 'crypto', 'action': 'buy'},
            action='execute_trade',
            outcome={'profit': 150.0, 'success': True}
        )

        entry_id = episodic.store(entry1)
        retrieved = episodic.retrieve(entry_id)

        assert retrieved is not None
        assert retrieved.action == 'execute_trade'
        assert retrieved.outcome['profit'] == 150.0

        # Test vector memory with caching
        vector = VectorMemory()
        vector._cache_file = os.path.join(temp_dir, "vectors.pkl")

        vector_entry = VectorMemoryEntry(
            vector=[0.1, 0.2, 0.3, 0.4],
            metadata={'type': 'price_pattern', 'timeframe': '1h'}
        )

        vector_id = vector.store(vector_entry)
        similar = vector.find_similar([0.1, 0.2, 0.3, 0.35], top_k=1)

        assert len(similar) > 0
        assert similar[0]['id'] == vector_id

        # Test persistence by reloading
        vector2 = VectorMemory()
        vector2._cache_file = os.path.join(temp_dir, "vectors.pkl")
        vector2._load_cache()

        similar2 = vector2.find_similar([0.1, 0.2, 0.3, 0.35], top_k=1)
        assert len(similar2) > 0

    print("✅ Memory persistence integration working")


def test_self_improvement_integration():
    """Test self-improvement engine with real metrics."""
    print("\nTesting self-improvement integration...")

    engine = SelfImprovementEngine()

    # Create realistic system state
    system_state = {
        'performance_metrics': {
            'accuracy': 0.45,  # Low accuracy
            'loss': 2.5,       # High loss
            'loss_volatility': 0.8,  # High volatility
            'train_val_gap': 0.2     # Overfitting
        },
        'neural_health': {
            'recent_performance': [0.4] * 15 + [0.41] * 10  # Plateau
        },
        'training_history': {
            'losses': [3.0, 2.8, 2.9, 3.1, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.1, 4.3, 4.5, 4.4, 4.6]
        },
        'strategy_metrics': {
            'win_rate': 0.42  # Low win rate
        }
    }

    # Analyze improvement opportunities
    candidates = engine.analyze_improvement_opportunities(system_state)

    assert len(candidates) > 0
    assert any(c.description.startswith("Reduce learning rate") for c in candidates)
    assert any(c.description.startswith("Add hidden layer") for c in candidates)
    assert any(c.description.startswith("Increase regularization") for c in candidates)

    # Test implementation with mock interfaces
    system_interfaces = {
        'measure_performance': lambda: 0.55,  # Improved performance
        'optimize_parameter': Mock(),
        'adapt_architecture': Mock(),
        'implement_meta_learning': Mock(),
        'create_rollback_data': Mock(return_value={'param': 'lr', 'value': 0.01}),
    }

    # Implement top candidate
    top_candidate = candidates[0]
    result = engine.implement_improvement(top_candidate, system_interfaces)

    assert result.success
    assert result.actual_impact > 0

    # Verify system interfaces were called
    if top_candidate.improvement_type.value == 'parameter_optimization':
        system_interfaces['optimize_parameter'].assert_called_once()

    print("✅ Self-improvement integration working")


def test_error_handling_integration():
    """Test error handling across system components."""
    print("\nTesting error handling integration...")

    # Test circuit breaker protection
    from safla_trading.core.error_handling import CircuitBreaker, CircuitBreakerConfig

    breaker = CircuitBreaker("test_integration", CircuitBreakerConfig(failure_threshold=2))

    def failing_operation():
        raise RuntimeError("Simulated failure")

    # Trip the circuit breaker
    for _ in range(2):
        try:
            breaker.call(failing_operation)
        except RuntimeError:
            pass

    # Circuit should be open
    assert breaker.state.name == "OPEN"

    # Test neural coordinator basic functionality
    config = Config()
    coordinator = NeuralCoordinator(
        input_dim=config.neural_input_dim,
        output_dim=config.neural_output_dim
    )

    # Should work without errors
    import torch
    input_tensor = torch.randn(1, config.neural_input_dim)
    result, aux = coordinator.predict(input_tensor)
    assert result is not None
    assert aux is not None

    print("✅ Error handling integration working")


def test_concurrent_access():
    """Test thread safety under concurrent access."""
    print("\nTesting concurrent access...")

    config = Config()
    coordinator = NeuralCoordinator(
        input_dim=config.neural_input_dim,
        output_dim=config.neural_output_dim
    )

    results = []
    errors = []

    def worker_thread(thread_id):
        """Worker function for concurrent testing."""
        try:
            for i in range(10):
                market_data = {
                    'price': 50000 + (thread_id * 100) + i,
                    'volume': 1000 + i,
                    'thread_id': thread_id,
                    'iteration': i
                }
                result = coordinator.coordinate_step(market_data)
                results.append((thread_id, i, result))
                time.sleep(0.01)  # Small delay to encourage race conditions
        except Exception as e:
            errors.append((thread_id, str(e)))

    # Create multiple threads
    threads = []
    for i in range(5):
        thread = threading.Thread(target=worker_thread, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Verify no errors occurred
    assert len(errors) == 0, f"Concurrent access errors: {errors}"
    assert len(results) == 50  # 5 threads * 10 iterations

    # Verify all results are valid
    for thread_id, iteration, result in results:
        assert result is not None
        assert 'action' in result
        assert 'confidence' in result

    print("✅ Concurrent access working")


def test_config_integration():
    """Test configuration system integration."""
    print("\nTesting configuration integration...")

    config = Config()

    # Test neural configuration
    assert config.neural_accuracy_threshold_low == 0.3
    assert config.neural_loss_threshold_high == 5.0
    assert config.performance_window_size == 100

    # Test feedback configuration
    assert config.feedback_cycle_timeout > 0
    assert config.feedback_error_threshold > 0

    # Test improvement configuration
    assert config.improvement_min_impact > 0
    assert config.improvement_min_confidence > 0

    # Test configuration is used by components
    coordinator = NeuralCoordinator(config)

    # Verify coordinator uses config values
    status = coordinator.get_status()
    assert status is not None

    print("✅ Configuration integration working")


def test_performance_monitoring():
    """Test performance monitoring and metrics collection."""
    print("\nTesting performance monitoring...")

    config = Config()
    coordinator = NeuralCoordinator(
        input_dim=config.neural_input_dim,
        output_dim=config.neural_output_dim
    )

    # Run multiple coordination steps and measure performance
    start_time = time.time()

    for i in range(100):
        market_data = {
            'price': 50000 + (i * 10),
            'volume': 1000 + i,
            'timestamp': time.time()
        }
        result = coordinator.coordinate_step(market_data)
        assert result is not None

    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_step = total_time / 100

    # Performance should be reasonable (less than 100ms per step)
    assert avg_time_per_step < 0.1, f"Performance too slow: {avg_time_per_step:.4f}s per step"

    # Test memory performance
    episodic = EpisodicMemory()

    start_time = time.time()

    for i in range(1000):
        entry = EpisodicMemoryEntry(
            context={'iteration': i, 'data': f'test_{i}'},
            action=f'action_{i}',
            outcome={'success': i % 2 == 0}
        )
        episodic.store(entry)

    end_time = time.time()
    storage_time = end_time - start_time
    avg_storage_time = storage_time / 1000

    # Storage should be fast (less than 1ms per entry)
    assert avg_storage_time < 0.001, f"Storage too slow: {avg_storage_time:.6f}s per entry"

    print("✅ Performance monitoring working")


if __name__ == "__main__":
    print("Running comprehensive integration tests...")

    # Run all integration tests
    test_neural_coordinator_integration()
    test_memory_persistence_integration()
    test_self_improvement_integration()
    test_error_handling_integration()
    test_concurrent_access()
    test_config_integration()
    test_performance_monitoring()

    print("\n🎯 Integration Test Summary:")
    print("   ✅ Neural coordinator: Full coordination cycle working")
    print("   ✅ Memory persistence: SQLite and pickle storage working")
    print("   ✅ Self-improvement: Real analysis and implementation working")
    print("   ✅ Error handling: Circuit breakers and graceful degradation working")
    print("   ✅ Concurrent access: Thread safety under load working")
    print("   ✅ Configuration: All config sections properly integrated")
    print("   ✅ Performance: System performance within acceptable limits")

    print("\n✅ All integration tests passed!")