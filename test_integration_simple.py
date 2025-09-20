#!/usr/bin/env python3
"""Simplified integration tests for SAFLA trading system."""

import os
import tempfile
import threading
import time
import torch

from safla_trading.config import Config
from safla_trading.core.neural_coordinator import NeuralCoordinator
from safla_trading.core.self_improvement import SelfImprovementEngine
from safla_trading.core.error_handling import (
    ErrorHandler, ErrorSeverity, ErrorCategory, CircuitBreaker, CircuitBreakerConfig
)
from safla_trading.memory import EpisodicMemory, VectorMemory, EpisodicMemoryEntry, VectorMemoryEntry


def test_basic_coordinator_functionality():
    """Test basic neural coordinator functionality."""
    print("Testing basic coordinator functionality...")

    config = Config()
    coordinator = NeuralCoordinator(
        input_dim=config.neural_input_dim,
        output_dim=config.neural_output_dim
    )

    # Test prediction
    input_tensor = torch.randn(1, config.neural_input_dim)
    result, aux = coordinator.predict(input_tensor)

    assert result is not None
    assert aux is not None
    assert result.shape[0] == 1
    assert result.shape[1] == config.neural_output_dim

    print("✅ Basic coordinator functionality working")


def test_memory_operations():
    """Test memory system operations."""
    print("\nTesting memory operations...")

    # Test episodic memory
    episodic = EpisodicMemory()
    entry = EpisodicMemoryEntry(
        context={'action': 'test', 'price': 50000},
        action='execute_trade',
        outcome={'success': True, 'profit': 100}
    )

    entry_id = episodic.store(entry)
    retrieved = episodic.retrieve(entry_id)

    assert retrieved is not None
    assert retrieved.action == 'execute_trade'
    assert retrieved.outcome['profit'] == 100

    # Test vector memory
    vector = VectorMemory()
    vector_entry = VectorMemoryEntry(
        vector=[1.0, 0.0, 0.0],
        metadata={'type': 'pattern', 'id': 1}
    )

    vector_id = vector.store(vector_entry)
    similar = vector.search([0.9, 0.1, 0.0], limit=1)

    assert len(similar) > 0

    print("✅ Memory operations working")


def test_error_handling():
    """Test error handling functionality."""
    print("\nTesting error handling...")

    # Test error handler
    handler = ErrorHandler()

    try:
        raise ValueError("Test error")
    except Exception as e:
        event = handler.handle_error(
            e,
            context={'test': 'context'},
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.COMPUTATION,
            attempt_recovery=False
        )

    assert event.error_type == "ValueError"
    assert event.error_message == "Test error"

    # Test circuit breaker
    breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=2))

    # Trip the breaker
    for _ in range(2):
        try:
            breaker.call(lambda: exec('raise ValueError("test")'))
        except ValueError:
            pass

    assert breaker.state.name == "OPEN"

    print("✅ Error handling working")


def test_self_improvement():
    """Test self-improvement engine."""
    print("\nTesting self-improvement...")

    engine = SelfImprovementEngine()

    system_state = {
        'performance_metrics': {
            'accuracy': 0.4,
            'loss': 3.0,
            'loss_volatility': 0.8,
            'train_val_gap': 0.2
        },
        'neural_health': {
            'recent_performance': [0.4] * 20
        },
        'training_history': {
            'losses': [2.0, 2.5, 3.0, 3.5, 4.0]
        },
        'strategy_metrics': {
            'win_rate': 0.4
        }
    }

    candidates = engine.analyze_improvement_opportunities(system_state)
    assert len(candidates) > 0

    print("✅ Self-improvement working")


def test_configuration():
    """Test configuration system."""
    print("\nTesting configuration...")

    config = Config()

    # Test basic config access
    assert config.system_name is not None
    assert config.initial_balance > 0
    assert config.neural_input_dim > 0
    assert config.neural_output_dim > 0

    # Test neural config
    assert config.neural_accuracy_threshold_low == 0.3
    assert config.neural_loss_threshold_high == 5.0

    print("✅ Configuration working")


def test_concurrent_memory_access():
    """Test concurrent memory access."""
    print("\nTesting concurrent memory access...")

    memory = EpisodicMemory()
    results = []
    errors = []

    def worker(worker_id):
        try:
            for i in range(10):
                entry = EpisodicMemoryEntry(
                    context={'worker': worker_id, 'iteration': i},
                    action=f'action_{worker_id}_{i}',
                    outcome={'success': True}
                )
                entry_id = memory.store(entry)
                retrieved = memory.retrieve(entry_id)
                results.append((worker_id, i, retrieved is not None))
        except Exception as e:
            errors.append((worker_id, str(e)))

    threads = []
    for i in range(3):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    assert len(errors) == 0, f"Concurrent errors: {errors}"
    assert len(results) == 30  # 3 workers * 10 iterations

    print("✅ Concurrent memory access working")


def test_persistence_with_errors():
    """Test memory persistence with error conditions."""
    print("\nTesting persistence with errors...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with valid directory
        memory = EpisodicMemory()
        memory._use_persistence = True
        memory._db_path = os.path.join(temp_dir, "test.db")

        entry = EpisodicMemoryEntry(
            context={'test': 'persistence'},
            action='test_action',
            outcome={'result': 'success'}
        )

        entry_id = memory.store(entry)
        retrieved = memory.retrieve(entry_id)

        assert retrieved is not None
        assert retrieved.action == 'test_action'

    # Test with invalid directory (should gracefully degrade)
    memory2 = EpisodicMemory()
    memory2._use_persistence = True
    memory2._db_path = "/tmp/nonexistent_dir/test.db"

    entry2 = EpisodicMemoryEntry(
        context={'test': 'fallback'},
        action='fallback_action',
        outcome={'result': 'success'}
    )

    # Should still work in-memory
    entry_id2 = memory2.store(entry2)
    retrieved2 = memory2.retrieve(entry_id2)

    assert retrieved2 is not None
    assert retrieved2.action == 'fallback_action'

    print("✅ Persistence with errors working")


if __name__ == "__main__":
    print("Running simplified integration tests...\n")

    try:
        test_basic_coordinator_functionality()
        test_memory_operations()
        test_error_handling()
        test_self_improvement()
        test_configuration()
        test_concurrent_memory_access()
        test_persistence_with_errors()

        print("\n🎯 Integration Test Summary:")
        print("   ✅ Basic coordinator: Neural prediction working")
        print("   ✅ Memory operations: Storage and retrieval working")
        print("   ✅ Error handling: Circuit breakers and error tracking working")
        print("   ✅ Self-improvement: Analysis and candidate generation working")
        print("   ✅ Configuration: YAML config loading working")
        print("   ✅ Concurrent access: Thread-safe memory operations working")
        print("   ✅ Error resilience: Graceful degradation working")

        print("\n✅ All integration tests passed!")

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()