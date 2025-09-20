#!/usr/bin/env python3
"""Comprehensive unit tests for SAFLA trading system components."""

import json
import os
import tempfile
import threading
import time
from unittest.mock import Mock, patch, MagicMock
import pytest

from safla_trading.config import Config
from safla_trading.core.neural_coordinator import NeuralCoordinator
from safla_trading.core.safla_neural import SAFLANeuralNetwork
from safla_trading.core.self_improvement import (
    SelfImprovementEngine, ImprovementType, ImprovementCandidate
)
from safla_trading.core.error_handling import (
    ErrorHandler, ErrorSeverity, ErrorCategory, CircuitBreaker,
    CircuitBreakerConfig, handle_errors, retry_on_error, RetryConfig
)
from safla_trading.memory import EpisodicMemory, VectorMemory, EpisodicMemoryEntry, VectorMemoryEntry


class TestSAFLANeuralNetwork:
    """Unit tests for SaflaNeuralNetwork."""

    def test_initialization(self):
        """Test neural network initialization."""
        config = Config()
        network = SAFLANeuralNetwork(
            input_dim=10,
            hidden_dim=64,
            output_dim=3
        )

        assert network.input_dim == 10
        assert network.hidden_dim == 64
        assert network.output_dim == 3
        assert len(network.layers) > 0

    def test_forward_pass(self):
        """Test forward pass through network."""
        config = Config()
        network = SAFLANeuralNetwork(
            input_dim=5,
            hidden_dim=10,
            output_dim=2
        )

        import torch
        input_tensor = torch.randn(1, 5)
        output = network(input_tensor)

        assert output.shape == (1, 2)
        assert not torch.isnan(output).any()

    def test_add_hidden_layer(self):
        """Test adding hidden layers."""
        config = Config()
        network = SAFLANeuralNetwork(
            input_dim=5,
            hidden_dim=10,
            output_dim=2
        )

        original_layer_count = len(network.layers)
        network.add_hidden_layer(15)
        new_layer_count = len(network.layers)

        assert new_layer_count > original_layer_count

    def test_layer_statistics(self):
        """Test layer statistics collection."""
        config = Config()
        network = SAFLANeuralNetwork(
            input_dim=5,
            hidden_dims=[10, 8],
            output_dim=2,
            config=config
        )

        stats = network.get_layer_stats()
        assert 'total_layers' in stats
        assert 'total_parameters' in stats
        assert stats['total_layers'] > 0
        assert stats['total_parameters'] > 0


class TestNeuralCoordinator:
    """Unit tests for NeuralCoordinator."""

    def test_initialization(self):
        """Test coordinator initialization."""
        config = Config()
        coordinator = NeuralCoordinator(
            input_dim=config.neural_input_dim,
            output_dim=config.neural_output_dim
        )

        assert coordinator.config == config
        assert coordinator.coordination_state is not None
        assert hasattr(coordinator, '_state_lock')

    def test_thread_safe_state_access(self):
        """Test thread-safe state access methods."""
        config = Config()
        coordinator = NeuralCoordinator(
            input_dim=config.neural_input_dim,
            output_dim=config.neural_output_dim
        )

        # Test setting and getting state
        coordinator._set_state_safe('test_key', 'test_value')
        value = coordinator._get_state_safe('test_key')

        assert value == 'test_value'

        # Test default value
        default_value = coordinator._get_state_safe('nonexistent_key', 'default')
        assert default_value == 'default'

    def test_market_data_processing(self):
        """Test market data processing."""
        config = Config()
        coordinator = NeuralCoordinator(
            input_dim=config.neural_input_dim,
            output_dim=config.neural_output_dim
        )

        market_data = {
            'price': 50000.0,
            'volume': 1500.0,
            'timestamp': time.time()
        }

        # Should not raise exception
        result = coordinator.coordinate_step(market_data)
        assert result is not None

    def test_status_reporting(self):
        """Test status reporting."""
        config = Config()
        coordinator = NeuralCoordinator(
            input_dim=config.neural_input_dim,
            output_dim=config.neural_output_dim
        )

        status = coordinator.get_status()

        assert 'active' in status
        assert 'neural_health' in status
        assert 'coordination_metrics' in status
        assert isinstance(status['active'], bool)


class TestSelfImprovementEngine:
    """Unit tests for SelfImprovementEngine."""

    def test_initialization(self):
        """Test engine initialization."""
        engine = SelfImprovementEngine()

        assert engine.improvement_candidates == []
        assert engine.improvement_results == []

    def test_improvement_analysis(self):
        """Test improvement opportunity analysis."""
        engine = SelfImprovementEngine()

        # Test with poor performance metrics
        system_state = {
            'performance_metrics': {
                'accuracy': 0.4,  # Low
                'loss': 3.0,      # High
                'loss_volatility': 0.6,
                'train_val_gap': 0.18  # Overfitting
            },
            'neural_health': {
                'recent_performance': [0.39] * 15  # Plateau
            },
            'training_history': {
                'losses': [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8]  # Increasing
            },
            'strategy_metrics': {
                'win_rate': 0.42  # Low
            }
        }

        candidates = engine.analyze_improvement_opportunities(system_state)

        assert len(candidates) > 0
        # Should identify learning rate issue
        assert any('learning rate' in c.description.lower() for c in candidates)
        # Should identify overfitting
        assert any('regularization' in c.description.lower() for c in candidates)

    def test_improvement_implementation(self):
        """Test improvement implementation."""
        engine = SelfImprovementEngine()

        candidate = ImprovementCandidate(
            improvement_type=ImprovementType.PARAMETER_OPTIMIZATION,
            description="Test optimization",
            impact_estimate=0.1,
            confidence=0.8,
            parameters={'parameter': 'learning_rate', 'action': 'reduce', 'factor': 0.5}
        )

        # Mock system interfaces
        system_interfaces = {
            'measure_performance': Mock(side_effect=[0.5, 0.6]),  # Before and after
            'optimize_parameter': Mock(),
            'create_rollback_data': Mock(return_value={'test': 'data'})
        }

        result = engine.implement_improvement(candidate, system_interfaces)

        assert result.success
        assert result.actual_impact > 0
        system_interfaces['optimize_parameter'].assert_called_once()

    def test_evaluation_metrics(self):
        """Test improvement evaluation."""
        engine = SelfImprovementEngine()

        # Add some mock results
        from safla_trading.core.self_improvement import ImprovementResult
        engine.improvement_results = [
            ImprovementResult(
                candidate=Mock(),
                success=True,
                actual_impact=0.1,
                implementation_time=1.0
            ),
            ImprovementResult(
                candidate=Mock(),
                success=False,
                actual_impact=0.0,
                implementation_time=0.5
            )
        ]

        evaluation = engine.evaluate_improvements()

        assert evaluation['total_improvements'] == 2
        assert evaluation['successful_improvements'] == 1
        assert evaluation['average_impact'] == 0.05


class TestErrorHandling:
    """Unit tests for error handling framework."""

    def test_error_handler_initialization(self):
        """Test error handler initialization."""
        handler = ErrorHandler()

        assert handler.error_history is not None
        assert handler.error_counts is not None
        assert handler.circuit_breakers is not None

    def test_error_recording(self):
        """Test error recording and tracking."""
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
        assert event.severity == ErrorSeverity.HIGH
        assert event.category == ErrorCategory.COMPUTATION

        # Check stats
        stats = handler.get_error_stats()
        assert stats['total_errors'] == 1
        assert 'ValueError' in stats['error_counts']

    def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.1)
        breaker = CircuitBreaker("test", config)

        # Test successful calls
        result = breaker.call(lambda: "success")
        assert result == "success"

        # Test failures
        for _ in range(2):
            try:
                breaker.call(lambda: exec('raise ValueError("test")'))
            except ValueError:
                pass

        # Should be open now
        assert breaker.state.name == "OPEN"

        # Test recovery after timeout
        time.sleep(0.2)
        breaker.call(lambda: "recovered")
        assert breaker.state.name == "CLOSED"

    def test_error_decorators(self):
        """Test error handling decorators."""
        call_count = [0]

        @handle_errors(severity=ErrorSeverity.LOW, reraise=False)
        def test_function():
            call_count[0] += 1
            raise RuntimeError("Test error")

        result = test_function()
        assert result is None
        assert call_count[0] == 1

    def test_retry_decorator(self):
        """Test retry decorator."""
        attempt_count = [0]

        @retry_on_error(
            config=RetryConfig(max_attempts=3, base_delay=0.01),
            exceptions=ValueError
        )
        def flaky_function():
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert attempt_count[0] == 3


class TestMemorySystem:
    """Unit tests for memory systems."""

    def test_episodic_memory_basic_operations(self):
        """Test basic episodic memory operations."""
        memory = EpisodicMemory()

        entry = EpisodicMemoryEntry(
            context={'action': 'buy', 'price': 50000},
            action='execute_trade',
            outcome={'profit': 100, 'success': True}
        )

        # Store and retrieve
        entry_id = memory.store(entry)
        retrieved = memory.retrieve(entry_id)

        assert retrieved is not None
        assert retrieved.action == 'execute_trade'
        assert retrieved.outcome['profit'] == 100

    def test_vector_memory_similarity_search(self):
        """Test vector memory similarity search."""
        memory = VectorMemory()

        # Store several vectors
        vectors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.9, 0.1, 0.0]  # Similar to first
        ]

        for i, vec in enumerate(vectors):
            entry = VectorMemoryEntry(
                vector=vec,
                metadata={'index': i, 'type': f'pattern_{i}'}
            )
            memory.store(entry)

        # Search for similar to first vector
        similar = memory.search([0.95, 0.05, 0.0], limit=2)

        assert len(similar) > 0
        # similar returns tuples of (entry, score)
        assert all(len(result) == 2 for result in similar)

    def test_memory_persistence_error_handling(self):
        """Test memory persistence with error conditions."""
        memory = EpisodicMemory()
        memory._use_persistence = True
        memory._db_path = "/invalid/nonexistent/path/test.db"

        # Should still work in-memory despite invalid path
        entry = EpisodicMemoryEntry(
            context={'test': 'data'},
            action='test_action',
            outcome={'result': 'success'}
        )

        entry_id = memory.store(entry)
        retrieved = memory.retrieve(entry_id)

        assert retrieved is not None
        assert retrieved.action == 'test_action'


class TestConfiguration:
    """Unit tests for configuration system."""

    def test_config_loading(self):
        """Test configuration loading."""
        config = Config()

        # Test that config loads properly
        assert config.system_name is not None
        assert config.initial_balance > 0
        assert len(config.symbols) > 0

    def test_config_values(self):
        """Test specific configuration values."""
        config = Config()

        # Test neural config
        assert config.neural_accuracy_threshold_low == 0.3
        assert config.neural_loss_threshold_high == 5.0
        assert config.performance_window_size == 100

        # Test feedback config
        assert config.feedback_cycle_timeout > 0
        assert config.feedback_error_threshold > 0

        # Test improvement config
        assert config.improvement_min_impact > 0
        assert config.improvement_min_confidence > 0


def run_unit_tests():
    """Run all unit tests."""
    print("Running comprehensive unit tests...\n")

    test_classes = [
        TestSAFLANeuralNetwork,
        TestNeuralCoordinator,
        TestSelfImprovementEngine,
        TestErrorHandling,
        TestMemorySystem,
        TestConfiguration
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"Testing {test_class.__name__}...")
        instance = test_class()

        # Get all test methods
        test_methods = [method for method in dir(instance) if method.startswith('test_')]

        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(instance, method_name)
                method()
                passed_tests += 1
                print(f"  ✅ {method_name}")
            except Exception as e:
                print(f"  ❌ {method_name}: {e}")

    print(f"\n🎯 Unit Test Summary:")
    print(f"   Tests run: {total_tests}")
    print(f"   Tests passed: {passed_tests}")
    print(f"   Tests failed: {total_tests - passed_tests}")
    print(f"   Success rate: {(passed_tests/total_tests)*100:.1f}%")

    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_unit_tests()
    if success:
        print("\n✅ All unit tests passed!")
    else:
        print("\n❌ Some unit tests failed!")