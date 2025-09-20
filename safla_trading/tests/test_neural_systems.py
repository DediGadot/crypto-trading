"""Tests for SAFLA neural systems."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from safla_trading.core import (
    SAFLANeuralNetwork, FeedbackLoopManager, SelfImprovementEngine, NeuralCoordinator
)
from safla_trading.core.feedback_loops import FeedbackSignal, FeedbackType, LearningCycle
from safla_trading.core.self_improvement import ImprovementCandidate, ImprovementType
from safla_trading.memory import MemoryManager


class TestSAFLANeuralNetwork:
    """Test SAFLA neural network."""

    def setup_method(self):
        """Set up test fixtures."""
        self.input_dim = 64
        self.output_dim = 8
        self.network = SAFLANeuralNetwork(
            input_dim=self.input_dim,
            output_dim=self.output_dim
        )

    def test_network_initialization(self):
        """Test network initialization."""
        assert self.network.input_dim == self.input_dim
        assert self.network.output_dim == self.output_dim
        assert len(self.network.layers) > 0

        # Check parameter count
        total_params = sum(p.numel() for p in self.network.parameters())
        assert total_params > 0

    def test_forward_pass(self):
        """Test forward pass through network."""
        batch_size = 32
        inputs = torch.randn(batch_size, self.input_dim)

        outputs, aux_outputs = self.network.forward(inputs)

        # Check output shape
        assert outputs.shape == (batch_size, self.output_dim)

        # Check auxiliary outputs
        assert 'confidence' in aux_outputs
        assert 'uncertainty' in aux_outputs
        assert 'meta_learning_signal' in aux_outputs

        # Check output ranges
        confidence = aux_outputs['confidence']
        assert torch.all(confidence >= 0) and torch.all(confidence <= 1)

    def test_training_step(self):
        """Test training step."""
        batch_size = 16
        batch_data = {
            'inputs': torch.randn(batch_size, self.input_dim),
            'targets': torch.randn(batch_size, self.output_dim)
        }

        # Perform training step
        metrics = self.network.train_step(batch_data)

        # Check metrics
        assert 'total_loss' in metrics
        assert 'primary_loss' in metrics
        assert 'confidence_loss' in metrics
        assert 'accuracy' in metrics

        # Check that loss is finite
        assert torch.isfinite(torch.tensor(metrics['total_loss']))

    def test_prediction_with_confidence(self):
        """Test prediction with confidence estimation."""
        inputs = torch.randn(1, self.input_dim)

        prediction, confidence = self.network.predict_with_confidence(inputs)

        # Check prediction shape
        assert prediction.shape == (1, self.output_dim)

        # Check confidence range
        assert 0 <= confidence <= 1

    def test_adaptation_to_feedback(self):
        """Test network adaptation based on feedback."""
        feedback = {
            'performance_degradation': 0.2,
            'overfitting': True
        }

        adapted = self.network.adapt_to_feedback(feedback)

        # Should trigger adaptation
        assert adapted

        # Check adaptation history
        assert len(self.network.training_history['adaptation_events']) > 0

    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        import tempfile
        import os

        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()

        try:
            # Save checkpoint
            self.network.save_checkpoint(temp_file.name)

            # Create new network and load
            new_network = SAFLANeuralNetwork(self.input_dim, self.output_dim)
            new_network.load_checkpoint(temp_file.name)

            # Compare parameters
            for p1, p2 in zip(self.network.parameters(), new_network.parameters()):
                assert torch.allclose(p1, p2)

        finally:
            os.unlink(temp_file.name)

    def test_network_state(self):
        """Test network state retrieval."""
        state = self.network.get_network_state()

        assert 'parameters' in state
        assert 'learning_rate' in state
        assert 'training_history' in state
        assert 'performance_stats' in state

        # Check parameter count is correct
        expected_params = sum(p.numel() for p in self.network.parameters())
        assert state['parameters'] == expected_params


class TestFeedbackLoopManager:
    """Test feedback loop manager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.feedback_manager = FeedbackLoopManager()

    def teardown_method(self):
        """Clean up test fixtures."""
        self.feedback_manager.shutdown()

    def test_feedback_addition(self):
        """Test adding feedback signals."""
        feedback = FeedbackSignal(
            feedback_type=FeedbackType.PERFORMANCE,
            value=0.8,
            metadata={'source': 'test'},
            confidence=0.9
        )

        initial_count = len(self.feedback_manager.feedback_buffer)
        self.feedback_manager.add_feedback(feedback)

        assert len(self.feedback_manager.feedback_buffer) == initial_count + 1
        assert self.feedback_manager.learning_metrics['feedback_processed'] > 0

    def test_learning_cycle_management(self):
        """Test learning cycle management."""
        context = {'test': 'context'}

        # Start cycle
        cycle_id = self.feedback_manager.start_learning_cycle(context)
        assert cycle_id is not None
        assert self.feedback_manager.current_cycle is not None

        # Add feedback during cycle
        feedback = FeedbackSignal(
            feedback_type=FeedbackType.SUCCESS,
            value=0.9
        )
        self.feedback_manager.add_feedback(feedback)

        # End cycle
        completed_cycle = self.feedback_manager.end_learning_cycle()
        assert completed_cycle is not None
        assert completed_cycle.cycle_id == cycle_id
        assert len(completed_cycle.feedback_signals) > 0

    def test_performance_analysis(self):
        """Test performance analysis."""
        # Add various feedback signals
        feedback_signals = [
            FeedbackSignal(FeedbackType.PERFORMANCE, 0.8),
            FeedbackSignal(FeedbackType.ERROR, 0.3),
            FeedbackSignal(FeedbackType.SUCCESS, 0.9),
        ]

        for signal in feedback_signals:
            self.feedback_manager.add_feedback(signal)

        # Analyze performance
        analysis = self.feedback_manager.performance_analyzer.analyze_performance(feedback_signals)

        assert 'trends' in analysis
        assert 'patterns' in analysis
        assert 'recommendations' in analysis

    def test_learning_insights(self):
        """Test learning insights generation."""
        # Create and complete a learning cycle
        cycle_id = self.feedback_manager.start_learning_cycle({'test': 'insights'})

        # Add feedback
        feedback = FeedbackSignal(FeedbackType.PERFORMANCE, 0.75)
        self.feedback_manager.add_feedback(feedback)

        # Complete cycle
        self.feedback_manager.end_learning_cycle()

        # Get insights
        insights = self.feedback_manager.get_learning_insights()

        assert isinstance(insights, dict)
        if insights.get('message') != 'No learning cycles available':
            assert 'cycle_summary' in insights

    def test_feedback_observers(self):
        """Test feedback observer system."""
        observed_feedback = []

        def observer(feedback):
            observed_feedback.append(feedback)

        # Add observer
        self.feedback_manager.add_feedback_observer(observer)

        # Add feedback
        feedback = FeedbackSignal(FeedbackType.PERFORMANCE, 0.7)
        self.feedback_manager.add_feedback(feedback)

        # Check observer was called
        assert len(observed_feedback) == 1
        assert observed_feedback[0] == feedback

        # Remove observer
        self.feedback_manager.remove_feedback_observer(observer)

        # Add more feedback
        self.feedback_manager.add_feedback(FeedbackSignal(FeedbackType.SUCCESS, 0.9))

        # Observer should not be called again
        assert len(observed_feedback) == 1

    def test_system_status(self):
        """Test system status reporting."""
        status = self.feedback_manager.get_system_status()

        assert 'learning_metrics' in status
        assert 'feedback_buffer_size' in status
        assert 'cycles_completed' in status
        assert 'current_cycle_active' in status


class TestSelfImprovementEngine:
    """Test self-improvement engine."""

    def setup_method(self):
        """Set up test fixtures."""
        self.improvement_engine = SelfImprovementEngine()

    def teardown_method(self):
        """Clean up test fixtures."""
        self.improvement_engine.shutdown()

    def test_improvement_opportunity_analysis(self):
        """Test improvement opportunity analysis."""
        system_state = {
            'performance_metrics': {
                'accuracy': 0.6,
                'loss_volatility': 0.4,
                'train_val_gap': 0.15
            },
            'model': Mock(spec=nn.Module),
            'strategies': [{'id': 'test_strategy', 'name': 'test'}]
        }

        candidates = self.improvement_engine.analyze_improvement_opportunities(system_state)

        assert isinstance(candidates, list)
        # Should identify some improvement opportunities
        assert len(candidates) > 0

        # Check candidate structure
        for candidate in candidates:
            assert isinstance(candidate, ImprovementCandidate)
            assert hasattr(candidate, 'improvement_type')
            assert hasattr(candidate, 'impact_estimate')
            assert hasattr(candidate, 'confidence')

    def test_improvement_implementation(self):
        """Test improvement implementation."""
        candidate = ImprovementCandidate(
            improvement_type=ImprovementType.PARAMETER_OPTIMIZATION,
            description="Test improvement",
            impact_estimate=0.1,
            confidence=0.8,
            implementation_cost=0.2,
            parameters={'parameter': 'learning_rate'}
        )

        # Mock system interface
        system_interface = {
            'optimize_parameter': Mock(return_value=None),
            'measure_performance': Mock(return_value=0.75),
            'create_rollback_data': Mock(return_value={'test': 'data'}),
            'rollback_improvement': Mock(return_value=None)
        }

        result = self.improvement_engine.implement_improvement(candidate, system_interface)

        assert result.candidate == candidate
        assert isinstance(result.success, bool)
        assert isinstance(result.actual_impact, float)

    def test_improvement_evaluation(self):
        """Test improvement evaluation."""
        # Simulate some improvement results
        candidate = ImprovementCandidate(
            improvement_type=ImprovementType.PARAMETER_OPTIMIZATION,
            description="Test improvement",
            impact_estimate=0.1,
            confidence=0.8,
            implementation_cost=0.2
        )

        # Add to results manually for testing
        from safla_trading.core.self_improvement import ImprovementResult
        result = ImprovementResult(
            candidate=candidate,
            success=True,
            actual_impact=0.12,
            implementation_time=1.5
        )
        self.improvement_engine.improvement_results.append(result)

        # Evaluate improvements
        evaluation = self.improvement_engine.evaluate_improvements()

        assert 'total_improvements' in evaluation
        assert 'successful_improvements' in evaluation
        assert 'average_impact' in evaluation

    def test_meta_insights(self):
        """Test meta-learning insights."""
        insights = self.improvement_engine.get_meta_insights()

        assert 'improvement_patterns' in insights
        assert 'success_factors' in insights
        assert 'failure_modes' in insights
        assert 'optimization_opportunities' in insights

    def test_system_status(self):
        """Test system status reporting."""
        status = self.improvement_engine.get_system_status()

        assert 'improvement_candidates' in status
        assert 'implemented_improvements' in status
        assert 'improvement_results' in status
        assert 'recent_success_rate' in status


class TestNeuralCoordinator:
    """Test neural coordinator integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.input_dim = 32
        self.output_dim = 4
        self.coordinator = NeuralCoordinator(
            input_dim=self.input_dim,
            output_dim=self.output_dim
        )

    def teardown_method(self):
        """Clean up test fixtures."""
        self.coordinator.shutdown()

    def test_coordinator_initialization(self):
        """Test coordinator initialization."""
        assert self.coordinator.neural_network is not None
        assert self.coordinator.feedback_manager is not None
        assert self.coordinator.improvement_engine is not None

        # Check system interfaces
        assert 'optimize_parameter' in self.coordinator.system_interfaces
        assert 'measure_performance' in self.coordinator.system_interfaces

    def test_coordinated_prediction(self):
        """Test coordinated prediction with all components."""
        inputs = torch.randn(1, self.input_dim)
        context = {'market_state': 'testing'}

        prediction, metadata = self.coordinator.predict(inputs, context)

        # Check prediction
        assert prediction.shape == (1, self.output_dim)

        # Check metadata
        assert 'confidence' in metadata
        assert 'uncertainty' in metadata
        assert 'neural_health' in metadata
        assert 'cycle_id' in metadata

        # Should have started a learning cycle
        assert self.coordinator.coordination_state['active_learning_cycle'] is not None

    def test_coordinated_training(self):
        """Test coordinated training step."""
        batch_data = {
            'inputs': torch.randn(8, self.input_dim),
            'targets': torch.randn(8, self.output_dim)
        }
        feedback_context = {'training': 'test'}

        metrics = self.coordinator.train_step(batch_data, feedback_context)

        # Check training metrics
        assert 'total_loss' in metrics
        assert 'accuracy' in metrics

        # Should have generated feedback
        assert len(self.coordinator.feedback_manager.feedback_buffer) > 0

    def test_feedback_provision(self):
        """Test external feedback provision."""
        self.coordinator.provide_feedback(
            feedback_type=FeedbackType.PERFORMANCE,
            value=0.85,
            metadata={'source': 'external'},
            context={'test': 'feedback'}
        )

        # Check feedback was added
        assert len(self.coordinator.feedback_manager.feedback_buffer) > 0

        # Check recent feedback
        recent_feedback = list(self.coordinator.feedback_manager.feedback_buffer)[-1]
        assert recent_feedback.value == 0.85
        assert recent_feedback.source == 'external'

    def test_analyze_and_improve(self):
        """Test analysis and improvement cycle."""
        # Add some performance data
        for _ in range(10):
            self.coordinator.coordination_state['recent_performance'].append(0.7)

        # Perform analysis and improvement
        results = self.coordinator.analyze_and_improve()

        assert 'system_state' in results
        assert 'improvement_candidates' in results
        assert 'implementations' in results

    def test_learning_insights(self):
        """Test comprehensive learning insights."""
        insights = self.coordinator.get_learning_insights()

        assert 'neural_network' in insights
        assert 'feedback_loops' in insights
        assert 'self_improvement' in insights
        assert 'coordination' in insights
        assert 'performance_summary' in insights

    def test_system_health(self):
        """Test system health monitoring."""
        health = self.coordinator.get_system_health()

        assert 'overall_status' in health
        assert 'neural_network' in health
        assert 'feedback_system' in health
        assert 'improvement_engine' in health
        assert 'coordination' in health

        # Check neural health
        neural_health = health['neural_network']
        assert 'status' in neural_health
        assert neural_health['status'] in ['healthy', 'degraded', 'critical']

    def test_immediate_adaptation(self):
        """Test immediate adaptation triggers."""
        # Simulate poor training performance
        batch_data = {
            'inputs': torch.randn(4, self.input_dim),
            'targets': torch.randn(4, self.output_dim)
        }

        # Mock poor performance
        with patch.object(self.coordinator.neural_network, 'train_step') as mock_train:
            mock_train.return_value = {
                'total_loss': 10.0,  # Very high loss
                'accuracy': 0.1,     # Very low accuracy
                'primary_loss': 9.0,
                'confidence_loss': 0.5,
                'uncertainty_loss': 0.3,
                'meta_loss': 0.2,
                'confidence': 0.3
            }

            initial_adaptations = len(self.coordinator.coordination_state['adaptation_history'])
            self.coordinator.train_step(batch_data)

            # Should trigger immediate adaptation
            assert len(self.coordinator.coordination_state['adaptation_history']) > initial_adaptations

    def test_architecture_adaptation(self):
        """Test architecture adaptation adds a layer."""
        initial_params = sum(p.numel() for p in self.coordinator.neural_network.parameters())
        adaptation_params = {'action': 'add_hidden_layer', 'layer_size': 64}
        self.coordinator._adapt_architecture(adaptation_params)
        new_params = sum(p.numel() for p in self.coordinator.neural_network.parameters())
        assert new_params > initial_params  # New layer added

    def test_performance_tracking(self):
        """Test performance tracking functionality."""
        # Add some predictions
        for i in range(5):
            inputs = torch.randn(1, self.input_dim)
            self.coordinator.predict(inputs)

        # Check performance tracker
        tracker = self.coordinator.performance_tracker
        assert len(tracker.prediction_history) > 0

        # Get current metrics
        metrics = tracker.get_current_metrics()
        assert isinstance(metrics, dict)

    def test_health_monitoring(self):
        """Test health monitoring functionality."""
        health_monitor = self.coordinator.health_monitor
        neural_health = health_monitor.check_neural_health(self.coordinator.neural_network)

        assert 'status' in neural_health
        assert 'issues' in neural_health
        assert 'parameter_stats' in neural_health


class TestIntegration:
    """Test integration between all neural components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.memory_manager = MemoryManager()
        self.coordinator = NeuralCoordinator(
            input_dim=64,
            output_dim=8,
            memory_manager=self.memory_manager
        )

    def teardown_method(self):
        """Clean up test fixtures."""
        self.coordinator.shutdown()
        self.memory_manager.shutdown()

    def test_memory_integrated_prediction(self):
        """Test prediction with memory integration."""
        # Store some context in memory
        experience = {
            "context": {"market": "bull", "strategy": "momentum"},
            "action": "buy",
            "outcome": {"success": True, "profit": 0.05}
        }
        self.memory_manager.create_comprehensive_memory(experience)

        # Make prediction with similar context
        inputs = torch.randn(1, 64)
        context = {"market": "bull", "strategy": "momentum"}

        prediction, metadata = self.coordinator.predict(inputs, context)

        # Should indicate memory context was used
        assert metadata['memory_context_used']

    def test_full_learning_cycle(self):
        """Test complete learning cycle with all components."""
        # Start with prediction
        inputs = torch.randn(2, 64)
        context = {"test": "full_cycle"}

        prediction, _ = self.coordinator.predict(inputs, context)

        # Perform training
        batch_data = {
            'inputs': inputs,
            'targets': torch.randn(2, 8)
        }
        train_metrics = self.coordinator.train_step(batch_data, context)

        # Provide external feedback
        self.coordinator.provide_feedback(
            FeedbackType.PERFORMANCE,
            0.8,
            context=context
        )

        # Analyze and improve
        improvement_results = self.coordinator.analyze_and_improve()

        # Check that all components participated
        assert train_metrics is not None
        assert improvement_results is not None

        # Check memory has stored experiences
        memory_status = self.memory_manager.get_system_status()
        assert memory_status['memory_systems']['episodic']['size'] > 0

    def test_adaptive_learning_with_memory(self):
        """Test adaptive learning with memory feedback."""
        # Create multiple learning experiences
        for i in range(5):
            # Create experience
            experience = {
                "context": {"iteration": i, "performance": 0.5 + i * 0.1},
                "action": f"action_{i}",
                "outcome": {"success": i > 2, "score": 0.5 + i * 0.1}
            }
            self.memory_manager.create_comprehensive_memory(experience)

            # Train with improving performance
            batch_data = {
                'inputs': torch.randn(4, 64),
                'targets': torch.randn(4, 8)
            }
            self.coordinator.train_step(batch_data)

            # Provide feedback
            self.coordinator.provide_feedback(
                FeedbackType.PERFORMANCE,
                0.5 + i * 0.1,
                context={"iteration": i}
            )

        # Analyze system state
        insights = self.coordinator.get_learning_insights()

        # Should show learning progress
        assert 'neural_network' in insights
        assert 'memory_integration' in insights

        # Memory should have accumulated knowledge
        memory_insights = insights['memory_integration']
        if memory_insights['status'] != 'not_available':
            assert memory_insights['memory_utilization']['memory_systems']['episodic']['size'] >= 5


if __name__ == "__main__":
    pytest.main([__file__])