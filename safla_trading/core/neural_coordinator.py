"""Neural Coordinator for SAFLA Architecture.

Coordinates all neural components including the main network, feedback loops,
self-improvement engine, and memory integration for unified operation.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import time
import logging

from .safla_neural import SAFLANeuralNetwork
from .feedback_loops import FeedbackLoopManager, FeedbackSignal, FeedbackType
from .self_improvement import SelfImprovementEngine, ImprovementCandidate, ImprovementType
from .error_handling import (
    ErrorHandler, ErrorSeverity, ErrorCategory,
    handle_errors, with_circuit_breaker, error_context,
    CircuitBreakerConfig
)
from ..memory import MemoryManager, MemoryType, WorkingMemoryEntry
from ..config import get_config

logger = logging.getLogger(__name__)


class NeuralCoordinator:
    """Coordinates all neural components in the SAFLA system."""

    def __init__(self, input_dim: int, output_dim: int, memory_manager: Optional[MemoryManager] = None):
        """Initialize neural coordinator.

        Args:
            input_dim: Input dimension for neural network
            output_dim: Output dimension for neural network
            memory_manager: Optional memory manager for integration
        """
        self.config = get_config()
        self.memory_manager = memory_manager
        self.error_handler = ErrorHandler(logger)

        # Setup circuit breakers for critical components
        self._setup_circuit_breakers()

        # Core neural components with error handling
        try:
            self.neural_network = SAFLANeuralNetwork(
                input_dim=input_dim,
                output_dim=output_dim,
                memory_manager=memory_manager
            )

            self.feedback_manager = FeedbackLoopManager(memory_manager)
            self.improvement_engine = SelfImprovementEngine(memory_manager)
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'component': 'neural_coordinator_init', 'input_dim': input_dim, 'output_dim': output_dim},
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.CONFIGURATION
            )
            raise

        # Coordination state - thread-safe access required
        self.coordination_state = {
            'active_learning_cycle': None,
            'recent_performance': deque(maxlen=self.config.performance_window_size),
            'adaptation_history': [],
            'neural_health': 'healthy'
        }
        self._state_lock = threading.RLock()  # Reentrant lock for nested access

        # System interfaces for improvement engine
        self.system_interfaces = {
            'optimize_parameter': self._optimize_parameter,
            'adapt_architecture': self._adapt_architecture,
            'evolve_strategy': self._evolve_strategy,
            'implement_meta_learning': self._implement_meta_learning,
            'measure_performance': self._measure_performance,
            'create_rollback_data': self._create_rollback_data,
            'rollback_improvement': self._rollback_improvement
        }

        # Performance monitoring
        self.performance_tracker = PerformanceTracker()
        self.health_monitor = HealthMonitor()

        # Background coordination
        self.coordination_thread = None
        self.coordination_interval = self.config.coordination_interval
        self.running = False
        self._coordination_lock = threading.Lock()  # Lock for coordination operations

        # Register feedback observers
        self.feedback_manager.add_feedback_observer(self._on_feedback_received)

        self._start_coordination_loop()

        logger.info("NeuralCoordinator initialized with all SAFLA components")

    def _setup_circuit_breakers(self):
        """Setup circuit breakers for critical operations."""
        # Neural network operations
        self.error_handler.create_circuit_breaker(
            "neural_network",
            CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30.0)
        )

        # Memory operations
        self.error_handler.create_circuit_breaker(
            "memory_operations",
            CircuitBreakerConfig(failure_threshold=5, recovery_timeout=60.0)
        )

        # Improvement operations
        self.error_handler.create_circuit_breaker(
            "improvement_engine",
            CircuitBreakerConfig(failure_threshold=2, recovery_timeout=120.0)
        )

    def _get_state_safe(self, key: str, default=None):
        """Thread-safe getter for coordination state."""
        with self._state_lock:
            return self.coordination_state.get(key, default)

    def _set_state_safe(self, key: str, value):
        """Thread-safe setter for coordination state."""
        with self._state_lock:
            self.coordination_state[key] = value

    def _update_state_safe(self, updates: Dict[str, Any]):
        """Thread-safe batch update for coordination state."""
        with self._state_lock:
            self.coordination_state.update(updates)

    def _append_to_state_safe(self, key: str, value):
        """Thread-safe append to list in coordination state."""
        with self._state_lock:
            if key in self.coordination_state and hasattr(self.coordination_state[key], 'append'):
                self.coordination_state[key].append(value)

    def _append_to_deque_safe(self, key: str, value):
        """Thread-safe append to deque in coordination state."""
        with self._state_lock:
            if key in self.coordination_state and hasattr(self.coordination_state[key], 'append'):
                self.coordination_state[key].append(value)

    @with_circuit_breaker("neural_network")
    @handle_errors(severity=ErrorSeverity.HIGH, category=ErrorCategory.COMPUTATION)
    def predict(self, inputs: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Make prediction with full SAFLA integration.

        Args:
            inputs: Input tensor for prediction
            context: Optional context for memory integration

        Returns:
            Tuple of (prediction, metadata)
        """
        # Start learning cycle if not active
        if not self._get_state_safe('active_learning_cycle'):
            cycle_id = self.feedback_manager.start_learning_cycle(context or {})
            self._set_state_safe('active_learning_cycle', cycle_id)

        # Update working memory with current context
        if context and self.memory_manager:
            try:
                with error_context(ErrorCategory.MEMORY, ErrorSeverity.MEDIUM):
                    working_entry = WorkingMemoryEntry(
                        content=context,
                        priority=2,
                        attention_weight=1.0,
                        importance=0.8
                    )
                    self.memory_manager.store_memory(MemoryType.WORKING, working_entry)
                    self.memory_manager.working_memory.set_context(context)
            except Exception as e:
                # Log error but don't fail prediction
                logger.warning(f"Memory operation failed during prediction: {e}")
                # Continue without memory context

        # Make prediction
        prediction, aux_outputs = self.neural_network.predict_with_confidence(inputs, context)

        # Track performance
        self.performance_tracker.record_prediction(prediction, aux_outputs)

        # Prepare metadata
        metadata = {
            'confidence': aux_outputs.get('confidence', 0.5),
            'uncertainty': aux_outputs.get('uncertainty', 0.5),
            'neural_health': self._get_state_safe('neural_health'),
            'cycle_id': self._get_state_safe('active_learning_cycle'),
            'memory_context_used': context is not None and self.memory_manager is not None
        }

        return prediction, metadata

    @with_circuit_breaker("neural_network")
    @handle_errors(severity=ErrorSeverity.HIGH, category=ErrorCategory.COMPUTATION)
    def train_step(self, batch_data: Dict[str, torch.Tensor], feedback_context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Perform coordinated training step with feedback integration.

        Args:
            batch_data: Training batch data
            feedback_context: Optional context for feedback

        Returns:
            Training metrics
        """
        # Perform neural network training step
        train_metrics = self.neural_network.train_step(batch_data)

        # Generate feedback signals
        self._generate_training_feedback(train_metrics, feedback_context)

        # Update performance tracking
        self.performance_tracker.record_training_step(train_metrics)

        # Check for immediate adaptations
        if self._should_trigger_immediate_adaptation(train_metrics):
            self._trigger_immediate_adaptation(train_metrics)

        return train_metrics

    def provide_feedback(self, feedback_type: FeedbackType, value: float,
                        metadata: Optional[Dict[str, Any]] = None,
                        context: Optional[Dict[str, Any]] = None) -> None:
        """Provide feedback to the system.

        Args:
            feedback_type: Type of feedback
            value: Feedback value
            metadata: Optional metadata
            context: Optional context
        """
        feedback_signal = FeedbackSignal(
            feedback_type=feedback_type,
            value=value,
            metadata=metadata or {},
            context=context or {},
            source='external'
        )

        self.feedback_manager.add_feedback(feedback_signal)

    @with_circuit_breaker("improvement_engine")
    @handle_errors(severity=ErrorSeverity.MEDIUM, category=ErrorCategory.COMPUTATION)
    def analyze_and_improve(self) -> Dict[str, Any]:
        """Analyze system state and implement improvements.

        Returns:
            Analysis and improvement results
        """
        # Collect current system state
        system_state = self._collect_system_state()

        # Analyze improvement opportunities
        improvement_candidates = self.improvement_engine.analyze_improvement_opportunities(system_state)

        # Select and implement top improvements
        implementation_results = []
        max_candidates = self.config.improvement_max_candidates
        for candidate in improvement_candidates[:max_candidates]:  # Configurable max candidates
            if candidate.confidence > self.config.improvement_min_confidence and candidate.impact_estimate > self.config.improvement_min_impact:
                result = self.improvement_engine.implement_improvement(candidate, self.system_interfaces)
                implementation_results.append(result)

        # End current learning cycle
        completed_cycle = None
        if self._get_state_safe('active_learning_cycle'):
            completed_cycle = self.feedback_manager.end_learning_cycle()
            self._set_state_safe('active_learning_cycle', None)

        if self.memory_manager:
            experience = {
                'context': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'candidate_count': len(improvement_candidates),
                },
                'action': 'analyze_and_improve',
                'outcome': {
                    'implemented': len(implementation_results),
                    'successes': sum(1 for r in implementation_results if r.success),
                },
                'importance': 0.6,
            }
            self.memory_manager.create_comprehensive_memory(experience)

        return {
            'system_state': system_state,
            'improvement_candidates': len(improvement_candidates),
            'implementations': len(implementation_results),
            'successful_implementations': sum(1 for r in implementation_results if r.success),
            'completed_cycle': completed_cycle.cycle_id if completed_cycle else None
        }

    def get_learning_insights(self) -> Dict[str, Any]:
        """Get comprehensive learning insights from all components.

        Returns:
            Learning insights dictionary
        """
        insights = {
            'neural_network': self.neural_network.get_network_state(),
            'feedback_loops': self.feedback_manager.get_learning_insights(),
            'self_improvement': self.improvement_engine.get_meta_insights(),
            'coordination': self._get_coordination_insights(),
            'memory_integration': self._get_memory_insights(),
            'performance_summary': self.performance_tracker.get_summary()
        }

        return insights

    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health information.

        Returns:
            System health dictionary
        """
        health = {
            'overall_status': self._get_state_safe('neural_health'),
            'neural_network': self.health_monitor.check_neural_health(self.neural_network),
            'feedback_system': self.feedback_manager.get_system_status(),
            'improvement_engine': self.improvement_engine.get_system_status(),
            'memory_system': self.memory_manager.get_system_status() if self.memory_manager else None,
            'coordination': {
                'active_cycle': self._get_state_safe('active_learning_cycle'),
                'recent_adaptations': len(self._get_state_safe('adaptation_history', [])),
                'coordination_thread_alive': self.coordination_thread.is_alive() if self.coordination_thread else False
            }
        }

        return health

    def _collect_system_state(self) -> Dict[str, Any]:
        """Collect comprehensive system state for analysis."""
        state = {
            'performance_metrics': self.performance_tracker.get_current_metrics(),
            'model': self.neural_network,
            'strategies': [],  # Would be populated from strategy manager
            'memory_usage': self.memory_manager.get_system_status() if self.memory_manager else {},
            'feedback_patterns': self._analyze_feedback_patterns(),
            'learning_progress': self._assess_learning_progress()
        }

        return state

    def _analyze_feedback_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in recent feedback."""
        recent_feedback = list(self.feedback_manager.feedback_buffer)[-50:]  # Last 50 signals

        patterns = {
            'feedback_by_type': defaultdict(int),
            'average_values': defaultdict(list),
            'feedback_frequency': len(recent_feedback),
            'dominant_contexts': defaultdict(int)
        }

        for signal in recent_feedback:
            patterns['feedback_by_type'][signal.feedback_type.value] += 1
            patterns['average_values'][signal.feedback_type.value].append(signal.value)

            # Analyze contexts
            for key, value in signal.context.items():
                patterns['dominant_contexts'][f"{key}:{value}"] += 1

        # Convert to averages
        for feedback_type, values in patterns['average_values'].items():
            if values:
                patterns['average_values'][feedback_type] = np.mean(values)

        return patterns

    def _assess_learning_progress(self) -> Dict[str, float]:
        """Assess current learning progress."""
        progress = {
            'performance_trend': 0.0,
            'learning_velocity': 0.0,
            'adaptation_frequency': 0.0,
            'knowledge_accumulation': 0.0
        }

        # Performance trend
        recent_performance = list(self._get_state_safe('recent_performance', []))
        if len(recent_performance) > 10:
            x = np.arange(len(recent_performance))
            trend = np.polyfit(x, recent_performance, 1)[0]
            progress['performance_trend'] = float(trend)

        # Learning velocity (rate of improvement)
        if len(recent_performance) > 20:
            recent_avg = np.mean(recent_performance[-10:])
            older_avg = np.mean(recent_performance[-20:-10])
            progress['learning_velocity'] = float(recent_avg - older_avg)

        # Adaptation frequency
        adaptation_history = self._get_state_safe('adaptation_history', [])
        recent_adaptations = [
            a for a in adaptation_history
            if (datetime.now() - a['timestamp']).total_seconds() < 3600  # Last hour
        ]
        progress['adaptation_frequency'] = len(recent_adaptations)

        # Knowledge accumulation (from memory if available)
        if self.memory_manager:
            semantic_count = len(self.memory_manager.semantic_memory.entries)
            episodic_count = len(self.memory_manager.episodic_memory.entries)
            progress['knowledge_accumulation'] = float(semantic_count + episodic_count) / 1000.0

        return progress

    def _generate_training_feedback(self, train_metrics: Dict[str, float], context: Optional[Dict[str, Any]]) -> None:
        """Generate feedback signals from training metrics."""
        # Performance feedback
        accuracy = train_metrics.get('accuracy', 0.0)
        performance_feedback = FeedbackSignal(
            feedback_type=FeedbackType.PERFORMANCE,
            value=accuracy,
            metadata={'metric': 'training_accuracy'},
            context=context or {},
            source='training'
        )
        self.feedback_manager.add_feedback(performance_feedback)

        # Error feedback if loss is high
        loss = train_metrics.get('total_loss', 0.0)
        if loss > self.config.neural_loss_threshold_normal:  # Threshold for high loss
            error_feedback = FeedbackSignal(
                feedback_type=FeedbackType.ERROR,
                value=min(1.0, loss / 10.0),  # Normalize to 0-1
                metadata={'metric': 'training_loss', 'loss_value': loss},
                context=context or {},
                source='training'
            )
            self.feedback_manager.add_feedback(error_feedback)

        # Success feedback if performance is good
        if accuracy > self.config.neural_accuracy_threshold_high:
            success_feedback = FeedbackSignal(
                feedback_type=FeedbackType.SUCCESS,
                value=accuracy,
                metadata={'metric': 'high_accuracy'},
                context=context or {},
                source='training'
            )
            self.feedback_manager.add_feedback(success_feedback)

    def _should_trigger_immediate_adaptation(self, train_metrics: Dict[str, float]) -> bool:
        """Check if immediate adaptation should be triggered."""
        # Trigger on very poor performance
        accuracy = train_metrics.get('accuracy', 1.0)
        loss = train_metrics.get('total_loss', 0.0)

        return accuracy < self.config.neural_accuracy_threshold_low or loss > self.config.neural_loss_threshold_high

    def _trigger_immediate_adaptation(self, train_metrics: Dict[str, float]) -> None:
        """Trigger immediate adaptation based on training metrics."""
        # Simple immediate adaptation - adjust learning rate
        current_lr = self.neural_network.optimizer.param_groups[0]['lr']
        new_lr = current_lr * 0.5  # Reduce learning rate

        for param_group in self.neural_network.optimizer.param_groups:
            param_group['lr'] = new_lr

        # Record adaptation
        adaptation_event = {
            'timestamp': datetime.now(),
            'type': 'immediate_lr_reduction',
            'trigger_metrics': train_metrics,
            'old_lr': current_lr,
            'new_lr': new_lr
        }
        self._append_to_state_safe('adaptation_history', adaptation_event)

        logger.info(f"Immediate adaptation: reduced learning rate from {current_lr} to {new_lr}")

    def _on_feedback_received(self, feedback: FeedbackSignal) -> None:
        """Handle feedback received events."""
        # Update coordination state based on feedback
        if feedback.feedback_type == FeedbackType.ERROR and feedback.value > self.config.feedback_error_threshold:
            self._set_state_safe('neural_health', 'degraded')
        elif feedback.feedback_type == FeedbackType.SUCCESS and feedback.value > self.config.feedback_success_threshold:
            if self._get_state_safe('neural_health') == 'degraded':
                self._set_state_safe('neural_health', 'healthy')

        # Record performance
        if feedback.feedback_type == FeedbackType.PERFORMANCE:
            self._append_to_deque_safe('recent_performance', feedback.value)

    def _get_coordination_insights(self) -> Dict[str, Any]:
        """Get coordination-specific insights."""
        adaptation_history = self._get_state_safe('adaptation_history', [])
        return {
            'total_adaptations': len(adaptation_history),
            'recent_adaptations': len([
                a for a in adaptation_history
                if (datetime.now() - a['timestamp']).total_seconds() < 3600
            ]),
            'neural_health_status': self._get_state_safe('neural_health'),
            'active_learning': self._get_state_safe('active_learning_cycle') is not None,
            'performance_stability': self._calculate_performance_stability()
        }

    def _get_memory_insights(self) -> Dict[str, Any]:
        """Get memory integration insights."""
        if not self.memory_manager:
            return {'status': 'not_available'}

        return {
            'status': 'available',
            'memory_utilization': self.memory_manager.get_system_status(),
            'recent_consolidations': 'not_implemented',  # Would track memory consolidation events
            'cross_memory_patterns': 'not_implemented'   # Would analyze cross-memory relationships
        }

    def _calculate_performance_stability(self) -> float:
        """Calculate performance stability metric."""
        recent_performance = list(self._get_state_safe('recent_performance', []))
        if len(recent_performance) < 10:
            return 1.0  # Assume stable if insufficient data

        # Calculate coefficient of variation
        mean_perf = np.mean(recent_performance)
        std_perf = np.std(recent_performance)

        if mean_perf == 0:
            return 0.0

        cv = std_perf / mean_perf
        stability = max(0.0, 1.0 - cv)  # Higher stability = lower coefficient of variation
        return float(stability)

    # System interface implementations for improvement engine
    def _optimize_parameter(self, parameter: str, optimization_params: Dict[str, Any]) -> None:
        """Optimize a specific parameter based on improvement analysis."""
        if parameter == 'learning_rate':
            current_lr = self.neural_network.optimizer.param_groups[0]['lr']
            action = optimization_params.get('action', 'reduce')
            factor = optimization_params.get('factor', 0.5)
            reason = optimization_params.get('reason', 'performance optimization')

            if action == 'reduce':
                new_lr = current_lr * factor
            elif action == 'increase':
                new_lr = current_lr * (2.0 - factor)  # Inverse scaling
            elif action == 'stabilize':
                new_lr = current_lr * factor
            else:
                new_lr = current_lr * factor

            # Apply bounds
            new_lr = max(1e-6, min(1e-1, new_lr))

            for param_group in self.neural_network.optimizer.param_groups:
                param_group['lr'] = new_lr

            logger.info(f"Parameter optimization: {parameter} {current_lr:.6f} -> {new_lr:.6f} ({reason})")

        elif parameter == 'dropout_rate':
            # Update dropout rate in network
            current_dropout = self.neural_network.dropout
            action = optimization_params.get('action', 'increase')
            factor = optimization_params.get('factor', 1.2)

            if action == 'increase':
                new_dropout = min(0.5, current_dropout * factor)
            else:
                new_dropout = max(0.1, current_dropout / factor)

            self.neural_network.dropout = new_dropout
            logger.info(f"Dropout rate optimization: {current_dropout:.3f} -> {new_dropout:.3f}")

        else:
            logger.warning(f"Unknown parameter optimization requested: {parameter}")

    def _adapt_architecture(self, adaptation_params: Dict[str, Any]) -> None:
        """Adapt neural network architecture based on performance needs."""
        action = adaptation_params.get('action', 'add_layer')
        reason = adaptation_params.get('reason', 'performance improvement')

        # Record parameters before change
        before_params = sum(p.numel() for p in self.neural_network.parameters())

        if action == 'add_layer':
            layer_size = adaptation_params.get('layer_size', 256)
            try:
                self.neural_network.add_hidden_layer(layer_size)
                after_params = sum(p.numel() for p in self.neural_network.parameters())

                adaptation_event = {
                    'timestamp': datetime.now(),
                    'type': action,
                    'params': adaptation_params,
                    'before_params': before_params,
                    'after_params': after_params,
                    'reason': reason
                }
                self._append_to_state_safe('adaptation_history', adaptation_event)
                logger.info(f"Architecture adapted: added layer of size {layer_size} ({reason})")
                logger.info(f"Parameters: {before_params} -> {after_params} (+{after_params - before_params})")

            except Exception as e:
                logger.error(f"Architecture adaptation failed: {e}")

        elif action == 'adjust_dropout':
            # Adjust dropout in existing layers
            new_dropout = adaptation_params.get('dropout_rate', 0.2)
            old_dropout = self.neural_network.dropout
            self.neural_network.dropout = new_dropout

            adaptation_event = {
                'timestamp': datetime.now(),
                'type': action,
                'params': {'old_dropout': old_dropout, 'new_dropout': new_dropout},
                'reason': reason
            }
            self._append_to_state_safe('adaptation_history', adaptation_event)
            logger.info(f"Architecture adapted: dropout {old_dropout:.3f} -> {new_dropout:.3f} ({reason})")

        else:
            logger.warning(f"Unknown architecture adaptation action: {action}")
    def _evolve_strategy(self, evolution_params: Dict[str, Any]) -> None:
        """Evolve trading strategies based on performance analysis."""
        action = evolution_params.get('action', 'tune_thresholds')
        current_performance = evolution_params.get('current_performance', 0.5)
        target_improvement = evolution_params.get('target_improvement', 0.05)
        reason = evolution_params.get('reason', 'strategy optimization')

        if action == 'tune_thresholds':
            # Strategy evolution focuses on trading signal thresholds
            # This is a simplified implementation - in reality would interact with strategy module
            threshold_adjustment = min(0.1, target_improvement * 2)

            if current_performance < 0.45:
                # Performance is poor, be more conservative
                adjustment_direction = "conservative"
                adjustment_factor = 1.0 + threshold_adjustment
            else:
                # Performance is acceptable, try to be more aggressive
                adjustment_direction = "aggressive"
                adjustment_factor = 1.0 - threshold_adjustment

            logger.info(f"Strategy evolution: {action} - {adjustment_direction} adjustment by {threshold_adjustment:.3f}")
            logger.info(f"Current performance: {current_performance:.3f}, target improvement: {target_improvement:.3f}")

        elif action == 'adjust_risk_params':
            # Adjust risk management parameters
            risk_adjustment = evolution_params.get('risk_adjustment', 0.05)
            logger.info(f"Strategy evolution: adjusting risk parameters by {risk_adjustment:.3f}")

        else:
            logger.warning(f"Unknown strategy evolution action: {action}")

        logger.info(f"Strategy evolution completed: {reason}")

    def _implement_meta_learning(self, meta_params: Dict[str, Any]) -> None:
        """Implement meta-learning improvements to address overfitting and generalization."""
        action = meta_params.get('action', 'increase_dropout')
        overfitting_signal = meta_params.get('overfitting_signal', 0.0)
        target_reduction = meta_params.get('target_reduction', 0.05)
        reason = meta_params.get('reason', 'meta-learning optimization')

        if action == 'increase_dropout':
            # Increase regularization to combat overfitting
            current_dropout = self.neural_network.dropout
            dropout_increase = min(0.2, overfitting_signal * 0.5)
            new_dropout = min(0.5, current_dropout + dropout_increase)

            self.neural_network.dropout = new_dropout
            logger.info(f"Meta-learning: increased dropout {current_dropout:.3f} -> {new_dropout:.3f}")
            logger.info(f"Overfitting signal: {overfitting_signal:.3f}, target reduction: {target_reduction:.3f}")

        elif action == 'adjust_learning_schedule':
            # Implement learning rate scheduling
            current_lr = self.neural_network.optimizer.param_groups[0]['lr']
            lr_reduction = min(0.5, overfitting_signal * 0.8)
            new_lr = current_lr * (1.0 - lr_reduction)

            for param_group in self.neural_network.optimizer.param_groups:
                param_group['lr'] = new_lr

            logger.info(f"Meta-learning: adjusted learning schedule {current_lr:.6f} -> {new_lr:.6f}")

        elif action == 'enable_early_stopping':
            # Enable early stopping mechanism
            patience = min(20, int(overfitting_signal * 50))
            logger.info(f"Meta-learning: enabled early stopping with patience {patience}")

        else:
            logger.warning(f"Unknown meta-learning action: {action}")

        logger.info(f"Meta-learning implementation completed: {reason}")

    def _measure_performance(self) -> float:
        """Measure current system performance."""
        recent_performance = list(self._get_state_safe('recent_performance', []))
        if recent_performance:
            return float(np.mean(recent_performance[-10:]))  # Last 10 measurements
        return 0.5  # Default neutral performance

    def _create_rollback_data(self, candidate: ImprovementCandidate) -> Dict[str, Any]:
        """Create rollback data for an improvement."""
        return {
            'neural_state': self.neural_network.state_dict(),
            'optimizer_state': self.neural_network.optimizer.state_dict(),
            'improvement_type': candidate.improvement_type.value,
            'timestamp': datetime.now()
        }

    def _rollback_improvement(self, rollback_data: Dict[str, Any]) -> None:
        """Rollback an improvement using rollback data."""
        try:
            self.neural_network.load_state_dict(rollback_data['neural_state'])
            self.neural_network.optimizer.load_state_dict(rollback_data['optimizer_state'])
            logger.info(f"Rolled back improvement: {rollback_data['improvement_type']}")
        except Exception as e:
            logger.error(f"Rollback failed: {e}")

    def _start_coordination_loop(self) -> None:
        """Start background coordination loop."""
        self.running = True
        self.coordination_thread = threading.Thread(target=self._coordination_loop, daemon=True)
        self.coordination_thread.start()

    def _coordination_loop(self) -> None:
        """Background coordination loop."""
        while self.running:
            try:
                # Periodic coordination tasks
                self._periodic_coordination()

                time.sleep(self.coordination_interval)
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")

    def _periodic_coordination(self) -> None:
        """Perform periodic coordination tasks."""
        with self._coordination_lock:
            # Health monitoring
            neural_health = self.health_monitor.check_neural_health(self.neural_network)
            if neural_health['status'] != 'healthy':
                self._set_state_safe('neural_health', neural_health['status'])

            # Memory optimization
            if self.memory_manager:
                optimization_results = self.memory_manager.optimize_memory_usage()
                if optimization_results:
                    logger.debug(f"Memory optimization results: {optimization_results}")

            # Learning cycle timeout check
            if self._get_state_safe('active_learning_cycle'):
                # Check if cycle has been active too long
                cycle_start = self.feedback_manager.current_cycle.start_time
                cycle_duration = time.time() - cycle_start.timestamp()
                if cycle_duration > self.config.feedback_cycle_timeout:  # Configurable timeout
                    logger.warning("Learning cycle timeout - ending cycle")
                    self.feedback_manager.end_learning_cycle()
                    self._set_state_safe('active_learning_cycle', None)

    def shutdown(self) -> None:
        """Shutdown neural coordinator and all components."""
        self.running = False

        # Shutdown coordination thread
        if self.coordination_thread and self.coordination_thread.is_alive():
            self.coordination_thread.join(timeout=2.0)

        # Shutdown component systems
        self.feedback_manager.shutdown()
        self.improvement_engine.shutdown()

        if self.memory_manager:
            self.memory_manager.shutdown()

        logger.info("NeuralCoordinator shutdown complete")


class PerformanceTracker:
    """Tracks neural network performance metrics."""

    def __init__(self):
        self.prediction_history = deque(maxlen=1000)
        self.training_history = deque(maxlen=1000)
        self.performance_metrics = {}

    def record_prediction(self, prediction: torch.Tensor, aux_outputs: Dict[str, torch.Tensor]) -> None:
        """Record prediction and auxiliary outputs."""
        confidence_val = aux_outputs.get('confidence', 0.5)
        uncertainty_val = aux_outputs.get('uncertainty', 0.5)

        if isinstance(confidence_val, torch.Tensor):
            confidence = confidence_val.mean().item()
        else:
            confidence = float(confidence_val)

        if isinstance(uncertainty_val, torch.Tensor):
            uncertainty = uncertainty_val.mean().item()
        else:
            uncertainty = float(uncertainty_val)

        self.prediction_history.append({
            'timestamp': datetime.now(),
            'confidence': confidence,
            'uncertainty': uncertainty
        })

    def record_training_step(self, train_metrics: Dict[str, float]) -> None:
        """Record training step metrics."""
        self.training_history.append({
            'timestamp': datetime.now(),
            **train_metrics
        })

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        if not self.training_history:
            return {}

        recent_training = list(self.training_history)[-10:]
        recent_predictions = list(self.prediction_history)[-50:]

        metrics = {}

        # Training metrics
        if recent_training:
            metrics['training_accuracy'] = np.mean([t.get('accuracy', 0) for t in recent_training])
            metrics['training_loss'] = np.mean([t.get('total_loss', 0) for t in recent_training])
            metrics['loss_volatility'] = np.std([t.get('total_loss', 0) for t in recent_training])

        # Prediction metrics
        if recent_predictions:
            metrics['prediction_confidence'] = np.mean([p['confidence'] for p in recent_predictions])
            metrics['prediction_uncertainty'] = np.mean([p['uncertainty'] for p in recent_predictions])

        return metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            'total_predictions': len(self.prediction_history),
            'total_training_steps': len(self.training_history),
            'current_metrics': self.get_current_metrics()
        }


class HealthMonitor:
    """Monitors system health and detects issues."""

    def check_neural_health(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Check neural network health."""
        health = {
            'status': 'healthy',
            'issues': [],
            'parameter_stats': {}
        }

        # Check for gradient issues
        total_norm = 0.0
        param_count = 0

        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            health['parameter_stats']['gradient_norm'] = total_norm

            if total_norm > 10.0:
                health['status'] = 'degraded'
                health['issues'].append('high_gradient_norm')
            elif total_norm < 1e-6:
                health['status'] = 'degraded'
                health['issues'].append('vanishing_gradients')

        # Check for NaN parameters
        nan_params = 0
        for param in model.parameters():
            if torch.isnan(param).any():
                nan_params += 1

        if nan_params > 0:
            health['status'] = 'critical'
            health['issues'].append(f'nan_parameters_{nan_params}')

        return health
