"""SAFLA Core Neural Components.

Contains the core neural network architectures, feedback loops,
and self-improvement mechanisms for the SAFLA trading system.
"""

from .safla_neural import SAFLANeuralNetwork
from .feedback_loops import FeedbackLoopManager, FeedbackSignal, FeedbackType
from .self_improvement import SelfImprovementEngine, ImprovementCandidate, ImprovementType
from .neural_coordinator import NeuralCoordinator

__all__ = [
    "SAFLANeuralNetwork",
    "FeedbackLoopManager",
    "FeedbackSignal",
    "FeedbackType",
    "SelfImprovementEngine",
    "ImprovementCandidate",
    "ImprovementType",
    "NeuralCoordinator"
]