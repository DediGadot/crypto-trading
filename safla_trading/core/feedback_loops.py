"""Simplified feedback loop manager used by the coordinator."""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional

from datetime import datetime


class FeedbackType(Enum):
    PERFORMANCE = "performance"
    ERROR = "error"
    SUCCESS = "success"
    MARKET = "market"
    USER = "user"


@dataclass
class FeedbackSignal:
    feedback_type: FeedbackType
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    source: str = "internal"


@dataclass
class LearningCycle:
    cycle_id: str
    context: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime] = None
    feedback_signals: List[FeedbackSignal] = field(default_factory=list)


class PerformanceAnalyzer:
    def analyze_performance(self, signals: List[FeedbackSignal]) -> Dict[str, Any]:
        if not signals:
            return {
                "trends": {},
                "patterns": {},
                "recommendations": [],
            }
        summary: Dict[str, List[float]] = {}
        for signal in signals:
            summary.setdefault(signal.feedback_type.value, []).append(signal.value)
        trends = {key: sum(values) / len(values) for key, values in summary.items()}
        patterns = {
            key: {
                "count": len(values),
                "mean": sum(values) / len(values),
            }
            for key, values in summary.items()
        }
        recommendations = []
        if trends.get(FeedbackType.ERROR.value, 0) > 0.5:
            recommendations.append("Investigate high error feedback")
        if trends.get(FeedbackType.PERFORMANCE.value, 0) < 0.5:
            recommendations.append("Consider parameter tuning")
        return {
            "trends": trends,
            "patterns": patterns,
            "recommendations": recommendations,
        }


class FeedbackLoopManager:
    def __init__(self, memory_manager: Any = None):
        self.feedback_buffer: Deque[FeedbackSignal] = deque(maxlen=1_000)
        self.completed_cycles: List[LearningCycle] = []
        self.current_cycle: Optional[LearningCycle] = None
        self.feedback_observers: List[Callable[[FeedbackSignal], None]] = []
        self.performance_analyzer = PerformanceAnalyzer()
        self.learning_metrics: Dict[str, Any] = {
            "feedback_processed": 0,
            "cycles_started": 0,
            "cycles_completed": 0,
        }
        self.memory_manager = memory_manager

    def add_feedback(self, feedback: FeedbackSignal) -> None:
        self.feedback_buffer.append(feedback)
        self.learning_metrics["feedback_processed"] += 1
        if self.current_cycle is not None:
            self.current_cycle.feedback_signals.append(feedback)
        for observer in list(self.feedback_observers):
            observer(feedback)

    def start_learning_cycle(self, context: Dict[str, Any]) -> str:
        cycle_id = uuid.uuid4().hex
        self.current_cycle = LearningCycle(
            cycle_id=cycle_id,
            context=dict(context),
            start_time=datetime.now(),
        )
        self.learning_metrics["cycles_started"] += 1
        return cycle_id

    def end_learning_cycle(self) -> Optional[LearningCycle]:
        if not self.current_cycle:
            return None
        self.current_cycle.end_time = datetime.now()
        self.completed_cycles.append(self.current_cycle)
        self.learning_metrics["cycles_completed"] += 1
        finished = self.current_cycle
        self.current_cycle = None
        return finished

    def add_feedback_observer(self, observer: Callable[[FeedbackSignal], None]) -> None:
        if observer not in self.feedback_observers:
            self.feedback_observers.append(observer)

    def remove_feedback_observer(self, observer: Callable[[FeedbackSignal], None]) -> None:
        if observer in self.feedback_observers:
            self.feedback_observers.remove(observer)

    def get_learning_insights(self) -> Dict[str, Any]:
        if not self.completed_cycles:
            return {"message": "No learning cycles available"}
        latest = self.completed_cycles[-1]
        analysis = self.performance_analyzer.analyze_performance(latest.feedback_signals)
        return {
            "cycle_summary": {
                "cycle_id": latest.cycle_id,
                "feedback_count": len(latest.feedback_signals),
                "duration_seconds": (
                    (latest.end_time or datetime.now()) - latest.start_time
                ).total_seconds(),
            },
            "analysis": analysis,
        }

    def get_system_status(self) -> Dict[str, Any]:
        return {
            "learning_metrics": dict(self.learning_metrics),
            "feedback_buffer_size": len(self.feedback_buffer),
            "cycles_completed": len(self.completed_cycles),
            "current_cycle_active": self.current_cycle is not None,
        }

    def shutdown(self) -> None:
        self.feedback_buffer.clear()
        self.feedback_observers.clear()
        self.current_cycle = None


__all__ = [
    "FeedbackType",
    "FeedbackSignal",
    "LearningCycle",
    "FeedbackLoopManager",
]
