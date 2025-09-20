"""
PERFORMANCE MONITORING SYSTEM
Real-time monitoring of system performance and resource usage
"""

import asyncio
import time
import psutil
import gc
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque
import weakref

from ..logging_system import TradeLogger
from ..utils.circuit_breaker import circuit_manager


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    api_requests_per_second: float
    active_tasks: int
    circuit_breaker_stats: Dict
    gc_stats: Dict


class PerformanceMonitor:
    """
    REAL-TIME PERFORMANCE MONITORING
    Tracks system resources and API performance
    """

    def __init__(self, logger: Optional[TradeLogger] = None):
        """Initialize performance monitor

        Args:
            logger: Trade logger instance
        """
        self.logger = logger
        self.running = False
        self.monitor_task = None

        # Metrics storage
        self.metrics_history = deque(maxlen=1000)  # Last 1000 readings
        self.alert_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 90.0,
            'api_requests_per_second': 20.0,  # Binance rate limit consideration
            'active_tasks': 100
        }

        # Request tracking
        self.request_times = deque(maxlen=100)  # Last 100 request timestamps
        self.api_call_count = 0
        self._lock = threading.Lock()

        # Alert callbacks
        self.alert_callbacks: List[Callable] = []

        # Baseline metrics
        self.process = psutil.Process()
        self.net_io_start = psutil.net_io_counters()

    async def start_monitoring(self, interval_seconds: float = 5.0):
        """Start performance monitoring

        Args:
            interval_seconds: Monitoring interval
        """
        if self.running:
            return

        self.running = True
        self.monitor_task = asyncio.create_task(
            self._monitor_loop(interval_seconds)
        )

        if self.logger:
            self.logger.log_system_event(
                'monitor', 'started',
                {'interval_seconds': interval_seconds}
            )

    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        if self.logger:
            self.logger.log_system_event(
                'monitor', 'stopped',
                {'total_readings': len(self.metrics_history)}
            )

    async def _monitor_loop(self, interval: float):
        """Main monitoring loop

        Args:
            interval: Sleep interval between readings
        """
        while self.running:
            try:
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)

                # Check for alerts
                await self._check_alerts(metrics)

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.logger:
                    self.logger.log_error(
                        'monitor', 'collection_error',
                        f"Error collecting metrics: {e}",
                        exception=e
                    )
                await asyncio.sleep(interval)

    async def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics

        Returns:
            Performance metrics snapshot
        """
        # CPU and memory
        cpu_percent = self.process.cpu_percent()
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = self.process.memory_percent()

        # Network I/O
        net_io = psutil.net_io_counters()
        net_bytes_sent = net_io.bytes_sent - self.net_io_start.bytes_sent
        net_bytes_recv = net_io.bytes_recv - self.net_io_start.bytes_recv

        # API request rate
        with self._lock:
            current_time = time.time()
            # Remove old requests (older than 1 minute)
            while self.request_times and current_time - self.request_times[0] > 60:
                self.request_times.popleft()

            api_requests_per_second = len(self.request_times) / 60.0

        # Active async tasks
        active_tasks = len([task for task in asyncio.all_tasks()
                          if not task.done()])

        # Circuit breaker stats
        circuit_stats = circuit_manager.get_all_stats()

        # Garbage collection stats
        gc_stats = {
            'collections': gc.get_stats(),
            'objects': len(gc.get_objects()),
            'garbage': len(gc.garbage)
        }

        return PerformanceMetrics(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            memory_percent=memory_percent,
            network_bytes_sent=net_bytes_sent,
            network_bytes_recv=net_bytes_recv,
            api_requests_per_second=api_requests_per_second,
            active_tasks=active_tasks,
            circuit_breaker_stats=circuit_stats,
            gc_stats=gc_stats
        )

    async def _check_alerts(self, metrics: PerformanceMetrics):
        """Check for performance alerts

        Args:
            metrics: Current metrics
        """
        alerts = []

        # Check CPU
        if metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")

        # Check memory
        if metrics.memory_percent > self.alert_thresholds['memory_percent']:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")

        # Check API rate
        if metrics.api_requests_per_second > self.alert_thresholds['api_requests_per_second']:
            alerts.append(f"High API rate: {metrics.api_requests_per_second:.1f} req/s")

        # Check task count
        if metrics.active_tasks > self.alert_thresholds['active_tasks']:
            alerts.append(f"High task count: {metrics.active_tasks}")

        # Check circuit breakers
        for name, stats in metrics.circuit_breaker_stats.items():
            if stats['state'] == 'open':
                alerts.append(f"Circuit breaker OPEN: {name}")

        # Trigger alerts
        if alerts:
            alert_data = {
                'alerts': alerts,
                'metrics': asdict(metrics)
            }

            if self.logger:
                self.logger.log_system_event(
                    'monitor', 'performance_alert',
                    alert_data
                )

            # Call registered callbacks
            for callback in self.alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert_data)
                    else:
                        callback(alert_data)
                except Exception as e:
                    if self.logger:
                        self.logger.log_error(
                            'monitor', 'callback_error',
                            f"Error in alert callback: {e}",
                            exception=e
                        )

    def record_api_call(self):
        """Record an API call for rate tracking"""
        with self._lock:
            self.request_times.append(time.time())
            self.api_call_count += 1

    def add_alert_callback(self, callback: Callable):
        """Add alert callback function

        Args:
            callback: Function to call on alerts
        """
        self.alert_callbacks.append(callback)

    def get_current_stats(self) -> Dict:
        """Get current performance statistics

        Returns:
            Current stats dictionary
        """
        if not self.metrics_history:
            return {}

        latest = self.metrics_history[-1]

        # Calculate averages over last 10 readings
        recent_metrics = list(self.metrics_history)[-10:]
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_mb for m in recent_metrics) / len(recent_metrics)

        return {
            'latest_metrics': asdict(latest),
            'averages': {
                'cpu_percent': avg_cpu,
                'memory_mb': avg_memory
            },
            'totals': {
                'api_calls': self.api_call_count,
                'readings': len(self.metrics_history)
            },
            'alerts': {
                'thresholds': self.alert_thresholds,
                'callbacks_registered': len(self.alert_callbacks)
            }
        }

    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report

        Returns:
            Performance report
        """
        if not self.metrics_history:
            return {'error': 'No metrics collected yet'}

        metrics = list(self.metrics_history)

        # Calculate statistics
        cpu_values = [m.cpu_percent for m in metrics]
        memory_values = [m.memory_mb for m in metrics]

        report = {
            'summary': {
                'collection_period_minutes': (metrics[-1].timestamp - metrics[0].timestamp) / 60,
                'total_readings': len(metrics),
                'api_calls_total': self.api_call_count
            },
            'cpu': {
                'current': cpu_values[-1],
                'average': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            },
            'memory': {
                'current_mb': memory_values[-1],
                'average_mb': sum(memory_values) / len(memory_values),
                'max_mb': max(memory_values),
                'min_mb': min(memory_values)
            },
            'circuit_breakers': metrics[-1].circuit_breaker_stats,
            'alerts_triggered': sum(1 for m in metrics
                                  if m.cpu_percent > self.alert_thresholds['cpu_percent'] or
                                     m.memory_percent > self.alert_thresholds['memory_percent'])
        }

        return report


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor(logger: Optional[TradeLogger] = None) -> PerformanceMonitor:
    """Get global performance monitor instance

    Args:
        logger: Trade logger instance

    Returns:
        Performance monitor
    """
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor(logger)
    return _global_monitor


async def start_global_monitoring(interval_seconds: float = 5.0,
                                logger: Optional[TradeLogger] = None):
    """Start global performance monitoring

    Args:
        interval_seconds: Monitoring interval
        logger: Trade logger instance
    """
    monitor = get_performance_monitor(logger)
    await monitor.start_monitoring(interval_seconds)


async def stop_global_monitoring():
    """Stop global performance monitoring"""
    global _global_monitor
    if _global_monitor:
        await _global_monitor.stop_monitoring()