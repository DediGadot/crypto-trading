#!/usr/bin/env python3
"""Performance optimizations for SAFLA trading system."""

import torch
import time
from typing import Dict, List, Optional, Tuple, Any
from functools import lru_cache
import threading

from safla_trading.core.safla_neural import SAFLANeuralNetwork
from safla_trading.config import Config


class OptimizedSAFLANeuralNetwork(SAFLANeuralNetwork):
    """Performance-optimized version of SAFLANeuralNetwork."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Performance optimizations
        self._batch_cache: Dict[int, torch.Tensor] = {}
        self._cache_lock = threading.Lock()
        self._warmup_done = False

        # Pre-compile model for inference
        self._optimize_for_inference()

    def _optimize_for_inference(self):
        """Apply inference optimizations."""
        # Set to evaluation mode
        self.eval()

        # Enable optimizations for inference
        torch.set_grad_enabled(False)

        # Compile model if PyTorch 2.0+ is available
        try:
            if hasattr(torch, 'compile'):
                self.forward = torch.compile(self.forward, mode="max-autotune")
        except Exception:
            pass  # Fallback if compilation fails

    def warmup(self, batch_sizes: List[int] = None):
        """Warmup the model with common batch sizes."""
        if self._warmup_done:
            return

        if batch_sizes is None:
            batch_sizes = [1, 8, 16, 32]

        print("Warming up neural network for optimal performance...")

        with torch.no_grad():
            for batch_size in batch_sizes:
                dummy_input = torch.randn(batch_size, self.input_dim)

                # Run several warmup iterations
                for _ in range(5):
                    _ = super().forward(dummy_input)

        self._warmup_done = True
        print("Neural network warmup completed.")

    def predict_batch(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Optimized batch prediction."""
        if not self._warmup_done:
            self.warmup()

        batch_size = inputs.shape[0]

        # Use cached tensors for common batch sizes
        with self._cache_lock:
            if batch_size not in self._batch_cache:
                self._batch_cache[batch_size] = torch.zeros(
                    batch_size, self.output_dim,
                    dtype=inputs.dtype, device=inputs.device
                )

        with torch.no_grad():
            outputs = self.forward(inputs)

        # Create auxiliary outputs efficiently
        aux_outputs = {
            'batch_size': batch_size,
            'inference_mode': True,
            'optimized': True
        }

        return outputs, aux_outputs


class BatchProcessor:
    """Efficient batch processing for neural operations."""

    def __init__(self, model: SAFLANeuralNetwork, max_batch_size: int = 64):
        self.model = model
        self.max_batch_size = max_batch_size
        self._input_queue: List[torch.Tensor] = []
        self._queue_lock = threading.Lock()

    def add_to_batch(self, input_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """Add input to batch queue and process when full."""
        with self._queue_lock:
            self._input_queue.append(input_tensor)

            if len(self._input_queue) >= self.max_batch_size:
                return self._process_batch()

        return None

    def _process_batch(self) -> torch.Tensor:
        """Process accumulated batch."""
        if not self._input_queue:
            return None

        # Stack inputs into batch
        batch_input = torch.stack(self._input_queue)
        self._input_queue.clear()

        # Process batch
        with torch.no_grad():
            batch_output = self.model(batch_input)

        return batch_output

    def flush_batch(self) -> Optional[torch.Tensor]:
        """Process remaining items in queue."""
        with self._queue_lock:
            if self._input_queue:
                return self._process_batch()
        return None


class MemoryOptimizer:
    """Memory usage optimization utilities."""

    @staticmethod
    def optimize_tensor_memory():
        """Optimize PyTorch tensor memory usage."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Set memory management settings
        torch.set_num_threads(4)  # Limit CPU threads

    @staticmethod
    @lru_cache(maxsize=128)
    def get_cached_tensor(shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Get cached tensor of specified shape."""
        return torch.zeros(shape, dtype=dtype)

    @staticmethod
    def clear_tensor_cache():
        """Clear cached tensors."""
        MemoryOptimizer.get_cached_tensor.cache_clear()


class ConcurrentPredictor:
    """Thread-safe concurrent prediction handler."""

    def __init__(self, model: SAFLANeuralNetwork, max_workers: int = 4):
        self.model = model
        self.max_workers = max_workers
        self._prediction_lock = threading.RLock()
        self._stats = {
            'total_predictions': 0,
            'total_time': 0.0,
            'concurrent_requests': 0
        }

    def predict_concurrent(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Thread-safe prediction with performance tracking."""
        start_time = time.perf_counter()

        with self._prediction_lock:
            self._stats['concurrent_requests'] += 1

            try:
                # Perform prediction
                with torch.no_grad():
                    output = self.model(input_tensor)

                # Update statistics
                end_time = time.perf_counter()
                self._stats['total_predictions'] += 1
                self._stats['total_time'] += (end_time - start_time)

                aux_outputs = {
                    'prediction_id': self._stats['total_predictions'],
                    'processing_time': end_time - start_time,
                    'concurrent_requests': self._stats['concurrent_requests'],
                    'average_time': self._stats['total_time'] / self._stats['total_predictions']
                }

                return output, aux_outputs

            finally:
                self._stats['concurrent_requests'] -= 1

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._prediction_lock:
            avg_time = (self._stats['total_time'] / self._stats['total_predictions']
                       if self._stats['total_predictions'] > 0 else 0)

            return {
                'total_predictions': self._stats['total_predictions'],
                'average_prediction_time': avg_time,
                'predictions_per_second': 1.0 / avg_time if avg_time > 0 else 0,
                'current_concurrent_requests': self._stats['concurrent_requests']
            }


def apply_global_optimizations():
    """Apply global performance optimizations."""
    print("Applying global performance optimizations...")

    # PyTorch optimizations
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)

    # Enable optimized kernels
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    except Exception:
        pass  # CUDA not available

    # Memory optimizations
    MemoryOptimizer.optimize_tensor_memory()

    print("Global optimizations applied.")


def benchmark_optimization_impact():
    """Benchmark the impact of optimizations."""
    print("\nBenchmarking optimization impact...\n")

    config = Config()

    # Test standard model
    print("Testing standard model...")
    standard_model = SAFLANeuralNetwork(
        input_dim=config.neural_input_dim,
        hidden_dim=config.neural_hidden_dim,
        output_dim=config.neural_output_dim
    )

    test_input = torch.randn(1, config.neural_input_dim)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = standard_model(test_input)

    # Benchmark standard
    start_time = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            _ = standard_model(test_input)
    standard_time = time.perf_counter() - start_time

    # Test optimized model
    print("Testing optimized model...")
    optimized_model = OptimizedSAFLANeuralNetwork(
        input_dim=config.neural_input_dim,
        hidden_dim=config.neural_hidden_dim,
        output_dim=config.neural_output_dim
    )
    optimized_model.warmup()

    # Benchmark optimized
    start_time = time.perf_counter()
    for _ in range(100):
        _ = optimized_model.predict_batch(test_input)
    optimized_time = time.perf_counter() - start_time

    # Results
    improvement = ((standard_time - optimized_time) / standard_time) * 100

    print(f"\n📊 OPTIMIZATION RESULTS:")
    print(f"  Standard model: {standard_time:.4f}s (100 predictions)")
    print(f"  Optimized model: {optimized_time:.4f}s (100 predictions)")
    print(f"  Performance improvement: {improvement:.1f}%")
    print(f"  Speed factor: {standard_time/optimized_time:.2f}x")

    # Test batch processing
    print("\nTesting batch processing optimization...")

    batch_input = torch.randn(32, config.neural_input_dim)

    # Standard batch processing
    start_time = time.perf_counter()
    with torch.no_grad():
        _ = standard_model(batch_input)
    standard_batch_time = time.perf_counter() - start_time

    # Optimized batch processing
    start_time = time.perf_counter()
    _ = optimized_model.predict_batch(batch_input)
    optimized_batch_time = time.perf_counter() - start_time

    batch_improvement = ((standard_batch_time - optimized_batch_time) / standard_batch_time) * 100

    print(f"\n📊 BATCH PROCESSING RESULTS:")
    print(f"  Standard batch: {standard_batch_time:.4f}s (32 samples)")
    print(f"  Optimized batch: {optimized_batch_time:.4f}s (32 samples)")
    print(f"  Batch improvement: {batch_improvement:.1f}%")
    print(f"  Batch throughput: {32/optimized_batch_time:.0f} samples/sec")


if __name__ == "__main__":
    print("SAFLA Trading System Performance Optimizations\n")

    # Apply global optimizations
    apply_global_optimizations()

    # Benchmark optimization impact
    benchmark_optimization_impact()

    print("\n✅ Performance optimization analysis completed!")