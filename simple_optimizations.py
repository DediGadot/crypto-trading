#!/usr/bin/env python3
"""Simple and effective performance optimizations for SAFLA trading system."""

import torch
import time
import gc
from typing import Dict, List, Optional, Tuple, Any
import threading

from safla_trading.core.safla_neural import SAFLANeuralNetwork
from safla_trading.config import Config


class PerformanceConfig:
    """Configuration for performance optimizations."""

    def __init__(self):
        # Neural network optimizations
        self.enable_torch_no_grad = True
        self.use_torch_jit = False  # Disabled due to compatibility issues
        self.tensor_cache_size = 100

        # Memory optimizations
        self.auto_gc_frequency = 1000  # Run GC every N operations
        self.clear_cuda_cache = True

        # Threading optimizations
        self.torch_num_threads = 4
        self.torch_num_interop_threads = 2


class SimpleNeuralOptimizer:
    """Simple neural network performance optimizer."""

    def __init__(self, model: SAFLANeuralNetwork, config: PerformanceConfig = None):
        self.model = model
        self.config = config or PerformanceConfig()
        self.operation_count = 0
        self._tensor_cache = {}
        self._cache_lock = threading.Lock()

        self._apply_basic_optimizations()

    def _apply_basic_optimizations(self):
        """Apply basic, safe optimizations."""
        # Set model to evaluation mode for inference
        self.model.eval()

        # Configure PyTorch threading
        torch.set_num_threads(self.config.torch_num_threads)
        torch.set_num_interop_threads(self.config.torch_num_interop_threads)

        # Enable CUDNN optimizations if available
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True

    def predict_optimized(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Optimized prediction with performance enhancements."""
        start_time = time.perf_counter()

        # Increment operation count
        self.operation_count += 1

        # Use no_grad context for inference
        if self.config.enable_torch_no_grad:
            with torch.no_grad():
                output = self.model(input_tensor)
        else:
            output = self.model(input_tensor)

        # Periodic garbage collection
        if (self.operation_count % self.config.auto_gc_frequency == 0):
            self._periodic_cleanup()

        end_time = time.perf_counter()

        aux_outputs = {
            'operation_id': self.operation_count,
            'processing_time': end_time - start_time,
            'optimized': True,
            'model_mode': 'eval' if not self.model.training else 'train'
        }

        return output, aux_outputs

    def predict_batch_optimized(self, input_batch: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Optimized batch prediction."""
        batch_size = input_batch.shape[0]
        start_time = time.perf_counter()

        # Get or create cached output tensor
        output_key = f"output_{batch_size}_{self.model.output_dim}"

        with self._cache_lock:
            if output_key not in self._tensor_cache:
                self._tensor_cache[output_key] = torch.zeros(
                    batch_size, self.model.output_dim,
                    dtype=input_batch.dtype,
                    device=input_batch.device
                )

        # Perform batch prediction
        with torch.no_grad():
            output = self.model(input_batch)

        end_time = time.perf_counter()
        processing_time = end_time - start_time

        aux_outputs = {
            'batch_size': batch_size,
            'processing_time': processing_time,
            'throughput': batch_size / processing_time,
            'time_per_sample': processing_time / batch_size,
            'optimized_batch': True
        }

        return output, aux_outputs

    def _periodic_cleanup(self):
        """Periodic cleanup to maintain performance."""
        # Run garbage collection
        gc.collect()

        # Clear CUDA cache if available
        if self.config.clear_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Clear tensor cache if it gets too large
        with self._cache_lock:
            if len(self._tensor_cache) > self.config.tensor_cache_size:
                self._tensor_cache.clear()

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'total_operations': self.operation_count,
            'cache_size': len(self._tensor_cache),
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'model_mode': 'eval' if not self.model.training else 'train',
            'torch_threads': torch.get_num_threads(),
            'optimizations_enabled': {
                'no_grad': self.config.enable_torch_no_grad,
                'auto_gc': True,
                'tensor_cache': True
            }
        }


class MemoryProfiler:
    """Simple memory usage profiler."""

    def __init__(self):
        self.memory_snapshots = []

    def snapshot(self, label: str = ""):
        """Take a memory snapshot."""
        snapshot = {
            'label': label,
            'timestamp': time.time(),
            'allocated_memory': torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0,
            'cached_memory': torch.cuda.memory_reserved() / 1024 / 1024 if torch.cuda.is_available() else 0
        }
        self.memory_snapshots.append(snapshot)
        return snapshot

    def get_memory_diff(self, start_label: str, end_label: str) -> Dict[str, float]:
        """Get memory difference between two snapshots."""
        start_snapshot = next((s for s in self.memory_snapshots if s['label'] == start_label), None)
        end_snapshot = next((s for s in self.memory_snapshots if s['label'] == end_label), None)

        if not start_snapshot or not end_snapshot:
            return {'error': 'Snapshots not found'}

        return {
            'allocated_diff_mb': end_snapshot['allocated_memory'] - start_snapshot['allocated_memory'],
            'cached_diff_mb': end_snapshot['cached_memory'] - start_snapshot['cached_memory'],
            'time_diff_sec': end_snapshot['timestamp'] - start_snapshot['timestamp']
        }


def benchmark_simple_optimizations():
    """Benchmark simple optimization techniques."""
    print("Benchmarking simple performance optimizations...\n")

    config = Config()
    perf_config = PerformanceConfig()

    # Create models
    standard_model = SAFLANeuralNetwork(
        input_dim=config.neural_input_dim,
        hidden_dim=config.neural_hidden_dim,
        output_dim=config.neural_output_dim
    )

    optimizer = SimpleNeuralOptimizer(standard_model, perf_config)
    profiler = MemoryProfiler()

    # Test data
    single_input = torch.randn(1, config.neural_input_dim)
    batch_input = torch.randn(32, config.neural_input_dim)

    print("Testing single prediction performance...")

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = standard_model(single_input)
        _ = optimizer.predict_optimized(single_input)

    # Benchmark standard single predictions
    profiler.snapshot("standard_single_start")
    start_time = time.perf_counter()

    for _ in range(1000):
        with torch.no_grad():
            _ = standard_model(single_input)

    standard_single_time = time.perf_counter() - start_time
    profiler.snapshot("standard_single_end")

    # Benchmark optimized single predictions
    profiler.snapshot("optimized_single_start")
    start_time = time.perf_counter()

    for _ in range(1000):
        _ = optimizer.predict_optimized(single_input)

    optimized_single_time = time.perf_counter() - start_time
    profiler.snapshot("optimized_single_end")

    print("Testing batch prediction performance...")

    # Benchmark standard batch predictions
    profiler.snapshot("standard_batch_start")
    start_time = time.perf_counter()

    for _ in range(100):
        with torch.no_grad():
            _ = standard_model(batch_input)

    standard_batch_time = time.perf_counter() - start_time
    profiler.snapshot("standard_batch_end")

    # Benchmark optimized batch predictions
    profiler.snapshot("optimized_batch_start")
    start_time = time.perf_counter()

    for _ in range(100):
        _ = optimizer.predict_batch_optimized(batch_input)

    optimized_batch_time = time.perf_counter() - start_time
    profiler.snapshot("optimized_batch_end")

    # Calculate improvements
    single_improvement = ((standard_single_time - optimized_single_time) / standard_single_time) * 100
    batch_improvement = ((standard_batch_time - optimized_batch_time) / standard_batch_time) * 100

    # Print results
    print("\n" + "="*60)
    print("SIMPLE OPTIMIZATION RESULTS")
    print("="*60)

    print(f"\n🚀 SINGLE PREDICTION PERFORMANCE:")
    print(f"  Standard: {standard_single_time:.4f}s (1000 predictions)")
    print(f"  Optimized: {optimized_single_time:.4f}s (1000 predictions)")
    print(f"  Improvement: {single_improvement:.1f}%")
    print(f"  Throughput: {1000/optimized_single_time:.0f} predictions/sec")

    print(f"\n📦 BATCH PREDICTION PERFORMANCE:")
    print(f"  Standard: {standard_batch_time:.4f}s (100 batches of 32)")
    print(f"  Optimized: {optimized_batch_time:.4f}s (100 batches of 32)")
    print(f"  Improvement: {batch_improvement:.1f}%")
    print(f"  Batch throughput: {3200/optimized_batch_time:.0f} samples/sec")

    # Performance stats
    stats = optimizer.get_performance_stats()
    print(f"\n📊 OPTIMIZER STATISTICS:")
    print(f"  Total operations: {stats['total_operations']}")
    print(f"  Cache size: {stats['cache_size']}")
    print(f"  Model parameters: {stats['model_parameters']:,}")
    print(f"  PyTorch threads: {stats['torch_threads']}")

    # Memory analysis (if CUDA available)
    if torch.cuda.is_available():
        single_memory = profiler.get_memory_diff("standard_single_start", "optimized_single_end")
        batch_memory = profiler.get_memory_diff("standard_batch_start", "optimized_batch_end")

        print(f"\n💾 MEMORY USAGE:")
        print(f"  Single prediction memory: {single_memory.get('allocated_diff_mb', 0):.1f} MB")
        print(f"  Batch prediction memory: {batch_memory.get('allocated_diff_mb', 0):.1f} MB")

    print("\n✅ Simple optimization benchmark completed!")

    return {
        'single_improvement_percent': single_improvement,
        'batch_improvement_percent': batch_improvement,
        'optimized_single_throughput': 1000/optimized_single_time,
        'optimized_batch_throughput': 3200/optimized_batch_time,
        'stats': stats
    }


if __name__ == "__main__":
    print("SAFLA Trading System - Simple Performance Optimizations\n")

    try:
        results = benchmark_simple_optimizations()

        print(f"\n🎯 OPTIMIZATION SUMMARY:")

        if results['single_improvement_percent'] > 0:
            print(f"  ✅ Single predictions improved by {results['single_improvement_percent']:.1f}%")
        else:
            print(f"  ⚠️  Single predictions: {abs(results['single_improvement_percent']):.1f}% overhead")

        if results['batch_improvement_percent'] > 0:
            print(f"  ✅ Batch predictions improved by {results['batch_improvement_percent']:.1f}%")
        else:
            print(f"  ⚠️  Batch predictions: {abs(results['batch_improvement_percent']):.1f}% overhead")

        print(f"  📈 Optimized throughput: {results['optimized_single_throughput']:.0f} single/sec, "
              f"{results['optimized_batch_throughput']:.0f} batch samples/sec")

    except Exception as e:
        print(f"❌ Optimization benchmark failed: {e}")
        import traceback
        traceback.print_exc()