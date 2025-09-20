#!/usr/bin/env python3
"""Performance benchmarking and optimization tests for SAFLA trading system."""

import gc
import time
import threading
import psutil
import os
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor
import torch

from safla_trading.config import Config
from safla_trading.core.neural_coordinator import NeuralCoordinator
from safla_trading.core.safla_neural import SAFLANeuralNetwork
from safla_trading.memory import EpisodicMemory, VectorMemory, EpisodicMemoryEntry, VectorMemoryEntry


class PerformanceBenchmark:
    """Performance benchmarking suite."""

    def __init__(self):
        self.config = Config()
        self.results: Dict[str, Dict[str, Any]] = {}

    def measure_time(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Measure execution time and system resources."""
        process = psutil.Process(os.getpid())

        # Initial measurements
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_cpu_percent = process.cpu_percent()
        start_time = time.perf_counter()

        # Execute function
        result = func(*args, **kwargs)

        # Final measurements
        end_time = time.perf_counter()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        end_cpu_percent = process.cpu_percent()

        return {
            'result': result,
            'execution_time': end_time - start_time,
            'memory_used': end_memory - start_memory,
            'cpu_usage': (start_cpu_percent + end_cpu_percent) / 2,
            'peak_memory': end_memory
        }

    def benchmark_neural_network_inference(self, batch_sizes: List[int] = None) -> Dict[str, Any]:
        """Benchmark neural network inference performance."""
        if batch_sizes is None:
            batch_sizes = [1, 8, 32, 64]

        print("Benchmarking neural network inference...")

        network = SAFLANeuralNetwork(
            input_dim=self.config.neural_input_dim,
            hidden_dim=self.config.neural_hidden_dim,
            output_dim=self.config.neural_output_dim
        )

        results = {}

        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, self.config.neural_input_dim)

            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = network(input_tensor)

            # Benchmark
            def inference():
                with torch.no_grad():
                    return network(input_tensor)

            perf = self.measure_time(inference)

            results[f'batch_{batch_size}'] = {
                'batch_size': batch_size,
                'time_per_sample': perf['execution_time'] / batch_size,
                'throughput': batch_size / perf['execution_time'],
                'memory_mb': perf['memory_used'],
                'total_time': perf['execution_time']
            }

            print(f"  Batch {batch_size}: {perf['execution_time']:.4f}s "
                  f"({batch_size/perf['execution_time']:.1f} samples/sec)")

        return results

    def benchmark_memory_operations(self, num_operations: int = 1000) -> Dict[str, Any]:
        """Benchmark memory system operations."""
        print(f"Benchmarking memory operations ({num_operations} ops)...")

        # Episodic memory benchmark
        episodic = EpisodicMemory()
        entries = []

        def store_episodic_entries():
            for i in range(num_operations):
                entry = EpisodicMemoryEntry(
                    context={'iteration': i, 'data': f'benchmark_{i}'},
                    action=f'action_{i}',
                    outcome={'success': i % 2 == 0, 'value': i * 0.1}
                )
                entry_id = episodic.store(entry)
                entries.append(entry_id)

        episodic_store_perf = self.measure_time(store_episodic_entries)

        def retrieve_episodic_entries():
            retrieved = []
            for entry_id in entries:
                retrieved.append(episodic.retrieve(entry_id))
            return retrieved

        episodic_retrieve_perf = self.measure_time(retrieve_episodic_entries)

        # Vector memory benchmark
        vector = VectorMemory(dimension=128)
        vector_entries = []

        def store_vector_entries():
            for i in range(num_operations):
                vector_data = [float(j + i) for j in range(128)]
                entry = VectorMemoryEntry(
                    vector=vector_data,
                    metadata={'index': i, 'type': 'benchmark'}
                )
                entry_id = vector.store(entry)
                vector_entries.append(entry_id)

        vector_store_perf = self.measure_time(store_vector_entries)

        def search_vector_entries():
            query = [1.0] * 128
            results = []
            for _ in range(100):  # 100 searches
                results.extend(vector.search(query, limit=5))
            return results

        vector_search_perf = self.measure_time(search_vector_entries)

        return {
            'episodic': {
                'store_ops_per_sec': num_operations / episodic_store_perf['execution_time'],
                'retrieve_ops_per_sec': num_operations / episodic_retrieve_perf['execution_time'],
                'store_time_per_op': episodic_store_perf['execution_time'] / num_operations,
                'retrieve_time_per_op': episodic_retrieve_perf['execution_time'] / num_operations,
                'memory_mb': episodic_store_perf['memory_used']
            },
            'vector': {
                'store_ops_per_sec': num_operations / vector_store_perf['execution_time'],
                'search_ops_per_sec': 100 / vector_search_perf['execution_time'],
                'store_time_per_op': vector_store_perf['execution_time'] / num_operations,
                'search_time_per_op': vector_search_perf['execution_time'] / 100,
                'memory_mb': vector_store_perf['memory_used']
            }
        }

    def benchmark_concurrent_access(self, num_threads: int = 4, ops_per_thread: int = 100) -> Dict[str, Any]:
        """Benchmark concurrent access performance."""
        print(f"Benchmarking concurrent access ({num_threads} threads, {ops_per_thread} ops each)...")

        coordinator = NeuralCoordinator(
            input_dim=self.config.neural_input_dim,
            output_dim=self.config.neural_output_dim
        )

        results = []
        errors = []

        def worker_task(worker_id: int):
            worker_times = []
            try:
                for i in range(ops_per_thread):
                    start = time.perf_counter()

                    input_tensor = torch.randn(1, self.config.neural_input_dim)
                    result, aux = coordinator.predict(input_tensor)

                    end = time.perf_counter()
                    worker_times.append(end - start)

                results.append((worker_id, worker_times))
            except Exception as e:
                errors.append((worker_id, str(e)))

        # Run concurrent benchmark
        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_task, i) for i in range(num_threads)]
            for future in futures:
                future.result()

        end_time = time.perf_counter()

        # Analyze results
        all_times = []
        for worker_id, times in results:
            all_times.extend(times)

        total_operations = len(all_times)
        total_time = end_time - start_time

        return {
            'total_operations': total_operations,
            'total_time': total_time,
            'operations_per_second': total_operations / total_time,
            'average_latency': sum(all_times) / len(all_times) if all_times else 0,
            'min_latency': min(all_times) if all_times else 0,
            'max_latency': max(all_times) if all_times else 0,
            'error_count': len(errors),
            'success_rate': (total_operations - len(errors)) / total_operations if total_operations > 0 else 0
        }

    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns."""
        print("Benchmarking memory usage patterns...")

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create components and measure memory
        coordinator = NeuralCoordinator(
            input_dim=self.config.neural_input_dim,
            output_dim=self.config.neural_output_dim
        )
        coordinator_memory = process.memory_info().rss / 1024 / 1024 - initial_memory

        episodic = EpisodicMemory()
        vector = VectorMemory()
        memory_systems_memory = process.memory_info().rss / 1024 / 1024 - initial_memory - coordinator_memory

        # Load with data and measure growth
        for i in range(1000):
            # Add episodic entries
            entry = EpisodicMemoryEntry(
                context={'load_test': i},
                action=f'load_action_{i}',
                outcome={'result': i}
            )
            episodic.store(entry)

            # Add vector entries
            vector_entry = VectorMemoryEntry(
                vector=[float(j + i) for j in range(128)],
                metadata={'load_index': i}
            )
            vector.store(vector_entry)

        loaded_memory = process.memory_info().rss / 1024 / 1024
        data_memory = loaded_memory - initial_memory - coordinator_memory - memory_systems_memory

        # Force garbage collection and measure
        gc.collect()
        gc_memory = process.memory_info().rss / 1024 / 1024

        return {
            'initial_memory_mb': initial_memory,
            'coordinator_memory_mb': coordinator_memory,
            'memory_systems_mb': memory_systems_memory,
            'data_memory_mb': data_memory,
            'total_memory_mb': loaded_memory,
            'after_gc_mb': gc_memory,
            'memory_efficiency': (data_memory / loaded_memory) if loaded_memory > 0 else 0
        }

    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark."""
        print("Starting comprehensive performance benchmark...\n")

        # Run all benchmarks
        neural_results = self.benchmark_neural_network_inference()
        memory_results = self.benchmark_memory_operations()
        concurrent_results = self.benchmark_concurrent_access()
        memory_usage_results = self.benchmark_memory_usage()

        comprehensive_results = {
            'neural_inference': neural_results,
            'memory_operations': memory_results,
            'concurrent_access': concurrent_results,
            'memory_usage': memory_usage_results,
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
                'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
                'torch_version': torch.__version__
            }
        }

        self.results = comprehensive_results
        return comprehensive_results

    def print_performance_report(self):
        """Print a formatted performance report."""
        if not self.results:
            print("No benchmark results available. Run benchmark first.")
            return

        print("\n" + "="*60)
        print("SAFLA TRADING SYSTEM PERFORMANCE REPORT")
        print("="*60)

        # Neural inference performance
        print("\n🧠 NEURAL INFERENCE PERFORMANCE:")
        neural = self.results['neural_inference']
        for batch_key, data in neural.items():
            print(f"  {batch_key.replace('_', ' ').title()}: "
                  f"{data['throughput']:.1f} samples/sec, "
                  f"{data['time_per_sample']*1000:.2f}ms per sample")

        # Memory operations performance
        print("\n💾 MEMORY OPERATIONS PERFORMANCE:")
        memory = self.results['memory_operations']
        print(f"  Episodic Store: {memory['episodic']['store_ops_per_sec']:.0f} ops/sec")
        print(f"  Episodic Retrieve: {memory['episodic']['retrieve_ops_per_sec']:.0f} ops/sec")
        print(f"  Vector Store: {memory['vector']['store_ops_per_sec']:.0f} ops/sec")
        print(f"  Vector Search: {memory['vector']['search_ops_per_sec']:.0f} ops/sec")

        # Concurrent access performance
        print("\n🔄 CONCURRENT ACCESS PERFORMANCE:")
        concurrent = self.results['concurrent_access']
        print(f"  Throughput: {concurrent['operations_per_second']:.1f} ops/sec")
        print(f"  Average Latency: {concurrent['average_latency']*1000:.2f}ms")
        print(f"  Success Rate: {concurrent['success_rate']*100:.1f}%")

        # Memory usage
        print("\n📊 MEMORY USAGE:")
        usage = self.results['memory_usage']
        print(f"  Coordinator: {usage['coordinator_memory_mb']:.1f} MB")
        print(f"  Memory Systems: {usage['memory_systems_mb']:.1f} MB")
        print(f"  Data Storage: {usage['data_memory_mb']:.1f} MB")
        print(f"  Total: {usage['total_memory_mb']:.1f} MB")
        print(f"  Memory Efficiency: {usage['memory_efficiency']*100:.1f}%")

        # System info
        print("\n🖥️  SYSTEM INFO:")
        sys_info = self.results['system_info']
        print(f"  CPU Cores: {sys_info['cpu_count']}")
        print(f"  Total Memory: {sys_info['memory_total_gb']:.1f} GB")
        print(f"  Python: {sys_info['python_version']}")
        print(f"  PyTorch: {sys_info['torch_version']}")

        print("\n" + "="*60)


def main():
    """Run performance benchmarks."""
    benchmark = PerformanceBenchmark()

    try:
        results = benchmark.run_comprehensive_benchmark()
        benchmark.print_performance_report()

        # Performance assessment
        print("\n📈 PERFORMANCE ASSESSMENT:")

        # Neural inference assessment
        batch_1_throughput = results['neural_inference']['batch_1']['throughput']
        if batch_1_throughput > 1000:
            print("  ✅ Neural inference: Excellent (>1000 samples/sec)")
        elif batch_1_throughput > 500:
            print("  ⚠️  Neural inference: Good (>500 samples/sec)")
        else:
            print("  ❌ Neural inference: Needs optimization (<500 samples/sec)")

        # Memory operations assessment
        episodic_ops = results['memory_operations']['episodic']['store_ops_per_sec']
        if episodic_ops > 10000:
            print("  ✅ Memory operations: Excellent (>10k ops/sec)")
        elif episodic_ops > 5000:
            print("  ⚠️  Memory operations: Good (>5k ops/sec)")
        else:
            print("  ❌ Memory operations: Needs optimization (<5k ops/sec)")

        # Concurrent performance assessment
        concurrent_throughput = results['concurrent_access']['operations_per_second']
        if concurrent_throughput > 100:
            print("  ✅ Concurrent access: Excellent (>100 ops/sec)")
        elif concurrent_throughput > 50:
            print("  ⚠️  Concurrent access: Good (>50 ops/sec)")
        else:
            print("  ❌ Concurrent access: Needs optimization (<50 ops/sec)")

        print("\n✅ Performance benchmark completed successfully!")

    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()