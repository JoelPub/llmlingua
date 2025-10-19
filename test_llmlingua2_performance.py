#!/usr/bin/env python3
"""
Performance benchmark tests for llmlingua2.

This file contains performance benchmarks and validation tests
to ensure llmlingua2 meets performance requirements.
"""

import sys
import os
import unittest
import time
import statistics
import tracemalloc
from typing import List, Dict, Any

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llmlingua2_config import LLMLingua2Config, CompressionMethod
from llmlingua2_compressor import LLMLingua2Compressor
from llmlingua2_results import CompressionResult

class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests for llmlingua2."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = LLMLingua2Config()
        self.compressor = LLMLingua2Compressor(self.config)
        
        # Test data sets
        self.short_prompts = [
            "Hello world",
            "Test prompt",
            "Quick compression",
            "Short text",
            "AI is amazing"
        ]
        
        self.medium_prompts = [
            "This is a medium length prompt that contains multiple sentences and should demonstrate the compression capabilities of llmlingua2 effectively.",
            "Machine learning algorithms can process large amounts of data to identify patterns and make predictions with remarkable accuracy.",
            "Natural language processing enables computers to understand, interpret, and generate human language in a meaningful way.",
            "The advancement of artificial intelligence has transformed many industries and continues to evolve rapidly.",
            "Data science combines statistics, programming, and domain expertise to extract insights from complex datasets."
        ]
        
        self.long_prompts = [
            "Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines that work and react like humans. AI is interdisciplinary in nature, with applications in various fields such as healthcare, finance, transportation, and education. Machine learning, a subset of AI, focuses on the development of algorithms that can learn and improve from experience without being explicitly programmed. Deep learning, a further subset, uses neural networks with multiple layers to analyze various forms of data. " * 3,
            "Climate change represents one of the most pressing challenges of our time. The scientific consensus is clear: human activities, particularly the burning of fossil fuels and deforestation, are the primary drivers of global warming. The consequences of climate change are far-reaching, including rising sea levels, extreme weather events, loss of biodiversity, and disruptions to agriculture and food systems. Addressing this crisis requires coordinated global action, technological innovation, and changes in individual and collective behavior. " * 2,
            "The internet has revolutionized how we communicate, work, and access information. From its origins as a military and academic network, the internet has grown into a global infrastructure that connects billions of devices and users worldwide. The rise of social media platforms, e-commerce, streaming services, and cloud computing has transformed business models and created new opportunities for innovation and connection. However, this digital revolution also brings challenges related to privacy, security, digital divide, and the spread of misinformation. " * 2
        ]
    
    def test_compression_speed_benchmarks(self):
        """Test compression speed benchmarks."""
        test_cases = [
            ("Short prompts", self.short_prompts, 0.1),  # Max 0.1s per prompt
            ("Medium prompts", self.medium_prompts, 0.5),  # Max 0.5s per prompt
            ("Long prompts", self.long_prompts, 2.0)  # Max 2s per prompt
        ]
        
        for case_name, prompts, max_time_per_prompt in test_cases:
            with self.subTest(case=case_name):
                processing_times = []
                
                for prompt in prompts:
                    start_time = time.time()
                    result = self.compressor.compress(prompt)
                    end_time = time.time()
                    
                    processing_time = end_time - start_time
                    processing_times.append(processing_time)
                    
                    # Individual prompt should not exceed max time
                    self.assertLess(
                        processing_time,
                        max_time_per_prompt,
                        f"Prompt processing time ({processing_time:.3f}s) exceeds maximum ({max_time_per_prompt}s)"
                    )
                
                # Average processing time should be reasonable
                avg_time = statistics.mean(processing_times)
                self.assertLess(
                    avg_time,
                    max_time_per_prompt * 0.8,
                    f"Average processing time ({avg_time:.3f}s) too high for {case_name}"
                )
                
                print(f"\n{case_name} Performance:")
                print(f"  Average time: {avg_time:.3f}s")
                print(f"  Max time: {max(processing_times):.3f}s")
                print(f"  Min time: {min(processing_times):.3f}s")
    
    def test_batch_processing_performance(self):
        """Test batch processing performance."""
        batch_sizes = [10, 50, 100, 500]
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                # Create batch of prompts
                prompts = [f"Batch test prompt {i} with some content to compress." 
                          for i in range(batch_size)]
                
                # Time batch processing
                start_time = time.time()
                results = self.compressor.batch_compress(prompts, show_progress=False)
                batch_time = time.time() - start_time
                
                # Verify all results
                self.assertEqual(len(results), batch_size)
                
                # Calculate average time per prompt
                avg_time_per_prompt = batch_time / batch_size
                
                # Batch processing should be efficient
                max_avg_time = 0.05  # 50ms per prompt average
                self.assertLess(
                    avg_time_per_prompt,
                    max_avg_time,
                    f"Batch processing too slow: {avg_time_per_prompt:.3f}s per prompt"
                )
                
                print(f"\nBatch size {batch_size}:")
                print(f"  Total time: {batch_time:.3f}s")
                print(f"  Avg per prompt: {avg_time_per_prompt:.3f}s")
    
    def test_memory_usage_benchmarks(self):
        """Test memory usage benchmarks."""
        # Start memory tracing
        tracemalloc.start()
        
        try:
            # Test with large batch
            large_batch = [f"Memory test prompt {i} " * 20 for i in range(1000)]
            
            # Measure memory before
            snapshot1 = tracemalloc.take_snapshot()
            
            # Process batch
            results = self.compressor.batch_compress(large_batch, show_progress=False)
            
            # Measure memory after
            snapshot2 = tracemalloc.take_snapshot()
            
            # Calculate memory difference
            stats = snapshot2.compare_to(snapshot1, 'lineno')
            total_memory_increase = sum(stat.size_diff for stat in stats)
            
            # Memory increase should be reasonable (less than 100MB for 1000 prompts)
            max_memory_increase = 100 * 1024 * 1024  # 100MB
            self.assertLess(
                total_memory_increase,
                max_memory_increase,
                f"Memory usage too high: {total_memory_increase / (1024*1024):.1f}MB"
            )
            
            print(f"\nMemory usage for 1000 prompts:")
            print(f"  Total increase: {total_memory_increase / (1024*1024):.2f}MB")
            print(f"  Per prompt: {total_memory_increase / 1000 / 1024:.2f}KB")
            
        finally:
            tracemalloc.stop()
    
    def test_scalability_performance(self):
        """Test scalability with increasing input sizes."""
        base_prompt = "This is a scalability test prompt. "
        multipliers = [1, 10, 50, 100, 500]
        
        processing_times = []
        memory_usages = []
        
        for multiplier in multipliers:
            with self.subTest(multiplier=multiplier):
                # Create prompt of varying length
                prompt = base_prompt * multiplier
                
                # Measure processing time
                start_time = time.time()
                result = self.compressor.compress(prompt)
                processing_time = time.time() - start_time
                
                processing_times.append(processing_time)
                
                # Verify result
                self.assertIsInstance(result, CompressionResult)
                self.assertGreater(result.compression_ratio, 0)
                
                print(f"\nMultiplier {multiplier} (length: {len(prompt)} chars):")
                print(f"  Processing time: {processing_time:.3f}s")
                print(f"  Compression ratio: {result.compression_ratio:.3f}")
        
        # Processing time should scale sub-linearly
        # (due to optimizations in the compression algorithm)
        for i in range(1, len(processing_times)):
            time_ratio = processing_times[i] / processing_times[i-1]
            multiplier_ratio = multipliers[i] / multipliers[i-1]
            
            # Time increase should be less than multiplier increase
            self.assertLess(
                time_ratio,
                multiplier_ratio * 1.5,  # Allow 50% overhead
                f"Processing time doesn't scale well: {time_ratio:.2f}x vs {multiplier_ratio:.2f}x"
                )
    
    def test_concurrent_processing_performance(self):
        """Test performance with concurrent processing simulation."""
        import threading
        import queue
        
        def worker(prompt_queue, result_queue, worker_id):
            """Worker function for concurrent processing."""
            while True:
                try:
                    prompt = prompt_queue.get_nowait()
                    result = self.compressor.compress(prompt)
                    result_queue.put((worker_id, result))
                except queue.Empty:
                    break
        
        # Create test prompts
        num_prompts = 100
        prompts = [f"Concurrent test prompt {i}" for i in range(num_prompts)]
        
        # Create queues
        prompt_queue = queue.Queue()
        result_queue = queue.Queue()
        
        # Add prompts to queue
        for prompt in prompts:
            prompt_queue.put(prompt)
        
        # Start timing
        start_time = time.time()
        
        # Create and start worker threads
        num_workers = 4
        workers = []
        for i in range(num_workers):
            worker_thread = threading.Thread(
                target=worker, 
                args=(prompt_queue, result_queue, i)
            )
            worker_thread.start()
            workers.append(worker_thread)
        
        # Wait for all workers to complete
        for worker in workers:
            worker.join()
        
        # End timing
        total_time = time.time() - start_time
        
        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        # Verify all prompts were processed
        self.assertEqual(len(results), num_prompts)
        
        # Calculate performance metrics
        avg_time_per_prompt = total_time / num_prompts
        
        print(f"\nConcurrent processing ({num_workers} workers):")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Avg per prompt: {avg_time_per_prompt:.3f}s")
        print(f"  Throughput: {num_prompts / total_time:.1f} prompts/s")
        
        # Concurrent processing should be efficient
        self.assertLess(
            avg_time_per_prompt,
            0.1,  # 100ms per prompt
            "Concurrent processing too slow"
        )
    
    def test_compression_quality_vs_performance(self):
        """Test the trade-off between compression quality and performance."""
        prompt = "This is a comprehensive test prompt designed to evaluate the trade-off between compression quality and processing performance in llmlingua2. The prompt contains multiple sentences with various types of content to ensure a thorough evaluation of the compression algorithm's effectiveness."
        
        quality_results = []
        
        for method in CompressionMethod:
            config = LLMLingua2Config(compression_method=method.value)
            compressor = LLMLingua2Compressor(config)
            
            # Measure performance
            start_time = time.time()
            result = compressor.compress(prompt)
            processing_time = time.time() - start_time
            
            quality_results.append({
                'method': method.value,
                'compression_ratio': result.compression_ratio,
                'processing_time': processing_time,
                'confidence': result.confidence_score or 0,
                'efficiency': result.calculate_efficiency_score()
            })
        
        # Analyze trade-offs
        print(f"\nQuality vs Performance Trade-off:")
        for result in quality_results:
            print(f"  {result['method']}:")
            print(f"    Compression ratio: {result['compression_ratio']:.3f}")
            print(f"    Processing time: {result['processing_time']:.3f}s")
            print(f"    Confidence: {result['confidence']:.3f}")
            print(f"    Efficiency: {result['efficiency']:.3f}")
        
        # Verify that aggressive methods are faster
        aggressive = next(r for r in quality_results if r['method'] == 'aggressive')
        conservative = next(r for r in quality_results if r['method'] == 'conservative')
        
        self.assertLess(
            aggressive['processing_time'],
            conservative['processing_time'] * 2,
            "Aggressive method not significantly faster"
        )
        
        # Verify that conservative methods have higher confidence
        self.assertGreater(
            conservative['confidence'],
            aggressive['confidence'] * 0.8,
            "Conservative method doesn't maintain higher confidence"
        )

class TestPerformanceValidation(unittest.TestCase):
    """Performance validation tests against requirements."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = LLMLingua2Config()
        self.compressor = LLMLingua2Compressor(self.config)
    
    def test_minimum_compression_ratio(self):
        """Test that minimum compression ratio requirements are met."""
        test_prompts = [
            "Short prompt for testing minimum compression requirements.",
            "This is a medium length prompt that should demonstrate adequate compression capabilities when processed by llmlingua2 compression algorithms.",
            "This is a much longer prompt designed to test the compression capabilities of llmlingua2 with substantial content that can be effectively compressed while maintaining the core meaning and important information from the original text. " * 3
        ]
        
        for prompt in test_prompts:
            with self.subTest(length=len(prompt)):
                result = self.compressor.compress(prompt)
                
                # Minimum compression ratio should be achieved
                min_ratio = 0.1  # At least 10% compression
                self.assertGreater(
                    result.compression_ratio,
                    min_ratio,
                    f"Compression ratio ({result.compression_ratio:.3f}) below minimum ({min_ratio})"
                )
    
    def test_maximum_processing_time(self):
        """Test that maximum processing time requirements are met."""
        # Test with various prompt lengths
        test_cases = [
            ("Short", "Short prompt", 0.05),  # Max 50ms
            ("Medium", "This is a medium length prompt with multiple sentences that should be processed efficiently by the compression algorithm.", 0.2),  # Max 200ms
            ("Long", "This is a long prompt " * 50, 1.0)  # Max 1s
        ]
        
        for case_name, prompt, max_time in test_cases:
            with self.subTest(case=case_name):
                start_time = time.time()
                result = self.compressor.compress(prompt)
                processing_time = time.time() - start_time
                
                self.assertLess(
                    processing_time,
                    max_time,
                    f"Processing time ({processing_time:.3f}s) exceeds maximum ({max_time}s)"
                )
    
    def test_memory_efficiency(self):
        """Test memory efficiency requirements."""
        import gc
        
        # Force garbage collection before test
        gc.collect()
        
        # Start memory tracing
        tracemalloc.start()
        
        try:
            # Take initial snapshot
            snapshot1 = tracemalloc.take_snapshot()
            
            # Process a large batch
            large_batch = [f"Memory efficiency test {i} " * 10 for i in range(500)]
            results = self.compressor.batch_compress(large_batch, show_progress=False)
            
            # Take final snapshot
            snapshot2 = tracemalloc.take_snapshot()
            
            # Calculate memory usage
            stats = snapshot2.compare_to(snapshot1, 'lineno')
            total_memory = sum(stat.size_diff for stat in stats)
            
            # Memory per prompt should be reasonable
            memory_per_prompt = total_memory / len(large_batch)
            max_memory_per_prompt = 50 * 1024  # 50KB per prompt
            
            self.assertLess(
                memory_per_prompt,
                max_memory_per_prompt,
                f"Memory usage per prompt too high: {memory_per_prompt / 1024:.1f}KB"
            )
            
            print(f"\nMemory efficiency:")
            print(f"  Total memory: {total_memory / (1024*1024):.2f}MB")
            print(f"  Per prompt: {memory_per_prompt / 1024:.1f}KB")
            
        finally:
            tracemalloc.stop()
    
    def test_throughput_requirements(self):
        """Test throughput requirements."""
        num_prompts = 100
        prompts = [f"Throughput test prompt {i}" for i in range(num_prompts)]
        
        # Measure batch processing throughput
        start_time = time.time()
        results = self.compressor.batch_compress(prompts, show_progress=False)
        total_time = time.time() - start_time
        
        # Calculate throughput
        throughput = num_prompts / total_time
        
        # Minimum throughput requirement
        min_throughput = 10  # 10 prompts per second
        self.assertGreater(
            throughput,
            min_throughput,
            f"Throughput ({throughput:.1f} prompts/s) below minimum ({min_throughput})"
        )
        
        print(f"\nThroughput performance:")
        print(f"  Total prompts: {num_prompts}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Throughput: {throughput:.1f} prompts/s")
    
    def test_scalability_requirements(self):
        """Test scalability requirements."""
        base_prompt = "Scalability test prompt. "
        sizes = [10, 50, 100, 500]
        
        processing_times = []
        
        for size in sizes:
            prompt = base_prompt * size
            
            start_time = time.time()
            result = self.compressor.compress(prompt)
            processing_time = time.time() - start_time
            
            processing_times.append(processing_time)
        
        # Check that processing time scales reasonably
        # (should not be exponential)
        for i in range(1, len(processing_times)):
            size_ratio = sizes[i] / sizes[i-1]
            time_ratio = processing_times[i] / processing_times[i-1]
            
            # Time increase should be proportional to size increase
            # (allow some overhead for larger inputs)
            max_acceptable_ratio = size_ratio * 1.5
            
            self.assertLess(
                time_ratio,
                max_acceptable_ratio,
                f"Processing time doesn't scale well from size {sizes[i-1]} to {sizes[i]}"
            )
        
        print(f"\nScalability performance:")
        for size, time_taken in zip(sizes, processing_times):
            print(f"  Size {size}: {time_taken:.3f}s")

class TestPerformanceRegression(unittest.TestCase):
    """Performance regression tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = LLMLingua2Config()
        self.compressor = LLMLingua2Compressor(self.config)
        
        # Baseline performance metrics (these should be updated as the system evolves)
        self.baseline_metrics = {
            'short_prompt_time': 0.01,  # 10ms for short prompt
            'medium_prompt_time': 0.05,  # 50ms for medium prompt
            'long_prompt_time': 0.2,  # 200ms for long prompt
            'batch_throughput': 20,  # 20 prompts/second for batch processing
            'memory_per_prompt': 10240  # 10KB per prompt
        }
    
    def test_short_prompt_regression(self):
        """Test performance regression for short prompts."""
        prompt = "Short regression test prompt"
        
        # Warm up
        self.compressor.compress(prompt)
        
        # Measure performance
        times = []
        for _ in range(10):
            start_time = time.time()
            result = self.compressor.compress(prompt)
            processing_time = time.time() - start_time
            times.append(processing_time)
        
        avg_time = statistics.mean(times)
        
        # Should not be significantly worse than baseline
        max_acceptable_time = self.baseline_metrics['short_prompt_time'] * 2
        
        self.assertLess(
            avg_time,
            max_acceptable_time,
            f"Short prompt performance regression: {avg_time:.3f}s vs baseline {self.baseline_metrics['short_prompt_time']}s"
        )
        
        print(f"\nShort prompt performance: {avg_time:.3f}s (baseline: {self.baseline_metrics['short_prompt_time']}s)")
    
    def test_batch_throughput_regression(self):
        """Test batch throughput regression."""
        num_prompts = 50
        prompts = [f"Batch regression test {i}" for i in range(num_prompts)]
        
        # Measure batch processing
        start_time = time.time()
        results = self.compressor.batch_compress(prompts, show_progress=False)
        total_time = time.time() - start_time
        
        # Calculate throughput
        throughput = num_prompts / total_time
        
        # Should not be significantly worse than baseline
        min_acceptable_throughput = self.baseline_metrics['batch_throughput'] * 0.5
        
        self.assertGreater(
            throughput,
            min_acceptable_throughput,
            f"Batch throughput regression: {throughput:.1f} vs baseline {self.baseline_metrics['batch_throughput']}"
        )
        
        print(f"\nBatch throughput: {throughput:.1f} prompts/s (baseline: {self.baseline_metrics['batch_throughput']})")
    
    def test_memory_usage_regression(self):
        """Test memory usage regression."""
        import gc
        
        gc.collect()
        tracemalloc.start()
        
        try:
            snapshot1 = tracemalloc.take_snapshot()
            
            # Process batch
            prompts = [f"Memory regression test {i}" for i in range(100)]
            results = self.compressor.batch_compress(prompts, show_progress=False)
            
            snapshot2 = tracemalloc.take_snapshot()
            
            # Calculate memory usage
            stats = snapshot2.compare_to(snapshot1, 'lineno')
            total_memory = sum(stat.size_diff for stat in stats)
            memory_per_prompt = total_memory / len(prompts)
            
            # Should not be significantly worse than baseline
            max_acceptable_memory = self.baseline_metrics['memory_per_prompt'] * 2
            
            self.assertLess(
                memory_per_prompt,
                max_acceptable_memory,
                f"Memory usage regression: {memory_per_prompt:.0f}B vs baseline {self.baseline_metrics['memory_per_prompt']}B"
            )
            
            print(f"\nMemory usage: {memory_per_prompt:.0f}B/prompt (baseline: {self.baseline_metrics['memory_per_prompt']}B)")
            
        finally:
            tracemalloc.stop()

if __name__ == '__main__':
    # Run all performance tests
    unittest.main(verbosity=2)
