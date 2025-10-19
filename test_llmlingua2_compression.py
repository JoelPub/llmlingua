#!/usr/bin/env python3
"""
Compression accuracy and quality tests for llmlingua2.

This file contains tests specifically focused on the compression functionality,
including accuracy, quality, and edge cases.
"""

import sys
import os
import unittest
import time

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llmlingua2_config import LLMLingua2Config, CompressionMethod
from llmlingua2_compressor import LLMLingua2Compressor
from llmlingua2_results import CompressionResult
from llmlingua2_utils import calculate_compression_metrics

class TestCompressionAccuracy(unittest.TestCase):
    """Test cases for compression accuracy and quality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = LLMLingua2Config()
        self.compressor = LLMLingua2Compressor(self.config)
    
    def test_compression_preserves_meaning(self):
        """Test that compression preserves the core meaning of text."""
        test_cases = [
            {
                "original": "The quick brown fox jumps over the lazy dog.",
                "min_keywords": ["quick", "fox", "jumps", "dog"]
            },
            {
                "original": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
                "min_keywords": ["machine", "learning", "artificial", "intelligence", "computers"]
            },
            {
                "original": "Climate change refers to long-term shifts in temperatures and weather patterns.",
                "min_keywords": ["climate", "change", "temperatures", "weather"]
            }
        ]
        
        for case in test_cases:
            with self.subTest(case=case):
                result = self.compressor.compress(case["original"])
                compressed_lower = result.compressed_prompt.lower()
                
                # Check that at least some key words are preserved
                preserved_keywords = [
                    keyword for keyword in case["min_keywords"]
                    if keyword.lower() in compressed_lower
                ]
                
                # At least 50% of keywords should be preserved
                min_preserved = len(case["min_keywords"]) // 2
                self.assertGreaterEqual(
                    len(preserved_keywords),
                    min_preserved,
                    f"Not enough keywords preserved in: {result.compressed_prompt}"
                )
    
    def test_compression_ratios_by_method(self):
        """Test that different compression methods produce different ratios."""
        prompt = "This is a moderately long test prompt that should demonstrate different compression ratios when processed with different compression methods available in llmlingua2."
        
        results = {}
        for method in CompressionMethod:
            config = LLMLingua2Config(compression_method=method.value)
            compressor = LLMLingua2Compressor(config)
            result = compressor.compress(prompt)
            results[method] = result
        
        # Verify that aggressive compression produces higher ratios than conservative
        self.assertGreater(
            results[CompressionMethod.AGGRESSIVE].compression_ratio,
            results[CompressionMethod.CONSERVATIVE].compression_ratio
        )
        
        # Verify that all methods produce some compression
        for method, result in results.items():
            self.assertGreater(result.compression_ratio, 0, f"No compression achieved with {method.value}")
            self.assertLessEqual(result.compression_ratio, 1, f"Invalid compression ratio with {method.value}")
    
    def test_target_token_respect(self):
        """Test that compressor respects target token limits."""
        prompt = "This is a test prompt with multiple words that should be compressed to respect the target token limit specified in the configuration."
        
        target_tokens = [5, 10, 15, 20]
        
        for target in target_tokens:
            with self.subTest(target=target):
                config = LLMLingua2Config(target_token=target)
                compressor = LLMLingua2Compressor(config)
                result = compressor.compress(prompt)
                
                self.assertLessEqual(
                    result.compressed_token_count,
                    target,
                    f"Compressed result ({result.compressed_token_count}) exceeds target ({target})"
                )
    
    def test_compression_consistency(self):
        """Test that compression is consistent across multiple runs."""
        prompt = "Consistency test prompt for llmlingua2 compression functionality."
        config = LLMLingua2Config()
        
        results = []
        for _ in range(5):  # Run 5 times
            compressor = LLMLingua2Compressor(config)
            result = compressor.compress(prompt)
            results.append(result)
        
        # All results should have the same token counts (deterministic)
        token_counts = [r.compressed_token_count for r in results]
        self.assertEqual(len(set(token_counts)), 1, "Inconsistent compression results")
        
        # All results should have similar compression ratios
        ratios = [r.compression_ratio for r in results]
        ratio_variance = max(ratios) - min(ratios)
        self.assertLess(ratio_variance, 0.1, "High variance in compression ratios")
    
    def test_edge_cases(self):
        """Test compression edge cases."""
        edge_cases = [
            "",  # Empty string
            "Single",  # Single word
            "a b c d e f g",  # Single characters
            "   ",  # Only whitespace
            "This\nhas\nnewlines\n",  # With newlines
            "Special chars: !@#$%^&*()_+",  # Special characters
            "1234567890",  # Only numbers
            "UPPERCASE TEXT",  # All uppercase
            "lowercase text",  # All lowercase
            "Mixed Case Text With Numbers 123",  # Mixed case and numbers
        ]
        
        for case in edge_cases:
            with self.subTest(case=case):
                result = self.compressor.compress(case)
                
                # Basic validation
                self.assertIsInstance(result, CompressionResult)
                self.assertGreaterEqual(result.compression_ratio, 0)
                self.assertLessEqual(result.compression_ratio, 1)
                self.assertGreaterEqual(result.processing_time, 0)
                
                # For empty input, output should be empty
                if case == "":
                    self.assertEqual(result.compressed_prompt, "")
                    self.assertEqual(result.compressed_token_count, 0)
    
    def test_long_text_compression(self):
        """Test compression of long texts."""
        # Generate a long text
        long_text = "This is a sentence. " * 100  # 100 sentences
        
        config = LLMLingua2Config(target_token=50)
        compressor = LLMLingua2Compressor(config)
        result = compressor.compress(long_text)
        
        # Should compress significantly
        self.assertGreater(result.compression_ratio, 0.5)
        self.assertLessEqual(result.compressed_token_count, 50)
        
        # Processing time should be reasonable
        self.assertLess(result.processing_time, 5.0)  # Less than 5 seconds
    
    def test_short_text_compression(self):
        """Test compression of very short texts."""
        short_texts = [
            "Hi",
            "Hello world",
            "A B C",
            "Test"
        ]
        
        for text in short_texts:
            with self.subTest(text=text):
                result = self.compressor.compress(text)
                
                # For very short texts, compression might be minimal
                self.assertGreaterEqual(result.compressed_token_count, 1)
                self.assertLessEqual(result.compressed_token_count, result.original_token_count)
    
    def test_repeated_content_compression(self):
        """Test compression of text with repeated content."""
        repeated_text = "Repeat this. " * 20
        
        result = self.compressor.compress(repeated_text)
        
        # Should effectively compress repeated content
        self.assertGreater(result.compression_ratio, 0.3)
        
        # Compressed text should be much shorter
        self.assertLess(result.compressed_token_count, result.original_token_count * 0.7)
    
    def test_multilingual_content(self):
        """Test compression of multilingual content."""
        multilingual_texts = [
            "Bonjour le monde",  # French
            "Hola mundo",  # Spanish
            "Hallo Welt",  # German
            "こんにちは世界",  # Japanese
            "你好世界",  # Chinese
            "Привет мир",  # Russian
        ]
        
        for text in multilingual_texts:
            with self.subTest(text=text):
                result = self.compressor.compress(text)
                
                # Should handle non-English text without errors
                self.assertIsInstance(result, CompressionResult)
                self.assertGreaterEqual(result.compressed_token_count, 0)
                self.assertLessEqual(result.compressed_token_count, result.original_token_count)
    
    def test_technical_content(self):
        """Test compression of technical content."""
        technical_texts = [
            "The function def calculate_sum(a, b): return a + b adds two numbers.",
            "HTTP status code 200 indicates OK response.",
            "SQL query: SELECT * FROM users WHERE id = 1;",
            "Machine learning model with 0.001 learning rate.",
            "API endpoint POST /api/v1/users creates new user."
        ]
        
        for text in technical_texts:
            with self.subTest(text=text):
                result = self.compressor.compress(text)
                
                # Should preserve technical keywords
                compressed_lower = result.compressed_prompt.lower()
                technical_keywords = ["function", "http", "sql", "model", "api"]
                
                preserved_count = sum(1 for keyword in technical_keywords if keyword in compressed_lower)
                self.assertGreaterEqual(preserved_count, 1, 
                    f"No technical keywords preserved in: {result.compressed_prompt}")
    
    def test_confidence_score_reasonableness(self):
        """Test that confidence scores are reasonable."""
        prompt = "This is a test prompt for confidence score validation."
        
        results = []
        for method in CompressionMethod:
            config = LLMLingua2Config(compression_method=method.value)
            compressor = LLMLingua2Compressor(config)
            result = compressor.compress(prompt)
            results.append(result)
        
        for result in results:
            # Confidence score should be between 0 and 1
            if result.confidence_score is not None:
                self.assertGreaterEqual(result.confidence_score, 0)
                self.assertLessEqual(result.confidence_score, 1)
                
                # Higher compression should generally have lower confidence
                if result.compression_ratio > 0.6:
                    self.assertLess(result.confidence_score, 0.9, 
                        f"High confidence for aggressive compression: {result.confidence_score}")

class TestCompressionPerformance(unittest.TestCase):
    """Test cases for compression performance metrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = LLMLingua2Config()
        self.compressor = LLMLingua2Compressor(self.config)
    
    def test_processing_time_scaling(self):
        """Test that processing time scales reasonably with input size."""
        texts = [
            "Short text.",
            "This is a medium length text with more words to process.",
            "This is a longer text that contains multiple sentences and should take more time to process due to the increased number of words and complexity. " * 5,
            "This is a very long text " * 50
        ]
        
        processing_times = []
        for text in texts:
            start_time = time.time()
            result = self.compressor.compress(text)
            end_time = time.time()
            actual_time = end_time - start_time
            
            processing_times.append({
                'length': len(text),
                'tokens': result.original_token_count,
                'time': actual_time,
                'reported_time': result.processing_time
            })
        
        # Processing time should generally increase with text length
        # (but not necessarily linearly due to various optimizations)
        for i in range(1, len(processing_times)):
            self.assertGreater(processing_times[i]['time'], 0)
            self.assertGreater(processing_times[i]['reported_time'], 0)
        
        # Reported time should be close to actual time (within 10ms)
        for item in processing_times:
            time_diff = abs(item['time'] - item['reported_time'])
            self.assertLess(time_diff, 0.01, 
                f"Reported time ({item['reported_time']}) differs significantly from actual time ({item['time']})")
    
    def test_batch_processing_efficiency(self):
        """Test that batch processing is more efficient than individual processing."""
        prompts = [
            "First prompt for batch testing.",
            "Second prompt with more content.",
            "Third prompt that is longer than the previous ones.",
            "Fourth prompt for comprehensive batch testing.",
            "Fifth and final prompt in this batch test."
        ]
        
        # Time individual processing
        start_time = time.time()
        individual_results = []
        for prompt in prompts:
            result = self.compressor.compress(prompt)
            individual_results.append(result)
        individual_time = time.time() - start_time
        
        # Time batch processing
        start_time = time.time()
        batch_results = self.compressor.batch_compress(prompts, show_progress=False)
        batch_time = time.time() - start_time
        
        # Batch should be faster (or at least not significantly slower)
        # Allow some overhead for batch processing
        self.assertLess(batch_time, individual_time * 1.5,
            f"Batch processing ({batch_time:.3f}s) much slower than individual ({individual_time:.3f}s)")
        
        # Results should be equivalent
        self.assertEqual(len(batch_results), len(individual_results))
        for batch_result, individual_result in zip(batch_results, individual_results):
            self.assertEqual(batch_result.original_prompt, individual_result.original_prompt)
            self.assertEqual(batch_result.compressed_token_count, individual_result.compressed_token_count)
    
    def test_memory_usage_large_batch(self):
        """Test memory usage with large batch processing."""
        # Create a large batch of prompts
        prompts = ["Test prompt number " + str(i) for i in range(1000)]
        
        # This should not cause memory issues
        try:
            results = self.compressor.batch_compress(prompts, show_progress=False)
            self.assertEqual(len(results), len(prompts))
            
            # Verify all results are valid
            for result in results:
                self.assertIsInstance(result, CompressionResult)
                self.assertGreaterEqual(result.compression_ratio, 0)
                self.assertLessEqual(result.compression_ratio, 1)
                
        except MemoryError:
            self.fail("MemoryError during large batch processing")
    
    def test_compression_quality_vs_speed(self):
        """Test the trade-off between compression quality and speed."""
        prompt = "This is a test prompt to evaluate the trade-off between compression quality and processing speed in llmlingua2."
        
        quality_results = []
        for method in CompressionMethod:
            config = LLMLingua2Config(compression_method=method.value)
            compressor = LLMLingua2Compressor(config)
            
            # Measure time
            start_time = time.time()
            result = compressor.compress(prompt)
            end_time = time.time()
            actual_time = end_time - start_time
            
            quality_results.append({
                'method': method.value,
                'compression_ratio': result.compression_ratio,
                'processing_time': actual_time,
                'confidence': result.confidence_score or 0,
                'efficiency': result.calculate_efficiency_score()
            })
        
        # Generally, more aggressive compression should be faster but lower quality
        aggressive = next(r for r in quality_results if r['method'] == 'aggressive')
        conservative = next(r for r in quality_results if r['method'] == 'conservative')
        
        # Aggressive should generally be faster
        self.assertLess(aggressive['processing_time'], conservative['processing_time'] * 2,
            "Aggressive compression not significantly faster than conservative")
        
        # Conservative should generally have higher confidence
        self.assertGreater(conservative['confidence'], aggressive['confidence'] * 0.8,
            "Conservative compression doesn't maintain higher confidence")

class TestCompressionMetrics(unittest.TestCase):
    """Test cases for compression metrics and calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = LLMLingua2Config()
        self.compressor = LLMLingua2Compressor(self.config)
    
    def test_compression_metrics_calculation(self):
        """Test compression metrics calculation accuracy."""
        original = "The quick brown fox jumps over the lazy dog."
        compressed = "Quick fox jumps dog."
        
        metrics = calculate_compression_metrics(original, compressed)
        
        # Verify all expected metrics are present
        expected_metrics = [
            'original_char_count', 'compressed_char_count',
            'original_token_count', 'compressed_token_count',
            'char_compression_ratio', 'token_compression_ratio',
            'space_saving_percent'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Verify metric values are reasonable
        self.assertGreater(metrics['original_char_count'], metrics['compressed_char_count'])
        self.assertGreater(metrics['original_token_count'], metrics['compressed_token_count'])
        self.assertGreater(metrics['char_compression_ratio'], 0)
        self.assertGreater(metrics['token_compression_ratio'], 0)
        self.assertGreater(metrics['space_saving_percent'], 0)
        
        # Verify ratios are between 0 and 1
        self.assertGreaterEqual(metrics['char_compression_ratio'], 0)
        self.assertLessEqual(metrics['char_compression_ratio'], 1)
        self.assertGreaterEqual(metrics['token_compression_ratio'], 0)
        self.assertLessEqual(metrics['token_compression_ratio'], 1)
    
    def test_efficiency_score_calculation(self):
        """Test efficiency score calculation."""
        # Create test results with different characteristics
        high_quality_result = CompressionResult(
            original_prompt="Test",
            compressed_prompt="Test",
            original_token_count=100,
            compressed_token_count=60,
            compression_ratio=0.4,
            processing_time=0.2,
            method_used="balanced",
            model_used="test"
        )
        
        fast_result = CompressionResult(
            original_prompt="Test",
            compressed_prompt="Test",
            original_token_count=100,
            compressed_token_count=70,
            compression_ratio=0.3,
            processing_time=0.05,
            method_used="fast",
            model_used="test"
        )
        
        # Calculate efficiency scores
        high_quality_score = high_quality_result.calculate_efficiency_score()
        fast_score = fast_result.calculate_efficiency_score()
        
        # Scores should be between 0 and 1
        self.assertGreaterEqual(high_quality_score, 0)
        self.assertLessEqual(high_quality_score, 1)
        self.assertGreaterEqual(fast_score, 0)
        self.assertLessEqual(fast_score, 1)
        
        # High quality should generally have better efficiency
        # (since compression ratio is weighted more heavily)
        self.assertGreaterEqual(high_quality_score, fast_score * 0.8)
    
    def test_result_comparison_accuracy(self):
        """Test result comparison functionality."""
        # Create test results
        result_a = CompressionResult(
            original_prompt="Test",
            compressed_prompt="Test A",
            original_token_count=100,
            compressed_token_count=50,
            compression_ratio=0.5,
            processing_time=0.1,
            method_used="method_a",
            model_used="test"
        )
        
        result_b = CompressionResult(
            original_prompt="Test",
            compressed_prompt="Test B",
            original_token_count=100,
            compressed_token_count=40,
            compression_ratio=0.6,
            processing_time=0.15,
            method_used="method_b",
            model_used="test"
        )
        
        # Test ratio-prioritized comparison
        self.assertTrue(result_b.is_better_than(result_a, prioritize_ratio=True))
        self.assertFalse(result_a.is_better_than(result_b, prioritize_ratio=True))
        
        # Test speed-prioritized comparison
        self.assertTrue(result_a.is_better_than(result_b, prioritize_ratio=False))
        self.assertFalse(result_b.is_better_than(result_a, prioritize_ratio=False))
        
        # Test equal results
        self.assertFalse(result_a.is_better_than(result_a))
        self.assertFalse(result_b.is_better_than(result_b))

if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
