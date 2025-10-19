#!/usr/bin/env python3
"""
Basic functionality tests for llmlingua2.

This file contains unit tests for core functionality of llmlingua2.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llmlingua2_config import LLMLingua2Config, CompressionMethod, ModelType
from llmlingua2_compressor import LLMLingua2Compressor
from llmlingua2_results import CompressionResult
from llmlingua2_utils import (
    calculate_compression_metrics,
    validate_compression_result,
    sanitize_prompt,
    get_available_models,
    estimate_compression_ratio
)

class TestLLMLingua2Config(unittest.TestCase):
    """Test cases for LLMLingua2Config class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = LLMLingua2Config()
        
        self.assertEqual(config.model_name, "microsoft/llmlingua-2")
        self.assertEqual(config.device_map, "auto")
        self.assertEqual(config.target_token, 512)
        self.assertEqual(config.compression_method, CompressionMethod.BALANCED.value)
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.top_p, 0.9)
        self.assertEqual(config.max_length, 1024)
        self.assertEqual(config.min_length, 50)
        self.assertTrue(config.preserve_structure)
        self.assertIsNone(config.custom_model_path)
        self.assertEqual(config.batch_size, 1)
        self.assertIsNone(config.cache_dir)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = LLMLingua2Config(
            model_name="custom-model",
            device_map="cpu",
            target_token=256,
            compression_method=CompressionMethod.AGGRESSIVE.value,
            temperature=0.5,
            top_p=0.8,
            max_length=512,
            min_length=25,
            preserve_structure=False,
            custom_model_path="/path/to/model",
            batch_size=8,
            cache_dir="/cache/dir"
        )
        
        self.assertEqual(config.model_name, "custom-model")
        self.assertEqual(config.device_map, "cpu")
        self.assertEqual(config.target_token, 256)
        self.assertEqual(config.compression_method, CompressionMethod.AGGRESSIVE.value)
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.top_p, 0.8)
        self.assertEqual(config.max_length, 512)
        self.assertEqual(config.min_length, 25)
        self.assertFalse(config.preserve_structure)
        self.assertEqual(config.custom_model_path, "/path/to/model")
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.cache_dir, "/cache/dir")
    
    def test_compression_method_validation(self):
        """Test compression method validation."""
        # Valid methods
        for method in CompressionMethod:
            config = LLMLingua2Config(compression_method=method.value)
            self.assertEqual(config.compression_method, method.value)
        
        # Invalid method
        with self.assertRaises(ValueError):
            LLMLingua2Config(compression_method="invalid_method")
    
    def test_device_map_validation(self):
        """Test device map validation."""
        # Valid device maps
        for device in ["cpu", "cuda", "mps", "auto"]:
            config = LLMLingua2Config(device_map=device)
            self.assertEqual(config.device_map, device)
        
        # Invalid device map
        with self.assertRaises(ValueError):
            LLMLingua2Config(device_map="invalid_device")
    
    def test_positive_integer_validation(self):
        """Test positive integer validation."""
        # Valid values
        config = LLMLingua2Config(target_token=100, max_length=200, min_length=10, batch_size=4)
        self.assertEqual(config.target_token, 100)
        self.assertEqual(config.max_length, 200)
        self.assertEqual(config.min_length, 10)
        self.assertEqual(config.batch_size, 4)
        
        # Invalid values
        with self.assertRaises(ValueError):
            LLMLingua2Config(target_token=0)
        with self.assertRaises(ValueError):
            LLMLingua2Config(target_token=-1)
    
    def test_float_range_validation(self):
        """Test float range validation."""
        # Valid values
        config = LLMLingua2Config(temperature=0.5, top_p=0.8)
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.top_p, 0.8)
        
        # Invalid values
        with self.assertRaises(ValueError):
            LLMLingua2Config(temperature=-0.1)
        with self.assertRaises(ValueError):
            LLMLingua2Config(temperature=1.1)
        with self.assertRaises(ValueError):
            LLMLingua2Config(top_p=-0.1)
        with self.assertRaises(ValueError):
            LLMLingua2Config(top_p=1.1)
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = LLMLingua2Config()
        self.assertTrue(config.validate())
        
        # Test with invalid config
        invalid_config = LLMLingua2Config()
        invalid_config.compression_method = "invalid"
        self.assertFalse(invalid_config.validate())
    
    def test_config_save_load(self):
        """Test configuration save and load functionality."""
        config = LLMLingua2Config(
            model_name="test-model",
            target_token=128,
            compression_method=CompressionMethod.CONSERVATIVE.value
        )
        
        # Save configuration
        config_file = "test_config.json"
        config.save(config_file)
        self.assertTrue(os.path.exists(config_file))
        
        # Load configuration
        loaded_config = LLMLingua2Config.load(config_file)
        self.assertEqual(loaded_config.model_name, "test-model")
        self.assertEqual(loaded_config.target_token, 128)
        self.assertEqual(loaded_config.compression_method, CompressionMethod.CONSERVATIVE.value)
        
        # Clean up
        os.remove(config_file)
    
    def test_config_update(self):
        """Test configuration update functionality."""
        config = LLMLingua2Config()
        
        # Update valid parameters
        config.update(target_token=256, temperature=0.8)
        self.assertEqual(config.target_token, 256)
        self.assertEqual(config.temperature, 0.8)
        
        # Update invalid parameter
        with self.assertRaises(AttributeError):
            config.update(invalid_param="value")

class TestLLMLingua2Compressor(unittest.TestCase):
    """Test cases for LLMLingua2Compressor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = LLMLingua2Config()
        self.compressor = LLMLingua2Compressor(self.config)
        self.test_prompt = "This is a test prompt for compression testing."
    
    def test_compressor_initialization(self):
        """Test compressor initialization."""
        self.assertIsNotNone(self.compressor.config)
        self.assertIsNotNone(self.compressor.model)
        self.assertIsNotNone(self.compressor.tokenizer)
        self.assertEqual(self.compressor.stats['total_compressions'], 0)
    
    def test_basic_compression(self):
        """Test basic compression functionality."""
        result = self.compressor.compress(self.test_prompt)
        
        self.assertIsInstance(result, CompressionResult)
        self.assertEqual(result.original_prompt, self.test_prompt)
        self.assertGreater(result.original_token_count, 0)
        self.assertGreaterEqual(result.compressed_token_count, 0)
        self.assertGreaterEqual(result.compression_ratio, 0)
        self.assertLessEqual(result.compression_ratio, 1)
        self.assertGreater(result.processing_time, 0)
        self.assertEqual(result.method_used, self.config.compression_method)
        self.assertEqual(result.model_used, self.config.model_name)
    
    def test_compression_with_overrides(self):
        """Test compression with parameter overrides."""
        target_token = 5
        method = CompressionMethod.AGGRESSIVE.value
        
        result = self.compressor.compress(
            self.test_prompt,
            target_token=target_token,
            method=method
        )
        
        self.assertLessEqual(result.compressed_token_count, target_token)
        self.assertEqual(result.method_used, method)
    
    def test_batch_compression(self):
        """Test batch compression functionality."""
        prompts = [
            "First test prompt",
            "Second test prompt with more words",
            "Third test prompt that is even longer than the previous ones"
        ]
        
        results = self.compressor.batch_compress(prompts, show_progress=False)
        
        self.assertEqual(len(results), len(prompts))
        for i, result in enumerate(results):
            self.assertIsInstance(result, CompressionResult)
            self.assertEqual(result.original_prompt, prompts[i])
    
    def test_empty_prompt_handling(self):
        """Test handling of empty prompts."""
        result = self.compressor.compress("")
        
        self.assertEqual(result.original_token_count, 0)
        self.assertEqual(result.compressed_token_count, 0)
        self.assertEqual(result.compressed_prompt, "")
    
    def test_stats_tracking(self):
        """Test statistics tracking."""
        # Compress a few prompts
        self.compressor.compress("First prompt")
        self.compressor.compress("Second prompt")
        
        stats = self.compressor.get_stats()
        
        self.assertEqual(stats['total_compressions'], 2)
        self.assertGreater(stats['total_tokens_saved'], 0)
        self.assertGreater(stats['total_processing_time'], 0)
    
    def test_config_reset(self):
        """Test configuration reset functionality."""
        new_config = LLMLingua2Config(
            target_token=100,
            compression_method=CompressionMethod.AGGRESSIVE.value
        )
        
        self.compressor.reset_config(new_config)
        
        self.assertEqual(self.compressor.config.target_token, 100)
        self.assertEqual(self.compressor.config.compression_method, CompressionMethod.AGGRESSIVE.value)
    
    def test_estimate_compression(self):
        """Test compression estimation functionality."""
        estimate = self.compressor.estimate_compression(self.test_prompt)
        
        self.assertIn('original_tokens', estimate)
        self.assertIn('estimated_compressed_tokens', estimate)
        self.assertIn('estimated_compression_ratio', estimate)
        self.assertIn('estimated_tokens_saved', estimate)
        self.assertIn('method', estimate)
        self.assertIn('confidence', estimate)
        
        self.assertGreater(estimate['original_tokens'], 0)
        self.assertGreaterEqual(estimate['estimated_compressed_tokens'], 0)
        self.assertGreaterEqual(estimate['estimated_compression_ratio'], 0)
        self.assertLessEqual(estimate['estimated_compression_ratio'], 1)

class TestCompressionResult(unittest.TestCase):
    """Test cases for CompressionResult class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.result = CompressionResult(
            original_prompt="Original prompt",
            compressed_prompt="Compressed prompt",
            original_token_count=10,
            compressed_token_count=5,
            compression_ratio=0.5,
            processing_time=0.1,
            method_used="balanced",
            model_used="test-model",
            confidence_score=0.8
        )
    
    def test_result_initialization(self):
        """Test result initialization."""
        self.assertEqual(self.result.original_prompt, "Original prompt")
        self.assertEqual(self.result.compressed_prompt, "Compressed prompt")
        self.assertEqual(self.result.original_token_count, 10)
        self.assertEqual(self.result.compressed_token_count, 5)
        self.assertEqual(self.result.compression_ratio, 0.5)
        self.assertEqual(self.result.processing_time, 0.1)
        self.assertEqual(self.result.method_used, "balanced")
        self.assertEqual(self.result.model_used, "test-model")
        self.assertEqual(self.result.confidence_score, 0.8)
    
    def test_result_validation(self):
        """Test result validation."""
        # Valid result should not raise exception
        CompressionResult(
            original_prompt="Test",
            compressed_prompt="Test",
            original_token_count=10,
            compressed_token_count=5,
            compression_ratio=0.5,
            processing_time=0.1,
            method_used="test",
            model_used="test"
        )
        
        # Invalid cases
        with self.assertRaises(ValueError):
            CompressionResult(
                original_prompt="Test",
                compressed_prompt="Test",
                original_token_count=5,
                compressed_token_count=10,  # More than original
                compression_ratio=0.5,
                processing_time=0.1,
                method_used="test",
                model_used="test"
            )
        
        with self.assertRaises(ValueError):
            CompressionResult(
                original_prompt="Test",
                compressed_prompt="Test",
                original_token_count=10,
                compressed_token_count=5,
                compression_ratio=1.5,  # > 1
                processing_time=0.1,
                method_used="test",
                model_used="test"
            )
        
        with self.assertRaises(ValueError):
            CompressionResult(
                original_prompt="Test",
                compressed_prompt="Test",
                original_token_count=10,
                compressed_token_count=5,
                compression_ratio=0.5,
                processing_time=-0.1,  # Negative
                method_used="test",
                model_used="test"
            )
    
    def test_dict_conversion(self):
        """Test dictionary conversion."""
        result_dict = self.result.to_dict()
        
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict['original_prompt'], "Original prompt")
        self.assertEqual(result_dict['compressed_token_count'], 5)
        
        # Test from_dict
        recreated_result = CompressionResult.from_dict(result_dict)
        self.assertEqual(recreated_result.original_prompt, self.result.original_prompt)
        self.assertEqual(recreated_result.compressed_token_count, self.result.compressed_token_count)
    
    def test_summary_generation(self):
        """Test summary generation."""
        summary = self.result.get_summary()
        
        self.assertIn('original_tokens', summary)
        self.assertIn('compressed_tokens', summary)
        self.assertIn('tokens_saved', summary)
        self.assertIn('compression_ratio', summary)
        self.assertIn('compression_percent', summary)
        self.assertIn('processing_time', summary)
        self.assertIn('method', summary)
        self.assertIn('model', summary)
        self.assertIn('confidence', summary)
        
        self.assertEqual(summary['original_tokens'], 10)
        self.assertEqual(summary['compressed_tokens'], 5)
        self.assertEqual(summary['tokens_saved'], 5)
        self.assertEqual(summary['compression_ratio'], 0.5)
        self.assertEqual(summary['compression_percent'], 50.0)
    
    def test_efficiency_score(self):
        """Test efficiency score calculation."""
        score = self.result.calculate_efficiency_score()
        
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
        self.assertIsInstance(score, float)
    
    def test_result_comparison(self):
        """Test result comparison functionality."""
        # Create a better result (higher compression ratio)
        better_result = CompressionResult(
            original_prompt="Original prompt",
            compressed_prompt="Compressed prompt",
            original_token_count=10,
            compressed_token_count=3,  # Better compression
            compression_ratio=0.7,
            processing_time=0.1,
            method_used="balanced",
            model_used="test-model"
        )
        
        # Create a worse result (lower compression ratio)
        worse_result = CompressionResult(
            original_prompt="Original prompt",
            compressed_prompt="Compressed prompt",
            original_token_count=10,
            compressed_token_count=8,  # Worse compression
            compression_ratio=0.2,
            processing_time=0.1,
            method_used="balanced",
            model_used="test-model"
        )
        
        # Test comparison
        self.assertTrue(better_result.is_better_than(self.result))
        self.assertTrue(self.result.is_better_than(worse_result))
        self.assertFalse(worse_result.is_better_than(better_result))
        
        # Test with speed prioritization
        faster_result = CompressionResult(
            original_prompt="Original prompt",
            compressed_prompt="Compressed prompt",
            original_token_count=10,
            compressed_token_count=5,
            compression_ratio=0.5,
            processing_time=0.05,  # Faster
            method_used="balanced",
            model_used="test-model"
        )
        
        self.assertTrue(faster_result.is_better_than(self.result, prioritize_ratio=False))

class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_calculate_compression_metrics(self):
        """Test compression metrics calculation."""
        original = "This is the original text with several words"
        compressed = "This is compressed"
        
        metrics = calculate_compression_metrics(original, compressed)
        
        self.assertIn('original_char_count', metrics)
        self.assertIn('compressed_char_count', metrics)
        self.assertIn('original_token_count', metrics)
        self.assertIn('compressed_token_count', metrics)
        self.assertIn('char_compression_ratio', metrics)
        self.assertIn('token_compression_ratio', metrics)
        self.assertIn('space_saving_percent', metrics)
        
        self.assertGreater(metrics['original_char_count'], metrics['compressed_char_count'])
        self.assertGreater(metrics['original_token_count'], metrics['compressed_token_count'])
        self.assertGreater(metrics['char_compression_ratio'], 0)
        self.assertGreater(metrics['token_compression_ratio'], 0)
        self.assertGreater(metrics['space_saving_percent'], 0)
    
    def test_validate_compression_result(self):
        """Test compression result validation."""
        valid_result = CompressionResult(
            original_prompt="Test",
            compressed_prompt="Test",
            original_token_count=10,
            compressed_token_count=5,
            compression_ratio=0.5,
            processing_time=0.1,
            method_used="test",
            model_used="test"
        )
        
        self.assertTrue(validate_compression_result(valid_result))
        self.assertFalse(validate_compression_result("not a result"))
        
        # Test invalid result creation - should raise ValueError
        with self.assertRaises(ValueError):
            CompressionResult(
                original_prompt="Test",
                compressed_prompt="Test",
                original_token_count=5,
                compressed_token_count=10,  # Invalid
                compression_ratio=0.5,
                processing_time=0.1,
                method_used="test",
                model_used="test"
            )
    
    def test_sanitize_prompt(self):
        """Test prompt sanitization."""
        # Test with null bytes
        prompt_with_null = "Test\x00prompt\x00with\x00nulls"
        sanitized = sanitize_prompt(prompt_with_null)
        self.assertNotIn('\x00', sanitized)
        
        # Test with extra whitespace
        prompt_with_spaces = "  Test   prompt   with   extra   spaces  "
        sanitized = sanitize_prompt(prompt_with_spaces)
        self.assertEqual(sanitized, "Test prompt with extra spaces")
        
        # Test empty prompt
        self.assertEqual(sanitize_prompt(""), "")
    
    def test_get_available_models(self):
        """Test available models function."""
        models = get_available_models()
        
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)
        
        for model in models:
            self.assertIsInstance(model, str)
            self.assertGreater(len(model), 0)
    
    def test_estimate_compression_ratio(self):
        """Test compression ratio estimation."""
        prompt = "This is a test prompt for compression ratio estimation"
        
        for method in CompressionMethod:
            ratio = estimate_compression_ratio(prompt, method)
            
            self.assertGreaterEqual(ratio, 0.1)
            self.assertLessEqual(ratio, 0.9)
            self.assertIsInstance(ratio, float)

if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
