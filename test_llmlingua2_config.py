#!/usr/bin/env python3
"""
Configuration validation tests for llmlingua2.

This file contains tests specifically focused on configuration management,
validation, and parameter handling.
"""

import sys
import os
import unittest
import json
import tempfile
from unittest.mock import patch, mock_open

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llmlingua2_config import LLMLingua2Config, CompressionMethod, ModelType
from llmlingua2_utils import load_compression_config

class TestLLMLingua2ConfigValidation(unittest.TestCase):
    """Test cases for LLMLingua2Config validation."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = LLMLingua2Config()
        
        # Test all default values
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
    
    def test_compression_method_validation(self):
        """Test compression method parameter validation."""
        # Valid methods
        valid_methods = [method.value for method in CompressionMethod]
        
        for method in valid_methods:
            config = LLMLingua2Config(compression_method=method)
            self.assertEqual(config.compression_method, method)
            self.assertTrue(config.validate())
        
        # Invalid method
        with self.assertRaises(ValueError):
            LLMLingua2Config(compression_method="invalid_method")
        
        # Test validation after creation
        config = LLMLingua2Config()
        config.compression_method = "invalid"
        self.assertFalse(config.validate())
    
    def test_device_map_validation(self):
        """Test device map parameter validation."""
        # Valid device maps
        valid_devices = ["cpu", "cuda", "mps", "auto"]
        
        for device in valid_devices:
            config = LLMLingua2Config(device_map=device)
            self.assertEqual(config.device_map, device)
            self.assertTrue(config.validate())
        
        # Invalid device map
        with self.assertRaises(ValueError):
            LLMLingua2Config(device_map="invalid_device")
        
        # Test validation after creation
        config = LLMLingua2Config()
        config.device_map = "invalid"
        self.assertFalse(config.validate())
    
    def test_positive_integer_parameters(self):
        """Test positive integer parameter validation."""
        # Test target_token
        with self.assertRaises(ValueError):
            LLMLingua2Config(target_token=0)
        with self.assertRaises(ValueError):
            LLMLingua2Config(target_token=-1)
        
        # Test max_length
        with self.assertRaises(ValueError):
            LLMLingua2Config(max_length=0)
        with self.assertRaises(ValueError):
            LLMLingua2Config(max_length=-1)
        
        # Test min_length
        with self.assertRaises(ValueError):
            LLMLingua2Config(min_length=0)
        with self.assertRaises(ValueError):
            LLMLingua2Config(min_length=-1)
        
        # Test batch_size
        with self.assertRaises(ValueError):
            LLMLingua2Config(batch_size=0)
        with self.assertRaises(ValueError):
            LLMLingua2Config(batch_size=-1)
        
        # Valid values should work
        config = LLMLingua2Config(
            target_token=100,
            max_length=200,
            min_length=10,
            batch_size=4
        )
        self.assertTrue(config.validate())
    
    def test_float_range_parameters(self):
        """Test float range parameter validation."""
        # Test temperature
        with self.assertRaises(ValueError):
            LLMLingua2Config(temperature=-0.1)
        with self.assertRaises(ValueError):
            LLMLingua2Config(temperature=1.1)
        
        # Test top_p
        with self.assertRaises(ValueError):
            LLMLingua2Config(top_p=-0.1)
        with self.assertRaises(ValueError):
            LLMLingua2Config(top_p=1.1)
        
        # Valid values should work
        config = LLMLingua2Config(temperature=0.5, top_p=0.8)
        self.assertTrue(config.validate())
    
    def test_length_constraints(self):
        """Test length constraint validation."""
        # max_length should be >= min_length
        with self.assertRaises(ValueError):
            LLMLingua2Config(max_length=50, min_length=100)
        
        # Valid configuration
        config = LLMLingua2Config(max_length=200, min_length=50)
        self.assertTrue(config.validate())
        
        # Test validation after modification
        config.max_length = 25  # Now less than min_length
        self.assertFalse(config.validate())
    
    def test_boolean_parameters(self):
        """Test boolean parameter validation."""
        # preserve_structure should be boolean
        config = LLMLingua2Config(preserve_structure=True)
        self.assertTrue(config.preserve_structure)
        self.assertTrue(config.validate())
        
        config = LLMLingua2Config(preserve_structure=False)
        self.assertFalse(config.preserve_structure)
        self.assertTrue(config.validate())
    
    def test_optional_parameters(self):
        """Test optional parameter validation."""
        # custom_model_path can be None or string
        config = LLMLingua2Config(custom_model_path=None)
        self.assertIsNone(config.custom_model_path)
        self.assertTrue(config.validate())
        
        config = LLMLingua2Config(custom_model_path="/path/to/model")
        self.assertEqual(config.custom_model_path, "/path/to/model")
        self.assertTrue(config.validate())
        
        # cache_dir can be None or string
        config = LLMLingua2Config(cache_dir=None)
        self.assertIsNone(config.cache_dir)
        self.assertTrue(config.validate())
        
        config = LLMLingua2Config(cache_dir="/cache/dir")
        self.assertEqual(config.cache_dir, "/cache/dir")
        self.assertTrue(config.validate())
    
    def test_model_name_validation(self):
        """Test model name parameter validation."""
        # Model name should be non-empty string
        with self.assertRaises(ValueError):
            LLMLingua2Config(model_name="")
        
        # Valid model names
        valid_models = [
            "microsoft/llmlingua-2",
            "custom/model",
            "test-model-v1"
        ]
        
        for model in valid_models:
            config = LLMLingua2Config(model_name=model)
            self.assertEqual(config.model_name, model)
            self.assertTrue(config.validate())
    
    def test_comprehensive_validation(self):
        """Test comprehensive configuration validation."""
        # Valid configuration
        config = LLMLingua2Config(
            model_name="test-model",
            device_map="cpu",
            target_token=256,
            compression_method=CompressionMethod.STANDARD.value,
            temperature=0.6,
            top_p=0.8,
            max_length=512,
            min_length=25,
            preserve_structure=True,
            custom_model_path="/path/to/model",
            batch_size=4,
            cache_dir="/cache/dir"
        )
        self.assertTrue(config.validate())
        
        # Invalid configuration (multiple issues)
        invalid_config = LLMLingua2Config()
        invalid_config.compression_method = "invalid"
        invalid_config.device_map = "invalid"
        invalid_config.target_token = -1
        invalid_config.temperature = 1.5
        invalid_config.max_length = 50
        invalid_config.min_length = 100
        
        self.assertFalse(invalid_config.validate())
    
    def test_validation_error_messages(self):
        """Test that validation provides meaningful error messages."""
        config = LLMLingua2Config()
        
        # Test individual validation errors
        config.compression_method = "invalid"
        validation_result = config.validate()
        self.assertIsInstance(validation_result, bool)
        self.assertFalse(validation_result)
        
        # Reset and test another error
        config.compression_method = CompressionMethod.BALANCED.value
        config.target_token = -1
        validation_result = config.validate()
        self.assertFalse(validation_result)

class TestLLMLingua2ConfigSerialization(unittest.TestCase):
    """Test cases for configuration serialization and deserialization."""
    
    def test_config_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = LLMLingua2Config(
            model_name="test-model",
            target_token=256,
            compression_method=CompressionMethod.AGGRESSIVE.value,
            temperature=0.5,
            custom_model_path="/path/to/model"
        )
        
        config_dict = config.dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['model_name'], "test-model")
        self.assertEqual(config_dict['target_token'], 256)
        self.assertEqual(config_dict['compression_method'], CompressionMethod.AGGRESSIVE.value)
        self.assertEqual(config_dict['temperature'], 0.5)
        self.assertEqual(config_dict['custom_model_path'], "/path/to/model")
        
        # Check that all parameters are included
        expected_keys = [
            'model_name', 'device_map', 'target_token', 'compression_method',
            'temperature', 'top_p', 'max_length', 'min_length',
            'preserve_structure', 'custom_model_path', 'batch_size', 'cache_dir'
        ]
        
        for key in expected_keys:
            self.assertIn(key, config_dict)
    
    def test_config_from_dict(self):
        """Test configuration from dictionary creation."""
        config_dict = {
            'model_name': 'custom-model',
            'device_map': 'cuda',
            'target_token': 128,
            'compression_method': CompressionMethod.CONSERVATIVE.value,
            'temperature': 0.8,
            'top_p': 0.95,
            'max_length': 256,
            'min_length': 10,
            'preserve_structure': False,
            'custom_model_path': '/custom/path',
            'batch_size': 8,
            'cache_dir': '/custom/cache'
        }
        
        config = LLMLingua2Config.from_dict(config_dict)
        
        self.assertEqual(config.model_name, 'custom-model')
        self.assertEqual(config.device_map, 'cuda')
        self.assertEqual(config.target_token, 128)
        self.assertEqual(config.compression_method, CompressionMethod.CONSERVATIVE.value)
        self.assertEqual(config.temperature, 0.8)
        self.assertEqual(config.top_p, 0.95)
        self.assertEqual(config.max_length, 256)
        self.assertEqual(config.min_length, 10)
        self.assertFalse(config.preserve_structure)
        self.assertEqual(config.custom_model_path, '/custom/path')
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.cache_dir, '/custom/cache')
    
    def test_config_save_load_file(self):
        """Test configuration save and load to/from file."""
        config = LLMLingua2Config(
            model_name="test-save-load",
            target_token=64,
            compression_method=CompressionMethod.STANDARD.value,
            temperature=0.3
        )
        
        # Test with temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            # Save configuration
            config.save(temp_file)
            self.assertTrue(os.path.exists(temp_file))
            
            # Load configuration
            loaded_config = LLMLingua2Config.load(temp_file)
            
            # Verify loaded configuration
            self.assertEqual(loaded_config.model_name, "test-save-load")
            self.assertEqual(loaded_config.target_token, 64)
            self.assertEqual(loaded_config.compression_method, CompressionMethod.STANDARD.value)
            self.assertEqual(loaded_config.temperature, 0.3)
            
            # Verify other parameters are at defaults
            self.assertEqual(loaded_config.device_map, "auto")
            self.assertEqual(loaded_config.top_p, 0.9)
            self.assertTrue(loaded_config.preserve_structure)
            
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_config_save_load_json_format(self):
        """Test that saved configuration is valid JSON."""
        config = LLMLingua2Config(
            model_name="json-test",
            target_token=100,
            compression_method=CompressionMethod.BALANCED.value
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            # Save configuration
            config.save(temp_file)
            
            # Verify file contains valid JSON
            with open(temp_file, 'r') as f:
                loaded_json = json.load(f)
            
            self.assertIsInstance(loaded_json, dict)
            self.assertEqual(loaded_json['model_name'], "json-test")
            self.assertEqual(loaded_json['target_token'], 100)
            self.assertEqual(loaded_json['compression_method'], CompressionMethod.BALANCED.value)
            
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_load_from_invalid_file(self):
        """Test loading configuration from invalid file."""
        # Test with non-existent file
        with self.assertRaises(FileNotFoundError):
            LLMLingua2Config.load("/non/existent/file.json")
        
        # Test with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_file = f.name
        
        try:
            with self.assertRaises(json.JSONDecodeError):
                LLMLingua2Config.load(temp_file)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_load_from_incomplete_config(self):
        """Test loading configuration with missing parameters."""
        # Create incomplete config JSON
        incomplete_config = {
            'model_name': 'incomplete-model',
            'target_token': 200
            # Missing many required parameters
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(incomplete_config, f)
            temp_file = f.name
        
        try:
            # Should load with defaults for missing parameters
            loaded_config = LLMLingua2Config.load(temp_file)
            
            self.assertEqual(loaded_config.model_name, 'incomplete-model')
            self.assertEqual(loaded_config.target_token, 200)
            
            # Should have defaults for missing parameters
            self.assertEqual(loaded_config.device_map, 'auto')
            self.assertEqual(loaded_config.compression_method, CompressionMethod.BALANCED.value)
            self.assertEqual(loaded_config.temperature, 0.7)
            
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

class TestLLMLingua2ConfigUpdate(unittest.TestCase):
    """Test cases for configuration update functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = LLMLingua2Config()
    
    def test_update_single_parameter(self):
        """Test updating a single parameter."""
        original_model = self.config.model_name
        
        self.config.update(model_name="updated-model")
        
        self.assertEqual(self.config.model_name, "updated-model")
        # Other parameters should remain unchanged
        self.assertEqual(self.config.device_map, "auto")
        self.assertEqual(self.config.target_token, 512)
    
    def test_update_multiple_parameters(self):
        """Test updating multiple parameters."""
        self.config.update(
            model_name="multi-update",
            target_token=256,
            compression_method=CompressionMethod.AGGRESSIVE.value,
            temperature=0.4
        )
        
        self.assertEqual(self.config.model_name, "multi-update")
        self.assertEqual(self.config.target_token, 256)
        self.assertEqual(self.config.compression_method, CompressionMethod.AGGRESSIVE.value)
        self.assertEqual(self.config.temperature, 0.4)
        
        # Unchanged parameters
        self.assertEqual(self.config.device_map, "auto")
        self.assertEqual(self.config.top_p, 0.9)
    
    def test_update_with_invalid_parameter(self):
        """Test updating with invalid parameter name."""
        with self.assertRaises(AttributeError):
            self.config.update(invalid_parameter="value")
        
        # Config should remain unchanged
        self.assertEqual(self.config.model_name, "microsoft/llmlingua-2")
        self.assertTrue(self.config.validate())
    
    def test_update_with_invalid_value(self):
        """Test updating with invalid parameter value."""
        # Try to update with invalid compression method
        with self.assertRaises(ValueError):
            self.config.update(compression_method="invalid")
        
        # Config should remain unchanged
        self.assertEqual(self.config.compression_method, CompressionMethod.BALANCED.value)
        self.assertTrue(self.config.validate())
    
    def test_update_preserves_validation(self):
        """Test that update maintains configuration validation."""
        # Start with valid config
        self.assertTrue(self.config.validate())
        
        # Update with valid values
        self.config.update(
            target_token=100,
            temperature=0.5,
            device_map="cpu"
        )
        
        # Should still be valid
        self.assertTrue(self.config.validate())
        
        # Update with invalid value
        with self.assertRaises(ValueError):
            self.config.update(target_token=-1)
        
        # Should still be valid (update failed)
        self.assertTrue(self.config.validate())
        self.assertEqual(self.config.target_token, 100)

class TestLLMLingua2ConfigEdgeCases(unittest.TestCase):
    """Test cases for configuration edge cases."""
    
    def test_boundary_values(self):
        """Test boundary values for numeric parameters."""
        # Test boundary values that should be valid
        config = LLMLingua2Config(
            temperature=0.0,  # Minimum valid
            top_p=1.0,  # Maximum valid
            target_token=1,  # Minimum valid
            max_length=1,  # Minimum valid
            min_length=1,  # Minimum valid
            batch_size=1  # Minimum valid
        )
        self.assertTrue(config.validate())
        
        config = LLMLingua2Config(
            temperature=1.0,  # Maximum valid
            top_p=0.0,  # Minimum valid
            target_token=10000,  # Large but valid
            max_length=10000,  # Large but valid
            min_length=1,  # Minimum valid
            batch_size=1000  # Large but valid
        )
        self.assertTrue(config.validate())
    
    def test_empty_string_handling(self):
        """Test handling of empty strings."""
        # model_name cannot be empty
        with self.assertRaises(ValueError):
            LLMLingua2Config(model_name="")
        
        # custom_model_path can be empty string (treated as None)
        config = LLMLingua2Config(custom_model_path="")
        self.assertIsNone(config.custom_model_path)
        self.assertTrue(config.validate())
        
        # cache_dir can be empty string (treated as None)
        config = LLMLingua2Config(cache_dir="")
        self.assertIsNone(config.cache_dir)
        self.assertTrue(config.validate())
    
    def test_whitespace_handling(self):
        """Test handling of whitespace in string parameters."""
        # Should trim whitespace
        config = LLMLingua2Config(
            model_name="  model-with-spaces  ",
            device_map="  cpu  ",
            custom_model_path="  /path/with/spaces  "
        )
        
        self.assertEqual(config.model_name, "model-with-spaces")
        self.assertEqual(config.device_map, "cpu")
        self.assertEqual(config.custom_model_path, "/path/with/spaces")
        self.assertTrue(config.validate())
    
    def test_case_sensitivity(self):
        """Test case sensitivity for string parameters."""
        # compression_method should be case-sensitive
        with self.assertRaises(ValueError):
            LLMLingua2Config(compression_method="CONSERVATIVE")  # Should be lowercase
        
        # device_map should be case-sensitive
        with self.assertRaises(ValueError):
            LLMLingua2Config(device_map="CPU")  # Should be lowercase
        
        # Valid lowercase should work
        config = LLMLingua2Config(
            compression_method=CompressionMethod.CONSERVATIVE.value,
            device_map="cpu"
        )
        self.assertTrue(config.validate())
    
    def test_special_characters(self):
        """Test handling of special characters in string parameters."""
        # model_name can contain special characters
        config = LLMLingua2Config(
            model_name="model-v2.0_special@chars",
            custom_model_path="/path/with/special_chars/@#$%",
            cache_dir="/cache/with/special/chars"
        )
        
        self.assertEqual(config.model_name, "model-v2.0_special@chars")
        self.assertEqual(config.custom_model_path, "/path/with/special_chars/@#$%")
        self.assertEqual(config.cache_dir, "/cache/with/special/chars")
        self.assertTrue(config.validate())
    
    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        # Should handle unicode characters
        config = LLMLingua2Config(
            model_name="模型-中文",
            custom_model_path="/路径/中文",
            cache_dir="/缓存/中文"
        )
        
        self.assertEqual(config.model_name, "模型-中文")
        self.assertEqual(config.custom_model_path, "/路径/中文")
        self.assertEqual(config.cache_dir, "/缓存/中文")
        self.assertTrue(config.validate())

class TestUtilityConfigFunctions(unittest.TestCase):
    """Test cases for utility configuration functions."""
    
    def test_load_compression_config_function(self):
        """Test the load_compression_config utility function."""
        config = LLMLingua2Config(
            model_name="utility-test",
            target_token=128,
            compression_method=CompressionMethod.STANDARD.value
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.save(f.name)
            temp_file = f.name
        
        try:
            # Test utility function
            loaded_config = load_compression_config(temp_file)
            
            self.assertIsInstance(loaded_config, LLMLingua2Config)
            self.assertEqual(loaded_config.model_name, "utility-test")
            self.assertEqual(loaded_config.target_token, 128)
            self.assertEqual(loaded_config.compression_method, CompressionMethod.STANDARD.value)
            
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_load_compression_config_nonexistent(self):
        """Test load_compression_config with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_compression_config("/non/existent/file.json")

if __name__ == '__main__':
    # Run all configuration tests
    unittest.main(verbosity=2)
