#!/usr/bin/env python3
"""
Main implementation and test file for llmlingua2.

This file demonstrates the usage of the llmlingua2 compression system
and provides example test cases.
"""

import sys
import os
from typing import List

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llmlingua2_config import LLMLingua2Config, CompressionMethod
from llmlingua2_compressor import LLMLingua2Compressor
from llmlingua2_results import CompressionResult
from llmlingua2_utils import (
    get_available_models, 
    get_system_info, 
    format_processing_time
)

def initialize_llmlingua2(config: LLMLingua2Config) -> LLMLingua2Compressor:
    """
    Initialize llmlingua2 compressor with configuration.
    
    Args:
        config: LLMLingua2Config object
        
    Returns:
        Initialized LLMLingua2Compressor instance
    """
    print("Initializing LLMLingua2 Compressor...")
    print(f"Model: {config.model_name}")
    print(f"Device: {config.device_map}")
    print(f"Method: {config.compression_method}")
    print("-" * 50)
    
    compressor = LLMLingua2Compressor(config)
    return compressor

def compress_prompt_llmlingua2(prompt: str, config: LLMLingua2Config) -> CompressionResult:
    """
    Main compression function for llmlingua2.
    
    Args:
        prompt: Input text prompt to compress
        config: LLMLingua2Config object
        
    Returns:
        CompressionResult object
    """
    compressor = initialize_llmlingua2(config)
    print(f"Original prompt ({len(prompt.split())} tokens):")
    print(f"'{prompt}'")
    print()
    
    result = compressor.compress(prompt)
    
    print(f"Compressed prompt ({result.compressed_token_count} tokens):")
    print(f"'{result.compressed_prompt}'")
    print()
    
    print("Compression Summary:")
    print(result)
    print()
    
    return result

def batch_compress_prompts(prompts: List[str], config: LLMLingua2Config) -> List[CompressionResult]:
    """
    Batch processing function for multiple prompts.
    
    Args:
        prompts: List of input prompts
        config: LLMLingua2Config object
        
    Returns:
        List of CompressionResult objects
    """
    compressor = initialize_llmlingua2(config)
    print(f"Batch compressing {len(prompts)} prompts...")
    print("-" * 50)
    
    results = compressor.batch_compress(prompts, show_progress=True)
    
    print("\nBatch Compression Results:")
    for i, result in enumerate(results):
        print(f"\nPrompt {i+1}:")
        print(f"  Original: {result.original_token_count} tokens")
        print(f"  Compressed: {result.compressed_token_count} tokens")
        print(f"  Ratio: {result.compression_ratio:.3f}")
        print(f"  Time: {format_processing_time(result.processing_time)}")
    
    return results

def test_different_methods():
    """Test compression with different methods."""
    prompt = "This is a sample prompt that we will use to test different compression methods in llmlingua2. The goal is to see how each method performs in terms of compression ratio and quality."
    
    methods = [
        CompressionMethod.CONSERVATIVE,
        CompressionMethod.BALANCED,
        CompressionMethod.STANDARD,
        CompressionMethod.AGGRESSIVE
    ]
    
    print("Testing Different Compression Methods")
    print("=" * 50)
    print(f"Original prompt ({len(prompt.split())} tokens):")
    print(f"'{prompt}'")
    print()
    
    results = []
    for method in methods:
        config = LLMLingua2Config(compression_method=method.value)
        compressor = LLMLingua2Compressor(config)
        result = compressor.compress(prompt)
        results.append(result)
        
        print(f"{method.value.upper()} Method:")
        print(f"  Compressed: {result.compressed_prompt}")
        print(f"  Tokens: {result.compressed_token_count}")
        print(f"  Ratio: {result.compression_ratio:.3f}")
        print(f"  Confidence: {result.confidence_score:.3f}")
        print()
    
    # Find the best result
    best_result = max(results, key=lambda r: r.calculate_efficiency_score())
    print(f"Best method by efficiency score: {best_result.method_used}")
    print(f"Efficiency score: {best_result.calculate_efficiency_score():.4f}")

def test_configuration_options():
    """Test different configuration options."""
    prompt = "The quick brown fox jumps over the lazy dog. This sentence contains various words and demonstrates how different configuration parameters affect the compression process in llmlingua2."
    
    print("Testing Configuration Options")
    print("=" * 50)
    
    # Test different target tokens
    target_tokens = [10, 20, 30]
    for target in target_tokens:
        config = LLMLingua2Config(target_token=target)
        compressor = LLMLingua2Compressor(config)
        result = compressor.compress(prompt)
        
        print(f"Target tokens: {target}")
        print(f"  Actual compressed: {result.compressed_token_count} tokens")
        print(f"  Result: '{result.compressed_prompt}'")
        print()

def main():
    """Main function to run llmlingua2 tests."""
    print("LLMLingua2 Test Suite")
    print("=" * 50)
    
    # Display system information
    print("System Information:")
    sys_info = get_system_info()
    for key, value in sys_info.items():
        print(f"  {key}: {value}")
    print()
    
    # Display available models
    print("Available Models:")
    models = get_available_models()
    for model in models:
        print(f"  - {model}")
    print()
    
    # Test 1: Basic compression
    print("Test 1: Basic Compression")
    print("-" * 30)
    prompt = "Hello, this is a simple test prompt for llmlingua2 compression. We want to see how well it works with basic text."
    config = LLMLingua2Config()
    result = compress_prompt_llmlingua2(prompt, config)
    
    # Test 2: Batch compression
    print("\nTest 2: Batch Compression")
    print("-" * 30)
    prompts = [
        "This is the first prompt in our batch test.",
        "Here comes the second prompt with slightly more content to compress.",
        "The third prompt is even longer and contains more information that needs to be compressed efficiently by llmlingua2.",
        "Final prompt in the batch, testing how the system handles multiple inputs."
    ]
    batch_results = batch_compress_prompts(prompts, config)
    
    # Test 3: Different methods
    print("\nTest 3: Different Compression Methods")
    print("-" * 30)
    test_different_methods()
    
    # Test 4: Configuration options
    print("\nTest 4: Configuration Options")
    print("-" * 30)
    test_configuration_options()
    
    # Display final statistics
    print("\nFinal Statistics")
    print("-" * 30)
    compressor = LLMLingua2Compressor(config)
    stats = compressor.get_stats()
    print(f"Total compressions: {stats['total_compressions']}")
    print(f"Total tokens saved: {stats['total_tokens_saved']}")
    print(f"Average compression ratio: {stats['average_compression_ratio']:.3f}")
    print(f"Total processing time: {format_processing_time(stats['total_processing_time'])}")
    
    print("\nLLMLingua2 Test Suite Complete!")

if __name__ == "__main__":
    main()
