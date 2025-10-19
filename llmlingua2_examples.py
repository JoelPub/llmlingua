#!/usr/bin/env python3
"""
Example usage and test cases for llmlingua2.

This file provides comprehensive examples of how to use llmlingua2
for various text compression scenarios.
"""

import sys
import os
import json
from typing import List, Dict, Any

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llmlingua2_config import LLMLingua2Config, CompressionMethod
from llmlingua2_compressor import LLMLingua2Compressor
from llmlingua2_results import CompressionResult
from llmlingua2_utils import (
    calculate_compression_metrics,
    save_compression_result,
    load_compression_config,
    get_system_info
)

def example_basic_usage():
    """Example of basic llmlingua2 usage."""
    print("Example 1: Basic Usage")
    print("=" * 40)
    
    # Create a simple prompt
    prompt = """
    Natural language processing (NLP) is a subfield of linguistics, computer science, 
    and artificial intelligence concerned with the interactions between computers 
    and human language. It focuses on how to program computers to process and analyze 
    large amounts of natural language data.
    """
    
    # Initialize with default configuration
    config = LLMLingua2Config()
    compressor = LLMLingua2Compressor(config)
    
    # Compress the prompt
    result = compressor.compress(prompt)
    
    print("Original prompt:")
    print(prompt.strip())
    print(f"\nOriginal tokens: {result.original_token_count}")
    
    print("\nCompressed prompt:")
    print(result.compressed_prompt)
    print(f"\nCompressed tokens: {result.compressed_token_count}")
    print(f"Compression ratio: {result.compression_ratio:.3f}")
    print(f"Processing time: {result.processing_time:.3f}s")
    print()

def example_different_methods():
    """Example comparing different compression methods."""
    print("Example 2: Comparing Compression Methods")
    print("=" * 40)
    
    prompt = """
    Machine learning is a method of data analysis that automates analytical model building. 
    It is a branch of artificial intelligence based on the idea that systems can learn 
    from data, identify patterns and make decisions with minimal human intervention.
    """
    
    methods = [
        ("Conservative", CompressionMethod.CONSERVATIVE.value),
        ("Balanced", CompressionMethod.BALANCED.value),
        ("Standard", CompressionMethod.STANDARD.value),
        ("Aggressive", CompressionMethod.AGGRESSIVE.value)
    ]
    
    results = []
    
    for method_name, method_value in methods:
        config = LLMLingua2Config(compression_method=method_value)
        compressor = LLMLingua2Compressor(config)
        result = compressor.compress(prompt)
        results.append((method_name, result))
        
        print(f"\n{method_name} Method:")
        print(f"  Compressed: {result.compressed_prompt}")
        print(f"  Tokens: {result.compressed_token_count}/{result.original_token_count}")
        print(f"  Ratio: {result.compression_ratio:.3f}")
        print(f"  Confidence: {result.confidence_score:.3f}")
    
    # Find the most efficient method
    best_method, best_result = max(results, key=lambda x: x[1].calculate_efficiency_score())
    print(f"\nMost efficient method: {best_method}")
    print(f"Efficiency score: {best_result.calculate_efficiency_score():.4f}")
    print()

def example_batch_processing():
    """Example of batch processing multiple prompts."""
    print("Example 3: Batch Processing")
    print("=" * 40)
    
    prompts = [
        "The sun is the star at the center of the Solar System.",
        "Photosynthesis is a process used by plants to convert light energy into chemical energy.",
        "Gravity is a fundamental force that attracts objects with mass toward each other.",
        "Evolution is the change in heritable characteristics of biological populations over time.",
        "Quantum mechanics is a fundamental theory in physics that describes nature at atomic scales."
    ]
    
    config = LLMLingua2Config(
        compression_method=CompressionMethod.BALANCED.value,
        target_token=15
    )
    
    compressor = LLMLingua2Compressor(config)
    results = compressor.batch_compress(prompts, show_progress=True)
    
    print("\nBatch Compression Results:")
    total_original = 0
    total_compressed = 0
    
    for i, result in enumerate(results):
        total_original += result.original_token_count
        total_compressed += result.compressed_token_count
        
        print(f"\nPrompt {i+1}:")
        print(f"  Original: {result.original_prompt}")
        print(f"  Compressed: {result.compressed_prompt}")
        print(f"  Tokens: {result.compressed_token_count}/{result.original_token_count}")
    
    overall_ratio = 1 - (total_compressed / total_original) if total_original > 0 else 0
    print(f"\nBatch Summary:")
    print(f"  Total original tokens: {total_original}")
    print(f"  Total compressed tokens: {total_compressed}")
    print(f"  Overall compression ratio: {overall_ratio:.3f}")
    print()

def example_configuration_management():
    """Example of configuration management."""
    print("Example 4: Configuration Management")
    print("=" * 40)
    
    # Create custom configuration
    config = LLMLingua2Config(
        model_name="microsoft/llmlingua-2-medium",
        device_map="cpu",
        target_token=100,
        compression_method=CompressionMethod.STANDARD.value,
        temperature=0.5,
        top_p=0.8,
        batch_size=4
    )
    
    # Save configuration to file
    config_file = "example_config.json"
    config.save(config_file)
    print(f"Configuration saved to {config_file}")
    
    # Load configuration from file
    loaded_config = load_compression_config(config_file)
    print(f"Configuration loaded from {config_file}")
    print(f"Model: {loaded_config.model_name}")
    print(f"Target tokens: {loaded_config.target_token}")
    print(f"Method: {loaded_config.compression_method}")
    
    # Use the loaded configuration
    compressor = LLMLingua2Compressor(loaded_config)
    
    prompt = "This is a test prompt to demonstrate configuration management in llmlingua2."
    result = compressor.compress(prompt)
    
    print(f"\nCompression result with loaded config:")
    print(f"  Compressed: {result.compressed_prompt}")
    print(f"  Ratio: {result.compression_ratio:.3f}")
    print()
    
    # Clean up
    if os.path.exists(config_file):
        os.remove(config_file)

def example_result_analysis():
    """Example of detailed result analysis."""
    print("Example 5: Result Analysis")
    print("=" * 40)
    
    prompt = """
    Climate change refers to long-term shifts in temperatures and weather patterns. 
    These shifts may be natural, such as through variations in the solar cycle. 
    But since the 1800s, human activities have been the main driver of climate change, 
    primarily due to burning fossil fuels (like coal, oil, and gas), which produces 
    heat-trapping gases.
    """
    
    config = LLMLingua2Config(
        compression_method=CompressionMethod.BALANCED.value,
        target_token=30
    )
    
    compressor = LLMLingua2Compressor(config)
    result = compressor.compress(prompt)
    
    # Display detailed analysis
    print("Detailed Compression Analysis:")
    print(f"Original prompt: {result.original_prompt}")
    print(f"Compressed prompt: {result.compressed_prompt}")
    print()
    
    # Calculate additional metrics
    metrics = calculate_compression_metrics(result.original_prompt, result.compressed_prompt)
    
    print("Compression Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print("\nResult Summary:")
    summary = result.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print(f"\nEfficiency Score: {result.calculate_efficiency_score():.4f}")
    
    # Save result for later analysis
    result_file = "compression_result.json"
    result.save_to_file(result_file)
    print(f"\nResult saved to {result_file}")
    
    # Clean up
    if os.path.exists(result_file):
        os.remove(result_file)
    
    print()

def example_advanced_configuration():
    """Example of advanced configuration options."""
    print("Example 6: Advanced Configuration")
    print("=" * 40)
    
    # Test different target token counts
    prompt = "Advanced configuration allows fine-tuning of compression parameters for optimal results."
    target_tokens = [5, 10, 15, 20]
    
    print("Testing different target token counts:")
    for target in target_tokens:
        config = LLMLingua2Config(target_token=target)
        compressor = LLMLingua2Compressor(config)
        result = compressor.compress(prompt)
        
        print(f"\nTarget: {target} tokens")
        print(f"  Actual: {result.compressed_token_count} tokens")
        print(f"  Result: '{result.compressed_prompt}'")
        print(f"  Confidence: {result.confidence_score:.3f}")
    
    print("\nTesting different temperature values:")
    temperatures = [0.1, 0.5, 0.9]
    
    for temp in temperatures:
        config = LLMLingua2Config(temperature=temp)
        compressor = LLMLingua2Compressor(config)
        result = compressor.compress(prompt)
        
        print(f"\nTemperature: {temp}")
        print(f"  Result: '{result.compressed_prompt}'")
        print(f"  Confidence: {result.confidence_score:.3f}")
    
    print()

def example_real_world_scenario():
    """Example of a real-world compression scenario."""
    print("Example 7: Real-World Scenario - Document Summarization")
    print("=" * 40)
    
    # Simulate a long document
    document = """
    Introduction to Artificial Intelligence
    
    Artificial Intelligence (AI) refers to the simulation of human intelligence in machines 
    that are programmed to think like humans and mimic their actions. The term may also be 
    applied to any machine that exhibits traits associated with a human mind such as learning 
    and problem-solving.
    
    History of AI
    
    The concept of artificial intelligence dates back to antiquity, but the modern field of 
    AI research was founded in 1956 at a conference at Dartmouth College. Early AI research 
    in the 1950s explored topics like problem solving and symbolic methods. In the 1960s, 
    the US Department of Defense took interest in this type of work and began training 
    computers to mimic basic human reasoning.
    
    Applications of AI
    
    AI has numerous applications across various industries. In healthcare, AI is used for 
    medical diagnosis, drug development, and patient care. In finance, AI algorithms help 
    in fraud detection, algorithmic trading, and risk management. In transportation, 
    self-driving cars represent one of the most significant AI applications.
    
    Future of AI
    
    The future of AI holds immense potential. Experts predict that AI will continue to 
    advance and become more integrated into our daily lives. However, there are also 
    concerns about the ethical implications of AI, including issues related to privacy, 
    bias, and job displacement.
    
    Conclusion
    
    Artificial Intelligence represents one of the most transformative technologies of our time. 
    As AI continues to evolve, it will undoubtedly reshape how we live, work, and interact 
    with technology. The key to harnessing AI's potential lies in responsible development 
    and deployment of these powerful technologies.
    """
    
    print("Original document length: {} words".format(len(document.split())))
    
    # Configure for aggressive compression to create a summary
    config = LLMLingua2Config(
        compression_method=CompressionMethod.AGGRESSIVE.value,
        target_token=50,
        preserve_structure=True
    )
    
    compressor = LLMLingua2Compressor(config)
    result = compressor.compress(document)
    
    print("\nCompressed Summary:")
    print(result.compressed_prompt)
    print(f"\nSummary length: {result.compressed_token_count} words")
    print(f"Compression ratio: {result.compression_ratio:.3f}")
    print(f"Confidence score: {result.confidence_score:.3f}")
    
    # Compare with conservative compression
    conservative_config = LLMLingua2Config(
        compression_method=CompressionMethod.CONSERVATIVE.value,
        target_token=100
    )
    
    conservative_compressor = LLMLingua2Compressor(conservative_config)
    conservative_result = conservative_compressor.compress(document)
    
    print("\nConservative Compression (more detailed):")
    print(conservative_result.compressed_prompt)
    print(f"\nLength: {conservative_result.compressed_token_count} words")
    print(f"Compression ratio: {conservative_result.compression_ratio:.3f}")
    print()

def main():
    """Run all examples."""
    print("LLMLingua2 Examples")
    print("=" * 50)
    
    # Display system information
    print("System Information:")
    sys_info = get_system_info()
    for key, value in sys_info.items():
        print(f"  {key}: {value}")
    print()
    
    # Run all examples
    example_basic_usage()
    example_different_methods()
    example_batch_processing()
    example_configuration_management()
    example_result_analysis()
    example_advanced_configuration()
    example_real_world_scenario()
    
    print("All examples completed!")

if __name__ == "__main__":
    main()
