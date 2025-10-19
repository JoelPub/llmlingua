# LLMLingua & LLMLingua2

LLMLingua and LLMLingua2 are powerful text prompt compression libraries designed to optimize prompts for large language models (LLMs). This repository contains implementations for both versions, allowing users to choose between them based on their specific needs.

## Table of Contents

- [LLMLingua (Version 1)](#llmlingua-version-1)
- [LLMLingua2 (Version 2)](#llmlingua2-version-2)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Comparison](#comparison)
- [Contributing](#contributing)

## LLMLingua (Version 1)

The original LLMLingua implementation provides robust text compression capabilities with proven performance and reliability.

### Installation (LLMLingua v1)

```bash
pip install llmlingua
```

### Basic Usage (LLMLingua v1)

```python
# See test.py for basic usage example
python test.py
```

## LLMLingua2 (Version 2)

LLMLingua2 is the next-generation text compression system that offers enhanced performance, new compression methods, and improved configuration options while maintaining focus on text prompt compression.

### Features

- **Enhanced Compression Algorithms**: Improved compression ratios with multiple method options
- **Flexible Configuration**: Extensive configuration options for fine-tuning
- **Batch Processing**: Efficient batch compression for multiple prompts
- **Performance Analytics**: Built-in statistics and performance metrics
- **Multiple Model Support**: Support for different model types and sizes
- **Confidence Scoring**: Quality assessment for compression results

### Installation (LLMLingua2)

#### Option 1: Install from requirements file

```bash
pip install -r requirements_llmlingua2.txt
```

#### Option 2: Install individual packages

```bash
pip install llmlingua2 transformers>=4.35.0 torch>=2.0.0 accelerate>=0.24.0 sentencepiece>=0.1.99 numpy>=1.24.0 pandas>=2.0.0 tqdm>=4.65.0 pydantic>=2.0.0
```

### Quick Start (LLMLingua2)

```python
from llmlingua2_config import LLMLingua2Config
from llmlingua2_compressor import LLMLingua2Compressor

# Initialize with default configuration
config = LLMLingua2Config()
compressor = LLMLingua2Compressor(config)

# Compress a prompt
prompt = "This is a sample prompt that we want to compress using LLMLingua2."
result = compressor.compress(prompt)

print(f"Original: {result.original_token_count} tokens")
print(f"Compressed: {result.compressed_token_count} tokens")
print(f"Compression ratio: {result.compression_ratio:.3f}")
print(f"Compressed text: {result.compressed_prompt}")
```

### Usage Examples (LLMLingua2)

#### Basic Compression

```python
from llmlingua2_config import LLMLingua2Config, CompressionMethod
from llmlingua2_compressor import LLMLingua2Compressor

# Create configuration
config = LLMLingua2Config(
    compression_method=CompressionMethod.BALANCED.value,
    target_token=100
)

# Initialize compressor
compressor = LLMLingua2Compressor(config)

# Compress text
prompt = "Your long text prompt goes here..."
result = compressor.compress(prompt)
```

#### Batch Processing

```python
prompts = [
    "First prompt to compress",
    "Second prompt to compress",
    "Third prompt to compress"
]

results = compressor.batch_compress(prompts, show_progress=True)

for i, result in enumerate(results):
    print(f"Prompt {i+1}: {result.compressed_token_count}/{result.original_token_count} tokens")
```

#### Different Compression Methods

```python
from llmlingua2_config import CompressionMethod

methods = [
    CompressionMethod.CONSERVATIVE,  # Minimal compression, high quality
    CompressionMethod.BALANCED,       # Balanced compression and quality
    CompressionMethod.STANDARD,       # Standard compression
    CompressionMethod.AGGRESSIVE      # Maximum compression
]

for method in methods:
    config = LLMLingua2Config(compression_method=method.value)
    compressor = LLMLingua2Compressor(config)
    result = compressor.compress(prompt)
    print(f"{method.value}: {result.compression_ratio:.3f} ratio")
```

#### Configuration Management

```python
# Create custom configuration
config = LLMLingua2Config(
    model_name="microsoft/llmlingua-2-medium",
    device_map="cpu",
    target_token=150,
    compression_method=CompressionMethod.STANDARD.value,
    temperature=0.7,
    top_p=0.9,
    batch_size=4
)

# Save configuration
config.save("my_config.json")

# Load configuration later
loaded_config = LLMLingua2Config.load("my_config.json")
```

### Running Examples and Tests

#### Run LLMLingua2 Test Suite

```bash
python llmlingua2_test.py
```

#### Run LLMLingua2 Examples

```bash
python llmlingua2_examples.py
```

## API Reference (LLMLingua2)

### LLMLingua2Config

Configuration class for LLMLingua2 compression parameters.

#### Parameters

- `model_name` (str): Model identifier for compression (default: "microsoft/llmlingua-2")
- `device_map` (str): Device mapping - "cpu", "cuda", "mps", or "auto" (default: "auto")
- `target_token` (int): Target token count for compression (default: 512)
- `compression_method` (str): Compression method - "conservative", "balanced", "standard", "aggressive", or "custom" (default: "balanced")
- `temperature` (float): Sampling temperature (default: 0.7)
- `top_p` (float): Nucleus sampling parameter (default: 0.9)
- `max_length` (int): Maximum output length (default: 1024)
- `min_length` (int): Minimum output length (default: 50)
- `preserve_structure` (bool): Whether to preserve text structure (default: True)
- `custom_model_path` (str, optional): Path to custom model
- `batch_size` (int): Batch processing size (default: 1)
- `cache_dir` (str, optional): Directory for model caching

#### Methods

- `save(filepath)`: Save configuration to JSON file
- `load(filepath)`: Load configuration from JSON file
- `validate()`: Validate configuration parameters
- `update(**kwargs)`: Update configuration parameters

### LLMLingua2Compressor

Main compression class for LLMLingua2.

#### Methods

- `compress(prompt, target_token=None, method=None)`: Compress a single prompt
- `batch_compress(prompts, show_progress=True)`: Compress multiple prompts
- `get_stats()`: Get compression statistics
- `reset_config(config)`: Reset compressor configuration
- `estimate_compression(prompt)`: Estimate compression performance

### CompressionResult

Data class containing compression results.

#### Attributes

- `original_prompt`: Original input prompt
- `compressed_prompt`: Compressed output prompt
- `original_token_count`: Number of tokens in original prompt
- `compressed_token_count`: Number of tokens in compressed prompt
- `compression_ratio`: Compression ratio (0.0 to 1.0)
- `processing_time`: Time taken for compression
- `method_used`: Compression method used
- `model_used`: Model used for compression
- `confidence_score`: Quality confidence score (optional)
- `metadata`: Additional metadata dictionary

#### Methods

- `to_dict()`: Convert to dictionary
- `from_dict(data)`: Create from dictionary
- `save_to_file(filepath)`: Save to JSON file
- `load_from_file(filepath)`: Load from JSON file
- `get_summary()`: Get result summary
- `calculate_efficiency_score()`: Calculate efficiency score
- `is_better_than(other, prioritize_ratio=True)`: Compare with another result

## Comparison

### LLMLingua vs LLMLingua2

| Feature | LLMLingua v1 | LLMLingua2 |
|---------|--------------|------------|
| Compression Methods | Basic | Advanced (Conservative, Balanced, Standard, Aggressive, Custom) |
| Configuration Options | Limited | Extensive |
| Batch Processing | Basic | Enhanced with progress tracking |
| Performance Analytics | Basic | Comprehensive statistics |
| Model Support | Single model | Multiple model types |
| Confidence Scoring | No | Yes |
| Configuration Persistence | No | Yes (save/load) |
| Error Handling | Basic | Advanced with validation |

### When to Use Which Version

**Use LLMLingua v1 when:**
- You need a simple, proven solution
- You have existing code that uses v1
- You require minimal configuration
- You're working with legacy systems

**Use LLMLingua2 when:**
- You need the best compression ratios
- You require fine-tuned control over compression
- You're processing large batches of prompts
- You need detailed performance analytics
- You want to experiment with different compression methods

## Contributing

We welcome contributions to both LLMLingua and LLMLingua2! Please feel free to submit issues and pull requests.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Install development dependencies:
   ```bash
   pip install -r requirements_llmlingua2.txt
   ```
4. Run tests to ensure everything works:
   ```bash
   python llmlingua2_test.py
   python llmlingua2_examples.py
   ```
5. Make your changes and add tests if applicable
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Microsoft Research for the original LLMLingua concept
- Contributors to the LLMLingua2 implementation
- The open-source community for their valuable feedback and contributions
