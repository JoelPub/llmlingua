import time
import json
from typing import List, Dict, Any, Optional
from .llmlingua2_config import LLMLingua2Config, CompressionMethod, CompressionResult

def calculate_compression_metrics(original: str, compressed: str) -> Dict[str, float]:
    """
    Calculate compression statistics between original and compressed text.
    
    Args:
        original: The original text string
        compressed: The compressed text string
        
    Returns:
        Dictionary containing compression metrics
    """
    original_chars = len(original)
    compressed_chars = len(compressed)
    
    # Simple word-based tokenization for estimation
    original_tokens = len(original.split())
    compressed_tokens = len(compressed.split())
    
    char_ratio = compressed_chars / original_chars if original_chars > 0 else 0
    token_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 0
    
    return {
        "original_char_count": original_chars,
        "compressed_char_count": compressed_chars,
        "original_token_count": original_tokens,
        "compressed_token_count": compressed_tokens,
        "char_compression_ratio": round(1 - char_ratio, 4),
        "token_compression_ratio": round(1 - token_ratio, 4),
        "space_saving_percent": round((1 - char_ratio) * 100, 2)
    }

def validate_compression_result(result: CompressionResult) -> bool:
    """
    Validate the compression result structure and values.
    
    Args:
        result: CompressionResult object to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(result, CompressionResult):
        return False
    
    # Check required fields
    required_fields = [
        'original_prompt', 'compressed_prompt', 'original_token_count',
        'compressed_token_count', 'compression_ratio', 'processing_time',
        'method_used', 'model_used'
    ]
    
    for field in required_fields:
        if not hasattr(result, field):
            return False
    
    # Validate value ranges
    if result.compression_ratio < 0 or result.compression_ratio > 1:
        return False
    
    if result.processing_time < 0:
        return False
    
    if result.compressed_token_count > result.original_token_count:
        return False
    
    return True

def save_compression_result(result: CompressionResult, filepath: str):
    """
    Save compression result to a JSON file.
    
    Args:
        result: CompressionResult object to save
        filepath: Path to save the result file
    """
    result_dict = result.to_dict() if hasattr(result, 'to_dict') else result.__dict__
    with open(filepath, 'w') as f:
        json.dump(result_dict, f, indent=4, default=str)

def load_compression_config(filepath: str) -> LLMLingua2Config:
    """
    Load llmlingua2 configuration from a JSON file.
    
    Args:
        filepath: Path to the configuration file
        
    Returns:
        LLMLingua2Config object
    """
    return LLMLingua2Config.load(filepath)

def get_available_models() -> List[str]:
    """
    Get list of available compression models.
    
    Returns:
        List of model identifiers
    """
    # This would typically query a model registry or API
    # For now, returning a static list of known models
    return [
        "microsoft/llmlingua-2-small",
        "microsoft/llmlingua-2-medium", 
        "microsoft/llmlingua-2-large",
        "microsoft/llmlingua-2-xlarge"
    ]

def estimate_compression_ratio(prompt: str, method: CompressionMethod) -> float:
    """
    Estimate compression ratio for a given prompt and method.
    
    Args:
        prompt: Input text prompt
        method: Compression method to use
        
    Returns:
        Estimated compression ratio (0.0 to 1.0)
    """
    # Simple heuristic-based estimation
    prompt_length = len(prompt)
    word_count = len(prompt.split())
    
    # Base ratios for different methods
    method_ratios = {
        CompressionMethod.CONSERVATIVE: 0.7,
        CompressionMethod.BALANCED: 0.5,
        CompressionMethod.STANDARD: 0.4,
        CompressionMethod.AGGRESSIVE: 0.3,
        CompressionMethod.CUSTOM: 0.45
    }
    
    base_ratio = method_ratios.get(method, 0.5)
    
    # Adjust based on prompt characteristics
    if word_count > 500:
        base_ratio *= 0.9  # Slightly better compression for longer texts
    elif word_count < 50:
        base_ratio *= 1.1  # Slightly worse for very short texts
    
    return max(0.1, min(0.9, base_ratio))

def format_processing_time(seconds: float) -> str:
    """
    Format processing time in human-readable format.
    
    Args:
        seconds: Processing time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"

def sanitize_prompt(prompt: str) -> str:
    """
    Sanitize input prompt by removing problematic characters.
    
    Args:
        prompt: Raw input prompt
        
    Returns:
        Sanitized prompt string
    """
    # Remove null bytes and other problematic characters
    sanitized = prompt.replace('\x00', '')
    # Normalize whitespace
    sanitized = ' '.join(sanitized.split())
    return sanitized.strip()

def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging and optimization.
    
    Returns:
        Dictionary containing system information
    """
    import platform
    import psutil
    
    return {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "cpu_count": psutil.cpu_count(),
        "python_version": platform.python_version()
    }
