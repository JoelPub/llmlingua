import time
import logging
from typing import List, Optional, Dict, Any
from tqdm import tqdm

from llmlingua2_config import LLMLingua2Config, CompressionMethod
from llmlingua2_results import CompressionResult
from llmlingua2_utils import (
    calculate_compression_metrics, 
    validate_compression_result, 
    sanitize_prompt,
    estimate_compression_ratio
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMLingua2Compressor:
    """
    Main compression class for llmlingua2.
    
    This class handles to core compression functionality, including
    single prompt compression, batch processing, and statistics tracking.
    """
    
    def __init__(self, config: Optional[LLMLingua2Config] = None):
        """
        Initialize compressor with configuration.
        
        Args:
            config: LLMLingua2Config object. If None, uses default config.
        """
        self.config = config or LLMLingua2Config()
        self.model = None
        self.tokenizer = None
        self.stats = {
            'total_compressions': 0,
            'total_tokens_saved': 0,
            'average_compression_ratio': 0.0,
            'total_processing_time': 0.0
        }
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize compression model and tokenizer."""
        try:
            # This is a placeholder for actual model initialization
            # In a real implementation, this would load to actual model
            logger.info(f"Initializing model: {self.config.model_name}")
            logger.info(f"Device map: {self.config.device_map}")
            
            # Simulate model loading
            self.model = f"MockModel({self.config.model_name})"
            self.tokenizer = f"MockTokenizer({self.config.model_name})"
            
            logger.info("Model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def compress(self, prompt: str, 
                target_token: Optional[int] = None,
                method: Optional[str] = None) -> CompressionResult:
        """
        Compress a single prompt.
        
        Args:
            prompt: Input text prompt to compress
            target_token: Optional override for target token count
            method: Optional override for compression method
            
        Returns:
            CompressionResult object with compression details
        """
        start_time = time.time()
        
        # Sanitize input
        prompt = sanitize_prompt(prompt)
        
        # Use provided overrides or config defaults
        target_token = target_token or self.config.target_token
        method = method or self.config.compression_method
        
        try:
            # Calculate original token count (simplified)
            original_token_count = len(prompt.split())
            
            # Simulate compression process
            # In a real implementation, this would use to actual model
            compressed_prompt = self._simulate_compression(prompt, target_token, method)
            
            # Calculate compressed token count
            compressed_token_count = len(compressed_prompt.split())
            
            # Calculate compression ratio
            compression_ratio = 1 - (compressed_token_count / original_token_count) if original_token_count > 0 else 0
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create result object
            result = CompressionResult(
                original_prompt=prompt,
                compressed_prompt=compressed_prompt,
                original_token_count=original_token_count,
                compressed_token_count=compressed_token_count,
                compression_ratio=compression_ratio,
                processing_time=processing_time,
                method_used=method,
                model_used=self.config.model_name,
                confidence_score=self._calculate_confidence_score(prompt, compressed_prompt),
                metadata={
                    'target_token': target_token,
                    'preserve_structure': self.config.preserve_structure,
                    'temperature': self.config.temperature,
                    'top_p': self.config.top_p
                }
            )
            
            # Update statistics
            self._update_stats(result)
            
            # Validate result
            if not validate_compression_result(result):
                logger.warning("Compression result validation failed")
            
            return result
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise
    
    def batch_compress(self, prompts: List[str], 
                      show_progress: bool = True) -> List[CompressionResult]:
        """
        Compress multiple prompts in batch.
        
        Args:
            prompts: List of input prompts to compress
            show_progress: Whether to show progress bar
            
        Returns:
            List of CompressionResult objects
        """
        results = []
        
        # Create progress bar
        iterator = tqdm(prompts, desc="Compressing prompts") if show_progress else prompts
        
        for prompt in iterator:
            try:
                result = self.compress(prompt)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to compress prompt: {e}")
                # Create a failed result
                results.append(CompressionResult(
                    original_prompt=prompt,
                    compressed_prompt="",
                    original_token_count=len(prompt.split()),
                    compressed_token_count=0,
                    compression_ratio=0.0,
                    processing_time=0.0,
                    method_used=self.config.compression_method,
                    model_used=self.config.model_name,
                    metadata={'error': str(e)}
                ))
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get compression statistics.
        
        Returns:
            Dictionary containing compression statistics
        """
        stats = self.stats.copy()
        
        # Calculate average compression ratio
        if stats['total_compressions'] > 0:
            stats['average_compression_ratio'] = (
                stats['total_tokens_saved'] / 
                (stats['total_tokens_saved'] + stats.get('total_compressed_tokens', 0))
                if (stats['total_tokens_saved'] + stats.get('total_compressed_tokens', 0)) > 0 else 0
            )
        
        stats['config'] = self.config.dict()
        return stats
    
    def reset_config(self, config: LLMLingua2Config):
        """
        Reset compressor configuration.
        
        Args:
            config: New LLMLingua2Config object
        """
        self.config = config
        self._initialize_model()
        logger.info("Configuration reset successfully")
    
    def _simulate_compression(self, prompt: str, target_token: int, method: str) -> str:
        """
        Simulate compression process.
        
        This is a placeholder for actual compression logic.
        In a real implementation, this would use to loaded model.
        
        Args:
            prompt: Input prompt
            target_token: Target token count
            method: Compression method
            
        Returns:
            Compressed prompt string
        """
        words = prompt.split()
        
        # Simple compression simulation based on method
        if method == CompressionMethod.CONSERVATIVE.value:
            # Keep 80% of words
            keep_ratio = 0.8
        elif method == CompressionMethod.BALANCED.value:
            # Keep 60% of words
            keep_ratio = 0.6
        elif method == CompressionMethod.STANDARD.value:
            # Keep 50% of words
            keep_ratio = 0.5
        elif method == CompressionMethod.AGGRESSIVE.value:
            # Keep 30% of words
            keep_ratio = 0.3
        else:  # CUSTOM
            # Keep 45% of words
            keep_ratio = 0.45
        
        # Calculate number of words to keep
        words_to_keep = max(int(len(words) * keep_ratio), 1)
        
        # Ensure we don't exceed target token count
        if target_token > 0:
            words_to_keep = min(words_to_keep, target_token)
        
        # Select words to keep (simple approach: take from beginning)
        compressed_words = words[:words_to_keep]
        
        # Join words back into a string
        compressed_prompt = ' '.join(compressed_words)
        
        return compressed_prompt
    
    def _calculate_confidence_score(self, original: str, compressed: str) -> float:
        """
        Calculate a confidence score for the compression quality.
        
        Args:
            original: Original prompt
            compressed: Compressed prompt
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Simple heuristic-based confidence calculation
        original_words = set(original.lower().split())
        compressed_words = set(compressed.lower().split())
        
        # Calculate word overlap
        if len(original_words) == 0:
            return 0.0
        
        overlap = len(original_words.intersection(compressed_words))
        overlap_ratio = overlap / len(original_words)
        
        # Adjust based on compression ratio
        compression_ratio = 1 - (len(compressed.split()) / len(original.split())) if len(original.split()) > 0 else 0
        
        # Combine metrics
        confidence = (overlap_ratio * 0.7 + compression_ratio * 0.3)
        
        return round(min(1.0, max(0.0, confidence)), 4)
    
    def _update_stats(self, result: CompressionResult):
        """Update compression statistics."""
        self.stats['total_compressions'] += 1
        self.stats['total_tokens_saved'] += (
            result.original_token_count - result.compressed_token_count
        )
        self.stats['total_processing_time'] += result.processing_time
        
        # Track total compressed tokens for ratio calculation
        if 'total_compressed_tokens' not in self.stats:
            self.stats['total_compressed_tokens'] = 0
        self.stats['total_compressed_tokens'] += result.compressed_token_count
    
    def estimate_compression(self, prompt: str) -> Dict[str, Any]:
        """
        Estimate compression performance for a prompt without actually compressing.
        
        Args:
            prompt: Input prompt to analyze
            
        Returns:
            Dictionary with compression estimates
        """
        method = CompressionMethod(self.config.compression_method)
        estimated_ratio = estimate_compression_ratio(prompt, method)
        
        original_tokens = len(prompt.split())
        estimated_tokens = int(original_tokens * (1 - estimated_ratio))
        
        return {
            'original_tokens': original_tokens,
            'estimated_compressed_tokens': estimated_tokens,
            'estimated_compression_ratio': estimated_ratio,
            'estimated_tokens_saved': original_tokens - estimated_tokens,
            'method': self.config.compression_method,
            'confidence': 'medium'  # Placeholder
        }
