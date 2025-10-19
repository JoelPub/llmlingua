import json
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from llmlingua2_config import CompressionMethod

@dataclass
class CompressionResult:
    """
    Data class to store compression results and metadata.
    """
    original_prompt: str
    compressed_prompt: str
    original_token_count: int
    compressed_token_count: int
    compression_ratio: float
    processing_time: float
    method_used: str
    model_used: str
    confidence_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate compression result after initialization."""
        if self.compressed_token_count > self.original_token_count:
            raise ValueError("Compressed token count cannot be greater than original token count")
        
        if not 0 <= self.compression_ratio <= 1:
            raise ValueError("Compression ratio must be between 0 and 1")
        
        if self.processing_time < 0:
            raise ValueError("Processing time cannot be negative")
        
        if self.confidence_score is not None and not 0 <= self.confidence_score <= 1:
            raise ValueError("Confidence score must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert compression result to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompressionResult':
        """Create a CompressionResult from a dictionary."""
        return cls(**data)

    def save_to_file(self, filepath: str):
        """Save compression result to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4, default=str)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'CompressionResult':
        """Load a compression result from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of compression result."""
        return {
            "original_tokens": self.original_token_count,
            "compressed_tokens": self.compressed_token_count,
            "tokens_saved": self.original_token_count - self.compressed_token_count,
            "compression_ratio": self.compression_ratio,
            "compression_percent": round(self.compression_ratio * 100, 2),
            "processing_time": self.processing_time,
            "method": self.method_used,
            "model": self.model_used,
            "confidence": self.confidence_score
        }

    def __str__(self) -> str:
        """String representation of compression result."""
        summary = self.get_summary()
        return (
            f"CompressionResult:\n"
            f"  Method: {summary['method']}\n"
            f"  Model: {summary['model']}\n"
            f"  Original Tokens: {summary['original_tokens']}\n"
            f"  Compressed Tokens: {summary['compressed_tokens']}\n"
            f"  Tokens Saved: {summary['tokens_saved']}\n"
            f"  Compression Ratio: {summary['compression_ratio']:.3f} ({summary['compression_percent']}%)\n"
            f"  Processing Time: {summary['processing_time']:.3f}s\n"
            f"  Confidence Score: {summary['confidence'] or 'N/A'}"
        )

    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return f"CompressionResult(original_tokens={self.original_token_count}, " \
               f"compressed_tokens={self.compressed_token_count}, " \
               f"ratio={self.compression_ratio:.3f}, " \
               f"method='{self.method_used}', " \
               f"model='{self.model_used}')"

    def is_better_than(self, other: 'CompressionResult', 
                      prioritize_ratio: bool = True) -> bool:
        """
        Compare this result with another compression result.
        
        Args:
            other: Another CompressionResult to compare against
            prioritize_ratio: If True, prioritize compression ratio over speed
            
        Returns:
            True if this result is better than other
        """
        if prioritize_ratio:
            # Higher compression ratio is better (more compression)
            if self.compression_ratio > other.compression_ratio:
                return True
            elif self.compression_ratio == other.compression_ratio:
                # If ratios are equal, faster processing is better
                return self.processing_time < other.processing_time
            else:
                return False
        else:
            # Faster processing is better
            if self.processing_time < other.processing_time:
                return True
            elif self.processing_time == other.processing_time:
                # If times are equal, higher compression ratio is better
                return self.compression_ratio > other.compression_ratio
            else:
                return False

    def calculate_efficiency_score(self) -> float:
        """
        Calculate an efficiency score combining compression ratio and speed.
        
        Returns:
            Efficiency score (higher is better)
        """
        # Normalize processing time (lower is better, so we use inverse)
        # Add small constant to avoid division by zero
        time_score = 1.0 / (self.processing_time + 0.001)
        
        # Weight compression ratio more heavily (0.7) vs speed (0.3)
        efficiency = (0.7 * self.compression_ratio + 0.3 * min(time_score, 1.0))
        
        return round(efficiency, 4)
