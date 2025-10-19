from enum import Enum
from typing import Optional, Dict, Any
import json
from pydantic import BaseModel, validator

class CompressionMethod(Enum):
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"
    CUSTOM = "custom"

class ModelType(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    CUSTOM = "custom"

class LLMLingua2Config(BaseModel):
    model_name: str = "microsoft/llmlingua-2"
    device_map: str = "auto"
    target_token: int = 512
    compression_method: str = CompressionMethod.BALANCED.value
    temperature: float = 0.7
    top_p: float = 0.9
    max_length: int = 1024
    min_length: int = 50
    preserve_structure: bool = True
    custom_model_path: Optional[str] = None
    batch_size: int = 1
    cache_dir: Optional[str] = None

    @validator('compression_method')
    def validate_compression_method(cls, v):
        allowed_methods = [method.value for method in CompressionMethod]
        if v not in allowed_methods:
            raise ValueError(f"compression_method must be one of {allowed_methods}")
        return v

    @validator('device_map')
    def validate_device_map(cls, v):
        allowed_devices = ["cpu", "cuda", "mps", "auto"]
        if v not in allowed_devices:
            raise ValueError(f"device_map must be one of {allowed_devices}")
        return v

    @validator('target_token', 'max_length', 'min_length', 'batch_size')
    def validate_positive_integers(cls, v):
        if v <= 0:
            raise ValueError("Value must be a positive integer")
        return v

    @validator('temperature', 'top_p')
    def validate_float_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Value must be between 0.0 and 1.0")
        return v

    def validate(self) -> bool:
        try:
            LLMLingua2Config(**self.dict())
            return True
        except Exception:
            return False

    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.dict(), f, indent=4)

    @classmethod
    def load(cls, filepath: str) -> 'LLMLingua2Config':
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
        # Re-validate after update
        self.validate()
