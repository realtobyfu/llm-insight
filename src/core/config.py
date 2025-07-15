"""Configuration management for the interpretability toolkit"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ModelConfig:
    """Configuration for model-specific settings"""
    
    name: str
    architecture: str
    max_sequence_length: int = 512
    batch_size: int = 8
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    dtype: str = "float32"
    cache_dir: Optional[Path] = None


@dataclass
class APIConfig:
    """Configuration for API settings"""
    
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "INFO"
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    timeout: int = 300  # 5 minutes


@dataclass
class CacheConfig:
    """Configuration for caching settings"""
    
    enabled: bool = True
    backend: str = "redis"  # redis or memory
    redis_url: str = "redis://localhost:6379"
    ttl: int = 3600  # 1 hour
    max_size: int = 1000  # max number of cached items


@dataclass
class LoggingConfig:
    """Configuration for logging settings"""
    
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[Path] = Path("logs/app.log")
    rotation: str = "daily"
    retention: int = 7  # days


@dataclass
class MonitoringConfig:
    """Configuration for monitoring settings"""
    
    enabled: bool = True
    metrics_port: int = 8001
    wandb_project: Optional[str] = os.environ.get("WANDB_PROJECT")
    wandb_entity: Optional[str] = os.environ.get("WANDB_ENTITY")
    opentelemetry_endpoint: Optional[str] = None


class Config:
    """Main configuration class"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config.yaml")
        self._config = self._load_config()
        
        # Initialize sub-configurations
        self.model = ModelConfig(**self._config.get("model", {}))
        self.api = APIConfig(**self._config.get("api", {}))
        self.cache = CacheConfig(**self._config.get("cache", {}))
        self.logging = LoggingConfig(**self._config.get("logging", {}))
        self.monitoring = MonitoringConfig(**self._config.get("monitoring", {}))
        
        # Paths
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.logs_dir = self.project_root / "logs"
        
        # Create directories
        self._create_directories()
    
    def _load_config(self) -> Dict:
        """Load configuration from file"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _create_directories(self):
        """Create necessary directories"""
        for dir_path in [self.data_dir, self.models_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def save(self):
        """Save current configuration to file"""
        config_dict = {
            "model": {
                "name": self.model.name,
                "architecture": self.model.architecture,
                "max_sequence_length": self.model.max_sequence_length,
                "batch_size": self.model.batch_size,
                "device": self.model.device,
                "dtype": self.model.dtype,
            },
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "workers": self.api.workers,
                "log_level": self.api.log_level,
                "cors_origins": self.api.cors_origins,
            },
            "cache": {
                "enabled": self.cache.enabled,
                "backend": self.cache.backend,
                "redis_url": self.cache.redis_url,
                "ttl": self.cache.ttl,
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "file": str(self.logging.file) if self.logging.file else None,
            },
            "monitoring": {
                "enabled": self.monitoring.enabled,
                "metrics_port": self.monitoring.metrics_port,
                "wandb_project": self.monitoring.wandb_project,
            },
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables"""
        config = cls()
        
        # Override with environment variables
        config.model.device = os.environ.get("MODEL_DEVICE", config.model.device)
        config.api.host = os.environ.get("API_HOST", config.api.host)
        config.api.port = int(os.environ.get("API_PORT", config.api.port))
        config.cache.redis_url = os.environ.get("REDIS_URL", config.cache.redis_url)
        config.logging.level = os.environ.get("LOG_LEVEL", config.logging.level)
        
        return config