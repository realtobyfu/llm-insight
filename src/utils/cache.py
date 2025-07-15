"""Caching utilities for the interpretability toolkit"""

import hashlib
import json
import pickle
from typing import Any, Dict, Optional, Union

import redis
from redis.exceptions import RedisError

from ..core.config import CacheConfig
from .logger import get_logger

logger = get_logger(__name__)


class CacheManager:
    """Manages caching for analysis results"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.enabled = config.enabled
        self.backend = config.backend
        self.ttl = config.ttl
        
        if self.enabled and self.backend == "redis":
            try:
                self.redis_client = redis.from_url(
                    config.redis_url,
                    decode_responses=False  # We'll handle encoding/decoding
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Redis cache initialized successfully")
            except RedisError as e:
                logger.error(f"Failed to connect to Redis: {e}")
                logger.warning("Falling back to in-memory cache")
                self.backend = "memory"
                self.memory_cache = {}
        else:
            self.memory_cache = {}
    
    def _generate_key(self, prefix: str, params: Dict[str, Any]) -> str:
        """Generate cache key from parameters"""
        # Sort parameters for consistent hashing
        sorted_params = json.dumps(params, sort_keys=True)
        hash_digest = hashlib.md5(sorted_params.encode()).hexdigest()
        return f"{prefix}:{hash_digest}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.enabled:
            return None
        
        try:
            if self.backend == "redis":
                value = self.redis_client.get(key)
                if value:
                    return pickle.loads(value)
            else:
                return self.memory_cache.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        if not self.enabled:
            return False
        
        ttl = ttl or self.ttl
        
        try:
            if self.backend == "redis":
                serialized = pickle.dumps(value)
                return self.redis_client.setex(key, ttl, serialized)
            else:
                self.memory_cache[key] = value
                # Simple size limit for memory cache
                if len(self.memory_cache) > self.config.max_size:
                    # Remove oldest entries
                    for k in list(self.memory_cache.keys())[:10]:
                        del self.memory_cache[k]
                return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if not self.enabled:
            return False
        
        try:
            if self.backend == "redis":
                return bool(self.redis_client.delete(key))
            else:
                return self.memory_cache.pop(key, None) is not None
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries"""
        if not self.enabled:
            return False
        
        try:
            if self.backend == "redis":
                self.redis_client.flushdb()
            else:
                self.memory_cache.clear()
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "enabled": self.enabled,
            "backend": self.backend,
        }
        
        if self.enabled:
            if self.backend == "redis":
                try:
                    info = self.redis_client.info()
                    stats.update({
                        "keys": self.redis_client.dbsize(),
                        "memory_used": info.get("used_memory_human", "N/A"),
                        "hits": info.get("keyspace_hits", 0),
                        "misses": info.get("keyspace_misses", 0),
                    })
                except Exception as e:
                    logger.error(f"Failed to get Redis stats: {e}")
            else:
                stats.update({
                    "keys": len(self.memory_cache),
                    "max_size": self.config.max_size,
                })
        
        return stats


def cached_analysis(cache_manager: CacheManager, prefix: str = "analysis"):
    """Decorator for caching analysis results"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract cacheable parameters
            cache_params = {
                "func": func.__name__,
                "args": str(args[1:]),  # Skip self
                "kwargs": str(kwargs),
            }
            
            # Generate cache key
            cache_key = cache_manager._generate_key(prefix, cache_params)
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache_manager.set(cache_key, result)
            logger.debug(f"Cached result for {func.__name__}")
            
            return result
        
        return wrapper
    return decorator