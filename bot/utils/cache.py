"""Advanced caching system for optimization."""

import hashlib
import json
import time
from typing import Optional, Any, Dict
from functools import wraps
from loguru import logger
import redis
from bot.config import settings


class CacheManager:
    """Redis-based cache manager with fallback to in-memory cache."""

    def __init__(self):
        self.redis_client = None
        self.memory_cache: Dict[str, tuple] = {}  # key -> (value, expiry)

        try:
            if settings.redis_url:
                self.redis_client = redis.from_url(
                    settings.redis_url,
                    decode_responses=True
                )
                self.redis_client.ping()
                logger.info("Redis cache connected")
        except Exception as e:
            logger.warning(f"Redis not available, using memory cache: {e}")

    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        # Create a unique key based on function arguments
        key_data = json.dumps({
            'args': args,
            'kwargs': sorted(kwargs.items())
        }, sort_keys=True, default=str)

        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"{prefix}:{key_hash}"

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            if self.redis_client:
                value = self.redis_client.get(key)
                if value:
                    return json.loads(value)
            else:
                # Memory cache
                if key in self.memory_cache:
                    value, expiry = self.memory_cache[key]
                    if time.time() < expiry:
                        return value
                    else:
                        del self.memory_cache[key]
        except Exception as e:
            logger.error(f"Cache get error: {e}")

        return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: int = 3600
    ) -> bool:
        """Set value in cache with TTL."""
        try:
            if self.redis_client:
                self.redis_client.setex(
                    key,
                    ttl,
                    json.dumps(value, default=str)
                )
            else:
                # Memory cache
                self.memory_cache[key] = (value, time.time() + ttl)

                # Clean expired entries periodically
                if len(self.memory_cache) > 1000:
                    self._clean_memory_cache()

            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            if self.redis_client:
                self.redis_client.delete(key)
            else:
                if key in self.memory_cache:
                    del self.memory_cache[key]
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    def _clean_memory_cache(self):
        """Clean expired entries from memory cache."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, expiry) in self.memory_cache.items()
            if current_time >= expiry
        ]
        for key in expired_keys:
            del self.memory_cache[key]

    def clear_user_cache(self, user_id: int):
        """Clear all cache entries for a user."""
        try:
            if self.redis_client:
                pattern = f"*:user_{user_id}:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            else:
                # Memory cache - clear matching keys
                user_keys = [
                    key for key in self.memory_cache.keys()
                    if f"user_{user_id}" in key
                ]
                for key in user_keys:
                    del self.memory_cache[key]

            logger.info(f"Cleared cache for user {user_id}")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")


# Global cache instance
cache = CacheManager()


def cached(prefix: str, ttl: int = 3600):
    """
    Decorator for caching function results.

    Args:
        prefix: Cache key prefix
        ttl: Time to live in seconds

    Example:
        @cached('ai_response', ttl=1800)
        async def get_ai_response(prompt, model):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache._generate_key(prefix, *args, **kwargs)

            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return cached_value

            # Call function
            result = await func(*args, **kwargs)

            # Store in cache
            cache.set(cache_key, result, ttl)
            logger.debug(f"Cache miss, stored: {cache_key}")

            return result

        return wrapper
    return decorator


class SmartCache:
    """Smart caching with response optimization."""

    def __init__(self):
        self.cache = cache

    def get_similar_responses(
        self,
        prompt: str,
        model: str,
        similarity_threshold: float = 0.9
    ) -> Optional[str]:
        """
        Get cached response for similar prompts.

        This can save API calls for very similar questions.
        """
        # Normalize prompt
        normalized_prompt = self._normalize_prompt(prompt)

        # Create cache key
        cache_key = f"prompt:{model}:{hashlib.md5(normalized_prompt.encode()).hexdigest()}"

        return self.cache.get(cache_key)

    def cache_response(
        self,
        prompt: str,
        model: str,
        response: str,
        tokens: int,
        ttl: int = 7200
    ):
        """Cache AI response with metadata."""
        normalized_prompt = self._normalize_prompt(prompt)
        cache_key = f"prompt:{model}:{hashlib.md5(normalized_prompt.encode()).hexdigest()}"

        data = {
            'response': response,
            'tokens': tokens,
            'timestamp': time.time(),
            'model': model
        }

        self.cache.set(cache_key, data, ttl)

    def _normalize_prompt(self, prompt: str) -> str:
        """Normalize prompt for better cache matching."""
        # Convert to lowercase
        normalized = prompt.lower()

        # Remove extra whitespace
        normalized = ' '.join(normalized.split())

        # Remove punctuation at the end
        normalized = normalized.rstrip('.,!?;:')

        return normalized

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        if cache.redis_client:
            try:
                info = cache.redis_client.info('stats')
                return {
                    'hits': info.get('keyspace_hits', 0),
                    'misses': info.get('keyspace_misses', 0),
                    'hit_rate': info.get('keyspace_hits', 0) /
                               max(info.get('keyspace_hits', 0) +
                                   info.get('keyspace_misses', 0), 1)
                }
            except:
                pass

        return {
            'hits': 0,
            'misses': 0,
            'hit_rate': 0,
            'memory_entries': len(cache.memory_cache)
        }


# Global smart cache instance
smart_cache = SmartCache()
