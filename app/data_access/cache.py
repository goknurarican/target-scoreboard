# Copyright (c) 2025 Göknur Arıcan
# All rights reserved. Licensed for internal evaluation only.
# See LICENSE-EVALUATION.md for terms.

"""
Production-ready caching utilities for VantAI Target Scoreboard.
Supports in-memory, file-cache, and optional Redis with TTL and stable keys.
"""

import asyncio
import hashlib
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
from datetime import datetime, timedelta

import pandas as pd

# Redis support (optional)
try:
    import redis.asyncio as aioredis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None

# Configure logger
logger = logging.getLogger(__name__)

# Configuration from environment
CACHE_DIR = Path(os.getenv("CACHE_DIR", "cache"))
REDIS_URL = os.getenv("REDIS_URL")
CACHE_NAMESPACE = os.getenv("CACHE_NAMESPACE", "vantai_v1")

# Per-source TTL configuration (seconds)
DEFAULT_TTL_CONFIG = {
    "opentargets": 24 * 3600,  # 24 hours
    "stringdb": 7 * 24 * 3600,  # 7 days
    "expression_atlas": 7 * 24 * 3600,  # 7 days
    "alphafold": 7 * 24 * 3600,  # 7 days
    "pubmed": 7 * 24 * 3600,  # 7 days
    "default": 3 * 3600  # 3 hours
}


class CacheKeyBuilder:
    """Builds stable, normalized cache keys."""

    @staticmethod
    def build_key(source: str, method: str, **params) -> str:
        """
        Build stable cache key with format: <SOURCE>::<METHOD>::<normalized_params>

        Args:
            source: Data source (opentargets, stringdb, etc.)
            method: API method name
            **params: Method parameters

        Returns:
            Stable cache key string
        """
        # Normalize parameters for consistent keys
        normalized_params = {}
        for key, value in params.items():
            if isinstance(value, str):
                normalized_params[key] = value.strip().lower()
            elif isinstance(value, (list, tuple)):
                normalized_params[key] = sorted([str(v).strip().lower() for v in value])
            else:
                normalized_params[key] = value

        # Create deterministic string
        param_str = json.dumps(normalized_params, sort_keys=True, default=str)

        # Build key with namespace
        key = f"{CACHE_NAMESPACE}::{source.upper()}::{method.upper()}::{param_str}"

        # Hash for length control
        return hashlib.sha256(key.encode()).hexdigest()


class ProductionCacheManager:
    """
    Production-ready cache manager with multiple backends and structured logging.
    """

    def __init__(
            self,
            cache_dir: Optional[Path] = None,
            redis_url: Optional[str] = None,
            ttl_config: Optional[Dict[str, int]] = None
    ):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for file cache (None = use CACHE_DIR env)
            redis_url: Redis connection URL (None = use REDIS_URL env or disable)
            ttl_config: Per-source TTL configuration
        """
        self.cache_dir = cache_dir or CACHE_DIR
        self.redis_url = redis_url or REDIS_URL
        self.ttl_config = ttl_config or DEFAULT_TTL_CONFIG.copy()

        # Initialize backends
        self._redis_client: Optional[aioredis.Redis] = None
        self._memory_cache: Dict[str, Dict[str, Any]] = {}

        # Setup file cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "data").mkdir(exist_ok=True)
        (self.cache_dir / "meta").mkdir(exist_ok=True)

        self.key_builder = CacheKeyBuilder()

    async def _get_redis(self) -> Optional[aioredis.Redis]:
        """Get Redis client if available and configured."""
        if not REDIS_AVAILABLE or not self.redis_url:
            return None

        if self._redis_client is None:
            try:
                self._redis_client = aioredis.from_url(self.redis_url)
                # Test connection
                await self._redis_client.ping()
                logger.info("Redis cache connected", extra={"redis_url": self.redis_url})
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}", extra={"redis_url": self.redis_url})
                self._redis_client = None

        return self._redis_client

    def _get_ttl(self, source: str) -> int:
        """Get TTL for source, with fallback to default."""
        return self.ttl_config.get(source.lower(), self.ttl_config["default"])

    def _get_file_paths(self, key: str) -> tuple[Path, Path]:
        """Get file paths for data and metadata."""
        data_path = self.cache_dir / "data" / f"{key}.pkl"
        meta_path = self.cache_dir / "meta" / f"{key}.json"
        return data_path, meta_path

    async def get(self, key: str) -> Optional[Any]:
        """
        Get cached value from available backends (Redis -> file -> memory).

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        start_time = time.time()

        # Try Redis first
        redis_client = await self._get_redis()
        if redis_client:
            try:
                data = await redis_client.get(key)
                if data:
                    logger.debug("Cache hit (Redis)", extra={"key": key, "fetch_ms": (time.time() - start_time) * 1000})
                    return pickle.loads(data)
            except Exception as e:
                logger.warning(f"Redis get error: {e}", extra={"key": key})

        # Try file cache
        data_path, meta_path = self._get_file_paths(key)
        if data_path.exists() and meta_path.exists():
            try:
                # Check if still valid
                with open(meta_path, 'r') as f:
                    meta = json.load(f)

                cache_time = datetime.fromisoformat(meta["timestamp"])
                ttl = meta.get("ttl", self.ttl_config["default"])

                if (datetime.utcnow() - cache_time).total_seconds() < ttl:
                    with open(data_path, 'rb') as f:
                        data = pickle.load(f)
                    logger.debug("Cache hit (file)", extra={"key": key, "fetch_ms": (time.time() - start_time) * 1000})
                    return data
                else:
                    # Expired, clean up
                    data_path.unlink(missing_ok=True)
                    meta_path.unlink(missing_ok=True)

            except Exception as e:
                logger.warning(f"File cache read error: {e}", extra={"key": key})

        # Try memory cache
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            cache_time = datetime.fromisoformat(entry["timestamp"])
            ttl = entry.get("ttl", self.ttl_config["default"])

            if (datetime.utcnow() - cache_time).total_seconds() < ttl:
                logger.debug("Cache hit (memory)", extra={"key": key, "fetch_ms": (time.time() - start_time) * 1000})
                return entry["data"]
            else:
                # Expired
                del self._memory_cache[key]

        logger.debug("Cache miss", extra={"key": key, "fetch_ms": (time.time() - start_time) * 1000})
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set cached value in all available backends.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None = use default for source)

        Returns:
            True if successfully cached
        """
        if ttl is None:
            # Extract source from key to get appropriate TTL
            source = key.split("::")[1].lower() if "::" in key else "default"
            ttl = self._get_ttl(source)

        timestamp = datetime.utcnow().isoformat()

        # Serialize once
        try:
            serialized_data = pickle.dumps(value)
        except Exception as e:
            logger.error(f"Cache serialization error: {e}", extra={"key": key})
            return False

        success_count = 0

        # Store in Redis
        redis_client = await self._get_redis()
        if redis_client:
            try:
                await redis_client.setex(key, ttl, serialized_data)
                success_count += 1
                logger.debug("Cache set (Redis)", extra={"key": key, "ttl": ttl})
            except Exception as e:
                logger.warning(f"Redis set error: {e}", extra={"key": key})

        # Store in file cache
        try:
            data_path, meta_path = self._get_file_paths(key)

            with open(data_path, 'wb') as f:
                f.write(serialized_data)

            meta = {
                "timestamp": timestamp,
                "ttl": ttl,
                "size_bytes": len(serialized_data),
                "key": key
            }

            with open(meta_path, 'w') as f:
                json.dump(meta, f)

            success_count += 1
            logger.debug("Cache set (file)", extra={"key": key, "ttl": ttl})

        except Exception as e:
            logger.warning(f"File cache set error: {e}", extra={"key": key})

        # Store in memory cache
        try:
            self._memory_cache[key] = {
                "data": value,
                "timestamp": timestamp,
                "ttl": ttl
            }
            success_count += 1
            logger.debug("Cache set (memory)", extra={"key": key, "ttl": ttl})
        except Exception as e:
            logger.warning(f"Memory cache set error: {e}", extra={"key": key})

        return success_count > 0

    async def delete(self, key: str) -> bool:
        """Delete key from all cache backends."""
        success_count = 0

        # Delete from Redis
        redis_client = await self._get_redis()
        if redis_client:
            try:
                deleted = await redis_client.delete(key)
                if deleted:
                    success_count += 1
            except Exception as e:
                logger.warning(f"Redis delete error: {e}", extra={"key": key})

        # Delete from file cache
        try:
            data_path, meta_path = self._get_file_paths(key)
            if data_path.exists():
                data_path.unlink()
                success_count += 1
            if meta_path.exists():
                meta_path.unlink()
        except Exception as e:
            logger.warning(f"File cache delete error: {e}", extra={"key": key})

        # Delete from memory
        if key in self._memory_cache:
            del self._memory_cache[key]
            success_count += 1

        return success_count > 0

    async def with_cache(
            self,
            source: str,
            method: str,
            fetch_coro: Callable[[], Any],
            ttl: Optional[int] = None,
            **params
    ) -> Any:
        """
        Cache-aware async wrapper for data fetching.

        Args:
            source: Data source identifier
            method: Method name
            fetch_coro: Async callable that fetches data
            ttl: Custom TTL override
            **params: Method parameters for key building

        Returns:
            Cached or freshly fetched data
        """
        # Build stable cache key
        key = self.key_builder.build_key(source, method, **params)

        # Try cache first
        cached_data = await self.get(key)
        if cached_data is not None:
            return cached_data

        # Fetch fresh data
        start_time = time.time()
        try:
            fresh_data = await fetch_coro()
            fetch_time = (time.time() - start_time) * 1000

            # Cache the result
            await self.set(key, fresh_data, ttl)

            logger.info(
                f"Fresh data cached for {source}.{method}",
                extra={
                    "source": source,
                    "method": method,
                    "fetch_ms": fetch_time,
                    "key": key[:16] + "..." if len(key) > 16 else key
                }
            )

            return fresh_data

        except Exception as e:
            logger.error(
                f"Cache fetch error for {source}.{method}: {e}",
                extra={"source": source, "method": method, "params": params}
            )
            raise

    async def clear_expired(self) -> int:
        """Clear expired entries from file cache. Returns count cleared."""
        cleared_count = 0

        # Clear file cache expired entries
        meta_dir = self.cache_dir / "meta"
        if meta_dir.exists():
            for meta_file in meta_dir.glob("*.json"):
                try:
                    with open(meta_file, 'r') as f:
                        meta = json.load(f)

                    cache_time = datetime.fromisoformat(meta["timestamp"])
                    ttl = meta.get("ttl", self.ttl_config["default"])

                    if (datetime.utcnow() - cache_time).total_seconds() >= ttl:
                        # Expired, delete both files
                        key = meta.get("key", "")
                        if await self.delete(key):
                            cleared_count += 1

                except Exception as e:
                    logger.warning(f"Error clearing expired cache {meta_file}: {e}")

        # Clear memory cache expired entries
        current_time = datetime.utcnow()
        expired_keys = []

        for key, entry in self._memory_cache.items():
            try:
                cache_time = datetime.fromisoformat(entry["timestamp"])
                ttl = entry.get("ttl", self.ttl_config["default"])

                if (current_time - cache_time).total_seconds() >= ttl:
                    expired_keys.append(key)
            except Exception:
                expired_keys.append(key)  # Remove malformed entries

        for key in expired_keys:
            del self._memory_cache[key]
            cleared_count += 1

        if cleared_count > 0:
            logger.info(f"Cleared {cleared_count} expired cache entries")

        return cleared_count

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            "backends": {
                "redis": {"available": False, "connected": False},
                "file": {"available": True, "entries": 0, "size_bytes": 0},
                "memory": {"available": True, "entries": len(self._memory_cache)}
            },
            "ttl_config": self.ttl_config,
            "namespace": CACHE_NAMESPACE
        }

        # Redis stats
        redis_client = await self._get_redis()
        if redis_client:
            stats["backends"]["redis"]["available"] = True
            try:
                await redis_client.ping()
                stats["backends"]["redis"]["connected"] = True
            except Exception:
                stats["backends"]["redis"]["connected"] = False

        # File cache stats
        try:
            data_dir = self.cache_dir / "data"
            if data_dir.exists():
                files = list(data_dir.glob("*.pkl"))
                stats["backends"]["file"]["entries"] = len(files)
                stats["backends"]["file"]["size_bytes"] = sum(f.stat().st_size for f in files)
        except Exception as e:
            logger.warning(f"Error getting file cache stats: {e}")

        return stats

    async def close(self):
        """Close connections and cleanup."""
        if self._redis_client:
            try:
                await self._redis_client.close()
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")
            finally:
                self._redis_client = None


# ========================
# Global cache instance
# ========================

_cache_manager: Optional[ProductionCacheManager] = None


async def get_cache_manager() -> ProductionCacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = ProductionCacheManager()
        logger.info("Cache manager initialized", extra=await _cache_manager.get_stats())
    return _cache_manager


# ========================
# Convenience functions
# ========================

async def cached_fetch(
        source: str,
        method: str,
        fetch_coro: Callable[[], Any],
        ttl: Optional[int] = None,
        **params
) -> Any:
    """
    Convenience function for cache-aware data fetching.

    Args:
        source: Data source identifier
        method: Method name
        fetch_coro: Async callable that fetches data
        ttl: Custom TTL override
        **params: Method parameters for key building

    Returns:
        Cached or freshly fetched data
    """
    cache_manager = await get_cache_manager()
    return await cache_manager.with_cache(source, method, fetch_coro, ttl, **params)


async def cache_get(key: str) -> Optional[Any]:
    """Get value from cache."""
    cache_manager = await get_cache_manager()
    return await cache_manager.get(key)


async def cache_set(key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """Set value in cache."""
    cache_manager = await get_cache_manager()
    return await cache_manager.set(key, value, ttl)


async def cache_delete(key: str) -> bool:
    """Delete key from cache."""
    cache_manager = await get_cache_manager()
    return await cache_manager.delete(key)


async def cache_clear_expired() -> int:
    """Clear all expired cache entries."""
    cache_manager = await get_cache_manager()
    return await cache_manager.clear_expired()


async def cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    cache_manager = await get_cache_manager()
    return await cache_manager.get_stats()


async def cleanup_cache():
    """Cleanup cache connections."""
    global _cache_manager
    if _cache_manager:
        await _cache_manager.close()
        _cache_manager = None


# ========================
# Legacy compatibility (keeping existing interface)
# ========================

class CacheManager:
    """Legacy cache manager for backward compatibility."""

    def __init__(self, cache_dir: str = "data_demo/cache", default_ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.default_ttl = default_ttl
        logger.warning("Using legacy CacheManager. Consider migrating to ProductionCacheManager.")

    def get_json(self, cache_key: str, ttl: Optional[int] = None) -> Optional[Dict]:
        """Legacy JSON cache getter."""
        # This is a synchronous wrapper - consider migration to async
        return None  # Simplified for Phase 1

    def set_json(self, cache_key: str, data: Dict, ttl: Optional[int] = None) -> bool:
        """Legacy JSON cache setter."""
        return False  # Simplified for Phase 1


# Global legacy instance for backward compatibility
cache_manager = CacheManager()


# Legacy convenience functions
def cache_json(key: str, data: Dict = None, ttl: int = None) -> Optional[Dict]:
    """Legacy JSON cache function."""
    if data is not None:
        cache_manager.set_json(key, data, ttl)
        return data
    else:
        return cache_manager.get_json(key, ttl)


def cache_dataframe(key: str, df: pd.DataFrame = None, ttl: int = None) -> Optional[pd.DataFrame]:
    """Legacy DataFrame cache function."""
    return None  # Simplified for Phase 1


def cache_object(key: str, obj: Any = None, ttl: int = None) -> Optional[Any]:
    """Legacy object cache function."""
    return None  # Simplified for Phase 1


def clear_cache(key: str = None) -> Union[bool, int]:
    """Legacy cache clear function."""
    return 0  # Simplified for Phase 1


def get_cache_info() -> Dict[str, Any]:
    """Legacy cache info function."""
    return {"legacy": True}  # Simplified for Phase 1