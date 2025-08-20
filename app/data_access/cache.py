"""
Caching utilities for API responses and computed results.
Supports both JSON and Parquet formats with TTL (time-to-live) functionality.
"""

import json
import pickle
import hashlib
import time
from pathlib import Path
from typing import Any, Optional, Dict, Union
import pandas as pd

class CacheManager:
    def __init__(self, cache_dir: str = "data_demo/cache", default_ttl: int = 3600):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory to store cache files
            default_ttl: Default time-to-live in seconds (1 hour default)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl

        # Create subdirectories for different cache types
        (self.cache_dir / "json").mkdir(exist_ok=True)
        (self.cache_dir / "parquet").mkdir(exist_ok=True)
        (self.cache_dir / "pickle").mkdir(exist_ok=True)
        (self.cache_dir / "metadata").mkdir(exist_ok=True)

    def _generate_cache_key(self, key: str) -> str:
        """Generate a hashed cache key from input string."""
        return hashlib.md5(key.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str, cache_type: str = "json") -> Path:
        """Get the full path for a cache file."""
        hashed_key = self._generate_cache_key(cache_key)
        extension = {
            "json": ".json",
            "parquet": ".parquet",
            "pickle": ".pkl"
        }.get(cache_type, ".json")

        return self.cache_dir / cache_type / f"{hashed_key}{extension}"

    def _get_metadata_path(self, cache_key: str) -> Path:
        """Get path for cache metadata file."""
        hashed_key = self._generate_cache_key(cache_key)
        return self.cache_dir / "metadata" / f"{hashed_key}.meta"

    def _is_cache_valid(self, cache_key: str, ttl: Optional[int] = None) -> bool:
        """Check if cached data is still valid (not expired)."""
        metadata_path = self._get_metadata_path(cache_key)

        if not metadata_path.exists():
            return False

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            cache_time = metadata.get("timestamp", 0)
            used_ttl = ttl if ttl is not None else metadata.get("ttl", self.default_ttl)

            return (time.time() - cache_time) < used_ttl

        except Exception:
            return False

    def _save_metadata(self, cache_key: str, ttl: int, data_type: str, size: int):
        """Save cache metadata."""
        metadata_path = self._get_metadata_path(cache_key)

        metadata = {
            "timestamp": time.time(),
            "ttl": ttl,
            "data_type": data_type,
            "size_bytes": size,
            "cache_key": cache_key
        }

        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
        except Exception as e:
            print(f"Warning: Could not save cache metadata: {e}")

    def get_json(self, cache_key: str, ttl: Optional[int] = None) -> Optional[Dict]:
        """Retrieve JSON data from cache."""
        if not self._is_cache_valid(cache_key, ttl):
            return None

        cache_path = self._get_cache_path(cache_key, "json")

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading JSON cache: {e}")
            return None

    def set_json(self, cache_key: str, data: Dict, ttl: Optional[int] = None) -> bool:
        """Store JSON data in cache."""
        cache_path = self._get_cache_path(cache_key, "json")
        used_ttl = ttl if ttl is not None else self.default_ttl

        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)

            # Save metadata
            size = cache_path.stat().st_size
            self._save_metadata(cache_key, used_ttl, "json", size)

            return True

        except Exception as e:
            print(f"Error writing JSON cache: {e}")
            return False

    def get_parquet(self, cache_key: str, ttl: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Retrieve DataFrame from parquet cache."""
        if not self._is_cache_valid(cache_key, ttl):
            return None

        cache_path = self._get_cache_path(cache_key, "parquet")

        if not cache_path.exists():
            return None

        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Error reading Parquet cache: {e}")
            return None

    def set_parquet(self, cache_key: str, df: pd.DataFrame, ttl: Optional[int] = None) -> bool:
        """Store DataFrame in parquet cache."""
        cache_path = self._get_cache_path(cache_key, "parquet")
        used_ttl = ttl if ttl is not None else self.default_ttl

        try:
            df.to_parquet(cache_path, index=False)

            # Save metadata
            size = cache_path.stat().st_size
            self._save_metadata(cache_key, used_ttl, "parquet", size)

            return True

        except Exception as e:
            print(f"Error writing Parquet cache: {e}")
            return False

    def get_pickle(self, cache_key: str, ttl: Optional[int] = None) -> Optional[Any]:
        """Retrieve pickled object from cache."""
        if not self._is_cache_valid(cache_key, ttl):
            return None

        cache_path = self._get_cache_path(cache_key, "pickle")

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error reading pickle cache: {e}")
            return None

    def set_pickle(self, cache_key: str, obj: Any, ttl: Optional[int] = None) -> bool:
        """Store pickled object in cache."""
        cache_path = self._get_cache_path(cache_key, "pickle")
        used_ttl = ttl if ttl is not None else self.default_ttl

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(obj, f)

            # Save metadata
            size = cache_path.stat().st_size
            self._save_metadata(cache_key, used_ttl, "pickle", size)

            return True

        except Exception as e:
            print(f"Error writing pickle cache: {e}")
            return False

    def delete(self, cache_key: str) -> bool:
        """Delete cached data and metadata."""
        hashed_key = self._generate_cache_key(cache_key)

        deleted_any = False

        # Delete all possible cache files
        for cache_type, extension in [("json", ".json"), ("parquet", ".parquet"), ("pickle", ".pkl")]:
            cache_path = self.cache_dir / cache_type / f"{hashed_key}{extension}"
            if cache_path.exists():
                try:
                    cache_path.unlink()
                    deleted_any = True
                except Exception as e:
                    print(f"Error deleting cache file {cache_path}: {e}")

        # Delete metadata
        metadata_path = self._get_metadata_path(cache_key)
        if metadata_path.exists():
            try:
                metadata_path.unlink()
                deleted_any = True
            except Exception as e:
                print(f"Error deleting metadata file: {e}")

        return deleted_any

    def clear_expired(self) -> int:
        """Clear all expired cache entries. Returns number of entries cleared."""
        cleared_count = 0

        # Check all metadata files
        metadata_dir = self.cache_dir / "metadata"
        if metadata_dir.exists():
            for metadata_file in metadata_dir.glob("*.meta"):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)

                    cache_time = metadata.get("timestamp", 0)
                    ttl = metadata.get("ttl", self.default_ttl)

                    if (time.time() - cache_time) >= ttl:
                        # Cache is expired, delete it
                        cache_key = metadata.get("cache_key", "")
                        if self.delete(cache_key):
                            cleared_count += 1

                except Exception as e:
                    print(f"Error checking cache expiry for {metadata_file}: {e}")

        return cleared_count

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "total_entries": 0,
            "total_size_bytes": 0,
            "by_type": {},
            "expired_entries": 0
        }

        # Check all metadata files
        metadata_dir = self.cache_dir / "metadata"
        if metadata_dir.exists():
            for metadata_file in metadata_dir.glob("*.meta"):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)

                    data_type = metadata.get("data_type", "unknown")
                    size = metadata.get("size_bytes", 0)
                    cache_time = metadata.get("timestamp", 0)
                    ttl = metadata.get("ttl", self.default_ttl)

                    stats["total_entries"] += 1
                    stats["total_size_bytes"] += size

                    if data_type not in stats["by_type"]:
                        stats["by_type"][data_type] = {"count": 0, "size_bytes": 0}

                    stats["by_type"][data_type]["count"] += 1
                    stats["by_type"][data_type]["size_bytes"] += size

                    # Check if expired
                    if (time.time() - cache_time) >= ttl:
                        stats["expired_entries"] += 1

                except Exception as e:
                    print(f"Error reading cache stats from {metadata_file}: {e}")

        return stats

# Global cache manager instance
cache_manager = CacheManager()

# Convenience functions for common caching patterns
def cache_json(key: str, data: Dict = None, ttl: int = None) -> Optional[Dict]:
    """Get or set JSON cache data."""
    if data is not None:
        cache_manager.set_json(key, data, ttl)
        return data
    else:
        return cache_manager.get_json(key, ttl)

def cache_dataframe(key: str, df: pd.DataFrame = None, ttl: int = None) -> Optional[pd.DataFrame]:
    """Get or set DataFrame cache data."""
    if df is not None:
        cache_manager.set_parquet(key, df, ttl)
        return df
    else:
        return cache_manager.get_parquet(key, ttl)

def cache_object(key: str, obj: Any = None, ttl: int = None) -> Optional[Any]:
    """Get or set pickled object cache data."""
    if obj is not None:
        cache_manager.set_pickle(key, obj, ttl)
        return obj
    else:
        return cache_manager.get_pickle(key, ttl)

def clear_cache(key: str = None) -> Union[bool, int]:
    """Clear specific cache entry or all expired entries."""
    if key is not None:
        return cache_manager.delete(key)
    else:
        return cache_manager.clear_expired()

def get_cache_info() -> Dict[str, Any]:
    """Get cache statistics."""
    return cache_manager.get_cache_stats()