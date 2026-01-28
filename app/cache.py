"""
Smart caching system for price comparator.
Provides per-platform caching with TTL and stale-while-revalidate support.
"""
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
import threading


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    data: List[Dict]
    timestamp: float
    ttl: float
    hits: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has exceeded its TTL."""
        return time.time() - self.timestamp > self.ttl
    
    @property
    def is_stale(self) -> bool:
        """Check if entry is stale (past 80% of TTL) but not expired."""
        age = time.time() - self.timestamp
        return age > (self.ttl * 0.8) and age <= self.ttl
    
    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.timestamp


class CacheManager:
    """
    LRU cache manager with per-platform TTL support.
    
    Features:
    - Per-platform cache entries
    - Different TTLs for quick-commerce vs e-commerce
    - LRU eviction when max entries reached
    - Thread-safe operations
    - Cache statistics tracking
    """
    
    # TTL settings (in seconds)
    QUICK_COMMERCE_TTL = 300  # 5 minutes for quick commerce (prices change more often)
    ECOMMERCE_TTL = 900       # 15 minutes for e-commerce
    
    # Platform categorization
    QUICK_COMMERCE_PLATFORMS = {"Amazon Fresh", "Flipkart Minutes", "JioMart Quick", "BigBasket", "Zepto", "Instamart", "Blinkit"}
    ECOMMERCE_PLATFORMS = {"Amazon", "Flipkart", "JioMart"}
    
    def __init__(self, max_entries: int = 1000):
        """Initialize cache manager."""
        self.max_entries = max_entries
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "stale_hits": 0,
            "evictions": 0,
        }
    
    def _make_key(self, platform: str, query: str, pincode: str) -> str:
        """Generate cache key from platform, query, and pincode."""
        # Normalize query (lowercase, strip whitespace)
        normalized_query = query.lower().strip()
        key_string = f"{platform}:{normalized_query}:{pincode}"
        # Use hash for consistent key length
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_ttl(self, platform: str) -> float:
        """Get appropriate TTL for platform."""
        if platform in self.QUICK_COMMERCE_PLATFORMS:
            return self.QUICK_COMMERCE_TTL
        return self.ECOMMERCE_TTL
    
    def _evict_if_needed(self):
        """Evict oldest entries if cache is full."""
        while len(self._cache) >= self.max_entries:
            # Remove oldest (first) item
            self._cache.popitem(last=False)
            self._stats["evictions"] += 1
    
    def get(self, platform: str, query: str, pincode: str) -> Tuple[Optional[List[Dict]], bool]:
        """
        Get cached results for a platform/query/pincode combination.
        
        Returns:
            Tuple of (results, is_stale)
            - results: List of product dicts or None if not found/expired
            - is_stale: True if results are stale (should revalidate in background)
        """
        key = self._make_key(platform, query, pincode)
        
        with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None, False
            
            entry = self._cache[key]
            
            # Check if expired
            if entry.is_expired:
                del self._cache[key]
                self._stats["misses"] += 1
                return None, False
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.hits += 1
            
            is_stale = entry.is_stale
            if is_stale:
                self._stats["stale_hits"] += 1
            else:
                self._stats["hits"] += 1
            
            return entry.data, is_stale
    
    def set(self, platform: str, query: str, pincode: str, results: List[Dict]):
        """Cache results for a platform/query/pincode combination."""
        key = self._make_key(platform, query, pincode)
        ttl = self._get_ttl(platform)
        
        with self._lock:
            self._evict_if_needed()
            
            self._cache[key] = CacheEntry(
                data=results,
                timestamp=time.time(),
                ttl=ttl
            )
            # Move to end (most recently used)
            self._cache.move_to_end(key)
    
    def invalidate(self, platform: str, query: str, pincode: str):
        """Invalidate a specific cache entry."""
        key = self._make_key(platform, query, pincode)
        
        with self._lock:
            if key in self._cache:
                del self._cache[key]
    
    def invalidate_platform(self, platform: str):
        """Invalidate all entries for a specific platform."""
        with self._lock:
            keys_to_delete = [
                key for key, entry in self._cache.items()
                # We can't easily reverse the hash, so we'll need to store platform in entry
            ]
            # Note: This is a simplified version - in production, 
            # you'd want to store platform in the entry for efficient invalidation
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats = {
                "hits": 0,
                "misses": 0,
                "stale_hits": 0,
                "evictions": 0,
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"] + self._stats["stale_hits"]
            hit_rate = (self._stats["hits"] + self._stats["stale_hits"]) / total_requests if total_requests > 0 else 0
            
            # Calculate cache size and entry ages
            entries = []
            current_time = time.time()
            for key, entry in self._cache.items():
                entries.append({
                    "age_seconds": round(current_time - entry.timestamp, 1),
                    "hits": entry.hits,
                    "is_stale": entry.is_stale,
                    "ttl_remaining": round(entry.ttl - (current_time - entry.timestamp), 1)
                })
            
            return {
                "entries": len(self._cache),
                "max_entries": self.max_entries,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "stale_hits": self._stats["stale_hits"],
                "evictions": self._stats["evictions"],
                "hit_rate": round(hit_rate * 100, 1),
                "quick_commerce_ttl": self.QUICK_COMMERCE_TTL,
                "ecommerce_ttl": self.ECOMMERCE_TTL,
            }


# Global cache instance
cache = CacheManager(max_entries=500)

