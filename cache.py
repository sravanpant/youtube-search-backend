# backend/cache.py
import hashlib
import json
import pickle
from typing import Any, Callable, Optional, TypeVar

from fastapi import FastAPI, Request
from redis import Redis, ConnectionPool
from models import VideoDetails
from pydantic import BaseModel

T = TypeVar("T")

class RedisCache:
    def __init__(self, redis_url: str, expire_time: int = 3600):
        """
        Initialize Redis cache
        
        Args:
            redis_url: Redis connection URL (e.g., redis://localhost:6379)
            expire_time: Default cache expiration time in seconds
        """
        self.pool = ConnectionPool.from_url(redis_url)
        self.redis = Redis(connection_pool=self.pool)
        self.default_expire = expire_time
        
    def _get_cache_key(self, func_name: str, args: Any = None, kwargs: Any = None) -> str:
        """Generate a unique cache key based on function name and parameters"""
        key_parts = [func_name]
        
        if args:
            key_parts.append(str(args))
        if kwargs:
            # Sort kwargs to ensure consistent key generation
            sorted_kwargs = sorted(kwargs.items())
            key_parts.append(str(sorted_kwargs))
            
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def cached(self, expire: Optional[int] = None):
        """Decorator to cache function results in Redis"""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            async def wrapper(*args, **kwargs):
                # Extract the request from kwargs if it exists
                request = kwargs.get('request', None)
                if isinstance(request, Request):
                    # Don't include the request in the cache key
                    kwargs_for_key = {k: v for k, v in kwargs.items() if k != 'request'}
                else:
                    kwargs_for_key = kwargs
                    
                # Generate cache key
                cache_key = self._get_cache_key(func.__name__, args, kwargs_for_key)
                
                # Check if result is in cache
                cached_result = self.redis.get(cache_key)
                if cached_result:
                    try:
                        return pickle.loads(cached_result)
                    except Exception as e:
                        print(f"Error deserializing cached result: {str(e)}")
                
                # If not in cache, execute the function
                result = await func(*args, **kwargs)
                
                # Cache the result
                try:
                    self.redis.set(
                        cache_key, 
                        pickle.dumps(result), 
                        ex=expire or self.default_expire
                    )
                except Exception as e:
                    print(f"Error caching result: {str(e)}")
                    
                return result
            return wrapper
        return decorator
    
    def invalidate_prefix(self, prefix: str):
        """Invalidate all cache keys with a given prefix"""
        cursor = 0
        while True:
            cursor, keys = self.redis.scan(cursor, match=f"{prefix}*", count=100)
            if keys:
                self.redis.delete(*keys)
            if cursor == 0:
                break
    
    def get_video_cache_key(self, video_id: str) -> str:
        """Generate a cache key for a specific video"""
        return f"video:{video_id}"
    
    async def get_video(self, video_id: str) -> Optional[VideoDetails]:
        """Get video details from cache"""
        cache_key = self.get_video_cache_key(video_id)
        cached_data = self.redis.get(cache_key)
        if cached_data:
            try:
                video_dict = json.loads(cached_data)
                return VideoDetails(**video_dict)
            except Exception as e:
                print(f"Error deserializing video from cache: {str(e)}")
        return None
    
    def store_video(self, video: VideoDetails, expire: Optional[int] = None):
        """Store video details in cache"""
        cache_key = self.get_video_cache_key(video.videoId)
        try:
            # Convert to dict before storing in Redis
            video_json = json.dumps(video.dict())
            self.redis.set(
                cache_key, 
                video_json, 
                ex=expire or self.default_expire
            )
        except Exception as e:
            print(f"Error storing video in cache: {str(e)}")
    
    def get_search_cache_key(self, payload):
        """Generate a cache key for search requests"""
        # Sort all parameters to ensure consistent key generation
        payload_dict = payload.dict()
        sorted_items = sorted(payload_dict.items())
        key_string = "search:" + str(sorted_items)
        return hashlib.md5(key_string.encode()).hexdigest()
