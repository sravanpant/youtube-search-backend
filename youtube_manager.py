# backend/youtube_manager.py

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from collections import deque
import time

class YouTubeAPIManager:
    def __init__(self, api_keys=None):
       # Use provided keys or empty list
        # If no keys provided, try to get them from environment variables
        if api_keys is None:
            import os
            # Try to load environment variables for API keys
            api_keys = []
            for i in range(1, 22):  # Look for keys numbered 1-21
                key = os.getenv(f"YOUTUBE_API_KEY_{i}")
                if key:
                    api_keys.append(key)
        
        # Filter out None values (only once)
        self.api_keys = [key for key in api_keys if key]
        
        # Rest of initialization
        self.available_keys = deque(range(len(self.api_keys)))
        self.quota_exceeded_keys = {}
        self.key_usage_count = {i: 0 for i in range(len(self.api_keys))}
        self.key_failure_count = {i: 0 for i in range(len(self.api_keys))}
        self.quota_reset_period = 24 * 60 * 60  # 24 hours
        self.current_key_index = self.available_keys[0] if self.available_keys else None
        self.youtube_client = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize YouTube client with current API key."""
        if self.current_key_index is None:
            self._restore_expired_quota_keys()
            if not self.available_keys:
                raise ValueError("No valid YouTube API keys available")
            self.current_key_index = self.available_keys[0]
        
        print(f"🔑 Initializing YouTube client with API key {self.current_key_index + 1}")

        self.youtube_client = build(
            "youtube", "v3", developerKey=self.api_keys[self.current_key_index]
        )

    def _restore_expired_quota_keys(self) -> None:
        """Check for and restore any keys whose quota should have reset."""
        current_time = time.time()
        restored_keys = []

        for key_idx, exceeded_time in list(self.quota_exceeded_keys.items()):
            # If enough time has passed since quota was exceeded
            if current_time - exceeded_time > self.quota_reset_period:
                self.available_keys.append(key_idx)
                restored_keys.append(key_idx)

        # Remove restored keys from the exceeded list
        for key_idx in restored_keys:
            del self.quota_exceeded_keys[key_idx]

        if restored_keys:
            print(f"Restored {len(restored_keys)} API keys with reset quota")

    def _mark_key_quota_exceeded(self) -> None:
        """Mark current key as having exceeded quota and select next key."""
        if self.current_key_index is not None:
            # Move from available to quota exceeded
            self.available_keys.remove(self.current_key_index)
            self.quota_exceeded_keys[self.current_key_index] = time.time()
            self.key_failure_count[self.current_key_index] += 1

            print(f"API key {self.current_key_index + 1} marked as quota exceeded")

            # Select next key
            if self.available_keys:
                self.current_key_index = self.available_keys[0]
                self._initialize_client()
            else:
                # No available keys, force check for restored keys
                self._restore_expired_quota_keys()
                if self.available_keys:
                    self.current_key_index = self.available_keys[0]
                    self._initialize_client()
                else:
                    self.current_key_index = None

    def get_client(self):
        """Get current YouTube client, ensuring a valid one is available."""
        if self.current_key_index is None or self.youtube_client is None:
            self._restore_expired_quota_keys()
            if self.available_keys:
                self.current_key_index = self.available_keys[0]
                self._initialize_client()
            else:
                raise Exception("All API keys have exceeded their quota")

        # Track usage
        self.key_usage_count[self.current_key_index] += 1
        return self.youtube_client

    async def execute_with_retry(self, operation):
        """Execute YouTube API operation with efficient key rotation on quota exceeded."""
        # First, make sure we have an available key
        if not self.available_keys:
            self._restore_expired_quota_keys()
            if not self.available_keys:
                raise Exception(
                    "All API keys have exceeded their quota. Please try again later."
                )

        # Try all possible keys
        initial_key_count = len(self.available_keys)
        attempts = 0

        while attempts < initial_key_count:
            try:
                # Get a fresh client with the current best key
                client = self.get_client()
                return await operation(client)
            except HttpError as e:
                if e.resp.status in [403, 429] and "quota" in str(e).lower():
                    print(f"Quota exceeded for key {self.current_key_index + 1}")
                    self._mark_key_quota_exceeded()
                    attempts += 1

                    # If we've exhausted all keys, raise exception
                    if not self.available_keys:
                        raise Exception("All API keys have exceeded their quota")
                else:
                    # Other API error, not quota-related
                    raise e
            except Exception as e:
                raise e

        raise Exception("All available API keys failed")

    def get_key_stats(self):
        """Return statistics about key usage for monitoring."""
        return {
            "available_keys": len(self.available_keys),
            "quota_exceeded_keys": len(self.quota_exceeded_keys),
            "current_key": self.current_key_index,
            "usage_stats": self.key_usage_count,
            "failure_stats": self.key_failure_count,
        }
