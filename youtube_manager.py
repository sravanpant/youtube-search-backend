# youtube_manager.py
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from collections import deque
import time
import asyncio
import threading


class YouTubeAPIManager:
    # Quota costs for each method
    QUOTA_COSTS = {
        "search.list": 100,
        "videos.list": 1,
        "channels.list": 1,
        "playlistItems.list": 1,
        "playlists.list": 1,
        "activities.list": 1,
        "commentThreads.list": 1,
        "comments.list": 1,
        "subscriptions.list": 1,
        # Default for any method not specified
        "default": 1,
    }

    # Daily quota limit per key
    DAILY_QUOTA_LIMIT = 10000

    # Threshold to start warming up next key
    QUOTA_WARNING_THRESHOLD = 0.9  # 90%

    # Threshold to switch to next key
    QUOTA_SWITCH_THRESHOLD = 0.95  # 95%

    def __init__(self, api_keys=None):
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

        # Two-tier key management system
        self.available_keys = deque(range(len(self.api_keys)))
        self.quota_exceeded_keys = {}  # key_index -> timestamp when quota was exceeded

        # Enhanced stats tracking
        self.key_usage_count = {i: 0 for i in range(len(self.api_keys))}
        self.key_failure_count = {i: 0 for i in range(len(self.api_keys))}
        self.key_quota_used = {
            i: 0 for i in range(len(self.api_keys))
        }  # Track quota units used
        self.key_usage_history = {
            i: [] for i in range(len(self.api_keys))
        }  # Track operations

        # Quota reset time (in seconds) - typically 24 hours for YouTube
        self.quota_reset_period = 24 * 60 * 60  # 24 hours

        # Initialize client with first available key
        self.current_key_index = self.available_keys[0] if self.available_keys else None
        self.next_key_index = None  # For preemptive warming
        self.youtube_client = None
        self.next_youtube_client = None  # Prewarmed client
        self._initialize_client()

        # Lock for thread safety during key rotation
        self.lock = threading.RLock()

        # Start key rotation monitor in background
        self.monitor_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_keys, daemon=True)
        self.monitor_thread.start()

    def _initialize_client(self, key_index=None, for_next=False):
        """Initialize YouTube client with specified key index"""

        if key_index is None:
            key_index = self.current_key_index

        if key_index is None:
            self._restore_expired_quota_keys()
            if not self.available_keys:
                raise ValueError("No valid YouTube API keys available")
            key_index = self.available_keys[0]

        # Log key initialization
        target = "NEXT" if for_next else "CURRENT"
        print(f"🔑 Initializing {target} YouTube client with API key {key_index + 1}")

        # Optional: Add a masked key log for verification
        key = self.api_keys[key_index]
        masked_key = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "****"
        print(f"   Key fingerprint: {masked_key}")

        client = build("youtube", "v3", developerKey=self.api_keys[key_index])

        if for_next:
            self.next_key_index = key_index
            self.next_youtube_client = client
            print(f"🔄 Pre-warmed client with key {key_index + 1} is ready")
        else:
            self.current_key_index = key_index
            self.youtube_client = client

        return client

    def _restore_expired_quota_keys(self):
        """Check for and restore any keys whose quota should have reset."""
        with self.lock:
            current_time = time.time()
            restored_keys = []

            for key_idx, exceeded_time in list(self.quota_exceeded_keys.items()):
                # If enough time has passed since quota was exceeded
                if current_time - exceeded_time > self.quota_reset_period:
                    self.available_keys.append(key_idx)
                    restored_keys.append(key_idx)
                    # Reset quota tracking for this key
                    self.key_quota_used[key_idx] = 0
                    self.key_usage_history[key_idx] = []

            # Remove restored keys from the exceeded list
            for key_idx in restored_keys:
                del self.quota_exceeded_keys[key_idx]

            if restored_keys:
                print(f"✅ Restored {len(restored_keys)} API keys with reset quota")

    def _monitor_keys(self):
        """Background thread to monitor key usage and prepare next key"""
        while self.monitor_active:
            try:
                if self.current_key_index is not None:
                    quota_used = self.key_quota_used.get(self.current_key_index, 0)
                    quota_percentage = quota_used / self.DAILY_QUOTA_LIMIT

                    # If we're approaching quota limit, prepare next key
                    if (
                        quota_percentage >= self.QUOTA_WARNING_THRESHOLD
                        and self.next_key_index is None
                    ):
                        print(
                            f"⚠️ Key {self.current_key_index + 1} at {quota_percentage:.1%} quota usage, preparing next key"
                        )
                        self._prepare_next_key()

                    # If we've crossed the switch threshold, switch immediately
                    if quota_percentage >= self.QUOTA_SWITCH_THRESHOLD:
                        print(
                            f"🔄 Proactively switching from key {self.current_key_index + 1} at {quota_percentage:.1%} quota"
                        )
                        self._switch_to_next_key()
            except Exception as e:
                print(f"Error in key monitor: {str(e)}")

            # Check every few seconds
            time.sleep(5)

    def _prepare_next_key(self):
        """Prepare the next key in advance"""
        with self.lock:
            # Find next available key
            available_keys = list(self.available_keys)
            if len(available_keys) <= 1:
                # Try to restore expired keys
                self._restore_expired_quota_keys()
                available_keys = list(self.available_keys)

            if len(available_keys) <= 1:
                print("⚠️ No additional keys available for prewarming")
                return

            # Find a key that isn't the current one
            for key_idx in available_keys:
                if key_idx != self.current_key_index:
                    # Initialize it as the next client
                    self._initialize_client(key_idx, for_next=True)
                    return

    def _switch_to_next_key(self):
        """Switch to the prewarmed key or find a new one"""
        with self.lock:
            # If we have a prewarmed key, use it
            if self.next_key_index is not None and self.next_youtube_client is not None:
                old_key = self.current_key_index

                # Switch clients
                self.current_key_index = self.next_key_index
                self.youtube_client = self.next_youtube_client

                # Reset next client
                self.next_key_index = None
                self.next_youtube_client = None

                print(
                    f"🔄 Switched from key {old_key + 1} to key {self.current_key_index + 1}"
                )
                return True

            # If no prewarmed key, find next available
            elif len(self.available_keys) > 1:
                # Find a different key
                old_key = self.current_key_index
                for key_idx in list(self.available_keys):
                    if key_idx != self.current_key_index:
                        self._initialize_client(key_idx)
                        print(
                            f"🔄 Switched from key {old_key + 1} to key {self.current_key_index + 1}"
                        )
                        return True

            return False

    def _mark_key_quota_exceeded(self):
        """Mark current key as having exceeded quota and select next key."""
        with self.lock:
            if self.current_key_index is not None:
                # Move from available to quota exceeded
                try:
                    self.available_keys.remove(self.current_key_index)
                except ValueError:
                    # Key might have been removed by another thread
                    pass

                self.quota_exceeded_keys[self.current_key_index] = time.time()
                self.key_failure_count[self.current_key_index] += 1
                self.key_quota_used[self.current_key_index] = (
                    self.DAILY_QUOTA_LIMIT
                )  # Mark as fully used

                print(
                    f"⛔ API key {self.current_key_index + 1} marked as quota exceeded"
                )

                # Switch to prewarmed key or find next key
                if not self._switch_to_next_key():
                    # No available keys, force check for restored keys
                    self._restore_expired_quota_keys()
                    if self.available_keys:
                        self.current_key_index = self.available_keys[0]
                        self._initialize_client()
                    else:
                        self.current_key_index = None

    def _track_operation_cost(self, operation_name, result=None):
        """Track the quota cost of an operation"""
        with self.lock:
            if self.current_key_index is None:
                return

            # Determine the operation type
            method_name = "default"
            if "." in operation_name:
                method_name = operation_name
            else:
                # Try to infer from the name
                if "search" in operation_name.lower():
                    method_name = "search.list"
                elif "video" in operation_name.lower():
                    method_name = "videos.list"
                elif "channel" in operation_name.lower():
                    method_name = "channels.list"

            # Calculate quota cost
            cost = self.QUOTA_COSTS.get(method_name, self.QUOTA_COSTS["default"])

            # Increase based on result count for paginated results
            if result and "pageInfo" in result and "totalResults" in result["pageInfo"]:
                items_per_page = result.get("pageInfo", {}).get("resultsPerPage", 0)
                if items_per_page > 0:
                    total_items = result["pageInfo"]["totalResults"]
                    # Adjust for pagination
                    estimated_pages = (
                        total_items + items_per_page - 1
                    ) // items_per_page
                    # But limit to what was actually in this response
                    actual_items = len(result.get("items", []))
                    if actual_items > 0:
                        cost = cost * (actual_items / items_per_page)

            # Record the operation and cost
            timestamp = time.time()
            self.key_usage_history[self.current_key_index].append(
                {"timestamp": timestamp, "method": method_name, "cost": cost}
            )

            # Update total quota used
            self.key_quota_used[self.current_key_index] += cost
            quota_percentage = (
                self.key_quota_used[self.current_key_index] / self.DAILY_QUOTA_LIMIT
            )

            # Log every 5% increment or if over warning threshold
            if (quota_percentage * 100) % 5 < (
                cost / self.DAILY_QUOTA_LIMIT * 100
            ) or quota_percentage >= self.QUOTA_WARNING_THRESHOLD:
                print(
                    f"📊 Key {self.current_key_index + 1} quota usage: {quota_percentage:.1%} ({self.key_quota_used[self.current_key_index]:.0f} units)"
                )

            return cost

    def get_client(self):
        """Get current YouTube client, ensuring a valid one is available."""
        with self.lock:
            if self.current_key_index is None or self.youtube_client is None:
                self._restore_expired_quota_keys()
                if self.available_keys:
                    self.current_key_index = self.available_keys[0]
                    self._initialize_client()
                else:
                    raise Exception("All API keys have exceeded their quota")

            # Track usage
            self.key_usage_count[self.current_key_index] += 1

            # Only log every 10th use to avoid excessive logging
            usage_count = self.key_usage_count[self.current_key_index]
            if usage_count % 10 == 0:
                quota_used = self.key_quota_used.get(self.current_key_index, 0)
                quota_percentage = quota_used / self.DAILY_QUOTA_LIMIT
                print(
                    f"🔑 Using API key {self.current_key_index + 1} (used {usage_count} times, {quota_percentage:.1%} quota)"
                )

            return self.youtube_client

    async def execute_with_retry(self, operation, method_name="default"):
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
        starting_key = self.current_key_index

        while attempts < initial_key_count:
            try:
                # Get a fresh client with the current best key
                client = self.get_client()

                # Execute the operation
                result = await operation(client)

                # Track the operation cost
                self._track_operation_cost(method_name, result)

                # Log key change if different from starting key
                if self.current_key_index != starting_key:
                    print(
                        f"🔄 Switched from key {starting_key + 1} to key {self.current_key_index + 1}"
                    )

                return result

            except HttpError as e:
                if e.resp.status in [403, 429] and "quota" in str(e).lower():
                    print(f"⚠️ Quota exceeded for key {self.current_key_index + 1}")
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
        stats = {
            "available_keys": len(self.available_keys),
            "available_key_indices": list(self.available_keys),
            "quota_exceeded_keys": len(self.quota_exceeded_keys),
            "quota_exceeded_indices": list(self.quota_exceeded_keys.keys()),
            "current_key": self.current_key_index,
            "next_key": self.next_key_index,
            "usage_stats": dict(self.key_usage_count),
            "failure_stats": dict(self.key_failure_count),
            "quota_usage": {},
            "quota_percentage": {},
        }

        # Calculate quota statistics
        for key_idx in range(len(self.api_keys)):
            quota_used = self.key_quota_used.get(key_idx, 0)
            percentage = (quota_used / self.DAILY_QUOTA_LIMIT) * 100
            stats["quota_usage"][key_idx] = round(quota_used, 2)
            stats["quota_percentage"][key_idx] = round(percentage, 2)

        # Add estimated remaining quota for current key
        if self.current_key_index is not None:
            remaining = max(
                0,
                self.DAILY_QUOTA_LIMIT
                - self.key_quota_used.get(self.current_key_index, 0),
            )
            stats["current_key_remaining_quota"] = round(remaining, 2)

            # Estimate how many more operations of each type can be performed
            stats["estimated_operations_remaining"] = {
                "search.list": int(remaining / self.QUOTA_COSTS["search.list"]),
                "videos.list": int(remaining / self.QUOTA_COSTS["videos.list"]),
                "channels.list": int(remaining / self.QUOTA_COSTS["channels.list"]),
            }

        return stats

    def shutdown(self):
        """Clean shutdown of the manager"""
        self.monitor_active = False
        if hasattr(self, "monitor_thread") and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
