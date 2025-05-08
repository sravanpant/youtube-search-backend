# backend/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from googleapiclient.discovery import build
from googleapiclient.discovery import build as google_build
from googleapiclient.errors import HttpError
import os
from dotenv import load_dotenv
import asyncio
from cache import RedisCache
from typing import List
from models import SearchRequest, VideoDetails, DateFilterOption
from utils import (
    extract_video_id_from_url,
    calculate_relevancy_score,
    extract_brand_related_urls,
)
from typing import Optional
from datetime import datetime, timedelta
import time
import json
import hashlib
from redis import Redis

load_dotenv()

# Load environment variables
YOUTUBE_API_KEY_1 = os.getenv("YOUTUBE_API_KEY_1")
YOUTUBE_API_KEY_2 = os.getenv("YOUTUBE_API_KEY_2")
YOUTUBE_API_KEY_3 = os.getenv("YOUTUBE_API_KEY_3")
YOUTUBE_API_KEY_4 = os.getenv("YOUTUBE_API_KEY_4")
YOUTUBE_API_KEY_5 = os.getenv("YOUTUBE_API_KEY_5")
YOUTUBE_API_KEY_6 = os.getenv("YOUTUBE_API_KEY_6")
YOUTUBE_API_KEY_7 = os.getenv("YOUTUBE_API_KEY_7")
YOUTUBE_API_KEY_8 = os.getenv("YOUTUBE_API_KEY_8")
YOUTUBE_API_KEY_9 = os.getenv("YOUTUBE_API_KEY_9")
YOUTUBE_API_KEY_10 = os.getenv("YOUTUBE_API_KEY_10")
YOUTUBE_API_KEY_11 = os.getenv("YOUTUBE_API_KEY_11")
YOUTUBE_API_KEY_12 = os.getenv("YOUTUBE_API_KEY_12")
YOUTUBE_API_KEY_13 = os.getenv("YOUTUBE_API_KEY_13")
YOUTUBE_API_KEY_14 = os.getenv("YOUTUBE_API_KEY_14")
YOUTUBE_API_KEY_15 = os.getenv("YOUTUBE_API_KEY_15")
YOUTUBE_API_KEY_16 = os.getenv("YOUTUBE_API_KEY_16")
YOUTUBE_API_KEY_17 = os.getenv("YOUTUBE_API_KEY_17")
YOUTUBE_API_KEY_18 = os.getenv("YOUTUBE_API_KEY_18")
YOUTUBE_API_KEY_19 = os.getenv("YOUTUBE_API_KEY_19")
YOUTUBE_API_KEY_20 = os.getenv("YOUTUBE_API_KEY_20")
YOUTUBE_API_KEY_21 = os.getenv("YOUTUBE_API_KEY_21")
GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
cache = RedisCache(REDIS_URL)

redis_client = Redis(connection_pool=cache.pool)


class YouTubeAPIManager:
    def __init__(self):
        self.api_keys = [
            YOUTUBE_API_KEY_1,
            YOUTUBE_API_KEY_2,
            YOUTUBE_API_KEY_3,
            YOUTUBE_API_KEY_4,
            YOUTUBE_API_KEY_5,
            YOUTUBE_API_KEY_6,
            YOUTUBE_API_KEY_7,
            YOUTUBE_API_KEY_8,
            YOUTUBE_API_KEY_9,
            YOUTUBE_API_KEY_10,
            YOUTUBE_API_KEY_11,
            YOUTUBE_API_KEY_12,
            YOUTUBE_API_KEY_13,
            YOUTUBE_API_KEY_14,
            YOUTUBE_API_KEY_15,
            YOUTUBE_API_KEY_16,
            YOUTUBE_API_KEY_17,
            YOUTUBE_API_KEY_18,
            YOUTUBE_API_KEY_19,
            YOUTUBE_API_KEY_20,
            YOUTUBE_API_KEY_21,
        ]
        # Filter out None values
        self.api_keys = [key for key in self.api_keys if key]
        self.current_key_index = 0
        self.youtube_client = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize YouTube client with current API key."""
        if self.api_keys:
            self.youtube_client = build(
                "youtube", "v3", developerKey=self.api_keys[self.current_key_index]
            )
        else:
            raise ValueError("No valid YouTube API keys available")

    def switch_to_next_key(self) -> bool:
        """Switch to next available API key. Returns True if successful."""
        self.current_key_index += 1
        if self.current_key_index >= len(self.api_keys):
            return False
        self._initialize_client()
        return True

    def get_client(self):
        """Get current YouTube client."""
        return self.youtube_client

    async def execute_with_retry(self, operation):
        """Execute YouTube API operation with automatic key switching on quota exceeded."""
        max_retries = len(self.api_keys)
        retries = 0

        while retries < max_retries:
            try:
                return await operation(self.youtube_client)
            except HttpError as e:
                if e.resp.status in [403, 429] and "quota" in str(e).lower():
                    print(
                        f"Quota exceeded for key {self.current_key_index + 1}, switching to next key..."
                    )
                    if not self.switch_to_next_key():
                        raise Exception("All API keys have exceeded their quota")
                    retries += 1
                else:
                    raise e
            except Exception as e:
                raise e

        raise Exception("All API keys have been exhausted")


app = FastAPI()

# Create a global instance of the YouTube API manager
youtube_manager = YouTubeAPIManager()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://youtube-search-frontend.vercel.app",
    ],  # Adjust this to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def process_video(
    video_id: str,
    keywords: List[str],
    all_videos: List[VideoDetails],
    min_views: int,
    publish_after: Optional[datetime] = None,
    publish_before: Optional[datetime] = None,
    country_code: Optional[str] = None,
) -> None:
    """
    Process a single video and add it to all_videos if relevant.
    """
    try:
        # First check if video is in cache
        cached_video = await cache.get_video(video_id)
        if cached_video:
            # If video is in cache and meets criteria, add it to results
            if (
                cached_video.viewCount >= min_views
                and cached_video.relevancy_score >= 3
                and (not country_code or cached_video.country == country_code)
            ):

                # Check date filters if applicable
                if publish_after or publish_before:
                    try:
                        from datetime import timezone

                        publish_time = datetime.fromisoformat(
                            cached_video.publishTime.replace("Z", "+00:00")
                        )

                        if publish_after and publish_after.tzinfo is None:
                            publish_after = publish_after.replace(tzinfo=timezone.utc)
                        if publish_before and publish_before.tzinfo is None:
                            publish_before = publish_before.replace(tzinfo=timezone.utc)

                        if (publish_after and publish_time < publish_after) or (
                            publish_before and publish_time > publish_before
                        ):
                            return
                    except Exception as e:
                        print(f"Error parsing cached publish date: {str(e)}")

                all_videos.append(cached_video)
                return

        # If video is not in cache or doesn't meet criteria, fetch from YouTube API
        # Get video details using the manager
        async def get_video_details(youtube):
            return (
                youtube.videos()
                .list(
                    part="statistics,contentDetails,snippet,recordingDetails",
                    id=video_id,
                )
                .execute()
            )

        video_response = await youtube_manager.execute_with_retry(get_video_details)

        if not video_response.get("items"):
            return

        video_data = video_response["items"][0]

        # Check video publish date against filters
        if publish_after or publish_before:
            try:
                publish_time_str = video_data["snippet"].get("publishedAt")
                if publish_time_str:
                    # Ensure consistent timezone handling - make both dates timezone-aware
                    from datetime import timezone

                    # Parse the video's publish time
                    publish_time = datetime.fromisoformat(
                        publish_time_str.replace("Z", "+00:00")
                    )

                    # Make sure publish_after and publish_before are timezone-aware
                    if publish_after and publish_after.tzinfo is None:
                        publish_after = publish_after.replace(tzinfo=timezone.utc)

                    if publish_before and publish_before.tzinfo is None:
                        publish_before = publish_before.replace(tzinfo=timezone.utc)

                    # Now compare the dates (all are timezone-aware)
                    if publish_after and publish_time < publish_after:
                        return

                    if publish_before and publish_time > publish_before:
                        return
            except Exception as e:
                print(f"Error parsing publish date for video {video_id}: {str(e)}")
                # Continue processing this video even if date comparison fails

        # Extract country information
        video_country = None
        recording_details = video_data.get("recordingDetails", {})
        if recording_details and recording_details.get("locationDescription"):
            # Try to extract country from location description
            location_desc = recording_details.get("locationDescription", "")
            video_country = (
                location_desc.split(",")[-1].strip()
                if "," in location_desc
                else location_desc
            )

        # Also check snippet country (video uploaded from)
        snippet_country = video_data.get("snippet", {}).get("country")
        if snippet_country:
            video_country = snippet_country

        # Check if country code filter is applied and matches
        if (
            country_code
            and video_country
            and video_country.upper() != country_code.upper()
        ):
            # Country doesn't match the filter, skip this video
            return

        # Calculate relevancy score
        relevancy_score = calculate_relevancy_score(video_data, keywords)

        if relevancy_score < 3:  # Minimum relevancy threshold
            return

        # Check minimum views
        view_count = int(video_data["statistics"].get("viewCount", 0))
        if view_count < min_views:
            return

        # Rest of the processing
        channel_id = video_data["snippet"]["channelId"]
        channel_title = video_data["snippet"]["channelTitle"].lower()

        # Skip if channel name contains any of the keywords
        if any(kw in channel_title for kw in keywords):
            return

        try:

            async def get_channel_details(youtube):
                return (
                    youtube.channels().list(part="statistics", id=channel_id).execute()
                )

            channel_response = await youtube_manager.execute_with_retry(
                get_channel_details
            )
            channel_stats = channel_response.get("items", [{}])[0].get("statistics", {})
        except:
            channel_stats = {}

        # Extract brand links
        description = video_data["snippet"].get("description", "")
        brand_links = extract_brand_related_urls(description, keywords)

        # Parse duration
        try:
            duration = video_data["contentDetails"][
                "duration"
            ]  # Keep the ISO 8601 format
        except:
            duration = "PT0S"  # Default duration if not available

        # Create video details
        video_details = VideoDetails(
            videoId=video_id,
            title=video_data["snippet"].get("title", ""),
            channelTitle=video_data["snippet"].get("channelTitle", ""),
            channelId=channel_id,
            publishTime=video_data["snippet"].get("publishedAt", ""),
            viewCount=view_count,
            likeCount=int(video_data["statistics"].get("likeCount", 0)),
            commentCount=int(video_data["statistics"].get("commentCount", 0)),
            subscriberCount=int(channel_stats.get("subscriberCount", 0)),
            duration=duration,
            description=description,
            thumbnails=video_data["snippet"].get("thumbnails", {}),
            videoLink=f"https://www.youtube.com/watch?v={video_id}",
            channelLink=f"https://www.youtube.com/channel/{channel_id}",
            relevancy_score=relevancy_score,
            brand_links=brand_links,
            country=video_country,  # Add country info to the VideoDetails
        )

        cache.store_video(video_details)
        all_videos.append(video_details)

    except Exception as e:
        print(f"Error processing video {video_id}: {str(e)}")


@cache.cached(expire=1800)
@app.post("/search", response_model=List[VideoDetails])
async def search_videos(payload: SearchRequest):

    start_time = time.time()

    # Generate a cache key based on the exact search parameters
    payload_dict = payload.model_dump()
    # Convert datetime objects to strings for consistent serialization
    if payload_dict.get("custom_date_from"):
        payload_dict["custom_date_from"] = payload_dict["custom_date_from"].isoformat()
    if payload_dict.get("custom_date_to"):
        payload_dict["custom_date_to"] = payload_dict["custom_date_to"].isoformat()

    # Create a consistent string representation and hash it
    payload_str = json.dumps(payload_dict, sort_keys=True)
    cache_key = f"search:{hashlib.md5(payload_str.encode()).hexdigest()}"

    print(f"Search cache key: {cache_key}")

    # Try to get complete results from cache
    try:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            print(f"🎉 CACHE HIT! Returning cached results")
            # Deserialize the cached data
            results_dict = json.loads(cached_data)
            results = [VideoDetails(**item) for item in results_dict]

            # Log performance
            elapsed = time.time() - start_time
            print(f"Cache hit response time: {elapsed:.4f} seconds")

            return results
        else:
            print("Cache miss, performing search...")
    except Exception as e:
        print(f"Error checking cache: {str(e)}")

    # If we reach here, it's a cache miss - execute the regular search logic

    try:
        if not youtube_manager.api_keys:
            raise HTTPException(
                status_code=500,
                detail="No YouTube API keys available. Please check environment variables.",
            )

        if not GOOGLE_CSE_API_KEY or not GOOGLE_CSE_ID:
            raise HTTPException(
                status_code=500,
                detail="Missing Google CSE credentials. Please check environment variables.",
            )

        google_cse = google_build("customsearch", "v1", developerKey=GOOGLE_CSE_API_KEY)

        country_code = None
        if (
            payload.country_code
            and payload.country_code.lower() != "string"
            and len(payload.country_code) == 2
        ):
            country_code = payload.country_code.upper()

        # Initialize string format dates for YouTube API
        publish_after_str = None
        publish_before_str = None

        # Initialize datetime objects for process_video function
        publish_after_dt = None
        publish_before_dt = None

        now = datetime.utcnow()

        if payload.date_filter != DateFilterOption.ALL_TIME:
            if payload.date_filter == DateFilterOption.CUSTOM:
                if payload.custom_date_from:
                    publish_after_str = payload.custom_date_from.isoformat() + "Z"
                    publish_after_dt = payload.custom_date_from
                if payload.custom_date_to:
                    publish_before_str = payload.custom_date_to.isoformat() + "Z"
                    publish_before_dt = payload.custom_date_to
            else:
                # Calculate publish dates based on filter option
                if payload.date_filter == DateFilterOption.PAST_HOUR:
                    publish_after_dt = now - timedelta(hours=1)
                elif payload.date_filter == DateFilterOption.PAST_3_HOURS:
                    publish_after_dt = now - timedelta(hours=3)
                elif payload.date_filter == DateFilterOption.PAST_6_HOURS:
                    publish_after_dt = now - timedelta(hours=6)
                elif payload.date_filter == DateFilterOption.PAST_12_HOURS:
                    publish_after_dt = now - timedelta(hours=12)
                elif payload.date_filter == DateFilterOption.PAST_24_HOURS:
                    publish_after_dt = now - timedelta(days=1)
                elif payload.date_filter == DateFilterOption.PAST_7_DAYS:
                    publish_after_dt = now - timedelta(days=7)
                elif payload.date_filter == DateFilterOption.PAST_30_DAYS:
                    publish_after_dt = now - timedelta(days=30)
                elif payload.date_filter == DateFilterOption.PAST_90_DAYS:
                    publish_after_dt = now - timedelta(days=90)
                elif payload.date_filter == DateFilterOption.PAST_180_DAYS:
                    publish_after_dt = now - timedelta(days=180)

                # Create string format for YouTube API
                if publish_after_dt:
                    publish_after_str = publish_after_dt.isoformat() + "Z"

        # Clean and split keywords
        keywords = [
            keyword.strip().lower()
            for keyword in payload.brand_name.split(",")
            if keyword.strip()
        ]

        all_videos = []
        seen_video_ids = set()
        tasks = []

        for keyword in keywords:
            try:
                # YouTube API search
                youtube_search_queries = [
                    f'"{keyword}"',
                    f'"{keyword}" review',
                    f'"{keyword}" tutorial',
                    f'"{keyword}" guide',
                ]

                for query in youtube_search_queries:
                    try:

                        async def search_videos(youtube):
                            search_params = {
                                "q": query,
                                "part": "id,snippet",
                                "maxResults": 50,
                                "type": "video",
                                "safeSearch": "none",
                                "relevanceLanguage": "en",
                            }

                            # Add date filters to YouTube API call if specified
                            if publish_after_str:
                                search_params["publishedAfter"] = publish_after_str
                            if publish_before_str:
                                search_params["publishedBefore"] = publish_before_str

                            # Add region code if specified (search within country)
                            if payload.country_code:
                                search_params["regionCode"] = country_code

                            return youtube.search().list(**search_params).execute()

                        search_response = await youtube_manager.execute_with_retry(
                            search_videos
                        )

                        for item in search_response.get("items", []):
                            video_id = item["id"].get("videoId")
                            if video_id and video_id not in seen_video_ids:
                                seen_video_ids.add(video_id)
                                # Pass datetime objects to process_video
                                tasks.append(
                                    process_video(
                                        video_id,
                                        keywords,
                                        all_videos,
                                        payload.min_views,
                                        publish_after_dt,
                                        publish_before_dt,
                                        payload.country_code,
                                    )
                                )

                    except Exception as e:
                        print(f"YouTube API search error: {str(e)}")

                # Google Custom Search
                if GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID:
                    google_search_queries = [
                        f'site:youtube.com "{keyword}"',
                        f'site:youtube.com/watch "{keyword}"',
                    ]

                    for query in google_search_queries:
                        try:
                            for start_index in range(1, 91, 10):
                                try:
                                    search_results = (
                                        google_cse.cse()
                                        .list(
                                            q=query, cx=GOOGLE_CSE_ID, start=start_index
                                        )
                                        .execute()
                                    )

                                    for item in search_results.get("items", []):
                                        video_id = extract_video_id_from_url(
                                            item["link"]
                                        )
                                        if video_id and video_id not in seen_video_ids:
                                            seen_video_ids.add(video_id)
                                            # Pass datetime objects to process_video for Google CSE results too
                                            tasks.append(
                                                process_video(
                                                    video_id,
                                                    keywords,
                                                    all_videos,
                                                    payload.min_views,
                                                    publish_after_dt,
                                                    publish_before_dt,
                                                    payload.country_code,
                                                )
                                            )

                                    if len(tasks) >= payload.max_results * 3:
                                        break

                                except HttpError as e:
                                    if e.resp.status in [
                                        403,
                                        429,
                                    ]:  # Quota exceeded or rate limit
                                        print(
                                            f"Google CSE quota exceeded or rate limited: {str(e)}"
                                        )
                                        break
                                    else:
                                        print(f"Google CSE error: {str(e)}")
                                        continue

                        except Exception as e:
                            print(f"Error in Google CSE search: {str(e)}")
                            continue

            except Exception as e:
                print(f"Error processing keyword '{keyword}': {str(e)}")

        # Execute all tasks
        await asyncio.gather(*tasks)

        # Sort videos by relevancy score and view count
        all_videos.sort(key=lambda x: (x.relevancy_score, x.viewCount), reverse=True)

        # Return exactly max_results videos or all available if less than max_results
        final_results = all_videos[: payload.max_results] if all_videos else []

        # Cache the results before returning
        try:
            # Serialize the VideoDetails objects
            results_dict = [result.dict() for result in final_results]
            serialized_data = json.dumps(results_dict)

            # Store in Redis with 1 hour expiration
            redis_client.setex(cache_key, 3600, serialized_data)  # 1 hour in seconds
            print(f"✅ Cached {len(final_results)} results with key: {cache_key}")
        except Exception as e:
            print(f"Error caching results: {str(e)}")

        # Log performance
        elapsed = time.time() - start_time
        print(f"Full search response time: {elapsed:.4f} seconds")

        return final_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.get("/redis-test")
async def test_redis():
    try:
        # Get Redis client
        redis_client = Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379"))

        # Set a test value
        redis_client.set("test_key", "Hello from Redis!")

        # Get the value back
        value = redis_client.get("test_key")

        return {
            "redis_connected": True,
            "test_value": value.decode("utf-8") if value else None,
            "keys_in_db": len(redis_client.keys("*")),
            "sample_keys": [k.decode("utf-8") for k in redis_client.keys("*")[:5]],
        }
    except Exception as e:
        return {"redis_connected": False, "error": str(e)}


@app.get("/cache-info/{key}")
async def get_cache_info(key: str = None):
    try:
        if key:
            # Get info about a specific key
            value = redis_client.get(key)
            return {
                "key": key,
                "exists": value is not None,
                "size": len(value) if value else 0,
                "ttl": redis_client.ttl(key),
            }
        else:
            # Get general cache info
            keys = redis_client.keys("search:*")
            return {
                "total_search_keys": len(keys),
                "sample_keys": [k.decode("utf-8") for k in keys[:10]],
                "memory_usage": redis_client.info("memory"),
            }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
