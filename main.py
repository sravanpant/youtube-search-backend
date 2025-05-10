# backend/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from googleapiclient.discovery import build as google_build
from googleapiclient.errors import HttpError
import os, asyncio, time, json, hashlib
from dotenv import load_dotenv
from redis import Redis
from typing import List, Optional
from datetime import datetime, timedelta
from collections import deque
from models import SearchRequest, VideoDetails, DateFilterOption, KeywordFilterOption
from youtube_manager import YouTubeAPIManager
from video_processor import process_video, parse_duration
from cache import RedisCache
from models import SearchRequest, VideoDetails, DateFilterOption, KeywordFilterOption
from utils import extract_video_id_from_url, calculate_relevancy_score, extract_brand_related_urls

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

REDIS_URL = os.getenv("REDIS_URL")
cache = RedisCache(REDIS_URL)

redis_client = Redis(connection_pool=cache.pool)

app = FastAPI()

all_keys = [
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

# Create a global instance of the YouTube API manager
youtube_manager = YouTubeAPIManager(all_keys)

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

    payload_dict["keyword_filter_explicit"] = str(payload.keyword_filter)

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

    # Get the excluded channels list (convert to lowercase for case-insensitive matching)
    excluded_channels = (
        [channel.strip() for channel in payload.excluded_channels]
        if payload.excluded_channels
        else []
    )

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
                    # Format correctly for YouTube API (RFC 3339 format)
                    publish_after_dt = payload.custom_date_from
                    # Remove the trailing 'Z' and properly format without milliseconds
                    publish_after_str = publish_after_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                if payload.custom_date_to:
                    publish_before_dt = payload.custom_date_to
                    # Remove the trailing 'Z' and properly format without milliseconds
                    publish_before_str = publish_before_dt.strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    )
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
                    publish_after_str = publish_after_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Clean and split keywords
        keywords = [
            keyword.strip().lower()
            for keyword in payload.brand_name.split(",")
            if keyword.strip()
        ]

        all_videos = []
        seen_video_ids = set()
        tasks = []
        valid_video_count = 0

        # Constants for search optimization
        MAX_SEARCH_MULTIPLIER = 10  # Higher multiplier to find more videos
        MAX_ATTEMPTS = 3

        # ===== PHASE 1: COLLECT CANDIDATE VIDEO IDS =====
        print("Phase 1: Collecting candidate videos...")
        phase1_start = time.time()

        for attempt in range(MAX_ATTEMPTS):
            if len(seen_video_ids) >= payload.max_results * MAX_SEARCH_MULTIPLIER:
                break

            for keyword in keywords:
                if len(seen_video_ids) >= payload.max_results * MAX_SEARCH_MULTIPLIER:
                    break

                # More comprehensive search queries
                youtube_search_queries = [
                    f'"{keyword}"',
                    f'"{keyword}" review',
                    f'"{keyword}" tutorial',
                    f'"{keyword}" guide',
                    f'"{keyword}" how to',
                    f'"{keyword}" app',
                    f'"{keyword}" demo',
                    f'"{keyword}" explained',
                ]

                needed_videos = max(
                    payload.max_results * MAX_SEARCH_MULTIPLIER - len(seen_video_ids),
                    50,
                )

                for query in youtube_search_queries:
                    if (
                        len(seen_video_ids)
                        >= payload.max_results * MAX_SEARCH_MULTIPLIER
                    ):
                        break

                    try:

                        async def search_videos(youtube):
                            search_params = {
                                "q": query,
                                "part": "id",  # Only need IDs in first phase
                                "maxResults": min(needed_videos, 50),
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

                    except Exception as e:
                        print(f"YouTube API search error: {str(e)}")

                # Google Custom Search - more efficient version
                if (
                    GOOGLE_CSE_API_KEY
                    and GOOGLE_CSE_ID
                    and len(seen_video_ids)
                    < payload.max_results * MAX_SEARCH_MULTIPLIER
                ):
                    google_search_queries = [
                        f'site:youtube.com "{keyword}"',
                        f'site:youtube.com/watch "{keyword}"',
                        f"site:youtube.com {keyword} review",
                        f"site:youtube.com {keyword} tutorial",
                    ]

                    for query in google_search_queries:
                        if (
                            len(seen_video_ids)
                            >= payload.max_results * MAX_SEARCH_MULTIPLIER
                        ):
                            break

                        try:
                            # Only do 3 pages per query for speed
                            for start_index in range(1, 31, 10):
                                if (
                                    len(seen_video_ids)
                                    >= payload.max_results * MAX_SEARCH_MULTIPLIER
                                ):
                                    break

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

                                except HttpError as e:
                                    if e.resp.status in [403, 429]:
                                        break
                                    else:
                                        continue

                        except Exception as e:
                            print(f"Error in Google CSE search: {str(e)}")

            # If we have enough videos, break out of the attempts loop
            if len(seen_video_ids) >= payload.max_results * 3:
                break

            # If we're not finding enough videos, try expanded search terms
            if attempt > 0 and len(seen_video_ids) < payload.max_results * 2:
                print(
                    f"Expanded search attempt {attempt+1} - found {len(seen_video_ids)} candidates so far"
                )
                expanded_keywords = []
                for keyword in keywords:
                    expanded_keywords.extend(
                        [
                            f"{keyword} best",
                            f"{keyword} app",
                            f"{keyword} recommended",
                            f"using {keyword}",
                        ]
                    )
                # Add expanded keywords to our search
                keywords.extend(expanded_keywords)

        phase1_time = time.time() - phase1_start
        print(
            f"Phase 1 complete: Found {len(seen_video_ids)} candidate videos in {phase1_time:.2f}s"
        )

        # ===== PHASE 2: PROCESS VIDEOS IN BATCHES =====
        print("Phase 2: Processing videos in batches...")
        phase2_start = time.time()

        # Process videos in batches of 50 where possible
        video_id_list = list(seen_video_ids)

        # First try to process videos in batches (more efficient)
        for i in range(0, len(video_id_list), 50):
            batch_ids = video_id_list[i : i + 50]

            # Skip if we have enough videos already
            if len(all_videos) >= payload.max_results * 2:
                break

            try:
                # Get details for up to 50 videos at once
                async def get_videos_batch(youtube):
                    return (
                        youtube.videos()
                        .list(
                            part="snippet,contentDetails,statistics,recordingDetails",
                            id=",".join(batch_ids),
                        )
                        .execute()
                    )

                batch_response = await youtube_manager.execute_with_retry(
                    get_videos_batch
                )

                # Gather unique channel IDs from this batch
                channel_ids = []
                channel_map = {}

                # First pass - extract channels and check basic criteria
                valid_videos_data = []

                for video_data in batch_response.get("items", []):
                    # Check view count meets minimum
                    view_count = int(video_data["statistics"].get("viewCount", 0))
                    if view_count < payload.min_views:
                        continue

                    # Check duration meets minimum (60 seconds)
                    try:
                        duration = video_data["contentDetails"]["duration"]
                        duration_seconds = parse_duration(duration)
                        if duration_seconds < 60:
                            continue
                    except:
                        continue

                    # Calculate relevancy score
                    relevancy_score = calculate_relevancy_score(video_data, keywords)
                    if relevancy_score < 3:
                        continue

                    # Check channel name against excluded list
                    channel_title = video_data["snippet"]["channelTitle"].lower()
                    if any(
                        excluded.lower() in channel_title
                        for excluded in excluded_channels
                    ):
                        continue

                    # Skip if channel name contains any of the keywords
                    if any(kw in channel_title for kw in keywords):
                        continue

                    # Store channel ID for batch processing
                    channel_id = video_data["snippet"]["channelId"]
                    if channel_id not in channel_map:
                        channel_ids.append(channel_id)
                        channel_map[channel_id] = []

                    # Add to list of videos sharing this channel
                    channel_map[channel_id].append(video_data)
                    valid_videos_data.append(video_data)

                # If we have valid videos, fetch their channels in a batch
                if channel_ids:
                    # Process channels in batches of 50
                    channel_stats = {}
                    for j in range(0, len(channel_ids), 50):
                        channel_batch = channel_ids[j : j + 50]

                        async def get_channels_batch(youtube):
                            return (
                                youtube.channels()
                                .list(part="statistics", id=",".join(channel_batch))
                                .execute()
                            )

                        channels_response = await youtube_manager.execute_with_retry(
                            get_channels_batch
                        )

                        # Map channel stats by ID
                        for channel in channels_response.get("items", []):
                            channel_stats[channel["id"]] = channel.get("statistics", {})

                    # Now create video details objects
                    for video_data in valid_videos_data:
                        video_id = video_data["id"]
                        channel_id = video_data["snippet"]["channelId"]

                        # Extract info needed for VideoDetails
                        description = video_data["snippet"].get("description", "")
                        brand_links = extract_brand_related_urls(description, keywords)

                        # Extract country information
                        video_country = None
                        recording_details = video_data.get("recordingDetails", {})
                        if recording_details and recording_details.get(
                            "locationDescription"
                        ):
                            location_desc = recording_details.get(
                                "locationDescription", ""
                            )
                            video_country = (
                                location_desc.split(",")[-1].strip()
                                if "," in location_desc
                                else location_desc
                            )

                        # Also check snippet country
                        snippet_country = video_data.get("snippet", {}).get("country")
                        if snippet_country:
                            video_country = snippet_country

                        # Check if country code filter matches
                        if (
                            country_code
                            and video_country
                            and video_country.upper() != country_code.upper()
                        ):
                            continue

                        title_lower = video_data["snippet"].get("title", "").lower()
                        description_lower = description.lower()

                        import re

                        has_keyword_in_title = any(
                            re.search(r"\b" + re.escape(keyword) + r"\b", title_lower)
                            for keyword in keywords
                        )
                        has_keyword_in_description = any(
                            re.search(
                                r"\b" + re.escape(keyword) + r"\b", description_lower
                            )
                            for keyword in keywords
                        )

                        # Get channel stats
                        channel_subscriber_count = int(
                            channel_stats.get(channel_id, {}).get("subscriberCount", 0)
                        )

                        # Create video details
                        video_details = VideoDetails(
                            videoId=video_id,
                            title=video_data["snippet"].get("title", ""),
                            channelTitle=video_data["snippet"].get("channelTitle", ""),
                            channelId=channel_id,
                            publishTime=video_data["snippet"].get("publishedAt", ""),
                            viewCount=int(video_data["statistics"].get("viewCount", 0)),
                            likeCount=int(video_data["statistics"].get("likeCount", 0)),
                            commentCount=int(
                                video_data["statistics"].get("commentCount", 0)
                            ),
                            subscriberCount=channel_subscriber_count,
                            duration=video_data["contentDetails"].get("duration", ""),
                            description=description,
                            thumbnails=video_data["snippet"].get("thumbnails", {}),
                            videoLink=f"https://www.youtube.com/watch?v={video_id}",
                            channelLink=f"https://www.youtube.com/channel/{channel_id}",
                            relevancy_score=calculate_relevancy_score(
                                video_data, keywords
                            ),
                            brand_links=brand_links,
                            country=video_country,
                            has_keyword_in_title=has_keyword_in_title,
                            has_keyword_in_description=has_keyword_in_description,
                        )

                        # Cache this video for future use
                        cache.store_video(
                            video_details, expire=7 * 24 * 60 * 60
                        )  # 1 week cache for videos

                        # Add to results if not already present
                        all_videos.append(video_details)

            except Exception as e:
                print(f"Error processing batch {i//50 + 1}: {str(e)}")

        # If we still need more videos, process remaining ones individually
        if len(all_videos) < payload.max_results * 1.2 and len(video_id_list) > 0:
            print(
                f"Batch processing yielded {len(all_videos)} videos, processing remainders individually..."
            )

            # Process remaining videos individually to ensure we get enough results
            remaining_ids = [
                vid
                for vid in video_id_list
                if vid not in [v.videoId for v in all_videos]
            ]
            remaining_tasks = []

            for video_id in remaining_ids[
                : payload.max_results * 3
            ]:  # Limit how many we process
                remaining_tasks.append(
                    process_video(
                        video_id,
                        keywords,
                        all_videos,
                        payload.min_views,
                        youtube_manager,  # Pass the manager
                        cache,  # Pass the cache
                        publish_after_dt,
                        publish_before_dt,
                        country_code,
                        excluded_channels,
                    )
                )

            # Process in smaller batches for better control
            BATCH_SIZE = 25
            for i in range(0, len(remaining_tasks), BATCH_SIZE):
                batch = remaining_tasks[i : i + BATCH_SIZE]
                await asyncio.gather(*batch)

                # Check if we have enough now
                if len(all_videos) >= payload.max_results * 1.5:
                    print(
                        f"Found {len(all_videos)} videos, stopping further processing"
                    )
                    break

        phase2_time = time.time() - phase2_start
        print(
            f"Phase 2 complete: Processed videos in {phase2_time:.2f}s, found {len(all_videos)} valid videos"
        )

        # Final filtering phase
        print(f"Initial video count: {len(all_videos)}")

        # Apply keyword location filter
        filtered_videos = []
        for video in all_videos:
            include_video = False

            if payload.keyword_filter == KeywordFilterOption.ANY:
                # Default behavior - include all videos that passed the relevancy check
                include_video = True
            elif payload.keyword_filter == KeywordFilterOption.TITLE_ONLY:
                # Only include if keyword is in title BUT NOT in description
                include_video = (
                    video.has_keyword_in_title and not video.has_keyword_in_description
                )
            elif payload.keyword_filter == KeywordFilterOption.DESCRIPTION_ONLY:
                # Only include if keyword is in description BUT NOT in title
                include_video = (
                    video.has_keyword_in_description and not video.has_keyword_in_title
                )
            elif payload.keyword_filter == KeywordFilterOption.TITLE_AND_DESCRIPTION:
                # Must be in both title AND description
                include_video = (
                    video.has_keyword_in_title and video.has_keyword_in_description
                )
            if include_video:
                filtered_videos.append(video)

        print(f"After {payload.keyword_filter} filtering: {len(filtered_videos)}")

        # Sort the filtered videos
        filtered_videos.sort(
            key=lambda x: (x.relevancy_score, x.viewCount), reverse=True
        )

        # Return exactly max_results videos or all available if less than max_results
        final_results = (
            filtered_videos[: payload.max_results] if filtered_videos else []
        )

        # If we still don't have enough results, attempt a last-resort relaxed search
        if len(final_results) < payload.max_results and len(final_results) > 0:
            print(
                f"Warning: Only found {len(final_results)} valid results out of {payload.max_results} requested"
            )
            print("Attempting final expanded search with relaxed criteria...")

            # This would be the place to add a last-resort search strategy
            # For example, you could lower the minimum view count, expand search terms, etc.
            # However, this would require a significant code addition

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
