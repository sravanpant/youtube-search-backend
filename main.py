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
from utils import (
    extract_video_id_from_url,
    calculate_relevancy_score,
    extract_brand_related_urls,
)
import pandas as pd


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
@app.post("/search", response_model=List[dict])
async def search_videos_snippet_only(payload: SearchRequest):
    start_time = time.time()

    # Generate cache key
    payload_dict = payload.model_dump()

    def json_serializable(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    # Handle datetime objects for caching
    if payload_dict.get("custom_date_from"):
        payload_dict["custom_date_from"] = payload_dict["custom_date_from"].isoformat()
    if payload_dict.get("custom_date_to"):
        payload_dict["custom_date_to"] = payload_dict["custom_date_to"].isoformat()

    payload_str = json.dumps(payload_dict, sort_keys=True, default=json_serializable)
    cache_key = f"search:{hashlib.md5(payload_str.encode()).hexdigest()}"
    print(f"Search cache key: {cache_key}")

    # Try to get results from cache
    try:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            print("🎉 CACHE HIT! Returning cached results")
            results = json.loads(cached_data)
            elapsed = time.time() - start_time
            print(f"Cache hit response time: {elapsed:.4f} seconds")
            return results[: payload.max_results]
        else:
            print("Cache miss, performing search...")
    except Exception as e:
        print(f"Error checking cache: {str(e)}")

    # Extract brand name keywords for searching
    keywords = [
        kw.strip().lower() for kw in payload.brand_name.split(",") if kw.strip()
    ]

    # Define CSV filename based on brand name
    csv_filename = f"videos_data.csv"

    # ---- DYNAMIC COLLECTION TARGET BASED ON MAX_RESULTS ----
    if payload.max_results <= 50:
        COLLECTION_TARGET = 500
    elif 50 < payload.max_results <= 200:
        COLLECTION_TARGET = 1000
    elif 200 < payload.max_results <= 300:
        COLLECTION_TARGET = 1500
    else:
        COLLECTION_TARGET = 2000

    print(
        f"Setting collection target to {COLLECTION_TARGET} videos for max_results={payload.max_results}"
    )

    all_results = []
    seen_video_ids = set()

    # Search query variations
    query_templates = [
        '"{}" brand',
        '"{}"',
        '"{}" review',
        '"{}" tutorial',
        '"{}" guide',
        '"{}" how to',
        '"{}" app',
        '"{}" demo',
        '"{}" explained',
    ]

    # Search phase start time
    search_start = time.time()

    # IMPROVED FUNCTION: Detect if a channel is an official brand channel
    def is_official_brand_channel(channel_title, keywords):
        """
        Enhanced detection of official brand channels based on various patterns.
        """
        channel_title_lower = channel_title.lower()

        # Common languages that might be used in channel names
        languages = [
            "hindi",
            "english",
            "telugu",
            "tamil",
            "kannada",
            "malayalam",
            "bengali",
            "marathi",
            "gujarati",
            "punjabi",
            "urdu",
            "odia",
            "assamese",
            "spanish",
            "french",
            "german",
            "chinese",
            "japanese",
        ]

        # Common patterns that indicate official channels
        official_patterns = [
            "official",
            ".com",
            "app",
            "official channel",
            "official page",
            "learn with",
            "tech",
            "technology",
            "digital",
            "support",
            "help",
            "customer service",
            "tutorials",
            "studio",
            "hq",
            "headquarters",
            "global",
            "india",
            "us",
            "uk",
            "tv",
            "media",
            "videos",
            "channel",
        ]

        # Finance/Investment specific patterns (for brands like Groww)
        finance_patterns = [
            "invest",
            "investing",
            "investor",
            "investment",
            "finance",
            "financial",
            "wealth",
            "money",
            "stocks",
            "stock market",
            "mutual fund",
            "mutual funds",
            "mf",
            "amc",
            "asset",
            "management",
            "portfolio",
            "trading",
            "trader",
            "market",
            "equity",
            "crypto",
            "thrive",
            "grow",
            "savings",
            "sip",
            "capital",
            "wealth",
        ]

        # Common prepositions and conjunctions used in channel names
        connectors = ["with", "by", "for", "on", "in", "and", "at", "&", "+", "-"]

        for keyword in keywords:
            # 1. EXACT MATCH - Most obvious case
            if channel_title_lower == keyword:
                return True

            # 2. BRAND NAME IS A SIGNIFICANT PART OF THE CHANNEL NAME
            # Check if brand is at beginning, middle, or end with word boundaries
            import re

            keyword_pattern = r"\b" + re.escape(keyword) + r"\b"
            if re.search(keyword_pattern, channel_title_lower):

                # 3. BRAND + SOMETHING or SOMETHING + BRAND
                # Check for patterns like "[brand] [anything]" or "[anything] [brand]"
                words = channel_title_lower.split()
                if len(words) >= 2:
                    # If brand is first or last word in channel name
                    if words[0] == keyword or words[-1] == keyword:
                        return True

                    # If brand is anywhere in the name
                    for i, word in enumerate(words):
                        if word == keyword:
                            # Check words around the keyword
                            # If surrounded by finance terms
                            for w in words:
                                if w in finance_patterns or any(
                                    p in w for p in finance_patterns
                                ):
                                    return True

                            # Check for patterns with connectors like "with", "by", etc.
                            if i > 0 and words[i - 1] in connectors:
                                return True
                            if i < len(words) - 1 and words[i + 1] in connectors:
                                return True

                # 4. BRAND NAME WITH SPECIFIC ADDITIONS
                # Check for key patterns that strongly suggest official status

                # Financial/Investment services patterns
                for pattern in finance_patterns:
                    if f"{keyword} {pattern}" in channel_title_lower:
                        return True
                    if f"{pattern} {keyword}" in channel_title_lower:
                        return True

                    # Also check with connectors
                    for connector in connectors:
                        # "mutual funds with groww", "thrive by groww", etc.
                        if f"{pattern} {connector} {keyword}" in channel_title_lower:
                            return True
                        if f"{keyword} {connector} {pattern}" in channel_title_lower:
                            return True

                # Official channel patterns
                for pattern in official_patterns:
                    if f"{keyword} {pattern}" in channel_title_lower:
                        return True
                    if f"{pattern} {keyword}" in channel_title_lower:
                        return True

                # Language-specific channels
                for lang in languages:
                    if f"{keyword} {lang}" in channel_title_lower:
                        return True
                    if f"{lang} {keyword}" in channel_title_lower:
                        return True

            # 5. ACRONYM DETECTION
            # For brands with multiple words, check for acronym usage
            if " " in keyword:
                # Create acronym from multi-word brand name
                acronym = "".join(word[0] for word in keyword.split())
                if acronym.lower() in channel_title_lower:
                    return True

        return False

    try:
        # Run searches until we collect enough videos
        for keyword in keywords:
            if len(all_results) >= COLLECTION_TARGET:
                break

            print(f"Processing keyword: {keyword}")

            for template in query_templates:
                if len(all_results) >= COLLECTION_TARGET:
                    break

                query = template.format(keyword)
                print(f"Searching for: {query}")

                # Track pagination
                next_page_token = None
                page_count = 0

                # Get up to 5 pages of results per query
                while page_count < 5 and len(all_results) < COLLECTION_TARGET:
                    page_count += 1

                    try:
                        # Define the search request
                        async def run_search(youtube):
                            params = {
                                "q": query,
                                "part": "snippet",
                                "maxResults": 50,
                                "type": "video",
                                "relevanceLanguage": "en",
                            }
                            if next_page_token:
                                params["pageToken"] = next_page_token
                            return youtube.search().list(**params).execute()

                        # Execute the search
                        search_response = await youtube_manager.execute_with_retry(
                            run_search, "search.list"
                        )

                        # Process results
                        items = search_response.get("items", [])
                        next_page_token = search_response.get("nextPageToken")

                        print(f"Found {len(items)} results on page {page_count}")

                        # Extract data from each result
                        for item in items:
                            video_id = item["id"].get("videoId")

                            # Skip if we've seen this video before
                            if not video_id or video_id in seen_video_ids:
                                continue

                            seen_video_ids.add(video_id)

                            # Extract snippet data
                            snippet = item.get("snippet", {})
                            title = snippet.get("title", "")
                            description = snippet.get("description", "")
                            channel_title = snippet.get("channelTitle", "")
                            channel_id = snippet.get("channelId", "")

                            # IMPROVED RELEVANCY CHECK using word boundaries (from the old code)
                            title_lower = title.lower()
                            description_lower = description.lower()

                            # Check for exact keyword matches with word boundaries
                            import re

                            has_keyword_in_title = any(
                                re.search(r"\b" + re.escape(kw) + r"\b", title_lower)
                                for kw in keywords
                            )
                            has_keyword_in_description = any(
                                re.search(
                                    r"\b" + re.escape(kw) + r"\b", description_lower
                                )
                                for kw in keywords
                            )

                            # Skip if not relevant (improved filtering)
                            if not (has_keyword_in_title or has_keyword_in_description):
                                continue

                            # IMPROVED OFFICIAL CHANNEL DETECTION
                            if is_official_brand_channel(channel_title, keywords):
                                continue

                            video_link = f"https://www.youtube.com/watch?v={video_id}"
                            channel_link = (
                                f"https://www.youtube.com/channel/{channel_id}"
                            )

                            # Create a clean result object with only the fields we want
                            result = {
                                "videoId": video_id,
                                "title": title,
                                "description": description,
                                "publishTime": snippet.get("publishedAt", ""),
                                "channelId": channel_id,
                                "channelTitle": channel_title,
                                "thumbnails": snippet.get("thumbnails", {}),
                                "viewCount": 0,
                                "likeCount": 0,
                                "commentCount": 0,
                                "subscriberCount": 0,
                                "duration": "",
                                "durationSeconds": 0,
                                "country": "",
                                "videoLink": video_link,
                                "channelLink": channel_link,
                                "brand_links": [],
                                "has_keyword_in_title": has_keyword_in_title,
                                "has_keyword_in_description": has_keyword_in_description,
                                "has_sponsored_links": False,
                                "relevancy_score": 0,  # Will be calculated in Phase 2
                            }

                            all_results.append(result)

                        # If no more pages or we have enough videos, stop pagination
                        if not next_page_token or len(all_results) >= COLLECTION_TARGET:
                            break

                    except Exception as e:
                        print(
                            f"Error on page {page_count} for query '{query}': {str(e)}"
                        )
                        break

        # Search phase complete
        search_time = time.time() - search_start
        print(
            f"Phase 1 complete: collected {len(all_results)} videos in {search_time:.2f} seconds"
        )

        # ===== PHASE 2: Get Video Details (Duration and View Count) CONCURRENTLY =====
        details_start = time.time()

        # Create a video ID to index mapping for quick updates
        video_id_to_index = {video["videoId"]: i for i, video in enumerate(all_results)}

        # Process in batches of 50 (YouTube API maximum)
        video_ids = list(seen_video_ids)

        # Create batches of 50 videos
        batches = []
        for i in range(0, len(video_ids), 50):
            batches.append(video_ids[i : i + 50])

        total_batches = len(batches)
        print(
            f"Phase 2: Fetching details for {len(video_ids)} videos in {total_batches} concurrent batches"
        )

        # Define a function to process one batch
        async def process_batch(batch_id, video_batch):
            batch_start = time.time()

            try:
                # Fetch details for this batch
                async def get_video_details(youtube):
                    return (
                        youtube.videos()
                        .list(
                            part="contentDetails,statistics", id=",".join(video_batch)
                        )
                        .execute()
                    )

                details_response = await youtube_manager.execute_with_retry(
                    get_video_details, "videos.list"
                )

                # Process the batch results
                batch_results = {}  # Store results to update atomically later

                for item in details_response.get("items", []):
                    video_id = item["id"]

                    # Extract content details and statistics
                    content_details = item.get("contentDetails", {})
                    statistics = item.get("statistics", {})
                    snippet = item.get("snippet", {})

                    # Get ISO 8601 duration
                    duration = content_details.get("duration", "PT0S")
                    view_count = int(statistics.get("viewCount", 0))
                    like_count = int(statistics.get("likeCount", 0))

                    relevancy_score = calculate_relevancy_score(
                        {"snippet": snippet}, keywords
                    )

                    # Convert to seconds
                    try:
                        duration_seconds = parse_duration(duration)
                    except Exception:
                        duration_seconds = 0

                    # Store in batch results
                    batch_results[video_id] = {
                        "duration": duration,
                        "durationSeconds": duration_seconds,
                        "viewCount": view_count,
                        "likeCount": like_count,
                        "relevancy_score": relevancy_score,
                    }

                batch_time = time.time() - batch_start
                print(
                    f"✅ Batch {batch_id+1}/{total_batches} completed in {batch_time:.2f}s"
                )

                return batch_results

            except Exception as e:
                print(f"❌ Error processing batch {batch_id+1}: {str(e)}")
                return {}  # Return empty dict on error

        # Create concurrent tasks for all batches
        batch_tasks = []
        for i, batch in enumerate(batches):
            task = process_batch(i, batch)
            batch_tasks.append(task)

        # Execute all batch tasks concurrently
        print(f"Starting concurrent execution of {len(batch_tasks)} batch tasks...")
        batch_results_list = await asyncio.gather(*batch_tasks)

        # Update all_results with the batch results
        for batch_results in batch_results_list:
            for video_id, details in batch_results.items():
                index = video_id_to_index.get(video_id)
                if index is not None:
                    all_results[index]["duration"] = details["duration"]
                    all_results[index]["durationSeconds"] = details["durationSeconds"]
                    all_results[index]["viewCount"] = details["viewCount"]
                    all_results[index]["likeCount"] = details["likeCount"]
                    all_results[index]["relevancy_score"] = details["relevancy_score"]

        details_time = time.time() - details_start
        print(
            f"Phase 2 complete: fetched video details concurrently in {details_time:.2f} seconds"
        )

        # ===== PHASE 3: Get Channel Details (Subscriber Count and Country) CONCURRENTLY =====
        channel_details_start = time.time()

        # Extract unique channel IDs from all videos
        unique_channel_ids = set(
            video["channelId"] for video in all_results if video.get("channelId")
        )
        channel_ids_list = list(unique_channel_ids)

        print(
            f"Phase 3: Found {len(channel_ids_list)} unique channels to fetch data for"
        )

        # Create a channelId to details mapping
        channel_id_to_details = {}

        # Create batches of 50 channels (YouTube API limit)
        channel_batches = []
        for i in range(0, len(channel_ids_list), 50):
            channel_batches.append(channel_ids_list[i : i + 50])

        total_channel_batches = len(channel_batches)
        print(
            f"Phase 3: Fetching details for {len(channel_ids_list)} channels in {total_channel_batches} concurrent batches"
        )

        # Define a function to process one batch of channels
        async def process_channel_batch(batch_id, channel_batch):
            batch_start = time.time()

            try:
                # Fetch details for this batch of channels
                async def get_channel_details(youtube):
                    return (
                        youtube.channels()
                        .list(part="statistics,snippet", id=",".join(channel_batch))
                        .execute()
                    )

                channel_response = await youtube_manager.execute_with_retry(
                    get_channel_details, "channels.list"
                )

                # Process the batch results
                batch_results = {}  # Store results to update atomically later

                for item in channel_response.get("items", []):
                    channel_id = item["id"]

                    # Extract statistics and snippet data
                    statistics = item.get("statistics", {})
                    snippet = item.get("snippet", {})
                    branding = item.get("brandingSettings", {}).get("channel", {})

                    # Get subscriber count and country
                    subscriber_count = int(statistics.get("subscriberCount", 0))
                    country = snippet.get("country", "")
                    keywords = snippet.get("keywords", "")
                    channel_link = f"https://www.youtube.com/channel/{channel_id}"
                    channel_description = snippet.get("description", "")
                    channel_title = snippet.get("title", "")
                    channel_type = branding.get("profileType", "")

                    # Determine if this is a music channel, comedy channel, etc.
                    is_music_channel = any(
                        term in channel_title.lower() or term in keywords.lower()
                        for term in [
                            "music",
                            "songs",
                            "record",
                            "artist",
                            "band",
                            "singer",
                        ]
                    )
                    is_meme_channel = any(
                        term in channel_title.lower() or term in keywords.lower()
                        for term in ["meme", "funny", "comedy", "laugh", "humor"]
                    )

                    # Store in batch results
                    batch_results[channel_id] = {
                        "subscriberCount": subscriber_count,
                        "country": country,
                        "channelLink": channel_link,
                        "is_music_channel": is_music_channel,
                        "is_meme_channel": is_meme_channel,
                    }

                batch_time = time.time() - batch_start
                print(
                    f"✅ Channel Batch {batch_id+1}/{total_channel_batches} completed in {batch_time:.2f}s"
                )

                return batch_results

            except Exception as e:
                print(f"❌ Error processing channel batch {batch_id+1}: {str(e)}")
                return {}  # Return empty dict on error

        # Create concurrent tasks for all channel batches
        channel_batch_tasks = []
        for i, batch in enumerate(channel_batches):
            task = process_channel_batch(i, batch)
            channel_batch_tasks.append(task)

        # Execute all channel batch tasks concurrently
        print(
            f"Starting concurrent execution of {len(channel_batch_tasks)} channel batch tasks..."
        )
        channel_batch_results_list = await asyncio.gather(*channel_batch_tasks)

        # Combine all channel batch results into a single dictionary
        for batch_results in channel_batch_results_list:
            channel_id_to_details.update(batch_results)

        # Now update all videos with their channel's details
        for video in all_results:
            channel_id = video.get("channelId", "")
            if channel_id and channel_id in channel_id_to_details:
                details = channel_id_to_details[channel_id]
                video["subscriberCount"] = details["subscriberCount"]
                video["country"] = details.get("country", "") or video["country"]
                video["channelLink"] = details["channelLink"]

                # Filter out music videos and meme videos with lower relevancy scores
                if (
                    details.get("is_music_channel") or details.get("is_meme_channel")
                ) and video["relevancy_score"] < 5:
                    video["relevancy_score"] = max(0, video["relevancy_score"] - 3)
            else:
                # Default values if channel details not found
                video["subscriberCount"] = 0
                video["country"] = ""
                video["channelLink"] = f"https://www.youtube.com/channel/{channel_id}"

        channel_details_time = time.time() - channel_details_start
        print(
            f"Phase 3 complete: fetched channel details concurrently in {channel_details_time:.2f} seconds"
        )

        # ===== PHASE 4: Extract Brand-Related URLs from Descriptions Check for Keywords CONCURRENTLY =====
        text_analysis_start = time.time()

        # Import re and urlparse if not already imported
        import re
        from urllib.parse import urlparse

        # First, ensure we have the list of keywords from the payload
        keywords = [kw.strip() for kw in payload.brand_name.split(",") if kw.strip()]
        print(f"Phase 4: Extracting brand URLs using keywords: {keywords}")

        # Process keywords for better matching
        processed_keywords = []
        for keyword in keywords:
            # Clean each keyword and convert to lowercase for case-insensitive matching
            clean_keyword = keyword.lower().strip()
            processed_keywords.append(clean_keyword)

        # Create batches of videos for processing (50 per batch)
        video_batches = []
        for i in range(0, len(all_results), 50):
            video_batches.append(all_results[i : i + 50])

        total_analysis_batches = len(video_batches)
        print(
            f"Phase 4: Processing text in {total_analysis_batches} concurrent batches"
        )

        # Define a function to process one batch of videos
        async def process_text_batch(batch_id, videos_batch):
            batch_start = time.time()
            loop = asyncio.get_running_loop()

            try:
                # Process each video in the batch
                def process_batch_content():
                    results = {}
                    for video in videos_batch:
                        video_id = video["videoId"]
                        title = video.get("title", "").lower()
                        description = video.get("description", "").lower()

                        # Check for keywords in title and description
                        has_keyword_in_title = any(
                            keyword in title for keyword in processed_keywords
                        )
                        has_keyword_in_description = any(
                            keyword in description for keyword in processed_keywords
                        )

                        # Extract brand-related URLs from the description
                        brand_urls = extract_brand_related_urls(description, keywords)

                        # Store results
                        results[video_id] = {
                            "brand_links": brand_urls,
                            "has_keyword_in_title": has_keyword_in_title,
                            "has_keyword_in_description": has_keyword_in_description,
                            "has_sponsored_links": len(brand_urls)
                            > 0,  # Flag if brand links exist
                        }

                    return results

                # Execute the CPU-bound work in a thread pool
                batch_results = await loop.run_in_executor(None, process_batch_content)

                batch_time = time.time() - batch_start
                print(
                    f"✅ Text Analysis Batch {batch_id+1}/{total_analysis_batches} completed in {batch_time:.2f}s"
                )

                return batch_results

            except Exception as e:
                print(f"❌ Error processing text analysis batch {batch_id+1}: {str(e)}")
                return {}  # Return empty dict on error

        # Create concurrent tasks for all text analysis batches
        text_batch_tasks = []
        for i, batch in enumerate(video_batches):
            task = process_text_batch(i, batch)
            text_batch_tasks.append(task)

        # Execute all text analysis tasks concurrently
        print(
            f"Starting concurrent execution of {len(text_batch_tasks)} text analysis tasks..."
        )
        text_batch_results_list = await asyncio.gather(*text_batch_tasks)

        # Process text analysis results
        video_id_to_text_analysis = {}
        for batch_results in text_batch_results_list:
            video_id_to_text_analysis.update(batch_results)

        # Update all videos with their text analysis results
        for video in all_results:
            video_id = video["videoId"]
            if video_id in video_id_to_text_analysis:
                analysis = video_id_to_text_analysis[video_id]
                video["brand_links"] = analysis["brand_links"]
                video["has_keyword_in_title"] = analysis["has_keyword_in_title"]
                video["has_keyword_in_description"] = analysis[
                    "has_keyword_in_description"
                ]
                video["has_sponsored_links"] = analysis["has_sponsored_links"]
            else:
                video["brand_links"] = []
                video["has_keyword_in_title"] = False
                video["has_keyword_in_description"] = False
                video["has_sponsored_links"] = False

        text_analysis_time = time.time() - text_analysis_start
        print(
            f"Phase 4 complete: processed all text content in {text_analysis_time:.2f} seconds"
        )

        # Update the total time report to include brand links extraction phase
        total_time = time.time() - start_time
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"  - Search phase: {search_time:.2f}s")
        print(f"  - Video details phase: {details_time:.2f}s")
        print(f"  - Channel details phase: {channel_details_time:.2f}s")
        print(f"  - Brand links extraction: {text_analysis_time:.2f}s")

        # After all phases complete, store data in CSV
        try:
            # Convert results to pandas dataframe
            csv_start = time.time()

            # Normalize nested structures for CSV storage
            flattened_results = []
            for video in all_results:
                # Create a copy to avoid modifying original data
                flat_video = video.copy()

                # Convert nested thumbnails to just the high resolution URL
                if "thumbnails" in flat_video and "high" in flat_video["thumbnails"]:
                    flat_video["thumbnail_url"] = flat_video["thumbnails"]["high"].get(
                        "url", ""
                    )
                else:
                    flat_video["thumbnail_url"] = ""

                # Convert brand_links list to string
                if "brand_links" in flat_video and isinstance(
                    flat_video["brand_links"], list
                ):
                    flat_video["brand_links_str"] = "|".join(flat_video["brand_links"])
                else:
                    flat_video["brand_links_str"] = ""

                # Remove complex nested structures before CSV storage
                flat_video.pop("thumbnails", None)
                flat_video.pop("brand_links", None)

                flattened_results.append(flat_video)

            # Create dataframe and save to CSV
            df = pd.DataFrame(flattened_results)
            df.to_csv(csv_filename, index=False)
            print(
                f"✅ Saved {len(df)} videos to {csv_filename} in {time.time() - csv_start:.2f}s"
            )

            # Apply filters to the collected data
            filtered_results = apply_filters(df, payload)

            # Cache the filtered results
            try:
                serialized_data = json.dumps(
                    filtered_results, default=json_serializable
                )
                redis_client.setex(cache_key, 3600, serialized_data)
                print(f"✅ Cached {len(filtered_results)} filtered results")
            except Exception as e:
                print(f"Error caching filtered results: {str(e)}")

            return filtered_results

        except Exception as e:
            import traceback

            print(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error saving or filtering data: {str(e)}"
            )

    except Exception as e:
        import traceback

        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


# Function to apply filters using pandas
def apply_filters(df, payload):
    """Apply all filters from SearchRequest to the pandas DataFrame"""
    filter_start = time.time()
    print(f"Applying filters to {len(df)} videos...")

    # Create a copy to avoid modifying the original dataframe
    filtered_df = df.copy()

    # 0. PERMANENT FILTER: Remove videos shorter than 1 minute
    initial_count = len(filtered_df)
    # First ensure duration data exists (fill missing values with 0)
    if "durationSeconds" not in filtered_df.columns:
        filtered_df["durationSeconds"] = 0
    else:
        filtered_df["durationSeconds"] = filtered_df["durationSeconds"].fillna(0)

    filtered_df = filtered_df[
        filtered_df["durationSeconds"] >= 60
    ]  # 60 seconds = 1 minute
    removed_count = initial_count - len(filtered_df)
    print(f"Removed {removed_count} videos shorter than 1 minute")
    print(f"After minimum duration filter: {len(filtered_df)} videos")

    # 1. Apply minimum views filter
    if payload.min_views and payload.min_views > 0:
        filtered_df = filtered_df[filtered_df["viewCount"] >= payload.min_views]
        print(f"After min_views filter: {len(filtered_df)} videos")

    # 2. Apply date filter
    if payload.date_filter != DateFilterOption.ALL_TIME:
        # Convert publishTime to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(filtered_df["publishTime"]):
            filtered_df["publishTime"] = pd.to_datetime(filtered_df["publishTime"])

        # Make sure we use timezone-aware datetime for comparison
        import pytz

        # Create a timezone-aware "now"
        now = datetime.now(pytz.UTC)

        if (
            payload.date_filter == DateFilterOption.CUSTOM
            and payload.custom_date_from
            and payload.custom_date_to
        ):
            # Make custom dates timezone-aware if they aren't already
            from_date = payload.custom_date_from
            to_date = payload.custom_date_to

            if from_date.tzinfo is None:
                from_date = from_date.replace(tzinfo=pytz.UTC)
            if to_date.tzinfo is None:
                to_date = to_date.replace(tzinfo=pytz.UTC)

            filtered_df = filtered_df[
                (filtered_df["publishTime"] >= from_date)
                & (filtered_df["publishTime"] <= to_date)
            ]
        else:
            # Predefined time ranges
            time_deltas = {
                DateFilterOption.PAST_HOUR: timedelta(hours=1),
                DateFilterOption.PAST_3_HOURS: timedelta(hours=3),
                DateFilterOption.PAST_6_HOURS: timedelta(hours=6),
                DateFilterOption.PAST_12_HOURS: timedelta(hours=12),
                DateFilterOption.PAST_24_HOURS: timedelta(days=1),
                DateFilterOption.PAST_7_DAYS: timedelta(days=7),
                DateFilterOption.PAST_30_DAYS: timedelta(days=30),
                DateFilterOption.PAST_90_DAYS: timedelta(days=90),
                DateFilterOption.PAST_180_DAYS: timedelta(days=180),
            }

            if payload.date_filter in time_deltas:
                cutoff_date = now - time_deltas[payload.date_filter]
                filtered_df = filtered_df[filtered_df["publishTime"] >= cutoff_date]

        print(f"After date filter: {len(filtered_df)} videos")

    # 3. Apply country filter
    if payload.country_code:
        if "country" in filtered_df.columns:
            # Filter for videos from the specified country
            country_filter = filtered_df["country"] == payload.country_code
            filtered_df = filtered_df[country_filter]
            print(f"After country filter: {len(filtered_df)} videos")

    # 4. Apply excluded channels filter
    if payload.excluded_channels and len(payload.excluded_channels) > 0:
        # Convert to lowercase for case-insensitive comparison
        excluded_channels = [channel.lower() for channel in payload.excluded_channels]
        filtered_df = filtered_df[
            ~filtered_df["channelTitle"].str.lower().isin(excluded_channels)
        ]
        print(f"After excluded channels filter: {len(filtered_df)} videos")

    # 5. Apply keyword location filter
    if payload.keyword_filter != KeywordFilterOption.ANY:
        # Create keyword regex pattern for case-insensitive matching
        keywords = [
            kw.strip().lower() for kw in payload.brand_name.split(",") if kw.strip()
        ]

        if payload.keyword_filter == KeywordFilterOption.TITLE_ONLY:
            filtered_df = filtered_df[filtered_df["has_keyword_in_title"] == True]
        elif payload.keyword_filter == KeywordFilterOption.DESCRIPTION_ONLY:
            filtered_df = filtered_df[filtered_df["has_keyword_in_description"] == True]
        elif payload.keyword_filter == KeywordFilterOption.TITLE_AND_DESCRIPTION:
            filtered_df = filtered_df[
                (filtered_df["has_keyword_in_title"] == True)
                & (filtered_df["has_keyword_in_description"] == True)
            ]

        print(f"After keyword location filter: {len(filtered_df)} videos")

    # 6. Sort by relevancy score and view count
    # First ensure the relevancy_score column exists
    if "relevancy_score" not in filtered_df.columns:
        filtered_df["relevancy_score"] = 0

    # Sort by relevancy score (high to low) and then by view count (high to low)
    filtered_df = filtered_df.sort_values(
        ["relevancy_score", "viewCount"], ascending=[False, False]
    )

    # Convert back to dictionary format for the API response
    result_records = filtered_df.head(payload.max_results).to_dict("records")

    # Reconstruct nested structures with proper type checking
    for record in result_records:
        # Reconstruct thumbnails
        if "thumbnail_url" in record:
            record["thumbnails"] = {"high": {"url": record["thumbnail_url"]}}
            record.pop("thumbnail_url", None)

        # Reconstruct brand_links with proper type checking
        if "brand_links_str" in record:
            try:
                # Check if it's a string before splitting
                if (
                    isinstance(record["brand_links_str"], str)
                    and record["brand_links_str"]
                ):
                    record["brand_links"] = record["brand_links_str"].split("|")
                else:
                    record["brand_links"] = []
            except:
                # Fallback for any errors
                record["brand_links"] = []
            record.pop("brand_links_str", None)

    filter_time = time.time() - filter_start
    print(
        f"Filtering completed in {filter_time:.2f}s. Returning {len(result_records)} videos."
    )

    return result_records


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


@app.delete("/cache/{cache_key}")
async def delete_cache(cache_key: str):
    """
    Delete a specific cache entry from Redis.

    Args:
        cache_key: The cache key to delete (without the 'search:' prefix)

    Returns:
        JSON response with deletion status
    """
    try:
        # Add 'search:' prefix if not already present
        full_key = (
            cache_key if cache_key.startswith("search:") else f"search:{cache_key}"
        )

        # Check if key exists first
        exists = redis_client.exists(full_key)

        if exists:
            # Delete the key
            deleted = redis_client.delete(full_key)
            return {
                "success": True,
                "message": f"Cache entry '{full_key}' successfully deleted",
                "deleted_count": deleted,
            }
        else:
            return {
                "success": False,
                "message": f"Cache entry '{full_key}' not found in Redis",
            }
    except Exception as e:
        return {"success": False, "message": f"Error deleting cache entry: {str(e)}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
