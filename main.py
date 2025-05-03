from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from googleapiclient.discovery import build
from googleapiclient.discovery import build as google_build
from googleapiclient.errors import HttpError
import os
from dotenv import load_dotenv
import asyncio
import isodate
from typing import List
from models import SearchRequest, VideoDetails
from utils import (
    extract_video_id_from_url,
    calculate_relevancy_score,
    extract_brand_related_urls,
)
from typing import Optional
import random

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
    allow_origins=["*"],  # Adjust this to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def process_video(
    video_id: str, keywords: List[str], all_videos: List[VideoDetails], min_views: int
) -> None:
    """Process a single video and add it to all_videos if relevant."""
    try:
        # Get video details using the manager
        async def get_video_details(youtube):
            return (
                youtube.videos()
                .list(part="statistics,contentDetails,snippet", id=video_id)
                .execute()
            )

        video_response = await youtube_manager.execute_with_retry(get_video_details)

        if not video_response.get("items"):
            return

        video_data = video_response["items"][0]
        # Calculate relevancy score
        relevancy_score = calculate_relevancy_score(video_data, keywords)

        if relevancy_score < 3:  # Minimum relevancy threshold
            return

        # Check minimum views
        view_count = int(video_data["statistics"].get("viewCount", 0))
        if view_count < min_views:
            return

        # Get channel details
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
        )

        all_videos.append(video_details)

    except Exception as e:
        print(f"Error processing video {video_id}: {str(e)}")


@app.post("/search", response_model=List[VideoDetails])
async def search_videos(payload: SearchRequest):
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
                            return (
                                youtube.search()
                                .list(
                                    q=query,
                                    part="id,snippet",
                                    maxResults=50,
                                    type="video",
                                    safeSearch="none",
                                    relevanceLanguage="en",
                                )
                                .execute()
                            )

                        search_response = await youtube_manager.execute_with_retry(
                            search_videos
                        )

                        for item in search_response.get("items", []):
                            video_id = item["id"].get("videoId")
                            if video_id and video_id not in seen_video_ids:
                                seen_video_ids.add(video_id)
                                tasks.append(
                                    process_video(
                                        video_id,
                                        keywords,
                                        all_videos,
                                        payload.min_views,
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
                                            tasks.append(
                                                process_video(
                                                    video_id,
                                                    keywords,
                                                    all_videos,
                                                    payload.min_views,
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
        return all_videos[: payload.max_results] if all_videos else []

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
