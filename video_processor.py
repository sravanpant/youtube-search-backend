# backend/video_processor.py

from typing import List, Optional
from datetime import datetime, timezone
from models import VideoDetails
from utils import extract_brand_related_urls, calculate_relevancy_score


def parse_duration(duration_str):
    """Convert ISO 8601 duration string to seconds"""
    import re
    import isodate  # You may need to add this to requirements.txt

    try:
        # Use isodate if available
        return int(isodate.parse_duration(duration_str).total_seconds())
    except (ImportError, ValueError):
        # Fallback to manual parsing
        hours = 0
        minutes = 0
        seconds = 0

        # Extract hours, minutes, seconds using regex
        hour_match = re.search(r"(\d+)H", duration_str)
        if hour_match:
            hours = int(hour_match.group(1))

        minute_match = re.search(r"(\d+)M", duration_str)
        if minute_match:
            minutes = int(minute_match.group(1))

        second_match = re.search(r"(\d+)S", duration_str)
        if second_match:
            seconds = int(second_match.group(1))

        return hours * 3600 + minutes * 60 + seconds


async def process_video(
    video_id: str,
    keywords: List[str],
    all_videos: List[VideoDetails],
    min_views: int,
    youtube_manager,
    cache,
    publish_after: Optional[datetime] = None,
    publish_before: Optional[datetime] = None,
    country_code: Optional[str] = None,
    excluded_channels: Optional[List[str]] = None,
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

                # Check if excluded_channels parameter is provided and if this channel should be excluded
                if excluded_channels and any(
                    excluded.lower() in cached_video.channelTitle.lower()
                    for excluded in excluded_channels
                ):
                    return

                # Check date filters if applicable
                if publish_after or publish_before:
                    try:
                        from datetime import timezone

                        # Parse the cached video's publish time
                        publish_time_str = cached_video.publishTime
                        if publish_time_str:
                            # Normalize the format by removing 'Z' and adding timezone info
                            if publish_time_str.endswith("Z"):
                                publish_time_str = publish_time_str[:-1] + "+00:00"

                            # Parse to datetime object
                            publish_time = datetime.fromisoformat(publish_time_str)

                            # Make dates timezone aware
                            if publish_after and publish_after.tzinfo is None:
                                publish_after = publish_after.replace(
                                    tzinfo=timezone.utc
                                )

                            if publish_before and publish_before.tzinfo is None:
                                publish_before = publish_before.replace(
                                    tzinfo=timezone.utc
                                )

                            # Compare dates
                            if publish_after and publish_time < publish_after:
                                return

                            if publish_before and publish_time > publish_before:
                                return
                    except Exception as e:
                        print(f"Error parsing cached publish date: {str(e)}")

                # Update keyword presence flags for cached videos
                title_lower = cached_video.title.lower()
                description_lower = cached_video.description.lower()

                import re

                cached_video.has_keyword_in_title = any(
                    re.search(r"\b" + re.escape(keyword) + r"\b", title_lower)
                    for keyword in keywords
                )
                cached_video.has_keyword_in_description = any(
                    re.search(r"\b" + re.escape(keyword) + r"\b", description_lower)
                    for keyword in keywords
                )

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

                    # Normalize the format by removing 'Z' and adding timezone info
                    if publish_time_str.endswith("Z"):
                        publish_time_str = publish_time_str[:-1] + "+00:00"

                    # Parse the video's publish time
                    publish_time = datetime.fromisoformat(publish_time_str)

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

        # Check if channel is in the excluded list (case-insensitive)
        if excluded_channels and any(
            excluded.lower() in channel_title.lower() for excluded in excluded_channels
        ):
            print(f"Skipping video from excluded channel: {channel_title}")
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

        try:
            duration = video_data["contentDetails"][
                "duration"
            ]  # Keep the ISO 8601 format

            # Check if video is too short (less than 60 seconds)
            duration_seconds = parse_duration(duration)
            if duration_seconds < 60:
                # print(
                #     f"Skipping video {video_id} - duration too short: {duration_seconds}s"
                # )
                return
        except:
            duration = "PT0S"  # Default duration if not available

        title_lower = video_data["snippet"].get("title", "").lower()
        description_lower = description.lower()

        import re

        has_keyword_in_title = any(
            re.search(r"\b" + re.escape(keyword) + r"\b", title_lower)
            for keyword in keywords
        )
        has_keyword_in_description = any(
            re.search(r"\b" + re.escape(keyword) + r"\b", description_lower)
            for keyword in keywords
        )

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
            country=video_country,
            has_keyword_in_title=has_keyword_in_title,  # Add the new field
            has_keyword_in_description=has_keyword_in_description,  # Add the new field
        )

        cache.store_video(video_details)
        all_videos.append(video_details)

    except Exception as e:
        print(f"Error processing video {video_id}: {str(e)}")
