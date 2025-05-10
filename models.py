# backend/models.py

from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime
from enum import Enum

class KeywordFilterOption(str, Enum):
    ANY = "any"  # Keywords can be anywhere (default)
    TITLE_ONLY = "title_only"  # Keywords must be in title
    DESCRIPTION_ONLY = "description_only"  # Keywords must be in description
    TITLE_AND_DESCRIPTION = "title_and_description"  # Keywords must be in both


class DateFilterOption(str, Enum):
    PAST_HOUR = "past_hour"
    PAST_3_HOURS = "past_3_hours" 
    PAST_6_HOURS = "past_6_hours"
    PAST_12_HOURS = "past_12_hours"
    PAST_24_HOURS = "past_24_hours"
    PAST_7_DAYS = "past_7_days"
    PAST_30_DAYS = "past_30_days"
    PAST_90_DAYS = "past_90_days"
    PAST_180_DAYS = "past_180_days"
    CUSTOM = "custom"
    ALL_TIME = "all_time"

class SearchRequest(BaseModel):
    brand_name: str
    max_results: int = 30
    min_views: Optional[int] = 1000
    date_filter: DateFilterOption = DateFilterOption.ALL_TIME
    custom_date_from: Optional[datetime] = None
    custom_date_to: Optional[datetime] = None
    country_code: Optional[str] = None  # ISO 3166-1 alpha-2 country code (e.g., "US", "GB")
    excluded_channels: Optional[List[str]] = []  # List of channel names to exclude from results
    keyword_filter: KeywordFilterOption = KeywordFilterOption.ANY  # New field for keyword location filtering

class VideoDetails(BaseModel):
    videoId: str
    title: str
    description: str
    channelTitle: str
    channelId: str
    publishTime: str
    viewCount: Optional[int] = 0
    likeCount: Optional[int] = 0
    subscriberCount: Optional[int] = 0
    duration: Optional[str] = None
    thumbnails: Dict[str, Any]
    videoLink: str
    channelLink: Optional[str]
    relevancy_score: int = 0
    brand_links: List[str] = []
    country: Optional[str] = None  
    has_keyword_in_title: bool = False  # New field: indicates if keyword is in title
    has_keyword_in_description: bool = False  # New field: indicates if keyword is in description
    has_sponsored_links: bool = False