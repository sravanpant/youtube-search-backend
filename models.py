from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class SearchRequest(BaseModel):
    brand_name: str
    max_results: int = 30
    min_views: Optional[int] = 1000

class VideoDetails(BaseModel):
    videoId: str
    title: str
    channelTitle: str
    channelId: str
    publishTime: str
    viewCount: Optional[int] = 0
    likeCount: Optional[int] = 0
    commentCount: Optional[int] = 0
    subscriberCount: Optional[int] = 0
    duration: Optional[str] = None
    description: str
    thumbnails: Dict[str, Any]
    videoLink: str
    channelLink: Optional[str]
    relevancy_score: int = 0
    brand_links: List[str] = []