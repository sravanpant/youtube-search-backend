
# async def search_videos(payload: SearchRequest):
#     start_time = time.time()

#     # Generate a cache key based on the exact search parameters
#     payload_dict = payload.model_dump()
#     if payload_dict.get("custom_date_from"):
#         payload_dict["custom_date_from"] = payload_dict["custom_date_from"].isoformat()
#     if payload_dict.get("custom_date_to"):
#         payload_dict["custom_date_to"] = payload_dict["custom_date_to"].isoformat()
#     payload_dict["keyword_filter_explicit"] = str(payload.keyword_filter)

#     # Create a consistent string representation and hash it
#     payload_str = json.dumps(payload_dict, sort_keys=True)
#     cache_key = f"search:{hashlib.md5(payload_str.encode()).hexdigest()}"
#     print(f"Search cache key: {cache_key}")

#     # Try to get complete results from cache
#     try:
#         cached_data = redis_client.get(cache_key)
#         if cached_data:
#             print(f"🎉 CACHE HIT! Returning cached results")
#             results_dict = json.loads(cached_data)
#             results = [VideoDetails(**item) for item in results_dict]
#             elapsed = time.time() - start_time
#             print(f"Cache hit response time: {elapsed:.4f} seconds")
#             return results
#         else:
#             print("Cache miss, performing search...")
#     except Exception as e:
#         print(f"Error checking cache: {str(e)}")

#     try:
#         # Clean and split keywords
#         keywords = [
#             keyword.strip().lower()
#             for keyword in payload.brand_name.split(",")
#             if keyword.strip()
#         ]

#         # Storage for all videos
#         all_videos = []
#         seen_video_ids = set()

#         # Target collection size - always aim for at least 500 videos
#         COLLECTION_TARGET = 500

#         # Enhanced search terms for each keyword
#         for keyword in keywords:
#             # Create comprehensive search queries
#             youtube_search_queries = [
#                 f'"{keyword}"',
#                 f'"{keyword}" review',
#                 f'"{keyword}" tutorial',
#                 f'"{keyword}" guide',
#                 f'"{keyword}" how to',
#                 f'"{keyword}" app',
#                 f'"{keyword}" demo',
#                 f'"{keyword}" explained',
#             ]

#             for query in youtube_search_queries:
#                 # Skip if we have enough videos
#                 if len(all_videos) >= COLLECTION_TARGET:
#                     print(f"✅ Collection target of {COLLECTION_TARGET} videos reached")
#                     break

#                 try:
#                     print(f"Searching for: {query}")

#                     # Search for videos with this query
#                     async def search_videos(youtube):
#                         return (
#                             youtube.search()
#                             .list(
#                                 q=query,
#                                 part="snippet",
#                                 maxResults=50,  # YouTube API max per request
#                                 type="video",
#                             )
#                             .execute()
#                         )

#                     search_response = await youtube_manager.execute_with_retry(
#                         search_videos, "search.list"
#                     )

#                     print(
#                         f"Found {len(search_response.get('items', []))} results for '{query}'"
#                     )

#                     # Process search results
#                     for item in search_response.get("items", []):
#                         video_id = item["id"].get("videoId")
#                         if video_id and video_id not in seen_video_ids:
#                             seen_video_ids.add(video_id)

#                             # Get full video details
#                             async def get_video_details(youtube):
#                                 return (
#                                     youtube.videos()
#                                     .list(
#                                         part="snippet,contentDetails,statistics",
#                                         id=video_id,
#                                     )
#                                     .execute()
#                                 )

#                             video_response = await youtube_manager.execute_with_retry(
#                                 get_video_details, "videos.list"
#                             )

#                             if not video_response.get("items"):
#                                 continue

#                             video_data = video_response["items"][0]

#                             # Get channel details for subscriber count
#                             channel_id = video_data["snippet"]["channelId"]

#                             async def get_channel_stats(youtube):
#                                 return (
#                                     youtube.channels()
#                                     .list(part="statistics", id=channel_id)
#                                     .execute()
#                                 )

#                             channel_response = await youtube_manager.execute_with_retry(
#                                 get_channel_stats, "channels.list"
#                             )

#                             channel_stats = channel_response.get("items", [{}])[0].get(
#                                 "statistics", {}
#                             )

#                             # Extract information
#                             description = video_data["snippet"].get("description", "")
#                             title = video_data["snippet"].get("title", "")

#                             # Check for keyword presence
#                             title_lower = title.lower()
#                             description_lower = description.lower()

#                             import re

#                             has_keyword_in_title = any(
#                                 re.search(r"\b" + re.escape(kw) + r"\b", title_lower)
#                                 for kw in keywords
#                             )
#                             has_keyword_in_description = any(
#                                 re.search(
#                                     r"\b" + re.escape(kw) + r"\b", description_lower
#                                 )
#                                 for kw in keywords
#                             )

#                             # Extract brand links
#                             brand_links = extract_brand_related_urls(
#                                 description, keywords
#                             )

#                             # Create video details
#                             video_details = VideoDetails(
#                                 videoId=video_id,
#                                 title=title,
#                                 channelTitle=video_data["snippet"].get(
#                                     "channelTitle", ""
#                                 ),
#                                 channelId=channel_id,
#                                 publishTime=video_data["snippet"].get(
#                                     "publishedAt", ""
#                                 ),
#                                 viewCount=int(
#                                     video_data["statistics"].get("viewCount", 0)
#                                 ),
#                                 likeCount=int(
#                                     video_data["statistics"].get("likeCount", 0)
#                                 ),
#                                 commentCount=int(
#                                     video_data["statistics"].get("commentCount", 0)
#                                 ),
#                                 subscriberCount=int(
#                                     channel_stats.get("subscriberCount", 0)
#                                 ),
#                                 duration=video_data["contentDetails"].get(
#                                     "duration", ""
#                                 ),
#                                 description=description,
#                                 thumbnails=video_data["snippet"].get("thumbnails", {}),
#                                 videoLink=f"https://www.youtube.com/watch?v={video_id}",
#                                 channelLink=f"https://www.youtube.com/channel/{channel_id}",
#                                 relevancy_score=calculate_relevancy_score(
#                                     video_data, keywords
#                                 ),
#                                 brand_links=brand_links,
#                                 country=video_data.get("snippet", {}).get("country"),
#                                 has_keyword_in_title=has_keyword_in_title,
#                                 has_keyword_in_description=has_keyword_in_description,
#                             )

#                             # Store the video
#                             all_videos.append(video_details)

#                             # Cache this video for future use
#                             cache.store_video(
#                                 video_details, expire=7 * 24 * 60 * 60
#                             )  # 1 week cache

#                             # Print progress every 50 videos
#                             if len(all_videos) % 50 == 0:
#                                 print(f"Collected {len(all_videos)} videos so far...")

#                             # Check if we've reached the collection target
#                             if len(all_videos) >= COLLECTION_TARGET:
#                                 break

#                 except Exception as e:
#                     print(f"Error searching for query '{query}': {str(e)}")

#         # Save all videos to a JSON file
#         try:
#             videos_json = [v.dict() for v in all_videos]
#             timestamp = int(time.time())
#             keywords_slug = "_".join(keywords)[:50]  # Limit length of filename
#             filename = f"search_{keywords_slug}_{timestamp}.json"
#             with open(filename, "w") as f:
#                 json.dump(videos_json, f, indent=2)
#             print(f"Saved {len(all_videos)} videos to {filename}")
#         except Exception as e:
#             print(f"Error saving videos to JSON: {str(e)}")

#         # Sort by relevancy score (for better presentation)
#         all_videos.sort(key=lambda x: x.relevancy_score, reverse=True)

#         # Take only the requested number for API response
#         final_results = all_videos[: payload.max_results] if all_videos else []

#         # Cache the final results (store all videos in the cache)
#         try:
#             results_dict = [result.dict() for result in all_videos]  # Cache ALL videos
#             serialized_data = json.dumps(results_dict)
#             redis_client.setex(cache_key, 3600, serialized_data)  # 1 hour expiration
#             print(f"✅ Cached {len(all_videos)} videos with key: {cache_key}")
#         except Exception as e:
#             print(f"Error caching results: {str(e)}")

#         # Log performance
#         elapsed = time.time() - start_time
#         print(f"Total search time: {elapsed:.2f} seconds")
#         print(
#             f"Collected {len(all_videos)} total videos, returning {len(final_results)} as requested"
#         )

#         return final_results

#     except Exception as e:
#         import traceback

#         print(traceback.format_exc())
#         raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")