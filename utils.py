# backend/utils.py

from typing import List, Optional
from urllib.parse import urlparse, parse_qs
import re

def extract_video_id_from_url(url: str) -> Optional[str]:
    """Extract YouTube video ID from various YouTube URL formats."""
    try:
        parsed_url = urlparse(url)
        if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
            if parsed_url.path == '/watch':
                return parse_qs(parsed_url.query)['v'][0]
            elif parsed_url.path.startswith('/shorts/'):
                return parsed_url.path.split('/')[2]
        elif parsed_url.hostname == 'youtu.be':
            return parsed_url.path[1:]
    except:
        return None
    return None

def calculate_relevancy_score(video_data: dict, keywords: List[str]) -> int:
    """Calculate relevancy score for a video."""
    score = 0
    title = video_data["snippet"].get("title", "").lower()
    description = video_data["snippet"].get("description", "").lower()

    for keyword in keywords:
        # Title matches
        if f'"{keyword}"' in f'"{title}"':  # Exact phrase match
            score += 10
        elif all(word in title for word in keyword.split()):  # All words present
            score += 5
        elif keyword in title:  # Partial match
            score += 3

        # Description matches
        if f'"{keyword}"' in f'"{description}"':
            score += 5
        elif all(word in description for word in keyword.split()):
            score += 3
        elif keyword in description:
            score += 1

    # Additional relevancy factors
    relevant_terms = ['app', 'download', 'tutorial', 'guide', 'review', 'mobile', 'official']
    for term in relevant_terms:
        if term in title or term in description:
            score += 1

    return score

def extract_brand_related_urls(description: str, keywords: List[str]) -> List[str]:
    """Extract URLs from description that might be related to any of the brand keywords."""
    # Clean the description first
    description = description.replace('\n', ' ').replace('\r', ' ')
    
    # Common domains to exclude
    excluded_domains = {
        'facebook.com', 'fb.com', 'instagram.com', 'twitter.com', 'youtube.com',
        'tiktok.com', 'linkedin.com', 'pinterest.com', 'reddit.com', 't.me',
        'telegram.me', 'whatsapp.com', 'snapchat.com', 'forms.gle', 'bit.ly',
        'youtu.be', 'discord.gg', 'discord.com', 'play.google.com', 'apple.co',
        'spf.bio'
    }

    # Regex to find URLs in text - improved pattern
    url_pattern = re.compile(
        r'https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\$$\$$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )

    # Find all URLs in cleaned description
    urls = url_pattern.findall(description)
    
    # Process brand keywords
    processed_keywords = set()
    for keyword in keywords:
        # Clean each keyword
        clean_keyword = keyword.lower().strip()
        # Add both original and no-space versions
        processed_keywords.add(clean_keyword)
        processed_keywords.add(clean_keyword.replace(" ", ""))
        # Add variations
        if "sahi" in clean_keyword:
            processed_keywords.update(["sahi", "sahitrade", "sahiapp"])

    brand_related_urls = []
    for url in urls:
        try:
            # Clean the URL
            url = url.strip().rstrip('.,/)]}').split(' ')[0]
            if not url.startswith('http'):
                continue
                
            # Parse the URL
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]

            # Skip excluded domains
            if domain in excluded_domains:
                continue

            # Create a clean version of the URL for matching
            url_text = f"{domain} {parsed_url.path}".lower()

            # Special handling for app download links
            if any(d in domain for d in ['onelink.me', 'page.link']):
                for keyword in processed_keywords:
                    if keyword in url_text:
                        brand_related_urls.append(url)
                        break
                continue

            # Check if URL is brand-related
            is_brand_link = False
            for keyword in processed_keywords:
                # Check domain and path
                if keyword in domain or keyword in parsed_url.path.lower():
                    is_brand_link = True
                    break
                # Check full URL text
                if keyword in url_text:
                    is_brand_link = True
                    break

            if is_brand_link:
                brand_related_urls.append(url)

        except Exception as e:
            print(f"Error processing URL {url}: {str(e)}")
            continue

    return list(set(brand_related_urls))  # Remove duplicates