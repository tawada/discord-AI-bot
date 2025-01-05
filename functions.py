import re

url_pattern_raw = (
    r'https?://'          # http:// or https://
    r'(?:[a-zA-Z0-9$-_@.&+]|[!*()\']|%[0-9a-fA-F]{2})+'
)
url_pattern = re.compile(url_pattern_raw)


def contains_url(text_including_url: str) -> bool:
    """Uses regex to check if a string has a URL."""
    return bool(re.search(url_pattern, text_including_url))


def extract_urls(text_including_url: str) -> list:
    """Uses regex to extract URLs from a string."""
    found_urls = re.findall(url_pattern, text_including_url)
    # 末尾にくっついている可能性がある句読点などを除去
    cleaned_urls = [url.rstrip(',.!?') for url in found_urls]
    return cleaned_urls


def contains_knowledge_request_keywords(text: str) -> bool:
    """Checks if a string contains any of the
       keywords for knowledge requests.
    """
    keywords = ["教えて", "誰", "何", "調べて"]
    return any(keyword in text for keyword in keywords)


def is_image_attachment(attachment) -> bool:
    """Checks if an attachment is an image."""
    extensions = (".png", ".jpg", ".jpeg")
    return attachment.content_type.startswith("image") and (
        attachment.filename.endswith(extensions)
        or attachment.url.endswith(extensions)
    )
