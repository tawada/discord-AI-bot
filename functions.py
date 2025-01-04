import re


def has_url(text_including_url: str) -> bool:
    """Uses regex to check if a string has a URL."""
    return bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text_including_url))


def get_urls(text_including_url: str) -> list:
    """Uses regex to extract URLs from a string."""
    return re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text_including_url)


def is_knowledge_request(text: str) -> bool:
    """Checks if a string contains any of the keywords for knowledge requests."""
    keywords = ["教えて", "誰", "何", "調べて"]
    return any(keyword in text for keyword in keywords)


def is_image_attachment(attachment) -> bool:
    """Checks if an attachment is an image."""
    extensions = (".png", ".jpg", ".jpeg")
    return attachment.content_type.startswith("image") and (
        attachment.filename.endswith(extensions) or attachment.url.endswith(extensions)
    )
