import re

url_pattern_raw = (
    r"https?://"  # http:// or https://
    r"(?:[a-zA-Z0-9$-_@.&+]|[!*()\']|%[0-9a-fA-F]{2})+"
)
url_pattern = re.compile(url_pattern_raw)
brackets_pattern = re.compile(r"\(.*?\)|（.*?）", flags=re.DOTALL)


def contains_url(text_including_url: str) -> bool:
    """Uses regex to check if a string has a URL."""
    return bool(re.search(url_pattern, text_including_url))


def extract_urls(text_including_url: str) -> list:
    """Uses regex to extract URLs from a string."""
    found_urls = re.findall(url_pattern, text_including_url)
    # 末尾にくっついている可能性がある句読点などを除去
    cleaned_urls = [url.rstrip(",.!?") for url in found_urls]
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
        attachment.filename.endswith(extensions) or attachment.url.endswith(extensions)
    )


def is_youtube_link(text: str) -> bool:
    """Checks if a string is a YouTube link."""
    return "youtube.com" in text or "youtu.be" in text


def is_twitter_link(text: str) -> bool:
    """Checks if a string is a Twitter link."""
    return "x.com" in text or "twitter.com" in text


def remove_brackets_and_spaces(text: str) -> str:
    """Removes brackets and spaces from a string."""

    # 丸括弧と大括弧（全角）の中にある文字も含めて削除する（改行も含めて非貪欲マッチ）
    text_no_brackets = re.sub(brackets_pattern, "", text)

    # 各行の前後空白を削除したうえで、空行は削除する
    lines = [line.strip() for line in text_no_brackets.split("\n")]
    lines = [line for line in lines if line]  # 空行の削除

    # 改行で再度結合する
    removed_text = "\n".join(lines)

    # もし空ならば、1個目の括弧を残す
    if not removed_text:
        removed_text = re.sub(brackets_pattern, "", text, count=1)

    return removed_text


def limit_sentences(text: str, max_sentences: int) -> str:
    """2文以内・100文字以内に収める

    100文字で単純に打ち切るのではなく改行で区切る
    """
    sentences = text.split("\n")
    if len(sentences) > max_sentences:
        text = "\n".join(sentences[:max_sentences])

    # 100文字以内に収める．ただし，改行で区切る
    if len(text) > 100:
        text = text[:100]
        # 最後の改行までの文字数を取得
        last_newline_index = text.rfind("\n")
        text = text[:last_newline_index]

    return text
