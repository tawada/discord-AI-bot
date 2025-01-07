import logging

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def summarize_webpage(url, ai_client):
    """Gemini クライアントでウェブページを要約。YouTube や X (twitter) は別関数へ。"""
    if "youtube.com" in url or "youtu.be" in url:
        return summarize_youtube(url)

    if "x.com" in url or "twitter.com" in url:
        return summarize_x(url, ai_client)

    # 通常のウェブページ
    response = requests.get(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 "
                "Mobile/15E148 Safari/604.1"
            )
        }
    )
    logger.debug(response.status_code)
    logger.debug(response.text[:50])

    soup = BeautifulSoup(response.text, "html.parser")

    # Collect text from certain tags
    content_tags = ['title', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
    extracted_text = []

    for tag in content_tags:
        extracted_text.extend(element.get_text() for element in soup.find_all(tag))

    # もしも上記タグが一つも見つからなかった場合、body 全体などをフォールバックで取得
    if not extracted_text or not any(t.strip() for t in extracted_text):
        fallback_text = soup.get_text().strip()
        if fallback_text:
            extracted_text = [fallback_text]

    combined_text = " ".join(extracted_text)

    # Trim to a reasonable length for processing, with slight buffer for word completion
    max_length = 4096
    if len(combined_text) > max_length:
        combined_text = combined_text[:max_length]

    if not combined_text.strip():
        return "検索結果から有用な情報を取得できませんでした。"

    logger.info(f"Extracted text for summarization: {combined_text[:100]}...")

    # Geminiで要約（テキストが長いとエラーになりやすいので切り詰め）
    messages = [
        {"role": "user", "content": combined_text},
        {"role": "system", "content": "Please summarize the webpage."},
    ]
    try:
        summarized_text = (
            ai_client.chat.completions.create(
                model="gemini-1.5-flash",
                messages=messages,
            )
            .choices[0]
            .message.content
        )
    except RuntimeError as err:
        logger.exception(err)
        raise
    return summarized_text


def summarize_youtube(url):
    """YouTube のページを簡易要約"""
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    logger.debug(response.status_code)
    logger.debug(response.text)
    html = response.text
    logger.debug(html[:50])

    idx_s = html.find("<title>")
    idx_e = html.find("</title>")
    if idx_s == -1 or idx_e == -1:
        return "Unknown Title"
    title = html[idx_s + 7: idx_e]
    logger.debug(f"Title: {title}")
    return "This youtube video is " + title


def summarize_x(url, ai_client):
    """X (旧 Twitter) のページをテキストとして簡易要約 (Gemini 使用)"""
    response = requests.get(url, headers={"User-Agent": "bot"})
    logger.debug(response.status_code)
    logger.debug(response.text)
    html = response.text
    logger.debug(html[:50])

    soup = BeautifulSoup(html, "html.parser")
    meta_tags = soup.find_all("meta")
    # 元のコードの意図をくみ取って「メタタグを処理」する
    return summarize_from_meta_tags(meta_tags, ai_client)

def summarize_from_meta_tags(meta_tags, ai_client):
    """
    メタタグを処理して要約を作成する。
      - og:image → summarize_image(...) で画像要約 (OpenAI)
      - og:title, og:description → そのまま文字列としてまとめる
      - 最後に "This is a summary of the webpage: ..." と連結して返す
    """
    property_handlers = {
        "og:image": lambda meta: 
            summarize_image(meta.get("content"), ai_client),
        "og:title": lambda meta: str(meta),
        "og:description": lambda meta: str(meta),
    }

    summarized_text = ""
    for meta in meta_tags:
        logger.debug(meta)
        # property が無い or property_handlers に無い場合はスキップ
        prop = meta.get("property")
        if not prop:
            continue
        handler = property_handlers.get(prop)
        if handler:
            summarized_text += handler(meta) + "\n"

    return "This is a summary of the webpage: " + summarized_text


def summarize_image(url, ai_client):
    """画像認識は Gemini では不可なので OpenAI 側 (gpt-4o) を使用"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Summarize the image."},
                {"type": "image_url", "image_url": {"url": url}},
            ]
        },
    ]
    try:
        summarized_text = (
            ai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
            )
            .choices[0]
            .message.content
        )
    except RuntimeError as err:
        logger.exception(err)
        raise
    return summarized_text
