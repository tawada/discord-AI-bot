import logging

import requests
from bs4 import BeautifulSoup

import functions

logger = logging.getLogger(__name__)

text_model = "gemini-2.0-flash"
image_model = "gpt-4o"


def summarize_webpage(url, ai_client):
    """Gemini クライアントでウェブページを要約。YouTube や X (twitter) は別関数へ。"""
    if functions.is_youtube_link(url):
        return summarize_youtube(url)

    if functions.is_twitter_link(url):
        return summarize_x(url, ai_client)

    return summarize_normal_website(url, ai_client)


def summarize_normal_website(url: str, ai_client):
    """
    通常サイトの要約フロー。HTML を取得し、テキスト抽出し、Gemini で要約。
    """
    response = fetch_html(url)
    logger.debug(response.status_code)
    logger.debug(response.text[:50])

    combined_text = extract_text_from_html(response.text)

    # テキストが空の場合はエラーメッセージを返す
    if not combined_text.strip():
        return "検索結果から有用な情報を取得できませんでした。"

    logger.info(f"Extracted text for summarization: {combined_text[:100]}...")

    # AI (Gemini) による要約
    return summarize_with_gemini(combined_text, ai_client)


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
                model=image_model,
                messages=messages,
            )
            .choices[0]
            .message.content
        )
    except RuntimeError as err:
        logger.exception(err)
        raise
    return summarized_text


def fetch_html(url: str):
    """HTTP GET で HTML を取得して返す。"""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 "
            "Mobile/15E148 Safari/604.1"
        )
    }
    return requests.get(url, headers=headers)


def extract_text_from_html(html: str) -> str:
    """
    BeautifulSoup で HTML をパースし、タイトルや見出し、段落などからテキストを抽出する。
    """
    soup = BeautifulSoup(html, "html.parser")

    # 必要に応じて抽出するタグを列挙
    content_tags = ['title', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
    extracted_texts = []

    for tag in content_tags:
        extracted_texts.extend(element.get_text() for element in soup.find_all(tag))

    # タグからの抽出が空の場合は、ページ全体のテキストをフォールバック取得
    if not any(t.strip() for t in extracted_texts):
        fallback_text = soup.get_text().strip()
        if fallback_text:
            extracted_texts = [fallback_text]

    combined_text = " ".join(extracted_texts)

    # 長すぎる場合に切り詰める (Gemini 側で処理制限があるなら)
    max_length = 4096
    if len(combined_text) > max_length:
        combined_text = combined_text[:max_length]

    return combined_text


def summarize_with_gemini(text: str, ai_client):
    """
    Gemini でテキストを要約する。
    """
    messages = [
        {"role": "user", "content": text},
        {"role": "system", "content": "Please summarize the webpage."},
    ]
    try:
        result = ai_client.chat.completions.create(
            model=text_model,
            messages=messages,
        )
        return result.choices[0].message.content
    except RuntimeError as err:
        logger.exception(err)
        raise
