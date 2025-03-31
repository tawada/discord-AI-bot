import requests
from bs4 import BeautifulSoup
from loguru import logger
from typing import Dict, Any, List, Optional

import functions

# グローバル変数の代わりに定数として定義
DEFAULT_TEXT_MODEL = "gemini-2.0-flash"
DEFAULT_IMAGE_MODEL = "gpt-4o"
MAX_TEXT_LENGTH = 4096
USER_AGENT = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Mobile/15E148 Safari/604.1"


def summarize_webpage(url: str, ai_client: Any, text_model: str = DEFAULT_TEXT_MODEL) -> str:
    """ウェブページを要約。URLに応じて適切な要約方法を選択。

    Args:
        url: 要約するウェブページのURL
        ai_client: AI APIクライアント
        text_model: テキスト要約に使用するモデル名

    Returns:
        str: 要約されたテキスト
    """
    if functions.is_youtube_link(url):
        return summarize_youtube(url)

    if functions.is_twitter_link(url):
        return summarize_x(url, ai_client)

    return summarize_normal_website(url, ai_client, text_model)


def summarize_normal_website(url: str, ai_client: Any, text_model: str = DEFAULT_TEXT_MODEL) -> str:
    """通常サイトの要約フロー。HTML を取得し、テキスト抽出し、AI で要約。

    Args:
        url: 要約するウェブページのURL
        ai_client: AI APIクライアント
        text_model: テキスト要約に使用するモデル名

    Returns:
        str: 要約されたテキスト
    """
    try:
        response = fetch_html(url)
        logger.debug(f"Status code: {response.status_code}")
        logger.debug(f"Response prefix: {response.text[:50]}")

        combined_text = extract_text_from_html(response.text)

        # テキストが空の場合はエラーメッセージを返す
        if not combined_text.strip():
            return "ウェブページから有用な情報を取得できませんでした。"

        logger.info(f"Extracted text for summarization: {combined_text[:100]}...")

        # AI による要約
        return summarize_with_ai(combined_text, ai_client, text_model)
    except Exception as e:
        logger.exception(f"Error summarizing website {url}: {e}")
        return f"ウェブページの要約中にエラーが発生しました: {str(e)}"


def summarize_youtube(url: str) -> str:
    """YouTube のページを簡易要約

    Args:
        url: YouTubeのURL

    Returns:
        str: 動画タイトル
    """
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT})
        logger.debug(f"Status code: {response.status_code}")
        html = response.text
        logger.debug(f"HTML prefix: {html[:50]}")

        idx_s = html.find("<title>")
        idx_e = html.find("</title>")
        if idx_s == -1 or idx_e == -1:
            return "YouTubeのタイトルを取得できませんでした。"
        title = html[idx_s + 7 : idx_e]
        logger.debug(f"Title: {title}")
        return f"YouTube動画: {title}"
    except Exception as e:
        logger.exception(f"Error summarizing YouTube {url}: {e}")
        return "YouTube動画の情報取得中にエラーが発生しました。"


def summarize_x(url: str, ai_client: Any) -> str:
    """X (旧 Twitter) のページをテキストとして簡易要約

    Args:
        url: XのURL
        ai_client: AI APIクライアント

    Returns:
        str: X投稿の要約
    """
    try:
        response = requests.get(url, headers={"User-Agent": "bot"})
        logger.debug(f"Status code: {response.status_code}")
        html = response.text
        logger.debug(f"HTML prefix: {html[:50]}")

        soup = BeautifulSoup(html, "html.parser")
        meta_tags = soup.find_all("meta")
        return summarize_from_meta_tags(meta_tags, ai_client)
    except Exception as e:
        logger.exception(f"Error summarizing X post {url}: {e}")
        return "X投稿の情報取得中にエラーが発生しました。"


def summarize_from_meta_tags(meta_tags: List, ai_client: Any) -> str:
    """メタタグを処理して要約を作成する

    Args:
        meta_tags: BeautifulSoupで抽出されたmetaタグリスト
        ai_client: AI APIクライアント

    Returns:
        str: メタ情報から抽出した要約
    """
    property_handlers = {
        "og:image": lambda meta: summarize_image(meta.get("content"), ai_client),
        "og:title": lambda meta: f"タイトル: {meta.get('content', '')}",
        "og:description": lambda meta: f"説明: {meta.get('content', '')}",
    }

    summarized_parts = []
    for meta in meta_tags:
        logger.debug(meta)
        prop = meta.get("property")
        if not prop:
            continue
        handler = property_handlers.get(prop)
        if handler:
            summarized_parts.append(handler(meta))

    if not summarized_parts:
        return "X投稿からメタ情報を取得できませんでした。"
    
    return "X投稿の要約:\n" + "\n".join(summarized_parts)


def summarize_image(url: str, ai_client: Any, image_model: str = DEFAULT_IMAGE_MODEL) -> str:
    """画像を要約する

    Args:
        url: 画像のURL
        ai_client: AI APIクライアント
        image_model: 画像分析に使用するモデル名

    Returns:
        str: 画像の要約
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "画像を日本語で要約してください。"},
                {"type": "image_url", "image_url": {"url": url}},
            ],
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
        return summarized_text
    except Exception as err:
        logger.exception(f"Error summarizing image {url}: {err}")
        raise RuntimeError(f"画像の要約中にエラーが発生しました: {str(err)}")


def fetch_html(url: str) -> requests.Response:
    """HTTP GET で HTML を取得して返す

    Args:
        url: 取得するURL

    Returns:
        requests.Response: HTTPレスポンス
    """
    headers = {"User-Agent": USER_AGENT}
    return requests.get(url, headers=headers)


def extract_text_from_html(html: str) -> str:
    """HTMLからテキストを抽出する

    Args:
        html: 抽出元のHTML文字列

    Returns:
        str: 抽出されたテキスト
    """
    soup = BeautifulSoup(html, "html.parser")

    # 必要に応じて抽出するタグを列挙
    content_tags = ["title", "p", "h1", "h2", "h3", "h4", "h5", "h6"]
    extracted_texts = []

    for tag in content_tags:
        extracted_texts.extend(element.get_text() for element in soup.find_all(tag))

    # タグからの抽出が空の場合は、ページ全体のテキストをフォールバック取得
    if not any(t.strip() for t in extracted_texts):
        fallback_text = soup.get_text().strip()
        if fallback_text:
            extracted_texts = [fallback_text]

    combined_text = " ".join(text.strip() for text in extracted_texts if text.strip())

    # 長すぎる場合に切り詰める
    if len(combined_text) > MAX_TEXT_LENGTH:
        combined_text = combined_text[:MAX_TEXT_LENGTH]

    return combined_text


def summarize_with_ai(text: str, ai_client: Any, text_model: str = DEFAULT_TEXT_MODEL) -> str:
    """AIでテキストを要約する

    Args:
        text: 要約するテキスト
        ai_client: AI APIクライアント
        text_model: 要約に使用するモデル名

    Returns:
        str: 要約されたテキスト
    """
    if not text.strip():
        return "要約するテキストがありません。"
        
    messages = [
        {"role": "user", "content": text},
        {"role": "system", "content": "このウェブページの内容を日本語で簡潔に要約してください。"},
    ]
    try:
        result = ai_client.chat.completions.create(
            model=text_model,
            messages=messages,
        )
        return result.choices[0].message.content
    except Exception as err:
        logger.exception(f"Error summarizing text with AI: {err}")
        raise RuntimeError(f"テキスト要約中にエラーが発生しました: {str(err)}")