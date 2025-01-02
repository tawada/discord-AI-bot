import logging
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def summarize_webpage(url, gemini_client):
    """Gemini クライアントでウェブページを要約"""
    # YouTube や X (twitter) もテキストとして要約する想定
    # 必要に応じて分岐

    if "youtube.com" in url or "youtu.be" in url:
        return summarize_youtube(url, gemini_client)

    if "x.com" in url or "twitter.com" in url:
        return summarize_x(url, gemini_client)

    # 通常のウェブページ要約
    response = requests.get(
        url, 
        headers={
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Mobile/15E148 Safari/604.1"
        }
    )
    logger.debug(response.status_code)
    logger.debug(response.text)
    html = response.text
    logger.debug(html[:50])

    # モデルに送るテキストが長すぎるとエラーになりやすいので適当にカット
    html = html[:4096]

    messages = [
        {"role": "user", "content": html},
        {"role": "system", "content": "Please summarize the webpage."},
    ]
    try:
        # Gemini 側で要約（モデルは gemini-1.5-flash でもよいが、呼び出し側で指定してもOK）
        summarized_text = (
            gemini_client.chat.completions.create(
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


def summarize_youtube(url, gemini_client):
    """YouTube のタイトルなどをテキストとして要約 (Gemini を使用)"""
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    logger.debug(response.status_code)
    logger.debug(response.text)
    html = response.text
    logger.debug(html[:50])

    idx_s = html.find("<title>")
    idx_e = html.find("</title>")
    title = html[idx_s + 7: idx_e]
    logger.debug(f"Title: {title}")

    # 簡易的にテキストレスポンス
    messages = [
        {"role": "user", "content": f"This is a YouTube page. Title is: {title}.\nPlease give a short summary."},
    ]
    try:
        summarized_text = (
            gemini_client.chat.completions.create(
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


def summarize_x(url, gemini_client):
    """X (旧 Twitter) のページをテキストとして要約 (Gemini を使用)"""
    response = requests.get(url, headers={"User-Agent": "bot"})
    logger.debug(response.status_code)
    logger.debug(response.text)
    html = response.text
    logger.debug(html[:50])

    soup = BeautifulSoup(html, "html.parser")
    meta_tags = soup.find_all("meta")
    return summarize_from_meta_tags(meta_tags, gemini_client)


def summarize_from_meta_tags(meta_tags, gemini_client):
    """meta タグから文章を抜き出して Gemini で要約"""
    # メタタグのプロパティごとに処理を分ける
    # ここはサンプルとして何らかの抽出を行う例
    contents = []
    for meta in meta_tags:
        logger.debug(meta)
        if meta.get("property") == "og:title":
            contents.append("Title: " + (meta.get("content") or ""))
        elif meta.get("property") == "og:description":
            contents.append("Description: " + (meta.get("content") or ""))

    # まとめて要約
    joined_contents = "\n".join(contents)
    messages = [
        {"role": "user", "content": joined_contents},
        {"role": "system", "content": "Please summarize this information about the page."},
    ]
    try:
        summarized_text = (
            gemini_client.chat.completions.create(
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


def summarize_image(url, openai_client):
    """画像認識は Gemini でできないため、OpenAI 側 (model=gpt-4o) を使う例"""
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
            openai_client.chat.completions.create(
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
