import logging

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def summarize_webpage(url, openai_client):
    """"""
    if "youtube.com" in url:
        return summarize_youtube(url, openai_client)

    if "x.com" in url or "twitter.com" in url:
        return summarize_x(url, openai_client)

    # Get the webpage
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Mobile/15E148 Safari/604.1"})
    logger.debug(response.status_code)
    logger.debug(response.text)
    html = response.text
    logger.debug(html[:50])

    # Limit the length of the HTML to 4096 characters
    html = html[:4096]

    messages = [
        {"role": "user", "content": html},
        {"role": "system", "content": "Summarize the webpage."},
    ]
    try:
        summarized_text = (
            openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
            )
            .choices[0]
            .message.content
        )
    except RuntimeError as err:
        logger.exception(err)
        raise
    return summarized_text


def summarize_youtube(url, openai_client):
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    logger.debug(response.status_code)
    logger.debug(response.text)
    html = response.text
    logger.debug(html[:50])

    idx_s = html.find("<title>")
    idx_e = html.find("</title>")
    title = html[idx_s + 7: idx_e]
    logger.debug(f"Title: {title}")
    return "This youtube video is " + title


def summarize_x(url, openai_client):
    response = requests.get(url, headers={"User-Agent": "bot"})
    logger.debug(response.status_code)
    logger.debug(response.text)
    html = response.text
    logger.debug(html[:50])

    soup = BeautifulSoup(html, "html.parser")

    summarized_text = ""

    # metaタグを列挙
    for meta in soup.find_all("meta"):
        # propertyがog:imageなら画像なのでsummarize_image
        if meta.get("property") == "og:image":
            summarized_text += summarize_image(meta.get("content"), openai_client) + "\n"
            continue
        # propertyがog:titleならタイトル
        if meta.get("property") == "og:title":
            summarized_text += str(meta) + "\n"
            continue
        # propertyがog:descriptionなら説明文
        if meta.get("property") == "og:description":
            summarized_text += str(meta) + "\n"
            continue
        logger.debug(meta.get("content"))
    return "This is a summary of the webpage: " + summarized_text


def summarize_image(url, openai_client):
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
                model="gpt-4-turbo",
                messages=messages,
            )
            .choices[0]
            .message.content
        )
    except RuntimeError as err:
        logger.exception(err)
        raise
    return summarized_text
