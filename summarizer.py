import logging

import requests

logger = logging.getLogger(__name__)


def summarize_webpage(url, openai_client):
    # Get the webpage
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Mobile/15E148 Safari/604.1"})
    logger.debug(response.status_code)
    logger.debug(response.text)
    html = response.text
    logger.debug(html[:50])

    if url.startswith("https://youtube.com/") or url.startswith(
        "https://www.youtube.com/"
    ):
        idx_s = html.find("<title>")
        idx_e = html.find("</title>")
        title = html[idx_s + 7 : idx_e]
        logger.debug(f"Title: {title}")
        return title

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
