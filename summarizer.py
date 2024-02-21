import requests
import logging

logger = logging.getLogger(__name__)


def summarize_webpage(url, openai_client):
    # Get the webpage
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    logger.debug(response.status_code)
    logger.debug(response.text)
    html = response.text
    logger.debug(html[:50])

    messages = [
        {"role": "user", "content": html},
        {"role": "system", "content": "Summarize the webpage."},
    ]
    try:
        summarized_text = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
        ).choices[0].message.content
    except RuntimeError as err:
        logger.exception(err)
        raise
    return summarized_text
