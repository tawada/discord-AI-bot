import logging
from typing import List, Dict, Any

from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)


def search_and_summarize(user_question: str, ai_client: Any, text_model: str) -> str:
    """DuckDuckGo APIを使って検索し、出てきた情報をまとめてLLMで要約する"""
    # 検索queryを作成
    messages = [
        {"role": "user", "content": user_question},
        {
            "role": "system",
            "content": (
                f"上記のユーザの質問「{user_question}」に対して、"
                "検索するべき単語を抽出してください。回答は単語のみで構いません。"
            ),
        },
    ]
    try:
        response = ai_client.chat.completions.create(
            model=text_model,
            messages=messages,
        )
        query = response.choices[0].message.content
    except Exception as e:
        logger.exception(e)
        query = user_question

    logger.info(f"Searching for: {query}")
    # DuckDuckGo で検索
    with DDGS() as ddgs:
        results = list(ddgs.text(
            keywords=query,
            region='jp-jp',
            max_results=10
        ))

    if not results:
        return "検索結果が見つかりませんでした。"
    
    # 検索結果をまとめる
    combined_text = f"{results}"

    # モデルに送りすぎるとエラーになりやすいので適当にカット
    combined_text = combined_text[:4096]

    logger.info(f"Search snippets (trimmed): {combined_text[:100]}...")

    if not combined_text.strip():
        return "検索結果から有用な情報を取得できませんでした。"

    # LLMで要約
    messages = [
        {"role": "user", "content": combined_text},
        {
            "role": "system",
            "content": (
                f"上記の検索結果に基づいて、ユーザの質問「{query}」に答えるための"
                "簡潔な日本語要約を作成してください。"
            ),
        },
    ]
    try:
        response = ai_client.chat.completions.create(
            model=text_model,
            messages=messages,
        )
        summary = response.choices[0].message.content
    except Exception as e:
        logger.exception(e)
        summary = "検索の要約時にエラーが発生しました。"
    return summary


def is_search_needed(user_message: str, ai_client: Any, text_model: str) -> bool:
    """Determine if a search is needed based on the user's message."""
    # Define keywords or patterns that indicate a search is needed
    search_keywords = ["教えて", "とは", "何", "どうやって", "方法"]
    if any(keyword in user_message for keyword in search_keywords):
        return True

    messages = [
        {"role": "user", "content": user_message},
        {
            "role": "system",
            "content": (
                '上記のユーザーの発言に適切に答えるためにインターネット検索が必要であれば True、不要であれば False を返してください。必ず "True" または "False" のいずれかのみを返してください。'
            ),
        },
    ]

    try:
        response = ai_client.chat.completions.create(
            model=text_model,
            messages=messages,
        )
        if "False" not in response.choices[0].message.content:
            return True
    except Exception as e:
        logger.exception(e)
    return False