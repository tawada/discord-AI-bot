from typing import Any, Dict, List
import json
from duckduckgo_search import DDGS
from loguru import logger

# 検索に関する定数
MAX_SEARCH_RESULTS = 10
MAX_TEXT_LENGTH = 4096
SEARCH_KEYWORDS = ["教えて", "とは", "何", "どうやって", "方法"]


def search_and_summarize(user_question: str, ai_client: Any, text_model: str) -> str:
    """DuckDuckGo APIを使って検索し、出てきた情報をまとめてLLMで要約する
    
    Args:
        user_question: ユーザーからの質問
        ai_client: AI APIクライアント
        text_model: 使用するAIモデル名
        
    Returns:
        str: 検索結果の要約
    """
    # 検索queryを作成
    search_query = extract_search_query(user_question, ai_client, text_model)
    if not search_query:
        search_query = user_question
    
    logger.info(f"Searching for: {search_query}")
    
    # DuckDuckGo で検索
    search_results = perform_search(search_query)
    
    if not search_results:
        return "検索結果が見つかりませんでした。"
    
    # 検索結果を整形
    formatted_results = format_search_results(search_results)
    logger.info(f"Search snippets (trimmed): {formatted_results[:100]}...")
    
    if not formatted_results.strip():
        return "検索結果から有用な情報を取得できませんでした。"
    
    # LLMで要約
    return summarize_search_results(formatted_results, user_question, ai_client, text_model)


def extract_search_query(user_question: str, ai_client: Any, text_model: str) -> str:
    """ユーザーの質問から検索クエリを抽出
    
    Args:
        user_question: ユーザーからの質問
        ai_client: AI APIクライアント
        text_model: 使用するAIモデル名
        
    Returns:
        str: 検索クエリ
    """
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
        return response.choices[0].message.content
    except Exception as e:
        logger.exception(f"Failed to extract search query: {e}")
        return ""


def perform_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> List[Dict]:
    """DuckDuckGoで検索を実行
    
    Args:
        query: 検索クエリ
        max_results: 最大結果数
        
    Returns:
        List[Dict]: 検索結果リスト
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(keywords=query, region="jp-jp", max_results=max_results))
        return results
    except Exception as e:
        logger.exception(f"Search failed: {e}")
        return []


def format_search_results(results: List[Dict]) -> str:
    """検索結果を整形
    
    Args:
        results: 検索結果リスト
        
    Returns:
        str: 整形された検索結果テキスト
    """
    try:
        # リスト内の各結果を整形
        formatted_items = []
        for item in results:
            title = item.get('title', 'No Title')
            body = item.get('body', 'No Content')
            href = item.get('href', 'No URL')
            formatted_items.append(f"タイトル: {title}\n内容: {body}\nURL: {href}\n")
        
        # 全ての結果を連結
        combined_text = "\n".join(formatted_items)
        
        # 長さを制限
        if len(combined_text) > MAX_TEXT_LENGTH:
            combined_text = combined_text[:MAX_TEXT_LENGTH]
        
        return combined_text
    except Exception as e:
        logger.exception(f"Error formatting search results: {e}")
        return str(results)[:MAX_TEXT_LENGTH]  # フォールバック


def summarize_search_results(results_text: str, query: str, ai_client: Any, text_model: str) -> str:
    """検索結果をLangChainで要約
    
    Args:
        results_text: 整形された検索結果テキスト
        query: 元の検索クエリ
        ai_client: AI APIクライアント（未使用、互換のため残す）
        text_model: 使用するAIモデル名
        
    Returns:
        str: 要約されたテキスト
    """
    if not results_text.strip():
        return "要約する検索結果がありません。"
    try:
        from config import load_config
        from langchain.chat_models import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        from langchain.chains import LLMChain
        config = load_config()
        openai_api_key = config.openai_api_key
        if not openai_api_key:
            return "OpenAI APIキーが設定されていません。"
        llm = ChatOpenAI(model=text_model, openai_api_key=openai_api_key)
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"以下の検索結果に基づいて、ユーザの質問『{query}』に答えるための簡潔な日本語要約を作成してください。"),
            ("user", "{input_text}")
        ])
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run({"input_text": results_text})
        return result
    except Exception as e:
        logger.exception(f"LangChainによる検索要約失敗: {e}")
        return "検索の要約時にエラーが発生しました。"


def is_search_needed(user_message: str, ai_client: Any, text_model: str) -> bool:
    """ユーザーメッセージに基づいて検索が必要かどうかを判断
    
    Args:
        user_message: ユーザーからのメッセージ
        ai_client: AI APIクライアント
        text_model: 使用するAIモデル名
        
    Returns:
        bool: 検索が必要な場合はTrue
    """
    # キーワードに基づく基本的なチェック
    if any(keyword in user_message for keyword in SEARCH_KEYWORDS):
        return True
    
    # AIによる高度な判断
    return ai_check_if_search_needed(user_message, ai_client, text_model)


def ai_check_if_search_needed(user_message: str, ai_client: Any, text_model: str) -> bool:
    """AIを使用して検索が必要かどうかをより高度に判断
    
    Args:
        user_message: ユーザーからのメッセージ
        ai_client: AI APIクライアント
        text_model: 使用するAIモデル名
        
    Returns:
        bool: 検索が必要な場合はTrue
    """
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
        result = response.choices[0].message.content.strip()
        return "True" in result
    except Exception as e:
        logger.exception(f"AI check for search need failed: {e}")
        return False