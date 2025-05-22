from unittest.mock import MagicMock, patch
import pytest
import summarizer

def make_openai_like_response(content="Image content described", role="assistant"):
    class Message:
        def __init__(self, content):
            self.content = content
            self.role = role
    class Choice:
        def __init__(self, message):
            self.message = message
    class Response:
        def __init__(self, choices):
            self.choices = choices
    return Response([Choice(Message(content))])

@pytest.fixture(autouse=True)
def patch_hybrid_ai_create():
    with patch("ai_client.HybridAIClient.create", side_effect=lambda *a, **kw: make_openai_like_response()):
        yield

@pytest.fixture(autouse=True)
def patch_llmchain():
    with patch("summarizer.LLMChain") as mock_llmchain:
        mock_chain = MagicMock()
        mock_chain.run.return_value = "Dummy summary of example.com"
        mock_llmchain.return_value = mock_chain
        yield

def test_summarize_webpage_for_youtube(mocker):
    test_url = "https://www.youtube.com/watch?v=test"
    title = "Test Title"

    # Mock the requests.get function
    def mock_get(url, headers):
        class MockResponse:
            status_code = 200
            text = f"<title>{title}</title>"

        return MockResponse()

    mocker.patch("requests.get", side_effect=mock_get)
    summarized_text = summarizer.summarize_webpage(test_url, None)
    assert summarized_text == "YouTube動画: " + title


def test_summarize_webpage_normal_site():
    test_url = "https://example.com"
    html_content = (
        "<html><head><title>Example Domain</title></head><body>Hello</body></html>"
    )

    def mock_get(url, headers):
        class MockResponse:
            status_code = 200
            text = html_content

        return MockResponse()

    with patch("requests.get", side_effect=mock_get):
        summarized_text = summarizer.summarize_webpage(test_url, None)
        assert "Dummy summary of example.com" in summarized_text


def test_summarize_youtube_direct():
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    with patch("requests.get") as mock_requests:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<title>Rick Astley - Never Gonna Give You Up</title>"
        mock_requests.return_value = mock_response

        title = summarizer.summarize_youtube(url)
        assert "Rick Astley" in title


def test_summarize_x(mocker):
    # テスト用に x(Twitter) とわかるURL
    url = "https://x.com/someone/status/123456789"
    mock_html = """
    <html>
    <head>
      <meta property="og:title" content="This is a tweet" />
      <meta property="og:description" content="Tweet description" />
      <meta property="og:image" content="https://example.com/image.png" />
    </head>
    <body>Some tweet content</body>
    </html>
    """

    def mock_get(url, headers):
        class MockResponse:
            status_code = 200
            text = mock_html

        return MockResponse()

    mocker.patch("requests.get", side_effect=mock_get)
    # ai_clientにcreateメソッドを持つダミーを渡す
    class DummyAIClient:
        def create(self, *a, **kw):
            class Message:
                def __init__(self, content):
                    self.content = content
                    self.role = "assistant"
            class Choice:
                def __init__(self, message):
                    self.message = message
            class Response:
                def __init__(self, choices):
                    self.choices = choices
            return Response([Choice(Message("Image summary"))])
    mock_ai_client = DummyAIClient()

    result = summarizer.summarize_x(url, mock_ai_client)
    assert "X投稿の要約" in result
    assert "Tweet description" in result
    assert "Image summary" in result


def test_summarize_image(mocker):
    url = "https://example.com/image.png"
    # ai_clientにcreateメソッドを持つダミーを渡す
    class DummyAIClient:
        def create(self, *a, **kw):
            class Message:
                def __init__(self, content):
                    self.content = content
                    self.role = "assistant"
            class Choice:
                def __init__(self, message):
                    self.message = message
            class Response:
                def __init__(self, choices):
                    self.choices = choices
            return Response([Choice(Message("Image content described"))])
    mock_ai_client = DummyAIClient()

    summary = summarizer.summarize_image(url, mock_ai_client)
    assert "Image content described" in summary
