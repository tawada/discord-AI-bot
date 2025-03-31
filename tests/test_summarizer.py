from unittest.mock import MagicMock, patch

import pytest

import summarizer


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


def test_summarize_webpage_normal_site(mocker):
    test_url = "https://example.com"
    html_content = (
        "<html><head><title>Example Domain</title></head><body>Hello</body></html>"
    )

    # requests.get のモック
    def mock_get(url, headers):
        class MockResponse:
            status_code = 200
            text = html_content

        return MockResponse()

    mocker.patch("requests.get", side_effect=mock_get)

    mock_ai_client = MagicMock()
    mock_ai_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="Dummy summary of example.com"))
    ]

    summarized_text = summarizer.summarize_webpage(test_url, mock_ai_client)
    assert "Dummy summary of example.com" in summarized_text
    mock_ai_client.chat.completions.create.assert_called_once()


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
    mock_ai_client = MagicMock()
    mock_ai_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="Image summary"))
    ]

    result = summarizer.summarize_x(url, mock_ai_client)
    assert "X投稿の要約" in result
    assert "Tweet description" in result
    assert "Image summary" in result


def test_summarize_image(mocker):
    url = "https://example.com/image.png"
    mock_ai_client = MagicMock()
    mock_ai_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="Image content described"))
    ]

    summary = summarizer.summarize_image(url, mock_ai_client)
    assert "Image content described" in summary
    mock_ai_client.chat.completions.create.assert_called_once()
