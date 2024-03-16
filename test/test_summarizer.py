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
    assert summarized_text == title
