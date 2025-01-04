import pytest
from unittest.mock import MagicMock

import functions


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Check out https://example.com now!", True),
        ("No link here!", False),
        ("Multiple links: http://foo.com and https://bar.com", True),
    ],
)
def test_contains_url(text, expected):
    assert functions.contains_url(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("See this https://example.com", ["https://example.com"]),
        ("Two links: http://foo.com, https://bar.com", ["http://foo.com", "https://bar.com"]),
        ("No link", []),
    ],
)
def test_extract_urls(text, expected):
    assert functions.extract_urls(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Pythonについて教えて", True),
        ("調べて欲しい", True),
        ("何をすればいい？", True),
        ("これは普通の文章です", False),
        ("誰が主人公？", True),
    ],
)
def test_contains_knowledge_request_keywords(text, expected):
    assert functions.contains_knowledge_request_keywords(text) == expected


def test_is_image_attachment():
    attachment_mock = MagicMock()
    attachment_mock.filename = "test.jpg"
    attachment_mock.url = "https://cdn.discordapp.com/attachments/test.jpg"
    attachment_mock.content_type = "image/jpeg"

    assert functions.is_image_attachment(attachment_mock) is True

    # PNG
    attachment_mock.filename = "some.png"
    attachment_mock.url = "http://somewhere/some.png"
    attachment_mock.content_type = "image/png"
    assert functions.is_image_attachment(attachment_mock) is True

    # not image
    attachment_mock.filename = "file.txt"
    attachment_mock.url = "http://somewhere/file.txt"
    attachment_mock.content_type = "text/plain"
    assert functions.is_image_attachment(attachment_mock) is False
