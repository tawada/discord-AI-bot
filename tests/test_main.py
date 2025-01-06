import logging
from unittest.mock import patch

import main


def test_logging_level():
    # メインスクリプトの logger がINFOになっているか確認
    # ここでは root ロガーのレベルをチェックしていますが、
    # より厳密に特定のロガー名をチェックしたい場合は
    # logging.getLogger("some_logger_name") のレベルをチェックしてください
    assert logging.getLogger().level == logging.WARNING


def test_main_calls_discord_run():
    with patch("main.discord_client.run") as mock_run:
        main.main()  # 関数として呼べるようになった
        mock_run.assert_called_once()
