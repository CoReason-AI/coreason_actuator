import sys
from unittest.mock import patch


def test_logger_mkdir() -> None:
    """Test that the logger creates the logs directory if it does not exist."""
    with patch("pathlib.Path.exists", return_value=False), patch("pathlib.Path.mkdir") as mock_mkdir:
        if "coreason_actuator.utils.logger" in sys.modules:
            del sys.modules["coreason_actuator.utils.logger"]

        with patch("loguru.logger.add"), patch("loguru.logger.remove"):
            import coreason_actuator.utils.logger  # noqa: F401

            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
