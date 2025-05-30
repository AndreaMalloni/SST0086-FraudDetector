import logging
from pathlib import Path

import pytest

from fraud_detector.core.logger import LoggingManager


@pytest.mark.parametrize("name", ["Training", "Analysis", "Processing"])
def test_logger_init_and_get(name: str, tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    logger = LoggingManager.init_logger(name, log_dir=log_dir, rotation="size")

    assert logger.name == name
    assert logger.level == logging.INFO
    assert log_dir.exists()
    assert any(log_dir.iterdir()), f"Expected log files in {log_dir}"

    retrieved_logger = LoggingManager().get_logger(name)
    assert retrieved_logger is logger


def test_init_logger_invalid_name_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported logger type"):
        LoggingManager.init_logger("InvalidLogger")
