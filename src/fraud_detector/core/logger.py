from datetime import datetime
import logging
from logging import Handler, Logger
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Literal

from rich.logging import RichHandler


class LoggingManager:
    _instance = None
    _loggers: dict[str, Logger] = {}
    _supported: list[str] = ["Training", "Running", "Processing", "Analysis"]

    def __new__(cls) -> "LoggingManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def supported_loggers(self) -> list[str]:
        return self._supported

    def get_logger(self, name: str) -> Logger:
        if name not in self._loggers:
            raise RuntimeError(f"Logger '{name}' not initialized. Call init_logger() first.")
        return self._loggers[name]

    @staticmethod
    def init_logger(
        name: str,
        log_dir: Path = Path("./logs"),
        enabled: bool = True,
        rotation: Literal["size", "time"] = "size",
        max_bytes: int = 5 * 1024 * 1024,
        when: str = "midnight",
        backup_count: int = 5,
    ) -> Logger:
        if name not in LoggingManager._supported:
            raise ValueError(f"Unsupported logger type: {name}")

        if name in LoggingManager._loggers:
            return LoggingManager._loggers[name]

        log_filename = f"[{name}] {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_path = log_path / log_filename

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | [%(module)s.%(funcName)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        if enabled:
            console_handler = RichHandler(rich_tracebacks=True, show_path=False)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        if rotation == "size":
            file_handler: Handler = RotatingFileHandler(
                file_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
        elif rotation == "time":
            file_handler = TimedRotatingFileHandler(
                file_path, when=when, backupCount=backup_count, encoding="utf-8"
            )
        else:
            raise ValueError("Invalid rotation type. Use 'size' or 'time'.")

        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        LoggingManager._loggers[name] = logger
        return logger
