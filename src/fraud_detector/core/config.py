import json
from pathlib import Path
from typing import Any, Optional

from jsonschema import ValidationError, validate
import yaml

__all__ = ["ConfigManager"]


class ConfigManager:
    _instance: Optional["ConfigManager"] = None
    _config: Optional[dict[str, Any]] = None

    def __new__(cls) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, config_path: Path) -> dict[str, Any]:
        """Load and validate the global configuration file.

        Args:
            config_path: Path to the configuration file

        Returns:
            The loaded configuration dictionary

        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValueError: If the config file is invalid or doesn't match the schema
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")

        # Load the global schema
        schema_path = Path("config/.schemas/global.json")
        if not schema_path.exists():
            raise FileNotFoundError(f"Global schema not found at {schema_path}")

        with schema_path.open() as f:
            schema = json.load(f)

        with config_path.open() as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid config format in {config_path}. Expected a dictionary.")

        try:
            validate(instance=data, schema=schema)
        except ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e!s}") from e

        self._config = data
        return data

    def get_command_config(self, command: str) -> dict[str, Any]:
        """Get the configuration for a specific command.

        Args:
            command: The command name (train, explain, or analyze)

        Returns:
            The configuration dictionary for the specified command

        Raises:
            RuntimeError: If configuration is not loaded
            KeyError: If the command configuration is not found
        """
        if self._config is None:
            raise RuntimeError("Configuration not loaded. Call load() first.")

        if command not in self._config:
            raise KeyError(f"Configuration for command '{command}' not found")

        return self._config[command]

    @staticmethod
    def available_configs() -> dict[str, list[str]]:
        """Get available configuration files.

        Returns:
            Dictionary mapping command names to lists of available configuration files
        """
        config_dir = Path("config")
        result: dict[str, list[str]] = {"train": [], "explain": [], "analyze": []}

        # Look for configs in command subdirectories
        for cmd in result:
            cmd_dir = config_dir / cmd
            if cmd_dir.exists():
                for config_path in cmd_dir.glob("*.y*ml"):
                    result[cmd].append(config_path.stem)

        return result

    @property
    def config(self) -> dict[str, Any]:
        """Get the full configuration.

        Returns:
            The complete configuration dictionary

        Raises:
            RuntimeError: If configuration is not loaded
        """
        if self._config is None:
            raise RuntimeError("Configuration not loaded. Call load() first.")
        return self._config
