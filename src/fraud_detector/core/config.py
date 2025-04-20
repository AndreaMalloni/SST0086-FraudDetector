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
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")

        if config_path.stem not in {
            name for names in ConfigManager.available_configs().values() for name in names
        }:
            raise ValueError(f"Cannot load unknown config: {config_path}.") from None

        with config_path.open() as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid config format in {config_path}. Expected a dictionary.")

        self._config = data
        return data

    @staticmethod
    def available_configs() -> dict[str, list[str]]:
        config_dir = Path("config")
        schemas_dir = config_dir / ".schemas"
        result: dict[str, list[str]] = {}

        for schema_path in schemas_dir.glob("*.json"):
            schema_name = schema_path.stem
            with schema_path.open() as f:
                schema = json.load(f)

            valid_configs = []
            for config_path in config_dir.glob("*.y*ml"):
                with config_path.open() as f:
                    try:
                        config = yaml.safe_load(f)
                        validate(instance=config, schema=schema)
                        valid_configs.append(config_path.stem)
                    except ValidationError:
                        continue

            result[schema_name] = valid_configs

        return result

    @property
    def config(self) -> dict[str, Any]:
        if self._config is None:
            raise RuntimeError("Configuration not loaded. Call load() first.")
        return self._config
