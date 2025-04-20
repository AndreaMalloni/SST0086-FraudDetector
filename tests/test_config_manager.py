import json
from pathlib import Path

import pytest
import yaml

from fraud_detector.core.config import ConfigManager


@pytest.fixture
def dummy_config(tmp_path: Path) -> tuple[Path, Path]:
    config_dir = tmp_path / "config"
    schema_dir = config_dir / ".schemas"
    config_dir.mkdir()
    schema_dir.mkdir()

    schema_path = schema_dir / "train.json"
    schema = {
        "type": "object",
        "properties": {"learning_rate": {"type": "number"}},
        "required": ["learning_rate"],
    }
    schema_path.write_text(json.dumps(schema))

    config_path = config_dir / "valid.yaml"
    config = {"learning_rate": 0.01}
    config_path.write_text(yaml.dump(config))

    return tmp_path, config_path


def test_available_configs(
    dummy_config: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    base_path, _ = dummy_config
    monkeypatch.chdir(base_path)

    configs = ConfigManager.available_configs()
    assert "train" in configs
    assert "valid" in configs["train"]


def test_load_valid_config(
    dummy_config: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    base_path, config_path = dummy_config
    monkeypatch.chdir(base_path)

    cm = ConfigManager()
    data = cm.load(config_path)
    assert isinstance(data, dict)
    assert data["learning_rate"] == 0.01


def test_load_invalid_path_raises() -> None:
    cm = ConfigManager()
    with pytest.raises(FileNotFoundError):
        cm.load(Path("non_existent.yaml"))
