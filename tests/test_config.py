"""Config loading: every failure path is a ConfigError that names what went wrong."""

from __future__ import annotations

from pathlib import Path

import pytest

from textclf.config import Config, ServeSettings, load_config
from textclf.exceptions import ConfigError


def test_defaults_match_committed_yaml() -> None:
    """configs/train.yaml is the documented set of knobs; it must equal the pydantic defaults."""
    cfg = load_config(Path(__file__).parents[1] / "configs" / "train.yaml")
    assert cfg == Config()


def test_empty_yaml_gives_defaults(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert load_config(path) == Config()


def test_override_is_applied(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("train:\n  epochs: 1\n  lr: 0.1\n")
    cfg = load_config(path)
    assert cfg.train.epochs == 1
    assert cfg.train.lr == pytest.approx(0.1)


def test_unknown_key_is_config_error(tmp_path: Path) -> None:
    path = tmp_path / "typo.yaml"
    path.write_text("train:\n  epoch: 3\n")  # 'epoch' instead of 'epochs'
    with pytest.raises(ConfigError) as info:
        load_config(path)
    assert "epoch" in info.value.message
    assert info.value.context["path"] == str(path)
    assert info.value.exit_code == 2


def test_missing_file_is_config_error(tmp_path: Path) -> None:
    path = tmp_path / "nope.yaml"
    with pytest.raises(ConfigError) as info:
        load_config(path)
    assert info.value.message == "Config file not found"
    assert info.value.context == {"path": str(path)}
    assert isinstance(info.value.__cause__, FileNotFoundError)


def test_malformed_yaml_is_config_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("train: [unclosed\n")
    with pytest.raises(ConfigError, match="Malformed YAML"):
        load_config(path)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("data", "max_len", 4),  # below ge=8
        ("data", "val_fraction", 0.9),  # above lt=0.5
        ("model", "dropout", 1.0),  # lt=1
        ("model", "name", "bert"),  # not in Literal
        ("train", "lr", 0),  # gt=0
        ("train", "epochs", 0),  # ge=1
    ],
)
def test_out_of_range_value_names_the_field(
    tmp_path: Path, section: str, field: str, value: object
) -> None:
    path = tmp_path / "range.yaml"
    path.write_text(f"{section}:\n  {field}: {value}\n")
    with pytest.raises(ConfigError) as info:
        load_config(path)
    assert field in info.value.message
    assert info.value.context["path"] == str(path)


def test_serve_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEXTCLF_MODEL_URI", "file:///tmp/m")
    monkeypatch.setenv("TEXTCLF_PORT", "9000")
    monkeypatch.setenv("TEXTCLF_TORCH_THREADS", "4")
    s = ServeSettings()
    assert s.model_uri == "file:///tmp/m"
    assert s.port == 9000
    assert s.torch_threads == 4
    assert s.host == "127.0.0.1"  # secure default; the container overrides it
