# ABOUTME: CLI-level tests for model-shelf subcommands
# ABOUTME: Covers init's graceful failure when the config file is externally managed

import argparse
from pathlib import Path

from model_shelf import cli
from model_shelf.resolver import Config


def _init_args(path: Path) -> argparse.Namespace:
    return argparse.Namespace(path=str(path), config=None)


def test_init_pins_explicit_path_in_config(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("MODEL_SHELF_CONFIG", str(config_path))

    rc = cli.cmd_init(_init_args(tmp_path / "shelf"), Config(shelf_root=None))

    assert rc == 0
    contents = config_path.read_text()
    assert "shelf_root" in contents
    assert str(tmp_path / "shelf") in contents


def test_init_reports_managed_config_instead_of_traceback(
    tmp_path: Path, monkeypatch, capsys
):
    # A config manager (home-manager, chezmoi, ...) may provision
    # ~/.config/model-shelf/config.toml as a read-only symlink; rewriting it
    # then raises PermissionError. Injected at the seam rather than via chmod,
    # which silently no-ops when tests run as root (CI containers).
    def _refuse(*args, **kwargs):
        raise PermissionError(13, "Read-only file system")

    monkeypatch.setattr(cli, "write_config", _refuse)

    rc = cli.cmd_init(_init_args(tmp_path / "shelf"), Config(shelf_root=None))

    assert rc == 1
    err = capsys.readouterr().err
    assert "managed" in err
    assert "MODEL_SHELF_CONFIG" in err
    # The shelf itself was still created — only the pin failed.
    assert (tmp_path / "shelf" / "gguf").is_dir()
