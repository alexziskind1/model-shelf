import tomllib
from pathlib import Path

from model_shelf import config as config_mod
from model_shelf.config import (
    bootstrap_default_config,
    load_config,
    writable_config_path,
    write_config,
)


def test_bootstrap_creates_default_config(tmp_path: Path):
    target = tmp_path / "config.toml"
    assert not target.exists()

    bootstrap_default_config(target)

    contents = target.read_text()
    assert "shelf_root" in contents
    assert "allow_downloads = true" in contents
    # Cache concept dropped in v0.4 — no hf_cache_root anymore.
    assert "hf_cache_root" not in contents


def test_bootstrap_is_idempotent(tmp_path: Path):
    target = tmp_path / "config.toml"
    target.write_text(
        'shelf_root      = "/custom/shelf"\n'
        'hf_cache_root   = "/custom/cache"\n'
        "allow_downloads = false\n"
    )
    original = target.read_text()

    bootstrap_default_config(target)

    assert target.read_text() == original


def test_load_config_bootstraps_when_nothing_found(
    tmp_path: Path, monkeypatch
):
    """First-run bootstrap: write a default config, then auto-discover the shelf."""
    monkeypatch.delenv("MODEL_SHELF_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    user_config = tmp_path / "user-config.toml"
    monkeypatch.setattr(config_mod, "USER_CONFIG", user_config)
    # Isolate discovery from the real /Volumes/ on this machine.
    fake_volumes = tmp_path / "Volumes"
    fake_volumes.mkdir()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    import model_shelf.resolver as resolver_mod
    monkeypatch.setattr(
        resolver_mod,
        "discover_primary_shelf",
        lambda **_: fake_home / ".cache" / "model-shelf" / "models",
    )

    cfg = load_config()

    assert user_config.is_file()
    # No shelf_root line (only comments mentioning it) — pure discovery.
    non_comment_lines = [
        line for line in user_config.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert all("shelf_root" not in line for line in non_comment_lines)
    # Discovery returned the internal default since fake_volumes is empty.
    assert cfg.shelf_root == fake_home / ".cache" / "model-shelf" / "models"
    assert cfg.allow_downloads is True


def test_write_config_writes_shelf_root_when_given(tmp_path: Path):
    target = tmp_path / "config.toml"
    write_config(target, shelf_root=Path("/some/shelf"), allow_downloads=False)
    contents = target.read_text()
    assert 'shelf_root      = "/some/shelf"' in contents
    assert "allow_downloads = false" in contents


def test_write_config_omits_shelf_root_when_none(tmp_path: Path):
    """v0.12: an unpinned config has no shelf_root line — discovery handles it."""
    target = tmp_path / "config.toml"
    write_config(target, shelf_root=None, allow_downloads=True)
    lines = [
        line for line in target.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    # The only non-comment line should be allow_downloads.
    assert lines == ["allow_downloads = true"]


def test_writable_config_path_falls_back_to_user_config(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MODEL_SHELF_CONFIG", raising=False)
    user_config = tmp_path / "user-config.toml"
    monkeypatch.setattr(config_mod, "USER_CONFIG", user_config)
    monkeypatch.chdir(tmp_path)

    assert writable_config_path() == user_config


def test_load_config_ignores_cwd_config_toml(tmp_path: Path, monkeypatch):
    """Regression test: a project-local config.toml in cwd must NOT hijack load_config.

    Previously, an unrelated tool's config.toml could silently override the
    user-level Model Shelf config. v0.7.1 removed that fallback.
    """
    monkeypatch.delenv("MODEL_SHELF_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    # Drop a foreign config.toml in cwd that would have hijacked us pre-0.7.1.
    (tmp_path / "config.toml").write_text(
        'shelf_root = "/some/foreign/path"\n'
        "allow_downloads = false\n"
    )
    # Point USER_CONFIG at our own file with a known value.
    user_config = tmp_path / "user-config.toml"
    user_config.write_text(
        f'shelf_root = "{tmp_path / "the-real-shelf"}"\n'
        "allow_downloads = true\n"
    )
    monkeypatch.setattr(config_mod, "USER_CONFIG", user_config)

    cfg = load_config()

    # Must use the USER_CONFIG, not the cwd-local file.
    assert str(cfg.shelf_root) == str(tmp_path / "the-real-shelf")
    assert cfg.allow_downloads is True


def test_writable_config_path_respects_env(tmp_path: Path, monkeypatch):
    explicit = tmp_path / "explicit.toml"
    monkeypatch.setenv("MODEL_SHELF_CONFIG", str(explicit))
    assert writable_config_path() == explicit


def test_writable_config_path_respects_arg(tmp_path: Path):
    explicit = tmp_path / "explicit.toml"
    assert writable_config_path(explicit) == explicit


def test_write_config_windows_path_roundtrips(tmp_path: Path):
    """A Windows-style path with backslashes must produce valid, loadable TOML."""
    cfg_path = tmp_path / "config.toml"
    # On POSIX, Path(r"L:\models") keeps the backslash as a literal character
    # in its string form, giving a real backslash to escape — no Windows-only
    # path type needed to reproduce the bug.
    write_config(cfg_path, shelf_root=Path(r"L:\models"))

    # Must NOT raise tomllib.TOMLDecodeError ("Unescaped '\' in a string").
    with open(cfg_path, "rb") as f:
        data = tomllib.load(f)

    assert data["shelf_root"] == r"L:\models"
    assert data["allow_downloads"] is True


def test_write_config_path_with_quote_roundtrips(tmp_path: Path):
    """Defensive: a path containing a double-quote stays valid TOML."""
    cfg_path = tmp_path / "config.toml"
    write_config(cfg_path, shelf_root=Path('/tmp/wei"rd/models'))
    with open(cfg_path, "rb") as f:
        data = tomllib.load(f)
    assert data["shelf_root"] == '/tmp/wei"rd/models'


def test_write_config_control_char_roundtrips(tmp_path: Path):
    """A control char in the value must be escaped to valid TOML, not passed through."""
    cfg_path = tmp_path / "config.toml"
    write_config(cfg_path, shelf_root=Path("/tmp/a\x01b/models"))
    with open(cfg_path, "rb") as f:
        data = tomllib.load(f)
    assert data["shelf_root"] == "/tmp/a\x01b/models"
