"""Load (or first-run bootstrap) Model Shelf config from a TOML file.

Lookup order:
    1. Explicit path passed to load_config().
    2. $MODEL_SHELF_CONFIG.
    3. ~/.config/model-shelf/config.toml  (user-level default).
    4. None of the above -> bootstrap a default at (3) and load it.

Note: a `config.toml` in the current working directory is NOT used.
A project-local fallback used to exist but was a footgun — any unrelated
tool's config.toml in cwd would silently hijack Model Shelf's config.
For project-scoped overrides, use `--config <path>` or `$MODEL_SHELF_CONFIG`.
"""

from __future__ import annotations

import os
import re
import sys
import tomllib
from pathlib import Path

from model_shelf.resolver import Config


USER_CONFIG = Path.home() / ".config" / "model-shelf" / "config.toml"


def _read(path: Path) -> Config:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    # `hf_cache_root` (pre-0.4) and `shelf_id` (0.9 only) from older configs
    # are silently ignored.
    shelf_root_str = data.get("shelf_root")
    shelf_root = Path(shelf_root_str).expanduser() if shelf_root_str else None
    return Config(
        shelf_root=shelf_root,
        allow_downloads=bool(data.get("allow_downloads", True)),
    )


_TOML_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _toml_basic_string(value: str) -> str:
    """Encode a string as a valid TOML basic (double-quoted) string.

    Escapes backslash and double-quote, the named control escapes
    (\\n/\\r/\\t), and any remaining control characters as \\uXXXX — the set
    TOML 1.0 forbids unescaped in a basic string. Critical on Windows, where
    shelf paths like ``L:\\models`` would otherwise emit an unescaped
    backslash and make config.toml unparseable.
    """
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    escaped = _TOML_CTRL_RE.sub(lambda m: f"\\u{ord(m.group()):04X}", escaped)
    return f'"{escaped}"'


def write_config(
    path: Path,
    *,
    shelf_root: Path | None = None,
    allow_downloads: bool = True,
) -> None:
    """Write a config file at `path`. Overwrites if present.

    If `shelf_root` is None, the file omits the field — Model Shelf will
    auto-discover a primary at runtime instead of pinning a specific path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Model Shelf config.",
        "# shelf_root is optional. If omitted, the primary is auto-discovered",
        "# from any mounted volume with a ModelShelf/models folder - /Volumes/*",
        "# on macOS, /mnt/* and /media/* on Linux - falling back to",
        "# ~/.cache/model-shelf/models. Set it explicitly to pin a location.",
    ]
    if shelf_root is not None:
        lines.append(f"shelf_root      = {_toml_basic_string(str(shelf_root))}")
    lines.append(f"allow_downloads = {'true' if allow_downloads else 'false'}")
    path.write_text("\n".join(lines) + "\n")


def writable_config_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Where `model-shelf init` should write the config (mirrors load_config)."""
    if path is not None:
        return Path(path)
    if env := os.environ.get("MODEL_SHELF_CONFIG"):
        return Path(env)
    return USER_CONFIG


def bootstrap_default_config(path: Path | None = None) -> Path:
    """Create a default config at `path` (or USER_CONFIG) if missing. Returns the path."""
    if path is None:
        path = USER_CONFIG
    if path.is_file():
        return path
    write_config(path, shelf_root=None, allow_downloads=True)
    print(
        f"model-shelf: created default config at {path}\n"
        "model-shelf: run `model-shelf init` to create your shelf "
        "(picks an external drive if one is plugged in, else uses internal).",
        file=sys.stderr,
    )
    return path


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    cfg = _load_raw(path)
    if cfg.shelf_root is None:
        # No pinned path — auto-discover.
        from model_shelf.resolver import discover_primary_shelf
        cfg.shelf_root = discover_primary_shelf()
    else:
        # Pinned path — try to relocate if the configured volume isn't around.
        from model_shelf.relocate import relocate_shelf
        cfg.shelf_root = relocate_shelf(cfg.shelf_root)
    return cfg


def _load_raw(path: str | os.PathLike[str] | None) -> Config:
    if path is not None:
        target = Path(path)
        if target.is_file():
            return _read(target)
        return _read(bootstrap_default_config(target))
    if env := os.environ.get("MODEL_SHELF_CONFIG"):
        target = Path(env)
        if target.is_file():
            return _read(target)
        return _read(bootstrap_default_config(target))
    if USER_CONFIG.is_file():
        return _read(USER_CONFIG)
    return _read(bootstrap_default_config())
