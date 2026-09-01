"""
Resolve a Hugging Face model to a local file or directory path on the curated shelf.

Supported formats:
    gguf         single .gguf file       (llama.cpp / Ollama / LM Studio)
    mlx          directory of files      (MLX, MLX-LM)
    safetensors  directory of files      (transformers, vLLM, exllamav2)

Lookup order:
    1. Curated shelf      (shelf_root / <format> / ...)
    2. Download from Hugging Face directly into the shelf (if allow_downloads).

Downloads use `huggingface_hub`'s `local_dir` parameter, so files land
directly in the shelf at the friendly path — no parallel cache, no blob
chasing, no temp folder to clean up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from model_shelf.mounts import iter_volumes, split_mount_path


SUPPORTED_FORMATS = ("gguf", "mlx", "safetensors")

SHELF_LEAF_NAME = "ModelShelf/models"  # convention used by detect_storage_candidates

# Files we want when downloading a safetensors repo. Skips .bin twins.
SAFETENSORS_ALLOW_PATTERNS = [
    "*.safetensors",
    "*.safetensors.index.json",
    "*.json",
    "tokenizer*",
    "*.txt",
    "*.md",
]


class StorageNotAvailableError(RuntimeError):
    """The configured shelf is unreachable (unmounted drive or uninitialized shelf)."""


class ShelfNotInitializedError(StorageNotAvailableError):
    """shelf_root is set in config but the directory doesn't exist yet."""


@dataclass
class Config:
    # Optional — if None when load_config is called, it's auto-discovered.
    # After load_config returns, shelf_root is always set.
    shelf_root: Path | None = None
    allow_downloads: bool = True


@dataclass
class ResolveResult:
    status: str               # "found" | "downloaded" | "missing"
    source: str               # "local_shelf" | "huggingface" | "none"
    format: str               # "gguf" | "mlx" | "safetensors"
    path: Path | None         # file for gguf, directory for mlx/safetensors
    checks: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "source": self.source,
            "format": self.format,
            "path": str(self.path) if self.path else None,
            "checks": list(self.checks),
        }


def _check_volume_available(config: Config) -> None:
    """Raise StorageNotAvailableError if shelf_root's volume is unmounted."""
    root = config.shelf_root
    split = split_mount_path(root)
    if split is None:
        return
    mount_root, volume_name, _subpath = split
    volume = mount_root / volume_name
    if not volume.exists():
        raise StorageNotAvailableError(
            f"shelf_root is set to {root}\n"
            f"but the volume '{volume}' is not mounted.\n"
            "Plug the drive in or mount the share, or update your config "
            "(~/.config/model-shelf/config.toml) to point at different storage."
        )


def check_storage_available(config: Config) -> None:
    """Raise if the volume is unmounted OR the shelf directory hasn't been created."""
    _check_volume_available(config)
    if not config.shelf_root.is_dir():
        raise ShelfNotInitializedError(
            f"shelf_root is set to {config.shelf_root}\n"
            "but that folder doesn't exist. Run `model-shelf init` to create it, "
            "or check your config (~/.config/model-shelf/config.toml)."
        )


def init_shelf(config: Config) -> list[Path]:
    """Create shelf_root + gguf/mlx/safetensors subfolders. Returns newly-created paths."""
    _check_volume_available(config)
    created: list[Path] = []
    for path in [config.shelf_root, *(config.shelf_root / fmt for fmt in SUPPORTED_FORMATS)]:
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
            created.append(path)
    return created


def detect_format(repo_id: str) -> str:
    """
    Heuristic format detection from the repo id.

    Splits the name on `-_./` boundaries and looks for format tokens anywhere
    in the name (not just as a suffix), so repos like `Qwen3-4B-MLX-4bit` are
    correctly classified as MLX.

    Examples:
        Qwen/Qwen3-14B-GGUF             -> gguf
        mlx-community/Qwen3-14B-4bit    -> mlx
        Qwen/Qwen3-4B-MLX-4bit          -> mlx
        lmstudio-community/X-MLX-4bit   -> mlx
        Qwen/Qwen3-14B                  -> safetensors
    """
    import re

    parts = repo_id.split("/")
    name = parts[-1].lower()
    tokens = set(filter(None, re.split(r"[-_./]", name)))

    if "gguf" in tokens:
        return "gguf"

    org = parts[0].lower() if len(parts) > 1 else ""
    if org == "mlx-community" or "mlx" in tokens:
        return "mlx"

    return "safetensors"


def _split_repo_id(repo_id: str) -> tuple[str, str]:
    """Return (publisher, repo). Raises ValueError if repo_id lacks a slash."""
    if "/" not in repo_id:
        raise ValueError(
            f"repo_id must be in 'publisher/repo' format (e.g. Qwen/Qwen3-14B-GGUF), got: {repo_id!r}"
        )
    publisher, _, repo = repo_id.partition("/")
    return publisher, repo


def hf_filename(repo_id: str, quant: str) -> str:
    """Filename inside a Hugging Face GGUF repo, preserving original case.

    If the repo name already contains the quant (e.g. `Qwen3-0.6B-Q4_K_M-GGUF`),
    don't append it again — that produces a nonexistent file on the Hub.
    """
    name = repo_id.split("/")[-1]
    if name.lower().endswith("-gguf"):
        name = name[: -len("-gguf")]
    if quant.lower() in name.lower():
        return f"{name}.gguf"
    return f"{name}-{quant}.gguf"


def shelf_path_gguf(shelf_root: Path, repo_id: str, quant: str) -> Path:
    """Shelf path for a GGUF model: <root>/gguf/<publisher>/<repo>/<file>.gguf."""
    publisher, repo = _split_repo_id(repo_id)
    return shelf_root / "gguf" / publisher / repo / hf_filename(repo_id, quant)


def shelf_path_snapshot(shelf_root: Path, repo_id: str, fmt: str) -> Path:
    """Shelf path for an mlx/safetensors model dir: <root>/<fmt>/<publisher>/<repo>/."""
    publisher, repo = _split_repo_id(repo_id)
    return shelf_root / fmt / publisher / repo


def _looks_like_model_dir(path: Path) -> bool:
    """Curated shelf hit for directory formats: dir exists with a config.json inside."""
    return path.is_dir() and (path / "config.json").is_file()


def list_shelf_candidates(config: Config) -> list[Path]:
    """Every plausible shelf root to search at resolve time.

    Order: configured primary first (if set), then every mounted volume with a
    ModelShelf/models folder, then ~/.cache/model-shelf/models if it exists.
    De-duplicated by resolved path.
    """
    seen: set[Path] = set()
    out: list[Path] = []

    def add(p: Path) -> None:
        try:
            key = p.resolve() if p.exists() else p
        except OSError:
            key = p
        if key in seen:
            return
        seen.add(key)
        out.append(p)

    if config.shelf_root is not None:
        add(config.shelf_root)

    for vol in iter_volumes():
        candidate = vol / "ModelShelf" / "models"
        if candidate.is_dir():
            add(candidate)

    internal = Path.home() / ".cache" / "model-shelf" / "models"
    if internal.is_dir():
        add(internal)

    return out


def discover_primary_shelf(
    *,
    volumes_dir: Path | None = None,
    home: Path | None = None,
) -> Path:
    """Pick a default primary shelf when the config doesn't pin one.

    Preference: the first mounted volume with a `ModelShelf/models` folder —
    `/Volumes/*` on macOS, `/mnt/*` and `/media/*` on Linux — else the internal
    default `~/.cache/model-shelf/models`. The returned path may not exist yet;
    downstream `check_storage_available` will surface that.

    `volumes_dir` (restrict the scan to a single root) and `home` are
    overridable for testing.
    """
    if home is None:
        home = Path.home()
    roots = None if volumes_dir is None else [volumes_dir]
    for vol in iter_volumes(roots):
        candidate = vol / "ModelShelf" / "models"
        if candidate.is_dir():
            return candidate
    return home / ".cache" / "model-shelf" / "models"


def resolve_model(
    config: Config,
    repo_id: str,
    *,
    format: str | None = None,
    quant: str | None = None,
) -> ResolveResult:
    check_storage_available(config)
    fmt = format or detect_format(repo_id)
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"unsupported format: {fmt!r}")
    if fmt == "gguf" and quant is None:
        raise ValueError("--quant is required for gguf format")

    if fmt == "gguf":
        return _resolve_gguf(config, repo_id, quant)
    return _resolve_snapshot(config, repo_id, fmt)


def _resolve_gguf(config: Config, repo_id: str, quant: str) -> ResolveResult:
    """Check every available shelf for the model. Download into the primary on miss."""
    checks: list[dict] = []

    for parent in list_shelf_candidates(config):
        candidate = shelf_path_gguf(parent, repo_id, quant)
        shelf = parent / "gguf"
        if candidate.is_file():
            checks.append({"location": "shelf", "root": str(shelf), "result": "hit"})
            return ResolveResult(
                status="found", source="local_shelf", format="gguf",
                path=candidate, checks=checks,
            )
        checks.append({"location": "shelf", "root": str(shelf), "result": "miss"})

    if not config.allow_downloads:
        return ResolveResult(
            status="missing", source="none", format="gguf",
            path=None, checks=checks,
        )

    # Download into the primary shelf at <root>/gguf/<publisher>/<repo>/.
    final = shelf_path_gguf(config.shelf_root, repo_id, quant)
    final.parent.mkdir(parents=True, exist_ok=True)
    hf_name = hf_filename(repo_id, quant)
    hf_hub_download(
        repo_id=repo_id,
        filename=hf_name,
        local_dir=str(final.parent),
    )

    return ResolveResult(
        status="downloaded", source="huggingface", format="gguf",
        path=final, checks=checks,
    )


def _resolve_snapshot(config: Config, repo_id: str, fmt: str) -> ResolveResult:
    """Check every available shelf for the model dir. Download into primary on miss."""
    checks: list[dict] = []

    for parent in list_shelf_candidates(config):
        candidate = shelf_path_snapshot(parent, repo_id, fmt)
        shelf = parent / fmt
        if _looks_like_model_dir(candidate):
            checks.append({"location": "shelf", "root": str(shelf), "result": "hit"})
            return ResolveResult(
                status="found", source="local_shelf", format=fmt,
                path=candidate, checks=checks,
            )
        checks.append({"location": "shelf", "root": str(shelf), "result": "miss"})

    if not config.allow_downloads:
        return ResolveResult(
            status="missing", source="none", format=fmt,
            path=None, checks=checks,
        )

    # Download into the primary shelf at <root>/<fmt>/<publisher>/<repo>/.
    final = shelf_path_snapshot(config.shelf_root, repo_id, fmt)
    final.mkdir(parents=True, exist_ok=True)
    allow_patterns = SAFETENSORS_ALLOW_PATTERNS if fmt == "safetensors" else None
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(final),
        allow_patterns=allow_patterns,
    )

    return ResolveResult(
        status="downloaded", source="huggingface", format=fmt,
        path=final, checks=checks,
    )
