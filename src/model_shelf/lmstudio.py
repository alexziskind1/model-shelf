"""LM Studio library scanning and migration helpers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from model_shelf.resolver import (
    Config,
    SUPPORTED_FORMATS,
    list_shelf_candidates,
    shelf_path_gguf,
    shelf_path_snapshot,
)


IGNORED_DIR_NAMES = {
    ".cache",
    ".git",
    "__pycache__",
    "tmp",
    "temp",
    "downloads",
    "incomplete",
}

PARTIAL_SUFFIXES = (".part", ".partial", ".tmp", ".download", ".crdownload")


@dataclass
class LmStudioRootDiscovery:
    selected: Path | None
    candidates: list[Path] = field(default_factory=list)
    source: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": str(self.selected) if self.selected else None,
            "candidates": [str(p) for p in self.candidates],
            "source": self.source,
        }


@dataclass
class LmStudioItem:
    status: str
    format: str | None
    publisher: str | None
    repo: str | None
    source_path: Path
    target_path: Path | None
    size_bytes: int
    is_link: bool = False
    link_target: Path | None = None
    reason: str | None = None

    @property
    def model(self) -> str:
        if self.publisher and self.repo:
            if self.format == "gguf":
                return f"{self.publisher}/{self.repo}/{self.source_path.name}"
            return f"{self.publisher}/{self.repo}"
        return self.source_path.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "format": self.format,
            "publisher": self.publisher,
            "repo": self.repo,
            "model": self.model,
            "source_path": str(self.source_path),
            "target_path": str(self.target_path) if self.target_path else None,
            "size_bytes": self.size_bytes,
            "is_link": self.is_link,
            "link_target": str(self.link_target) if self.link_target else None,
            "reason": self.reason,
        }


@dataclass
class LmStudioScanResult:
    lmstudio_root: Path
    shelf_roots: list[Path]
    items: list[LmStudioItem]
    discovery: LmStudioRootDiscovery | None = None

    @property
    def summary(self) -> dict[str, int]:
        counts = {
            "scanned": len(self.items),
            "lmstudio_only": 0,
            "in_shelf": 0,
            "duplicated": 0,
            "lmstudio_link": 0,
            "shelf_link": 0,
            "ambiguous": 0,
            "unsupported": 0,
            "error": 0,
        }
        for item in self.items:
            counts[item.status] = counts.get(item.status, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "lmstudio_root": str(self.lmstudio_root),
            "shelf_roots": [str(p) for p in self.shelf_roots],
            "discovery": self.discovery.to_dict() if self.discovery else None,
            "items": [item.to_dict() for item in self.items],
            "summary": self.summary,
        }


@dataclass
class MigrationOperation:
    action: str
    source_path: Path
    target_path: Path
    link_path: Path | None = None
    status: str = "planned"
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "source_path": str(self.source_path),
            "target_path": str(self.target_path),
            "link_path": str(self.link_path) if self.link_path else None,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass
class MigrationResult:
    scan: LmStudioScanResult
    operations: list[MigrationOperation]
    applied: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "scan": self.scan.to_dict(),
            "operations": [op.to_dict() for op in self.operations],
        }


@dataclass
class LinkResult:
    scan: LmStudioScanResult
    item: LmStudioItem | None
    operation: MigrationOperation | None
    applied: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "item": self.item.to_dict() if self.item else None,
            "operation": self.operation.to_dict() if self.operation else None,
            "scan": self.scan.to_dict(),
        }


def default_lmstudio_model_roots(home: Path | None = None) -> list[Path]:
    if home is None:
        home = Path.home()
    candidates = [
        home / ".cache" / "lm-studio" / "models",
        home / ".lmstudio" / "models",
    ]
    if os.name == "nt":
        candidates.insert(0, home / ".cache" / "lm-studio" / "models")
    return _dedupe_paths(candidates)


def discover_lmstudio_root(
    explicit_root: Path | None = None,
    *,
    home: Path | None = None,
    use_lms: bool = False,
) -> LmStudioRootDiscovery:
    """Find the LM Studio models directory without requiring LM Studio to run."""
    if explicit_root is not None:
        root = _normalize_lmstudio_root(explicit_root.expanduser())
        return LmStudioRootDiscovery(selected=root, candidates=[root], source="explicit")

    candidates: list[Path] = []
    candidates.extend(_configured_lmstudio_model_roots(home=home))
    if use_lms:
        candidates.extend(_roots_from_lms_inventory())
    candidates.extend(default_lmstudio_model_roots(home=home))
    candidates = _dedupe_paths(candidates)

    existing = [p for p in candidates if p.is_dir()]
    selected = existing[0] if existing else (candidates[0] if candidates else None)
    source = "none"
    if selected is not None:
        if selected in _configured_lmstudio_model_roots(home=home):
            source = "settings"
        elif use_lms and selected in _roots_from_lms_inventory():
            source = "lms"
        else:
            source = "default"
    return LmStudioRootDiscovery(selected=selected, candidates=candidates, source=source)


def scan_lmstudio(
    config: Config,
    *,
    lmstudio_root: Path | None = None,
    format: str | None = None,
    use_lms: bool = False,
) -> LmStudioScanResult:
    if format is not None and format not in SUPPORTED_FORMATS:
        raise ValueError(f"unsupported format: {format!r}")

    discovery = discover_lmstudio_root(lmstudio_root, use_lms=use_lms)
    if discovery.selected is None:
        raise ValueError("could not determine LM Studio models directory")
    root = discovery.selected.expanduser()
    if not root.is_dir():
        raise ValueError(f"LM Studio models directory does not exist: {root}")

    shelf_roots = list_shelf_candidates(config)
    items = _scan_items(root, config.shelf_root, shelf_roots)
    if format is not None:
        items = [item for item in items if item.format == format]
    return LmStudioScanResult(
        lmstudio_root=root,
        shelf_roots=shelf_roots,
        items=items,
        discovery=discovery,
    )


def migrate_lmstudio(
    config: Config,
    *,
    model: str | None = None,
    filename: str | None = None,
    lmstudio_root: Path | None = None,
    format: str | None = None,
    use_lms: bool = False,
    link: bool = False,
    dry_run: bool = True,
    yes: bool = True,
) -> MigrationResult:
    if config.shelf_root is None:
        raise ValueError("shelf_root is not configured")
    scan = scan_lmstudio(
        config,
        lmstudio_root=lmstudio_root,
        format=format,
        use_lms=use_lms,
    )
    operations = build_migration_plan(scan, link=link, model=model, filename=filename)
    applied = False
    if not dry_run:
        if not yes:
            raise ValueError("migrate requires --yes when not using --dry-run")
        for op in operations:
            _apply_operation(op)
        applied = True
    return MigrationResult(scan=scan, operations=operations, applied=applied)


def link_lmstudio_model(
    config: Config,
    model: str,
    *,
    filename: str | None = None,
    lmstudio_root: Path | None = None,
    format: str | None = None,
    use_lms: bool = False,
    dry_run: bool = True,
    yes: bool = False,
    replace_link: bool = False,
) -> LinkResult:
    scan = scan_lmstudio(
        config,
        lmstudio_root=lmstudio_root,
        format=format,
        use_lms=use_lms,
    )
    item = _find_link_item(scan, model, filename=filename)
    op = MigrationOperation(
        action="link_into_lmstudio",
        source_path=item.target_path,
        target_path=item.source_path,
        link_path=item.source_path,
    )
    applied = False
    if not dry_run:
        _apply_link_into_lmstudio(op, replace_link=replace_link)
        applied = True
    return LinkResult(scan=scan, item=item, operation=op, applied=applied)


def _find_link_item(
    scan: LmStudioScanResult,
    model: str,
    *,
    filename: str | None = None,
) -> LmStudioItem:
    matches = [
        item for item in scan.items
        if item.status == "in_shelf"
        and item.target_path is not None
        and _model_matches(item, model, scan, filename=filename)
    ]
    if not matches:
        raise ValueError(f"no in_shelf model matched: {model}")
    if len(matches) > 1:
        choices = ", ".join(item.model for item in matches[:5])
        suffix = "" if len(matches) <= 5 else ", ..."
        raise ValueError(f"model matched multiple in_shelf entries; add --filename or --format: {choices}{suffix}")
    return matches[0]


def _model_matches(
    item: LmStudioItem,
    model: str,
    scan: LmStudioScanResult,
    *,
    filename: str | None = None,
) -> bool:
    candidates = {item.model}
    if item.publisher and item.repo:
        candidates.add(f"{item.publisher}/{item.repo}")
    try:
        candidates.add(str(item.source_path.relative_to(scan.lmstudio_root)))
    except ValueError:
        pass
    for shelf_root in scan.shelf_roots:
        try:
            candidates.add(str(item.target_path.relative_to(shelf_root)))
        except (AttributeError, ValueError):
            pass
    if model not in candidates:
        return False
    if filename is None:
        return True
    return item.source_path.name == filename or (item.target_path is not None and item.target_path.name == filename)


def build_migration_plan(
    scan: LmStudioScanResult,
    *,
    link: bool = False,
    model: str | None = None,
    filename: str | None = None,
) -> list[MigrationOperation]:
    operations: list[MigrationOperation] = []
    items = _filter_migration_items(scan, model=model, filename=filename)
    for item in items:
        if link:
            operations.append(
                MigrationOperation(
                    action="link_into_shelf",
                    source_path=item.source_path,
                    target_path=item.target_path,
                    link_path=item.target_path,
                )
            )
        else:
            operations.append(
                MigrationOperation(
                    action="move_to_shelf_and_link_back",
                    source_path=item.source_path,
                    target_path=item.target_path,
                    link_path=item.source_path,
                )
            )
    return operations


def _filter_migration_items(
    scan: LmStudioScanResult,
    *,
    model: str | None = None,
    filename: str | None = None,
) -> list[LmStudioItem]:
    items = [
        item for item in scan.items
        if item.status == "lmstudio_only" and item.target_path is not None
    ]
    if model is None:
        return items
    matches = [
        item for item in items
        if _model_matches(item, model, scan, filename=filename)
    ]
    if not matches:
        raise ValueError(f"no lmstudio_only model matched: {model}")
    if len(matches) > 1:
        choices = ", ".join(item.model for item in matches[:5])
        suffix = "" if len(matches) <= 5 else ", ..."
        raise ValueError(f"model matched multiple lmstudio_only entries; add --filename or --format: {choices}{suffix}")
    return matches


def _scan_items(
    root: Path,
    primary_shelf: Path,
    shelf_roots: list[Path],
) -> list[LmStudioItem]:
    items: list[LmStudioItem] = []
    seen_dirs: set[Path] = set()

    for publisher_dir in _iter_visible_dirs(root):
        for repo_dir in _iter_visible_dirs(publisher_dir):
            if repo_dir in seen_dirs:
                continue
            seen_dirs.add(repo_dir)
            items.extend(_scan_repo_dir(root, publisher_dir.name, repo_dir.name, repo_dir, primary_shelf, shelf_roots))

    loose_ggufs = [
        p for p in root.rglob("*.gguf")
        if _is_scannable_file(p) and len(_safe_relative_parts(root, p)) < 3
    ]
    for path in sorted(loose_ggufs, key=lambda p: str(p).lower()):
        items.append(
            LmStudioItem(
                status="ambiguous",
                format="gguf",
                publisher=None,
                repo=None,
                source_path=path,
                target_path=None,
                size_bytes=_path_size(path),
                is_link=path.is_symlink(),
                link_target=_readlink(path),
                reason="GGUF file is not under <publisher>/<repo>/",
            )
        )

    _add_shelf_only_items(items, root, primary_shelf, shelf_roots)
    return sorted(items, key=lambda item: (item.status, item.model.lower(), str(item.source_path).lower()))


def _scan_repo_dir(
    root: Path,
    publisher: str,
    repo: str,
    repo_dir: Path,
    primary_shelf: Path,
    shelf_roots: list[Path],
) -> list[LmStudioItem]:
    items: list[LmStudioItem] = []
    ggufs = sorted(
        (p for p in repo_dir.glob("*.gguf") if _is_scannable_file(p)),
        key=lambda p: p.name.lower(),
    )
    for gguf in ggufs:
        target = shelf_path_gguf(primary_shelf, f"{publisher}/{repo}", _quant_from_gguf_name(gguf.name))
        if target.name != gguf.name:
            target = primary_shelf / "gguf" / publisher / repo / gguf.name
        target = _target_for_scan(target, primary_shelf, shelf_roots)
        items.append(_classify_item("gguf", publisher, repo, gguf, target, shelf_roots))

    if _looks_like_snapshot_dir(repo_dir):
        fmt = _detect_snapshot_format(publisher, repo, repo_dir)
        if fmt is None:
            target = None
            status = "ambiguous"
            reason = "directory could be MLX or safetensors"
        else:
            target = _target_for_scan(
                shelf_path_snapshot(primary_shelf, f"{publisher}/{repo}", fmt),
                primary_shelf,
                shelf_roots,
            )
            status = _status_for_path(repo_dir, target, shelf_roots)
            reason = None
        items.append(
            LmStudioItem(
                status=status,
                format=fmt,
                publisher=publisher,
                repo=repo,
                source_path=repo_dir,
                target_path=target,
                size_bytes=_path_size(repo_dir),
                is_link=repo_dir.is_symlink(),
                link_target=_readlink(repo_dir),
                reason=reason,
            )
        )
    elif not ggufs and _has_modelish_files(repo_dir):
        items.append(
            LmStudioItem(
                status="unsupported",
                format=None,
                publisher=publisher,
                repo=repo,
                source_path=repo_dir,
                target_path=None,
                size_bytes=_path_size(repo_dir),
                is_link=repo_dir.is_symlink(),
                link_target=_readlink(repo_dir),
                reason="model directory is missing config.json or supported weights",
            )
        )

    return items


def _classify_item(
    fmt: str,
    publisher: str,
    repo: str,
    source: Path,
    target: Path,
    shelf_roots: list[Path],
) -> LmStudioItem:
    return LmStudioItem(
        status=_status_for_path(source, target, shelf_roots),
        format=fmt,
        publisher=publisher,
        repo=repo,
        source_path=source,
        target_path=target,
        size_bytes=_path_size(source),
        is_link=source.is_symlink(),
        link_target=_readlink(source),
    )


def _status_for_path(source: Path, target: Path, shelf_roots: list[Path]) -> str:
    if source.is_symlink():
        link_target = source.resolve()
        if any(_is_relative_to(link_target, shelf.resolve()) for shelf in shelf_roots if shelf.exists()):
            return "lmstudio_link"
    if target.exists():
        if target.is_symlink():
            try:
                if source.exists() and target.resolve() == source.resolve():
                    return "shelf_link"
            except OSError:
                pass
        return "duplicated"
    return "lmstudio_only"


def _target_for_scan(target: Path, primary_shelf: Path, shelf_roots: list[Path]) -> Path:
    try:
        rel = target.relative_to(primary_shelf)
    except ValueError:
        return target
    for shelf_root in shelf_roots:
        candidate = shelf_root / rel
        if candidate.exists() or candidate.is_symlink():
            return candidate
    return target


def _add_shelf_only_items(
    items: list[LmStudioItem],
    lmstudio_root: Path,
    primary_shelf: Path,
    shelf_roots: list[Path],
) -> None:
    seen_targets = {
        item.target_path.resolve()
        for item in items
        if item.target_path is not None and item.target_path.exists()
    }
    for shelf_root in shelf_roots:
        if not shelf_root.is_dir():
            continue
        for item in _iter_shelf_items(lmstudio_root, primary_shelf, shelf_root):
            if item.target_path is None:
                continue
            try:
                key = item.target_path.resolve()
            except OSError:
                key = item.target_path
            if key in seen_targets:
                continue
            if item.source_path.exists() or item.source_path.is_symlink():
                continue
            seen_targets.add(key)
            items.append(item)


def _iter_shelf_items(
    lmstudio_root: Path,
    primary_shelf: Path,
    shelf_root: Path,
) -> list[LmStudioItem]:
    items: list[LmStudioItem] = []
    gguf_root = shelf_root / "gguf"
    if gguf_root.is_dir():
        for path in sorted(gguf_root.glob("*/*/*.gguf"), key=lambda p: str(p).lower()):
            if not _is_scannable_file(path):
                continue
            rel = path.relative_to(gguf_root)
            publisher, repo = rel.parts[0], rel.parts[1]
            source = lmstudio_root / publisher / repo / path.name
            target = shelf_root / "gguf" / publisher / repo / path.name
            items.append(
                LmStudioItem(
                    status="in_shelf",
                    format="gguf",
                    publisher=publisher,
                    repo=repo,
                    source_path=source,
                    target_path=target,
                    size_bytes=_path_size(path),
                    is_link=path.is_symlink(),
                    link_target=_readlink(path),
                )
            )
    for fmt in ("mlx", "safetensors"):
        fmt_root = shelf_root / fmt
        if not fmt_root.is_dir():
            continue
        for publisher_dir in _iter_visible_dirs(fmt_root):
            for repo_dir in _iter_visible_dirs(publisher_dir):
                if not _looks_like_snapshot_dir(repo_dir):
                    continue
                publisher = publisher_dir.name
                repo = repo_dir.name
                source = lmstudio_root / publisher / repo
                target = shelf_root / fmt / publisher / repo
                items.append(
                    LmStudioItem(
                        status="in_shelf",
                        format=fmt,
                        publisher=publisher,
                        repo=repo,
                        source_path=source,
                        target_path=target,
                        size_bytes=_path_size(repo_dir),
                        is_link=repo_dir.is_symlink(),
                        link_target=_readlink(repo_dir),
                    )
                )
    return items


def _apply_operation(op: MigrationOperation) -> None:
    if op.status != "planned":
        return
    if op.target_path.exists() or op.target_path.is_symlink():
        op.status = "skipped"
        op.reason = "target already exists"
        return
    op.target_path.parent.mkdir(parents=True, exist_ok=True)

    if op.action == "link_into_shelf":
        op.target_path.symlink_to(op.source_path)
        op.status = "applied"
        return

    if op.action != "move_to_shelf_and_link_back":
        op.status = "error"
        op.reason = f"unknown action: {op.action}"
        return

    try:
        op.source_path.rename(op.target_path)
    except OSError:
        if op.source_path.is_dir():
            shutil.copytree(op.source_path, op.target_path, symlinks=True)
        else:
            shutil.copy2(op.source_path, op.target_path)
        if _path_size(op.source_path) != _path_size(op.target_path):
            op.status = "error"
            op.reason = "copy verification failed"
            return
        if op.source_path.is_dir():
            shutil.rmtree(op.source_path)
        else:
            op.source_path.unlink()
    op.source_path.symlink_to(op.target_path, target_is_directory=op.target_path.is_dir())
    op.status = "applied"


def _apply_link_into_lmstudio(op: MigrationOperation, *, replace_link: bool = False) -> None:
    if op.status != "planned":
        return
    if op.source_path is None or not op.source_path.exists():
        op.status = "error"
        op.reason = "shelf target does not exist"
        return
    if op.target_path.exists() or op.target_path.is_symlink():
        if not op.target_path.is_symlink():
            op.status = "skipped"
            op.reason = "LM Studio path already exists"
            return
        if not replace_link:
            op.status = "skipped"
            op.reason = "LM Studio symlink already exists; pass --replace-link to replace it"
            return
        op.target_path.unlink()
    op.target_path.parent.mkdir(parents=True, exist_ok=True)
    op.target_path.symlink_to(op.source_path, target_is_directory=op.source_path.is_dir())
    op.status = "applied"


def _looks_like_snapshot_dir(path: Path) -> bool:
    return (path / "config.json").is_file() and any(
        p.is_file() and p.suffix == ".safetensors"
        for p in path.rglob("*.safetensors")
        if _is_scannable_file(p)
    )


def _detect_snapshot_format(publisher: str, repo: str, path: Path) -> str | None:
    tokens = f"{publisher} {repo}".lower().replace("_", "-").split("-")
    if publisher.lower() == "mlx-community" or "mlx" in tokens:
        return "mlx"
    if any(p.name in {"model.safetensors", "model-00001-of-00001.safetensors"} for p in path.glob("*.safetensors")):
        return "safetensors"
    if any(path.glob("*.safetensors.index.json")):
        return "safetensors"
    return "safetensors"


def _quant_from_gguf_name(name: str) -> str:
    stem = name.removesuffix(".gguf")
    parts = stem.split("-")
    if not parts:
        return stem
    last = parts[-1]
    if last.upper().startswith("Q") or last.lower() in {"f16", "bf16", "iq4_xs"}:
        return last
    return last


def _configured_lmstudio_model_roots(home: Path | None = None) -> list[Path]:
    if home is None:
        home = Path.home()
    config_dirs = [
        home / "Library" / "Application Support" / "LM Studio",
        home / ".config" / "LM Studio",
        home / ".lmstudio",
    ]
    out: list[Path] = []
    for config_dir in config_dirs:
        if not config_dir.is_dir():
            continue
        for path in sorted(config_dir.rglob("*.json"), key=lambda p: str(p).lower()):
            if not _is_scannable_file(path):
                continue
            out.extend(_paths_from_json_file(path))
    return _dedupe_paths(out)


def _paths_from_json_file(path: Path) -> list[Path]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return []
    paths: list[Path] = []

    def walk(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                walk(child, str(child_key))
            return
        if isinstance(value, list):
            for child in value:
                walk(child, key)
            return
        if not isinstance(value, str):
            return
        normalized_key = key.lower()
        if "model" not in normalized_key:
            return
        if not any(token in normalized_key for token in ("path", "dir", "root", "folder", "location")):
            return
        candidate = _normalize_lmstudio_root(Path(value).expanduser())
        paths.append(candidate)

    walk(data)
    return paths


def _roots_from_lms_inventory() -> list[Path]:
    try:
        completed = subprocess.run(
            ["lms", "ls", "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return []
    paths: list[Path] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                walk(child)
            return
        if isinstance(value, list):
            for child in value:
                walk(child)
            return
        if not isinstance(value, str):
            return
        path = Path(value).expanduser()
        if path.exists():
            parts = path.parts
            if "models" in parts:
                idx = parts.index("models")
                paths.append(Path(*parts[: idx + 1]))

    walk(data)
    return _dedupe_paths(paths)


def _iter_visible_dirs(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(
        (
            p for p in path.iterdir()
            if p.is_dir()
            and not p.name.startswith(".")
            and p.name not in IGNORED_DIR_NAMES
        ),
        key=lambda p: p.name.lower(),
    )


def _is_scannable_file(path: Path) -> bool:
    return (
        not path.name.startswith(".")
        and not any(part in IGNORED_DIR_NAMES for part in path.parts)
        and not path.name.lower().endswith(PARTIAL_SUFFIXES)
    )


def _has_modelish_files(path: Path) -> bool:
    return any(
        p.is_file()
        and _is_scannable_file(p)
        and p.suffix.lower() in {".gguf", ".safetensors", ".bin", ".json"}
        for p in path.rglob("*")
    )


def _path_size(path: Path) -> int:
    try:
        if path.is_file() or path.is_symlink():
            return path.stat().st_size
        total = 0
        for child in path.rglob("*"):
            if child.is_file() and _is_scannable_file(child):
                total += child.stat().st_size
        return total
    except OSError:
        return 0


def _readlink(path: Path) -> Path | None:
    if not path.is_symlink():
        return None
    try:
        return Path(os.readlink(path))
    except OSError:
        return None


def _safe_relative_parts(root: Path, path: Path) -> tuple[str, ...]:
    try:
        return path.relative_to(root).parts
    except ValueError:
        return ()


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        expanded = path.expanduser()
        key = str(expanded)
        if key in seen:
            continue
        seen.add(key)
        out.append(expanded)
    return out


def _normalize_lmstudio_root(path: Path) -> Path:
    """Return the LM Studio library root for paths under a `models` directory.

    LM Studio settings can contain either the configured models directory or a
    recently used model/repo path below it. Scanning from the deeper path loses
    the publisher/repo context, so collapse those paths back to the enclosing
    `models` directory.
    """
    parts = path.parts
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "models":
            return Path(*parts[: i + 1])
    if path.name != "models" and (path / "models").is_dir():
        return path / "models"
    return path


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
