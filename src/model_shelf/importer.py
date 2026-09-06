"""Plan and execute imports of existing models onto the curated shelf.

The core knows nothing about *where* models come from — adapters in
``sources.py`` turn a source location into ``ImportCandidate`` records, and
this module maps each candidate to its canonical shelf target and transfers
it (copy by default, move on request). No symlinks: a moved/copied model is
a plain file or directory the shelf owns.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from model_shelf.resolver import (
    _looks_like_model_dir,  # intentional reuse: stays the single source of "is this a complete model dir?"
    shelf_path_gguf,
    shelf_path_snapshot,
)


@dataclass
class ImportCandidate:
    source_path: Path        # file (gguf) or directory (mlx/safetensors)
    repo_id: str | None      # publisher/repo; None means it must be bound
    format: str              # "gguf" | "mlx" | "safetensors"
    quant: str | None        # gguf only
    size_bytes: int
    source_label: str        # "dir" | "hf" | "ollama" | "lmstudio"


@dataclass
class ImportOperation:
    candidate: ImportCandidate
    target_path: Path | None  # None when status == "needs_binding"
    action: str               # "copy" | "move" | "skip"
    status: str               # pending|already_in_shelf|conflict|needs_binding|done|failed
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "repo_id": self.candidate.repo_id,
            "format": self.candidate.format,
            "source_label": self.candidate.source_label,
            "source_path": str(self.candidate.source_path),
            "target_path": str(self.target_path) if self.target_path else None,
            "action": self.action,
            "status": self.status,
            "size_bytes": self.candidate.size_bytes,
            "note": self.note,
        }


@dataclass
class ImportReport:
    operations: list[ImportOperation] = field(default_factory=list)
    bytes_imported: int = 0
    bytes_reclaimable: int = 0

    def to_dict(self) -> dict:
        return {
            "operations": [op.to_dict() for op in self.operations],
            "bytes_imported": self.bytes_imported,
            "bytes_reclaimable": self.bytes_reclaimable,
        }


def _target_for(candidate: ImportCandidate, shelf_root: Path) -> Path:
    if candidate.format == "gguf":
        return shelf_path_gguf(shelf_root, candidate.repo_id, candidate.quant or "")
    return shelf_path_snapshot(shelf_root, candidate.repo_id, candidate.format)


def plan_import(
    candidates: list[ImportCandidate],
    shelf_root: Path,
) -> list[ImportOperation]:
    """Map each candidate to a shelf target and decide copy vs. skip.

    Never plans an overwrite: an existing complete target becomes
    ``already_in_shelf``; an existing incomplete/size-mismatched target
    becomes ``conflict`` (skipped, left untouched).
    """
    ops: list[ImportOperation] = []
    for c in candidates:
        if c.repo_id is None or (c.format == "gguf" and c.quant is None):
            note = (
                "no repo_id could be inferred — bind it interactively"
                if c.repo_id is None
                else "gguf candidate has no quant token — bind it interactively"
            )
            ops.append(ImportOperation(c, None, "skip", "needs_binding", note))
            continue

        target = _target_for(c, shelf_root)

        if c.format == "gguf":
            if target.is_file():
                same = target.stat().st_size == c.size_bytes
                ops.append(ImportOperation(
                    c, target, "skip",
                    "already_in_shelf" if same else "conflict",
                    "" if same else "target exists with a different size",
                ))
                continue
        else:
            if target.exists():
                complete = _looks_like_model_dir(target)
                ops.append(ImportOperation(
                    c, target, "skip",
                    "already_in_shelf" if complete else "conflict",
                    "" if complete else "target exists but looks incomplete",
                ))
                continue

        ops.append(ImportOperation(c, target, "copy", "pending"))
    return ops


def _real(path: Path) -> Path:
    """Resolve symlinks to the real on-disk path (HF cache stores blobs as symlinks)."""
    try:
        return path.resolve()
    except OSError:
        return path


def _copy_into(src: Path, dst: Path) -> None:
    """Copy real content of src to dst, following symlinks, for file or directory."""
    real = _real(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if real.is_dir():
        # symlinks=False → dereference any inner symlinks (HF blob links) into real files
        shutil.copytree(real, dst, symlinks=False, dirs_exist_ok=True)
    else:
        shutil.copy2(real, dst)


def _verify(dst: Path, fmt: str, expected_bytes: int | None = None) -> bool:
    """A transferred target is valid if the shelf would treat it as a real model.

    For gguf, also reject a truncated copy by comparing against the expected
    byte count when known (copy preserves size, so a mismatch means corruption).
    """
    if fmt == "gguf":
        if not (dst.is_file() and dst.stat().st_size > 0):
            return False
        return expected_bytes is None or dst.stat().st_size == expected_bytes
    return _looks_like_model_dir(dst)


def _remove_partial(target: Path) -> None:
    """Best-effort cleanup of a half-written target after a failed transfer."""
    try:
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        elif target.exists() or target.is_symlink():
            target.unlink(missing_ok=True)
    except OSError:
        pass


def execute_import(
    operations: list[ImportOperation],
    *,
    action: str = "copy",
    dry_run: bool = True,
) -> ImportReport:
    """Apply pending operations. copy (default) or move; dry_run plans only.

    move = copy real content -> verify -> delete the original source. The
    source is only removed after the target verifies complete, so an
    interrupted move never destroys the only copy. Never overwrites: a target
    that exists at execute time (race with planning) is left untouched as a
    conflict. A failed transfer cleans up its partial target so a re-run does
    not mistake it for a completed import.

    Note: source_path is expected to be a real file or directory (the adapters
    never yield a directory symlink), so move deletion is unambiguous.
    """
    report = ImportReport(operations=operations)

    for op in operations:
        if op.status != "pending" or op.target_path is None:
            continue
        op.action = action

        if dry_run:
            report.bytes_imported += op.candidate.size_bytes
            if action == "move":
                report.bytes_reclaimable += op.candidate.size_bytes
            continue

        # Never overwrite: if the target appeared between plan and execute, skip.
        if op.target_path.exists():
            op.status = "conflict"
            op.note = "target appeared between plan and execute"
            continue

        try:
            _copy_into(op.candidate.source_path, op.target_path)
            if not _verify(op.target_path, op.candidate.format, op.candidate.size_bytes):
                raise RuntimeError("post-copy verification failed (incomplete target)")
            if action == "move":
                src = op.candidate.source_path
                if src.is_dir() and not src.is_symlink():
                    shutil.rmtree(src)
                else:
                    src.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001 — record, don't abort the batch
            _remove_partial(op.target_path)
            op.status = "failed"
            op.note = (str(e).strip().splitlines() or [type(e).__name__])[-1]
            continue

        op.status = "done"
        report.bytes_imported += op.candidate.size_bytes
        if action == "move":
            report.bytes_reclaimable += op.candidate.size_bytes

    return report
