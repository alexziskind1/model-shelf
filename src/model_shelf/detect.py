"""Detect plausible Model Shelf locations on the user's machine.

Scans every mount root (/Volumes on macOS, /mnt and /media on Linux) for
external drives, plus the user's home directory for an existing or default
internal shelf. Returns a ranked list of candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from model_shelf.mounts import iter_volumes


@dataclass
class StorageCandidate:
    path: Path           # full shelf_root path
    label: str           # short display name (volume name or "internal")
    existing: bool       # does this path already contain a ModelShelf folder?
    is_external: bool    # external drive vs internal


def detect_storage_candidates(
    volumes_dir: Path | None = None,
    home: Path | None = None,
) -> list[StorageCandidate]:
    """Return ranked candidates.

    Order: external drives with existing shelves first, then external drives
    (new-shelf defaults), then the internal default.

    `volumes_dir` restricts the scan to a single mount root (used by tests).
    """
    if home is None:
        home = Path.home()

    candidates: list[StorageCandidate] = []

    roots = None if volumes_dir is None else [volumes_dir]
    for vol in iter_volumes(roots):
        shelf_path = vol / "ModelShelf" / "models"
        candidates.append(
            StorageCandidate(
                path=shelf_path,
                label=vol.name,
                existing=shelf_path.is_dir(),
                is_external=True,
            )
        )

    internal = home / ".cache" / "model-shelf" / "models"
    candidates.append(
        StorageCandidate(
            path=internal,
            label="internal",
            existing=internal.is_dir(),
            is_external=False,
        )
    )

    # Sort: existing first (preserves prior order within group).
    candidates.sort(key=lambda c: (not c.existing, not c.is_external))
    return candidates
