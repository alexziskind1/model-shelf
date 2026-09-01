"""Find the shelf even if its drive was renamed or remounted under a different name.

Convention-based: if your shelf was at `/Volumes/Lexar/ModelShelf/models` and
that path no longer resolves, scan every mounted volume — `/Volumes/*` on
macOS, `/mnt/*` and `/media/*` on Linux — for one whose `ModelShelf/models`
directory exists, and use that. First match wins.

No identity tracking, no marker files — Model Shelf trusts the folder
convention. The simplest thing that could possibly work.
"""

from __future__ import annotations

from pathlib import Path

from model_shelf.mounts import iter_volumes, split_mount_path


def _extract_volume_subpath(shelf_root: Path) -> tuple[Path, str] | None:
    """If shelf_root is `<root>/<volume>/<sub...>`, return (root, sub).

    Returns None when shelf_root isn't under a mount root, or names a volume
    with no subpath below it (nothing to match other drives against).
    """
    split = split_mount_path(shelf_root)
    if split is None:
        return None
    root, _volume, subpath = split
    if not subpath:
        return None
    return root, subpath


def find_shelf_at_subpath(
    subpath: str,
    volumes_dir: Path | None = None,
) -> Path | None:
    """Return the first mounted volume that has `<vol>/<subpath>` as a directory.

    `volumes_dir` restricts the scan to a single root (used by tests);
    by default every root in `mount_roots()` is searched.
    """
    roots = None if volumes_dir is None else [volumes_dir]
    for vol in iter_volumes(roots):
        candidate = vol / subpath
        if candidate.is_dir():
            return candidate
    return None


def relocate_shelf(shelf_root: Path) -> Path:
    """Return the effective shelf_root, possibly relocated to a renamed/swapped drive.

    - If shelf_root exists, return it.
    - If shelf_root is under a mount root and some other mounted volume has the
      same subpath, return that volume's path. The search spans every mount
      root, so a shelf that moves between /Volumes and /mnt still resolves.
    - Otherwise return shelf_root unchanged (downstream code will surface the
      appropriate "not mounted" / "not initialized" error).
    """
    if shelf_root.is_dir():
        return shelf_root
    extracted = _extract_volume_subpath(shelf_root)
    if extracted is None:
        return shelf_root
    _root, subpath = extracted
    found = find_shelf_at_subpath(subpath)
    return found if found is not None else shelf_root
