"""Where mounted volumes live, per platform.

macOS puts every mounted volume under /Volumes. Linux uses /mnt for manual
and fstab mounts — the usual home for a NAS share — and /media for
auto-mounted removable media, often nested one level under the current
user (e.g. /media/alex/MyDrive), so that per-user directory is itself a root.

Model Shelf trusts the `<volume>/ModelShelf/models` folder convention rather
than tracking drive identity, so all it needs to know is which directories
hold volumes.
"""

from __future__ import annotations

import getpass
from collections.abc import Iterator
from pathlib import Path

# Ordered by preference. Discovery walks these in order, so on a machine that
# somehow has shelves under both, /Volumes wins over /mnt.
_BASE_ROOTS = ("/Volumes", "/mnt", "/media")


def mount_roots() -> list[Path]:
    """Directories whose immediate children are mounted volumes."""
    roots = [Path(r) for r in _BASE_ROOTS]
    try:
        user = getpass.getuser()
    except Exception:
        # No passwd entry and no LOGNAME/USER — happens in some containers.
        user = ""
    if user:
        roots.append(Path("/media") / user)
    return roots


def iter_volumes(roots: list[Path] | None = None) -> Iterator[Path]:
    """Yield mounted volume directories in a deterministic order.

    Roots are walked in `mount_roots()` order and each root's children sorted
    alphabetically. Symlinks are skipped so macOS's `/Volumes/Macintosh HD -> /`
    never becomes a candidate.
    """
    if roots is None:
        roots = mount_roots()
    for root in roots:
        if not root.is_dir():
            continue
        try:
            entries = sorted(root.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            continue
        for vol in entries:
            if vol.is_symlink() or not vol.is_dir():
                continue
            yield vol


def split_mount_path(path: Path) -> tuple[Path, str, str] | None:
    """Split `<root>/<volume>/<sub...>` for a known mount root.

    Returns (root, volume_name, subpath) — subpath is "" when `path` is the
    volume itself — or None if `path` isn't under a mount root at all.

    Longest root wins, so /media/<user>/Drive/... resolves against
    /media/<user> rather than /media.
    """
    for root in sorted(mount_roots(), key=lambda p: len(p.parts), reverse=True):
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if not rel.parts:
            continue
        return root, rel.parts[0], "/".join(rel.parts[1:])
    return None
