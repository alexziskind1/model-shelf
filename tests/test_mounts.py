"""Tests for multi-platform mount-root discovery.

macOS mounts volumes under /Volumes; Linux uses /mnt (fstab / manual mounts,
where a NAS share usually lands) and /media (auto-mounted removable media,
often nested one level under the current user).
"""

import getpass
from pathlib import Path

import model_shelf.mounts as mounts_mod
from model_shelf.mounts import iter_volumes, mount_roots, split_mount_path
from model_shelf.relocate import relocate_shelf
from model_shelf.resolver import discover_primary_shelf


# --- mount_roots -----------------------------------------------------------

def test_mount_roots_covers_macos_and_linux():
    roots = [str(p) for p in mount_roots()]
    assert "/Volumes" in roots
    assert "/mnt" in roots
    assert "/media" in roots


def test_mount_roots_includes_per_user_media_dir():
    """udisks2 mounts removable media at /media/<user>/<label>."""
    roots = [str(p) for p in mount_roots()]
    assert f"/media/{getpass.getuser()}" in roots


def test_volumes_is_preferred_over_linux_roots():
    """Order is preference order: a macOS shelf wins over a /mnt one."""
    roots = [str(p) for p in mount_roots()]
    assert roots.index("/Volumes") < roots.index("/mnt")


# --- split_mount_path ------------------------------------------------------

def test_split_mount_path_volumes():
    assert split_mount_path(Path("/Volumes/Lexar/ModelShelf/models")) == (
        Path("/Volumes"),
        "Lexar",
        "ModelShelf/models",
    )


def test_split_mount_path_mnt():
    assert split_mount_path(Path("/mnt/nas/ModelShelf/models")) == (
        Path("/mnt"),
        "nas",
        "ModelShelf/models",
    )


def test_split_mount_path_prefers_longest_root():
    """/media/<user>/Drive must resolve against /media/<user>, not /media."""
    user = getpass.getuser()
    root, volume, subpath = split_mount_path(
        Path(f"/media/{user}/MyDrive/ModelShelf/models")
    )
    assert root == Path(f"/media/{user}")
    assert volume == "MyDrive"
    assert subpath == "ModelShelf/models"


def test_split_mount_path_volume_itself_has_empty_subpath():
    assert split_mount_path(Path("/mnt/nas")) == (Path("/mnt"), "nas", "")


def test_split_mount_path_returns_none_outside_mount_roots():
    assert split_mount_path(Path("/opt/models")) is None


def test_split_mount_path_returns_none_for_bare_root():
    assert split_mount_path(Path("/media")) is None


# --- iter_volumes ----------------------------------------------------------

def test_iter_volumes_walks_every_root_in_order(tmp_path, monkeypatch):
    volumes = tmp_path / "Volumes"
    volumes.mkdir()
    mnt = tmp_path / "mnt"
    mnt.mkdir()
    (volumes / "MacDrive").mkdir()
    (mnt / "nas").mkdir()
    monkeypatch.setattr(mounts_mod, "mount_roots", lambda: [volumes, mnt])

    assert [v.name for v in iter_volumes()] == ["MacDrive", "nas"]


def test_iter_volumes_sorts_alphabetically_within_a_root(tmp_path, monkeypatch):
    mnt = tmp_path / "mnt"
    mnt.mkdir()
    (mnt / "b-share").mkdir()
    (mnt / "A-share").mkdir()
    monkeypatch.setattr(mounts_mod, "mount_roots", lambda: [mnt])

    assert [v.name for v in iter_volumes()] == ["A-share", "b-share"]


def test_iter_volumes_skips_symlinks(tmp_path, monkeypatch):
    """macOS's /Volumes/Macintosh HD -> / must never become a candidate."""
    volumes = tmp_path / "Volumes"
    volumes.mkdir()
    (tmp_path / "real-root").mkdir()
    (volumes / "Macintosh HD").symlink_to(tmp_path / "real-root")
    (volumes / "Lexar").mkdir()
    monkeypatch.setattr(mounts_mod, "mount_roots", lambda: [volumes])

    assert [v.name for v in iter_volumes()] == ["Lexar"]


def test_iter_volumes_ignores_roots_that_do_not_exist(tmp_path, monkeypatch):
    """A Mac has no /mnt and a Linux box has no /Volumes — neither should error."""
    present = tmp_path / "mnt"
    present.mkdir()
    (present / "nas").mkdir()
    monkeypatch.setattr(
        mounts_mod, "mount_roots", lambda: [tmp_path / "absent", present]
    )

    assert [v.name for v in iter_volumes()] == ["nas"]


# --- integration: discovery and relocation across roots --------------------

def test_discover_primary_shelf_finds_a_linux_mnt_shelf(tmp_path, monkeypatch):
    """The whole point: a NAS mounted at /mnt/nas is auto-discovered."""
    mnt = tmp_path / "mnt"
    shelf = mnt / "nas" / "ModelShelf" / "models"
    shelf.mkdir(parents=True)
    monkeypatch.setattr(mounts_mod, "mount_roots", lambda: [mnt])

    assert discover_primary_shelf(home=tmp_path / "home") == shelf


def test_discover_primary_shelf_falls_back_to_internal_on_linux(tmp_path, monkeypatch):
    mnt = tmp_path / "mnt"
    mnt.mkdir()
    (mnt / "unrelated").mkdir()
    home = tmp_path / "home"
    monkeypatch.setattr(mounts_mod, "mount_roots", lambda: [mnt])

    assert discover_primary_shelf(home=home) == (
        home / ".cache" / "model-shelf" / "models"
    )


def test_relocate_finds_shelf_that_moved_to_another_root(tmp_path, monkeypatch):
    """Shelf pinned under /Volumes but now mounted under /mnt still resolves."""
    volumes = tmp_path / "Volumes"
    volumes.mkdir()
    mnt = tmp_path / "mnt"
    shelf = mnt / "nas" / "ModelShelf" / "models"
    shelf.mkdir(parents=True)
    monkeypatch.setattr(mounts_mod, "mount_roots", lambda: [volumes, mnt])

    configured = volumes / "OldDrive" / "ModelShelf" / "models"  # not mounted
    assert relocate_shelf(configured) == shelf
