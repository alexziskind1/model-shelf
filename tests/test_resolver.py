from pathlib import Path

import pytest

from model_shelf.resolver import (
    Config,
    ShelfNotInitializedError,
    StorageNotAvailableError,
    check_storage_available,
    detect_format,
    hf_filename,
    init_shelf,
    list_shelf_candidates,
    resolve_model,
    shelf_path_gguf,
    shelf_path_snapshot,
)


# --- filename helpers ------------------------------------------------------

def test_hf_filename_preserves_case():
    assert hf_filename("Qwen/Qwen3-14B-GGUF", "Q4_K_M") == "Qwen3-14B-Q4_K_M.gguf"


def test_hf_filename_doesnt_double_append_quant():
    """Repos that already carry the quant in their name (e.g. -Q4_K_M-GGUF)
    should not get the quant appended again."""
    assert (
        hf_filename("rippertnt/Qwen3-0.6B-Q4_K_M-GGUF", "Q4_K_M")
        == "Qwen3-0.6B-Q4_K_M.gguf"
    )


def test_shelf_path_gguf_uses_publisher_repo_nesting():
    root = Path("/shelf")
    assert (
        shelf_path_gguf(root, "Qwen/Qwen3-14B-GGUF", "Q4_K_M")
        == root / "gguf" / "Qwen" / "Qwen3-14B-GGUF" / "Qwen3-14B-Q4_K_M.gguf"
    )


def test_shelf_path_snapshot_uses_publisher_repo_nesting():
    root = Path("/shelf")
    assert (
        shelf_path_snapshot(root, "mlx-community/Qwen3-14B-4bit", "mlx")
        == root / "mlx" / "mlx-community" / "Qwen3-14B-4bit"
    )


def test_repo_id_without_slash_errors():
    root = Path("/shelf")
    with pytest.raises(ValueError, match="publisher/repo"):
        shelf_path_gguf(root, "no-slash", "Q4_K_M")


# --- format detection -------------------------------------------------------

def test_detect_format_gguf():
    assert detect_format("Qwen/Qwen3-14B-GGUF") == "gguf"
    assert detect_format("bartowski/Qwen_Qwen3-14B-gguf") == "gguf"


def test_detect_format_mlx():
    assert detect_format("mlx-community/Qwen3-14B-4bit") == "mlx"
    assert detect_format("Qwen/Qwen3-14B-MLX") == "mlx"
    assert detect_format("Qwen/Qwen3-4B-MLX-4bit") == "mlx"
    assert detect_format("lmstudio-community/Qwen3-4B-MLX-4bit") == "mlx"


def test_detect_format_safetensors_default():
    assert detect_format("Qwen/Qwen3-14B") == "safetensors"
    assert detect_format("meta-llama/Llama-3.1-8B-Instruct") == "safetensors"


# --- resolver: gguf path ----------------------------------------------------

def _config(tmp_path: Path, *, allow_downloads: bool = False) -> Config:
    shelf = tmp_path / "shelf"
    shelf.mkdir(parents=True, exist_ok=True)
    return Config(shelf_root=shelf, allow_downloads=allow_downloads)


def test_gguf_shelf_hit(tmp_path: Path):
    cfg = _config(tmp_path)
    target = cfg.shelf_root / "gguf" / "Qwen" / "Qwen3-14B-GGUF"
    target.mkdir(parents=True)
    f = target / "Qwen3-14B-Q4_K_M.gguf"
    f.write_bytes(b"x")

    result = resolve_model(cfg, "Qwen/Qwen3-14B-GGUF", quant="Q4_K_M")

    assert result.status == "found"
    assert result.source == "local_shelf"
    assert result.format == "gguf"
    assert result.path == f


def test_gguf_missing_when_downloads_disabled(tmp_path: Path):
    cfg = _config(tmp_path)
    result = resolve_model(cfg, "Qwen/Qwen3-14B-GGUF", quant="Q4_K_M")

    assert result.status == "missing"
    assert result.format == "gguf"
    assert result.checks
    assert all(c["result"] == "miss" for c in result.checks)


def test_gguf_requires_quant(tmp_path: Path):
    cfg = _config(tmp_path)
    with pytest.raises(ValueError, match="quant"):
        resolve_model(cfg, "Qwen/Qwen3-14B-GGUF")


# --- resolver: mlx path -----------------------------------------------------

def test_mlx_shelf_hit_requires_config_json(tmp_path: Path):
    cfg = _config(tmp_path)
    target = cfg.shelf_root / "mlx" / "mlx-community" / "Qwen3-14B-4bit"
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}")
    (target / "model.safetensors").write_bytes(b"x")

    result = resolve_model(cfg, "mlx-community/Qwen3-14B-4bit")

    assert result.status == "found"
    assert result.source == "local_shelf"
    assert result.format == "mlx"
    assert result.path == target


def test_mlx_empty_dir_does_not_count(tmp_path: Path):
    cfg = _config(tmp_path)
    (cfg.shelf_root / "mlx" / "mlx-community" / "Qwen3-14B-4bit").mkdir(parents=True)
    # No config.json on purpose.

    result = resolve_model(cfg, "mlx-community/Qwen3-14B-4bit")

    assert result.status == "missing"


# --- resolver: safetensors path --------------------------------------------

def test_safetensors_shelf_hit(tmp_path: Path):
    cfg = _config(tmp_path)
    target = cfg.shelf_root / "safetensors" / "Qwen" / "Qwen3-14B"
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}")
    (target / "model.safetensors").write_bytes(b"x")

    result = resolve_model(cfg, "Qwen/Qwen3-14B")

    assert result.status == "found"
    assert result.format == "safetensors"
    assert result.path == target


def test_format_override(tmp_path: Path):
    cfg = _config(tmp_path)
    target = cfg.shelf_root / "safetensors" / "Qwen" / "Qwen3-14B-GGUF"
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}")
    (target / "model.safetensors").write_bytes(b"x")

    result = resolve_model(cfg, "Qwen/Qwen3-14B-GGUF", format="safetensors")

    assert result.status == "found"
    assert result.format == "safetensors"


# --- storage availability precheck -----------------------------------------

NONEXISTENT_VOLUME = Path("/Volumes/__model_shelf_test_xyz_does_not_exist__")


def test_storage_check_passes_for_initialized_shelf(tmp_path: Path):
    check_storage_available(Config(shelf_root=tmp_path))


def test_storage_check_errors_for_uninitialized_shelf(tmp_path: Path):
    cfg = Config(shelf_root=tmp_path / "nope")
    with pytest.raises(ShelfNotInitializedError, match="doesn't exist"):
        check_storage_available(cfg)


def test_storage_check_errors_for_unmounted_volume():
    cfg = Config(shelf_root=NONEXISTENT_VOLUME / "models")
    with pytest.raises(StorageNotAvailableError, match="not mounted"):
        check_storage_available(cfg)


def test_volume_check_runs_before_shelf_check():
    cfg = Config(shelf_root=NONEXISTENT_VOLUME / "models")
    with pytest.raises(StorageNotAvailableError) as exc:
        check_storage_available(cfg)
    assert "not mounted" in str(exc.value)


def test_resolve_model_errors_when_volume_unmounted():
    cfg = Config(shelf_root=NONEXISTENT_VOLUME / "models", allow_downloads=False)
    with pytest.raises(StorageNotAvailableError):
        resolve_model(cfg, "Qwen/Qwen3-14B-GGUF", quant="Q4_K_M")


# --- init_shelf -----------------------------------------------------------

def test_init_shelf_creates_subfolders(tmp_path: Path):
    cfg = Config(shelf_root=tmp_path / "newshelf")
    created = init_shelf(cfg)
    assert len(created) == 4
    assert (cfg.shelf_root / "gguf").is_dir()
    assert (cfg.shelf_root / "mlx").is_dir()
    assert (cfg.shelf_root / "safetensors").is_dir()


def test_init_shelf_is_idempotent(tmp_path: Path):
    cfg = Config(shelf_root=tmp_path / "newshelf")
    init_shelf(cfg)
    assert init_shelf(cfg) == []


def test_init_shelf_errors_on_unmounted_volume():
    cfg = Config(shelf_root=NONEXISTENT_VOLUME / "models")
    with pytest.raises(StorageNotAvailableError):
        init_shelf(cfg)


# --- multi-shelf lookup ---------------------------------------------------

def _patch_candidates(monkeypatch, paths: list):
    import model_shelf.resolver as resolver_mod
    monkeypatch.setattr(
        resolver_mod,
        "list_shelf_candidates",
        lambda cfg: list(paths),
    )


def test_gguf_lookup_hits_additional_shelf(tmp_path: Path, monkeypatch):
    primary = tmp_path / "primary"
    extra = tmp_path / "extra"
    primary.mkdir()
    target = extra / "gguf" / "Qwen" / "Qwen3-14B-GGUF"
    target.mkdir(parents=True)
    (target / "Qwen3-14B-Q4_K_M.gguf").write_bytes(b"x")
    _patch_candidates(monkeypatch, [primary, extra])

    cfg = Config(shelf_root=primary, allow_downloads=False)
    result = resolve_model(cfg, "Qwen/Qwen3-14B-GGUF", quant="Q4_K_M")

    assert result.status == "found"
    assert result.source == "local_shelf"
    assert result.path == target / "Qwen3-14B-Q4_K_M.gguf"
    assert [c["result"] for c in result.checks] == ["miss", "hit"]


def test_gguf_lookup_prefers_primary_when_both_have_file(tmp_path: Path, monkeypatch):
    primary = tmp_path / "primary"
    extra = tmp_path / "extra"
    for parent in (primary, extra):
        target = parent / "gguf" / "Qwen" / "Qwen3-14B-GGUF"
        target.mkdir(parents=True)
        (target / "Qwen3-14B-Q4_K_M.gguf").write_bytes(b"x")
    _patch_candidates(monkeypatch, [primary, extra])

    cfg = Config(shelf_root=primary, allow_downloads=False)
    result = resolve_model(cfg, "Qwen/Qwen3-14B-GGUF", quant="Q4_K_M")

    assert result.path == primary / "gguf" / "Qwen" / "Qwen3-14B-GGUF" / "Qwen3-14B-Q4_K_M.gguf"
    assert len(result.checks) == 1


def test_mlx_lookup_hits_additional_shelf(tmp_path: Path, monkeypatch):
    primary = tmp_path / "primary"
    extra = tmp_path / "extra"
    primary.mkdir()
    target = extra / "mlx" / "mlx-community" / "Qwen3-14B-4bit"
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}")
    (target / "model.safetensors").write_bytes(b"x")
    _patch_candidates(monkeypatch, [primary, extra])

    cfg = Config(shelf_root=primary, allow_downloads=False)
    result = resolve_model(cfg, "mlx-community/Qwen3-14B-4bit")

    assert result.status == "found"
    assert result.path == target


# --- regression: _looks_like_model_dir partial-download checks ---------


def test_partial_dir_not_a_hit(tmp_path: Path):
    """Directory with config.json but no weight files must NOT be a shelf hit."""
    cfg = _config(tmp_path)
    target = cfg.shelf_root / "mlx" / "mlx-community" / "Qwen3-14B-4bit"
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}")
    # No .safetensors file — this is a partial download.

    result = resolve_model(cfg, "mlx-community/Qwen3-14B-4bit")

    assert result.status == "missing"


def test_sharded_partial_dir_not_a_hit(tmp_path: Path):
    """Directory with config.json + index but missing shards must NOT be a shelf hit."""
    cfg = _config(tmp_path)
    target = cfg.shelf_root / "mlx" / "mlx-community" / "Qwen3-14B-4bit"
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}")
    (target / "model.safetensors.index.json").write_text(
        '{"weight_map": {"layer0": "model-00001.safetensors", "layer1": "model-00002.safetensors"}}'
    )
    # Only one of two shards present — still a partial download.
    (target / "model-00001.safetensors").write_bytes(b"x")
    # model-00002.safetensors is intentionally missing.

    result = resolve_model(cfg, "mlx-community/Qwen3-14B-4bit")

    assert result.status == "missing"
