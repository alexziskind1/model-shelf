import json
from pathlib import Path

from model_shelf.sources import scan_dir


def test_scan_dir_gguf_infers_repo_id(tmp_path: Path):
    root = tmp_path / "drive"
    repo = root / "Qwen" / "Qwen3-14B-GGUF"
    repo.mkdir(parents=True)
    (repo / "Qwen3-14B-Q4_K_M.gguf").write_bytes(b"x")
    cands = scan_dir(root)
    assert len(cands) == 1
    c = cands[0]
    assert c.repo_id == "Qwen/Qwen3-14B-GGUF"
    assert c.format == "gguf"
    assert c.quant == "Q4_K_M"
    assert c.source_label == "dir"


def test_scan_dir_snapshot_infers_repo_id(tmp_path: Path):
    root = tmp_path / "drive"
    repo = root / "mlx-community" / "Qwen3-14B-4bit"
    repo.mkdir(parents=True)
    (repo / "config.json").write_text("{}")
    (repo / "model.safetensors").write_bytes(b"x")
    cands = scan_dir(root)
    assert len(cands) == 1
    assert cands[0].repo_id == "mlx-community/Qwen3-14B-4bit"
    assert cands[0].format == "mlx"


def test_scan_dir_flat_gguf_needs_binding(tmp_path: Path):
    root = tmp_path / "drive"
    root.mkdir()
    (root / "some-model-Q4_K_M.gguf").write_bytes(b"x")  # no publisher/repo nesting
    cands = scan_dir(root)
    assert len(cands) == 1
    assert cands[0].repo_id is None
    assert cands[0].format == "gguf"
    assert cands[0].quant == "Q4_K_M"


def test_scan_dir_deeply_nested_needs_binding(tmp_path: Path):
    """A model dir nested deeper than <root>/pub/repo can't be inferred → repo_id None."""
    root = tmp_path / "drive"
    deep = root / "pub" / "repo" / "snapshots" / "abc"
    deep.mkdir(parents=True)
    (deep / "config.json").write_text("{}")
    (deep / "model.safetensors").write_bytes(b"x")
    cands = scan_dir(root)
    assert len(cands) == 1
    assert cands[0].repo_id is None


from model_shelf.sources import scan_hf_cache


def _make_hf_repo(hub: Path, mangled: str, files: dict[str, bytes]) -> None:
    snap = hub / mangled / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    blobs = hub / mangled / "blobs"
    blobs.mkdir(parents=True)
    (hub / mangled / "refs").mkdir(parents=True)
    (hub / mangled / "refs" / "main").write_text("abc123")
    for name, data in files.items():
        blob = blobs / name
        blob.write_bytes(data)
        (snap / name).symlink_to(blob)


def test_scan_hf_cache_gguf(tmp_path: Path):
    hub = tmp_path / "hub"
    _make_hf_repo(hub, "models--Qwen--Qwen3-14B-GGUF",
                  {"Qwen3-14B-Q4_K_M.gguf": b"data"})
    cands = scan_hf_cache(home=tmp_path, hub=hub)
    assert len(cands) == 1
    c = cands[0]
    assert c.repo_id == "Qwen/Qwen3-14B-GGUF"
    assert c.format == "gguf"
    assert c.quant == "Q4_K_M"
    assert c.source_label == "hf"
    assert c.source_path.name == "Qwen3-14B-Q4_K_M.gguf"


def test_scan_hf_cache_safetensors(tmp_path: Path):
    hub = tmp_path / "hub"
    _make_hf_repo(hub, "models--Qwen--Qwen3-14B",
                  {"config.json": b"{}", "model.safetensors": b"w"})
    cands = scan_hf_cache(home=tmp_path, hub=hub)
    assert len(cands) == 1
    assert cands[0].repo_id == "Qwen/Qwen3-14B"
    assert cands[0].format == "safetensors"
    assert cands[0].source_path.is_dir()


from model_shelf.sources import scan_ollama


def _make_ollama(models: Path, manifest_rel: str, blob_hex: str, data: bytes) -> None:
    digest = f"sha256:{blob_hex}"
    blob = models / "blobs" / f"sha256-{blob_hex}"
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(data)
    manifest = models / "manifests" / manifest_rel
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "layers": [
            {"mediaType": "application/vnd.ollama.image.license", "digest": "sha256:dead", "size": 1},
            {"mediaType": "application/vnd.ollama.image.model", "digest": digest, "size": len(data)},
        ]
    }))


def test_scan_ollama_hf_pull_maps_real_repo_id(tmp_path: Path):
    models = tmp_path / "models"
    _make_ollama(models, "hf.co/Qwen/Qwen3-14B-GGUF/Q4_K_M", "a" * 8, b"weights")
    cands = scan_ollama(home=tmp_path, models_root=models)
    assert len(cands) == 1
    c = cands[0]
    assert c.repo_id == "Qwen/Qwen3-14B-GGUF"
    assert c.quant == "Q4_K_M"
    assert c.format == "gguf"
    assert c.source_label == "ollama"
    assert c.source_path.read_bytes() == b"weights"


def test_scan_ollama_library_maps_synthetic_repo_id(tmp_path: Path):
    models = tmp_path / "models"
    _make_ollama(models, "registry.ollama.ai/library/llama3/8b", "b" * 8, b"w")
    cands = scan_ollama(home=tmp_path, models_root=models)
    assert len(cands) == 1
    assert cands[0].repo_id == "ollama-library/llama3"
    assert cands[0].quant == "8b"
    assert cands[0].format == "gguf"


def test_scan_ollama_library_without_model_needs_binding(tmp_path: Path):
    """A library manifest missing the model-name segment can't be mapped → repo_id None."""
    models = tmp_path / "models"
    _make_ollama(models, "registry.ollama.ai/library/llama3", "c" * 8, b"w")
    cands = scan_ollama(home=tmp_path, models_root=models)
    assert len(cands) == 1
    assert cands[0].repo_id is None


from model_shelf.sources import discover_lmstudio_root, scan_lmstudio


def test_discover_lmstudio_root_from_default(tmp_path: Path):
    models = tmp_path / ".lmstudio" / "models"
    models.mkdir(parents=True)
    assert discover_lmstudio_root(home=tmp_path) == models


def test_discover_lmstudio_root_from_settings(tmp_path: Path):
    home = tmp_path
    (home / ".lmstudio").mkdir(parents=True)
    custom = tmp_path / "external" / "lmstudio-models"
    custom.mkdir(parents=True)
    (home / ".lmstudio" / "settings.json").write_text(
        json.dumps({"downloadsFolder": str(custom)})
    )
    assert discover_lmstudio_root(home=home) == custom


def test_scan_lmstudio_finds_models(tmp_path: Path):
    models = tmp_path / ".lmstudio" / "models"
    repo = models / "lmstudio-community" / "Qwen3-14B-GGUF"
    repo.mkdir(parents=True)
    (repo / "Qwen3-14B-Q4_K_M.gguf").write_bytes(b"x")
    cands = scan_lmstudio(home=tmp_path)
    assert len(cands) == 1
    assert cands[0].repo_id == "lmstudio-community/Qwen3-14B-GGUF"
    assert cands[0].format == "gguf"
    assert cands[0].source_label == "lmstudio"


def test_discover_lmstudio_root_from_settings_with_models_subdir(tmp_path: Path):
    """downloadsFolder containing a models/ subdir → that subdir is preferred."""
    home = tmp_path
    (home / ".lmstudio").mkdir(parents=True)
    custom = tmp_path / "external" / "lmstudio-downloads"
    models_sub = custom / "models"
    models_sub.mkdir(parents=True)
    (home / ".lmstudio" / "settings.json").write_text(
        json.dumps({"downloadsFolder": str(custom)})
    )
    assert discover_lmstudio_root(home=home) == models_sub


def test_discover_lmstudio_root_malformed_settings_falls_back(tmp_path: Path):
    """Malformed settings.json must not crash — fall back to the default models dir."""
    models = tmp_path / ".lmstudio" / "models"
    models.mkdir(parents=True)
    (tmp_path / ".lmstudio" / "settings.json").write_text("{ not valid json")
    assert discover_lmstudio_root(home=tmp_path) == models
