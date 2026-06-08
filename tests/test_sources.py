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
