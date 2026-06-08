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
