from pathlib import Path

from model_shelf.importer import (
    ImportCandidate,
    ImportOperation,
    plan_import,
)


def _cand(repo_id, fmt, src, *, quant=None, size=1):
    return ImportCandidate(
        source_path=src, repo_id=repo_id, format=fmt,
        quant=quant, size_bytes=size, source_label="dir",
    )


def test_plan_gguf_pending(tmp_path: Path):
    shelf = tmp_path / "shelf"
    src = tmp_path / "Qwen3-14B-Q4_K_M.gguf"
    src.write_bytes(b"x")
    ops = plan_import([_cand("Qwen/Qwen3-14B-GGUF", "gguf", src, quant="Q4_K_M", size=1)], shelf)
    assert len(ops) == 1
    op = ops[0]
    assert op.action == "copy"
    assert op.status == "pending"
    assert op.target_path == shelf / "gguf" / "Qwen" / "Qwen3-14B-GGUF" / "Qwen3-14B-Q4_K_M.gguf"


def test_plan_snapshot_pending(tmp_path: Path):
    shelf = tmp_path / "shelf"
    src = tmp_path / "model"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "model.safetensors").write_bytes(b"x")
    ops = plan_import([_cand("mlx-community/Qwen3-14B-4bit", "mlx", src)], shelf)
    assert ops[0].status == "pending"
    assert ops[0].action == "copy"
    assert ops[0].target_path == shelf / "mlx" / "mlx-community" / "Qwen3-14B-4bit"


def test_plan_already_in_shelf_gguf(tmp_path: Path):
    shelf = tmp_path / "shelf"
    target = shelf / "gguf" / "Qwen" / "Qwen3-14B-GGUF" / "Qwen3-14B-Q4_K_M.gguf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")  # size 1, matches candidate
    src = tmp_path / "Qwen3-14B-Q4_K_M.gguf"
    src.write_bytes(b"x")
    ops = plan_import([_cand("Qwen/Qwen3-14B-GGUF", "gguf", src, quant="Q4_K_M", size=1)], shelf)
    assert ops[0].status == "already_in_shelf"
    assert ops[0].action == "skip"


def test_plan_conflict_gguf_size_mismatch(tmp_path: Path):
    shelf = tmp_path / "shelf"
    target = shelf / "gguf" / "Qwen" / "Qwen3-14B-GGUF" / "Qwen3-14B-Q4_K_M.gguf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"xxxxx")  # size 5, candidate says 1 → mismatch
    src = tmp_path / "Qwen3-14B-Q4_K_M.gguf"
    src.write_bytes(b"x")
    ops = plan_import([_cand("Qwen/Qwen3-14B-GGUF", "gguf", src, quant="Q4_K_M", size=1)], shelf)
    assert ops[0].status == "conflict"
    assert ops[0].action == "skip"


def test_plan_conflict_snapshot_incomplete(tmp_path: Path):
    # Target exists but is NOT a valid model dir (no config.json) — stays correct
    # whether or not #5's stricter _looks_like_model_dir is present.
    shelf = tmp_path / "shelf"
    target = shelf / "mlx" / "mlx-community" / "Qwen3-14B-4bit"
    target.mkdir(parents=True)
    (target / "README.md").write_text("leftover, no config.json")
    src = tmp_path / "model"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "model.safetensors").write_bytes(b"x")
    ops = plan_import([_cand("mlx-community/Qwen3-14B-4bit", "mlx", src)], shelf)
    assert ops[0].status == "conflict"
    assert ops[0].action == "skip"


def test_plan_needs_binding(tmp_path: Path):
    shelf = tmp_path / "shelf"
    src = tmp_path / "mystery.gguf"
    src.write_bytes(b"x")
    ops = plan_import([_cand(None, "gguf", src, quant="Q4_K_M")], shelf)
    assert ops[0].status == "needs_binding"
    assert ops[0].target_path is None
    assert ops[0].action == "skip"


def test_plan_gguf_without_quant_needs_binding(tmp_path: Path):
    shelf = tmp_path / "shelf"
    src = tmp_path / "weird-name.gguf"
    src.write_bytes(b"x")
    ops = plan_import([_cand("Qwen/Qwen3-14B-GGUF", "gguf", src, quant=None)], shelf)
    assert ops[0].status == "needs_binding"
    assert ops[0].target_path is None
    assert ops[0].action == "skip"


def test_operation_to_dict_roundtrips(tmp_path: Path):
    shelf = tmp_path / "shelf"
    src = tmp_path / "Qwen3-14B-Q4_K_M.gguf"
    src.write_bytes(b"x")
    op = plan_import([_cand("Qwen/Qwen3-14B-GGUF", "gguf", src, quant="Q4_K_M", size=1)], shelf)[0]
    d = op.to_dict()
    assert d["repo_id"] == "Qwen/Qwen3-14B-GGUF"
    assert d["format"] == "gguf"
    assert d["action"] == "copy"
    assert d["status"] == "pending"
    assert d["target_path"].endswith("Qwen3-14B-Q4_K_M.gguf")
    assert d["size_bytes"] == 1
