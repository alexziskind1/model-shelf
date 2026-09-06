from pathlib import Path

from model_shelf.importer import (
    ImportCandidate,
    ImportOperation,
    ImportReport,
    execute_import,
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


def test_execute_dry_run_creates_nothing(tmp_path: Path):
    shelf = tmp_path / "shelf"
    src = tmp_path / "Qwen3-14B-Q4_K_M.gguf"
    src.write_bytes(b"abc")
    ops = plan_import([_cand("Qwen/Qwen3-14B-GGUF", "gguf", src, quant="Q4_K_M", size=3)], shelf)
    report = execute_import(ops, action="copy", dry_run=True)
    assert not (shelf / "gguf").exists()
    assert report.bytes_imported == 3  # planned
    assert ops[0].status == "pending"


def test_execute_copy_gguf(tmp_path: Path):
    shelf = tmp_path / "shelf"
    src = tmp_path / "Qwen3-14B-Q4_K_M.gguf"
    src.write_bytes(b"abc")
    ops = plan_import([_cand("Qwen/Qwen3-14B-GGUF", "gguf", src, quant="Q4_K_M", size=3)], shelf)
    report = execute_import(ops, action="copy", dry_run=False)
    target = shelf / "gguf" / "Qwen" / "Qwen3-14B-GGUF" / "Qwen3-14B-Q4_K_M.gguf"
    assert target.is_file() and target.read_bytes() == b"abc"
    assert src.exists()  # copy leaves source
    assert ops[0].status == "done"
    assert report.bytes_imported == 3
    assert report.bytes_reclaimable == 0


def test_execute_move_gguf_removes_source(tmp_path: Path):
    shelf = tmp_path / "shelf"
    src = tmp_path / "Qwen3-14B-Q4_K_M.gguf"
    src.write_bytes(b"abc")
    ops = plan_import([_cand("Qwen/Qwen3-14B-GGUF", "gguf", src, quant="Q4_K_M", size=3)], shelf)
    report = execute_import(ops, action="move", dry_run=False)
    target = shelf / "gguf" / "Qwen" / "Qwen3-14B-GGUF" / "Qwen3-14B-Q4_K_M.gguf"
    assert target.is_file()
    assert not src.exists()  # move removes source
    assert report.bytes_reclaimable == 3


def test_execute_copy_snapshot_dir(tmp_path: Path):
    shelf = tmp_path / "shelf"
    src = tmp_path / "model"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "model.safetensors").write_bytes(b"weights")
    ops = plan_import([_cand("Qwen/Qwen3-14B", "safetensors", src, size=7)], shelf)
    execute_import(ops, action="copy", dry_run=False)
    target = shelf / "safetensors" / "Qwen" / "Qwen3-14B"
    assert (target / "config.json").is_file()
    assert (target / "model.safetensors").read_bytes() == b"weights"
    assert src.exists()


def test_execute_idempotent_second_run(tmp_path: Path):
    shelf = tmp_path / "shelf"
    src = tmp_path / "Qwen3-14B-Q4_K_M.gguf"
    src.write_bytes(b"abc")
    cand = _cand("Qwen/Qwen3-14B-GGUF", "gguf", src, quant="Q4_K_M", size=3)
    execute_import(plan_import([cand], shelf), action="copy", dry_run=False)
    ops2 = plan_import([cand], shelf)
    assert ops2[0].status == "already_in_shelf"
    report2 = execute_import(ops2, action="copy", dry_run=False)
    assert report2.bytes_imported == 0


def test_execute_resolves_symlinked_source_on_copy(tmp_path: Path):
    """HF cache stores snapshot files as symlinks to blobs — copy must follow them."""
    shelf = tmp_path / "shelf"
    blob = tmp_path / "blob"
    blob.write_bytes(b"realdata")
    link = tmp_path / "Qwen3-14B-Q4_K_M.gguf"
    link.symlink_to(blob)
    ops = plan_import([_cand("Qwen/Qwen3-14B-GGUF", "gguf", link, quant="Q4_K_M", size=8)], shelf)
    execute_import(ops, action="copy", dry_run=False)
    target = shelf / "gguf" / "Qwen" / "Qwen3-14B-GGUF" / "Qwen3-14B-Q4_K_M.gguf"
    assert not target.is_symlink()  # real file, not a dangling link
    assert target.read_bytes() == b"realdata"


def test_execute_move_keeps_source_when_verify_fails(tmp_path: Path, monkeypatch):
    """verify-before-delete: if verification fails, the source must survive and the
    partial target must be cleaned up."""
    import model_shelf.importer as importer_mod

    shelf = tmp_path / "shelf"
    src = tmp_path / "Qwen3-14B-Q4_K_M.gguf"
    src.write_bytes(b"abc")
    ops = plan_import([_cand("Qwen/Qwen3-14B-GGUF", "gguf", src, quant="Q4_K_M", size=3)], shelf)

    monkeypatch.setattr(importer_mod, "_verify", lambda *a, **k: False)
    report = execute_import(ops, action="move", dry_run=False)

    target = shelf / "gguf" / "Qwen" / "Qwen3-14B-GGUF" / "Qwen3-14B-Q4_K_M.gguf"
    assert src.exists()              # source survived
    assert ops[0].status == "failed"
    assert not target.exists()       # partial target cleaned up
    assert report.bytes_reclaimable == 0
