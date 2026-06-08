from pathlib import Path

import pytest

from model_shelf.cli import main


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Pin the shelf to a tmp dir via env config; no real /Volumes scanning."""
    cfg = tmp_path / "config.toml"
    shelf = tmp_path / "shelf"
    for fmt in ("gguf", "mlx", "safetensors"):
        (shelf / fmt).mkdir(parents=True)
    cfg.write_text(f"shelf_root = {shelf.as_posix()!r}\nallow_downloads = false\n")
    monkeypatch.setenv("MODEL_SHELF_CONFIG", str(cfg))
    return shelf


def test_import_dir_apply_copies(tmp_path, capsys, _isolate):
    shelf = _isolate
    src = tmp_path / "drive" / "Qwen" / "Qwen3-14B-GGUF"
    src.mkdir(parents=True)
    (src / "Qwen3-14B-Q4_K_M.gguf").write_bytes(b"abc")
    rc = main(["import", str(tmp_path / "drive"), "--apply"])
    assert rc == 0
    target = shelf / "gguf" / "Qwen" / "Qwen3-14B-GGUF" / "Qwen3-14B-Q4_K_M.gguf"
    assert target.read_bytes() == b"abc"


def test_import_dry_run_default_copies_nothing(tmp_path, capsys, _isolate):
    shelf = _isolate
    src = tmp_path / "drive" / "Qwen" / "Qwen3-14B-GGUF"
    src.mkdir(parents=True)
    (src / "Qwen3-14B-Q4_K_M.gguf").write_bytes(b"abc")
    rc = main(["import", str(tmp_path / "drive")])
    assert rc == 0
    assert not (shelf / "gguf" / "Qwen").exists()
    assert "dry-run" in capsys.readouterr().out.lower()


def test_import_ollama_move_is_rejected(tmp_path, capsys, _isolate):
    rc = main(["import", "--from", "ollama", "--move"])
    assert rc == 2
    assert "move" in capsys.readouterr().err.lower()


def test_import_requires_source(capsys, _isolate):
    rc = main(["import"])
    assert rc == 2
    err = capsys.readouterr().err.lower()
    assert "path" in err or "from" in err


def test_import_json_skips_unbound(tmp_path, capsys, _isolate):
    import json as _json
    src = tmp_path / "drive"
    src.mkdir()
    (src / "loose-Q4_K_M.gguf").write_bytes(b"x")  # flat → needs_binding
    rc = main(["import", str(src), "--json"])
    out = _json.loads(capsys.readouterr().out)
    assert out["operations"][0]["status"] == "needs_binding"
    assert rc == 0


def test_import_from_hf_dispatch(tmp_path, capsys, _isolate, monkeypatch):
    """--from hf routes through _IMPORT_SCANNERS and the candidate reaches the plan."""
    import json as _json
    import model_shelf.cli as cli_mod
    from model_shelf.importer import ImportCandidate

    blob = tmp_path / "blob.gguf"
    blob.write_bytes(b"abc")
    fake = [ImportCandidate(
        source_path=blob, repo_id="Qwen/Qwen3-14B-GGUF", format="gguf",
        quant="Q4_K_M", size_bytes=3, source_label="hf",
    )]
    monkeypatch.setitem(cli_mod._IMPORT_SCANNERS, "hf", lambda: fake)

    rc = main(["import", "--from", "hf", "--json"])
    out = _json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["operations"][0]["source_label"] == "hf"
    assert out["operations"][0]["repo_id"] == "Qwen/Qwen3-14B-GGUF"
    assert out["operations"][0]["status"] == "pending"  # dry-run default
