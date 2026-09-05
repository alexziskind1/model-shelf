from pathlib import Path

import pytest

from model_shelf.cli import (
    LMSTUDIO_WINDOWS_UNSUPPORTED,
    _print_lmstudio_scan,
    cmd_lmstudio_link,
    cmd_lmstudio_migrate,
)
from model_shelf.lmstudio import (
    MigrationOperation,
    _apply_link_into_lmstudio,
    build_migration_plan,
    discover_lmstudio_root,
    link_lmstudio_model,
    migrate_lmstudio,
    scan_lmstudio,
)
from model_shelf.resolver import Config


@pytest.fixture(autouse=True)
def _isolate_shelf_candidates(monkeypatch):
    monkeypatch.setattr(
        "model_shelf.lmstudio.list_shelf_candidates",
        lambda cfg: [cfg.shelf_root],
    )


def _cfg(tmp_path: Path) -> Config:
    shelf = tmp_path / "shelf"
    shelf.mkdir()
    return Config(shelf_root=shelf, allow_downloads=False)


def _lm_root(tmp_path: Path) -> Path:
    root = tmp_path / "lmstudio" / "models"
    root.mkdir(parents=True)
    return root


def test_discovers_explicit_lmstudio_root(tmp_path: Path):
    root = _lm_root(tmp_path)

    result = discover_lmstudio_root(root)

    assert result.selected == root
    assert result.source == "explicit"


def test_explicit_lmstudio_child_path_normalizes_to_models_root(tmp_path: Path):
    root = _lm_root(tmp_path)
    child = root / "DevQuasar-8" / "mistralai.Mistral-7B-v0.3-GGUF"
    child.mkdir(parents=True)

    result = discover_lmstudio_root(child)

    assert result.selected == root
    assert result.source == "explicit"


def test_discovers_lmstudio_root_from_settings_json(tmp_path: Path):
    home = tmp_path / "home"
    model_root = home / "Models" / "LM Studio"
    model_root.mkdir(parents=True)
    settings_dir = home / "Library" / "Application Support" / "LM Studio"
    settings_dir.mkdir(parents=True)
    (settings_dir / "settings.json").write_text(
        f'{{"modelSavePath": "{model_root}"}}'
    )

    result = discover_lmstudio_root(home=home)

    assert result.selected == model_root
    assert result.source == "settings"


def test_settings_lmstudio_child_path_normalizes_to_models_root(tmp_path: Path):
    home = tmp_path / "home"
    model_root = home / ".lmstudio" / "models"
    child = model_root / "DevQuasar-8" / "mistralai.Mistral-7B-v0.3-GGUF"
    child.mkdir(parents=True)
    settings_dir = home / "Library" / "Application Support" / "LM Studio"
    settings_dir.mkdir(parents=True)
    (settings_dir / "settings.json").write_text(
        f'{{"lastModelPath": "{child}"}}'
    )

    result = discover_lmstudio_root(home=home)

    assert result.selected == model_root
    assert result.source == "settings"


def test_scan_reports_gguf_lmstudio_only(tmp_path: Path):
    cfg = _cfg(tmp_path)
    root = _lm_root(tmp_path)
    repo = root / "Qwen" / "Qwen3-14B-GGUF"
    repo.mkdir(parents=True)
    gguf = repo / "Qwen3-14B-Q4_K_M.gguf"
    gguf.write_bytes(b"gguf")

    result = scan_lmstudio(cfg, lmstudio_root=root)

    assert len(result.items) == 1
    item = result.items[0]
    assert item.status == "lmstudio_only"
    assert item.format == "gguf"
    assert item.target_path == cfg.shelf_root / "gguf" / "Qwen" / "Qwen3-14B-GGUF" / gguf.name


def test_scan_reports_existing_shelf_copy_as_duplicated(tmp_path: Path):
    cfg = _cfg(tmp_path)
    root = _lm_root(tmp_path)
    repo = root / "Qwen" / "Qwen3-14B-GGUF"
    repo.mkdir(parents=True)
    gguf = repo / "Qwen3-14B-Q4_K_M.gguf"
    gguf.write_bytes(b"gguf")
    target = cfg.shelf_root / "gguf" / "Qwen" / "Qwen3-14B-GGUF"
    target.mkdir(parents=True)
    (target / gguf.name).write_bytes(b"gguf")

    result = scan_lmstudio(cfg, lmstudio_root=root)

    assert result.items[0].status == "duplicated"


def test_scan_reports_shelf_only_model_as_in_shelf(tmp_path: Path):
    cfg = _cfg(tmp_path)
    root = _lm_root(tmp_path)
    target = cfg.shelf_root / "gguf" / "Qwen" / "Qwen3-14B-GGUF"
    target.mkdir(parents=True)
    (target / "Qwen3-14B-Q4_K_M.gguf").write_bytes(b"gguf")

    result = scan_lmstudio(cfg, lmstudio_root=root)

    assert len(result.items) == 1
    item = result.items[0]
    assert item.status == "in_shelf"
    assert item.source_path == root / "Qwen" / "Qwen3-14B-GGUF" / "Qwen3-14B-Q4_K_M.gguf"
    assert item.target_path == target / "Qwen3-14B-Q4_K_M.gguf"


def test_scan_reports_lmstudio_link_to_shelf(tmp_path: Path):
    cfg = _cfg(tmp_path)
    root = _lm_root(tmp_path)
    target = cfg.shelf_root / "gguf" / "Qwen" / "Qwen3-14B-GGUF"
    target.mkdir(parents=True)
    shelf_file = target / "Qwen3-14B-Q4_K_M.gguf"
    shelf_file.write_bytes(b"gguf")
    repo = root / "Qwen" / "Qwen3-14B-GGUF"
    repo.mkdir(parents=True)
    (repo / shelf_file.name).symlink_to(shelf_file)

    result = scan_lmstudio(cfg, lmstudio_root=root)

    assert result.items[0].status == "lmstudio_link"


def test_scan_reports_shelf_link_to_lmstudio(tmp_path: Path):
    cfg = _cfg(tmp_path)
    root = _lm_root(tmp_path)
    repo = root / "Qwen" / "Qwen3-14B-GGUF"
    repo.mkdir(parents=True)
    source = repo / "Qwen3-14B-Q4_K_M.gguf"
    source.write_bytes(b"gguf")
    target = cfg.shelf_root / "gguf" / "Qwen" / "Qwen3-14B-GGUF"
    target.mkdir(parents=True)
    (target / source.name).symlink_to(source)

    result = scan_lmstudio(cfg, lmstudio_root=root)

    assert result.items[0].status == "shelf_link"


def test_scan_detects_mlx_and_safetensors(tmp_path: Path):
    cfg = _cfg(tmp_path)
    root = _lm_root(tmp_path)
    mlx = root / "mlx-community" / "Qwen3-14B-4bit"
    mlx.mkdir(parents=True)
    (mlx / "config.json").write_text("{}")
    (mlx / "model.safetensors").write_bytes(b"mlx")
    st = root / "Qwen" / "Qwen3-14B"
    st.mkdir(parents=True)
    (st / "config.json").write_text("{}")
    (st / "model-00001-of-00001.safetensors").write_bytes(b"st")

    result = scan_lmstudio(cfg, lmstudio_root=root)

    by_model = {item.model: item for item in result.items}
    assert by_model["mlx-community/Qwen3-14B-4bit"].format == "mlx"
    assert by_model["Qwen/Qwen3-14B"].format == "safetensors"


def test_build_migration_plan_defaults_to_move_and_link_back(tmp_path: Path):
    cfg = _cfg(tmp_path)
    root = _lm_root(tmp_path)
    repo = root / "Qwen" / "Qwen3-14B-GGUF"
    repo.mkdir(parents=True)
    source = repo / "Qwen3-14B-Q4_K_M.gguf"
    source.write_bytes(b"gguf")
    scan = scan_lmstudio(cfg, lmstudio_root=root)

    plan = build_migration_plan(scan)

    assert len(plan) == 1
    assert plan[0].action == "move_to_shelf_and_link_back"
    assert plan[0].source_path == source
    assert plan[0].link_path == source


def test_migrate_dry_run_does_not_change_files(tmp_path: Path):
    cfg = _cfg(tmp_path)
    root = _lm_root(tmp_path)
    repo = root / "Qwen" / "Qwen3-14B-GGUF"
    repo.mkdir(parents=True)
    source = repo / "Qwen3-14B-Q4_K_M.gguf"
    source.write_bytes(b"gguf")

    result = migrate_lmstudio(cfg, lmstudio_root=root, dry_run=True)

    assert result.applied is False
    assert source.is_file()
    assert not result.operations[0].target_path.exists()


def test_migrate_moves_file_to_shelf_and_links_back(tmp_path: Path):
    cfg = _cfg(tmp_path)
    root = _lm_root(tmp_path)
    repo = root / "Qwen" / "Qwen3-14B-GGUF"
    repo.mkdir(parents=True)
    source = repo / "Qwen3-14B-Q4_K_M.gguf"
    source.write_bytes(b"gguf")

    result = migrate_lmstudio(cfg, lmstudio_root=root, dry_run=False, yes=True)

    op = result.operations[0]
    assert result.applied is True
    assert op.status == "applied"
    assert op.target_path.is_file()
    assert source.is_symlink()
    assert source.resolve() == op.target_path


def test_migrate_cli_single_model_applies_by_default(tmp_path: Path):
    cfg = _cfg(tmp_path)
    root = _lm_root(tmp_path)
    repo = root / "Qwen" / "Qwen3-14B-GGUF"
    repo.mkdir(parents=True)
    source = repo / "Qwen3-14B-Q4_K_M.gguf"
    source.write_bytes(b"gguf")

    class Args:
        model = "Qwen/Qwen3-14B-GGUF"
        filename = source.name
        lmstudio_root = str(root)
        format = None
        use_lms = False
        link = False
        dry_run = False
        yes = False
        json = False

    rc = cmd_lmstudio_migrate(Args, cfg)

    target = cfg.shelf_root / "gguf" / "Qwen" / "Qwen3-14B-GGUF" / source.name
    assert rc == 0
    assert target.is_file()
    assert source.is_symlink()
    assert source.resolve() == target


def test_migrate_cli_single_model_dry_run_does_not_move(tmp_path: Path):
    cfg = _cfg(tmp_path)
    root = _lm_root(tmp_path)
    repo = root / "Qwen" / "Qwen3-14B-GGUF"
    repo.mkdir(parents=True)
    source = repo / "Qwen3-14B-Q4_K_M.gguf"
    source.write_bytes(b"gguf")

    class Args:
        model = "Qwen/Qwen3-14B-GGUF"
        filename = source.name
        lmstudio_root = str(root)
        format = None
        use_lms = False
        link = False
        dry_run = True
        yes = False
        json = False

    rc = cmd_lmstudio_migrate(Args, cfg)

    target = cfg.shelf_root / "gguf" / "Qwen" / "Qwen3-14B-GGUF" / source.name
    assert rc == 0
    assert source.is_file()
    assert not target.exists()


def test_migrate_cli_reports_unsupported_on_windows(tmp_path: Path, monkeypatch, capsys):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr("model_shelf.cli.os.name", "nt")

    class Args:
        model = "Qwen/Qwen3-14B-GGUF"
        filename = None
        lmstudio_root = None
        format = None
        use_lms = False
        link = False
        dry_run = False
        yes = False
        json = False

    rc = cmd_lmstudio_migrate(Args, cfg)

    assert rc == 2
    assert LMSTUDIO_WINDOWS_UNSUPPORTED in capsys.readouterr().err


def test_migrate_link_leaves_lmstudio_file_and_links_shelf(tmp_path: Path):
    cfg = _cfg(tmp_path)
    root = _lm_root(tmp_path)
    repo = root / "Qwen" / "Qwen3-14B-GGUF"
    repo.mkdir(parents=True)
    source = repo / "Qwen3-14B-Q4_K_M.gguf"
    source.write_bytes(b"gguf")

    result = migrate_lmstudio(
        cfg,
        lmstudio_root=root,
        link=True,
        dry_run=False,
        yes=True,
    )

    op = result.operations[0]
    assert source.is_file()
    assert op.target_path.is_symlink()
    assert op.target_path.resolve() == source


def test_link_dry_run_plans_single_in_shelf_model(tmp_path: Path):
    cfg = _cfg(tmp_path)
    root = _lm_root(tmp_path)
    target = cfg.shelf_root / "mlx" / "mlx-community" / "Qwen3-14B-4bit"
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}")
    (target / "model.safetensors").write_bytes(b"mlx")

    result = link_lmstudio_model(
        cfg,
        "mlx-community/Qwen3-14B-4bit",
        lmstudio_root=root,
        dry_run=True,
    )

    assert result.applied is False
    assert result.item.status == "in_shelf"
    assert result.operation.action == "link_into_lmstudio"
    assert result.operation.source_path == target
    assert result.operation.target_path == root / "mlx-community" / "Qwen3-14B-4bit"
    assert not result.operation.target_path.exists()


def test_link_apply_creates_lmstudio_symlink_to_shelf(tmp_path: Path):
    cfg = _cfg(tmp_path)
    root = _lm_root(tmp_path)
    target = cfg.shelf_root / "mlx" / "mlx-community" / "Qwen3-14B-4bit"
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}")
    (target / "model.safetensors").write_bytes(b"mlx")

    result = link_lmstudio_model(
        cfg,
        "mlx-community/Qwen3-14B-4bit",
        lmstudio_root=root,
        dry_run=False,
        yes=True,
    )

    op = result.operation
    assert result.applied is True
    assert op.status == "applied"
    assert op.target_path.is_symlink()
    assert op.target_path.resolve() == target


def test_link_cli_applies_by_default(tmp_path: Path):
    cfg = _cfg(tmp_path)
    root = _lm_root(tmp_path)
    target = cfg.shelf_root / "mlx" / "mlx-community" / "Qwen3-14B-4bit"
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}")
    (target / "model.safetensors").write_bytes(b"mlx")

    class Args:
        model = "mlx-community/Qwen3-14B-4bit"
        filename = None
        lmstudio_root = str(root)
        format = None
        use_lms = False
        dry_run = False
        yes = False
        replace_link = False
        json = False

    rc = cmd_lmstudio_link(Args, cfg)

    link = root / "mlx-community" / "Qwen3-14B-4bit"
    assert rc == 0
    assert link.is_symlink()
    assert link.resolve() == target


def test_link_cli_dry_run_does_not_create_symlink(tmp_path: Path):
    cfg = _cfg(tmp_path)
    root = _lm_root(tmp_path)
    target = cfg.shelf_root / "mlx" / "mlx-community" / "Qwen3-14B-4bit"
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}")
    (target / "model.safetensors").write_bytes(b"mlx")

    class Args:
        model = "mlx-community/Qwen3-14B-4bit"
        filename = None
        lmstudio_root = str(root)
        format = None
        use_lms = False
        dry_run = True
        yes = False
        replace_link = False
        json = False

    rc = cmd_lmstudio_link(Args, cfg)

    assert rc == 0
    assert not (root / "mlx-community" / "Qwen3-14B-4bit").exists()


def test_link_cli_reports_unsupported_on_windows(tmp_path: Path, monkeypatch, capsys):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr("model_shelf.cli.os.name", "nt")

    class Args:
        model = "mlx-community/Qwen3-14B-4bit"
        filename = None
        lmstudio_root = None
        format = None
        use_lms = False
        dry_run = False
        yes = False
        replace_link = False
        json = False

    rc = cmd_lmstudio_link(Args, cfg)

    assert rc == 2
    assert LMSTUDIO_WINDOWS_UNSUPPORTED in capsys.readouterr().err


def test_link_matches_gguf_by_repo_and_filename(tmp_path: Path):
    cfg = _cfg(tmp_path)
    root = _lm_root(tmp_path)
    target = cfg.shelf_root / "gguf" / "Qwen" / "Qwen3-14B-GGUF"
    target.mkdir(parents=True)
    shelf_file = target / "Qwen3-14B-Q4_K_M.gguf"
    shelf_file.write_bytes(b"gguf")

    result = link_lmstudio_model(
        cfg,
        "Qwen/Qwen3-14B-GGUF",
        filename=shelf_file.name,
        lmstudio_root=root,
    )

    assert result.operation.source_path == shelf_file
    assert result.operation.target_path == root / "Qwen" / "Qwen3-14B-GGUF" / shelf_file.name


def test_link_requires_in_shelf_model_not_existing_lmstudio_copy(tmp_path: Path):
    cfg = _cfg(tmp_path)
    root = _lm_root(tmp_path)
    target = cfg.shelf_root / "gguf" / "Qwen" / "Qwen3-14B-GGUF"
    target.mkdir(parents=True)
    shelf_file = target / "Qwen3-14B-Q4_K_M.gguf"
    shelf_file.write_bytes(b"gguf")
    repo = root / "Qwen" / "Qwen3-14B-GGUF"
    repo.mkdir(parents=True)
    (repo / shelf_file.name).write_bytes(b"local")

    with pytest.raises(ValueError, match="no in_shelf model matched"):
        link_lmstudio_model(
            cfg,
            "Qwen/Qwen3-14B-GGUF",
            filename=shelf_file.name,
            lmstudio_root=root,
            dry_run=False,
            yes=True,
        )


def test_link_replace_link_replaces_existing_lmstudio_symlink(tmp_path: Path):
    shelf_file = tmp_path / "shelf" / "model.gguf"
    shelf_file.parent.mkdir()
    shelf_file.write_bytes(b"gguf")
    other = tmp_path / "other.gguf"
    other.write_bytes(b"other")
    link = tmp_path / "lmstudio" / "model.gguf"
    link.parent.mkdir()
    link.symlink_to(other)
    op = MigrationOperation(
        action="link_into_lmstudio",
        source_path=shelf_file,
        target_path=link,
        link_path=link,
    )

    _apply_link_into_lmstudio(op, replace_link=True)

    assert op.status == "applied"
    assert link.is_symlink()
    assert link.resolve() == shelf_file


def test_scan_human_output_uses_relative_paths_by_default(tmp_path: Path, capsys):
    cfg = _cfg(tmp_path)
    root = _lm_root(tmp_path)
    repo = root / "Qwen" / "Qwen3-14B-GGUF"
    repo.mkdir(parents=True)
    gguf = repo / "Qwen3-14B-Q4_K_M.gguf"
    gguf.write_bytes(b"gguf")
    scan = scan_lmstudio(cfg, lmstudio_root=root)

    _print_lmstudio_scan(scan)

    out = capsys.readouterr().out
    assert "path: Qwen/Qwen3-14B-GGUF/Qwen3-14B-Q4_K_M.gguf" in out
    assert f"path: {gguf}" not in out


def test_scan_human_output_can_show_full_paths(tmp_path: Path, capsys):
    cfg = _cfg(tmp_path)
    root = _lm_root(tmp_path)
    repo = root / "Qwen" / "Qwen3-14B-GGUF"
    repo.mkdir(parents=True)
    gguf = repo / "Qwen3-14B-Q4_K_M.gguf"
    gguf.write_bytes(b"gguf")
    scan = scan_lmstudio(cfg, lmstudio_root=root)

    _print_lmstudio_scan(scan, full_paths=True)

    out = capsys.readouterr().out
    assert f"path: {gguf}" in out
