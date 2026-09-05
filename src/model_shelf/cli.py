"""Command-line interface for Model Shelf."""

from __future__ import annotations

import argparse
import json
import os
import sys

from pathlib import Path

from model_shelf.config import load_config, writable_config_path, write_config
from model_shelf.detect import StorageCandidate, detect_storage_candidates
from model_shelf.lmstudio import link_lmstudio_model, migrate_lmstudio, scan_lmstudio
from model_shelf.resolver import (
    SUPPORTED_FORMATS,
    Config,
    StorageNotAvailableError,
    check_storage_available,
    detect_format,
    init_shelf,
    list_shelf_candidates,
    resolve_model,
)
from model_shelf.search import find_models


LMSTUDIO_WINDOWS_UNSUPPORTED = (
    "LM Studio migrate/link are not supported on Windows yet. "
    "Windows needs explicit junction/hardlink/symlink handling to avoid unsafe links. "
    "Use `model-shelf lmstudio scan` to inspect models."
)


def _fmt_size(n_bytes: int) -> str:
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _print_result_pretty(repo_id: str, result) -> None:
    for c in result.checks:
        marker = "HIT" if c["result"] == "hit" else "miss"
        print(f"  {c['location']:<6} {c['root']:<48} {marker}")
    if result.status == "downloaded":
        print(f"  fetch  huggingface.co/{repo_id:<33} downloaded")
    print()
    print(f"  status      {result.status}")
    print(f"  source      {result.source}")
    print(f"  format      {result.format}")
    print(f"  path        {result.path}")


def cmd_resolve(args: argparse.Namespace, cfg: Config) -> int:
    if args.no_download:
        cfg.allow_downloads = False
    fmt = args.format or detect_format(args.repo_id)
    result = resolve_model(cfg, args.repo_id, format=fmt, quant=args.quant)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        _print_result_pretty(args.repo_id, result)
    return 0 if result.status != "missing" else 1


def _format_choice_label(c: StorageCandidate) -> str:
    tag = "existing" if c.existing else "new"
    where = "external" if c.is_external else "internal"
    return f"[{where}] {c.label} ({tag})  —  {c.path}"


def _pick_candidate_interactively(
    candidates: list[StorageCandidate],
) -> Path | None:
    import questionary

    choices = [questionary.Choice(_format_choice_label(c), value=c.path) for c in candidates]
    choices.append(questionary.Choice("Enter a custom path...", value="__custom__"))

    chosen = questionary.select(
        "Where do you want your Model Shelf?",
        choices=choices,
    ).ask()

    if chosen is None:
        return None  # user hit Ctrl-C
    if chosen == "__custom__":
        path_str = questionary.path(
            "Enter shelf path:",
            default=str(Path.home() / ".cache" / "model-shelf" / "models"),
        ).ask()
        if not path_str:
            return None
        return Path(path_str).expanduser()
    return chosen


def _resolve_shelf_path(args: argparse.Namespace) -> Path | None:
    """Decide which shelf_root to use for `init`. Returns None on user-abort."""
    if args.path:
        return Path(args.path).expanduser().resolve()

    candidates = detect_storage_candidates()
    existing_external = [c for c in candidates if c.existing and c.is_external]
    external = [c for c in candidates if c.is_external]
    internal = next((c for c in candidates if not c.is_external), None)

    # Strong signal: exactly one external drive already has a ModelShelf —
    # use it without prompting.
    if len(existing_external) == 1:
        chosen = existing_external[0]
        print(f"model-shelf: using existing shelf at {chosen.path}")
        return chosen.path

    is_tty = sys.stdin.isatty()

    # Multiple existing external shelves — prompt to disambiguate, error if non-TTY.
    if len(existing_external) > 1:
        if not is_tty:
            print("error: multiple existing external shelves; specify a path explicitly:", file=sys.stderr)
            for c in existing_external:
                print(f"  - {c.path}", file=sys.stderr)
            return None
        picked = _pick_candidate_interactively(candidates)
        if picked is None:
            print("model-shelf: cancelled", file=sys.stderr)
        return picked

    # No existing external shelves.
    # If no external drives are connected, fall back to the internal default.
    if not external:
        if internal is None:
            print("error: could not determine a shelf location", file=sys.stderr)
            return None
        print(f"model-shelf: no external drives detected; using internal storage at {internal.path}")
        return internal.path

    # External drives present but none with an existing shelf — prompt.
    if not is_tty:
        if internal is None:
            print("error: not interactive and no shelf location could be inferred", file=sys.stderr)
            return None
        print(f"model-shelf: not interactive; using internal storage at {internal.path}", file=sys.stderr)
        return internal.path

    picked = _pick_candidate_interactively(candidates)
    if picked is None:
        print("model-shelf: cancelled", file=sys.stderr)
    return picked


def cmd_init(args: argparse.Namespace, cfg: Config) -> int:
    chosen = _resolve_shelf_path(args)
    if chosen is None:
        return 1
    new_root = chosen.expanduser().resolve()
    cfg.shelf_root = new_root
    created = init_shelf(cfg)

    config_path = writable_config_path(args.config)
    if args.path:
        # Explicit path → pin it in the config.
        write_config(
            config_path,
            shelf_root=new_root,
            allow_downloads=cfg.allow_downloads,
        )
        print(f"model-shelf: wrote {config_path}")
        print(f"            shelf_root = {new_root}  (pinned)")
    else:
        # Auto-detected → don't pin. Discovery will find this shelf at runtime,
        # which means swapping drives (or renaming) Just Works.
        print(f"model-shelf: using {new_root}  (auto-discovered, not pinned in config)")

    if not created:
        print(f"model-shelf: shelf at {cfg.shelf_root} already initialized")
        return 0
    print(f"model-shelf: initialized shelf at {cfg.shelf_root}")
    for path in created:
        print(f"  + {path}")
    return 0


def cmd_find(args: argparse.Namespace, cfg: Config) -> int:
    results = find_models(args.query, format=args.format, limit=args.limit)
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return 0 if results else 1
    if not results:
        print("(no results)")
        return 1
    for r in results:
        print(f"  [{r.format:<11}] {r.repo_id:<55} {r.downloads:>10,} downloads")
    return 0


def _print_shelf_contents(root: Path) -> None:
    """Walk publisher/repo nesting and list files/dirs at each level."""
    for fmt in SUPPORTED_FORMATS:
        sub = root / fmt
        print(f"\n  {fmt}/")
        if not sub.exists():
            print("    (empty)")
            continue
        publishers = sorted(
            p for p in sub.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
        if not publishers:
            print("    (empty)")
            continue
        for publisher in publishers:
            repos = sorted(
                r for r in publisher.iterdir()
                if r.is_dir() and not r.name.startswith("._")
            )
            if not repos:
                continue
            for repo in repos:
                if fmt == "gguf":
                    files = sorted(
                        f for f in repo.glob("*.gguf") if not f.name.startswith("._")
                    )
                    for f in files:
                        print(
                            f"    {publisher.name}/{repo.name}/{f.name}  "
                            f"({_fmt_size(f.stat().st_size)})"
                        )
                else:
                    print(f"    {publisher.name}/{repo.name}/")


def cmd_list(args: argparse.Namespace, cfg: Config) -> int:
    check_storage_available(cfg)
    candidates = list_shelf_candidates(cfg)
    primary = cfg.shelf_root
    for i, root in enumerate(candidates):
        tag = "primary" if root == primary or root.resolve() == primary.resolve() else "additional"
        if i > 0:
            print()
        print(f"shelf  {root}  ({tag})")
        _print_shelf_contents(root)
    return 0


def _display_path(path: Path, root: Path, *, full_paths: bool) -> str:
    if full_paths:
        return str(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _print_lmstudio_scan(result, *, full_paths: bool = False) -> None:
    print(f"lmstudio  {result.lmstudio_root}")
    for i, root in enumerate(result.shelf_roots):
        label = "shelf" if i == 0 else "shelf+"
        print(f"{label:<9} {root}")
    print()
    print(f"  {'status':<14} {'format':<11} {'size':>10}  model")
    for item in result.items:
        fmt = item.format or "unknown"
        print(
            f"  {item.status:<14} {fmt:<11} {_fmt_size(item.size_bytes):>10}  "
            f"{item.model}"
        )
        print(
            f"  {'':<14} {'':<11} {'':>10}  path: "
            f"{_display_path(item.source_path, result.lmstudio_root, full_paths=full_paths)}"
        )
        if item.target_path:
            shelf_root = result.shelf_roots[0] if result.shelf_roots else item.target_path.parent
            print(
                f"  {'':<14} {'':<11} {'':>10}  shelf: "
                f"{_display_path(item.target_path, shelf_root, full_paths=full_paths)}"
            )
        if item.is_link and item.link_target:
            print(f"  {'':<14} {'':<11} {'':>10}  link: {item.link_target}")
        if item.reason:
            print(f"  {'':<14} {'':<11} {'':>10}  reason: {item.reason}")
    s = result.summary
    print()
    print(
        "summary: "
        f"{s.get('scanned', 0)} scanned, "
        f"{s.get('in_shelf', 0)} in shelf, "
        f"{s.get('lmstudio_only', 0)} lmstudio only, "
        f"{s.get('duplicated', 0)} duplicated, "
        f"{s.get('lmstudio_link', 0)} lmstudio links, "
        f"{s.get('shelf_link', 0)} shelf links, "
        f"{s.get('ambiguous', 0)} ambiguous, "
        f"{s.get('unsupported', 0)} unsupported"
    )


def cmd_lmstudio_scan(args: argparse.Namespace, cfg: Config) -> int:
    result = scan_lmstudio(
        cfg,
        lmstudio_root=Path(args.lmstudio_root).expanduser() if args.lmstudio_root else None,
        format=args.format,
        use_lms=args.use_lms,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        _print_lmstudio_scan(result, full_paths=args.full_paths)
    return 0


def _print_lmstudio_migrate(result) -> None:
    mode = "applied" if result.applied else "dry-run"
    print(f"migration  {mode}")
    print(f"lmstudio   {result.scan.lmstudio_root}")
    print()
    if not result.operations:
        print("(no LM Studio-only supported models to migrate)")
    else:
        print(f"  {'status':<9} {'action':<30} source -> target")
        for op in result.operations:
            print(f"  {op.status:<9} {op.action:<30} {op.source_path} -> {op.target_path}")
            if op.link_path:
                print(f"  {'':<9} {'link':<30} {op.link_path} -> {op.target_path}")
            if op.reason:
                print(f"  {'':<9} {'reason':<30} {op.reason}")
    print()
    s = result.scan.summary
    print(
        "summary: "
        f"{s.get('lmstudio_only', 0)} LM Studio-only supported models, "
        f"{len(result.operations)} planned operations"
    )


def cmd_lmstudio_migrate(args: argparse.Namespace, cfg: Config) -> int:
    if os.name == "nt":
        print(f"model-shelf: {LMSTUDIO_WINDOWS_UNSUPPORTED}", file=sys.stderr)
        return 2
    dry_run = args.dry_run if args.model else (args.dry_run or not args.yes)
    result = migrate_lmstudio(
        cfg,
        model=args.model,
        filename=args.filename,
        lmstudio_root=Path(args.lmstudio_root).expanduser() if args.lmstudio_root else None,
        format=args.format,
        use_lms=args.use_lms,
        link=args.link,
        dry_run=dry_run,
        yes=args.yes or bool(args.model),
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        _print_lmstudio_migrate(result)
    return 0


def _print_lmstudio_link(result) -> None:
    mode = "applied" if result.applied else "dry-run"
    print(f"link       {mode}")
    print(f"lmstudio   {result.scan.lmstudio_root}")
    if result.operation is None:
        print()
        print("(no link operation)")
        return
    op = result.operation
    print()
    print(f"  {'status':<9} {'action':<20} shelf -> lmstudio")
    print(f"  {op.status:<9} {op.action:<20} {op.source_path} -> {op.target_path}")
    if op.reason:
        print(f"  {'':<9} {'reason':<20} {op.reason}")


def cmd_lmstudio_link(args: argparse.Namespace, cfg: Config) -> int:
    if os.name == "nt":
        print(f"model-shelf: {LMSTUDIO_WINDOWS_UNSUPPORTED}", file=sys.stderr)
        return 2
    dry_run = args.dry_run
    result = link_lmstudio_model(
        cfg,
        args.model,
        filename=args.filename,
        lmstudio_root=Path(args.lmstudio_root).expanduser() if args.lmstudio_root else None,
        format=args.format,
        use_lms=args.use_lms,
        dry_run=dry_run,
        yes=args.yes,
        replace_link=args.replace_link,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        _print_lmstudio_link(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="model-shelf",
        description="Local-first resolver for Hugging Face models (gguf, mlx, safetensors).",
    )
    parser.add_argument(
        "--config",
        help="path to config.toml (default: $MODEL_SHELF_CONFIG or ./config.toml)",
        default=None,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_resolve = sub.add_parser("resolve", help="resolve a model to a local path")
    p_resolve.add_argument("repo_id", help='e.g. "Qwen/Qwen3-14B-GGUF"')
    p_resolve.add_argument(
        "--format",
        choices=SUPPORTED_FORMATS,
        default=None,
        help="model format (auto-detected from repo_id if omitted)",
    )
    p_resolve.add_argument(
        "--quant",
        default=None,
        help='quantization tag, required for gguf (e.g. "Q4_K_M")',
    )
    p_resolve.add_argument(
        "--no-download",
        action="store_true",
        help="never reach out to Hugging Face, even on a cache miss",
    )
    p_resolve.add_argument("--json", action="store_true", help="emit JSON")

    sub.add_parser("list", help="list models on the curated shelf")

    p_find = sub.add_parser(
        "find",
        help="search Hugging Face Hub for models matching a query",
    )
    p_find.add_argument("query", help='e.g. "qwen 3 4b mlx 4-bit"')
    p_find.add_argument(
        "--format",
        choices=SUPPORTED_FORMATS,
        default=None,
        help="filter to a single format",
    )
    p_find.add_argument("--limit", type=int, default=10, help="max results (default 10)")
    p_find.add_argument("--json", action="store_true", help="emit JSON")

    p_init = sub.add_parser(
        "init",
        help="create the shelf directory (optionally at a new path)",
    )
    p_init.add_argument(
        "path",
        nargs="?",
        default=None,
        help="shelf location (writes to config); omit to use existing config",
    )

    p_lmstudio = sub.add_parser(
        "lmstudio",
        help="scan or migrate models from an LM Studio installation",
    )
    lmstudio_sub = p_lmstudio.add_subparsers(dest="lmstudio_command", required=True)

    p_lm_scan = lmstudio_sub.add_parser(
        "scan",
        help="scan LM Studio models and report Model Shelf availability",
    )
    p_lm_scan.add_argument(
        "--lmstudio-root",
        default=None,
        help="override LM Studio models directory",
    )
    p_lm_scan.add_argument(
        "--format",
        choices=SUPPORTED_FORMATS,
        default=None,
        help="filter to a single format",
    )
    p_lm_scan.add_argument(
        "--use-lms",
        action="store_true",
        help="also consult `lms ls --json` during root discovery",
    )
    p_lm_scan.add_argument("--json", action="store_true", help="emit JSON")
    p_lm_scan.add_argument(
        "--full-paths",
        action="store_true",
        help="show absolute paths instead of paths relative to displayed roots",
    )

    p_lm_migrate = lmstudio_sub.add_parser(
        "migrate",
        help=argparse.SUPPRESS if os.name == "nt" else "move or link LM Studio models into Model Shelf",
    )
    p_lm_migrate.add_argument(
        "model",
        nargs="?",
        default=None,
        help="optional single model to migrate, e.g. publisher/repo or publisher/repo/file.gguf",
    )
    p_lm_migrate.add_argument(
        "--filename",
        default=None,
        help="GGUF filename when model is given as publisher/repo",
    )
    p_lm_migrate.add_argument(
        "--lmstudio-root",
        default=None,
        help="override LM Studio models directory",
    )
    p_lm_migrate.add_argument(
        "--format",
        choices=SUPPORTED_FORMATS,
        default=None,
        help="filter to a single format",
    )
    p_lm_migrate.add_argument(
        "--use-lms",
        action="store_true",
        help="also consult `lms ls --json` during root discovery",
    )
    p_lm_migrate.add_argument(
        "--link",
        action="store_true",
        help="leave LM Studio paths in place and link Model Shelf to them",
    )
    p_lm_migrate.add_argument(
        "--dry-run",
        action="store_true",
        help="show planned operations without changing files",
    )
    p_lm_migrate.add_argument(
        "--yes",
        action="store_true",
        help="apply bulk changes; single-model migrate applies by default unless --dry-run is passed",
    )
    p_lm_migrate.add_argument("--json", action="store_true", help="emit JSON")

    p_lm_link = lmstudio_sub.add_parser(
        "link",
        help=argparse.SUPPRESS if os.name == "nt" else "link a single Model Shelf-only model into LM Studio",
    )
    p_lm_link.add_argument(
        "model",
        help="model id to link, e.g. publisher/repo or publisher/repo/file.gguf",
    )
    p_lm_link.add_argument(
        "--filename",
        default=None,
        help="GGUF filename when model is given as publisher/repo",
    )
    p_lm_link.add_argument(
        "--lmstudio-root",
        default=None,
        help="override LM Studio models directory",
    )
    p_lm_link.add_argument(
        "--format",
        choices=SUPPORTED_FORMATS,
        default=None,
        help="filter to a single format",
    )
    p_lm_link.add_argument(
        "--use-lms",
        action="store_true",
        help="also consult `lms ls --json` during root discovery",
    )
    p_lm_link.add_argument(
        "--dry-run",
        action="store_true",
        help="show planned operation without changing files",
    )
    p_lm_link.add_argument(
        "--yes",
        action="store_true",
        help="accepted for compatibility; link applies by default unless --dry-run is passed",
    )
    p_lm_link.add_argument(
        "--replace-link",
        action="store_true",
        help="replace an existing LM Studio symlink; never replaces full files or directories",
    )
    p_lm_link.add_argument("--json", action="store_true", help="emit JSON")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    try:
        if args.command == "resolve":
            return cmd_resolve(args, cfg)
        if args.command == "list":
            return cmd_list(args, cfg)
        if args.command == "init":
            return cmd_init(args, cfg)
        if args.command == "find":
            return cmd_find(args, cfg)
        if args.command == "lmstudio":
            if args.lmstudio_command == "scan":
                return cmd_lmstudio_scan(args, cfg)
            if args.lmstudio_command == "migrate":
                return cmd_lmstudio_migrate(args, cfg)
            if args.lmstudio_command == "link":
                return cmd_lmstudio_link(args, cfg)
    except StorageNotAvailableError as e:
        print(f"model-shelf: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"model-shelf: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        # Catch HF API errors (RepositoryNotFoundError, RemoteEntryNotFoundError,
        # HTTPStatusError, etc.) and surface a one-line message instead of a
        # traceback. Last line of the message is usually the most informative.
        msg = str(e).strip().splitlines()[-1] if str(e).strip() else type(e).__name__
        print(f"model-shelf: {type(e).__name__}: {msg}", file=sys.stderr)
        return 3
    return 2


if __name__ == "__main__":
    sys.exit(main())
