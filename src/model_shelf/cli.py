"""Command-line interface for Model Shelf."""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

from model_shelf.config import load_config, writable_config_path, write_config
from model_shelf.detect import StorageCandidate, detect_storage_candidates
from model_shelf.importer import execute_import, plan_import
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
from model_shelf.sources import scan_dir, scan_hf_cache, scan_lmstudio, scan_ollama


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


# The "dir" source is handled via the positional PATH branch in cmd_import,
# so it is intentionally not in this table (which maps only the --from choices).
_IMPORT_SCANNERS = {
    "hf": lambda: scan_hf_cache(),
    "ollama": lambda: scan_ollama(),
    "lmstudio": lambda: scan_lmstudio(),
}


def _bind_repo_ids(candidates: list) -> None:
    """Interactively bind repo_id for candidates that lack one (TTY only)."""
    import questionary

    for c in candidates:
        if c.repo_id is not None:
            continue
        answer = questionary.text(
            f"repo_id for {c.source_path.name} (publisher/repo, blank to skip):"
        ).ask()
        if answer and "/" in answer:
            c.repo_id = answer.strip()


def _print_import_plan(report, *, dry_run: bool) -> None:
    if not report.operations:
        print("  (nothing to import)")
        return
    for op in report.operations:
        tgt = op.target_path if op.target_path else "—"
        print(f"  [{op.status:<16}] {op.action:<5} {op.candidate.source_label:<8} "
              f"{op.candidate.repo_id or '?'}")
        print(f"       {op.candidate.source_path}")
        print(f"    -> {tgt}  ({_fmt_size(op.candidate.size_bytes)})")
        if op.note:
            print(f"       note: {op.note}")
    head = "DRY-RUN — would import" if dry_run else "imported"
    print(f"\n  {head} {_fmt_size(report.bytes_imported)}")
    if report.bytes_reclaimable:
        print(f"  reclaimable on source: {_fmt_size(report.bytes_reclaimable)}")
    if dry_run:
        print("  re-run with --apply to execute.")


def cmd_import(args: argparse.Namespace, cfg: Config) -> int:
    """Import existing models into the shelf (copy/move, no symlinks).

    Returns 1 only if a transfer failed; idempotent/again-nothing-to-do runs return 0.
    """
    check_storage_available(cfg)
    action = "move" if args.move else "copy"

    if bool(args.path) == bool(args.source_from):
        print("model-shelf: provide exactly one of PATH or --from", file=sys.stderr)
        return 2
    if args.source_from == "ollama" and action == "move":
        print("model-shelf: --move is not supported for --from ollama "
              "(Ollama's content-addressed store would break); use --copy.",
              file=sys.stderr)
        return 2

    if args.path:
        candidates = scan_dir(Path(args.path).expanduser())
    else:
        candidates = _IMPORT_SCANNERS[args.source_from]()

    interactive = sys.stdin.isatty() and not args.json
    if interactive:
        _bind_repo_ids(candidates)

    ops = plan_import(candidates, cfg.shelf_root)
    dry_run = not args.apply
    report = execute_import(ops, action=action, dry_run=dry_run)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        _print_import_plan(report, dry_run=dry_run)

    # rc contract: a hard transfer failure is the only error. Idempotent re-runs
    # (all already_in_shelf), empty scans, conflicts and needs_binding are shown
    # in the output but are NOT script-failing — matches cp/rsync semantics.
    failed = any(op.status == "failed" for op in report.operations)
    return 1 if failed else 0


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

    p_import = sub.add_parser(
        "import",
        help="import existing models into the shelf (copy/move, no symlinks)",
    )
    p_import.add_argument(
        "path", nargs="?", default=None,
        help="a directory to scan (mutually exclusive with --from)",
    )
    p_import.add_argument(
        "--from", dest="source_from", default=None,
        choices=["hf", "ollama", "lmstudio"],
        help="import from a known source instead of a path",
    )
    p_import.add_argument("--copy", dest="move", action="store_false", default=False,
                          help="copy into the shelf (default)")
    p_import.add_argument("--move", dest="move", action="store_true",
                          help="move into the shelf, freeing source space "
                               "(rejected for --from ollama)")
    p_import.add_argument("--apply", action="store_true",
                          help="execute the plan (default is dry-run)")
    p_import.add_argument("--json", action="store_true", help="emit JSON")

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
        if args.command == "import":
            return cmd_import(args, cfg)
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
