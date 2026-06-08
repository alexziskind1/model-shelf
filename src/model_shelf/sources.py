"""Source adapters: turn an existing model location into ImportCandidates.

Each adapter scans one kind of store (a plain directory, the Hugging Face
cache, Ollama's blob store, or LM Studio's model dir) and yields
``ImportCandidate`` records. The core (``importer.py``) does the rest.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from model_shelf.importer import ImportCandidate
from model_shelf.resolver import detect_format

_GGUF_QUANT_RE = re.compile(r"(IQ\d[\w]*|Q\d[\w]*|BF16|F16|F32)$", re.IGNORECASE)


def quant_from_gguf_name(filename: str) -> str | None:
    """Extract the quant token from a GGUF filename, e.g. Qwen3-14B-Q4_K_M -> Q4_K_M."""
    stem = filename[:-5] if filename.lower().endswith(".gguf") else filename
    m = _GGUF_QUANT_RE.search(stem)
    return m.group(1) if m else None


def _dir_size(path: Path) -> int:
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def _repo_id_from_parents(model_path: Path, root: Path) -> str | None:
    """If model_path sits at <root>/<publisher>/<repo>[/...], return 'publisher/repo'."""
    try:
        rel = model_path.relative_to(root)
    except ValueError:
        return None
    parts = rel.parts
    if model_path.is_file():
        parts = parts[:-1]  # drop the filename, keep dir nesting
    if len(parts) == 2:
        return f"{parts[0]}/{parts[1]}"
    return None


def _snapshot_dir_has_weights(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    return any(
        f.suffix in (".safetensors", ".bin", ".npz") for f in path.iterdir() if f.is_file()
    )


def scan_dir(root: Path, *, source_label: str = "dir") -> list[ImportCandidate]:
    """Find GGUF files and model directories under root, deepest match wins.

    A GGUF file is one candidate. A directory with config.json + weights is one
    candidate (and its inner files are not scanned again). repo_id is inferred
    from <publisher>/<repo> nesting, else left None (needs binding).
    """
    root = root.expanduser()
    if not root.is_dir():
        return []

    candidates: list[ImportCandidate] = []
    claimed_dirs: set[Path] = set()

    # Directories first (so we can skip their inner gguf/safetensors files).
    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        if any(d.is_relative_to(c) for c in claimed_dirs):
            continue
        if _snapshot_dir_has_weights(d):
            repo_id = _repo_id_from_parents(d, root)
            fmt = detect_format(repo_id) if repo_id else "safetensors"
            if fmt == "gguf":  # a dir is never gguf; fall back
                fmt = "safetensors"
            candidates.append(ImportCandidate(
                source_path=d, repo_id=repo_id, format=fmt,
                quant=None, size_bytes=_dir_size(d), source_label=source_label,
            ))
            claimed_dirs.add(d)

    for f in sorted(p for p in root.rglob("*.gguf") if p.is_file()):
        if any(f.is_relative_to(c) for c in claimed_dirs):
            continue
        repo_id = _repo_id_from_parents(f, root)
        candidates.append(ImportCandidate(
            source_path=f, repo_id=repo_id, format="gguf",
            quant=quant_from_gguf_name(f.name),
            size_bytes=f.stat().st_size, source_label=source_label,
        ))

    return candidates


def _unmangle_hf_repo(mangled: str) -> str | None:
    """models--Qwen--Qwen3-14B-GGUF -> Qwen/Qwen3-14B-GGUF.

    HF mangles '/' to '--'; a repo id has exactly one '/'. split('--', 1)
    keeps any hyphens inside the repo name intact (Qwen3-14B-GGUF).
    """
    if not mangled.startswith("models--"):
        return None
    rest = mangled[len("models--"):]
    parts = rest.split("--", 1)
    if len(parts) < 2 or not parts[0] or not parts[1]:
        return None
    return f"{parts[0]}/{parts[1]}"


def _active_snapshot(repo_dir: Path) -> Path | None:
    """Return the snapshot dir for refs/main, else the first snapshot present."""
    ref = repo_dir / "refs" / "main"
    snaps = repo_dir / "snapshots"
    if ref.is_file():
        target = snaps / ref.read_text().strip()
        if target.is_dir():
            return target
    if snaps.is_dir():
        for s in sorted(snaps.iterdir()):
            if s.is_dir():
                return s
    return None


def scan_hf_cache(
    *,
    home: Path | None = None,
    hub: Path | None = None,
) -> list[ImportCandidate]:
    """Scan ~/.cache/huggingface/hub. gguf -> file candidate, else dir candidate."""
    if hub is None:
        hub = (home or Path.home()) / ".cache" / "huggingface" / "hub"
    if not hub.is_dir():
        return []

    candidates: list[ImportCandidate] = []
    for repo_dir in sorted(hub.iterdir()):
        if not repo_dir.is_dir() or not repo_dir.name.startswith("models--"):
            continue
        repo_id = _unmangle_hf_repo(repo_dir.name)
        if repo_id is None:
            continue
        snap = _active_snapshot(repo_dir)
        if snap is None:
            continue

        fmt = detect_format(repo_id)
        if fmt == "gguf":
            for f in sorted(snap.glob("*.gguf")):
                candidates.append(ImportCandidate(
                    source_path=f, repo_id=repo_id, format="gguf",
                    quant=quant_from_gguf_name(f.name),
                    size_bytes=f.resolve().stat().st_size, source_label="hf",
                ))
        else:
            if _snapshot_dir_has_weights(snap):
                candidates.append(ImportCandidate(
                    source_path=snap, repo_id=repo_id, format=fmt,
                    quant=None, size_bytes=_dir_size(snap), source_label="hf",
                ))
    return candidates


_OLLAMA_MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"


def _ollama_repo_and_quant(manifest_rel_parts: tuple[str, ...]) -> tuple[str | None, str]:
    """Map a manifest path under manifests/ to (repo_id, quant/tag).

    manifests/hf.co/<user>/<repo>/<tag>             -> (user/repo, tag)
    manifests/registry.ollama.ai/library/<m>/<tag>  -> (ollama-library/<m>, tag)
    other registries                                -> (None, tag)  # needs binding
    """
    parts = manifest_rel_parts
    if len(parts) < 2:
        return None, ""
    registry, *rest, tag = parts
    if registry == "hf.co" and len(rest) >= 2:
        return f"{rest[0]}/{rest[1]}", tag
    if registry == "registry.ollama.ai" and len(rest) >= 2 and rest[0] == "library":
        return f"ollama-library/{rest[1]}", tag
    if registry == "registry.ollama.ai" and len(rest) == 1 and rest[0] != "library":
        return f"ollama-library/{rest[0]}", tag
    return None, tag


def scan_ollama(
    *,
    home: Path | None = None,
    models_root: Path | None = None,
) -> list[ImportCandidate]:
    """Scan ~/.ollama/models. Each manifest's model layer -> a gguf candidate.

    Ollama blobs are content-addressed; the source_path points at the raw blob.
    Import must copy (the CLI rejects --move for this source to keep the CAS
    intact).
    """
    if models_root is None:
        models_root = (home or Path.home()) / ".ollama" / "models"
    manifests = models_root / "manifests"
    blobs = models_root / "blobs"
    if not manifests.is_dir():
        return []

    candidates: list[ImportCandidate] = []
    for manifest in sorted(p for p in manifests.rglob("*") if p.is_file()):
        try:
            data = json.loads(manifest.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        layer = next(
            (lay for lay in data.get("layers", [])
             if lay.get("mediaType") == _OLLAMA_MODEL_MEDIA_TYPE),
            None,
        )
        if layer is None:
            continue
        digest = str(layer.get("digest", ""))
        if ":" not in digest:
            continue
        blob = blobs / f"sha256-{digest.split(':', 1)[1]}"
        if not blob.is_file():
            continue

        rel_parts = manifest.relative_to(manifests).parts
        repo_id, quant = _ollama_repo_and_quant(rel_parts)
        candidates.append(ImportCandidate(
            source_path=blob, repo_id=repo_id, format="gguf",
            quant=quant or None, size_bytes=blob.stat().st_size,
            source_label="ollama",
        ))
    return candidates


def discover_lmstudio_root(home: Path | None = None) -> Path | None:
    """Find LM Studio's models directory.

    Order: ~/.lmstudio/settings.json downloadsFolder (+/models if needed),
    then the default ~/.lmstudio/models. Returns None if nothing is found.
    (Discovery heuristic adapted from PR #3 by @bhorrock.)
    """
    home = home or Path.home()
    base = home / ".lmstudio"

    settings = base / "settings.json"
    if settings.is_file():
        try:
            data = json.loads(settings.read_text())
        except (json.JSONDecodeError, OSError):
            data = {}
        folder = data.get("downloadsFolder")
        if folder:
            p = Path(folder).expanduser()
            if (p / "models").is_dir():
                return p / "models"
            if p.is_dir():
                return p

    default = base / "models"
    return default if default.is_dir() else None


def scan_lmstudio(
    *,
    home: Path | None = None,
    root: Path | None = None,
) -> list[ImportCandidate]:
    """Scan LM Studio's model dir. Its layout is publisher/repo, so this is
    scan_dir against the discovered root with source_label 'lmstudio'."""
    if root is None:
        root = discover_lmstudio_root(home)
    if root is None:
        return []
    return scan_dir(root, source_label="lmstudio")
