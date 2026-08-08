# Model Shelf

A local-first resolver for Hugging Face models — GGUF, MLX, and safetensors. Your agent checks your curated library before downloading.

```
> model-shelf resolve "Qwen/Qwen3-14B-GGUF" --quant Q4_K_M

  shelf  /Volumes/MyDrive/ModelShelf/models/gguf          HIT

  status      found
  source      local_shelf
  format      gguf
  path        /Volumes/MyDrive/ModelShelf/models/gguf/Qwen/Qwen3-14B-GGUF/Qwen3-14B-Q4_K_M.gguf
```

## Why

Local AI workflows download the same model files over and over — across tools, runtimes, and machines. Model Shelf gives you one curated library at a path you own, and one shell command that asks: *do I already have this locally?*

- Handles **GGUF, MLX, and safetensors** through one CLI — format auto-detected from the repo id.
- **Publisher/repo layout** that mirrors Hugging Face Hub (and matches LM Studio's expected structure) — `gguf/Qwen/Qwen3-14B-GGUF/Qwen3-14B-Q4_K_M.gguf` instead of `models--Qwen--Qwen3-14B-GGUF/snapshots/abc.../...`.
- Works with **any** storage you already mount: external SSD, Thunderbolt DAS, NAS, or just an internal folder.
- Downloads land **directly** in the shelf at the friendly path — no parallel Hugging Face cache to manage or clean up.
- A single shell command (`model-shelf resolve … --json`) means any agent that can run shell commands can plug it in — no special protocols, no extra server.

## Install

### Claude Code (one command)

```
/plugin install model-shelf@alexziskind1/model-shelf
```

That's it. The plugin installs a [skill](skills/resolve/SKILL.md) that tells the agent to always resolve through Model Shelf, plus a SessionStart hook that auto-installs the CLI via `uv` on first session. Requires [`uv`](https://docs.astral.sh/uv/) — install with `curl -LsSf https://astral.sh/uv/install.sh | sh` if you don't have it.

### Anywhere else

```bash
uv tool install git+https://github.com/alexziskind1/model-shelf
# or
pip install git+https://github.com/alexziskind1/model-shelf
```

Requires Python 3.11+.

## Configure

Just run:

```bash
model-shelf init
```

What happens depends on what Model Shelf can see:

- **Your external drive already has a `ModelShelf/models/` folder** → it uses that one silently. No prompt, no questions. (Most common: you've used this drive before.)
- **External drives are connected but none has a shelf yet** → interactive picker (arrow keys) showing each drive plus an internal-storage option and a custom-path escape hatch.
- **No external drives connected** → falls back to internal storage (`~/.cache/model-shelf/models`) and tells you.
- **Need explicit control** → `model-shelf init /path/to/shelf` skips detection and uses that path.

Whichever path is picked, Model Shelf creates the three format subfolders under it. Downloads then nest by publisher and repo (mirrors Hugging Face Hub, matches LM Studio's layout):

```
models/
├── gguf/
│   └── Qwen/
│       └── Qwen3-14B-GGUF/
│           └── Qwen3-14B-Q4_K_M.gguf
├── mlx/
│   └── mlx-community/
│       └── Qwen3-14B-4bit/
└── safetensors/
    └── Qwen/
        └── Qwen3-14B/
```

By default `init` does **not** pin a path in the config — discovery handles drive swaps and renames automatically. Pass an explicit path (`model-shelf init /path/to/shelf`) only when you want to pin a specific location in the config.

Switch shelves later by re-running `init`. Override which config file is used with `$MODEL_SHELF_CONFIG` or `--config <path>`. The user-level config (`~/.config/model-shelf/config.toml`) is the only implicit lookup — Model Shelf does not pick up a `./config.toml` from your current directory, so unrelated tools' configs can't accidentally hijack it.

> If you pass a path under `/Volumes/<name>/` and `<name>` isn't currently mounted, `init` fails with a clear error instead of silently writing to the internal SSD.

### Multi-shelf lookup

Model Shelf treats every shelf it can see locally as fair game when resolving a model. On every `resolve` it checks:

1. The primary shelf (configured `shelf_root` if pinned, or auto-discovered if not).
2. Every mounted `/Volumes/*/ModelShelf/models/` directory (any external drive with a shelf).
3. The internal default at `~/.cache/model-shelf/models` (if it exists).

First hit wins. Downloads on a miss still go to the primary. So you can plug in any drive that has a ModelShelf folder, rename your main drive, or have multiple shelves spread across drives — if the file is local *anywhere*, it's used.

### Pinned vs unpinned config

By default the config doesn't pin a specific path. The user-level config looks like:

```toml
allow_downloads = true
```

That's it — no `shelf_root` line. At runtime Model Shelf auto-discovers a primary (first external `/Volumes/*/ModelShelf/models`, else internal). Swap drives, rename them, plug in a different drive entirely — nothing in the config needs to change.

If you *want* to pin a specific path (say, you have two external drives and want downloads to land on a particular one), run `model-shelf init <path>` — that writes `shelf_root` to the config explicitly. Running `model-shelf init` without an argument never pins.

## CLI

```bash
# Setup: auto-detects external drives, prompts or auto-picks, writes config + creates dirs.
model-shelf init

# Or skip detection and use an explicit path:
model-shelf init /Volumes/MyDrive/ModelShelf/models

# Search Hugging Face for a loose query — for when you don't know the exact repo id.
model-shelf find "qwen 3 4b mlx 4-bit" --format mlx --limit 5

# GGUF (format auto-detected; --quant required for gguf)
model-shelf resolve "Qwen/Qwen3-14B-GGUF" --quant Q4_K_M

# MLX (auto-detected from mlx-community/* or *-mlx)
model-shelf resolve "mlx-community/Qwen3-14B-4bit"

# Safetensors (default when nothing else matches)
model-shelf resolve "Qwen/Qwen3-14B"

# Force a specific format
model-shelf resolve "Qwen/Qwen3-14B" --format safetensors

# Never reach out to the network, even on a miss.
model-shelf resolve "Qwen/Qwen3-14B-GGUF" --quant Q4_K_M --no-download

# Emit JSON for scripting.
model-shelf resolve "Qwen/Qwen3-14B-GGUF" --quant Q4_K_M --json

# List what's on the curated shelf (all three format subfolders).
model-shelf list
```

Exit codes: `0` on found/downloaded, `1` on missing.

## Agent integration

If you installed via `/plugin install`, you're done — the bundled skill tells the agent to always call `model-shelf resolve` before any Hugging Face download, and the SessionStart hook keeps the CLI installed. You may want to pre-allow the CLI in permissions so the agent doesn't prompt every time:

```json
{
  "permissions": {
    "allow": ["Bash(model-shelf resolve:*)", "Bash(model-shelf list:*)"]
  }
}
```

For any other agent: copy [`skills/resolve/SKILL.md`](skills/resolve/SKILL.md) into wherever your agent reads instructions from, and allow `model-shelf resolve` as a tool. The agent calls:

```
model-shelf resolve "Qwen/Qwen3-14B-GGUF" --quant Q4_K_M --json
```

and gets back:

```json
{
  "status": "found",
  "source": "hf_cache",
  "path":   "/Volumes/MyDrive/ModelShelf/hf-cache/.../Qwen3-14B-Q4_K_M.gguf",
  "checks": [ ... ]
}
```

## How it works

Format is detected from the repo id (override with `--format`):

| Repo pattern | Format | Notes |
|---|---|---|
| `*-GGUF` (case-insensitive) | `gguf` | requires `--quant` |
| `mlx-community/*` or `*-mlx` | `mlx` | directory of files |
| anything else | `safetensors` | directory of files |

For every resolve request:

1. **Curated shelf** — looks in `shelf_root/<format>/`. Hit → return.
2. **Download** — if `allow_downloads = true`, calls `huggingface_hub` with `local_dir` pointed at the shelf, so the file lands directly at the friendly path. For GGUF, a single rename normalizes the HF capitalization to lowercase. Otherwise returns `status="missing"`.

No parallel cache to manage. `huggingface_hub` writes a small hidden `.cache/huggingface/` subfolder inside the shelf for download metadata (resumability) — it's filtered out of `model-shelf list`.

Curated-shelf paths:

| Repo | Quant | Path under `shelf_root` |
|---|---|---|
| `Qwen/Qwen3-14B-GGUF` | `Q4_K_M` | `gguf/Qwen/Qwen3-14B-GGUF/Qwen3-14B-Q4_K_M.gguf` |
| `meta-llama/Llama-3.1-8B-Instruct-GGUF` | `Q5_K_M` | `gguf/meta-llama/Llama-3.1-8B-Instruct-GGUF/Llama-3.1-8B-Instruct-Q5_K_M.gguf` |
| `mlx-community/Qwen3-14B-4bit` | — | `mlx/mlx-community/Qwen3-14B-4bit/` |
| `Qwen/Qwen3-14B` | — | `safetensors/Qwen/Qwen3-14B/` |

A directory-format shelf hit requires the directory to exist **and** contain a `config.json` — that's the minimal "this is actually a model" sanity check.

## Storage backends

The code is storage-agnostic. Examples of where you might point the two roots:

```toml
# External SSD / Thunderbolt DAS
shelf_root = "/Volumes/MyDAS/ModelShelf/models"

# NAS mount
shelf_root = "/mnt/nas/ai-models"

# Plain internal folder
shelf_root = "~/.cache/model-shelf/models"

# Shared local model root — the same gguf/ and mlx/ subfolders other local
# runtimes read (llama.cpp via LLAMA_CACHE, oMLX via its model_dir), so one
# retention sweep covers every runtime's weights
shelf_root = "~/models"
```

### Managed (read-only) config

If `~/.config/model-shelf/config.toml` is provisioned by a config manager
(home-manager, chezmoi, ...), it may be a read-only symlink. Resolution and
downloads work normally, but `model-shelf init <path>` cannot rewrite the pin
and fails with a clear error. Change `shelf_root` in the manager instead, or
point `$MODEL_SHELF_CONFIG` / `--config` at a writable file for a one-off
override.

## Status

v0.13 — GGUF, MLX, and safetensors via CLI + Python lib. **Publisher/repo nested layout** that mirrors the Hugging Face Hub (and matches what LM Studio expects). Config is unpinned by default: `shelf_root` is optional, auto-discovered at runtime from any mounted `/Volumes/*/ModelShelf/models` (else internal). `model-shelf init <path>` pins; `model-shelf init` without an argument does not. Multi-shelf lookup: every `resolve` checks the primary plus every mounted drive with a ModelShelf folder plus the internal default. `model-shelf find <query>` searches Hugging Face for loose natural-language queries. Mount precheck refuses to write if the configured volume isn't mounted. Roadmap: `verify` subcommand, quantized-safetensors variants (AWQ/GPTQ).

## License

MIT
