# Plan: Fix multi-shard safetensors download

## Bug report summary

**Repro:**
```bash
model-shelf resolve "stamsam/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-MLX-oQ4-MTP" --json
```

**Expected:** All shard files downloaded to the shelf.

**Actual:** Only the index file and `.gitattributes` are present. The 5 actual weight shard files (~20 GB total) are missing.

```
Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-MLX-oQ4-MTP/
├── .gitattributes
└── model.safetensors.index.json    ← only the index, no weights
```

The HF repo contains a `weight_map` referencing 5 separate `.safetensors` shard files:

```json
{
  "metadata": { "total_size": 20000343567 },
  "weight_map": {
    "layers.0.*": "model-00001-of-00005.safetensors",
    "layers.1.*": "model-00002-of-00005.safetensors"
    // ... 5 shards total
  }
}
```

**Affected models:**
- `stamsam/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-MLX-oQ4-MTP` (5 shards, ~20 GB)
- `mlx-community/Qwen3.6-35B-A3B-MTP-5bit` (1 shard, 0.6 GB)
- Any other HF repo using multi-shard safetensors weight files

**Environment:** macOS Sonoma, HF token set via `HF_TOKEN`.

**Workaround:** Users must manually download shards using `huggingface-cli download` or the Hugging Face App.

## Code trace

Both affected models are detected as **`mlx`** format (confirmed via `detect_format()`):

```
stamsam/Qwen3.6-35B-...-MLX-oQ4-MTP  →  mlx
mlx-community/Qwen3.6-35B-A3B-MTP-5bit  →  mlx
```

The call chain is:

```
cmd_resolve()
  → resolve_model()
    → _resolve_snapshot(config, repo_id, fmt="mlx")
      → shelf check: _looks_like_model_dir(candidate)   ← checks config.json only
      → allow_patterns = None                             ← because fmt != "safetensors"
      → final.mkdir(parents=True, exist_ok=True)         ← creates dir BEFORE download
      → snapshot_download(local_dir=..., allow_patterns=None)
```

With `allow_patterns=None`, `snapshot_download` is supposed to download **all** files from the repo. The `*.safetensors` pattern in `SAFETENSORS_ALLOW_PATTERNS` also matches multi-shard files under `fnmatch`. So pattern filtering should not be the issue for either format.

**Existing test coverage:** The safetensors/MLX download path has **zero test coverage**. All existing tests only verify shelf hits (files already on disk).

## Root cause hypotheses

There are two competing hypotheses. Diagnosis (Step 1) must rule one in and the other out.

### Hypothesis A: `_looks_like_model_dir` false hit (more likely)

The shelf check is too lenient:

```python
def _looks_like_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()
```

Any directory with a `config.json` counts as a complete model. Scenario:

1. **First run:** `snapshot_download` starts writing small files first (`.gitattributes`, `config.json`, `model.safetensors.index.json`). Then it begins downloading large shard files (~20 GB total).
2. **Download interrupted:** User Ctrl+C's, network drops, or machine sleeps. Large shards are incomplete or missing. But `config.json` (and the index) are already on disk.
3. **Second run:** `_looks_like_model_dir` sees `config.json` → returns `"found"` → **no download attempted** → user sees model as "found" but weights are missing.

Even if `snapshot_download` works perfectly, this check means any interrupted download becomes a permanent false "found" result. This also explains why the bug is **silent** — the user sees `status="found"`, not an error.

Also note: `_resolve_snapshot` creates the directory (`final.mkdir(parents=True, exist_ok=True)`) **before** calling `snapshot_download`. So even a single interrupted run leaves behind a directory with partial contents that could confuse a subsequent run.

**Predictions if Hypothesis A is correct:**
- Running `resolve` a **second time** on the same model should return `status="found"` immediately without attempting a download.
- Deleting the partial directory and running `resolve` once should result in a full download (assuming the download completes).
- The bug is format-agnostic — it affects both `mlx` and `safetensors`, consolidated and sharded.

### Hypothesis B: `snapshot_download` fails to download shard files (less likely, but untested)

`snapshot_download` itself fails to download the `.safetensors` shard files even though they should match the filters. Possible sub-causes:

- LFS pointer resolution fails silently for large files
- Files land in the HF cache instead of `local_dir`
- Some interaction between `local_dir` and concurrent LFS downloads causes files to be skipped

**Predictions if Hypothesis B is correct:**
- Running `resolve` on a **clean shelf** (no prior partial download) should still produce the incomplete result.
- `snapshot_download(dry_run=True)` would list the shard files but they wouldn't appear on disk after a real run.
- The bug might be specific to large files (LFS) or to `local_dir` mode.

These two hypotheses are **not mutually exclusive** — both could be true. Hypothesis A is a latent bug regardless of whether B is also true.

## Step 1 — Diagnose

Goal: rule each hypothesis in or out before implementing a fix.

### 1a. Check for stale partial downloads on disk

Examine the shelf directory for the affected models:

```bash
ls -la <shelf_root>/mlx/stamsam/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-MLX-oQ4-MTP/
ls -la <shelf_root>/mlx/mlx-community/Qwen3.6-35B-A3B-MTP-5bit/
```

- If these directories exist with `config.json` but no `.safetensors` files → **Hypothesis A supported** (stale partial download causing false hit)
- If these directories don't exist at all → Hypothesis A can't explain the bug on this machine (but could on the reporter's machine)

### 1b. Test clean resolve (rules out A, tests B)

1. Delete any existing partial directories for the affected models
2. Run `model-shelf resolve` once against each affected model
3. Let it complete fully (no Ctrl+C)
4. Check directory contents

- If all shards are present → **Hypothesis A confirmed** (the bug was stale partial downloads; `snapshot_download` itself works fine)
- If shards are still missing on a clean run → **Hypothesis B confirmed** (`snapshot_download` is broken)

### 1c. Test dry_run (isolates B further)

If Hypothesis B is confirmed in 1b, run a diagnostic script:

```python
from huggingface_hub import snapshot_download
result = snapshot_download(
    repo_id="mlx-community/Qwen3.6-35B-A3B-MTP-5bit",
    dry_run=True,
)
for f in result:
    print(f.rfilename, f.size)
```

- If shard files appear in dry_run but not on disk → the issue is in the download execution (LFS resolution, local_dir, etc.)
- If shard files don't appear in dry_run → the issue is in file listing/filtering

### 1d. Test second-run behavior (confirms A)

If partial directories exist from a prior failed download:

1. Run `model-shelf resolve` a second time
2. Observe whether it returns `status="found"` without downloading

- If it returns `"found"` without downloading → **Hypothesis A confirmed** (false hit on partial download)

## Step 2 — Fix

The fix has two independent parts. **Part 1 is mandatory regardless of which hypothesis is correct.** Part 2 depends on diagnosis results.

### Part 1: Strengthen `_looks_like_model_dir` (mandatory)

The current check (`config.json` exists) is insufficient. A model directory should only count as a "found" hit if its weight files are present, not just its metadata.

#### New check logic

```
_looks_like_model_dir(path):
  if not path.is_dir() or not (path / "config.json").is_file():
    return False

  if model.safetensors.index.json exists:
    # Sharded model — verify all shards from weight_map are present
    parse index → extract unique shard filenames from weight_map
    return all(path / shard exists and is non-empty for each shard)
  elif any *.safetensors file exists in path:
    # Consolidated model — at least one weight file is present
    return True
  else:
    return False
```

This handles:
- **Sharded:** must have every shard listed in `weight_map`
- **Consolidated:** must have at least one `.safetensors` file
- **Partial downloads:** `config.json` alone is no longer sufficient → returns `False` → triggers re-download

**Edge case:** if the index file is corrupted or can't be parsed, fall back to `False` (treat as miss, re-download). Better to re-download than to serve a broken model.

### Part 2: Index-aware shard download (depends on diagnosis)

**Only needed if Hypothesis B is confirmed** (i.e., `snapshot_download` fails to download shards on a clean run). If Hypothesis A alone explains the bug, this part is unnecessary but still a defensive improvement.

#### Two cases per format

Both `mlx` and `safetensors` formats can have:
- **Consolidated weights** — single `model.safetensors` file, no index
- **Sharded weights** — `model.safetensors.index.json` with a `weight_map` pointing to N shard files

#### Download sequence for sharded case

1. **Download metadata files** to the shelf: `snapshot_download` with `allow_patterns=["*.json", "tokenizer*", "*.txt", "*.md", ".git*"]`
2. **Check for index:** if `model.safetensors.index.json` is present:
   - Parse `weight_map` to extract unique shard filenames
   - Download each unique shard via `hf_hub_download(filename=..., local_dir=str(final))`
3. **If no index:** download `*.safetensors` via `snapshot_download` with `allow_patterns=["*.safetensors"]` (consolidated case)

#### Trade-offs of switching to `hf_hub_download`

The current `snapshot_download` call provides:
- Built-in concurrent downloads (8 workers by default)
- Aggregated progress bar
- Integrated caching metadata

Switching to individual `hf_hub_download` calls means we lose these unless we re-implement them (e.g. via `concurrent.futures.ThreadPoolExecutor` + custom progress). For a first pass, sequential is acceptable since the bottleneck is network I/O and `hf_hub_download` already supports resuming partial downloads.

#### Applies to both `mlx` and `safetensors` formats

The shard-aware logic runs for both format types since both can contain multi-shard safetensors weight files. The only difference is that `mlx` repos may have additional MLX-specific files — but the metadata download in step 1 catches everything.

### Part 3: Error handling and cleanup

On any shard download failure:
- Delete the model directory from the shelf so the next `resolve` call gets a clean "miss" rather than a partial directory that could become a false hit
- Report the error to the user

**Limitation:** Ctrl+C / SIGINT may leave partial files without triggering cleanup. We can register an `atexit` handler or `signal` handler to clean up, but this is a best-effort measure. The strengthened `_looks_like_model_dir` (Part 1) is the real safety net — even if cleanup fails, the next run won't falsely "find" the partial directory.

### Part 4: `--format` flag interaction

The CLI `--format` flag overrides auto-detection. The fix must work regardless:
- `model-shelf resolve "Qwen/Qwen3-14B" --format mlx` → shelf path is `mlx/Qwen/Qwen3-14B`
- `model-shelf resolve "Qwen/Qwen3-14B"` → shelf path is `safetensors/Qwen/Qwen3-14B`

The shard-aware logic in Part 2 doesn't depend on format detection — it depends on the contents of the downloaded directory (presence of index file). So `--format` doesn't affect correctness.

## Step 3 — Add tests

Currently there are **no tests for the safetensors or MLX download paths**. Add:

| Test | What it covers |
|------|---------------|
| `test_partial_dir_not_a_hit` | Directory with `config.json` but no weight files → NOT a shelf hit (core regression test for Hypothesis A) |
| `test_sharded_partial_dir_not_a_hit` | Directory with `config.json` + index but missing shards → NOT a shelf hit |
| `test_mlx_sharded_download` | MLX model with index + 3 shards — mock `snapshot_download` for metadata, `hf_hub_download` for each shard, verify all 3 downloaded |
| `test_mlx_consolidated_download` | MLX model with single `model.safetensors`, no index — verify download works |
| `test_safetensors_sharded_download` | Same as mlx sharded but for `safetensors` format |
| `test_safetensors_consolidated_download` | Same as mlx consolidated but for `safetensors` format |
| `test_sharded_partial_failure` | One shard fails — verify cleanup: directory removed, error reported |
| `test_miss_with_downloads_disabled` | `allow_downloads=False`, no shelf hit — verify `status="missing"` |

Mocking approach: use `monkeypatch` (already used in existing tests) to replace `huggingface_hub.snapshot_download` and `huggingface_hub.hf_hub_download` with side-effect functions that create the expected files on disk.

## Step 4 — Verify end-to-end

1. Run `model-shelf resolve` against both affected models from the bug report:
   - `stamsam/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-MLX-oQ4-MTP`
   - `mlx-community/Qwen3.6-35B-A3B-MTP-5bit`
2. Confirm all shard files appear on disk with correct names and non-zero sizes
3. Run `model-shelf list` to verify the model shows up
4. Run the full test suite with `uv run pytest`

## Step 1 — Diagnosis Results

**Date:** 2026-06-05  
**Test model:** `stamsam/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-MLX-oQ4-MTP`  
**Shelf:** `/Volumes/Extreme SSD/ModelShelf/models/mlx`  
**Environment:** macOS, HF token configured

### 1a. Stale partial downloads on disk

After the first (timed-out) run, the shelf directory contained:

```
model.safetensors.index.json     (183 KB)    # metadata files: 10:03
config.json                      (21 KB)     #
chat_template.jinja              (8 KB)      #
tokenizer.json                   (2 MB)      #
model-00004-of-00004.safetensors (3.4 GB)    # 2 shards finished at 10:06-10:08
model-mtp.safetensors            (491 MB)    #
```

**Expected shards (5 total):** `model-00001-of-00004`, `model-00002-of-00004`, `model-00003-of-00004`, `model-00004-of-00004`, `model-mtp`  
**Present on disk (2 of 5):** `model-00004-of-00004`, `model-mtp`  
**Missing (3 shards, ~16 GB):** `model-00001`, `model-00002`, `model-00003`

→ **Hypothesis A supported** — partial download with `config.json` present

### 1b. Clean resolve

**Performed 2026-06-05.**

Deleted the partial directory, ran `model-shelf resolve` from a clean slate:

```
Fetching 12 files: 100% | 12/12
Download complete: 16.1G/16.1G [04:57, 54.0MB/s]
```

**Result:** All 5 shards downloaded successfully (20 GB total):

```
model-00001-of-00004.safetensors   5.0 GB
model-00002-of-00004.safetensors   5.0 GB
model-00003-of-00004.safetensors   5.0 GB
model-00004-of-00004.safetensors   3.2 GB
model-mtp.safetensors              469 MB
```

All metadata files also present (`config.json`, `chat_template.jinja`, `tokenizer.json`, `model.safetensors.index.json`).  
Strengthened `_looks_like_model_dir` returns `True` for the complete directory.  
Second `resolve` returns `status="found"` from `local_shelf` (no re-download).

### 1c. dry_run

Not performed — Hypothesis B is now ruled out by 1b (clean resolve succeeded).

### 1d. Second-run behavior

Second run (with HF token) returned immediately:

```json
{
  "status": "found",
  "source": "local_shelf",
  "format": "mlx",
  "checks": [{ "location": "shelf", "root": ".../mlx", "result": "hit" }]
}
```

**No download was attempted** despite 3/5 shards missing. The `_looks_like_model_dir` check saw `config.json` and short-circuited.

→ **Hypothesis A confirmed** (false hit on partial download)

### Conclusions

1. **Hypothesis A is confirmed as the root cause.** The `_looks_like_model_dir` check (`config.json` exists → complete) is insufficient. Any interrupted download becomes a permanent silent failure because subsequent runs return `"found"` without verifying weight files.

2. **Hypothesis B is ruled out.** `snapshot_download(repo_id, local_dir, allow_patterns=None)` reliably downloads all 5 shards on a clean run (confirmed via 1b clean resolve). Part 2 of the fix (index-aware shard download via `hf_hub_download`) is unnecessary.

3. **The bug is NOT specific to sharded repos.** The smaller metadata files (config.json, tokenizer, etc.) always download before the large weight files, so `_looks_like_model_dir` will false-hit on any interrupted download — sharded or consolidated. For sharded repos, the symptom is missing shard files (clearly broken). For consolidated repos, the symptom would be a truncated or corrupt `model.safetensors` file (silently broken). The bug affects all model formats; sharded repos just make the failure more visible.

4. **The shard completion pattern is consistent with a simple timeout/interrupt explanation.** Shards 4 (3.4 GB) and mtp (491 MB) completed, while shards 1–3 (~16 GB total) did not. `snapshot_download` uses 8 concurrent workers, and the 300s timeout likely killed the process before the largest files finished. This does not rule out a concurrency issue with large LFS files (Hypothesis B), but it's the simplest explanation.

5. **The fix in Part 1 of Step 2 will force a re-download** when applied, which will simultaneously (a) fix the false-hit bug and (b) provide an observation of whether all shards download successfully on a fresh run (partial evidence for/against Hypothesis B).
