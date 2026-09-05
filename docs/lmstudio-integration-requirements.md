# LM Studio Integration Requirements

## Summary

Add a `model-shelf lmstudio` command group with two subcommands:

- `model-shelf lmstudio scan`: inspect the current LM Studio model library and report which models are already represented in Model Shelf.
- `model-shelf lmstudio migrate`: move or link LM Studio models into Model Shelf, then preserve LM Studio compatibility by placing links at the LM Studio paths.

The integration should support every Model Shelf format: GGUF files, MLX model directories, and safetensors model directories.

## Background

Model Shelf already stores models under a Hugging Face-like publisher/repo layout:

```text
<shelf_root>/
  gguf/<publisher>/<repo>/<file>.gguf
  mlx/<publisher>/<repo>/
  safetensors/<publisher>/<repo>/
```

This is close to LM Studio's local model organization. LM Studio's official CLI can list installed models with `lms ls --json`, and LM Studio documents that the CLI reflects the model directory configured in the app's My Models tab. The CLI also supports importing local model files with move, copy, hard-link, and symbolic-link modes.

For this feature, Model Shelf should remain the source of truth for canonical storage. LM Studio should either contain symlinks back to Model Shelf, or, when requested, be linked into Model Shelf without changing the original LM Studio library.

## Goals

- Discover the LM Studio model directory automatically.
- Scan LM Studio's local model files without downloading anything.
- Classify models as GGUF, MLX, safetensors, unsupported, duplicated, or ambiguous.
- Identify whether each LM Studio model is already present in any Model Shelf candidate.
- Provide human-readable and JSON reports suitable for scripting.
- Migrate LM Studio-only models into the primary Model Shelf using safe, resumable filesystem operations.
- Preserve LM Studio usability after migration by replacing moved LM Studio paths with links to Model Shelf.
- Support dry runs and explicit confirmation for operations that move files or replace paths.

## Non-Goals

- Do not download absent models from Hugging Face during scan or migrate.
- Do not rewrite LM Studio application settings in the first version.
- Do not require LM Studio to be running.
- Do not require the `lms` CLI, although it may be used when present.
- Do not infer Hugging Face metadata from the network.
- Do not deduplicate by file hash by default. Hashing multi-GB models should be opt-in.

## Command Shape

```bash
model-shelf lmstudio scan [options]
model-shelf lmstudio migrate [options]
```

Recommended shared options:

```text
--lmstudio-root <path>     Override discovered LM Studio model directory.
--format <format>          Limit to gguf, mlx, or safetensors.
--json                     Emit machine-readable JSON.
--verbose                  Include skipped files and detection details.
```

Recommended scan options:

```text
--use-lms                  Prefer `lms ls --json` inventory when available.
--hash                     Compute content hashes for duplicate detection.
```

Recommended migrate options:

```text
--dry-run                  Show planned operations without changing files. Default for first release.
--yes                      Apply changes without interactive confirmation.
--link                     Leave LM Studio paths unmodified and create Model Shelf links to them.
--replace-existing-links   Replace existing symlinks when they point somewhere else.
--on-conflict <policy>     skip | error | overwrite-link. Default: skip.
--hash                     Verify same-sized conflicts by hash before skipping or linking.
```

## LM Studio Root Discovery

Discovery should produce an ordered list of candidate roots and clearly report which one was selected.

Recommended lookup order:

1. `--lmstudio-root <path>`, if provided.
2. A configured LM Studio model directory from LM Studio settings, when discoverable.
3. The current LM Studio CLI inventory from `lms ls --json`, when `lms` exists and has been initialized.
4. Default filesystem locations:
   - macOS/Linux: `~/.cache/lm-studio/models`
   - Windows: `%USERPROFILE%\.cache\lm-studio\models`
5. Legacy compatibility fallback:
   - `~/.lmstudio/models`

Implementation note: the exact LM Studio settings file and key should be verified during implementation against real current installations. The first version should tolerate multiple possible settings files and fail open to the default path if the setting cannot be read.

When multiple roots appear valid, `scan` should report all valid roots and select the configured/default root unless the user passes `--lmstudio-root`. `migrate` should fail with a clear message if multiple roots contain models and no explicit root is selected.

## Model Detection Rules

### GGUF

A GGUF model is any non-hidden `*.gguf` file below the LM Studio root.

Required metadata:

- `format`: `gguf`
- `source_path`: file path
- `size_bytes`
- `publisher`
- `repo`
- `filename`
- `quant`, when inferable from filename
- `target_path`: `shelf_root/gguf/<publisher>/<repo>/<filename>`

Publisher/repo should be inferred from the relative path when it follows LM Studio's expected nesting:

```text
<lmstudio_root>/<publisher>/<repo>/<file>.gguf
```

If a GGUF file is outside that layout, mark it `ambiguous` unless the user supplies an override in a future mapping file.

### MLX

An MLX model is a directory containing `config.json` plus MLX-specific weight or tokenizer artifacts, such as `*.safetensors`, tokenizer files, and an MLX-indicative repo or metadata signal.

Recommended detection:

- Path contains a publisher/repo pair.
- Directory contains `config.json`.
- Directory name or publisher/repo metadata indicates MLX, for example publisher `mlx-community`, repo token `mlx`, or an LM Studio inventory entry identifying it as MLX.

Target path:

```text
shelf_root/mlx/<publisher>/<repo>/
```

If a directory could be either MLX or generic safetensors, prefer LM Studio CLI metadata when available. Without metadata, classify as `ambiguous` instead of guessing.

### Safetensors

A safetensors model is a directory containing `config.json` and at least one `*.safetensors` file, including sharded models with `*.safetensors.index.json`.

Target path:

```text
shelf_root/safetensors/<publisher>/<repo>/
```

Safetensors detection should ignore nested `.cache`, temporary download folders, hidden files, and partial downloads.

## Scan Report

Human output should be compact and table-like:

```text
lmstudio  /Users/me/.cache/lm-studio/models
shelf     /Volumes/MyDrive/ModelShelf/models

  status         format       model
  duplicated     gguf         Qwen/Qwen3-14B-GGUF/Qwen3-14B-Q4_K_M.gguf
  lmstudio_only  mlx          mlx-community/Qwen3-14B-4bit
  ambiguous      unknown      local/foo
  unsupported    unknown      some-folder

summary: 12 scanned, 2 in shelf, 3 lmstudio only, 4 duplicated, 1 ambiguous, 1 unsupported
```

JSON output should include:

```json
{
  "lmstudio_root": "/Users/me/.cache/lm-studio/models",
  "shelf_roots": ["/Volumes/MyDrive/ModelShelf/models"],
  "items": [
    {
      "status": "lmstudio_only",
      "format": "gguf",
      "publisher": "Qwen",
      "repo": "Qwen3-14B-GGUF",
      "source_path": "...",
      "target_path": "...",
      "size_bytes": 123,
      "is_link": false,
      "link_target": null,
      "reason": null
    }
  ],
  "summary": {
    "scanned": 1,
    "in_shelf": 0,
    "lmstudio_only": 1,
    "duplicated": 0,
    "lmstudio_link": 0,
    "shelf_link": 0,
    "ambiguous": 0,
    "unsupported": 0
  }
}
```

Status values:

- `lmstudio_only`: supported model is present in LM Studio and not represented in Model Shelf.
- `in_shelf`: supported model is present in Model Shelf and not present in LM Studio.
- `duplicated`: both LM Studio and Model Shelf contain full copies at the expected paths.
- `lmstudio_link`: LM Studio path is a symlink to a Model Shelf path.
- `shelf_link`: Model Shelf path is a symlink to the LM Studio path.
- `ambiguous`: format or publisher/repo cannot be determined safely.
- `unsupported`: file or directory is not one of the supported Model Shelf formats.
- `error`: path could not be inspected.

## Migration Semantics

Default migration mode should move supported `lmstudio_only` models into the primary Model Shelf, then create a symlink at the original LM Studio location pointing to the new Model Shelf path.

For a GGUF file:

```text
before:
  <lmstudio_root>/<publisher>/<repo>/<file>.gguf

after:
  <shelf_root>/gguf/<publisher>/<repo>/<file>.gguf
  <lmstudio_root>/<publisher>/<repo>/<file>.gguf -> <shelf_root>/gguf/<publisher>/<repo>/<file>.gguf
```

For MLX and safetensors directories:

```text
before:
  <lmstudio_root>/<publisher>/<repo>/

after:
  <shelf_root>/<format>/<publisher>/<repo>/
  <lmstudio_root>/<publisher>/<repo> -> <shelf_root>/<format>/<publisher>/<repo>
```

With `--link`, reverse the ownership direction:

```text
<lmstudio_root>/<publisher>/<repo>/... remains in place
<shelf_root>/<format>/<publisher>/<repo> -> <lmstudio_root>/<publisher>/<repo>
```

For GGUF with `--link`, link the file into the Model Shelf GGUF repo directory and create parent directories as needed.

## Safety Requirements

- `migrate` must support `--dry-run`; first implementation should default to dry-run unless `--yes` is passed.
- Never delete source data as a standalone step.
- Use same-filesystem `Path.rename()` when possible; otherwise use copy-then-verify-then-replace-with-link.
- Cross-device migration must verify at least size and mtime before replacing the original path with a link.
- If `--hash` is passed, verify a full content hash before considering a copied file safe.
- Skip partial or temporary downloads.
- Refuse to overwrite a non-link existing target unless `--on-conflict overwrite-link` applies to an existing link only.
- If an operation fails mid-migration, leave either the original source or the target copy intact and report remediation instructions.
- Do not follow symlink loops during scanning.
- Treat hidden directories and `.cache` directories as ignored by default.

## Conflict Handling

Target path exists:

- LM Studio path is a symlink into Model Shelf: status `lmstudio_link`, no action.
- Model Shelf path is a symlink to LM Studio: status `shelf_link`, no action.
- Same size and optional hash matches: skip move and optionally relink LM Studio with confirmation.
- Both locations contain full copies: status `duplicated`, default action `skip`.
- Existing target is a symlink: replace only with `--replace-existing-links` or compatible `--on-conflict`.

Source path is already a symlink:

- If it points into any Model Shelf root, status `lmstudio_link`.
- If it points elsewhere and `--link` is requested, link Model Shelf to the resolved source.
- If it points elsewhere and move mode is requested, skip by default to avoid moving through an alias.

## Windows Linking Considerations

Windows support is more complex than macOS/Linux because there is no single link type that safely covers all Model Shelf formats.

Current recommendation:

- Support `model-shelf lmstudio scan` on Windows.
- Hide `model-shelf lmstudio migrate` and `model-shelf lmstudio link` from Windows help.
- If invoked directly, return a clear unsupported message explaining that Windows needs explicit junction/hardlink/symlink handling first.

Why mutation is disabled for now:

- Python `Path.symlink_to()` maps to Windows symbolic links, which often require Developer Mode or elevated privileges.
- Windows symlinks can target files or directories, but privilege and policy behavior varies by machine.
- Junctions are often a better fit for directory models because they are directory-only links and are widely compatible with apps that expect a normal folder.
- Junctions can point to another local drive, including an external drive, but drive-letter instability can break them if Windows remounts the drive under a different letter.
- Junctions do not support GGUF files because GGUF models are single files.
- Hard links can support GGUF-like file linking, but only when source and target are on the same NTFS volume; they do not work across drives.
- External drives may be NTFS, exFAT, FAT, ReFS, network-mounted, or encrypted, and each has different link behavior.
- A disconnected external drive leaves a junction/symlink target inaccessible while the LM Studio path may still appear to exist.
- Removing a junction/symlink should remove only the link, but users may not visually distinguish links from real directories/files in all tools.

Future Windows implementation should add a dedicated link strategy abstraction instead of calling `Path.symlink_to()` directly:

```text
--link-type auto | symlink | junction | hardlink
```

Recommended `auto` behavior:

- macOS/Linux files and directories: symlink.
- Windows MLX/safetensors directories: junction by default.
- Windows GGUF files on the same NTFS volume: hard link or symlink depending on user preference.
- Windows GGUF files across volumes: symlink only if available; otherwise fail with guidance.
- Windows external drive paths: warn when using drive-letter targets and, where practical, prefer stable volume GUID paths.

Future Windows tests should cover:

- Directory junction creation and deletion.
- File symlink creation with and without Developer Mode, where test environments allow it.
- Same-volume hard links for GGUF files.
- Cross-volume GGUF behavior.
- Broken junction/symlink detection when the target is missing.
- Existing full file/directory conflicts.
- Existing junction/symlink replacement with explicit confirmation.

## Implementation Plan

Recommended modules:

- `src/model_shelf/lmstudio.py`: discovery, scan data model, migration planning.
- `src/model_shelf/links.py`: symlink and conflict helpers, kept generic for future integrations.
- `tests/test_lmstudio.py`: root discovery, detection, scan statuses, migration plans, symlink behavior.

Recommended CLI integration:

- Add `lmstudio` subparser in `cli.py`.
- Keep command handlers small and delegate logic to `lmstudio.py`.
- Reuse `SUPPORTED_FORMATS`, `list_shelf_candidates`, `shelf_path_gguf`, and `shelf_path_snapshot`.
- Reuse `_fmt_size` for readable output.

Recommended internal flow:

1. Discover LM Studio root.
2. Load Model Shelf config and check primary shelf availability.
3. Walk LM Studio root and build `LmStudioItem` records.
4. Compute target shelf paths.
5. Compare against all shelf candidates.
6. Render scan report or build migration plan.
7. For migrate, apply the plan only after dry-run/confirmation checks.

## Testing Requirements

Unit tests should cover:

- Root discovery with explicit path, default path, legacy path, and missing path.
- GGUF file detection under publisher/repo layout.
- MLX directory detection with `config.json`.
- Safetensors directory detection with sharded and unsharded weights.
- Ambiguous directory handling.
- Existing Model Shelf hit across primary and additional shelves.
- Move-mode plan generation.
- `--link` plan generation.
- Conflict policies.
- Existing symlink classification.
- Cross-device fallback behavior using mocked rename failure.
- JSON report schema stability.

Manual tests should cover:

- A real LM Studio install with `lms ls --json`.
- A custom LM Studio model directory configured in the app.
- A shelf on an external drive under `/Volumes`.
- Migration while LM Studio is closed.
- LM Studio restart after migration to confirm linked models still appear.

## Open Questions

- Should `migrate` default to dry-run forever, or only for the first release?
- Should Model Shelf prefer LM Studio's `lms ls --json` inventory over filesystem scanning when both are available?
- Do we want a future `--copy` mode, or should ownership always become either Model Shelf-owned move mode or LM Studio-owned `--link` mode?
- Should ambiguous directories be resolvable through a user mapping file?
- Should exact duplicate detection use hashes by default for files below a size threshold?
- Should Windows junctions/hard links be supported in addition to symlinks?

## Documentation Updates

After implementation, update `README.md` with:

- A short "LM Studio integration" section.
- Examples for `scan`, `migrate --dry-run`, `migrate --yes`, and `migrate --link`.
- A warning to close LM Studio before migration.
- A note that scan/migrate do not download from Hugging Face.

## References

- [LM Studio CLI docs](https://lmstudio.ai/docs/cli): `lms` ships with LM Studio, requires LM Studio to have been run once, and reflects the current model directory set in the app's My Models tab.
- [LM Studio `lms ls` docs](https://lmstudio.ai/docs/cli/local-models/ls): `lms ls --json` lists downloaded local models and can include detailed local inventory.
- [LM Studio `lms import` docs](https://lmstudio.ai/docs/cli/local-models/import): `lms import` supports move, copy, hard-link, symbolic-link, and dry-run import behavior.
