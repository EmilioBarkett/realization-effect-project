# Canonical preflight artifact index

The single path and provenance source of truth for the Wave 1–4 model-side
preflight is:

`configs/construct_benchmark/preflight_campaigns/waves1_4_preflight_artifact_index_v1.json`

Validate it before any model or RunPod work:

```bash
./venv/bin/python scripts/validate_preflight_artifact_index.py
```

The command prints one row for every wave × model pair and exits nonzero if a
row is pending or if a path is missing, absolute, outside the repository,
under a traversal/symlink escape, under a failed/superseded root, or has a
changed SHA-256. It also checks inventory-manifest output links, selection
source-inventory links, model revisions, construct sets, gate IDs, and
preflight run-config links. It never calls an API or launches a GPU.

The path policy applies to the indexed top-level artifacts and the inventory
manifest `output_path`/`combined_path` link used to resolve the frozen input.
Some preserved frozen manifests contain absolute paths inside historical
`generated_downstream`, `parent_inventory`, `prompt_audit`, `sources`, or
`audit_path` metadata. Those nested values are explicitly opaque provenance;
they are nonportable records and are excluded from execution resolution. A
clean checkout or RunPod executor must resolve only the repository-relative
paths recorded by the canonical index.

The checked-in index is ready for preflight artifact resolution: both Wave 1
model-specific v4 run configs and the normalized v4 model selections for
Waves 2–4 are recorded with hashes recomputed from the local files. This is
only a provenance/readiness result. The existing historical, failed,
superseded, or full-production paths are not substitutes and must not be
copied into this index.

The indexed Wave 1 selections retain all behavior and steering IDs and apply
only a deterministic collateral rebalance: start from the frozen model-specific
stable-hash selections, remove the highest-ranked selected IDs from an
overrepresented `correct_option` stratum, and add the lowest-ranked unselected
IDs from the deficient stratum. The resulting 16 collateral IDs are exactly
8/8 for correct options 1/2 for both models. No model output or prompt text is
used in this adjustment.

## Completing an entry

The Waves 2–4 implementation owner should update only this index when a new
artifact has been reviewed and frozen:

1. Use a repository-relative path for every artifact; do not use `..`, an
   absolute path, a URI, or a path below a failed/superseded root.
2. Change that artifact's `status` from `pending` to `ready` and record the
   SHA-256 of the exact file bytes.
3. For an inventory, populate its nested ready `manifest` path and SHA-256.
   The manifest's relative `output_path`/`combined_path` must resolve to the
   indexed inventory and its declared output hash must agree.
4. For a model selection, ensure `source_inventory` and
   `source_inventory_sha256` point to the same indexed inventory. Its model,
   revision, construct IDs, gate ID, and gate hash must agree with the row. The
   index's `sha256` is the hash of the selection file itself; when present,
   `selection_sha256` separately records and verifies the selection manifest's
   canonical selection digest.
5. Ensure the gate and preflight run config remain the registered artifacts
   for that row and that their model, revision, construct IDs, and gate ID
   agree. Do not add thresholds or execution settings to this index.
6. Run the validator, the focused index tests, `git diff --check`, and the
   repository's full checks before GPU provisioning.

The index is a provenance guard, not an execution configuration. A `READY`
matrix means only that the referenced files are present and internally
consistent; it does not mean that behavioral or steering gates passed and it
does not authorize confirmatory claims.
