#!/usr/bin/env python3
"""Plan and materialize deterministic benchmark inventory shards.

This command performs no model/API work.  It writes only new shard files and
immutable manifests below the requested output directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.distributed_contracts import canonical_hash, load_json_object  # noqa: E402
from construct_benchmark.sharding import (  # noqa: E402
    build_shard_plan,
    write_shard_outputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create deterministic, pair-preserving benchmark inventory shards without model calls."
    )
    parser.add_argument("--inventory", type=Path, required=True, help="Frozen CSV, JSONL, or JSON inventory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--shard-count",
        "--shards",
        dest="shard_count",
        type=int,
        default=None,
        help="Physical shard count (3, 4, or 5).",
    )
    parser.add_argument(
        "--worker-count",
        "--workers",
        dest="worker_count",
        type=int,
        default=None,
        help="Subprocess slots (3, 4, or 5); a four-construct/three-worker run emits four shards.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-config", type=Path, default=None)
    parser.add_argument("--run-config-hash", default=None)
    parser.add_argument("--run-mode", default="test")
    confirmatory = parser.add_mutually_exclusive_group()
    confirmatory.add_argument("--confirmatory", dest="confirmatory", action="store_true")
    confirmatory.add_argument("--non-confirmatory", dest="confirmatory", action="store_false")
    parser.set_defaults(confirmatory=None)
    parser.add_argument("--parent-manifest", type=Path, default=None)
    parser.add_argument("--construct-id", action="append", dest="construct_ids", default=None)
    parser.add_argument("--expected-request-ids", default=None)
    parser.add_argument("--expected-observation-ids", default=None)
    parser.add_argument("--split-construct-id", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the deterministic plan without writing.")
    return parser


def _parent_ids(path: Path | None, *, key: str) -> set[str] | None:
    if path is None:
        return None
    payload = load_json_object(path, label="parent manifest")
    values: set[str] = set()
    for candidate in (
        key,
        f"expected_{key}",
        f"parent_{key}",
        "owned_request_ids" if key == "request_ids" else "owned_observation_ids",
    ):
        raw = payload.get(candidate)
        if isinstance(raw, list):
            values.update(str(item) for item in raw)
    shards = payload.get("shards")
    if isinstance(shards, list):
        for shard in shards:
            if isinstance(shard, dict):
                candidate = "owned_request_ids" if key == "request_ids" else "expected_observation_ids"
                raw = shard.get(candidate, shard.get(key))
                if isinstance(raw, list):
                    values.update(str(item) for item in raw)
    return values or None


def _run_config_hash(path: Path | None, explicit: str | None) -> str | None:
    if explicit is not None:
        return explicit
    if path is None:
        return None
    return canonical_hash(load_json_object(path, label="run config"))


def main() -> None:
    args = _parser().parse_args()
    confirmatory = args.confirmatory
    if confirmatory is None:
        confirmatory = args.run_mode == "full"
    expected_construct_ids = args.construct_ids or _parent_ids(args.parent_manifest, key="construct_ids")
    expected_request_ids = args.expected_request_ids or _parent_ids(args.parent_manifest, key="request_ids")
    expected_observation_ids = args.expected_observation_ids or _parent_ids(
        args.parent_manifest, key="observation_ids"
    )
    plan = build_shard_plan(
        args.inventory,
        shard_count=args.shard_count,
        worker_count=args.worker_count,
        seed=args.seed,
        run_config_hash=_run_config_hash(args.run_config, args.run_config_hash),
        run_mode=args.run_mode,
        confirmatory=confirmatory,
        expected_request_ids=expected_request_ids,
        expected_construct_ids=expected_construct_ids,
        expected_observation_ids=expected_observation_ids,
        split_construct_id=args.split_construct_id,
    )
    if args.dry_run:
        print(json.dumps(plan.to_mapping(), indent=2, sort_keys=True))
        return
    suffix = args.inventory.suffix or ".jsonl"
    report = write_shard_outputs(plan, args.output_dir, inventory_suffix=suffix)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
