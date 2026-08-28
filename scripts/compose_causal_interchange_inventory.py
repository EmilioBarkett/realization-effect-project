#!/usr/bin/env python3
"""Compose a deterministic matched-episode inventory for C1 residual interchange.

The causal inventory is deliberately separate from the activation inventory.
Each request combines one held-out positive/negative induction pair with one
independent steering-task prompt that is identical for both receivers.  The
model-side runner then swaps the induction state at the registered boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import sys

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.config import load_construct_specs  # noqa: E402
from construct_benchmark.prompts import load_prompt_records  # noqa: E402


SCHEMA_VERSION = "0.1.0"
INDUCTION_SPLIT = "direction_heldout"
TASK_SPLIT = "steering_eval"
SYSTEM_PROMPT = (
    "Respond with only the final answer requested by the user. Do not explain "
    "your reasoning. Output no words, labels, punctuation, markdown, or extra text."
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _metadata(record: Any) -> dict[str, Any]:
    value = getattr(record, "metadata", {})
    return dict(value) if isinstance(value, dict) else {}


def _condition_map(spec: Any) -> tuple[str, str]:
    readout = spec.expected_direction["readout"]
    positive = str(readout["positive_condition"])
    negative = str(readout["negative_condition"])
    if positive == negative:
        raise ValueError(f"{spec.construct_id} has identical positive and negative conditions.")
    return positive, negative


def _group_probe_pairs(records: Iterable[Any], *, construct_id: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(dict)
    for record in records:
        if (
            record.construct_id == construct_id
            and record.split == INDUCTION_SPLIT
            and record.prompt_role == "probe"
        ):
            pair_id = str(record.pair_id or "")
            if not pair_id:
                raise ValueError(f"{record.prompt_id} has no pair_id in {INDUCTION_SPLIT}.")
            condition_id = str(record.condition_id)
            if condition_id in grouped[pair_id]:
                raise ValueError(f"Duplicate {construct_id} induction condition {pair_id}/{condition_id}.")
            grouped[pair_id][condition_id] = record
    return dict(grouped)


def _task_records(records: Iterable[Any], *, construct_id: str) -> list[Any]:
    return sorted(
        (
            record
            for record in records
            if record.construct_id == construct_id
            and record.split == TASK_SPLIT
            and record.prompt_role == "steering"
        ),
        key=lambda record: record.prompt_id,
    )


def _compose_requests(
    records: list[Any],
    specs: dict[str, Any],
    *,
    mode: str,
    pairs_per_construct: int | None,
    tasks_per_construct: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if mode not in {"test", "full"}:
        raise ValueError("mode must be 'test' or 'full'.")
    requests: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for construct_id in sorted(specs):
        spec = specs[construct_id]
        positive, negative = _condition_map(spec)
        pairs = _group_probe_pairs(records, construct_id=construct_id)
        valid_pairs: list[tuple[str, dict[str, Any]]] = []
        for pair_id in sorted(pairs):
            conditions = pairs[pair_id]
            if set(conditions) != {positive, negative}:
                raise ValueError(
                    f"{construct_id} pair {pair_id} must contain exactly "
                    f"{positive!r} and {negative!r}; found {sorted(conditions)}."
                )
            valid_pairs.append((pair_id, conditions))
        tasks = _task_records(records, construct_id=construct_id)
        if not valid_pairs or not tasks:
            raise ValueError(f"{construct_id} has no complete held-out induction pairs or steering tasks.")
        pair_limit = pairs_per_construct if pairs_per_construct is not None else (2 if mode == "test" else len(valid_pairs))
        task_limit = tasks_per_construct if tasks_per_construct is not None else (2 if mode == "test" else len(tasks))
        if pair_limit < 1 or task_limit < 1:
            raise ValueError("pair/task limits must be positive.")
        selected_pairs = valid_pairs[:pair_limit]
        selected_tasks = tasks[:task_limit]
        if len(selected_pairs) < pair_limit or len(selected_tasks) < task_limit:
            raise ValueError(
                f"{construct_id} cannot satisfy requested limits: "
                f"pairs={pair_limit}/{len(valid_pairs)}, tasks={task_limit}/{len(tasks)}."
            )
        counts[construct_id] = {
            "available_pairs": len(valid_pairs),
            "selected_pairs": len(selected_pairs),
            "available_tasks": len(tasks),
            "selected_tasks": len(selected_tasks),
            "selected_requests": len(selected_pairs) * len(selected_tasks),
        }
        for pair_index, (pair_id, conditions) in enumerate(selected_pairs, start=1):
            positive_record = conditions[positive]
            negative_record = conditions[negative]
            for task_index, task in enumerate(selected_tasks, start=1):
                request_id = (
                    f"c1__{construct_id}__heldout_pair_{pair_index:03d}__"
                    f"steering_task_{task_index:03d}"
                )
                task_metadata = _metadata(task)
                requests.append(
                    {
                        "request_id": request_id,
                        "construct_id": construct_id,
                        "positive_source_prompt_id": positive_record.prompt_id,
                        "negative_source_prompt_id": negative_record.prompt_id,
                        "downstream_prompt_id": task.prompt_id,
                        "downstream_task_id": task_metadata.get("task_id", task.prompt_id),
                        "positive_condition": positive,
                        "negative_condition": negative,
                        "positive_induction_prompt": positive_record.prompt_text,
                        "negative_induction_prompt": negative_record.prompt_text,
                        "downstream_prompt": task.prompt_text,
                        "boundary_separator": "\n\n",
                        "prompt_format": "chat",
                        "system_prompt": SYSTEM_PROMPT,
                        "boundary_mode": "last_induction_token",
                        "intervention_timing": "prefill_only",
                        "metadata": {
                            "mode": mode,
                            "induction_split": INDUCTION_SPLIT,
                            "task_split": TASK_SPLIT,
                            "pair_id": pair_id,
                            "task_prompt_family": task.prompt_family,
                            "task_metadata": task_metadata.get("task_metadata", {}),
                        },
                    }
                )
    return requests, counts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-inventory", type=Path, required=True)
    parser.add_argument("--construct-spec", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("test", "full"), default="test")
    parser.add_argument("--pairs-per-construct", type=int, default=None)
    parser.add_argument("--tasks-per-construct", type=int, default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".manifest.json").exists():
        raise SystemExit(f"Output already exists; choose a new path: {args.output}")
    records = list(load_prompt_records(args.prompt_inventory))
    specs = load_construct_specs(args.construct_spec)
    requests, counts = _compose_requests(
        records,
        specs,
        mode=args.mode,
        pairs_per_construct=args.pairs_per_construct,
        tasks_per_construct=args.tasks_per_construct,
    )
    if not requests:
        raise SystemExit("No causal interchange requests were composed.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for request in requests:
            handle.write(json.dumps(request, ensure_ascii=True, sort_keys=True) + "\n")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "matched_episode_causal_inventory",
        "confirmatory": False,
        "mode": args.mode,
        "output": str(args.output),
        "prompt_inventory": str(args.prompt_inventory),
        "prompt_inventory_sha256": _sha256_file(args.prompt_inventory),
        "construct_ids": sorted(specs),
        "construct_spec_paths": {key: str(value) for key, value in sorted(zip(specs, args.construct_spec))},
        "induction_split": INDUCTION_SPLIT,
        "task_split": TASK_SPLIT,
        "prompt_format": "chat",
        "system_prompt_sha256": _sha256_text(SYSTEM_PROMPT),
        "boundary_mode": "last_induction_token",
        "intervention_timing": "prefill_only",
        "counts_by_construct": counts,
        "expected_request_count": len(requests),
        "request_ids": [request["request_id"] for request in requests],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "complete": True,
    }
    manifest["output_sha256"] = _sha256_file(args.output)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "requests": len(requests), "counts_by_construct": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
