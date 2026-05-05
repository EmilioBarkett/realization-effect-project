from __future__ import annotations

import csv
import copy
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


DEFAULT_PLAN = Path("configs/emotion_activation/final_inference_prompt_generation_v1.json")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PROMPT_PREFIX = "Read the following short scenario.\n\nScenario:\n"
PROMPT_SUFFIX = "\n\nDo not answer yet. Continue processing the scenario."
FORBIDDEN_EMOTION_LABELS = [
    "regret",
    "regretted",
    "regretting",
    "frustration",
    "frustrated",
    "desperation",
    "desperate",
    "temptation",
    "tempted",
    "anxiety",
    "anxious",
    "caution",
    "cautious",
    "relief",
    "relieved",
    "calm",
]
FORBIDDEN_NEUTRAL_AFFECT_TERMS = [
    "hope",
    "hoping",
    "fear",
    "worried",
    "worry",
    "nervous",
    "panic",
    "panicked",
    "dread",
    "upset",
    "manic",
]
FIRST_PERSON_PATTERN = re.compile(r"\b(?:i|me|my|mine|we|our|ours)\b", re.IGNORECASE)

CSV_FIELDNAMES = [
    "prompt_id",
    "prompt_text",
    "prompt_family",
    "source",
    "source_llm",
    "source_model",
    "generation_cell_id",
    "generation_batch_id",
    "domain",
    "concept_axis",
    "emotion",
    "emotion_intensity",
    "risk_orientation",
    "risk_intensity",
    "casino_context",
    "realization_frame",
    "outcome_valence",
    "behavior_target",
    "control_type",
    "contrast_role",
    "variant_id",
    "model_temperature",
    "model_seed",
    "expected_feature",
    "notes",
]


@dataclass(frozen=True)
class GenerationJob:
    plan_name: str
    model_alias: str
    model_id: str
    cell: dict[str, Any]
    metadata: dict[str, str]
    count: int

    @property
    def batch_id(self) -> str:
        parts = [
            self.plan_name,
            self.model_alias,
            self.cell["cell_id"],
            self.metadata.get("emotion", "none"),
            self.metadata.get("risk_orientation", "neutral"),
            self.metadata.get("realization_frame", "none"),
            self.metadata.get("outcome_valence", "none"),
            self.metadata.get("control_type", "none"),
            self.metadata.get("contrast_role", "none"),
        ]
        if self.cell.get("_batch_suffix"):
            parts.append(str(self.cell["_batch_suffix"]))
        return "__".join(_slugify(part) for part in parts if part)


RequestFn = Callable[[str, list[dict[str, str]], dict[str, Any]], dict[str, Any]]


def _slugify(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "none"


def load_generation_plan(path: str | Path = DEFAULT_PLAN) -> dict[str, Any]:
    plan_path = Path(path)
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(data.get("models"), list) or not data["models"]:
        raise ValueError(f"{plan_path} must define a non-empty models list.")
    if not isinstance(data.get("cells"), list) or not data["cells"]:
        raise ValueError(f"{plan_path} must define a non-empty cells list.")
    return data


def pilot_plan_one_job_per_cell(plan: dict[str, Any], *, count_per_model: int = 1) -> dict[str, Any]:
    """Return a small plan with one representative job per cell and model.

    This is intentionally different from `limit_jobs`: it samples across the
    full taxonomy instead of only taking the earliest expanded cells.
    """

    pilot = copy.deepcopy(plan)
    for cell in pilot["cells"]:
        cell["count_per_model"] = count_per_model
        for plural_key, singular_key in (
            ("emotions", "emotion"),
            ("risk_orientations", "risk_orientation"),
            ("control_types", "control_type"),
            ("contrast_roles", "contrast_role"),
            ("realization_frames", "realization_frame"),
            ("outcome_valences", "outcome_valence"),
        ):
            values = cell.pop(plural_key, None)
            if values:
                cell[singular_key] = values[0]
    return pilot


def _as_list(value: Any, *, default: str) -> list[str]:
    if value is None:
        return [default]
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise ValueError("Expected a string or list of strings in generation plan.")


def iter_generation_jobs(
    plan: dict[str, Any],
    *,
    model_aliases: set[str] | None = None,
    limit_jobs: int | None = None,
) -> Iterable[GenerationJob]:
    plan_name = str(plan.get("name", "openrouter_prompt_generation"))
    default_count = int(plan.get("default_count_per_cell_per_model", 3))
    emitted = 0

    for model in plan["models"]:
        alias = str(model["alias"])
        if model_aliases is not None and alias not in model_aliases:
            continue
        model_id = str(model["model"])

        for cell in plan["cells"]:
            count = int(cell.get("count_per_model", default_count))
            emotions = _as_list(cell.get("emotions"), default=str(cell.get("emotion", "none")))
            risks = _as_list(cell.get("risk_orientations"), default=str(cell.get("risk_orientation", "neutral")))
            frames = _as_list(cell.get("realization_frames"), default=str(cell.get("realization_frame", "none")))
            outcomes = _as_list(cell.get("outcome_valences"), default=str(cell.get("outcome_valence", "none")))
            control_types = _as_list(cell.get("control_types"), default=str(cell.get("control_type", "none")))
            contrast_roles = _as_list(cell.get("contrast_roles"), default=str(cell.get("contrast_role", "positive")))

            for emotion in emotions:
                for risk in risks:
                    for frame in frames:
                        for outcome in outcomes:
                            for control_type in control_types:
                                for contrast_role in contrast_roles:
                                    metadata = {
                                        "domain": str(cell.get("domain", "general")),
                                        "concept_axis": str(cell.get("concept_axis", "unspecified")),
                                        "emotion": emotion,
                                        "emotion_intensity": str(cell.get("emotion_intensity", "none")),
                                        "risk_orientation": risk,
                                        "risk_intensity": str(cell.get("risk_intensity", "none")),
                                        "casino_context": str(cell.get("casino_context", "none")),
                                        "realization_frame": frame,
                                        "outcome_valence": outcome,
                                        "behavior_target": str(cell.get("behavior_target", "none")),
                                        "control_type": control_type,
                                        "contrast_role": contrast_role,
                                        "expected_feature": str(cell.get("expected_feature", cell.get("concept_axis", ""))),
                                    }
                                    yield GenerationJob(
                                        plan_name=plan_name,
                                        model_alias=alias,
                                        model_id=model_id,
                                        cell=cell,
                                        metadata=metadata,
                                        count=count,
                                    )
                                    emitted += 1
                                    if limit_jobs is not None and emitted >= limit_jobs:
                                        return


def _response_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "sae_prompt_generation_batch",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["prompts"],
                "properties": {
                    "prompts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["variant_id", "prompt_text", "notes"],
                            "properties": {
                                "variant_id": {
                                    "type": "string",
                                    "description": "Short lowercase identifier unique within this batch.",
                                },
                                "prompt_text": {
                                    "type": "string",
                                    "description": "The full prompt to feed into residual-stream inference.",
                                },
                                "notes": {
                                    "type": "string",
                                    "description": "Brief rationale or quality note for later audit.",
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def build_generation_messages(plan: dict[str, Any], job: GenerationJob) -> list[dict[str, str]]:
    template = str(
        plan.get(
            "prompt_template",
            PROMPT_PREFIX + "{scenario}" + PROMPT_SUFFIX,
        )
    )
    system = (
        "You generate controlled prompts for sparse-autoencoder activation analysis. "
        "Return only JSON matching the requested schema. Do not include markdown. "
        "Avoid explicit labels like 'the person feels regret' unless the cell asks for lexical adversarial controls. "
        "Keep prompts natural, concise, and structurally consistent inside the batch. "
        "The dataset is designed to separate realization framing, emotion, risk orientation, casino context, "
        "and gambling-behavior cues, so do not collapse those concepts unless the cell explicitly combines them."
    )
    user_payload = {
        "task": (
            "Generate prompt scenarios for an SAE final inference dataset that will test whether "
            "realization framing shifts emotion/risk features that later predict gambling behavior."
        ),
        "count": job.count,
        "cell_id": job.cell["cell_id"],
        "metadata": job.metadata,
        "cell_instructions": job.cell.get("instructions", ""),
        "global_design_rules": plan.get("design_rules", []),
        "prompt_template": template,
        "output_requirement": (
            "Each prompt_text must be the full inference prompt. Put the generated scenario into "
            "the prompt_template before returning it. The returned prompt_text must start exactly with "
            f"{PROMPT_PREFIX!r} and end exactly with {PROMPT_SUFFIX!r}."
        ),
        "forbidden_surface_labels": (
            []
            if job.metadata.get("control_type") == "lexical_adversarial"
            else FORBIDDEN_EMOTION_LABELS
        ),
        "forbidden_terms": list(plan.get("forbidden_terms", []))
        + list(job.cell.get("forbidden_terms", [])),
        "style_requirement": "Use third-person scenario narration only. Do not write first-person monologue or dialogue.",
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, indent=2)},
    ]


def call_openrouter_chat_completion(
    model_id: str,
    messages: list[dict[str, str]],
    options: dict[str, Any],
) -> dict[str, Any]:
    api_key = str(options["api_key"])
    body = {
        "model": model_id,
        "messages": messages,
        "temperature": float(options.get("temperature", 0.8)),
        "max_tokens": int(options.get("max_tokens", 2500)),
        "response_format": _response_schema(),
    }
    if options.get("seed") is not None:
        body["seed"] = int(options["seed"])

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if options.get("http_referer"):
        headers["HTTP-Referer"] = str(options["http_referer"])
    if options.get("x_title"):
        headers["X-Title"] = str(options["x_title"])

    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(options.get("timeout", 120))) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ValueError(f"OpenRouter HTTP {exc.code}: {body}") from exc


def _extract_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenRouter response did not include choices.")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("OpenRouter response choice did not include a message.")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        return "".join(text_parts)
    raise ValueError("OpenRouter response message did not include string content.")


def parse_generated_prompts(response: dict[str, Any]) -> list[dict[str, str]]:
    content = _extract_content(response)
    data = json.loads(content)
    prompts = data.get("prompts")
    if not isinstance(prompts, list):
        raise ValueError("Model JSON response must contain a prompts list.")
    parsed: list[dict[str, str]] = []
    for index, row in enumerate(prompts, start=1):
        if not isinstance(row, dict):
            raise ValueError("Each generated prompt must be an object.")
        prompt_text = str(row.get("prompt_text", "")).strip()
        if not prompt_text:
            raise ValueError("Generated prompt_text cannot be empty.")
        parsed.append(
            {
                "variant_id": str(row.get("variant_id") or f"variant_{index:03d}").strip(),
                "prompt_text": prompt_text,
                "notes": str(row.get("notes", "")).strip(),
            }
        )
    return parsed


def _validate_generated_prompt_text(row: dict[str, str], job: GenerationJob) -> None:
    prompt_text = row["prompt_text"]
    if not prompt_text.startswith(PROMPT_PREFIX) or not prompt_text.endswith(PROMPT_SUFFIX):
        raise ValueError(
            f"{job.batch_id} prompt {row['variant_id']} did not use the exact prompt template."
        )

    if job.metadata.get("control_type") == "lexical_adversarial":
        return

    scenario = prompt_text.removeprefix(PROMPT_PREFIX).removesuffix(PROMPT_SUFFIX).lower()
    if FIRST_PERSON_PATTERN.search(scenario):
        raise ValueError(
            f"{job.batch_id} prompt {row['variant_id']} uses first-person narration."
        )

    leaked = [
        label
        for label in FORBIDDEN_EMOTION_LABELS
        if re.search(rf"\b{re.escape(label)}\b", scenario)
    ]
    if leaked:
        raise ValueError(
            f"{job.batch_id} prompt {row['variant_id']} contains explicit emotion label(s): "
            + ", ".join(leaked)
        )

    if job.metadata.get("emotion") == "none":
        affect_terms = [
            term
            for term in FORBIDDEN_NEUTRAL_AFFECT_TERMS
            if re.search(rf"\b{re.escape(term)}\b", scenario)
        ]
        if affect_terms:
            raise ValueError(
                f"{job.batch_id} prompt {row['variant_id']} contains affective term(s) "
                "despite emotion=none: " + ", ".join(affect_terms)
            )

    forbidden_terms = [
        str(term).strip().lower()
        for term in job.cell.get("_plan_forbidden_terms", [])
        if str(term).strip()
    ]
    forbidden_hits = [
        term
        for term in forbidden_terms
        if re.search(rf"\b{re.escape(term)}\b", scenario)
    ]
    if forbidden_hits:
        raise ValueError(
            f"{job.batch_id} prompt {row['variant_id']} contains forbidden term(s): "
            + ", ".join(forbidden_hits)
        )


def _request_with_retries(
    request_fn: RequestFn,
    model_id: str,
    messages: list[dict[str, str]],
    options: dict[str, Any],
) -> dict[str, Any]:
    retries = int(options.get("retries", 2))
    delay = float(options.get("retry_delay", 3))
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return request_fn(model_id, messages, options)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(delay * (2**attempt))
    assert last_error is not None
    raise last_error


def rows_for_job(
    plan: dict[str, Any],
    job: GenerationJob,
    *,
    request_fn: RequestFn = call_openrouter_chat_completion,
    options: dict[str, Any],
) -> list[dict[str, str]]:
    plan_forbidden_terms = list(plan.get("forbidden_terms", [])) + list(job.cell.get("forbidden_terms", []))
    if plan_forbidden_terms:
        job = GenerationJob(
            plan_name=job.plan_name,
            model_alias=job.model_alias,
            model_id=job.model_id,
            cell={**job.cell, "_plan_forbidden_terms": plan_forbidden_terms},
            metadata=job.metadata,
            count=job.count,
        )
    messages = build_generation_messages(plan, job)
    validation_retries = int(options.get("validation_retries", 2))
    last_error: Exception | None = None

    for attempt in range(validation_retries + 1):
        response = _request_with_retries(request_fn, job.model_id, messages, options)
        try:
            generated = parse_generated_prompts(response)
            if len(generated) > job.count:
                generated = generated[: job.count]
            if len(generated) != job.count:
                raise ValueError(
                    f"{job.batch_id} returned {len(generated)} prompts, expected {job.count}."
                )

            rows: list[dict[str, str]] = []
            for index, prompt in enumerate(generated, start=1):
                variant_id = _slugify(prompt["variant_id"] or f"variant_{index:03d}")
                prompt = {**prompt, "variant_id": variant_id}
                _validate_generated_prompt_text(prompt, job)
                prompt_id = f"{job.batch_id}__{variant_id}"
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "prompt_text": prompt["prompt_text"],
                        "prompt_family": str(job.cell.get("prompt_family", plan.get("name", "openrouter_generated"))),
                        "source": "openrouter_generated",
                        "source_llm": job.model_alias,
                        "source_model": job.model_id,
                        "generation_cell_id": str(job.cell["cell_id"]),
                        "generation_batch_id": job.batch_id,
                        "domain": job.metadata["domain"],
                        "concept_axis": job.metadata["concept_axis"],
                        "emotion": "" if job.metadata["emotion"] == "none" else job.metadata["emotion"],
                        "emotion_intensity": job.metadata["emotion_intensity"],
                        "risk_orientation": job.metadata["risk_orientation"],
                        "risk_intensity": job.metadata["risk_intensity"],
                        "casino_context": job.metadata["casino_context"],
                        "realization_frame": job.metadata["realization_frame"],
                        "outcome_valence": job.metadata["outcome_valence"],
                        "behavior_target": job.metadata["behavior_target"],
                        "control_type": job.metadata["control_type"],
                        "contrast_role": job.metadata["contrast_role"],
                        "variant_id": variant_id,
                        "model_temperature": str(options.get("temperature", "")),
                        "model_seed": str(options.get("seed", "")),
                        "expected_feature": job.metadata["expected_feature"],
                        "notes": prompt["notes"],
                    }
                )
            prompt_ids = [row["prompt_id"] for row in rows]
            duplicate_prompt_ids = sorted(
                prompt_id for prompt_id in set(prompt_ids) if prompt_ids.count(prompt_id) > 1
            )
            if duplicate_prompt_ids:
                raise ValueError(
                    f"{job.batch_id} returned duplicate prompt_id(s): "
                    + ", ".join(duplicate_prompt_ids[:5])
                )
            return rows
        except ValueError as exc:
            last_error = exc
            if attempt >= validation_retries:
                break
            messages = messages + [
                {
                    "role": "assistant",
                    "content": _extract_content(response),
                },
                {
                    "role": "user",
                    "content": (
                        "Regenerate the full batch. The previous batch failed validation: "
                        f"{exc}. Return exactly the same JSON schema, with exactly "
                        f"{job.count} prompts, and avoid the validation issue."
                    ),
                },
            ]

    assert last_error is not None
    raise last_error


def read_existing_prompt_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["prompt_id"] for row in csv.DictReader(handle) if row.get("prompt_id")}


def validate_unique_prompt_ids(path: Path) -> None:
    if not path.exists():
        return
    seen: set[str] = set()
    duplicates: list[str] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            prompt_id = row.get("prompt_id", "")
            if not prompt_id:
                continue
            if prompt_id in seen:
                duplicates.append(prompt_id)
            seen.add(prompt_id)
    if duplicates:
        sample = ", ".join(sorted(set(duplicates))[:5])
        raise ValueError(
            f"{path} contains duplicate prompt_id values; clean it before resuming: {sample}"
        )


def write_prompt_rows(path: Path, rows: Iterable[dict[str, str]], *, append: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    count = 0
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES, lineterminator="\n")
        if mode == "w":
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDNAMES})
            count += 1
    return count


def _iter_request_jobs(job: GenerationJob, *, max_prompts_per_request: int) -> Iterable[GenerationJob]:
    if max_prompts_per_request <= 0 or job.count <= max_prompts_per_request:
        yield job
        return

    remaining = job.count
    chunk_index = 1
    while remaining > 0:
        chunk_count = min(max_prompts_per_request, remaining)
        yield GenerationJob(
            plan_name=job.plan_name,
            model_alias=job.model_alias,
            model_id=job.model_id,
            cell={**job.cell, "_batch_suffix": f"part_{chunk_index:03d}"},
            metadata=job.metadata,
            count=chunk_count,
        )
        remaining -= chunk_count
        chunk_index += 1


def generate_prompt_csv(
    plan: dict[str, Any],
    output_path: Path,
    *,
    request_fn: RequestFn = call_openrouter_chat_completion,
    api_key: str,
    model_aliases: set[str] | None = None,
    limit_jobs: int | None = None,
    resume: bool = False,
) -> int:
    generation_options = dict(plan.get("generation", {}))
    generation_options["api_key"] = api_key
    if resume:
        validate_unique_prompt_ids(output_path)
    existing_ids = read_existing_prompt_ids(output_path) if resume else set()
    append = resume and output_path.exists()
    written = 0
    max_prompts_per_request = int(generation_options.get("max_prompts_per_request", 0) or 0)
    request_options = {**generation_options, "max_prompts_per_request": 0}

    for job in iter_generation_jobs(plan, model_aliases=model_aliases, limit_jobs=limit_jobs):
        parent_expected_ids = {
            prompt_id for prompt_id in existing_ids if prompt_id.startswith(f"{job.batch_id}__")
        }
        if resume and len(parent_expected_ids) >= job.count:
            print(f"skip existing batch {job.batch_id}")
            continue

        for request_job in _iter_request_jobs(
            job,
            max_prompts_per_request=max_prompts_per_request,
        ):
            parent_expected_ids = {
                prompt_id
                for prompt_id in existing_ids
                if prompt_id.startswith(f"{job.batch_id}__")
            }
            parent_remaining = job.count - len(parent_expected_ids)
            if parent_remaining <= 0:
                break

            expected_ids = {
                prompt_id
                for prompt_id in existing_ids
                if prompt_id.startswith(f"{request_job.batch_id}__")
            }
            if resume and len(expected_ids) >= request_job.count:
                print(f"skip existing batch {request_job.batch_id}")
                continue
            if request_job.count > parent_remaining:
                request_job = GenerationJob(
                    plan_name=request_job.plan_name,
                    model_alias=request_job.model_alias,
                    model_id=request_job.model_id,
                    cell=request_job.cell,
                    metadata=request_job.metadata,
                    count=parent_remaining,
                )
            rows = rows_for_job(
                plan,
                request_job,
                request_fn=request_fn,
                options=request_options,
            )
            rows = [row for row in rows if row["prompt_id"] not in existing_ids]
            if not rows:
                continue
            written += write_prompt_rows(output_path, rows, append=append or written > 0)
            append = True
            existing_ids.update(row["prompt_id"] for row in rows)
            print(f"wrote {len(rows)} prompts for {request_job.batch_id}")
    return written
