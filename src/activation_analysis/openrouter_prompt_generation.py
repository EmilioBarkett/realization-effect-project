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


DEFAULT_PLAN = Path("configs/activation_analysis/realization_vector_generation_v1.json")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PROMPT_PREFIX = "Read the following short scenario.\n\nScenario:\n"
PROMPT_SUFFIX = "\n\nDo not answer yet. Continue processing the scenario."
CASINO_BEHAVIOR_INSTRUCTION = (
    "Respond with exactly two integers on separate lines:\n"
    "Line 1: total session wager in CHF from 1 to 1000\n"
    "Line 2: machine risk preference from 1 to 5"
)
FINANCE_BEHAVIOR_INSTRUCTION = (
    "Respond with exactly two integers on separate lines:\n"
    "Line 1: allocation amount from 1 to 1000\n"
    "Line 2: risk preference from 1 to 5"
)
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
    "prompt_generation_mode",
    "prompt_framing",
    "split",
    "pair_id",
    "pair_role",
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
    "amount_bucket",
    "risk_context",
    "emotion_context",
    "confound_axis",
    "behavior_target",
    "asks_for_behavior",
    "expected_output_format",
    "control_type",
    "contrast_role",
    "variant_id",
    "model_temperature",
    "model_seed",
    "expected_feature",
    "expected_direction",
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
            self.cell.get("split", "unsplit"),
            self.metadata.get("domain", "general"),
            self.metadata.get("emotion", "none"),
            self.metadata.get("risk_orientation", "neutral"),
            self.metadata.get("realization_frame", "none"),
            self.metadata.get("outcome_valence", "none"),
            self.cell.get("amount_bucket", "amount_none"),
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
            ("domains", "domain"),
            ("splits", "split"),
            ("amount_buckets", "amount_bucket"),
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
            domains = _as_list(cell.get("domains"), default=str(cell.get("domain", "general")))
            splits = _as_list(cell.get("splits"), default=str(cell.get("split", "")))
            amount_buckets = _as_list(cell.get("amount_buckets"), default=str(cell.get("amount_bucket", "")))

            for domain in domains:
                for split in splits:
                    for amount_bucket in amount_buckets:
                        expanded_cell = {
                            **cell,
                            "domain": domain,
                            "split": split,
                            "amount_bucket": amount_bucket,
                        }
                        for emotion in emotions:
                            for risk in risks:
                                for frame in frames:
                                    for outcome in outcomes:
                                        for control_type in control_types:
                                            for contrast_role in contrast_roles:
                                                metadata = {
                                                    "domain": domain,
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
                                                    cell=expanded_cell,
                                                    metadata=metadata,
                                                    count=count,
                                                )
                                                emitted += 1
                                                if limit_jobs is not None and emitted >= limit_jobs:
                                                    return


def _single_prompt_response_schema() -> dict[str, Any]:
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


def _paired_contrast_response_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "activation_vector_paired_generation_batch",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["pairs"],
                "properties": {
                    "pairs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "pair_id",
                                "paper_open_prompt_text",
                                "realized_closed_prompt_text",
                                "notes",
                            ],
                            "properties": {
                                "pair_id": {
                                    "type": "string",
                                    "description": "Short lowercase identifier unique within this batch.",
                                },
                                "paper_open_prompt_text": {
                                    "type": "string",
                                    "description": "Full prompt for the open/unrealized/paper version.",
                                },
                                "realized_closed_prompt_text": {
                                    "type": "string",
                                    "description": "Full prompt for the closed/finalized/realized version.",
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


def _response_schema_for_plan(plan: dict[str, Any]) -> dict[str, Any]:
    mode = str(plan.get("generation_mode", "single_prompt"))
    if mode == "paired_contrast":
        return _paired_contrast_response_schema()
    if mode == "single_prompt":
        return _single_prompt_response_schema()
    raise ValueError(f"Unsupported generation_mode: {mode}")


def build_generation_messages(plan: dict[str, Any], job: GenerationJob) -> list[dict[str, str]]:
    template = str(
        plan.get(
            "prompt_template",
            PROMPT_PREFIX + "{scenario}" + PROMPT_SUFFIX,
        )
    )
    mode = str(plan.get("generation_mode", "single_prompt"))
    system = (
        "You generate controlled prompts for residual-stream activation analysis. "
        "Return only JSON matching the requested schema. Do not include markdown. "
        "Avoid explicit labels like 'the person feels regret' unless the cell asks for lexical adversarial controls. "
        "Keep prompts natural, concise, and structurally consistent inside the batch. "
        "The dataset is designed to separate realization framing, emotion, risk orientation, casino context, "
        "and gambling-behavior cues, so do not collapse those concepts unless the cell explicitly combines them."
    )
    user_payload = {
        "task": (
            "Generate prompt scenarios for an activation-vector dataset that will test whether "
            "realization framing is represented distinctly from risk/emotion and whether it predicts "
            "or changes risk-taking behavior."
        ),
        "count": job.count,
        "generation_mode": mode,
        "cell_id": job.cell["cell_id"],
        "metadata": job.metadata,
        "cell_instructions": job.cell.get("instructions", ""),
        "global_design_rules": plan.get("design_rules", []),
        "prompt_template": template,
        "output_requirement": (
            "Each returned prompt text must be the full prompt for residual-stream logging. Put the "
            "generated scenario into the prompt_template before returning it. Every prompt text must "
            f"start exactly with {PROMPT_PREFIX!r} and end exactly with {PROMPT_SUFFIX!r}."
        ),
        "paired_contrast_requirement": (
            "For generation_mode=paired_contrast, return exactly count matched pairs. Each pair must "
            "contain the same actor, domain, amount bucket, outcome valence, and event structure in both "
            "versions. The paper_open_prompt_text should frame the outcome as visible but still open, "
            "pending, held, or not settled. The realized_closed_prompt_text should frame the same outcome "
            "as sold, paid, received, cashed out, posted, submitted, finalized, or otherwise closed. "
            "Do not put labels like 'paper_open' or 'realized_closed' inside prompt_text."
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
        "response_format": options.get("response_schema", _single_prompt_response_schema()),
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


def _extract_content_or_empty(response: dict[str, Any]) -> str:
    try:
        return _extract_content(response)
    except ValueError:
        return ""


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


def parse_generated_pairs(response: dict[str, Any]) -> list[dict[str, str]]:
    content = _extract_content(response)
    data = json.loads(content)
    pairs = data.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("Model JSON response must contain a pairs list.")
    parsed: list[dict[str, str]] = []
    for index, row in enumerate(pairs, start=1):
        if not isinstance(row, dict):
            raise ValueError("Each generated pair must be an object.")
        pair_id = str(row.get("pair_id") or f"pair_{index:03d}").strip()
        paper_open = str(row.get("paper_open_prompt_text", "")).strip()
        realized_closed = str(row.get("realized_closed_prompt_text", "")).strip()
        if not paper_open or not realized_closed:
            raise ValueError("Generated pairs must include both prompt texts.")
        parsed.append(
            {
                "pair_id": pair_id,
                "paper_open_prompt_text": paper_open,
                "realized_closed_prompt_text": realized_closed,
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

    if str(job.cell.get("asks_for_behavior", "false")).lower() == "true":
        if not _has_complete_behavior_instruction(prompt_text):
            raise ValueError(
                f"{job.batch_id} prompt {row['variant_id']} is missing the behavior "
                "answer instruction with two integers and expected ranges."
            )

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


def _behavior_instruction_for_job(job: GenerationJob) -> str:
    target = str(job.cell.get("behavior_target", ""))
    domain = str(job.cell.get("domain", ""))
    if "wager" in target or "casino" in domain:
        return CASINO_BEHAVIOR_INSTRUCTION
    return FINANCE_BEHAVIOR_INSTRUCTION


def _has_complete_behavior_instruction(prompt_text: str) -> bool:
    normalized_prompt = re.sub(r"\s+", " ", prompt_text.lower())
    has_two_integer_instruction = "two integers" in normalized_prompt
    has_amount_range = bool(re.search(r"\b1\s*(?:-|to)\s*1000\b", normalized_prompt))
    has_risk_range = bool(re.search(r"\b1\s*(?:-|to)\s*5\b", normalized_prompt))
    return has_two_integer_instruction and has_amount_range and has_risk_range


def _ensure_behavior_instruction(prompt_text: str, job: GenerationJob) -> str:
    if str(job.cell.get("asks_for_behavior", "false")).lower() != "true":
        return prompt_text
    if _has_complete_behavior_instruction(prompt_text):
        return prompt_text
    if not prompt_text.endswith(PROMPT_SUFFIX):
        return prompt_text
    instruction = _behavior_instruction_for_job(job)
    return prompt_text.removesuffix(PROMPT_SUFFIX).rstrip() + "\n\n" + instruction + PROMPT_SUFFIX


def _scenario_word_count(prompt_text: str) -> int:
    scenario = prompt_text.removeprefix(PROMPT_PREFIX).removesuffix(PROMPT_SUFFIX)
    return len(re.findall(r"\b\w+\b", scenario))


def _row_base(
    *,
    plan: dict[str, Any],
    job: GenerationJob,
    prompt_id: str,
    prompt_text: str,
    variant_id: str,
    notes: str,
    options: dict[str, Any],
    metadata: dict[str, str] | None = None,
) -> dict[str, str]:
    row_metadata = metadata if metadata is not None else job.metadata
    return {
        "prompt_id": prompt_id,
        "prompt_text": prompt_text,
        "prompt_generation_mode": str(plan.get("generation_mode", "single_prompt")),
        "prompt_framing": str(job.cell.get("prompt_framing", plan.get("prompt_framing", "scenario_continuation"))),
        "split": str(job.cell.get("split", "")),
        "pair_id": "",
        "pair_role": "",
        "prompt_family": str(job.cell.get("prompt_family", plan.get("name", "openrouter_generated"))),
        "source": "openrouter_generated",
        "source_llm": job.model_alias,
        "source_model": job.model_id,
        "generation_cell_id": str(job.cell["cell_id"]),
        "generation_batch_id": job.batch_id,
        "domain": row_metadata["domain"],
        "concept_axis": row_metadata["concept_axis"],
        "emotion": "" if row_metadata["emotion"] == "none" else row_metadata["emotion"],
        "emotion_intensity": row_metadata["emotion_intensity"],
        "risk_orientation": row_metadata["risk_orientation"],
        "risk_intensity": row_metadata["risk_intensity"],
        "casino_context": row_metadata["casino_context"],
        "realization_frame": row_metadata["realization_frame"],
        "outcome_valence": row_metadata["outcome_valence"],
        "amount_bucket": str(job.cell.get("amount_bucket", "")),
        "risk_context": str(job.cell.get("risk_context", "")),
        "emotion_context": str(job.cell.get("emotion_context", "")),
        "confound_axis": str(job.cell.get("confound_axis", "")),
        "behavior_target": row_metadata["behavior_target"],
        "asks_for_behavior": str(job.cell.get("asks_for_behavior", "false")),
        "expected_output_format": str(job.cell.get("expected_output_format", "")),
        "control_type": row_metadata["control_type"],
        "contrast_role": row_metadata["contrast_role"],
        "variant_id": variant_id,
        "model_temperature": str(options.get("temperature", "")),
        "model_seed": str(options.get("seed", "")),
        "expected_feature": row_metadata["expected_feature"],
        "expected_direction": str(job.cell.get("expected_direction", row_metadata["expected_feature"])),
        "notes": notes,
    }


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
    mode = str(plan.get("generation_mode", "single_prompt"))
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
    options = {**options, "response_schema": _response_schema_for_plan(plan)}
    validation_retries = int(options.get("validation_retries", 2))
    last_error: Exception | None = None

    for attempt in range(validation_retries + 1):
        response = _request_with_retries(request_fn, job.model_id, messages, options)
        try:
            if mode == "paired_contrast":
                generated_pairs = parse_generated_pairs(response)
                if len(generated_pairs) > job.count:
                    generated_pairs = generated_pairs[: job.count]
                if len(generated_pairs) != job.count:
                    raise ValueError(
                        f"{job.batch_id} returned {len(generated_pairs)} pairs, expected {job.count}."
                    )

                rows: list[dict[str, str]] = []
                pair_ids: list[str] = []
                for index, pair in enumerate(generated_pairs, start=1):
                    pair_id = _slugify(pair["pair_id"] or f"pair_{index:03d}")
                    pair_ids.append(pair_id)
                    paper_prompt = {
                        "variant_id": f"{pair_id}_paper_open",
                        "prompt_text": pair["paper_open_prompt_text"],
                    }
                    closed_prompt = {
                        "variant_id": f"{pair_id}_realized_closed",
                        "prompt_text": pair["realized_closed_prompt_text"],
                    }
                    paper_metadata = {
                        **job.metadata,
                        "realization_frame": "paper_open",
                        "contrast_role": "paper_open",
                    }
                    closed_metadata = {
                        **job.metadata,
                        "realization_frame": "realized_closed",
                        "contrast_role": "realized_closed",
                    }
                    paper_job = GenerationJob(
                        plan_name=job.plan_name,
                        model_alias=job.model_alias,
                        model_id=job.model_id,
                        cell=job.cell,
                        metadata=paper_metadata,
                        count=job.count,
                    )
                    closed_job = GenerationJob(
                        plan_name=job.plan_name,
                        model_alias=job.model_alias,
                        model_id=job.model_id,
                        cell=job.cell,
                        metadata=closed_metadata,
                        count=job.count,
                    )
                    paper_prompt["prompt_text"] = _ensure_behavior_instruction(
                        paper_prompt["prompt_text"],
                        paper_job,
                    )
                    closed_prompt["prompt_text"] = _ensure_behavior_instruction(
                        closed_prompt["prompt_text"],
                        closed_job,
                    )
                    _validate_generated_prompt_text(paper_prompt, paper_job)
                    _validate_generated_prompt_text(closed_prompt, closed_job)
                    paper_words = _scenario_word_count(paper_prompt["prompt_text"])
                    closed_words = _scenario_word_count(closed_prompt["prompt_text"])
                    if abs(paper_words - closed_words) > int(job.cell.get("max_pair_word_delta", 25)):
                        raise ValueError(
                            f"{job.batch_id} pair {pair_id} has mismatched lengths: "
                            f"{paper_words} vs {closed_words} words."
                        )

                    for pair_role, prompt, metadata in (
                        ("paper_open", paper_prompt, paper_metadata),
                        ("realized_closed", closed_prompt, closed_metadata),
                    ):
                        variant_id = _slugify(prompt["variant_id"])
                        prompt_id = f"{job.batch_id}__{pair_id}__{pair_role}"
                        row = _row_base(
                            plan=plan,
                            job=job,
                            prompt_id=prompt_id,
                            prompt_text=prompt["prompt_text"],
                            variant_id=variant_id,
                            notes=pair["notes"],
                            options=options,
                            metadata=metadata,
                        )
                        row["pair_id"] = f"{job.batch_id}__{pair_id}"
                        row["pair_role"] = pair_role
                        rows.append(row)
                duplicate_pair_ids = sorted(
                    pair_id for pair_id in set(pair_ids) if pair_ids.count(pair_id) > 1
                )
                if duplicate_pair_ids:
                    raise ValueError(
                        f"{job.batch_id} returned duplicate pair_id(s): "
                        + ", ".join(duplicate_pair_ids[:5])
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

            if mode != "single_prompt":
                raise ValueError(f"Unsupported generation_mode: {mode}")

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
                prompt["prompt_text"] = _ensure_behavior_instruction(prompt["prompt_text"], job)
                _validate_generated_prompt_text(prompt, job)
                prompt_id = f"{job.batch_id}__{variant_id}"
                rows.append(
                    _row_base(
                        plan=plan,
                        job=job,
                        prompt_id=prompt_id,
                        prompt_text=prompt["prompt_text"],
                        variant_id=variant_id,
                        notes=prompt["notes"],
                        options=options,
                    )
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
            retry_messages = list(messages)
            previous_content = _extract_content_or_empty(response)
            if previous_content:
                retry_messages.append({"role": "assistant", "content": previous_content})
            retry_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Regenerate the full batch. The previous batch failed validation: "
                        f"{exc}. Return exactly the same JSON schema, with exactly "
                        f"{job.count} prompts, and avoid the validation issue. If your previous "
                        "response had no normal content field, return parseable JSON in the "
                        "message content now."
                    ),
                }
            )
            messages = retry_messages

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


def merge_prompt_csvs(input_paths: Iterable[Path], output_path: Path) -> int:
    """Merge model-specific prompt CSVs into one CSV with duplicate protection."""

    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    duplicates: list[str] = []
    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Prompt CSV not found: {input_path}")
        with input_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                continue
            missing = [field for field in ("prompt_id", "prompt_text") if field not in reader.fieldnames]
            if missing:
                raise ValueError(f"{input_path} is missing required columns: {', '.join(missing)}")
            for row in reader:
                prompt_id = row.get("prompt_id", "")
                if not prompt_id:
                    continue
                if prompt_id in seen_ids:
                    duplicates.append(prompt_id)
                seen_ids.add(prompt_id)
                rows.append({field: row.get(field, "") for field in CSV_FIELDNAMES})
    if duplicates:
        sample = ", ".join(sorted(set(duplicates))[:5])
        raise ValueError(f"Cannot merge CSVs with duplicate prompt_id values: {sample}")
    return write_prompt_rows(output_path, rows, append=False)


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
    rows_per_unit = 2 if str(plan.get("generation_mode", "single_prompt")) == "paired_contrast" else 1

    for job in iter_generation_jobs(plan, model_aliases=model_aliases, limit_jobs=limit_jobs):
        parent_expected_ids = {
            prompt_id for prompt_id in existing_ids if prompt_id.startswith(f"{job.batch_id}__")
        }
        if resume and len(parent_expected_ids) >= job.count * rows_per_unit:
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
            parent_remaining = job.count - (len(parent_expected_ids) // rows_per_unit)
            if parent_remaining <= 0:
                break

            expected_ids = {
                prompt_id
                for prompt_id in existing_ids
                if prompt_id.startswith(f"{request_job.batch_id}__")
            }
            if resume and len(expected_ids) >= request_job.count * rows_per_unit:
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
