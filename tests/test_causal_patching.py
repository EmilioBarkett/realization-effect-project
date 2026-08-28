from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from activation_analysis.causal_patching import (
    MatchedEpisode,
    compose_matched_episode,
    expected_observation_ids,
    validate_residual_interchange_output,
)


def test_matched_episode_requires_one_shared_downstream_task() -> None:
    episode = MatchedEpisode(
        request_id="episode_001",
        construct_id="realization_account_closure",
        positive_induction_prompt="The account is closed and the outcome is final.",
        negative_induction_prompt="The account remains open and the outcome can change.",
        downstream_prompt="Choose one action and return only its name.",
    )

    assert compose_matched_episode(
        episode.positive_induction_prompt,
        episode.downstream_prompt,
    ).endswith(episode.downstream_prompt)
    assert episode.to_mapping()["positive_source_prompt_id"] == "episode_001__positive_source"
    assert episode.to_mapping()["negative_source_prompt_id"] == "episode_001__negative_source"
    assert episode.to_mapping()["downstream_prompt_id"] == "episode_001__downstream"

    with pytest.raises(ValueError, match="must differ"):
        MatchedEpisode(
            request_id="bad",
            positive_induction_prompt="positive",
            negative_induction_prompt="negative",
            downstream_prompt="task",
            positive_condition="same",
            negative_condition="same",
        )


def test_expected_observation_ids_include_bidirectional_and_same_condition_controls() -> None:
    assert expected_observation_ids(["episode_001"], [3, 1], include_same_condition_controls=True) == [
        "episode_001__positive_to_negative__layer_01",
        "episode_001__positive_to_negative__layer_03",
        "episode_001__negative_to_positive__layer_01",
        "episode_001__negative_to_positive__layer_03",
        "episode_001__positive_to_positive__layer_01",
        "episode_001__positive_to_positive__layer_03",
        "episode_001__negative_to_negative__layer_01",
        "episode_001__negative_to_negative__layer_03",
    ]


def test_validator_refuses_truncated_output_unless_diagnostic_override(tmp_path: Path) -> None:
    output = tmp_path / "interchange.jsonl"
    record_id = "episode_001__positive_to_negative__layer_01"
    row = {
        "request": {"request_id": "episode_001"},
        "observations": [
            {"record_id": record_id, "request_id": "episode_001"},
        ],
    }
    output.write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "0.1.0",
        "manifest_type": "residual_interchange_output",
        "request_ids": ["episode_001"],
        "layers": [1],
        "expected_record_ids": [
            "episode_001__positive_to_negative__layer_01",
            "episode_001__negative_to_positive__layer_01",
        ],
        "expected_request_count": 1,
        "expected_observation_count": 2,
        "execution": {"include_same_condition_controls": False},
        "completed_request_ids": ["episode_001"],
        "completed_request_count": 1,
        "completed_observation_count": 1,
        "complete": False,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete"):
        validate_residual_interchange_output(output)

    validated = validate_residual_interchange_output(output, allow_incomplete_diagnostic=True)
    assert validated["complete"] is False


def test_fake_forward_patches_boundary_and_cleans_hooks() -> None:
    torch = pytest.importorskip("torch")
    from activation_analysis.causal_patching import MatchedEpisodeResidualPatcher

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def __init__(self) -> None:
            self.vocabulary: dict[str, int] = {}

        def _tokens(self, prompt: str) -> tuple[list[int], list[tuple[int, int]]]:
            tokens: list[int] = []
            offsets: list[tuple[int, int]] = []
            for match in re.finditer(r"\S+", prompt):
                token = match.group(0)
                token_id = self.vocabulary.setdefault(token, 10 + len(self.vocabulary))
                tokens.append(token_id)
                offsets.append((match.start(), match.end()))
            return tokens, offsets

        def __call__(
            self,
            prompts,
            *,
            padding=False,
            truncation=False,
            max_length=None,
            return_tensors=None,
            return_offsets_mapping=False,
        ):
            del truncation, max_length
            if isinstance(prompts, str):
                prompts = [prompts]
            token_rows = [self._tokens(prompt) for prompt in prompts]
            max_tokens = max(len(tokens) for tokens, _offsets in token_rows)
            input_ids = []
            attention_mask = []
            offset_mapping = []
            for tokens, offsets in token_rows:
                pad_count = max_tokens - len(tokens) if padding else 0
                input_ids.append(tokens + [self.pad_token_id] * pad_count)
                attention_mask.append([1] * len(tokens) + [0] * pad_count)
                offset_mapping.append(offsets + [(0, 0)] * pad_count)
            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if return_offsets_mapping:
                result["offset_mapping"] = offset_mapping
            if return_tensors == "pt":
                result = {
                    key: torch.tensor(value, dtype=torch.long)
                    for key, value in result.items()
                }
            return result

        def decode(self, token_ids, *, skip_special_tokens=True):
            del skip_special_tokens
            if hasattr(token_ids, "detach"):
                token_ids = token_ids.detach().cpu().tolist()
            return " ".join(f"generated_{int(token_id)}" for token_id in token_ids if int(token_id) != 0)

    class FakeBlock(torch.nn.Module):
        def __init__(self, offset: float) -> None:
            super().__init__()
            self.offset = offset

        def forward(self, hidden, **_kwargs):
            return hidden + self.offset + torch.cumsum(hidden, dim=1) * 0.01

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([FakeBlock(1.0), FakeBlock(2.0)])

        def forward(self, input_ids, attention_mask=None, **_kwargs):
            del attention_mask
            hidden = input_ids.float().unsqueeze(-1)
            for block in self.layers:
                hidden = block(hidden)
            return SimpleNamespace(last_hidden_state=hidden)

        def generate(self, input_ids, attention_mask=None, **_kwargs):
            hidden = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            generated_id = 100 + (int(round(float(hidden.last_hidden_state[0, -1, 0].item()) * 10)) % 10000)
            generated = torch.tensor([[generated_id]], dtype=input_ids.dtype)
            return torch.cat([input_ids, generated], dim=1)

    episode = MatchedEpisode(
        request_id="episode_001",
        construct_id="realization_account_closure",
        positive_induction_prompt="positive state extra",
        negative_induction_prompt="negative state",
        downstream_prompt="Choose action.",
    )
    tokenizer = FakeTokenizer()
    model = FakeModel()
    patcher = MatchedEpisodeResidualPatcher(
        model,
        tokenizer,
        device="cpu",
        block_path="layers",
        model_id="fake/model",
        tokenizer_id="fake/tokenizer",
    )

    result = patcher.run_episode(
        episode,
        layers=[1, 2],
        max_length=32,
        max_new_tokens=1,
        min_new_tokens=1,
        include_same_condition_controls=True,
    )

    observations = result["observations"]
    assert len(observations) == 8
    assert all(observation["patch_applied"] for observation in observations)
    assert all(observation["forward_calls"] == 1 for observation in observations)
    assert any(
        observation["patched_output"] != observation["receiver_baseline_output"]
        for observation in observations
        if observation["patch_direction"] in {"positive_to_negative", "negative_to_positive"}
    )
    assert all(
        observation["patched_output"] == observation["receiver_baseline_output"]
        for observation in observations
        if observation["patch_direction"] in {"positive_to_positive", "negative_to_negative"}
    )
    assert result["prompt_fingerprints"]["downstream_prompt_identical"] is True
    assert all(not block._forward_hooks for block in model.layers)
