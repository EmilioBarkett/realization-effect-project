import json
from pathlib import Path

from scripts.preflight_benchmark_run import build_preflight_report


ROOT = Path(__file__).resolve().parents[1]
RUN_CONFIG = ROOT / "configs/construct_benchmark/run_configs/wave1_realization_single_construct_smoke_v1.json"
CONSTRUCT_SPEC = ROOT / "configs/construct_benchmark/constructs/realization_account_closure_v1.json"
ANALYSIS_SPEC = ROOT / "configs/construct_benchmark/analysis_specs/rsc_benchmark_core_v1.json"


def _statuses(report: dict[str, object]) -> dict[str, str]:
    return {str(item["name"]): str(item["status"]) for item in report["checks"]}  # type: ignore[index]


def test_preflight_validates_cross_file_configuration_without_network_calls() -> None:
    report = build_preflight_report(
        run_config_path=RUN_CONFIG,
        construct_spec_paths=[CONSTRUCT_SPEC],
        analysis_spec_path=ANALYSIS_SPEC,
        env={},
    )

    statuses = _statuses(report)
    assert report["ready"] is True
    assert statuses["configuration_cross_validation"] == "pass"
    assert statuses["openrouter_api_key"] == "warn"
    assert statuses["model_configuration"] == "pass"


def test_preflight_rejects_placeholder_model_when_required(tmp_path: Path) -> None:
    payload = json.loads(RUN_CONFIG.read_text(encoding="utf-8"))
    payload["model"]["model_id"] = "REPLACE_WITH_LOCAL_MODEL"
    payload["model"]["tokenizer_id"] = "REPLACE_WITH_LOCAL_MODEL"
    placeholder_config = tmp_path / "placeholder_run_config.json"
    placeholder_config.write_text(json.dumps(payload), encoding="utf-8")

    report = build_preflight_report(
        run_config_path=placeholder_config,
        env={"OPENROUTER_API_KEY": "test-secret"},
        require_model=True,
        require_openrouter=True,
    )

    statuses = _statuses(report)
    assert report["ready"] is False
    assert statuses["model_configuration"] == "fail"
    assert statuses["openrouter_api_key"] == "pass"
    serialized = str(report)
    assert "test-secret" not in serialized
