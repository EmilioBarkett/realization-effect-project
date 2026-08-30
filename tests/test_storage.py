from __future__ import annotations

from pathlib import Path

import pytest

from construct_benchmark.config import load_run_config
from construct_benchmark.storage import (
    archive_destination,
    build_storage_manifest,
    build_sync_command,
    prepare_run_directories,
    resolve_storage_layout,
    validate_archive_uri,
    verify_checksums,
    write_checksums,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_CONFIG_PATH = ROOT / "configs/construct_benchmark/run_configs/wave1_four_construct_smoke_v1.json"


def test_run_config_contains_credential_free_storage_policy() -> None:
    run_config = load_run_config(RUN_CONFIG_PATH)

    assert run_config.storage["sync_tool"] == "aws"
    assert run_config.storage["archive_uri_env"] == "RSC_BENCH_ARCHIVE_URI"
    assert run_config.storage["sync_on_finalize"] is True
    assert "AWS_SECRET_ACCESS_KEY" not in run_config.to_mapping()["storage"]


def test_layout_preparation_and_checksums_are_deterministic(tmp_path: Path) -> None:
    run_config = load_run_config(RUN_CONFIG_PATH)
    layout = resolve_storage_layout(run_config, tmp_path)
    prepare_run_directories(run_config, layout)
    artifact = layout.raw_root / "worker" / "result.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text('{"value": 1}\n', encoding="utf-8")

    checksum_path = write_checksums(layout.run_root)
    assert checksum_path.is_file()
    summary = verify_checksums(layout.run_root)
    assert summary["valid"] is True
    assert summary["checked_files"] == 1

    artifact.write_text('{"value": 2}\n', encoding="utf-8")
    assert verify_checksums(layout.run_root)["valid"] is False


@pytest.mark.parametrize("digest", ["", "a" * 63, "g" * 64])
def test_verify_checksums_rejects_malformed_sha256_digests(tmp_path: Path, digest: str) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    artifact = run_root / "artifact.txt"
    artifact.write_text("artifact\n", encoding="utf-8")
    (run_root / "checksums.sha256").write_text(f"{digest}  artifact.txt\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid SHA-256 digest"):
        verify_checksums(run_root)


def test_verify_checksums_rejects_malformed_checksum_lines(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "checksums.sha256").write_text("not a checksum line\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid checksum line"):
        verify_checksums(run_root)


def test_verify_checksums_rejects_duplicate_paths(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    artifact = run_root / "artifact.txt"
    artifact.write_text("artifact\n", encoding="utf-8")
    write_checksums(run_root)
    checksum_path = run_root / "checksums.sha256"
    checksum_line = checksum_path.read_text(encoding="utf-8")
    checksum_path.write_text(checksum_line + checksum_line, encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate checksum path"):
        verify_checksums(run_root)


@pytest.mark.parametrize("relative", ["/outside.txt", "../outside.txt"])
def test_verify_checksums_rejects_absolute_and_traversal_paths(tmp_path: Path, relative: str) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "checksums.sha256").write_text(f"{'a' * 64}  {relative}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="absolute|traversal"):
        verify_checksums(run_root)


def test_verify_checksums_rejects_missing_artifact_entries(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "artifact.txt").write_text("artifact\n", encoding="utf-8")
    write_checksums(run_root)
    (run_root / "artifact.txt").unlink()

    with pytest.raises(ValueError, match="missing or non-artifact files"):
        verify_checksums(run_root)


def test_verify_checksums_rejects_unlisted_artifact_files(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "artifact.txt").write_text("artifact\n", encoding="utf-8")
    write_checksums(run_root)
    (run_root / "extra.txt").write_text("extra\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unlisted artifact files"):
        verify_checksums(run_root)


def test_verify_checksums_accepts_exact_artifact_coverage(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "nested" / "first.txt").parent.mkdir()
    (run_root / "nested" / "first.txt").write_text("first\n", encoding="utf-8")
    (run_root / "second.txt").write_text("second\n", encoding="utf-8")
    write_checksums(run_root)

    summary = verify_checksums(run_root)

    assert summary == {
        "run_root": str(run_root.resolve()),
        "checked_files": 2,
        "failures": [],
        "valid": True,
    }


def test_storage_manifest_records_layout_without_credentials(tmp_path: Path) -> None:
    run_config = load_run_config(RUN_CONFIG_PATH)
    layout = resolve_storage_layout(run_config, tmp_path)
    manifest = build_storage_manifest(
        run_config,
        layout,
        status="prepared",
        archive_uri="s3://example-bucket/rsc-bench",
    )

    assert manifest["archive"]["destination"] == (
        "s3://example-bucket/rsc-bench/wave1_four_construct_smoke_v1"
    )
    assert "AWS_SECRET_ACCESS_KEY" not in str(manifest)
    assert manifest["workspace"]["run_root"] == str(layout.run_root)


def test_sync_command_is_explicit_and_does_not_embed_credentials(tmp_path: Path) -> None:
    run_config = load_run_config(RUN_CONFIG_PATH)
    layout = resolve_storage_layout(run_config, tmp_path)
    command = build_sync_command(
        run_config,
        layout,
        "s3://example-bucket/rsc-bench",
        dry_run=True,
        endpoint_url="https://s3.example.test",
    )

    assert command[:5] == [
        "aws",
        "s3",
        "sync",
        str(layout.run_root),
        "s3://example-bucket/rsc-bench/wave1_four_construct_smoke_v1",
    ]
    assert "--dryrun" in command
    assert "AWS_SECRET_ACCESS_KEY" not in " ".join(command)


def test_archive_uri_rejects_embedded_credentials() -> None:
    with pytest.raises(ValueError, match="credential-free"):
        validate_archive_uri("s3://user:secret@example-bucket/rsc-bench")
    assert archive_destination("s3://example-bucket/rsc-bench/", "run_001") == (
        "s3://example-bucket/rsc-bench/run_001"
    )
