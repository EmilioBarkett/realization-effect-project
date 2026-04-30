#!/usr/bin/env python3
"""Live dashboard for monitoring block progress and running analysis on demand.

Usage:
  python scripts/block_dashboard.py
  python scripts/block_dashboard.py --target-trials 25 --refresh-seconds 5
"""

import argparse
import csv
import html
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from urllib.parse import parse_qs, quote_plus, urlparse


def _parse_run_number(value: Any) -> int:
    try:
        number = int(str(value).strip())
    except ValueError:
        return 0
    return number if number > 0 else 0


def _load_expected_conditions(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "condition" not in reader.fieldnames:
            return []
        return [str(row["condition"]).strip() for row in reader if str(row["condition"]).strip()]


def _coerce_int(value: str, default: int) -> int:
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed


def _coerce_float(value: str, default: float) -> float:
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed


def collect_block_stats(
    blocks_dir: Path,
    expected_conditions: List[str],
    target_trials: int,
    active_window_seconds: float,
) -> List[Dict[str, Any]]:
    expected_set = set(expected_conditions)
    stats: List[Dict[str, Any]] = []
    now = time.time()

    if not blocks_dir.exists():
        return stats

    for block_path in sorted(blocks_dir.glob("block__*.csv"), key=lambda p: p.name):
        if not block_path.is_file():
            continue

        rows = 0
        model = ""
        temperature = ""
        prompt_version = ""
        run_numbers_by_condition: Dict[str, Set[int]] = defaultdict(set)
        read_error = ""

        try:
            with block_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    rows += 1
                    if not model:
                        model = (row.get("model") or "").strip()
                        temperature = str(row.get("temperature") or "").strip()
                        prompt_version = (row.get("prompt_version") or "").strip()

                    condition = (row.get("condition") or "").strip()
                    run_number = _parse_run_number(row.get("run_number"))
                    if condition and run_number > 0:
                        run_numbers_by_condition[condition].add(run_number)
        except Exception as error:
            read_error = str(error)

        if not model:
            model = "(unknown)"
        if not temperature:
            temperature = "(unknown)"
        if not prompt_version:
            prompt_version = "(unknown)"

        st = block_path.stat()
        mtime = st.st_mtime
        age_seconds = max(0.0, now - mtime)
        active_recently = age_seconds <= active_window_seconds

        condition_names = sorted(run_numbers_by_condition.keys())
        counts = [len(run_numbers_by_condition[name]) for name in condition_names]
        per_condition_min = min(counts) if counts else 0
        per_condition_max = max(counts) if counts else 0
        covered_conditions = len(condition_names)
        total_unique_runs = sum(counts)

        if expected_conditions:
            missing_conditions = sorted(expected_set - set(condition_names))
            remaining_to_target = sum(
                max(0, target_trials - len(run_numbers_by_condition.get(condition, set())))
                for condition in expected_conditions
            )
            completed_target_runs = sum(
                min(target_trials, len(run_numbers_by_condition.get(condition, set())))
                for condition in expected_conditions
            )
            target_total_runs = len(expected_conditions) * target_trials
            progress_pct = (
                (completed_target_runs / target_total_runs) * 100.0 if target_total_runs > 0 else 0.0
            )
            schema_ok = not missing_conditions and covered_conditions == len(expected_conditions)
        else:
            missing_conditions = []
            remaining_to_target = 0
            target_total_runs = 0
            progress_pct = 0.0
            schema_ok = True

        deficits: List[Tuple[str, int]] = []
        if expected_conditions:
            for condition in expected_conditions:
                have = len(run_numbers_by_condition.get(condition, set()))
                deficit = max(0, target_trials - have)
                deficits.append((condition, deficit))
            deficits.sort(key=lambda item: (-item[1], item[0]))
        top_deficits = [item for item in deficits if item[1] > 0][:3]

        stats.append(
            {
                "file_name": block_path.name,
                "file_path": str(block_path),
                "model": model,
                "temperature": temperature,
                "prompt_version": prompt_version,
                "rows": rows,
                "total_unique_runs": total_unique_runs,
                "covered_conditions": covered_conditions,
                "per_condition_min": per_condition_min,
                "per_condition_max": per_condition_max,
                "remaining_to_target": remaining_to_target,
                "target_total_runs": target_total_runs,
                "progress_pct": progress_pct,
                "schema_ok": schema_ok,
                "missing_conditions": missing_conditions,
                "top_deficits": top_deficits,
                "mtime_iso": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "age_seconds": age_seconds,
                "active_recently": active_recently,
                "read_error": read_error,
            }
        )

    return stats


def _css() -> str:
    return """
    :root {
      --bg: #f4f7f8;
      --ink: #1f2a2e;
      --muted: #5f6f75;
      --card: #ffffff;
      --line: #d9e1e5;
      --ok: #2e7d32;
      --warn: #ef6c00;
      --bad: #c62828;
      --accent: #0f766e;
      --chip: #e7f2f1;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 10% 20%, #e9f4f3 0%, transparent 40%),
        radial-gradient(circle at 90% 0%, #fff0dc 0%, transparent 35%),
        var(--bg);
    }
    a { color: var(--accent); text-decoration: none; }
    a:hover { text-decoration: underline; }
    .wrap { max-width: 1400px; margin: 24px auto; padding: 0 16px 40px; }
    .header {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }
    h1 { margin: 0; font-size: 1.5rem; letter-spacing: 0.02em; }
    h2 { margin: 14px 0 8px; font-size: 1.15rem; }
    h3 { margin: 16px 0 8px; font-size: 1rem; }
    .sub { color: var(--muted); font-size: 0.95rem; }
    .tabs { display: flex; gap: 8px; margin: 8px 0 12px; }
    .tab {
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 999px;
      padding: 6px 12px;
      font-size: 0.9rem;
      color: var(--ink);
      text-decoration: none;
    }
    .tab.active {
      background: #dbefed;
      border-color: #b8d9d5;
      font-weight: 600;
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(5, minmax(150px, 1fr));
      gap: 10px;
      margin: 14px 0 16px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      box-shadow: 0 1px 0 rgba(0,0,0,0.02);
    }
    .card .k { font-size: 1.4rem; font-weight: 700; }
    .card .l { color: var(--muted); font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em; }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin: 8px 0 14px;
    }
    input, select, button {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      background: #fff;
      min-width: 180px;
      font-family: inherit;
      font-size: 0.9rem;
    }
    button {
      cursor: pointer;
      background: #dbefed;
      border-color: #b8d9d5;
      min-width: 140px;
      font-weight: 600;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
    }
    th, td {
      text-align: left;
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      font-size: 0.9rem;
      vertical-align: top;
    }
    th {
      background: #f9fbfc;
      position: sticky;
      top: 0;
      cursor: pointer;
      user-select: none;
      white-space: nowrap;
    }
    tr:last-child td { border-bottom: none; }
    .chip {
      display: inline-block;
      padding: 2px 7px;
      border-radius: 999px;
      background: var(--chip);
      border: 1px solid #cfe5e2;
      font-size: 0.78rem;
      margin-right: 4px;
      margin-bottom: 4px;
      white-space: nowrap;
    }
    .ok { color: var(--ok); font-weight: 600; }
    .warn { color: var(--warn); font-weight: 600; }
    .bad { color: var(--bad); font-weight: 600; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.82rem; }
    .foot { margin-top: 10px; color: var(--muted); font-size: 0.84rem; }
    .panel {
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      margin-bottom: 12px;
    }
    .output {
      white-space: pre-wrap;
      background: #f8fbfb;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      font-size: 0.84rem;
      max-height: 68vh;
      overflow: auto;
    }
    @media (max-width: 1050px) {
      .cards { grid-template-columns: repeat(2, minmax(150px, 1fr)); }
      th, td { font-size: 0.82rem; padding: 8px; }
      input, select, button { min-width: 140px; width: 100%; }
    }
    """


def _layout(title: str, body_html: str, active_tab: str) -> str:
    dashboard_class = "tab active" if active_tab == "dashboard" else "tab"
    analyze_class = "tab active" if active_tab == "analyze" else "tab"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>{_css()}</style>
</head>
<body>
  <div class="wrap">
    <div class="tabs">
      <a class="{dashboard_class}" href="/">Dashboard</a>
      <a class="{analyze_class}" href="/analyze">Analysis</a>
    </div>
    {body_html}
  </div>
</body>
</html>"""


def render_dashboard_html(
    *,
    stats: List[Dict[str, Any]],
    target_trials: int,
    refresh_seconds: int,
    active_window_seconds: float,
    expected_conditions: List[str],
) -> str:
    total_blocks = len(stats)
    active_blocks = sum(1 for row in stats if row["active_recently"])
    completed_blocks = sum(
        1
        for row in stats
        if row["remaining_to_target"] == 0 and row["schema_ok"] and not row["read_error"]
    )
    errored_blocks = sum(1 for row in stats if row["read_error"])
    incomplete_blocks = total_blocks - completed_blocks

    model_summary: Dict[str, Dict[str, int]] = defaultdict(lambda: {"blocks": 0, "remaining": 0})
    for row in stats:
        model_summary[row["model"]]["blocks"] += 1
        model_summary[row["model"]]["remaining"] += int(row["remaining_to_target"])

    model_summary_rows = sorted(
        (
            {
                "model": model,
                "blocks": values["blocks"],
                "remaining": values["remaining"],
            }
            for model, values in model_summary.items()
        ),
        key=lambda item: (item["remaining"], item["model"]),
    )

    blocks_json = json.dumps(stats)
    model_json = json.dumps(model_summary_rows)

    body = f"""
    <div class="header">
      <div>
        <h1>Model Block Dashboard</h1>
        <div class="sub">
          Target = {target_trials} trials/condition | refresh every {refresh_seconds}s |
          active window = {active_window_seconds:.0f}s | expected conditions = {len(expected_conditions)}
        </div>
      </div>
      <div class="sub">Loaded at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
    </div>

    <div class="cards">
      <div class="card"><div class="k">{total_blocks}</div><div class="l">Total Blocks</div></div>
      <div class="card"><div class="k">{active_blocks}</div><div class="l">Active Recently</div></div>
      <div class="card"><div class="k">{completed_blocks}</div><div class="l">Complete @ Target</div></div>
      <div class="card"><div class="k">{incomplete_blocks}</div><div class="l">Incomplete</div></div>
      <div class="card"><div class="k">{errored_blocks}</div><div class="l">Read Errors</div></div>
    </div>

    <div class="controls">
      <input id="searchInput" placeholder="Filter by model/prompt/temp/file..." />
      <select id="statusFilter">
        <option value="all">All statuses</option>
        <option value="active">Active recently</option>
        <option value="complete">Complete at target</option>
        <option value="incomplete">Incomplete</option>
        <option value="error">Read errors</option>
      </select>
    </div>

    <h3>By Model</h3>
    <table id="modelTable">
      <thead>
        <tr>
          <th data-key="model">Model</th>
          <th data-key="blocks">Blocks</th>
          <th data-key="remaining">Total Remaining</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>

    <h3>By Block</h3>
    <table id="blockTable">
      <thead>
        <tr>
          <th data-key="model">Model</th>
          <th data-key="temperature">Temp</th>
          <th data-key="prompt_version">Prompt</th>
          <th data-key="rows">Rows</th>
          <th data-key="covered_conditions">Conds</th>
          <th data-key="per_condition_min">Min/Cond</th>
          <th data-key="per_condition_max">Max/Cond</th>
          <th data-key="remaining_to_target">Remaining</th>
          <th data-key="progress_pct">Progress</th>
          <th data-key="active_recently">Active</th>
          <th data-key="mtime_iso">Updated</th>
          <th data-key="file_name">File</th>
          <th>Top Deficits</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
    <div class="foot">Tip: click headers to sort. This page auto-refreshes every {refresh_seconds} seconds.</div>

    <script>
      const blocks = {blocks_json};
      const modelRows = {model_json};
      const refreshMs = {refresh_seconds} * 1000;

      const searchInput = document.getElementById("searchInput");
      const statusFilter = document.getElementById("statusFilter");
      const blockBody = document.querySelector("#blockTable tbody");
      const modelBody = document.querySelector("#modelTable tbody");

      let blockSortKey = "remaining_to_target";
      let blockSortAsc = true;
      let modelSortKey = "remaining";
      let modelSortAsc = true;

      function compare(a, b, key, asc) {{
        const av = a[key];
        const bv = b[key];
        let result = 0;
        if (typeof av === "number" && typeof bv === "number") result = av - bv;
        else result = String(av).localeCompare(String(bv));
        return asc ? result : -result;
      }}

      function rowMatchesSearch(row, searchText) {{
        if (!searchText) return true;
        const hay = [row.model, row.temperature, row.prompt_version, row.file_name, row.mtime_iso].join(" ").toLowerCase();
        return hay.includes(searchText);
      }}

      function rowMatchesStatus(row, filterValue) {{
        if (filterValue === "all") return true;
        if (filterValue === "active") return !!row.active_recently;
        if (filterValue === "complete") return row.remaining_to_target === 0 && row.schema_ok && !row.read_error;
        if (filterValue === "incomplete") return row.remaining_to_target > 0 && !row.read_error;
        if (filterValue === "error") return !!row.read_error;
        return true;
      }}

      function deficitHtml(row) {{
        if (!row.top_deficits || row.top_deficits.length === 0) return '<span class="ok">none</span>';
        return row.top_deficits.map(([condition, deficit]) => `<span class="chip">${{condition}}: -${{deficit}}</span>`).join("");
      }}

      function statusHtml(row) {{
        if (row.read_error) return '<span class="bad">error</span>';
        if (row.active_recently) return '<span class="warn">active</span>';
        return '<span class="ok">idle</span>';
      }}

      function progressHtml(row) {{
        const pct = Number(row.progress_pct || 0);
        const cls = pct >= 100 ? "ok" : (pct >= 50 ? "warn" : "bad");
        return `<span class="${{cls}}">${{pct.toFixed(1)}}%</span>`;
      }}

      function renderModels() {{
        const sorted = [...modelRows].sort((a, b) => compare(a, b, modelSortKey, modelSortAsc));
        modelBody.innerHTML = sorted.map((row) => `
          <tr>
            <td>${{row.model}}</td>
            <td>${{row.blocks}}</td>
            <td>${{row.remaining}}</td>
          </tr>
        `).join("");
      }}

      function renderBlocks() {{
        const searchText = searchInput.value.trim().toLowerCase();
        const filterValue = statusFilter.value;
        const filtered = blocks.filter((row) => rowMatchesSearch(row, searchText)).filter((row) => rowMatchesStatus(row, filterValue));
        const sorted = filtered.sort((a, b) => compare(a, b, blockSortKey, blockSortAsc));
        blockBody.innerHTML = sorted.map((row) => `
          <tr>
            <td>${{row.model}}</td>
            <td>${{row.temperature}}</td>
            <td>${{row.prompt_version}}</td>
            <td>${{row.rows}}</td>
            <td>${{row.covered_conditions}}</td>
            <td>${{row.per_condition_min}}</td>
            <td>${{row.per_condition_max}}</td>
            <td>${{row.remaining_to_target}}</td>
            <td>${{progressHtml(row)}}</td>
            <td>${{statusHtml(row)}}</td>
            <td>${{row.mtime_iso}}</td>
            <td class="mono">${{row.file_name}}</td>
            <td>${{deficitHtml(row)}}</td>
          </tr>
        `).join("");
      }}

      function wireSort(tableId, onSort) {{
        document.querySelectorAll(`#${{tableId}} th[data-key]`).forEach((th) => {{
          th.addEventListener("click", () => onSort(th.dataset.key));
        }});
      }}

      wireSort("blockTable", (key) => {{
        if (blockSortKey === key) blockSortAsc = !blockSortAsc;
        else {{ blockSortKey = key; blockSortAsc = true; }}
        renderBlocks();
      }});
      wireSort("modelTable", (key) => {{
        if (modelSortKey === key) modelSortAsc = !modelSortAsc;
        else {{ modelSortKey = key; modelSortAsc = true; }}
        renderModels();
      }});

      searchInput.addEventListener("input", renderBlocks);
      statusFilter.addEventListener("change", renderBlocks);
      renderModels();
      renderBlocks();
      setTimeout(() => window.location.reload(), refreshMs);
    </script>
    """
    return _layout("Block Dashboard", body, "dashboard")


def list_analysis_datasets(project_root: Path) -> List[Path]:
    datasets: List[Path] = []
    results_dir = project_root / "results"
    if results_dir.exists():
        datasets.extend(path for path in results_dir.rglob("*.csv") if path.is_file())
    # Keep list deterministic and unique.
    deduped = sorted(set(path.resolve() for path in datasets))
    return deduped


def _relative_label(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _sanitize_dataset_path(project_root: Path, raw_value: str, allowed: List[Path]) -> Path:
    if not raw_value:
        raise ValueError("Dataset path is required.")
    path = Path(raw_value)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    else:
        path = path.resolve()
    allowed_set = {item.resolve() for item in allowed}
    if path not in allowed_set:
        raise ValueError("Selected dataset is not in the allowed results CSV list.")
    return path


def _run_analysis(
    *,
    python_executable: str,
    analyze_script: Path,
    dataset_path: Path,
    model: str,
    prompt_version: str,
    per_model: bool,
    cov_type: str,
    timeout_seconds: int,
    output_limit_chars: int,
) -> Dict[str, Any]:
    command = [python_executable, str(analyze_script), str(dataset_path)]
    if model:
        command.extend(["--model", model])
    if prompt_version:
        command.extend(["--prompt-version", prompt_version])
    if per_model:
        command.append("--per-model")
    command.extend(["--cov-type", cov_type])

    started_at = datetime.now(timezone.utc)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=str(analyze_script.parent),
            timeout=timeout_seconds,
            check=False,
        )
        combined_output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
        truncated = False
        if len(combined_output) > output_limit_chars:
            combined_output = (
                combined_output[:output_limit_chars]
                + "\n\n[Output truncated for display. Re-run from terminal for full output.]"
            )
            truncated = True

        ended_at = datetime.now(timezone.utc)
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "command": command,
            "dataset": str(dataset_path),
            "model": model,
            "prompt_version": prompt_version,
            "per_model": per_model,
            "cov_type": cov_type,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_seconds": round((ended_at - started_at).total_seconds(), 2),
            "output": combined_output.strip(),
            "truncated": truncated,
        }
    except subprocess.TimeoutExpired as error:
        ended_at = datetime.now(timezone.utc)
        partial = (error.stdout or "") + ("\n" + error.stderr if error.stderr else "")
        if len(partial) > output_limit_chars:
            partial = partial[:output_limit_chars] + "\n\n[Partial output truncated.]"
        return {
            "ok": False,
            "returncode": None,
            "command": command,
            "dataset": str(dataset_path),
            "model": model,
            "prompt_version": prompt_version,
            "per_model": per_model,
            "cov_type": cov_type,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_seconds": round((ended_at - started_at).total_seconds(), 2),
            "output": (partial.strip() + "\n\n[Timed out waiting for analysis to finish.]").strip(),
            "truncated": True,
        }


def render_analysis_html(
    *,
    project_root: Path,
    datasets: List[Path],
    selected_dataset: str,
    selected_model: str,
    selected_prompt_version: str,
    selected_per_model: bool,
    selected_cov_type: str,
    message: str,
    message_ok: bool,
    analysis_result: Dict[str, Any],
) -> str:
    dataset_options_html = []
    for path in datasets:
        rel = _relative_label(path, project_root)
        selected_attr = " selected" if rel == selected_dataset else ""
        dataset_options_html.append(
            f'<option value="{html.escape(rel)}"{selected_attr}>{html.escape(rel)}</option>'
        )
    options_text = "\n".join(dataset_options_html)

    message_html = ""
    if message:
        cls = "ok" if message_ok else "bad"
        message_html = (
            f'<div class="panel"><span class="{cls}">{html.escape(message)}</span></div>'
        )

    result_html = '<div class="panel sub">No analysis run yet in this dashboard session.</div>'
    if analysis_result:
        status_text = "Success" if analysis_result.get("ok") else "Failed"
        status_class = "ok" if analysis_result.get("ok") else "bad"
        command_text = " ".join(
            html.escape(str(part)) for part in analysis_result.get("command", [])
        )
        output_text = html.escape(analysis_result.get("output", "") or "(no output)")
        result_html = f"""
        <div class="panel">
          <div><span class="{status_class}">{status_text}</span></div>
          <div class="sub">Dataset: {html.escape(analysis_result.get("dataset", ""))}</div>
          <div class="sub">Duration: {analysis_result.get("duration_seconds", 0)}s</div>
          <div class="sub mono">Command: {command_text}</div>
        </div>
        <div class="panel">
          <h3>Output</h3>
          <div class="output">{output_text}</div>
        </div>
        """

    prompt_absolute_selected = " selected" if selected_prompt_version == "absolute" else ""
    prompt_balance_selected = " selected" if selected_prompt_version == "balance" else ""
    prompt_qualitative_selected = " selected" if selected_prompt_version == "qualitative" else ""
    prompt_blank_selected = " selected" if selected_prompt_version == "" else ""
    cov_hc0 = " selected" if selected_cov_type == "HC0" else ""
    cov_hc1 = " selected" if selected_cov_type == "HC1" else ""
    cov_hc2 = " selected" if selected_cov_type == "HC2" else ""
    cov_hc3 = " selected" if selected_cov_type == "HC3" else ""
    per_model_checked = " checked" if selected_per_model else ""

    body = f"""
    <div class="header">
      <div>
        <h1>Statistical Analysis</h1>
        <div class="sub">Run <span class="mono">scripts/analyze_realization_results.py</span> on demand and view output here.</div>
      </div>
      <div class="sub">{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
    </div>

    {message_html}

    <form class="panel" method="post" action="/analyze-run">
      <div class="controls">
        <select name="dataset">
          {options_text}
        </select>
        <input name="model" placeholder="Optional model filter (e.g. openai/gpt-4o)" value="{html.escape(selected_model)}" />
        <select name="prompt_version">
          <option value=""{prompt_blank_selected}>All prompt versions</option>
          <option value="absolute"{prompt_absolute_selected}>absolute</option>
          <option value="balance"{prompt_balance_selected}>balance</option>
          <option value="qualitative"{prompt_qualitative_selected}>qualitative</option>
        </select>
        <select name="cov_type">
          <option value="HC0"{cov_hc0}>HC0</option>
          <option value="HC1"{cov_hc1}>HC1</option>
          <option value="HC2"{cov_hc2}>HC2</option>
          <option value="HC3"{cov_hc3}>HC3</option>
        </select>
        <label style="display:flex;align-items:center;gap:8px;min-width:180px;">
          <input type="checkbox" name="per_model" value="1"{per_model_checked} style="min-width:auto;" />
          Per-model analysis
        </label>
        <button type="submit">Run Analysis</button>
      </div>
    </form>

    {result_html}
    """
    return _layout("Analysis Dashboard", body, "analyze")


def run_server(
    *,
    host: str,
    port: int,
    blocks_dir: Path,
    conditions_path: Path,
    default_target_trials: int,
    default_refresh_seconds: int,
    default_active_window_seconds: float,
    project_root: Path,
    analyze_script: Path,
    analysis_python_executable: str,
    analysis_timeout_seconds: int,
    analysis_output_limit_chars: int,
) -> None:
    expected_conditions = _load_expected_conditions(conditions_path)
    analysis_state: Dict[str, Any] = {}

    class DashboardHandler(BaseHTTPRequestHandler):
        def _send_html(self, body: str) -> None:
            payload = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _redirect(self, path: str) -> None:
            self.send_response(303)
            self.send_header("Location", path)
            self.end_headers()

        def _parse_params(self) -> Dict[str, List[str]]:
            parsed = urlparse(self.path)
            return parse_qs(parsed.query)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)

            if parsed.path in {"/", "/index.html"}:
                target_trials = max(
                    1,
                    _coerce_int(params.get("target", [str(default_target_trials)])[0], default_target_trials),
                )
                refresh_seconds = max(
                    1,
                    _coerce_int(params.get("refresh", [str(default_refresh_seconds)])[0], default_refresh_seconds),
                )
                active_window_seconds = max(
                    1.0,
                    _coerce_float(
                        params.get("active_window", [str(default_active_window_seconds)])[0],
                        default_active_window_seconds,
                    ),
                )

                stats = collect_block_stats(
                    blocks_dir=blocks_dir,
                    expected_conditions=expected_conditions,
                    target_trials=target_trials,
                    active_window_seconds=active_window_seconds,
                )
                html_text = render_dashboard_html(
                    stats=stats,
                    target_trials=target_trials,
                    refresh_seconds=refresh_seconds,
                    active_window_seconds=active_window_seconds,
                    expected_conditions=expected_conditions,
                )
                self._send_html(html_text)
                return

            if parsed.path == "/analyze":
                datasets = list_analysis_datasets(project_root)
                selected_dataset = params.get("dataset", [""])[0].strip()
                if not selected_dataset and datasets:
                    preferred = (project_root / "results" / "results.csv").resolve()
                    if preferred in datasets:
                        selected_dataset = _relative_label(preferred, project_root)
                    else:
                        selected_dataset = _relative_label(datasets[0], project_root)
                selected_model = params.get("model", [""])[0].strip()
                selected_prompt_version = params.get("prompt_version", [""])[0].strip()
                selected_cov_type = params.get("cov_type", ["HC3"])[0].strip() or "HC3"
                selected_per_model = params.get("per_model", ["0"])[0] in {"1", "true", "on", "yes"}
                message = params.get("msg", [""])[0]
                message_ok = params.get("ok", ["1"])[0] != "0"

                html_text = render_analysis_html(
                    project_root=project_root,
                    datasets=datasets,
                    selected_dataset=selected_dataset,
                    selected_model=selected_model,
                    selected_prompt_version=selected_prompt_version,
                    selected_per_model=selected_per_model,
                    selected_cov_type=selected_cov_type,
                    message=message,
                    message_ok=message_ok,
                    analysis_result=analysis_state.copy(),
                )
                self._send_html(html_text)
                return

            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/analyze-run":
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not found")
                return

            length = _coerce_int(self.headers.get("Content-Length", "0"), 0)
            body = self.rfile.read(max(0, length)).decode("utf-8", errors="replace")
            params = parse_qs(body)

            datasets = list_analysis_datasets(project_root)
            dataset_raw = params.get("dataset", [""])[0].strip()
            model = params.get("model", [""])[0].strip()
            prompt_version = params.get("prompt_version", [""])[0].strip()
            per_model = params.get("per_model", [""])[0].strip().lower() in {"1", "true", "on", "yes"}
            cov_type = params.get("cov_type", ["HC3"])[0].strip() or "HC3"
            if cov_type not in {"HC0", "HC1", "HC2", "HC3"}:
                cov_type = "HC3"

            try:
                dataset_path = _sanitize_dataset_path(project_root, dataset_raw, datasets)
            except Exception as error:
                message = quote_plus(f"Invalid dataset selection: {error}")
                self._redirect(f"/analyze?ok=0&msg={message}")
                return

            if not analyze_script.exists():
                message = quote_plus(f"Missing analyze script: {analyze_script}")
                self._redirect(f"/analyze?ok=0&msg={message}")
                return

            analysis_state.clear()
            analysis_state.update(
                _run_analysis(
                    python_executable=analysis_python_executable,
                    analyze_script=analyze_script,
                    dataset_path=dataset_path,
                    model=model,
                    prompt_version=prompt_version,
                    per_model=per_model,
                    cov_type=cov_type,
                    timeout_seconds=analysis_timeout_seconds,
                    output_limit_chars=analysis_output_limit_chars,
                )
            )

            ok = "1" if analysis_state.get("ok") else "0"
            message = "Analysis completed." if analysis_state.get("ok") else "Analysis failed."
            query = (
                f"ok={ok}"
                f"&msg={quote_plus(message)}"
                f"&dataset={quote_plus(_relative_label(dataset_path, project_root))}"
                f"&model={quote_plus(model)}"
                f"&prompt_version={quote_plus(prompt_version)}"
                f"&cov_type={quote_plus(cov_type)}"
                f"&per_model={'1' if per_model else '0'}"
            )
            self._redirect(f"/analyze?{query}")

        def log_message(self, *_args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"Dashboard running at http://{host}:{port}")
    print(
        "Dashboard query params: "
        "`?target=<n>` `&refresh=<sec>` `&active_window=<sec>` "
        f"(defaults: target={default_target_trials}, refresh={default_refresh_seconds}, "
        f"active_window={int(default_active_window_seconds)})"
    )
    print("Analysis UI: /analyze")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a local dashboard for block CSV progress.")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind (default: 8765).",
    )
    parser.add_argument(
        "--blocks-dir",
        type=Path,
        default=Path("results/blocks"),
        help="Directory containing block CSV files.",
    )
    parser.add_argument(
        "--conditions",
        type=Path,
        default=Path("configs/realization_effect/conditions.csv"),
        help="Conditions CSV used to define expected condition list.",
    )
    parser.add_argument(
        "--target-trials",
        type=int,
        default=100,
        help="Target trials per condition used for progress calculations (default: 100).",
    )
    parser.add_argument(
        "--refresh-seconds",
        type=int,
        default=10,
        help="Auto-refresh interval in seconds (default: 10).",
    )
    parser.add_argument(
        "--active-window-seconds",
        type=float,
        default=60.0,
        help="Mark block as active when updated within this many seconds (default: 60).",
    )
    parser.add_argument(
        "--analysis-python",
        type=str,
        default="",
        help="Python executable used for on-demand analysis (default: auto-detect venv, else current Python).",
    )
    parser.add_argument(
        "--analysis-timeout-seconds",
        type=int,
        default=600,
        help="Timeout for on-demand analysis commands (default: 600).",
    )
    parser.add_argument(
        "--analysis-output-limit-chars",
        type=int,
        default=200000,
        help="Max analysis output chars kept in UI state (default: 200000).",
    )
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    analyze_script = project_root / "scripts" / "analyze_realization_results.py"
    default_venv_python = project_root / "venv" / "bin" / "python"
    if args.analysis_python.strip():
        analysis_python_executable = args.analysis_python.strip()
    elif default_venv_python.exists():
        analysis_python_executable = str(default_venv_python)
    else:
        analysis_python_executable = sys.executable

    run_server(
        host=args.host,
        port=args.port,
        blocks_dir=args.blocks_dir,
        conditions_path=args.conditions,
        default_target_trials=max(1, args.target_trials),
        default_refresh_seconds=max(1, args.refresh_seconds),
        default_active_window_seconds=max(1.0, args.active_window_seconds),
        project_root=project_root,
        analyze_script=analyze_script,
        analysis_python_executable=analysis_python_executable,
        analysis_timeout_seconds=max(30, args.analysis_timeout_seconds),
        analysis_output_limit_chars=max(5000, args.analysis_output_limit_chars),
    )


if __name__ == "__main__":
    main()
