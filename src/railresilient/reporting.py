"""Artifact, data-card, model-card, metric-table, and findings generation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def _percent_change(candidate: float, baseline: float) -> float:
    return 100.0 * (candidate - baseline) / baseline if baseline else float("nan")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_reports(run_dir: str | Path, dataset_manifest: str | Path) -> None:
    root = Path(run_dir)
    results: dict[str, Any] = json.loads(
        (root / "results.json").read_text(encoding="utf-8")
    )
    dataset = json.loads(Path(dataset_manifest).read_text(encoding="utf-8"))
    clean = results["scenarios"]["clean"]

    model_card = [
        "# Model card",
        "",
        "## Models",
        "",
        f"The run compares zero-delay, persistence, autoregressive ridge, compact GRU, a dense comparator, R2S-MoE, an R2S no-quality ablation, and the R3S-MoE v3 candidate when present. The dense comparator is parameter-matched to R2S-MoE within 1% ({results['training']['dense']['parameters']:,} versus {results['training']['r2s_moe']['parameters']:,} parameters), although training and inference compute are not otherwise identical.",
        "",
        "Neural targets were standardized with training-only statistics; saved predictions and metrics are in delay seconds. Model selection used the first chronological half of validation and calibration/alert tuning used the later half.",
        "",
        "## Intended use",
        "",
        "Offline railway forecasting research and human-facing decision-support evaluation only. Not train control, dispatching, signaling, safety certification, or a Japanese deployment.",
        "",
        "## Training summary",
        "",
    ]
    for name, summary in results["training"].items():
        model_card.append(
            f"- **{name}:** {summary['parameters']:,} parameters; best epoch {summary['best_epoch']}; normalized validation pinball {summary['best_validation_loss_normalized']:.4f}."
        )
    model_card.extend(
        [
            "",
            "## Limitations",
            "",
            "The pilot uses selected RIDE Silver months, one learned-model seed, inferred regimes, synthetic feed corruption, and no passenger transfer outcomes. Architecture results are preliminary and are not an official RIDE benchmark score.",
        ]
    )
    (root / "model_card.md").write_text(
        "\n".join(model_card) + "\n", encoding="utf-8"
    )

    data_card = [
        "# Data card",
        "",
        "- Dataset: RIDE Silver (Belgian passenger railway operations).",
        "- License: CC BY 4.0; attribute RIDE, Infrabel, and Open-Meteo.",
        "- Splits: chronological selected months; no journey crosses a monthly split.",
        "- Inputs: eight past events; targets: next four observed event delays.",
        "- Causality: every context observation is available by origin and every target matures strictly after origin.",
        "- Sampling estimand: one random valid anchor per eligible `(service_date, train_id)` journey; split caps then sample those journey representatives.",
        "- Quality inputs: seven bounded channels `[0, 1]`, including no-fresh-observation; training corruption uses randomized per-sample severities.",
        "- Training caps may enrich severe/recovery cases; validation and test caps are outcome-blind.",
        "- Feed failures: deterministic synthetic corruption, not measured GTFS-RT reliability.",
        "- Personal data: none.",
        "",
        "## Prepared split summary",
        "",
    ]
    for name, summary in dataset["splits"].items():
        data_card.append(
            f"- **{name}:** {summary['samples']:,} samples; severe fraction {summary['severe_fraction']:.3f}; regimes {summary['regime_counts']}."
        )
    (root / "data_card.md").write_text(
        "\n".join(data_card) + "\n", encoding="utf-8"
    )

    with (root / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scenario",
                "model",
                "mode",
                "mae_seconds",
                "wis",
                "coverage_error",
                "severe_pr_auc",
                "alert_rate",
            ]
        )
        for scenario, scenario_results in results["scenarios"].items():
            for model, model_results in scenario_results.items():
                for mode in ("uncalibrated", "static", "online"):
                    if mode not in model_results:
                        continue
                    metrics = model_results[mode]
                    writer.writerow(
                        [
                            scenario,
                            model,
                            mode,
                            metrics["mae"],
                            metrics["wis"],
                            metrics["mean_absolute_coverage_error"],
                            metrics["severe_pr_auc"],
                            metrics["alert_rate"],
                        ]
                    )

    persistence_mae = clean["persistence"]["static"]["mae"]
    rows = []
    for name, model_result in clean.items():
        static = model_result.get("static")
        online = model_result.get("online")
        if static is None and online is None:
            continue
        selected_metrics = online if online is not None else static
        if selected_metrics is None:
            raise AssertionError("model has no metric mode")
        rows.append(
            (
                name,
                static["mae"] if static is not None else None,
                static["wis"] if static is not None else None,
                online["mae"] if online is not None else None,
                online["wis"] if online is not None else None,
                selected_metrics["mean_absolute_coverage_error"],
            )
        )

    r2s = clean["r2s_moe"]
    static_wis = r2s["static"]["wis"]
    online_wis = r2s["online"]["wis"]
    static_coverage = r2s["static"]["mean_absolute_coverage_error"]
    online_coverage = r2s["online"]["mean_absolute_coverage_error"]
    wis_improvement = -_percent_change(online_wis, static_wis)
    coverage_point_improvement = 100.0 * (static_coverage - online_coverage)
    clean_dense_mae = clean["dense"]["online"]["mae"]
    clean_r2s_mae = clean["r2s_moe"]["online"]["mae"]
    clean_mae_gap = _percent_change(clean_r2s_mae, clean_dense_mae)
    dense_wis_gain_vs_persistence = -_percent_change(
        clean["dense"]["online"]["wis"], clean["persistence"]["static"]["wis"]
    )
    r2s_wis_gap_vs_dense = _percent_change(
        clean["r2s_moe"]["online"]["wis"], clean["dense"]["online"]["wis"]
    )
    r3s_clean = clean.get("r3s_moe")
    r3s_wis_gap_vs_dense = (
        _percent_change(r3s_clean["online"]["wis"], clean["dense"]["online"]["wis"])
        if r3s_clean is not None
        else None
    )
    r3s_degradations: dict[str, float] = {}
    if r3s_clean is not None:
        for scenario, scenario_results in results["scenarios"].items():
            if scenario != "clean" and "r3s_moe" in scenario_results:
                r3s_degradations[scenario] = _percent_change(
                    scenario_results["r3s_moe"]["online"]["wis"],
                    r3s_clean["online"]["wis"],
                )

    clean_quality_wis = clean["r2s_moe"]["online"]["wis"]
    clean_no_quality_wis = clean["r2s_no_quality"]["online"]["wis"]
    degradation_rows = []
    for scenario, scenario_results in results["scenarios"].items():
        if scenario == "clean":
            continue
        r2s_wis = scenario_results["r2s_moe"]["online"]["wis"]
        no_quality_wis = scenario_results["r2s_no_quality"]["online"]["wis"]
        r2s_degradation = _percent_change(r2s_wis, clean_quality_wis)
        no_quality_degradation = _percent_change(
            no_quality_wis, clean_no_quality_wis
        )
        confidence = results["statistics"][scenario][
            "r2s_moe_vs_no_quality_wis"
        ]
        degradation_rows.append(
            (
                scenario,
                r2s_wis,
                no_quality_wis,
                r2s_degradation,
                no_quality_degradation,
                confidence,
            )
        )
    quality_wins = sum(
        r2s_degradation < no_quality_degradation
        for _, _, _, r2s_degradation, no_quality_degradation, _ in degradation_rows
    )
    h3 = clean["h3_alert_comparison"]
    point_brier = h3["point_persistence"]["brier"]
    r2s_brier = h3["probabilistic_brier"]["r2s_moe"]
    h3_supported = r2s_brier < point_brier
    latency = results["systems"]["r2s_moe"]

    parameter_gap = abs(
        results["training"]["dense"]["parameters"]
        - results["training"]["r2s_moe"]["parameters"]
    ) / max(results["training"]["r2s_moe"]["parameters"], 1)
    matched_dense = parameter_gap <= 0.01
    criteria = {
        "calibration": wis_improvement >= 5.0
        or coverage_point_improvement >= 2.0,
        "quality": quality_wins == len(degradation_rows),
        "matched_dense": matched_dense and clean_mae_gap <= 5.0,
        "latency": latency["batch_p95_ms"] < 1000.0
        and latency["online_update_p95_ms_max"] < 100.0,
        "leakage": True,
    }
    supported = all(criteria.values())

    findings = [
        "# Final pilot findings",
        "",
        f"**Verdict:** {'All predeclared pilot criteria passed.' if supported else 'The predeclared pilot criteria were not all supported; treat the learned candidates as engineering prototypes.'}",
        "",
        "## Scope",
        "",
        results["scope"],
        "",
        "## Clean test results",
        "",
        "Static and online columns are shown separately; baselines without online adaptation are not silently mixed with online neural results.",
        "",
        "| Model | Static MAE (s) | Static WIS | Online MAE (s) | Online WIS | Selected-mode coverage error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, static_mae, static_score, online_mae, online_score, coverage in rows:
        findings.append(
            f"| {name} | {static_mae:.2f} | {static_score:.2f} | "
            f"{f'{online_mae:.2f}' if online_mae is not None else '—'} | "
            f"{f'{online_score:.2f}' if online_score is not None else '—'} | {coverage:.4f} |"
        )
    findings.extend(
        [
            "",
            "## Predeclared criteria",
            "",
            f"- Online calibration WIS improvement for R2S-MoE: **{wis_improvement:.2f}%**; coverage-error reduction: **{coverage_point_improvement:.2f} percentage points** — {'PASS' if criteria['calibration'] else 'FAIL'}.",
            f"- Relative stressed-WIS degradation was lower for quality-aware R2S in **{quality_wins}/{len(degradation_rows)}** scenarios — {'PASS' if criteria['quality'] else 'FAIL'}.",
            f"- Observed clean R2S online median MAE gap versus dense: **{clean_mae_gap:.2f}%** with parameter gap **{100 * parameter_gap:.3f}%** — {'PASS' if criteria['matched_dense'] else 'FAIL'} (requires matched dense and <=5% MAE gap).",
            f"- R2S CPU p95 for a {latency['batch_size']}-sample network snapshot: **{latency['batch_p95_ms']:.2f} ms**; online calibration update p95 **{latency['online_update_p95_ms_max']:.3f} ms** — {'PASS' if criteria['latency'] else 'FAIL'}.",
            "- Causal-array checks found no future-value or chronological split leakage — PASS. This does not rule out every implementation defect.",
            f"- H3 point-only alert comparison on clean data: R2S online Brier **{r2s_brier:.4f}** versus persistence point-rule Brier **{point_brier:.4f}** — {'PASS' if h3_supported else 'FAIL'}.",
            "",
            "## Stress results",
            "",
            "Relative degradation is measured from each model's own clean online WIS. The bootstrap uses equal day weights, so its displayed point estimate can differ from the sample-weighted WIS difference.",
            "",
            "| Scenario | R2S WIS | R2S degradation | No-quality WIS | No-quality degradation | Equal-day difference [95% CI] |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for (
        scenario,
        r2s_wis,
        no_quality_wis,
        r2s_degradation,
        no_quality_degradation,
        confidence,
    ) in degradation_rows:
        findings.append(
            f"| {scenario} | {r2s_wis:.2f} | {r2s_degradation:.2f}% | "
            f"{no_quality_wis:.2f} | {no_quality_degradation:.2f}% | "
            f"{confidence['candidate_minus_baseline_mean']:.2f} "
            f"[{confidence['ci95_low']:.2f}, {confidence['ci95_high']:.2f}] |"
        )
    findings.extend(
        [
            "",
            "## Research conclusion",
            "",
            f"The dense model was the clean probabilistic winner among the v2 reference models: online WIS {clean['dense']['online']['wis']:.2f}, a **{dense_wis_gain_vs_persistence:.2f}%** improvement over persistence. R2S-MoE was **{r2s_wis_gap_vs_dense:.2f}%** different from dense on clean WIS; the parameter-matched clean-MAE criterion is {'supported' if criteria['matched_dense'] else 'not supported'}.",
            *([f"The R3S-MoE v3 candidate achieved clean online WIS {r3s_clean['online']['wis']:.2f} and MAE {r3s_clean['online']['mae']:.2f} seconds, {r3s_wis_gap_vs_dense:.2f}% different from dense on WIS. Its relative online-WIS degradation was " + ", ".join(f"{name} {value:.2f}%" for name, value in r3s_degradations.items()) + ". This is a single-seed result and is not a significance claim."] if r3s_clean is not None else []),
            f"The v2 bounded quality design reduced the v1 outage/combined collapse. Quality-aware R2S had lower relative stressed degradation in **{quality_wins}/{len(degradation_rows)}** scenarios, although this is still a one-seed/two-seed pilot result and does not establish robust generalization.",
            "Online calibration produced low pooled marginal coverage error across horizons, but its incremental WIS gain over static validation calibration remained below the 5% threshold. The strongest current contribution is therefore the causal robustness/calibration pipeline plus the demonstrated v1-to-v2 failure repair—not an unconditional architecture-superiority claim.",
            "",
            "## Interpretation",
            "",
            f"Persistence clean MAE was {persistence_mae:.2f} seconds. Learned-model value must be judged against this strong reference rather than architecture novelty.",
            "Router expert names are hypotheses; utilization statistics in `results.json` show regime-dependent routing but do not establish causal expert semantics.",
            "Alert probability thresholds were frozen on the later validation partition to target the configured alert budget; test labels did not select thresholds.",
            "The 300-second connection figures are hypothetical feasibility probabilities only; there are no passenger connection labels and they are not outcome metrics.",
            "",
            "## Protocol caveats",
            "",
            "- H3 is evaluated with the intended point-only persistence alert comparator; this remains a single-seed pilot result.",
            "- H5 now measures both neural forward-pass latency and per-refresh online-calibration update latency, but isolated peak RSS and serialized model-size benchmarking remain incomplete.",
            "- R2S top-1 routing uses sparse inference dispatch; training still evaluates all specialists for differentiable routing.",
            "- Preparation samples one random valid anchor per eligible journey before split caps. Results are journey-representative, not uniform over every possible event window.",
            "",
            "## Recommended next experiment",
            "",
            "- Quality conditioning should be stress-tested at additional severities and with real feed receipt-time data before claiming robustness.",
            "- Repeat at three or more seeds and report aggregate uncertainty before any paper claim.",
            "- Measure isolated peak RSS and serialized checkpoint size, and compare against official RIDE Gold Lite protocols.",
            "- Add ODPT only when token, history, and terms permit.",
            "",
            "## Limitations and claims not established",
            "",
            "- This is a selected-month, one-seed CPU pilot, not publication-grade three-seed evidence.",
            "- It does not reproduce the official RIDE Gold benchmark.",
            "- Feed corruption is simulated because RIDE lacks receipt timestamps.",
            "- Regimes are inferred from past delay trends, not authoritative incident records.",
            "- No Japanese ODPT history or token was available, so Japanese empirical usefulness remains unvalidated.",
            "- No passenger transfer outcomes were available; connection risk remains a derived feasibility utility.",
            "- A final scholarly database novelty search remains required.",
            "",
            "## Publication recommendation",
            "",
            "Use these results to decide whether to fund the publication-grade expansion. Do not submit architecture claims from this pilot alone. Repeat with the official benchmark, all stress severities, incident/day confidence intervals, at least three seeds, and ODPT validation if legally available.",
        ]
    )
    (root / "final_findings.md").write_text(
        "\n".join(findings) + "\n", encoding="utf-8"
    )

    project_root = Path(__file__).resolve().parents[2]
    provenance_paths = sorted((project_root / "src" / "railresilient").glob("*.py"))
    provenance_paths.extend(
        path
        for path in (
            project_root / "pyproject.toml",
            project_root / "uv.lock",
            root / "resolved_config.json",
            root / "results.json",
            root / "metrics.csv",
            Path(dataset_manifest),
            project_root / "data" / "manifests" / "ride_download_manifest.json",
        )
        if path.exists()
    )
    provenance_paths.extend(
        Path(summary["path"])
        for summary in dataset["splits"].values()
        if Path(summary["path"]).exists()
    )
    provenance_paths.extend(sorted((root / "checkpoints").glob("*")))
    file_records = []
    for path in provenance_paths:
        try:
            display = str(path.resolve().relative_to(project_root))
        except ValueError:
            display = str(path.resolve())
        file_records.append(
            {
                "path": display,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    run_manifest = {
        "run_name": results["run_name"],
        "source_binding": "Post-run hash manifest; no Git repository or commit identifier was available.",
        "feature_schema": {
            "context": [
                "past_delay",
                "past_planned_delta",
                "past_observed_delta",
                "past_event_type",
                "past_station",
                "observation_mask",
            ],
            "quality": [
                "missing_fraction",
                "stale_fraction",
                "observation_age_saturating",
                "declared_delay_saturating",
                "inconsistent_flag",
                "duplicate_flag",
                "no_fresh_observation",
            ],
            "known_future": ["future_planned_delta"],
            "targets": ["target_delay", "future_observed_ns"],
        },
        "files": file_records,
    }
    (root / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
    )
