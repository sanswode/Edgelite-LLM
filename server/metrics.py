from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .prefix_cache import kv_cache_size_mb


STRATEGY_LABELS = {
    "cloud_only": "Cloud-only",
    "edge_only": "Edge-only",
    "static_threshold": "Static-threshold",
    "semantic_cache_only": "Semantic-cache-only",
    "no_privacy": "No-privacy",
    "ours": "Ours",
    "full": "Full",
    "wo_scheduler": "w/o Scheduler",
    "wo_semantic_cache": "w/o Semantic Cache",
    "wo_kv_cache": "w/o KV Cache",
    "wo_privacy_gate": "w/o Privacy Gate",
}

ACTION_LABELS = {
    "local": "Local",
    "edge": "Edge",
    "cloud": "Cloud",
    "cache": "Cache",
    "draft_edge": "Draft->Edge",
}


def save_rows(rows: List[Dict[str, object]], path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def percentile(series: pd.Series, value: float) -> float:
    if series.empty:
        return 0.0
    return float(np.percentile(series.to_numpy(), value))


def summarize_by_group(frame: pd.DataFrame, group_keys: Iterable[str]) -> pd.DataFrame:
    summary = (
        frame.groupby(list(group_keys), dropna=False)
        .apply(
            lambda part: pd.Series(
                {
                    "request_count": len(part),
                    "mean_ttft_ms": part["ttft_ms"].mean(),
                    "mean_e2e_ms": part["e2e_ms"].mean(),
                    "p95_e2e_ms": percentile(part["e2e_ms"], 95),
                    "p99_e2e_ms": percentile(part["e2e_ms"], 99),
                    "mean_tpot_ms": part["tpot_ms"].mean(),
                    "mean_throughput_tps": part["throughput_tps"].mean(),
                    "cache_hit_ratio": part["semantic_hit"].mean(),
                    "prefix_hit_ratio": part["prefix_hit"].mean(),
                    "mean_bandwidth_bytes": part["bandwidth_bytes"].mean(),
                    "mean_vram_peak_gb": part["vram_peak_gb"].mean(),
                    "privacy_exposure_rate": part.loc[part["sensitive"] == 1, "privacy_exposed"].mean()
                    if (part["sensitive"] == 1).any()
                    else 0.0,
                    "mean_quality": part["quality_score"].mean(),
                }
            )
        )
        .reset_index()
    )
    return summary


def build_main_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = summarize_by_group(frame, ["network", "network_label", "strategy"])
    summary["strategy_label"] = summary["strategy"].map(STRATEGY_LABELS)
    return summary.sort_values(["network", "strategy"])


def build_privacy_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = summarize_by_group(frame, ["network", "network_label", "strategy"])
    summary["strategy_label"] = summary["strategy"].map(STRATEGY_LABELS)
    return summary.sort_values(["network", "strategy"])


def build_semantic_threshold_summary(frame: pd.DataFrame, baseline_mean_e2e: float) -> pd.DataFrame:
    summary = summarize_by_group(frame, ["network", "network_label", "semantic_threshold"])
    summary["latency_gain_pct"] = (baseline_mean_e2e - summary["mean_e2e_ms"]) / baseline_mean_e2e * 100.0
    return summary.sort_values("semantic_threshold")


def build_ablation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = summarize_by_group(frame, ["strategy"])
    summary["strategy_label"] = summary["strategy"].map(STRATEGY_LABELS)
    return summary.sort_values("strategy")


def build_action_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["network", "network_label", "route"]).size().reset_index(name="count")
    totals = grouped.groupby(["network"])["count"].transform("sum")
    grouped["ratio"] = grouped["count"] / totals
    grouped["route_label"] = grouped["route"].map(ACTION_LABELS)
    return grouped.sort_values(["network", "route"])


def build_vram_budget(configs: Dict[str, object]) -> pd.DataFrame:
    models = configs["models"]["vram_figure_models"]
    hardware = configs["hardware"]
    contexts = [1024, 2048, 4096, 8192]
    rows = []
    for model in models:
        for context_tokens in contexts:
            weight_gb = (model["params_billion"] * 1_000_000_000 * model["quant_bits"] / 8.0) / (1024.0 ** 3)
            weight_gb *= 1.0 + model["overhead"]
            kv_mb = (
                2
                * model["layers"]
                * context_tokens
                * model["kv_heads"]
                * model["head_dim"]
                * 2
                / (1024.0 * 1024.0)
            )
            total_gb = weight_gb + kv_mb / 1024.0 + hardware["runtime_gb"] + hardware["fragment_gb"]
            rows.append(
                {
                    "model_name": model["name"],
                    "context_tokens": context_tokens,
                    "weight_gb": weight_gb,
                    "kv_gb": kv_mb / 1024.0,
                    "total_gb": total_gb,
                    "safe_limit_gb": hardware["gpu_vram_gb"] * hardware["safe_utilization"],
                }
            )
    return pd.DataFrame(rows)


def dataframe_to_markdown(frame: pd.DataFrame, float_columns: Iterable[str]) -> str:
    display = frame.copy()
    for column in float_columns:
        if column in display.columns:
            display[column] = display[column].map(lambda value: f"{value:.3f}" if pd.notna(value) else "")

    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def write_report(
    base_dir: Path,
    main_summary: pd.DataFrame,
    privacy_summary: pd.DataFrame,
    semantic_summary: pd.DataFrame,
    ablation_summary: pd.DataFrame,
) -> None:
    report_path = base_dir / "results" / "report.md"

    weak_rows = main_summary[(main_summary["network"] == "weak") & (main_summary["strategy"].isin(["cloud_only", "ours"]))]
    weak_lookup = {row["strategy"]: row for row in weak_rows.to_dict("records")}
    weak_gain = 0.0
    if "cloud_only" in weak_lookup and "ours" in weak_lookup:
        weak_gain = (
            weak_lookup["cloud_only"]["p95_e2e_ms"] - weak_lookup["ours"]["p95_e2e_ms"]
        ) / weak_lookup["cloud_only"]["p95_e2e_ms"] * 100.0

    privacy_rows = privacy_summary[privacy_summary["strategy"].isin(["no_privacy", "ours"])]
    privacy_lookup = {row["strategy"]: row for row in privacy_rows.groupby("strategy")["privacy_exposure_rate"].mean().reset_index().to_dict("records")}
    privacy_drop = 0.0
    if "no_privacy" in privacy_lookup and "ours" in privacy_lookup:
        privacy_drop = (
            privacy_lookup["no_privacy"]["privacy_exposure_rate"] - privacy_lookup["ours"]["privacy_exposure_rate"]
        ) / max(privacy_lookup["no_privacy"]["privacy_exposure_rate"], 1e-6) * 100.0

    best_threshold_row = semantic_summary.sort_values("latency_gain_pct", ascending=False).iloc[0]

    report = f"""# Experiment Report

## Highlights

- Weak-network P95 latency improvement of `Ours` over `Cloud-only`: **{weak_gain:.2f}%**
- Privacy exposure reduction of `Ours` over `No-privacy`: **{privacy_drop:.2f}%**
- Best semantic-cache threshold in this run: **{best_threshold_row['semantic_threshold']:.2f}**

## Main Summary

{dataframe_to_markdown(main_summary, ['mean_ttft_ms', 'mean_e2e_ms', 'p95_e2e_ms', 'p99_e2e_ms', 'cache_hit_ratio', 'prefix_hit_ratio', 'privacy_exposure_rate', 'mean_quality'])}

## Privacy Summary

{dataframe_to_markdown(privacy_summary, ['mean_e2e_ms', 'privacy_exposure_rate', 'mean_quality'])}

## Semantic Threshold Sweep

{dataframe_to_markdown(semantic_summary, ['semantic_threshold', 'cache_hit_ratio', 'mean_e2e_ms', 'latency_gain_pct'])}

## Ablation Summary

{dataframe_to_markdown(ablation_summary, ['mean_e2e_ms', 'p95_e2e_ms', 'cache_hit_ratio', 'prefix_hit_ratio', 'privacy_exposure_rate', 'mean_quality'])}
"""
    report_path.write_text(report, encoding="utf-8")
