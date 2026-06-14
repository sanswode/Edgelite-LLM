from __future__ import annotations

import copy
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BASE_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "scripts"))

from client.replay_requests import load_requests
from plot_statistical_results import generate_statistical_figures
from run_experiment import generate_general_requests, run_batch, stable_seed
from server.config import load_all_configs, load_json_compatible_yaml
from server.entities import NetworkProfile, Request
from server.metrics import STRATEGY_LABELS, build_action_distribution, dataframe_to_markdown, summarize_by_group


RESULTS_DIR = BASE_DIR / "results" / "statistical"
DATASET_DIR = RESULTS_DIR / "datasets"
FIGURES_DIR = RESULTS_DIR / "figures"

SUMMARY_METRICS = [
    "request_count",
    "mean_ttft_ms",
    "mean_e2e_ms",
    "p95_e2e_ms",
    "p99_e2e_ms",
    "mean_tpot_ms",
    "mean_throughput_tps",
    "cache_hit_ratio",
    "prefix_hit_ratio",
    "mean_bandwidth_bytes",
    "mean_vram_peak_gb",
    "privacy_exposure_rate",
    "mean_quality",
]

MODEL_SCALE_B = {
    "Qwen2.5-0.5B": 0.5,
    "Llama-3.2-1B": 1.0,
    "TinyLlama-1.1B": 1.1,
    "Gemma-2-2B": 2.0,
    "Qwen2.5-3B": 3.0,
}


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_networks(configs: Dict[str, object]) -> List[NetworkProfile]:
    return [NetworkProfile(**profile) for profile in configs["network_profiles"]["profiles"]]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clone_request(request: Request, **updates: object) -> Request:
    payload = request.__dict__.copy()
    payload.update(updates)
    return Request(**payload)


def base_requests_for_seed(seed: int, request_count: int) -> List[Request]:
    path = DATASET_DIR / f"seed_{seed}_requests_{request_count}.jsonl"
    generate_general_requests(path, request_count, stable_seed(seed, "statistical-dataset", request_count))
    return load_requests(path)


def apply_context_profile(requests: List[Request], context_target: int, seed: int) -> List[Request]:
    rng = random.Random(stable_seed(seed, "context", context_target))
    context_scale = context_target / 2048.0
    adjusted: List[Request] = []
    for index, request in enumerate(requests):
        prompt_tokens = int(request.prompt_tokens)
        prefix_tokens = int(request.prefix_tokens)
        output_tokens = int(request.output_tokens)
        difficulty = float(request.difficulty)
        quality_target = float(request.quality_target)

        if request.category == "long_context":
            prompt_tokens = int(context_target * rng.uniform(0.86, 1.12))
            prefix_tokens = int(prompt_tokens * rng.uniform(0.42, 0.60))
            output_tokens = int(output_tokens * (0.94 + 0.12 * context_scale))
            difficulty = clamp(difficulty + 0.03 * math.log2(max(context_scale, 0.5)), 0.18, 0.98)
            quality_target = clamp(quality_target + 0.01 * math.log2(max(context_scale, 0.5)), 0.55, 0.99)
        elif request.category == "rag":
            factor = 0.80 + 0.28 * context_scale
            prompt_tokens = int(prompt_tokens * factor)
            prefix_tokens = int(prefix_tokens * (0.84 + 0.24 * context_scale))
        elif request.category == "ordinary":
            prompt_tokens = int(prompt_tokens * (0.96 + 0.08 * context_scale))
        elif request.category == "privacy":
            prompt_tokens = int(prompt_tokens * (0.94 + 0.10 * context_scale))

        prefix_tokens = max(24, min(prefix_tokens, prompt_tokens - 12))
        prompt_tokens = max(prefix_tokens + 12, prompt_tokens)
        output_tokens = max(24, output_tokens)
        adjusted.append(
            clone_request(
                request,
                request_id=f"{request.request_id}-ctx{context_target}-{index:03d}",
                prompt_tokens=prompt_tokens,
                prefix_tokens=prefix_tokens,
                output_tokens=output_tokens,
                difficulty=round(difficulty, 4),
                quality_target=round(quality_target, 4),
            )
        )
    return adjusted


def apply_arrival_profile(requests: List[Request], arrival_mean_ms: float, seed: int) -> List[Request]:
    rng = random.Random(stable_seed(seed, "arrival", arrival_mean_ms))
    arrival_ms = 0.0
    adjusted: List[Request] = []
    for index, request in enumerate(requests):
        inter_arrival = rng.expovariate(1.0 / max(arrival_mean_ms, 1.0))
        if rng.random() < 0.12:
            inter_arrival = rng.uniform(arrival_mean_ms * 0.08, arrival_mean_ms * 0.28)
        arrival_ms += inter_arrival
        adjusted.append(
            clone_request(
                request,
                request_id=f"{request.request_id}-arr{int(arrival_mean_ms)}-{index:03d}",
                arrival_ms=round(arrival_ms, 3),
            )
        )
    return adjusted


def prepare_requests(seed: int, request_count: int, arrival_mean_ms: float, context_target: int) -> List[Request]:
    requests = base_requests_for_seed(seed, request_count)
    requests = apply_context_profile(requests, context_target=context_target, seed=seed)
    requests = apply_arrival_profile(requests, arrival_mean_ms=arrival_mean_ms, seed=seed)
    return requests


def build_configs(
    base_configs: Dict[str, object],
    seed: int,
    strategies: Iterable[str],
    cloud_parallelism: float,
    edge_profile: Dict[str, object] | None = None,
    allowed_routes: Iterable[str] | None = None,
) -> Dict[str, object]:
    configs = copy.deepcopy(base_configs)
    configs["hardware"]["seed"] = seed
    configs["hardware"]["parallelism"] = {
        "local": 1.0,
        "edge": 5.0,
        "cloud": float(cloud_parallelism),
    }
    configs["experiment"]["strategies"] = list(strategies)
    if allowed_routes is None:
        configs["experiment"].pop("allowed_routes", None)
    else:
        configs["experiment"]["allowed_routes"] = list(allowed_routes)
    if edge_profile is not None:
        configs["models"]["models"]["edge"] = copy.deepcopy(edge_profile)
    return configs


def append_metadata(rows: List[Dict[str, object]], metadata: Dict[str, object]) -> List[Dict[str, object]]:
    for row in rows:
        row.update(metadata)
        row["strategy_label"] = STRATEGY_LABELS.get(str(row["strategy"]), str(row["strategy"]))
    return rows


def summarize_seed(frame: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    summary = summarize_by_group(frame, group_cols)
    if "strategy" in summary.columns:
        summary["strategy_label"] = summary["strategy"].map(STRATEGY_LABELS).fillna(summary["strategy"])
    return summary


def metric_ci(values: pd.Series) -> tuple[float, float, int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return 0.0, 0.0, 0
    mean = float(numeric.mean())
    if len(numeric) == 1:
        return mean, 0.0, 1
    std = float(numeric.std(ddof=1))
    ci95 = float(stats.t.ppf(0.975, len(numeric) - 1) * std / math.sqrt(len(numeric)))
    return mean, ci95, len(numeric)


def aggregate_seed_summary(seed_summary: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for keys, part in seed_summary.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = {column: value for column, value in zip(group_cols, keys)}
        for metric in SUMMARY_METRICS:
            mean, ci95, n = metric_ci(part[metric])
            row[metric] = mean
            row[f"{metric}_ci95"] = ci95
            row[f"{metric}_seed_n"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_single_metric(seed_frame: pd.DataFrame, group_cols: List[str], metric: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for keys, part in seed_frame.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        mean, ci95, n = metric_ci(part[metric])
        row = {column: value for column, value in zip(group_cols, keys)}
        row[metric] = mean
        row[f"{metric}_ci95"] = ci95
        row[f"{metric}_seed_n"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def safe_paired_ttest(baseline: pd.Series, ours: pd.Series) -> tuple[float, float]:
    diff = pd.to_numeric(baseline, errors="coerce") - pd.to_numeric(ours, errors="coerce")
    diff = diff.dropna()
    if diff.empty:
        return 0.0, 1.0
    if len(diff) == 1:
        return 0.0, 1.0
    if float(diff.std(ddof=1)) == 0.0:
        return (math.inf, 0.0) if float(diff.iloc[0]) != 0.0 else (0.0, 1.0)
    test = stats.ttest_rel(baseline, ours, nan_policy="omit")
    return float(test.statistic), float(test.pvalue)


def significance_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def paired_significance(
    seed_summary: pd.DataFrame,
    group_cols: List[str],
    metric_cols: Iterable[str],
    baselines: Iterable[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    ours_frame = seed_summary[seed_summary["strategy"] == "ours"]
    for keys, ours_part in ours_frame.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        group_payload = {column: value for column, value in zip(group_cols, keys)}
        for baseline in baselines:
            baseline_part = seed_summary[
                (seed_summary["strategy"] == baseline)
                & np.logical_and.reduce(
                    [seed_summary[column] == value for column, value in group_payload.items()]
                )
            ]
            merged = ours_part.merge(
                baseline_part,
                on=["seed", *group_cols],
                suffixes=("_ours", "_baseline"),
            )
            if merged.empty:
                continue
            for metric in metric_cols:
                baseline_values = merged[f"{metric}_baseline"]
                ours_values = merged[f"{metric}_ours"]
                statistic, p_value = safe_paired_ttest(baseline_values, ours_values)
                diff = pd.to_numeric(baseline_values, errors="coerce") - pd.to_numeric(ours_values, errors="coerce")
                diff = diff.dropna()
                effect_d = 0.0
                if len(diff) > 1 and float(diff.std(ddof=1)) > 0.0:
                    effect_d = float(diff.mean() / diff.std(ddof=1))
                if metric == "privacy_exposure_rate":
                    improvement_pct = float(diff.mean() * 100.0)
                else:
                    baseline_mean = max(float(pd.to_numeric(baseline_values, errors="coerce").mean()), 1e-9)
                    improvement_pct = float(diff.mean() / baseline_mean * 100.0)
                row = dict(group_payload)
                row.update(
                    {
                        "baseline": baseline,
                        "baseline_label": STRATEGY_LABELS.get(baseline, baseline),
                        "metric": metric,
                        "ours_mean": float(pd.to_numeric(ours_values, errors="coerce").mean()),
                        "baseline_mean": float(pd.to_numeric(baseline_values, errors="coerce").mean()),
                        "improvement_pct": improvement_pct,
                        "t_statistic": statistic,
                        "p_value": p_value,
                        "effect_size_d": effect_d,
                        "significance": significance_stars(p_value),
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def select_representative_seed(seed_summary: pd.DataFrame, group_cols: List[str], rule_text: str) -> pd.DataFrame:
    ours = seed_summary[seed_summary["strategy"] == "ours"].copy()
    ascending = [True] * len(group_cols) + [False, True, True]
    ours = ours.sort_values([*group_cols, "mean_quality", "mean_e2e_ms", "seed"], ascending=ascending)
    representative = ours.drop_duplicates(group_cols, keep="first").reset_index(drop=True)
    representative["selection_rule"] = rule_text
    return representative


def representative_action_distribution(main_frame: pd.DataFrame, representative: pd.DataFrame) -> pd.DataFrame:
    selected = main_frame[main_frame["strategy"] == "ours"].merge(
        representative[["seed", "network", "network_label"]],
        on=["seed", "network", "network_label"],
        how="inner",
    )
    return build_action_distribution(selected)


def write_markdown_report(
    stat_config: Dict[str, object],
    main_overall_ci: pd.DataFrame,
    significance: pd.DataFrame,
    arrival_ci: pd.DataFrame,
    worker_ci: pd.DataFrame,
    context_ci: pd.DataFrame,
    model_ci: pd.DataFrame,
    representative: pd.DataFrame,
) -> None:
    key_sig = significance[
        (significance["metric"] == "mean_e2e_ms")
        & (significance["baseline"].isin(["cloud_only", "edge_only", "static_threshold"]))
    ].copy()
    key_sig = key_sig.sort_values(["network", "baseline"])

    report = [
        "# Statistical Experiment Upgrade Report",
        "",
        "## Setup",
        "",
        f"- Runtime: `D:\\anaconda\\envs\\edgeLLM\\python.exe`",
        f"- Seeds: `{', '.join(str(seed) for seed in stat_config['seed_values'])}`",
        f"- Strategies: `{', '.join(stat_config['strategies'])}`",
        f"- Requests per network profile: `{int(stat_config['request_count_per_network'])}`",
        f"- Representative-seed rule: `{stat_config['representative_seed_rule']}`",
        "",
        "## Overall Main Comparison",
        "",
        dataframe_to_markdown(
            main_overall_ci,
            [
                "mean_ttft_ms",
                "mean_e2e_ms",
                "p95_e2e_ms",
                "cache_hit_ratio",
                "mean_bandwidth_bytes",
                "privacy_exposure_rate",
                "mean_quality",
            ],
        ),
        "",
        "## Paired Significance Tests",
        "",
        dataframe_to_markdown(
            key_sig,
            ["ours_mean", "baseline_mean", "improvement_pct", "t_statistic", "p_value", "effect_size_d"],
        ),
        "",
        "## Arrival Sweep",
        "",
        dataframe_to_markdown(
            arrival_ci,
            ["mean_e2e_ms", "mean_e2e_ms_ci95", "p95_e2e_ms", "cache_hit_ratio", "mean_quality"],
        ),
        "",
        "## Worker Sweep",
        "",
        dataframe_to_markdown(
            worker_ci,
            ["mean_e2e_ms", "mean_e2e_ms_ci95", "p95_e2e_ms", "cache_hit_ratio", "mean_quality"],
        ),
        "",
        "## Context Sweep",
        "",
        dataframe_to_markdown(
            context_ci,
            ["mean_e2e_ms", "mean_e2e_ms_ci95", "p95_e2e_ms", "cache_hit_ratio", "mean_quality"],
        ),
        "",
        "## Cross-Model Four-Strategy Comparison",
        "",
        dataframe_to_markdown(
            model_ci,
            [
                "model_scale_b",
                "mean_e2e_ms",
                "mean_e2e_ms_ci95",
                "p95_e2e_ms",
                "required_vram_peak_gb",
                "mean_quality",
                "privacy_exposure_rate",
            ],
        ),
        "",
        "## Representative Seeds",
        "",
        dataframe_to_markdown(representative, ["mean_e2e_ms", "mean_quality"]),
    ]
    (RESULTS_DIR / "report.md").write_text("\n".join(report), encoding="utf-8")


def run_main_sweep(
    base_configs: Dict[str, object],
    stat_config: Dict[str, object],
    networks: List[NetworkProfile],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    default_arrival = float(stat_config["arrival_mean_ms"][stat_config["scenario_defaults"]["arrival_label"]])
    default_workers = float(stat_config["cloud_parallelism"][stat_config["scenario_defaults"]["worker_label"]])
    default_context = int(stat_config["context_targets"][stat_config["scenario_defaults"]["context_label"]])
    strategies = list(stat_config["strategies"])
    request_count = int(stat_config["request_count_per_network"])

    rows: List[Dict[str, object]] = []
    for seed in stat_config["seed_values"]:
        requests = prepare_requests(seed, request_count, default_arrival, default_context)
        configs = build_configs(base_configs, seed, strategies, default_workers)
        batch_rows = run_batch(
            configs=configs,
            requests=requests,
            networks=networks,
            strategies=strategies,
            experiment_name="stat_main",
        )
        rows.extend(
            append_metadata(
                batch_rows,
                {
                    "seed": seed,
                    "scenario_family": "main",
                    "arrival_label": stat_config["scenario_defaults"]["arrival_label"],
                    "worker_label": stat_config["scenario_defaults"]["worker_label"],
                    "context_label": stat_config["scenario_defaults"]["context_label"],
                },
            )
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS_DIR / "main_raw_logs.csv", index=False)

    seed_summary = summarize_seed(frame, ["seed", "network", "network_label", "strategy"])
    seed_summary.to_csv(RESULTS_DIR / "main_seed_summary.csv", index=False)

    ci_summary = aggregate_seed_summary(seed_summary, ["network", "network_label", "strategy", "strategy_label"])
    ci_summary.to_csv(RESULTS_DIR / "main_ci_summary.csv", index=False)

    overall_seed = summarize_seed(frame, ["seed", "strategy"])
    overall_seed.to_csv(RESULTS_DIR / "main_overall_seed_summary.csv", index=False)

    overall_ci = aggregate_seed_summary(overall_seed, ["strategy", "strategy_label"])
    overall_ci.to_csv(RESULTS_DIR / "main_overall_ci_summary.csv", index=False)

    significance = paired_significance(
        seed_summary=seed_summary,
        group_cols=["network", "network_label"],
        metric_cols=["mean_e2e_ms", "p95_e2e_ms", "privacy_exposure_rate", "mean_quality"],
        baselines=["cloud_only", "edge_only", "static_threshold"],
    )
    significance.to_csv(RESULTS_DIR / "main_significance.csv", index=False)

    representative = select_representative_seed(
        seed_summary=seed_summary,
        group_cols=["network", "network_label"],
        rule_text=str(stat_config["representative_seed_rule"]),
    )
    representative.to_csv(RESULTS_DIR / "representative_seeds.csv", index=False)

    action_distribution = representative_action_distribution(frame, representative)
    action_distribution.to_csv(RESULTS_DIR / "representative_action_distribution.csv", index=False)

    return frame, seed_summary, ci_summary, overall_ci, significance


def run_factor_sweep(
    base_configs: Dict[str, object],
    stat_config: Dict[str, object],
    networks: List[NetworkProfile],
    factor_name: str,
    factor_values: Dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    defaults = stat_config["scenario_defaults"]
    default_arrival = float(stat_config["arrival_mean_ms"][defaults["arrival_label"]])
    default_workers = float(stat_config["cloud_parallelism"][defaults["worker_label"]])
    default_context = int(stat_config["context_targets"][defaults["context_label"]])
    strategies = list(stat_config["strategies"])
    request_count = int(stat_config["request_count_per_network"])

    rows: List[Dict[str, object]] = []
    for label, value in factor_values.items():
        for seed in stat_config["seed_values"]:
            arrival_mean = default_arrival
            worker_count = default_workers
            context_target = default_context
            if factor_name == "arrival_label":
                arrival_mean = float(value)
            elif factor_name == "worker_label":
                worker_count = float(value)
            elif factor_name == "context_label":
                context_target = int(value)

            requests = prepare_requests(seed, request_count, arrival_mean, context_target)
            configs = build_configs(base_configs, seed, strategies, worker_count)
            batch_rows = run_batch(
                configs=configs,
                requests=requests,
                networks=networks,
                strategies=strategies,
                experiment_name=f"stat_{factor_name}",
            )
            rows.extend(
                append_metadata(
                    batch_rows,
                    {
                        "seed": seed,
                        "scenario_family": factor_name,
                        "arrival_label": label if factor_name == "arrival_label" else defaults["arrival_label"],
                        "worker_label": label if factor_name == "worker_label" else defaults["worker_label"],
                        "context_label": label if factor_name == "context_label" else defaults["context_label"],
                    },
                )
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS_DIR / f"{factor_name}_raw_logs.csv", index=False)

    seed_summary = summarize_seed(frame, ["seed", factor_name, "strategy"])
    seed_summary.to_csv(RESULTS_DIR / f"{factor_name}_seed_summary.csv", index=False)

    ci_summary = aggregate_seed_summary(seed_summary, [factor_name, "strategy", "strategy_label"])
    ci_summary.to_csv(RESULTS_DIR / f"{factor_name}_ci_summary.csv", index=False)

    return frame, seed_summary, ci_summary


def run_model_strategy_sweep(
    base_configs: Dict[str, object],
    stat_config: Dict[str, object],
    networks: List[NetworkProfile],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    defaults = stat_config["scenario_defaults"]
    default_arrival = float(stat_config["arrival_mean_ms"][defaults["arrival_label"]])
    default_workers = float(stat_config["cloud_parallelism"][defaults["worker_label"]])
    default_context = int(stat_config["context_targets"][defaults["context_label"]])
    request_count = int(stat_config["request_count_per_network"])
    strategies = list(stat_config["strategies"])
    model_compare = stat_config["model_compare"]
    edge_profiles = model_compare["edge_profiles"]
    allowed_routes = model_compare["allowed_routes"]

    rows: List[Dict[str, object]] = []
    for model_label, edge_profile in edge_profiles.items():
        for seed in stat_config["seed_values"]:
            requests = prepare_requests(seed, request_count, default_arrival, default_context)
            configs = build_configs(
                base_configs=base_configs,
                seed=seed,
                strategies=strategies,
                cloud_parallelism=default_workers,
                edge_profile=edge_profile,
                allowed_routes=allowed_routes,
            )
            batch_rows = run_batch(
                configs=configs,
                requests=requests,
                networks=networks,
                strategies=strategies,
                experiment_name=f"model_strategy_{model_label}",
            )
            rows.extend(
                append_metadata(
                    batch_rows,
                    {
                        "seed": seed,
                        "scenario_family": "model_strategy",
                        "model_label": model_label,
                        "model_scale_b": MODEL_SCALE_B.get(model_label, math.nan),
                        "arrival_label": defaults["arrival_label"],
                        "worker_label": defaults["worker_label"],
                        "context_label": defaults["context_label"],
                    },
                )
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS_DIR / "model_strategy_raw_logs.csv", index=False)

    seed_summary = summarize_seed(frame, ["seed", "model_label", "model_scale_b", "strategy"])
    seed_summary.to_csv(RESULTS_DIR / "model_strategy_seed_summary.csv", index=False)

    ci_summary = aggregate_seed_summary(seed_summary, ["model_label", "model_scale_b", "strategy", "strategy_label"])
    ci_summary.to_csv(RESULTS_DIR / "model_strategy_ci_summary.csv", index=False)

    network_seed_summary = summarize_seed(frame, ["seed", "model_label", "model_scale_b", "network", "network_label", "strategy"])
    network_seed_summary.to_csv(RESULTS_DIR / "model_strategy_network_seed_summary.csv", index=False)

    network_ci_summary = aggregate_seed_summary(
        network_seed_summary,
        ["model_label", "model_scale_b", "network", "network_label", "strategy", "strategy_label"],
    )
    network_ci_summary.to_csv(RESULTS_DIR / "model_strategy_network_ci_summary.csv", index=False)

    edge_routes = frame[frame["route"].isin(["local", "edge", "draft_edge"])].copy()
    peak_vram_seed = (
        edge_routes.groupby(["seed", "model_label", "model_scale_b", "strategy", "strategy_label"], as_index=False)
        .agg(required_vram_peak_gb=("vram_peak_gb", "max"))
    )
    peak_vram_seed.to_csv(RESULTS_DIR / "model_strategy_peak_vram_seed_summary.csv", index=False)

    peak_vram_ci = aggregate_single_metric(
        peak_vram_seed,
        ["model_label", "model_scale_b", "strategy", "strategy_label"],
        "required_vram_peak_gb",
    )
    peak_vram_ci.to_csv(RESULTS_DIR / "model_strategy_peak_vram_ci_summary.csv", index=False)

    representative = select_representative_seed(
        seed_summary=seed_summary,
        group_cols=["model_label", "model_scale_b"],
        rule_text=str(stat_config["representative_seed_rule"]),
    )
    representative.to_csv(RESULTS_DIR / "model_strategy_representative_seeds.csv", index=False)

    return frame, seed_summary, ci_summary


def main() -> None:
    ensure_dirs()
    base_configs = load_all_configs(BASE_DIR)
    stat_config = load_json_compatible_yaml(BASE_DIR / "configs" / "statistical_experiment.yaml")
    networks = load_networks(base_configs)

    _, _, _, main_overall_ci, significance = run_main_sweep(base_configs, stat_config, networks)

    _, _, arrival_ci = run_factor_sweep(
        base_configs=base_configs,
        stat_config=stat_config,
        networks=networks,
        factor_name="arrival_label",
        factor_values=stat_config["arrival_mean_ms"],
    )
    _, _, worker_ci = run_factor_sweep(
        base_configs=base_configs,
        stat_config=stat_config,
        networks=networks,
        factor_name="worker_label",
        factor_values=stat_config["cloud_parallelism"],
    )
    _, _, context_ci = run_factor_sweep(
        base_configs=base_configs,
        stat_config=stat_config,
        networks=networks,
        factor_name="context_label",
        factor_values=stat_config["context_targets"],
    )
    _, _, model_ci = run_model_strategy_sweep(base_configs, stat_config, networks)
    model_peak_vram = pd.read_csv(RESULTS_DIR / "model_strategy_peak_vram_ci_summary.csv")
    model_ci = model_ci.merge(
        model_peak_vram[["model_label", "strategy_label", "required_vram_peak_gb"]],
        on=["model_label", "strategy_label"],
        how="left",
    )

    representative = pd.read_csv(RESULTS_DIR / "representative_seeds.csv")
    write_markdown_report(
        stat_config=stat_config,
        main_overall_ci=main_overall_ci,
        significance=significance,
        arrival_ci=arrival_ci,
        worker_ci=worker_ci,
        context_ci=context_ci,
        model_ci=model_ci,
        representative=representative,
    )
    generate_statistical_figures(BASE_DIR)

    print("Statistical experiment upgrade completed.")
    print(RESULTS_DIR)


if __name__ == "__main__":
    main()
