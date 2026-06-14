from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


RESULTS_DIR = BASE_DIR / "results" / "statistical"
FIGURES_DIR = RESULTS_DIR / "figures"

STRATEGY_ORDER = ["Cloud-only", "Edge-only", "Static-threshold", "Ours"]
STRATEGY_COLORS = {
    "Cloud-only": "#c94f4f",
    "Edge-only": "#4c78a8",
    "Static-threshold": "#72b7b2",
    "Ours": "#54a24b",
}

NETWORK_ORDER = ["WiFi Good", "4G/5G", "Congested", "Weak"]
ARRIVAL_ORDER = ["arrival_high", "arrival_medium", "arrival_low"]
ARRIVAL_LABELS = {
    "arrival_high": "High rate",
    "arrival_medium": "Medium rate",
    "arrival_low": "Low rate",
}
WORKER_ORDER = ["workers_4", "workers_8", "workers_12"]
WORKER_LABELS = {
    "workers_4": "4 workers",
    "workers_8": "8 workers",
    "workers_12": "12 workers",
}
CONTEXT_ORDER = ["ctx_1024", "ctx_2048", "ctx_4096"]
CONTEXT_LABELS = {
    "ctx_1024": "1K ctx",
    "ctx_2048": "2K ctx",
    "ctx_4096": "4K ctx",
}
MODEL_ORDER = [
    "Qwen2.5-0.5B",
    "Llama-3.2-1B",
    "TinyLlama-1.1B",
    "Gemma-2-2B",
    "Qwen2.5-3B",
]


def save_figure(path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path_without_suffix.with_suffix(".svg"), format="svg")
    plt.savefig(path_without_suffix.with_suffix(".png"), dpi=180)
    plt.close()


def strategy_color(label: str) -> str:
    return STRATEGY_COLORS.get(label, "#999999")


def plot_main_errorbars(base_dir: Path) -> None:
    frame = pd.read_csv(base_dir / "results" / "statistical" / "main_ci_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    metrics = [
        ("mean_e2e_ms", "mean_e2e_ms_ci95", "Mean E2E Latency (ms)"),
        ("p95_e2e_ms", "p95_e2e_ms_ci95", "P95 E2E Latency (ms)"),
    ]

    x = np.arange(len(NETWORK_ORDER))
    width = 0.18

    for ax, (metric, ci_metric, ylabel) in zip(axes, metrics):
        for index, strategy in enumerate(STRATEGY_ORDER):
            part = (
                frame[frame["strategy_label"] == strategy]
                .set_index("network_label")
                .reindex(NETWORK_ORDER)
            )
            values = part[metric].to_numpy(dtype=float)
            errors = part[ci_metric].to_numpy(dtype=float)
            ax.bar(
                x + (index - 1.5) * width,
                values,
                width=width,
                yerr=errors,
                capsize=4,
                color=strategy_color(strategy),
                label=strategy,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(NETWORK_ORDER)
        ax.set_xlabel("Network Profile")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)

    axes[0].legend(title="Strategy", ncols=2)
    fig.suptitle("Main Comparison with 95% Confidence Intervals", fontsize=14, y=1.02)
    save_figure(base_dir / "results" / "statistical" / "figures" / "main_latency_errorbars")


def plot_parameter_sweeps(base_dir: Path) -> None:
    figure_specs = [
        (
            "arrival_label_ci_summary.csv",
            "arrival_label",
            ARRIVAL_ORDER,
            ARRIVAL_LABELS,
            "Arrival Rate Sweep",
        ),
        (
            "worker_label_ci_summary.csv",
            "worker_label",
            WORKER_ORDER,
            WORKER_LABELS,
            "Cloud Worker Sweep",
        ),
        (
            "context_label_ci_summary.csv",
            "context_label",
            CONTEXT_ORDER,
            CONTEXT_LABELS,
            "Context Length Sweep",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (filename, column, order, label_map, title) in zip(axes, figure_specs):
        frame = pd.read_csv(base_dir / "results" / "statistical" / filename)
        x = np.arange(len(order))
        for strategy in STRATEGY_ORDER:
            part = frame[frame["strategy_label"] == strategy].set_index(column).reindex(order)
            ax.errorbar(
                x,
                part["mean_e2e_ms"].to_numpy(dtype=float),
                yerr=part["mean_e2e_ms_ci95"].to_numpy(dtype=float),
                marker="o",
                linewidth=2.0,
                capsize=4,
                color=strategy_color(strategy),
                label=strategy,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([label_map[label] for label in order])
        ax.set_title(title)
        ax.set_xlabel("Setting")
        ax.set_ylabel("Mean E2E (ms)")

    axes[0].legend(title="Strategy", fontsize=9)
    fig.suptitle("Parameter Sweeps with 95% Confidence Intervals", fontsize=14, y=1.02)
    save_figure(base_dir / "results" / "statistical" / "figures" / "parameter_sweep_errorbars")


def plot_significance_heatmap(base_dir: Path) -> None:
    frame = pd.read_csv(base_dir / "results" / "statistical" / "main_significance.csv")
    frame = frame[frame["metric"] == "mean_e2e_ms"].copy()
    baseline_order = ["Cloud-only", "Edge-only", "Static-threshold"]
    frame = frame[frame["baseline_label"].isin(baseline_order)]
    value_pivot = frame.pivot(index="network_label", columns="baseline_label", values="improvement_pct").reindex(
        index=NETWORK_ORDER,
        columns=baseline_order,
    )
    pvalue_pivot = frame.pivot(index="network_label", columns="baseline_label", values="p_value").reindex(
        index=NETWORK_ORDER,
        columns=baseline_order,
    )
    star_pivot = frame.pivot(index="network_label", columns="baseline_label", values="significance").reindex(
        index=NETWORK_ORDER,
        columns=baseline_order,
    )

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    heatmap = ax.imshow(value_pivot.to_numpy(dtype=float), cmap="YlGn", aspect="auto")
    ax.set_xticks(np.arange(len(baseline_order)))
    ax.set_xticklabels(baseline_order)
    ax.set_yticks(np.arange(len(NETWORK_ORDER)))
    ax.set_yticklabels(NETWORK_ORDER)
    ax.set_title("Ours vs Baselines: Mean E2E Gain and Paired t-test")

    for row in range(len(NETWORK_ORDER)):
        for col in range(len(baseline_order)):
            gain = value_pivot.iloc[row, col]
            p_value = pvalue_pivot.iloc[row, col]
            stars = star_pivot.iloc[row, col]
            p_text = "<0.001" if p_value < 0.001 else f"{p_value:.3f}"
            ax.text(
                col,
                row,
                f"{gain:.1f}%\n{stars}\np={p_text}",
                ha="center",
                va="center",
                fontsize=9,
                color="#143642",
            )

    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label("Improvement over Baseline (%)")
    save_figure(base_dir / "results" / "statistical" / "figures" / "significance_heatmap")


def plot_representative_action_distribution(base_dir: Path) -> None:
    frame = pd.read_csv(base_dir / "results" / "statistical" / "representative_action_distribution.csv")
    if frame.empty:
        return
    pivot = frame.pivot(index="network_label", columns="route_label", values="ratio").fillna(0.0)
    route_order = [label for label in ["Local", "Edge", "Cloud", "Cache", "Draft->Edge"] if label in pivot.columns]
    pivot = pivot.reindex(index=NETWORK_ORDER, columns=route_order, fill_value=0.0)
    colors = {
        "Local": "#72b7b2",
        "Edge": "#4c78a8",
        "Cloud": "#c94f4f",
        "Cache": "#54a24b",
        "Draft->Edge": "#f58518",
    }
    ax = pivot.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 5),
        color=[colors[label] for label in route_order],
    )
    ax.set_title("Representative Ours Seed: Action Distribution by Network")
    ax.set_xlabel("Network Profile")
    ax.set_ylabel("Action Ratio")
    ax.legend(title="Route", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(base_dir / "results" / "statistical" / "figures" / "representative_action_distribution")


def plot_model_strategy_comparison(base_dir: Path) -> None:
    frame = pd.read_csv(base_dir / "results" / "statistical" / "model_strategy_ci_summary.csv")
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    metrics = [
        ("mean_e2e_ms", "mean_e2e_ms_ci95", "Mean E2E Latency (ms)"),
        ("mean_quality", "mean_quality_ci95", "Quality Proxy"),
    ]
    x = np.arange(len(MODEL_ORDER))
    width = 0.18

    for ax, (metric, ci_metric, ylabel) in zip(axes, metrics):
        for index, strategy in enumerate(STRATEGY_ORDER):
            part = (
                frame[frame["strategy_label"] == strategy]
                .set_index("model_label")
                .reindex(MODEL_ORDER)
            )
            ax.bar(
                x + (index - 1.5) * width,
                part[metric].to_numpy(dtype=float),
                width=width,
                yerr=part[ci_metric].to_numpy(dtype=float),
                capsize=4,
                color=strategy_color(strategy),
                label=strategy,
            )
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(MODEL_ORDER, rotation=15)
    axes[1].set_xlabel("Edge Model")
    axes[0].legend(title="Strategy", ncols=2)
    fig.suptitle("Cross-Model Comparison: Four Strategies with 95% Confidence Intervals", fontsize=14, y=1.01)
    save_figure(base_dir / "results" / "statistical" / "figures" / "model_strategy_comparison")


def plot_model_tradeoff(base_dir: Path) -> None:
    frame = pd.read_csv(base_dir / "results" / "statistical" / "model_strategy_ci_summary.csv")
    peak_vram = pd.read_csv(base_dir / "results" / "statistical" / "model_strategy_peak_vram_ci_summary.csv")
    ours = frame[frame["strategy_label"] == "Ours"].copy()
    ours = ours.merge(
        peak_vram[["model_label", "strategy_label", "required_vram_peak_gb"]],
        on=["model_label", "strategy_label"],
        how="left",
    )
    ours["model_label"] = pd.Categorical(ours["model_label"], categories=MODEL_ORDER, ordered=True)
    ours = ours.sort_values("model_label")

    fig, ax = plt.subplots(figsize=(10.5, 6))
    max_vram = max(float(ours["required_vram_peak_gb"].max()), 1e-6)
    sizes = ours["required_vram_peak_gb"] / max_vram * 900 + 120
    scatter = ax.scatter(
        ours["model_scale_b"],
        ours["mean_e2e_ms"],
        s=sizes,
        c=ours["mean_quality"],
        cmap="viridis",
        alpha=0.84,
        edgecolors="#333333",
        linewidths=0.8,
    )

    for row in ours.itertuples(index=False):
        ax.annotate(
            row.model_label,
            (row.model_scale_b, row.mean_e2e_ms),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_title("Ours: Model Scale-Latency-Quality-VRAM Trade-off")
    ax.set_xlabel("Model Scale (B parameters)")
    ax.set_ylabel("Mean E2E Latency (ms)")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Quality Proxy")

    example_sizes = sorted(
        {
            round(float(ours["required_vram_peak_gb"].min()), 2),
            round(float(ours["required_vram_peak_gb"].median()), 2),
            round(float(ours["required_vram_peak_gb"].max()), 2),
        }
    )
    handles = [
        ax.scatter([], [], s=value / max_vram * 900 + 120, color="#999999", alpha=0.6, edgecolors="#333333")
        for value in example_sizes
    ]
    ax.legend(
        handles,
        [f"{value:.2f} GB VRAM" for value in example_sizes],
        title="Bubble Size",
        loc="lower right",
    )
    save_figure(base_dir / "results" / "statistical" / "figures" / "model_scale_tradeoff_statistical")


def generate_statistical_figures(base_dir: Path | None = None) -> None:
    base_dir = BASE_DIR if base_dir is None else base_dir
    plt.style.use("seaborn-v0_8-whitegrid")
    plot_main_errorbars(base_dir)
    plot_parameter_sweeps(base_dir)
    plot_significance_heatmap(base_dir)
    plot_representative_action_distribution(base_dir)
    plot_model_strategy_comparison(base_dir)
    plot_model_tradeoff(base_dir)


if __name__ == "__main__":
    generate_statistical_figures()
