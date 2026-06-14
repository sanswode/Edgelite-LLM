from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from server.metrics import ACTION_LABELS, STRATEGY_LABELS


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, format="svg")
    plt.savefig(path.with_suffix(".png"), dpi=180)
    plt.close()


def plot_p95_latency(base_dir: Path) -> None:
    frame = pd.read_csv(base_dir / "results" / "main_summary.csv")
    frame = frame[frame["strategy"].isin(["cloud_only", "edge_only", "static_threshold", "ours"])]
    pivot = frame.pivot(index="network_label", columns="strategy", values="p95_e2e_ms")
    pivot = pivot[["cloud_only", "edge_only", "static_threshold", "ours"]]
    pivot.columns = [STRATEGY_LABELS[column] for column in pivot.columns]
    ax = pivot.plot(kind="bar", figsize=(10, 5), color=["#c94f4f", "#4c78a8", "#72b7b2", "#54a24b"])
    ax.set_title("P95 End-to-End Latency Across Network Profiles")
    ax.set_xlabel("Network Profile")
    ax.set_ylabel("P95 E2E Latency (ms)")
    ax.legend(title="Strategy")
    save_figure(base_dir / "results" / "figures" / "p95_latency_by_network.svg")


def plot_cache_threshold(base_dir: Path) -> None:
    frame = pd.read_csv(base_dir / "results" / "semantic_threshold_sweep.csv")
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.plot(frame["semantic_threshold"], frame["cache_hit_ratio"], marker="o", linewidth=2.2, color="#4c78a8", label="Hit Ratio")
    ax2.plot(frame["semantic_threshold"], frame["latency_gain_pct"], marker="s", linewidth=2.2, color="#f58518", label="Latency Gain")

    ax1.set_title("Semantic Cache Threshold vs Hit Ratio and Latency Gain")
    ax1.set_xlabel("Semantic Threshold")
    ax1.set_ylabel("Hit Ratio")
    ax2.set_ylabel("Latency Gain over Cloud-only (%)")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")
    save_figure(base_dir / "results" / "figures" / "cache_hit_vs_latency_gain.svg")


def plot_privacy_exposure(base_dir: Path) -> None:
    frame = pd.read_csv(base_dir / "results" / "privacy_summary.csv")
    frame = frame[frame["strategy"].isin(["no_privacy", "ours"])]
    pivot = frame.pivot(index="network_label", columns="strategy", values="privacy_exposure_rate")
    pivot = pivot[["no_privacy", "ours"]]
    pivot.columns = [STRATEGY_LABELS[column] for column in pivot.columns]
    ax = pivot.plot(kind="bar", figsize=(9, 5), color=["#e45756", "#54a24b"])
    ax.set_title("Privacy Exposure Rate on Sensitive Prompts")
    ax.set_xlabel("Network Profile")
    ax.set_ylabel("Privacy Exposure Rate")
    ax.legend(title="Strategy")
    save_figure(base_dir / "results" / "figures" / "privacy_exposure.svg")


def plot_ablation(base_dir: Path) -> None:
    frame = pd.read_csv(base_dir / "results" / "ablation_summary.csv")
    frame["strategy_label"] = frame["strategy_label"].fillna(frame["strategy"])
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.bar(frame["strategy_label"], frame["mean_e2e_ms"], color="#4c78a8", alpha=0.8, label="Mean E2E")
    ax2.plot(frame["strategy_label"], frame["mean_quality"], color="#f58518", marker="o", linewidth=2, label="Mean Quality")

    ax1.set_title("Ablation Study: Latency and Quality")
    ax1.set_ylabel("Mean E2E Latency (ms)")
    ax2.set_ylabel("Mean Quality Score")
    ax1.set_xlabel("Ablation Setting")
    ax1.tick_params(axis="x", rotation=20)

    lines = [ax1.patches[0], ax2.get_lines()[0]]
    labels = ["Mean E2E", "Mean Quality"]
    ax1.legend(lines, labels, loc="upper left")
    save_figure(base_dir / "results" / "figures" / "ablation_latency_quality.svg")


def plot_action_distribution(base_dir: Path) -> None:
    frame = pd.read_csv(base_dir / "results" / "action_distribution.csv")
    pivot = frame.pivot(index="network_label", columns="route_label", values="ratio").fillna(0.0)
    desired = ["Local", "Edge", "Cloud", "Cache", "Draft->Edge"]
    pivot = pivot.reindex(columns=[name for name in desired if name in pivot.columns], fill_value=0.0)
    ax = pivot.plot(kind="bar", stacked=True, figsize=(10, 5), color=["#72b7b2", "#4c78a8", "#c94f4f", "#54a24b", "#f58518"])
    ax.set_title("Action Distribution of Ours Across Network Profiles")
    ax.set_xlabel("Network Profile")
    ax.set_ylabel("Action Ratio")
    ax.legend(title="Action", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(base_dir / "results" / "figures" / "action_distribution.svg")


def plot_vram_budget(base_dir: Path) -> None:
    frame = pd.read_csv(base_dir / "results" / "vram_budget.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    for model_name, part in frame.groupby("model_name"):
        ax.plot(part["context_tokens"], part["total_gb"], marker="o", linewidth=2, label=model_name)

    safe_limit = frame["safe_limit_gb"].iloc[0]
    ax.axhline(safe_limit, linestyle="--", color="#e45756", label=f"Safe Limit ({safe_limit:.2f} GB)")
    ax.set_title("Estimated VRAM Budget by Model Size and Context Length")
    ax.set_xlabel("Context Tokens")
    ax.set_ylabel("Estimated Total VRAM (GB)")
    ax.legend(loc="upper left", ncols=3)
    save_figure(base_dir / "results" / "figures" / "vram_budget.svg")


def generate_all_figures(base_dir: Path | None = None) -> None:
    base_dir = BASE_DIR if base_dir is None else base_dir
    plt.style.use("seaborn-v0_8-whitegrid")
    plot_p95_latency(base_dir)
    plot_cache_threshold(base_dir)
    plot_privacy_exposure(base_dir)
    plot_ablation(base_dir)
    plot_action_distribution(base_dir)
    plot_vram_budget(base_dir)


if __name__ == "__main__":
    generate_all_figures()
