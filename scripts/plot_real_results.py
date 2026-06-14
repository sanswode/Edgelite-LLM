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


STRATEGY_ORDER = ["Cloud-only", "Edge-only", "No-privacy", "Ours"]
STRATEGY_COLORS = {
    "Cloud-only": "#c94f4f",
    "Edge-only": "#4c78a8",
    "No-privacy": "#9c755f",
    "Ours": "#54a24b",
}


def save_figure(path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path_without_suffix.with_suffix(".svg"), format="svg")
    plt.savefig(path_without_suffix.with_suffix(".png"), dpi=180)
    plt.close()


def ordered_strategy_columns(pivot: pd.DataFrame) -> list[str]:
    return [label for label in STRATEGY_ORDER if label in pivot.columns]


def strategy_colors(columns: list[str]) -> list[str]:
    return [STRATEGY_COLORS.get(column, "#72b7b2") for column in columns]


def plot_grouped_metric(
    base_dir: Path,
    metric: str,
    filename: str,
    title: str,
    ylabel: str,
) -> None:
    frame = pd.read_csv(base_dir / "results" / "real_summary.csv")
    pivot = frame.pivot(index="network_label", columns="strategy_label", values=metric)
    columns = ordered_strategy_columns(pivot)
    if not columns:
        return
    pivot = pivot[columns]
    ax = pivot.plot(kind="bar", figsize=(10, 5), color=strategy_colors(columns))
    ax.set_title(title)
    ax.set_xlabel("Network Profile")
    ax.set_ylabel(ylabel)
    ax.legend(title="Strategy")
    save_figure(base_dir / "results" / "figures" / filename)


def plot_real_p95_latency(base_dir: Path) -> None:
    plot_grouped_metric(
        base_dir=base_dir,
        metric="p95_e2e_ms",
        filename="real_p95_latency_by_network",
        title="Real Run: P95 End-to-End Latency",
        ylabel="P95 E2E Latency (ms)",
    )


def plot_real_mean_e2e(base_dir: Path) -> None:
    plot_grouped_metric(
        base_dir=base_dir,
        metric="mean_e2e_ms",
        filename="real_mean_e2e_by_network",
        title="Real Run: Mean End-to-End Latency",
        ylabel="Mean E2E Latency (ms)",
    )


def plot_real_mean_ttft(base_dir: Path) -> None:
    plot_grouped_metric(
        base_dir=base_dir,
        metric="mean_ttft_ms",
        filename="real_mean_ttft_by_network",
        title="Real Run: Mean Time-to-First-Token",
        ylabel="Mean TTFT (ms)",
    )


def plot_real_bandwidth(base_dir: Path) -> None:
    plot_grouped_metric(
        base_dir=base_dir,
        metric="mean_bandwidth_bytes",
        filename="real_bandwidth_by_network",
        title="Real Run: Mean Transfer Bytes",
        ylabel="Mean Bandwidth Bytes",
    )


def plot_real_quality(base_dir: Path) -> None:
    plot_grouped_metric(
        base_dir=base_dir,
        metric="mean_quality",
        filename="real_quality_by_network",
        title="Real Run: Mean Response-Completeness Proxy",
        ylabel="Mean Quality Proxy",
    )


def plot_real_privacy_exposure(base_dir: Path) -> None:
    plot_grouped_metric(
        base_dir=base_dir,
        metric="privacy_exposure_rate",
        filename="real_privacy_exposure_by_network",
        title="Real Run: Privacy Exposure Rate",
        ylabel="Privacy Exposure Rate",
    )


def plot_real_cache_ratio(base_dir: Path) -> None:
    summary = pd.read_csv(base_dir / "results" / "real_summary.csv")
    raw = pd.read_csv(base_dir / "results" / "real_raw_logs.csv")

    ours_raw = raw[(raw["strategy"] == "ours") & (raw["status"] == "ok")].copy()
    hit_frame = (
        ours_raw.groupby("network_label", as_index=False)
        .agg(
            request_count=("request_id", "count"),
            cache_hits=("route", lambda values: int((values == "cache").sum())),
        )
    )
    hit_frame["cache_hit_ratio"] = hit_frame["cache_hits"] / hit_frame["request_count"]

    edge_summary = summary[summary["strategy"] == "edge_only"][["network_label", "mean_e2e_ms"]].rename(
        columns={"mean_e2e_ms": "edge_mean_e2e_ms"}
    )
    ours_summary = summary[summary["strategy"] == "ours"][["network_label", "mean_e2e_ms"]].rename(
        columns={"mean_e2e_ms": "ours_mean_e2e_ms"}
    )
    plot_frame = (
        hit_frame.merge(edge_summary, on="network_label", how="left")
        .merge(ours_summary, on="network_label", how="left")
        .sort_values("network_label")
    )
    plot_frame["latency_gain_pct"] = (
        (plot_frame["edge_mean_e2e_ms"] - plot_frame["ours_mean_e2e_ms"])
        / plot_frame["edge_mean_e2e_ms"]
        * 100.0
    )

    fig, ax1 = plt.subplots(figsize=(10, 5))
    bars = ax1.bar(
        plot_frame["network_label"],
        plot_frame["cache_hit_ratio"],
        color="#54a24b",
        width=0.58,
        label="Cache Hit Ratio (Ours)",
    )
    ax1.set_title(
        "Real Run: Cache Hits and Latency Gain\n"
        "Same 24-request replay per network profile; identical hit counts are expected.",
        pad=12,
    )
    ax1.set_xlabel("Network Profile")
    ax1.set_ylabel("Cache Hit Ratio")
    ax1.set_ylim(0.0, max(0.18, plot_frame["cache_hit_ratio"].max() + 0.04))

    for bar, row in zip(bars, plot_frame.itertuples(index=False)):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() - 0.01,
            f"{int(row.cache_hits)}/{int(row.request_count)}",
            ha="center",
            va="top",
            fontsize=10,
            color="white",
            fontweight="bold",
        )

    ax2 = ax1.twinx()
    ax2.plot(
        plot_frame["network_label"],
        plot_frame["latency_gain_pct"],
        color="#f58518",
        marker="o",
        linewidth=2.2,
        label="Mean E2E Gain vs Edge-only",
    )
    ax2.set_ylabel("Mean E2E Gain (%)")
    ax2.set_ylim(0.0, max(16.0, plot_frame["latency_gain_pct"].max() + 1.5))

    lines = [bars, ax2.get_lines()[0]]
    labels = ["Cache Hit Ratio (Ours)", "Mean E2E Gain vs Edge-only"]
    ax1.legend(lines, labels, loc="upper left")
    save_figure(base_dir / "results" / "figures" / "real_cache_hit_ratio")


def plot_real_action_distribution(base_dir: Path) -> None:
    frame = pd.read_csv(base_dir / "results" / "real_action_distribution.csv")
    if frame.empty:
        return
    pivot = frame.pivot(index="network_label", columns="route_label", values="ratio").fillna(0.0)
    order = [label for label in ["Edge", "Cache", "Cloud"] if label in pivot.columns]
    pivot = pivot[order]
    ax = pivot.plot(kind="bar", stacked=True, figsize=(10, 5), color=["#4c78a8", "#54a24b", "#c94f4f"])
    ax.set_title("Real Run: Action Distribution")
    ax.set_xlabel("Network Profile")
    ax.set_ylabel("Action Ratio")
    ax.legend(title="Route", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(base_dir / "results" / "figures" / "real_action_distribution")


def plot_real_cloud_queue_wait(base_dir: Path) -> None:
    raw = pd.read_csv(base_dir / "results" / "real_raw_logs.csv")
    cloud = raw[(raw["route"] == "cloud") & (raw["status"] == "ok")]
    if cloud.empty:
        return
    grouped = (
        cloud.groupby(["network_label", "strategy_label"], as_index=False)["queue_wait_ms"]
        .mean()
    )
    pivot = grouped.pivot(index="network_label", columns="strategy_label", values="queue_wait_ms").fillna(0.0)
    columns = ordered_strategy_columns(pivot)
    pivot = pivot[columns]
    ax = pivot.plot(kind="bar", figsize=(10, 5), color=strategy_colors(columns))
    ax.set_title("Real Run: Mean Cloud Queue Wait")
    ax.set_xlabel("Network Profile")
    ax.set_ylabel("Queue Wait (ms)")
    ax.legend(title="Strategy")
    save_figure(base_dir / "results" / "figures" / "real_cloud_queue_wait")


def generate_real_figures(base_dir: Path | None = None) -> None:
    base_dir = BASE_DIR if base_dir is None else base_dir
    plt.style.use("seaborn-v0_8-whitegrid")
    plot_real_p95_latency(base_dir)
    plot_real_mean_e2e(base_dir)
    plot_real_mean_ttft(base_dir)
    plot_real_bandwidth(base_dir)
    plot_real_quality(base_dir)
    plot_real_privacy_exposure(base_dir)
    plot_real_cache_ratio(base_dir)
    plot_real_action_distribution(base_dir)
    plot_real_cloud_queue_wait(base_dir)


if __name__ == "__main__":
    generate_real_figures()
