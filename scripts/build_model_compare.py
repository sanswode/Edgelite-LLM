from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results" / "model_compare"


MODEL_ORDER = [
    "Qwen2.5-0.5B",
    "Llama-3.2-1B",
    "TinyLlama-1.1B",
    "Gemma-2-2B",
    "Qwen3-1.7B",
    "Qwen2.5-3B",
]

MODEL_COLORS = {
    "Qwen2.5-0.5B": "#72b7b2",
    "Llama-3.2-1B": "#4c78a8",
    "TinyLlama-1.1B": "#e45756",
    "Gemma-2-2B": "#b279a2",
    "Qwen3-1.7B": "#54a24b",
    "Qwen2.5-3B": "#f58518",
}

MODEL_SCALE_B = {
    "Qwen2.5-0.5B": 0.5,
    "Llama-3.2-1B": 1.0,
    "TinyLlama-1.1B": 1.1,
    "Gemma-2-2B": 2.0,
    "Qwen3-1.7B": 1.7,
    "Qwen2.5-3B": 3.0,
}


def save_figure(path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path_without_suffix.with_suffix(".svg"), format="svg")
    plt.savefig(path_without_suffix.with_suffix(".png"), dpi=180)
    plt.close()


def collect_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for summary_path in sorted(RESULTS_DIR.glob("*/real_summary.csv")):
        model_dir = summary_path.parent.name
        model_label = {
            "qwen2p5_0p5b": "Qwen2.5-0.5B",
            "llama_1b": "Llama-3.2-1B",
            "tinyllama_1p1b": "TinyLlama-1.1B",
            "gemma2_2b": "Gemma-2-2B",
            "qwen3_1p7b": "Qwen3-1.7B",
            "qwen2p5_3b": "Qwen2.5-3B",
        }.get(model_dir, model_dir)
        summary = pd.read_csv(summary_path)
        raw = pd.read_csv(summary_path.parent / "real_raw_logs.csv")
        ours = summary[summary["strategy"] == "ours"].copy()
        if ours.empty:
            continue
        rows.append(
            {
                "model_label": model_label,
                "model_scale_b": MODEL_SCALE_B.get(model_label, math.nan),
                "mean_ttft_ms": ours["mean_ttft_ms"].mean(),
                "mean_e2e_ms": ours["mean_e2e_ms"].mean(),
                "p95_e2e_ms": ours["p95_e2e_ms"].mean(),
                "mean_bandwidth_bytes": ours["mean_bandwidth_bytes"].mean(),
                "mean_vram_peak_gb": ours["mean_vram_peak_gb"].mean(),
                "privacy_exposure_rate": ours["privacy_exposure_rate"].mean(),
                "mean_quality": ours["mean_quality"].mean(),
                "cache_hit_ratio": ours["cache_hit_ratio"].mean(),
                "cloud_route_ratio": float((raw["route"] == "cloud").mean()),
                "edge_route_ratio": float((raw["route"] == "edge").mean()),
                "cache_route_ratio": float((raw["route"] == "cache").mean()),
            }
        )
    return rows


def build_summary() -> pd.DataFrame:
    frame = pd.DataFrame(collect_rows())
    if frame.empty:
        raise SystemExit("No model comparison results found under results/model_compare/*/real_summary.csv")
    frame["model_label"] = pd.Categorical(frame["model_label"], categories=MODEL_ORDER, ordered=True)
    frame = frame.sort_values("model_label").reset_index(drop=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS_DIR / "model_compare_summary.csv", index=False)
    return frame


def plot_overview(frame: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [
        ("mean_e2e_ms", "Mean E2E (ms)"),
        ("p95_e2e_ms", "Mean P95 (ms)"),
        ("mean_quality", "Quality Proxy"),
        ("mean_vram_peak_gb", "VRAM Peak (GB)"),
    ]

    for ax, (metric, ylabel) in zip(axes.flatten(), metrics):
        colors = [MODEL_COLORS.get(model, "#72b7b2") for model in frame["model_label"]]
        bars = ax.bar(frame["model_label"], frame[metric], color=colors, width=0.62)
        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Edge Model")
        for bar, value in zip(bars, frame[metric]):
            label = f"{value:.3f}" if value < 10 else f"{value:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), label, ha="center", va="bottom", fontsize=9)

    fig.suptitle("Ours Strategy: Cross-Model Comparison on Real Edge Runs", fontsize=14, y=0.98)
    save_figure(RESULTS_DIR / "figures" / "model_compare_overview")


def plot_routes(frame: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    order = frame["model_label"].tolist()
    plot_frame = frame.set_index("model_label")[["edge_route_ratio", "cache_route_ratio", "cloud_route_ratio"]]
    ax = plot_frame.loc[order].plot(
        kind="bar",
        stacked=True,
        figsize=(10, 5),
        color=["#4c78a8", "#54a24b", "#c94f4f"],
    )
    ax.set_title("Ours Strategy: Route Distribution by Edge Model")
    ax.set_xlabel("Edge Model")
    ax.set_ylabel("Route Ratio")
    ax.legend(["Edge", "Cache", "Cloud"], title="Route", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(RESULTS_DIR / "figures" / "model_compare_routes")


def plot_tradeoff(frame: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plot_frame = frame.dropna(subset=["model_scale_b"]).copy()
    fig, ax = plt.subplots(figsize=(10.5, 6))

    size_scale = 900
    sizes = plot_frame["mean_vram_peak_gb"] / plot_frame["mean_vram_peak_gb"].max() * size_scale + 120
    scatter = ax.scatter(
        plot_frame["model_scale_b"],
        plot_frame["mean_e2e_ms"],
        s=sizes,
        c=plot_frame["mean_quality"],
        cmap="viridis",
        alpha=0.82,
        edgecolors="#333333",
        linewidths=0.8,
    )

    for row in plot_frame.itertuples(index=False):
        ax.annotate(
            row.model_label,
            (row.model_scale_b, row.mean_e2e_ms),
            xytext=(7, 6),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_title("Ours Strategy: Model Scale-Latency-Quality-VRAM Trade-off")
    ax.set_xlabel("Model Scale (B parameters, nominal)")
    ax.set_ylabel("Mean End-to-End Latency (ms)")

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Quality Proxy")

    legend_sizes = [2.3, 2.8, 3.5]
    handles = [
        ax.scatter([], [], s=value / plot_frame["mean_vram_peak_gb"].max() * size_scale + 120, color="#999999", alpha=0.6, edgecolors="#333333")
        for value in legend_sizes
    ]
    labels = [f"{value:.1f} GB VRAM" for value in legend_sizes]
    ax.legend(handles, labels, title="Bubble Size", loc="upper left")

    save_figure(RESULTS_DIR / "figures" / "model_scale_tradeoff")


def main() -> None:
    frame = build_summary()
    plot_overview(frame)
    plot_routes(frame)
    plot_tradeoff(frame)
    print(RESULTS_DIR / "model_compare_summary.csv")
    print(RESULTS_DIR / "figures")


if __name__ == "__main__":
    main()
