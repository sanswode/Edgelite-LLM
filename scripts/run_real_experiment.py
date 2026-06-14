from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from client.replay_requests import load_requests
from server.config import load_all_configs, load_json_compatible_yaml
from server.metrics import STRATEGY_LABELS, build_action_distribution, dataframe_to_markdown, summarize_by_group
from server.openai_runtime import BackendConfig, OpenAICompatibleBackend
from server.privacy_gate import score_prompt
from server.semantic_cache import SemanticCache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-model experiments against OpenAI-compatible local servers.")
    parser.add_argument("--dry-run", action="store_true", help="Only validate config and backend reachability.")
    parser.add_argument("--dataset", default="prompts", choices=["prompts", "privacy_prompts"], help="Which dataset to replay.")
    parser.add_argument("--limit", type=int, default=None, help="Override subset size.")
    return parser.parse_args()


def load_backend(config: Dict[str, object], name: str, timeout_s: int) -> OpenAICompatibleBackend | None:
    backend_cfg = config.get(name, {})
    if not backend_cfg or not backend_cfg.get("enabled", False):
        return None
    return OpenAICompatibleBackend(
        BackendConfig(
            name=name,
            backend=str(backend_cfg["backend"]),
            base_url=str(backend_cfg["base_url"]),
            model=str(backend_cfg["model"]),
            api_key=str(backend_cfg["api_key"]),
            label=str(backend_cfg["label"]),
            enabled=bool(backend_cfg["enabled"]),
        ),
        timeout_s=timeout_s,
    )


def choose_request_subset(requests: List, subset_size: int, seed: int) -> List:
    rng = random.Random(seed)
    if len(requests) <= subset_size:
        return list(requests)

    long_requests = [request for request in requests if request.category == "long_context"]
    privacy_requests = [request for request in requests if request.category == "privacy"]
    others = [request for request in requests if request.category not in {"long_context", "privacy"}]

    chosen: List = []
    chosen.extend(rng.sample(long_requests, min(max(12, subset_size // 6), len(long_requests))))
    chosen.extend(rng.sample(privacy_requests, min(max(16, subset_size // 5), len(privacy_requests))))
    remaining = max(0, subset_size - len(chosen))
    if remaining:
        chosen.extend(rng.sample(others, min(remaining, len(others))))
    rng.shuffle(chosen)
    return chosen[:subset_size]


def sleep_for_profile(profile: Dict[str, object], payload_bytes: int, phase: str) -> None:
    rtt_ms = float(profile["rtt_ms"])
    bandwidth_mbps = float(profile["bandwidth_mbps"])
    loss_pct = float(profile["loss_pct"])
    jitter_ms = float(profile["jitter_ms"])
    bytes_per_second = bandwidth_mbps * 1024 * 1024 / 8.0
    transfer_s = payload_bytes / max(bytes_per_second, 1.0)
    if phase == "before":
        sleep_s = rtt_ms / 2000.0 + transfer_s
    else:
        sleep_s = rtt_ms / 2000.0 + transfer_s + loss_pct / 200.0
    sleep_s += jitter_ms / 1000.0 * 0.08
    if sleep_s > 0:
        time.sleep(min(sleep_s, 2.0))


def select_route(strategy: str, request, privacy_score: float, semantic_hit: bool, cloud_available: bool, profile_name: str) -> str:
    if strategy == "edge_only":
        return "edge"
    if strategy == "cloud_only":
        return "cloud"
    if semantic_hit and privacy_score < 0.86:
        return "cache"
    if privacy_score >= 0.68:
        return "edge"
    if not cloud_available:
        return "edge"
    if profile_name in {"weak", "congested"} or request.prompt_tokens > 1600 or request.category == "long_context":
        return "edge"
    return "cloud"


def run_real() -> None:
    args = parse_args()
    configs = load_all_configs(BASE_DIR)
    real_config = load_json_compatible_yaml(BASE_DIR / "configs" / "real_runtime.yaml")
    request_cfg = real_config["request"]

    dataset_name = "privacy_prompts" if args.dataset == "privacy_prompts" else "prompts"
    request_path = BASE_DIR / "data" / f"{dataset_name}.jsonl"
    requests = load_requests(request_path)
    subset_size = args.limit or int(request_cfg["subset_size"])
    requests = choose_request_subset(requests, subset_size=subset_size, seed=int(request_cfg["seed"]))

    timeout_s = int(request_cfg["timeout_s"])
    edge_backend = load_backend(real_config, "edge", timeout_s)
    cloud_backend = load_backend(real_config, "cloud", timeout_s)

    if edge_backend is None:
        raise RuntimeError("Edge backend is not enabled in configs/real_runtime.yaml")

    network_profiles = {
        profile["name"]: profile
        for profile in configs["network_profiles"]["profiles"]
    }

    backends = {"edge": edge_backend, "cloud": cloud_backend}
    health_rows = []
    for name, backend in backends.items():
        if backend is None:
            continue
        ok, message = backend.healthcheck()
        health_rows.append({"backend": name, "ok": ok, "message": message, "base_url": backend.config.base_url})
    health_frame = pd.DataFrame(health_rows)
    health_frame.to_csv(BASE_DIR / "results" / "real_backend_health.csv", index=False)

    if args.dry_run:
        print("Dry run complete.")
        print(health_frame.to_string(index=False))
        return

    if not health_rows or not any(row["ok"] for row in health_rows):
        raise RuntimeError("No reachable backend detected. Start llama-server or vLLM first, then rerun.")

    warmup_requests = requests[: int(request_cfg["warmup_size"])]
    for backend in [edge_backend, cloud_backend]:
        if backend is None:
            continue
        ok, _ = backend.healthcheck()
        if not ok:
            continue
        for request in warmup_requests:
            backend.chat_completion(
                prompt=request.prompt,
                temperature=float(request_cfg["temperature"]),
                max_tokens=min(64, int(request_cfg["max_tokens"])),
            )

    semantic_cache = SemanticCache(capacity=96)
    rows: List[Dict[str, object]] = []
    cloud_ok = cloud_backend is not None and cloud_backend.healthcheck()[0]
    strategies = [strategy for strategy in real_config["strategies"] if strategy != "cloud_only" or cloud_ok]
    if not strategies:
        raise RuntimeError("No runnable strategies remain after backend availability filtering.")
    semantic_threshold = float(request_cfg["semantic_threshold"])

    for profile_name in ["wifi_good", "mobile_4g5g", "congested", "weak"]:
        profile = network_profiles[profile_name]
        for strategy in strategies:
            semantic_cache = SemanticCache(capacity=96)
            for request in requests:
                privacy_score, features = score_prompt(request.prompt)
                cache_entry, similarity = semantic_cache.lookup(request.prompt, semantic_threshold)
                semantic_hit = cache_entry is not None and request.category != "long_context"
                route = select_route(
                    strategy=strategy,
                    request=request,
                    privacy_score=privacy_score,
                    semantic_hit=semantic_hit,
                    cloud_available=cloud_ok,
                    profile_name=profile_name,
                )

                if route == "cloud" and not cloud_ok:
                    route = "edge"

                start_total = time.perf_counter()
                if route == "cache" and semantic_hit:
                    response_text = cache_entry.prompt_text
                    elapsed_ms = (time.perf_counter() - start_total) * 1000.0 + 12.0
                    row = {
                        "network": profile_name,
                        "network_label": profile["label"],
                        "strategy": strategy,
                        "strategy_label": STRATEGY_LABELS.get(strategy, strategy),
                        "route": route,
                        "backend": "cache",
                        "request_id": request.request_id,
                        "category": request.category,
                        "prompt_tokens": request.prompt_tokens,
                        "privacy_score": privacy_score,
                        "semantic_hit": 1,
                        "similarity": similarity,
                        "ttft_ms": max(8.0, elapsed_ms * 0.45),
                        "e2e_ms": max(15.0, elapsed_ms),
                        "throughput_tps": 999.0,
                        "output_tokens_est": len(response_text) // 4 if response_text else 0,
                        "bandwidth_bytes": 0,
                        "status": "ok",
                        "status_code": 200,
                        "error": "",
                        "privacy_exposed": 0,
                        "n_id": features["n_id"],
                        "n_loc": features["n_loc"],
                        "n_contact": features["n_contact"],
                        "n_domain": features["n_domain"],
                        "n_name": features["n_name"],
                    }
                    rows.append(row)
                    continue

                backend = backends.get(route)
                if backend is None:
                    continue

                payload_bytes = len(request.prompt.encode("utf-8"))
                if route == "cloud":
                    sleep_for_profile(profile, payload_bytes, "before")

                result = backend.chat_completion(
                    prompt=request.prompt,
                    temperature=float(request_cfg["temperature"]),
                    max_tokens=int(request_cfg["max_tokens"]),
                )

                if route == "cloud":
                    sleep_for_profile(profile, len(result.response_text.encode("utf-8")), "after")
                    added_network_ms = (
                        profile["rtt_ms"] + payload_bytes / (max(profile["bandwidth_mbps"], 1.0) * 125.0)
                    )
                else:
                    added_network_ms = 0.0

                effective_ttft_ms = result.ttft_ms + (profile["rtt_ms"] * 0.75 if route == "cloud" else 3.0)
                effective_e2e_ms = result.e2e_ms + added_network_ms
                privacy_exposed = int(route == "cloud" and privacy_score >= float(request_cfg["privacy_threshold"]))

                if result.status == "ok":
                    semantic_cache.store(
                        cache_key=request.request_id,
                        prompt=request.prompt,
                        response_route=route,
                        quality_score=1.0,
                    )

                rows.append(
                    {
                        "network": profile_name,
                        "network_label": profile["label"],
                        "strategy": strategy,
                        "strategy_label": STRATEGY_LABELS.get(strategy, strategy),
                        "route": route,
                        "backend": result.backend,
                        "request_id": request.request_id,
                        "category": request.category,
                        "prompt_tokens": request.prompt_tokens,
                        "privacy_score": privacy_score,
                        "semantic_hit": 0,
                        "similarity": similarity,
                        "ttft_ms": effective_ttft_ms,
                        "e2e_ms": effective_e2e_ms,
                        "throughput_tps": result.throughput_tps,
                        "output_tokens_est": result.output_tokens_est,
                        "bandwidth_bytes": result.bandwidth_bytes,
                        "status": result.status,
                        "status_code": result.status_code,
                        "error": result.error,
                        "privacy_exposed": privacy_exposed,
                        "n_id": features["n_id"],
                        "n_loc": features["n_loc"],
                        "n_contact": features["n_contact"],
                        "n_domain": features["n_domain"],
                        "n_name": features["n_name"],
                    }
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(BASE_DIR / "results" / "real_raw_logs.csv", index=False)

    ok_frame = frame[frame["status"] == "ok"].copy()
    if ok_frame.empty:
        raise RuntimeError("Real experiment finished but no successful requests were recorded.")

    summary = summarize_by_group(ok_frame, ["network", "network_label", "strategy", "strategy_label"])
    summary.to_csv(BASE_DIR / "results" / "real_summary.csv", index=False)

    action_distribution = build_action_distribution(ok_frame.rename(columns={"route": "route"}))
    action_distribution.to_csv(BASE_DIR / "results" / "real_action_distribution.csv", index=False)

    report = [
        "# Real Experiment Report",
        "",
        "## Backend Health",
        "",
        dataframe_to_markdown(health_frame, []),
        "",
        "## Summary",
        "",
        dataframe_to_markdown(
            summary,
            [
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
            ],
        ),
    ]
    (BASE_DIR / "results" / "real_report.md").write_text("\n".join(report), encoding="utf-8")

    print("Real experiment completed.")
    print(BASE_DIR / "results" / "real_summary.csv")


if __name__ == "__main__":
    run_real()
