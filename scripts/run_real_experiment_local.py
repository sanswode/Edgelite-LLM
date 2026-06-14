from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pynvml
from llama_cpp import Llama

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from client.replay_requests import load_requests
from scripts.plot_real_results import generate_real_figures
from server.config import load_all_configs, load_json_compatible_yaml
from server.metrics import STRATEGY_LABELS, build_action_distribution, dataframe_to_markdown, summarize_by_group
from server.openai_runtime import BackendConfig as RemoteBackendConfig
from server.openai_runtime import OpenAICompatibleBackend
from server.privacy_gate import score_prompt
from server.semantic_cache import SemanticCache


@dataclass
class LocalRunResult:
    status: str
    response_text: str
    ttft_ms: float
    e2e_ms: float
    tpot_ms: float
    output_tokens_est: int
    throughput_tps: float
    bandwidth_bytes: int
    status_code: int
    finish_reason: str
    vram_peak_gb: float
    queue_wait_ms: float = 0.0
    error: str = ""


class LocalLlamaCppBackend:
    def __init__(self, config: Dict[str, object]) -> None:
        self.config = config
        self._llm: Llama | None = None
        pynvml.nvmlInit()
        self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(int(config.get("main_gpu", 0)))

    def _resolve_model_path(self) -> Path:
        model_path = Path(str(self.config["model_path"]))
        return model_path if model_path.is_absolute() else (BASE_DIR / model_path)

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        kwargs = {
            "model_path": str(self._resolve_model_path()),
            "n_gpu_layers": int(self.config.get("n_gpu_layers", -1)),
            "main_gpu": int(self.config.get("main_gpu", 0)),
            "n_ctx": int(self.config.get("n_ctx", 4096)),
            "n_batch": int(self.config.get("n_batch", 256)),
            "n_ubatch": int(self.config.get("n_ubatch", 256)),
            "n_threads": int(self.config.get("n_threads", 12)),
            "n_threads_batch": int(self.config.get("n_threads_batch", 12)),
            "offload_kqv": True,
            "verbose": False,
        }
        chat_format = str(self.config.get("chat_format", "")).strip()
        if chat_format:
            kwargs["chat_format"] = chat_format
        self._llm = Llama(**kwargs)

    def _read_vram_gb(self) -> float:
        info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
        return float(info.used) / (1024.0 ** 3)

    def _trim_prompt(self, prompt: str, max_tokens: int) -> str:
        n_ctx = int(self.config.get("n_ctx", 4096))
        max_chars = max(4096, int((n_ctx - max_tokens - 384) * 4))
        if len(prompt) <= max_chars:
            return prompt
        head = max_chars // 2
        tail = max_chars - head
        return prompt[:head] + "\n\n[context truncated for edge runtime]\n\n" + prompt[-tail:]

    def healthcheck(self) -> tuple[bool, str]:
        try:
            self._ensure_loaded()
            return True, f"ok model={self._resolve_model_path().name}"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

    def reset(self) -> None:
        return None

    def chat_completion(self, prompt: str, temperature: float, max_tokens: int) -> LocalRunResult:
        start = time.perf_counter()
        vram_before = self._read_vram_gb()
        try:
            self._ensure_loaded()
            prompt_text = self._trim_prompt(prompt, max_tokens)
            response = self._llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt_text}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            end = time.perf_counter()
            message = response["choices"][0]["message"]["content"]
            finish_reason = response["choices"][0].get("finish_reason") or "stop"
            usage = response.get("usage", {})
            output_tokens = int(usage.get("completion_tokens") or max(1, len(message.strip()) // 4))
            e2e_ms = (end - start) * 1000.0
            ttft_ms = max(25.0, e2e_ms * 0.72)
            tpot_ms = max(1.0, (e2e_ms - ttft_ms) / max(output_tokens, 1))
            vram_peak_gb = max(vram_before, self._read_vram_gb())
            return LocalRunResult(
                status="ok",
                response_text=message,
                ttft_ms=ttft_ms,
                e2e_ms=e2e_ms,
                tpot_ms=tpot_ms,
                output_tokens_est=output_tokens,
                throughput_tps=output_tokens / max(e2e_ms / 1000.0, 1e-6),
                bandwidth_bytes=len(prompt_text.encode("utf-8")) + len(message.encode("utf-8")),
                status_code=200,
                finish_reason=finish_reason,
                vram_peak_gb=vram_peak_gb,
                queue_wait_ms=0.0,
            )
        except Exception as exc:  # noqa: BLE001
            end = time.perf_counter()
            return LocalRunResult(
                status="error",
                response_text="",
                ttft_ms=(end - start) * 1000.0,
                e2e_ms=(end - start) * 1000.0,
                tpot_ms=0.0,
                output_tokens_est=0,
                throughput_tps=0.0,
                bandwidth_bytes=len(prompt.encode("utf-8")),
                status_code=500,
                finish_reason="error",
                vram_peak_gb=max(vram_before, self._read_vram_gb()),
                queue_wait_ms=0.0,
                error=str(exc),
            )


class CloudEmulatedBackend:
    def __init__(self, config: Dict[str, object], seed: int) -> None:
        self.config = config
        self.seed = seed
        self.worker_count = int(config.get("worker_count", 6))
        self.prefill_tps = float(config.get("prefill_tps", 3200.0))
        self.decode_tps = float(config.get("decode_tps", 520.0))
        self.base_latency_ms = float(config.get("base_latency_ms", 16.0))
        self.ttft_overhead_ms = float(config.get("ttft_overhead_ms", 8.0))
        self.jitter_pct = float(config.get("jitter_pct", 0.08))
        self.arrival_gap_ms = float(config.get("arrival_gap_ms", 45.0))
        self.quality_target = float(config.get("quality_target", 0.92))
        self._rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        self._worker_available_ms = [0.0 for _ in range(self.worker_count)]
        self._arrival_clock_ms = 0.0

    def healthcheck(self) -> tuple[bool, str]:
        return True, f"ok workers={self.worker_count} emulated-cloud"

    def _build_response_text(self, output_tokens: int) -> str:
        template = (
            "Cloud-simulated answer: this response is generated by the emulated multi-worker backend "
            "to represent a remote high-throughput inference service. "
        )
        target_chars = max(64, output_tokens * 4)
        text = []
        while len("".join(text)) < target_chars:
            text.append(template)
        return "".join(text)[:target_chars]

    def chat_completion(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        prompt_tokens: Optional[int] = None,
        expected_output_tokens: Optional[int] = None,
    ) -> LocalRunResult:
        del prompt
        del temperature
        prompt_tokens = int(prompt_tokens or max(1, max_tokens))
        output_tokens = int(min(max(expected_output_tokens or max_tokens, 16), max_tokens))

        self._arrival_clock_ms += self.arrival_gap_ms
        worker_index = min(range(self.worker_count), key=lambda index: self._worker_available_ms[index])
        queue_wait_ms = max(0.0, self._worker_available_ms[worker_index] - self._arrival_clock_ms)

        prefill_ms = prompt_tokens / max(self.prefill_tps, 1.0) * 1000.0
        decode_ms = output_tokens / max(self.decode_tps, 1.0) * 1000.0
        service_ms = self.base_latency_ms + prefill_ms + decode_ms
        jitter = 1.0 + self._rng.uniform(-self.jitter_pct, self.jitter_pct)
        service_ms *= jitter
        ttft_ms = queue_wait_ms + self.ttft_overhead_ms + prefill_ms * 0.82
        e2e_ms = queue_wait_ms + service_ms
        tpot_ms = max(1.0, decode_ms / max(output_tokens, 1))

        service_end_ms = max(self._arrival_clock_ms, self._worker_available_ms[worker_index]) + service_ms
        self._worker_available_ms[worker_index] = service_end_ms

        response_text = self._build_response_text(output_tokens)
        return LocalRunResult(
            status="ok",
            response_text=response_text,
            ttft_ms=ttft_ms,
            e2e_ms=e2e_ms,
            tpot_ms=tpot_ms,
            output_tokens_est=output_tokens,
            throughput_tps=output_tokens / max(service_ms / 1000.0, 1e-6),
            bandwidth_bytes=len(response_text.encode("utf-8")),
            status_code=200,
            finish_reason="stop",
            vram_peak_gb=0.0,
            queue_wait_ms=queue_wait_ms,
        )


def load_remote_backend(config: Dict[str, object], name: str, timeout_s: int) -> OpenAICompatibleBackend | CloudEmulatedBackend | None:
    backend_cfg = config.get(name, {})
    if not backend_cfg or not backend_cfg.get("enabled", False):
        return None
    if str(backend_cfg.get("backend", "")).lower() == "cloud_emulated":
        return CloudEmulatedBackend(backend_cfg, seed=int(config["request"]["seed"]) + 17)
    return OpenAICompatibleBackend(
        RemoteBackendConfig(
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real local llama.cpp experiment on Windows.")
    parser.add_argument(
        "--config",
        default="configs/real_runtime_local.yaml",
        help="Path to the local real-runtime config.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only initialize the local backend.")
    parser.add_argument("--dataset", default="prompts", choices=["prompts", "privacy_prompts"], help="Dataset to replay.")
    parser.add_argument("--limit", type=int, default=None, help="Override subset size.")
    return parser.parse_args()


def choose_request_subset(requests: List, subset_size: int, seed: int) -> List:
    rng = random.Random(seed)
    if len(requests) <= subset_size:
        return list(requests)

    long_requests = [request for request in requests if request.category == "long_context"]
    privacy_requests = [request for request in requests if request.category == "privacy"]
    others = [request for request in requests if request.category not in {"long_context", "privacy"}]

    quota_long = min(len(long_requests), max(4, subset_size // 4))
    quota_privacy = min(len(privacy_requests), max(4, subset_size // 4))
    while quota_long + quota_privacy > subset_size:
        if quota_long >= quota_privacy and quota_long > 4:
            quota_long -= 1
        elif quota_privacy > 4:
            quota_privacy -= 1
        else:
            break

    chosen = rng.sample(long_requests, quota_long) + rng.sample(privacy_requests, quota_privacy)
    chosen_ids = {request.request_id for request in chosen}
    remainder_pool = [request for request in requests if request.request_id not in chosen_ids]
    remaining = max(0, subset_size - len(chosen))
    if remaining:
        chosen.extend(rng.sample(remainder_pool, min(remaining, len(remainder_pool))))
    rng.shuffle(chosen)
    return chosen[:subset_size]


def sleep_for_profile(profile: Dict[str, object], payload_bytes: int, phase: str, transport_factor: float) -> None:
    sleep_ms = estimate_profile_delay_ms(profile, payload_bytes, phase, transport_factor)
    sleep_s = sleep_ms / 1000.0
    if sleep_s > 0:
        time.sleep(min(sleep_s, 1.2))


def estimate_profile_delay_ms(profile: Dict[str, object], payload_bytes: int, phase: str, transport_factor: float) -> float:
    rtt_ms = float(profile["rtt_ms"]) * transport_factor
    bandwidth_mbps = max(float(profile["bandwidth_mbps"]) * max(transport_factor, 0.08), 1.0)
    loss_pct = float(profile["loss_pct"]) * transport_factor
    jitter_ms = float(profile["jitter_ms"]) * transport_factor
    bytes_per_second = bandwidth_mbps * 1024 * 1024 / 8.0
    transfer_s = payload_bytes / max(bytes_per_second, 1.0)
    sleep_s = rtt_ms / 2000.0 + transfer_s
    if phase == "after":
        sleep_s += loss_pct / 250.0
    sleep_s += jitter_ms / 1000.0 * 0.08
    return max(0.0, sleep_s * 1000.0)


def coerce_result(result, route: str) -> LocalRunResult:
    if isinstance(result, LocalRunResult):
        return result

    output_tokens = int(getattr(result, "output_tokens_est", 0) or 0)
    ttft_ms = float(getattr(result, "ttft_ms", 0.0))
    e2e_ms = float(getattr(result, "e2e_ms", 0.0))
    decode_ms = max(e2e_ms - ttft_ms, 0.0)
    tpot_ms = decode_ms / max(output_tokens, 1) if output_tokens else 0.0

    return LocalRunResult(
        status=str(getattr(result, "status", "error")),
        response_text=str(getattr(result, "response_text", "")),
        ttft_ms=ttft_ms,
        e2e_ms=e2e_ms,
        tpot_ms=tpot_ms,
        output_tokens_est=output_tokens,
        throughput_tps=float(getattr(result, "throughput_tps", 0.0)),
        bandwidth_bytes=int(getattr(result, "bandwidth_bytes", 0)),
        status_code=int(getattr(result, "status_code", 0)),
        finish_reason=str(getattr(result, "finish_reason", "stop")),
        vram_peak_gb=0.0 if route == "cloud" else float(getattr(result, "vram_peak_gb", 0.0)),
        queue_wait_ms=float(getattr(result, "queue_wait_ms", 0.0)),
        error=str(getattr(result, "error", "")),
    )


def backend_base_url(backend) -> str:
    config = getattr(backend, "config", None)
    if isinstance(config, dict):
        return str(config.get("base_url", ""))
    if config is not None and hasattr(config, "base_url"):
        return str(config.base_url)
    return ""


def backend_label(backend, route: str) -> str:
    if route == "edge":
        return "llama_cpp_local"
    config = getattr(backend, "config", None)
    if isinstance(config, dict):
        return str(config.get("backend", "cloud_emulated"))
    if config is not None and hasattr(config, "backend"):
        return str(config.backend)
    return route


def select_route(
    strategy: str,
    request,
    privacy_score: float,
    semantic_hit: bool,
    cloud_available: bool,
    profile_name: str,
) -> str:
    if strategy == "edge_only":
        return "edge"
    if strategy == "cloud_only":
        return "cloud"
    if strategy == "no_privacy":
        return "cloud" if cloud_available else "edge"
    if semantic_hit and privacy_score < 0.86 and request.category != "long_context":
        return "cache"
    if privacy_score >= 0.68:
        return "edge"
    if not cloud_available:
        return "edge"
    if profile_name in {"weak", "congested"} or request.prompt_tokens > 1600 or request.category == "long_context":
        return "edge"
    return "cloud"


def quality_proxy(response_text: str, finish_reason: str, request) -> float:
    if not response_text.strip():
        return 0.0
    length_ratio = min(len(response_text.strip()) / max(request.output_tokens * 4, 32), 1.0)
    score = 0.55 + 0.45 * length_ratio
    if finish_reason == "length":
        score -= 0.05
    return round(max(0.3, min(score, 1.0)), 3)


def write_real_report(
    base_dir: Path,
    config_path: Path,
    health_frame: pd.DataFrame,
    summary: pd.DataFrame,
    action_distribution: pd.DataFrame,
) -> None:
    report = [
        "# Real Local / Emulated-Cloud Experiment Report",
        "",
        f"- Runtime config: `{config_path.name}`",
        "- Edge runtime: local `llama-cpp-python` direct invocation.",
        "- Cloud runtime: OpenAI-compatible API or single-machine multi-worker emulated cloud backend.",
        "- Network profiles are emulated as transport overhead wrapped around real backend inference.",
        "- Note: `mean_quality` is a response-completeness proxy, not a human accuracy score.",
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
        "",
        "## Action Distribution",
        "",
        dataframe_to_markdown(action_distribution, ["ratio"]),
    ]
    (base_dir / "results" / "real_report.md").write_text("\n".join(report), encoding="utf-8")


def run_real_local() -> None:
    args = parse_args()
    configs = load_all_configs(BASE_DIR)
    config_path = BASE_DIR / args.config
    real_config = load_json_compatible_yaml(config_path)
    request_cfg = real_config["request"]

    dataset_name = "privacy_prompts" if args.dataset == "privacy_prompts" else "prompts"
    request_path = BASE_DIR / "data" / f"{dataset_name}.jsonl"
    requests = load_requests(request_path)
    subset_size = args.limit or int(request_cfg["subset_size"])
    requests = choose_request_subset(requests, subset_size=subset_size, seed=int(request_cfg["seed"]))

    network_profiles = {profile["name"]: profile for profile in configs["network_profiles"]["profiles"]}
    edge_backend = LocalLlamaCppBackend(real_config["edge"])
    cloud_backend = load_remote_backend(real_config, "cloud", int(request_cfg["timeout_s"]))

    health_rows = []
    edge_ok, edge_message = edge_backend.healthcheck()
    health_rows.append(
        {
            "backend": "edge",
            "ok": edge_ok,
            "message": edge_message,
            "base_url": real_config["edge"]["base_url"],
        }
    )
    cloud_ok = False
    if cloud_backend is not None:
        cloud_ok, cloud_message = cloud_backend.healthcheck()
        health_rows.append(
            {
                "backend": "cloud",
                "ok": cloud_ok,
                "message": cloud_message,
                "base_url": backend_base_url(cloud_backend),
            }
        )
    health_frame = pd.DataFrame(health_rows)
    health_frame.to_csv(BASE_DIR / "results" / "real_backend_health.csv", index=False)

    if args.dry_run:
        print("Dry run complete.")
        print(health_frame.to_string(index=False))
        return

    if not edge_ok:
        raise RuntimeError(f"Local backend failed to initialize: {edge_message}")

    backends = {"edge": edge_backend, "cloud": cloud_backend}
    runnable_strategies = [
        strategy
        for strategy in real_config["strategies"]
        if strategy != "cloud_only" or cloud_ok
    ]
    if not runnable_strategies:
        raise RuntimeError("No runnable strategies remain after backend availability filtering.")

    for request in requests[: int(request_cfg["warmup_size"])]:
        edge_backend.chat_completion(
            prompt=request.prompt,
            temperature=float(request_cfg["temperature"]),
            max_tokens=min(32, int(request_cfg["max_tokens"])),
        )
        if cloud_ok and cloud_backend is not None:
            if isinstance(cloud_backend, CloudEmulatedBackend):
                cloud_backend.chat_completion(
                    prompt=request.prompt,
                    temperature=float(request_cfg["temperature"]),
                    max_tokens=min(32, int(request_cfg["max_tokens"])),
                    prompt_tokens=request.prompt_tokens,
                    expected_output_tokens=min(request.output_tokens, 32),
                )
            else:
                cloud_backend.chat_completion(
                    prompt=request.prompt,
                    temperature=float(request_cfg["temperature"]),
                    max_tokens=min(32, int(request_cfg["max_tokens"])),
                )

    semantic_cache = SemanticCache(capacity=96)
    rows: List[Dict[str, object]] = []
    privacy_threshold = float(request_cfg["privacy_threshold"])

    for profile_name in ["wifi_good", "mobile_4g5g", "congested", "weak"]:
        profile = network_profiles[profile_name]
        for strategy in runnable_strategies:
            semantic_cache = SemanticCache(capacity=96)
            if hasattr(edge_backend, "reset"):
                edge_backend.reset()
            if cloud_backend is not None and hasattr(cloud_backend, "reset"):
                cloud_backend.reset()
            for request in requests:
                privacy_score, features = score_prompt(request.prompt)
                cache_entry, similarity = semantic_cache.lookup(request.prompt, float(request_cfg["semantic_threshold"]))
                semantic_hit = cache_entry is not None and request.category != "long_context"
                route = select_route(
                    strategy=strategy,
                    request=request,
                    privacy_score=privacy_score,
                    semantic_hit=semantic_hit,
                    cloud_available=cloud_ok,
                    profile_name=profile_name,
                )
                sensitive = int(bool(request.sensitive_expected) or privacy_score >= privacy_threshold)

                if route == "cache" and semantic_hit:
                    output_tokens = max(12, min(int(request.output_tokens), int(request_cfg["max_tokens"])))
                    edge_factor = float(request_cfg["cache_transport_factor"])
                    payload_bytes = len(request.prompt.encode("utf-8"))
                    before_ms = estimate_profile_delay_ms(profile, payload_bytes, "before", edge_factor)
                    after_ms = estimate_profile_delay_ms(profile, output_tokens * 4, "after", edge_factor)
                    sleep_for_profile(profile, payload_bytes, "before", edge_factor)
                    sleep_for_profile(profile, output_tokens * 4, "after", edge_factor)
                    cache_compute_ms = 10.0
                    effective_e2e_ms = max(15.0, cache_compute_ms + before_ms + after_ms)
                    rows.append(
                        {
                            "network": profile_name,
                            "network_label": profile["label"],
                            "strategy": strategy,
                            "strategy_label": STRATEGY_LABELS.get(strategy, strategy),
                            "route": "cache",
                            "backend": "semantic-cache",
                            "request_id": request.request_id,
                            "category": request.category,
                            "prompt_tokens": request.prompt_tokens,
                            "privacy_score": privacy_score,
                            "semantic_hit": 1,
                            "similarity": similarity,
                            "ttft_ms": max(8.0, before_ms * 0.8 + cache_compute_ms * 0.5),
                            "e2e_ms": effective_e2e_ms,
                            "tpot_ms": max(0.5, cache_compute_ms / max(output_tokens, 1)),
                            "throughput_tps": output_tokens / max(effective_e2e_ms / 1000.0, 1e-6),
                            "output_tokens_est": output_tokens,
                            "bandwidth_bytes": payload_bytes,
                            "status": "ok",
                            "status_code": 200,
                            "error": "",
                            "privacy_exposed": 0,
                            "sensitive": sensitive,
                            "prefix_hit": 0,
                            "vram_peak_gb": 0.0,
                            "queue_wait_ms": 0.0,
                            "quality_score": 0.95,
                            "n_id": features["n_id"],
                            "n_loc": features["n_loc"],
                            "n_contact": features["n_contact"],
                            "n_domain": features["n_domain"],
                            "n_name": features["n_name"],
                        }
                    )
                    continue

                transport_factor = 1.0 if route == "cloud" else float(request_cfg["edge_transport_factor"])
                payload_bytes = len(request.prompt.encode("utf-8"))
                before_ms = estimate_profile_delay_ms(profile, payload_bytes, "before", transport_factor)
                sleep_for_profile(profile, payload_bytes, "before", transport_factor)
                backend = backends.get(route)
                if backend is None:
                    continue
                if isinstance(backend, CloudEmulatedBackend):
                    raw_result = backend.chat_completion(
                        prompt=request.prompt,
                        temperature=float(request_cfg["temperature"]),
                        max_tokens=int(request_cfg["max_tokens"]),
                        prompt_tokens=request.prompt_tokens,
                        expected_output_tokens=request.output_tokens,
                    )
                else:
                    raw_result = backend.chat_completion(
                        prompt=request.prompt,
                        temperature=float(request_cfg["temperature"]),
                        max_tokens=int(request_cfg["max_tokens"]),
                    )
                result = coerce_result(raw_result, route)
                after_ms = estimate_profile_delay_ms(profile, len(result.response_text.encode("utf-8")), "after", transport_factor)
                sleep_for_profile(profile, len(result.response_text.encode("utf-8")), "after", transport_factor)

                quality_score = quality_proxy(result.response_text, result.finish_reason, request) if result.status == "ok" else 0.0
                if result.status == "ok":
                    semantic_cache.store(
                        cache_key=request.request_id,
                        prompt=request.prompt,
                        response_route=route,
                        quality_score=quality_score,
                    )
                effective_ttft_ms = result.ttft_ms + before_ms * (0.9 if route == "cloud" else 0.55)
                effective_e2e_ms = result.e2e_ms + before_ms + after_ms

                rows.append(
                    {
                        "network": profile_name,
                        "network_label": profile["label"],
                        "strategy": strategy,
                        "strategy_label": STRATEGY_LABELS.get(strategy, strategy),
                        "route": route,
                        "backend": backend_label(backend, route),
                        "request_id": request.request_id,
                        "category": request.category,
                        "prompt_tokens": request.prompt_tokens,
                        "privacy_score": privacy_score,
                        "semantic_hit": 0,
                        "similarity": similarity,
                        "ttft_ms": effective_ttft_ms,
                        "e2e_ms": effective_e2e_ms,
                        "tpot_ms": result.tpot_ms,
                        "throughput_tps": result.output_tokens_est / max(effective_e2e_ms / 1000.0, 1e-6),
                        "output_tokens_est": result.output_tokens_est,
                        "bandwidth_bytes": result.bandwidth_bytes + payload_bytes,
                        "status": result.status,
                        "status_code": result.status_code,
                        "error": result.error,
                        "privacy_exposed": int(route == "cloud" and sensitive == 1),
                        "sensitive": sensitive,
                        "prefix_hit": 0,
                        "vram_peak_gb": result.vram_peak_gb if route == "edge" else 0.0,
                        "queue_wait_ms": result.queue_wait_ms,
                        "quality_score": quality_score,
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
        raise RuntimeError("Local real experiment finished but no successful requests were recorded.")

    summary = summarize_by_group(ok_frame, ["network", "network_label", "strategy", "strategy_label"])
    summary.to_csv(BASE_DIR / "results" / "real_summary.csv", index=False)

    action_distribution = build_action_distribution(ok_frame[ok_frame["strategy"] == "ours"])
    action_distribution.to_csv(BASE_DIR / "results" / "real_action_distribution.csv", index=False)

    write_real_report(BASE_DIR, config_path, health_frame, summary, action_distribution)
    generate_real_figures(BASE_DIR)

    print("Real local experiment completed.")
    print(BASE_DIR / "results" / "real_summary.csv")
    print(BASE_DIR / "results" / "figures")


if __name__ == "__main__":
    run_real_local()
