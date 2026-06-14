from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Optional

from .entities import ModelProfile, NetworkProfile, Request, SimulationResult
from .prefix_cache import PrefixCache, kv_cache_size_mb


@dataclass
class RuntimeState:
    available_at_ms: Dict[str, float]


class LLMSimulator:
    def __init__(
        self,
        models: Dict[str, ModelProfile],
        hardware: Dict[str, float],
        rng: random.Random,
    ) -> None:
        self.models = models
        self.hardware = hardware
        self.rng = rng
        self.safe_vram_gb = hardware["gpu_vram_gb"] * hardware["safe_utilization"]
        self.fragment_gb = hardware["fragment_gb"]
        parallelism = hardware.get("parallelism", {})
        self.parallelism = {
            "local": float(parallelism.get("local", 1.0)),
            "edge": float(parallelism.get("edge", 5.0)),
            "cloud": float(parallelism.get("cloud", 10.0)),
        }

    def build_state(self) -> RuntimeState:
        return RuntimeState(available_at_ms={"local": 0.0, "edge": 0.0, "cloud": 0.0})

    @staticmethod
    def _bytes_per_ms(bandwidth_mbps: float) -> float:
        if bandwidth_mbps <= 0:
            return 1e12
        return bandwidth_mbps * 125.0

    @staticmethod
    def _bandwidth_profile(route: str, network: NetworkProfile) -> tuple[float, float, float]:
        if route == "cloud":
            return network.rtt_ms, max(0.5, network.bandwidth_mbps), network.loss_pct
        if route in {"edge", "draft_edge"}:
            return 3.0 + network.rtt_ms * 0.06, max(120.0, network.bandwidth_mbps * 4.0), network.loss_pct * 0.12
        return 0.0, 1e9, 0.0

    def _quality_score(
        self,
        route: str,
        request: Request,
        similarity: float,
        cached_quality: float,
        prefix_hit: bool,
    ) -> float:
        if route == "cache":
            value = cached_quality - (1.0 - similarity) * 0.08
            return max(0.55, min(0.99, value))

        if route == "draft_edge":
            value = 0.91 - request.difficulty * 0.09
            if prefix_hit:
                value += 0.015
            return max(0.6, min(0.96, value))

        model = self.models["edge" if route == "draft_edge" else route]
        value = model.quality_base - request.difficulty * model.difficulty_penalty
        if request.category == "long_context" and route == "local":
            value -= 0.06
        if request.category == "privacy" and route == "cloud":
            value += 0.01
        if prefix_hit:
            value += 0.01
        return max(0.45, min(0.99, value))

    def _vram_peak_gb(self, route: str, active_tokens: int) -> float:
        if route == "cloud" or route == "cache":
            return 0.0
        model_key = "edge" if route == "draft_edge" else route
        model = self.models[model_key]
        kv_gb = kv_cache_size_mb(active_tokens, model) / 1024.0
        return model.weight_gb + model.runtime_gb + self.fragment_gb + kv_gb

    def simulate(
        self,
        request: Request,
        route: str,
        network: NetworkProfile,
        state: RuntimeState,
        prefix_cache: PrefixCache,
        privacy_score: float,
        sensitive: bool,
        enable_prefix_cache: bool,
        similarity: float = 0.0,
        cached_quality: float = 0.0,
        commit: bool = True,
    ) -> SimulationResult:
        if route == "cache":
            base = 18.0 + request.output_tokens * 0.22
            if commit:
                base *= 1.0 + self.rng.uniform(-0.04, 0.04)
            quality = self._quality_score(route, request, similarity, cached_quality, prefix_hit=False)
            return SimulationResult(
                route="cache",
                ttft_ms=max(8.0, base * 0.35),
                e2e_ms=max(20.0, base),
                tpot_ms=max(0.4, base * 0.01),
                throughput_tps=max(60.0, request.output_tokens / max(base / 1000.0, 1e-6)),
                bandwidth_bytes=0,
                vram_peak_gb=0.0,
                quality_score=quality,
                privacy_exposed=int(sensitive and privacy_score > 0.0 and False),
                semantic_hit=1,
                prefix_hit=0,
                similarity=similarity,
                queue_wait_ms=0.0,
                cache_saved_ms=0.0,
                privacy_score=privacy_score,
                sensitive=int(sensitive),
                notes={"memory_risk": 0.0, "privacy_risk": 0.0},
            )

        model_route = "edge" if route == "draft_edge" else route
        model = self.models[model_route]
        prefix_entry = None
        if enable_prefix_cache:
            prefix_entry = prefix_cache.lookup(model_route, request.prefix_id) if commit else prefix_cache.peek(model_route, request.prefix_id)
        prefix_hit = int(prefix_entry is not None)
        cache_saved_ms = request.prefix_tokens / model.prefill_tps * 1000.0 if prefix_hit else 0.0
        effective_prompt_tokens = max(24, request.prompt_tokens - (request.prefix_tokens if prefix_hit else 0))

        prompt_bytes = max(len(request.prompt.encode("utf-8")), request.prompt_tokens * 3)
        response_bytes = max(120, request.output_tokens * 5)
        rtt_ms, bandwidth_mbps, loss_pct = self._bandwidth_profile(route, network)
        bytes_per_ms = self._bytes_per_ms(bandwidth_mbps)

        if route == "cloud":
            effective_prompt_bytes = prompt_bytes
            effective_response_bytes = response_bytes
        elif route in {"edge", "draft_edge"}:
            effective_prompt_bytes = int(prompt_bytes * 0.28)
            effective_response_bytes = int(response_bytes * 0.28)
        else:
            effective_prompt_bytes = 0
            effective_response_bytes = 0

        upload_ms = effective_prompt_bytes / bytes_per_ms
        download_ms = effective_response_bytes / bytes_per_ms

        if route == "draft_edge":
            local = self.models["local"]
            local_draft_ms = (request.prompt_tokens / local.prefill_tps + request.output_tokens * 0.30 / local.decode_tps) * 1000.0
            edge_refine_ms = (effective_prompt_tokens * 0.60 / model.prefill_tps + request.output_tokens * 0.70 / model.decode_tps) * 1000.0
            prefill_ms = request.prompt_tokens * 0.55 / local.prefill_tps * 1000.0 + effective_prompt_tokens * 0.30 / model.prefill_tps * 1000.0
            decode_ms = request.output_tokens * 0.70 / model.decode_tps * 1000.0
            service_ms = local_draft_ms + edge_refine_ms
        else:
            prefill_ms = effective_prompt_tokens / model.prefill_tps * 1000.0
            decode_ms = request.output_tokens / model.decode_tps * 1000.0
            service_ms = prefill_ms + decode_ms

        queue_key = model_route
        queue_wait_ms = max(0.0, state.available_at_ms[queue_key] - request.arrival_ms)
        network_penalty_ms = rtt_ms + upload_ms + download_ms + loss_pct * 25.0
        jitter_ms = network.jitter_ms * (0.5 + self.rng.random()) if commit else network.jitter_ms * 0.75
        ttft_ms = queue_wait_ms + upload_ms + rtt_ms * 0.75 + prefill_ms + jitter_ms * 0.35
        e2e_ms = queue_wait_ms + network_penalty_ms + service_ms + jitter_ms

        if commit:
            noise = 1.0 + self.rng.uniform(-0.035, 0.035)
            ttft_ms *= noise
            e2e_ms *= noise
            service_window_ms = service_ms / self.parallelism.get(queue_key, 1.0)
            service_end = max(state.available_at_ms[queue_key], request.arrival_ms) + service_window_ms
            state.available_at_ms[queue_key] = service_end

        active_tokens = effective_prompt_tokens + min(request.output_tokens, 256)
        vram_peak_gb = self._vram_peak_gb(route, active_tokens)
        quality_score = self._quality_score(route, request, similarity, cached_quality, prefix_hit=bool(prefix_hit))
        memory_risk = max(0.0, vram_peak_gb / max(self.safe_vram_gb, 1e-6))
        privacy_risk = 0.0
        if route == "cloud":
            privacy_risk = privacy_score
        elif route in {"edge", "draft_edge"}:
            privacy_risk = privacy_score * 0.22
        elif route == "local":
            privacy_risk = privacy_score * 0.06

        if commit and enable_prefix_cache and request.prefix_tokens >= 96 and model_route in self.models:
            prefix_cache.maybe_store(
                route=model_route,
                prefix_id=request.prefix_id,
                token_count=request.prefix_tokens,
                privacy_risk=privacy_score if sensitive else 0.0,
                saved_ms=request.prefix_tokens / model.prefill_tps * 1000.0,
            )

        return SimulationResult(
            route=route,
            ttft_ms=ttft_ms,
            e2e_ms=e2e_ms,
            tpot_ms=decode_ms / max(request.output_tokens, 1),
            throughput_tps=request.output_tokens / max(decode_ms / 1000.0, 1e-6),
            bandwidth_bytes=effective_prompt_bytes + effective_response_bytes,
            vram_peak_gb=vram_peak_gb,
            quality_score=quality_score,
            privacy_exposed=int(sensitive and route == "cloud"),
            semantic_hit=0,
            prefix_hit=prefix_hit,
            similarity=similarity,
            queue_wait_ms=queue_wait_ms,
            cache_saved_ms=cache_saved_ms,
            privacy_score=privacy_score,
            sensitive=int(sensitive),
            notes={"memory_risk": memory_risk, "privacy_risk": privacy_risk},
        )
