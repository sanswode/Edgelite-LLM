from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from .entities import NetworkProfile, Request
from .prefix_cache import PrefixCache
from .simulator import LLMSimulator, RuntimeState


@dataclass
class StrategyFlags:
    use_scheduler: bool
    use_semantic_cache: bool
    use_prefix_cache: bool
    use_privacy_gate: bool


class Scheduler:
    def __init__(
        self,
        weights: Dict[str, float],
        privacy_threshold: float,
        safe_vram_gb: float,
        allowed_routes: Optional[Iterable[str]] = None,
    ) -> None:
        self.weights = weights
        self.privacy_threshold = privacy_threshold
        self.safe_vram_gb = safe_vram_gb
        self.allowed_routes = tuple(allowed_routes or ("local", "edge", "cloud", "draft_edge"))

    def _normalize_route(self, route: str) -> str:
        if route in self.allowed_routes:
            return route
        if route == "local" and "edge" in self.allowed_routes:
            return "edge"
        if route == "draft_edge" and "edge" in self.allowed_routes:
            return "edge"
        if route == "cloud" and "edge" in self.allowed_routes:
            return "edge"
        return self.allowed_routes[0]

    def static_route(self, request: Request, network: NetworkProfile, privacy_score: float, use_privacy_gate: bool) -> str:
        if use_privacy_gate and privacy_score > self.privacy_threshold:
            return self._normalize_route("edge")
        if request.prompt_tokens > 1700 or network.rtt_ms >= 70 or request.category in {"long_context", "privacy"}:
            return self._normalize_route("edge")
        if request.difficulty < 0.24 and request.prompt_tokens < 220:
            return self._normalize_route("local")
        return self._normalize_route("cloud")

    def choose_route(
        self,
        request: Request,
        network: NetworkProfile,
        state: RuntimeState,
        simulator: LLMSimulator,
        prefix_cache: PrefixCache,
        privacy_score: float,
        sensitive: bool,
        flags: StrategyFlags,
    ) -> str:
        candidate_routes: Iterable[str] = self.allowed_routes
        best_route = "edge"
        best_cost = float("inf")

        for route in candidate_routes:
            if route == "cloud" and flags.use_privacy_gate and privacy_score > self.privacy_threshold:
                continue

            preview = simulator.simulate(
                request=request,
                route=route,
                network=network,
                state=state,
                prefix_cache=prefix_cache,
                privacy_score=privacy_score,
                sensitive=sensitive,
                enable_prefix_cache=flags.use_prefix_cache,
                commit=False,
            )

            if route in {"local", "edge", "draft_edge"} and preview.vram_peak_gb > self.safe_vram_gb:
                continue

            latency_term = preview.e2e_ms / 1000.0
            bandwidth_term = preview.bandwidth_bytes / 1_000_000.0
            memory_term = min(2.0, preview.notes.get("memory_risk", 0.0))
            privacy_term = preview.notes.get("privacy_risk", 0.0)
            quality_term = max(0.0, request.quality_target - preview.quality_score)

            cost = (
                self.weights["alpha"] * latency_term
                + self.weights["beta"] * bandwidth_term
                + self.weights["gamma"] * memory_term
                + self.weights["lambda"] * privacy_term
                + self.weights["eta"] * quality_term
            )

            if route == "cloud":
                cost += network.rtt_ms / 700.0
                cost += max(0.0, 10.0 - network.bandwidth_mbps) / 45.0
                if network.rtt_ms > 70:
                    cost += 0.12
            if route == "edge":
                cost -= min(0.10, network.rtt_ms / 1800.0)
            if route == "edge" and request.prompt_tokens > 1200:
                cost -= 0.05
            if route == "local" and request.difficulty > 0.55:
                cost += 0.10
            if route == "draft_edge" and 0.25 <= request.difficulty <= 0.75:
                cost -= 0.04

            if cost < best_cost:
                best_cost = cost
                best_route = route

        return best_route
