from __future__ import annotations

import random
from dataclasses import asdict
from typing import Dict

from .entities import ModelProfile, NetworkProfile, Request
from .prefix_cache import PrefixCache
from .privacy_gate import score_prompt
from .scheduler import Scheduler, StrategyFlags
from .semantic_cache import SemanticCache
from .simulator import LLMSimulator


class ExperimentGateway:
    def __init__(self, configs: Dict[str, object], seed: int) -> None:
        model_config = configs["models"]["models"]
        self.models = {
            key: ModelProfile(**value)
            for key, value in model_config.items()
        }
        self.hardware = configs["hardware"]
        experiment = configs["experiment"]
        self.semantic_threshold = float(experiment["semantic_threshold"])
        self.privacy_threshold = float(experiment["privacy_threshold"])
        self.semantic_cache = SemanticCache(capacity=int(experiment["cache_capacity"]))
        self.prefix_cache = PrefixCache(
            capacity_mb=float(self.hardware["prefix_cache_capacity_mb"]),
            models=self.models,
        )
        self.rng = random.Random(seed)
        self.simulator = LLMSimulator(self.models, self.hardware, self.rng)
        self.state = self.simulator.build_state()
        self.scheduler = Scheduler(
            weights=experiment["scheduler_weights"],
            privacy_threshold=self.privacy_threshold,
            safe_vram_gb=self.hardware["gpu_vram_gb"] * self.hardware["safe_utilization"],
            allowed_routes=experiment.get("allowed_routes"),
        )

    @staticmethod
    def flags_for_strategy(strategy: str) -> StrategyFlags:
        mapping = {
            "cloud_only": StrategyFlags(False, False, False, False),
            "edge_only": StrategyFlags(False, False, False, False),
            "static_threshold": StrategyFlags(False, False, False, True),
            "semantic_cache_only": StrategyFlags(False, True, False, False),
            "no_privacy": StrategyFlags(True, True, True, False),
            "ours": StrategyFlags(True, True, True, True),
            "full": StrategyFlags(True, True, True, True),
            "wo_scheduler": StrategyFlags(False, True, True, True),
            "wo_semantic_cache": StrategyFlags(True, False, True, True),
            "wo_kv_cache": StrategyFlags(True, True, False, True),
            "wo_privacy_gate": StrategyFlags(True, True, True, False),
        }
        return mapping[strategy]

    def handle_request(
        self,
        request: Request,
        network: NetworkProfile,
        strategy: str,
        semantic_threshold: float | None = None,
    ) -> Dict[str, object]:
        flags = self.flags_for_strategy(strategy)
        threshold = self.semantic_threshold if semantic_threshold is None else semantic_threshold
        privacy_score, features = score_prompt(request.prompt)
        sensitive = request.sensitive_expected or privacy_score >= self.privacy_threshold

        semantic_entry = None
        similarity = 0.0
        semantic_eligible = (
            flags.use_semantic_cache
            and request.category != "long_context"
            and not (flags.use_privacy_gate and privacy_score >= self.privacy_threshold * 0.9)
        )
        if semantic_eligible:
            semantic_entry, similarity = self.semantic_cache.lookup(request.prompt, threshold)

        required_similarity = threshold
        if request.category == "rag":
            required_similarity += 0.02
        if request.category == "privacy":
            required_similarity += 0.05

        if semantic_eligible and semantic_entry is not None and similarity >= required_similarity:
            allow_cache = not flags.use_privacy_gate or privacy_score < min(0.92, self.privacy_threshold + 0.18)
            if allow_cache:
                result = self.simulator.simulate(
                    request=request,
                    route="cache",
                    network=network,
                    state=self.state,
                    prefix_cache=self.prefix_cache,
                    privacy_score=privacy_score,
                    sensitive=sensitive,
                    enable_prefix_cache=False,
                    similarity=similarity,
                    cached_quality=semantic_entry.quality_score,
                    commit=True,
                )
                return self._build_row(request, network, strategy, threshold, features, result)

        route = self._choose_route(request, network, strategy, flags, privacy_score, sensitive)
        result = self.simulator.simulate(
            request=request,
            route=route,
            network=network,
            state=self.state,
            prefix_cache=self.prefix_cache,
            privacy_score=privacy_score,
            sensitive=sensitive,
            enable_prefix_cache=flags.use_prefix_cache,
            commit=True,
        )

        if semantic_eligible and route != "cache":
            self.semantic_cache.store(
                cache_key=request.request_id,
                prompt=request.prompt,
                response_route=route,
                quality_score=result.quality_score,
            )

        return self._build_row(request, network, strategy, threshold, features, result)

    def _choose_route(
        self,
        request: Request,
        network: NetworkProfile,
        strategy: str,
        flags: StrategyFlags,
        privacy_score: float,
        sensitive: bool,
    ) -> str:
        if strategy == "cloud_only":
            return "cloud"
        if strategy == "edge_only":
            return "edge"
        if strategy in {"static_threshold", "wo_scheduler"}:
            return self.scheduler.static_route(request, network, privacy_score, flags.use_privacy_gate)
        if strategy == "semantic_cache_only":
            return "cloud"
        return self.scheduler.choose_route(
            request=request,
            network=network,
            state=self.state,
            simulator=self.simulator,
            prefix_cache=self.prefix_cache,
            privacy_score=privacy_score,
            sensitive=sensitive,
            flags=flags,
        )

    @staticmethod
    def _build_row(
        request: Request,
        network: NetworkProfile,
        strategy: str,
        threshold: float,
        features: Dict[str, int],
        result,
    ) -> Dict[str, object]:
        return {
            "request_id": request.request_id,
            "strategy": strategy,
            "network": network.name,
            "network_label": network.label,
            "route": result.route,
            "category": request.category,
            "topic": request.topic,
            "cluster_id": request.cluster_id,
            "variant_id": request.variant_id,
            "prompt_tokens": request.prompt_tokens,
            "output_tokens": request.output_tokens,
            "prefix_tokens": request.prefix_tokens,
            "difficulty": request.difficulty,
            "quality_target": request.quality_target,
            "arrival_ms": request.arrival_ms,
            "semantic_threshold": threshold,
            "privacy_score": result.privacy_score,
            "sensitive": result.sensitive,
            "semantic_hit": result.semantic_hit,
            "prefix_hit": result.prefix_hit,
            "similarity": result.similarity,
            "ttft_ms": result.ttft_ms,
            "e2e_ms": result.e2e_ms,
            "tpot_ms": result.tpot_ms,
            "throughput_tps": result.throughput_tps,
            "bandwidth_bytes": result.bandwidth_bytes,
            "vram_peak_gb": result.vram_peak_gb,
            "quality_score": result.quality_score,
            "privacy_exposed": result.privacy_exposed,
            "queue_wait_ms": result.queue_wait_ms,
            "cache_saved_ms": result.cache_saved_ms,
            "n_id": features["n_id"],
            "n_loc": features["n_loc"],
            "n_contact": features["n_contact"],
            "n_domain": features["n_domain"],
            "n_name": features["n_name"],
            "memory_risk": result.notes.get("memory_risk", 0.0),
            "privacy_risk": result.notes.get("privacy_risk", 0.0),
        }
