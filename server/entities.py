from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Request:
    request_id: str
    prompt: str
    category: str
    topic: str
    cluster_id: str
    variant_id: int
    prompt_tokens: int
    output_tokens: int
    prefix_id: str
    prefix_tokens: int
    difficulty: float
    quality_target: float
    arrival_ms: float
    sensitive_expected: bool


@dataclass
class NetworkProfile:
    name: str
    label: str
    rtt_ms: float
    bandwidth_mbps: float
    loss_pct: float
    jitter_ms: float


@dataclass
class ModelProfile:
    name: str
    prefill_tps: float
    decode_tps: float
    quality_base: float
    difficulty_penalty: float
    weight_gb: float
    runtime_gb: float
    layers: int
    kv_heads: int
    head_dim: int
    bytes_per_element: int


@dataclass
class SimulationResult:
    route: str
    ttft_ms: float
    e2e_ms: float
    tpot_ms: float
    throughput_tps: float
    bandwidth_bytes: int
    vram_peak_gb: float
    quality_score: float
    privacy_exposed: int
    semantic_hit: int
    prefix_hit: int
    similarity: float
    queue_wait_ms: float
    cache_saved_ms: float
    privacy_score: float
    sensitive: int
    notes: Dict[str, float] = field(default_factory=dict)


@dataclass
class RunArtifacts:
    rows: List[Dict[str, object]]
