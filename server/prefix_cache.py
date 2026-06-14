from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .entities import ModelProfile


def kv_cache_size_mb(token_count: int, model: ModelProfile) -> float:
    bytes_used = (
        2
        * model.layers
        * token_count
        * model.kv_heads
        * model.head_dim
        * model.bytes_per_element
    )
    return bytes_used / (1024.0 * 1024.0)


@dataclass
class PrefixCacheEntry:
    prefix_id: str
    route: str
    token_count: int
    size_mb: float
    privacy_risk: float
    saved_ms: float
    hits: int = 0
    accesses: int = 1
    last_used: int = 0

    def utility(self) -> float:
        probability = (self.hits + 1.0) / (self.accesses + 2.0)
        return probability * (self.saved_ms + 0.12 * self.token_count) - 0.18 * self.size_mb - 35.0 * self.privacy_risk


class PrefixCache:
    def __init__(self, capacity_mb: float, models: Dict[str, ModelProfile]) -> None:
        self.capacity_mb = capacity_mb
        self.models = models
        self.entries: Dict[Tuple[str, str], PrefixCacheEntry] = {}
        self.clock = 0

    def total_size_mb(self) -> float:
        return sum(entry.size_mb for entry in self.entries.values())

    def peek(self, route: str, prefix_id: str) -> Optional[PrefixCacheEntry]:
        return self.entries.get((route, prefix_id))

    def lookup(self, route: str, prefix_id: str) -> Optional[PrefixCacheEntry]:
        key = (route, prefix_id)
        entry = self.entries.get(key)
        if entry is None:
            return None
        self.clock += 1
        entry.hits += 1
        entry.accesses += 1
        entry.last_used = self.clock
        return entry

    def maybe_store(
        self,
        route: str,
        prefix_id: str,
        token_count: int,
        privacy_risk: float,
        saved_ms: float,
    ) -> None:
        if route not in self.models:
            return

        model = self.models[route]
        size_mb = kv_cache_size_mb(token_count, model)
        if size_mb > self.capacity_mb * 0.65:
            return

        key = (route, prefix_id)
        self.clock += 1
        if key in self.entries:
            entry = self.entries[key]
            entry.token_count = token_count
            entry.privacy_risk = privacy_risk
            entry.saved_ms = max(entry.saved_ms, saved_ms)
            entry.accesses += 1
            entry.last_used = self.clock
            return

        while self.total_size_mb() + size_mb > self.capacity_mb and self.entries:
            evict_key = min(self.entries.items(), key=lambda item: item[1].utility())[0]
            del self.entries[evict_key]

        if self.total_size_mb() + size_mb > self.capacity_mb:
            return

        self.entries[key] = PrefixCacheEntry(
            prefix_id=prefix_id,
            route=route,
            token_count=token_count,
            size_mb=size_mb,
            privacy_risk=privacy_risk,
            saved_ms=saved_ms,
            last_used=self.clock,
        )
