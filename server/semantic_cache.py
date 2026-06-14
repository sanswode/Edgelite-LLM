from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


def normalize_text(text: str) -> str:
    chars = []
    for char in text.lower():
        if char.isalnum() or "\u4e00" <= char <= "\u9fff":
            chars.append(char)
    return "".join(chars)


def char_ngrams(text: str, n: int = 2) -> Counter[str]:
    clean = normalize_text(text)
    if len(clean) < n:
        return Counter({clean: 1}) if clean else Counter()
    return Counter(clean[index : index + n] for index in range(len(clean) - n + 1))


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot_product = sum(left[token] * right[token] for token in left.keys() & right.keys())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


@dataclass
class CacheEntry:
    prompt_text: str
    vector: Counter[str]
    response_route: str
    quality_score: float
    last_used: int
    hits: int


class SemanticCache:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.entries: Dict[str, CacheEntry] = {}
        self.counter = 0

    def lookup(self, prompt: str, threshold: float) -> Tuple[Optional[CacheEntry], float]:
        query_vector = char_ngrams(prompt)
        best_key = None
        best_similarity = 0.0
        for key, entry in self.entries.items():
            similarity = cosine_similarity(query_vector, entry.vector)
            if similarity > best_similarity:
                best_similarity = similarity
                best_key = key

        if best_key is None or best_similarity < threshold:
            return None, best_similarity

        self.counter += 1
        entry = self.entries[best_key]
        entry.hits += 1
        entry.last_used = self.counter
        return entry, best_similarity

    def store(self, cache_key: str, prompt: str, response_route: str, quality_score: float) -> None:
        self.counter += 1
        if cache_key in self.entries:
            entry = self.entries[cache_key]
            entry.prompt_text = prompt
            entry.vector = char_ngrams(prompt)
            entry.response_route = response_route
            entry.quality_score = quality_score
            entry.last_used = self.counter
            return

        if len(self.entries) >= self.capacity:
            lru_key = min(self.entries.items(), key=lambda item: item[1].last_used)[0]
            del self.entries[lru_key]

        self.entries[cache_key] = CacheEntry(
            prompt_text=prompt,
            vector=char_ngrams(prompt),
            response_route=response_route,
            quality_score=quality_score,
            last_used=self.counter,
            hits=0,
        )
