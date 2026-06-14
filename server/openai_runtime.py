from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, Optional

import requests


@dataclass
class BackendConfig:
    name: str
    backend: str
    base_url: str
    model: str
    api_key: str
    label: str
    enabled: bool = True


@dataclass
class RealRunResult:
    route: str
    backend: str
    status: str
    response_text: str
    ttft_ms: float
    e2e_ms: float
    output_tokens_est: int
    throughput_tps: float
    bandwidth_bytes: int
    status_code: int
    finish_reason: str
    error: str = ""


class OpenAICompatibleBackend:
    def __init__(self, config: BackendConfig, timeout_s: int) -> None:
        self.config = config
        self.timeout_s = timeout_s

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def healthcheck(self) -> tuple[bool, str]:
        base_url = self.config.base_url.rstrip("/")
        endpoints = ["/health", "/v1/models"]
        last_message = "unreachable"
        for endpoint in endpoints:
            try:
                response = requests.get(
                    f"{base_url}{endpoint}",
                    timeout=min(self.timeout_s, 10),
                )
                if response.ok:
                    return True, f"ok via {endpoint}"
                last_message = f"http {response.status_code} via {endpoint}"
                if response.status_code != 404:
                    return False, last_message
            except Exception as exc:  # noqa: BLE001
                last_message = f"{endpoint}: {exc}"
        return False, last_message

    @staticmethod
    def estimate_token_count(text: str) -> int:
        stripped = text.strip()
        if not stripped:
            return 0
        return max(1, int(len(stripped) / 4))

    @staticmethod
    def build_chatml_prompt(prompt: str) -> str:
        return (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    @staticmethod
    def trim_prompt(prompt: str, max_chars: int = 6000) -> str:
        if len(prompt) <= max_chars:
            return prompt
        head = max_chars // 2
        tail = max_chars - head
        return (
            prompt[:head]
            + "\n\n[context truncated for edge runtime]\n\n"
            + prompt[-tail:]
        )

    def chat_completion(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> RealRunResult:
        use_completion_api = self.config.backend.lower() == "llama.cpp"
        prompt_text = self.trim_prompt(prompt) if use_completion_api else prompt
        request_prompt = self.build_chatml_prompt(prompt_text) if use_completion_api else prompt_text
        payload = (
            {
                "model": self.config.model,
                "prompt": request_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            }
            if use_completion_api
            else {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }
        )
        endpoint = "/v1/completions" if use_completion_api else "/v1/chat/completions"

        start = time.perf_counter()
        first_token_at: Optional[float] = None
        output_parts: list[str] = []
        finish_reason = ""
        status_code = 0

        try:
            if use_completion_api:
                response = requests.post(
                    f"{self.config.base_url.rstrip('/')}{endpoint}",
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout_s,
                )
                status_code = response.status_code
                response.raise_for_status()
                chunk = response.json()
                choices = chunk.get("choices", [])
                if choices:
                    output_parts.append(choices[0].get("text", ""))
                    finish_reason = choices[0].get("finish_reason") or finish_reason
            else:
                with requests.post(
                    f"{self.config.base_url.rstrip('/')}{endpoint}",
                    headers=self.headers,
                    json=payload,
                    stream=True,
                    timeout=self.timeout_s,
                ) as response:
                    status_code = response.status_code
                    response.raise_for_status()
                    for raw_line in response.iter_lines(decode_unicode=True):
                        if not raw_line:
                            continue
                        line = raw_line.strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                            output_parts.append(content)
                        finish_reason = choices[0].get("finish_reason") or finish_reason

            end = time.perf_counter()
            text = "".join(output_parts)
            output_tokens_est = self.estimate_token_count(text)
            ttft_ms = (end - start) * 1000.0 if use_completion_api else ((first_token_at or end) - start) * 1000.0
            e2e_ms = (end - start) * 1000.0
            throughput_tps = output_tokens_est / max(e2e_ms / 1000.0, 1e-6)
            return RealRunResult(
                route=self.config.name,
                backend=self.config.backend,
                status="ok",
                response_text=text,
                ttft_ms=ttft_ms,
                e2e_ms=e2e_ms,
                output_tokens_est=output_tokens_est,
                throughput_tps=throughput_tps,
                bandwidth_bytes=len(request_prompt.encode("utf-8")) + len(text.encode("utf-8")),
                status_code=status_code,
                finish_reason=finish_reason or "stop",
            )
        except Exception as exc:  # noqa: BLE001
            end = time.perf_counter()
            return RealRunResult(
                route=self.config.name,
                backend=self.config.backend,
                status="error",
                response_text="",
                ttft_ms=(end - start) * 1000.0,
                e2e_ms=(end - start) * 1000.0,
                output_tokens_est=0,
                throughput_tps=0.0,
                bandwidth_bytes=len(request_prompt.encode("utf-8")),
                status_code=status_code,
                finish_reason="error",
                error=str(exc),
            )
