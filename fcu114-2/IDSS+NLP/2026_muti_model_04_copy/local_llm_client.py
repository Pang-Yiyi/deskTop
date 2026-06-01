"""Local LLM client for Ollama and llama.cpp backends.

The system keeps this module small and dependency-free so the Streamlit demo
can still run when no local model server is available. Callers should catch
LocalLLMError and fall back to deterministic rules.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LocalLLMError(RuntimeError):
    """Raised when the configured local LLM backend cannot return a response."""


def local_llm_enabled() -> bool:
    """Return whether local LLM calls should be attempted."""
    value = os.getenv("ENABLE_LOCAL_LLM", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class LocalLLMConfig:
    backend: str = os.getenv("LLM_BACKEND", "ollama").strip().lower()
    model: str = os.getenv("LLM_MODEL", "qwen3:4b")
    ollama_url: str = os.getenv("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")
    llamacpp_url: str = os.getenv("LLAMACPP_CHAT_URL", "http://localhost:8080/v1/chat/completions")
    timeout_seconds: float = float(os.getenv("LOCAL_LLM_TIMEOUT", "60"))


class LocalLLMClient:
    """HTTP client for local chat-completion style model servers."""

    def __init__(self, config: LocalLLMConfig | None = None) -> None:
        self.config = config or LocalLLMConfig()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 900,
        format_json: bool = False,
        think: bool = False,
    ) -> str:
        if not local_llm_enabled():
            raise LocalLLMError("Local LLM is disabled by ENABLE_LOCAL_LLM.")

        if self.config.backend == "ollama":
            return self._chat_ollama(messages, temperature=temperature,
                                     format_json=format_json, think=think)
        if self.config.backend in {"llamacpp", "llama.cpp", "llama_cpp"}:
            return self._chat_llamacpp(messages, temperature=temperature, max_tokens=max_tokens)
        raise LocalLLMError(f"Unsupported local LLM backend: {self.config.backend}")

    def _chat_ollama(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        format_json: bool,
        think: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if format_json:
            payload["format"] = "json"
        if think:
            payload["think"] = True   # Qwen3 extended thinking

        data = self._post_json(self.config.ollama_url, payload)
        msg = data.get("message", {})

        # Qwen3 thinking mode: thinking field OR <think>...</think> in content
        thinking_text = msg.get("thinking", "")
        content = msg.get("content", "")

        if not content and not thinking_text:
            raise LocalLLMError("Ollama returned an empty response.")

        # 若 thinking 在 content 內（某些版本），保留完整原文供呼叫端解析
        return str(content).strip()

    def _chat_llamacpp(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = self._post_json(self.config.llamacpp_url, payload)
        choices = data.get("choices") or []
        if not choices:
            raise LocalLLMError("llama.cpp returned no choices.")
        content = choices[0].get("message", {}).get("content")
        if not content:
            raise LocalLLMError("llama.cpp returned an empty response.")
        return str(content).strip()

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise LocalLLMError(f"Local LLM HTTP error {exc.code}: {exc.reason}") from exc
        except URLError as exc:
            raise LocalLLMError(f"Local LLM connection error: {exc.reason}") from exc
        except TimeoutError as exc:
            raise LocalLLMError("Local LLM request timed out.") from exc
        except json.JSONDecodeError as exc:
            raise LocalLLMError("Local LLM returned invalid JSON from HTTP API.") from exc
