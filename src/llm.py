import os
import re
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from langchain_openai import ChatOpenAI


SUPPORTED_LLM_BACKENDS = ("openai", "llama_cpp")
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:8080/v1"
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _read_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LLMConfig:
    backend: str = "openai"
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    disable_thinking: bool = True

    @classmethod
    def from_inputs(
        cls,
        backend: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        disable_thinking: bool | None = None,
    ) -> "LLMConfig":
        resolved_backend = (backend or os.getenv("LLM_BACKEND") or "openai").strip().lower()
        if resolved_backend not in SUPPORTED_LLM_BACKENDS:
            raise ValueError(
                f"Unsupported LLM backend '{resolved_backend}'. "
                f"Choose one of: {', '.join(SUPPORTED_LLM_BACKENDS)}"
            )

        default_model = "gpt-4o-mini" if resolved_backend == "openai" else "Qwen3.5-4B"
        resolved_model = (model or os.getenv("LLM_MODEL") or default_model).strip()
        resolved_base_url = base_url or os.getenv("LLM_BASE_URL")
        if resolved_backend == "llama_cpp" and not resolved_base_url:
            resolved_base_url = DEFAULT_LOCAL_BASE_URL

        if temperature is None:
            temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))

        if disable_thinking is None:
            disable_thinking = _read_bool_env("LLM_DISABLE_THINKING", True)

        resolved_api_key = api_key or os.getenv("LLM_API_KEY")
        if resolved_backend == "llama_cpp" and not resolved_api_key:
            resolved_api_key = "EMPTY"

        return cls(
            backend=resolved_backend,
            model=resolved_model,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            temperature=temperature,
            disable_thinking=disable_thinking,
        )

    @property
    def normalized_base_url(self) -> str | None:
        if not self.base_url:
            return None
        normalized = self.base_url.rstrip("/")
        if not normalized.endswith("/v1"):
            normalized = f"{normalized}/v1"
        return normalized

    @property
    def llama_cpp_root_url(self) -> str | None:
        normalized = self.normalized_base_url
        if not normalized:
            return None
        return normalized.removesuffix("/v1")


def describe_backend(config: LLMConfig) -> str:
    details = [f"backend={config.backend}", f"model={config.model}"]
    if config.backend == "llama_cpp" and config.normalized_base_url:
        details.append(f"base_url={config.normalized_base_url}")
    return ", ".join(details)


def validate_config(config: LLMConfig) -> None:
    if config.backend == "openai":
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required when using the OpenAI backend. "
                "Set it in .env or switch to --llm-backend llama_cpp."
            )
        return

    if not config.normalized_base_url:
        raise RuntimeError(
            "A llama.cpp base URL is required. "
            "Pass --llm-base-url or set LLM_BASE_URL."
        )


def ensure_llama_cpp_server(config: LLMConfig, timeout_seconds: float = 5.0) -> None:
    validate_config(config)
    if config.backend != "llama_cpp":
        return

    candidate_urls = [
        f"{config.llama_cpp_root_url}/health",
        f"{config.normalized_base_url}/models",
        f"{config.llama_cpp_root_url}/models",
    ]

    headers = {"Authorization": f"Bearer {config.api_key or 'EMPTY'}"}
    for url in candidate_urls:
        try:
            req = request.Request(url, headers=headers, method="GET")
            with request.urlopen(req, timeout=timeout_seconds) as response:
                if 200 <= response.status < 300:
                    return
        except (error.HTTPError, error.URLError, ValueError):
            continue

    raise RuntimeError(
        "Could not reach the local llama.cpp server. "
        f"Expected a healthy endpoint near {config.normalized_base_url}. "
        "Start llama-server first, then retry."
    )


def build_chat_model(config: LLMConfig) -> ChatOpenAI:
    validate_config(config)

    model_kwargs: dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
    }

    if config.backend == "openai":
        model_kwargs["openai_api_key"] = config.api_key or os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(**model_kwargs)

    ensure_llama_cpp_server(config)
    model_kwargs["openai_api_key"] = config.api_key or "EMPTY"
    model_kwargs["openai_api_base"] = config.normalized_base_url
    if config.disable_thinking:
        model_kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False}
        }
    return ChatOpenAI(**model_kwargs)


def extract_text_from_response(response: Any) -> str:
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return clean_response_text(content)

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return clean_response_text("\n".join(part for part in parts if part))

    return clean_response_text(str(content))


def clean_response_text(text: str) -> str:
    cleaned = THINK_BLOCK_RE.sub("", text).strip()
    if cleaned:
        return cleaned

    return text.replace("<think>", "").replace("</think>", "").strip()
