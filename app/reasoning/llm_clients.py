from __future__ import annotations

import os

import requests

from app.reasoning.models import ChatLLMConfig


class ChatLLMClient:
    """Tiny provider wrapper used by reasoning and evaluation.

    This keeps generation/judging logic inside app modules instead of scripts.
    The OpenAI path is intentionally OpenAI-compatible so it can work with
    OpenAI or any server exposing the same chat completions API.
    """

    def __init__(self, config: ChatLLMConfig):
        self.config = config

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.config.provider == "openai":
            return self._generate_openai(system_prompt, user_prompt)
        if self.config.provider == "ollama":
            return self._generate_ollama(system_prompt, user_prompt)
        raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

    @classmethod
    def from_env(
        cls,
        prefix: str,
        default_provider: str = "ollama",
        default_model: str = "qwen2.5:3b",
    ) -> "ChatLLMClient":
        provider = os.getenv(f"{prefix}_PROVIDER", default_provider).strip().lower()
        model = os.getenv(f"{prefix}_MODEL", default_model).strip()
        temperature = float(os.getenv(f"{prefix}_TEMPERATURE", "0.0"))
        timeout = int(os.getenv(f"{prefix}_TIMEOUT_SECONDS", "180"))
        base_url = os.getenv(f"{prefix}_BASE_URL") or None
        api_key = os.getenv(f"{prefix}_API_KEY") or None
        return cls(
            ChatLLMConfig(
                provider=provider,  # type: ignore[arg-type]
                model=model,
                temperature=temperature,
                timeout_seconds=timeout,
                base_url=base_url,
                api_key=api_key,
            )
        )

    def _generate_openai(self, system_prompt: str, user_prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(
            api_key=self.config.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=self.config.base_url or os.getenv("OPENAI_BASE_URL") or None,
        )
        response = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
        )
        return response.choices[0].message.content or ""

    def _generate_ollama(self, system_prompt: str, user_prompt: str) -> str:
        base_url = (self.config.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        prompt = f"{system_prompt}\n\n{user_prompt}".strip()
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.config.temperature},
            },
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        return str(response.json().get("response") or "")
