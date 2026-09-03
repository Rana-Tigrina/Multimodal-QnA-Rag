"""
Unified Model Service:
- Cloud: Groq (qwen/qwen3.8-27b) with reasoning_effort="none", temperature=0.66, max_completion_tokens=4096, top_p=0.95
- Local: llama.cpp (bartowski/Ling-3.0-tiny-GGUF:Q4_K_M)
"""
import os
import sys
import logging
from pathlib import Path
from typing import Generator

from config import (
    LLM_PROVIDER,
    GROQ_API_KEY,
    GROQ_MODEL,
    LOCAL_MODEL_REPO,
    LOCAL_MODEL_FILE,
    LOCAL_MODELS_DIR,
    LOCAL_MODEL_PATH,
    LOG_LEVEL,
    LOG_FORMAT,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("model_service")

_groq_client = None
_llama_instance = None
_llama_loading = False


def is_local_model_downloaded() -> bool:
    """Check if the GGUF model file is already present on local disk."""
    return LOCAL_MODEL_PATH.exists() and LOCAL_MODEL_PATH.stat().st_size > 100_000_000


def get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_API_KEY or os.getenv("GROQ_API_KEY", ""))
    return _groq_client


def get_llama_instance():
    global _llama_instance, _llama_loading
    if _llama_instance is not None:
        return _llama_instance

    _llama_loading = True
    try:
        from llama_cpp import Llama
        LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        if not LOCAL_MODEL_PATH.exists():
            logger.info(f"Local model not found at {LOCAL_MODEL_PATH}. Downloading {LOCAL_MODEL_FILE} from HuggingFace ({LOCAL_MODEL_REPO})...")
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=LOCAL_MODEL_REPO,
                filename=LOCAL_MODEL_FILE,
                local_dir=LOCAL_MODELS_DIR,
            )

        logger.info(f"Loading local llama.cpp model from {LOCAL_MODEL_PATH}...")
        _llama_instance = Llama(
            model_path=str(LOCAL_MODEL_PATH),
            n_ctx=4096,
            n_threads=max(1, (os.cpu_count() or 4) - 1),
            verbose=False,
        )
        logger.info("Local llama.cpp Ling 3.0 Tiny model loaded successfully!")
        return _llama_instance
    finally:
        _llama_loading = False


def stream_chat_completion(
    messages: list[dict],
    provider: str | None = None,
    temperature: float = 0.66,
    max_tokens: int = 4096,
    top_p: float = 0.95,
) -> Generator[str, None, None]:
    """
    Unified streaming generator yielding delta text tokens.
    """
    prov = (provider or LLM_PROVIDER).lower()

    if prov in ("llamacpp", "local"):
        logger.info(f"Streaming response via Local llama.cpp ({LOCAL_MODEL_FILE})...")
        llm = get_llama_instance()
        response = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=True,
        )
        for chunk in response:
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content
    else:
        logger.info(f"Streaming response via Groq Cloud ({GROQ_MODEL}) without reasoning effort...")
        client = get_groq_client()
        completion = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=top_p,
            reasoning_effort="none",
            stream=True,
            stop=None,
        )
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta:
                content = chunk.choices[0].delta.content
                if content:
                    yield content


def generate_chat_text(
    messages: list[dict],
    provider: str | None = None,
    temperature: float = 0.66,
    max_tokens: int = 500,
    top_p: float = 0.95,
) -> str:
    """
    Unified non-streaming completion for query rewriting, HyDE, etc.
    """
    prov = (provider or LLM_PROVIDER).lower()

    if prov in ("llamacpp", "local"):
        llm = get_llama_instance()
        resp = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=False,
        )
        return resp["choices"][0]["message"].get("content", "").strip()
    else:
        client = get_groq_client()
        resp = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=top_p,
            reasoning_effort="none",
            stream=False,
            stop=None,
        )
        return resp.choices[0].message.content or ""
