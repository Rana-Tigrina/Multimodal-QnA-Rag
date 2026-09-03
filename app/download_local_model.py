"""
Helper script to pre-download the local GGUF model for offline llama.cpp inference.
Model: bartowski/Ling-3.0-tiny-GGUF (Ling-3.0-tiny-Q4_K_M.gguf)
"""
import sys
from pathlib import Path

from config import (
    LOCAL_MODEL_REPO,
    LOCAL_MODEL_FILE,
    LOCAL_MODELS_DIR,
    LOCAL_MODEL_PATH,
)

def download():
    print("=" * 65)
    print("  Downloading Local llama.cpp Model")
    print(f"  Repository: {LOCAL_MODEL_REPO}")
    print(f"  File:       {LOCAL_MODEL_FILE}")
    print(f"  Target Dir: {LOCAL_MODELS_DIR}")
    print("=" * 65)

    LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if LOCAL_MODEL_PATH.exists() and LOCAL_MODEL_PATH.stat().st_size > 100_000_000:
        size_gb = round(LOCAL_MODEL_PATH.stat().st_size / (1024 ** 3), 2)
        print(f"✅ Model already downloaded at: {LOCAL_MODEL_PATH} ({size_gb} GB)")
        return

    print("Downloading from Hugging Face... (approx 4.68 GB)")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id=LOCAL_MODEL_REPO,
        filename=LOCAL_MODEL_FILE,
        local_dir=LOCAL_MODELS_DIR,
    )
    print(f"\n✅ Download complete! Model saved to:\n  {path}")


if __name__ == "__main__":
    download()
