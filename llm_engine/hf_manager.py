from __future__ import annotations

import gc
import json
import logging
import re
import threading
from pathlib import Path
from typing import Optional, Type, TypeVar

import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download
from pydantic import BaseModel

from config import (
    EMBEDDING_DIR,
    HF_EMBEDDING_REPO_ID,
    HF_LLM_FILENAME,
    HF_LLM_REPO_ID,
    LLM_DIR,
    LLM_N_BATCH,
    LLM_N_CTX,
    LLM_N_THREADS,
    LLM_USE_MMAP,
)

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LocalHFEngine:
    """
    Thread-safe Singleton managing in-process Hugging Face models:
    1. Qwen2.5-3B-Instruct-GGUF via llama-cpp-python (~1.93 GB disk, <2.2 GB RAM)
    2. sentence-transformers/all-MiniLM-L6-v2 on CPU (~90 MB RAM)
    Enforces offline-first execution (local_files_only=True once downloaded).
    """

    _instance: Optional["LocalHFEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "LocalHFEngine":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._llm = None
        self._embedder = None
        self._llm_lock = threading.Lock()
        self._embed_lock = threading.Lock()
        self._initialized = True

    # ------------------------------------------------------------------------
    # Model Provisioning (Offline-First with First-Run HF Hub Download)
    # ------------------------------------------------------------------------
    def ensure_llm_weights(self) -> Path:
        local_gguf_path = LLM_DIR / HF_LLM_FILENAME
        if local_gguf_path.exists() and local_gguf_path.stat().st_size > 100_000_000:
            return local_gguf_path

        try:
            # Attempt local cache first
            downloaded = hf_hub_download(
                repo_id=HF_LLM_REPO_ID,
                filename=HF_LLM_FILENAME,
                local_dir=str(LLM_DIR),
                local_files_only=True,
            )
            return Path(downloaded)
        except Exception:
            logger.info("Downloading %s/%s from Hugging Face Hub...", HF_LLM_REPO_ID, HF_LLM_FILENAME)
            downloaded = hf_hub_download(
                repo_id=HF_LLM_REPO_ID,
                filename=HF_LLM_FILENAME,
                local_dir=str(LLM_DIR),
                local_files_only=False,
            )
            return Path(downloaded)

    def ensure_embedding_weights(self) -> Path:
        model_subfolder = EMBEDDING_DIR / "all-MiniLM-L6-v2"
        config_file = model_subfolder / "config.json"
        if config_file.exists():
            return model_subfolder

        try:
            snapshot_download(
                repo_id=HF_EMBEDDING_REPO_ID,
                local_dir=str(model_subfolder),
                local_files_only=True,
            )
        except Exception:
            logger.info("Downloading %s from Hugging Face Hub...", HF_EMBEDDING_REPO_ID)
            snapshot_download(
                repo_id=HF_EMBEDDING_REPO_ID,
                local_dir=str(model_subfolder),
                local_files_only=False,
            )
        return model_subfolder

    # ------------------------------------------------------------------------
    # Lifecycle Management (Load / Unload)
    # ------------------------------------------------------------------------
    def load_llm(self):
        with self._llm_lock:
            if self._llm is not None:
                return self._llm

            from llama_cpp import Llama

            gguf_path = self.ensure_llm_weights()
            logger.info(
                "Loading GGUF LLM from %s (n_ctx=%d, n_batch=%d, n_threads=%d, mmap=%s)",
                gguf_path,
                LLM_N_CTX,
                LLM_N_BATCH,
                LLM_N_THREADS,
                LLM_USE_MMAP,
            )
            self._llm = Llama(
                model_path=str(gguf_path),
                n_ctx=LLM_N_CTX,
                n_batch=LLM_N_BATCH,
                n_threads=LLM_N_THREADS,
                use_mmap=LLM_USE_MMAP,
                n_gpu_layers=0,  # Strictly CPU-only ceiling
                verbose=False,
            )
            return self._llm

    def unload_llm(self) -> bool:
        """
        Explicitly releases the GGUF model from memory and forces garbage collection
        so RAM is reclaimed before CPU-intensive RL training.
        """
        with self._llm_lock:
            if self._llm is not None:
                logger.info("Unloading Qwen2.5-3B GGUF LLM and reclaiming RAM...")
                try:
                    if hasattr(self._llm, "close"):
                        self._llm.close()
                except Exception:
                    pass
                del self._llm
                self._llm = None
                gc.collect()
                return True
            gc.collect()
            return False

    def is_llm_loaded(self) -> bool:
        return self._llm is not None

    def load_embedder(self):
        with self._embed_lock:
            if self._embedder is not None:
                return self._embedder

            from sentence_transformers import SentenceTransformer

            embed_path = self.ensure_embedding_weights()
            logger.info("Loading SentenceTransformer from local disk: %s", embed_path)
            self._embedder = SentenceTransformer(
                str(embed_path),
                device="cpu",
                local_files_only=True,
            )
            return self._embedder

    # ------------------------------------------------------------------------
    # Embedding & Inference Methods
    # ------------------------------------------------------------------------
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        embedder = self.load_embedder()
        embeddings = embedder.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_model: Type[T],
        temperature: float = 0.05,
        max_tokens: int = 1200,
    ) -> T:
        """
        Executes grammar-constrained JSON Schema generation using llama-cpp-python
        and validates the output directly into the target Pydantic model.
        """
        llm = self.load_llm()
        json_schema = schema_model.model_json_schema()

        with self._llm_lock:
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_object",
                    "schema": json_schema,
                },
                temperature=temperature,
                max_tokens=max_tokens,
            )

        raw_content: str = response["choices"][0]["message"]["content"]
        try:
            return schema_model.model_validate_json(raw_content)
        except Exception:
            # Fallback extraction if whitespace or markdown fences wrap the JSON object
            match = re.search(r"(\{.*\})", raw_content, flags=re.DOTALL)
            if match:
                return schema_model.model_validate(json.loads(match.group(1)))
            raise

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.25,
        max_tokens: int = 768,
    ) -> str:
        """
        Generates natural-language commentary using Qwen2.5-3B-Instruct-GGUF.
        """
        llm = self.load_llm()
        with self._llm_lock:
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return response["choices"][0]["message"]["content"].strip()