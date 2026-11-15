from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Literal

import numpy as np
from pydantic import BaseModel, Field

from config import (
    ALL_CATEGORIES,
    DEFAULT_REGEX_RULES,
    SIMILARITY_THRESHOLD,
)
from database.db import (
    get_all_corrections,
    get_verified_transactions,
    normalize_merchant_key,
)
from llm_engine.hf_manager import LocalHFEngine

logger = logging.getLogger(__name__)

CategoryLiteral = Literal[
    "Housing",
    "Utilities",
    "Groceries",
    "Dining & Cafe",
    "Transport & Fuel",
    "Shopping & Retail",
    "Healthcare",
    "Subscriptions & Media",
    "Entertainment & Leisure",
    "Travel",
    "Miscellaneous",
    "Salary & Payroll",
    "Investments & Dividends",
    "Refunds & Transfers",
]


class SingleCategorySchema(BaseModel):
    category: CategoryLiteral = Field(description="Best matching financial category")
    is_recurring: bool = Field(description="Whether this merchant represents a recurring bill/subscription")


class HybridCategorizer:
    """
    3-Tier Privacy-First Local Categorizer (Scoped per User):
    Tier 1: User's personal `category_corrections` overrides + deterministic regex rules (0ms).
    Tier 2: Cosine similarity against the user's verified past transactions (`all-MiniLM-L6-v2` >= 0.78).
    Tier 3: Local Qwen2.5-3B-Instruct-GGUF with JSON Schema enforcement when similarity < 0.78.
    """

    def __init__(self, user_id: int) -> None:
        self.user_id = user_id
        self.hf_engine = LocalHFEngine()
        self._compiled_regex = [
            (re.compile(pattern, re.IGNORECASE), category)
            for pattern, category in DEFAULT_REGEX_RULES.items()
        ]

    def categorize_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not records:
            return []

        user_corrections = get_all_corrections(self.user_id)
        verified_txs = get_verified_transactions(self.user_id)

        verified_exemplars: List[tuple[str, str, bool]] = []
        seen_keys: set[str] = set()
        for tx in verified_txs:
            key = normalize_merchant_key(f"{tx.merchant} {tx.raw_description}")
            if key and key not in seen_keys:
                seen_keys.add(key)
                verified_exemplars.append((key, tx.category, tx.is_recurring))

        exemplar_embeddings: np.ndarray | None = None
        if verified_exemplars:
            exemplar_texts = [item[0] for item in verified_exemplars]
            exemplar_embeddings = self.hf_engine.encode_texts(exemplar_texts)

        unresolved_indices: List[int] = []

        # Tier 1: User Corrections Table & Deterministic Regex Rules
        for idx, row in enumerate(records):
            merchant_text = str(row.get("merchant") or "")
            raw_desc = str(row.get("raw_description") or merchant_text)
            combined_norm = normalize_merchant_key(f"{merchant_text} {raw_desc}")
            merchant_norm = normalize_merchant_key(merchant_text)

            matched_cat: str | None = None

            if merchant_norm in user_corrections:
                matched_cat = user_corrections[merchant_norm]
            else:
                for pattern_key, corr_cat in user_corrections.items():
                    if pattern_key and (pattern_key in combined_norm or combined_norm in pattern_key):
                        matched_cat = corr_cat
                        break

            if matched_cat is None:
                for regex, rule_cat in self._compiled_regex:
                    if regex.search(f"{merchant_text} {raw_desc}"):
                        matched_cat = rule_cat
                        break

            if matched_cat is not None:
                row["category"] = matched_cat
                row["categorization_tier"] = "Tier 1 (Rule/Override)"
                if matched_cat in ("Housing", "Utilities", "Subscriptions & Media"):
                    row["is_recurring"] = True
            else:
                unresolved_indices.append(idx)

        if not unresolved_indices:
            return records

        # Tier 2: Local SentenceTransformer Cosine Similarity (>= 0.78)
        llm_fallback_indices: List[int] = []
        if exemplar_embeddings is not None and len(exemplar_embeddings) > 0:
            query_texts = [
                normalize_merchant_key(
                    f"{records[i].get('merchant', '')} {records[i].get('raw_description', '')}"
                )
                for i in unresolved_indices
            ]
            query_embeddings = self.hf_engine.encode_texts(query_texts)
            sim_matrix = np.matmul(query_embeddings, exemplar_embeddings.T)

            for local_pos, record_idx in enumerate(unresolved_indices):
                best_exemplar_idx = int(np.argmax(sim_matrix[local_pos]))
                best_score = float(sim_matrix[local_pos, best_exemplar_idx])
                if best_score >= SIMILARITY_THRESHOLD:
                    _, matched_cat, matched_recurring = verified_exemplars[best_exemplar_idx]
                    records[record_idx]["category"] = matched_cat
                    records[record_idx]["is_recurring"] = bool(
                        records[record_idx].get("is_recurring") or matched_recurring
                    )
                    records[record_idx]["categorization_tier"] = f"Tier 2 (Semantic {best_score:.2f})"
                else:
                    llm_fallback_indices.append(record_idx)
        else:
            llm_fallback_indices = unresolved_indices

        # Tier 3: Local Qwen2.5-3B-Instruct GGUF Fallback
        system_prompt = (
            "You are a local financial transaction classifier. Classify the transaction into "
            f"exactly one of these valid categories: {', '.join(ALL_CATEGORIES)}."
        )
        for record_idx in llm_fallback_indices:
            row = records[record_idx]
            user_prompt = (
                f"Merchant: {row.get('merchant')}\n"
                f"Description: {row.get('raw_description')}\n"
                f"Amount: {row.get('amount')} ({row.get('tx_type')})"
            )
            try:
                prediction: SingleCategorySchema = self.hf_engine.generate_structured(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    schema_model=SingleCategorySchema,
                    temperature=0.0,
                    max_tokens=160,
                )
                row["category"] = prediction.category
                row["is_recurring"] = bool(row.get("is_recurring") or prediction.is_recurring)
                row["categorization_tier"] = "Tier 3 (Qwen2.5-3B GGUF)"
            except Exception as exc:
                logger.warning("Tier 3 LLM categorization fallback error (%s); defaulting category.", exc)
                row["category"] = (
                    "Salary & Payroll" if row.get("tx_type") == "credit" else "Miscellaneous"
                )
                row["categorization_tier"] = "Tier 3 Fallback (Default)"

        return records