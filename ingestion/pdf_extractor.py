from __future__ import annotations

import io
import logging
import re
from datetime import date, datetime
from typing import Any, Dict, List, Literal

import numpy as np
import pandas as pd
import pdfplumber
from PIL import Image
from pydantic import BaseModel, Field

from config import EXTRACTION_BATCH_SIZE, MIN_PAGE_TEXT_CHARS
from llm_engine.hf_manager import LocalHFEngine

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Schemas for Grammar-Constrained GGUF Extraction
# ============================================================================
class ExtractedTransactionItem(BaseModel):
    date: str = Field(
        description="Transaction date normalized to YYYY-MM-DD format"
    )
    merchant: str = Field(
        description="Clean merchant, UPI payee, or counterparty name without long reference IDs"
    )
    raw_description: str = Field(
        description="Original statement narration or line description"
    )
    amount: float = Field(
        description="Absolute positive numeric transaction amount (ignore running balance)"
    )
    tx_type: Literal["debit", "credit"] = Field(
        description="'debit' for withdrawals/DR/UPI-paid/purchases, 'credit' for deposits/CR/salary/refunds"
    )
    is_recurring: bool = Field(
        default=False,
        description="True if subscription, rent, utility, insurance, SIP, or repeating monthly bill",
    )


class BatchTransactionSchema(BaseModel):
    transactions: List[ExtractedTransactionItem] = Field(
        default_factory=list,
        description="List of parsed financial transactions extracted from the statement lines",
    )


# ============================================================================
# Two-Stage Document Parser (pdfplumber + RapidOCR ONNX Runtime Fallback)
# ============================================================================
class StatementExtractor:
    def __init__(self) -> None:
        self._ocr_engine = None
        self.hf_engine = LocalHFEngine()

    def _get_ocr_engine(self):
        if self._ocr_engine is None:
            try:
                from rapidocr import RapidOCR
            except ImportError:
                from rapidocr_onnxruntime import RapidOCR

            self._ocr_engine = RapidOCR()
        return self._ocr_engine

    def _ocr_image_array(self, img_array: np.ndarray) -> List[str]:
        ocr = self._get_ocr_engine()
        raw_out = ocr(img_array)

        if hasattr(raw_out, "boxes") and hasattr(raw_out, "txts"):
            if raw_out.boxes is None or raw_out.txts is None:
                return []
            scores = raw_out.scores if raw_out.scores is not None else [1.0] * len(raw_out.txts)
            triplets = list(zip(raw_out.boxes, raw_out.txts, scores))
        elif isinstance(raw_out, tuple):
            result = raw_out[0]
            if not result:
                return []
            triplets = result
        else:
            return []

        sorted_boxes = sorted(
            triplets,
            key=lambda item: (round(float(item[0][0][1]) / 12.0), float(item[0][0][0])),
        )
        lines: List[str] = []
        current_line_tokens: List[str] = []
        current_y_bucket: int | None = None

        for box, text, _score in sorted_boxes:
            y_bucket = round(float(box[0][1]) / 12.0)
            if current_y_bucket is None or abs(y_bucket - current_y_bucket) <= 1:
                current_line_tokens.append(str(text).strip())
                current_y_bucket = y_bucket
            else:
                lines.append(" ".join(current_line_tokens))
                current_line_tokens = [str(text).strip()]
                current_y_bucket = y_bucket

        if current_line_tokens:
            lines.append(" ".join(current_line_tokens))
        return lines

    def extract_raw_lines_from_pdf(self, file_bytes: bytes, password: str = "") -> List[str]:
        """
        Stage A: Native digital extraction with `pdfplumber` (tables + text lines).
        Stage B: Automatic fallback to `rapidocr` via `pypdfium2` on scanned pages.
        Handles encrypted bank statements if password is provided.
        """
        extracted_lines: List[str] = []
        scanned_page_indices: List[int] = []

        try:
            with pdfplumber.open(io.BytesIO(file_bytes), password=password or None) as pdf:
                for idx, page in enumerate(pdf.pages):
                    page_lines: List[str] = []

                    tables = page.extract_tables() or []
                    for table in tables:
                        for row in table:
                            if not row:
                                continue
                            cleaned_cells = [
                                re.sub(r"\s+", " ", str(cell).strip())
                                for cell in row
                                if cell is not None and str(cell).strip()
                            ]
                            if len(cleaned_cells) >= 2:
                                page_lines.append(" | ".join(cleaned_cells))

                    raw_text = (page.extract_text() or "").strip()
                    if len(raw_text) >= MIN_PAGE_TEXT_CHARS:
                        for line in raw_text.splitlines():
                            clean_line = re.sub(r"\s+", " ", line).strip()
                            if clean_line and clean_line not in page_lines:
                                page_lines.append(clean_line)
                        extracted_lines.extend(page_lines)
                    else:
                        scanned_page_indices.append(idx)

        except Exception as e:
            err_str = repr(e).lower() + " " + str(e).lower()
            if "password" in err_str or "encrypt" in err_str:
                raise ValueError(
                    "This PDF is password-protected. Please enter the correct PDF password below the file uploader."
                ) from e
            raise ValueError(f"Failed to read PDF: {str(e)}") from e

        if scanned_page_indices:
            import pypdfium2 as pdfium

            pdf_doc = pdfium.PdfDocument(file_bytes, password=password or None)
            for page_idx in scanned_page_indices:
                logger.info("Running Stage B RapidOCR fallback on scanned PDF page %d", page_idx + 1)
                page = pdf_doc[page_idx]
                bitmap = page.render(scale=2.0)
                pil_image = bitmap.to_pil()
                ocr_lines = self._ocr_image_array(np.array(pil_image))
                extracted_lines.extend(ocr_lines)

        return self._filter_candidate_transaction_lines(extracted_lines)

    def extract_raw_lines_from_image(self, file_bytes: bytes) -> List[str]:
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        ocr_lines = self._ocr_image_array(np.array(pil_img))
        return self._filter_candidate_transaction_lines(ocr_lines)

    @staticmethod
    def _filter_candidate_transaction_lines(lines: List[str]) -> List[str]:
        """
        Retains statement lines containing a currency amount (supports both Indian 1,25,000.00
        and standard 125,000.00 formats, as well as Rs./INR/₹/$ symbols).
        """
        amount_pattern = re.compile(
            r"(?:[-+$€£₹]|Rs\.?|INR)?\s*\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})\b",
            re.IGNORECASE,
        )
        ignore_header_pattern = re.compile(
            r"^(page\s+\d+|opening\s+balance|closing\s+balance|beginning\s+balance|ending\s+balance|statement\s+of\s+account|statement\s+period|account\s+summary|brought\s+forward|carried\s+forward)",
            re.IGNORECASE,
        )
        candidates: List[str] = []
        for line in lines:
            stripped = line.strip()
            if len(stripped) < 8 or ignore_header_pattern.search(stripped):
                continue
            if amount_pattern.search(stripped):
                candidates.append(stripped)
        return candidates

    # ------------------------------------------------------------------------
    # Grammar-Constrained Hugging Face Batch Parser (15 Lines / Batch)
    # ------------------------------------------------------------------------
    def parse_lines_with_hf_llm(self, candidate_lines: List[str]) -> List[Dict[str, Any]]:
        if not candidate_lines:
            return []

        system_prompt = (
            "You are a precision Indian & global bank statement parser. Extract every valid financial transaction "
            "from the provided statement lines. Normalize dates to YYYY-MM-DD (note: Indian bank statements often "
            f"use DD/MM/YYYY or DD-MM-YYYY; assume year {date.today().year} if omitted). "
            "For UPI narrations like 'UPI/SWIGGY/1234567/Pay', extract 'Swiggy' as the clean merchant. "
            "If a line shows both transaction amount and closing balance, extract ONLY the transaction amount (not the closing balance). "
            "Set tx_type to 'debit' for withdrawals/DR/paid or 'credit' for deposits/CR/received, "
            "and mark is_recurring=true for rent, utilities, subscriptions, or SIPs."
        )

        parsed_records: List[Dict[str, Any]] = []
        for start_idx in range(0, len(candidate_lines), EXTRACTION_BATCH_SIZE):
            batch = candidate_lines[start_idx : start_idx + EXTRACTION_BATCH_SIZE]
            numbered_lines = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(batch))
            user_prompt = f"Extract all transactions from these {len(batch)} bank statement lines:\n\n{numbered_lines}"

            try:
                structured_result: BatchTransactionSchema = self.hf_engine.generate_structured(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    schema_model=BatchTransactionSchema,
                    temperature=0.0,
                    max_tokens=1500,
                )
                for item in structured_result.transactions:
                    parsed_records.append(
                        {
                            "date": self._normalize_date_str(item.date),
                            "merchant": self._clean_upi_merchant(item.merchant),
                            "raw_description": item.raw_description.strip() or item.merchant.strip(),
                            "amount": round(abs(float(item.amount)), 2),
                            "tx_type": item.tx_type,
                            "is_recurring": bool(item.is_recurring),
                            "source": "ai_extracted",
                        }
                    )
            except Exception as exc:
                logger.warning("HF structured batch parse failed (%s); using deterministic line fallback.", exc)
                for line in batch:
                    fallback_item = self._deterministic_line_parse(line)
                    if fallback_item:
                        parsed_records.append(fallback_item)

        return parsed_records

    @staticmethod
    def _clean_upi_merchant(merchant: str) -> str:
        text = (merchant or "").strip()
        # Extract clean payee from UPI-style strings: UPI/PAYEE/REF/... or UPI-PAYEE-...
        upi_match = re.match(r"^UPI[/-](?:CR[/-]|DR[/-])?(?:\d+[/-])?([A-Za-z0-9\s.&]+?)(?:[/-]|$)", text, re.I)
        if upi_match and len(upi_match.group(1).strip()) >= 2:
            return upi_match.group(1).strip().title()
        return text or "Unknown Merchant"

    @staticmethod
    def _normalize_date_str(raw_date: str) -> str:
        raw_date = (raw_date or "").strip()
        # Prioritize DD/MM/YYYY and DD-MM-YYYY for Indian statements alongside ISO YYYY-MM-DD
        for fmt in (
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%d-%m-%Y",
            "%d/%m/%y",
            "%d-%m-%y",
            "%d %b %Y",
            "%d-%b-%Y",
            "%d %b %y",
            "%d-%b-%y",
            "%m/%d/%Y",
            "%m/%d/%y",
            "%b %d, %Y",
            "%b %d %Y",
        ):
            try:
                return datetime.strptime(raw_date, fmt).date().isoformat()
            except ValueError:
                continue
        m = re.match(r"^(\d{1,2})[/-](\d{1,2})$", raw_date)
        if m:
            first, second = int(m.group(1)), int(m.group(2))
            # Support DD/MM if first > 12, else DD/MM default
            day, month = (first, second) if second <= 12 else (second, first)
            try:
                return date(date.today().year, month, day).isoformat()
            except ValueError:
                pass
        return date.today().isoformat()

    @classmethod
    def _deterministic_line_parse(cls, line: str) -> Dict[str, Any] | None:
        date_match = re.search(
            r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{1,2}[\s-](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,-]*\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:,?\s+\d{4})?)\b",
            line,
            re.IGNORECASE,
        )
        # Match Indian (1,25,000.00) and Western (125,000.00) decimal amounts
        amount_matches = list(
            re.finditer(
                r"([-+]?)\s*(?:[$€£₹]|Rs\.?|INR)?\s*(\d{1,3}(?:,\d{2,3})*\.\d{2})\s*(Dr|Cr|DR|CR)?\b",
                line,
            )
        )
        if not amount_matches:
            return None

        # If a statement row has [Transaction Amount, Closing Balance], pick the penultimate amount
        target_amt = amount_matches[-2] if len(amount_matches) >= 2 else amount_matches[-1]
        sign_str = target_amt.group(1)
        val = float(target_amt.group(2).replace(",", ""))
        dr_cr_flag = (target_amt.group(3) or "").upper()

        parsed_date = cls._normalize_date_str(date_match.group(1)) if date_match else date.today().isoformat()

        desc = line
        if date_match:
            desc = desc.replace(date_match.group(0), " ")
        desc = desc[: target_amt.start()].replace("|", " ").strip()
        desc = re.sub(r"\s+", " ", desc).strip(" -|") or "Statement Entry"

        is_credit = (
            sign_str == "+"
            or dr_cr_flag == "CR"
            or bool(
                re.search(
                    r"\b(credit|deposit|payroll|salary|refund|dividend|neft\s*cr|imps\s*cr|ach\s*cr|by\s*transfer)\b",
                    line,
                    re.IGNORECASE,
                )
            )
        )
        if dr_cr_flag == "DR":
            is_credit = False

        return {
            "date": parsed_date,
            "merchant": cls._clean_upi_merchant(desc[:48]),
            "raw_description": line,
            "amount": round(abs(val), 2),
            "tx_type": "credit" if is_credit else "debit",
            "is_recurring": bool(
                re.search(
                    r"\b(netflix|spotify|hotstar|rent|electric|bescom|water|internet|airtel|jio|icloud|gym|sip)\b",
                    line,
                    re.IGNORECASE,
                )
            ),
            "source": "ai_extracted",
        }

    def parse_csv_bytes(self, file_bytes: bytes) -> List[Dict[str, Any]]:
        df = pd.read_csv(io.BytesIO(file_bytes))
        col_map = {str(c).lower().strip(): c for c in df.columns}

        date_col = next(
            (col_map[k] for k in col_map if any(w in k for w in ("date", "txn date", "value date"))),
            None,
        )
        desc_col = next(
            (
                col_map[k]
                for k in col_map
                if any(
                    w in k
                    for w in ("merchant", "description", "narration", "particulars", "remarks", "memo", "name", "payee")
                )
            ),
            None,
        )
        # Support Indian bank CSVs that split Withdrawal (Debit) and Deposit (Credit) into 2 columns
        withdrawal_col = next(
            (col_map[k] for k in col_map if any(w in k for w in ("withdrawal", "debit", "dr amt"))),
            None,
        )
        deposit_col = next(
            (col_map[k] for k in col_map if any(w in k for w in ("deposit", "credit", "cr amt"))),
            None,
        )
        amt_col = next(
            (col_map[k] for k in col_map if any(w in k for w in ("amount", "value", "total", "txn amt"))),
            None,
        )
        type_col = next(
            (col_map[k] for k in col_map if any(w in k for w in ("type", "dr/cr", "cr/dr"))),
            None,
        )

        if not (date_col and desc_col and (amt_col or withdrawal_col or deposit_col)):
            lines = [
                " | ".join(str(v) for v in row.values if pd.notna(v))
                for _, row in df.iterrows()
            ]
            return self.parse_lines_with_hf_llm(lines)

        records: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            desc_str = str(row[desc_col]).strip() if pd.notna(row[desc_col]) else "Statement Entry"
            numeric_amt = 0.0
            tx_type = "debit"

            if withdrawal_col and deposit_col and amt_col is None:
                w_val = self._parse_currency_number(row.get(withdrawal_col))
                d_val = self._parse_currency_number(row.get(deposit_col))
                if w_val > 0:
                    numeric_amt = w_val
                    tx_type = "debit"
                elif d_val > 0:
                    numeric_amt = d_val
                    tx_type = "credit"
                else:
                    continue
            else:
                target_col = amt_col or withdrawal_col or deposit_col
                numeric_amt = self._parse_currency_number(row.get(target_col))
                if numeric_amt == 0.0:
                    continue
                if type_col and pd.notna(row[type_col]):
                    t_str = str(row[type_col]).lower().strip()
                    tx_type = "credit" if ("cr" in t_str or "cred" in t_str or "dep" in t_str) else "debit"
                else:
                    tx_type = (
                        "credit"
                        if numeric_amt > 0
                        and re.search(r"\b(payroll|salary|deposit|refund|dividend|neft\s*cr|imps\s*cr)\b", desc_str, re.I)
                        else "debit"
                    )

            records.append(
                {
                    "date": self._normalize_date_str(str(row[date_col])),
                    "merchant": self._clean_upi_merchant(re.sub(r"\s+", " ", desc_str)[:50]),
                    "raw_description": desc_str,
                    "amount": round(abs(numeric_amt), 2),
                    "tx_type": tx_type,
                    "is_recurring": False,
                    "source": "ai_extracted",
                }
            )
        return records

    @staticmethod
    def _parse_currency_number(val: Any) -> float:
        if val is None or pd.isna(val):
            return 0.0
        cleaned = re.sub(r"[^0-9.-]", "", str(val))
        if not cleaned or cleaned in ("-", "."):
            return 0.0
        try:
            return abs(float(cleaned))
        except ValueError:
            return 0.0