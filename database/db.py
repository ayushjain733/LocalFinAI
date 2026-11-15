from __future__ import annotations

import hashlib
import hmac
import io
import json
import re
import secrets
import zipfile
from contextlib import contextmanager
from datetime import date, datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Tuple

import pandas as pd
from sqlalchemy import delete as sql_delete, event, text
from sqlmodel import Session, SQLModel, create_engine, select

from config import (
    DATABASE_URL,
    DEFAULT_CURRENCY_LABEL,
    ESSENTIAL_CATEGORIES,
    RL_MODEL_DIR,
    SUPPORTED_CURRENCIES,
)
from database.models import CategoryCorrection, MonthlyForecast, Transaction, User

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.close()


def init_db() -> None:
    """
    Initializes the multi-user SQLite schema and non-destructively migrates
    existing `users` tables to include offline account recovery columns.
    """
    with engine.begin() as conn:
        tx_table_check = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='transactions';")
        ).fetchone()
        if tx_table_check is not None:
            tx_cols = [
                row[1]
                for row in conn.execute(text("PRAGMA table_info(transactions);")).fetchall()
            ]
            if "user_id" not in tx_cols:
                conn.execute(text("DROP TABLE IF EXISTS transactions;"))
                conn.execute(text("DROP TABLE IF EXISTS category_corrections;"))
                conn.execute(text("DROP TABLE IF EXISTS monthly_forecasts;"))

        user_table_check = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='users';")
        ).fetchone()
        if user_table_check is not None:
            user_cols = [
                row[1]
                for row in conn.execute(text("PRAGMA table_info(users);")).fetchall()
            ]
            if "security_question" not in user_cols:
                conn.execute(text("ALTER TABLE users ADD COLUMN security_question TEXT;"))
            if "security_answer_hash" not in user_cols:
                conn.execute(text("ALTER TABLE users ADD COLUMN security_answer_hash TEXT;"))
            if "recovery_key_hash" not in user_cols:
                conn.execute(text("ALTER TABLE users ADD COLUMN recovery_key_hash TEXT;"))

    SQLModel.metadata.create_all(engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


# ============================================================================
# Local User Authentication, Offline Recovery & Account Deletion
# ============================================================================
def _hash_secret(secret_value: str, salt_hex: str) -> str:
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        secret_value.encode("utf-8"),
        bytes.fromhex(salt_hex),
        120_000,
    )
    return dk.hex()


def _normalize_security_answer(answer: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]", "", (answer or "").strip().lower())
    return cleaned


def _normalize_recovery_key(key_str: str) -> str:
    cleaned = re.sub(r"[^A-Z0-9]", "", (key_str or "").strip().upper())
    return cleaned


def _generate_recovery_key() -> str:
    alphabet = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    blocks = [
        "".join(secrets.choice(alphabet) for _ in range(4))
        for _ in range(3)
    ]
    return f"LFAI-{'-'.join(blocks)}"


def create_user(
    username: str,
    password: str,
    display_name: str = "",
    preferred_currency: str = DEFAULT_CURRENCY_LABEL,
    security_question: str = "",
    security_answer: str = "",
) -> Tuple[User, str]:
    clean_user = re.sub(r"\s+", "", (username or "").strip().lower())
    if len(clean_user) < 3:
        raise ValueError("Username must be at least 3 characters long (no spaces).")
    if len(password or "") < 6:
        raise ValueError("Password must be at least 6 characters long.")

    norm_answer = _normalize_security_answer(security_answer)
    if not security_question or len(norm_answer) < 2:
        raise ValueError(
            "Please select a security question and provide an answer (at least 2 alphanumeric characters) for account recovery."
        )

    clean_display = (display_name or clean_user).strip()

    with get_session() as session:
        existing = session.exec(
            select(User).where(User.username == clean_user)
        ).first()
        if existing is not None:
            raise ValueError(f"Username '{clean_user}' is already registered.")

        salt_hex = secrets.token_hex(16)
        pw_hash = _hash_secret(password, salt_hex)

        raw_recovery_key = _generate_recovery_key()
        norm_recovery_key = _normalize_recovery_key(raw_recovery_key)

        ans_hash = _hash_secret(norm_answer, salt_hex)
        rec_hash = _hash_secret(norm_recovery_key, salt_hex)

        new_user = User(
            username=clean_user,
            display_name=clean_display,
            password_hash=pw_hash,
            salt=salt_hex,
            preferred_currency=preferred_currency,
            security_question=security_question.strip(),
            security_answer_hash=ans_hash,
            recovery_key_hash=rec_hash,
        )
        session.add(new_user)
        session.commit()
        session.refresh(new_user)
        return new_user, raw_recovery_key


def authenticate_user(username: str, password: str) -> Optional[User]:
    clean_user = re.sub(r"\s+", "", (username or "").strip().lower())
    if not clean_user or not password:
        return None

    with get_session() as session:
        user = session.exec(
            select(User).where(User.username == clean_user)
        ).first()
        if user is None:
            return None

        candidate_hash = _hash_secret(password, user.salt)
        if hmac.compare_digest(candidate_hash, user.password_hash):
            return user
        return None


def get_user_recovery_profile(username: str) -> Optional[Dict[str, Any]]:
    clean_user = re.sub(r"\s+", "", (username or "").strip().lower())
    if not clean_user:
        return None

    with get_session() as session:
        user = session.exec(
            select(User).where(User.username == clean_user)
        ).first()
        if user is None:
            return None

        return {
            "username": user.username,
            "display_name": user.display_name,
            "security_question": user.security_question,
            "has_security_question": bool(user.security_question and user.security_answer_hash),
            "has_recovery_key": bool(user.recovery_key_hash),
        }


def reset_user_password_offline(
    username: str,
    new_password: str,
    verification_method: str,
    verification_input: str,
) -> User:
    clean_user = re.sub(r"\s+", "", (username or "").strip().lower())
    if len(new_password or "") < 6:
        raise ValueError("New password must be at least 6 characters long.")

    with get_session() as session:
        user = session.exec(
            select(User).where(User.username == clean_user)
        ).first()
        if user is None:
            raise ValueError("Account not found.")

        if verification_method == "recovery_key":
            if not user.recovery_key_hash:
                raise ValueError("No Master Recovery Key is configured on this account.")
            norm_key = _normalize_recovery_key(verification_input)
            if not norm_key:
                raise ValueError("Please enter your Master Recovery Key.")
            candidate_hash = _hash_secret(norm_key, user.salt)
            if not hmac.compare_digest(candidate_hash, user.recovery_key_hash):
                raise ValueError("Invalid Master Recovery Key. Please check the code and try again.")

        elif verification_method == "security_question":
            if not user.security_answer_hash:
                raise ValueError("No Security Question is configured on this account.")
            norm_ans = _normalize_security_answer(verification_input)
            if not norm_ans:
                raise ValueError("Please enter your security answer.")
            candidate_hash = _hash_secret(norm_ans, user.salt)
            if not hmac.compare_digest(candidate_hash, user.security_answer_hash):
                raise ValueError("Incorrect answer to the security question.")
        else:
            raise ValueError("Unsupported verification method.")

        user.password_hash = _hash_secret(new_password, user.salt)
        session.add(user)
        session.commit()
        session.refresh(user)
        return user


def configure_or_rotate_recovery_credentials(
    user_id: int,
    security_question: str,
    security_answer: str,
) -> str:
    norm_answer = _normalize_security_answer(security_answer)
    if not security_question or len(norm_answer) < 2:
        raise ValueError("Please provide a valid security answer (at least 2 alphanumeric characters).")

    with get_session() as session:
        user = session.get(User, user_id)
        if user is None:
            raise ValueError("User not found.")

        raw_recovery_key = _generate_recovery_key()
        norm_key = _normalize_recovery_key(raw_recovery_key)

        user.security_question = security_question.strip()
        user.security_answer_hash = _hash_secret(norm_answer, user.salt)
        user.recovery_key_hash = _hash_secret(norm_key, user.salt)
        session.add(user)
        session.commit()
        return raw_recovery_key


def delete_user_account(user_id: int, password: str) -> Dict[str, Any]:
    if not password:
        raise ValueError("Please enter your password to confirm account deletion.")

    with get_session() as session:
        user = session.get(User, user_id)
        if user is None:
            raise ValueError("User account not found.")

        candidate_hash = _hash_secret(password, user.salt)
        if not hmac.compare_digest(candidate_hash, user.password_hash):
            raise ValueError("Incorrect password. Account deletion aborted.")

        username = user.username
        tx_count = len(
            session.exec(select(Transaction).where(Transaction.user_id == user_id)).all()
        )
        corr_count = len(
            session.exec(select(CategoryCorrection).where(CategoryCorrection.user_id == user_id)).all()
        )
        fc_count = len(
            session.exec(select(MonthlyForecast).where(MonthlyForecast.user_id == user_id)).all()
        )

        session.exec(sql_delete(Transaction).where(Transaction.user_id == user_id))
        session.exec(sql_delete(CategoryCorrection).where(CategoryCorrection.user_id == user_id))
        session.exec(sql_delete(MonthlyForecast).where(MonthlyForecast.user_id == user_id))
        session.delete(user)
        session.commit()

    rl_checkpoint = RL_MODEL_DIR / f"ppo_expense_user_{user_id}.zip"
    rl_removed = False
    if rl_checkpoint.exists():
        try:
            rl_checkpoint.unlink()
            rl_removed = True
        except Exception:
            pass

    return {
        "username": username,
        "deleted_transactions": tx_count,
        "deleted_rules": corr_count,
        "deleted_forecasts": fc_count,
        "deleted_rl_model": rl_removed,
    }


def update_user_currency(user_id: int, currency_label: str) -> None:
    with get_session() as session:
        user = session.get(User, user_id)
        if user is not None and user.preferred_currency != currency_label:
            user.preferred_currency = currency_label
            session.add(user)
            session.commit()


# ============================================================================
# Full Account Backup (Export) & Restore (Import)
# ============================================================================
def export_user_backup_bytes(user_id: int) -> bytes:
    """
    Exports a user's complete workspace into an in-memory `.zip` archive containing:
    - `workspace_data.json`: All verified transactions, learned merchant category
      rules, monthly RL forecasts, and account currency preference.
    - `rl_agent/ppo_expense_model.zip`: The user's trained PPO policy checkpoint (if present).
    """
    with get_session() as session:
        user = session.get(User, user_id)
        if user is None:
            raise ValueError("User not found.")

        txs = session.exec(
            select(Transaction)
            .where(Transaction.user_id == user_id)
            .order_by(Transaction.date.asc(), Transaction.id.asc())
        ).all()
        corrs = session.exec(
            select(CategoryCorrection)
            .where(CategoryCorrection.user_id == user_id)
            .order_by(CategoryCorrection.raw_merchant_pattern.asc())
        ).all()
        forecasts = session.exec(
            select(MonthlyForecast)
            .where(MonthlyForecast.user_id == user_id)
            .order_by(MonthlyForecast.year_month.asc(), MonthlyForecast.category.asc())
        ).all()

        payload = {
            "schema_version": "1.0",
            "app": "LocalFinAI",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "account": {
                "username": user.username,
                "display_name": user.display_name,
                "preferred_currency": user.preferred_currency,
            },
            "transactions": [
                {
                    "date": tx.date.isoformat(),
                    "merchant": tx.merchant,
                    "raw_description": tx.raw_description,
                    "amount": round(float(tx.amount), 2),
                    "tx_type": tx.tx_type,
                    "category": tx.category,
                    "is_recurring": bool(tx.is_recurring),
                    "source": tx.source,
                    "verified_by_user": bool(tx.verified_by_user),
                }
                for tx in txs
            ],
            "category_corrections": [
                {
                    "raw_merchant_pattern": c.raw_merchant_pattern,
                    "corrected_category": c.corrected_category,
                }
                for c in corrs
            ],
            "monthly_forecasts": [
                {
                    "year_month": f.year_month,
                    "category": f.category,
                    "baseline_spend": round(float(f.baseline_spend), 2),
                    "rl_multiplier": round(float(f.rl_multiplier), 4),
                    "predicted_spend": round(float(f.predicted_spend), 2),
                    "actual_spend": round(float(f.actual_spend), 2)
                    if f.actual_spend is not None
                    else None,
                    "reward": round(float(f.reward), 4) if f.reward is not None else None,
                }
                for f in forecasts
            ],
        }

    buffer = io.BytesIO()
    rl_checkpoint = RL_MODEL_DIR / f"ppo_expense_user_{user_id}.zip"

    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("workspace_data.json", json.dumps(payload, indent=2))
        if rl_checkpoint.exists():
            zf.write(rl_checkpoint, arcname="rl_agent/ppo_expense_model.zip")

    return buffer.getvalue()


def restore_user_backup_bytes(
    user_id: int,
    file_bytes: bytes,
    replace_existing: bool = False,
) -> Dict[str, Any]:
    """
    Restores a user's workspace from a `.zip` or `.json` backup file.
    - When `replace_existing=False` (Merge Mode), deduplicates transactions and upserts
      merchant rules and monthly forecasts.
    - When `replace_existing=True` (Replace Mode), wipes existing workspace records
      before importing the backup.
    """
    if not file_bytes:
        raise ValueError("Uploaded backup file is empty.")

    rl_model_bytes: Optional[bytes] = None
    try:
        if zipfile.is_zipfile(io.BytesIO(file_bytes)):
            with zipfile.ZipFile(io.BytesIO(file_bytes), mode="r") as zf:
                names = zf.namelist()
                json_entry = next(
                    (n for n in names if n.endswith("workspace_data.json") or n.endswith(".json")),
                    None,
                )
                if json_entry is None:
                    raise ValueError("No `workspace_data.json` manifest found inside the ZIP archive.")
                raw_json = zf.read(json_entry).decode("utf-8")
                payload = json.loads(raw_json)

                if "rl_agent/ppo_expense_model.zip" in names:
                    rl_model_bytes = zf.read("rl_agent/ppo_expense_model.zip")
        else:
            payload = json.loads(file_bytes.decode("utf-8"))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Invalid backup archive or JSON format: {exc}") from exc

    if not isinstance(payload, dict) or "transactions" not in payload:
        raise ValueError("Backup file is missing required `transactions` data.")

    tx_list = payload.get("transactions") or []
    corr_list = payload.get("category_corrections") or []
    fc_list = payload.get("monthly_forecasts") or []
    account_meta = payload.get("account") or {}

    restored_tx = 0
    skipped_duplicates = 0
    restored_rules = 0
    restored_forecasts = 0
    restored_currency: Optional[str] = None

    with get_session() as session:
        user = session.get(User, user_id)
        if user is None:
            raise ValueError("Target user account not found.")

        pref_curr = account_meta.get("preferred_currency")
        if pref_curr and pref_curr in SUPPORTED_CURRENCIES:
            user.preferred_currency = pref_curr
            restored_currency = pref_curr
            session.add(user)

        if replace_existing:
            session.exec(sql_delete(Transaction).where(Transaction.user_id == user_id))
            session.exec(sql_delete(CategoryCorrection).where(CategoryCorrection.user_id == user_id))
            session.exec(sql_delete(MonthlyForecast).where(MonthlyForecast.user_id == user_id))
            session.flush()

        # Build deduplication fingerprint set for existing transactions in Merge Mode
        existing_fingerprints: set[Tuple[str, str, float, str, str]] = set()
        if not replace_existing:
            current_txs = session.exec(
                select(Transaction).where(Transaction.user_id == user_id)
            ).all()
            for etx in current_txs:
                existing_fingerprints.add(
                    (
                        etx.date.isoformat(),
                        etx.merchant.strip().lower(),
                        round(float(etx.amount), 2),
                        etx.tx_type.lower(),
                        etx.raw_description.strip().lower(),
                    )
                )

        # 1. Restore Transactions
        for item in tx_list:
            parsed_date = _parse_date_safe(item.get("date"))
            merchant = str(item.get("merchant") or "Unknown Merchant").strip()
            raw_desc = str(item.get("raw_description") or merchant).strip()
            try:
                amount = round(abs(float(item.get("amount") or 0.0)), 2)
            except (ValueError, TypeError):
                continue
            if amount <= 0.0:
                continue

            tx_type = str(item.get("tx_type") or "debit").lower().strip()
            if tx_type not in ("debit", "credit"):
                tx_type = "debit"

            category = str(item.get("category") or "Miscellaneous").strip()
            is_recurring = bool(item.get("is_recurring", False))
            source = str(item.get("source") or "manual").strip()
            if source not in ("ai_extracted", "manual"):
                source = "manual"

            fp = (
                parsed_date.isoformat(),
                merchant.lower(),
                amount,
                tx_type,
                raw_desc.lower(),
            )
            if not replace_existing and fp in existing_fingerprints:
                skipped_duplicates += 1
                continue

            existing_fingerprints.add(fp)
            session.add(
                Transaction(
                    user_id=user_id,
                    date=parsed_date,
                    merchant=merchant,
                    raw_description=raw_desc,
                    amount=amount,
                    tx_type=tx_type,
                    category=category,
                    is_recurring=is_recurring,
                    source=source,
                    verified_by_user=bool(item.get("verified_by_user", True)),
                )
            )
            restored_tx += 1

        # 2. Restore Category Corrections
        existing_corrs = {
            c.raw_merchant_pattern: c
            for c in session.exec(
                select(CategoryCorrection).where(CategoryCorrection.user_id == user_id)
            ).all()
        }
        for c_item in corr_list:
            pattern = normalize_merchant_key(str(c_item.get("raw_merchant_pattern") or ""))
            cat = str(c_item.get("corrected_category") or "").strip()
            if len(pattern) < 2 or not cat:
                continue
            if pattern in existing_corrs:
                existing_corrs[pattern].corrected_category = cat
                existing_corrs[pattern].updated_at = datetime.now(timezone.utc)
                session.add(existing_corrs[pattern])
            else:
                new_c = CategoryCorrection(
                    user_id=user_id,
                    raw_merchant_pattern=pattern,
                    corrected_category=cat,
                )
                session.add(new_c)
                existing_corrs[pattern] = new_c
            restored_rules += 1

        # 3. Restore Monthly Forecasts
        existing_fcs = {
            (f.year_month, f.category): f
            for f in session.exec(
                select(MonthlyForecast).where(MonthlyForecast.user_id == user_id)
            ).all()
        }
        for f_item in fc_list:
            ym = str(f_item.get("year_month") or "").strip()
            cat = str(f_item.get("category") or "").strip()
            if not ym or not cat:
                continue
            baseline = round(float(f_item.get("baseline_spend") or 0.0), 2)
            mult = round(float(f_item.get("rl_multiplier") or 0.0), 4)
            pred = round(float(f_item.get("predicted_spend") or 0.0), 2)

            key = (ym, cat)
            if key in existing_fcs:
                fc_obj = existing_fcs[key]
                fc_obj.baseline_spend = baseline
                fc_obj.rl_multiplier = mult
                fc_obj.predicted_spend = pred
                fc_obj.updated_at = datetime.now(timezone.utc)
                session.add(fc_obj)
            else:
                new_fc = MonthlyForecast(
                    user_id=user_id,
                    year_month=ym,
                    category=cat,
                    baseline_spend=baseline,
                    rl_multiplier=mult,
                    predicted_spend=pred,
                )
                session.add(new_fc)
                existing_fcs[key] = new_fc
            restored_forecasts += 1

        session.commit()

    # 4. Restore PPO RL Model Checkpoint if included in the ZIP archive
    restored_rl_model = False
    if rl_model_bytes:
        rl_checkpoint_path = RL_MODEL_DIR / f"ppo_expense_user_{user_id}.zip"
        try:
            rl_checkpoint_path.write_bytes(rl_model_bytes)
            restored_rl_model = True
        except Exception:
            restored_rl_model = False

    sync_forecasts_with_actuals(user_id)

    return {
        "restored_transactions": restored_tx,
        "skipped_duplicates": skipped_duplicates,
        "restored_rules": restored_rules,
        "restored_forecasts": restored_forecasts,
        "restored_rl_model": restored_rl_model,
        "preferred_currency": restored_currency,
    }


# ============================================================================
# Helpers & Per-User Category Corrections
# ============================================================================
def normalize_merchant_key(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s&]", " ", (text or "").lower())
    cleaned = re.sub(r"\b\d{3,}\b", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _parse_date_safe(raw_date: Any) -> date:
    if isinstance(raw_date, datetime):
        return raw_date.date()
    if isinstance(raw_date, date):
        return raw_date
    if isinstance(raw_date, pd.Timestamp):
        return raw_date.date()
    if isinstance(raw_date, str) and raw_date.strip():
        try:
            return date.fromisoformat(raw_date.strip()[:10])
        except ValueError:
            try:
                return pd.to_datetime(raw_date).date()
            except Exception:
                return date.today()
    return date.today()


def upsert_category_correction(user_id: int, merchant_pattern: str, corrected_category: str) -> None:
    norm_key = normalize_merchant_key(merchant_pattern)
    if len(norm_key) < 2:
        return

    with get_session() as session:
        existing = session.exec(
            select(CategoryCorrection).where(
                CategoryCorrection.user_id == user_id,
                CategoryCorrection.raw_merchant_pattern == norm_key,
            )
        ).first()
        if existing:
            existing.corrected_category = corrected_category
            existing.updated_at = datetime.now(timezone.utc)
            session.add(existing)
        else:
            session.add(
                CategoryCorrection(
                    user_id=user_id,
                    raw_merchant_pattern=norm_key,
                    corrected_category=corrected_category,
                )
            )
        session.commit()


def get_all_corrections(user_id: int) -> Dict[str, str]:
    with get_session() as session:
        rows = session.exec(
            select(CategoryCorrection).where(CategoryCorrection.user_id == user_id)
        ).all()
        return {r.raw_merchant_pattern: r.corrected_category for r in rows}


# ============================================================================
# Per-User Transaction Staging Commit & Full Ledger CRUD
# ============================================================================
def commit_verified_transactions(
    user_id: int,
    staged_rows: List[Dict[str, Any]],
    initial_ai_categories: Optional[Dict[int, str]] = None,
) -> Dict[str, int]:
    inserted_count = 0
    correction_count = 0

    with get_session() as session:
        for idx, row in enumerate(staged_rows):
            if row.get("delete_row") is True:
                continue

            parsed_date = _parse_date_safe(row.get("date"))
            merchant = str(row.get("merchant") or "Unknown Merchant").strip()
            raw_desc = str(row.get("raw_description") or merchant).strip()
            try:
                amount = abs(float(row.get("amount") or 0.0))
            except (ValueError, TypeError):
                amount = 0.0
            if amount <= 0.0:
                continue

            tx_type = str(row.get("tx_type") or "debit").lower().strip()
            if tx_type not in ("debit", "credit"):
                tx_type = "debit"

            category = str(row.get("category") or "Miscellaneous").strip()
            is_recurring = bool(
                False if pd.isna(row.get("is_recurring")) else row.get("is_recurring", False)
            )
            source = str(row.get("source") or "manual").strip()

            tx = Transaction(
                user_id=user_id,
                date=parsed_date,
                merchant=merchant,
                raw_description=raw_desc,
                amount=round(amount, 2),
                tx_type=tx_type,
                category=category,
                is_recurring=is_recurring,
                source=source if source in ("ai_extracted", "manual") else "manual",
                verified_by_user=True,
            )
            session.add(tx)
            inserted_count += 1

            original_cat = (initial_ai_categories or {}).get(idx)
            if (original_cat and original_cat != category) or source == "manual":
                norm_key = normalize_merchant_key(merchant)
                if len(norm_key) >= 2:
                    existing_corr = session.exec(
                        select(CategoryCorrection).where(
                            CategoryCorrection.user_id == user_id,
                            CategoryCorrection.raw_merchant_pattern == norm_key,
                        )
                    ).first()
                    if existing_corr:
                        existing_corr.corrected_category = category
                        existing_corr.updated_at = datetime.now(timezone.utc)
                        session.add(existing_corr)
                    else:
                        session.add(
                            CategoryCorrection(
                                user_id=user_id,
                                raw_merchant_pattern=norm_key,
                                corrected_category=category,
                            )
                        )
                    correction_count += 1

        session.commit()

    sync_forecasts_with_actuals(user_id)
    return {"inserted": inserted_count, "learned_corrections": correction_count}


def sync_ledger_crud(
    user_id: int,
    edited_df: pd.DataFrame,
    visible_ids: Optional[List[int]] = None,
) -> Dict[str, int]:
    inserted = 0
    updated = 0
    deleted = 0
    learned = 0

    with get_session() as session:
        existing_txs = session.exec(
            select(Transaction).where(
                Transaction.user_id == user_id,
                Transaction.verified_by_user == True,  # noqa: E712
            )
        ).all()
        existing_map: Dict[int, Transaction] = {
            tx.id: tx for tx in existing_txs if tx.id is not None
        }
        scope_ids = set(visible_ids) if visible_ids is not None else set(existing_map.keys())

        kept_ids: set[int] = set()
        rows = edited_df.to_dict(orient="records") if not edited_df.empty else []

        for row in rows:
            raw_id = row.get("id")
            has_valid_id = (
                raw_id is not None
                and not pd.isna(raw_id)
                and str(raw_id).strip() != ""
            )
            marked_for_delete = bool(
                False if pd.isna(row.get("delete_row")) else row.get("delete_row", False)
            )

            if has_valid_id:
                tx_id = int(float(raw_id))
                if marked_for_delete:
                    if tx_id in existing_map:
                        session.delete(existing_map[tx_id])
                        deleted += 1
                    continue

                kept_ids.add(tx_id)
                if tx_id in existing_map:
                    tx_obj = existing_map[tx_id]
                    new_date = _parse_date_safe(row.get("date"))
                    new_merchant = str(row.get("merchant") or "Unknown Merchant").strip()
                    new_raw_desc = str(row.get("raw_description") or new_merchant).strip()
                    try:
                        new_amount = round(abs(float(row.get("amount") or 0.0)), 2)
                    except (ValueError, TypeError):
                        new_amount = tx_obj.amount

                    new_tx_type = str(row.get("tx_type") or "debit").lower().strip()
                    if new_tx_type not in ("debit", "credit"):
                        new_tx_type = "debit"

                    new_cat = str(row.get("category") or "Miscellaneous").strip()
                    new_rec = bool(
                        False if pd.isna(row.get("is_recurring")) else row.get("is_recurring", False)
                    )
                    new_source = str(row.get("source") or tx_obj.source).strip()
                    if new_source not in ("ai_extracted", "manual"):
                        new_source = "manual"

                    category_changed = tx_obj.category != new_cat
                    row_changed = (
                        tx_obj.date != new_date
                        or tx_obj.merchant != new_merchant
                        or tx_obj.raw_description != new_raw_desc
                        or round(tx_obj.amount, 2) != new_amount
                        or tx_obj.tx_type != new_tx_type
                        or category_changed
                        or tx_obj.is_recurring != new_rec
                        or tx_obj.source != new_source
                    )

                    if row_changed:
                        tx_obj.date = new_date
                        tx_obj.merchant = new_merchant
                        tx_obj.raw_description = new_raw_desc
                        tx_obj.amount = new_amount
                        tx_obj.tx_type = new_tx_type
                        tx_obj.category = new_cat
                        tx_obj.is_recurring = new_rec
                        tx_obj.source = new_source
                        session.add(tx_obj)
                        updated += 1

                    if category_changed:
                        norm_key = normalize_merchant_key(new_merchant)
                        if len(norm_key) >= 2:
                            existing_corr = session.exec(
                                select(CategoryCorrection).where(
                                    CategoryCorrection.user_id == user_id,
                                    CategoryCorrection.raw_merchant_pattern == norm_key,
                                )
                            ).first()
                            if existing_corr:
                                existing_corr.corrected_category = new_cat
                                existing_corr.updated_at = datetime.now(timezone.utc)
                                session.add(existing_corr)
                            else:
                                session.add(
                                    CategoryCorrection(
                                        user_id=user_id,
                                        raw_merchant_pattern=norm_key,
                                        corrected_category=new_cat,
                                    )
                                )
                            learned += 1
            else:
                if marked_for_delete:
                    continue
                try:
                    new_amount = round(abs(float(row.get("amount") or 0.0)), 2)
                except (ValueError, TypeError):
                    new_amount = 0.0
                if new_amount <= 0.0:
                    continue

                new_date = _parse_date_safe(row.get("date"))
                new_merchant = str(row.get("merchant") or "Manual Entry").strip()
                new_raw_desc = str(row.get("raw_description") or new_merchant).strip()
                new_tx_type = str(row.get("tx_type") or "debit").lower().strip()
                if new_tx_type not in ("debit", "credit"):
                    new_tx_type = "debit"
                new_cat = str(row.get("category") or "Miscellaneous").strip()
                new_rec = bool(
                    False if pd.isna(row.get("is_recurring")) else row.get("is_recurring", False)
                )

                new_tx = Transaction(
                    user_id=user_id,
                    date=new_date,
                    merchant=new_merchant,
                    raw_description=new_raw_desc,
                    amount=new_amount,
                    tx_type=new_tx_type,
                    category=new_cat,
                    is_recurring=new_rec,
                    source="manual",
                    verified_by_user=True,
                )
                session.add(new_tx)
                inserted += 1

        for tx_id in scope_ids:
            if tx_id not in kept_ids and tx_id in existing_map:
                tx_obj = existing_map[tx_id]
                if tx_obj in session:
                    session.delete(tx_obj)
                    deleted += 1

        session.commit()

    sync_forecasts_with_actuals(user_id)
    return {
        "inserted": inserted,
        "updated": updated,
        "deleted": deleted,
        "learned": learned,
    }


def clear_all_transactions(user_id: int) -> int:
    with get_session() as session:
        txs = session.exec(
            select(Transaction).where(Transaction.user_id == user_id)
        ).all()
        count = len(txs)
        session.exec(
            sql_delete(Transaction).where(Transaction.user_id == user_id)
        )
        session.exec(
            sql_delete(MonthlyForecast).where(MonthlyForecast.user_id == user_id)
        )
        session.commit()
        return count


def get_verified_transactions(
    user_id: int,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> List[Transaction]:
    with get_session() as session:
        stmt = select(Transaction).where(
            Transaction.user_id == user_id,
            Transaction.verified_by_user == True,  # noqa: E712
        )
        if start_date:
            stmt = stmt.where(Transaction.date >= start_date)
        if end_date:
            stmt = stmt.where(Transaction.date <= end_date)
        stmt = stmt.order_by(Transaction.date.desc(), Transaction.id.desc())
        return list(session.exec(stmt).all())


def get_transactions_dataframe(user_id: int, verified_only: bool = True) -> pd.DataFrame:
    with get_session() as session:
        stmt = select(Transaction).where(Transaction.user_id == user_id)
        if verified_only:
            stmt = stmt.where(Transaction.verified_by_user == True)  # noqa: E712
        stmt = stmt.order_by(Transaction.date.desc(), Transaction.id.desc())
        rows = session.exec(stmt).all()

    if not rows:
        return pd.DataFrame(
            columns=[
                "id",
                "user_id",
                "date",
                "merchant",
                "raw_description",
                "amount",
                "tx_type",
                "category",
                "is_recurring",
                "source",
                "verified_by_user",
            ]
        )

    records = [r.model_dump() for r in rows]
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["year_month"] = df["date"].dt.strftime("%Y-%m")
    return df


# ============================================================================
# Per-User RL Forecasts & Actuals Synchronization
# ============================================================================
def compute_rl_reward(category: str, predicted_spend: float, actual_spend: float) -> float:
    base_error = abs(predicted_spend - actual_spend) / max(actual_spend, 1.0)
    if category in ESSENTIAL_CATEGORIES and predicted_spend < actual_spend:
        base_error *= 1.5
    return -float(round(base_error, 4))


def save_monthly_forecast(
    user_id: int,
    year_month: str,
    category: str,
    baseline_spend: float,
    rl_multiplier: float,
    predicted_spend: float,
) -> None:
    with get_session() as session:
        existing = session.exec(
            select(MonthlyForecast).where(
                MonthlyForecast.user_id == user_id,
                MonthlyForecast.year_month == year_month,
                MonthlyForecast.category == category,
            )
        ).first()
        if existing:
            existing.baseline_spend = round(baseline_spend, 2)
            existing.rl_multiplier = round(rl_multiplier, 4)
            existing.predicted_spend = round(predicted_spend, 2)
            existing.updated_at = datetime.now(timezone.utc)
            session.add(existing)
        else:
            session.add(
                MonthlyForecast(
                    user_id=user_id,
                    year_month=year_month,
                    category=category,
                    baseline_spend=round(baseline_spend, 2),
                    rl_multiplier=round(rl_multiplier, 4),
                    predicted_spend=round(predicted_spend, 2),
                )
            )
        session.commit()
    sync_forecasts_with_actuals(user_id)


def sync_forecasts_with_actuals(user_id: int) -> None:
    df = get_transactions_dataframe(user_id=user_id, verified_only=True)
    with get_session() as session:
        forecasts = session.exec(
            select(MonthlyForecast).where(MonthlyForecast.user_id == user_id)
        ).all()
        if df.empty:
            for f in forecasts:
                f.actual_spend = None
                f.reward = None
                session.add(f)
            session.commit()
            return

        debits = df[df["tx_type"] == "debit"]
        actual_lookup: Dict[tuple[str, str], float] = {}
        if not debits.empty:
            grouped = (
                debits.groupby(["year_month", "category"], as_index=False)["amount"]
                .sum()
                .rename(columns={"amount": "actual_spend"})
            )
            actual_lookup = {
                (row["year_month"], row["category"]): float(row["actual_spend"])
                for _, row in grouped.iterrows()
            }

        for f in forecasts:
            key = (f.year_month, f.category)
            if key in actual_lookup:
                actual = round(actual_lookup[key], 2)
                f.actual_spend = actual
                f.reward = compute_rl_reward(f.category, f.predicted_spend, actual)
                f.updated_at = datetime.now(timezone.utc)
                session.add(f)
            else:
                f.actual_spend = None
                f.reward = None
                session.add(f)
        session.commit()


def get_forecasts_dataframe(user_id: int) -> pd.DataFrame:
    with get_session() as session:
        rows = session.exec(
            select(MonthlyForecast)
            .where(MonthlyForecast.user_id == user_id)
            .order_by(MonthlyForecast.year_month.desc(), MonthlyForecast.category.asc())
        ).all()
    if not rows:
        return pd.DataFrame(
            columns=[
                "year_month",
                "category",
                "baseline_spend",
                "rl_multiplier",
                "predicted_spend",
                "actual_spend",
                "reward",
            ]
        )
    return pd.DataFrame([r.model_dump() for r in rows])