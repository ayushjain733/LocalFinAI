from datetime import date as dt_date, datetime as dt_datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


def utc_now() -> dt_datetime:
    return dt_datetime.now(timezone.utc)


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(unique=True, index=True, description="Unique login username")
    display_name: str = Field(description="User display name")
    password_hash: str = Field(description="PBKDF2-HMAC-SHA256 password digest")
    salt: str = Field(description="Per-user cryptographic salt")
    preferred_currency: str = Field(default="INR (₹)", description="Saved currency preference")
    security_question: Optional[str] = Field(
        default=None, description="Selected offline recovery question"
    )
    security_answer_hash: Optional[str] = Field(
        default=None, description="PBKDF2-HMAC-SHA256 digest of normalized security answer"
    )
    recovery_key_hash: Optional[str] = Field(
        default=None, description="PBKDF2-HMAC-SHA256 digest of LFAI-XXXX-XXXX-XXXX recovery key"
    )
    created_at: dt_datetime = Field(default_factory=utc_now)


class Transaction(SQLModel, table=True):
    __tablename__ = "transactions"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True, foreign_key="users.id")
    date: dt_date = Field(index=True, description="Transaction posting date")
    merchant: str = Field(index=True, description="Normalized merchant name")
    raw_description: str = Field(description="Original bank statement line text")
    amount: float = Field(description="Positive transaction value in local currency")
    tx_type: str = Field(default="debit", description="'debit' (expense) or 'credit' (income)")
    category: str = Field(index=True, description="Assigned expense or income category")
    is_recurring: bool = Field(default=False, description="Fixed monthly recurring flag")
    source: str = Field(default="ai_extracted", description="'ai_extracted' | 'manual'")
    verified_by_user: bool = Field(default=False, index=True)
    created_at: dt_datetime = Field(default_factory=utc_now)


class CategoryCorrection(SQLModel, table=True):
    __tablename__ = "category_corrections"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True, foreign_key="users.id")
    raw_merchant_pattern: str = Field(
        index=True,
        description="Lowercased normalized merchant pattern overridden by the user",
    )
    corrected_category: str = Field(description="User-verified target category")
    updated_at: dt_datetime = Field(default_factory=utc_now)


class MonthlyForecast(SQLModel, table=True):
    __tablename__ = "monthly_forecasts"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True, foreign_key="users.id")
    year_month: str = Field(index=True, description="Format YYYY-MM")
    category: str = Field(index=True, description="Expense category name")
    baseline_spend: float = Field(description="3M WMA + Seasonality statistical baseline")
    rl_multiplier: float = Field(description="RL action adjustment in [-0.5, +0.5]")
    predicted_spend: float = Field(description="Final RL predicted category spend")
    actual_spend: Optional[float] = Field(default=None, description="Realized spend once month closes")
    reward: Optional[float] = Field(default=None, description="Computed RL reward signal")
    updated_at: dt_datetime = Field(default_factory=utc_now)