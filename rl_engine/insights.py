from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from config import DEFAULT_CURRENCY_SYMBOL
from database.db import get_transactions_dataframe
from llm_engine.hf_manager import LocalHFEngine


def compute_spend_analytics(
    user_id: int,
    currency_symbol: str = DEFAULT_CURRENCY_SYMBOL,
) -> Dict[str, Any]:
    """
    Computes exact deterministic totals, category shares, Month-over-Month deltas,
    and top merchants for `user_id` in Pandas/SQL first before any LLM summarization.
    """
    df = get_transactions_dataframe(user_id=user_id, verified_only=True)
    if df.empty:
        return {
            "has_data": False,
            "total_income": 0.0,
            "total_spend": 0.0,
            "net_cashflow": 0.0,
            "savings_rate_pct": 0.0,
            "latest_month": "N/A",
            "category_table": pd.DataFrame(),
            "top_merchants": pd.DataFrame(),
            "compact_prompt_summary": "No verified transactions available.",
        }

    months = sorted(df["year_month"].unique())
    latest_month = months[-1]
    prev_month = months[-2] if len(months) >= 2 else None

    latest_df = df[df["year_month"] == latest_month]
    total_income = float(latest_df[latest_df["tx_type"] == "credit"]["amount"].sum())
    total_spend = float(latest_df[latest_df["tx_type"] == "debit"]["amount"].sum())
    net_cashflow = total_income - total_spend
    savings_rate_pct = (net_cashflow / total_income * 100.0) if total_income > 0 else 0.0

    latest_debits = latest_df[latest_df["tx_type"] == "debit"]
    if latest_debits.empty:
        cat_merged = pd.DataFrame(
            columns=["category", "current_spend", "prev_spend", "share_pct", "mom_delta_amt", "mom_delta_pct"]
        )
        top_merchants = pd.DataFrame(columns=["merchant", "total_spend", "tx_count"])
    else:
        cat_curr = (
            latest_debits.groupby("category", as_index=False)["amount"]
            .sum()
            .rename(columns={"amount": "current_spend"})
        )

        if prev_month:
            prev_debits = df[(df["year_month"] == prev_month) & (df["tx_type"] == "debit")]
            cat_prev = (
                prev_debits.groupby("category", as_index=False)["amount"]
                .sum()
                .rename(columns={"amount": "prev_spend"})
            )
            cat_merged = pd.merge(cat_curr, cat_prev, on="category", how="outer").fillna(0.0)
        else:
            cat_merged = cat_curr.copy()
            cat_merged["prev_spend"] = 0.0

        cat_merged["share_pct"] = (
            cat_merged["current_spend"] / max(total_spend, 1.0) * 100.0
        ).round(1)
        cat_merged["mom_delta_amt"] = (cat_merged["current_spend"] - cat_merged["prev_spend"]).round(2)
        cat_merged["mom_delta_pct"] = cat_merged.apply(
            lambda r: round(((r["current_spend"] - r["prev_spend"]) / r["prev_spend"]) * 100.0, 1)
            if r["prev_spend"] > 0
            else 0.0,
            axis=1,
        )
        cat_merged = cat_merged.sort_values("current_spend", ascending=False).reset_index(drop=True)

        top_merchants = (
            latest_debits.groupby("merchant", as_index=False)
            .agg(total_spend=("amount", "sum"), tx_count=("id", "count"))
            .sort_values("total_spend", ascending=False)
            .head(5)
            .reset_index(drop=True)
        )

    cs = currency_symbol
    compact_lines = [
        f"Reporting Month: {latest_month} (Previous Month: {prev_month or 'None'}) | Currency: {cs}",
        f"Total Income: {cs}{total_income:,.2f} | Total Spend: {cs}{total_spend:,.2f} | Net Cashflow: {cs}{net_cashflow:,.2f} ({savings_rate_pct:.1f}% savings rate)",
        f"\nCategory Breakdown (Current {cs} | Share % | MoM Delta {cs} | MoM Delta %):",
    ]
    for _, r in cat_merged.iterrows():
        compact_lines.append(
            f"- {r['category']}: {cs}{r['current_spend']:,.2f} ({r['share_pct']:.1f}%) | "
            f"MoM: {cs}{r['mom_delta_amt']:+,.2f} ({r['mom_delta_pct']:+.1f}%)"
        )

    compact_lines.append(f"\nTop 5 Merchants by Spend ({cs}):")
    for _, m in top_merchants.iterrows():
        compact_lines.append(f"- {m['merchant']}: {cs}{m['total_spend']:,.2f} across {int(m['tx_count'])} transactions")

    return {
        "has_data": True,
        "latest_month": latest_month,
        "prev_month": prev_month,
        "total_income": round(total_income, 2),
        "total_spend": round(total_spend, 2),
        "net_cashflow": round(net_cashflow, 2),
        "savings_rate_pct": round(savings_rate_pct, 1),
        "category_table": cat_merged,
        "top_merchants": top_merchants,
        "compact_prompt_summary": "\n".join(compact_lines),
    }


def generate_ai_spend_commentary(
    analytics: Dict[str, Any],
    currency_symbol: str = DEFAULT_CURRENCY_SYMBOL,
) -> str:
    if not analytics.get("has_data"):
        return "Add or upload transactions in the Ledger tab first to generate AI financial insights."

    hf_engine = LocalHFEngine()
    system_prompt = (
        "You are a fiduciary personal finance analyst running 100% locally. "
        f"All monetary figures are in '{currency_symbol}'. Always use '{currency_symbol}' when quoting amounts. "
        "Using ONLY the exact deterministic SQL/Pandas metrics provided, write a concise 3-part briefing: "
        "(1) Cashflow & Savings Efficiency, (2) Notable Month-over-Month Category Shifts, and "
        "(3) Two concrete cost-optimization actions citing specific merchants or categories."
    )
    return hf_engine.generate_text(
        system_prompt=system_prompt,
        user_prompt=analytics["compact_prompt_summary"],
        temperature=0.2,
        max_tokens=600,
    )