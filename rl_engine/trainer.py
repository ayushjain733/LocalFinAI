from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from config import (
    EXPENSE_CATEGORIES,
    RL_MODEL_DIR,
    SEASONALITY_INDEX,
)
from database.db import (
    get_transactions_dataframe,
    save_monthly_forecast,
)
from llm_engine.hf_manager import LocalHFEngine
from rl_engine.env import PersonalExpenseForecastEnv

logger = logging.getLogger(__name__)


def compute_baseline_forecast(
    recent_3m_spends: List[float],
    target_month: int,
    category: str,
) -> float:
    if not recent_3m_spends:
        return 0.0
    if len(recent_3m_spends) == 1:
        wma = float(recent_3m_spends[0])
    elif len(recent_3m_spends) == 2:
        wma = 0.4 * float(recent_3m_spends[0]) + 0.6 * float(recent_3m_spends[1])
    else:
        last_three = recent_3m_spends[-3:]
        wma = (
            0.2 * float(last_three[0])
            + 0.3 * float(last_three[1])
            + 0.5 * float(last_three[2])
        )

    seasonal_factor = SEASONALITY_INDEX.get(target_month, {}).get(category, 1.0)
    return round(max(0.0, wma * seasonal_factor), 2)


def generate_synthetic_training_samples(num_months: int = 24, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Cold-Start Synthetic Bootstrapping Generator used internally to pre-train
    the PPO neural network weights on seasonal dynamics before fine-tuning on the
    user's real SQLite ledger. Does NOT insert any fake rows into the user's database.
    """
    rng = np.random.default_rng(seed)
    category_profiles = {
        "Housing": (1850.0, 0.02, 0.98),
        "Utilities": (210.0, 0.14, 0.85),
        "Groceries": (580.0, 0.10, 0.15),
        "Dining & Cafe": (320.0, 0.22, 0.05),
        "Transport & Fuel": (240.0, 0.16, 0.20),
        "Shopping & Retail": (350.0, 0.28, 0.05),
        "Healthcare": (140.0, 0.25, 0.40),
        "Subscriptions & Media": (95.0, 0.04, 0.95),
        "Entertainment & Leisure": (170.0, 0.25, 0.10),
        "Travel": (260.0, 0.38, 0.00),
        "Miscellaneous": (130.0, 0.30, 0.05),
    }

    monthly_income = 7200.0
    history_by_cat: Dict[str, List[float]] = {cat: [] for cat in EXPENSE_CATEGORIES}
    samples: List[Dict[str, Any]] = []

    for step_idx in range(num_months):
        month_num = (step_idx % 12) + 1
        inflation_drift = 1.0 + (0.0025 * step_idx)

        for cat in EXPENSE_CATEGORIES:
            base_mean, noise_std, rec_ratio = category_profiles[cat]
            season_mult = SEASONALITY_INDEX[month_num][cat]
            essential_surge = 1.06 if cat in ("Housing", "Utilities", "Groceries") else 1.0
            actual = max(
                15.0,
                base_mean
                * season_mult
                * inflation_drift
                * essential_surge
                * float(rng.normal(1.0, noise_std)),
            )

            past = history_by_cat[cat]
            if len(past) >= 3:
                recent_3 = past[-3:]
                rolling_mean = float(np.mean(recent_3))
                rolling_std = float(np.std(recent_3))
                baseline = compute_baseline_forecast(recent_3, month_num, cat)
                prev_month = ((month_num - 2) % 12) + 1
                prev_baseline = compute_baseline_forecast(
                    past[-4:-1] if len(past) >= 4 else recent_3, prev_month, cat
                )
                last_err_ratio = (prev_baseline - past[-1]) / max(past[-1], 1.0)

                samples.append(
                    {
                        "month": month_num,
                        "category": cat,
                        "rolling_3m_mean": rolling_mean,
                        "rolling_3m_std": rolling_std,
                        "fixed_recurring_ratio": float(np.clip(rec_ratio + rng.normal(0, 0.03), 0.0, 1.0)),
                        "last_month_error_ratio": float(last_err_ratio),
                        "income_to_spend_ratio": float(monthly_income / max(rolling_mean * 6.0, 1000.0)),
                        "baseline_forecast": baseline,
                        "actual_spend": round(actual, 2),
                    }
                )
            history_by_cat[cat].append(actual)

    return samples


def build_real_ledger_samples(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty:
        return []

    debits = df[df["tx_type"] == "debit"].copy()
    credits = df[df["tx_type"] == "credit"].copy()
    if debits.empty:
        return []

    monthly_income_series = credits.groupby("year_month")["amount"].sum()
    avg_monthly_income = (
        float(monthly_income_series.mean())
        if not monthly_income_series.empty
        else float(debits.groupby("year_month")["amount"].sum().mean() * 1.25)
    )

    monthly_cat = (
        debits.groupby(["year_month", "category"], as_index=False)
        .agg(
            total_spend=("amount", "sum"),
            recurring_spend=("amount", lambda s: s[debits.loc[s.index, "is_recurring"]].sum()),
        )
        .sort_values("year_month")
    )

    all_months = sorted(debits["year_month"].unique())
    if len(all_months) < 2:
        return []

    samples: List[Dict[str, Any]] = []
    for cat in EXPENSE_CATEGORIES:
        cat_df = monthly_cat[monthly_cat["category"] == cat].set_index("year_month")
        series_spend: List[float] = []
        series_rec_ratio: List[float] = []

        for ym in all_months:
            if ym in cat_df.index:
                tot = float(cat_df.loc[ym, "total_spend"])
                rec = float(cat_df.loc[ym, "recurring_spend"])
                rec_ratio = rec / max(tot, 1.0)
            else:
                tot = 0.0
                rec_ratio = 0.0

            month_int = int(ym.split("-")[1])
            if len(series_spend) >= 1 and tot > 0:
                recent = series_spend[-3:]
                r_mean = float(np.mean(recent))
                r_std = float(np.std(recent))
                baseline = compute_baseline_forecast(recent, month_int, cat)
                last_err = (baseline - series_spend[-1]) / max(series_spend[-1], 1.0)

                samples.append(
                    {
                        "month": month_int,
                        "category": cat,
                        "rolling_3m_mean": r_mean,
                        "rolling_3m_std": r_std,
                        "fixed_recurring_ratio": float(np.mean(series_rec_ratio[-3:])),
                        "last_month_error_ratio": float(last_err),
                        "income_to_spend_ratio": float(avg_monthly_income / max(r_mean * 6.0, 500.0)),
                        "baseline_forecast": max(baseline, 10.0),
                        "actual_spend": tot,
                    }
                )
            series_spend.append(tot)
            series_rec_ratio.append(rec_ratio)

    return samples


class RLForecastManager:
    """
    Manages per-user Stable-Baselines3 PPO training (`MlpPolicy`, `net_arch=[64, 64]`),
    cold-start synthetic pre-training, incremental fine-tuning, and next-month inference.
    """

    def __init__(self, user_id: int) -> None:
        self.user_id = user_id
        self.hf_engine = LocalHFEngine()
        self.model_path: Path = RL_MODEL_DIR / f"ppo_expense_user_{user_id}.zip"
        self._model: PPO | None = None

    def load_or_initialize_agent(self) -> PPO:
        if self._model is not None:
            return self._model

        synth_samples = generate_synthetic_training_samples(num_months=24, seed=42 + self.user_id)
        dummy_env = PersonalExpenseForecastEnv(synth_samples)

        if self.model_path.exists():
            self._model = PPO.load(str(self.model_path), env=dummy_env, device="cpu")
            return self._model

        self.train_or_finetune(total_timesteps=3072, use_synthetic_bootstrap=True)
        return self._model  # type: ignore[return-value]

    def train_or_finetune(
        self,
        total_timesteps: int = 4096,
        use_synthetic_bootstrap: bool = True,
    ) -> Dict[str, Any]:
        llm_was_unloaded = self.hf_engine.unload_llm()

        df = get_transactions_dataframe(user_id=self.user_id, verified_only=True)
        real_samples = build_real_ledger_samples(df)
        synth_samples = generate_synthetic_training_samples(num_months=24, seed=42 + self.user_id)

        if real_samples and use_synthetic_bootstrap:
            training_samples = synth_samples + (real_samples * 3)
            mode = f"Hybrid Fine-Tuning ({len(real_samples)} real + {len(synth_samples)} synthetic priors)"
        elif real_samples:
            training_samples = real_samples * 4
            mode = f"Ledger Fine-Tuning ({len(real_samples)} verified transitions)"
        else:
            training_samples = synth_samples
            mode = f"Cold-Start Policy Initialization ({len(synth_samples)} seasonal transitions)"

        env = PersonalExpenseForecastEnv(training_samples)

        if self.model_path.exists():
            model = PPO.load(str(self.model_path), env=env, device="cpu")
            model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
        else:
            model = PPO(
                policy="MlpPolicy",
                env=env,
                policy_kwargs=dict(net_arch=[64, 64]),
                learning_rate=3e-4,
                n_steps=min(128, max(32, len(training_samples))),
                batch_size=32,
                gamma=0.95,
                ent_coef=0.01,
                verbose=0,
                device="cpu",
            )
            model.learn(total_timesteps=total_timesteps)

        model.save(str(self.model_path))
        self._model = model

        eval_rewards: List[float] = []
        obs, _ = env.reset(seed=101)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(action)
            eval_rewards.append(float(reward))
            done = terminated or truncated

        return {
            "mode": mode,
            "timesteps": total_timesteps,
            "mean_reward": round(float(np.mean(eval_rewards)), 4),
            "llm_unloaded_for_ram": llm_was_unloaded,
            "model_path": str(self.model_path),
        }

    def predict_next_month_expenses(self, target_year_month: str | None = None) -> pd.DataFrame:
        """
        Generates next-month category forecasts strictly from the user's verified
        debit transactions. If the user has no debit transactions yet, returns an
        empty DataFrame (preserving a clean blank slate for new accounts).
        """
        df = get_transactions_dataframe(user_id=self.user_id, verified_only=True)
        if df.empty:
            return pd.DataFrame()

        debits = df[df["tx_type"] == "debit"]
        if debits.empty:
            return pd.DataFrame()

        model = self.load_or_initialize_agent()

        if target_year_month is None:
            today = date.today()
            next_month = 1 if today.month == 12 else today.month + 1
            next_year = today.year + 1 if today.month == 12 else today.year
            target_year_month = f"{next_year:04d}-{next_month:02d}"

        target_month_int = int(target_year_month.split("-")[1])
        credits = df[df["tx_type"] == "credit"]

        avg_income = (
            float(credits.groupby("year_month")["amount"].sum().mean())
            if not credits.empty
            else float(debits.groupby("year_month")["amount"].sum().mean() * 1.2)
        )
        scale_norm = max(float(debits["amount"].max()), 3000.0)

        rows: List[Dict[str, Any]] = []
        for cat in EXPENSE_CATEGORIES:
            cat_debits = debits[debits["category"] == cat]
            if cat_debits.empty:
                continue

            monthly_totals = (
                cat_debits.groupby("year_month")["amount"].sum().sort_index().tolist()
            )
            recent_3 = monthly_totals[-3:]
            r_mean = float(np.mean(recent_3))
            r_std = float(np.std(recent_3))
            rec_spend = float(cat_debits[cat_debits["is_recurring"]]["amount"].sum())
            tot_spend = float(cat_debits["amount"].sum())
            rec_ratio = rec_spend / max(tot_spend, 1.0)
            baseline = compute_baseline_forecast(recent_3, target_month_int, cat)

            obs = PersonalExpenseForecastEnv.build_observation_vector(
                month=target_month_int,
                category=cat,
                rolling_3m_mean=r_mean,
                rolling_3m_std=r_std,
                fixed_recurring_ratio=rec_ratio,
                last_month_error_ratio=0.0,
                income_to_spend_ratio=avg_income / max(r_mean * 6.0, 500.0),
                scale_norm=scale_norm,
            )
            action, _ = model.predict(obs, deterministic=True)
            multiplier = float(np.clip(action[0], -0.5, 0.5))
            predicted = round(max(0.0, baseline * (1.0 + multiplier)), 2)

            save_monthly_forecast(
                user_id=self.user_id,
                year_month=target_year_month,
                category=cat,
                baseline_spend=baseline,
                rl_multiplier=multiplier,
                predicted_spend=predicted,
            )

            rows.append(
                {
                    "year_month": target_year_month,
                    "category": cat,
                    "rolling_3m_avg": round(r_mean, 2),
                    "baseline_spend": baseline,
                    "rl_multiplier": round(multiplier, 4),
                    "rl_adjustment_pct": f"{multiplier * 100:+.1f}%",
                    "predicted_spend": predicted,
                }
            )

        return pd.DataFrame(rows)