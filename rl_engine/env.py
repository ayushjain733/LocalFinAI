from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import ESSENTIAL_CATEGORIES, EXPENSE_CATEGORIES


class PersonalExpenseForecastEnv(gym.Env):
    """
    Custom Gymnasium Environment for Next-Month Category Spend Forecasting.

    Observation Space (Box, length = 2 + len(EXPENSE_CATEGORIES) + 5 = 18):
        [month_sin, month_cos, *category_id_one_hot, rolling_3m_mean,
         rolling_3m_std, fixed_recurring_ratio, last_month_error_ratio, income_to_spend_ratio]

    Action Space (Box, shape=(1,), low=-0.5, high=+0.5):
        Continuous multiplier `a` in [-0.5, +0.5] scaling the statistical baseline forecast:
        `predicted_spend = baseline_forecast * (1.0 + a)`

    Reward Function:
        `reward = -abs(predicted_spend - actual_spend) / max(actual_spend, 1.0)`
        With a 1.5x penalty multiplier when `predicted_spend < actual_spend` on
        essential categories (Housing, Utilities, Groceries).
    """

    metadata = {"render_modes": []}

    def __init__(self, samples: List[Dict[str, Any]]) -> None:
        super().__init__()
        if not samples:
            raise ValueError("PersonalExpenseForecastEnv requires at least one training transition sample.")

        self.samples = samples
        self.num_categories = len(EXPENSE_CATEGORIES)
        self.obs_dim = 2 + self.num_categories + 5

        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-0.5], dtype=np.float32),
            high=np.array([0.5], dtype=np.float32),
            dtype=np.float32,
        )

        self._current_idx: int = 0
        self._indices: np.ndarray = np.arange(len(self.samples))

    @staticmethod
    def build_observation_vector(
        month: int,
        category: str,
        rolling_3m_mean: float,
        rolling_3m_std: float,
        fixed_recurring_ratio: float,
        last_month_error_ratio: float,
        income_to_spend_ratio: float,
        scale_norm: float = 3000.0,
    ) -> np.ndarray:
        angle = 2.0 * math.pi * (float(month) / 12.0)
        month_sin = math.sin(angle)
        month_cos = math.cos(angle)

        one_hot = np.zeros(len(EXPENSE_CATEGORIES), dtype=np.float32)
        if category in EXPENSE_CATEGORIES:
            one_hot[EXPENSE_CATEGORIES.index(category)] = 1.0

        mean_norm = float(np.clip(rolling_3m_mean / scale_norm, 0.0, 5.0))
        std_norm = float(np.clip(rolling_3m_std / (scale_norm * 0.5), 0.0, 5.0))
        rec_ratio = float(np.clip(fixed_recurring_ratio, 0.0, 1.0))
        err_ratio = float(np.clip(last_month_error_ratio, -2.0, 2.0))
        inc_ratio = float(np.clip(income_to_spend_ratio / 5.0, 0.0, 5.0))

        obs = np.concatenate(
            [
                np.array([month_sin, month_cos], dtype=np.float32),
                one_hot,
                np.array([mean_norm, std_norm, rec_ratio, err_ratio, inc_ratio], dtype=np.float32),
            ]
        )
        return obs.astype(np.float32)

    def _get_obs(self, sample: Dict[str, Any]) -> np.ndarray:
        return self.build_observation_vector(
            month=int(sample["month"]),
            category=str(sample["category"]),
            rolling_3m_mean=float(sample["rolling_3m_mean"]),
            rolling_3m_std=float(sample["rolling_3m_std"]),
            fixed_recurring_ratio=float(sample["fixed_recurring_ratio"]),
            last_month_error_ratio=float(sample["last_month_error_ratio"]),
            income_to_spend_ratio=float(sample["income_to_spend_ratio"]),
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.np_random.shuffle(self._indices)
        self._current_idx = 0
        sample = self.samples[int(self._indices[self._current_idx])]
        return self._get_obs(sample), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        sample = self.samples[int(self._indices[self._current_idx])]
        multiplier = float(np.clip(action[0], -0.5, 0.5))

        baseline_forecast = float(sample["baseline_forecast"])
        actual_spend = float(sample["actual_spend"])
        category = str(sample["category"])

        predicted_spend = max(0.0, baseline_forecast * (1.0 + multiplier))
        relative_error = abs(predicted_spend - actual_spend) / max(actual_spend, 1.0)

        # 1.5x penalty when under-predicting essential categories
        if category in ESSENTIAL_CATEGORIES and predicted_spend < actual_spend:
            relative_error *= 1.5

        reward = -float(relative_error)

        self._current_idx += 1
        terminated = self._current_idx >= len(self.samples)
        truncated = False

        if not terminated:
            next_sample = self.samples[int(self._indices[self._current_idx])]
            next_obs = self._get_obs(next_sample)
        else:
            next_obs = np.zeros(self.obs_dim, dtype=np.float32)

        info = {
            "category": category,
            "baseline_forecast": baseline_forecast,
            "predicted_spend": predicted_spend,
            "actual_spend": actual_spend,
            "multiplier": multiplier,
            "reward": reward,
        }
        return next_obs, reward, terminated, truncated, info