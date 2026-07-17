"""Признаки лида из истории событий (74 шт. на выходе `EventFeatureBuilder.build`).

Против утечки берутся только события с `event_ts < assignment_ts`: более поздние попадают
в окно таргета и содержат ответ.

Группы признаков:
- window_counts (42): счётчики по типам событий + всего + "горячие" (chat_open, call_click)
  в окнах 1/3/7/14/30/90 дней до назначения;
- recency (7): дни с последнего события — общий, по каждому типу, по горячим;
- diversity (7): широта интереса — nunique / энтропия / доля топа по контекстам (`ctx_seq`)
  и источникам (`src_slot`);
- span (3): возраст первого события, размах истории, средний возраст;
- price (6): статистики `item_price_log` просмотренного и |отклонение| от цены лида
  (средней и последней);
- bigrams (4): переходы между соседними событиями — всего, в горячие, повторы, доля в горячие;
- dynamics (5): отношения активности короткое/длинное окно (1d/7d, 3d/14d, 7d/30d)
  и доля горячих (90d, 7d).

Лиды без событий получают 0 по счётчикам и `RECENCY_FILL_DAYS` по recency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EVENT_TYPES = ["item_view", "search", "favorite", "chat_open", "call_click"]
HOT_TYPES = ["chat_open", "call_click"]

WINDOWS: list[tuple[str, float]] = [
    ("1d", 1.0),
    ("3d", 3.0),
    ("7d", 7.0),
    ("14d", 14.0),
    ("30d", 30.0),
    ("90d", 90.0),
]

RECENCY_FILL_DAYS = 999.0
LEAD_KEY = "lead_id"
ASSIGNMENT_TS = "assignment_ts"
EVENT_TS = "event_ts"
EVENT_TYPE = "event_type"
EVENT_PRICE = "item_price_log"


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0, np.nan)).fillna(0.0)


def _group_entropy(events: pd.DataFrame, column: str) -> pd.Series:
    counts = events.groupby([LEAD_KEY, column]).size().rename("count").reset_index()
    total = counts.groupby(LEAD_KEY)["count"].transform("sum")
    probability = counts["count"] / total
    counts["plogp"] = -probability * np.log(probability)
    return counts.groupby(LEAD_KEY)["plogp"].sum()


def _group_top_share(events: pd.DataFrame, column: str) -> pd.Series:
    counts = events.groupby([LEAD_KEY, column]).size().rename("count").reset_index()
    top = counts.groupby(LEAD_KEY)["count"].max()
    total = counts.groupby(LEAD_KEY)["count"].sum()
    return _safe_ratio(top, total)


@dataclass(frozen=True)
class EventFeatureConfig:
    reference_price_column: str = EVENT_PRICE
    slot_column: str = "src_slot"
    context_column: str = "ctx_seq"


class EventFeatureBuilder:
    def __init__(self, config: EventFeatureConfig | None = None) -> None:
        self._config = config or EventFeatureConfig()

    def build(self, events: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
        valid = self._prepare_valid_events(events, keys)
        blocks = [
            self._window_counts(valid),
            self._recency(valid),
            self._diversity(valid),
            self._span(valid),
            self._price(valid, keys),
            self._bigrams(valid),
        ]
        features = pd.concat(blocks, axis=1)
        features = self._dynamics(features)
        return self._finalize(features, keys)

    def _prepare_valid_events(self, events: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
        joined = events.merge(keys[[LEAD_KEY, ASSIGNMENT_TS]], on=LEAD_KEY, how="inner")
        valid = joined[joined[EVENT_TS] < joined[ASSIGNMENT_TS]].copy()
        valid["age_days"] = (valid[ASSIGNMENT_TS] - valid[EVENT_TS]).dt.total_seconds() / 86400.0
        valid["is_hot"] = valid[EVENT_TYPE].isin(HOT_TYPES)
        return valid

    def _window_counts(self, valid: pd.DataFrame) -> pd.DataFrame:
        blocks = []
        for name, days in WINDOWS:
            window = valid[valid["age_days"] <= days]
            by_type = pd.crosstab(window[LEAD_KEY], window[EVENT_TYPE])
            by_type = by_type.reindex(columns=EVENT_TYPES, fill_value=0)
            by_type.columns = [f"ev_{event_type}_{name}" for event_type in by_type.columns]
            total = window.groupby(LEAD_KEY).size().rename(f"ev_total_{name}")
            hot = window[window["is_hot"]].groupby(LEAD_KEY).size().rename(f"ev_hot_{name}")
            blocks.extend([by_type, total, hot])
        return pd.concat(blocks, axis=1)

    def _recency(self, valid: pd.DataFrame) -> pd.DataFrame:
        overall = valid.groupby(LEAD_KEY)["age_days"].min().rename("ev_recency")
        per_type = valid.groupby([LEAD_KEY, EVENT_TYPE])["age_days"].min().unstack()
        per_type = per_type.reindex(columns=EVENT_TYPES)
        per_type.columns = [f"ev_recency_{event_type}" for event_type in per_type.columns]
        hot = valid[valid["is_hot"]].groupby(LEAD_KEY)["age_days"].min().rename("ev_recency_hot")
        return pd.concat([overall, per_type, hot], axis=1)

    def _diversity(self, valid: pd.DataFrame) -> pd.DataFrame:
        slot = self._config.slot_column
        context = self._config.context_column
        result = pd.DataFrame(
            {
                "ev_type_nunique": valid.groupby(LEAD_KEY)[EVENT_TYPE].nunique(),
                "ev_slot_nunique": valid.groupby(LEAD_KEY)[slot].nunique(),
                "ev_ctx_nunique": valid.groupby(LEAD_KEY)[context].nunique(),
                "ev_ctx_entropy": _group_entropy(valid, context),
                "ev_slot_entropy": _group_entropy(valid, slot),
                "ev_ctx_top_share": _group_top_share(valid, context),
                "ev_slot_top_share": _group_top_share(valid, slot),
            }
        )
        return result

    def _span(self, valid: pd.DataFrame) -> pd.DataFrame:
        grouped = valid.groupby(LEAD_KEY)["age_days"]
        first_age = grouped.max().rename("ev_first_age")
        recency = grouped.min()
        result = pd.DataFrame({"ev_first_age": first_age})
        result["ev_span_days"] = first_age - recency
        result["ev_mean_age"] = grouped.mean()
        return result

    def _bigrams(self, valid: pd.DataFrame) -> pd.DataFrame:
        ordered = valid.sort_values([LEAD_KEY, EVENT_TS])
        previous_type = ordered.groupby(LEAD_KEY)[EVENT_TYPE].shift(1)
        transitions = ordered.assign(previous_type=previous_type).dropna(subset=["previous_type"])
        total = transitions.groupby(LEAD_KEY).size().rename("ev_bg_total")
        to_hot = (
            transitions[transitions[EVENT_TYPE].isin(HOT_TYPES)]
            .groupby(LEAD_KEY)
            .size()
            .rename("ev_bg_to_hot")
        )
        repeat = (
            transitions[transitions["previous_type"] == transitions[EVENT_TYPE]]
            .groupby(LEAD_KEY)
            .size()
            .rename("ev_bg_repeat")
        )
        result = pd.concat([total, to_hot, repeat], axis=1)
        result["ev_bg_to_hot_ratio"] = _safe_ratio(result["ev_bg_to_hot"], result["ev_bg_total"])
        return result

    def _price(self, valid: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
        price = self._config.reference_price_column
        grouped = valid.groupby(LEAD_KEY)[EVENT_PRICE]
        result = pd.DataFrame(
            {
                "ev_price_mean": grouped.mean(),
                "ev_price_std": grouped.std(),
                "ev_price_min": grouped.min(),
                "ev_price_max": grouped.max(),
            }
        )
        reference = keys.set_index(LEAD_KEY)[price]
        result["ev_price_diff_mean"] = (result["ev_price_mean"] - reference).abs()
        last_price = valid.sort_values("age_days").groupby(LEAD_KEY)[EVENT_PRICE].first()
        result["ev_price_diff_last"] = (last_price - reference).abs()
        return result

    def _dynamics(self, features: pd.DataFrame) -> pd.DataFrame:
        features["ev_ratio_1_7"] = _safe_ratio(features["ev_total_1d"], features["ev_total_7d"])
        features["ev_ratio_3_14"] = _safe_ratio(features["ev_total_3d"], features["ev_total_14d"])
        features["ev_ratio_7_30"] = _safe_ratio(features["ev_total_7d"], features["ev_total_30d"])
        features["ev_hot_ratio"] = _safe_ratio(features["ev_hot_90d"], features["ev_total_90d"])
        features["ev_hot_ratio_7d"] = _safe_ratio(features["ev_hot_7d"], features["ev_total_7d"])
        return features

    def _finalize(self, features: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
        features = features.reindex(keys[LEAD_KEY].to_numpy())
        recency_columns = [column for column in features.columns if "recency" in column]
        features[recency_columns] = features[recency_columns].fillna(RECENCY_FILL_DAYS)
        features = features.fillna(0.0)
        features.index.name = LEAD_KEY
        return features.reset_index()
