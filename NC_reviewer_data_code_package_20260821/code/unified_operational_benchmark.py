#!/usr/bin/env python
"""Unified strict-prefix benchmark for operational epidemic transition alerts.

This analysis reuses the independently generated, paired outbreak/no-outbreak
renewal-process trajectories and full prospective Morlet-w6/V31.1 prefix outputs from
the formal mother-wavelet validation. Every comparator is evaluated on the
same observed count sequences. Thresholds are calibrated on development
no-outbreak curves and frozen before outer validation.

The benchmark is deliberately separate from the core phase-estimation study.
Its synthetic T1 is the reporting-delay-adjusted day on which latent Rt crosses
1. Its synthetic T2 is the reporting-delay-adjusted day on which the Rt ramp
reaches 90% of its post-transition level. These benchmark-specific landmarks
define an operational response window; they do not replace the framework's
primary phase definitions.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, gammaln
from scipy.stats import gamma as gamma_distribution


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "母小波确认" / "outputs_v2" / "20260716_103352_formal_v2"
DEFAULT_CACHE = Path(__file__).resolve().parent / "cache_v31_full_morlet_w6"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs_unified_operational_benchmark_v31_full"

METHODS = (
    "EEMD_CWT_first",
    "EEMD_CWT_stable",
    "EARS_C2",
    "EARS_C3",
    "EWMA",
    "CUSUM",
    "growth_rate",
    "rolling_quantile",
    "Rt_P_gt_1",
    "Poisson_GLR",
    "BOCPD",
    "Farrington_style_GLM",
)

METHOD_LABELS = {
    "EEMD_CWT_first": "Full V31.1 Morlet-w6: first indication",
    "EEMD_CWT_stable": "Full V31.1 Morlet-w6: sustained support",
    "EARS_C2": "EARS C2",
    "EARS_C3": "EARS C3",
    "EWMA": "EWMA",
    "CUSUM": "CUSUM",
    "growth_rate": "3-day growth rate",
    "rolling_quantile": "Rolling quantile",
    "Rt_P_gt_1": "Cori-style P(Rt>1)",
    "Poisson_GLR": "Sequential Poisson GLR",
    "BOCPD": "Bayesian online changepoint",
    "Farrington_style_GLM": "Farrington-style quasi-Poisson GLM",
}

REGIMES = (
    {"regime": "burden_low", "criterion": "alert_days_per_100d", "target": 1.0},
    {"regime": "burden_high", "criterion": "alert_days_per_100d", "target": 3.0},
    {"regime": "specificity_90", "criterion": "null_curve_fpr", "target": 0.10},
)


@dataclass
class Curve:
    curve_id: str
    pair_id: str
    split: str
    kind: str
    archetype: str
    gt: float
    truth_t1: float | None
    truth_t2: float | None
    observed: np.ndarray
    rt: np.ndarray
    start: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--v31-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--development-only", action="store_true")
    parser.add_argument(
        "--threshold-locks", type=Path, default=None,
        help="Frozen development_threshold_locks.csv for one-pass validation.",
    )
    return parser.parse_args()


def robust_scale(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 1.0
    med = float(np.median(x))
    mad = 1.4826 * float(np.median(np.abs(x - med)))
    sd = float(np.std(x, ddof=1))
    return max(mad, 0.25 * sd, 1.0e-8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def derive_t2(rt: np.ndarray, t1_latent: int, r_pre: float, r_post: float) -> int:
    target = float(r_pre + 0.90 * (r_post - r_pre))
    candidates = np.flatnonzero((np.arange(len(rt)) >= int(t1_latent)) & (rt >= target))
    if candidates.size:
        return int(candidates[0])
    return int(min(len(rt) - 1, t1_latent + 1))


def load_curves(source: Path, split: str) -> List[Curve]:
    metadata = pd.read_csv(source / f"{split}_curve_metadata.csv")
    archive = np.load(source / f"{split}_curve_arrays.npz")
    curves: List[Curve] = []
    for index, row in metadata.iterrows():
        observed = np.asarray(archive[f"observed_{index}"], dtype=float)
        rt = np.asarray(archive[f"rt_{index}"], dtype=float)
        gt = float(row.GT)
        start = max(24, int(math.ceil(5.0 * gt)))
        if row.kind == "outbreak":
            t1_latent = int(row.truth_T1_latent)
            t2_latent = derive_t2(rt, t1_latent, float(row.r_pre), float(row.r_post))
            delay = int(row.delay_median)
            truth_t1 = float(min(len(observed) - 1, t1_latent + delay))
            truth_t2 = float(min(len(observed) - 1, t2_latent + delay))
            truth_t2 = max(truth_t2, truth_t1 + 1.0)
        else:
            truth_t1 = None
            truth_t2 = None
        curves.append(
            Curve(
                curve_id=str(row.curve_id),
                pair_id=str(row.pair_id),
                split=split,
                kind=str(row.kind),
                archetype=str(row.archetype),
                gt=gt,
                truth_t1=truth_t1,
                truth_t2=truth_t2,
                observed=observed,
                rt=rt,
                start=start,
            )
        )
    return curves


def generation_weights(gt: float, maximum: int | None = None) -> np.ndarray:
    shape = 1.0 / 0.45**2
    scale = gt / shape
    maximum = maximum or max(8, int(np.ceil(5 * gt)))
    edges = np.arange(maximum + 1, dtype=float) + 0.5
    cdf = gamma_distribution.cdf(edges, a=shape, scale=scale)
    weights = np.diff(np.r_[0.0, cdf])
    weights = np.maximum(weights, 0.0)
    return weights / max(weights.sum(), 1.0e-12)


def ears_scores(x: np.ndarray, start: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(x)
    c2 = np.full(n, np.nan)
    c3 = np.full(n, np.nan)
    for t in range(max(start, 9), n):
        baseline = x[t - 9 : t - 2]
        mean = float(np.mean(baseline))
        sd = max(float(np.std(baseline, ddof=1)), math.sqrt(mean + 1.0), 1.0)
        c2[t] = (x[t] - mean) / sd
        recent = c2[max(start, t - 2) : t + 1]
        c3[t] = float(np.nansum(np.maximum(recent - 1.0, 0.0)))
    return c2, c3


def ewma_score(x: np.ndarray, start: int, alpha: float = 0.30) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    state = float(np.mean(x[max(0, start - 7) : start])) if start > 0 else float(x[0])
    for t in range(start, n):
        history = x[max(0, t - 28) : t]
        if history.size < 7:
            continue
        mean = float(np.median(history))
        sd = robust_scale(history)
        state = alpha * float(x[t]) + (1.0 - alpha) * state
        denom = max(sd * math.sqrt(alpha / (2.0 - alpha)), 1.0)
        out[t] = (state - mean) / denom
    return out


def cusum_score(x: np.ndarray, start: int, allowance: float = 0.5) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    statistic = 0.0
    for t in range(start, n):
        history = x[max(0, t - 28) : t]
        if history.size < 7:
            continue
        centre = float(np.median(history))
        scale = max(robust_scale(history), math.sqrt(centre + 1.0), 1.0)
        statistic = max(0.0, statistic + (float(x[t]) - centre) / scale - allowance)
        out[t] = statistic
    return out


def growth_score(x: np.ndarray, start: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    for t in range(max(start, 6), n):
        current = float(np.sum(x[t - 2 : t + 1]))
        previous = float(np.sum(x[t - 5 : t - 2]))
        out[t] = math.log((current + 0.5) / (previous + 0.5))
    return out


def rolling_quantile_score(x: np.ndarray, start: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    for t in range(start, n):
        history = x[max(0, t - 28) : t]
        if history.size < 14:
            continue
        q95 = float(np.quantile(history, 0.95))
        out[t] = (float(x[t]) - q95) / max(robust_scale(history), 1.0)
    return out


def rt_probability_score(x: np.ndarray, start: int, gt: float) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    weights = generation_weights(gt)
    infectiousness = np.zeros(n)
    for t in range(1, n):
        lag = min(t, len(weights))
        infectiousness[t] = float(np.dot(x[t - lag : t][::-1], weights[:lag]))
    window = max(3, int(round(gt)))
    prior_shape, prior_rate = 1.0, 1.0
    for t in range(max(start, window), n):
        left = t - window + 1
        shape = prior_shape + float(np.sum(x[left : t + 1]))
        rate = prior_rate + float(np.sum(infectiousness[left : t + 1]))
        out[t] = float(gamma_distribution.sf(1.0, a=shape, scale=1.0 / max(rate, 1.0e-9)))
    return out


def poisson_loglik(segment: np.ndarray) -> float:
    if segment.size == 0:
        return -np.inf
    mean = max(float(np.mean(segment)), 1.0e-9)
    return float(np.sum(segment * math.log(mean) - mean - gammaln(segment + 1.0)))


def poisson_glr_score(x: np.ndarray, start: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    for t in range(start, n):
        left = max(0, t - 41)
        segment = x[left : t + 1]
        if segment.size < 21:
            continue
        null_ll = poisson_loglik(segment)
        best = 0.0
        for cut in range(7, len(segment) - 6):
            before = segment[:cut]
            after = segment[cut:]
            if float(np.mean(after)) <= float(np.mean(before)):
                continue
            best = max(best, 2.0 * (poisson_loglik(before) + poisson_loglik(after) - null_ll))
        out[t] = math.sqrt(max(best, 0.0))
    return out


def log_nb_predictive(count: float, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return (
        gammaln(alpha + count)
        - gammaln(alpha)
        - gammaln(count + 1.0)
        + alpha * np.log(beta / (beta + 1.0))
        + count * np.log(1.0 / (beta + 1.0))
    )


def bocpd_score(x: np.ndarray, start: int, hazard: float = 1.0 / 50.0) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    run_prob = np.array([1.0])
    alpha = np.array([1.0])
    beta = np.array([1.0])
    for t, count in enumerate(x):
        log_pred = log_nb_predictive(float(count), alpha, beta)
        log_pred -= float(np.max(log_pred))
        pred = np.exp(log_pred)
        cp = float(np.sum(run_prob * pred * hazard))
        growth = run_prob * pred * (1.0 - hazard)
        updated = np.r_[cp, growth]
        updated /= max(float(updated.sum()), 1.0e-300)
        new_alpha = np.r_[1.0, alpha + float(count)]
        new_beta = np.r_[1.0, beta + 1.0]
        run_prob, alpha, beta = updated, new_alpha, new_beta
        if t >= start:
            short_mass = float(np.sum(run_prob[: min(4, len(run_prob))]))
            recent_mean = float(np.mean(x[max(0, t - 2) : t + 1]))
            baseline = float(np.mean(x[max(0, t - 28) : max(0, t - 3)]))
            upward = max(0.0, math.log((recent_mean + 0.5) / (baseline + 0.5)))
            out[t] = short_mass * upward
    return out


def farrington_design(days: np.ndarray, centre: float) -> np.ndarray:
    scaled = (days - centre) / 28.0
    dow = (days.astype(int) % 7)
    columns = [
        np.ones(len(days)),
        scaled,
        np.sin(2.0 * np.pi * days / 7.0),
        np.cos(2.0 * np.pi * days / 7.0),
    ]
    columns.extend((dow == value).astype(float) for value in range(1, 7))
    return np.column_stack(columns)


def fit_poisson_ridge(y: np.ndarray, design: np.ndarray, ridge: float = 1.0e-4) -> np.ndarray:
    beta = np.zeros(design.shape[1])
    beta[0] = math.log(float(np.mean(y)) + 0.1)
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    for _ in range(8):
        eta = np.clip(design @ beta, -8.0, 12.0)
        mu = np.exp(eta)
        weights = np.maximum(mu, 1.0e-6)
        working = eta + (y - mu) / weights
        lhs = design.T @ (weights[:, None] * design) + penalty
        rhs = design.T @ (weights * working)
        try:
            updated = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            updated = np.linalg.pinv(lhs) @ rhs
        if float(np.max(np.abs(updated - beta))) < 1.0e-6:
            beta = updated
            break
        beta = updated
    return beta


def farrington_style_score(x: np.ndarray, start: int, step: int = 3) -> np.ndarray:
    """Short-history quasi-Poisson analogue, not canonical Farrington Flexible."""
    n = len(x)
    out = np.full(n, np.nan)
    for t in range(start, n, max(1, int(step))):
        end = t - 2
        left = max(0, end - 83)
        days = np.arange(left, end + 1)
        if days.size < 28:
            continue
        y = x[days]
        design = farrington_design(days.astype(float), float(t))
        beta = fit_poisson_ridge(y, design)
        mu = float(np.exp(np.clip(farrington_design(np.array([float(t)]), float(t)) @ beta, -8.0, 12.0))[0])
        fitted = np.exp(np.clip(design @ beta, -8.0, 12.0))
        df = max(1, len(y) - design.shape[1])
        dispersion = max(1.0, float(np.sum((y - fitted) ** 2 / np.maximum(fitted, 1.0e-6)) / df))
        out[t] = (float(x[t]) - mu) / math.sqrt(max(dispersion * mu, 1.0))
    return out


def eemd_score(
    cache: pd.DataFrame,
    curve_id: str,
    n: int,
    column: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a strict-prefix full-V31.1 evidence stream without backdating."""
    group = cache[cache.curve_id == curve_id].sort_values("day")
    score = np.full(n, np.nan)
    opportunity = np.zeros(n, dtype=bool)
    values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
    days = pd.to_numeric(group.day, errors="coerce").to_numpy(dtype=float)
    for value, day_value in zip(values, days):
        if not np.isfinite(value) or not np.isfinite(day_value):
            continue
        day = int(day_value)
        if 0 <= day < n:
            score[day] = float(value)
            opportunity[day] = True
    return score, opportunity


def compute_scores(
    curve: Curve,
    cache: pd.DataFrame,
    step: int = 3,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    x = np.asarray(curve.observed, dtype=float)
    n = len(x)
    c2, c3 = ears_scores(x, curve.start)
    eemd_first, first_opportunity = eemd_score(
        cache, curve.curve_id, n, "first_score"
    )
    eemd_stable, stable_opportunity = eemd_score(
        cache, curve.curve_id, n, "stable_score"
    )
    scheduled = np.zeros(n, dtype=bool)
    scheduled[np.arange(curve.start, n, max(1, int(step)))] = True
    scores = {
        "EEMD_CWT_first": eemd_first,
        "EEMD_CWT_stable": eemd_stable,
        "EARS_C2": c2,
        "EARS_C3": c3,
        "EWMA": ewma_score(x, curve.start),
        "CUSUM": cusum_score(x, curve.start),
        "growth_rate": growth_score(x, curve.start),
        "rolling_quantile": rolling_quantile_score(x, curve.start),
        "Rt_P_gt_1": rt_probability_score(x, curve.start, curve.gt),
        "Poisson_GLR": poisson_glr_score(x, curve.start),
        "BOCPD": bocpd_score(x, curve.start),
        "Farrington_style_GLM": farrington_style_score(x, curve.start, step=step),
    }
    opportunities = {method: scheduled.copy() for method in METHODS}
    opportunities["EEMD_CWT_first"] = first_opportunity & scheduled
    opportunities["EEMD_CWT_stable"] = stable_opportunity & scheduled
    for method in METHODS:
        opportunities[method] &= np.isfinite(scores[method])
    return scores, opportunities


def compute_curve_payload(payload: Tuple[Curve, pd.DataFrame, int]):
    """Worker for one independent curve."""
    curve, cache_group, step = payload
    scores, opportunities = compute_scores(curve, cache_group, step=step)
    return curve.curve_id, scores, opportunities


def compute_split_scores(
    curves: Sequence[Curve],
    cache: pd.DataFrame,
    step: int,
    workers: int,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]:
    cache_groups = {
        str(curve_id): group.copy()
        for curve_id, group in cache.groupby("curve_id", sort=False)
    }
    payloads = [
        (curve, cache_groups.get(curve.curve_id, cache.iloc[0:0].copy()), step)
        for curve in curves
    ]
    score_map: Dict[str, Dict[str, np.ndarray]] = {}
    opportunity_map: Dict[str, Dict[str, np.ndarray]] = {}
    pool = None
    if workers <= 1:
        iterator = map(compute_curve_payload, payloads)
    else:
        pool = ThreadPoolExecutor(max_workers=workers)
        iterator = pool.map(compute_curve_payload, payloads, chunksize=1)
    try:
        for index, (curve_id, scores, opportunities) in enumerate(iterator, start=1):
            score_map[curve_id] = scores
            opportunity_map[curve_id] = opportunities
            if index % 20 == 0 or index == len(curves):
                print(f"computed {index}/{len(curves)} curves", flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    return score_map, opportunity_map

def alarm_trace(
    score: np.ndarray,
    opportunity: np.ndarray,
    threshold: float,
    cooldown: int = 7,
    reset_updates: int = 2,
) -> Tuple[List[int], np.ndarray, List[int]]:
    """Apply the same prospective reset/cooldown state machine to all methods."""
    n = len(score)
    active_days = np.zeros(n, dtype=bool)
    events: List[int] = []
    durations: List[int] = []
    active = False
    below_run = 0
    cooldown_until = -1
    episode_start: int | None = None
    update_days = np.flatnonzero(opportunity)
    for position, day_value in enumerate(update_days):
        day = int(day_value)
        next_day = int(update_days[position + 1]) if position + 1 < len(update_days) else n
        above = bool(np.isfinite(score[day]) and float(score[day]) >= threshold)
        if above:
            below_run = 0
            if not active and day >= cooldown_until:
                active = True
                episode_start = day
                events.append(day)
        elif active:
            below_run += 1
            if below_run >= max(1, int(reset_updates)):
                active = False
                end = day
                if episode_start is not None:
                    durations.append(max(1, end - episode_start))
                episode_start = None
                cooldown_until = day + max(0, int(cooldown))
                below_run = 0
        if active:
            active_days[day:next_day] = True
    if active and episode_start is not None:
        durations.append(max(1, n - episode_start))
    return events, active_days, durations


def burden_metrics(
    curves: Sequence[Curve],
    score_map: Mapping[str, Mapping[str, np.ndarray]],
    opportunity_map: Mapping[str, Mapping[str, np.ndarray]],
    method: str,
    threshold: float,
) -> Dict[str, float]:
    null_curves = [curve for curve in curves if curve.kind == "no_outbreak"]
    n_events = 0
    n_alert_curves = 0
    alert_days = 0
    denominator = 0
    all_durations: List[int] = []
    for curve in null_curves:
        scores = score_map[curve.curve_id][method]
        opportunities = opportunity_map[curve.curve_id][method]
        events, active, durations = alarm_trace(scores, opportunities, threshold)
        n_events += len(events)
        n_alert_curves += int(bool(events))
        alert_days += int(np.sum(active[curve.start:]))
        all_durations.extend(durations)
        denominator += max(1, len(scores) - curve.start)
    return {
        "events_per_100d": 100.0 * n_events / max(denominator, 1),
        "null_curve_fpr": n_alert_curves / max(len(null_curves), 1),
        "alert_days_per_100d": 100.0 * alert_days / max(denominator, 1),
        "mean_episode_duration_days": (
            float(np.mean(all_durations)) if all_durations else 0.0
        ),
    }

def candidate_thresholds(
    curves: Sequence[Curve],
    score_map: Mapping[str, Mapping[str, np.ndarray]],
    opportunity_map: Mapping[str, Mapping[str, np.ndarray]],
    method: str,
) -> np.ndarray:
    parts = []
    for curve in curves:
        if curve.kind != "no_outbreak":
            continue
        values = score_map[curve.curve_id][method]
        mask = opportunity_map[curve.curve_id][method]
        selected = values[mask]
        if selected.size:
            parts.append(selected)
    if not parts:
        return np.array([np.inf])
    pooled = np.concatenate(parts)
    candidates = np.unique(pooled[np.isfinite(pooled)])
    return np.r_[np.inf, candidates[::-1], -np.inf]


def calibrate_thresholds(
    curves: Sequence[Curve],
    score_map: Mapping[str, Mapping[str, np.ndarray]],
    opportunity_map: Mapping[str, Mapping[str, np.ndarray]],
) -> pd.DataFrame:
    """Lock thresholds by monotone search over every observed null score.

    For both alert-day burden and null-curve FPR, lowering the threshold can
    only add above-threshold support under the shared state machine. We use a
    binary search for the crossing and then evaluate adjacent exact score
    thresholds, avoiding an imprecise quantile grid and exhaustive replay.
    """
    rows = []
    for method in METHODS:
        candidates = candidate_thresholds(curves, score_map, opportunity_map, method)
        metric_cache: Dict[int, Dict[str, float]] = {}

        def metrics_at(index: int) -> Dict[str, float]:
            index = int(np.clip(index, 0, len(candidates) - 1))
            if index not in metric_cache:
                metric_cache[index] = burden_metrics(
                    curves,
                    score_map,
                    opportunity_map,
                    method,
                    float(candidates[index]),
                )
            return metric_cache[index]

        for regime in REGIMES:
            criterion = str(regime["criterion"])
            target = float(regime["target"])
            left, right = 0, len(candidates) - 1
            while left < right:
                middle = (left + right) // 2
                if float(metrics_at(middle)[criterion]) >= target:
                    right = middle
                else:
                    left = middle + 1
            crossing = left
            neighbor_indices = sorted(
                set(
                    int(np.clip(index, 0, len(candidates) - 1))
                    for index in range(crossing - 3, crossing + 4)
                )
            )
            evaluated = [
                {
                    "threshold": float(candidates[index]),
                    **metrics_at(index),
                }
                for index in neighbor_indices
            ]
            best = min(
                evaluated,
                key=lambda item: (
                    abs(float(item[criterion]) - target),
                    float(item[criterion]) > target,
                    -float(item["threshold"]),
                ),
            )
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "regime": regime["regime"],
                    "criterion": criterion,
                    "target": target,
                    "threshold": best["threshold"],
                    "development_events_per_100d": best["events_per_100d"],
                    "development_null_curve_fpr": best["null_curve_fpr"],
                    "development_alert_days_per_100d": best["alert_days_per_100d"],
                    "development_mean_episode_duration_days": best[
                        "mean_episode_duration_days"
                    ],
                }
            )
    return pd.DataFrame(rows)
def curve_result(
    curve: Curve,
    method: str,
    regime: str,
    threshold: float,
    score: np.ndarray,
    opportunity: np.ndarray,
) -> Dict:
    events, active, durations = alarm_trace(score, opportunity, threshold)
    alert_days = int(np.sum(active[curve.start:]))
    result = {
        "curve_id": curve.curve_id,
        "pair_id": curve.pair_id,
        "kind": curve.kind,
        "archetype": curve.archetype,
        "GT": curve.gt,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "regime": regime,
        "threshold": threshold,
        "surveillance_days": max(1, len(score) - curve.start),
        "n_events": len(events),
        "any_event": bool(events),
        "first_event_day": events[0] if events else np.nan,
        "alert_days": alert_days,
        "alert_day_proportion": alert_days / max(1, len(score) - curve.start),
        "mean_episode_duration_days": (
            float(np.mean(durations)) if durations else 0.0
        ),
        "truth_T1_observable": curve.truth_t1,
        "truth_T2_observable": curve.truth_t2,
        "premature": False,
        "detected_in_window": False,
        "miss_before_T2": False,
        "first_effective_alert_day": np.nan,
        "lead_to_T2_days": np.nan,
        "signed_timing_from_T1_days": np.nan,
    }
    if curve.kind != "outbreak":
        return result
    lower = float(curve.truth_t1 - 0.5 * curve.gt)
    upper = float(curve.truth_t2)
    effective = [day for day in events if lower <= day < upper]
    result["premature"] = any(day < lower for day in events)
    result["detected_in_window"] = bool(effective)
    result["miss_before_T2"] = not bool(effective)
    if effective:
        day = int(effective[0])
        result["first_effective_alert_day"] = day
        result["lead_to_T2_days"] = upper - day
        result["signed_timing_from_T1_days"] = day - float(curve.truth_t1)
    return result

def evaluate_thresholds(
    curves: Sequence[Curve],
    score_map: Mapping[str, Mapping[str, np.ndarray]],
    opportunity_map: Mapping[str, Mapping[str, np.ndarray]],
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for lock in thresholds.itertuples(index=False):
        for curve in curves:
            rows.append(
                curve_result(
                    curve,
                    lock.method,
                    lock.regime,
                    float(lock.threshold),
                    score_map[curve.curve_id][lock.method],
                    opportunity_map[curve.curve_id][lock.method],
                )
            )
    return pd.DataFrame(rows)


def summarize_group(group: pd.DataFrame) -> Dict[str, float]:
    outbreak = group[group.kind == "outbreak"]
    null = group[group.kind == "no_outbreak"]
    detected = outbreak[outbreak.detected_in_window]
    return {
        "n_outbreak": int(len(outbreak)),
        "n_null": int(len(null)),
        "detection_before_T2": float(outbreak.detected_in_window.mean()),
        "miss_rate": float(outbreak.miss_before_T2.mean()),
        "premature_curve_rate": float(outbreak.premature.mean()),
        "median_lead_to_T2_days": float(detected.lead_to_T2_days.median()) if len(detected) else np.nan,
        "lead_q1_days": float(detected.lead_to_T2_days.quantile(0.25)) if len(detected) else np.nan,
        "lead_q3_days": float(detected.lead_to_T2_days.quantile(0.75)) if len(detected) else np.nan,
        "null_events_per_100d": float(100.0 * null.n_events.sum() / max(null.surveillance_days.sum(), 1)),
        "null_curve_fpr": float(null.any_event.mean()),
        "null_specificity": float(1.0 - null.any_event.mean()),
        "null_alert_day_proportion": float(
            np.average(null.alert_day_proportion, weights=null.surveillance_days)
        ),
        "null_alert_days_per_100d": float(
            100.0 * null.alert_days.sum() / max(null.surveillance_days.sum(), 1)
        ),
        "null_mean_episode_duration_days": float(
            null.alert_days.sum() / max(null.n_events.sum(), 1)
        ),
    }


def stratified_resample(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    parts = []
    for (_, _), group in frame.groupby(["kind", "archetype"], sort=False):
        indices = rng.integers(0, len(group), len(group))
        parts.append(group.iloc[indices])
    return pd.concat(parts, ignore_index=True)


def summarize_with_bootstrap(
    results: pd.DataFrame, n_boot: int, seed: int
) -> pd.DataFrame:
    rows = []
    ci_metrics = (
        "detection_before_T2",
        "miss_rate",
        "premature_curve_rate",
        "median_lead_to_T2_days",
        "null_events_per_100d",
        "null_curve_fpr",
        "null_alert_days_per_100d",
        "null_mean_episode_duration_days",
    )
    for group_index, ((method, regime), group) in enumerate(results.groupby(["method", "regime"], sort=False)):
        point = summarize_group(group)
        rng = np.random.default_rng(seed + 1009 * group_index)
        draws = {metric: [] for metric in ci_metrics}
        for _ in range(n_boot):
            summary = summarize_group(stratified_resample(group, rng))
            for metric in ci_metrics:
                draws[metric].append(summary[metric])
        row = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "regime": regime,
            **point,
        }
        for metric in ci_metrics:
            values = np.asarray(draws[metric], dtype=float)
            values = values[np.isfinite(values)]
            row[f"{metric}_ci_low"] = float(np.quantile(values, 0.025)) if values.size else np.nan
            row[f"{metric}_ci_high"] = float(np.quantile(values, 0.975)) if values.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def forward_known_score(score: np.ndarray, opportunity: np.ndarray, start: int) -> np.ndarray:
    out = np.full(len(score), np.nan)
    last = np.nan
    for day in range(start, len(score)):
        if opportunity[day] and np.isfinite(score[day]):
            last = float(score[day])
        out[day] = last
    # Before the first available update the method is unavailable, not strong
    # negative evidence. Missing values are later imputed to the development
    # baseline for discrimination/calibration and availability is reported.
    return out


def classification_arrays(
    curves: Sequence[Curve],
    score_map: Mapping[str, Mapping[str, np.ndarray]],
    opportunity_map: Mapping[str, Mapping[str, np.ndarray]],
    method: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores, labels, clusters = [], [], []
    for cluster_index, curve in enumerate(curves):
        known = forward_known_score(
            score_map[curve.curve_id][method],
            opportunity_map[curve.curve_id][method],
            curve.start,
        )
        if curve.kind == "outbreak":
            end = int(min(len(known), math.ceil(float(curve.truth_t2))))
            days = np.arange(curve.start, max(curve.start + 1, end))
            lower = float(curve.truth_t1 - 0.5 * curve.gt)
            y = ((days >= lower) & (days < float(curve.truth_t2))).astype(int)
        else:
            days = np.arange(curve.start, len(known))
            y = np.zeros(len(days), dtype=int)
        scores.append(known[days])
        labels.append(y)
        clusters.append(np.full(len(days), cluster_index))
    return np.concatenate(scores), np.concatenate(labels), np.concatenate(clusters)


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    positives = int(y.sum())
    if positives == 0:
        return np.nan
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    score_sorted = score[order]
    ap = 0.0
    tp = fp = 0
    index = 0
    while index < len(y_sorted):
        end = index + 1
        while end < len(y_sorted) and score_sorted[end] == score_sorted[index]:
            end += 1
        group_pos = int(y_sorted[index:end].sum())
        group_total = end - index
        tp += group_pos
        fp += group_total - group_pos
        if group_pos:
            ap += (group_pos / positives) * (tp / max(tp + fp, 1))
        index = end
    return float(ap)


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    ridge: float = 1.0e-6,
    nonnegative_slope: bool = False,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    design = np.column_stack([np.ones(len(x)), x])

    def objective(beta: np.ndarray) -> Tuple[float, np.ndarray]:
        eta = np.clip(design @ beta, -30.0, 30.0)
        p = expit(eta)
        loss = float(np.sum(np.logaddexp(0.0, eta) - y * eta) + 0.5 * ridge * beta[1] ** 2)
        gradient = design.T @ (p - y)
        gradient[1] += ridge * beta[1]
        return loss, gradient

    prevalence = float(np.clip(np.mean(y), 1.0e-6, 1.0 - 1.0e-6))
    initial = np.array([math.log(prevalence / (1.0 - prevalence)), 0.0])
    result = minimize(
        lambda beta: objective(beta)[0],
        initial,
        jac=lambda beta: objective(beta)[1],
        method="L-BFGS-B" if nonnegative_slope else "BFGS",
        bounds=[(None, None), (0.0, None)] if nonnegative_slope else None,
    )
    return np.asarray(result.x, dtype=float)


def discrimination_and_calibration(
    development: Sequence[Curve],
    validation: Sequence[Curve],
    dev_scores: Mapping[str, Mapping[str, np.ndarray]],
    dev_opportunities: Mapping[str, Mapping[str, np.ndarray]],
    val_scores: Mapping[str, Mapping[str, np.ndarray]],
    val_opportunities: Mapping[str, Mapping[str, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        dev_x, dev_y, _ = classification_arrays(
            development, dev_scores, dev_opportunities, method
        )
        val_x, val_y, _ = classification_arrays(
            validation, val_scores, val_opportunities, method
        )
        dev_finite = np.isfinite(dev_x)
        val_finite = np.isfinite(val_x)
        centre = float(np.median(dev_x[dev_finite])) if np.any(dev_finite) else 0.0
        scale = robust_scale(dev_x[dev_finite]) if np.any(dev_finite) else 1.0
        dev_filled = np.where(dev_finite, dev_x, centre)
        val_filled = np.where(val_finite, val_x, centre)
        dev_z = (dev_filled - centre) / scale
        val_z = (val_filled - centre) / scale
        beta = fit_logistic(dev_z, dev_y, nonnegative_slope=True)
        probability = expit(np.clip(beta[0] + beta[1] * val_z, -30.0, 30.0))
        brier = float(np.mean((probability - val_y) ** 2))
        clipped = np.clip(probability, 1.0e-6, 1.0 - 1.0e-6)
        logit_probability = np.log(clipped / (1.0 - clipped))
        recalibration = fit_logistic(logit_probability, val_y)
        rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n_validation_curve_days": int(len(val_y)),
                "validation_positive_prevalence": float(np.mean(val_y)),
                "PR_AUC": average_precision(val_y, val_filled),
                "development_score_availability": float(np.mean(dev_finite)),
                "validation_score_availability": float(np.mean(val_finite)),
                "Brier_score": brier,
                "calibration_intercept": float(recalibration[0]),
                "calibration_slope": float(recalibration[1]),
                "development_platt_intercept": float(beta[0]),
                "development_platt_slope": float(beta[1]),
                "score_centre": centre,
                "score_scale": scale,
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap_comparisons(
    results: pd.DataFrame, n_boot: int, seed: int
) -> pd.DataFrame:
    rows = []
    reference = "EEMD_CWT_stable"
    for regime_index, regime in enumerate(results.regime.unique()):
        subset = results[(results.regime == regime) & (results.kind == "outbreak")]
        pivot_detect = subset.pivot(index="curve_id", columns="method", values="detected_in_window")
        pivot_premature = subset.pivot(index="curve_id", columns="method", values="premature")
        pivot_lead = subset.pivot(index="curve_id", columns="method", values="lead_to_T2_days")
        metadata = subset.drop_duplicates("curve_id").set_index("curve_id")[["archetype"]]
        for method_index, method in enumerate(METHODS):
            if method == reference:
                continue
            common = pivot_detect.index.intersection(metadata.index)
            point_detection = float(
                pivot_detect.loc[common, method].astype(float).mean()
                - pivot_detect.loc[common, reference].astype(float).mean()
            )
            point_premature = float(
                pivot_premature.loc[common, method].astype(float).mean()
                - pivot_premature.loc[common, reference].astype(float).mean()
            )
            both = pivot_lead.loc[common, [method, reference]].dropna()
            point_lead = float((both[method] - both[reference]).median()) if len(both) else np.nan
            rng = np.random.default_rng(seed + 10007 * regime_index + 101 * method_index)
            detection_draws, premature_draws, lead_draws = [], [], []
            for _ in range(n_boot):
                sampled = []
                for _, ids in metadata.loc[common].groupby("archetype"):
                    values = ids.index.to_numpy()
                    sampled.extend(values[rng.integers(0, len(values), len(values))])
                detection_draws.append(
                    float(
                        pivot_detect.loc[sampled, method].astype(float).mean()
                        - pivot_detect.loc[sampled, reference].astype(float).mean()
                    )
                )
                premature_draws.append(
                    float(
                        pivot_premature.loc[sampled, method].astype(float).mean()
                        - pivot_premature.loc[sampled, reference].astype(float).mean()
                    )
                )
                lead = pivot_lead.loc[sampled, [method, reference]].dropna()
                if len(lead):
                    lead_draws.append(float((lead[method] - lead[reference]).median()))
            rows.append(
                {
                    "regime": regime,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "reference": reference,
                    "detection_difference": point_detection,
                    "detection_difference_ci_low": float(np.quantile(detection_draws, 0.025)),
                    "detection_difference_ci_high": float(np.quantile(detection_draws, 0.975)),
                    "premature_difference": point_premature,
                    "premature_difference_ci_low": float(np.quantile(premature_draws, 0.025)),
                    "premature_difference_ci_high": float(np.quantile(premature_draws, 0.975)),
                    "median_paired_lead_difference_days": point_lead,
                    "median_paired_lead_difference_ci_low": (
                        float(np.quantile(lead_draws, 0.025)) if lead_draws else np.nan
                    ),
                    "median_paired_lead_difference_ci_high": (
                        float(np.quantile(lead_draws, 0.975)) if lead_draws else np.nan
                    ),
                    "n_both_detected": int(len(both)),
                }
            )
    return pd.DataFrame(rows)


def save_score_arrays(
    path: Path,
    curves: Sequence[Curve],
    score_map: Mapping[str, Mapping[str, np.ndarray]],
    opportunity_map: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    arrays: Dict[str, np.ndarray] = {}
    for index, curve in enumerate(curves):
        arrays[f"scores_{index}"] = np.vstack([score_map[curve.curve_id][method] for method in METHODS])
        arrays[f"opportunities_{index}"] = np.vstack(
            [opportunity_map[curve.curve_id][method] for method in METHODS]
        ).astype(np.uint8)
    np.savez_compressed(path, **arrays)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    cache_root = args.v31_cache.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    development = load_curves(source, "development")
    validation = [] if args.development_only else load_curves(source, "validation")
    dev_features = pd.read_csv(
        cache_root / "development_v31_full_scores.csv",
        keep_default_na=True,
    )
    val_features = (
        pd.DataFrame()
        if args.development_only
        else pd.read_csv(
            cache_root / "validation_v31_full_scores.csv",
            keep_default_na=True,
        )
    )

    dev_scores, dev_opportunities = compute_split_scores(
        development, dev_features, step=3, workers=max(1, args.workers)
    )
    if args.development_only:
        thresholds = calibrate_thresholds(development, dev_scores, dev_opportunities)
        apparent_results = evaluate_thresholds(
            development, dev_scores, dev_opportunities, thresholds
        )
        apparent_summary = summarize_with_bootstrap(
            apparent_results, args.bootstrap, args.seed
        )
        apparent_discrimination = discrimination_and_calibration(
            development,
            development,
            dev_scores,
            dev_opportunities,
            dev_scores,
            dev_opportunities,
        )
        thresholds.to_csv(
            output / "development_threshold_locks.csv",
            index=False,
            encoding="utf-8-sig",
        )
        apparent_results.to_csv(
            output / "development_apparent_curve_level.csv",
            index=False,
            encoding="utf-8-sig",
        )
        apparent_summary.to_csv(
            output / "development_apparent_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        apparent_discrimination.to_csv(
            output / "development_apparent_discrimination_calibration.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(apparent_summary.to_string(index=False))
        print(apparent_discrimination.to_string(index=False))
        return

    val_scores, val_opportunities = compute_split_scores(
        validation, val_features, step=3, workers=max(1, args.workers)
    )

    if args.threshold_locks is not None:
        thresholds = pd.read_csv(args.threshold_locks.resolve())
    else:
        thresholds = calibrate_thresholds(development, dev_scores, dev_opportunities)
    curve_results = evaluate_thresholds(validation, val_scores, val_opportunities, thresholds)
    summary = summarize_with_bootstrap(curve_results, args.bootstrap, args.seed)
    discrimination = discrimination_and_calibration(
        development,
        validation,
        dev_scores,
        dev_opportunities,
        val_scores,
        val_opportunities,
    )
    pairwise = paired_bootstrap_comparisons(curve_results, args.bootstrap, args.seed + 500003)

    thresholds.to_csv(output / "development_threshold_locks.csv", index=False, encoding="utf-8-sig")
    curve_results.to_csv(output / "outer_validation_curve_level.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "outer_validation_summary.csv", index=False, encoding="utf-8-sig")
    discrimination.to_csv(output / "discrimination_calibration_summary.csv", index=False, encoding="utf-8-sig")
    pairwise.to_csv(output / "paired_comparisons_vs_eemd_stable.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "method": "Farrington Flexible",
                "status": "not_primary_eligible",
                "reason": "Canonical historical and seasonal baselines are unavailable in 220-day short-history trajectories.",
                "implemented_proxy": "Farrington_style_GLM",
            },
            {
                "method": "Retrospective PELT",
                "status": "excluded_from_operational_primary",
                "reason": "A full-series retrospective changepoint is not a prospective alert; sequential Poisson GLR and BOCPD were used.",
                "implemented_proxy": "Poisson_GLR; BOCPD",
            },
        ]
    ).to_csv(output / "method_eligibility.csv", index=False, encoding="utf-8-sig")

    save_score_arrays(output / "development_score_arrays.npz", development, dev_scores, dev_opportunities)
    save_score_arrays(output / "validation_score_arrays.npz", validation, val_scores, val_opportunities)
    pd.DataFrame(
        [
            {
                "array_file": f"{split}_score_arrays.npz",
                "array_index": index,
                "curve_id": curve.curve_id,
                "pair_id": curve.pair_id,
                "split": curve.split,
                "kind": curve.kind,
                "archetype": curve.archetype,
                "GT": curve.gt,
                "truth_T1_observable": curve.truth_t1,
                "truth_T2_observable": curve.truth_t2,
                "common_start": curve.start,
            }
            for split, curves in (("development", development), ("validation", validation))
            for index, curve in enumerate(curves)
        ]
    ).to_csv(output / "score_array_manifest.csv", index=False, encoding="utf-8-sig")

    protocol = {
        "analysis": "unified strict-prefix operational benchmark",
        "version": "2.0-full-v31.1-morlet-w6",
        "seed": args.seed,
        "bootstrap_replicates": args.bootstrap,
        "source_directory": str(source),
        "v31_cache_directory": str(cache_root),
        "development": {
            "outbreak": sum(c.kind == "outbreak" for c in development),
            "matched_no_outbreak": sum(c.kind == "no_outbreak" for c in development),
        },
        "outer_validation": {
            "outbreak": sum(c.kind == "outbreak" for c in validation),
            "matched_no_outbreak": sum(c.kind == "no_outbreak" for c in validation),
        },
        "archetypes": sorted({curve.archetype for curve in validation}),
        "data_model": "integer renewal process with negative-binomial transmission, ascertainment, reporting delay and weekday batching",
        "T1": "latent Rt crossing 1 plus simulated median reporting delay",
        "T2": "latent Rt ramp reaching 90% of the post-transition level plus simulated median reporting delay",
        "effective_window": "[T1 - 0.5 GT, T2)",
        "lead_time": "T2 minus first alert in the effective window; misses remain misses and are not assigned zero lead",
        "threshold_calibration": "development no-outbreak curves only; frozen before outer validation",
        "threshold_lock_file": str(args.threshold_locks.resolve()) if args.threshold_locks else None,
        "threshold_lock_sha256": sha256(args.threshold_locks.resolve()) if args.threshold_locks else None,
        "regimes": list(REGIMES),
        "common_update_interval_days": 3,
        "alarm_state": {
            "reset_after_below_threshold_updates": 2,
            "cooldown_days": 7,
            "primary_equal_load_measure": "active alert days per 100 surveillance days",
            "secondary_load_measures": [
                "alert episodes per 100 surveillance days",
                "mean alert episode duration",
                "proportion of null curves with any alert",
            ],
        },
        "EEMD_CWT_first": "after six strict-prefix updates: max(full-V31.1 structural strength - 0.04 * candidate age in generation intervals, valid raw-growth margin)",
        "EEMD_CWT_stable": "trailing minimum across three consecutive scheduled updates; a sustained-support tier that is never backdated",
        "PR_AUC_target": "curve-day membership in the effective response window, evaluated before T2 and on all no-outbreak surveillance days",
        "calibration": "monotone Platt mapping fitted on development curve-days; unavailable pre-update scores are imputed to the development baseline, not an extreme sentinel; Brier score and calibration intercept/slope are evaluated in outer validation",
       "claim_boundaries": [
            "This is internal synthetic validation, not external accuracy validation.",
            "EEMD-CWT rows use the complete prospective V31.1 Morlet-w6 feature and gate chain, not the mother-wavelet single-band energy qualification proxy.",
            "Farrington_style_GLM is a short-history quasi-Poisson analogue, not canonical Farrington Flexible; canonical Farrington is conditionally ineligible without multi-year seasonal history.",
            "Sequential GLR and BOCPD are operational changepoint comparators; retrospective full-series PELT is not treated as an online alert.",
            "The benchmark-specific T1 and T2 do not replace the core framework's phase definitions.",
       ],
        "source_hashes": {
            name: sha256(source / name)
            for name in (
                "development_curve_metadata.csv",
                "development_curve_arrays.npz",
                "validation_curve_metadata.csv",
                "validation_curve_arrays.npz",
            )
        },
        "v31_cache_hashes": {
            name: sha256(cache_root / name)
            for name in (
                "development_v31_full_scores.csv",
                "validation_v31_full_scores.csv",
                "development_v31_full_audit.csv",
                "validation_v31_full_audit.csv",
            )
        },
    }
    json_write(output / "protocol_and_provenance.json", protocol)
    print(summary.to_string(index=False))
    print(discrimination.to_string(index=False))


if __name__ == "__main__":
    main()
