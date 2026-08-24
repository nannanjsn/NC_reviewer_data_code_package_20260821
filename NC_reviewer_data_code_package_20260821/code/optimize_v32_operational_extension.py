"""Development-locked V32 operational extension for full V31.1 evidence.

This script does not modify or rerun the V31.1 phase-segmentation algorithm.
It reads the frozen strict-prefix Morlet-w6/V31.1 score cache and the saved
benchmark score streams. Candidate operational readouts are selected only on
the development split by paired, archetype-stratified cross-validation.

The primary selection endpoint is outbreak detection in the prespecified
observable [T1 - 0.5 GT, T2) response window under three operational
constraints: 1 and 3 alert-days per 100 surveillance days, and 90% null-curve
specificity. PR-AUC is a secondary guardrail, not the optimization endpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

import unified_operational_benchmark as ub


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_SOURCE = ROOT / "母小波确认" / "outputs_v2" / "20260716_103352_formal_v2"
DEFAULT_BENCHMARK = HERE / "outputs_unified_operational_benchmark_v31_full"
DEFAULT_OUTPUT = HERE / "outputs_v32_operational_extension"

REGIMES = (
    ("burden_low", "alert_days_per_100d", 1.0),
    ("burden_high", "alert_days_per_100d", 3.0),
    ("specificity_90", "null_curve_fpr", 0.10),
)


@dataclass(frozen=True)
class Policy:
    name: str
    mode: str
    review_days: int = 0
    cooldown_days: int = 7
    reset_updates: int = 2


POLICIES = (
    Policy("legacy_reset2", "legacy", reset_updates=2),
    Policy("review_1d", "review", review_days=1, cooldown_days=7),
    Policy("review_3d", "review", review_days=3, cooldown_days=7),
    Policy("review_5d", "review", review_days=5, cooldown_days=7),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--evaluate-seen-validation",
        action="store_true",
        help=(
            "Diagnostic only: apply the development lock to the already-seen "
            "validation split. Never label this as a new independent validation."
        ),
    )
    return parser.parse_args()


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


def load_saved_split(
    benchmark: Path,
    source: Path,
    split: str,
) -> Tuple[
    list[ub.Curve],
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, np.ndarray]],
]:
    manifest = pd.read_csv(benchmark / "score_array_manifest.csv")
    manifest = manifest[manifest.split == split].sort_values("array_index")
    archive = np.load(benchmark / f"{split}_score_arrays.npz")
    curves = ub.load_curves(source, split)
    curve_by_id = {curve.curve_id: curve for curve in curves}
    score_map: Dict[str, Dict[str, np.ndarray]] = {}
    opportunity_map: Dict[str, Dict[str, np.ndarray]] = {}
    for row in manifest.itertuples(index=False):
        curve_id = str(row.curve_id)
        index = int(row.array_index)
        matrix = np.asarray(archive[f"scores_{index}"], dtype=float)
        opportunities = np.asarray(
            archive[f"opportunities_{index}"], dtype=bool
        )
        if matrix.shape[0] != len(ub.METHODS):
            raise ValueError("Saved score matrix does not match benchmark method order")
        score_map[curve_id] = {
            method: matrix[j].copy() for j, method in enumerate(ub.METHODS)
        }
        opportunity_map[curve_id] = {
            method: opportunities[j].copy()
            for j, method in enumerate(ub.METHODS)
        }
    missing = sorted(set(curve_by_id) - set(score_map))
    if missing:
        raise ValueError(f"Missing saved scores for {len(missing)} curves")
    return curves, score_map, opportunity_map


def robust_location_scale(values: Iterable[float]) -> Tuple[float, float]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    median = float(np.median(x))
    mad = 1.4826 * float(np.median(np.abs(x - median)))
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    return median, max(mad, 0.25 * sd, 1.0e-6)


def development_null_scalers(
    curves: Sequence[ub.Curve],
    scores: Mapping[str, Mapping[str, np.ndarray]],
    opportunities: Mapping[str, Mapping[str, np.ndarray]],
) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for method in ("EEMD_CWT_first", "EEMD_CWT_stable", "growth_rate", "Rt_P_gt_1"):
        pooled = []
        for curve in curves:
            if curve.kind != "no_outbreak":
                continue
            values = scores[curve.curve_id][method]
            mask = opportunities[curve.curve_id][method]
            pooled.extend(values[mask])
        out[method] = robust_location_scale(pooled)
    return out


def scheduled_aggregate(
    values: np.ndarray,
    opportunity: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    output = np.full(len(values), np.nan)
    valid_days = np.flatnonzero(opportunity & np.isfinite(values))
    history: list[float] = []
    state = np.nan
    for day_value in valid_days:
        day = int(day_value)
        value = float(values[day])
        history.append(value)
        if mode == "first":
            result = value
        elif mode == "mean2":
            result = float(np.mean(history[-2:])) if len(history) >= 2 else np.nan
        elif mode == "median3":
            result = float(np.median(history[-3:])) if len(history) >= 3 else np.nan
        elif mode == "mean3":
            result = float(np.mean(history[-3:])) if len(history) >= 3 else np.nan
        elif mode == "ewma05":
            state = value if not np.isfinite(state) else 0.5 * value + 0.5 * state
            result = float(state) if len(history) >= 2 else np.nan
        else:
            raise ValueError(mode)
        if np.isfinite(result):
            output[day] = result
    return output, np.isfinite(output)


def standardized(
    values: np.ndarray,
    centre: float,
    scale: float,
) -> np.ndarray:
    return (np.asarray(values, dtype=float) - centre) / scale


def make_candidates(
    curves: Sequence[ub.Curve],
    base_scores: Mapping[str, Mapping[str, np.ndarray]],
    base_opportunities: Mapping[str, Mapping[str, np.ndarray]],
    scalers: Mapping[str, Tuple[float, float]],
) -> Tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, object]],
]:
    candidate_scores: Dict[str, Dict[str, np.ndarray]] = {}
    candidate_opportunities: Dict[str, Dict[str, np.ndarray]] = {}
    definitions: Dict[str, Dict[str, object]] = {}

    pure_modes = ("first", "mean2", "median3", "mean3", "ewma05")
    for mode in pure_modes:
        name = f"pure_{mode}"
        definitions[name] = {
            "family": "pure_V31.1_operational_readout",
            "formula": f"{mode} aggregation of scheduled V31.1 first-indication evidence",
        }
    definitions["pure_frozen_stable"] = {
        "family": "pure_V31.1_operational_readout",
        "formula": "frozen V31.1 sustained-support score (three-update minimum)",
    }
    for growth_weight in (0.25, 0.50, 0.75):
        name = f"fusion_growth_w{int(100 * growth_weight):02d}"
        definitions[name] = {
            "family": "V32_structural_growth_extension",
            "formula": (
                f"{1-growth_weight:.2f} * standardized V31.1 mean-of-two evidence "
                f"+ {growth_weight:.2f} * standardized 3-day growth evidence"
            ),
            "growth_weight": growth_weight,
        }
    for rt_weight in (0.25, 0.50, 0.65, 0.75, 0.90):
        name = f"fusion_rt_w{int(100 * rt_weight):02d}"
        definitions[name] = {
            "family": "V32_structural_Rt_extension",
            "formula": (
                f"{1-rt_weight:.2f} * standardized V31.1 mean-of-two evidence "
                f"+ {rt_weight:.2f} * standardized online P(Rt>1) evidence"
            ),
            "rt_weight": rt_weight,
        }
        definitions[f"fusion_rt_first_w{int(100 * rt_weight):02d}"] = {
            "family": "V32_structural_Rt_extension",
            "formula": (
                f"{1-rt_weight:.2f} * standardized V31.1 first-indication evidence "
                f"+ {rt_weight:.2f} * standardized online P(Rt>1) evidence"
            ),
            "rt_weight": rt_weight,
        }

    for curve in curves:
        curve_id = curve.curve_id
        candidate_scores[curve_id] = {}
        candidate_opportunities[curve_id] = {}
        first = base_scores[curve_id]["EEMD_CWT_first"]
        first_opportunity = base_opportunities[curve_id]["EEMD_CWT_first"]
        aggregates = {}
        for mode in pure_modes:
            score, opportunity = scheduled_aggregate(first, first_opportunity, mode)
            name = f"pure_{mode}"
            aggregates[mode] = (score, opportunity)
            candidate_scores[curve_id][name] = score
            candidate_opportunities[curve_id][name] = opportunity
        stable = base_scores[curve_id]["EEMD_CWT_stable"].copy()
        stable_opportunity = base_opportunities[curve_id]["EEMD_CWT_stable"].copy()
        candidate_scores[curve_id]["pure_frozen_stable"] = stable
        candidate_opportunities[curve_id]["pure_frozen_stable"] = stable_opportunity

        structure, structure_opportunity = aggregates["mean2"]
        s_centre, s_scale = scalers["EEMD_CWT_first"]
        structure_z = standardized(structure, s_centre, s_scale)
        for comparator, weights, prefix in (
            ("growth_rate", (0.25, 0.50, 0.75), "fusion_growth_w"),
            ("Rt_P_gt_1", (0.25, 0.50, 0.65, 0.75, 0.90), "fusion_rt_w"),
        ):
            comp = base_scores[curve_id][comparator]
            comp_opportunity = base_opportunities[curve_id][comparator]
            c_centre, c_scale = scalers[comparator]
            comp_z = standardized(comp, c_centre, c_scale)
            common = structure_opportunity & comp_opportunity
            for weight in weights:
                name = f"{prefix}{int(100 * weight):02d}"
                fused = np.full(len(structure), np.nan)
                fused[common] = (
                    (1.0 - weight) * structure_z[common]
                    + weight * comp_z[common]
                )
                candidate_scores[curve_id][name] = fused
                candidate_opportunities[curve_id][name] = common.copy()
        rt = base_scores[curve_id]["Rt_P_gt_1"]
        rt_opportunity = base_opportunities[curve_id]["Rt_P_gt_1"]
        rt_centre, rt_scale = scalers["Rt_P_gt_1"]
        rt_z = standardized(rt, rt_centre, rt_scale)
        first_score, first_available = aggregates["first"]
        first_z = standardized(first_score, s_centre, s_scale)
        first_common = first_available & rt_opportunity
        for weight in (0.25, 0.50, 0.65, 0.75, 0.90):
            name = f"fusion_rt_first_w{int(100 * weight):02d}"
            fused = np.full(len(first_score), np.nan)
            fused[first_common] = (
                (1.0 - weight) * first_z[first_common]
                + weight * rt_z[first_common]
            )
            candidate_scores[curve_id][name] = fused
            candidate_opportunities[curve_id][name] = first_common.copy()
    return candidate_scores, candidate_opportunities, definitions


def alarm_trace_policy(
    score: np.ndarray,
    opportunity: np.ndarray,
    threshold: float,
    policy: Policy,
) -> Tuple[list[int], np.ndarray, list[int]]:
    if policy.mode == "legacy":
        return ub.alarm_trace(
            score,
            opportunity,
            threshold,
            cooldown=policy.cooldown_days,
            reset_updates=policy.reset_updates,
        )
    if policy.mode != "review":
        raise ValueError(policy.mode)
    n = len(score)
    active = np.zeros(n, dtype=bool)
    events: list[int] = []
    durations: list[int] = []
    armed = True
    cooldown_until = -1
    for day_value in np.flatnonzero(opportunity):
        day = int(day_value)
        above = bool(np.isfinite(score[day]) and score[day] >= threshold)
        if not above:
            armed = True
            continue
        if armed and day >= cooldown_until:
            end = min(n, day + max(1, policy.review_days))
            events.append(day)
            active[day:end] = True
            durations.append(end - day)
            cooldown_until = end + max(0, policy.cooldown_days)
            armed = False
    return events, active, durations


def null_metrics(
    curves: Sequence[ub.Curve],
    scores: Mapping[str, Mapping[str, np.ndarray]],
    opportunities: Mapping[str, Mapping[str, np.ndarray]],
    candidate: str,
    threshold: float,
    policy: Policy,
) -> Dict[str, float]:
    selected = [curve for curve in curves if curve.kind == "no_outbreak"]
    events = curves_with_event = alert_days = surveillance_days = 0
    durations: list[int] = []
    for curve in selected:
        event_days, active, episode_lengths = alarm_trace_policy(
            scores[curve.curve_id][candidate],
            opportunities[curve.curve_id][candidate],
            threshold,
            policy,
        )
        events += len(event_days)
        curves_with_event += int(bool(event_days))
        alert_days += int(np.sum(active[curve.start:]))
        surveillance_days += max(1, len(active) - curve.start)
        durations.extend(episode_lengths)
    return {
        "events_per_100d": 100.0 * events / max(1, surveillance_days),
        "null_curve_fpr": curves_with_event / max(1, len(selected)),
        "alert_days_per_100d": 100.0 * alert_days / max(1, surveillance_days),
        "mean_episode_duration_days": float(np.mean(durations)) if durations else 0.0,
    }


def threshold_candidates(
    curves: Sequence[ub.Curve],
    scores: Mapping[str, Mapping[str, np.ndarray]],
    opportunities: Mapping[str, Mapping[str, np.ndarray]],
    candidate: str,
) -> np.ndarray:
    pooled = []
    for curve in curves:
        if curve.kind != "no_outbreak":
            continue
        values = scores[curve.curve_id][candidate]
        mask = opportunities[curve.curve_id][candidate]
        pooled.extend(values[mask])
    finite = np.asarray(pooled, dtype=float)
    finite = finite[np.isfinite(finite)]
    return np.r_[np.inf, np.unique(finite)[::-1], -np.inf]


def calibrate_one(
    curves: Sequence[ub.Curve],
    scores: Mapping[str, Mapping[str, np.ndarray]],
    opportunities: Mapping[str, Mapping[str, np.ndarray]],
    candidate: str,
    policy: Policy,
    criterion: str,
    target: float,
) -> Tuple[float, Dict[str, float]]:
    candidates = threshold_candidates(curves, scores, opportunities, candidate)
    cache: Dict[int, Dict[str, float]] = {}

    def at(index: int) -> Dict[str, float]:
        index = int(np.clip(index, 0, len(candidates) - 1))
        if index not in cache:
            cache[index] = null_metrics(
                curves,
                scores,
                opportunities,
                candidate,
                float(candidates[index]),
                policy,
            )
        return cache[index]

    left, right = 0, len(candidates) - 1
    while left < right:
        middle = (left + right) // 2
        if at(middle)[criterion] >= target:
            right = middle
        else:
            left = middle + 1
    nearby = range(max(0, left - 4), min(len(candidates), left + 5))
    best_index = min(
        nearby,
        key=lambda index: (
            abs(at(index)[criterion] - target),
            at(index)[criterion] > target,
            -float(candidates[index]),
        ),
    )
    return float(candidates[best_index]), at(best_index)


def evaluate(
    curves: Sequence[ub.Curve],
    scores: Mapping[str, Mapping[str, np.ndarray]],
    opportunities: Mapping[str, Mapping[str, np.ndarray]],
    candidate: str,
    threshold: float,
    policy: Policy,
) -> Dict[str, float]:
    outbreak = [curve for curve in curves if curve.kind == "outbreak"]
    null = [curve for curve in curves if curve.kind == "no_outbreak"]
    detected: list[bool] = []
    premature: list[bool] = []
    leads: list[float] = []
    for curve in outbreak:
        events, _, _ = alarm_trace_policy(
            scores[curve.curve_id][candidate],
            opportunities[curve.curve_id][candidate],
            threshold,
            policy,
        )
        lower = float(curve.truth_t1 - 0.5 * curve.gt)
        upper = float(curve.truth_t2)
        effective = [day for day in events if lower <= day < upper]
        detected.append(bool(effective))
        premature.append(any(day < lower for day in events))
        if effective:
            leads.append(upper - effective[0])
    burden = null_metrics(
        null, scores, opportunities, candidate, threshold, policy
    )
    return {
        "n_outbreak": len(outbreak),
        "n_null": len(null),
        "detection_before_T2": float(np.mean(detected)),
        "miss_rate": 1.0 - float(np.mean(detected)),
        "premature_curve_rate": float(np.mean(premature)),
        "median_lead_to_T2_days": float(np.median(leads)) if leads else np.nan,
        **burden,
    }


def paired_folds(curves: Sequence[ub.Curve], n_folds: int) -> Dict[str, int]:
    archetype = {curve.pair_id: curve.archetype for curve in curves}
    folds: Dict[str, int] = {}
    for name in sorted(set(archetype.values())):
        pairs = sorted(pair for pair, value in archetype.items() if value == name)
        for index, pair in enumerate(pairs):
            folds[pair] = index % n_folds
    return folds


def classification_pr_auc(
    curves: Sequence[ub.Curve],
    scores: Mapping[str, Mapping[str, np.ndarray]],
    opportunities: Mapping[str, Mapping[str, np.ndarray]],
    candidate: str,
) -> float:
    x, y, _ = ub.classification_arrays(curves, scores, opportunities, candidate)
    baseline = float(np.nanmedian(x))
    return float(
        ub.average_precision(y, np.where(np.isfinite(x), x, baseline))
    )


def cross_validated_grid(
    curves: Sequence[ub.Curve],
    scores: Mapping[str, Mapping[str, np.ndarray]],
    opportunities: Mapping[str, Mapping[str, np.ndarray]],
    definitions: Mapping[str, Mapping[str, object]],
    n_folds: int,
) -> pd.DataFrame:
    fold_map = paired_folds(curves, n_folds)
    rows = []
    for candidate in definitions:
        overall_pr = classification_pr_auc(
            curves, scores, opportunities, candidate
        )
        for policy in POLICIES:
            fold_rows = []
            for fold_index in range(n_folds):
                training = [
                    curve for curve in curves
                    if fold_map[curve.pair_id] != fold_index
                ]
                held_out = [
                    curve for curve in curves
                    if fold_map[curve.pair_id] == fold_index
                ]
                for regime, criterion, target in REGIMES:
                    threshold, training_burden = calibrate_one(
                        training,
                        scores,
                        opportunities,
                        candidate,
                        policy,
                        criterion,
                        target,
                    )
                    metrics = evaluate(
                        held_out,
                        scores,
                        opportunities,
                        candidate,
                        threshold,
                        policy,
                    )
                    fold_rows.append(
                        {
                            "candidate": candidate,
                            "family": definitions[candidate]["family"],
                            "policy": policy.name,
                            "fold": fold_index,
                            "regime": regime,
                            "criterion": criterion,
                            "target": target,
                            "threshold": threshold,
                            "development_training_constraint": training_burden[criterion],
                            **metrics,
                        }
                    )
            detail = pd.DataFrame(fold_rows)
            for regime, group in detail.groupby("regime", sort=False):
                rows.append(
                    {
                        "candidate": candidate,
                        "family": definitions[candidate]["family"],
                        "policy": policy.name,
                        "regime": regime,
                        "overall_development_PR_AUC": overall_pr,
                        "mean_fold_detection_before_T2": float(
                            group.detection_before_T2.mean()
                        ),
                        "minimum_fold_detection_before_T2": float(
                            group.detection_before_T2.min()
                        ),
                        "mean_fold_median_lead_days": float(
                            group.median_lead_to_T2_days.mean()
                        ),
                        "mean_fold_premature_curve_rate": float(
                            group.premature_curve_rate.mean()
                        ),
                        "mean_fold_null_alert_days_per_100d": float(
                            group.alert_days_per_100d.mean()
                        ),
                        "mean_fold_null_curve_fpr": float(
                            group.null_curve_fpr.mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def select_candidate(summary: pd.DataFrame) -> Tuple[str, str, pd.DataFrame]:
    wide = summary.pivot_table(
        index=["candidate", "family", "policy", "overall_development_PR_AUC"],
        columns="regime",
        values="mean_fold_detection_before_T2",
    ).reset_index()
    regime_columns = [item[0] for item in REGIMES]
    wide["mean_detection"] = wide[regime_columns].mean(axis=1)
    wide["minimum_regime_detection"] = wide[regime_columns].min(axis=1)
    baseline_pr = float(
        wide.loc[
            (wide.candidate == "pure_frozen_stable")
            & (wide.policy == "legacy_reset2"),
            "overall_development_PR_AUC",
        ].iloc[0]
    )
    # PR-AUC is a guardrail: allow a small trade for the endpoint that is
    # actually claimed, but reject candidates with materially degraded ranking.
    wide["PR_AUC_guardrail"] = wide.overall_development_PR_AUC >= 0.90 * baseline_pr
    eligible = wide[wide.PR_AUC_guardrail].copy()
    eligible = eligible.sort_values(
        [
            "mean_detection",
            "minimum_regime_detection",
            "specificity_90",
            "burden_low",
            "overall_development_PR_AUC",
        ],
        ascending=False,
    )
    selected = eligible.iloc[0]
    return str(selected.candidate), str(selected.policy), wide


def full_development_locks(
    curves: Sequence[ub.Curve],
    scores: Mapping[str, Mapping[str, np.ndarray]],
    opportunities: Mapping[str, Mapping[str, np.ndarray]],
    candidate: str,
    policy: Policy,
) -> pd.DataFrame:
    rows = []
    for regime, criterion, target in REGIMES:
        threshold, burden = calibrate_one(
            curves,
            scores,
            opportunities,
            candidate,
            policy,
            criterion,
            target,
        )
        rows.append(
            {
                "candidate": candidate,
                "policy": policy.name,
                "regime": regime,
                "criterion": criterion,
                "target": target,
                "threshold": threshold,
                **{f"development_{key}": value for key, value in burden.items()},
            }
        )
    return pd.DataFrame(rows)


def apply_locks(
    curves: Sequence[ub.Curve],
    scores: Mapping[str, Mapping[str, np.ndarray]],
    opportunities: Mapping[str, Mapping[str, np.ndarray]],
    locks: pd.DataFrame,
    candidate: str,
    policy: Policy,
) -> pd.DataFrame:
    rows = []
    for lock in locks.itertuples(index=False):
        rows.append(
            {
                "candidate": candidate,
                "policy": policy.name,
                "regime": lock.regime,
                **evaluate(
                    curves,
                    scores,
                    opportunities,
                    candidate,
                    float(lock.threshold),
                    policy,
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    benchmark = args.benchmark.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    development, development_base, development_base_opportunity = load_saved_split(
        benchmark, source, "development"
    )
    scalers = development_null_scalers(
        development, development_base, development_base_opportunity
    )
    development_scores, development_opportunities, definitions = make_candidates(
        development,
        development_base,
        development_base_opportunity,
        scalers,
    )
    cv_summary = cross_validated_grid(
        development,
        development_scores,
        development_opportunities,
        definitions,
        max(2, int(args.folds)),
    )
    selected_candidate, selected_policy_name, selection_table = select_candidate(
        cv_summary
    )
    selected_policy = next(
        policy for policy in POLICIES if policy.name == selected_policy_name
    )
    locks = full_development_locks(
        development,
        development_scores,
        development_opportunities,
        selected_candidate,
        selected_policy,
    )
    development_result = apply_locks(
        development,
        development_scores,
        development_opportunities,
        locks,
        selected_candidate,
        selected_policy,
    )

    cv_summary.to_csv(output / "development_cv_regime_summary.csv", index=False)
    selection_table.to_csv(output / "development_candidate_ranking.csv", index=False)
    locks.to_csv(output / "development_threshold_locks.csv", index=False)
    development_result.to_csv(
        output / "development_locked_apparent_performance.csv", index=False
    )

    seen_validation_result = None
    if args.evaluate_seen_validation:
        validation, validation_base, validation_base_opportunity = load_saved_split(
            benchmark, source, "validation"
        )
        validation_scores, validation_opportunities, _ = make_candidates(
            validation,
            validation_base,
            validation_base_opportunity,
            scalers,
        )
        seen_validation_result = apply_locks(
            validation,
            validation_scores,
            validation_opportunities,
            locks,
            selected_candidate,
            selected_policy,
        )
        seen_validation_result.to_csv(
            output / "seen_validation_diagnostic_not_independent.csv", index=False
        )

    lock = {
        "status": "development_locked_requires_new_independent_validation",
        "core_algorithm_modified": False,
        "selected_candidate": selected_candidate,
        "selected_policy": selected_policy_name,
        "selected_definition": definitions[selected_candidate],
        "selection_endpoint": (
            "mean paired-fold T2-pre detection across 1 and 3 alert-days/100d "
            "and 90% null-curve specificity"
        ),
        "PR_AUC_guardrail": (
            "development PR-AUC must be at least 90% of frozen stable-score baseline"
        ),
        "development_only_for_selection": True,
        "seen_validation_is_diagnostic_only": bool(args.evaluate_seen_validation),
        "requires_new_seed_outer_validation": True,
        "development_null_scalers": {
            method: {"centre": centre, "scale": scale}
            for method, (centre, scale) in scalers.items()
        },
        "source_hashes": {
            "development_score_arrays.npz": sha256(
                benchmark / "development_score_arrays.npz"
            ),
            "development_curve_arrays.npz": sha256(
                source / "development_curve_arrays.npz"
            ),
        },
    }
    json_write(output / "v32_development_lock.json", lock)
    print(json.dumps(lock, ensure_ascii=False, indent=2))
    print(locks.to_string(index=False))
    print(development_result.to_string(index=False))
    if seen_validation_result is not None:
        print("SEEN VALIDATION DIAGNOSTIC -- NOT INDEPENDENT")
        print(seen_validation_result.to_string(index=False))


if __name__ == "__main__":
    main()
