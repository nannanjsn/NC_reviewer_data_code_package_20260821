"""One-pass independent outer evaluation of the V32 operational extension.

The V31.1 phase-segmentation core and its saved strict-prefix Morlet-w6
outputs are read-only inputs.  All thresholds and calibration maps are fitted
from the original development split.  The new-seed outer split is evaluated
once.  A common one-day review policy is used for every method in the primary
matched-burden comparison so that alarm duration cannot favour V32.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit

import optimize_v32_operational_extension as v32
import unified_operational_benchmark as ub


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_DEVELOPMENT_SOURCE = (
    ROOT / "母小波确认" / "outputs_v2" / "20260716_103352_formal_v2"
)
DEFAULT_BENCHMARK = HERE / "outputs_unified_operational_benchmark_v31_full"
DEFAULT_V32_LOCK = HERE / "outputs_v32_operational_extension"
DEFAULT_OUTER_SOURCE = HERE / "v32_independent_outer_seed_2026073003"
DEFAULT_OUTER_CACHE = HERE / "cache_v32_independent_outer_seed_2026073003"
DEFAULT_OUTPUT = HERE / "outputs_v32_independent_outer_seed_2026073003"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-source", type=Path, default=DEFAULT_DEVELOPMENT_SOURCE)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--v32-lock", type=Path, default=DEFAULT_V32_LOCK)
    parser.add_argument("--outer-source", type=Path, default=DEFAULT_OUTER_SOURCE)
    parser.add_argument("--outer-cache", type=Path, default=DEFAULT_OUTER_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def curve_result_policy(
    curve: ub.Curve,
    method: str,
    method_label: str,
    regime: str,
    threshold: float,
    score: np.ndarray,
    opportunity: np.ndarray,
    policy: v32.Policy,
) -> Dict[str, object]:
    events, active, durations = v32.alarm_trace_policy(
        score, opportunity, threshold, policy
    )
    surveillance_days = max(1, len(score) - curve.start)
    alert_days = int(np.sum(active[curve.start:]))
    result: Dict[str, object] = {
        "curve_id": curve.curve_id,
        "pair_id": curve.pair_id,
        "kind": curve.kind,
        "archetype": curve.archetype,
        "GT": curve.gt,
        "method": method,
        "method_label": method_label,
        "policy": policy.name,
        "regime": regime,
        "threshold": threshold,
        "surveillance_days": surveillance_days,
        "n_events": len(events),
        "any_event": bool(events),
        "first_event_day": events[0] if events else np.nan,
        "alert_days": alert_days,
        "alert_day_proportion": alert_days / surveillance_days,
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


def calibrate_common_policy(
    curves: Sequence[ub.Curve],
    scores: Mapping[str, Mapping[str, np.ndarray]],
    opportunities: Mapping[str, Mapping[str, np.ndarray]],
    policy: v32.Policy,
) -> pd.DataFrame:
    rows = []
    for method in ub.METHODS:
        for regime, criterion, target in v32.REGIMES:
            threshold, burden = v32.calibrate_one(
                curves,
                scores,
                opportunities,
                method,
                policy,
                criterion,
                target,
            )
            rows.append(
                {
                    "method": method,
                    "method_label": ub.METHOD_LABELS[method],
                    "policy": policy.name,
                    "regime": regime,
                    "criterion": criterion,
                    "target": target,
                    "threshold": threshold,
                    **{f"development_{key}": value for key, value in burden.items()},
                }
            )
    return pd.DataFrame(rows)


def evaluate_common_policy(
    curves: Sequence[ub.Curve],
    scores: Mapping[str, Mapping[str, np.ndarray]],
    opportunities: Mapping[str, Mapping[str, np.ndarray]],
    locks: pd.DataFrame,
    policy: v32.Policy,
) -> pd.DataFrame:
    rows = []
    for lock in locks.itertuples(index=False):
        for curve in curves:
            rows.append(
                curve_result_policy(
                    curve,
                    str(lock.method),
                    str(lock.method_label),
                    str(lock.regime),
                    float(lock.threshold),
                    scores[curve.curve_id][str(lock.method)],
                    opportunities[curve.curve_id][str(lock.method)],
                    policy,
                )
            )
    return pd.DataFrame(rows)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, label, policy, regime), group in results.groupby(
        ["method", "method_label", "policy", "regime"], sort=False
    ):
        rows.append(
            {
                "method": method,
                "method_label": label,
                "policy": policy,
                "regime": regime,
                **ub.summarize_group(group),
            }
        )
    return pd.DataFrame(rows)


def single_method_discrimination(
    development: Sequence[ub.Curve],
    validation: Sequence[ub.Curve],
    dev_scores: Mapping[str, Mapping[str, np.ndarray]],
    dev_opportunities: Mapping[str, Mapping[str, np.ndarray]],
    val_scores: Mapping[str, Mapping[str, np.ndarray]],
    val_opportunities: Mapping[str, Mapping[str, np.ndarray]],
    method: str,
    method_label: str,
) -> Dict[str, object]:
    dev_x, dev_y, _ = ub.classification_arrays(
        development, dev_scores, dev_opportunities, method
    )
    val_x, val_y, _ = ub.classification_arrays(
        validation, val_scores, val_opportunities, method
    )
    dev_finite = np.isfinite(dev_x)
    val_finite = np.isfinite(val_x)
    centre = float(np.median(dev_x[dev_finite])) if np.any(dev_finite) else 0.0
    scale = ub.robust_scale(dev_x[dev_finite]) if np.any(dev_finite) else 1.0
    dev_filled = np.where(dev_finite, dev_x, centre)
    val_filled = np.where(val_finite, val_x, centre)
    dev_z = (dev_filled - centre) / scale
    val_z = (val_filled - centre) / scale
    beta = ub.fit_logistic(dev_z, dev_y, nonnegative_slope=True)
    probability = expit(np.clip(beta[0] + beta[1] * val_z, -30.0, 30.0))
    brier = float(np.mean((probability - val_y) ** 2))
    clipped = np.clip(probability, 1.0e-6, 1.0 - 1.0e-6)
    recalibration = ub.fit_logistic(np.log(clipped / (1.0 - clipped)), val_y)
    return {
        "method": method,
        "method_label": method_label,
        "n_validation_curve_days": int(len(val_y)),
        "validation_positive_prevalence": float(np.mean(val_y)),
        "PR_AUC": ub.average_precision(val_y, val_filled),
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


def main() -> None:
    args = parse_args()
    development_source = args.development_source.resolve()
    benchmark = args.benchmark.resolve()
    v32_lock_root = args.v32_lock.resolve()
    outer_source = args.outer_source.resolve()
    outer_cache = args.outer_cache.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    development, dev_base, dev_base_opportunities = v32.load_saved_split(
        benchmark, development_source, "development"
    )
    validation = ub.load_curves(outer_source, "validation")
    validation_cache = pd.read_csv(
        outer_cache / "validation_v31_full_scores.csv", keep_default_na=True
    )
    val_base, val_base_opportunities = ub.compute_split_scores(
        validation, validation_cache, step=3, workers=max(1, int(args.workers))
    )
    ub.save_score_arrays(
        output / "validation_score_arrays.npz",
        validation,
        val_base,
        val_base_opportunities,
    )

    scalers = v32.development_null_scalers(
        development, dev_base, dev_base_opportunities
    )
    dev_candidates, dev_candidate_opportunities, definitions = v32.make_candidates(
        development, dev_base, dev_base_opportunities, scalers
    )
    val_candidates, val_candidate_opportunities, _ = v32.make_candidates(
        validation, val_base, val_base_opportunities, scalers
    )
    lock_meta = json.loads(
        (v32_lock_root / "v32_development_lock.json").read_text(encoding="utf-8")
    )
    selected = str(lock_meta["selected_candidate"])
    policy = next(
        item for item in v32.POLICIES
        if item.name == str(lock_meta["selected_policy"])
    )
    if policy.name != "review_1d":
        raise ValueError("Primary fair comparison requires the locked one-day review policy")

    common_locks = calibrate_common_policy(
        development, dev_base, dev_base_opportunities, policy
    )
    common_locks.to_csv(
        output / "development_common_policy_threshold_locks.csv", index=False
    )
    comparator_results = evaluate_common_policy(
        validation, val_base, val_base_opportunities, common_locks, policy
    )
    comparator_results.to_csv(output / "comparator_outer_curve_level.csv", index=False)

    v32_locks = pd.read_csv(v32_lock_root / "development_threshold_locks.csv")
    v32_rows = []
    v32_label = "V32 EEMD-CWT + online Rt support"
    for lock in v32_locks.itertuples(index=False):
        for curve in validation:
            v32_rows.append(
                curve_result_policy(
                    curve,
                    "EEMD_CWT_V32",
                    v32_label,
                    str(lock.regime),
                    float(lock.threshold),
                    val_candidates[curve.curve_id][selected],
                    val_candidate_opportunities[curve.curve_id][selected],
                    policy,
                )
            )
    v32_results = pd.DataFrame(v32_rows)
    v32_results.to_csv(output / "v32_outer_curve_level.csv", index=False)
    all_results = pd.concat([comparator_results, v32_results], ignore_index=True)
    all_results.to_csv(output / "all_method_outer_curve_level.csv", index=False)
    summary = summarize_results(all_results)
    summary.to_csv(output / "all_method_outer_summary.csv", index=False)

    discrimination = ub.discrimination_and_calibration(
        development,
        validation,
        dev_base,
        dev_base_opportunities,
        val_base,
        val_base_opportunities,
    )
    v32_discrimination = single_method_discrimination(
        development,
        validation,
        dev_candidates,
        dev_candidate_opportunities,
        val_candidates,
        val_candidate_opportunities,
        selected,
        v32_label,
    )
    v32_discrimination["method"] = "EEMD_CWT_V32"
    discrimination = pd.concat(
        [discrimination, pd.DataFrame([v32_discrimination])], ignore_index=True
    )
    discrimination.to_csv(
        output / "discrimination_calibration_new_outer.csv", index=False
    )

    manifest_rows = []
    for index, curve in enumerate(validation):
        manifest_rows.append(
            {
                "array_file": "validation_score_arrays.npz",
                "array_index": index,
                "curve_id": curve.curve_id,
                "pair_id": curve.pair_id,
                "kind": curve.kind,
                "archetype": curve.archetype,
                "GT": curve.gt,
                "truth_T1_observable": curve.truth_t1,
                "truth_T2_observable": curve.truth_t2,
                "common_start": curve.start,
            }
        )
    pd.DataFrame(manifest_rows).to_csv(output / "score_array_manifest.csv", index=False)

    provenance = {
        "analysis": "V32 new-seed independent outer validation",
        "core_algorithm_modified": False,
        "v31_role": "read-only strict-prefix Morlet-w6 score cache",
        "selected_candidate": selected,
        "selected_definition": definitions[selected],
        "common_alarm_policy": policy.name,
        "threshold_source": "original development split only",
        "outer_used_for_selection_or_thresholding": False,
        "outer_seed": 2026073003,
        "outer_curves": len(validation),
        "outer_outbreak": sum(curve.kind == "outbreak" for curve in validation),
        "outer_no_outbreak": sum(curve.kind == "no_outbreak" for curve in validation),
        "source_hashes": {
            "outer_metadata": ub.sha256(
                outer_source / "validation_curve_metadata.csv"
            ),
            "outer_arrays": ub.sha256(
                outer_source / "validation_curve_arrays.npz"
            ),
            "outer_v31_cache": ub.sha256(
                outer_cache / "validation_v31_full_scores.csv"
            ),
            "v32_lock": ub.sha256(v32_lock_root / "v32_development_lock.json"),
        },
        "claim_boundary": (
            "V32 is an additive operational extension and does not replace the "
            "V31.1 phase segmentation or its primary validation."
        ),
    }
    (output / "protocol_and_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)
    print(discrimination.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
