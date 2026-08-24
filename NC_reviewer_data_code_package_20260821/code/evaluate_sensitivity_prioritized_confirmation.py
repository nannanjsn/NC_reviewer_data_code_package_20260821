#!/usr/bin/env python
"""One-pass evaluation of the prelocked sensitivity-prioritized tier."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd

import evaluate_locked_practical_low_burden_confirmation as statistics
import evaluate_v32_independent_outer as ev
import optimize_v32_1_concordance_extension as opt
import optimize_v32_operational_extension as v32
import unified_operational_benchmark as ub


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
LOCK_PATH = PACKAGE_ROOT / "config" / "confirmation_lock_sensitivity_prioritized_tier_20260811.json"
SOURCE = PACKAGE_ROOT / "data" / "confirmatory_inputs"
CACHE = PACKAGE_ROOT / "data" / "confirmatory_cache"
DEVELOPMENT_SOURCE = PACKAGE_ROOT / "data" / "development_wavelet"
DEVELOPMENT_BENCHMARK = PACKAGE_ROOT / "data" / "development_benchmark"
OUTPUT = PACKAGE_ROOT / "reproduced_outputs" / "confirmatory"
PRIMARY_METHOD = "EEMD_CWT_Rt_concordance"
PRIMARY_LABEL = "EEMD-CWT/Rt concordance"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {OUTPUT}")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("status") != "locked_before_fresh_seed_generation":
        raise ValueError("The operating tier was not locked before fresh-seed generation")
    generation = json.loads((SOURCE / "generation_lock.json").read_text(encoding="utf-8"))
    if int(generation["seed"]) != int(lock["new_seed"]):
        raise ValueError("Generated seed does not match prelock")

    development, dev_base, dev_opportunities = v32.load_saved_split(
        DEVELOPMENT_BENCHMARK, DEVELOPMENT_SOURCE, "development"
    )
    validation = ub.load_curves(SOURCE, "validation")
    cache = pd.read_csv(CACHE / "validation_v31_full_scores.csv")
    val_base, val_opportunities = ub.compute_split_scores(
        validation, cache, step=3, workers=4
    )
    scalers = v32.development_null_scalers(development, dev_base, dev_opportunities)
    candidates, candidate_opportunities, definitions = opt.make_concordance_candidates(
        validation, val_base, val_opportunities, scalers
    )

    candidate = str(lock["primary_method"]["candidate"])
    primary_threshold = float(lock["primary_method"]["development_locked_threshold"])
    comparator_thresholds = {
        str(key): float(value) for key, value in lock["locked_comparator_thresholds"].items()
    }
    endpoints = {
        "observable_expansion_primary": 1.0,
        "original_strict_T2_stress_test": 0.0,
    }
    rows = []
    for endpoint, shift_gt in endpoints.items():
        for curve in validation:
            evaluated = (
                replace(curve, truth_t2=float(curve.truth_t2 + shift_gt * curve.gt))
                if curve.kind == "outbreak" else curve
            )
            primary_row = ev.curve_result_policy(
                evaluated, PRIMARY_METHOD, PRIMARY_LABEL,
                "sensitivity_prioritized_target_2p25", primary_threshold,
                candidates[curve.curve_id][candidate],
                candidate_opportunities[curve.curve_id][candidate], opt.POLICY
            )
            primary_row.update(endpoint=endpoint, T2_shift_GT=shift_gt)
            rows.append(primary_row)
            for method, threshold in comparator_thresholds.items():
                row = ev.curve_result_policy(
                    evaluated, method, ub.METHOD_LABELS[method],
                    "sensitivity_prioritized_target_2p25", threshold,
                    val_base[curve.curve_id][method],
                    val_opportunities[curve.curve_id][method], opt.POLICY
                )
                row.update(endpoint=endpoint, T2_shift_GT=shift_gt)
                rows.append(row)

    results = pd.DataFrame(rows)
    summaries = []
    for endpoint, frame in results.groupby("endpoint", sort=False):
        summary = ev.summarize_results(frame)
        summary["endpoint"] = endpoint
        ci_rows = []
        for method, group in frame[frame.kind == "outbreak"].groupby("method"):
            hits = int(group.detected_in_window.sum())
            low, high = statistics.wilson(hits, len(group))
            ci_rows.append({
                "method": method, "detected_n": hits,
                "sensitivity_ci_low": low, "sensitivity_ci_high": high,
            })
        summaries.append(summary.merge(pd.DataFrame(ci_rows), on="method", how="left"))
    summary = pd.concat(summaries, ignore_index=True)
    primary_frame = results[results.endpoint == "observable_expansion_primary"].copy()
    paired = statistics.paired_comparisons(primary_frame, list(comparator_thresholds))
    archetype = (
        primary_frame[primary_frame.kind == "outbreak"]
        .groupby(["method", "method_label", "archetype"], as_index=False)
        .agg(n=("curve_id", "size"), detected_n=("detected_in_window", "sum"),
             sensitivity=("detected_in_window", "mean"))
    )

    primary = summary[
        (summary.endpoint == "observable_expansion_primary") &
        (summary.method == PRIMARY_METHOD)
    ].iloc[0]
    versus_rt = paired[paired.comparator_method == "Rt_P_gt_1"].iloc[0]
    criteria = lock["success_criteria"]
    checks = {
        "point_sensitivity_at_least_0p80": bool(
            primary.detection_before_T2 >= criteria["primary_point_sensitivity_at_least"]
        ),
        "paired_difference_versus_Rt_positive": bool(versus_rt.paired_difference > 0.0),
        "null_burden_within_prelocked_maximum": bool(
            primary.null_alert_days_per_100d <= criteria["actual_alert_days_per_100d_at_most"]
        ),
    }
    decision = {
        "all_prelocked_success_criteria_met": bool(all(checks.values())),
        "checks": checks,
        "primary_detected_n": int(primary.detected_n),
        "primary_n_outbreak": int(primary.n_outbreak),
        "primary_sensitivity": float(primary.detection_before_T2),
        "primary_sensitivity_95ci": [
            float(primary.sensitivity_ci_low), float(primary.sensitivity_ci_high)
        ],
        "primary_median_lead_days": float(primary.median_lead_to_T2_days),
        "full_null_alert_days_per_100d": float(primary.null_alert_days_per_100d),
        "day_level_specificity": float(1.0 - primary.null_alert_days_per_100d / 100.0),
        "paired_difference_versus_Rt": float(versus_rt.paired_difference),
        "paired_difference_versus_Rt_95ci": [
            float(versus_rt.paired_difference_ci_low),
            float(versus_rt.paired_difference_ci_high),
        ],
        "mcnemar_exact_p_versus_Rt": float(versus_rt.mcnemar_exact_p),
        "interpretation_boundary": lock["interpretation_boundary"],
    }

    OUTPUT.mkdir(parents=True, exist_ok=False)
    results.to_csv(OUTPUT / "locked_curve_level_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT / "locked_endpoint_summary.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(OUTPUT / "paired_primary_vs_comparators.csv", index=False, encoding="utf-8-sig")
    archetype.to_csv(OUTPUT / "primary_sensitivity_by_archetype.csv", index=False, encoding="utf-8-sig")
    (OUTPUT / "prelocked_success_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    provenance = {
        "threshold_search_on_fresh_seed": False,
        "endpoint_search_on_fresh_seed": False,
        "operating_target_search_on_fresh_seed": False,
        "lock_hash": ub.sha256(LOCK_PATH),
        "metadata_hash": ub.sha256(SOURCE / "validation_curve_metadata.csv"),
        "arrays_hash": ub.sha256(SOURCE / "validation_curve_arrays.npz"),
        "cache_hash": ub.sha256(CACHE / "validation_v31_full_scores.csv"),
        "primary_candidate_definition": definitions[candidate],
        "common_alarm_policy": opt.POLICY.name,
    }
    (OUTPUT / "protocol_and_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
