#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate the pre-locked practical low-burden point on a fresh seed.

This script intentionally contains no threshold search.  It reads the lock
written before generation of the 2026081015 validation set, applies the same
one-day review/seven-day cooldown policy to every method, and reports strict
pre-T2 event sensitivity and no-transition alert burden.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import evaluate_v32_independent_outer as ev
import optimize_v32_1_concordance_extension as opt
import optimize_v32_operational_extension as v32
import unified_operational_benchmark as ub


HERE = Path(__file__).resolve().parent
LOCK_PATH = HERE / "confirmation_lock_practical_low_burden_20260810.json"
DEVELOPMENT_SOURCE = ev.DEFAULT_DEVELOPMENT_SOURCE
BENCHMARK = HERE / "outputs_unified_operational_benchmark_v31_full"
OUTER_SOURCE = HERE / "practical_low_burden_confirmation_seed_2026081015"
OUTER_CACHE = HERE / "cache_practical_low_burden_confirmation_seed_2026081015"
OUTPUT = HERE / "outputs_practical_low_burden_confirmation_seed_2026081015"
REGIME = "practical_low_burden_1p5"
PRIMARY_METHOD = "EEMD_CWT_Rt_concordance"
PRIMARY_LABEL = "EEMD-CWT/Rt concordance"


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def paired_comparisons(results: pd.DataFrame, comparators: list[str], n_boot: int = 5000) -> pd.DataFrame:
    outbreak = results[results.kind == "outbreak"].copy()
    primary = outbreak[outbreak.method == PRIMARY_METHOD][
        ["curve_id", "pair_id", "archetype", "detected_in_window"]
    ].rename(columns={"detected_in_window": "primary_detected"})
    rng = np.random.default_rng(2026081015)
    rows: list[dict[str, object]] = []
    for method in comparators:
        other = outbreak[outbreak.method == method][
            ["curve_id", "detected_in_window"]
        ].rename(columns={"detected_in_window": "comparator_detected"})
        paired = primary.merge(other, on="curve_id", how="inner")
        paired["difference"] = (
            paired.primary_detected.astype(float) - paired.comparator_detected.astype(float)
        )
        draws: list[float] = []
        grouped = [group.reset_index(drop=True) for _, group in paired.groupby("archetype", sort=True)]
        for _ in range(n_boot):
            sampled = []
            for group in grouped:
                sampled.append(group.iloc[rng.integers(0, len(group), len(group))])
            draws.append(float(pd.concat(sampled, ignore_index=True).difference.mean()))
        discordant_primary = int((paired.primary_detected & ~paired.comparator_detected).sum())
        discordant_comparator = int((~paired.primary_detected & paired.comparator_detected).sum())
        discordant_total = discordant_primary + discordant_comparator
        p_value = (
            float(binomtest(discordant_primary, discordant_total, 0.5, alternative="two-sided").pvalue)
            if discordant_total else 1.0
        )
        rows.append(
            {
                "primary_method": PRIMARY_METHOD,
                "comparator_method": method,
                "n_pairs": len(paired),
                "primary_sensitivity": float(paired.primary_detected.mean()),
                "comparator_sensitivity": float(paired.comparator_detected.mean()),
                "paired_difference": float(paired.difference.mean()),
                "paired_difference_ci_low": float(np.quantile(draws, 0.025)),
                "paired_difference_ci_high": float(np.quantile(draws, 0.975)),
                "primary_only_detected": discordant_primary,
                "comparator_only_detected": discordant_comparator,
                "mcnemar_exact_p": p_value,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("status") != "locked_before_new_seed_generation":
        raise ValueError("The analysis lock does not document pre-generation locking")

    development, dev_base, dev_opportunities = v32.load_saved_split(
        BENCHMARK, DEVELOPMENT_SOURCE, "development"
    )
    validation = ub.load_curves(OUTER_SOURCE, "validation")
    cache = pd.read_csv(OUTER_CACHE / "validation_v31_full_scores.csv")
    val_base, val_opportunities = ub.compute_split_scores(
        validation, cache, step=3, workers=4
    )
    scalers = v32.development_null_scalers(development, dev_base, dev_opportunities)
    val_candidates, val_candidate_opportunities, definitions = opt.make_concordance_candidates(
        validation, val_base, val_opportunities, scalers
    )

    candidate = str(lock["primary_method"]["candidate"])
    primary_threshold = float(lock["primary_method"]["development_locked_threshold"])
    policy = opt.POLICY
    comparator_thresholds = {
        str(method): float(threshold)
        for method, threshold in lock["locked_comparator_thresholds"].items()
    }
    rows: list[dict[str, object]] = []
    for curve in validation:
        rows.append(
            ev.curve_result_policy(
                curve,
                PRIMARY_METHOD,
                PRIMARY_LABEL,
                REGIME,
                primary_threshold,
                val_candidates[curve.curve_id][candidate],
                val_candidate_opportunities[curve.curve_id][candidate],
                policy,
            )
        )
        for method, threshold in comparator_thresholds.items():
            rows.append(
                ev.curve_result_policy(
                    curve,
                    method,
                    ub.METHOD_LABELS[method],
                    REGIME,
                    threshold,
                    val_base[curve.curve_id][method],
                    val_opportunities[curve.curve_id][method],
                    policy,
                )
            )
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT / "locked_curve_level_results.csv", index=False, encoding="utf-8-sig")
    summary = ev.summarize_results(results)
    ci_rows = []
    for row in summary.itertuples(index=False):
        method_rows = results[(results.method == row.method) & (results.kind == "outbreak")]
        successes = int(method_rows.detected_in_window.sum())
        low, high = wilson(successes, len(method_rows))
        ci_rows.append(
            {
                "method": row.method,
                "detected_n": successes,
                "sensitivity_ci_low": low,
                "sensitivity_ci_high": high,
            }
        )
    summary = summary.merge(pd.DataFrame(ci_rows), on="method", how="left")
    summary.to_csv(OUTPUT / "locked_summary.csv", index=False, encoding="utf-8-sig")

    archetype = (
        results[results.kind == "outbreak"]
        .groupby(["method", "method_label", "archetype"], as_index=False)
        .agg(n=("curve_id", "size"), detected_n=("detected_in_window", "sum"), sensitivity=("detected_in_window", "mean"))
    )
    archetype.to_csv(OUTPUT / "sensitivity_by_archetype.csv", index=False, encoding="utf-8-sig")
    paired = paired_comparisons(results, list(comparator_thresholds))
    paired.to_csv(OUTPUT / "paired_sensitivity_comparisons.csv", index=False, encoding="utf-8-sig")

    primary = summary[summary.method == PRIMARY_METHOD].iloc[0]
    versus_rt = paired[paired.comparator_method == "Rt_P_gt_1"].iloc[0]
    criteria = lock["success_criteria"]
    decisions = {
        "sensitivity_at_least_0p60": bool(primary.detection_before_T2 >= float(criteria["primary_sensitivity_at_least"])),
        "paired_difference_versus_Rt_positive": bool(versus_rt.paired_difference > float(criteria["paired_difference_versus_Rt_greater_than"])),
        "actual_burden_at_most_1p80": bool(primary.null_alert_days_per_100d <= float(criteria["actual_alert_days_per_100_upper_tolerance"])),
    }
    decision = {
        "all_prelocked_success_criteria_met": bool(all(decisions.values())),
        "criteria": decisions,
        "primary_sensitivity": float(primary.detection_before_T2),
        "primary_detected_n": int(primary.detected_n),
        "primary_sensitivity_95ci": [float(primary.sensitivity_ci_low), float(primary.sensitivity_ci_high)],
        "actual_alert_days_per_100_null_days": float(primary.null_alert_days_per_100d),
        "null_curve_false_positive_rate": float(primary.null_curve_fpr),
        "median_lead_to_T2_days": float(primary.median_lead_to_T2_days),
        "paired_difference_versus_Rt": float(versus_rt.paired_difference),
        "paired_difference_versus_Rt_95ci": [float(versus_rt.paired_difference_ci_low), float(versus_rt.paired_difference_ci_high)],
        "mcnemar_exact_p_versus_Rt": float(versus_rt.mcnemar_exact_p),
        "interpretation_boundary": "Fresh-seed confirmation under the same simulation generator; not external clinical validation and not a replacement for the original prespecified 1.0-burden result.",
    }
    (OUTPUT / "prelocked_success_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    provenance = {
        "analysis": "Pre-locked practical low-burden fresh-seed confirmation",
        "lock_file": str(LOCK_PATH),
        "threshold_search_on_new_seed": False,
        "primary_candidate": candidate,
        "primary_definition": definitions[candidate],
        "common_alarm_policy": policy.name,
        "validation_seed": int(lock["new_seed"]),
        "validation_curves": len(validation),
        "source_hashes": {
            "lock": ub.sha256(LOCK_PATH),
            "metadata": ub.sha256(OUTER_SOURCE / "validation_curve_metadata.csv"),
            "arrays": ub.sha256(OUTER_SOURCE / "validation_curve_arrays.npz"),
            "eemd_cwt_cache": ub.sha256(OUTER_CACHE / "validation_v31_full_scores.csv"),
        },
    }
    (OUTPUT / "protocol_and_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(paired.to_string(index=False), flush=True)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
