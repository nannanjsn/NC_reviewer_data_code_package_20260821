"""Development-only optimization of a concordance-aware V32.1 score.

This is an additive operational layer.  It does not alter V31.1, its phase
landmarks, or any existing validation.  Candidate selection uses only the
original development split and the common one-day review policy.  The already
unblinded V32 outer split is deliberately not loaded.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

import optimize_v32_operational_extension as v32
import unified_operational_benchmark as ub


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "母小波确认" / "outputs_v2" / "20260716_103352_formal_v2"
BENCHMARK = HERE / "outputs_unified_operational_benchmark_v31_full"
OUTPUT = HERE / "outputs_v32_1_concordance_extension"
POLICY = next(item for item in v32.POLICIES if item.name == "review_1d")


def make_concordance_candidates(
    curves: Sequence[ub.Curve],
    base_scores: Mapping[str, Mapping[str, np.ndarray]],
    base_opportunities: Mapping[str, Mapping[str, np.ndarray]],
    scalers: Mapping[str, Tuple[float, float]],
) -> Tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, object]],
]:
    scores: Dict[str, Dict[str, np.ndarray]] = {}
    opportunities: Dict[str, Dict[str, np.ndarray]] = {}
    definitions: Dict[str, Dict[str, object]] = {
        "pure_rt": {
            "family": "online_Rt_reference",
            "formula": "standardized online P(Rt>1)",
        },
        "linear_rt_w75": {
            "family": "V32_linear_reference",
            "formula": "0.25 structural mean-of-two z + 0.75 Rt z",
        },
    }
    for bonus in (0.15, 0.25, 0.40, 0.60, 0.80):
        definitions[f"rt_positive_bonus_b{int(100*bonus):02d}"] = {
            "family": "V32.1_positive_structural_support",
            "formula": f"Rt z + {bonus:.2f} * max(structural mean-of-two z, 0)",
            "bonus_weight": bonus,
        }
        definitions[f"rt_softplus_bonus_b{int(100*bonus):02d}"] = {
            "family": "V32.1_soft_structural_support",
            "formula": f"Rt z + {bonus:.2f} * softplus(structural mean-of-two z)",
            "bonus_weight": bonus,
        }
    for penalty in (0.15, 0.25, 0.40, 0.60):
        definitions[f"rt_concordance_p{int(100*penalty):02d}"] = {
            "family": "V32.1_bidirectional_concordance",
            "formula": (
                f"Rt z + {penalty:.2f} * clipped structural mean-of-two z"
            ),
            "concordance_weight": penalty,
        }
    for mix in (0.25, 0.40, 0.60):
        definitions[f"soft_and_m{int(100*mix):02d}"] = {
            "family": "V32.1_soft_AND",
            "formula": (
                f"{1-mix:.2f} * min(Rt z, structural z) "
                f"+ {mix:.2f} * mean(Rt z, structural z)"
            ),
            "mean_weight": mix,
        }

    s_centre, s_scale = scalers["EEMD_CWT_first"]
    rt_centre, rt_scale = scalers["Rt_P_gt_1"]
    for curve in curves:
        curve_id = curve.curve_id
        first = base_scores[curve_id]["EEMD_CWT_first"]
        first_opp = base_opportunities[curve_id]["EEMD_CWT_first"]
        structure, structure_opp = v32.scheduled_aggregate(first, first_opp, "mean2")
        structure_z = v32.standardized(structure, s_centre, s_scale)
        rt = base_scores[curve_id]["Rt_P_gt_1"]
        rt_opp = base_opportunities[curve_id]["Rt_P_gt_1"]
        rt_z = v32.standardized(rt, rt_centre, rt_scale)
        common = structure_opp & rt_opp & np.isfinite(structure_z) & np.isfinite(rt_z)
        scores[curve_id] = {}
        opportunities[curve_id] = {}

        def add(name: str, values: np.ndarray) -> None:
            out = np.full(len(values), np.nan)
            out[common] = values[common]
            scores[curve_id][name] = out
            opportunities[curve_id][name] = common.copy()

        add("pure_rt", rt_z)
        add("linear_rt_w75", 0.25 * structure_z + 0.75 * rt_z)
        positive = np.maximum(structure_z, 0.0)
        softplus = np.logaddexp(0.0, np.clip(structure_z, -20.0, 20.0))
        clipped = np.clip(structure_z, -2.0, 3.0)
        minimum = np.minimum(rt_z, structure_z)
        mean = 0.5 * (rt_z + structure_z)
        for bonus in (0.15, 0.25, 0.40, 0.60, 0.80):
            add(f"rt_positive_bonus_b{int(100*bonus):02d}", rt_z + bonus * positive)
            add(f"rt_softplus_bonus_b{int(100*bonus):02d}", rt_z + bonus * softplus)
        for penalty in (0.15, 0.25, 0.40, 0.60):
            add(f"rt_concordance_p{int(100*penalty):02d}", rt_z + penalty * clipped)
        for mix in (0.25, 0.40, 0.60):
            add(f"soft_and_m{int(100*mix):02d}", (1.0 - mix) * minimum + mix * mean)
    return scores, opportunities, definitions


def development_cv(
    curves: Sequence[ub.Curve],
    scores: Mapping[str, Mapping[str, np.ndarray]],
    opportunities: Mapping[str, Mapping[str, np.ndarray]],
    definitions: Mapping[str, Mapping[str, object]],
    n_folds: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fold_map = v32.paired_folds(curves, n_folds)
    detail_rows = []
    for candidate in definitions:
        pr_auc = v32.classification_pr_auc(curves, scores, opportunities, candidate)
        for fold in range(n_folds):
            training = [curve for curve in curves if fold_map[curve.pair_id] != fold]
            held_out = [curve for curve in curves if fold_map[curve.pair_id] == fold]
            for regime, criterion, target in v32.REGIMES:
                threshold, training_burden = v32.calibrate_one(
                    training,
                    scores,
                    opportunities,
                    candidate,
                    POLICY,
                    criterion,
                    target,
                )
                metrics = v32.evaluate(
                    held_out,
                    scores,
                    opportunities,
                    candidate,
                    threshold,
                    POLICY,
                )
                detail_rows.append(
                    {
                        "candidate": candidate,
                        "family": definitions[candidate]["family"],
                        "fold": fold,
                        "regime": regime,
                        "criterion": criterion,
                        "target": target,
                        "threshold": threshold,
                        "overall_development_PR_AUC": pr_auc,
                        "training_constraint": training_burden[criterion],
                        **metrics,
                    }
                )
    detail = pd.DataFrame(detail_rows)
    rows = []
    for (candidate, family, regime), group in detail.groupby(
        ["candidate", "family", "regime"], sort=False
    ):
        rows.append(
            {
                "candidate": candidate,
                "family": family,
                "policy": POLICY.name,
                "regime": regime,
                "overall_development_PR_AUC": float(
                    group.overall_development_PR_AUC.iloc[0]
                ),
                "mean_fold_detection_before_T2": float(
                    group.detection_before_T2.mean()
                ),
                "minimum_fold_detection_before_T2": float(
                    group.detection_before_T2.min()
                ),
                "mean_fold_median_lead_days": float(
                    group.median_lead_to_T2_days.mean()
                ),
                "mean_fold_null_alert_days_per_100d": float(
                    group.alert_days_per_100d.mean()
                ),
                "mean_fold_null_curve_fpr": float(group.null_curve_fpr.mean()),
            }
        )
    return detail, pd.DataFrame(rows)


def select_candidate(summary: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    wide = summary.pivot_table(
        index=["candidate", "family", "overall_development_PR_AUC"],
        columns="regime",
        values="mean_fold_detection_before_T2",
    ).reset_index()
    regimes = [item[0] for item in v32.REGIMES]
    wide["mean_detection"] = wide[regimes].mean(axis=1)
    wide["minimum_regime_detection"] = wide[regimes].min(axis=1)
    rt_pr = float(wide.loc[wide.candidate == "pure_rt", "overall_development_PR_AUC"].iloc[0])
    wide["PR_AUC_guardrail"] = wide.overall_development_PR_AUC >= 0.95 * rt_pr
    eligible = wide[wide.PR_AUC_guardrail].sort_values(
        [
            "minimum_regime_detection",
            "mean_detection",
            "specificity_90",
            "burden_low",
            "overall_development_PR_AUC",
        ],
        ascending=False,
    )
    return str(eligible.iloc[0].candidate), wide


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    development, base_scores, base_opportunities = v32.load_saved_split(
        BENCHMARK, SOURCE, "development"
    )
    scalers = v32.development_null_scalers(
        development, base_scores, base_opportunities
    )
    scores, opportunities, definitions = make_concordance_candidates(
        development, base_scores, base_opportunities, scalers
    )
    detail, summary = development_cv(
        development, scores, opportunities, definitions, n_folds=5
    )
    selected, ranking = select_candidate(summary)
    locks = v32.full_development_locks(
        development, scores, opportunities, selected, POLICY
    )
    apparent = v32.apply_locks(
        development, scores, opportunities, locks, selected, POLICY
    )
    detail.to_csv(OUTPUT / "development_cv_fold_detail.csv", index=False)
    summary.to_csv(OUTPUT / "development_cv_regime_summary.csv", index=False)
    ranking.sort_values(
        ["minimum_regime_detection", "mean_detection"], ascending=False
    ).to_csv(OUTPUT / "development_candidate_ranking.csv", index=False)
    locks.to_csv(OUTPUT / "development_threshold_locks.csv", index=False)
    apparent.to_csv(OUTPUT / "development_locked_apparent_performance.csv", index=False)
    lock = {
        "status": "development_locked_requires_new_independent_validation",
        "version": "V32.1-concordance",
        "core_algorithm_modified": False,
        "selected_candidate": selected,
        "selected_policy": POLICY.name,
        "selected_definition": definitions[selected],
        "selection_endpoint": (
            "maximize the minimum paired-fold T2-pre detection across 1 and 3 "
            "alert-days/100d and 90% null-curve specificity, then mean detection"
        ),
        "PR_AUC_guardrail": "at least 95% of pure online Rt development PR-AUC",
        "development_only_for_selection": True,
        "previous_outer_split_loaded": False,
        "requires_new_seed_outer_validation": True,
        "development_null_scalers": {
            key: {"centre": value[0], "scale": value[1]}
            for key, value in scalers.items()
        },
        "source_hashes": {
            "development_score_arrays.npz": ub.sha256(
                BENCHMARK / "development_score_arrays.npz"
            ),
            "development_curve_arrays.npz": ub.sha256(
                SOURCE / "development_curve_arrays.npz"
            ),
        },
    }
    (OUTPUT / "v32_1_development_lock.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(ranking.sort_values(
        ["minimum_regime_detection", "mean_detection"], ascending=False
    ).head(20).to_string(index=False), flush=True)
    print(locks.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
