from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASE_MODULE_NAME = "epidemic_phase_segmentation_v31_1_archetype_adaptive"
REQUIRED_RUNTIME_PACKAGES = ("numpy", "pandas", "matplotlib", "scipy")
INFLUENZA_TARGET_WINDOW_START = "2025-09-01"
INFLUENZA_TARGET_WINDOW_END = "2026-03-31"
CROSS_PATHOGEN_DATA_DIR = THIS_DIR.parent

base = None
np = None
pd = None


def bootstrap_site_packages() -> List[str]:
    added = []
    exe_dir = Path(sys.executable).resolve().parent
    candidates = [
        exe_dir / "Lib" / "site-packages",
        exe_dir.parent / "Lib" / "site-packages",
        Path(sys.prefix) / "Lib" / "site-packages",
        Path(sys.base_prefix) / "Lib" / "site-packages",
    ]
    seen = set()
    for path in candidates:
        norm = str(path)
        if norm in seen:
            continue
        seen.add(norm)
        if path.exists() and path.is_dir() and norm not in sys.path:
            sys.path.append(norm)
            added.append(norm)
    return added


def missing_runtime_packages() -> List[str]:
    missing = []
    for package_name in REQUIRED_RUNTIME_PACKAGES:
        if importlib.util.find_spec(package_name) is None:
            missing.append(package_name)
    return missing


def print_env_check() -> None:
    added_paths = bootstrap_site_packages()
    print(f"[env] Python executable: {sys.executable}")
    if added_paths:
        print(f"[env] Added site-packages: {', '.join(added_paths)}")
    missing = missing_runtime_packages()
    if missing:
        print(f"[env] Missing packages: {', '.join(missing)}")
        print("[env] Suggested fix:")
        print(f"       {sys.executable} -m pip install {' '.join(missing)}")
    else:
        print("[env] Core packages detected: numpy, pandas, matplotlib, scipy")


def ensure_runtime_ready() -> None:
    global base, np, pd
    if base is not None and np is not None and pd is not None:
        return

    bootstrap_site_packages()
    missing = missing_runtime_packages()
    if missing:
        raise ModuleNotFoundError(
            "Current Python environment is missing required packages: "
            f"{', '.join(missing)}. "
            f"Use this interpreter to install them: {sys.executable} -m pip install {' '.join(missing)}"
        )

    import numpy as _np
    import pandas as _pd

    try:
        _base = importlib.import_module(BASE_MODULE_NAME)
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", "") or str(exc)
        raise ModuleNotFoundError(
            "Failed to import the original V31.1 core module because a runtime dependency "
            f"is missing: {missing_name}. Please run this wrapper with the same Python environment "
            "used for the original epidemic-phase scripts."
        ) from exc

    np = _np
    pd = _pd
    base = _base


FINAL_RUN_TAG = "final_20260624"
VERSION_TAG = f"v31_1_cross_pathogen_extension_v1_{FINAL_RUN_TAG}"
OUTPUT_ROOT = THIS_DIR / f"outputs_{FINAL_RUN_TAG}"
OUTPUT_BENCH = OUTPUT_ROOT / "stage0_cross_pathogen_benchmark"
OUTPUT_STAGE1 = OUTPUT_ROOT / "stage1_cross_pathogen_development"
OUTPUT_STAGE2 = OUTPUT_ROOT / "stage2_cross_pathogen_validation"
OUTPUT_STAGE3 = OUTPUT_ROOT / "stage3_cross_pathogen_application"

PRIMARY_CORE_ARCHETYPE_NAMES: Tuple[str, ...] = (
    "influenza_like",
    "influenza_seasonal_high_baseline",
    "moderate_coronavirus",
    "chikungunya_vector_borne",
    "mpox_specific",
)
SENSITIVITY_ARCHETYPE_NAMES: Tuple[str, ...] = (
    "high_transmissibility_coronavirus",
    "mpv_like_contact_network",
    "mpv_like_refined_contact_network",
)
ALL_CROSS_PATHOGEN_ARCHETYPE_NAMES: Tuple[str, ...] = PRIMARY_CORE_ARCHETYPE_NAMES + SENSITIVITY_ARCHETYPE_NAMES
MPV_TUNED_VARIANT = "mpv_tuned"


def configure_output_dirs() -> None:
    for path in [OUTPUT_ROOT, OUTPUT_BENCH, OUTPUT_STAGE1, OUTPUT_STAGE2, OUTPUT_STAGE3]:
        path.mkdir(parents=True, exist_ok=True)

    base.VERSION_TAG = VERSION_TAG
    base.OUTPUT_BENCH = str(OUTPUT_BENCH)
    base.OUTPUT_STAGE1 = str(OUTPUT_STAGE1)
    base.OUTPUT_STAGE2 = str(OUTPUT_STAGE2)
    base.OUTPUT_STAGE3 = str(OUTPUT_STAGE3)
    base.CACHE_PATH = os.path.join(base.OUTPUT_STAGE1, f"eemd_cwt_structural_cache_{VERSION_TAG}.npz")
    base.BEST_PARAMS_PATH = os.path.join(base.OUTPUT_STAGE1, f"best_params_{VERSION_TAG}.json")
    base.T1_FOCUSED_PARAMS_PATH = os.path.join(base.OUTPUT_STAGE2, f"t1_focused_best_params_{VERSION_TAG}.json")
    base.PAPER_TABLES_PATH = os.path.join(base.OUTPUT_STAGE3, f"paper_ready_tables_{VERSION_TAG}.xlsx")


def install_safe_batch_generator() -> None:
    if getattr(base.SIRSimulator.batch_generate_archetype, "_cross_pathogen_safe_patch", False):
        return

    def _safe_batch_generate_archetype(
        self,
        n_curves=180,
        days=300,
        add_structured_noise=True,
        N=100000,
        I0=5,
        min_Tp: Optional[int] = None,
        scenario_set: Optional[List[Dict]] = None,
    ):
        scenarios = scenario_set or self.ARCHETYPE_SCENARIOS
        counts = self._allocate_counts_from_weights(scenarios, n_curves)
        curves = []
        curve_id = 0
        generation_stats: List[Dict[str, object]] = []

        def effective_min_tp(sc: Dict) -> Optional[int]:
            if min_Tp is None:
                return None
            arch = str(sc.get("archetype", sc.get("label", ""))).lower()
            floor = int(min_Tp)
            if "high_transmissibility" in arch or "fast_respiratory" in arch:
                return min(floor, 28)
            if "influenza" in arch:
                return min(floor, 30)
            if "moderate_coronavirus" in arch:
                return min(floor, 35)
            return floor

        def make_curve(sc: Dict, suffix: str = ""):
            R0 = self._sample_range(sc["R0_range"])
            GT = self._sample_range(sc["GT_range"])
            SI = self._sample_range(sc["SI_range"])
            nst = self._sample_range(sc["noise_std_range"])
            rd = self._sample_int_range(sc["report_delay_range"])
            c = self.generate_sir_curve(
                days=days,
                R0=R0,
                GT=GT,
                N=N,
                I0=I0,
                noise_std=nst,
                report_delay=rd,
                add_structured_noise=add_structured_noise,
                structural_noise_profile=sc["structural_noise_profile"],
                SI=SI,
                archetype=sc["archetype"],
            )
            c["scenario"] = sc["label"] + suffix
            c["archetype"] = sc["archetype"]
            c["SI"] = SI
            c["scenario_weight"] = float(sc["weight"])
            c["structural_noise_profile"] = sc["structural_noise_profile"]
            c["transmission_route"] = sc.get("route", "")
            c["source_framework"] = "pathogen_archetype_design"
            return c

        for sc, n_sc in zip(scenarios, counts):
            sc_count = 0
            attempts = 0
            rejected_min_tp = 0
            max_attempts = max(40, n_sc * 40)
            min_tp_eff = effective_min_tp(sc)
            while sc_count < n_sc and attempts < max_attempts:
                attempts += 1
                c = make_curve(sc)
                if min_tp_eff is not None and c["truth_Tp"] < min_tp_eff:
                    rejected_min_tp += 1
                    continue
                c["curve_id"] = curve_id
                curves.append(c)
                curve_id += 1
                sc_count += 1
            generation_stats.append(
                dict(
                    archetype=sc["archetype"],
                    target=int(n_sc),
                    accepted=int(sc_count),
                    attempts=int(attempts),
                    rejected_min_tp=int(rejected_min_tp),
                    min_tp_eff=(None if min_tp_eff is None else int(min_tp_eff)),
                    phase="allocated",
                )
            )

        weights = np.array([float(s.get("weight", 1.0)) for s in scenarios], dtype=float)
        weights = weights / max(float(weights.sum()), 1e-10)
        supplemental_attempts = 0
        supplemental_accepts = 0
        supplemental_rejected = 0
        max_supplemental_attempts = max(200, n_curves * max(60, 15 * len(scenarios)))
        while len(curves) < n_curves and supplemental_attempts < max_supplemental_attempts:
            supplemental_attempts += 1
            sc = scenarios[int(self.rng.choice(np.arange(len(scenarios)), p=weights))]
            c = make_curve(sc, suffix="_supplemental")
            min_tp_eff = effective_min_tp(sc)
            if min_tp_eff is not None and c["truth_Tp"] < min_tp_eff:
                supplemental_rejected += 1
                continue
            c["curve_id"] = curve_id
            curves.append(c)
            curve_id += 1
            supplemental_accepts += 1

        generation_stats.append(
            dict(
                archetype="__supplemental__",
                target=max(0, int(n_curves) - sum(int(row["accepted"]) for row in generation_stats)),
                accepted=int(supplemental_accepts),
                attempts=int(supplemental_attempts),
                rejected_min_tp=int(supplemental_rejected),
                min_tp_eff=(None if min_Tp is None else int(min_Tp)),
                phase="supplemental",
            )
        )

        stats_text = " | ".join(
            (
                f"{row['archetype']}: accepted={row['accepted']}/{row['target']}, "
                f"attempts={row['attempts']}, reject_minTp={row['rejected_min_tp']}, "
                f"minTp={row['min_tp_eff']}"
            )
            for row in generation_stats
        )
        print(f"[generator] days={days}, n_target={n_curves}, stats -> {stats_text}")

        if len(curves) < n_curves:
            raise RuntimeError(
                "Cross-pathogen batch generation did not finish within the safeguarded "
                f"attempt budget. Accepted {len(curves)}/{n_curves} curves. {stats_text}"
            )
        return curves[:n_curves]

    _safe_batch_generate_archetype._cross_pathogen_safe_patch = True
    base.SIRSimulator.batch_generate_archetype = _safe_batch_generate_archetype
    print("[cross-pathogen] Installed safeguarded batch_generate_archetype patch")


def _normalized_weights(scenarios: List[Dict]) -> List[Dict]:
    total = sum(float(sc.get("weight", 0.0)) for sc in scenarios)
    if total <= 0:
        even = 1.0 / max(len(scenarios), 1)
        for sc in scenarios:
            sc["weight"] = even
        return scenarios
    for sc in scenarios:
        sc["weight"] = float(sc.get("weight", 0.0)) / total
    return scenarios


def register_cross_pathogen_archetypes() -> List[Dict]:
    scenarios = copy.deepcopy(base.SIRSimulator.ARCHETYPE_SCENARIOS)
    existing = {str(sc.get("archetype", "")).lower() for sc in scenarios}
    if "influenza_seasonal_high_baseline" not in existing:
        scenarios.append(
            dict(
                archetype="influenza_seasonal_high_baseline",
                label="archetype_influenza_seasonal_high_baseline",
                route="respiratory_seasonal",
                R0_range=(1.15, 1.68),
                GT_range=(2.2, 3.5),
                SI_range=(2.1, 3.5),
                report_delay_range=(1, 5),
                noise_std_range=(0.14, 0.28),
                structural_noise_profile="influenza_high_baseline",
                weight=0.12,
            )
        )
    if "chikungunya_vector_borne" not in existing:
        scenarios.append(
            dict(
                archetype="chikungunya_vector_borne",
                label="archetype_chikungunya_vector_borne",
                route="vector_borne",
                R0_range=(1.8, 4.2),
                GT_range=(4.5, 7.5),
                SI_range=(4.0, 7.5),
                report_delay_range=(3, 8),
                noise_std_range=(0.22, 0.45),
                structural_noise_profile="vector_borne",
                weight=0.14,
            )
        )
    if "mpv_like_refined_contact_network" not in existing:
        scenarios.append(
            dict(
                archetype="mpv_like_refined_contact_network",
                label="archetype_mpv_like_refined_contact_network",
                route="contact_network_refined",
                R0_range=(1.20, 2.35),
                GT_range=(4.8, 7.1),
                SI_range=(5.6, 8.8),
                report_delay_range=(1, 4),
                noise_std_range=(0.10, 0.24),
                structural_noise_profile="contact_network_refined",
                weight=0.12,
            )
        )
    if "mpox_specific" not in existing:
        scenarios.append(
            dict(
                archetype="mpox_specific",
                label="archetype_mpox_specific",
                route="close_contact_clustered",
                R0_range=(1.12, 1.72),
                GT_range=(5.9, 7.9),
                SI_range=(6.2, 9.0),
                report_delay_range=(2, 4),
                noise_std_range=(0.09, 0.20),
                structural_noise_profile="mpox_specific",
                weight=0.12,
            )
        )

    base.SIRSimulator.STRUCTURED_NOISE_PROFILES["vector_borne"] = dict(
        weekly_amp=(0.05, 0.16),
        shift_prob=0.55,
        shift_factor=(0.72, 1.35),
        pulse_prob=0.45,
        pulse_window_frac=0.55,
        pulse_amp=(0.015, 0.060),
        spike_lambda_scale=1.15,
        spike_amp=(1.4, 4.4),
        plateau_prob=0.45,
        plateau_scale=(1.8, 4.5),
        mult_noise=(0.05, 0.12),
        additive_scale=0.035,
    )
    base.SIRSimulator.STRUCTURED_NOISE_PROFILES["contact_network_refined"] = dict(
        weekly_amp=(0.015, 0.060),
        shift_prob=0.20,
        shift_factor=(0.90, 1.12),
        pulse_prob=0.18,
        pulse_window_frac=0.20,
        pulse_amp=(0.005, 0.020),
        spike_lambda_scale=0.55,
        spike_amp=(1.05, 1.90),
        plateau_prob=0.14,
        plateau_scale=(1.0, 1.8),
        mult_noise=(0.018, 0.060),
        additive_scale=0.012,
    )
    base.SIRSimulator.STRUCTURED_NOISE_PROFILES["mpox_specific"] = dict(
        weekly_amp=(0.015, 0.060),
        shift_prob=0.22,
        shift_factor=(0.90, 1.14),
        pulse_prob=0.18,
        pulse_window_frac=0.24,
        pulse_amp=(0.004, 0.018),
        spike_lambda_scale=0.58,
        spike_amp=(1.05, 1.85),
        plateau_prob=0.16,
        plateau_scale=(1.1, 2.0),
        mult_noise=(0.018, 0.055),
        additive_scale=0.011,
    )

    base.SIRSimulator.ARCHETYPE_SCENARIOS = _normalized_weights(scenarios)
    base.CORE_ARCHETYPE_NAMES = PRIMARY_CORE_ARCHETYPE_NAMES
    return base.SIRSimulator.ARCHETYPE_SCENARIOS


def export_extended_design_table(default_n: int = 180) -> pd.DataFrame:
    design_df = base.SIRSimulator.archetype_design_table(default_n=default_n)
    out_path = OUTPUT_STAGE1 / "cross_pathogen_archetype_design.csv"
    design_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[cross-pathogen] Archetype design exported: {out_path}")
    return design_df


def _select_extended_scenarios(selected_archetypes: Optional[List[str]] = None) -> List[Dict]:
    wanted = {name.lower() for name in PRIMARY_CORE_ARCHETYPE_NAMES}
    if selected_archetypes:
        requested = {name.lower() for name in selected_archetypes}
        allowed = {name.lower() for name in ALL_CROSS_PATHOGEN_ARCHETYPE_NAMES}
        unknown = sorted(requested - allowed)
        if unknown:
            raise RuntimeError(f"Unknown cross-pathogen archetypes requested: {unknown}")
        wanted = requested
    scenarios = [
        copy.deepcopy(sc)
        for sc in base.SIRSimulator.ARCHETYPE_SCENARIOS
        if str(sc.get("archetype", "")).lower() in wanted
    ]
    found = {str(sc.get("archetype", "")).lower() for sc in scenarios}
    missing = sorted(wanted - found)
    if missing:
        raise RuntimeError(f"Missing cross-pathogen archetypes: {missing}")
    return scenarios


def _is_refined_mpv_curve(curve: Optional[Dict]) -> bool:
    curve = curve or {}
    text = " ".join(
        [
            str(curve.get("archetype", "")),
            str(curve.get("scenario", "")),
            str(curve.get("label", "")),
            str(curve.get("route", "")),
            str(curve.get("structural_noise_profile", "")),
        ]
    ).lower()
    return "mpv_like_refined" in text or "contact_network_refined" in text


def _is_mpox_specific_curve(curve: Optional[Dict]) -> bool:
    curve = curve or {}
    text = " ".join(
        [
            str(curve.get("archetype", "")),
            str(curve.get("scenario", "")),
            str(curve.get("label", "")),
            str(curve.get("route", "")),
            str(curve.get("structural_noise_profile", "")),
            str(curve.get("disease", "")),
            str(curve.get("preset", "")),
        ]
    ).lower()
    return (
        "mpox_specific" in text
        or "mpox" in text
        or "monkeypox" in text
        or _is_refined_mpv_curve(curve)
    )


def _is_influenza_curve(curve: Optional[Dict]) -> bool:
    curve = curve or {}
    text = " ".join(
        [
            str(curve.get("archetype", "")),
            str(curve.get("scenario", "")),
            str(curve.get("label", "")),
            str(curve.get("route", "")),
            str(curve.get("structural_noise_profile", "")),
        ]
    ).lower()
    return "influenza" in text or "h1n1" in text or "flu_like" in text


def _is_seasonal_influenza_curve(curve: Optional[Dict]) -> bool:
    curve = curve or {}
    text = " ".join(
        [
            str(curve.get("archetype", "")),
            str(curve.get("scenario", "")),
            str(curve.get("label", "")),
            str(curve.get("route", "")),
            str(curve.get("structural_noise_profile", "")),
        ]
    ).lower()
    return "influenza_seasonal" in text or "high_baseline" in text or "seasonal" in text


def _compute_mpox_truth_refs(simulator, truth_I, GT):
    sigma = max(2, int(np.ceil(GT * 0.5)))
    refs = simulator._compute_truth_references(truth_I, GT=GT, sigma=sigma)
    smooth = np.maximum(
        base.gaussian_filter1d(np.asarray(truth_I, dtype=float), sigma=max(1.5, float(GT) * 0.30)),
        0.0,
    )
    if smooth.size == 0:
        return refs

    peak = float(np.max(smooth)) if np.max(smooth) > 0 else 1.0
    baseline_end = max(7, int(np.ceil(1.95 * GT)))
    window = max(3, int(np.ceil(0.70 * GT)))
    search_end = min(
        len(smooth) - 2,
        max(
            baseline_end + window + 1,
            min(
                int(refs.T2_dyn) - 2,
                max(int(0.55 * refs.Tp), baseline_end + window + 2),
            ),
        ),
    )
    abs_thr = max(peak * 0.010, 0.35)
    early_growth = None
    for i in range(baseline_end, max(baseline_end + 1, search_end - window + 1)):
        seg = smooth[i:i + window + 1]
        if len(seg) < window + 1 or seg[-1] <= abs_thr:
            continue
        diffs = np.diff(seg)
        growth_ratio = seg[-1] / max(seg[0], 0.20)
        positive_share = float(np.mean(diffs > -0.003 * peak))
        if positive_share >= 0.84 and growth_ratio >= 1.22:
            early_growth = i
            break

    growth_cap = min(int(refs.T1_growth), int(refs.T1_struct), int(refs.T2_dyn) - 3)
    if int(refs.Tp) >= int(np.ceil(20.0 * GT)) or int(refs.T1_growth) >= baseline_end + int(np.ceil(3.0 * GT)):
        growth_cap = min(growth_cap, baseline_end + int(np.ceil(2.4 * GT)))
    if early_growth is None:
        early_growth = max(baseline_end, int(refs.T1_struct) - int(np.ceil(0.45 * GT)))
    T1_growth = max(baseline_end, min(int(early_growth), growth_cap))

    struct_target = T1_growth + max(1, int(np.ceil(0.55 * GT)))
    struct_cap = min(int(refs.T1_struct), int(refs.T2_dyn) - 1, int(refs.T2_dom) - 1, int(refs.Tp) - 2)
    T1_struct = max(T1_growth + 1, min(struct_target, struct_cap))

    t2_gap = max(int(np.ceil(4.2 * GT)), 9)
    T2_dyn = min(int(refs.T2_dyn), T1_growth + t2_gap)
    T2_dyn = max(T1_struct + 2, min(T2_dyn, int(refs.Tp) - 2))

    t2_dom_target = T2_dyn + max(1, int(np.ceil(0.75 * GT)))
    T2_dom = min(int(refs.T2_dom), t2_dom_target)
    T2_dom = max(T1_struct + 2, min(T2_dom, int(refs.Tp) - 1))

    return base.TruthReferences(
        T1_growth=int(T1_growth),
        T1_struct=int(T1_struct),
        T2_dyn=int(T2_dyn),
        T2_dom=int(T2_dom),
        Tp=int(refs.Tp),
    )


def _compute_seasonal_influenza_truth_refs(simulator, truth_I, GT):
    sigma = max(2, int(np.ceil(GT * 0.45)))
    refs = simulator._compute_truth_references(truth_I, GT=GT, sigma=sigma)
    smooth = np.maximum(
        base.gaussian_filter1d(np.asarray(truth_I, dtype=float), sigma=max(1.2, float(GT) * 0.28)),
        0.0,
    )
    if smooth.size == 0:
        return refs

    peak = float(np.max(smooth)) if np.max(smooth) > 0 else 1.0
    peak_idx = int(np.argmax(smooth))
    long_shoulder_mode = peak_idx >= max(80, int(np.ceil(28.0 * GT)))
    baseline_end = max(7, int(np.ceil(2.0 * GT)))
    window = max(3, int(np.ceil(0.85 * GT)))
    early_probe_idx = min(len(smooth) - 1, max(baseline_end, int(np.floor(0.22 * peak_idx))))
    early_level_ratio = float(smooth[early_probe_idx] / max(peak, 1e-6))
    search_end = min(
        len(smooth) - 2,
        max(
            baseline_end + window + 1,
            min(
                int(refs.T2_dyn) - 2,
                max(int(0.52 * refs.Tp), baseline_end + window + 2),
            ),
        ),
    )
    abs_thr = max(peak * 0.016, 0.36)
    early_growth = None
    for i in range(baseline_end, max(baseline_end + 1, search_end - window + 1)):
        seg = smooth[i:i + window + 1]
        if len(seg) < window + 1 or seg[-1] <= abs_thr:
            continue
        diffs = np.diff(seg)
        growth_ratio = seg[-1] / max(seg[0], 0.20)
        positive_share = float(np.mean(diffs > -0.0025 * peak))
        if positive_share >= 0.82 and growth_ratio >= 1.18:
            early_growth = i
            break

    growth_floor = max(
        baseline_end + int(np.ceil(1.3 * GT)),
        int(np.floor(peak_idx * (0.20 if long_shoulder_mode else 0.16))),
    )
    peak_based_cap = max(
        growth_floor + 1,
        int(np.floor(peak_idx * (0.56 if long_shoulder_mode else 0.46))),
    )
    gt_based_cap = baseline_end + int(np.ceil((6.2 if long_shoulder_mode else 4.8) * GT))
    growth_cap = min(
        int(refs.T1_growth),
        int(refs.T1_struct),
        int(refs.T2_dyn) - 4,
        peak_based_cap,
        gt_based_cap,
    )
    if not long_shoulder_mode:
        if early_level_ratio < 0.075:
            growth_floor = max(growth_floor, int(np.floor(0.24 * peak_idx)))
            growth_cap = max(growth_floor, growth_cap)
        elif early_level_ratio > 0.16:
            growth_cap = min(growth_cap, max(growth_floor + 1, int(np.floor(0.20 * peak_idx))))
    growth_cap = max(growth_floor, growth_cap)
    if early_growth is None:
        early_growth = min(
            growth_cap,
            max(
                growth_floor,
                min(int(refs.T1_growth), int(refs.T1_struct)),
            ),
        )
    T1_growth = max(growth_floor, min(int(early_growth), growth_cap))

    struct_target = T1_growth + max(1, int(np.ceil((0.72 if long_shoulder_mode else 0.60) * GT)))
    struct_cap = min(
        max(T1_growth + 1, int(refs.T1_struct)),
        int(refs.T2_dyn) - 2,
        int(refs.T2_dom) - 2,
        int(refs.Tp) - 3,
    )
    T1_struct = max(T1_growth + 1, min(struct_target, struct_cap))

    t2_gap = max(int(np.ceil((5.0 if long_shoulder_mode else 4.2) * GT)), 10)
    t2_peak_cap = max(T1_struct + 2, int(np.floor(peak_idx * (0.72 if long_shoulder_mode else 0.62))))
    T2_dyn = min(int(refs.T2_dyn), T1_growth + t2_gap, t2_peak_cap)
    T2_dyn = max(T1_struct + 2, min(T2_dyn, int(refs.Tp) - 2))

    t2_dom_target = T2_dyn + max(1, int(np.ceil((0.82 if long_shoulder_mode else 0.70) * GT)))
    t2_dom_cap = max(
        T2_dyn,
        min(int(refs.T2_dom), int(np.floor(peak_idx * (0.80 if long_shoulder_mode else 0.70))), int(refs.Tp) - 1),
    )
    T2_dom = max(T2_dyn, min(t2_dom_target, t2_dom_cap))

    return base.TruthReferences(
        T1_growth=int(T1_growth),
        T1_struct=int(T1_struct),
        T2_dyn=int(T2_dyn),
        T2_dom=int(T2_dom),
        Tp=int(refs.Tp),
    )


def apply_influenza_prototype_rebuild() -> None:
    if getattr(base, "_cross_pathogen_influenza_rebuilt", False):
        return

    original_generate = base.SIRSimulator.generate_sir_curve
    original_build = base.build_archetype_adaptive_ep
    original_settings = base.adaptive_rolling_settings

    scenarios = getattr(base.SIRSimulator, "ARCHETYPE_SCENARIOS", [])
    for sc in scenarios:
        if str(sc.get("archetype", "")).lower() == "influenza_like":
            sc["R0_range"] = (1.25, 2.25)
            sc["GT_range"] = (2.0, 3.1)
            sc["SI_range"] = (1.9, 3.1)
            sc["report_delay_range"] = (1, 5)
            sc["noise_std_range"] = (0.10, 0.24)
            sc["structural_noise_profile"] = "influenza_pandemic_wave"
        elif str(sc.get("archetype", "")).lower() == "influenza_seasonal_high_baseline":
            sc["R0_range"] = (1.18, 1.58)
            sc["GT_range"] = (2.3, 3.2)
            sc["SI_range"] = (2.2, 3.2)
            sc["report_delay_range"] = (1, 4)
            sc["noise_std_range"] = (0.12, 0.22)
            sc["structural_noise_profile"] = "influenza_high_baseline"

    noise_profiles = getattr(base.SIRSimulator, "STRUCTURED_NOISE_PROFILES", {})
    noise_profiles["influenza_pandemic_wave"] = dict(
        weekly_amp=(0.04, 0.12),
        shift_prob=0.34,
        shift_factor=(0.88, 1.18),
        pulse_prob=0.28,
        pulse_window_frac=0.30,
        pulse_amp=(0.006, 0.024),
        spike_lambda_scale=0.68,
        spike_amp=(1.10, 2.20),
        plateau_prob=0.22,
        plateau_scale=(1.2, 2.3),
        mult_noise=(0.025, 0.075),
        additive_scale=0.014,
    )
    noise_profiles["influenza_high_baseline"] = dict(
        weekly_amp=(0.03, 0.10),
        shift_prob=0.26,
        shift_factor=(0.92, 1.14),
        pulse_prob=0.24,
        pulse_window_frac=0.26,
        pulse_amp=(0.004, 0.016),
        spike_lambda_scale=0.52,
        spike_amp=(1.08, 1.85),
        plateau_prob=0.26,
        plateau_scale=(1.2, 2.1),
        mult_noise=(0.022, 0.060),
        additive_scale=0.012,
    )

    def rebuilt_generate_sir_curve(self, days=200, R0=2.5, GT=5.0, N=100000, I0=5,
                                   noise_std=0.2, report_delay=0, add_structured_noise=True,
                                   structural_noise_profile="standard", SI=None, archetype=None):
        result = original_generate(
            self,
            days=days,
            R0=R0,
            GT=GT,
            N=N,
            I0=I0,
            noise_std=noise_std,
            report_delay=report_delay,
            add_structured_noise=add_structured_noise,
            structural_noise_profile=structural_noise_profile,
            SI=SI,
            archetype=archetype,
        )
        if not _is_influenza_curve({"archetype": archetype, "structural_noise_profile": structural_noise_profile}):
            return result

        truth = np.asarray(result.get("truth_I"), dtype=float).copy()
        observed = np.asarray(result.get("I"), dtype=float).copy()
        if truth.size == 0:
            return result

        peak = float(np.max(truth)) if np.max(truth) > 0 else 1.0
        peak_idx = int(np.argmax(truth))
        n_days = truth.size
        is_seasonal = _is_seasonal_influenza_curve(
            {"archetype": archetype, "structural_noise_profile": structural_noise_profile}
        )
        seasonal_long_shoulder = bool(is_seasonal and float(R0) <= 1.26)

        # Seasonal influenza needs a persistent background and a noisier
        # shoulder; pandemic-like influenza should keep a cleaner wave shape.
        if is_seasonal:
            if seasonal_long_shoulder:
                baseline_level = peak * self.rng.uniform(0.007, 0.015)
                drift_hi = self.rng.uniform(1.02, 1.14)
                shoulder_amp = peak * self.rng.uniform(0.010, 0.024)
                prewave_hi = self.rng.uniform(0.28, 0.56)
                prewave_obs_scale = 0.66
                variant_name = "influenza_seasonal_high_baseline_long_shoulder"
            else:
                baseline_level = peak * self.rng.uniform(0.004, 0.010)
                drift_hi = self.rng.uniform(0.98, 1.08)
                shoulder_amp = peak * self.rng.uniform(0.004, 0.014)
                prewave_hi = self.rng.uniform(0.14, 0.32)
                prewave_obs_scale = 0.54
                variant_name = "influenza_seasonal_high_baseline_fast"
        else:
            baseline_level = peak * self.rng.uniform(0.002, 0.010)
            drift_hi = self.rng.uniform(0.98, 1.12)
            shoulder_amp = peak * self.rng.uniform(0.008, 0.040)
            prewave_hi = self.rng.uniform(0.45, 0.95)
            prewave_obs_scale = 0.75
            variant_name = "influenza_pandemic_wave_rebuilt"
        baseline_drift = np.linspace(
            0.92,
            drift_hi,
            n_days,
            dtype=float,
        )
        endemic_floor = baseline_level * baseline_drift

        if is_seasonal and not seasonal_long_shoulder:
            shoulder_center_hi = max(10, int(min(max(peak_idx * 0.42, GT * 3.6), n_days - 8)))
            shoulder_center_lo = max(4, int(min(max(peak_idx * 0.22, GT * 1.8), shoulder_center_hi - 1)))
            shoulder_width = max(3, int(np.ceil(self.rng.uniform(0.9, 2.0) * GT)))
        elif is_seasonal and seasonal_long_shoulder:
            shoulder_center_hi = max(12, int(min(max(peak_idx * 0.68, GT * 5.2), n_days - 8)))
            shoulder_center_lo = max(5, int(min(max(peak_idx * 0.40, GT * 3.0), shoulder_center_hi - 1)))
            shoulder_width = max(3, int(np.ceil(self.rng.uniform(1.6, 3.6) * GT)))
        else:
            shoulder_center_hi = max(
                12,
                int(min(max(peak_idx * 0.78, GT * 5.6), n_days - 8)),
            )
            shoulder_center_lo = max(
                4,
                int(
                    min(
                        max(peak_idx * 0.32, GT * 2.5),
                        shoulder_center_hi - 1,
                    )
                ),
            )
            shoulder_width = max(3, int(np.ceil(self.rng.uniform(2.5, 7.5) * GT)))
        shoulder_center = int(self.rng.randint(shoulder_center_lo, shoulder_center_hi + 1))
        shoulder = shoulder_amp * np.exp(
            -0.5 * ((np.arange(n_days, dtype=float) - shoulder_center) / max(1.0, shoulder_width)) ** 2
        )

        prewave_end = max(
            10,
            min(
                n_days,
                int(
                    max(
                        (
                            peak_idx * self.rng.uniform(0.28, 0.42)
                            if (is_seasonal and not seasonal_long_shoulder)
                            else peak_idx * self.rng.uniform(0.46, 0.66)
                            if is_seasonal
                            else peak_idx * self.rng.uniform(0.55, 0.78)
                        ),
                        GT * (2.6 if (is_seasonal and not seasonal_long_shoulder) else 4.2 if is_seasonal else 7.0),
                    )
                ),
            ),
        )
        prewave = np.zeros(n_days, dtype=float)
        if prewave_end > 1:
            prewave[:prewave_end] = np.linspace(
                (
                    baseline_level * self.rng.uniform(0.03, 0.09)
                    if (is_seasonal and not seasonal_long_shoulder)
                    else baseline_level * self.rng.uniform(0.06, 0.14)
                    if is_seasonal
                    else baseline_level * self.rng.uniform(0.10, 0.35)
                ),
                baseline_level * prewave_hi,
                prewave_end,
            )

        truth_adj = np.maximum(truth + endemic_floor + shoulder + prewave, 0.0)
        observed_adj = np.maximum(observed + 0.95 * endemic_floor + shoulder + prewave_obs_scale * prewave, 0.0)

        sigma_s = max(2, int(np.ceil(GT * 0.5)))
        refs = (
            _compute_seasonal_influenza_truth_refs(self, truth_adj, GT=GT)
            if is_seasonal
            else self._compute_truth_references(truth_adj, GT=GT, sigma=sigma_s)
        )
        result.update(
            I=observed_adj,
            truth_I=truth_adj,
            truth_T1=refs.T1_growth,
            truth_T2=refs.T2_dom,
            truth_Tp=refs.Tp,
            truth_T1_growth=refs.T1_growth,
            truth_T1_struct=refs.T1_struct,
            truth_T2_dyn=refs.T2_dyn,
            truth_T2_dom=refs.T2_dom,
            prototype_variant=variant_name,
            prototype_baseline_level=float(baseline_level),
            prototype_shoulder_center=int(shoulder_center),
        )
        return result

    def rebuilt_build_archetype_adaptive_ep(base_ep, curve=None, prospective=False):
        ep = original_build(base_ep, curve=curve, prospective=prospective)
        if not _is_influenza_curve(curve):
            return ep

        is_seasonal = _is_seasonal_influenza_curve(curve)
        ep.t1_baseline_days = max(int(ep.t1_baseline_days), int(np.ceil((5.4 if is_seasonal else 4.6) * ep.GT)))
        if is_seasonal:
            ep.t1_structure_score_thresh = max(float(ep.t1_structure_score_thresh), 0.54)
            ep.t1_mid_z_thresh = max(float(ep.t1_mid_z_thresh), 1.00)
            ep.t1_relaxed_score_thresh = max(float(ep.t1_relaxed_score_thresh), 0.38)
            ep.t1_relaxed_mid_z_thresh = max(float(ep.t1_relaxed_mid_z_thresh), 0.61)
            ep.t1_prealert_mid_z_thresh = max(float(ep.t1_prealert_mid_z_thresh), 0.38)
            ep.t1_valid_strength_min = max(float(getattr(ep, "t1_valid_strength_min", 0.18)), 0.20)
        else:
            ep.t1_structure_score_thresh = min(float(ep.t1_structure_score_thresh), 0.47)
            ep.t1_mid_z_thresh = min(float(ep.t1_mid_z_thresh), 0.88)
            ep.t1_relaxed_score_thresh = min(float(ep.t1_relaxed_score_thresh), 0.33)
            ep.t1_relaxed_mid_z_thresh = min(float(ep.t1_relaxed_mid_z_thresh), 0.52)
            ep.t1_prealert_mid_z_thresh = min(float(ep.t1_prealert_mid_z_thresh), 0.26)
        ep.t2_dom_score_thresh = min(float(ep.t2_dom_score_thresh), 0.44 if is_seasonal else 0.42)

        if prospective:
            if is_seasonal:
                ep.prospective_min_start = max(
                    int(getattr(ep, "prospective_min_start", 0)),
                    int(np.ceil(2.60 * ep.GT + 0.45 * getattr(ep, "T_report", 2.0))),
                )
                ep.prospective_min_window_gt = min(max(float(ep.prospective_min_window_gt), 2.35), 2.70)
                ep.prospective_min_window_floor = max(int(ep.prospective_min_window_floor), 10)
                ep.prospective_t2_min_after_t1_gt = min(max(float(ep.prospective_t2_min_after_t1_gt), 1.15), 1.35)
                ep.prospective_t2_strength_min = min(max(float(ep.prospective_t2_strength_min), 0.37), 0.41)
                ep.t1_confirm_allow_same_window_as_first_alert = False
                ep.t1_first_alert_strength = max(float(getattr(ep, "t1_first_alert_strength", 0.14)), 0.18)
                ep.t1_early_confirm_strength = max(float(getattr(ep, "t1_early_confirm_strength", 0.22)), 0.28)
                ep.t1_single_confirm_strength = max(float(getattr(ep, "t1_single_confirm_strength", 0.46)), 0.52)
                ep.t1_first_alert_min_persist_gt = max(float(getattr(ep, "t1_first_alert_min_persist_gt", 0.0)), 0.16)
                ep.t1_early_confirm_min_persist_gt = max(float(getattr(ep, "t1_early_confirm_min_persist_gt", 0.18)), 0.32)
                ep.t1_confirm_single_min_persist_gt = max(float(getattr(ep, "t1_confirm_single_min_persist_gt", 0.18)), 0.34)
                ep.t1_flu_raw_immediate_strength = max(float(getattr(ep, "t1_flu_raw_immediate_strength", 0.30)), 0.40)
                ep.t1_raw_growth_z = max(float(getattr(ep, "t1_raw_growth_z", 1.9)), 2.00)
                ep.t1_raw_growth_min_persist_gt = max(float(getattr(ep, "t1_raw_growth_min_persist_gt", 0.25)), 0.34)
                ep.t1_raw_growth_confirm_strength = max(
                    float(getattr(ep, "t1_raw_growth_confirm_strength", 0.50)),
                    0.58,
                )
                ep.t1_estimate_history_quantile = min(
                    max(int(getattr(ep, "t1_estimate_history_quantile", 25)), 30),
                    40,
                )
                ep.t1_raw_growth_backcast_gt = min(float(getattr(ep, "t1_raw_growth_backcast_gt", 0.85)), 0.38)
                ep.t1_estimate_backtrack_gt = min(float(getattr(ep, "t1_estimate_backtrack_gt", 0.55)), 0.24)
                ep.t1_estimate_max_backtrack_gt = min(float(getattr(ep, "t1_estimate_max_backtrack_gt", 0.65)), 0.36)
                ep.t1_aux_stale_backcast_gt = min(float(getattr(ep, "t1_aux_stale_backcast_gt", 0.40)), 0.24)
                ep.t1_struct_stale_backcast_gt = min(float(getattr(ep, "t1_struct_stale_backcast_gt", 0.56)), 0.34)
            else:
                ep.prospective_min_start = max(
                    int(getattr(ep, "prospective_min_start", 0)),
                    int(np.ceil(2.05 * ep.GT + 0.35 * getattr(ep, "T_report", 2.0))),
                )
                ep.prospective_min_window_gt = min(max(float(ep.prospective_min_window_gt), 1.95), 2.25)
                ep.prospective_min_window_floor = max(int(ep.prospective_min_window_floor), 8)
                ep.prospective_t2_min_after_t1_gt = min(max(float(ep.prospective_t2_min_after_t1_gt), 0.95), 1.15)
                ep.prospective_t2_strength_min = min(max(float(ep.prospective_t2_strength_min), 0.33), 0.37)
            ep.prospective_tp_maturity_gt = min(
                max(float(ep.prospective_tp_maturity_gt), 1.55 if is_seasonal else 1.40),
                1.80 if is_seasonal else 1.60,
            )
            ep.prospective_tp_decline_frac = min(
                max(float(ep.prospective_tp_decline_frac), 0.055 if is_seasonal else 0.045),
                0.085 if is_seasonal else 0.075,
            )
        return ep

    def rebuilt_adaptive_rolling_settings(base_window, base_step, base_rounds, base_tol, ep, curve):
        settings = original_settings(base_window, base_step, base_rounds, base_tol, ep, curve)
        if not _is_seasonal_influenza_curve(curve):
            return settings

        gt = float(curve.get("GT", getattr(ep, "GT", 2.8))) if curve else float(getattr(ep, "GT", 2.8))
        tuned_window = int(np.ceil(13.0 * gt))
        settings["window_size"] = int(max(42, int(settings["window_size"]), tuned_window))
        settings["confirm_rounds"] = max(int(settings["confirm_rounds"]), 3)
        settings["confirm_tol"] = int(max(int(settings["confirm_tol"]), int(np.ceil(1.00 * gt))))
        return settings

    base.EpiParams.PRESETS["influenza_seasonal"] = dict(
        name="Seasonal Influenza (High Baseline)",
        GT=2.8,
        SI=2.5,
        T_inc=1.7,
        T_inf=5.2,
        T_report=3.0,
        T_week=7.0,
    )

    base.SIRSimulator.generate_sir_curve = rebuilt_generate_sir_curve
    base.build_archetype_adaptive_ep = rebuilt_build_archetype_adaptive_ep
    base.adaptive_rolling_settings = rebuilt_adaptive_rolling_settings
    base._cross_pathogen_influenza_rebuilt = True
    print("[cross-pathogen] Influenza prototype rebuild enabled.")


def apply_mpv_like_optimization() -> None:
    if getattr(base, "_cross_pathogen_mpv_tuned", False):
        return

    original_build = base.build_archetype_adaptive_ep
    original_settings = base.adaptive_rolling_settings
    scenarios = getattr(base.SIRSimulator, "ARCHETYPE_SCENARIOS", [])
    for sc in scenarios:
        if str(sc.get("archetype", "")).lower() == "mpv_like_contact_network":
            sc["R0_range"] = (1.05, 2.10)
            sc["GT_range"] = (5.5, 8.5)
            sc["SI_range"] = (6.5, 10.5)
            sc["report_delay_range"] = (2, 6)
            sc["noise_std_range"] = (0.16, 0.34)
            sc["structural_noise_profile"] = "contact_network_mpv_tuned"
            break
    noise_profiles = getattr(base.SIRSimulator, "STRUCTURED_NOISE_PROFILES", {})
    noise_profiles["contact_network_mpv_tuned"] = dict(
        weekly_amp=(0.03, 0.12),
        shift_prob=0.32,
        shift_factor=(0.82, 1.22),
        pulse_prob=0.30,
        pulse_window_frac=0.28,
        pulse_amp=(0.008, 0.040),
        spike_lambda_scale=0.85,
        spike_amp=(1.2, 3.0),
        plateau_prob=0.28,
        plateau_scale=(1.2, 2.8),
        mult_noise=(0.03, 0.09),
        additive_scale=0.022,
    )

    def tuned_build_archetype_adaptive_ep(base_ep, curve=None, prospective=False):
        ep = original_build(base_ep, curve=curve, prospective=prospective)
        profile = base.infer_archetype_adaptive_profile(curve, base_ep)
        if prospective and profile.get("mpv_like"):
            gt = float(profile.get("GT", getattr(ep, "GT", 7.0)))
            report_delay = float(profile.get("report_delay", getattr(ep, "T_report", 4.0)))
            ep.prospective_min_start = min(
                int(ep.prospective_min_start),
                max(8, int(np.ceil(1.35 * gt + 0.50 * report_delay))),
            )
            ep.t1_aux_min_strength = min(float(ep.t1_aux_min_strength), 0.13)
            ep.t1_aux_max_lead_gt = max(float(ep.t1_aux_max_lead_gt), 3.2)
            ep.t1_valid_strength_min = min(float(ep.t1_valid_strength_min), 0.20)
            ep.t1_first_alert_strength = min(float(ep.t1_first_alert_strength), 0.18)
            ep.t1_early_confirm_strength = min(float(ep.t1_early_confirm_strength), 0.27)
            ep.t1_fast_confirm_strength = min(float(ep.t1_fast_confirm_strength), 0.50)
            ep.t1_single_confirm_strength = min(float(ep.t1_single_confirm_strength), 0.47)
            ep.t1_first_alert_min_persist_gt = min(float(ep.t1_first_alert_min_persist_gt), 0.12)
            ep.t1_early_confirm_min_persist_gt = min(float(ep.t1_early_confirm_min_persist_gt), 0.28)
            ep.t1_confirm_single_min_persist_gt = min(float(ep.t1_confirm_single_min_persist_gt), 0.34)
            ep.t1_alert_graduation_gt = min(float(ep.t1_alert_graduation_gt), 0.55)
            ep.t1_alert_graduation_min_strength = min(float(ep.t1_alert_graduation_min_strength), 0.16)
            ep.t1_estimate_history_quantile = min(int(ep.t1_estimate_history_quantile), 24)
            ep.t1_raw_growth_backcast_gt = max(float(ep.t1_raw_growth_backcast_gt), 0.95)
            ep.t1_estimate_backtrack_gt = max(float(ep.t1_estimate_backtrack_gt), 0.90)
            ep.t1_estimate_max_backtrack_gt = max(float(ep.t1_estimate_max_backtrack_gt), 1.55)
            ep.t1_aux_stale_backcast_gt = max(float(ep.t1_aux_stale_backcast_gt), 0.85)
            ep.t1_struct_stale_backcast_gt = max(float(ep.t1_struct_stale_backcast_gt), 1.05)
            ep.prospective_min_window_gt = min(float(ep.prospective_min_window_gt), 2.55)
            ep.prospective_min_window_floor = min(int(ep.prospective_min_window_floor), 10)
            ep.prospective_t2_min_after_t1_gt = min(float(ep.prospective_t2_min_after_t1_gt), 1.35)
            ep.prospective_t2_strength_min = min(float(ep.prospective_t2_strength_min), 0.41)
            ep.prospective_t2_tail_quantile = min(float(ep.prospective_t2_tail_quantile), 0.38)
            ep.prospective_tp_maturity_gt = min(float(ep.prospective_tp_maturity_gt), 1.95)
            ep.prospective_tp_decline_frac = min(float(ep.prospective_tp_decline_frac), 0.10)
            ep.prospective_tp_fallback_quantile = 35.0
        return ep

    def tuned_adaptive_rolling_settings(base_window, base_step, base_rounds, base_tol, ep, curve):
        settings = original_settings(base_window, base_step, base_rounds, base_tol, ep, curve)
        profile = base.infer_archetype_adaptive_profile(curve, ep)
        if profile.get("mpv_like"):
            gt = float(profile.get("GT", getattr(ep, "GT", 7.0)))
            burden = float(profile.get("adaptive_burden", 0.0))
            tuned_window = int(np.ceil(8.8 * gt + 1.5 * burden))
            settings["window_size"] = int(max(78, min(settings["window_size"], tuned_window)))
            settings["confirm_rounds"] = min(int(settings["confirm_rounds"]), 2)
            settings["confirm_tol"] = int(max(base_step, min(int(settings["confirm_tol"]), int(np.ceil(0.90 * gt)))))
        return settings

    base.build_archetype_adaptive_ep = tuned_build_archetype_adaptive_ep
    base.adaptive_rolling_settings = tuned_adaptive_rolling_settings
    base._cross_pathogen_mpv_tuned = True
    print("[cross-pathogen] MPV-like local tuning enabled.")


def apply_mpox_specific_optimization() -> None:
    if getattr(base, "_cross_pathogen_mpox_specific_tuned", False):
        return

    original_infer = base.infer_archetype_adaptive_profile
    original_build = base.build_archetype_adaptive_ep
    original_settings = base.adaptive_rolling_settings
    original_generate = base.SIRSimulator.generate_sir_curve

    def tuned_infer_archetype_adaptive_profile(curve, ep):
        profile = original_infer(curve, ep)
        if not _is_mpox_specific_curve(curve):
            return profile

        tuned = dict(profile)
        tuned["mpox_specific"] = True
        tuned["mpv_like"] = True
        tuned["contact_like"] = True
        tuned["slow_contact_like"] = True
        tuned["hospital_like"] = False
        tuned["zoonotic_like"] = False
        tuned["vector_like"] = False
        tuned["adaptive_burden"] = float(min(max(float(tuned.get("adaptive_burden", 0.0)) + 0.08, 1.00), 1.95))
        tuned["adaptive_profile"] = (
            "conservative_slow_or_delayed" if tuned["adaptive_burden"] >= 1.20 else "moderate_adaptive"
        )
        return tuned

    def tuned_build_archetype_adaptive_ep(base_ep, curve=None, prospective=False):
        ep = original_build(base_ep, curve=curve, prospective=prospective)
        if not _is_mpox_specific_curve(curve):
            return ep

        gt = float(getattr(ep, "GT", 7.0))
        report_delay = float(getattr(ep, "T_report", 4.0))

        if prospective:
            ep.prospective_min_start = max(
                int(getattr(ep, "prospective_min_start", 0)),
                int(np.ceil(1.45 * gt + 0.75 * report_delay)),
            )
            ep.t1_confirm_allow_same_window_as_first_alert = False
            ep.t1_aux_min_strength = min(max(float(ep.t1_aux_min_strength), 0.15), 0.20)
            ep.t1_aux_max_lead_gt = min(max(float(ep.t1_aux_max_lead_gt), 1.3), 1.9)
            ep.t1_valid_strength_min = min(max(float(ep.t1_valid_strength_min), 0.21), 0.27)
            ep.t1_first_alert_strength = min(max(float(ep.t1_first_alert_strength), 0.20), 0.26)
            ep.t1_early_confirm_strength = min(max(float(ep.t1_early_confirm_strength), 0.28), 0.34)
            ep.t1_fast_confirm_strength = min(max(float(ep.t1_fast_confirm_strength), 0.47), 0.53)
            ep.t1_single_confirm_strength = min(max(float(ep.t1_single_confirm_strength), 0.45), 0.51)
            ep.t1_first_alert_min_persist_gt = min(max(float(ep.t1_first_alert_min_persist_gt), 0.24), 0.36)
            ep.t1_early_confirm_min_persist_gt = min(max(float(ep.t1_early_confirm_min_persist_gt), 0.34), 0.48)
            ep.t1_confirm_single_min_persist_gt = min(max(float(ep.t1_confirm_single_min_persist_gt), 0.38), 0.52)
            ep.t1_alert_graduation_gt = min(max(float(ep.t1_alert_graduation_gt), 0.54), 0.72)
            ep.t1_alert_graduation_min_strength = min(
                max(float(ep.t1_alert_graduation_min_strength), 0.16),
                0.22,
            )
            ep.t1_estimate_history_quantile = min(
                max(int(getattr(ep, "t1_estimate_history_quantile", 25)), 28),
                40,
            )
            ep.t1_raw_growth_backcast_gt = min(max(float(ep.t1_raw_growth_backcast_gt), 0.16), 0.24)
            ep.t1_estimate_backtrack_gt = min(max(float(ep.t1_estimate_backtrack_gt), 0.14), 0.24)
            ep.t1_estimate_max_backtrack_gt = min(max(float(ep.t1_estimate_max_backtrack_gt), 0.28), 0.46)
            ep.t1_aux_stale_backcast_gt = min(max(float(ep.t1_aux_stale_backcast_gt), 0.18), 0.30)
            ep.t1_struct_stale_backcast_gt = min(max(float(ep.t1_struct_stale_backcast_gt), 0.24), 0.42)
            ep.prospective_min_window_gt = min(max(float(ep.prospective_min_window_gt), 2.40), 2.90)
            ep.prospective_min_window_floor = max(int(ep.prospective_min_window_floor), 12)
            ep.prospective_t2_min_after_t1_gt = min(max(float(ep.prospective_t2_min_after_t1_gt), 1.75), 2.10)
            ep.prospective_t2_strength_min = min(max(float(ep.prospective_t2_strength_min), 0.46), 0.52)
            ep.prospective_t2_tail_quantile = min(max(float(ep.prospective_t2_tail_quantile), 0.44), 0.52)
            ep.prospective_t2_confirm_rounds = max(int(getattr(ep, "prospective_t2_confirm_rounds", 3)), 5)
            ep.prospective_t2_confirm_tol_gt = min(max(float(ep.prospective_t2_confirm_tol_gt), 1.15), 1.35)
            ep.prospective_tp_maturity_gt = min(max(float(ep.prospective_tp_maturity_gt), 2.00), 2.35)
            ep.prospective_tp_decline_frac = min(max(float(ep.prospective_tp_decline_frac), 0.08), 0.12)
            ep.prospective_tp_fallback_quantile = 35.0
        else:
            ep.t1_structure_score_thresh = min(float(ep.t1_structure_score_thresh), 0.48)
            ep.t1_mid_z_thresh = min(float(ep.t1_mid_z_thresh), 0.92)
            ep.t1_relaxed_score_thresh = min(float(ep.t1_relaxed_score_thresh), 0.38)
            ep.t2_dom_score_thresh = max(float(ep.t2_dom_score_thresh), 0.54)
            ep.t2_low_share_thresh_short = max(float(ep.t2_low_share_thresh_short), 0.34)
            ep.t2_low_share_thresh_medium = max(float(ep.t2_low_share_thresh_medium), 0.38)
            ep.t2_low_share_thresh_long = max(float(ep.t2_low_share_thresh_long), 0.42)
            ep.real_t2_low_share_short = ep.t2_low_share_thresh_short
            ep.real_t2_low_share_medium = ep.t2_low_share_thresh_medium
            ep.real_t2_low_share_long = ep.t2_low_share_thresh_long
        return ep

    def tuned_generate_sir_curve(self, days=200, R0=2.5, GT=5.0, N=100000, I0=5,
                                 noise_std=0.2, report_delay=0, add_structured_noise=True,
                                 structural_noise_profile="standard", SI=None, archetype=None):
        result = original_generate(
            self,
            days=days,
            R0=R0,
            GT=GT,
            N=N,
            I0=I0,
            noise_std=noise_std,
            report_delay=report_delay,
            add_structured_noise=add_structured_noise,
            structural_noise_profile=structural_noise_profile,
            SI=SI,
            archetype=archetype,
        )
        if not _is_mpox_specific_curve({"archetype": archetype, "structural_noise_profile": structural_noise_profile}):
            return result

        truth = np.asarray(result.get("truth_I"), dtype=float).copy()
        observed = np.asarray(result.get("I"), dtype=float).copy()
        if truth.size == 0:
            return result

        peak = float(np.max(truth)) if np.max(truth) > 0 else 1.0
        peak_idx = int(np.argmax(truth))
        n_days = truth.size
        x = np.arange(n_days, dtype=float)

        cluster_center = int(
            min(
                max(np.ceil(2.4 * GT), peak_idx * 0.30),
                max(np.ceil(4.2 * GT), peak_idx * 0.46),
            )
        )
        cluster_width = max(3, int(np.ceil(self.rng.uniform(0.75, 1.25) * GT)))
        cluster_amp = peak * self.rng.uniform(0.028, 0.075)
        early_cluster = cluster_amp * np.exp(-0.5 * ((x - cluster_center) / max(1.0, cluster_width)) ** 2)

        ramp_end = int(
            min(
                n_days,
                max(
                    cluster_center + cluster_width + 2,
                    min(peak_idx * self.rng.uniform(0.58, 0.72), 7.6 * GT),
                ),
            )
        )
        early_ramp = np.zeros(n_days, dtype=float)
        if ramp_end > 2:
            early_ramp[:ramp_end] = np.linspace(
                0.0,
                peak * self.rng.uniform(0.014, 0.040),
                ramp_end,
            )

        tail_start = int(
            min(
                n_days - 1,
                max(cluster_center + 2 * cluster_width + 4, peak_idx * self.rng.uniform(0.74, 0.86)),
            )
        )
        tail_scale = np.ones(n_days, dtype=float)
        if tail_start < n_days - 3:
            decay = np.linspace(1.0, self.rng.uniform(0.62, 0.82), n_days - tail_start)
            tail_scale[tail_start:] = decay

        truth_adj = np.maximum((truth + early_cluster + early_ramp) * tail_scale, 0.0)
        observed_adj = np.maximum((observed + 0.92 * early_cluster + 0.88 * early_ramp) * tail_scale, 0.0)

        refs = _compute_mpox_truth_refs(self, truth_adj, GT=GT)
        result.update(
            I=observed_adj,
            truth_I=truth_adj,
            truth_T1=refs.T1_growth,
            truth_T2=refs.T2_dom,
            truth_Tp=refs.Tp,
            truth_T1_growth=refs.T1_growth,
            truth_T1_struct=refs.T1_struct,
            truth_T2_dyn=refs.T2_dyn,
            truth_T2_dom=refs.T2_dom,
            prototype_variant="mpox_specific_rebuilt_v1",
            prototype_cluster_center=int(cluster_center),
            prototype_tail_start=int(tail_start),
        )
        return result

    def tuned_adaptive_rolling_settings(base_window, base_step, base_rounds, base_tol, ep, curve):
        settings = original_settings(base_window, base_step, base_rounds, base_tol, ep, curve)
        if not _is_mpox_specific_curve(curve):
            return settings

        gt = float(curve.get("GT", getattr(ep, "GT", 7.0))) if curve else float(getattr(ep, "GT", 7.0))
        tuned_window = int(np.ceil(12.0 * gt))
        settings["window_size"] = int(max(104, int(settings["window_size"]), tuned_window))
        settings["confirm_rounds"] = max(int(settings["confirm_rounds"]), 3)
        settings["confirm_tol"] = int(
            max(
                int(settings["confirm_tol"]),
                int(np.ceil(1.00 * gt)),
            )
        )
        return settings

    base.infer_archetype_adaptive_profile = tuned_infer_archetype_adaptive_profile
    base.build_archetype_adaptive_ep = tuned_build_archetype_adaptive_ep
    base.adaptive_rolling_settings = tuned_adaptive_rolling_settings
    base.SIRSimulator.generate_sir_curve = tuned_generate_sir_curve
    base._cross_pathogen_mpox_specific_tuned = True
    print("[cross-pathogen] Mpox-specific local tuning enabled.")


def summarize_t1_focus_metrics(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df is None or len(detail_df) == 0:
        return pd.DataFrame()

    rows = []
    for archetype, group in detail_df.groupby("core_archetype", dropna=False):
        work = group.copy()
        t1_err = pd.to_numeric(work.get("T1_err"), errors="coerce")
        gt = pd.to_numeric(work.get("GT"), errors="coerce")
        lead_before_t2 = pd.to_numeric(work.get("lead_before_T2"), errors="coerce")
        first_lead_before_t2 = pd.to_numeric(work.get("lead_before_T2_by_first_alert"), errors="coerce")
        t1_confirmed = pd.to_numeric(work.get("T1_confirmed"), errors="coerce")
        truth_t2 = pd.to_numeric(work.get("truth_T2_dom"), errors="coerce")
        valid_err = t1_err.dropna()
        if len(valid_err) == 0:
            continue
        rows.append(
            dict(
                core_archetype=archetype,
                n=int(len(work)),
                T1_median_MAE=float(valid_err.median()),
                T1_mean_MAE=float(valid_err.mean()),
                T1_within_1GT=float(((t1_err <= gt) & t1_err.notna() & gt.notna()).mean()),
                T1_within_2GT=float(((t1_err <= 2 * gt) & t1_err.notna() & gt.notna()).mean()),
                T1_confirm_before_T2=float(((lead_before_t2 >= 0) & lead_before_t2.notna()).mean()),
                T1_first_alert_before_T2=float(((first_lead_before_t2 >= 0) & first_lead_before_t2.notna()).mean()),
                T1_IQR=float(valid_err.quantile(0.75) - valid_err.quantile(0.25)),
                T1_p90=float(valid_err.quantile(0.90)),
                T1_confirmed_within_1GT=float(((pd.to_numeric(work.get("T1_delay"), errors="coerce") <= gt) & gt.notna()).mean()),
                T1_confirmed_within_2GT=float(((pd.to_numeric(work.get("T1_delay"), errors="coerce") <= 2 * gt) & gt.notna()).mean()),
                T1_first_alert_within_1GT=float(((pd.to_numeric(work.get("T1_first_alert_err"), errors="coerce") <= gt) & gt.notna()).mean()),
                T1_first_alert_within_2GT=float(((pd.to_numeric(work.get("T1_first_alert_err"), errors="coerce") <= 2 * gt) & gt.notna()).mean()),
                T1_confirmed_missing=float((t1_confirmed.isna() | truth_t2.isna()).mean()),
            )
        )
    return pd.DataFrame(rows).sort_values("core_archetype").reset_index(drop=True)


def _export_stage_protocol_manifest(
    out_path: Path,
    rows: List[Dict[str, object]],
) -> pd.DataFrame:
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_path, index=False, encoding="utf-8-sig")
    return manifest


def run_extended_core_validation(
    ep_opt: base.EpiParams,
    n_per_archetype: int = 40,
    seed: int = 5151,
    eemd_trials_default: int = 12,
    output_label: str = "cross_pathogen_core_archetype",
    mpv_tuned: bool = False,
    selected_archetypes: Optional[List[str]] = None,
) -> Tuple[List[Dict], pd.DataFrame, pd.DataFrame]:
    selected_label = selected_archetypes or list(PRIMARY_CORE_ARCHETYPE_NAMES)
    print("\n" + "#" * 72)
    print("  CROSS-PATHOGEN STAGE 2: locked internal validation on fresh simulated curves")
    print("  Archetypes: " + " + ".join(selected_label))
    print(f"  n_per_archetype={n_per_archetype}")
    print("#" * 72)

    scenario_settings = {
        "high_transmissibility_coronavirus": dict(
            days=220,
            min_Tp=22,
            window_size=72,
            step=1,
            confirm_rounds=2,
            eemd_trials=8,
        ),
        "influenza_like": dict(days=180, min_Tp=24, window_size=70, step=1, confirm_rounds=2, eemd_trials=8),
        "influenza_seasonal_high_baseline": dict(
            days=220,
            min_Tp=30,
            window_size=84,
            step=1,
            confirm_rounds=2,
            eemd_trials=8,
        ),
        "moderate_coronavirus": dict(days=300, min_Tp=35, window_size=100, step=2, confirm_rounds=2, eemd_trials=eemd_trials_default),
        "chikungunya_vector_borne": dict(days=260, min_Tp=36, window_size=105, step=2, confirm_rounds=3, eemd_trials=eemd_trials_default),
        "mpox_specific": dict(
            days=320,
            min_Tp=40,
            window_size=116,
            step=2,
            confirm_rounds=3,
            eemd_trials=eemd_trials_default,
        ),
        "mpv_like_contact_network": dict(
            days=300,
            min_Tp=42,
            window_size=(108 if mpv_tuned else 120),
            step=2,
            confirm_rounds=(2 if mpv_tuned else 3),
            eemd_trials=eemd_trials_default,
        ),
        "mpv_like_refined_contact_network": dict(
            days=300,
            min_Tp=42,
            window_size=96,
            step=2,
            confirm_rounds=2,
            eemd_trials=eemd_trials_default,
        ),
    }

    design_rows = []
    all_results: List[Dict] = []
    all_frames: List[pd.DataFrame] = []
    selected = _select_extended_scenarios(selected_archetypes=selected_archetypes)
    for sc in selected:
        cfg = scenario_settings[sc["archetype"]]
        design_rows.append(
            dict(
                role="cross_pathogen_core",
                archetype=sc["archetype"],
                R=base.SIRSimulator._format_range(sc["R0_range"]),
                GT=base.SIRSimulator._format_range(sc["GT_range"]),
                SI=base.SIRSimulator._format_range(sc["SI_range"]),
                report_delay=base.SIRSimulator._format_range(sc["report_delay_range"]),
                noise=base.SIRSimulator._format_range(sc["noise_std_range"]),
                structural_noise=sc["structural_noise_profile"],
                scenario_weight=float(sc["weight"]),
                sample_n=int(n_per_archetype),
                rationale="cross-pathogen transportability core set",
            )
        )

    design_df = pd.DataFrame(design_rows)
    design_path = OUTPUT_STAGE2 / f"{output_label}_design.csv"
    design_df.to_csv(design_path, index=False, encoding="utf-8-sig")
    protocol_path = OUTPUT_STAGE2 / f"{output_label}_protocol_manifest.csv"
    _export_stage_protocol_manifest(
        protocol_path,
        [
            dict(
                stage="stage2_core_validation",
                data_role="locked_internal_validation",
                data_source="fresh_simulated_curves",
                same_exact_sequences_as_stage1="no",
                same_archetype_family_as_stage1="yes",
                parameter_status="locked_before_validation",
                validation_claim="internal_simulation_validation_only",
                random_seed_base=int(seed),
                n_per_archetype=int(n_per_archetype),
                selected_archetypes=";".join(selected_label),
            )
        ],
    )
    detail_csv = OUTPUT_STAGE2 / f"{output_label}_all_detail.csv"
    summary_csv = OUTPUT_STAGE2 / f"{output_label}_summary.csv"
    focus_csv = OUTPUT_STAGE2 / f"{output_label}_t1_focus_metrics.csv"
    progress_csv = OUTPUT_STAGE2 / f"{output_label}_progress.csv"

    for idx, sc in enumerate(selected):
        arch = sc["archetype"]
        cfg = scenario_settings[arch]
        sim = base.SIRSimulator(seed=seed + idx * 101)
        val_curves = sim.batch_generate_archetype(
            n_curves=n_per_archetype,
            days=int(cfg["days"]),
            add_structured_noise=True,
            min_Tp=int(cfg["min_Tp"]),
            scenario_set=[sc],
        )
        label = f"cross_{arch}_validation"
        base.SIRSimulator.print_truth_distribution(val_curves, label)
        base.SignalDiagnostics.diagnose_signal_complexity(val_curves, label)
        validator = base.RollingWindowValidator(
            ep_opt,
            window_size=int(cfg["window_size"]),
            step=int(cfg["step"]),
            eemd_trials=int(cfg["eemd_trials"]),
            confirm_rounds=int(cfg["confirm_rounds"]),
            confirm_tol=max(3, int(np.ceil(1.0 * ep_opt.GT))),
        )
        val_results, val_df = validator.batch_validate(
            val_curves,
            label=f"{label} ({VERSION_TAG})",
            verbose=True,
        )
        val_df["validation_role"] = "cross_pathogen_core"
        val_df["core_archetype"] = arch
        val_df["base_window_size"] = int(cfg["window_size"])
        val_df["base_step"] = int(cfg["step"])
        val_df["base_confirm_rounds"] = int(cfg["confirm_rounds"])
        val_df["eemd_trials_used"] = int(cfg["eemd_trials"])
        detail_path = OUTPUT_STAGE2 / f"{output_label}_{arch}_detail.csv"
        val_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
        all_results.extend(val_results)
        all_frames.append(val_df)
        core_df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
        summary_df = base._summarize_core_archetype_validation(core_df) if len(core_df) else pd.DataFrame()
        focus_df = summarize_t1_focus_metrics(core_df) if len(core_df) else pd.DataFrame()
        core_df.to_csv(detail_csv, index=False, encoding="utf-8-sig")
        summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        focus_df.to_csv(focus_csv, index=False, encoding="utf-8-sig")
        progress_df = pd.DataFrame(
            {
                "completed_archetype": [item["archetype"] for item in selected[: idx + 1]],
                "completed_count": list(range(1, idx + 2)),
                "total_selected": [len(selected)] * (idx + 1),
            }
        )
        progress_df.to_csv(progress_csv, index=False, encoding="utf-8-sig")
        print(f"[cross-pathogen] Incremental aggregate saved after {arch}")

    core_df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    summary_df = base._summarize_core_archetype_validation(core_df) if len(core_df) else pd.DataFrame()
    core_df.to_csv(detail_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    focus_df = summarize_t1_focus_metrics(core_df)
    focus_df.to_csv(focus_csv, index=False, encoding="utf-8-sig")

    print(f"[cross-pathogen] Validation design:  {design_path}")
    print(f"[cross-pathogen] Validation protocol:{protocol_path}")
    print(f"[cross-pathogen] Validation detail:  {detail_csv}")
    print(f"[cross-pathogen] Validation summary: {summary_csv}")
    print(f"[cross-pathogen] Validation progress: {progress_csv}")
    if len(focus_df):
        print(f"[cross-pathogen] T1 focus metrics:   {focus_csv}")
        print(focus_df.to_string(index=False))
    return all_results, core_df, summary_df


def run_cross_pathogen_prospective_validation(
    ep_opt: base.EpiParams,
    n_t1_tune: int = 12,
    n_t1_validation: int = 40,
    n_prospective_validation: int = 80,
    t1_fast_mode: bool = True,
) -> Tuple[base.EpiParams, Dict[str, object]]:
    print("\n" + "#" * 72)
    print("  CROSS-PATHOGEN STAGE 2B: unified prospective validation on fresh curves")
    print(
        f"  T1 tune/val = {n_t1_tune}/{n_t1_validation}  "
        f"prospective validation = {n_prospective_validation}"
    )
    print("#" * 72)

    protocol_path = OUTPUT_STAGE2 / f"cross_pathogen_prospective_protocol_manifest_{VERSION_TAG}.csv"
    _export_stage_protocol_manifest(
        protocol_path,
        [
            dict(
                stage="stage2b_t1_focused_optimization",
                data_role="t1_runtime_refinement",
                data_source="fresh_simulated_curves",
                same_exact_sequences_as_stage1="no",
                same_archetype_family_as_stage1="yes",
                parameter_status="temporary_t1_runtime_search",
                validation_claim="internal_tuning_only",
                random_seed_base=2026,
                n_curves=int(n_t1_tune),
                fast_mode=("yes" if t1_fast_mode else "no"),
                archetype_library="expanded_cross_pathogen_core",
            ),
            dict(
                stage="stage2b_unified_prospective_validation",
                data_role="locked_internal_validation",
                data_source="fresh_simulated_curves",
                same_exact_sequences_as_stage1="no",
                same_archetype_family_as_stage1="yes",
                parameter_status="locked_before_validation",
                validation_claim="internal_simulation_validation_only",
                random_seed_base=999,
                n_curves=int(n_prospective_validation),
                fast_mode=("yes" if t1_fast_mode else "no"),
                archetype_library="expanded_cross_pathogen_core",
            ),
        ],
    )

    ep_t1, t1_params, t1_val_df = base.run_t1_focused_optimization(
        ep_opt,
        n_tune=n_t1_tune,
        n_val=n_t1_validation,
        fast_mode=t1_fast_mode,
    )

    stage2_window = int(t1_params.get("window_size", 100)) if t1_params else 100
    stage2_step = int(t1_params.get("step", 2)) if t1_params else 2
    stage2_rounds = int(t1_params.get("confirm_rounds", 2)) if t1_params else 2
    val_results, val_df = base.run_stage2_validation(
        ep_t1,
        n_val=n_prospective_validation,
        window_size=stage2_window,
        step=stage2_step,
        confirm_rounds=stage2_rounds,
    )

    outputs: Dict[str, object] = {
        "t1_params": t1_params,
        "t1_validation": t1_val_df,
        "rolling_validation_results": val_results,
        "rolling_validation": val_df,
    }

    if t1_val_df is not None and len(t1_val_df):
        t1_path = OUTPUT_STAGE2 / f"cross_pathogen_t1_focused_validation_{VERSION_TAG}.csv"
        t1_val_df.to_csv(t1_path, index=False, encoding="utf-8-sig")
        print(f"[cross-pathogen] T1-focused validation: {t1_path}")

    if val_df is not None and len(val_df):
        val_df = val_df.copy()
        val_df["validation_role"] = "cross_pathogen_unified_prospective"
        val_df["archetype_library"] = "expanded_cross_pathogen_core"
        val_path = OUTPUT_STAGE2 / f"cross_pathogen_unified_prospective_validation_{VERSION_TAG}.csv"
        val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
        print(f"[cross-pathogen] Unified prospective detail:  {val_path}")

        summary_df = base._summarize_t1_validation(val_df, "cross_pathogen_unified_prospective")
        if summary_df is not None and len(summary_df):
            summary_df = summary_df.copy()
            summary_df["validation_role"] = "cross_pathogen_unified_prospective"
            summary_df["archetype_library"] = "expanded_cross_pathogen_core"
            summary_path = OUTPUT_STAGE2 / f"cross_pathogen_unified_prospective_validation_summary_{VERSION_TAG}.csv"
            summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
            print(f"[cross-pathogen] Unified prospective summary: {summary_path}")
            outputs["rolling_validation_summary"] = summary_df

    return ep_t1, outputs


def run_stage1_development(
    n_dev: int = 120,
    n_stage2_tune: int = 24,
    force_cache: bool = False,
    force_grid: bool = True,
) -> base.EpiParams:
    # This extension frequently changes the archetype library; always refresh the
    # structural cache unless the caller explicitly opts into a different choice.
    force_cache = True if not force_cache else force_cache
    ep_base = base.EpiParams(preset="omicron")
    ep_opt, _dev_curves, _tune_curves, _cache, _grid_df = base.run_stage1_development(
        ep_base,
        n_dev=n_dev,
        n_stage2_tune=n_stage2_tune,
        force_cache=force_cache,
        force_grid=force_grid,
    )
    protocol_path = OUTPUT_STAGE1 / f"stage1_protocol_manifest_{VERSION_TAG}.csv"
    _export_stage_protocol_manifest(
        protocol_path,
        [
            dict(
                stage="stage1_development",
                data_role="parameter_development",
                data_source="simulated_archetype_curves",
                same_exact_sequences_as_stage2="no",
                same_archetype_family_as_stage2="yes",
                parameter_status="optimized_on_development_plus_stage2_tune",
                validation_claim="development_only_not_for_final_claim",
                random_seed_base=42,
                n_curves=int(n_dev),
                scenario_scope="primary_cross_pathogen_archetype_library",
            ),
            dict(
                stage="stage1_rerank",
                data_role="prospective_rerank_tune",
                data_source="fresh_simulated_prospective_curves",
                same_exact_sequences_as_stage2="no",
                same_archetype_family_as_stage2="partly_yes",
                parameter_status="used_for_model_reranking_not_final_validation",
                validation_claim="tuning_only_not_external_validation",
                random_seed_base=42,
                n_curves=int(n_stage2_tune),
                scenario_scope="prospective_generator",
            ),
        ],
    )
    print(f"[cross-pathogen] Stage1 protocol manifest exported: {protocol_path}")
    return ep_opt


def locked_parameter_candidates() -> List[Path]:
    candidates = [
        Path(base.BEST_PARAMS_PATH),
        Path(base.T1_FOCUSED_PARAMS_PATH),
        PROJECT_ROOT / "stage1_fusion_development_v31_1_archetype_adaptive" / "best_params_v31_1_archetype_adaptive.json",
        PROJECT_ROOT / "stage2_fusion_validation_v31_1_archetype_adaptive" / "t1_focused_best_params_v31_1_archetype_adaptive.json",
        PROJECT_ROOT / "submission_package_20260617" / "05_data" / "development" / "best_params_paper1_v31_1_v1.json",
    ]
    ordered: List[Path] = []
    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


def load_locked_params_with_fallback() -> base.EpiParams:
    ep = base.EpiParams(preset="omicron")
    loaded_paths: List[Path] = []
    for path in locked_parameter_candidates():
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                params = json.load(f)
            ep.apply_override(params)
            loaded_paths.append(path)
        except Exception as exc:
            print(f"[cross-pathogen] Failed to load locked parameters {path}: {exc}")
    if loaded_paths:
        print(
            "[cross-pathogen] Loaded locked parameters in order: "
            + " -> ".join(str(path) for path in loaded_paths)
        )
    else:
        print("[cross-pathogen] No locked parameter file found; using default Omicron EpiParams.")
    ep.score_smooth_sigma = max(1.0, float(getattr(ep, "energy_smooth_sigma", 1.5)) * 0.7)
    ep.real_t2_low_share_short = ep.t2_low_share_thresh_short
    ep.real_t2_low_share_medium = ep.t2_low_share_thresh_medium
    ep.real_t2_low_share_long = ep.t2_low_share_thresh_long
    return ep


def load_or_build_params(args: argparse.Namespace) -> base.EpiParams:
    if args.reuse_locked_params:
        ep = load_locked_params_with_fallback()
        ep.print_summary()
        return ep
    return run_stage1_development(
        n_dev=args.n_dev,
        n_stage2_tune=args.n_stage2_tune,
        force_cache=args.force_cache,
        force_grid=args.force_grid,
    )


def _find_first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _normalize_text_label(value: object) -> str:
    text = str(value or "")
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _choose_first_matching_column(columns: List[str], keyword_groups: List[Tuple[str, ...]]) -> Optional[str]:
    normalized = {col: _normalize_text_label(col) for col in columns}
    for keywords in keyword_groups:
        for col, norm in normalized.items():
            if all(keyword in norm for keyword in keywords):
                return col
    return None


def _parse_cn_month_day_series(values: "pd.Series", default_year: int) -> "pd.Series":
    parsed = pd.to_datetime(values, errors="coerce")
    mask = parsed.isna()
    if not mask.any():
        return parsed

    pattern = re.compile(r"^\s*(\d{1,2})鏈?\d{1,2})鏃s*$")
    raw_values = values.astype(str)
    for idx in values.index[mask]:
        text = raw_values.loc[idx]
        match = pattern.match(text)
        if match is None:
            continue
        month = int(match.group(1))
        day = int(match.group(2))
        try:
            parsed.loc[idx] = pd.Timestamp(year=default_year, month=month, day=day)
        except ValueError:
            continue
    return parsed


def _parse_cn_month_day_series(values: "pd.Series", default_year: int) -> "pd.Series":
    parsed = pd.to_datetime(values, errors="coerce")
    mask = parsed.isna()
    if not mask.any():
        return parsed

    # Some exported case workbooks mix proper dates with loose "month-day" text and blanks.
    pattern = re.compile(r"^\s*(\d{1,2})\D+(\d{1,2})\D*$")
    raw_values = values.fillna("").astype(str)
    for idx in values.index[mask]:
        text = raw_values.loc[idx].strip()
        if not text:
            continue
        match = pattern.match(text)
        if match is None:
            continue
        month = int(match.group(1))
        day = int(match.group(2))
        try:
            parsed.loc[idx] = pd.Timestamp(year=default_year, month=month, day=day)
        except ValueError:
            continue
    return parsed


def _build_daily_counts_from_dates(work: pd.DataFrame, date_col: str, count_col: Optional[str] = None,
                                   date_parser=None) -> pd.DataFrame:
    daily = work.copy()
    if date_parser is None:
        daily["date"] = pd.to_datetime(daily[date_col], errors="coerce")
    else:
        daily["date"] = date_parser(daily[date_col])
    if count_col is None:
        daily["cases"] = 1
    else:
        daily["cases"] = pd.to_numeric(daily[count_col], errors="coerce")
    daily = daily[daily["date"].notna() & daily["cases"].notna()].copy()
    if daily.empty:
        raise ValueError("No valid rows remained after date/count filtering.")

    daily["date"] = daily["date"].dt.normalize()
    daily["cases"] = daily["cases"].astype(int)
    daily = daily.groupby("date", as_index=False)["cases"].sum().sort_values("date")

    full_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = (
        daily.set_index("date")
        .reindex(full_dates, fill_value=0)
        .rename_axis("date")
        .reset_index()
    )
    daily["day_index"] = np.arange(len(daily), dtype=int)
    return daily[["day_index", "date", "cases"]]


def _prepend_zero_baseline(daily: pd.DataFrame, days: int) -> pd.DataFrame:
    days = max(0, int(days))
    if days <= 0 or daily.empty:
        out = daily.copy()
        out["day_index"] = np.arange(len(out), dtype=int)
        return out
    first_date = pd.to_datetime(daily["date"]).min()
    baseline_dates = pd.date_range(end=first_date - pd.Timedelta(days=1), periods=days, freq="D")
    baseline = pd.DataFrame({"date": baseline_dates, "cases": np.zeros(days, dtype=int)})
    work = daily.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    combined = pd.concat([baseline, work[["date", "cases"]]], ignore_index=True)
    combined = combined.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    combined["cases"] = pd.to_numeric(combined["cases"], errors="coerce").fillna(0).round().astype(int)
    combined["day_index"] = np.arange(len(combined), dtype=int)
    return combined[["day_index", "date", "cases"]]


def _derive_tp_confidence(result: Dict, ep_d: base.EpiParams) -> Tuple[str, str]:
    tp = int(result["Tp_global"])
    n_full = int(result.get("N_full", result.get("N", 0)))
    gt = max(float(ep_d.GT), 1.0)
    tail_after_peak = max(n_full - 1 - tp, 0)
    tp_low, tp_high = result.get("intervals", {}).get("Tp", (tp, tp))
    if pd.isna(tp_low) or pd.isna(tp_high):
        interval_width = np.nan
    else:
        interval_width = max(int(tp_high) - int(tp_low), 0)

    if tail_after_peak < int(np.ceil(1.0 * gt)):
        return "low", "post_peak_tail_too_short"
    if np.isfinite(interval_width) and interval_width > int(np.ceil(2.0 * gt)):
        return "low", "peak_interval_too_wide"
    if tail_after_peak < int(np.ceil(2.0 * gt)):
        return "medium", "post_peak_tail_borderline"
    if np.isfinite(interval_width) and interval_width > int(np.ceil(1.2 * gt)):
        return "medium", "peak_interval_moderately_wide"
    return "high", "peak_supported"


def _derive_case_scope(result: Dict, ep_d: base.EpiParams) -> Tuple[str, str]:
    n_full = int(result.get("N_full", result.get("N", 0)))
    gt = max(float(ep_d.GT), 1.0)
    tp = int(result["Tp_global"])
    tail_after_peak = max(n_full - 1 - tp, 0)
    if tail_after_peak < int(np.ceil(1.0 * gt)):
        return "exploratory_phase_only", "insufficient_post_peak_tail"
    if n_full < int(np.ceil(6.0 * gt)):
        return "phase_focused_borderline", "short_total_duration"
    return "full_process_candidate", "structure_complete_enough"


def _legacy_broken_read_mpox_sheet_to_daily_series_shadowed(excel_path: Path, sheet_name: str) -> pd.DataFrame:
    return _read_mpox_sheet_to_daily_series(excel_path, sheet_name)


def _build_single_series_summary(result: Dict, disease_name: str, series_label: str,
                                 ep_d: base.EpiParams) -> pd.DataFrame:
    conf = result.get("confidence", {})
    tp_conf, tp_note = _derive_tp_confidence(result, ep_d)
    case_scope, case_scope_note = _derive_case_scope(result, ep_d)
    comparator = base.SignalDiagnostics.energy_threshold_baseline_from_result(
        result, ep_d, search_start=result.get("bg_end", 0), sig_valid_start=result.get("sig_valid_start")
    )
    short_alert_class = base.SignalDiagnostics.classify_short_alert(
        result.get("signal", np.zeros(result.get("N_full", result.get("N", 0)))),
        result["T1"], result["T2"], result["Tp"], onset_day=result.get("sig_valid_start")
    )
    intervention_window = base.SignalDiagnostics.classify_intervention_window(
        result["T1_global"], result["T2_global"], result["Tp_global"], ep_d.GT
    )

    row = {
        "city": series_label,
        "region": series_label,
        "disease": disease_name,
        "GT": ep_d.GT,
        "SI": ep_d.SI,
        "bg_end": result.get("bg_end", 0),
        "bg_mean": round(result.get("bg_mean", 0.0), 3),
        "bg_std": round(result.get("bg_std", 1.0), 3),
        "sig_valid_start": result.get("sig_valid_start", 0),
        "bimodal_takeoff": result.get("bimodal_takeoff"),
        "T1": result["T1_global"],
        "T2": result["T2_global"],
        "T_peak": result["Tp_global"],
        "Tp": result["Tp_global"],
        "T3": result["T3_global"],
        "T1_low": result.get("intervals", {}).get("T1", (np.nan, np.nan))[0],
        "T1_high": result.get("intervals", {}).get("T1", (np.nan, np.nan))[1],
        "T2_low": result.get("intervals", {}).get("T2", (np.nan, np.nan))[0],
        "T2_high": result.get("intervals", {}).get("T2", (np.nan, np.nan))[1],
        "Tp_low": result.get("intervals", {}).get("Tp", (np.nan, np.nan))[0],
        "Tp_high": result.get("intervals", {}).get("Tp", (np.nan, np.nan))[1],
        "pre_T1_days": max(result["T1_global"], 0),
        "alert_days": max(result["T2_global"] - result["T1_global"], 0),
        "rise_days": max(result["Tp_global"] - result["T2_global"], 0),
        "post_peak_decline_days": max(result["T3_global"] - result["Tp_global"], 0),
        "resolution_days": max(result["N_full"] - result["T3_global"], 0),
        "alert_days_GT_norm": round(max(result["T2_global"] - result["T1_global"], 0) / max(ep_d.GT, 1e-10), 3),
        "alert_days_SI_norm": round(max(result["T2_global"] - result["T1_global"], 0) / max(ep_d.SI, 1e-10), 3),
        "T1_strength": round(conf.get("t1_strength", 0.0), 3),
        "T2_strength": round(conf.get("t2_strength", 0.0), 3),
        "T1_raw_z": round(conf.get("t1_raw_z", 0.0), 3),
        "T2_dom_gap": round(conf.get("t2_dom_gap", 0.0), 3),
        "T1_confidence": conf.get("T1", "none"),
        "T2_confidence": conf.get("T2", "none"),
        "Tp_confidence": tp_conf,
        "T3_confidence": conf.get("T3", "none"),
        "Tp_confidence_note": tp_note,
        "Tp_tail_after_peak_days": max(result["N_full"] - 1 - result["Tp_global"], 0),
        "Tp_interval_width": (
            result.get("intervals", {}).get("Tp", (np.nan, np.nan))[1]
            - result.get("intervals", {}).get("Tp", (np.nan, np.nan))[0]
            if pd.notna(result.get("intervals", {}).get("Tp", (np.nan, np.nan))[0])
            and pd.notna(result.get("intervals", {}).get("Tp", (np.nan, np.nan))[1])
            else np.nan
        ),
        "T1_to_T2_days": max(result["T2_global"] - result["T1_global"], 0),
        "T1_to_peak_days": max(result["Tp_global"] - result["T1_global"], 0),
        "T1_lead_peak": max(result["Tp_global"] - result["T1_global"], 0),
        "T1_minus_sig_valid_start": int(result["T1"] - result.get("sig_valid_start", 0)),
        "short_alert_class": short_alert_class,
        "intervention_window": intervention_window,
        "case_scope": case_scope,
        "case_scope_note": case_scope_note,
        "confidence": conf.get("overall", "none"),
        "method_structural": VERSION_TAG,
        "T1_energy": comparator.get("T1_energy"),
        "T2_energy": comparator.get("T2_energy"),
        "Tp_energy": comparator.get("Tp_energy"),
        "alert_days_energy": comparator.get("alert_days_energy"),
        "T1_energy_delta_vs_struct": comparator.get("T1_energy_delta_vs_struct"),
        "T2_energy_delta_vs_struct": comparator.get("T2_energy_delta_vs_struct"),
        "Tp_energy_delta_vs_struct": comparator.get("Tp_energy_delta_vs_struct"),
    }
    return pd.DataFrame([row])


def _legacy_run_mpox_external_application_shadowed(ep_opt: base.EpiParams) -> Dict[str, pd.DataFrame]:
    mpox_candidates = [
        CROSS_PATHOGEN_DATA_DIR / "猴痘数据.xlsx",
        CROSS_PATHOGEN_DATA_DIR / "mpox_data_tmp.xlsx",
    ]
    excel_path = _find_first_existing(mpox_candidates)
    if excel_path is None:
        raise FileNotFoundError(
            "Mpox source file not found. Expected one of: "
            + ", ".join(str(p) for p in mpox_candidates)
        )

    print("\n" + "#" * 72)
    print("  CROSS-PATHOGEN STAGE 3: mpox external application")
    print(f"  Source workbook: {excel_path}")
    print("#" * 72)

    cases = [
        dict(sheet="23年猴痘发病日期", label="Mpox 2023", preset="mpox", fallback="chikungunya"),
        dict(sheet="26年猴痘发病日期", label="Mpox 2026", preset="mpox", fallback="chikungunya"),
    ]

    outputs: Dict[str, pd.DataFrame] = {}
    summary_frames: List[pd.DataFrame] = []
    method_frames: List[pd.DataFrame] = []

    for case in cases:
        daily = _read_mpox_sheet_to_daily_series(excel_path, case["sheet"])
        signal = daily["cases"].to_numpy(dtype=float)

        try:
            ep_d = base.EpiParams(preset=case["preset"])
        except Exception:
            ep_d = base.EpiParams(preset=case["fallback"])

        ep_d.apply_override({
            "energy_smooth_sigma": ep_opt.energy_smooth_sigma,
            "score_smooth_sigma": ep_opt.score_smooth_sigma,
            "t1_structure_score_thresh": ep_opt.t1_structure_score_thresh,
            "t1_mid_z_thresh": ep_opt.t1_mid_z_thresh,
            "t1_mid_high_ratio_z": max(0.50, ep_opt.t1_mid_high_ratio_z),
            "t1_centroid_z_thresh": ep_opt.t1_centroid_z_thresh,
            "t1_relaxed_score_thresh": ep_opt.t1_relaxed_score_thresh,
            "t1_valid_strength_min": ep_opt.t1_valid_strength_min,
            "t1_confirm_ewma_k": ep_opt.t1_confirm_ewma_k,
            "t1_fast_confirm_strength": ep_opt.t1_fast_confirm_strength,
            "t1_single_confirm_strength": ep_opt.t1_single_confirm_strength,
            "t1_fast_local_margin": ep_opt.t1_fast_local_margin,
            "t1_confirm_span_gt": ep_opt.t1_confirm_span_gt,
            "t2_dom_score_thresh": ep_opt.t2_dom_score_thresh,
            "t2_low_share_thresh_short": ep_opt.t2_low_share_thresh_short,
            "t2_low_share_thresh_medium": ep_opt.t2_low_share_thresh_medium,
            "t2_low_share_thresh_long": ep_opt.t2_low_share_thresh_long,
        })
        ep_d.apply_override(base._t1_runtime_overrides(ep_opt))

        detector = base.RealDataStructuralDetector(ep_d, eemd_trials=120, stability_runs=0)
        result = detector.segment_realdata(signal, city_name=case["label"], global_offset=0, do_ensemble=False)

        loader_like = SimpleNamespace(dates=list(pd.to_datetime(daily["date"]).dt.to_pydatetime()))
        prefix = case["label"].lower().replace(" ", "_")
        detail_df = daily.copy()
        detail_df["disease"] = case["label"]
        detail_df["T1"] = result["T1_global"]
        detail_df["T2"] = result["T2_global"]
        detail_df["Tp"] = result["Tp_global"]
        detail_df["T3"] = result["T3_global"]

        summary_df = _build_single_series_summary(result, case["label"], case["label"], ep_d)
        method_df = base._build_method_comparison_table(summary_df, source_label=case["label"], gt=ep_d.GT)

        detail_path = OUTPUT_STAGE3 / f"{prefix}_daily_series_{VERSION_TAG}.csv"
        summary_path = OUTPUT_STAGE3 / f"{prefix}_phase_seg_{VERSION_TAG}.csv"
        method_path = OUTPUT_STAGE3 / f"{prefix}_method_comparison_{VERSION_TAG}.csv"
        figure_path = OUTPUT_STAGE3 / f"{prefix}_segmentation_{VERSION_TAG}.png"

        detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        method_df.to_csv(method_path, index=False, encoding="utf-8-sig")
        base.Visualizer.plot_realdata_segmentation(result, loader_like, save_path=str(figure_path), title=case["label"])

        print(
            f"[mpox] {case['label']}: N={len(signal)}  total_cases={int(signal.sum())}  "
            f"T1={result['T1_global']}  T2={result['T2_global']}  Tp={result['Tp_global']}  "
            f"T3={result['T3_global']}  conf={result['confidence'].get('overall', 'none')}"
        )

        summary_frames.append(summary_df)
        method_frames.append(method_df)
        outputs[f"{prefix}_daily"] = detail_df
        outputs[f"{prefix}_summary"] = summary_df
        outputs[f"{prefix}_method"] = method_df

    combined_summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    combined_method = pd.concat(method_frames, ignore_index=True) if method_frames else pd.DataFrame()
    if len(combined_summary):
        combined_summary.to_csv(
            OUTPUT_STAGE3 / f"mpox_external_application_summary_{VERSION_TAG}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        outputs["mpox_external_application_summary"] = combined_summary
    if len(combined_method):
        combined_method.to_csv(
            OUTPUT_STAGE3 / f"mpox_external_application_method_long_{VERSION_TAG}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        outputs["mpox_external_application_method_long"] = combined_method
    return outputs


def _read_mpox_sheet_to_daily_series(excel_path: Path, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(excel_path, sheet_name=sheet_name)
    work = raw.copy()
    work.columns = [str(col).strip() for col in work.columns]

    normalized_map = {col: _normalize_text_label(col) for col in work.columns}

    def _find_column(keyword_groups: list[tuple[str, ...]]) -> Optional[str]:
        for keywords in keyword_groups:
            for original, normalized in normalized_map.items():
                if all(keyword in normalized for keyword in keywords):
                    return original
        return None

    date_col = _find_column([
        ("onset", "date"),
        ("report", "date"),
        ("date",),
        ("riqi",),
        ("shijian",),
        ("fabing",),
    ])
    count_col = _find_column([
        ("case", "count"),
        ("cases",),
        ("count",),
        ("shuliang",),
        ("bingli",),
        ("geshu",),
    ])

    if date_col is None:
        for candidate in work.columns:
            parsed = pd.to_datetime(work[candidate], errors="coerce")
            if parsed.notna().sum() >= max(3, len(work) // 5):
                date_col = candidate
                break

    if date_col is None:
        raise ValueError(f"Sheet {sheet_name} is missing a readable date column: {list(work.columns)}")

    return _build_daily_counts_from_dates(work, date_col=date_col, count_col=count_col)


def _read_chikungunya_workbook_to_daily_series(excel_path: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    try:
        workbook = pd.ExcelFile(excel_path)
    except Exception as exc:
        raise ValueError(
            "Chikungunya workbook could not be opened by pandas. "
            "It is likely still encrypted or saved in a legacy/non-standard Excel container. "
            "Please open it with password 0595 and save/export a normal .xlsx copy, "
            "then place it in the same folder as `chikungunya_data_decrypted.xlsx` or `chikungunya_data_tmp.xlsx`."
        ) from exc

    available_sheets = workbook.sheet_names
    target_sheet = sheet_name
    if target_sheet is None:
        normalized_map = {name: _normalize_text_label(name) for name in available_sheets}
        preferred_groups = [
            ("病例", "一览"),
            ("鐥呬緥",),
            ("鎶ュ憡",),
            ("鏄庣粏",),
            ("sheet1",),
        ]
        for keywords in preferred_groups:
            for candidate, normalized in normalized_map.items():
                if all(keyword in normalized for keyword in keywords):
                    target_sheet = candidate
                    break
            if target_sheet is not None:
                break
        if target_sheet is None and available_sheets:
            target_sheet = available_sheets[0]

    if target_sheet is None:
        raise ValueError(f"Chikungunya workbook {excel_path} does not contain any readable sheets.")

    raw = pd.read_excel(excel_path, sheet_name=target_sheet)
    work = raw.copy()
    work.columns = [str(col).strip() for col in work.columns]

    date_col = _choose_first_matching_column(
        list(work.columns),
        [
            ("鏃ユ湡",),
            ("鍙戠梾", "鏃ユ湡"),
            ("鍙戠梾",),
            ("灏辫瘖", "鏃ユ湡"),
            ("鎶ュ憡", "鏃ユ湡"),
            ("璇婃柇", "鏃ユ湡"),
            ("onset", "date"),
            ("illness", "onset"),
            ("date",),
        ],
    )
    count_col = _choose_first_matching_column(
        list(work.columns),
        [
            ("病例数",),
            ("病例", "数"),
            ("鏁伴噺",),
            ("count",),
            ("cases",),
        ],
    )
    if date_col is None:
        raise ValueError(
            f"Chikungunya workbook sheet {target_sheet} is missing a readable date column: {list(work.columns)}"
        )
    return _build_daily_counts_from_dates(
        work,
        date_col=date_col,
        count_col=count_col,
        date_parser=lambda series: _parse_cn_month_day_series(series, default_year=2025),
    )


def _read_influenza_summary_workbook_to_daily_series(excel_path: Path,
                                                     sheet_name: str = "省市日新增") -> pd.DataFrame:
    raw = pd.read_excel(excel_path, sheet_name=sheet_name)
    work = raw.copy()
    work.columns = [str(col).strip() for col in work.columns]

    province_col = "省/直辖市" if "省/直辖市" in work.columns else _choose_first_matching_column(
        list(work.columns),
        [("省",), ("province",)],
    )
    city_col = "市" if "市" in work.columns else None
    if province_col is None:
        raise ValueError(f"Influenza workbook sheet {sheet_name} is missing a readable province column: {list(work.columns)}")

    date_columns: List[str] = []
    for col in work.columns:
        if col in {province_col, city_col}:
            continue
        if pd.notna(pd.to_datetime(col, errors="coerce")):
            date_columns.append(col)
    if not date_columns:
        raise ValueError(f"Influenza workbook sheet {sheet_name} has no readable date columns.")

    # Keep province-level totals only to avoid double counting province + city rows.
    if city_col is not None and city_col in work.columns:
        city_text = work[city_col].fillna("").astype(str).str.strip()
        province_only = work[city_text == ""]
    else:
        province_only = work.copy()
    if province_only.empty:
        raise ValueError("Influenza workbook does not contain any province-level rows after filtering.")

    numeric_block = province_only[date_columns].apply(pd.to_numeric, errors="coerce").fillna(0).clip(lower=0)
    daily_cases = numeric_block.sum(axis=0)
    daily = pd.DataFrame({
        "date": pd.to_datetime(daily_cases.index, errors="coerce"),
        "cases": daily_cases.to_numpy(dtype=float),
    })
    daily = daily[daily["date"].notna()].sort_values("date").copy()
    if daily.empty:
        raise ValueError("Influenza workbook has no valid dated rows after parsing.")
    daily["date"] = daily["date"].dt.normalize()
    daily["cases"] = daily["cases"].round().astype(int)
    daily["day_index"] = np.arange(len(daily), dtype=int)
    return daily[["day_index", "date", "cases"]]


def _parse_date_columns(columns: List[object]) -> List[object]:
    return [col for col in columns if pd.notna(pd.to_datetime(col, errors="coerce"))]


def _make_unique_labels(labels: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    unique: List[str] = []
    for raw in labels:
        label = str(raw).strip() or "unnamed_region"
        count = seen.get(label, 0)
        seen[label] = count + 1
        unique.append(label if count == 0 else f"{label}_{count + 1}")
    return unique


def _write_realdata_matrix(selected: pd.DataFrame,
                           date_cols: List[object],
                           labels: List[str],
                           out_path: Path) -> pd.DataFrame:
    values = selected[date_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
    matrix = values.T
    matrix.index = pd.to_datetime([str(col) for col in date_cols], errors="coerce")
    matrix = matrix.loc[pd.notna(matrix.index)].copy()
    matrix.columns = _make_unique_labels(labels)
    matrix.to_excel(out_path)
    return matrix


def _build_daily_series_from_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    daily_cases = matrix.sum(axis=1)
    daily = pd.DataFrame({
        "date": pd.to_datetime(daily_cases.index, errors="coerce"),
        "cases": pd.to_numeric(daily_cases, errors="coerce").fillna(0.0).to_numpy(dtype=float),
    })
    daily = daily[daily["date"].notna()].sort_values("date").copy()
    if daily.empty:
        raise ValueError("Influenza matrix has no valid dated rows after aggregation.")
    daily["date"] = daily["date"].dt.normalize()
    daily["cases"] = daily["cases"].round().astype(int)
    daily["day_index"] = np.arange(len(daily), dtype=int)
    return daily[["day_index", "date", "cases"]]


def _prepare_influenza_province_city_workbook(excel_path: Path,
                                              sheet_name: Optional[str] = None) -> Dict[str, object]:
    source = pd.read_excel(excel_path, sheet_name=sheet_name or 0)
    source.columns = [str(col).strip() for col in source.columns]

    province_col = "省/直辖市" if "省/直辖市" in source.columns else _choose_first_matching_column(
        list(source.columns),
        [("province",), ("sheng",), ("region",)],
    )
    city_col = "市" if "市" in source.columns else _choose_first_matching_column(
        list(source.columns),
        [("city",), ("shi",), ("unit",)],
    )
    level_col = "城市等级" if "城市等级" in source.columns else _choose_first_matching_column(
        list(source.columns),
        [("level",), ("cengji",)],
    )
    if province_col is None or city_col is None:
        raise ValueError(
            f"Influenza workbook sheet is missing readable province/city columns: {list(source.columns)}"
        )

    date_cols = _parse_date_columns(list(source.columns))
    if not date_cols:
        raise ValueError("Influenza workbook has no readable date columns.")
    window_start = pd.Timestamp(INFLUENZA_TARGET_WINDOW_START)
    window_end = pd.Timestamp(INFLUENZA_TARGET_WINDOW_END)
    filtered_date_cols = [
        col for col in date_cols
        if window_start <= pd.to_datetime(str(col), errors="coerce") <= window_end
    ]
    if not filtered_date_cols:
        raise ValueError(
            "Influenza workbook has no readable date columns inside the target window "
            f"{INFLUENZA_TARGET_WINDOW_START} to {INFLUENZA_TARGET_WINDOW_END}."
        )
    date_cols = filtered_date_cols

    work = source.copy()
    work[province_col] = work[province_col].astype(str).str.strip()
    city_values = work[city_col].astype("string").fillna("").str.strip()
    is_city_row = city_values != ""
    direct_municipalities = {"北京", "上海", "天津", "重庆"}
    is_direct_municipality = (~is_city_row) & work[province_col].isin(direct_municipalities)
    city_like = work.loc[is_city_row | is_direct_municipality].copy()
    province_rows = work.loc[~is_city_row].copy()
    if city_like.empty or province_rows.empty:
        raise ValueError("Influenza workbook could not be separated into city-like and province-level matrices.")

    city_names = city_like[city_col].astype("string").fillna("").str.strip()
    city_labels = []
    for province, city in zip(city_like[province_col].astype(str), city_names):
        city_labels.append(province if not city else f"{province}_{city}")
    province_labels = province_rows[province_col].astype(str).str.strip().tolist()

    values = city_like[date_cols].apply(pd.to_numeric, errors="coerce")
    qc = city_like[[province_col, city_col] + ([level_col] if level_col else [])].copy()
    qc["analysis_label"] = city_labels
    qc["total_cases"] = values.sum(axis=1, skipna=True).astype(float)
    qc["missing_rate"] = values.isna().mean(axis=1).astype(float)
    qc["zero_day_rate"] = values.fillna(0).eq(0).mean(axis=1).astype(float)

    prep_dir = OUTPUT_STAGE3 / "prepared_real_input"
    prep_dir.mkdir(parents=True, exist_ok=True)
    city_matrix_path = prep_dir / f"influenza_2025_2026_city_direct_municipality_matrix_{VERSION_TAG}.xlsx"
    province_matrix_path = prep_dir / f"influenza_2025_2026_province_matrix_{VERSION_TAG}.xlsx"
    qc_path = prep_dir / f"influenza_2025_2026_city_qc_{VERSION_TAG}.csv"

    city_matrix = _write_realdata_matrix(city_like, date_cols, city_labels, city_matrix_path)
    province_matrix = _write_realdata_matrix(province_rows, date_cols, province_labels, province_matrix_path)
    qc.to_csv(qc_path, index=False, encoding="utf-8-sig")

    return {
        "city_matrix_path": city_matrix_path,
        "province_matrix_path": province_matrix_path,
        "city_matrix": city_matrix,
        "province_matrix": province_matrix,
        "qc_path": qc_path,
        "city_qc": qc,
        "start_date": city_matrix.index.min().date(),
        "end_date": city_matrix.index.max().date(),
    }


def _summarize_real_application_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    work = df.copy()
    if "confidence" in work.columns:
        work = work[work["confidence"].isin(["high", "medium", "low"])].copy()
    if work.empty:
        return pd.DataFrame([{"dataset": label, "n_units": 0}])

    row: Dict[str, object] = {"dataset": label, "n_units": int(len(work))}
    for col in ["T1", "T2", "T_peak", "Tp", "alert_days", "T1_to_peak_days", "T1_raw_z", "T2_dom_gap"]:
        if col in work.columns:
            vals = pd.to_numeric(work[col], errors="coerce").dropna()
            if len(vals):
                row[f"{col}_median"] = float(vals.median())
                row[f"{col}_iqr_low"] = float(vals.quantile(0.25))
                row[f"{col}_iqr_high"] = float(vals.quantile(0.75))
    if "confidence" in work.columns:
        conf = work["confidence"].dropna().astype(str)
        row["high_medium_confidence_rate"] = float(conf.isin(["high", "medium"]).mean()) if len(conf) else np.nan
    return pd.DataFrame([row])


def _run_realdata_matrix_application(ep_opt: base.EpiParams,
                                     matrix_path: Path,
                                     start_date,
                                     label: str,
                                     disease_name: str,
                                     preset: str,
                                     fallback: str,
                                     plot_examples: int = 3) -> Dict[str, pd.DataFrame]:
    try:
        ep_real = base.EpiParams(preset=preset)
    except Exception:
        ep_real = base.EpiParams(preset=fallback)
    _apply_runtime_overrides_for_realdata(ep_real, ep_opt)

    loader = base.RealDataLoader(str(matrix_path), start_date, label)
    detector = base.RealDataStructuralDetector(ep_real, eemd_trials=120, stability_runs=0)
    results, city_df = detector.batch_segment_realdata(loader, verbose=True)
    national_result = detector.segment_realdata(
        loader.national,
        city_name=f"{label}_national",
        global_offset=0,
        do_ensemble=False,
    )

    out_label = label.lower().replace(" ", "_")
    city_df = city_df.copy()
    city_df["disease"] = disease_name
    city_df["source_group"] = label
    city_df.to_csv(OUTPUT_STAGE3 / f"{out_label}_phase_segmentation.csv", index=False, encoding="utf-8-sig")
    city_df.to_excel(OUTPUT_STAGE3 / f"{out_label}_phase_segmentation.xlsx", index=False)

    overview_df = _summarize_real_application_table(city_df, label)
    overview_df["disease"] = disease_name
    overview_df.to_csv(OUTPUT_STAGE3 / f"{out_label}_real_application_summary.csv", index=False, encoding="utf-8-sig")

    base.Visualizer.plot_realdata_segmentation(
        national_result,
        loader,
        save_path=str(OUTPUT_STAGE3 / f"{out_label}_national_phase_segmentation.png"),
        title=f"{label} national",
    )

    valid_results = [r for r in results if r.get("confidence", {}).get("overall") != "none"]
    for idx, result in enumerate(valid_results[: max(0, plot_examples)]):
        safe_name = base.Visualizer._safe_filename(result["city"])
        base.Visualizer.plot_realdata_segmentation(
            result,
            loader,
            save_path=str(OUTPUT_STAGE3 / f"{out_label}_{idx + 1:02d}_{safe_name}.png"),
            title=f"{label} - {result['city']}",
        )

    method_df = base._build_method_comparison_table(city_df, source_label=label, gt=ep_real.GT)
    method_df.to_csv(
        OUTPUT_STAGE3 / f"{out_label}_method_comparison_{VERSION_TAG}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return {
        f"{out_label}_phase_segmentation": city_df,
        f"{out_label}_summary": city_df,
        f"{out_label}_overview": overview_df,
        f"{out_label}_method": method_df,
    }


def _apply_runtime_overrides_for_realdata(ep_d: base.EpiParams, ep_opt: base.EpiParams) -> None:
    ep_d.apply_override({
        "energy_smooth_sigma": ep_opt.energy_smooth_sigma,
        "score_smooth_sigma": ep_opt.score_smooth_sigma,
        "t1_structure_score_thresh": ep_opt.t1_structure_score_thresh,
        "t1_mid_z_thresh": ep_opt.t1_mid_z_thresh,
        "t1_mid_high_ratio_z": max(0.50, ep_opt.t1_mid_high_ratio_z),
        "t1_centroid_z_thresh": ep_opt.t1_centroid_z_thresh,
        "t1_relaxed_score_thresh": ep_opt.t1_relaxed_score_thresh,
        "t1_valid_strength_min": ep_opt.t1_valid_strength_min,
        "t1_confirm_ewma_k": ep_opt.t1_confirm_ewma_k,
        "t1_fast_confirm_strength": ep_opt.t1_fast_confirm_strength,
        "t1_single_confirm_strength": ep_opt.t1_single_confirm_strength,
        "t1_fast_local_margin": ep_opt.t1_fast_local_margin,
        "t1_confirm_span_gt": ep_opt.t1_confirm_span_gt,
        "t2_dom_score_thresh": ep_opt.t2_dom_score_thresh,
        "t2_low_share_thresh_short": ep_opt.t2_low_share_thresh_short,
        "t2_low_share_thresh_medium": ep_opt.t2_low_share_thresh_medium,
        "t2_low_share_thresh_long": ep_opt.t2_low_share_thresh_long,
    })
    ep_d.apply_override(base._t1_runtime_overrides(ep_opt))
    if str(getattr(ep_d, "preset", "")).lower() == "influenza_seasonal":
        ep_d.apply_override({
            "t1_baseline_days": max(int(ep_d.t1_baseline_days), int(np.ceil(5.4 * ep_d.GT))),
            "t1_structure_score_thresh": min(float(ep_d.t1_structure_score_thresh), 0.50),
            "t1_mid_z_thresh": min(float(ep_d.t1_mid_z_thresh), 0.94),
            "t1_relaxed_score_thresh": min(float(ep_d.t1_relaxed_score_thresh), 0.35),
            "t1_relaxed_mid_z_thresh": min(float(ep_d.t1_relaxed_mid_z_thresh), 0.58),
            "t1_prealert_mid_z_thresh": min(float(ep_d.t1_prealert_mid_z_thresh), 0.34),
            "t2_dom_score_thresh": min(float(ep_d.t2_dom_score_thresh), 0.44),
            "t1_valid_strength_min": min(float(getattr(ep_d, "t1_valid_strength_min", 0.24)), 0.18),
            "t1_first_alert_strength": min(float(getattr(ep_d, "t1_first_alert_strength", 0.24)), 0.14),
            "t1_first_alert_local_margin": min(float(getattr(ep_d, "t1_first_alert_local_margin", -0.08)), -0.14),
            "t1_first_alert_min_persist_gt": min(float(getattr(ep_d, "t1_first_alert_min_persist_gt", 0.20)), 0.0),
            "t1_early_confirm_strength": min(float(getattr(ep_d, "t1_early_confirm_strength", 0.30)), 0.22),
            "t1_early_confirm_min_persist_gt": min(float(getattr(ep_d, "t1_early_confirm_min_persist_gt", 0.35)), 0.18),
            "t1_single_confirm_strength": min(float(getattr(ep_d, "t1_single_confirm_strength", 0.56)), 0.46),
            "t1_confirm_single_min_persist_gt": min(float(getattr(ep_d, "t1_confirm_single_min_persist_gt", 0.35)), 0.18),
            "t1_confirm_allow_same_window_as_first_alert": True,
            "t1_estimate_backtrack_gt": min(float(getattr(ep_d, "t1_estimate_backtrack_gt", 0.75)), 0.55),
            "t1_estimate_max_backtrack_gt": min(float(getattr(ep_d, "t1_estimate_max_backtrack_gt", 0.75)), 0.65),
            "stage3_rolling_window_floor": min(int(getattr(ep_d, "stage3_rolling_window_floor", 42)), 34),
            "stage3_rolling_window_cap": min(int(getattr(ep_d, "stage3_rolling_window_cap", 65)), 52),
            "stage3_rolling_window_frac": min(float(getattr(ep_d, "stage3_rolling_window_frac", 0.45)), 0.30),
            "stage3_rolling_confirm_rounds": min(int(getattr(ep_d, "stage3_rolling_confirm_rounds", 1)), 1),
            "stage3_rolling_step": 1,
        })


def _apply_seasonal_influenza_stage3_overrides(ep_roll: base.EpiParams) -> None:
    ep_roll.apply_override({
        "prospective_min_window_gt": min(float(getattr(ep_roll, "prospective_min_window_gt", 2.5)), 1.55),
        "prospective_min_window_floor": min(int(getattr(ep_roll, "prospective_min_window_floor", 10)), 6),
        "prospective_t2_confirm_rounds": 1,
        "prospective_t2_confirm_tol_gt": min(float(getattr(ep_roll, "prospective_t2_confirm_tol_gt", 1.25)), 0.95),
        "prospective_t2_strength_min": min(float(getattr(ep_roll, "prospective_t2_strength_min", 0.36)), 0.30),
        "prospective_t2_tail_quantile": min(float(getattr(ep_roll, "prospective_t2_tail_quantile", 0.20)), 0.12),
        "prospective_tp_confirm_rounds": 1,
        "prospective_tp_confirm_tol_gt": min(float(getattr(ep_roll, "prospective_tp_confirm_tol_gt", 2.0)), 1.25),
        "prospective_tp_maturity_gt": min(float(getattr(ep_roll, "prospective_tp_maturity_gt", 1.25)), 0.95),
        "prospective_tp_decline_frac": min(float(getattr(ep_roll, "prospective_tp_decline_frac", 0.05)), 0.03),
        "t1_first_alert_strength": min(float(getattr(ep_roll, "t1_first_alert_strength", 0.16)), 0.12),
        "t1_first_alert_local_margin": min(float(getattr(ep_roll, "t1_first_alert_local_margin", -0.12)), -0.16),
        "t1_first_alert_min_persist_gt": min(float(getattr(ep_roll, "t1_first_alert_min_persist_gt", 0.0)), 0.0),
        "t1_raw_growth_z": min(float(getattr(ep_roll, "t1_raw_growth_z", 1.9)), 1.7),
        "t1_raw_growth_min_persist_gt": min(float(getattr(ep_roll, "t1_raw_growth_min_persist_gt", 0.45)), 0.25),
        "t1_raw_growth_confirm_strength": min(float(getattr(ep_roll, "t1_raw_growth_confirm_strength", 0.58)), 0.50),
        "t1_confirm_allow_same_window_as_first_alert": True,
        "t1_confirm_span_gt": min(float(getattr(ep_roll, "t1_confirm_span_gt", 0.75)), 0.55),
    })


def _run_single_real_series_case(ep_opt: base.EpiParams, daily: pd.DataFrame, label: str,
                                 disease_name: str, preset: str, fallback: str) -> Dict[str, pd.DataFrame]:
    signal = daily["cases"].to_numpy(dtype=float)
    try:
        ep_d = base.EpiParams(preset=preset)
    except Exception:
        ep_d = base.EpiParams(preset=fallback)
    _apply_runtime_overrides_for_realdata(ep_d, ep_opt)

    detector = base.RealDataStructuralDetector(ep_d, eemd_trials=120, stability_runs=0)
    result = detector.segment_realdata(signal, city_name=label, global_offset=0, do_ensemble=False)

    loader_like = SimpleNamespace(dates=list(pd.to_datetime(daily["date"]).dt.to_pydatetime()))
    prefix = label.lower().replace(" ", "_")
    detail_df = daily.copy()
    detail_df["disease"] = disease_name
    detail_df["series_label"] = label
    detail_df["T1"] = result["T1_global"]
    detail_df["T2"] = result["T2_global"]
    detail_df["Tp"] = result["Tp_global"]
    detail_df["T3"] = result["T3_global"]

    summary_df = _build_single_series_summary(result, disease_name, label, ep_d)
    method_df = base._build_method_comparison_table(summary_df, source_label=label, gt=ep_d.GT)

    detail_path = OUTPUT_STAGE3 / f"{prefix}_daily_series_{VERSION_TAG}.csv"
    summary_path = OUTPUT_STAGE3 / f"{prefix}_phase_seg_{VERSION_TAG}.csv"
    method_path = OUTPUT_STAGE3 / f"{prefix}_method_comparison_{VERSION_TAG}.csv"
    figure_path = OUTPUT_STAGE3 / f"{prefix}_segmentation_{VERSION_TAG}.png"

    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    method_df.to_csv(method_path, index=False, encoding="utf-8-sig")
    base.Visualizer.plot_realdata_segmentation(result, loader_like, save_path=str(figure_path), title=label)

    print(
        f"[{disease_name}] {label}: N={len(signal)}  total_cases={int(signal.sum())}  "
        f"T1={result['T1_global']}  T2={result['T2_global']}  Tp={result['Tp_global']}  "
        f"T3={result['T3_global']}  conf={result['confidence'].get('overall', 'none')}"
    )
    return {
        "daily": detail_df,
        "summary": summary_df,
        "method": method_df,
    }


def _build_single_series_rolling_reference(ep_opt: base.EpiParams, daily: pd.DataFrame, summary_df: pd.DataFrame,
                                           disease_name: str, city_label: str, preset: str,
                                           fallback: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        ep_roll = base.EpiParams(preset=preset)
    except Exception:
        ep_roll = base.EpiParams(preset=fallback)
    _apply_runtime_overrides_for_realdata(ep_roll, ep_opt)
    ep_roll.apply_override(base._stage3_rolling_overrides(ep_roll))
    if str(getattr(ep_roll, "preset", "")).lower() == "influenza_seasonal":
        _apply_seasonal_influenza_stage3_overrides(ep_roll)

    signal = daily["cases"].to_numpy(dtype=float)
    ref = summary_df.iloc[0]
    rolling_step = max(1, int(getattr(ep_roll, "stage3_rolling_step", 2)))
    real_rolling_window = min(
        int(getattr(ep_roll, "stage3_rolling_window_cap", 65)),
        max(
            int(getattr(ep_roll, "stage3_rolling_window_floor", 42)),
            int(np.ceil(len(signal) * float(getattr(ep_roll, "stage3_rolling_window_frac", 0.45))))
        )
    )
    stage3_rounds = int(getattr(ep_roll, "stage3_rolling_confirm_rounds", 1))
    row = base._stage3_rolling_reference_row(
        signal,
        ep_roll,
        disease=disease_name,
        city=city_label,
        T1_ref=int(ref["T1"]),
        T2_ref=int(ref["T2"]),
        Tp_ref=int(ref["Tp"]),
        curve_id=city_label,
        window_size=real_rolling_window,
        step=rolling_step,
        eemd_trials=8,
        confirm_rounds=stage3_rounds,
        sig_valid_start=ref.get("sig_valid_start"),
        bg_end=ref.get("bg_end"),
        T1_low=ref.get("T1_low"),
        T1_high=ref.get("T1_high"),
    )
    rolling_df = pd.DataFrame([row])
    rolling_summary = base._summarize_stage3_rolling_reference(rolling_df)
    return rolling_df, rolling_summary


def _write_combined_real_application_outputs(prefix: str, outputs: Dict[str, pd.DataFrame]) -> None:
    summary_frames = [frame for key, frame in outputs.items() if key.endswith("_summary")]
    method_frames = [frame for key, frame in outputs.items() if key.endswith("_method")]
    combined_summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    combined_method = pd.concat(method_frames, ignore_index=True) if method_frames else pd.DataFrame()
    if len(combined_summary):
        combined_summary.to_csv(
            OUTPUT_STAGE3 / f"{prefix}_external_application_summary_{VERSION_TAG}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        outputs[f"{prefix}_external_application_summary"] = combined_summary
    if len(combined_method):
        combined_method.to_csv(
            OUTPUT_STAGE3 / f"{prefix}_external_application_method_long_{VERSION_TAG}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        outputs[f"{prefix}_external_application_method_long"] = combined_method


def run_mpox_external_application(ep_opt: base.EpiParams) -> Dict[str, pd.DataFrame]:
    mpox_dir = CROSS_PATHOGEN_DATA_DIR
    mpox_candidates = [
        mpox_dir / "猴痘数据.xlsx",
        mpox_dir / "mpox_data_tmp.xlsx",
    ]
    excel_path = _find_first_existing(mpox_candidates)
    if excel_path is None:
        raise FileNotFoundError(
            "Mpox source file not found. Expected one of: "
            + ", ".join(str(p) for p in mpox_candidates)
        )

    workbook = pd.ExcelFile(excel_path)
    cases = [
        dict(sheet="23年猴痘发病日期", label="Mpox 2023", preset="mpox", fallback="chikungunya"),
        dict(sheet="26年猴痘发病日期", label="Mpox 2026", preset="mpox", fallback="chikungunya"),
    ]
    missing = [case["sheet"] for case in cases if case["sheet"] not in workbook.sheet_names]
    if missing:
        raise ValueError(f"Mpox workbook sheets not found: {missing}; available sheets={workbook.sheet_names}")

    print("\n" + "#" * 72)
    print("  CROSS-PATHOGEN STAGE 3: mpox external application")
    print(f"  Source workbook: {excel_path}")
    print("#" * 72)

    outputs: Dict[str, pd.DataFrame] = {}
    for case in cases:
        daily = _read_mpox_sheet_to_daily_series(excel_path, case["sheet"])
        case_outputs = _run_single_real_series_case(
            ep_opt,
            daily=daily,
            label=case["label"],
            disease_name="mpox",
            preset=case["preset"],
            fallback=case["fallback"],
        )
        prefix = case["label"].lower().replace(" ", "_")
        outputs[f"{prefix}_daily"] = case_outputs["daily"]
        outputs[f"{prefix}_summary"] = case_outputs["summary"]
        outputs[f"{prefix}_method"] = case_outputs["method"]

    _write_combined_real_application_outputs("mpox", outputs)
    return outputs


def run_chikungunya_external_application(ep_opt: base.EpiParams) -> Dict[str, pd.DataFrame]:
    chik_dir = CROSS_PATHOGEN_DATA_DIR
    chik_candidates = [
        chik_dir / "2025泉州基孔每日数据.xlsx",
        chik_dir / "chikungunya_data_decrypted.xlsx",
        chik_dir / "chikungunya_data_tmp.xlsx",
        chik_dir / "0595-tpyrced_泉州基孔肯雅热网络报告病例一览表20250904(24时)—110例.xlsx",
    ]
    excel_path = _find_first_existing(chik_candidates)
    if excel_path is None:
        raise FileNotFoundError(
            "Chikungunya source file not found. Expected one of: "
            + ", ".join(str(p) for p in chik_candidates)
        )

    print("\n" + "#" * 72)
    print("  CROSS-PATHOGEN STAGE 3: chikungunya external application")
    print(f"  Source workbook: {excel_path}")
    print("#" * 72)

    daily = _read_chikungunya_workbook_to_daily_series(excel_path)
    baseline_days = int(np.ceil(3.0 * base.EpiParams(preset="chikungunya").GT))
    daily = _prepend_zero_baseline(daily, baseline_days)
    case_outputs = _run_single_real_series_case(
        ep_opt,
        daily=daily,
        label="Chikungunya Quanzhou 2025",
        disease_name="chikungunya",
        preset="chikungunya",
        fallback="chikungunya",
    )
    outputs = {
        "chikungunya_quanzhou_2025_daily": case_outputs["daily"],
        "chikungunya_quanzhou_2025_summary": case_outputs["summary"],
        "chikungunya_quanzhou_2025_method": case_outputs["method"],
    }
    _write_combined_real_application_outputs("chikungunya", outputs)
    return outputs


def _legacy_run_influenza_external_application_shadowed(ep_opt: base.EpiParams) -> Dict[str, pd.DataFrame]:
    flu_candidates = [
        PROJECT_ROOT / "influenza_daily_cases_summary_province_city1.xlsx",
        CROSS_PATHOGEN_DATA_DIR / "influenza_daily_cases_summary_province_city1.xlsx",
    ]
    excel_path = _find_first_existing(flu_candidates)
    if excel_path is None:
        raise FileNotFoundError(
            "Influenza source file not found. Expected one of: "
            + ", ".join(str(p) for p in flu_candidates)
        )

    print("\n" + "#" * 72)
    print("  CROSS-PATHOGEN STAGE 3: influenza 2025-2026 external application")
    print(f"  Source workbook: {excel_path}")
    print("#" * 72)

    daily = _read_influenza_summary_workbook_to_daily_series(excel_path)
    case_outputs = _run_single_real_series_case(
        ep_opt,
        daily=daily,
        label="Influenza China 2025-2026",
        disease_name="influenza",
        preset="influenza_seasonal",
        fallback="influenza_h1n1",
    )
    rolling_df, rolling_summary = _build_single_series_rolling_reference(
        ep_opt,
        daily=daily,
        summary_df=case_outputs["summary"],
        disease_name="Influenza 2025-2026",
        city_label="Influenza China 2025-2026",
        preset="influenza_seasonal",
        fallback="influenza_h1n1",
    )
    rolling_df.to_csv(
        OUTPUT_STAGE3 / f"influenza_2025_2026_rolling_reference_validation_{VERSION_TAG}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rolling_summary.to_csv(
        OUTPUT_STAGE3 / f"influenza_2025_2026_rolling_reference_summary_{VERSION_TAG}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    outputs = {
        "influenza_2025_2026_daily": case_outputs["daily"],
        "influenza_2025_2026_summary": case_outputs["summary"],
        "influenza_2025_2026_method": case_outputs["method"],
        "influenza_2025_2026_rolling_validation": rolling_df,
        "influenza_2025_2026_rolling_summary": rolling_summary,
    }
    _write_combined_real_application_outputs("influenza", outputs)
    return outputs


def run_influenza_external_application(ep_opt: base.EpiParams) -> Dict[str, pd.DataFrame]:
    flu_candidates = [
        PROJECT_ROOT / "influenza_daily_cases_summary_province_city1.xlsx",
        CROSS_PATHOGEN_DATA_DIR / "influenza_daily_cases_summary_province_city1.xlsx",
    ]
    excel_path = _find_first_existing(flu_candidates)
    if excel_path is None:
        raise FileNotFoundError(
            "Influenza source file not found. Expected one of: "
            + ", ".join(str(p) for p in flu_candidates)
        )

    print("\n" + "#" * 72)
    print("  CROSS-PATHOGEN STAGE 3: influenza 2025-2026 external application")
    print(f"  Source workbook: {excel_path}")
    print("#" * 72)

    prepared = _prepare_influenza_province_city_workbook(excel_path)
    daily = _build_daily_series_from_matrix(prepared["province_matrix"])
    national_outputs = _run_single_real_series_case(
        ep_opt,
        daily=daily,
        label="Influenza China 2025-2026",
        disease_name="influenza",
        preset="influenza_seasonal",
        fallback="influenza_h1n1",
    )
    city_outputs = _run_realdata_matrix_application(
        ep_opt,
        matrix_path=prepared["city_matrix_path"],
        start_date=prepared["start_date"],
        label="Influenza 2025-2026 City Direct Municipality",
        disease_name="influenza",
        preset="influenza_seasonal",
        fallback="influenza_h1n1",
    )
    province_outputs = _run_realdata_matrix_application(
        ep_opt,
        matrix_path=prepared["province_matrix_path"],
        start_date=prepared["start_date"],
        label="Influenza 2025-2026 Province",
        disease_name="influenza",
        preset="influenza_seasonal",
        fallback="influenza_h1n1",
    )
    rolling_df, rolling_summary = _build_single_series_rolling_reference(
        ep_opt,
        daily=daily,
        summary_df=national_outputs["summary"],
        disease_name="Influenza 2025-2026",
        city_label="Influenza China 2025-2026",
        preset="influenza_seasonal",
        fallback="influenza_h1n1",
    )
    rolling_df.to_csv(
        OUTPUT_STAGE3 / f"influenza_2025_2026_rolling_reference_validation_{VERSION_TAG}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rolling_summary.to_csv(
        OUTPUT_STAGE3 / f"influenza_2025_2026_rolling_reference_summary_{VERSION_TAG}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    outputs = {
        "influenza_2025_2026_daily": national_outputs["daily"],
        "influenza_2025_2026_summary": national_outputs["summary"],
        "influenza_2025_2026_method": national_outputs["method"],
        "influenza_city_qc": prepared["city_qc"],
        "influenza_2025_2026_rolling_validation": rolling_df,
        "influenza_2025_2026_rolling_summary": rolling_summary,
    }
    outputs.update(city_outputs)
    outputs.update(province_outputs)
    _write_combined_real_application_outputs("influenza", outputs)
    return outputs


def _validate_stage3_result_integrity(outputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    checks: List[Dict[str, object]] = []

    def _record(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    expected_keys = [
        "cross_pathogen_real_case_summary",
        "cross_pathogen_real_case_overview",
        "cross_pathogen_real_case_method_long",
    ]
    for key in expected_keys:
        _record(key, key in outputs and isinstance(outputs.get(key), pd.DataFrame) and len(outputs[key]) > 0,
                "present" if key in outputs else "missing")

    summary_df = outputs.get("cross_pathogen_real_case_summary")
    if isinstance(summary_df, pd.DataFrame) and len(summary_df):
        scope_values = set(summary_df.get("source_group", pd.Series(dtype=str)).astype(str))
        _record(
            "influenza_dual_level_present",
            ("Influenza 2025-2026 City Direct Municipality" in scope_values)
            and ("Influenza 2025-2026 Province" in scope_values),
            ",".join(sorted(scope_values)),
        )
        required_cols = {"T1", "T2", "Tp", "confidence", "source_group"}
        _record(
            "summary_required_columns",
            required_cols.issubset(summary_df.columns),
            ",".join(summary_df.columns),
        )
    else:
        _record("influenza_dual_level_present", False, "summary_missing")
        _record("summary_required_columns", False, "summary_missing")

    integrity_df = pd.DataFrame(checks)
    integrity_df.to_csv(
        OUTPUT_STAGE3 / f"stage3_integrity_checks_{VERSION_TAG}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return integrity_df


def run_existing_core_real_application(ep_opt: base.EpiParams) -> Dict[str, object]:
    print("\n" + "#" * 72)
    print("  CROSS-PATHOGEN STAGE 3: existing core real-world application")
    print(f"  H1N1 source: {base.DATA_PATH_H1N1}")
    print(f"  COVID source: {base.DATA_PATH_COVID}")
    print("#" * 72)
    return base.run_stage3_real_application(ep_opt, run_rolling_reference=True)


def build_cross_pathogen_real_case_outputs(existing_results: Optional[Dict[str, object]],
                                           mpox_outputs: Optional[Dict[str, pd.DataFrame]],
                                           chik_outputs: Optional[Dict[str, pd.DataFrame]],
                                           influenza_outputs: Optional[Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    combined_summary_frames: List[pd.DataFrame] = []
    combined_method_frames: List[pd.DataFrame] = []
    case_overview_rows: List[Dict[str, object]] = []
    rolling_frames: List[pd.DataFrame] = []

    def _preserve_or_fill_source_group(frame: pd.DataFrame, fallback_group: str) -> pd.DataFrame:
        work = frame.copy()
        if "source_group" not in work.columns:
            work["source_group"] = fallback_group
            return work
        source_vals = work["source_group"].fillna("").astype(str).str.strip()
        missing_mask = source_vals.eq("")
        if missing_mask.any():
            work.loc[missing_mask, "source_group"] = fallback_group
        return work

    if existing_results:
        for key in ["H1N1 2009", "COVID-19 Omicron"]:
            payload = existing_results.get(key)
            if isinstance(payload, dict) and isinstance(payload.get("summary"), pd.DataFrame):
                summary_df = payload["summary"].copy()
                summary_df["source_group"] = key
                combined_summary_frames.append(summary_df)
            if isinstance(payload, dict) and isinstance(payload.get("national"), dict):
                nat = payload["national"]
                nat_conf = nat.get("confidence")
                nat_conf_overall = nat_conf.get("overall") if isinstance(nat_conf, dict) else nat_conf
                nat_alert_days = nat.get("alert_days")
                if nat_alert_days is None and isinstance(nat_conf, dict):
                    nat_alert_days = nat_conf.get("alert_days")
                nat_t1_to_peak = nat.get("T1_lead_peak", nat.get("T1_to_peak_days"))
                if nat_t1_to_peak is None:
                    t1_val = nat.get("T1")
                    tp_val = nat.get("Tp")
                    if isinstance(t1_val, (int, float)) and isinstance(tp_val, (int, float)):
                        nat_t1_to_peak = tp_val - t1_val
                case_overview_rows.append({
                    "case_label": key,
                    "region": f"{key} national",
                    "disease": key,
                    "source_group": key,
                    "T1": nat.get("T1"),
                    "T2": nat.get("T2"),
                    "Tp": nat.get("Tp"),
                    "T3": nat.get("T3"),
                    "alert_days": nat_alert_days,
                    "T1_to_peak_days": nat_t1_to_peak,
                    "confidence": nat_conf_overall,
                    "T1_confidence": nat_conf.get("T1") if isinstance(nat_conf, dict) else None,
                    "T2_confidence": nat_conf.get("T2") if isinstance(nat_conf, dict) else None,
                    "Tp_confidence": None,
                    "T3_confidence": nat_conf.get("T3") if isinstance(nat_conf, dict) else None,
                    "Tp_confidence_note": None,
                    "case_scope": None,
                    "case_scope_note": None,
                    "short_alert_class": nat.get("short_alert_class"),
                    "intervention_window": nat.get("intervention_window"),
                })
        method_long = existing_results.get("method_comparison_long")
        if isinstance(method_long, pd.DataFrame) and len(method_long):
            keep_labels = {"H1N1 2009", "COVID-19 Omicron"}
            method_df = method_long.copy()
            if "disease" in method_df.columns:
                method_df = method_df[method_df["disease"].astype(str).isin(keep_labels)].copy()
            elif "source" in method_df.columns:
                method_df = method_df[method_df["source"].astype(str).isin(keep_labels)].copy()
            method_df["source_group"] = "existing_core"
            if len(method_df):
                combined_method_frames.append(method_df)
        rolling_validation = existing_results.get("stage3_rolling_reference_validation")
        if isinstance(rolling_validation, pd.DataFrame) and len(rolling_validation):
            keep_labels = {"H1N1 2009", "COVID-19 Omicron"}
            rolling_df = rolling_validation.copy()
            if "disease" in rolling_df.columns:
                rolling_df = rolling_df[rolling_df["disease"].astype(str).isin(keep_labels)].copy()
            if len(rolling_df):
                rolling_df["source_group"] = "existing_core"
                rolling_frames.append(rolling_df)

    for payload, group_name in [
        (mpox_outputs, "mpox_extension"),
        (chik_outputs, "chikungunya_extension"),
        (influenza_outputs, "influenza_extension"),
    ]:
        if not payload:
            continue
        for key, frame in payload.items():
            if not isinstance(frame, pd.DataFrame) or not len(frame):
                continue
            if "rolling_validation" in key:
                rolling_df = _preserve_or_fill_source_group(frame, group_name)
                rolling_frames.append(rolling_df)
            elif "rolling_summary" in key:
                continue
            elif key.endswith("_summary"):
                summary_df = _preserve_or_fill_source_group(frame, group_name)
                label_cols = [c for c in ["region", "city"] if c in summary_df.columns]
                stage_cols = [c for c in ["T1", "T2", "Tp", "T3"] if c in summary_df.columns]
                if label_cols:
                    label_mask = summary_df[label_cols].fillna("").astype(str).apply(
                        lambda col: col.str.strip()
                    ).ne("").any(axis=1)
                    summary_df = summary_df[label_mask].copy()
                if stage_cols:
                    stage_mask = summary_df[stage_cols].notna().any(axis=1)
                    summary_df = summary_df[stage_mask].copy()
                if not len(summary_df):
                    continue
                combined_summary_frames.append(summary_df)
                for _, row in summary_df.iterrows():
                    case_overview_rows.append({
                        "case_label": row.get("region", row.get("city", row.get("source_group", group_name))),
                        "region": row.get("region", row.get("city")),
                        "disease": row.get("disease"),
                        "source_group": row.get("source_group", group_name),
                        "T1": row.get("T1"),
                        "T2": row.get("T2"),
                        "Tp": row.get("Tp"),
                        "T3": row.get("T3"),
                        "alert_days": row.get("alert_days"),
                        "T1_to_peak_days": row.get("T1_lead_peak", row.get("T1_to_peak_days")),
                        "confidence": row.get("confidence"),
                        "T1_confidence": row.get("T1_confidence"),
                        "T2_confidence": row.get("T2_confidence"),
                        "Tp_confidence": row.get("Tp_confidence"),
                        "T3_confidence": row.get("T3_confidence"),
                        "Tp_confidence_note": row.get("Tp_confidence_note"),
                        "case_scope": row.get("case_scope"),
                        "case_scope_note": row.get("case_scope_note"),
                        "short_alert_class": row.get("short_alert_class"),
                        "intervention_window": row.get("intervention_window"),
                    })
            elif key.endswith("_method"):
                method_df = _preserve_or_fill_source_group(frame, group_name)
                combined_method_frames.append(method_df)

    outputs: Dict[str, pd.DataFrame] = {}
    if combined_summary_frames:
        combined_summary = pd.concat(combined_summary_frames, ignore_index=True, sort=False)
        dedup_cols = [c for c in ["source_group", "region", "city", "disease", "T1", "T2", "Tp", "T3"] if c in combined_summary.columns]
        if dedup_cols:
            combined_summary = combined_summary.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
        combined_summary.to_csv(
            OUTPUT_STAGE3 / f"cross_pathogen_real_case_summary_{VERSION_TAG}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        outputs["cross_pathogen_real_case_summary"] = combined_summary
    if combined_method_frames:
        combined_method = pd.concat(combined_method_frames, ignore_index=True, sort=False)
        combined_method.to_csv(
            OUTPUT_STAGE3 / f"cross_pathogen_real_case_method_long_{VERSION_TAG}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        outputs["cross_pathogen_real_case_method_long"] = combined_method
    if rolling_frames:
        rolling_validation = pd.concat(rolling_frames, ignore_index=True, sort=False)
        rolling_summary = base._summarize_stage3_rolling_reference(rolling_validation)
        rolling_validation.to_csv(
            OUTPUT_STAGE3 / f"cross_pathogen_real_case_rolling_reference_validation_{VERSION_TAG}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        rolling_summary.to_csv(
            OUTPUT_STAGE3 / f"cross_pathogen_real_case_rolling_reference_summary_{VERSION_TAG}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        rolling_validation.to_csv(
            OUTPUT_STAGE3 / f"stage3_rolling_reference_validation_{VERSION_TAG}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        rolling_summary.to_csv(
            OUTPUT_STAGE3 / f"stage3_rolling_reference_summary_{VERSION_TAG}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        outputs["cross_pathogen_real_case_rolling_reference_validation"] = rolling_validation
        outputs["cross_pathogen_real_case_rolling_reference_summary"] = rolling_summary
    if case_overview_rows:
        case_overview = pd.DataFrame(case_overview_rows)
        preferred_cols = [
            "case_label",
            "region",
            "disease",
            "source_group",
            "T1",
            "T2",
            "Tp",
            "T3",
            "alert_days",
            "T1_to_peak_days",
            "confidence",
            "T1_confidence",
            "T2_confidence",
            "Tp_confidence",
            "T3_confidence",
            "Tp_confidence_note",
            "case_scope",
            "case_scope_note",
            "short_alert_class",
            "intervention_window",
        ]
        case_overview = case_overview[[c for c in preferred_cols if c in case_overview.columns]]
        dedup_cols = [c for c in ["case_label", "region", "disease", "source_group", "T1", "T2", "Tp", "T3"] if c in case_overview.columns]
        if dedup_cols:
            case_overview = case_overview.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
        case_overview.to_csv(
            OUTPUT_STAGE3 / f"cross_pathogen_real_case_overview_{VERSION_TAG}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        outputs["cross_pathogen_real_case_overview"] = case_overview
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-pathogen extension wrapper for the V31.1 EEMD-CWT phase-segmentation framework."
    )
    parser.add_argument(
        "--mode",
        choices=[
            "design",
            "stage1",
            "core-validation",
            "prospective-validation",
            "stage2-all",
            "mpox-application",
            "chikungunya-application",
            "influenza-application",
            "real-application",
            "all",
        ],
        default="design",
    )
    parser.add_argument("--n-dev", type=int, default=120)
    parser.add_argument("--n-stage2-tune", type=int, default=24)
    parser.add_argument("--n-per-archetype", type=int, default=40)
    parser.add_argument("--n-t1-tune", type=int, default=12)
    parser.add_argument("--n-t1-validation", type=int, default=40)
    parser.add_argument("--n-prospective-validation", type=int, default=80)
    parser.add_argument("--eemd-trials-default", type=int, default=12)
    parser.add_argument("--mpv-tuned", action="store_true")
    parser.add_argument("--archetypes", nargs="+", default=None)
    parser.add_argument("--reuse-locked-params", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--force-grid", action="store_true")
    parser.add_argument("--full-t1-search", action="store_true")
    parser.add_argument("--check-env", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_env:
        print_env_check()
        return

    ensure_runtime_ready()
    configure_output_dirs()
    install_safe_batch_generator()
    scenarios = register_cross_pathogen_archetypes()
    apply_influenza_prototype_rebuild()
    if args.mpv_tuned:
        apply_mpv_like_optimization()
    apply_mpox_specific_optimization()
    print(f"[cross-pathogen] Registered archetypes: {len(scenarios)}")

    export_extended_design_table(default_n=max(args.n_dev, len(scenarios) * 20))
    if args.mode == "design":
        print(f"[cross-pathogen] Design-only run completed. Output root: {OUTPUT_ROOT}")
        return

    ep_opt = load_or_build_params(args)
    existing_real_results: Optional[Dict[str, object]] = None
    mpox_outputs: Optional[Dict[str, pd.DataFrame]] = None
    chik_outputs: Optional[Dict[str, pd.DataFrame]] = None
    influenza_outputs: Optional[Dict[str, pd.DataFrame]] = None
    if args.mode in {"core-validation", "stage2-all", "all"}:
        run_extended_core_validation(
            ep_opt,
            n_per_archetype=args.n_per_archetype,
            eemd_trials_default=args.eemd_trials_default,
            output_label=(
                f"cross_pathogen_core_archetype_{MPV_TUNED_VARIANT}"
                if args.mpv_tuned else "cross_pathogen_core_archetype"
            ),
            mpv_tuned=args.mpv_tuned,
            selected_archetypes=args.archetypes,
        )
    if args.mode in {"prospective-validation", "stage2-all", "all"}:
        ep_opt, _ = run_cross_pathogen_prospective_validation(
            ep_opt,
            n_t1_tune=args.n_t1_tune,
            n_t1_validation=args.n_t1_validation,
            n_prospective_validation=args.n_prospective_validation,
            t1_fast_mode=not args.full_t1_search,
        )
    if args.mode in {"mpox-application", "real-application", "all"}:
        mpox_outputs = run_mpox_external_application(ep_opt)
    if args.mode == "chikungunya-application":
        chik_outputs = run_chikungunya_external_application(ep_opt)
    if args.mode == "influenza-application":
        influenza_outputs = run_influenza_external_application(ep_opt)
    if args.mode in {"real-application", "all"}:
        existing_real_results = run_existing_core_real_application(ep_opt)
        try:
            chik_outputs = run_chikungunya_external_application(ep_opt)
        except Exception as exc:
            print(f"[cross-pathogen] Warning: chikungunya application skipped: {exc}")
        try:
            influenza_outputs = run_influenza_external_application(ep_opt)
        except Exception as exc:
            print(f"[cross-pathogen] Warning: influenza application skipped: {exc}")
        stage3_outputs = build_cross_pathogen_real_case_outputs(
            existing_real_results,
            mpox_outputs,
            chik_outputs,
            influenza_outputs,
        )
        integrity_df = _validate_stage3_result_integrity(stage3_outputs)
        print(f"[cross-pathogen] Stage3 integrity checks exported: {OUTPUT_STAGE3 / f'stage3_integrity_checks_{VERSION_TAG}.csv'}")
        if not integrity_df["ok"].all():
            print(integrity_df.to_string(index=False))

    if args.mode in {"stage1", "all"} and args.mode != "all":
        print(f"[cross-pathogen] Stage1 development completed. Output root: {OUTPUT_ROOT}")
        return

    print(f"[cross-pathogen] Completed. Output root: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
