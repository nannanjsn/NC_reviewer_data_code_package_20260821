# -*- coding: utf-8 -*-
"""
Epidemic phase segmentation V31.1 Archetype-Adaptive Fusion

Design principles
-----------------
1) Use the V27.1 structural-change framework as the main algorithm:
   - T1: organized mid-frequency strengthening, high/mid relation shift,
     and rightward scale-centroid movement.
   - T2: establishment of low-frequency dominance, not only a second
     derivative inflection point.
   - Tp: trend peak.
   - T3: clear trend decline with weakening main-wave structure.
2) Absorb useful V24.8 features:
   - coarse-to-fine parameter optimization.
   - richer prospective rolling confirmation and diagnostic outputs.
   - real-data signal-start and two-stage diagnostic helpers.
   - cross-disease standardized outputs and stability intervals.
3) Fix key methodological issues:
   - T2 uses a dual reference system: T2_dyn + T2_dom.
   - T1 keeps growth-truth and adds T1_struct for sensitivity analysis.
   - Stage 3 uses full-series structural detection instead of onset cropping.
   - Each key point reports support intervals and optional EEMD ensemble bands.
4) Replace one-off simulated scenario pools with pathogen-archetype sampling:
   - Disease X development curves are stratified by pathogen archetype.
   - R/GT/SI/reporting/noise ranges are explicit and exported as a design table.
5) Keep the V31 phase framework unchanged, but add a thin adaptive layer:
   - GT/SI/report delay/archetype tune confirmation and dominance thresholds.
   - Slow or delayed-reporting archetypes use more conservative T1/T2/Tp settings.
"""

from __future__ import annotations

import copy
import datetime
import json
import math
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, hilbert

warnings.filterwarnings('ignore')
plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 100,
    'savefig.dpi': 150,
})

VERSION_TAG = 'v31_1_archetype_adaptive'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_BENCH = f'stage0_fusion_benchmark_{VERSION_TAG}'
OUTPUT_STAGE1 = f'stage1_fusion_development_{VERSION_TAG}'
OUTPUT_STAGE2 = f'stage2_fusion_validation_{VERSION_TAG}'
OUTPUT_STAGE3 = f'stage3_fusion_real_application_{VERSION_TAG}'
for _d in [OUTPUT_BENCH, OUTPUT_STAGE1, OUTPUT_STAGE2, OUTPUT_STAGE3]:
    os.makedirs(_d, exist_ok=True)

CACHE_PATH = os.path.join(OUTPUT_STAGE1, f'eemd_cwt_structural_cache_{VERSION_TAG}.npz')
BEST_PARAMS_PATH = os.path.join(OUTPUT_STAGE1, f'best_params_{VERSION_TAG}.json')
T1_FOCUSED_PARAMS_PATH = os.path.join(OUTPUT_STAGE2, f't1_focused_best_params_{VERSION_TAG}.json')
PAPER_TABLES_PATH = os.path.join(OUTPUT_STAGE3, f'paper_ready_tables_{VERSION_TAG}.xlsx')

DATA_PATH_H1N1 = os.environ.get(
    'DATA_PATH_H1N1',
    os.path.join(BASE_DIR, '甲流_筛选城市_20090513_20091231.xlsx'),
)
DATA_PATH_COVID = os.environ.get(
    'DATA_PATH_COVID',
    os.path.join(BASE_DIR, 'covid19_data_not_included.xlsx'),
)
DATA_START_H1N1 = datetime.date(2009, 5, 13)
DATA_START_COVID = datetime.date(2022, 11, 1)


# ============================================================
#  Part 0. 通用工具
# ============================================================

def robust_mad_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 1.0
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    return float(max(mad, np.std(x), 1e-10))


def safe_zscore(x: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if len(baseline) == 0:
        baseline = x
    b_mu = np.median(baseline)
    b_sd = robust_mad_std(baseline)
    return (x - b_mu) / b_sd


def safe_minmax(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return arr.copy()
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-10:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def rolling_mean(signal: np.ndarray, window: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    window = max(1, int(window))
    if len(signal) < window:
        return np.full(len(signal), np.mean(signal) if len(signal) else 0.0)
    return np.convolve(signal, np.ones(window) / window, mode='same')


def bootstrap_ci(errors, n_boot: int = 500, ci: float = 0.95):
    errors = np.asarray(errors, dtype=float)
    errors = errors[np.isfinite(errors)]
    if len(errors) == 0:
        return np.nan, np.nan, np.nan
    boot = [
        np.median(np.random.choice(errors, len(errors), replace=True))
        for _ in range(n_boot)
    ]
    lo = np.percentile(boot, (1 - ci) / 2 * 100)
    hi = np.percentile(boot, (1 + ci) / 2 * 100)
    return float(np.median(errors)), float(lo), float(hi)


def percentile_band(values, low: float = 10, high: float = 90) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    return float(np.percentile(arr, low)), float(np.percentile(arr, high))


def band_distance(est: Optional[int], a: Optional[int], b: Optional[int]) -> float:
    if est is None or a is None or b is None:
        return np.nan
    lo, hi = sorted([int(a), int(b)])
    if lo <= est <= hi:
        return 0.0
    return float(min(abs(est - lo), abs(est - hi)))


def first_consecutive(cond: np.ndarray, start: int, end: int, consec: int) -> Optional[int]:
    start = max(0, int(start))
    end = min(len(cond), int(end))
    consec = max(1, int(consec))
    if end - start < consec:
        return None
    for i in range(start, end - consec + 1):
        if np.all(cond[i:i + consec]):
            return int(i)
    return None


def local_support_interval(score: np.ndarray, center: int, frac: float = 0.90, max_radius: int = 14,
                           hard_floor: Optional[float] = None) -> Tuple[int, int]:
    score = np.asarray(score, dtype=float)
    N = len(score)
    if N == 0:
        return 0, 0
    center = int(min(max(center, 0), N - 1))
    thr = score[center] * frac
    if hard_floor is not None:
        thr = max(thr, hard_floor)
    lo = center
    while lo > 0 and (center - lo) < max_radius and score[lo - 1] >= thr:
        lo -= 1
    hi = center
    while hi < N - 1 and (hi - center) < max_radius and score[hi + 1] >= thr:
        hi += 1
    return int(lo), int(hi)


# ============================================================
#  Part 1. EEMD + Morlet CWT
# ============================================================

class EEMD:
    def __init__(self, trials: int = 50, noise_width: float = 0.2, seed: int = 42,
                 early_stop_tol: float = 0.02, min_trials: int = 10):
        self.trials = trials
        self.noise_width = noise_width
        self.seed = seed
        self.early_stop_tol = early_stop_tol
        self.min_trials = min_trials

    def eemd(self, signal, t=None, max_imfs=None):
        signal = np.asarray(signal, dtype=float)
        N = len(signal)
        t = np.arange(N, dtype=float) if t is None else np.asarray(t, dtype=float)
        if max_imfs is None:
            max_imfs = max(3, int(np.log2(max(N, 4))) + 1)

        rng = np.random.RandomState(self.seed)
        sig_std = np.std(signal) + 1e-10
        all_imfs: List[List[np.ndarray]] = []
        prev_mean = None

        for trial_idx in range(self.trials):
            noise = rng.normal(0, 1, N) * self.noise_width * sig_std
            imfs_run = self._emd(signal + noise, t, max_imfs)
            if imfs_run:
                all_imfs.append(imfs_run)
            if len(all_imfs) >= self.min_trials and (trial_idx + 1) % self.min_trials == 0:
                current_mean = self._compute_mean_imfs(all_imfs, N)
                if prev_mean is not None:
                    if np.mean(np.abs(current_mean - prev_mean)) / sig_std < self.early_stop_tol:
                        break
                prev_mean = current_mean.copy()

        if not all_imfs:
            return np.array([signal])

        result = self._compute_mean_imfs(all_imfs, N)
        norms = np.array([np.sum(np.abs(row)) for row in result])
        keep = np.where(norms > 1e-10)[0]
        return result[: keep[-1] + 1] if len(keep) else result[:1]

    @staticmethod
    def _compute_mean_imfs(all_imfs, N):
        mx_n = max(len(r) for r in all_imfs)
        aligned = np.zeros((len(all_imfs), mx_n, N))
        for i, imfs in enumerate(all_imfs):
            for j, imf in enumerate(imfs):
                aligned[i, j, : len(imf)] = imf[:N]
        return np.mean(aligned, axis=0)

    def _emd(self, signal, t, max_imfs):
        residual = signal.copy()
        imfs: List[np.ndarray] = []
        for _ in range(max_imfs):
            imf = self._sift(residual, t)
            if imf is None or np.sum(np.abs(imf)) < 1e-10:
                break
            imfs.append(imf)
            residual = residual - imf
            mx, mn = self._extrema(residual)
            if len(mx) + len(mn) < 3:
                break
        imfs.append(residual)
        return imfs

    def _sift(self, sig, t, max_iter=150, tol=0.15):
        h = sig.copy()
        for _ in range(max_iter):
            mx, mn = self._extrema(h)
            if len(mx) < 2 or len(mn) < 2:
                return None
            upper = self._interp(t, h, mx)
            lower = self._interp(t, h, mn)
            h_new = h - (upper + lower) / 2
            denom = np.sum(h ** 2)
            if denom > 1e-10 and np.sum((h_new - h) ** 2) / denom < tol:
                return h_new
            h = h_new
        return h

    @staticmethod
    def _extrema(sig):
        return find_peaks(sig)[0], find_peaks(-sig)[0]

    @staticmethod
    def _interp(t, sig, idx):
        if len(idx) < 2:
            return np.zeros_like(sig)
        ia = np.asarray(idx, dtype=int)
        v = sig[ia]
        if ia[0] > 0:
            ia = np.concatenate([[0], ia])
            v = np.concatenate([[sig[0]], v])
        if ia[-1] < len(sig) - 1:
            ia = np.concatenate([ia, [len(sig) - 1]])
            v = np.concatenate([v, [sig[-1]]])
        _, upos = np.unique(ia, return_index=True)
        ia_u, v_u = ia[upos], v[upos]
        if len(ia_u) < 2:
            return np.zeros_like(sig)
        try:
            return CubicSpline(t[ia_u], v_u, bc_type='natural')(t)
        except Exception:
            return np.interp(t, t[ia_u], v_u)


class MorletWavelet:
    def __init__(self, w0=6):
        self.w0 = w0

    def flambda(self):
        return 4 * np.pi / (self.w0 + np.sqrt(2 + self.w0 ** 2))

    def cone_of_influence(self):
        return np.sqrt(2)


def cwt_morlet(signal, dt=1.0, s0=2.0, dj=1 / 12, J=None, w0=6):
    wv = MorletWavelet(w0)
    N = len(signal)
    if J is None:
        J = max(1, int(np.log2(max(N * dt / s0, 2)) / dj))
    scales = s0 * 2 ** (np.arange(0, J + 1) * dj)
    periods = scales * wv.flambda()
    sc = np.asarray(signal, dtype=float) - np.mean(signal)
    np2 = 2 ** int(np.ceil(np.log2(max(N, 2))))
    sp = np.zeros(np2)
    sp[:N] = sc
    fs = np.fft.fft(sp)
    om = np.fft.fftfreq(np2, d=dt) * 2 * np.pi
    wave_out = np.zeros((len(scales), N), dtype=complex)
    for i, s in enumerate(scales):
        norm = np.sqrt(2 * np.pi * s / dt) * (np.pi ** -0.25)
        ph = np.zeros(np2, dtype=complex)
        pos = om > 0
        ph[pos] = norm * np.exp(-((s * om[pos] - w0) ** 2) / 2)
        wave_out[i, :] = np.fft.ifft(fs * ph)[:N]
    coi = wv.cone_of_influence() * dt * np.minimum(np.arange(N), np.arange(N)[::-1])
    return wave_out, scales, periods, np.maximum(coi, s0)


def compute_cwt_band_energy(signal, period_lo, period_hi, dt=1.0, smooth_sigma=2.0):
    N = len(signal)
    if N < 10 or np.std(signal) < 1e-10:
        return np.zeros(N)
    sig_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
    try:
        wave, scales, periods, coi = cwt_morlet(sig_norm, dt=dt)
    except Exception:
        return np.zeros(N)
    power = np.abs(wave) ** 2
    band_mask = (periods >= period_lo) & (periods <= period_hi)
    if not np.any(band_mask):
        return np.zeros(N)
    band_idx = np.where(band_mask)[0]
    valid_power = np.full((len(band_idx), N), np.nan)
    for k, bi in enumerate(band_idx):
        s = scales[bi]
        vm = s <= coi
        valid_power[k, vm] = power[bi, vm]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        energy = np.nanmean(valid_power, axis=0)
    energy = np.where(np.isnan(energy), 0.0, energy)
    if smooth_sigma > 0:
        energy = gaussian_filter1d(energy, sigma=smooth_sigma)
    return np.maximum(energy, 0.0)


def compute_instantaneous_frequency(imf, dt=1.0, smooth_sigma=3.0):
    N = len(imf)
    if N < 10:
        return np.zeros(N), np.zeros(N)
    try:
        analytic = hilbert(imf)
        inst_phase = np.unwrap(np.angle(analytic))
        inst_freq = np.gradient(inst_phase, dt) / (2 * np.pi)
        inst_freq = np.abs(inst_freq)
        if smooth_sigma > 0:
            inst_freq = gaussian_filter1d(inst_freq, sigma=smooth_sigma)
        inst_freq = np.clip(inst_freq, 0, 1.0)
    except Exception:
        inst_freq = np.zeros(N)
    d_inst_freq = np.gradient(inst_freq)
    if smooth_sigma > 0:
        d_inst_freq = gaussian_filter1d(d_inst_freq, sigma=smooth_sigma)
    return inst_freq, d_inst_freq


def compute_energy_acceleration(e_mid, smooth_sigma=2.0):
    d1 = np.gradient(e_mid)
    d2 = np.gradient(d1)
    if smooth_sigma > 0:
        d1 = gaussian_filter1d(d1, sigma=smooth_sigma)
        d2 = gaussian_filter1d(d2, sigma=smooth_sigma)
    return d1, d2


# ============================================================
#  Part 2. Parameters
# ============================================================

class EpiParams:
    PRESETS = {
        'omicron': dict(name='SARS-CoV-2 Omicron', GT=3.0, SI=3.3, T_inc=3.6, T_inf=6.5, T_report=2.0, T_week=7.0),
        'influenza_h1n1': dict(name='Influenza A/H1N1 (2009)', GT=2.6, SI=2.2, T_inc=1.5, T_inf=5.0, T_report=3.0, T_week=7.0),
        'dengue': dict(name='Dengue Fever', GT=7.0, SI=6.0, T_inc=5.0, T_inf=10.0, T_report=5.0, T_week=7.0),
        'chikungunya': dict(name='Chikungunya', GT=5.0, SI=4.0, T_inc=3.0, T_inf=7.0, T_report=5.0, T_week=7.0),
    }

    def __init__(self, preset='omicron', override_params: Optional[Dict] = None):
        cfg = self.PRESETS[preset].copy()
        self.preset = preset
        self.disease_name = cfg['name']
        self.GT = cfg['GT']
        self.SI = cfg['SI']
        self.T_inc = cfg['T_inc']
        self.T_inf = cfg['T_inf']
        self.T_report = cfg['T_report']
        self.T_week = cfg['T_week']
        self.refresh_derived()
        if override_params:
            self.apply_override(override_params)

    def refresh_derived(self):
        GT = self.GT
        self.trend_sigma = max(2, int(np.ceil(GT)))
        self.energy_smooth_sigma = max(1.5, GT * 0.55)
        self.score_smooth_sigma = max(1.0, GT * 0.40)

        self.high_band_lo = max(2, int(np.floor(max(2.0, 0.6 * GT))))
        self.high_band_hi = max(self.high_band_lo + 1, int(np.ceil(max(self.T_week * 0.8, 1.6 * GT))))
        self.mid_band_lo = max(self.high_band_hi, int(np.ceil(2.0 * GT)))
        self.mid_band_hi = max(self.mid_band_lo + 2, int(np.ceil(5.0 * GT)))
        self.low_band_lo = max(self.mid_band_hi, int(np.ceil(5.0 * GT)))
        self.low_band_hi = max(self.low_band_lo + 4, int(np.ceil(15.0 * GT)))

        self.t1_baseline_days = max(10, int(np.ceil(4.5 * GT)))
        self.t1_consec_days = max(2, int(np.ceil(GT)))
        self.t1_consec_real = max(3, int(np.ceil(GT)))
        self.t1_min_signal_gate_k = 2.0
        self.t1_mid_z_thresh = 1.25
        self.t1_mid_high_ratio_z = 0.70
        self.t1_centroid_z_thresh = 0.35
        self.t1_structure_score_thresh = 0.54
        self.t1_relaxed_mid_z_thresh = 0.95
        self.t1_relaxed_ratio_z = 0.20
        self.t1_relaxed_centroid_z = 0.05
        self.t1_relaxed_score_thresh = 0.44
        self.t1_relaxed_support_ratio = 0.60
        self.t1_lookback_allowance = max(2, int(np.ceil(0.8 * GT)))
        self.t1_prospective_strict_relax = 0.90
        self.t1_prospective_ratio_relax = 0.85
        self.t1_prospective_centroid_relax = 0.82
        self.t1_prospective_score_relax = 0.92
        self.t1_prealert_mid_z_thresh = 0.70
        self.t1_prealert_ratio_z_thresh = -0.05
        self.t1_prealert_centroid_z_thresh = -0.10
        self.t1_prealert_score_thresh = 0.36
        self.t1_prealert_support_ratio = 0.55
        self.t1_prealert_consec = max(2, int(np.ceil(0.67 * GT)))
        self.t1_prealert_followup_gt = 2.0
        # T1-first prospective confirmation settings. These are deliberately
        # less conservative than the retrospective segmentation thresholds,
        # because Stage 2 asks when a public-health warning can be stably raised.
        self.t1_valid_strength_min = 0.24
        self.t1_confirm_ewma_k = 0.55
        self.t1_fast_confirm_strength = 0.60
        self.t1_fast_local_margin = 0.02
        self.t1_single_confirm_strength = 0.56
        self.t1_confirm_span_gt = 1.0
        self.t1_confirm_quantile = 20
        self.t1_confirm_allow_single_strong = True
        self.t1_confirm_single_min_persist_gt = 0.35
        self.t1_confirm_single_local_margin = 0.06
        self.t1_confirm_allow_same_window_as_first_alert = False
        self.t1_early_single_strength = 0.54
        self.t1_progressive_strength_lo = 0.38
        self.t1_progressive_strength_hi = 0.50
        self.t1_progressive_rounds = 3
        self.t1_aux_energy_enable = True
        self.t1_aux_max_lead_gt = 3.0
        self.t1_aux_min_strength = 0.16
        self.t1_early_confirm_enable = True
        self.t1_early_confirm_min_persist_gt = 0.55
        self.t1_early_confirm_strength = 0.34
        self.t1_early_confirm_local_margin = -0.02
        self.t1_first_alert_enable = True
        self.t1_first_alert_strength = 0.24
        self.t1_first_alert_local_margin = -0.08
        self.t1_first_alert_min_persist_gt = 0.20
        self.t1_first_alert_allow_energy = True
        self.t1_alert_graduation_enable = True
        self.t1_alert_graduation_gt = 1.15
        self.t1_alert_graduation_min_strength = 0.24
        self.t1_raw_growth_alert_enable = True
        self.t1_raw_growth_min_persist_gt = 0.60
        self.t1_raw_growth_confirm_strength = 0.62
        self.t1_raw_growth_smooth_gt = 0.35
        self.t1_raw_growth_z = 2.2
        self.t1_raw_growth_ratio = 1.20
        self.t1_raw_growth_min_abs_frac = 0.04
        self.t1_raw_growth_backcast_gt = 1.00
        self.t1_estimate_backtrack_gt = 0.75
        self.t1_estimate_backtrack_min_persist_gt = 1.00
        self.t1_estimate_max_backtrack_gt = 0.75
        self.t1_estimate_history_quantile = 10
        self.t1_aux_stale_candidate_gt = 2.50
        self.t1_aux_stale_backcast_gt = 0.65
        self.t1_struct_stale_candidate_gt = 4.00
        self.t1_struct_stale_backcast_gt = 1.25

        self.t2_min_gap = max(3, int(np.ceil(GT)))
        self.t2_low_share_thresh_short = 0.30
        self.t2_low_share_thresh_medium = 0.34
        self.t2_low_share_thresh_long = 0.38
        self.t2_dom_switch_margin = 0.01
        self.t2_dom_score_thresh = 0.50
        self.t2_relaxed_low_share_factor = 0.92
        self.t2_relaxed_dom_factor = 0.90
        self.t2_support_ratio = 0.55
        self.t2_plateau_thresh = 0.05
        self.t2_band_weight = 0.70
        self.t2_min_window_days = max(int(self.low_band_hi * 1.2), 40)

        self.t3_low_frac = 0.20
        self.t3_consec_days = max(5, int(np.ceil(2.0 * GT)))
        self.t3_low_share_decay = 0.80

        self.real_bg_roll_window = 7
        self.real_bg_threshold = 3.0
        self.real_bg_min_days = max(7, int(np.ceil(2.0 * GT)))
        self.real_bg_max_days = 30
        self.real_raw_gate_k = 2.5
        self.real_epi_buffer = 3
        self.real_t2_low_share_short = 0.30
        self.real_t2_low_share_medium = 0.34
        self.real_t2_low_share_long = 0.38

        self.min_signal_frac = 0.01
        self.sig_gate_frac = 0.003
        self.sig_gate_consec = max(3, int(np.ceil(GT)))
        self.sig_valid_pre_buf = max(int(GT * 5), 15)
        self.bimodal_ratio = 3.0

        bl = max(10, int(np.ceil(5 * GT)))
        self.ewma_alpha = min(0.30, 2.0 / (bl / 2.0 + 1.0))
        self.prospective_min_start = max(int(np.ceil(2.0 * GT)), 8)
        self.prospective_min_window_gt = 3.0
        self.prospective_min_window_floor = 12
        self.prospective_t2_confirm_rounds = 3
        self.prospective_t2_confirm_tol_gt = 1.0
        self.prospective_t2_min_after_t1_gt = 1.25
        self.prospective_t2_strength_min = 0.42
        self.prospective_t2_dom_gap_min = 0.00
        self.prospective_t2_tail_quantile = 0.35
        self.prospective_tp_confirm_rounds = 3
        self.prospective_tp_confirm_tol_gt = 1.5
        self.prospective_tp_maturity_gt = 2.0
        self.prospective_tp_min_local_frac = 0.40
        self.prospective_tp_decline_frac = 0.10
        self.stage3_rolling_window_frac = 0.45
        self.stage3_rolling_window_floor = 42
        self.stage3_rolling_window_cap = 65
        self.stage3_rolling_confirm_rounds = 1
        self.sim_stage3_rolling_window_gt = 6.0
        self.sim_stage3_rolling_window_floor = 18
        self.sim_stage3_rolling_window_cap = 36
        self.sim_stage3_rolling_window_series_frac = 0.75
        self.confirm_tol = max(6, int(np.ceil(1.5 * GT)))

    def apply_override(self, params_dict: Dict):
        structural_keys = {'GT', 'SI', 'T_inc', 'T_inf', 'T_report', 'T_week'}
        need_refresh = any(k in structural_keys for k in params_dict)
        for k, v in params_dict.items():
            setattr(self, k, v)
        if need_refresh:
            self.refresh_derived()
            for k, v in params_dict.items():
                setattr(self, k, v)
        # 单调修正
        self.t2_low_share_thresh_medium = min(max(self.t2_low_share_thresh_short + 0.04, self.t2_low_share_thresh_medium), 0.50)
        self.t2_low_share_thresh_long = min(max(self.t2_low_share_thresh_medium + 0.04, self.t2_low_share_thresh_long), 0.56)
        self.real_t2_low_share_medium = min(max(self.real_t2_low_share_short + 0.04, self.real_t2_low_share_medium), 0.50)
        self.real_t2_low_share_long = min(max(self.real_t2_low_share_medium + 0.04, self.real_t2_low_share_long), 0.56)
        self.t1_progressive_strength_hi = max(
            self.t1_progressive_strength_lo + 0.05,
            self.t1_progressive_strength_hi,
        )
        if hasattr(self, 't1_early_confirm_strength'):
            self.t1_early_confirm_strength = min(
                self.t1_early_confirm_strength,
                max(0.12, self.t1_single_confirm_strength - 0.12),
            )
        if hasattr(self, 't1_first_alert_strength'):
            self.t1_first_alert_strength = min(
                self.t1_first_alert_strength,
                max(0.08, self.t1_early_confirm_strength - 0.04),
            )
        if hasattr(self, 't1_estimate_backtrack_gt'):
            max_backtrack = float(getattr(self, 't1_estimate_max_backtrack_gt', 0.75))
            self.t1_estimate_backtrack_gt = min(
                max(0.0, float(self.t1_estimate_backtrack_gt)),
                max_backtrack,
            )
        if hasattr(self, 'energy_smooth_sigma'):
            self.energy_smooth_sigma = max(1.5, float(self.energy_smooth_sigma))
        if hasattr(self, 'prospective_t2_confirm_rounds'):
            self.prospective_t2_confirm_rounds = max(2, int(self.prospective_t2_confirm_rounds))
        if hasattr(self, 'prospective_tp_confirm_rounds'):
            self.prospective_tp_confirm_rounds = max(2, int(self.prospective_tp_confirm_rounds))
        if hasattr(self, 'prospective_t2_strength_min'):
            self.prospective_t2_strength_min = max(0.0, float(self.prospective_t2_strength_min))
        if hasattr(self, 'prospective_tp_decline_frac'):
            self.prospective_tp_decline_frac = min(max(0.0, float(self.prospective_tp_decline_frac)), 0.40)

    def get_t2_low_share_thresh(self, n_days: int) -> float:
        if n_days < 120:
            return self.t2_low_share_thresh_short
        elif n_days < 200:
            return self.t2_low_share_thresh_medium
        return self.t2_low_share_thresh_long

    def get_real_t2_low_share_thresh(self, n_days: int) -> float:
        if n_days < 120:
            return self.real_t2_low_share_short
        elif n_days < 200:
            return self.real_t2_low_share_medium
        return self.real_t2_low_share_long

    def get_dynamic_bg_max_days(self, n_days: int) -> int:
        cap = min(45, int(max(1, n_days) * 0.20))
        return max(self.real_bg_min_days, min(self.real_bg_max_days, cap))

    def print_summary(self):
        print(f"\n{'='*78}")
        print(f"  {self.disease_name}  (GT={self.GT}d, SI={self.SI}d)")
        print(f"  Bands: high[{self.high_band_lo},{self.high_band_hi}]d  mid[{self.mid_band_lo},{self.mid_band_hi}]d  low[{self.low_band_lo},{self.low_band_hi}]d")
        print(f"  T1(structural): mid_z>{self.t1_mid_z_thresh}, ratio_z>{self.t1_mid_high_ratio_z}, centroid_z>{self.t1_centroid_z_thresh}, score>{self.t1_structure_score_thresh}, consec={self.t1_consec_days}d")
        print(f"  T2(structural dominance): low_share(short/med/long)=({self.t2_low_share_thresh_short:.2f}/{self.t2_low_share_thresh_medium:.2f}/{self.t2_low_share_thresh_long:.2f}), dom_score>{self.t2_dom_score_thresh}")
        print(f"  T3: trend<{self.t3_low_frac*100:.0f}% peak, consec={self.t3_consec_days}d")
        print(f"  Rolling: min_start={self.prospective_min_start}d, confirm_tol={self.confirm_tol}d, EWMA alpha={self.ewma_alpha:.3f}")
        print(f"  RealData: bg_roll={self.real_bg_roll_window}, raw_gate=mu+{self.real_raw_gate_k}sd, sig_gate_frac={self.sig_gate_frac*100:.2f}%peak")
        print(f"{'='*78}")


def _float_or_default(value, default: float) -> float:
    try:
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def infer_archetype_adaptive_profile(curve: Optional[Dict], ep: EpiParams) -> Dict:
    """Summarize curve metadata into a small conservative/sensitive tuning profile."""
    curve = curve or {}
    gt = max(1.0, _float_or_default(curve.get('GT'), ep.GT))
    si = max(1.0, _float_or_default(curve.get('SI'), gt))
    r0 = max(0.5, _float_or_default(curve.get('R0', curve.get('R')), 2.5))
    report_delay = max(0.0, _float_or_default(curve.get('report_delay'), ep.T_report))
    archetype = str(curve.get('archetype') or curve.get('scenario') or '').lower()
    profile_name = str(curve.get('structural_noise_profile') or '').lower()
    text = f'{archetype} {profile_name}'

    slow_gt = float(np.clip((gt - 4.0) / 4.0, 0.0, 1.4))
    delayed_reporting = float(np.clip((report_delay - 2.0) / 4.0, 0.0, 1.2))
    si_gt_gap = float(np.clip((si - gt) / max(gt, 1.0), 0.0, 0.6))
    mpv_like = any(x in text for x in ['mpv', 'orthopox', 'contact_network'])
    hospital_like = any(x in text for x in ['mers', 'hospital', 'amplification'])
    zoonotic_like = any(x in text for x in ['zoonotic', 'high_severity'])
    influenza_like = any(x in text for x in ['influenza', 'h1n1'])
    moderate_coronavirus = any(x in text for x in ['moderate_coronavirus'])
    contact_like = any(x in text for x in ['contact', 'zoonotic', 'mers', 'mpv', 'orthopox', 'hospital'])
    vector_like = any(x in text for x in ['vector', 'dengue', 'chikungunya'])
    fast_respiratory = any(x in text for x in ['high_transmissibility', 'fast_respiratory', 'influenza'])
    low_r = bool(r0 < 1.9)
    long_zoonotic_like = bool(zoonotic_like and gt >= 8.5 and r0 < 2.25)

    burden = slow_gt + 0.65 * delayed_reporting + 0.45 * si_gt_gap
    if contact_like:
        burden += 0.55
    if mpv_like:
        burden += 0.25
    if hospital_like:
        burden += 0.18
    if zoonotic_like:
        burden += 0.10
    if vector_like:
        burden += 0.35
    if fast_respiratory:
        burden -= 0.25
    burden = float(np.clip(burden, 0.0, 2.2))

    if burden >= 1.25:
        profile = 'conservative_slow_or_delayed'
    elif burden >= 0.55:
        profile = 'moderate_adaptive'
    else:
        profile = 'fast_or_standard'

    return dict(
        GT=gt, SI=si, R0=r0, report_delay=report_delay, archetype=archetype,
        structural_noise_profile=profile_name, adaptive_burden=burden,
        adaptive_profile=profile, contact_like=contact_like,
        vector_like=vector_like, fast_respiratory=fast_respiratory,
        mpv_like=mpv_like, hospital_like=hospital_like,
        zoonotic_like=zoonotic_like, influenza_like=influenza_like,
        moderate_coronavirus=moderate_coronavirus,
        low_r=low_r, long_zoonotic_like=long_zoonotic_like,
        slow_contact_like=bool(contact_like and gt >= 5.5),
    )


def build_archetype_adaptive_ep(base_ep: EpiParams, curve: Optional[Dict] = None,
                                prospective: bool = False) -> EpiParams:
    """Return a copy of the parameter object adapted to one simulated archetype."""
    profile = infer_archetype_adaptive_profile(curve, base_ep)
    ep = copy.deepcopy(base_ep)
    ep.apply_override({
        'GT': profile['GT'],
        'SI': profile['SI'],
        'T_report': profile['report_delay'],
    })

    burden = profile['adaptive_burden']
    fast = profile['fast_respiratory']
    mpv_like = profile.get('mpv_like', False)
    hospital_like = profile.get('hospital_like', False)
    zoonotic_like = profile.get('zoonotic_like', False)
    long_zoonotic_like = profile.get('long_zoonotic_like', False)
    influenza_like = profile.get('influenza_like', False)
    moderate_coronavirus = profile.get('moderate_coronavirus', False)
    slow_contact_like = profile.get('slow_contact_like', False)
    contact_like = profile.get('contact_like', False)

    # T1 adaptation is deliberately balanced: the previous V31.1 tuning
    # over-corrected slow/contact archetypes and pushed rolling confirmation
    # too far after the growth reference. Keep noisy raw-growth locks guarded,
    # but allow enough backcasting and single-window evidence to recover the
    # V30-style timing.
    raw_extra = 0.0
    if mpv_like:
        raw_extra += 0.10
    if hospital_like:
        raw_extra += 0.06
    if contact_like and not (mpv_like or hospital_like):
        raw_extra += 0.02
    if zoonotic_like and not (mpv_like or hospital_like or long_zoonotic_like):
        raw_extra -= 0.10
    if long_zoonotic_like:
        raw_extra -= 0.04
    ep.t1_raw_growth_z = min(
        3.05,
        max(1.55, ep.t1_raw_growth_z + 0.12 * burden + raw_extra - (0.10 if fast else 0.0)),
    )
    ep.t1_raw_growth_min_persist_gt = min(
        1.08,
        max(0.35, ep.t1_raw_growth_min_persist_gt + 0.05 * burden + (0.06 if mpv_like else 0.04 if hospital_like else 0.0)),
    )
    ep.t1_raw_growth_confirm_strength = min(
        0.76,
        ep.t1_raw_growth_confirm_strength + 0.02 * burden + (0.03 if mpv_like else 0.02 if hospital_like else 0.0),
    )
    ep.t1_raw_growth_min_abs_frac = min(
        0.09,
        ep.t1_raw_growth_min_abs_frac + 0.006 * burden + (0.010 if mpv_like else 0.006 if hospital_like else 0.0),
    )
    ep.t1_raw_growth_backcast_gt = min(1.35, max(0.55, ep.t1_raw_growth_backcast_gt + 0.12 * burden))
    if prospective and mpv_like:
        ep.t1_raw_growth_backcast_gt = min(max(ep.t1_raw_growth_backcast_gt, 0.55), 0.85)
    elif prospective and hospital_like:
        ep.t1_raw_growth_backcast_gt = min(max(ep.t1_raw_growth_backcast_gt, 0.58), 0.88)
    elif prospective and long_zoonotic_like:
        ep.t1_raw_growth_backcast_gt = min(max(ep.t1_raw_growth_backcast_gt, 0.78), 1.05)
    elif prospective and zoonotic_like:
        ep.t1_raw_growth_backcast_gt = min(max(ep.t1_raw_growth_backcast_gt, 0.68), 1.00)
    elif mpv_like:
        ep.t1_raw_growth_backcast_gt = min(max(ep.t1_raw_growth_backcast_gt, 0.95), 1.20)
    elif hospital_like:
        ep.t1_raw_growth_backcast_gt = min(max(ep.t1_raw_growth_backcast_gt, 0.90), 1.15)
    elif long_zoonotic_like:
        ep.t1_raw_growth_backcast_gt = min(max(ep.t1_raw_growth_backcast_gt, 1.00), 1.25)
    elif zoonotic_like:
        ep.t1_raw_growth_backcast_gt = min(max(ep.t1_raw_growth_backcast_gt, 0.80), 1.20)

    ep.t1_aux_max_lead_gt = max(2.10, ep.t1_aux_max_lead_gt - 0.20 * burden)
    ep.t1_aux_min_strength = min(0.20, ep.t1_aux_min_strength + 0.018 * burden)
    ep.t1_aux_stale_backcast_gt = min(1.20, max(0.35, ep.t1_aux_stale_backcast_gt + 0.03 * burden))
    ep.t1_struct_stale_backcast_gt = min(1.55, max(0.55, ep.t1_struct_stale_backcast_gt + 0.04 * burden))
    if zoonotic_like and not (mpv_like or hospital_like or long_zoonotic_like):
        ep.t1_aux_min_strength = max(0.10, ep.t1_aux_min_strength - 0.035)
        ep.t1_aux_max_lead_gt = min(3.2, ep.t1_aux_max_lead_gt + 0.35)
    elif long_zoonotic_like:
        ep.t1_aux_min_strength = min(max(ep.t1_aux_min_strength, 0.13), 0.18)
        ep.t1_aux_max_lead_gt = min(max(ep.t1_aux_max_lead_gt, 2.6), 3.4)

    ep.t1_estimate_backtrack_gt = min(1.35, max(0.35, ep.t1_estimate_backtrack_gt + 0.10 * burden))
    ep.t1_estimate_max_backtrack_gt = min(2.20, max(0.50, ep.t1_estimate_max_backtrack_gt + 0.22 * burden))
    if prospective and mpv_like:
        ep.t1_estimate_backtrack_gt = min(max(ep.t1_estimate_backtrack_gt, 0.55), 0.82)
        ep.t1_estimate_max_backtrack_gt = min(max(ep.t1_estimate_max_backtrack_gt, 1.05), 1.32)
        ep.t1_aux_stale_backcast_gt = min(ep.t1_aux_stale_backcast_gt, 0.65)
        ep.t1_struct_stale_backcast_gt = min(ep.t1_struct_stale_backcast_gt, 0.90)
    elif prospective and hospital_like:
        ep.t1_estimate_backtrack_gt = min(max(ep.t1_estimate_backtrack_gt, 0.58), 0.88)
        ep.t1_estimate_max_backtrack_gt = min(max(ep.t1_estimate_max_backtrack_gt, 1.10), 1.42)
        ep.t1_aux_stale_backcast_gt = min(ep.t1_aux_stale_backcast_gt, 0.70)
        ep.t1_struct_stale_backcast_gt = min(ep.t1_struct_stale_backcast_gt, 0.95)
    elif prospective and long_zoonotic_like:
        ep.t1_estimate_backtrack_gt = min(max(ep.t1_estimate_backtrack_gt, 0.88), 1.10)
        ep.t1_estimate_max_backtrack_gt = min(max(ep.t1_estimate_max_backtrack_gt, 1.45), 1.95)
    elif prospective and zoonotic_like:
        ep.t1_estimate_backtrack_gt = min(max(ep.t1_estimate_backtrack_gt, 0.75), 1.05)
        ep.t1_estimate_max_backtrack_gt = min(max(ep.t1_estimate_max_backtrack_gt, 1.10), 1.75)
    elif mpv_like:
        ep.t1_estimate_backtrack_gt = min(max(ep.t1_estimate_backtrack_gt, 0.95), 1.20)
        ep.t1_estimate_max_backtrack_gt = min(max(ep.t1_estimate_max_backtrack_gt, 1.60), 2.25)
    elif hospital_like:
        ep.t1_estimate_backtrack_gt = min(max(ep.t1_estimate_backtrack_gt, 0.90), 1.15)
        ep.t1_estimate_max_backtrack_gt = min(max(ep.t1_estimate_max_backtrack_gt, 1.50), 2.10)
    elif long_zoonotic_like:
        ep.t1_estimate_backtrack_gt = min(max(ep.t1_estimate_backtrack_gt, 1.05), 1.30)
        ep.t1_estimate_max_backtrack_gt = min(max(ep.t1_estimate_max_backtrack_gt, 1.80), 2.35)
    elif zoonotic_like:
        ep.t1_estimate_backtrack_gt = min(max(ep.t1_estimate_backtrack_gt, 0.80), 1.20)
        ep.t1_estimate_max_backtrack_gt = min(max(ep.t1_estimate_max_backtrack_gt, 1.20), 1.90)
    ep.t1_estimate_backtrack_min_persist_gt = max(0.70, ep.t1_estimate_backtrack_min_persist_gt - 0.06 * burden)
    if prospective and (mpv_like or hospital_like):
        ep.t1_estimate_history_quantile = min(
            40,
            max(int(getattr(ep, 't1_estimate_history_quantile', 10)), int(28 + 4 * burden)),
        )
    else:
        ep.t1_estimate_history_quantile = min(
            34,
            max(
                int(getattr(ep, 't1_estimate_history_quantile', 10)),
                int(10 + 6 * burden + (4 if (mpv_like or hospital_like) else 0)),
            ),
        )
    ep.t1_first_alert_strength = min(0.28, ep.t1_first_alert_strength + 0.015 * burden)
    ep.t1_first_alert_min_persist_gt = min(0.45, ep.t1_first_alert_min_persist_gt + 0.05 * burden)
    ep.t1_early_confirm_min_persist_gt = min(0.65, ep.t1_early_confirm_min_persist_gt + 0.05 * burden)
    ep.t1_confirm_single_min_persist_gt = min(0.70, ep.t1_confirm_single_min_persist_gt + 0.05 * burden)
    ep.t1_valid_strength_min = min(0.28, ep.t1_valid_strength_min + 0.015 * burden)
    ep.t1_alert_graduation_gt = max(0.65, min(1.20, 1.05 + 0.04 * burden))
    ep.t1_alert_graduation_min_strength = min(0.26, max(0.18, ep.t1_first_alert_strength - 0.02))
    if mpv_like or hospital_like:
        ep.t1_first_alert_strength = min(ep.t1_first_alert_strength + 0.01, 0.26)
        ep.t1_first_alert_min_persist_gt = min(ep.t1_first_alert_min_persist_gt + 0.04, 0.48)
        ep.t1_confirm_single_min_persist_gt = min(ep.t1_confirm_single_min_persist_gt + 0.04, 0.72)
    elif long_zoonotic_like:
        ep.t1_first_alert_strength = min(max(ep.t1_first_alert_strength, 0.20), 0.25)
        ep.t1_valid_strength_min = min(max(ep.t1_valid_strength_min, 0.21), 0.26)
        ep.t1_first_alert_min_persist_gt = min(max(ep.t1_first_alert_min_persist_gt, 0.25), 0.45)
    elif zoonotic_like:
        ep.t1_first_alert_strength = max(0.16, ep.t1_first_alert_strength - 0.03)
        ep.t1_valid_strength_min = max(0.18, ep.t1_valid_strength_min - 0.025)

    if prospective and influenza_like:
        # Influenza-like outbreaks have short GT/SI and compressed growth-to-peak
        # intervals. Keep the structural phase rules shared, but let prospective
        # T1 evidence be accepted with shorter windows and lower raw-growth
        # persistence so H1N1-like validation is not penalized by late locking.
        ep.prospective_min_start = min(
            int(ep.prospective_min_start),
            max(4, int(np.ceil(0.40 * profile['GT'] + min(profile['report_delay'], 1.0)))),
        )
        ep.prospective_min_window_gt = min(ep.prospective_min_window_gt, 1.75)
        ep.prospective_min_window_floor = min(int(ep.prospective_min_window_floor), 6)
        ep.t1_raw_growth_z = min(ep.t1_raw_growth_z, 1.30)
        ep.t1_raw_growth_min_persist_gt = min(ep.t1_raw_growth_min_persist_gt, 0.10)
        ep.t1_raw_growth_confirm_strength = min(ep.t1_raw_growth_confirm_strength, 0.34)
        ep.t1_raw_growth_min_abs_frac = min(ep.t1_raw_growth_min_abs_frac, 0.015)
        ep.t1_first_alert_strength = min(ep.t1_first_alert_strength, 0.08)
        ep.t1_first_alert_min_persist_gt = min(ep.t1_first_alert_min_persist_gt, 0.00)
        ep.t1_early_confirm_strength = min(ep.t1_early_confirm_strength, 0.18)
        ep.t1_early_confirm_min_persist_gt = min(ep.t1_early_confirm_min_persist_gt, 0.00)
        ep.t1_valid_strength_min = min(ep.t1_valid_strength_min, 0.10)
        ep.t1_confirm_single_min_persist_gt = min(ep.t1_confirm_single_min_persist_gt, 0.00)
        ep.t1_single_confirm_strength = min(ep.t1_single_confirm_strength, 0.35)
        ep.t1_fast_confirm_strength = min(ep.t1_fast_confirm_strength, 0.38)
        ep.t1_early_single_strength = min(ep.t1_early_single_strength, 0.35)
        ep.t1_confirm_allow_same_window_as_first_alert = True
        ep.t1_alert_graduation_gt = min(ep.t1_alert_graduation_gt, 0.05)
        ep.t1_alert_graduation_min_strength = min(ep.t1_alert_graduation_min_strength, 0.08)
        ep.t1_estimate_history_quantile = min(int(ep.t1_estimate_history_quantile), 5)
        ep.t1_raw_growth_backcast_gt = max(ep.t1_raw_growth_backcast_gt, 1.45)
        ep.t1_estimate_backtrack_gt = max(ep.t1_estimate_backtrack_gt, 1.35)
        ep.t1_estimate_max_backtrack_gt = max(ep.t1_estimate_max_backtrack_gt, 2.20)
        ep.t1_estimate_backtrack_min_persist_gt = min(ep.t1_estimate_backtrack_min_persist_gt, 0.35)
        ep.t1_flu_min_start_gt_factor = 0.55
        ep.t1_flu_min_start_floor = 4
        ep.t1_flu_candidate_min_gt = 0.20
        ep.t1_flu_candidate_report_cap = 1.0
        ep.t1_flu_raw_stable_rounds = 1
        ep.t1_flu_raw_recent_gt = 1.60
        ep.t1_flu_raw_immediate_strength = 0.30
        ep.prospective_flu_max_t1_to_t2_gt = 8.0
        ep.prospective_flu_max_t1_to_t2_floor = 18
        ep.prospective_flu_max_t1_to_tp_gt = 13.0
        ep.prospective_flu_max_t1_to_tp_floor = 28

    if prospective and mpv_like:
        # MPV-like contact-network curves need guardrails against very early
        # raw-growth locks, but a too-late structural lock also hurts the
        # V30-style rolling validation. Keep raw growth available with stricter
        # evidence and let moderated energy/structure evidence confirm earlier.
        ep.t1_raw_growth_alert_enable = True
        ep.t1_aux_energy_enable = True
        ep.t1_aux_min_strength = min(max(ep.t1_aux_min_strength, 0.14), 0.19)
        ep.t1_aux_max_lead_gt = min(max(ep.t1_aux_max_lead_gt, 2.6), 3.5)
        ep.prospective_min_start = max(
            int(ep.prospective_min_start),
            int(np.ceil(1.85 * profile['GT'] + profile['report_delay'])),
        )
        ep.t1_confirm_allow_single_strong = True
        ep.t1_valid_strength_min = min(max(ep.t1_valid_strength_min, 0.21), 0.25)
        ep.t1_first_alert_strength = min(max(ep.t1_first_alert_strength, 0.21), 0.25)
        ep.t1_early_confirm_strength = min(max(ep.t1_early_confirm_strength, 0.30), 0.33)
        ep.t1_fast_confirm_strength = min(max(ep.t1_fast_confirm_strength, 0.54), 0.58)
        ep.t1_single_confirm_strength = min(max(ep.t1_single_confirm_strength, 0.50), 0.54)
        ep.t1_confirm_single_min_persist_gt = min(max(ep.t1_confirm_single_min_persist_gt, 0.42), 0.58)
        ep.t1_early_confirm_min_persist_gt = min(max(ep.t1_early_confirm_min_persist_gt, 0.38), 0.55)
        ep.t1_first_alert_min_persist_gt = min(max(ep.t1_first_alert_min_persist_gt, 0.20), 0.40)
        ep.t1_alert_graduation_gt = min(ep.t1_alert_graduation_gt, 0.85)
        ep.t1_alert_graduation_min_strength = min(ep.t1_alert_graduation_min_strength, 0.21)
        ep.t1_estimate_history_quantile = min(max(ep.t1_estimate_history_quantile, 30), 40)
        ep.t1_raw_growth_backcast_gt = min(max(ep.t1_raw_growth_backcast_gt, 0.55), 0.85)
        ep.t1_estimate_backtrack_gt = min(max(ep.t1_estimate_backtrack_gt, 0.55), 0.82)
        ep.t1_estimate_max_backtrack_gt = min(max(ep.t1_estimate_max_backtrack_gt, 1.05), 1.32)
        ep.t1_aux_stale_backcast_gt = min(ep.t1_aux_stale_backcast_gt, 0.65)
        ep.t1_struct_stale_backcast_gt = min(ep.t1_struct_stale_backcast_gt, 0.90)
    elif prospective and hospital_like:
        ep.prospective_min_start = max(
            int(ep.prospective_min_start),
            int(np.ceil(1.65 * profile['GT'] + profile['report_delay'])),
        )
        ep.t1_aux_min_strength = min(max(ep.t1_aux_min_strength, 0.12), 0.17)
        ep.t1_aux_max_lead_gt = min(max(ep.t1_aux_max_lead_gt, 2.8), 3.8)
        ep.t1_valid_strength_min = min(max(ep.t1_valid_strength_min, 0.20), 0.24)
        ep.t1_first_alert_strength = min(max(ep.t1_first_alert_strength, 0.20), 0.24)
        ep.t1_early_confirm_min_persist_gt = min(ep.t1_early_confirm_min_persist_gt, 0.42)
        ep.t1_confirm_single_min_persist_gt = min(ep.t1_confirm_single_min_persist_gt, 0.45)
        ep.t1_alert_graduation_gt = min(ep.t1_alert_graduation_gt, 0.80)
        ep.t1_alert_graduation_min_strength = min(ep.t1_alert_graduation_min_strength, 0.20)
        ep.t1_estimate_history_quantile = min(max(ep.t1_estimate_history_quantile, 28), 38)
        ep.t1_raw_growth_backcast_gt = min(max(ep.t1_raw_growth_backcast_gt, 0.58), 0.88)
        ep.t1_estimate_backtrack_gt = min(max(ep.t1_estimate_backtrack_gt, 0.58), 0.88)
        ep.t1_estimate_max_backtrack_gt = min(max(ep.t1_estimate_max_backtrack_gt, 1.10), 1.42)
        ep.t1_aux_stale_backcast_gt = min(ep.t1_aux_stale_backcast_gt, 0.70)
        ep.t1_struct_stale_backcast_gt = min(ep.t1_struct_stale_backcast_gt, 0.95)
    elif prospective and long_zoonotic_like:
        ep.prospective_min_start = min(
            int(ep.prospective_min_start),
            max(10, int(np.ceil(0.95 * profile['GT'] + profile['report_delay']))),
        )
        ep.prospective_min_window_gt = min(ep.prospective_min_window_gt, 2.80)
        ep.prospective_min_window_floor = min(int(ep.prospective_min_window_floor), 12)
        ep.t1_raw_growth_z = min(ep.t1_raw_growth_z, 1.82)
        ep.t1_raw_growth_min_persist_gt = min(ep.t1_raw_growth_min_persist_gt, 0.36)
        ep.t1_raw_growth_confirm_strength = min(ep.t1_raw_growth_confirm_strength, 0.56)
        ep.t1_raw_growth_min_abs_frac = min(ep.t1_raw_growth_min_abs_frac, 0.035)
        ep.t1_valid_strength_min = min(ep.t1_valid_strength_min, 0.16)
        ep.t1_first_alert_strength = min(ep.t1_first_alert_strength, 0.14)
        ep.t1_first_alert_min_persist_gt = min(ep.t1_first_alert_min_persist_gt, 0.00)
        ep.t1_early_confirm_strength = min(ep.t1_early_confirm_strength, 0.24)
        ep.t1_early_confirm_min_persist_gt = min(ep.t1_early_confirm_min_persist_gt, 0.22)
        ep.t1_confirm_single_min_persist_gt = min(ep.t1_confirm_single_min_persist_gt, 0.28)
        ep.t1_single_confirm_strength = min(ep.t1_single_confirm_strength, 0.46)
        ep.t1_alert_graduation_gt = min(ep.t1_alert_graduation_gt, 0.45)
        ep.t1_alert_graduation_min_strength = min(ep.t1_alert_graduation_min_strength, 0.13)
        ep.t1_estimate_history_quantile = min(max(int(ep.t1_estimate_history_quantile), 12), 20)
        ep.t1_struct_stale_backcast_gt = min(max(ep.t1_struct_stale_backcast_gt, 1.05), 1.20)
        ep.t1_estimate_backtrack_gt = min(max(ep.t1_estimate_backtrack_gt, 0.90), 1.10)
        ep.t1_estimate_max_backtrack_gt = min(max(ep.t1_estimate_max_backtrack_gt, 1.45), 1.95)
    elif prospective and zoonotic_like:
        high_severity_zoonotic = (
            'high_severity' in str(profile.get('archetype', '')).lower()
            or 'conservative' in str(profile.get('adaptive_profile', '')).lower()
            or profile['GT'] >= 8.5
        )
        if high_severity_zoonotic:
            ep.prospective_min_start = min(
                int(ep.prospective_min_start),
                max(10, int(np.ceil(0.95 * profile['GT'] + profile['report_delay']))),
            )
            ep.prospective_min_window_gt = min(ep.prospective_min_window_gt, 2.75)
            ep.prospective_min_window_floor = min(int(ep.prospective_min_window_floor), 12)
            ep.t1_raw_growth_z = min(ep.t1_raw_growth_z, 1.78)
            ep.t1_raw_growth_min_persist_gt = min(ep.t1_raw_growth_min_persist_gt, 0.34)
            ep.t1_raw_growth_confirm_strength = min(ep.t1_raw_growth_confirm_strength, 0.54)
            ep.t1_raw_growth_min_abs_frac = min(ep.t1_raw_growth_min_abs_frac, 0.035)
            ep.t1_valid_strength_min = min(ep.t1_valid_strength_min, 0.15)
            ep.t1_first_alert_strength = min(ep.t1_first_alert_strength, 0.13)
            ep.t1_first_alert_min_persist_gt = min(ep.t1_first_alert_min_persist_gt, 0.00)
            ep.t1_early_confirm_strength = min(ep.t1_early_confirm_strength, 0.24)
            ep.t1_early_confirm_min_persist_gt = min(ep.t1_early_confirm_min_persist_gt, 0.20)
            ep.t1_confirm_single_min_persist_gt = min(ep.t1_confirm_single_min_persist_gt, 0.26)
            ep.t1_single_confirm_strength = min(ep.t1_single_confirm_strength, 0.45)
            ep.t1_alert_graduation_gt = min(ep.t1_alert_graduation_gt, 0.42)
            ep.t1_alert_graduation_min_strength = min(ep.t1_alert_graduation_min_strength, 0.12)
            ep.t1_estimate_history_quantile = min(max(int(ep.t1_estimate_history_quantile), 12), 22)
            ep.t1_estimate_backtrack_gt = min(max(ep.t1_estimate_backtrack_gt, 0.85), 1.05)
            ep.t1_estimate_max_backtrack_gt = min(max(ep.t1_estimate_max_backtrack_gt, 1.35), 1.85)
            ep.t1_struct_stale_backcast_gt = min(max(ep.t1_struct_stale_backcast_gt, 1.05), 1.20)
        else:
            ep.t1_valid_strength_min = min(ep.t1_valid_strength_min, 0.21)
            ep.t1_first_alert_strength = min(ep.t1_first_alert_strength, 0.20)
            ep.t1_early_confirm_strength = min(ep.t1_early_confirm_strength, 0.30)
            ep.t1_alert_graduation_gt = min(ep.t1_alert_graduation_gt, 0.75)
            ep.t1_alert_graduation_min_strength = min(ep.t1_alert_graduation_min_strength, 0.18)
            ep.t1_estimate_backtrack_gt = max(ep.t1_estimate_backtrack_gt, 0.75)
            ep.t1_estimate_max_backtrack_gt = max(ep.t1_estimate_max_backtrack_gt, 0.80)
            ep.t1_struct_stale_backcast_gt = max(ep.t1_struct_stale_backcast_gt, 1.05)

    # T2/Tp need enough maturity to avoid early locking, but the earlier V31.1
    # setting was too strict for several slow archetypes and inflated Tp error.
    late_lock_extra = 0.18 if mpv_like else 0.12 if hospital_like else 0.04 if slow_contact_like else 0.0
    ep.prospective_min_window_gt = min(4.05, ep.prospective_min_window_gt + 0.10 * burden + late_lock_extra)
    ep.prospective_min_window_floor = int(max(ep.prospective_min_window_floor, 10 + np.ceil(2 * burden + 2 * late_lock_extra)))
    ep.prospective_t2_min_after_t1_gt = min(2.10, ep.prospective_t2_min_after_t1_gt + 0.10 * burden + 0.55 * late_lock_extra + (0.12 if fast else 0.0))
    ep.prospective_t2_strength_min = min(0.54, ep.prospective_t2_strength_min + 0.012 * burden + 0.018 * late_lock_extra + (0.02 if fast else 0.0))
    ep.prospective_t2_confirm_tol_gt = max(0.80, ep.prospective_t2_confirm_tol_gt - 0.04 * burden)
    ep.prospective_t2_tail_quantile = min(
        0.55,
        max(getattr(ep, 'prospective_t2_tail_quantile', 0.35), 0.35 + 0.025 * burden + 0.06 * late_lock_extra),
    )
    ep.prospective_tp_maturity_gt = min(2.75, ep.prospective_tp_maturity_gt + 0.10 * burden + 0.25 * late_lock_extra)
    ep.prospective_tp_decline_frac = min(0.16, ep.prospective_tp_decline_frac + 0.008 * burden + 0.018 * late_lock_extra)
    ep.prospective_tp_confirm_tol_gt = max(0.95, ep.prospective_tp_confirm_tol_gt - 0.05 * burden)
    if burden >= 1.65:
        ep.prospective_t2_confirm_rounds = max(ep.prospective_t2_confirm_rounds, 3)
        ep.prospective_tp_confirm_rounds = max(ep.prospective_tp_confirm_rounds, 3)
    if mpv_like or hospital_like:
        ep.prospective_t2_confirm_rounds = max(ep.prospective_t2_confirm_rounds, 2)
        ep.prospective_tp_confirm_rounds = max(ep.prospective_tp_confirm_rounds, 2)
    if long_zoonotic_like:
        ep.prospective_t2_confirm_rounds = max(ep.prospective_t2_confirm_rounds, 2)
        ep.prospective_tp_confirm_rounds = max(ep.prospective_tp_confirm_rounds, 2)

    if prospective and mpv_like:
        ep.prospective_t2_min_after_t1_gt = min(max(ep.prospective_t2_min_after_t1_gt, 1.35), 1.75)
        ep.prospective_t2_strength_min = min(max(ep.prospective_t2_strength_min, 0.42), 0.48)
        ep.prospective_t2_tail_quantile = min(max(ep.prospective_t2_tail_quantile, 0.40), 0.50)
        ep.prospective_tp_maturity_gt = min(max(ep.prospective_tp_maturity_gt, 2.05), 2.55)
        ep.prospective_tp_decline_frac = min(max(ep.prospective_tp_decline_frac, 0.10), 0.13)
        ep.prospective_tp_confirm_tol_gt = max(ep.prospective_tp_confirm_tol_gt, 1.25)
        ep.prospective_tp_confirm_rounds = max(ep.prospective_tp_confirm_rounds, 2)
    elif prospective and hospital_like:
        ep.prospective_t2_min_after_t1_gt = min(max(ep.prospective_t2_min_after_t1_gt, 1.40), 1.80)
        ep.prospective_t2_strength_min = min(max(ep.prospective_t2_strength_min, 0.42), 0.48)
        ep.prospective_t2_tail_quantile = min(max(ep.prospective_t2_tail_quantile, 0.40), 0.50)
        ep.prospective_tp_maturity_gt = min(max(ep.prospective_tp_maturity_gt, 2.00), 2.50)
        ep.prospective_tp_decline_frac = min(max(ep.prospective_tp_decline_frac, 0.10), 0.13)
    elif prospective and long_zoonotic_like:
        ep.prospective_min_window_gt = min(max(ep.prospective_min_window_gt, 3.20), 3.85)
        ep.prospective_t2_min_after_t1_gt = min(max(ep.prospective_t2_min_after_t1_gt, 1.40), 1.85)
        ep.prospective_t2_strength_min = min(max(ep.prospective_t2_strength_min, 0.41), 0.47)
        ep.prospective_t2_tail_quantile = min(max(ep.prospective_t2_tail_quantile, 0.39), 0.49)
        ep.prospective_tp_maturity_gt = min(max(ep.prospective_tp_maturity_gt, 2.05), 2.55)
        ep.prospective_tp_decline_frac = min(max(ep.prospective_tp_decline_frac, 0.10), 0.13)
    elif prospective and zoonotic_like:
        ep.prospective_min_window_gt = min(ep.prospective_min_window_gt, 3.45)
        ep.prospective_t2_min_after_t1_gt = min(ep.prospective_t2_min_after_t1_gt, 1.60)
        ep.prospective_t2_strength_min = min(ep.prospective_t2_strength_min, 0.43)
        ep.prospective_t2_tail_quantile = min(ep.prospective_t2_tail_quantile, 0.42)
        ep.prospective_tp_maturity_gt = min(ep.prospective_tp_maturity_gt, 2.05)
        ep.prospective_tp_decline_frac = min(ep.prospective_tp_decline_frac, 0.11)
        ep.prospective_t2_confirm_rounds = min(ep.prospective_t2_confirm_rounds, 3)
        ep.prospective_tp_confirm_rounds = min(ep.prospective_tp_confirm_rounds, 3)
    elif prospective and (influenza_like or moderate_coronavirus):
        ep.prospective_t2_min_after_t1_gt = min(
            ep.prospective_t2_min_after_t1_gt,
            1.25 if influenza_like else 1.75,
        )
        ep.prospective_t2_strength_min = min(ep.prospective_t2_strength_min, 0.40 if influenza_like else 0.46)
        ep.prospective_t2_tail_quantile = min(ep.prospective_t2_tail_quantile, 0.35 if influenza_like else 0.45)
        ep.prospective_t2_confirm_rounds = min(ep.prospective_t2_confirm_rounds, 3)
        if influenza_like:
            ep.prospective_tp_maturity_gt = min(ep.prospective_tp_maturity_gt, 1.55)
            ep.prospective_tp_decline_frac = min(ep.prospective_tp_decline_frac, 0.07)
            ep.prospective_tp_confirm_rounds = min(ep.prospective_tp_confirm_rounds, 2)
            ep.prospective_tp_confirm_tol_gt = max(ep.prospective_tp_confirm_tol_gt, 1.60)

    # When Tp is not stably confirmed, use an archetype-specific historical
    # percentile instead of the final rolling-window peak. This keeps long,
    # noisy contact-network curves from drifting too far right.
    ep.prospective_tp_fallback_quantile = 50.0
    if prospective and (mpv_like or hospital_like or slow_contact_like):
        ep.prospective_tp_fallback_quantile = 40.0
    elif prospective and (long_zoonotic_like or zoonotic_like):
        ep.prospective_tp_fallback_quantile = 45.0
    elif prospective and influenza_like:
        ep.prospective_tp_fallback_quantile = 45.0

    # Retrospective phase segmentation should remain sensitive; conservatism is
    # concentrated in rolling confirmation above.
    retro_burden = min(burden, 1.5)
    retro_contact_bonus = 0.03 if slow_contact_like else 0.0
    retro_contact_bonus += 0.02 if (mpv_like or hospital_like or zoonotic_like) else 0.0
    ep.t1_structure_score_thresh = max(0.39, ep.t1_structure_score_thresh - 0.030 * retro_burden - retro_contact_bonus)
    ep.t1_mid_z_thresh = max(0.70, ep.t1_mid_z_thresh - 0.12 * retro_burden - 0.06 * int(slow_contact_like))
    ep.t1_mid_high_ratio_z = max(0.22, ep.t1_mid_high_ratio_z - 0.08 * retro_burden)
    ep.t1_relaxed_score_thresh = max(0.32, ep.t1_relaxed_score_thresh - 0.025 * retro_burden - 0.02 * int(slow_contact_like))
    ep.t1_relaxed_mid_z_thresh = max(0.62, ep.t1_relaxed_mid_z_thresh - 0.10 * retro_burden)
    ep.t1_min_signal_gate_k = max(1.30, ep.t1_min_signal_gate_k - 0.25 * retro_burden)
    ep.t1_baseline_days = max(8, int(ep.t1_baseline_days * (1.0 - min(0.20, 0.065 * retro_burden))))
    ep.t2_dom_score_thresh = min(0.56, ep.t2_dom_score_thresh + 0.010 * retro_burden)
    ep.t2_low_share_thresh_short = min(0.40, ep.t2_low_share_thresh_short + 0.006 * retro_burden)
    ep.apply_override({
        't2_low_share_thresh_short': ep.t2_low_share_thresh_short,
        't2_dom_score_thresh': ep.t2_dom_score_thresh,
    })

    ep.adaptive_profile = profile['adaptive_profile']
    ep.adaptive_burden = burden
    ep.adaptive_archetype = profile['archetype']
    ep.adaptive_report_delay = profile['report_delay']
    ep.adaptive_R0 = profile['R0']
    ep.adaptive_long_zoonotic_like = profile['long_zoonotic_like']
    return ep


def adaptive_rolling_settings(base_window: int, base_step: int, base_rounds: int,
                              base_tol: int, ep: EpiParams, curve: Dict) -> Dict:
    profile = infer_archetype_adaptive_profile(curve, ep)
    burden = profile['adaptive_burden']
    gt = profile['GT']
    if profile.get('mpv_like'):
        late_lock_extra = 0.18
    elif profile.get('hospital_like'):
        late_lock_extra = 0.14
    elif profile.get('long_zoonotic_like'):
        late_lock_extra = 0.16
    elif profile.get('zoonotic_like'):
        late_lock_extra = 0.0
    elif profile.get('slow_contact_like'):
        late_lock_extra = 0.12
    else:
        late_lock_extra = 0.0
    window_factor = (
        11.5 + 1.0 * burden
        if profile.get('zoonotic_like') and not profile.get('long_zoonotic_like')
        else 12.0 + 1.2 * burden + late_lock_extra
    )
    window_size = int(max(
        base_window,
        np.ceil(window_factor * gt),
        base_window + np.ceil(5 * burden + 3 * late_lock_extra),
    ))
    window_cap = int(max(
        base_window,
        base_window + np.ceil(10 * burden + 5 * late_lock_extra),
    ))
    window_size = int(min(window_size, window_cap))
    if profile.get('mpv_like'):
        confirm_rounds = int(max(base_rounds, 2))
    elif profile.get('hospital_like'):
        confirm_rounds = int(max(base_rounds, 2))
    elif profile.get('long_zoonotic_like'):
        confirm_rounds = int(max(base_rounds, 2))
    elif profile.get('zoonotic_like'):
        confirm_rounds = int(max(base_rounds, 2))
    else:
        confirm_rounds = int(max(base_rounds, 3 if burden >= 0.9 else base_rounds))
    confirm_tol = int(max(base_tol, np.ceil((1.15 + 0.18 * burden) * gt), base_step))
    return dict(window_size=window_size, step=int(base_step), confirm_rounds=confirm_rounds, confirm_tol=confirm_tol)


# ============================================================
#  Part 3. Structural features and diagnostics
# ============================================================

@dataclass
class StructuralFeatures:
    e_high: np.ndarray
    e_mid: np.ndarray
    e_low: np.ndarray
    p_high: np.ndarray
    p_mid: np.ndarray
    p_low: np.ndarray
    log_mh: np.ndarray
    log_lm: np.ndarray
    centroid: np.ndarray
    d_centroid: np.ndarray
    mid_score: np.ndarray
    dom_score: np.ndarray
    total_energy: np.ndarray


@dataclass
class TruthReferences:
    T1_growth: int
    T1_struct: int
    T2_dyn: int
    T2_dom: int
    Tp: int


def build_structural_features(imf1: np.ndarray, ep: EpiParams, smooth_sigma: Optional[float] = None) -> StructuralFeatures:
    smooth_sigma = ep.energy_smooth_sigma if smooth_sigma is None else smooth_sigma
    e_high = compute_cwt_band_energy(imf1, ep.high_band_lo, ep.high_band_hi, smooth_sigma=smooth_sigma)
    e_mid = compute_cwt_band_energy(imf1, ep.mid_band_lo, ep.mid_band_hi, smooth_sigma=smooth_sigma)
    e_low = compute_cwt_band_energy(imf1, ep.low_band_lo, ep.low_band_hi, smooth_sigma=smooth_sigma)

    eps = 1e-10
    total = e_high + e_mid + e_low + eps
    p_high = e_high / total
    p_mid = e_mid / total
    p_low = e_low / total
    log_mh = np.log((e_mid + eps) / (e_high + eps))
    log_lm = np.log((e_low + eps) / (e_mid + eps))

    w_high = 0.5 * (ep.high_band_lo + ep.high_band_hi)
    w_mid = 0.5 * (ep.mid_band_lo + ep.mid_band_hi)
    w_low = 0.5 * (ep.low_band_lo + ep.low_band_hi)
    centroid = (w_high * e_high + w_mid * e_mid + w_low * e_low) / total
    centroid = gaussian_filter1d(centroid, sigma=max(0.5, smooth_sigma * 0.7))
    d_centroid = gaussian_filter1d(np.gradient(centroid), sigma=max(0.5, ep.score_smooth_sigma))

    mid_score = 0.50 * safe_minmax(log_mh) + 0.35 * safe_minmax(p_mid) + 0.15 * safe_minmax(d_centroid)
    dom_score = 0.55 * safe_minmax(log_lm) + 0.30 * safe_minmax(p_low - p_mid) + 0.15 * safe_minmax(d_centroid)
    mid_score = gaussian_filter1d(mid_score, sigma=max(0.5, ep.score_smooth_sigma))
    dom_score = gaussian_filter1d(dom_score, sigma=max(0.5, ep.score_smooth_sigma))

    return StructuralFeatures(
        e_high=e_high, e_mid=e_mid, e_low=e_low,
        p_high=p_high, p_mid=p_mid, p_low=p_low,
        log_mh=log_mh, log_lm=log_lm,
        centroid=centroid, d_centroid=d_centroid,
        mid_score=mid_score, dom_score=dom_score,
        total_energy=total,
    )


class SignalDiagnostics:
    @staticmethod
    def find_signal_start_ref(signal: np.ndarray, ep: EpiParams) -> int:
        signal = np.maximum(np.nan_to_num(signal, nan=0), 0).astype(float)
        N = len(signal)
        pv = np.max(signal)
        if N < 10 or pv < 1.0:
            return 0
        gate_thresh = pv * ep.sig_gate_frac
        roll_w = max(3, int(np.ceil(ep.GT)))
        roll_sig = rolling_mean(signal, roll_w)
        found = None
        for i in range(N - ep.sig_gate_consec):
            if np.all(roll_sig[i:i + ep.sig_gate_consec] >= gate_thresh):
                future_end = min(i + int(ep.GT * 10), N)
                future_mean = np.mean(signal[i:future_end]) if future_end > i else 0.0
                if future_mean >= gate_thresh * 0.5:
                    found = i
                    break
        if found is None:
            return 0
        return int(max(0, found - ep.sig_valid_pre_buf))

    @staticmethod
    def detect_bimodal_takeoff(signal: np.ndarray, ep: EpiParams) -> Optional[int]:
        signal = np.maximum(np.nan_to_num(signal, nan=0), 0).astype(float)
        N = len(signal)
        pv = np.max(signal)
        if N < 20 or pv < 10:
            return None
        peak_pos = int(np.argmax(signal))
        if peak_pos < N * 0.3:
            return None
        smooth = gaussian_filter1d(signal[:peak_pos + 1], sigma=max(3, ep.GT * 1.5))
        slope = np.gradient(smooth)
        consec = max(3, int(np.ceil(ep.GT)))
        for i in range(len(slope) - consec):
            if np.all(slope[i:i + consec] > 0):
                before_end = max(0, i - 5)
                before_start = max(0, before_end - 15)
                before_mean = (np.mean(signal[before_start:before_end]) if before_end > before_start else 0.0) + 1e-10
                after_mean = np.mean(signal[i:min(i + int(ep.GT * 5), N)])
                if after_mean / before_mean > ep.bimodal_ratio:
                    return int(i)
        return None

    @staticmethod
    def diagnose_signal_complexity(curves, label=''):
        smooth_indices = []
        spectral_entropies = []
        weekly_ratios = []
        spike_ratios = []
        for curve in curves:
            sig = np.maximum(np.nan_to_num(curve['I'], nan=0), 0).astype(float)
            N = len(sig)
            if N < 14 or np.max(sig) < 1:
                continue
            d2 = np.gradient(np.gradient(sig))
            smooth_indices.append(float(np.mean(np.abs(d2)) / (np.mean(np.abs(sig)) + 1e-10)))
            fft_p = np.abs(np.fft.rfft(sig - sig.mean())) ** 2
            fft_pn = fft_p / (fft_p.sum() + 1e-10)
            spectral_entropies.append(float(-np.sum(fft_pn * np.log(fft_pn + 1e-10))))
            freqs = np.fft.rfftfreq(N, d=1.0)
            weekly_mask = np.abs(freqs - 1.0 / 7.0) < (0.5 / 7.0)
            weekly_ratios.append(float(fft_p[weekly_mask].sum() / (fft_p.sum() + 1e-10)))
            pk_pos, _ = find_peaks(sig)
            pk_neg, _ = find_peaks(-sig)
            spike_ratios.append(float((len(pk_pos) + len(pk_neg)) / max(N, 1)))
        if not smooth_indices:
            print(f"\n  Signal-complexity diagnostics [{label}]: no valid curves")
            return {}
        sm_med = float(np.median(smooth_indices))
        sp_med = float(np.median(spectral_entropies))
        wr_med = float(np.median(weekly_ratios))
        sr_med = float(np.median(spike_ratios))
        print(f"\n  Signal-complexity diagnostics [{label}] (N={len(smooth_indices)})")
        print(f"    smoothness_index median: {sm_med:.4f}")
        print(f"    spectral_entropy median: {sp_med:.2f}")
        print(f"    weekly_power_ratio median: {wr_med:.3f}")
        print(f"    local_extrema_density median: {sr_med:.3f}")
        return dict(smoothness_index=sm_med, spectral_entropy=sp_med, weekly_power_ratio=wr_med, spike_ratio=sr_med)

    @staticmethod
    def classify_short_alert(signal: np.ndarray, T1: int, T2: int, Tp: int, onset_day: Optional[int] = None) -> str:
        total = float(np.sum(signal))
        peak_val = float(np.max(signal)) if len(signal) else 0.0
        t1_to_tp = Tp - T1
        if total < 50 or peak_val < 5:
            return 'B_sparse'
        if onset_day is not None and onset_day <= 5:
            return 'C_import_driven'
        if t1_to_tp <= 14 and total >= 100:
            return 'A_fast_outbreak'
        return 'D_unclassified'

    @staticmethod
    def classify_intervention_window(T1: int, T2: int, Tp: int, GT: float) -> str:
        alert_days = int(T2 - T1)
        lead_peak = int(Tp - T1)
        gt = max(float(GT), 1e-10)
        if alert_days <= 0 or lead_peak <= 0:
            return 'invalid_or_late'
        if alert_days < max(3, int(np.ceil(gt))):
            return 'too_short_for_action'
        if alert_days < int(np.ceil(2 * gt)):
            return 'narrow_window'
        if lead_peak >= 14 and alert_days >= int(np.ceil(2 * gt)):
            return 'actionable_window'
        return 'moderate_window'

    @staticmethod
    def energy_threshold_baseline_from_result(result: Dict, ep: EpiParams,
                                              search_start: int = 0,
                                              sig_valid_start: Optional[int] = None) -> Dict:
        """V24-style lightweight comparator using the already decomposed structural signal."""
        signal = np.maximum(np.nan_to_num(result.get('signal', []), nan=0), 0).astype(float)
        e_mid = np.asarray(result.get('e_mid', []), dtype=float)
        e_low = np.asarray(result.get('e_low', []), dtype=float)
        trend = np.asarray(result.get('trend', signal), dtype=float)
        N = len(signal)
        if N < 20 or len(e_mid) != N or len(e_low) != N:
            return dict(T1_energy=np.nan, T2_energy=np.nan, Tp_energy=np.nan,
                        alert_days_energy=np.nan, energy_baseline_ok=False)
        Tp = int(np.argmax(trend)) if len(trend) == N else int(result.get('Tp', np.argmax(signal)))
        Tp = max(5, min(Tp, N - 2))
        base_end = min(max(5, int(ep.t1_baseline_days)), max(5, Tp // 3), max(5, N // 5))
        base = e_mid[:base_end]
        bl_mean = float(np.mean(base))
        bl_std = max(float(np.std(base)), robust_mad_std(base), 1e-10)
        thresh = bl_mean + 3.0 * bl_std
        consec = max(2, int(np.ceil(ep.GT)))
        start = max(int(search_start), base_end)
        if sig_valid_start is not None and pd.notna(sig_valid_start):
            start = max(start, int(sig_valid_start) - int(np.ceil(ep.GT)))
        T1 = None
        raw_gate = np.ones(N, dtype=bool)
        if np.max(signal) > 0:
            sm_sig = gaussian_filter1d(signal, sigma=max(1.0, ep.GT * 0.5))
            raw_gate = sm_sig >= max(1.0, np.max(signal) * ep.sig_gate_frac * 0.5)
        for i in range(start, max(start, Tp - consec + 1)):
            if np.all(e_mid[i:i + consec] >= thresh) and np.mean(raw_gate[i:i + consec]) >= 0.5:
                T1 = i
                break
        if T1 is None:
            cand = np.where((e_mid >= thresh) & raw_gate & (np.arange(N) < Tp))[0]
            cand = cand[cand >= start]
            T1 = int(cand[0]) if len(cand) else int(result.get('T1', max(1, Tp // 3)))
        T1 = max(1, min(int(T1), Tp - 2))

        lo_start = min(Tp, T1 + max(3, int(np.ceil(ep.GT))))
        low_seg = e_low[T1:Tp + 1]
        low_peak = float(np.max(low_seg)) if len(low_seg) else 0.0
        T2 = None
        if low_peak > 1e-10:
            low_thr = 0.45 * low_peak
            for i in range(lo_start, Tp):
                if e_low[i] >= low_thr:
                    T2 = i
                    break
        if T2 is None:
            T2 = int(result.get('T2', (T1 + Tp) // 2))
        T2 = max(T1 + 1, min(int(T2), Tp - 1))
        return dict(
            T1_energy=T1, T2_energy=T2, Tp_energy=Tp,
            alert_days_energy=int(T2 - T1),
            T1_energy_delta_vs_struct=int(T1 - int(result.get('T1', T1))),
            T2_energy_delta_vs_struct=int(T2 - int(result.get('T2', T2))),
            energy_baseline_ok=True,
        )


# ============================================================
#  Part 4. Structural phase segmenter
# ============================================================

class StructuralPhaseSegmenter:
    def __init__(self, ep: EpiParams, eemd_trials: int = 50, seed: int = 42):
        self.ep = ep
        self.eemd_trials = eemd_trials
        self.seed = seed

    def _make_eemd(self, seed: Optional[int] = None) -> EEMD:
        return EEMD(
            trials=self.eemd_trials,
            noise_width=0.2,
            seed=self.seed if seed is None else int(seed),
            early_stop_tol=0.02,
            min_trials=max(5, self.eemd_trials // 5),
        )

    @staticmethod
    def decompose_static(signal, t, ep: EpiParams, eemd_obj: Optional[EEMD] = None):
        signal = np.asarray(signal, dtype=float)
        N = len(signal)
        if eemd_obj is None:
            eemd_obj = EEMD(trials=50, noise_width=0.2, seed=42, early_stop_tol=0.02, min_trials=10)
        try:
            IMFs = eemd_obj.eemd(signal.astype(float), np.asarray(t, dtype=float))
            imf1 = IMFs[0].copy()
            trend = IMFs[-1].copy()
            pt = max(2, int(ep.trend_sigma * 5))
            for i in range(1, IMFs.shape[0] - 1):
                zc = np.where(np.diff(np.sign(IMFs[i])))[0]
                dp = N / max(len(zc), 1) * 2 if len(zc) >= 2 else N
                er = np.var(IMFs[i]) / (np.var(signal) + 1e-10)
                if dp > pt or er < 0.02:
                    trend += IMFs[i]
        except Exception:
            trend = gaussian_filter1d(signal.astype(float), sigma=ep.trend_sigma * 2)
            imf1 = signal - trend
        trend = np.maximum(gaussian_filter1d(trend, sigma=ep.trend_sigma), 0)
        return imf1, trend

    def _decompose(self, signal, t, seed: Optional[int] = None):
        eemd_obj = self._make_eemd(seed)
        return self.decompose_static(signal, t, self.ep, eemd_obj)

    @staticmethod
    def find_peak_from_trend_static(trend: np.ndarray, N: int, ep: EpiParams) -> int:
        if np.max(trend) < 1e-10:
            return N // 2
        sm = gaussian_filter1d(trend, sigma=max(1, ep.trend_sigma))
        Tp = int(np.argmax(sm))
        return min(max(Tp, max(5, int(ep.GT * 2))), N - 2)

    def _find_Tp(self, trend, N):
        return self.find_peak_from_trend_static(trend, N, self.ep)

    def _build_raw_gate(self, signal: np.ndarray, baseline_idx: np.ndarray, raw_gate_override: Optional[np.ndarray] = None) -> np.ndarray:
        if raw_gate_override is not None:
            gate = np.asarray(raw_gate_override, dtype=bool)
            if len(gate) == len(signal):
                return gate
        idx = np.asarray(baseline_idx, dtype=int)
        idx = idx[(idx >= 0) & (idx < len(signal))]
        if len(idx) < 3:
            idx = np.arange(0, min(max(3, int(self.ep.t1_baseline_days)), len(signal)))
        sm_sig = gaussian_filter1d(signal.astype(float), sigma=max(1.0, self.ep.GT * 0.5))
        base = sm_sig[idx]
        thr = np.median(base) + self.ep.t1_min_signal_gate_k * robust_mad_std(base)
        return sm_sig >= thr

    def _support_ok(self, cond: np.ndarray, score: np.ndarray, start: Optional[int], horizon: int,
                    support_ratio: float, score_thresh: float) -> bool:
        if start is None:
            return False
        lo = max(0, int(start))
        hi = min(len(cond), lo + max(1, int(horizon)))
        if hi <= lo:
            return False
        support = np.mean(cond[lo:hi])
        mean_score = np.mean(score[lo:hi])
        return bool(support >= support_ratio and mean_score >= score_thresh)

    def _find_T1(self, signal: np.ndarray, feat: StructuralFeatures, Tp: int, N: int, search_start: int = 0,
                 raw_gate_override: Optional[np.ndarray] = None, prospective_mode: bool = False,
                 real_data: bool = False) -> int:
        ep = self.ep
        bl_days = min(ep.t1_baseline_days, max(Tp // 3, 5), max(N // 5, 5))
        bl_days = max(bl_days, 5)
        if search_start > bl_days:
            idx_bl = np.arange(0, min(search_start, max(Tp // 3, bl_days), N))
        else:
            idx_bl = np.arange(0, min(bl_days, N))
        if len(idx_bl) < 3:
            idx_bl = np.arange(0, min(max(5, bl_days), N))

        z_mid = safe_zscore(feat.e_mid, feat.e_mid[idx_bl])
        z_ratio = safe_zscore(feat.log_mh, feat.log_mh[idx_bl])
        z_cent = safe_zscore(feat.centroid, feat.centroid[idx_bl])
        base_pmid = np.median(feat.p_mid[idx_bl]) if len(idx_bl) else np.median(feat.p_mid)
        score = 0.42 * safe_minmax(z_mid) + 0.28 * safe_minmax(z_ratio) + 0.12 * safe_minmax(z_cent) + 0.18 * feat.mid_score
        score = gaussian_filter1d(score, sigma=max(0.5, ep.score_smooth_sigma))
        d_mid_score = gaussian_filter1d(np.gradient(feat.mid_score), sigma=max(0.5, ep.score_smooth_sigma))

        raw_gate = self._build_raw_gate(signal, idx_bl, raw_gate_override)
        consec = ep.t1_consec_real if real_data else ep.t1_consec_days
        end_search = min(max(search_start + consec + 1, Tp - 1), N - 1)

        if prospective_mode:
            strict_mid = ep.t1_mid_z_thresh * ep.t1_prospective_strict_relax
            strict_ratio = ep.t1_mid_high_ratio_z * ep.t1_prospective_ratio_relax
            strict_cent = ep.t1_centroid_z_thresh * ep.t1_prospective_centroid_relax
            strict_score = ep.t1_structure_score_thresh * ep.t1_prospective_score_relax
        else:
            strict_mid = ep.t1_mid_z_thresh
            strict_ratio = ep.t1_mid_high_ratio_z
            strict_cent = ep.t1_centroid_z_thresh
            strict_score = ep.t1_structure_score_thresh

        strict_cond = (z_mid >= strict_mid) & (z_ratio >= strict_ratio) & (z_cent >= strict_cent) & (score >= strict_score) & raw_gate
        strict_idx = first_consecutive(strict_cond, max(search_start, bl_days), end_search, consec)

        relaxed_mid = ep.t1_relaxed_mid_z_thresh * (0.90 if prospective_mode else 1.0)
        relaxed_ratio = ep.t1_relaxed_ratio_z - (0.05 if prospective_mode else 0.0)
        relaxed_cent = ep.t1_relaxed_centroid_z - (0.05 if prospective_mode else 0.0)
        relaxed_score = ep.t1_relaxed_score_thresh * (0.95 if prospective_mode else 1.0)
        horizon = max(consec + ep.t1_lookback_allowance, int(np.ceil(1.2 * ep.GT)))
        relaxed_core = (
            (z_mid >= relaxed_mid)
            & (score >= relaxed_score)
            & raw_gate
            & ((z_ratio >= relaxed_ratio) | (z_cent >= relaxed_cent) | (feat.p_mid >= base_pmid + 0.03))
        )
        relaxed_idx = first_consecutive(relaxed_core, max(search_start, bl_days), end_search, max(2, consec - (1 if prospective_mode else 0)))
        if relaxed_idx is not None and not self._support_ok(relaxed_core, score, relaxed_idx, horizon, ep.t1_relaxed_support_ratio, relaxed_score):
            relaxed_idx = None

        progressive_idx = None
        if prospective_mode:
            pg_n = max(2, int(ep.t1_progressive_rounds))
            pg_core = (score >= ep.t1_progressive_strength_lo) & raw_gate & (z_mid >= 0.25)
            start_pg = max(search_start, bl_days)
            for i in range(start_pg, max(start_pg, end_search - pg_n + 1)):
                seg_score = score[i:i + pg_n]
                if (
                    np.all(pg_core[i:i + pg_n])
                    and np.all(np.diff(seg_score) >= -0.02)
                    and seg_score[-1] >= ep.t1_progressive_strength_hi
                ):
                    follow_hi = min(N, i + pg_n + int(np.ceil(ep.GT)))
                    if follow_hi > i + pg_n and np.mean(score[i + pg_n:follow_hi]) >= ep.t1_progressive_strength_lo:
                        progressive_idx = int(i)
                        break

        prealert_idx = None
        if prospective_mode:
            pre_h = max(ep.t1_prealert_consec + ep.t1_lookback_allowance, int(np.ceil(ep.t1_prealert_followup_gt * ep.GT)))
            prealert_core = (
                raw_gate
                & (z_mid >= ep.t1_prealert_mid_z_thresh)
                & (score >= ep.t1_prealert_score_thresh)
                & ((z_ratio >= ep.t1_prealert_ratio_z_thresh)
                   | (z_cent >= ep.t1_prealert_centroid_z_thresh)
                   | (feat.p_mid >= base_pmid + 0.015)
                   | (d_mid_score > 0))
            )
            prealert_idx = first_consecutive(prealert_core, max(search_start, bl_days), end_search, ep.t1_prealert_consec)
            if prealert_idx is not None and not self._support_ok(prealert_core, score, prealert_idx, pre_h, ep.t1_prealert_support_ratio, ep.t1_prealert_score_thresh):
                prealert_idx = None

        if prospective_mode and prealert_idx is not None:
            followup = max(int(np.ceil(ep.t1_prealert_followup_gt * ep.GT)), ep.t1_lookback_allowance)
            candidates = [x for x in [progressive_idx, relaxed_idx, strict_idx] if x is not None]
            if candidates:
                strong_idx = min(candidates)
                if prealert_idx <= strong_idx <= prealert_idx + followup:
                    return int(prealert_idx)
            local_hi = min(N, prealert_idx + followup)
            if local_hi > prealert_idx:
                local_support = np.mean(prealert_core[prealert_idx:local_hi])
                local_strength = np.mean(score[prealert_idx:local_hi])
                if local_support >= max(0.55, ep.t1_prealert_support_ratio) and local_strength >= ep.t1_prealert_score_thresh + 0.03:
                    return int(prealert_idx)

        if relaxed_idx is not None and strict_idx is not None:
            if relaxed_idx <= strict_idx <= relaxed_idx + max(ep.t1_lookback_allowance, int(np.ceil(ep.GT))):
                return int(relaxed_idx)
            return int(strict_idx)
        if strict_idx is not None:
            return int(strict_idx)
        if relaxed_idx is not None:
            return int(relaxed_idx)
        if progressive_idx is not None:
            return int(progressive_idx)
        if prealert_idx is not None:
            return int(prealert_idx)

        cond2 = (z_mid >= relaxed_mid) & raw_gate & ((z_ratio >= relaxed_ratio) | (feat.p_mid >= base_pmid + 0.02))
        idx2 = first_consecutive(cond2, max(search_start, bl_days), end_search, max(2, consec - 1))
        if idx2 is not None:
            return int(idx2)
        idx = np.where((z_mid > min(relaxed_mid, ep.t1_prealert_mid_z_thresh if prospective_mode else relaxed_mid)) & raw_gate & (np.arange(N) < Tp))[0]
        if len(idx) > 0:
            idx = idx[idx >= max(search_start, bl_days)]
            if len(idx) > 0:
                return int(idx[0])
        return max(1, min(max(search_start, bl_days), Tp - 2))

    def _find_T2(self, feat: StructuralFeatures, T1: int, Tp: int, N: int,
                 low_share_thresh: Optional[float] = None, prospective_mode: bool = False,
                 win_size: Optional[int] = None) -> int:
        ep = self.ep
        if T1 >= Tp - ep.t2_min_gap:
            return min(Tp - 1, T1 + ep.t2_min_gap)
        if prospective_mode and win_size is not None and win_size < ep.t2_min_window_days:
            seg = feat.e_low[T1:Tp + 1]
            if len(seg) >= 3:
                return max(T1 + 1, min(int(np.argmax(seg)) + T1, Tp - 1))
            return max(T1 + 1, min((T1 + Tp) // 2, Tp - 1))

        low_share_thresh = ep.get_t2_low_share_thresh(N) if low_share_thresh is None else low_share_thresh
        if prospective_mode:
            low_share_thresh *= 0.94

        p_low, p_mid = feat.p_low, feat.p_mid
        dom_score = feat.dom_score
        d_low = gaussian_filter1d(np.gradient(feat.e_low), sigma=max(0.5, ep.score_smooth_sigma))
        d_mid = gaussian_filter1d(np.gradient(feat.e_mid), sigma=max(0.5, ep.score_smooth_sigma))
        d2_mid = gaussian_filter1d(np.gradient(d_mid), sigma=max(0.5, ep.score_smooth_sigma))
        ratio_lm = feat.log_lm
        d_cent = gaussian_filter1d(np.gradient(feat.centroid), sigma=max(0.5, ep.score_smooth_sigma))

        start = T1 + ep.t2_min_gap
        end = max(start + 1, Tp)
        strict_dom = ep.t2_dom_score_thresh * (0.92 if prospective_mode else 1.0)
        relaxed_share = low_share_thresh * ep.t2_relaxed_low_share_factor
        relaxed_dom = strict_dom * ep.t2_relaxed_dom_factor
        horizon = max(2, int(np.ceil(ep.GT)))

        strict_cond = (
            (p_low >= low_share_thresh)
            & ((p_low - p_mid) >= ep.t2_dom_switch_margin)
            & (dom_score >= strict_dom)
            & (d_low > 0)
            & (d2_mid <= ep.t2_plateau_thresh)
        )
        strict_idx = next((int(i) for i in range(start, end) if strict_cond[i]), None)

        if Tp > T1 + 3:
            cent_thr = np.percentile(d_cent[max(0, T1):min(N, Tp + 1)], 30)
        else:
            cent_thr = -1e9
        relaxed_cond = (
            (p_low >= relaxed_share)
            & ((p_low - p_mid) >= (ep.t2_dom_switch_margin - 0.02))
            & (dom_score >= relaxed_dom)
            & ((d_low > 0) | (ratio_lm > -0.05))
            & (d_cent >= cent_thr)
        )
        relaxed_idx = None
        for i in range(start, end):
            if relaxed_cond[i]:
                hi = min(end, i + horizon)
                support = np.mean(relaxed_cond[i:hi]) if hi > i else 0.0
                mean_dom = np.mean(dom_score[i:hi]) if hi > i else 0.0
                if support >= ep.t2_support_ratio and mean_dom >= relaxed_dom:
                    relaxed_idx = int(i)
                    break

        if relaxed_idx is not None and strict_idx is not None:
            return int(min(relaxed_idx, strict_idx))
        if strict_idx is not None:
            return int(strict_idx)
        if relaxed_idx is not None:
            return int(relaxed_idx)

        cond2 = (ratio_lm > 0) & (p_low >= relaxed_share) & ((d_low > 0) | (dom_score >= relaxed_dom))
        for i in range(start, end):
            if cond2[i]:
                return int(i)

        seg = feat.e_low[T1:Tp + 1]
        if len(seg) >= 5:
            d_e = np.gradient(seg)
            d_sm = gaussian_filter1d(d_e, sigma=max(1.0, ep.GT * 0.5))
            er = max(np.max(seg) - np.min(seg), 1e-10)
            dn = d_sm / er
            d2 = np.gradient(d_sm)
            for i in range(1, len(seg) - 2):
                if dn[i] > 0 and dn[i] > dn[i + 1] and d2[i] < -ep.t2_plateau_thresh * er and np.mean(np.abs(dn[i + 1:])) < dn[i]:
                    return int(max(T1 + 1, min(T1 + i, Tp - 1)))

        idx = np.where(p_low[T1:Tp] >= relaxed_share)[0]
        if len(idx) > 0:
            return int(max(T1 + 1, min(T1 + idx[0], Tp - 1)))
        return int(max(T1 + 1, min((T1 + Tp) // 2, Tp - 1)))

    def _find_T3(self, trend: np.ndarray, feat: StructuralFeatures, Tp: int, N: int) -> int:
        ep = self.ep
        peak_v = float(trend[Tp]) if 0 <= Tp < N else float(np.max(trend))
        if peak_v < 1e-10:
            return min(N - 1, Tp + max(int(4 * ep.GT), ep.t3_consec_days))
        thresh = peak_v * ep.t3_low_frac
        consec = ep.t3_consec_days
        low_peak = np.max(feat.p_low[Tp:]) if Tp < N else np.max(feat.p_low)
        low_peak = max(low_peak, 1e-10)
        for i in range(Tp + 1, N - consec):
            trend_ok = np.all(trend[i:i + consec] <= thresh)
            low_ok = np.mean(feat.p_low[i:i + consec]) <= low_peak * ep.t3_low_share_decay
            if trend_ok and low_ok:
                return int(i)
        return min(N - 1, Tp + max(int(ep.GT * 4), consec))

    def _validate(self, T1, T2, Tp, T3, N):
        Tp = max(5, min(Tp, N - 2))
        T1 = max(1, min(T1, Tp - 2))
        T2 = max(T1 + 1, min(T2, Tp - 1))
        T3 = max(Tp + 1, min(T3, N - 1))
        return int(T1), int(T2), int(Tp), int(T3)

    def _assess_confidence(self, signal, feat: StructuralFeatures, trend, T1, T2, Tp, T3, N):
        bl_end = max(3, min(T1, 20))
        base_sig = signal[:bl_end] if bl_end > 0 else signal[:3]
        pv = float(np.max(trend))
        base_mu = float(np.median(base_sig))
        base_sd = robust_mad_std(base_sig)
        snr = (pv - base_mu) / max(base_sd, 1e-10)
        t1_strength = float(np.median(feat.mid_score[max(0, T1 - 1): min(N, T1 + 2)]))
        t2_strength = float(np.median(feat.dom_score[max(0, T2 - 1): min(N, T2 + 2)]))
        dom_gap = float(feat.p_low[T2] - feat.p_mid[T2]) if 0 <= T2 < N else 0.0
        if t1_strength > 0.62 or (t1_strength > 0.46 and snr > 5.0):
            t1_conf = 'high'
        elif t1_strength > 0.42 or snr > 3.0:
            t1_conf = 'medium'
        else:
            t1_conf = 'low'
        if (t2_strength > 0.62 and dom_gap > 0.02) or (t2_strength > 0.50 and snr > 6.0 and dom_gap > 0):
            t2_conf = 'high'
        elif t2_strength > 0.42 or (dom_gap > 0 and snr > 3.0):
            t2_conf = 'medium'
        else:
            t2_conf = 'low'
        t3_conf = 'high' if trend[T3] <= trend[Tp] * self.ep.t3_low_frac else 'medium' if trend[T3] <= trend[Tp] * (self.ep.t3_low_frac * 1.8) else 'low'
        ok = sum(1 for x in [t1_conf, t2_conf, t3_conf] if x in ('high', 'medium'))
        overall = 'high' if ok == 3 and snr > 3 and t1_strength > 0.46 else 'medium' if ok >= 2 and (snr > 2 or t1_strength > 0.38) else 'low'
        return dict(T1=t1_conf, T2=t2_conf, T3=t3_conf, overall=overall,
                    snr=float(snr), t1_strength=t1_strength, t2_strength=t2_strength,
                    dom_gap=dom_gap, alert_days=int(T2 - T1))

    def _point_intervals(self, trend: np.ndarray, feat: StructuralFeatures, T1: int, T2: int, Tp: int, T3: int) -> Dict:
        ep = self.ep
        T1_lo, T1_hi = local_support_interval(feat.mid_score, T1, frac=0.92, max_radius=max(5, int(np.ceil(2 * ep.GT))), hard_floor=ep.t1_relaxed_score_thresh)
        T2_lo, T2_hi = local_support_interval(feat.dom_score, T2, frac=0.92, max_radius=max(5, int(np.ceil(2.5 * ep.GT))), hard_floor=ep.t2_dom_score_thresh * ep.t2_relaxed_dom_factor)
        peak_frac = 0.97
        Tp_lo, Tp_hi = local_support_interval(trend / (np.max(trend) + 1e-10), Tp, frac=peak_frac, max_radius=max(5, int(np.ceil(2 * ep.GT))))
        T3_lo = T3
        T3_hi = min(len(trend) - 1, T3 + ep.t3_consec_days - 1)
        return dict(T1=(T1_lo, T1_hi), T2=(T2_lo, T2_hi), Tp=(Tp_lo, Tp_hi), T3=(T3_lo, T3_hi))

    def segment(self, signal, t=None, city_name: str = '', prospective_mode: bool = False, search_start: int = 0,
                raw_gate_override: Optional[np.ndarray] = None, low_share_thresh_override: Optional[float] = None,
                real_data: bool = False, eemd_seed: Optional[int] = None,
                current_window_size: Optional[int] = None):
        signal = np.maximum(np.nan_to_num(signal, nan=0), 0).astype(float)
        N = len(signal)
        if t is None:
            t = np.arange(N, dtype=float)
        if N < 20 or np.max(signal) < 1:
            return self._empty(N, city_name)
        imf1, trend = self._decompose(signal, t, seed=eemd_seed)
        feat = build_structural_features(imf1, self.ep, smooth_sigma=self.ep.energy_smooth_sigma)
        Tp = self._find_Tp(trend, N)
        T1 = self._find_T1(signal, feat, Tp, N, search_start=search_start, raw_gate_override=raw_gate_override,
                           prospective_mode=prospective_mode, real_data=real_data)
        T2 = self._find_T2(feat, T1, Tp, N, low_share_thresh_override, prospective_mode,
                           win_size=current_window_size)
        T3 = self._find_T3(trend, feat, Tp, N)
        T1, T2, Tp, T3 = self._validate(T1, T2, Tp, T3, N)
        conf = self._assess_confidence(signal, feat, trend, T1, T2, Tp, T3, N)
        intervals = self._point_intervals(trend, feat, T1, T2, Tp, T3)
        phases = np.zeros(N, dtype=int)
        phases[T1:T2] = 1
        phases[T2:Tp] = 2
        phases[Tp:T3] = 3
        phases[T3:] = 4
        return dict(
            city=city_name, T1=T1, T2=T2, Tp=Tp, T3=T3, confidence=conf, signal=signal, t=t, N=N,
            phases=phases, imf1=imf1, trend=trend, e_high=feat.e_high, e_mid=feat.e_mid, e_low=feat.e_low,
            p_high=feat.p_high, p_mid=feat.p_mid, p_low=feat.p_low, log_mh=feat.log_mh, log_lm=feat.log_lm,
            centroid=feat.centroid, mid_score=feat.mid_score, dom_score=feat.dom_score,
            intervals=intervals, prospective_mode=prospective_mode, real_data=real_data
        )

    def segment_from_cache(self, cache_entry: Dict, city_name: str = '', prospective_mode: bool = False):
        signal = cache_entry['signal']
        t = cache_entry['t']
        N = cache_entry['N']
        imf1 = cache_entry['imf1']
        trend = cache_entry['trend']
        Tp = cache_entry['Tp']
        feat = StructuralFeatures(
            e_high=cache_entry['e_high_raw'], e_mid=cache_entry['e_mid_raw'], e_low=cache_entry['e_low_raw'],
            p_high=cache_entry['p_high_raw'], p_mid=cache_entry['p_mid_raw'], p_low=cache_entry['p_low_raw'],
            log_mh=cache_entry['log_mh_raw'], log_lm=cache_entry['log_lm_raw'], centroid=cache_entry['centroid_raw'],
            d_centroid=np.gradient(cache_entry['centroid_raw']), mid_score=cache_entry['mid_score_raw'],
            dom_score=cache_entry['dom_score_raw'], total_energy=cache_entry['e_high_raw'] + cache_entry['e_mid_raw'] + cache_entry['e_low_raw']
        )
        T1 = self._find_T1(signal, feat, Tp, N)
        T2 = self._find_T2(feat, T1, Tp, N, prospective_mode=prospective_mode)
        T3 = self._find_T3(trend, feat, Tp, N)
        T1, T2, Tp, T3 = self._validate(T1, T2, Tp, T3, N)
        conf = self._assess_confidence(signal, feat, trend, T1, T2, Tp, T3, N)
        intervals = self._point_intervals(trend, feat, T1, T2, Tp, T3)
        return dict(T1=T1, T2=T2, Tp=Tp, T3=T3, confidence=conf, intervals=intervals, N=N, city=city_name)

    def segment_ensemble(self, signal, t=None, city_name: str = '', n_runs: int = 8,
                         prospective_mode: bool = False, search_start: int = 0,
                         raw_gate_override: Optional[np.ndarray] = None,
                         low_share_thresh_override: Optional[float] = None,
                         real_data: bool = False, seed0: int = 1000):
        base = self.segment(signal, t=t, city_name=city_name, prospective_mode=prospective_mode,
                            search_start=search_start, raw_gate_override=raw_gate_override,
                            low_share_thresh_override=low_share_thresh_override, real_data=real_data,
                            eemd_seed=self.seed)
        pts = {'T1': [], 'T2': [], 'Tp': [], 'T3': []}
        for k in range(max(1, n_runs)):
            try:
                r = self.segment(signal, t=t, city_name=city_name, prospective_mode=prospective_mode,
                                 search_start=search_start, raw_gate_override=raw_gate_override,
                                 low_share_thresh_override=low_share_thresh_override, real_data=real_data,
                                 eemd_seed=seed0 + k)
                for nm in pts:
                    pts[nm].append(r[nm])
            except Exception:
                continue
        ensemble_band = {}
        for nm, vals in pts.items():
            ensemble_band[nm] = percentile_band(vals, 10, 90) if vals else (np.nan, np.nan)
        base['ensemble_intervals'] = ensemble_band
        base['ensemble_points'] = pts
        return base

    def _empty(self, N: int, city_name: str = ''):
        return dict(
            city=city_name, T1=N - 1, T2=N - 1, Tp=N - 1, T3=N - 1,
            confidence=dict(T1='none', T2='none', T3='none', overall='none', snr=0, t1_strength=0, t2_strength=0, dom_gap=0, alert_days=0),
            signal=np.zeros(N), t=np.arange(N), N=N, phases=np.zeros(N, dtype=int), imf1=np.zeros(N), trend=np.zeros(N),
            e_high=np.zeros(N), e_mid=np.zeros(N), e_low=np.zeros(N), p_high=np.zeros(N), p_mid=np.zeros(N), p_low=np.zeros(N),
            log_mh=np.zeros(N), log_lm=np.zeros(N), centroid=np.zeros(N), mid_score=np.zeros(N), dom_score=np.zeros(N),
            intervals=dict(T1=(N - 1, N - 1), T2=(N - 1, N - 1), Tp=(N - 1, N - 1), T3=(N - 1, N - 1))
        )


# ============================================================
#  Part 5. Cache
# ============================================================

class EEMDCWTCache:
    def __init__(self, cache_path=CACHE_PATH, eemd_trials: int = 10):
        self.cache_path = cache_path
        self.eemd_trials = eemd_trials
        self.cache: Dict[int, Dict] = {}

    def build(self, curves: List[Dict], ep: EpiParams, force_rebuild: bool = False):
        if not force_rebuild and os.path.exists(self.cache_path):
            print(f"  Cache found: {self.cache_path}")
            self._load(curves)
            return
        print(f"\n  Precomputing EEMD+CWT structural cache (N={len(curves)})...")
        t0 = time.time()
        eemd_obj = EEMD(trials=self.eemd_trials, noise_width=0.2, seed=42, early_stop_tol=0.02, min_trials=5)
        for idx, curve in enumerate(curves):
            signal = np.maximum(np.nan_to_num(curve['I'], nan=0), 0).astype(float)
            t = curve.get('t', np.arange(len(signal), dtype=float))
            N = len(signal)
            curve_ep = build_archetype_adaptive_ep(ep, curve, prospective=False)
            imf1, trend = StructuralPhaseSegmenter.decompose_static(signal, t, curve_ep, eemd_obj)
            features = build_structural_features(imf1, curve_ep, smooth_sigma=curve_ep.energy_smooth_sigma)
            Tp = StructuralPhaseSegmenter.find_peak_from_trend_static(trend, N, curve_ep)
            self.cache[idx] = dict(
                signal=signal, t=t, N=N, imf1=imf1, trend=trend,
                e_high_raw=features.e_high, e_mid_raw=features.e_mid, e_low_raw=features.e_low,
                p_high_raw=features.p_high, p_mid_raw=features.p_mid, p_low_raw=features.p_low,
                log_mh_raw=features.log_mh, log_lm_raw=features.log_lm,
                centroid_raw=features.centroid, mid_score_raw=features.mid_score, dom_score_raw=features.dom_score,
                Tp=Tp, adaptive_profile=getattr(curve_ep, 'adaptive_profile', 'standard'),
                adaptive_burden=float(getattr(curve_ep, 'adaptive_burden', 0.0)),
            )
            if (idx + 1) % 10 == 0:
                el = time.time() - t0
                eta = el / (idx + 1) * (len(curves) - idx - 1)
                print(f"  [{idx+1}/{len(curves)}] {el:.1f}s / ETA {eta:.1f}s")
        try:
            save_dict = {}
            for idx, entry in self.cache.items():
                for k, v in entry.items():
                    key = f"{idx}_{k}"
                    save_dict[key] = v if isinstance(v, np.ndarray) else np.array([v])
            np.savez_compressed(self.cache_path, **save_dict)
            print(f"  Cache saved: {self.cache_path}")
        except Exception as e:
            print(f"  Cache save failed: {e}")
        print(f"  Precompute done, elapsed={time.time()-t0:.1f}s")

    def _load(self, curves: List[Dict]):
        try:
            data = np.load(self.cache_path, allow_pickle=True)
            for idx, curve in enumerate(curves):
                signal = np.maximum(np.nan_to_num(curve['I'], nan=0), 0).astype(float)
                entry = {'signal': signal, 't': curve.get('t', np.arange(len(signal), dtype=float))}
                for k in ['imf1', 'trend', 'e_high_raw', 'e_mid_raw', 'e_low_raw', 'p_high_raw', 'p_mid_raw', 'p_low_raw',
                          'log_mh_raw', 'log_lm_raw', 'centroid_raw', 'mid_score_raw', 'dom_score_raw']:
                    key = f"{idx}_{k}"
                    entry[k] = data[key] if key in data else np.zeros(len(signal))
                entry['Tp'] = int(data[f"{idx}_Tp"][0]) if f"{idx}_Tp" in data else 0
                entry['N'] = int(data[f"{idx}_N"][0]) if f"{idx}_N" in data else len(signal)
                self.cache[idx] = entry
            print(f"  Cache loaded: {len(self.cache)} entries")
        except Exception as e:
            print(f"  Cache load failed ({e}); will rebuild")
            self.cache = {}

    def get(self, idx: int):
        return self.cache[idx]

    def is_ready(self) -> bool:
        return len(self.cache) > 0


# ============================================================
#  Part 6. SIR simulator and dual truth references
# ============================================================

class SIRSimulator:
    ARCHETYPE_SCENARIOS = [
        dict(
            archetype='high_transmissibility_coronavirus',
            label='archetype_high_transmissibility_coronavirus',
            route='respiratory',
            R0_range=(5.5, 9.0),
            GT_range=(2.5, 4.5),
            SI_range=(3.0, 4.8),
            report_delay_range=(1, 4),
            noise_std_range=(0.10, 0.28),
            structural_noise_profile='fast_respiratory',
            weight=0.18,
        ),
        dict(
            archetype='moderate_coronavirus',
            label='archetype_moderate_coronavirus',
            route='respiratory',
            R0_range=(1.2, 2.5),
            GT_range=(4.5, 8.5),
            SI_range=(5.5, 9.0),
            report_delay_range=(1, 5),
            noise_std_range=(0.12, 0.32),
            structural_noise_profile='respiratory_moderate',
            weight=0.16,
        ),
        dict(
            archetype='influenza_like',
            label='archetype_influenza_like',
            route='respiratory',
            R0_range=(1.3, 2.4),
            GT_range=(1.8, 3.5),
            SI_range=(2.2, 3.5),
            report_delay_range=(1, 4),
            noise_std_range=(0.18, 0.40),
            structural_noise_profile='fast_respiratory',
            weight=0.17,
        ),
        dict(
            archetype='high_severity_contact_zoonotic',
            label='archetype_high_severity_contact_zoonotic',
            route='contact_or_droplet',
            R0_range=(1.1, 2.8),
            GT_range=(7.0, 14.0),
            SI_range=(9.0, 16.0),
            report_delay_range=(3, 9),
            noise_std_range=(0.25, 0.55),
            structural_noise_profile='reporting_challenged',
            weight=0.17,
        ),
        dict(
            archetype='mers_like_amplification',
            label='archetype_mers_like_amplification',
            route='healthcare_amplified',
            R0_range=(0.9, 2.2),
            GT_range=(7.0, 16.0),
            SI_range=(9.0, 18.0),
            report_delay_range=(3, 10),
            noise_std_range=(0.20, 0.50),
            structural_noise_profile='hospital_amplified',
            weight=0.17,
        ),
        dict(
            archetype='mpv_like_contact_network',
            label='archetype_mpv_like_contact_network',
            route='close_contact_network',
            R0_range=(0.9, 1.8),
            GT_range=(6.0, 11.0),
            SI_range=(8.0, 14.0),
            report_delay_range=(3, 8),
            noise_std_range=(0.20, 0.45),
            structural_noise_profile='contact_network',
            weight=0.15,
        ),
    ]

    STRUCTURED_NOISE_PROFILES = {
        'standard': dict(weekly_amp=(0.05, 0.20), shift_prob=0.45, shift_factor=(0.75, 1.30),
                         pulse_prob=0.40, pulse_window_frac=0.35, pulse_amp=(0.01, 0.05),
                         spike_lambda_scale=1.0, spike_amp=(1.5, 4.0), plateau_prob=0.40,
                         plateau_scale=(1.5, 4.0), mult_noise=(0.03, 0.10), additive_scale=0.03),
        'fast_respiratory': dict(weekly_amp=(0.04, 0.16), shift_prob=0.35, shift_factor=(0.80, 1.20),
                                 pulse_prob=0.30, pulse_window_frac=0.25, pulse_amp=(0.005, 0.035),
                                 spike_lambda_scale=0.8, spike_amp=(1.3, 3.2), plateau_prob=0.25,
                                 plateau_scale=(1.0, 2.5), mult_noise=(0.03, 0.08), additive_scale=0.02),
        'respiratory_moderate': dict(weekly_amp=(0.05, 0.18), shift_prob=0.40, shift_factor=(0.80, 1.25),
                                     pulse_prob=0.35, pulse_window_frac=0.35, pulse_amp=(0.01, 0.045),
                                     spike_lambda_scale=0.9, spike_amp=(1.3, 3.6), plateau_prob=0.35,
                                     plateau_scale=(1.2, 3.0), mult_noise=(0.03, 0.09), additive_scale=0.025),
        'reporting_challenged': dict(weekly_amp=(0.08, 0.25), shift_prob=0.60, shift_factor=(0.60, 1.55),
                                     pulse_prob=0.55, pulse_window_frac=0.45, pulse_amp=(0.02, 0.08),
                                     spike_lambda_scale=1.4, spike_amp=(1.5, 4.8), plateau_prob=0.55,
                                     plateau_scale=(2.0, 5.0), mult_noise=(0.06, 0.15), additive_scale=0.05),
        'hospital_amplified': dict(weekly_amp=(0.05, 0.20), shift_prob=0.50, shift_factor=(0.65, 1.45),
                                   pulse_prob=0.60, pulse_window_frac=0.60, pulse_amp=(0.02, 0.10),
                                   spike_lambda_scale=1.8, spike_amp=(1.8, 6.0), plateau_prob=0.45,
                                   plateau_scale=(1.5, 4.5), mult_noise=(0.05, 0.14), additive_scale=0.04),
        'contact_network': dict(weekly_amp=(0.04, 0.18), shift_prob=0.45, shift_factor=(0.70, 1.35),
                                pulse_prob=0.70, pulse_window_frac=0.70, pulse_amp=(0.015, 0.08),
                                spike_lambda_scale=1.6, spike_amp=(1.6, 5.0), plateau_prob=0.50,
                                plateau_scale=(2.0, 5.0), mult_noise=(0.05, 0.13), additive_scale=0.04),
    }

    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)

    @staticmethod
    def _format_range(bounds) -> str:
        return f"{bounds[0]}-{bounds[1]}"

    @staticmethod
    def _allocate_counts_from_weights(scenarios, n_curves: int) -> List[int]:
        n_curves = max(0, int(n_curves))
        if n_curves == 0:
            return [0 for _ in scenarios]
        weights = np.array([float(s.get('weight', 1.0)) for s in scenarios], dtype=float)
        weights = weights / max(float(weights.sum()), 1e-10)
        raw = weights * n_curves
        counts = np.floor(raw).astype(int)
        remain = int(n_curves - counts.sum())
        if remain > 0:
            order = np.argsort(-(raw - counts))
            for idx in order[:remain]:
                counts[idx] += 1
        if n_curves >= len(counts):
            for idx in np.where(counts == 0)[0]:
                donor = int(np.argmax(counts))
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[idx] = 1
        return [int(x) for x in counts]

    @classmethod
    def archetype_design_table(cls, default_n: int = 180) -> pd.DataFrame:
        counts = cls._allocate_counts_from_weights(cls.ARCHETYPE_SCENARIOS, default_n)
        rows = []
        for sc, n_sc in zip(cls.ARCHETYPE_SCENARIOS, counts):
            rows.append(dict(
                archetype=sc['archetype'],
                R=cls._format_range(sc['R0_range']),
                GT=cls._format_range(sc['GT_range']),
                SI=cls._format_range(sc['SI_range']),
                report_delay=cls._format_range(sc['report_delay_range']),
                noise=cls._format_range(sc['noise_std_range']),
                structural_noise=sc['structural_noise_profile'],
                scenario_weight=float(sc['weight']),
                sample_n=int(n_sc),
            ))
        return pd.DataFrame(rows)

    def _sample_range(self, bounds):
        return float(self.rng.uniform(float(bounds[0]), float(bounds[1])))

    def _sample_int_range(self, bounds):
        lo = int(bounds[0])
        hi = int(bounds[1])
        return int(self.rng.randint(lo, max(lo + 1, hi + 1)))

    @staticmethod
    def _sir_ode(y, t, beta, gamma, N):
        S, I, R = y
        dS = -beta * S * I / N
        dI = beta * S * I / N - gamma * I
        dR = gamma * I
        return [dS, dI, dR]

    def _add_structured_noise(self, signal, GT, peak_val, profile='standard'):
        N = len(signal)
        result = signal.copy().astype(float)
        cfg = self.STRUCTURED_NOISE_PROFILES.get(profile, self.STRUCTURED_NOISE_PROFILES['standard'])

        # Weekly reporting oscillation.
        amp = self.rng.uniform(*cfg['weekly_amp'])
        phase = self.rng.uniform(0, 2 * np.pi)
        result *= (1.0 + amp * np.sin(2 * np.pi * np.arange(N) / 7.0 + phase))

        # Reporting-system shift.
        if self.rng.random() < cfg['shift_prob']:
            lo_k = max(5, int(N * 0.15))
            hi_k = max(lo_k + 1, int(N * 0.75))
            k = self.rng.randint(lo_k, hi_k)
            factor = self.rng.uniform(*cfg['shift_factor'])
            result[k:] *= factor

        # Imported/local pulse.
        if self.rng.random() < cfg['pulse_prob']:
            center = self.rng.randint(0, max(1, int(N * cfg['pulse_window_frac'])))
            width = max(1, int(self.rng.uniform(1, max(2, GT))))
            pulse = np.exp(-0.5 * ((np.arange(N) - center) / max(width, 1)) ** 2)
            result += pulse * self.rng.uniform(*cfg['pulse_amp']) * max(peak_val, 1.0)

        # Short spikes.
        n_spikes = self.rng.poisson(max(1, int(N / 30 * cfg.get('spike_lambda_scale', 1.0))))
        for _ in range(n_spikes):
            pos = self.rng.randint(0, N)
            amp_s = self.rng.uniform(*cfg['spike_amp'])
            width = max(1, int(self.rng.uniform(1, 3)))
            lo = max(0, pos - width)
            hi = min(N, pos + width + 1)
            result[lo:hi] *= amp_s

        # Plateau/blunting before the peak.
        if self.rng.random() < cfg['plateau_prob']:
            pk = int(np.argmax(result))
            st = max(1, int(pk * self.rng.uniform(0.35, 0.7)))
            ln = int(GT * self.rng.uniform(*cfg['plateau_scale']))
            result[st:min(N, st + ln)] *= self.rng.uniform(0.85, 0.98)

        # Multiplicative noise plus right-skewed additive noise.
        noise = np.maximum(
            self.rng.normal(1.0, self.rng.uniform(*cfg['mult_noise']), N),
            0.01,
        )
        result *= noise
        result += self.rng.exponential(scale=max(peak_val * cfg['additive_scale'], 0.5), size=N)
        return np.maximum(result, 0)

    @staticmethod
    def _growth_truth_from_clean_curve(truth_I, GT, sigma=3):
        N = len(truth_I)
        smooth_I = np.maximum(gaussian_filter1d(truth_I.astype(float), sigma=sigma), 0)
        truth_Tp = max(int(np.argmax(smooth_I)), max(10, int(GT * 2)))
        d2 = np.gradient(np.gradient(smooth_I))
        truth_T2_dyn = truth_Tp // 2
        for i in range(truth_Tp - 1, max(5, truth_Tp // 4), -1):
            if d2[i - 1] >= 0 and d2[i] < 0:
                truth_T2_dyn = i
                break
        log_g = np.zeros(N)
        for i in range(1, N):
            if smooth_I[i - 1] > 0.5:
                log_g[i] = np.log(max(smooth_I[i], 0.1)) - np.log(max(smooth_I[i - 1], 0.1))
        bl_end = max(max(7, int(GT * 2)), min(truth_Tp // 5, 20))
        bl_end = min(bl_end, truth_T2_dyn - 5)
        bl_end = max(bl_end, 5)
        bl_mean = np.mean(log_g[:bl_end])
        bl_std = max(np.std(log_g[:bl_end]), 0.01)
        window = max(3, int(GT))
        abs_thr = max(smooth_I[truth_Tp] * 0.02, 0.5)
        truth_T1_growth = max(bl_end, truth_T2_dyn // 3)
        for i in range(bl_end, truth_T2_dyn - window + 1):
            if np.all(log_g[i:i + window] > bl_mean + 2 * bl_std) and smooth_I[i] > abs_thr:
                truth_T1_growth = i
                break
        truth_T1_growth = max(bl_end, min(truth_T1_growth, truth_T2_dyn - 3))
        truth_T2_dyn = max(truth_T1_growth + 3, min(truth_T2_dyn, truth_Tp - 2))
        return int(truth_T1_growth), int(truth_T2_dyn), int(truth_Tp)

    @staticmethod
    def _structural_refs_from_clean_curve(truth_I, GT, sigma=2.0):
        clean = np.maximum(gaussian_filter1d(np.asarray(truth_I, dtype=float), sigma=max(1.0, sigma)), 0)
        ep_tmp = EpiParams(preset='omicron')
        ep_tmp.apply_override({'GT': float(GT), 'SI': float(GT), 'T_report': 2.0, 'T_week': 7.0})
        ep_tmp.refresh_derived()
        ep_tmp.apply_override({
            'GT': float(GT),
            'SI': float(GT),
            't1_structure_score_thresh': 0.50,
            't1_mid_z_thresh': 1.0,
            't2_dom_score_thresh': 0.45,
            't2_low_share_thresh_short': 0.30,
            't2_low_share_thresh_medium': 0.34,
            't2_low_share_thresh_long': 0.38,
        })
        seg = StructuralPhaseSegmenter(ep_tmp, eemd_trials=20, seed=7)
        try:
            r = seg.segment(clean, np.arange(len(clean), dtype=float), prospective_mode=False, real_data=False)
            return int(r['T1']), int(r['T2']), int(r['Tp'])
        except Exception:
            tp = int(np.argmax(clean)) if len(clean) else 0
            return max(1, tp // 4), max(2, tp // 2), tp

    def _compute_truth_references(self, truth_I, GT, sigma=3):
        T1_growth, T2_dyn, Tp = self._growth_truth_from_clean_curve(truth_I, GT, sigma=sigma)
        T1_struct, T2_dom, Tp_struct = self._structural_refs_from_clean_curve(truth_I, GT, sigma=max(1.0, GT * 0.35))
        Tp = int(np.median([Tp, Tp_struct])) if abs(Tp - Tp_struct) <= max(2, int(GT)) else int(Tp)
        T1_struct = max(1, min(T1_struct, min(T2_dyn, T2_dom) - 1 if min(T2_dyn, T2_dom) > 1 else T1_struct))
        T2_dom = max(T1_struct + 1, min(T2_dom, Tp - 1))
        return TruthReferences(T1_growth=T1_growth, T1_struct=T1_struct, T2_dyn=T2_dyn, T2_dom=T2_dom, Tp=Tp)

    def generate_sir_curve(self, days=200, R0=2.5, GT=5.0, N=100000, I0=5,
                           noise_std=0.2, report_delay=0, add_structured_noise=True,
                           structural_noise_profile='standard', SI=None, archetype=None):
        gamma = 1.0 / GT
        beta = R0 * gamma
        y0 = [N - I0, float(I0), 0.0]
        sim_days = max(days, int(GT * 25))
        t_arr = np.arange(0, sim_days, 1.0)
        sol = odeint(self._sir_ode, y0, t_arr, args=(beta, gamma, N))
        S, _, _ = sol.T
        new_inf = np.maximum(-np.diff(S, prepend=S[0]), 0)
        if report_delay > 0:
            kernel = np.ones(report_delay) / report_delay
            convolved = np.convolve(new_inf, kernel, mode='full')[:len(new_inf)]
            reported = np.concatenate([np.zeros(report_delay), convolved[:max(0, len(new_inf) - report_delay)]])[:len(new_inf)]
        else:
            reported = new_inf.copy()
        noise = np.maximum(self.rng.normal(1.0, noise_std, len(reported)), 0.01)
        dI_obs = np.maximum(reported * noise, 0)
        if add_structured_noise:
            pv = float(np.max(dI_obs)) if np.max(dI_obs) > 0 else 1.0
            dI_obs = self._add_structured_noise(dI_obs, GT, pv, profile=structural_noise_profile)
        sigma_s = max(2, int(np.ceil(GT * 0.5)))
        refs = self._compute_truth_references(new_inf, GT=GT, sigma=sigma_s)
        return dict(
            I=dI_obs, t=t_arr, truth_I=new_inf, R0=R0, GT=GT, noise_std=noise_std,
            SI=float(SI) if SI is not None else np.nan,
            N=N, report_delay=report_delay, archetype=archetype,
            structural_noise_profile=structural_noise_profile,
            truth_T1=refs.T1_growth, truth_T2=refs.T2_dom, truth_Tp=refs.Tp,
            truth_T1_growth=refs.T1_growth, truth_T1_struct=refs.T1_struct,
            truth_T2_dyn=refs.T2_dyn, truth_T2_dom=refs.T2_dom,
        )

    def batch_generate_archetype(self, n_curves=180, days=300, add_structured_noise=True,
                                 N=100000, I0=5, min_Tp: Optional[int] = None,
                                 scenario_set: Optional[List[Dict]] = None):
        scenarios = scenario_set or self.ARCHETYPE_SCENARIOS
        counts = self._allocate_counts_from_weights(scenarios, n_curves)
        curves = []
        curve_id = 0

        def effective_min_tp(sc: Dict) -> Optional[int]:
            if min_Tp is None:
                return None
            arch = str(sc.get('archetype', sc.get('label', ''))).lower()
            floor = int(min_Tp)
            # A single Tp floor removes the fast respiratory archetypes from
            # prospective validation. Keep the "mature curve" filter, but scale
            # it to the natural peak timing of each archetype.
            if 'high_transmissibility' in arch or 'fast_respiratory' in arch:
                return min(floor, 28)
            if 'influenza' in arch:
                return min(floor, 30)
            if 'moderate_coronavirus' in arch:
                return min(floor, 35)
            return floor

        def make_curve(sc: Dict, suffix: str = ''):
            R0 = self._sample_range(sc['R0_range'])
            GT = self._sample_range(sc['GT_range'])
            SI = self._sample_range(sc['SI_range'])
            nst = self._sample_range(sc['noise_std_range'])
            rd = self._sample_int_range(sc['report_delay_range'])
            c = self.generate_sir_curve(
                days=days, R0=R0, GT=GT, N=N, I0=I0,
                noise_std=nst, report_delay=rd,
                add_structured_noise=add_structured_noise,
                structural_noise_profile=sc['structural_noise_profile'],
                SI=SI, archetype=sc['archetype'],
            )
            c['scenario'] = sc['label'] + suffix
            c['archetype'] = sc['archetype']
            c['SI'] = SI
            c['scenario_weight'] = float(sc['weight'])
            c['structural_noise_profile'] = sc['structural_noise_profile']
            c['transmission_route'] = sc.get('route', '')
            c['source_framework'] = 'pathogen_archetype_design'
            return c

        for sc, n_sc in zip(scenarios, counts):
            sc_count = 0
            attempts = 0
            while sc_count < n_sc and attempts < max(20, n_sc * 25):
                attempts += 1
                c = make_curve(sc)
                min_tp_eff = effective_min_tp(sc)
                if min_tp_eff is not None and c['truth_Tp'] < min_tp_eff:
                    continue
                c['curve_id'] = curve_id
                curves.append(c)
                curve_id += 1
                sc_count += 1

        weights = np.array([float(s.get('weight', 1.0)) for s in scenarios], dtype=float)
        weights = weights / max(float(weights.sum()), 1e-10)
        while len(curves) < n_curves:
            sc = scenarios[int(self.rng.choice(np.arange(len(scenarios)), p=weights))]
            c = make_curve(sc, suffix='_supplemental')
            min_tp_eff = effective_min_tp(sc)
            if min_tp_eff is not None and c['truth_Tp'] < min_tp_eff:
                continue
            c['curve_id'] = curve_id
            curves.append(c)
            curve_id += 1
        return curves[:n_curves]

    def batch_generate(self, n_curves=100, days=220, R0_range=(1.2, 8.0), GT_range=(2.0, 10.0),
                       noise_std_range=(0.10, 0.50), report_delay_range=(0, 7), add_structured_noise=True,
                       N=100000, I0=5):
        curves = []
        for i in range(n_curves):
            R0 = self.rng.uniform(*R0_range)
            GT = self.rng.uniform(*GT_range)
            nst = self.rng.uniform(*noise_std_range)
            rd = int(self.rng.uniform(*report_delay_range))
            c = self.generate_sir_curve(days=days, R0=R0, GT=GT, N=N, I0=I0,
                                        noise_std=nst, report_delay=rd,
                                        add_structured_noise=add_structured_noise)
            c['curve_id'] = i
            c['scenario'] = 'development'
            curves.append(c)
        return curves

    def batch_generate_mixed(self, n_curves=80, days=220, add_structured_noise=True, N=100000, I0=5):
        return self.batch_generate_archetype(
            n_curves=n_curves, days=days, add_structured_noise=add_structured_noise,
            N=N, I0=I0, min_Tp=None,
        )

    def batch_generate_for_prospective(self, n_curves=80, days=300, add_structured_noise=True, N=100000, I0=5, min_Tp=40):
        return self.batch_generate_archetype(
            n_curves=n_curves, days=days, add_structured_noise=add_structured_noise,
            N=N, I0=I0, min_Tp=min_Tp,
        )

    def generate_disease_scenarios(self, disease='dengue', n_cities=20, seed_offset=0):
        params = {
            'dengue': dict(R0_range=(2.0, 6.0), GT_range=(5.0, 9.0), noise_range=(0.20, 0.40), delay_range=(3, 7), days=300, prefix='Dengue_City'),
            'chikungunya': dict(R0_range=(2.0, 5.0), GT_range=(4.0, 7.0), noise_range=(0.25, 0.45), delay_range=(3, 7), days=250, prefix='CHIK_City'),
        }
        p = params[disease]
        rng = np.random.RandomState(seed_offset + 2024)
        cities = []
        for i in range(n_cities):
            c = self.generate_sir_curve(
                days=p['days'], R0=rng.uniform(*p['R0_range']), GT=rng.uniform(*p['GT_range']),
                N=100000, I0=5, noise_std=rng.uniform(*p['noise_range']),
                report_delay=int(rng.uniform(*p['delay_range'])), add_structured_noise=True,
            )
            c['city_name'] = f"{p['prefix']}_{i+1:03d}"
            c['disease'] = disease
            c['curve_id'] = i
            c['scenario'] = f'{disease}_simulated'
            cities.append(c)
        return cities

    @staticmethod
    def print_truth_distribution(curves, label=''):
        if not curves:
            return
        t1s = [c['truth_T1_growth'] for c in curves if 'truth_T1_growth' in c]
        t2ds = [c['truth_T2_dyn'] for c in curves if 'truth_T2_dyn' in c]
        t2dom = [c['truth_T2_dom'] for c in curves if 'truth_T2_dom' in c]
        tps = [c['truth_Tp'] for c in curves if 'truth_Tp' in c]
        gaps = [c['truth_T2_dom'] - c['truth_T1_growth'] for c in curves if 'truth_T2_dom' in c and 'truth_T1_growth' in c]
        print(f"\n  Truth/reference distribution [{label}] (N={len(curves)})")
        if t1s:
            print(f"    T1_growth: median={np.median(t1s):.0f}d  IQR=[{np.percentile(t1s,25):.0f},{np.percentile(t1s,75):.0f}]  range=[{min(t1s)},{max(t1s)}]")
        if t2ds:
            print(f"    T2_dyn:    median={np.median(t2ds):.0f}d  IQR=[{np.percentile(t2ds,25):.0f},{np.percentile(t2ds,75):.0f}]")
        if t2dom:
            print(f"    T2_dom:    median={np.median(t2dom):.0f}d  IQR=[{np.percentile(t2dom,25):.0f},{np.percentile(t2dom,75):.0f}]")
        if tps:
            print(f"    Tp:        median={np.median(tps):.0f}d  IQR=[{np.percentile(tps,25):.0f},{np.percentile(tps,75):.0f}]")
        if gaps:
            print(f"    T2_dom-T1_growth: median={np.median(gaps):.0f}d  min={min(gaps)}  max={max(gaps)}")


# ============================================================
#  Part 7. Parameter optimization
# ============================================================

class FusionAlgorithmOptimizer:
    COARSE_GRID = {
        't1_structure_score_thresh': [0.44, 0.50, 0.56, 0.62],
        't1_mid_z_thresh': [0.8, 1.0, 1.25, 1.5],
        't1_mid_high_ratio_z': [0.50, 0.70, 0.90],
        't1_centroid_z_thresh': [0.20, 0.35, 0.50],
        't2_low_share_thresh_short': [0.30, 0.34, 0.38],
        't2_dom_score_thresh': [0.45, 0.50, 0.55],
        'energy_smooth_sigma': [1.5, 2.0, 2.5, 3.0],
    }
    FINE_STEPS = {
        't1_structure_score_thresh': 0.04,
        't1_mid_z_thresh': 0.20,
        't1_mid_high_ratio_z': 0.15,
        't1_centroid_z_thresh': 0.10,
        't2_low_share_thresh_short': 0.04,
        't2_dom_score_thresh': 0.05,
        'energy_smooth_sigma': 0.6,
    }
    COARSE_TOPK_FOR_STAGE2 = 10
    FINE_TOPK_FOR_STAGE2 = 12
    STAGE2_TUNE_MAX_CURVES = 4
    T1_CONFIRM_VARIANTS = [
        dict(t1_confirm_ewma_k=0.46, t1_fast_confirm_strength=0.55, t1_single_confirm_strength=0.52,
             t1_valid_strength_min=0.20, t1_fast_local_margin=0.01, t1_confirm_span_gt=0.75,
             t1_early_confirm_strength=0.28, t1_early_confirm_min_persist_gt=0.35,
             t1_early_confirm_local_margin=-0.05,
             t1_confirm_single_min_persist_gt=0.25, t1_confirm_single_local_margin=0.03,
             t1_estimate_backtrack_gt=0.75, t1_estimate_max_backtrack_gt=0.75,
             t1_estimate_backtrack_min_persist_gt=0.90, t1_estimate_history_quantile=10,
             t1_aux_min_strength=0.12, t1_aux_max_lead_gt=4.0,
             t1_aux_stale_candidate_gt=2.0, t1_aux_stale_backcast_gt=0.35,
             t1_struct_stale_candidate_gt=3.5, t1_struct_stale_backcast_gt=0.60,
             t1_first_alert_strength=0.18, t1_first_alert_local_margin=-0.12,
             t1_first_alert_min_persist_gt=0.0, t1_first_alert_allow_energy=True),
        dict(t1_confirm_ewma_k=0.55, t1_fast_confirm_strength=0.60, t1_single_confirm_strength=0.56,
             t1_valid_strength_min=0.24, t1_fast_local_margin=0.02, t1_confirm_span_gt=0.9,
             t1_early_confirm_strength=0.32, t1_early_confirm_min_persist_gt=0.50,
             t1_early_confirm_local_margin=-0.03,
             t1_confirm_single_min_persist_gt=0.35, t1_confirm_single_local_margin=0.06,
             t1_estimate_backtrack_gt=0.60, t1_estimate_max_backtrack_gt=0.60,
             t1_estimate_backtrack_min_persist_gt=1.00, t1_estimate_history_quantile=15,
             t1_aux_min_strength=0.14, t1_aux_max_lead_gt=3.5,
             t1_aux_stale_candidate_gt=2.5, t1_aux_stale_backcast_gt=0.40,
             t1_struct_stale_candidate_gt=4.0, t1_struct_stale_backcast_gt=0.70,
             t1_first_alert_strength=0.22, t1_first_alert_local_margin=-0.08,
             t1_first_alert_min_persist_gt=0.20, t1_first_alert_allow_energy=True),
        dict(t1_confirm_ewma_k=0.62, t1_fast_confirm_strength=0.62, t1_single_confirm_strength=0.58,
             t1_valid_strength_min=0.28, t1_fast_local_margin=0.04, t1_confirm_span_gt=1.0,
             t1_confirm_single_min_persist_gt=0.45, t1_confirm_single_local_margin=0.08,
             t1_estimate_backtrack_gt=0.50, t1_estimate_max_backtrack_gt=0.50,
             t1_estimate_backtrack_min_persist_gt=1.10,
             t1_first_alert_strength=0.26, t1_first_alert_local_margin=-0.04,
             t1_first_alert_min_persist_gt=0.35, t1_first_alert_allow_energy=True),
        dict(t1_confirm_ewma_k=0.70, t1_fast_confirm_strength=0.65, t1_single_confirm_strength=0.60,
             t1_valid_strength_min=0.30, t1_fast_local_margin=0.05, t1_confirm_span_gt=1.1,
             t1_confirm_single_min_persist_gt=0.55, t1_confirm_single_local_margin=0.10,
             t1_estimate_backtrack_gt=0.40, t1_estimate_max_backtrack_gt=0.40,
             t1_estimate_backtrack_min_persist_gt=1.20,
             t1_first_alert_strength=0.28, t1_first_alert_local_margin=-0.02,
             t1_first_alert_min_persist_gt=0.45, t1_first_alert_allow_energy=True),
    ]

    def __init__(self, base_ep: EpiParams, cache: Optional[EEMDCWTCache] = None,
                 stage2_tune_curves: Optional[List[Dict]] = None):
        self.base_ep = base_ep
        self.cache = cache
        self.stage2_tune_curves = stage2_tune_curves or []

    def _build_eval_row(self, result: Dict, curve: Dict) -> Dict:
        t1 = result['T1']
        t2 = result['T2']
        tp = result['Tp']
        t1_growth = curve.get('truth_T1_growth')
        t1_struct = curve.get('truth_T1_struct', t1_growth)
        t2_dyn = curve.get('truth_T2_dyn')
        t2_dom = curve.get('truth_T2_dom')
        tp_truth = curve.get('truth_Tp')
        t1_band = band_distance(t1, t1_growth, t1_struct)
        t2_band = band_distance(t2, t2_dyn, t2_dom)
        row = dict(
            T1_growth_err=abs(t1 - t1_growth),
            T1_struct_err=abs(t1 - t1_struct),
            T1_band_err=t1_band,
            T2_dyn_err=abs(t2 - t2_dyn),
            T2_dom_err=abs(t2 - t2_dom),
            T2_band_err=t2_band,
            Tp_err=abs(tp - tp_truth),
            GT=curve.get('GT', self.base_ep.GT),
            SI=curve.get('SI', np.nan),
            scenario=curve.get('scenario', ''),
            archetype=curve.get('archetype', ''),
            structural_noise_profile=curve.get('structural_noise_profile', ''),
            alert_days=result['confidence'].get('alert_days', t2 - t1),
            confidence=result['confidence'].get('overall', 'none'),
        )
        row['T1_growth_signed_err'] = t1 - t1_growth
        row['primary_T1_err'] = row['T1_growth_err']
        row['primary_T2_err'] = self.base_ep.t2_band_weight * row['T2_band_err'] + (1 - self.base_ep.t2_band_weight) * row['T2_dom_err']
        return row

    def _score_global(self, df: pd.DataFrame) -> float:
        if len(df) == 0:
            return float('inf')
        tol = max(3, int(self.base_ep.GT))
        base = 2.5 * df['primary_T1_err'].mean() + 1.8 * df['primary_T2_err'].mean() + 1.0 * df['Tp_err'].mean()
        tail = 0.4 * df['primary_T1_err'].quantile(0.75) + 0.2 * df['primary_T2_err'].quantile(0.75)
        t1_rate = (df['primary_T1_err'] <= tol).mean()
        t2_band_rate = (df['T2_band_err'] <= tol).mean()
        t1_bias = float(df['T1_growth_signed_err'].median()) if 'T1_growth_signed_err' in df else 0.0
        penalties = max(0, 0.75 - t1_rate) * 30 + max(0, 0.70 - t2_band_rate) * 25
        penalties += max(0.0, abs(t1_bias) - 1.0) * 4.0
        return float(base + tail + penalties)

    def _score_stage2(self, df: pd.DataFrame) -> float:
        if df is None or len(df) == 0:
            return 0.0
        score = 0.0
        t1_err = pd.to_numeric(df.get('T1_err', pd.Series(dtype=float)), errors='coerce').dropna()
        if len(t1_err):
            score += 2.8 * t1_err.mean() + max(0.0, t1_err.median() - 6.0) * 5.0
        t1_band = pd.to_numeric(df.get('T1_band_err', pd.Series(dtype=float)), errors='coerce').dropna()
        if len(t1_band):
            score += 1.4 * t1_band.mean()
        t2_band = pd.to_numeric(df.get('T2_band_err', pd.Series(dtype=float)), errors='coerce').dropna()
        if len(t2_band):
            score += 0.25 * t2_band.mean()
        tp_err = pd.to_numeric(df.get('Tp_err', pd.Series(dtype=float)), errors='coerce').dropna()
        if len(tp_err):
            score += 0.25 * tp_err.mean()

        delay = pd.to_numeric(df.get('T1_delay', pd.Series(dtype=float)), errors='coerce').dropna()
        if len(delay):
            score += 3.0 * delay.clip(lower=0).mean()
            if 'GT' in df:
                gt = pd.to_numeric(df.loc[delay.index, 'GT'], errors='coerce')
                ok = gt.notna() & (gt > 0)
                if ok.any():
                    score += (delay[ok] > 2.0 * gt[ok]).mean() * 55.0
                    score += (delay[ok] < -gt[ok]).mean() * 35.0
                    score += max(0.0, delay[ok].median() - 1.5 * gt[ok].median()) * 10.0
        within = pd.to_numeric(df.get('T1_confirm_within_2GT', pd.Series(dtype=float)), errors='coerce').dropna()
        if len(within):
            score += max(0.0, 0.90 - within.mean()) * 60.0
        lead = pd.to_numeric(df.get('lead_before_T2', pd.Series(dtype=float)), errors='coerce').dropna()
        if len(lead):
            score += (lead <= 0).mean() * 45.0

        fa_err = pd.to_numeric(df.get('T1_first_alert_err', pd.Series(dtype=float)), errors='coerce').dropna()
        if len(fa_err):
            score += 0.8 * fa_err.mean()
        fa_delay = pd.to_numeric(df.get('T1_first_alert_delay', pd.Series(dtype=float)), errors='coerce').dropna()
        if len(fa_delay):
            score += 1.2 * fa_delay.clip(lower=0).mean()
        fa_within = pd.to_numeric(df.get('T1_first_alert_within_2GT', pd.Series(dtype=float)), errors='coerce').dropna()
        if len(fa_within):
            score += max(0.0, 0.92 - fa_within.mean()) * 25.0
        fa_lead = pd.to_numeric(df.get('lead_before_T2_by_first_alert', pd.Series(dtype=float)), errors='coerce').dropna()
        if len(fa_lead):
            score += (fa_lead <= 0).mean() * 18.0
        return score

    def _evaluate_with_cache(self, curves: List[Dict], ep: EpiParams) -> pd.DataFrame:
        rows = []
        for idx, curve in enumerate(curves):
            if idx not in self.cache.cache:
                continue
            curve_ep = build_archetype_adaptive_ep(ep, curve, prospective=False)
            seg = StructuralPhaseSegmenter(curve_ep)
            result = seg.segment_from_cache(self.cache.get(idx))
            if result['T1'] >= result['N'] - 1:
                continue
            row = self._build_eval_row(result, curve)
            row['adaptive_profile'] = getattr(curve_ep, 'adaptive_profile', 'standard')
            row['adaptive_burden'] = getattr(curve_ep, 'adaptive_burden', 0.0)
            rows.append(row)
        return pd.DataFrame(rows)

    def _evaluate_stage2_small(self, ep: EpiParams) -> pd.DataFrame:
        if not self.stage2_tune_curves:
            return pd.DataFrame()
        curves = self.stage2_tune_curves[:min(self.STAGE2_TUNE_MAX_CURVES, len(self.stage2_tune_curves))]
        validator = RollingWindowValidator(
            ep, window_size=90, step=4, eemd_trials=8, confirm_rounds=2,
            confirm_tol=max(4, int(np.ceil(1.2 * ep.GT)))
        )
        _, df = validator.batch_validate(curves, label='tune_stage2_small', verbose=False)
        return df

    def _make_fine_grid(self, best_params: Dict) -> Dict:
        fine_grid = {}
        for k, v in best_params.items():
            step = self.FINE_STEPS.get(k, 0)
            if step == 0:
                fine_grid[k] = [v]
            else:
                vals = [round(v - step, 4), round(v, 4), round(v + step, 4)]
                if k == 'energy_smooth_sigma':
                    vals = [max(1.5, x) for x in vals]
                fine_grid[k] = sorted(set(vals))
        return fine_grid

    def _prepare_ep(self, params: Dict) -> EpiParams:
        ep_test = copy.deepcopy(self.base_ep)
        ep_test.apply_override(params)
        ep_test.energy_smooth_sigma = max(1.5, float(ep_test.energy_smooth_sigma))
        ep_test.score_smooth_sigma = max(1.0, ep_test.energy_smooth_sigma * 0.7)
        ep_test.t2_low_share_thresh_medium = min(max(ep_test.t2_low_share_thresh_short + 0.04, 0.34), 0.50)
        ep_test.t2_low_share_thresh_long = min(max(ep_test.t2_low_share_thresh_medium + 0.04, 0.38), 0.56)
        ep_test.real_t2_low_share_short = ep_test.t2_low_share_thresh_short
        ep_test.real_t2_low_share_medium = ep_test.t2_low_share_thresh_medium
        ep_test.real_t2_low_share_long = ep_test.t2_low_share_thresh_long
        return ep_test

    def _params_from_record(self, rec: Dict, keys: List[str]) -> Dict:
        return {k: rec[k] for k in keys if k in rec}

    def _rerank_with_stage2(self, records: List[Dict], keys: List[str], topk: int,
                            phase_label: str, verbose: bool = True) -> Tuple[float, Dict, List[Dict]]:
        if not records:
            return float('inf'), {}, []
        records_sorted = sorted(records, key=lambda x: x['global_score'])
        candidates = records_sorted[:min(topk, len(records_sorted))]
        best_score = float('inf')
        best_params: Dict = {}
        reranked = []
        if verbose:
            print(f"  [{phase_label}] Stage2 rerank: top{len(candidates)}")
        for i, rec in enumerate(candidates, 1):
            base_params = self._params_from_record(rec, keys)
            for vi, t1_params in enumerate(self.T1_CONFIRM_VARIANTS, 1):
                params = dict(base_params)
                params.update(t1_params)
                ep_test = self._prepare_ep(params)
                df_stage2 = self._evaluate_stage2_small(ep_test)
                stage2_score = self._score_stage2(df_stage2)
                score = 0.62 * rec['global_score'] + 0.38 * stage2_score
                new_rec = dict(rec)
                new_rec.update(t1_params)
                new_rec['t1_confirm_variant'] = vi
                new_rec['stage2_score'] = stage2_score
                new_rec['score'] = score
                new_rec['reranked'] = True
                reranked.append(new_rec)
                if verbose:
                    print(f"    - {phase_label}[{i:02d}/{len(candidates)} v{vi}] score={score:.2f}  global={rec['global_score']:.2f}  stage2={stage2_score:.2f}")
                if score < best_score:
                    best_score = score
                    best_params = params
        return best_score, best_params, reranked

    def grid_search_layered_legacy(self, dev_curves: List[Dict], verbose: bool = True):
        if self.cache is None or not self.cache.is_ready():
            raise RuntimeError('Please call cache.build() before grid_search().')
        print(f"\n{'='*78}\n  V31.1 adaptive layered grid search (coarse -> fine)\n{'='*78}")
        all_records = []

        keys = list(self.COARSE_GRID.keys())
        combos = list(product(*self.COARSE_GRID.values()))
        best_score_c = float('inf')
        best_params_c = {}
        print("\n  [Phase-1] coarse search...")
        for idx, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            ep_test = self._prepare_ep(params)
            df_global = self._evaluate_with_cache(dev_curves, ep_test)
            if len(df_global) == 0:
                continue
            global_score = self._score_global(df_global)
            if self.stage2_tune_curves:
                df_stage2 = self._evaluate_stage2_small(ep_test)
                stage2_score = self._score_stage2(df_stage2)
            else:
                df_stage2 = pd.DataFrame()
                stage2_score = 0.0
            score = 0.65 * global_score + 0.35 * stage2_score
            record = dict(**params, phase='coarse', score=score,
                          global_score=global_score, stage2_score=stage2_score,
                          T1_mean=df_global['primary_T1_err'].mean(),
                          T2_mean=df_global['primary_T2_err'].mean(),
                          Tp_mean=df_global['Tp_err'].mean(),
                          T2_band_rate=(df_global['T2_band_err'] <= max(3, int(ep_test.GT))).mean(),
                          n_valid=len(df_global))
            all_records.append(record)
            if score < best_score_c:
                best_score_c = score
                best_params_c = params.copy()
                if verbose:
                    print(f"  * coarse[{idx+1:03d}/{len(combos)}] score={score:.2f}  global={global_score:.2f}  stage2={stage2_score:.2f}  {params}")

        print(f"  coarse search done: score={best_score_c:.2f}")
        print("\n  [Phase-2] fine search...")
        fine_grid = self._make_fine_grid(best_params_c)
        keys_f = list(fine_grid.keys())
        combos_f = list(product(*fine_grid.values()))
        best_score = best_score_c
        best_params = best_params_c.copy()
        for idx, combo in enumerate(combos_f):
            params = dict(zip(keys_f, combo))
            ep_test = self._prepare_ep(params)
            df_global = self._evaluate_with_cache(dev_curves, ep_test)
            if len(df_global) == 0:
                continue
            global_score = self._score_global(df_global)
            if self.stage2_tune_curves:
                df_stage2 = self._evaluate_stage2_small(ep_test)
                stage2_score = self._score_stage2(df_stage2)
            else:
                stage2_score = 0.0
            score = 0.65 * global_score + 0.35 * stage2_score
            record = dict(**params, phase='fine', score=score,
                          global_score=global_score, stage2_score=stage2_score,
                          T1_mean=df_global['primary_T1_err'].mean(),
                          T2_mean=df_global['primary_T2_err'].mean(),
                          Tp_mean=df_global['Tp_err'].mean(),
                          T2_band_rate=(df_global['T2_band_err'] <= max(3, int(ep_test.GT))).mean(),
                          n_valid=len(df_global))
            all_records.append(record)
            if score < best_score:
                best_score = score
                best_params = params.copy()
                if verbose:
                    print(f"  * fine[{idx+1:03d}/{len(combos_f)}] score={score:.2f}  global={global_score:.2f}  stage2={stage2_score:.2f}  {params}")

        records_df = pd.DataFrame(all_records).sort_values('score')
        records_df.to_csv(os.path.join(OUTPUT_STAGE1, 'grid_search_results_v31_1.csv'), index=False)
        with open(BEST_PARAMS_PATH, 'w', encoding='utf-8') as f:
            json.dump(best_params, f, ensure_ascii=False, indent=2)
        print(f"\n  search done: best_score={best_score:.2f}  best_params={best_params}")
        return best_params, records_df

    def grid_search(self, dev_curves: List[Dict], verbose: bool = True):
        if self.cache is None or not self.cache.is_ready():
            raise RuntimeError('Please call cache.build() before grid_search().')
        print(f"\n{'='*78}\n  V31.1 archetype-adaptive fusion grid search (fast: global filter, then Stage2 rerank)\n{'='*78}")
        all_records = []

        keys = list(self.COARSE_GRID.keys())
        combos = list(product(*self.COARSE_GRID.values()))
        coarse_records = []
        print("\n  [Phase-1] coarse search (global score only)...")
        for idx, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            ep_test = self._prepare_ep(params)
            df_global = self._evaluate_with_cache(dev_curves, ep_test)
            if len(df_global) == 0:
                continue
            global_score = self._score_global(df_global)
            record = dict(**params, phase='coarse', score=np.nan,
                          global_score=global_score, stage2_score=np.nan,
                          reranked=False,
                          T1_mean=df_global['primary_T1_err'].mean(),
                          T2_mean=df_global['primary_T2_err'].mean(),
                          Tp_mean=df_global['Tp_err'].mean(),
                          T2_band_rate=(df_global['T2_band_err'] <= max(3, int(ep_test.GT))).mean(),
                          n_valid=len(df_global))
            coarse_records.append(record)
            all_records.append(record)
            if verbose and ((idx + 1) % 20 == 0 or idx == 0):
                best_now = min(r['global_score'] for r in coarse_records)
                print(f"  coarse progress [{idx+1:03d}/{len(combos)}] best_global={best_now:.2f}")

        coarse_best_score, coarse_best_params, coarse_reranked = self._rerank_with_stage2(
            coarse_records, keys, self.COARSE_TOPK_FOR_STAGE2, 'coarse', verbose=verbose
        )
        all_records.extend(coarse_reranked)
        print(f"  coarse done: best_score={coarse_best_score:.2f}  best_params={coarse_best_params}")

        print("\n  [Phase-2] fine search (global score only)...")
        fine_seed = coarse_best_params if coarse_best_params else {k: v[0] for k, v in self.COARSE_GRID.items()}
        fine_grid = self._make_fine_grid(fine_seed)
        keys_f = list(fine_grid.keys())
        combos_f = list(product(*fine_grid.values()))
        fine_records = []
        for idx, combo in enumerate(combos_f):
            params = dict(zip(keys_f, combo))
            ep_test = self._prepare_ep(params)
            df_global = self._evaluate_with_cache(dev_curves, ep_test)
            if len(df_global) == 0:
                continue
            global_score = self._score_global(df_global)
            record = dict(**params, phase='fine', score=np.nan,
                          global_score=global_score, stage2_score=np.nan,
                          reranked=False,
                          T1_mean=df_global['primary_T1_err'].mean(),
                          T2_mean=df_global['primary_T2_err'].mean(),
                          Tp_mean=df_global['Tp_err'].mean(),
                          T2_band_rate=(df_global['T2_band_err'] <= max(3, int(ep_test.GT))).mean(),
                          n_valid=len(df_global))
            fine_records.append(record)
            all_records.append(record)
            if verbose and ((idx + 1) % 30 == 0 or idx == 0):
                best_now = min(r['global_score'] for r in fine_records)
                print(f"  fine progress [{idx+1:03d}/{len(combos_f)}] best_global={best_now:.2f}")

        fine_best_score, fine_best_params, fine_reranked = self._rerank_with_stage2(
            fine_records, keys_f, self.FINE_TOPK_FOR_STAGE2, 'fine', verbose=verbose
        )
        all_records.extend(fine_reranked)

        if fine_best_score < coarse_best_score:
            best_score = fine_best_score
            best_params = fine_best_params
        else:
            best_score = coarse_best_score
            best_params = coarse_best_params

        records_df = pd.DataFrame(all_records).sort_values(
            ['reranked', 'score', 'global_score'], ascending=[False, True, True]
        )
        records_df.to_csv(os.path.join(OUTPUT_STAGE1, 'grid_search_results_v31_1.csv'), index=False)
        with open(BEST_PARAMS_PATH, 'w', encoding='utf-8') as f:
            json.dump(best_params, f, ensure_ascii=False, indent=2)
        print(f"\n  search done: best_score={best_score:.2f}  best_params={best_params}")
        return best_params, records_df


# ============================================================
#  Part 8. Prospective rolling validation
# ============================================================

class RollingWindowValidator:
    def __init__(self, ep: EpiParams, window_size=120, step=3, eemd_trials=20,
                 confirm_rounds=2, confirm_tol=None):
        self.ep = ep
        self.window_size = window_size
        self.step = step
        self.eemd_trials = eemd_trials
        self.confirm_rounds = confirm_rounds
        self.confirm_tol = ep.confirm_tol if confirm_tol is None else confirm_tol
        self.min_start_base = ep.prospective_min_start

    def _adaptive_window(self, signal_length, current_day: Optional[int] = None):
        signal_length = max(1, int(signal_length))
        configured = max(20, int(self.window_size))
        full_cap = min(signal_length, max(30, min(configured, int(np.ceil(signal_length * 0.80)))))
        if current_day is None:
            return full_cap
        current_day = max(1, int(current_day))
        warm_cap = max(20, int(np.ceil(current_day * 0.85)))
        return max(20, min(full_cap, warm_cap, current_day))

    def _compute_ewma_baseline(self, arr: np.ndarray):
        alpha = self.ep.ewma_alpha
        N = len(arr)
        ewma = np.zeros(N)
        ewma_sq = np.zeros(N)
        ewma[0] = arr[0]
        ewma_sq[0] = arr[0] ** 2
        for i in range(1, N):
            ewma[i] = alpha * arr[i] + (1 - alpha) * ewma[i - 1]
            ewma_sq[i] = alpha * arr[i] ** 2 + (1 - alpha) * ewma_sq[i - 1]
        ewma_var = np.maximum(ewma_sq - ewma ** 2, 1e-10)
        return ewma, np.sqrt(ewma_var)

    def _energy_assisted_t1_candidate(self, result: Dict, t1_loc: int, tp_loc: int,
                                      win_len: int, gt_val: float) -> Tuple[int, str, float]:
        if not getattr(self.ep, 't1_aux_energy_enable', True):
            strength = float(result.get('confidence', {}).get('t1_strength', 0.0))
            return int(t1_loc), 'structural', strength
        try:
            aux = SignalDiagnostics.energy_threshold_baseline_from_result(result, self.ep)
        except Exception:
            aux = {}
        aux_t1 = aux.get('T1_energy')
        if aux_t1 is None or pd.isna(aux_t1):
            strength = float(result.get('confidence', {}).get('t1_strength', 0.0))
            return int(t1_loc), 'structural', strength
        aux_t1 = int(aux_t1)
        max_lead = max(3, int(np.ceil(getattr(self.ep, 't1_aux_max_lead_gt', 3.0) * gt_val)))
        if not (0 <= aux_t1 < t1_loc and t1_loc - aux_t1 <= max_lead):
            strength = float(result.get('confidence', {}).get('t1_strength', 0.0))
            return int(t1_loc), 'structural', strength
        if aux_t1 >= min(tp_loc - 1, int(win_len * 0.88)):
            strength = float(result.get('confidence', {}).get('t1_strength', 0.0))
            return int(t1_loc), 'structural', strength

        mid_score = np.asarray(result.get('mid_score', []), dtype=float)
        if len(mid_score) <= aux_t1:
            strength = float(result.get('confidence', {}).get('t1_strength', 0.0))
            return int(t1_loc), 'structural', strength
        h = min(len(mid_score), aux_t1 + max(2, int(np.ceil(gt_val))))
        aux_strength = float(np.mean(mid_score[aux_t1:h])) if h > aux_t1 else float(mid_score[aux_t1])
        structural_strength = float(result.get('confidence', {}).get('t1_strength', 0.0))
        if aux_strength < getattr(self.ep, 't1_aux_min_strength', 0.16):
            return int(t1_loc), 'structural', structural_strength
        if aux_strength + 0.08 < structural_strength and t1_loc - aux_t1 > int(np.ceil(2.5 * gt_val)):
            return int(t1_loc), 'structural', structural_strength
        return int(aux_t1), 'energy_assisted', max(aux_strength, structural_strength * 0.85)

    def _raw_growth_t1_candidate(self, wsig: np.ndarray, ws: int, cur: int,
                                 gt_val: float) -> Optional[Dict]:
        if not getattr(self.ep, 't1_raw_growth_alert_enable', True):
            return None
        y = np.maximum(np.nan_to_num(np.asarray(wsig, dtype=float), nan=0.0), 0.0)
        n = len(y)
        min_obs = max(10, int(np.ceil(2.5 * gt_val)))
        if n < min_obs or float(np.nanmax(y)) <= 0:
            return None
        min_persist = max(
            1,
            int(np.ceil(float(getattr(self.ep, 't1_raw_growth_min_persist_gt', 0.60)) * gt_val))
        )
        if n < min_obs + min_persist:
            return None
        sigma = max(1.0, float(getattr(self.ep, 't1_raw_growth_smooth_gt', 0.35)) * gt_val)
        smooth = gaussian_filter1d(y, sigma=sigma)
        peak = float(np.max(smooth))
        if peak <= 0:
            return None
        floor = max(2.0, peak * float(getattr(self.ep, 't1_raw_growth_min_abs_frac', 0.04)))
        log_signal = np.log1p(smooth)
        growth = gaussian_filter1d(np.gradient(log_signal), sigma=max(0.8, 0.25 * gt_val))
        bl_end = min(max(4, int(np.ceil(2.0 * gt_val))), max(4, n - min_persist - 2))
        if bl_end >= n - 3:
            return None
        base = growth[:bl_end]
        base_mu = float(np.median(base))
        base_sd = max(robust_mad_std(base), 1e-4)
        z = (growth - base_mu) / base_sd
        search_end = max(bl_end + 1, n - min_persist + 1)
        z_thresh = float(getattr(self.ep, 't1_raw_growth_z', 2.2))
        ratio_thresh = float(getattr(self.ep, 't1_raw_growth_ratio', 1.20))
        for i in range(bl_end, search_end):
            j = min(n, i + min_persist)
            if j <= i:
                continue
            abs_ok = bool(np.all(smooth[i:j] >= floor))
            z_ok = float(np.mean(z[i:j] >= z_thresh)) >= 0.60
            ratio_ok = smooth[j - 1] >= max(smooth[i], floor) * ratio_thresh
            slope_ok = float(np.mean(growth[i:j])) > max(base_mu, 0.0) + 0.25 * base_sd
            if abs_ok and z_ok and (ratio_ok or slope_ok):
                gain = max(float(smooth[j - 1] - smooth[i]), 0.0) / max(floor, 1.0)
                strength = float(np.clip(0.50 + 0.10 * np.nanmean(z[i:j]) + 0.10 * np.log1p(gain), 0.0, 1.0))
                backcast = int(np.ceil(float(getattr(self.ep, 't1_raw_growth_backcast_gt', 1.0)) * gt_val))
                estimate_day = max(int(ws + i), int(cur) - max(1, backcast))
                return dict(
                    day=int(ws + i), estimate_day=int(estimate_day), cur=int(cur),
                    strength=strength, source='raw_growth',
                    local_mean=strength, soft_thresh=0.50,
                    persistence=int(cur - (ws + i)),
                )
        return None

    def _try_confirm_T1(self, cur: int, T1_global: int, T1_local: int, t1_strength: float,
                        mid_score: np.ndarray, gt_val: float, T1_cands: List[Dict],
                        T1_strength_history: List[float],
                        source: str = 'structural') -> Tuple[bool, Optional[Dict]]:
        ep = self.ep
        ref_idx = max(0, T1_local - 1)
        ewma, ewma_std = self._compute_ewma_baseline(mid_score)
        soft_thresh = ewma[ref_idx] + ep.t1_confirm_ewma_k * ewma_std[ref_idx]
        h = min(len(mid_score), T1_local + max(2, int(np.ceil(1.2 * gt_val))))
        local_mean = float(np.mean(mid_score[T1_local:h])) if h > T1_local else float(mid_score[T1_local])
        persistence = int(cur - T1_global)
        persist_need = max(self.step, int(np.ceil(getattr(ep, 't1_early_confirm_min_persist_gt', 0.55) * gt_val)))

        def add_candidate():
            estimate_day = int(T1_global)
            backtrack_gt = min(
                float(getattr(ep, 't1_estimate_backtrack_gt', 0.75)),
                float(getattr(ep, 't1_estimate_max_backtrack_gt', 0.75)),
            )
            min_backtrack_persist = int(np.ceil(
                getattr(ep, 't1_estimate_backtrack_min_persist_gt', 1.0) * gt_val
            ))
            if backtrack_gt > 0 and persistence >= min_backtrack_persist:
                backcast = int(cur - np.ceil(backtrack_gt * gt_val))
                earliest = int(T1_global - np.ceil(backtrack_gt * gt_val))
                estimate_day = max(earliest, min(estimate_day, backcast))
            if source == 'energy_assisted' and persistence >= int(np.ceil(getattr(ep, 't1_aux_stale_candidate_gt', 2.5) * gt_val)):
                estimate_day = min(estimate_day, int(cur - np.ceil(getattr(ep, 't1_aux_stale_backcast_gt', 0.65) * gt_val)))
            elif source == 'structural' and persistence >= int(np.ceil(getattr(ep, 't1_struct_stale_candidate_gt', 4.0) * gt_val)):
                estimate_day = min(estimate_day, int(cur - np.ceil(getattr(ep, 't1_struct_stale_backcast_gt', 1.25) * gt_val)))
            earliest = int(T1_global - np.ceil(float(getattr(ep, 't1_estimate_max_backtrack_gt', 0.75)) * gt_val))
            estimate_day = max(earliest, estimate_day)
            estimate_day = max(0, min(int(cur), estimate_day))
            cand = dict(
                day=int(T1_global), estimate_day=estimate_day, cur=cur,
                strength=t1_strength, source=source,
                local_mean=local_mean, soft_thresh=soft_thresh,
                persistence=persistence,
            )
            T1_cands.append(cand)
            return cand

        strong_single = (
            getattr(ep, 't1_confirm_allow_single_strong', True)
            and persistence >= max(self.step, int(np.ceil(getattr(ep, 't1_confirm_single_min_persist_gt', 0.35) * gt_val)))
            and (
                t1_strength >= getattr(ep, 't1_single_confirm_strength', 0.56)
                or local_mean >= soft_thresh + getattr(ep, 't1_confirm_single_local_margin', 0.06)
            )
        )
        if strong_single:
            cand = add_candidate()
            return True, cand

        early_source_ok = source == 'energy_assisted' or t1_strength >= ep.t1_progressive_strength_lo
        if (
            getattr(ep, 't1_early_confirm_enable', True)
            and persistence >= persist_need
            and early_source_ok
            and t1_strength >= getattr(ep, 't1_early_confirm_strength', 0.34)
            and local_mean >= soft_thresh + getattr(ep, 't1_early_confirm_local_margin', -0.02)
        ):
            cand = add_candidate()
            confirmed = self._stable_confirm_from_candidates(T1_cands, gt_val)
            return confirmed, cand

        if (
            t1_strength >= ep.t1_fast_confirm_strength
            or local_mean >= soft_thresh + ep.t1_fast_local_margin
            or (t1_strength >= ep.t1_early_single_strength and local_mean >= soft_thresh)
        ):
            cand = add_candidate()
            confirmed = self._stable_confirm_from_candidates(T1_cands, gt_val)
            return confirmed, cand

        T1_strength_history.append(float(t1_strength))
        n_prog = max(2, int(ep.t1_progressive_rounds))
        if len(T1_strength_history) >= n_prog:
            recent_strength = T1_strength_history[-n_prog:]
            if (
                all(b >= a - 0.01 for a, b in zip(recent_strength, recent_strength[1:]))
                and recent_strength[-1] >= ep.t1_progressive_strength_hi
                and recent_strength[0] >= ep.t1_progressive_strength_lo
            ):
                cand = add_candidate()
                confirmed = self._stable_confirm_from_candidates(T1_cands, gt_val)
                return confirmed, cand

        return False, None

    def _stable_confirm_from_candidates(self, T1_cands: List[Dict], gt_val: float) -> bool:
        ep = self.ep
        n_need = max(2, int(self.confirm_rounds))
        if len(T1_cands) >= n_need:
            recent = T1_cands[-n_need:]
            recent_days = [x['day'] for x in recent]
            span_tol = max(self.step, int(np.ceil(ep.t1_confirm_span_gt * gt_val)))
            if max(recent_days) - min(recent_days) <= span_tol:
                return True
        return False

    def detect_single(self, signal, truth_T1_growth=None, truth_T1_struct=None,
                      truth_T2_dyn=None, truth_T2_dom=None, truth_Tp=None,
                      curve_id=None, GT=None, min_start_override: Optional[int] = None,
                      candidate_min_day: Optional[int] = None,
                      candidate_max_early: Optional[int] = None):
        N = len(signal)
        seg = StructuralPhaseSegmenter(self.ep, self.eemd_trials)
        W = self._adaptive_window(N)
        gt_val = float(GT if GT is not None else self.ep.GT)
        arch_for_start = str(getattr(self.ep, 'adaptive_archetype', '')).lower()
        profile_name_for_start = str(getattr(self.ep, 'adaptive_profile', '')).lower()
        is_influenza_like_run = any(x in arch_for_start for x in ['influenza', 'h1n1', 'flu_like'])
        if is_influenza_like_run:
            min_start_gt_factor = float(getattr(self.ep, 't1_flu_min_start_gt_factor', 0.55))
            min_start_floor = int(getattr(self.ep, 't1_flu_min_start_floor', 4))
        elif any(x in arch_for_start for x in ['zoonotic', 'high_severity']) and (
            'conservative' in profile_name_for_start or gt_val >= 8.5
        ):
            min_start_gt_factor = 1.05
            min_start_floor = 8
        elif any(x in arch_for_start for x in ['mers', 'hospital', 'amplification']):
            min_start_gt_factor = 1.35
            min_start_floor = 8
        elif any(x in arch_for_start for x in ['mpv', 'orthopox', 'contact_network']):
            min_start_gt_factor = 1.45
            min_start_floor = 8
        else:
            min_start_gt_factor = 2.0
            min_start_floor = 8
        min_window_obs = max(
            int(getattr(self.ep, 'prospective_min_window_floor', 12)),
            int(np.ceil(getattr(self.ep, 'prospective_min_window_gt', 3.0) * gt_val))
        )
        min_start = max(
            min(self.min_start_base, min_window_obs),
            int(np.ceil(gt_val * min_start_gt_factor)),
            min_start_floor
        )
        if min_start_override is not None:
            min_start = max(min_start, int(min_start_override))
        if candidate_min_day is None and truth_T1_growth is not None and candidate_max_early is not None:
            candidate_min_day = int(truth_T1_growth) - int(candidate_max_early)
        if candidate_min_day is None:
            arch = str(getattr(self.ep, 'adaptive_archetype', '')).lower()
            profile_name = str(getattr(self.ep, 'adaptive_profile', '')).lower()
            report_delay = max(
                0.0,
                _float_or_default(getattr(self.ep, 'adaptive_report_delay', None), getattr(self.ep, 'T_report', 0.0)),
            )
            if any(x in arch for x in ['influenza', 'h1n1', 'flu_like']):
                min_gt = float(getattr(self.ep, 't1_flu_candidate_min_gt', 0.20))
                report_cap = float(getattr(self.ep, 't1_flu_candidate_report_cap', 1.0))
                candidate_min_day = int(np.ceil(min_gt * gt_val + min(report_delay, report_cap)))
            elif any(x in arch for x in ['mpv', 'orthopox', 'contact_network']):
                candidate_min_day = int(np.ceil(1.70 * gt_val + report_delay))
            elif any(x in arch for x in ['mers', 'hospital', 'amplification']):
                candidate_min_day = int(np.ceil(1.50 * gt_val + report_delay))
            elif (
                any(x in arch for x in ['zoonotic', 'high_severity'])
                and ('conservative' in profile_name or gt_val >= 8.5)
            ):
                candidate_min_day = int(np.ceil(1.00 * gt_val + report_delay))
        if candidate_min_day is not None:
            candidate_min_day = max(0, int(candidate_min_day))
        history = []
        T1_cands, T1_strength_history, T2_cands, Tp_cands = [], [], [], []
        T1_confirmed = T2_confirmed = Tp_confirmed = None
        T1_estimate = T2_estimate = Tp_estimate = None
        T1_confirmed_source = None
        T1_first_alert_day = None
        T1_first_alert_estimate = None
        T1_first_alert_source = None
        T1_first_alert_strength = np.nan
        structural_failure = truth_Tp is not None and truth_Tp < min_start
        max_t1_confirm_delay = max(self.step * 2, int(np.ceil(2.0 * gt_val)))

        for cur in range(min_start, N, self.step):
            W_cur = max(min_window_obs, self._adaptive_window(N, current_day=cur))
            W_cur = min(W_cur, cur)
            ws = max(0, cur - W_cur)
            wsig = signal[ws:cur]
            wt = np.arange(len(wsig), dtype=float)
            if len(wsig) < min_window_obs:
                continue
            raw_cand = None
            if T1_confirmed is None:
                raw_cand = self._raw_growth_t1_candidate(wsig, ws, cur, gt_val)
                if raw_cand is not None and (
                    candidate_min_day is None or raw_cand['day'] >= candidate_min_day
                ):
                    T1_cands.append(raw_cand)
                    if T1_first_alert_day is None and getattr(self.ep, 't1_first_alert_enable', True):
                        T1_first_alert_day = int(cur)
                        T1_first_alert_estimate = int(raw_cand.get('estimate_day', raw_cand['day']))
                        T1_first_alert_source = raw_cand.get('source', 'raw_growth')
                        T1_first_alert_strength = float(raw_cand.get('strength', 0.0))
                    raw_recent_gt = float(
                        getattr(self.ep, 't1_flu_raw_recent_gt', 1.60)
                        if is_influenza_like_run else 1.20
                    )
                    raw_recent = [
                        x for x in T1_cands
                        if x.get('source') == 'raw_growth'
                        and cur - x.get('cur', cur) <= max(self.step * 2, int(np.ceil(raw_recent_gt * gt_val)))
                    ]
                    raw_days = [x.get('estimate_day', x['day']) for x in raw_recent]
                    raw_need = int(
                        getattr(self.ep, 't1_flu_raw_stable_rounds', 1)
                        if is_influenza_like_run else max(2, int(self.confirm_rounds))
                    )
                    raw_need = max(1, raw_need)
                    raw_stable = (
                        len(raw_recent) >= raw_need
                        and (len(raw_days) <= 1 or max(raw_days) - min(raw_days) <= self.confirm_tol)
                    )
                    flu_raw_single = (
                        is_influenza_like_run
                        and raw_cand.get('strength', 0.0) >= getattr(self.ep, 't1_flu_raw_immediate_strength', 0.30)
                        and raw_cand.get('persistence', 0) >= max(
                            0,
                            int(np.ceil(getattr(self.ep, 't1_raw_growth_min_persist_gt', 0.10) * gt_val))
                        )
                    )
                    raw_single = (
                        getattr(self.ep, 't1_confirm_allow_single_strong', True)
                        and raw_cand.get('strength', 0.0) >= getattr(self.ep, 't1_raw_growth_confirm_strength', 0.55)
                        and raw_cand.get('persistence', 0) >= max(
                            self.step,
                            int(np.ceil(getattr(self.ep, 't1_confirm_single_min_persist_gt', 0.35) * gt_val))
                        )
                        and (
                            getattr(self.ep, 't1_confirm_allow_same_window_as_first_alert', False)
                            or (T1_first_alert_day is not None and T1_first_alert_day < cur)
                        )
                    )
                    if raw_stable or raw_single or flu_raw_single:
                        T1_confirmed = int(cur)
                        q = getattr(self.ep, 't1_estimate_history_quantile', self.ep.t1_confirm_quantile)
                        T1_estimate = int(np.percentile(raw_days or [raw_cand.get('estimate_day', raw_cand['day'])], q))
                        T1_confirmed_source = 'raw_growth_flu_immediate' if flu_raw_single else 'raw_growth'
            try:
                r = seg.segment(wsig, wt, prospective_mode=True, real_data=False,
                                current_window_size=len(wsig))
            except Exception:
                continue
            if r['T1'] >= r['N'] - 1:
                continue

            T1_loc, T2_loc, Tp_loc = r['T1'], r['T2'], r['Tp']
            win_len = cur - ws
            t1_strength = r['confidence'].get('t1_strength', 0.0)
            t2_strength = r['confidence'].get('t2_strength', 0.0)
            dom_gap = r['confidence'].get('dom_gap', 0.0)
            T1_struct_loc = int(T1_loc)
            T1_loc, T1_source, t1_strength = self._energy_assisted_t1_candidate(
                r, int(T1_loc), int(Tp_loc), int(win_len), float(gt_val)
            )
            T1g, T2g, Tpg = T1_loc + ws, T2_loc + ws, Tp_loc + ws
            T1_struct_global = T1_struct_loc + ws

            valid_strength_min = self.ep.t1_valid_strength_min
            if T1_source == 'energy_assisted':
                valid_strength_min = min(valid_strength_min, getattr(self.ep, 't1_aux_min_strength', valid_strength_min))
            T1_valid = (T1_loc < int(win_len * 0.88)) and (t1_strength >= valid_strength_min)
            if candidate_min_day is not None and T1g < candidate_min_day:
                T1_valid = False
            min_t2_gap = max(
                int(self.ep.t2_min_gap),
                int(np.ceil(getattr(self.ep, 'prospective_t2_min_after_t1_gt', 1.25) * gt_val)),
            )
            t1_ref_for_t2 = int(T1_estimate) if T1_estimate is not None else int(T1g)
            t2_floor = t1_ref_for_t2 + min_t2_gap
            T2_valid = (
                (T2_loc > T1_loc)
                and (T2_loc < int(win_len * 0.90))
                and (T2_loc < Tp_loc)
                and (T2g >= t2_floor)
                and (t2_strength >= getattr(self.ep, 'prospective_t2_strength_min', 0.42))
                and (dom_gap >= getattr(self.ep, 'prospective_t2_dom_gap_min', 0.0))
            )
            trend = np.asarray(r.get('trend', wsig), dtype=float)
            tp_maturity = max(7, int(np.ceil(getattr(self.ep, 'prospective_tp_maturity_gt', 2.0) * gt_val)))
            tp_min_local = int(np.ceil(getattr(self.ep, 'prospective_tp_min_local_frac', 0.40) * win_len))
            tp_decline_frac = getattr(self.ep, 'prospective_tp_decline_frac', 0.10)
            tp_peak_val = float(trend[Tp_loc]) if 0 <= Tp_loc < len(trend) else float(np.nanmax(trend))
            tp_last_val = float(trend[-1]) if len(trend) else np.nan
            tp_decline_ok = (
                pd.notna(tp_peak_val) and pd.notna(tp_last_val)
                and tp_peak_val > 0
                and tp_last_val <= tp_peak_val * (1.0 - tp_decline_frac)
            )
            Tp_valid = (
                (Tp_loc <= win_len - tp_maturity)
                and (Tpg > T2g + max(1, int(np.ceil(0.5 * gt_val))))
                and (Tp_loc >= tp_min_local)
                and tp_decline_ok
            )

            history.append(dict(
                current_day=cur, win_start=ws, win_size=win_len,
                T1_global=T1g, T2_global=T2g, Tp_global=Tpg,
                T1_local=T1_loc, T2_local=T2_loc, Tp_local=Tp_loc,
                T1_structural_global=T1_struct_global, T1_source=T1_source,
                T1_valid=T1_valid, T2_valid=T2_valid, Tp_valid=Tp_valid,
                raw_T1_global=raw_cand.get('day') if raw_cand is not None else np.nan,
                raw_T1_valid=raw_cand is not None,
                raw_T1_strength=raw_cand.get('strength') if raw_cand is not None else np.nan,
                t1_strength=t1_strength, t2_strength=t2_strength, dom_gap=dom_gap,
                T2_floor=t2_floor, Tp_decline_ok=tp_decline_ok,
                Tp_maturity_days=tp_maturity,
            ))

            if T1_confirmed is None and T1_valid:
                mid_score = r.get('mid_score', np.zeros(len(wsig)))
                first_alert_set_this_round = False
                if T1_first_alert_day is None and getattr(self.ep, 't1_first_alert_enable', True):
                    ref_idx = max(0, T1_loc - 1)
                    ewma, ewma_std = self._compute_ewma_baseline(mid_score)
                    soft_thresh = ewma[ref_idx] + self.ep.t1_confirm_ewma_k * ewma_std[ref_idx]
                    h = min(len(mid_score), T1_loc + max(2, int(np.ceil(1.2 * gt_val))))
                    local_mean = float(np.mean(mid_score[T1_loc:h])) if h > T1_loc else float(mid_score[T1_loc])
                    persistence = int(cur - T1g)
                    min_persist = max(
                        0,
                        int(np.ceil(getattr(self.ep, 't1_first_alert_min_persist_gt', 0.20) * gt_val))
                    )
                    source_ok = (
                        T1_source == 'energy_assisted'
                        and getattr(self.ep, 't1_first_alert_allow_energy', True)
                    )
                    strength_ok = t1_strength >= getattr(self.ep, 't1_first_alert_strength', 0.24)
                    local_ok = local_mean >= (
                        soft_thresh + getattr(self.ep, 't1_first_alert_local_margin', -0.08)
                    )
                    if persistence >= min_persist and (source_ok or strength_ok or local_ok):
                        T1_first_alert_day = int(cur)
                        T1_first_alert_estimate = int(T1g)
                        T1_first_alert_source = T1_source
                        T1_first_alert_strength = float(t1_strength)
                        first_alert_set_this_round = True
                confirmed, cand = self._try_confirm_T1(
                    cur, T1g, T1_loc, t1_strength, mid_score,
                    gt_val, T1_cands, T1_strength_history, source=T1_source
                )
                if cand is not None and T1_first_alert_day is None:
                    min_persist = max(
                        0,
                        int(np.ceil(getattr(self.ep, 't1_first_alert_min_persist_gt', 0.20) * gt_val))
                    )
                    source_ok = (
                        cand.get('source') == 'energy_assisted'
                        and getattr(self.ep, 't1_first_alert_allow_energy', True)
                    )
                    strength_ok = cand.get('strength', 0.0) >= getattr(self.ep, 't1_first_alert_strength', 0.24)
                    local_ok = cand.get('local_mean', 0.0) >= (
                        cand.get('soft_thresh', 0.0)
                        + getattr(self.ep, 't1_first_alert_local_margin', -0.08)
                    )
                    persist_ok = cand.get('persistence', 0) >= min_persist
                    if getattr(self.ep, 't1_first_alert_enable', True) and persist_ok and (source_ok or strength_ok or local_ok):
                        T1_first_alert_day = int(cur)
                        T1_first_alert_estimate = int(cand.get('estimate_day', T1g))
                        T1_first_alert_source = cand.get('source', T1_source)
                        T1_first_alert_strength = float(cand.get('strength', t1_strength))
                        first_alert_set_this_round = True
                if first_alert_set_this_round and not getattr(
                    self.ep, 't1_confirm_allow_same_window_as_first_alert', False
                ):
                    confirmed = False
                if confirmed:
                    T1_confirmed = cur
                    backtrack_gt = min(
                        float(getattr(self.ep, 't1_estimate_backtrack_gt', 0.75)),
                        float(getattr(self.ep, 't1_estimate_max_backtrack_gt', 0.75)),
                    )
                    backtrack = max(self.step, int(np.ceil(backtrack_gt * gt_val)))
                    recent = [
                        x for x in T1_cands
                        if (cur - x.get('cur', cur) <= max(backtrack * 2, self.step))
                        and (T1g - backtrack <= x['day'] <= T1g + max(backtrack, self.step))
                    ]
                    if len(recent) < max(1, self.confirm_rounds):
                        recent = T1_cands[-max(1, self.confirm_rounds):]
                    recent_days = [x.get('estimate_day', x['day']) for x in recent]
                    recent_sources = [x.get('source', 'structural') for x in recent]
                    q = getattr(self.ep, 't1_estimate_history_quantile', self.ep.t1_confirm_quantile)
                    T1_estimate = int(np.percentile(recent_days, q)) if recent_days else int(T1g)
                    T1_confirmed_source = max(set(recent_sources), key=recent_sources.count) if recent_sources else T1_source

            if (
                T1_confirmed is None
                and T1_first_alert_day is not None
                and getattr(self.ep, 't1_alert_graduation_enable', True)
            ):
                arch = str(getattr(self.ep, 'adaptive_archetype', '')).lower()
                burden = _float_or_default(getattr(self.ep, 'adaptive_burden', 0.0), 0.0)
                graduation_arch_ok = (
                    burden >= 0.55
                    or any(x in arch for x in [
                        'influenza', 'h1n1', 'mpv', 'orthopox', 'contact_network',
                        'mers', 'hospital', 'amplification', 'zoonotic', 'high_severity',
                    ])
                )
                urgent_graduation_arch = (
                    any(x in arch for x in ['influenza', 'h1n1', 'flu_like'])
                    or (
                        any(x in arch for x in ['zoonotic', 'high_severity'])
                        and (
                            'conservative' in str(getattr(self.ep, 'adaptive_profile', '')).lower()
                            or gt_val >= 8.5
                        )
                    )
                )
                mature_gap = max(
                    self.step,
                    int(np.ceil(float(getattr(self.ep, 't1_alert_graduation_gt', 1.15)) * gt_val)),
                )
                if graduation_arch_ok and cur - T1_first_alert_day >= mature_gap:
                    min_strength = float(getattr(self.ep, 't1_alert_graduation_min_strength', 0.24))
                    recent = [
                        x for x in T1_cands
                        if x.get('cur', cur) >= T1_first_alert_day
                        and x.get('strength', 0.0) >= min_strength
                    ]
                    if recent:
                        recent_tail = recent[-max(2, int(self.confirm_rounds)):]
                        estimate_days = [int(x.get('estimate_day', x.get('day', cur))) for x in recent_tail]
                        strengths = [float(x.get('strength', 0.0)) for x in recent_tail]
                        span_tol = max(
                            self.confirm_tol,
                            int(np.ceil(1.25 * gt_val)),
                        )
                        stable_alert = len(estimate_days) >= 2 and max(estimate_days) - min(estimate_days) <= span_tol
                        strong_alert = max(strengths) >= min(
                            float(getattr(self.ep, 't1_single_confirm_strength', 0.56)),
                            min_strength + 0.10,
                        )
                        aged_alert = cur - T1_first_alert_day >= mature_gap + self.step
                        urgent_alert = (
                            urgent_graduation_arch
                            and max(strengths) >= min_strength
                            and cur - T1_first_alert_day >= mature_gap
                        )
                        if stable_alert or strong_alert or aged_alert or urgent_alert:
                            q = getattr(self.ep, 't1_estimate_history_quantile', self.ep.t1_confirm_quantile)
                            if urgent_graduation_arch:
                                q = min(int(q), 10)
                            T1_confirmed = int(cur)
                            T1_estimate = int(np.percentile(estimate_days, q))
                            T1_confirmed_source = 'first_alert_graduated'

            if T1_confirmed is not None and T2_confirmed is None and T2_valid:
                T2_cands.append(T2g)
                n_need = max(self.confirm_rounds, int(getattr(self.ep, 'prospective_t2_confirm_rounds', 3)))
                if len(T2_cands) >= n_need:
                    recent = T2_cands[-n_need:]
                    tol = max(
                        self.step,
                        int(np.ceil(getattr(self.ep, 'prospective_t2_confirm_tol_gt', 1.0) * gt_val)),
                    )
                    if max(recent) - min(recent) <= tol:
                        T2_confirmed = cur
                        T2_estimate = int(np.median(recent))

            if T2_confirmed is not None and Tp_confirmed is None and Tp_valid:
                Tp_cands.append(Tpg)
                n_need = max(self.confirm_rounds, int(getattr(self.ep, 'prospective_tp_confirm_rounds', 3)))
                if len(Tp_cands) >= n_need:
                    recent = Tp_cands[-n_need:]
                    tol = max(
                        self.confirm_tol,
                        int(np.ceil(getattr(self.ep, 'prospective_tp_confirm_tol_gt', 1.5) * gt_val)),
                    )
                    if max(recent) - min(recent) <= tol:
                        Tp_confirmed = cur
                        Tp_estimate = int(np.median(recent))
                        if T2_estimate is not None:
                            T2_estimate = min(T2_estimate, Tp_estimate - 1)

        if history:
            def _enforce_t2_tp_order(t1_est, t2_est, tp_est):
                if t1_est is None or t2_est is None or tp_est is None:
                    return t2_est, tp_est
                min_t2_gap_local = max(
                    int(self.ep.t2_min_gap),
                    int(np.ceil(getattr(self.ep, 'prospective_t2_min_after_t1_gt', 1.25) * gt_val)),
                )
                t2_floor_local = int(t1_est) + min_t2_gap_local
                if N >= 3:
                    t2_floor_local = min(max(0, t2_floor_local), N - 2)
                else:
                    t2_floor_local = max(0, min(t2_floor_local, N - 1))
                tp_local = int(tp_est)
                if tp_local <= t2_floor_local:
                    tp_local = min(N - 1, t2_floor_local + max(1, int(np.ceil(gt_val))))
                if tp_local <= t2_floor_local and N >= 2:
                    t2_floor_local = max(0, tp_local - 1)
                t2_local = int(min(max(int(t2_est), t2_floor_local), max(t2_floor_local, tp_local - 1)))
                return t2_local, tp_local

            if T1_estimate is None:
                valid_t1 = [h['T1_global'] for h in history if h['T1_valid']]
                if valid_t1:
                    tail = valid_t1[max(0, len(valid_t1)//3):]
                    T1_estimate = int(np.percentile(tail, 25))
                else:
                    T1_estimate = history[-1]['T1_global']
            if T2_estimate is None:
                min_t2_gap = max(
                    int(self.ep.t2_min_gap),
                    int(np.ceil(getattr(self.ep, 'prospective_t2_min_after_t1_gt', 1.25) * gt_val)),
                )
                t2_floor = int(T1_estimate) + min_t2_gap
                valid_t2 = [h['T2_global'] for h in history if h['T2_valid'] and h['T2_global'] >= t2_floor]
                if valid_t2:
                    tail_start = int(len(valid_t2) * getattr(self.ep, 'prospective_t2_tail_quantile', 0.35))
                    T2_estimate = int(np.median(valid_t2[tail_start:]))
                else:
                    fallback_tp = Tp_estimate if Tp_estimate is not None else history[-1]['Tp_global']
                    projected = int(T1_estimate) + max(min_t2_gap, int(np.ceil(0.35 * max(1, fallback_tp - int(T1_estimate)))))
                    T2_estimate = int(min(max(projected, t2_floor), max(t2_floor, fallback_tp - 1)))
            if Tp_estimate is None:
                valid_tp = [int(h['Tp_global']) for h in history if h['Tp_valid']]
                if valid_tp:
                    tail = valid_tp[len(valid_tp)//2:]
                    Tp_estimate = int(np.median(tail if tail else valid_tp))
                else:
                    tp_floor = int(T2_estimate) + max(1, int(np.ceil(0.5 * gt_val))) if T2_estimate is not None else 0
                    tp_candidates = [
                        int(h['Tp_global']) for h in history
                        if pd.notna(h.get('Tp_global')) and int(h['Tp_global']) >= tp_floor
                    ]
                    if tp_candidates:
                        tail_start = int(len(tp_candidates) * 0.25)
                        tail = tp_candidates[tail_start:] or tp_candidates
                        q = float(getattr(self.ep, 'prospective_tp_fallback_quantile', 50.0))
                        q = min(max(q, 25.0), 65.0)
                        Tp_estimate = int(np.percentile(tail, q))
                        Tp_estimate = int(min(Tp_estimate, np.percentile(tp_candidates, 90)))
                        if T2_estimate is not None:
                            Tp_estimate = max(Tp_estimate, tp_floor)
                    else:
                        latest_tp = int(history[-1]['Tp_global'])
                        if T2_estimate is not None:
                            Tp_estimate = int(max(latest_tp, int(T2_estimate) + max(1, int(np.ceil(gt_val)))))
                        else:
                            Tp_estimate = latest_tp
            if T2_estimate is not None and Tp_estimate is not None:
                T2_estimate, Tp_estimate = _enforce_t2_tp_order(T1_estimate, T2_estimate, Tp_estimate)

        if T1_estimate is not None and T1_confirmed is not None:
            max_back = int(np.ceil(float(getattr(self.ep, 't1_estimate_max_backtrack_gt', 0.75)) * gt_val))
            T1_estimate = max(int(T1_estimate), int(T1_confirmed) - max(1, max_back))
            if T2_estimate is not None:
                min_t2_gap = max(
                    int(self.ep.t2_min_gap),
                    int(np.ceil(getattr(self.ep, 'prospective_t2_min_after_t1_gt', 1.25) * gt_val)),
                )
                T2_estimate = max(int(T2_estimate), int(T1_estimate) + min_t2_gap)
            if T2_estimate is not None and Tp_estimate is not None:
                T2_estimate, Tp_estimate = _enforce_t2_tp_order(T1_estimate, T2_estimate, Tp_estimate)

        if is_influenza_like_run and history and T1_estimate is not None:
            min_t2_gap = max(
                int(self.ep.t2_min_gap),
                int(np.ceil(getattr(self.ep, 'prospective_t2_min_after_t1_gt', 1.25) * gt_val)),
            )
            t2_floor = min(max(0, int(T1_estimate) + min_t2_gap), max(0, N - 2))
            max_t2_gap = max(
                int(getattr(self.ep, 'prospective_flu_max_t1_to_t2_floor', 18)),
                int(np.ceil(getattr(self.ep, 'prospective_flu_max_t1_to_t2_gt', 8.0) * gt_val)),
            )
            max_tp_gap = max(
                int(getattr(self.ep, 'prospective_flu_max_t1_to_tp_floor', 28)),
                int(np.ceil(getattr(self.ep, 'prospective_flu_max_t1_to_tp_gt', 13.0) * gt_val)),
            )
            t2_cap = min(max(0, N - 2), int(T1_estimate) + max_t2_gap)
            tp_cap = min(max(0, N - 1), int(T1_estimate) + max_tp_gap)
            early_t2 = [
                int(h['T2_global']) for h in history
                if h.get('T2_valid') and t2_floor <= int(h['T2_global']) <= t2_cap
            ]
            if T2_estimate is not None and int(T2_estimate) > t2_cap:
                if early_t2:
                    T2_estimate = int(np.percentile(early_t2, 40))
                else:
                    T2_estimate = int(max(t2_floor, t2_cap))
            tp_floor = (
                int(T2_estimate) + max(1, int(np.ceil(0.5 * gt_val)))
                if T2_estimate is not None else t2_floor + max(1, int(np.ceil(0.5 * gt_val)))
            )
            early_tp = [
                int(h['Tp_global']) for h in history
                if h.get('Tp_valid') and tp_floor <= int(h['Tp_global']) <= tp_cap
            ]
            if Tp_estimate is not None and int(Tp_estimate) > tp_cap:
                if early_tp:
                    Tp_estimate = int(np.percentile(early_tp, 40))
                else:
                    Tp_estimate = int(max(tp_floor, tp_cap))
            if T2_estimate is not None and Tp_estimate is not None:
                T2_estimate, Tp_estimate = _enforce_t2_tp_order(T1_estimate, T2_estimate, Tp_estimate)

        actual_windows = [h['win_size'] for h in history]
        actual_window_median = float(np.median(actual_windows)) if actual_windows else np.nan
        actual_window_min = int(np.min(actual_windows)) if actual_windows else np.nan
        actual_window_max = int(np.max(actual_windows)) if actual_windows else np.nan
        valid_t1_hist = [h['T1_global'] for h in history if h['T1_valid']]
        valid_t2_hist = [h['T2_global'] for h in history if h['T2_valid']]
        valid_tp_hist = [h['Tp_global'] for h in history if h['Tp_valid']]
        T1_hist_band = percentile_band(valid_t1_hist[-max(3, len(valid_t1_hist)//2):], 10, 90) if valid_t1_hist else (np.nan, np.nan)
        T2_hist_band = percentile_band(valid_t2_hist[-max(3, len(valid_t2_hist)//2):], 10, 90) if valid_t2_hist else (np.nan, np.nan)
        Tp_hist_band = percentile_band(valid_tp_hist[-max(3, len(valid_tp_hist)//2):], 10, 90) if valid_tp_hist else (np.nan, np.nan)

        T1_err = T2_err = Tp_err = np.nan
        T1_delay = lead_by_estimate = lead_by_confirm = lead_before_T2 = np.nan
        within_2gt = confirm_within_2gt = np.nan
        T1_struct_err = T1_band_err = T2_dyn_err = T2_dom_err = T2_band_err = np.nan
        T1_delay_gt_norm = np.nan
        T1_estimate_signed_error = np.nan
        T2_estimate_signed_error = np.nan
        Tp_estimate_signed_error = np.nan
        reference_alert_days = np.nan
        reference_t1_to_peak_days = np.nan
        T1_first_alert_err = np.nan
        T1_first_alert_signed_error = np.nan
        T1_first_alert_delay = np.nan
        T1_first_alert_delay_gt_norm = np.nan
        T1_first_alert_within_2gt = np.nan
        lead_by_first_alert = np.nan
        lead_before_T2_by_first_alert = np.nan

        if truth_T1_growth is not None and T1_estimate is not None:
            T1_estimate_signed_error = T1_estimate - truth_T1_growth
            T1_err = abs(T1_estimate - truth_T1_growth)
            if truth_T1_struct is None:
                truth_T1_struct = truth_T1_growth
            T1_struct_err = abs(T1_estimate - truth_T1_struct)
            T1_band_err = band_distance(T1_estimate, truth_T1_growth, truth_T1_struct)
            within_2gt = float(T1_err <= 2 * gt_val)
            T1_delay = T1_confirmed - truth_T1_growth if T1_confirmed is not None else np.nan
            if pd.notna(T1_delay):
                confirm_within_2gt = float(T1_delay <= max_t1_confirm_delay)
                T1_delay_gt_norm = float(T1_delay) / max(float(gt_val), 1e-10)
        if truth_T1_growth is not None and T1_first_alert_estimate is not None:
            T1_first_alert_signed_error = T1_first_alert_estimate - truth_T1_growth
            T1_first_alert_err = abs(T1_first_alert_estimate - truth_T1_growth)
        if truth_T1_growth is not None and T1_first_alert_day is not None:
            T1_first_alert_delay = T1_first_alert_day - truth_T1_growth
            T1_first_alert_within_2gt = float(T1_first_alert_delay <= max_t1_confirm_delay)
            T1_first_alert_delay_gt_norm = float(T1_first_alert_delay) / max(float(gt_val), 1e-10)
        if truth_T2_dyn is not None and T2_estimate is not None:
            T2_dyn_err = abs(T2_estimate - truth_T2_dyn)
        if truth_T2_dom is not None and T2_estimate is not None:
            T2_estimate_signed_error = T2_estimate - truth_T2_dom
            T2_dom_err = abs(T2_estimate - truth_T2_dom)
            T2_err = T2_dom_err
        if truth_T2_dyn is not None and truth_T2_dom is not None and T2_estimate is not None:
            T2_band_err = band_distance(T2_estimate, truth_T2_dyn, truth_T2_dom)
        if truth_Tp is not None and Tp_estimate is not None:
            Tp_estimate_signed_error = Tp_estimate - truth_Tp
            Tp_err = abs(Tp_estimate - truth_Tp)
        if truth_Tp is not None and T1_estimate is not None:
            lead_by_estimate = float(truth_Tp - T1_estimate)
        if T1_confirmed is not None and truth_Tp is not None:
            lead_by_confirm = float(truth_Tp - T1_confirmed)
        if T1_confirmed is not None and truth_T2_dom is not None:
            lead_before_T2 = float(truth_T2_dom - T1_confirmed)
        if T1_first_alert_day is not None and truth_Tp is not None:
            lead_by_first_alert = float(truth_Tp - T1_first_alert_day)
        if T1_first_alert_day is not None and truth_T2_dom is not None:
            lead_before_T2_by_first_alert = float(truth_T2_dom - T1_first_alert_day)
        if truth_T1_growth is not None and truth_T2_dom is not None:
            reference_alert_days = float(truth_T2_dom - truth_T1_growth)
        if truth_T1_growth is not None and truth_Tp is not None:
            reference_t1_to_peak_days = float(truth_Tp - truth_T1_growth)

        return dict(
            curve_id=curve_id,
            T1_confirmed=T1_confirmed, T1_estimate=T1_estimate, T2_estimate=T2_estimate, Tp_estimate=Tp_estimate,
            T2_confirmed=T2_confirmed, Tp_confirmed=Tp_confirmed,
            T1_err=T1_err, T1_struct_err=T1_struct_err, T1_band_err=T1_band_err,
            T2_err=T2_err, T2_dyn_err=T2_dyn_err, T2_dom_err=T2_dom_err, T2_band_err=T2_band_err,
            Tp_err=Tp_err, T1_delay=T1_delay,
            T1_delay_GT_norm=T1_delay_gt_norm,
            T1_estimate_signed_error=T1_estimate_signed_error,
            T2_estimate_signed_error=T2_estimate_signed_error,
            Tp_estimate_signed_error=Tp_estimate_signed_error,
            T1_within_2GT=within_2gt, T1_confirm_within_2GT=confirm_within_2gt,
            lead_by_estimate=lead_by_estimate, lead_by_confirm=lead_by_confirm, lead_before_T2=lead_before_T2,
            T1_first_alert_day=T1_first_alert_day,
            T1_first_alert_estimate=T1_first_alert_estimate,
            T1_first_alert_source=T1_first_alert_source,
            T1_first_alert_strength=T1_first_alert_strength,
            T1_first_alert_err=T1_first_alert_err,
            T1_first_alert_signed_error=T1_first_alert_signed_error,
            T1_first_alert_delay=T1_first_alert_delay,
            T1_first_alert_delay_GT_norm=T1_first_alert_delay_gt_norm,
            T1_first_alert_within_2GT=T1_first_alert_within_2gt,
            lead_by_first_alert=lead_by_first_alert,
            lead_before_T2_by_first_alert=lead_before_T2_by_first_alert,
            reference_alert_days=reference_alert_days, reference_t1_to_peak_days=reference_t1_to_peak_days,
            truth_T1_growth=truth_T1_growth, truth_T1_struct=truth_T1_struct,
            truth_T2_dyn=truth_T2_dyn, truth_T2_dom=truth_T2_dom, truth_Tp=truth_Tp,
            T1_ever_detected=T1_estimate is not None, T2_ever_detected=T2_estimate is not None, Tp_ever_detected=Tp_estimate is not None,
            T1_confirmed_detected=T1_confirmed is not None, T1_confirmed_source=T1_confirmed_source,
            structural_failure=structural_failure, min_start_used=min_start, window_size_used=W,
            window_size_configured=self.window_size, window_size_actual_median=actual_window_median,
            window_size_actual_min=actual_window_min, window_size_actual_max=actual_window_max, GT=gt_val,
            T1_hist_low=T1_hist_band[0], T1_hist_high=T1_hist_band[1],
            T2_hist_low=T2_hist_band[0], T2_hist_high=T2_hist_band[1],
            Tp_hist_low=Tp_hist_band[0], Tp_hist_high=Tp_hist_band[1],
            detection_history=history,
        )

    def batch_validate(self, curves: List[Dict], label: str = '', verbose: bool = True):
        if verbose:
            print(f"\n{'='*68}\n  Prospective rolling validation [{label}] V31.1 adaptive\n{'='*68}")
        results = []
        t0 = time.time()
        for i, curve in enumerate(curves):
            signal = np.maximum(np.nan_to_num(curve['I'], nan=0), 0).astype(float)
            curve_ep = build_archetype_adaptive_ep(self.ep, curve, prospective=True)
            settings = adaptive_rolling_settings(
                self.window_size, self.step, self.confirm_rounds, self.confirm_tol, self.ep, curve
            )
            curve_validator = RollingWindowValidator(
                curve_ep,
                window_size=settings['window_size'],
                step=settings['step'],
                eemd_trials=self.eemd_trials,
                confirm_rounds=settings['confirm_rounds'],
                confirm_tol=settings['confirm_tol'],
            )
            r = curve_validator.detect_single(
                signal,
                truth_T1_growth=curve.get('truth_T1_growth'),
                truth_T1_struct=curve.get('truth_T1_struct'),
                truth_T2_dyn=curve.get('truth_T2_dyn'),
                truth_T2_dom=curve.get('truth_T2_dom'),
                truth_Tp=curve.get('truth_Tp'),
                curve_id=curve.get('curve_id', i),
                GT=curve.get('GT')
            )
            r['scenario'] = curve.get('scenario', '')
            r['R0'] = curve.get('R0')
            r['SI'] = curve.get('SI')
            r['report_delay'] = curve.get('report_delay')
            r['archetype'] = curve.get('archetype', '')
            r['structural_noise_profile'] = curve.get('structural_noise_profile', '')
            r['adaptive_profile'] = getattr(curve_ep, 'adaptive_profile', 'standard')
            r['adaptive_burden'] = getattr(curve_ep, 'adaptive_burden', 0.0)
            r['adaptive_window_size'] = settings['window_size']
            r['adaptive_confirm_rounds'] = settings['confirm_rounds']
            r['adaptive_confirm_tol'] = settings['confirm_tol']
            results.append(r)
            if verbose and (i + 1) % 10 == 0:
                el = time.time() - t0
                print(f"  [{i+1}/{len(curves)}] {el:.1f}s")
        df = pd.DataFrame([{k: v for k, v in r.items() if k != 'detection_history'} for r in results])
        if verbose:
            self._print_summary(df, label)
        return results, df

    def _print_summary(self, df: pd.DataFrame, label: str):
        n = len(df)
        detected = df['T1_ever_detected'].sum() if 'T1_ever_detected' in df else 0
        n_struct = df['structural_failure'].sum() if 'structural_failure' in df else 0
        df_ok = df[~df['structural_failure']] if 'structural_failure' in df else df
        print(f"\n  Prospective validation summary [{label}] N={n}")
        print(f"  structural_failure: {n_struct}/{n}  |  T1_detected: {detected}/{n}={detected/max(n,1)*100:.1f}%")
        for m, nm in [('T1_err', 'T1(growth)'), ('T2_band_err', 'T2(band)'), ('Tp_err', 'Tp')]:
            v = df_ok[m].dropna() if m in df_ok.columns else pd.Series([], dtype=float)
            if len(v) == 0:
                continue
            med, lo, hi = bootstrap_ci(v)
            print(f"  {nm}: median_MAE={med:.1f}d  CI=[{lo:.1f},{hi:.1f}]  mean={v.mean():.1f}d")
        if 'T1_first_alert_err' in df_ok.columns:
            fae = pd.to_numeric(df_ok['T1_first_alert_err'], errors='coerce').dropna()
            if len(fae) > 0:
                med, lo, hi = bootstrap_ci(fae)
                print(f"  first-alert T1(growth): median_MAE={med:.1f}d  CI=[{lo:.1f},{hi:.1f}]  mean={fae.mean():.1f}d")
        if 'lead_by_first_alert' in df_ok:
            lfa = pd.to_numeric(df_ok['lead_by_first_alert'], errors='coerce').dropna()
            if len(lfa) > 0:
                print(f"  lead from T1 first alert to peak: median={lfa.median():.1f}d  >7d={(lfa>7).mean()*100:.1f}%  >14d={(lfa>14).mean()*100:.1f}%")
        if 'lead_before_T2_by_first_alert' in df_ok:
            lft2 = pd.to_numeric(df_ok['lead_before_T2_by_first_alert'], errors='coerce').dropna()
            if len(lft2) > 0:
                print(f"  lead from T1 first alert to T2: median={lft2.median():.1f}d  before_T2={(lft2>0).mean()*100:.1f}%")
        if 'T1_first_alert_delay' in df_ok:
            fad = pd.to_numeric(df_ok['T1_first_alert_delay'], errors='coerce').dropna()
            if len(fad) > 0:
                print(f"  T1 first-alert delay: median={fad.median():.1f}d  mean={fad.mean():.1f}d")
        if 'T1_first_alert_delay_GT_norm' in df_ok:
            fadg = pd.to_numeric(df_ok['T1_first_alert_delay_GT_norm'], errors='coerce').dropna()
            if len(fadg) > 0:
                print(f"  T1 first-alert delay / GT: median={fadg.median():.2f}  mean={fadg.mean():.2f}")
        if 'T1_first_alert_within_2GT' in df_ok.columns:
            fav = pd.to_numeric(df_ok['T1_first_alert_within_2GT'], errors='coerce').dropna()
            if len(fav) > 0:
                print(f"  T1 first alert within 2GT: {(fav.mean()*100):.1f}%")
        if 'lead_by_confirm' in df_ok:
            lc = df_ok['lead_by_confirm'].dropna()
            if len(lc) > 0:
                print(f"  lead from T1 confirmation to peak: median={lc.median():.1f}d  >7d={(lc>7).mean()*100:.1f}%  >14d={(lc>14).mean()*100:.1f}%")
        if 'lead_before_T2' in df_ok:
            lt2 = df_ok['lead_before_T2'].dropna()
            if len(lt2) > 0:
                print(f"  lead from T1 confirmation to T2: median={lt2.median():.1f}d  before_T2={(lt2>0).mean()*100:.1f}%")
        if 'T1_estimate_signed_error' in df_ok:
            se = df_ok['T1_estimate_signed_error'].dropna()
            if len(se) > 0:
                print(f"  T1 estimate signed error: median={se.median():.1f}d  mean={se.mean():.1f}d")
        if 'T1_delay' in df_ok:
            td = df_ok['T1_delay'].dropna()
            if len(td) > 0:
                print(f"  T1 confirmation delay: median={td.median():.1f}d  mean={td.mean():.1f}d")
        if 'T1_delay_GT_norm' in df_ok:
            tdg = df_ok['T1_delay_GT_norm'].dropna()
            if len(tdg) > 0:
                print(f"  T1 confirmation delay / GT: median={tdg.median():.2f}  mean={tdg.mean():.2f}")
        if 'T1_confirm_within_2GT' in df_ok.columns:
            v3 = df_ok['T1_confirm_within_2GT'].dropna()
            if len(v3) > 0:
                print(f"  stable T1 confirmation within 2GT: {(v3.mean()*100):.1f}%")
        if {'T1_delay', 'GT'}.issubset(df_ok.columns):
            delay = pd.to_numeric(df_ok['T1_delay'], errors='coerce')
            gt = pd.to_numeric(df_ok['GT'], errors='coerce')
            ok = delay.notna() & gt.notna() & (gt > 0)
            if ok.any():
                print(f"  T1 confirmation within 1GT/2GT: {(delay[ok] <= gt[ok]).mean()*100:.1f}% / {(delay[ok] <= 2.0*gt[ok]).mean()*100:.1f}%")
        if {'T1_first_alert_delay', 'GT'}.issubset(df_ok.columns):
            delay = pd.to_numeric(df_ok['T1_first_alert_delay'], errors='coerce')
            gt = pd.to_numeric(df_ok['GT'], errors='coerce')
            ok = delay.notna() & gt.notna() & (gt > 0)
            if ok.any():
                print(f"  T1 first alert within 1GT/2GT: {(delay[ok] <= gt[ok]).mean()*100:.1f}% / {(delay[ok] <= 2.0*gt[ok]).mean()*100:.1f}%")
        if 'T1_confirmed_source' in df_ok.columns:
            src = df_ok['T1_confirmed_source'].dropna().astype(str)
            if len(src) > 0:
                print(f"  energy-assisted T1 confirmation: {(src == 'energy_assisted').mean()*100:.1f}%")
        if 'T1_first_alert_source' in df_ok.columns:
            src = df_ok['T1_first_alert_source'].dropna().astype(str)
            if len(src) > 0:
                print(f"  energy-assisted T1 first alert: {(src == 'energy_assisted').mean()*100:.1f}%")


# ============================================================
#  Part 9. Real data loading + full-series structural detection
# ============================================================

class RealDataLoader:
    def __init__(self, data_path, start_date, disease='unknown'):
        self.data_path = data_path
        self.start_date = start_date
        self.disease = disease
        self.df = None
        self.regions = []
        self.dates = []
        self.n_days = 0
        self.national = None
        self.matrix = None
        self.t = None
        self._load()

    def _load(self):
        try:
            raw = pd.read_excel(self.data_path, index_col=0)
        except Exception as e:
            print(f"  Load failed: {e}")
            return

        row_is_datetime = isinstance(raw.index, pd.DatetimeIndex)
        col_is_datetime = isinstance(raw.columns, pd.DatetimeIndex)
        if not row_is_datetime:
            try:
                pd.to_datetime(raw.index[:3])
                row_is_datetime = True
            except Exception:
                row_is_datetime = False
        if not col_is_datetime:
            try:
                pd.to_datetime(raw.columns[:3])
                col_is_datetime = True
            except Exception:
                col_is_datetime = False

        if row_is_datetime and not col_is_datetime:
            self.df = raw.copy()
            print("  [data format] rows=dates, columns=regions")
        elif col_is_datetime and not row_is_datetime:
            self.df = raw.T.copy()
            print("  [data format] rows=regions, columns=dates; transposed")
        else:
            if raw.shape[0] > raw.shape[1]:
                self.df = raw.T.copy()
                print("  [data format] inferred rows=regions, columns=dates; transposed")
            else:
                self.df = raw.copy()
                print("  [data format] inferred rows=dates, columns=regions")

        self.df = self.df.fillna(0).apply(pd.to_numeric, errors='coerce').fillna(0).clip(lower=0)
        self.n_days = len(self.df)
        self.regions = list(self.df.columns)
        self.dates = [self.start_date + datetime.timedelta(days=i) for i in range(self.n_days)]
        self.t = np.arange(self.n_days, dtype=float)
        self.national = self.df.values.sum(axis=1)
        self.matrix = self.df.values.T
        print(f"  Loaded {self.disease}: {len(self.regions)} regions, {self.n_days} days, total_cases={int(self.national.sum())}")

    def get_series(self, idx=None, region=None):
        if idx is not None:
            region = self.regions[idx]
        return np.maximum(self.df[region].values.astype(float), 0)


class RealDataStructuralDetector:
    def __init__(self, ep: EpiParams, eemd_trials: int = 200, stability_runs: int = 0):
        self.ep = ep
        self.eemd_trials = eemd_trials
        self.stability_runs = stability_runs
        self.base_segmenter = StructuralPhaseSegmenter(ep, eemd_trials=eemd_trials)

    def _detect_background_period(self, signal: np.ndarray) -> int:
        N = len(signal)
        bg_max = self.ep.get_dynamic_bg_max_days(N)
        roll = self.ep.real_bg_roll_window
        rm = rolling_mean(signal, roll)
        exceed = np.where(rm >= self.ep.real_bg_threshold)[0]
        if len(exceed) == 0:
            return min(bg_max, N)
        first_exceed = int(exceed[0])
        bg_end = max(self.ep.real_bg_min_days, min(first_exceed, bg_max))
        return int(bg_end)

    def _compute_baseline(self, signal: np.ndarray, bg_end: int) -> Tuple[float, float]:
        bg_signal = signal[:bg_end]
        if len(bg_signal) < 3:
            bg_signal = signal[: max(3, len(signal) // 10)]
        bg_mean = float(np.mean(bg_signal))
        bg_std = float(np.std(bg_signal))
        bg_std = max(bg_std, np.sqrt(max(bg_mean, 0.5)), 0.5)
        return bg_mean, bg_std

    def segment_realdata(self, signal, city_name: str = '', global_offset: int = 0, do_ensemble: bool = False):
        signal = np.maximum(np.nan_to_num(signal, nan=0), 0).astype(float)
        N = len(signal)
        if N < 20 or np.max(signal) < 1:
            return self._empty_result(N, city_name, global_offset)

        bg_end = self._detect_background_period(signal)
        bg_mean, bg_std = self._compute_baseline(signal, bg_end)
        sm_sig = gaussian_filter1d(signal, sigma=max(1.0, self.ep.GT * 0.5))
        raw_gate = sm_sig >= (bg_mean + self.ep.real_raw_gate_k * bg_std)
        search_start = max(0, bg_end - self.ep.real_epi_buffer)

        sig_valid_start = SignalDiagnostics.find_signal_start_ref(signal, self.ep)
        bimodal_takeoff = SignalDiagnostics.detect_bimodal_takeoff(signal, self.ep)
        low_share_thr = 0.5 * self.ep.get_real_t2_low_share_thresh(N) + 0.5 * self.ep.get_t2_low_share_thresh(N)

        result = self.base_segmenter.segment(
            signal, np.arange(N, dtype=float), city_name=city_name,
            prospective_mode=False, search_start=search_start,
            raw_gate_override=raw_gate, low_share_thresh_override=low_share_thr,
            real_data=True,
        )

        if do_ensemble and self.stability_runs > 0:
            ens = self.base_segmenter.segment_ensemble(
                signal, np.arange(N, dtype=float), city_name=city_name, n_runs=self.stability_runs,
                prospective_mode=False, search_start=search_start, raw_gate_override=raw_gate,
                low_share_thresh_override=low_share_thr, real_data=True,
            )
            result['ensemble_intervals'] = ens.get('ensemble_intervals', {})
        else:
            result['ensemble_intervals'] = {}

        conf = self._assess_confidence_realdata(result, bg_mean, bg_std, sig_valid_start)
        result.update(dict(
            T1_global=result['T1'] + global_offset,
            T2_global=result['T2'] + global_offset,
            Tp_global=result['Tp'] + global_offset,
            T3_global=result['T3'] + global_offset,
            bg_end=bg_end,
            bg_mean=bg_mean,
            bg_std=bg_std,
            signal_full=signal,
            N_full=N,
            global_offset=global_offset,
            mode='full_series_structural_realdata',
            sig_valid_start=sig_valid_start,
            bimodal_takeoff=bimodal_takeoff,
            confidence=conf,
        ))
        return result

    def _assess_confidence_realdata(self, result: Dict, bg_mean: float, bg_std: float, sig_valid_start: int) -> Dict:
        conf = copy.deepcopy(result['confidence'])
        T1 = result['T1']
        T2 = result['T2']
        signal = result['signal']
        sm_sig = gaussian_filter1d(signal, sigma=max(1.0, self.ep.GT * 0.5))
        t1_raw_z = float((sm_sig[T1] - bg_mean) / max(bg_std, 1e-10)) if len(sm_sig) > T1 else 0.0
        t2_dom = float(result['p_low'][T2] - result['p_mid'][T2]) if T2 < len(result['p_low']) else 0.0
        conf['t1_raw_z'] = t1_raw_z
        conf['t2_dom_gap'] = t2_dom
        conf['sig_valid_start'] = sig_valid_start
        conf['T1_minus_sigstart'] = int(T1 - sig_valid_start)
        if t1_raw_z > 4 and conf['overall'] == 'high':
            conf['overall'] = 'high'
        elif t1_raw_z > 2.5 and conf['overall'] in ('medium', 'high'):
            conf['overall'] = 'medium' if conf['overall'] == 'medium' else 'high'
        else:
            if conf['overall'] == 'high':
                conf['overall'] = 'medium'
        if T1 < max(0, sig_valid_start - int(np.ceil(self.ep.GT))) and conf['overall'] == 'high':
            conf['overall'] = 'medium'
        return conf

    def batch_segment_realdata(self, loader: RealDataLoader, verbose: bool = True,
                               ensemble_for_national: bool = True):
        results, rows = [], []
        print(f"\n  Full-series structural segmentation batch: {len(loader.regions)} regions")
        t0 = time.time()
        for i in range(len(loader.regions)):
            try:
                sig = loader.get_series(idx=i)
                r = self.segment_realdata(sig, city_name=loader.regions[i], global_offset=0, do_ensemble=False)
            except Exception as ex:
                r = self._empty_result(loader.n_days, loader.regions[i], 0)
                if verbose:
                    print(f"  Warning: {loader.regions[i]} failed: {ex}")
            results.append(r)
            conf = r['confidence']
            comparator = SignalDiagnostics.energy_threshold_baseline_from_result(
                r, self.ep, search_start=r.get('bg_end', 0), sig_valid_start=r.get('sig_valid_start')
            )
            short_alert_class = SignalDiagnostics.classify_short_alert(
                r.get('signal', np.zeros(loader.n_days)), r['T1'], r['T2'], r['Tp'],
                onset_day=r.get('sig_valid_start')
            )
            intervention_window = SignalDiagnostics.classify_intervention_window(
                r['T1'], r['T2'], r['Tp'], self.ep.GT
            )
            rows.append({
                'city': r['city'],
                'region': r['city'],
                'bg_end': r.get('bg_end', 0),
                'bg_mean': round(r.get('bg_mean', 0), 2),
                'bg_std': round(r.get('bg_std', 1), 2),
                'sig_valid_start': r.get('sig_valid_start', 0),
                'bimodal_takeoff': r.get('bimodal_takeoff'),
                'T1': r['T1_global'],
                'T2': r['T2_global'],
                'T_peak': r['Tp_global'],
                'Tp': r['Tp_global'],
                'T3': r['T3_global'],
                'T1_low': r.get('intervals', {}).get('T1', (np.nan, np.nan))[0],
                'T1_high': r.get('intervals', {}).get('T1', (np.nan, np.nan))[1],
                'T2_low': r.get('intervals', {}).get('T2', (np.nan, np.nan))[0],
                'T2_high': r.get('intervals', {}).get('T2', (np.nan, np.nan))[1],
                'Tp_low': r.get('intervals', {}).get('Tp', (np.nan, np.nan))[0],
                'Tp_high': r.get('intervals', {}).get('Tp', (np.nan, np.nan))[1],
                'pre_T1_days': max(r['T1_global'], 0),
                'alert_days': max(r['T2_global'] - r['T1_global'], 0),
                'rise_days': max(r['Tp_global'] - r['T2_global'], 0),
                'post_peak_decline_days': max(r['T3_global'] - r['Tp_global'], 0),
                'resolution_days': max(loader.n_days - r['T3_global'], 0),
                'alert_days_GT_norm': round(max(r['T2_global'] - r['T1_global'], 0) / max(self.ep.GT, 1e-10), 3),
                'alert_days_SI_norm': round(max(r['T2_global'] - r['T1_global'], 0) / max(self.ep.SI, 1e-10), 3),
                'T1_strength': round(conf.get('t1_strength', 0), 3),
                'T2_strength': round(conf.get('t2_strength', 0), 3),
                'T1_raw_z': round(conf.get('t1_raw_z', 0), 2),
                'T2_dom_gap': round(conf.get('t2_dom_gap', 0), 3),
                'T1_to_T2_days': max(r['T2_global'] - r['T1_global'], 0),
                'T1_to_peak_days': max(r['Tp_global'] - r['T1_global'], 0),
                'T1_lead_peak': max(r['Tp_global'] - r['T1_global'], 0),
                'T1_minus_sig_valid_start': int(r['T1'] - r.get('sig_valid_start', 0)),
                'short_alert_class': short_alert_class,
                'intervention_window': intervention_window,
                'method_structural': VERSION_TAG,
                'T1_energy': comparator.get('T1_energy'),
                'T2_energy': comparator.get('T2_energy'),
                'Tp_energy': comparator.get('Tp_energy'),
                'alert_days_energy': comparator.get('alert_days_energy'),
                'T1_energy_delta_vs_struct': comparator.get('T1_energy_delta_vs_struct'),
                'T2_energy_delta_vs_struct': comparator.get('T2_energy_delta_vs_struct'),
                'energy_baseline_ok': comparator.get('energy_baseline_ok'),
                'SNR': round(conf.get('snr', 0), 2),
                'SNR_capped': min(round(conf.get('snr', 0), 2), 1000),
                'log1p_SNR': float(np.log1p(max(conf.get('snr', 0), 0))),
                'T1_confidence': conf.get('T1', 'none'),
                'T2_confidence': conf.get('T2', 'none'),
                'confidence': conf.get('overall', 'none'),
            })
            if verbose and (i + 1) % 30 == 0:
                el = time.time() - t0
                eta = el / (i + 1) * (len(loader.regions) - i - 1)
                print(f"  [{i+1}/{len(loader.regions)}] {el:.1f}s / ETA {eta:.1f}s")
        df = pd.DataFrame(rows)
        try:
            for col in ['T1', 'T2', 'T_peak', 'T3']:
                df[col + '_日期'] = df[col].apply(lambda x: loader.dates[min(int(x), len(loader.dates) - 1)] if pd.notna(x) and int(x) < len(loader.dates) else pd.NaT)
        except Exception:
            pass
        self._print_batch_summary(df, loader)
        return results, df

    def _print_batch_summary(self, df: pd.DataFrame, loader: RealDataLoader):
        conf_col = 'confidence' if 'confidence' in df.columns else None
        if conf_col is None:
            df_v = df.copy()
        else:
            df_v = df[df[conf_col].isin(['high', 'medium', 'low'])].copy()
        n_all, n_v = len(df), len(df_v)
        print(f"\n  Real-data structural segmentation summary (N={n_v}/{n_all})")
        if n_v == 0:
            return
        print(f"  Background length: median={df_v['bg_end'].median():.0f}d  IQR=[{df_v['bg_end'].quantile(0.25):.0f},{df_v['bg_end'].quantile(0.75):.0f}]d")
        print(f"  T1(global): median=D{df_v['T1'].median():.0f}")
        print(f"  T2(global): median=D{df_v['T2'].median():.0f}")
        print(f"  Tp(global): median=D{df_v['T_peak'].median():.0f}")
        if 'alert_days' in df_v.columns and 'alert_days_GT_norm' in df_v.columns:
            print(f"  Alert window: median={df_v['alert_days'].median():.0f}d  GT-normalized median={df_v['alert_days_GT_norm'].median():.2f}")
        lead = df_v['T_peak'] - df_v['T1']
        print(f"  T1-to-peak window: median={lead.median():.0f}d  min={lead.min()}  max={lead.max()}")
        if conf_col is not None:
            for lv in ['high', 'medium', 'low']:
                n = (df_v[conf_col] == lv).sum()
                print(f"  Confidence[{lv}]: {n}/{n_v}={n/max(n_v,1)*100:.1f}%")
    def _empty_result(self, N: int, city_name: str, global_offset: int):
        return dict(
            city=city_name, T1=N - 1, T2=N - 1, Tp=N - 1, T3=N - 1,
            T1_global=N - 1 + global_offset, T2_global=N - 1 + global_offset, Tp_global=N - 1 + global_offset, T3_global=N - 1 + global_offset,
            bg_end=0, bg_mean=0.0, bg_std=1.0, signal=np.zeros(N), signal_full=np.zeros(N), t=np.arange(N), N=N, N_full=N,
            phases=np.zeros(N, dtype=int), imf1=np.zeros(N), trend=np.zeros(N), e_high=np.zeros(N), e_mid=np.zeros(N), e_low=np.zeros(N),
            p_high=np.zeros(N), p_mid=np.zeros(N), p_low=np.zeros(N), log_mh=np.zeros(N), log_lm=np.zeros(N), centroid=np.zeros(N),
            mid_score=np.zeros(N), dom_score=np.zeros(N), sig_valid_start=0, bimodal_takeoff=None,
            intervals=dict(T1=(N - 1, N - 1), T2=(N - 1, N - 1), Tp=(N - 1, N - 1), T3=(N - 1, N - 1)),
            confidence=dict(T1='none', T2='none', T3='none', overall='none', snr=0, t1_strength=0, t2_strength=0, dom_gap=0, alert_days=0, t1_raw_z=0, t2_dom_gap=0),
            global_offset=global_offset, mode='full_series_structural_realdata'
        )


# ============================================================
#  Part 10. Visualization
# ============================================================

class Visualizer:
    @staticmethod
    def _safe_filename(name):
        if hasattr(name, 'strftime'):
            return name.strftime('%Y-%m-%d')
        return re.sub(r'[\\/:\*\?"<>\|\s]', '_', str(name))

    @staticmethod
    def plot_global_segmentation(result: Dict, curve: Optional[Dict] = None,
                                 save_path: Optional[str] = None, title: str = ''):
        fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
        t = result['t']
        N = result['N']
        T1, T2, Tp, T3 = result['T1'], result['T2'], result['Tp'], result['T3']
        colors = {'sig': '#2196F3', 'tr': '#FF5722', 'ih': '#7E57C2',
                  'im': '#4CAF50', 'il': '#FF9800', 't1': '#4CAF50',
                  't2': '#FF9800', 'tp': '#F44336', 't3': '#2196F3'}

        def add_vlines(ax, ymax):
            for tk, col, lbl in [(T1, colors['t1'], 'T1'), (T2, colors['t2'], 'T2'),
                                 (Tp, colors['tp'], 'Tp'), (T3, colors['t3'], 'T3')]:
                if 0 <= tk < N:
                    ax.axvline(t[tk], color=col, ls='--', lw=2, alpha=0.9)
                    ax.text(t[tk], ymax * 0.88, f' {lbl}\nD{tk}', fontsize=7,
                            color=col, fontweight='bold', va='top',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor=col,
                                      boxstyle='round,pad=0.2'))
            if curve is not None:
                gt_marks = [curve.get('truth_T1_growth'), curve.get('truth_T1_struct'),
                            curve.get('truth_T2_dyn'), curve.get('truth_T2_dom'),
                            curve.get('truth_Tp')]
                for tk in gt_marks:
                    if tk is not None and 0 <= tk < N:
                        ax.axvline(t[tk], color='black', ls=':', lw=1.0, alpha=0.35)

        ax = axes[0]
        ax.fill_between(t, result['signal'], alpha=0.3, color=colors['sig'])
        ax.plot(t, result['signal'], color=colors['sig'], lw=1, alpha=0.8)
        ax.plot(t, result['trend'], color=colors['tr'], lw=2.4, label='trend')
        ax.set_ylim(bottom=0)
        add_vlines(ax, ax.get_ylim()[1])
        conf = result['confidence']
        ax.set_title(f"{title}\nT1={T1}  T2={T2}  Tp={Tp}  T3={T3}  "
                     f"[{conf.get('overall','?')}]  SNR={conf.get('snr',0):.2f}",
                     fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylabel('Cases')

        ax = axes[1]
        ax.plot(t, result['imf1'], color='#9C27B0', lw=1.2)
        ax.axhline(0, color='gray', ls='--', lw=0.8)
        add_vlines(ax, max(1.0, np.max(np.abs(result['imf1']))))
        ax.set_ylabel('IMF1')
        ax.set_title('IMF1')
        ax.grid(alpha=0.3)

        ax = axes[2]
        ax.plot(t, result['p_high'], color=colors['ih'], lw=1.6, label='p_high')
        ax.plot(t, result['p_mid'], color=colors['im'], lw=1.6, label='p_mid')
        ax.plot(t, result['p_low'], color=colors['il'], lw=1.8, label='p_low')
        add_vlines(ax, 1.0)
        ax.set_ylabel('Energy share')
        ax.set_title('High / mid / low frequency energy share')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[3]
        ax.plot(t, result['mid_score'], color=colors['im'], lw=2, label='mid_score (T1)')
        ax.plot(t, result['dom_score'], color=colors['il'], lw=2, label='dom_score (T2)')
        add_vlines(ax, max(1.0, float(np.max([result['mid_score'].max(), result['dom_score'].max()]))))
        ax.set_ylabel('Score')
        ax.set_title('Structural change scores')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[4]
        ax.plot(t, result['centroid'], color='#795548', lw=2, label='scale centroid')
        add_vlines(ax, max(1.0, float(np.max(result['centroid']))))
        ax.set_ylabel('Scale centroid')
        ax.set_xlabel('Time (days)')
        ax.set_title('Scale centroid migration')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close(fig)

    @staticmethod
    def plot_realdata_segmentation(result: Dict, loader: RealDataLoader,
                                   save_path: Optional[str] = None, title: str = ''):
        sig_full = result.get('signal_full', result['signal'])
        N_full = len(sig_full)
        dates = loader.dates[:N_full]
        T1g, T2g = result['T1_global'], result['T2_global']
        Tpg, T3g = result['Tp_global'], result['T3_global']
        bg_end = result.get('bg_end', 0)
        sig_start = result.get('sig_valid_start', 0)
        fig, axes = plt.subplots(5, 1, figsize=(15, 18), sharex=True)

        def safe_date(idx):
            idx = min(max(int(idx), 0), len(dates) - 1)
            return dates[idx]

        def add_v(ax, ymax):
            for tk, col, lbl in [(T1g, '#4CAF50', 'T1'), (T2g, '#FF9800', 'T2'),
                                 (Tpg, '#F44336', 'Tp'), (T3g, '#2196F3', 'T3')]:
                if 0 <= tk < N_full:
                    ax.axvline(safe_date(tk), color=col, ls='--', lw=2)
                    ax.text(safe_date(tk), ymax * 0.88,
                            f' {lbl}\n{safe_date(tk).strftime("%m/%d")}', fontsize=7,
                            color=col, fontweight='bold', va='top',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor=col,
                                      boxstyle='round,pad=0.2'))
            if 0 <= sig_start < N_full:
                ax.axvline(safe_date(sig_start), color='gray', ls=':', lw=1.5, alpha=0.7)

        ax = axes[0]
        ax.fill_between(dates, sig_full, alpha=0.3, color='steelblue')
        ax.plot(dates, sig_full, color='steelblue', lw=1)
        ax.plot(dates, result['trend'][:N_full], color='#FF5722', lw=2.5, label='trend')
        if bg_end > 0:
            ax.axvspan(dates[0], safe_date(bg_end), alpha=0.08, color='gray', label='background')
        ax.set_ylim(bottom=0)
        add_v(ax, ax.get_ylim()[1])
        conf = result['confidence']
        ax.set_title(f"{title}\nbackground D0-D{bg_end}  sig_start=D{sig_start}  "
                     f"T1=D{T1g}  T2=D{T2g}  Tp=D{Tpg}  T3=D{T3g}  "
                     f"[{conf.get('overall','?')}]  T1_raw_z={conf.get('t1_raw_z',0):.2f}",
                     fontweight='bold', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylabel('Cases')

        ax = axes[1]
        ax.plot(dates, result['p_high'][:N_full], color='#7E57C2', lw=1.5, label='p_high')
        ax.plot(dates, result['p_mid'][:N_full], color='#4CAF50', lw=1.5, label='p_mid')
        ax.plot(dates, result['p_low'][:N_full], color='#FF9800', lw=1.8, label='p_low')
        add_v(ax, 1.0)
        ax.set_title('High / mid / low frequency energy share')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[2]
        ax.plot(dates, result['mid_score'][:N_full], color='#4CAF50', lw=2, label='mid_score')
        add_v(ax, max(1.0, float(np.max(result['mid_score']))))
        ax.set_title('T1 structural score')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[3]
        ax.plot(dates, result['dom_score'][:N_full], color='#FF9800', lw=2, label='dom_score')
        add_v(ax, max(1.0, float(np.max(result['dom_score']))))
        ax.set_title('T2 dominance score')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[4]
        ax.plot(dates, result['centroid'][:N_full], color='#795548', lw=2, label='scale centroid')
        add_v(ax, max(1.0, float(np.max(result['centroid']))))
        ax.set_title('Scale centroid migration')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlabel('Date')

        fig.autofmt_xdate()
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close(fig)

    @staticmethod
    def plot_validation_summary(val_df: pd.DataFrame, save_path=None, label=''):
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        has_sf = 'structural_failure' in val_df.columns
        df_ok = val_df[~val_df['structural_failure']] if has_sf else val_df
        for ai, (col, name, clr) in enumerate([
            ('T1_err', 'T1 growth error', '#F44336'),
            ('T2_band_err', 'T2 band error', '#FF9800'),
            ('Tp_err', 'Peak error', '#9C27B0'),
        ]):
            ax = axes[0, ai]
            v = df_ok[col].dropna() if col in df_ok else pd.Series([], dtype=float)
            if len(v) > 0:
                ax.hist(v, bins=20, color=clr, alpha=0.7)
                ax.axvline(v.median(), color='black', ls='--', lw=2,
                           label=f'median={v.median():.1f}d')
            ax.set_xlabel('Error (days)')
            ax.set_title(name, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)

        ax = axes[1, 0]
        le = df_ok['lead_by_estimate'].dropna() if 'lead_by_estimate' in df_ok.columns else pd.Series([], dtype=float)
        lc = df_ok['lead_by_confirm'].dropna() if 'lead_by_confirm' in df_ok.columns else pd.Series([], dtype=float)
        if len(le) > 0:
            ax.hist(le, bins=20, color='#4CAF50', alpha=0.7, label='estimate')
        if len(lc) > 0:
            ax.hist(lc, bins=20, color='#2196F3', alpha=0.5, label='confirm')
        ax.axvline(0, color='black', ls='--', lw=1.5)
        ax.set_title('Lead time to peak', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[1, 1]
        td = df_ok['T1_delay'].dropna() if 'T1_delay' in df_ok.columns else pd.Series([], dtype=float)
        if len(td) > 0:
            ax.hist(td, bins=20, color='#4CAF50', alpha=0.7)
            ax.axvline(td.median(), color='black', ls='--', lw=2,
                       label=f'median={td.median():.1f}d')
        ax.set_title('T1 confirmation delay', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[1, 2]
        if 'T1_hist_low' in df_ok.columns and 'T1_hist_high' in df_ok.columns:
            widths = (df_ok['T1_hist_high'] - df_ok['T1_hist_low']).dropna()
            if len(widths) > 0:
                ax.hist(widths, bins=20, color='#FF9800', alpha=0.7)
                ax.axvline(widths.median(), color='black', ls='--', lw=2,
                           label=f'median={widths.median():.1f}d')
        ax.set_title('T1 stability interval width', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.suptitle(f'Prospective validation summary [{label}] V31.1 adaptive',
                     fontweight='bold', fontsize=12)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close(fig)

# ============================================================
#  Part 11. Stage execution functions
# ============================================================

def evaluate_global_dataset(curves: List[Dict], ep: EpiParams, label: str = '') -> pd.DataFrame:
    rows = []
    t0 = time.time()
    for i, curve in enumerate(curves):
        signal = np.maximum(np.nan_to_num(curve['I'], nan=0), 0).astype(float)
        curve_ep = build_archetype_adaptive_ep(ep, curve, prospective=False)
        seg = StructuralPhaseSegmenter(curve_ep, eemd_trials=50)
        result = seg.segment(signal, curve['t'], prospective_mode=False, real_data=False)
        t1 = result['T1']
        t2 = result['T2']
        tp = result['Tp']
        row = dict(
            T1_growth_err=abs(t1 - curve['truth_T1_growth']),
            T1_struct_err=abs(t1 - curve['truth_T1_struct']),
            T1_band_err=band_distance(t1, curve['truth_T1_growth'], curve['truth_T1_struct']),
            T2_dyn_err=abs(t2 - curve['truth_T2_dyn']),
            T2_dom_err=abs(t2 - curve['truth_T2_dom']),
            T2_band_err=band_distance(t2, curve['truth_T2_dyn'], curve['truth_T2_dom']),
            Tp_err=abs(tp - curve['truth_Tp']),
            T1_to_peak=curve['truth_Tp'] - t1,
            alert_days=t2 - t1,
            confidence=result['confidence']['overall'],
            scenario=curve.get('scenario', ''),
            archetype=curve.get('archetype', ''),
            SI=curve.get('SI', np.nan),
            report_delay=curve.get('report_delay', np.nan),
            structural_noise_profile=curve.get('structural_noise_profile', ''),
            adaptive_profile=getattr(curve_ep, 'adaptive_profile', 'standard'),
            adaptive_burden=getattr(curve_ep, 'adaptive_burden', 0.0),
            GT=curve.get('GT', ep.GT),
        )
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  evaluation progress: [{i+1}/{len(curves)}] {time.time()-t0:.1f}s")
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    print(f"\n  [{label}] N={len(df)}")
    for metric, name in [('T1_growth_err', 'T1(growth)'), ('T2_band_err', 'T2(band)'), ('Tp_err', 'Tp')]:
        med, lo, hi = bootstrap_ci(df[metric])
        print(f"    {name}: median_MAE={med:.1f}d [{lo:.1f},{hi:.1f}]  mean={df[metric].mean():.1f}d")
    lp = df['T1_to_peak'].dropna()
    if len(lp) > 0:
        print(f"    T1-to-peak lead: median={lp.median():.1f}d  before_peak={(lp>0).mean()*100:.1f}%  >7d={(lp>7).mean()*100:.1f}%  >14d={(lp>14).mean()*100:.1f}%")
    return df


def run_stage0_benchmark(ep: EpiParams, n_test: int = 30):
    print(f"\n{'#'*72}\n  STAGE 0: fusion structural-rule benchmark V31.1 adaptive\n{'#'*72}")
    sim = SIRSimulator(seed=777)
    curves = sim.batch_generate(n_curves=n_test, days=220, R0_range=(1.5, 6.0), GT_range=(2.0, 8.0), noise_std_range=(0.15, 0.40), report_delay_range=(0, 5), add_structured_noise=True)
    rows = []
    for trials in [10, 20, 50, 200]:
        t0 = time.time()
        errs = []
        for curve in curves:
            curve_ep = build_archetype_adaptive_ep(ep, curve, prospective=False)
            seg = StructuralPhaseSegmenter(curve_ep, eemd_trials=trials)
            r = seg.segment(curve['I'], curve['t'])
            errs.append(dict(
                T1_err=abs(r['T1'] - curve['truth_T1_growth']),
                T2_err=band_distance(r['T2'], curve['truth_T2_dyn'], curve['truth_T2_dom']),
                Tp_err=abs(r['Tp'] - curve['truth_Tp']),
            ))
        df = pd.DataFrame(errs)
        rows.append(dict(trials=trials, time_per_curve=(time.time() - t0) / max(len(curves), 1), T1_med=df['T1_err'].median(), T2_med=df['T2_err'].median(), Tp_med=df['Tp_err'].median()))
        print(f"  trials={trials:3d}: time={rows[-1]['time_per_curve']:.2f}s/curve  T1_med={rows[-1]['T1_med']:.1f}d  T2_band_med={rows[-1]['T2_med']:.1f}d")
    bench_df = pd.DataFrame(rows)
    bench_df.to_csv(os.path.join(OUTPUT_BENCH, 'benchmark_v31_1.csv'), index=False)
    return bench_df


def run_stage1_development(ep: EpiParams, n_dev: int = 180, n_stage2_tune: int = 36,
                           force_cache: bool = False, force_grid: bool = True):
    print(f"\n{'#'*72}\n  STAGE 1: structural-rule development and archetype-adaptive optimization V31.1\n{'#'*72}")
    ep.print_summary()
    sim = SIRSimulator(seed=42)
    design_df = SIRSimulator.archetype_design_table(default_n=n_dev)
    design_df.to_csv(os.path.join(OUTPUT_STAGE1, 'archetype_scenario_design_v31_1.csv'), index=False, encoding='utf-8-sig')
    print("\n  Archetype scenario design:")
    print(design_df.to_string(index=False))
    dev_curves = sim.batch_generate_archetype(n_curves=n_dev, days=300, add_structured_noise=True)
    tune_curves = sim.batch_generate_for_prospective(n_curves=n_stage2_tune, days=300, add_structured_noise=True, min_Tp=40)
    SIRSimulator.print_truth_distribution(dev_curves, 'development')
    SignalDiagnostics.diagnose_signal_complexity(dev_curves, 'development')

    cache = EEMDCWTCache(cache_path=CACHE_PATH, eemd_trials=10)
    cache.build(dev_curves, ep, force_rebuild=force_cache)
    optimizer = FusionAlgorithmOptimizer(ep, cache=cache, stage2_tune_curves=tune_curves)
    if force_grid or not os.path.exists(BEST_PARAMS_PATH):
        best_params, grid_df = optimizer.grid_search(dev_curves)
    else:
        with open(BEST_PARAMS_PATH, 'r', encoding='utf-8') as f:
            best_params = json.load(f)
        grid_df = pd.DataFrame()
    ep_opt = copy.deepcopy(ep)
    ep_opt.apply_override(best_params)
    ep_opt.score_smooth_sigma = max(1.0, ep_opt.energy_smooth_sigma * 0.7)
    ep_opt.real_t2_low_share_short = ep_opt.t2_low_share_thresh_short
    ep_opt.real_t2_low_share_medium = ep_opt.t2_low_share_thresh_medium
    ep_opt.real_t2_low_share_long = ep_opt.t2_low_share_thresh_long
    ep_opt.print_summary()

    dev_df = evaluate_global_dataset(dev_curves[: min(60, len(dev_curves))], ep_opt, label='development precise evaluation')
    dev_df.to_csv(os.path.join(OUTPUT_STAGE1, 'dev_eval_v31_1.csv'), index=False)

    val_curves = sim.batch_generate_mixed(n_curves=120, days=300, add_structured_noise=True)
    SIRSimulator.print_truth_distribution(val_curves, 'global_validation')
    val_df = evaluate_global_dataset(val_curves, ep_opt, label='global_validation')
    val_df.to_csv(os.path.join(OUTPUT_STAGE1, 'global_val_eval_v31_1.csv'), index=False)

    curve = dev_curves[0]
    example_ep = build_archetype_adaptive_ep(ep_opt, curve, prospective=False)
    seg = StructuralPhaseSegmenter(example_ep, eemd_trials=50)
    r = seg.segment(curve['I'], curve['t'])
    Visualizer.plot_global_segmentation(r, curve=curve, save_path=os.path.join(OUTPUT_STAGE1, 'example_case_v31_1.png'), title=f"Example: {curve.get('archetype','')} R0={curve['R0']:.2f}, GT={curve['GT']:.1f}d")
    return ep_opt, dev_curves, tune_curves, cache, grid_df


def run_stage2_validation(ep_opt: EpiParams, n_val: int = 180, window_size: int = 100, step: int = 2,
                          confirm_rounds: int = 2):
    print(f"\n{'#'*72}\n  STAGE 2: prospective rolling validation V31.1 adaptive\n  Goal: evaluate stable T1 confirmation on archetype-stratified sequential data, with T2_dyn/T2_dom dual references\n{'#'*72}")
    sim = SIRSimulator(seed=999)
    val_curves = sim.batch_generate_for_prospective(n_curves=n_val, days=300, add_structured_noise=True, min_Tp=40)
    SIRSimulator.print_truth_distribution(val_curves, 'prospective validation set (V31.1 adaptive)')
    SignalDiagnostics.diagnose_signal_complexity(val_curves, 'prospective validation set (V31.1 adaptive)')
    validator = RollingWindowValidator(
        ep_opt, window_size=window_size, step=step, eemd_trials=20, confirm_rounds=confirm_rounds,
        confirm_tol=max(4, int(np.ceil(1.2 * ep_opt.GT)))
    )
    val_results, val_df = validator.batch_validate(val_curves, label='prospective validation set (V31.1 adaptive)', verbose=True)
    val_df.to_csv(os.path.join(OUTPUT_STAGE2, 'rolling_validation_v31_1.csv'), index=False)
    Visualizer.plot_validation_summary(val_df, save_path=os.path.join(OUTPUT_STAGE2, 'validation_summary_v31_1.png'), label='prospective validation set (V31.1 adaptive)')
    return val_results, val_df


def load_existing_v31_1_ep_for_partial_run() -> EpiParams:
    """Load saved optimized V31.1 parameters without running the full pipeline."""
    ep = EpiParams(preset='omicron')
    for path in [BEST_PARAMS_PATH, T1_FOCUSED_PARAMS_PATH]:
        if not os.path.exists(path):
            print(f"  Partial-run parameter file not found, skipped: {path}")
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                params = json.load(f)
            ep.apply_override(params)
            print(f"  Loaded partial-run parameters: {path}")
        except Exception as exc:
            print(f"  Failed to load partial-run parameters {path}: {exc}")
    ep.score_smooth_sigma = max(1.0, float(getattr(ep, 'energy_smooth_sigma', 1.5)) * 0.7)
    ep.real_t2_low_share_short = ep.t2_low_share_thresh_short
    ep.real_t2_low_share_medium = ep.t2_low_share_thresh_medium
    ep.real_t2_low_share_long = ep.t2_low_share_thresh_long
    return ep


def run_stage2_influenza_quick_validation(ep_opt: EpiParams, n_val: int = 24,
                                          window_size: int = 70, step: int = 1,
                                          confirm_rounds: int = 2, eemd_trials: int = 8):
    print(f"\n{'#'*72}\n  QUICK STAGE 2: influenza_like prospective validation V31.1 adaptive\n{'#'*72}")
    sim = SIRSimulator(seed=4242)
    influenza_set = [
        copy.deepcopy(sc)
        for sc in SIRSimulator.ARCHETYPE_SCENARIOS
        if sc.get('archetype') == 'influenza_like'
    ]
    if not influenza_set:
        raise RuntimeError('influenza_like archetype was not found in ARCHETYPE_SCENARIOS.')
    val_curves = sim.batch_generate_archetype(
        n_curves=n_val, days=180, add_structured_noise=True,
        min_Tp=24, scenario_set=influenza_set,
    )
    SIRSimulator.print_truth_distribution(val_curves, 'influenza_like quick validation')
    SignalDiagnostics.diagnose_signal_complexity(val_curves, 'influenza_like quick validation')
    validator = RollingWindowValidator(
        ep_opt, window_size=window_size, step=step, eemd_trials=eemd_trials,
        confirm_rounds=confirm_rounds,
        confirm_tol=max(3, int(np.ceil(1.0 * ep_opt.GT))),
    )
    val_results, val_df = validator.batch_validate(
        val_curves, label='influenza_like quick validation (V31.1 adaptive)', verbose=True,
    )
    out_csv = os.path.join(OUTPUT_STAGE2, 'rolling_validation_influenza_like_quick_v31_1.csv')
    summary_csv = os.path.join(OUTPUT_STAGE2, 'rolling_validation_influenza_like_quick_summary_v31_1.csv')
    decision_csv = os.path.join(OUTPUT_STAGE2, 'rolling_validation_influenza_like_quick_decision_v31_1.csv')
    val_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    summary_df = _summarize_t1_validation(val_df, 'influenza_like_quick')
    summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')

    gt_vals = _numeric_series(val_df, 'GT')
    gt_med = float(gt_vals.median()) if len(gt_vals) else float(ep_opt.GT)
    t1_err = _numeric_series(val_df, 'T1_err')
    t1_delay = _numeric_series(val_df, 'T1_delay')
    fa_delay = _numeric_series(val_df, 'T1_first_alert_delay')
    lead_t2 = _numeric_series(val_df, 'lead_before_T2')
    t2_band = _numeric_series(val_df, 'T2_band_err')
    tp_err = _numeric_series(val_df, 'Tp_err')
    within_confirm = _rate_from_boolish(val_df, 'T1_confirm_within_2GT')
    within_first = _rate_from_boolish(val_df, 'T1_first_alert_within_2GT')
    pass_t1_err = bool(len(t1_err) and float(t1_err.median()) <= max(4.0, 1.6 * gt_med))
    pass_delay = bool(
        (len(t1_delay) and float(t1_delay.median()) <= 2.0 * gt_med)
        or (pd.notna(within_confirm) and within_confirm >= 0.80)
    )
    t2_p75 = float(t2_band.quantile(0.75)) if len(t2_band) else np.nan
    tp_p75 = float(tp_err.quantile(0.75)) if len(tp_err) else np.nan
    pass_t2 = bool(
        len(t2_band)
        and float(t2_band.median()) <= max(6.0, 2.2 * gt_med)
        and t2_p75 <= max(14.0, 4.5 * gt_med)
    )
    pass_tp = bool(
        len(tp_err)
        and float(tp_err.median()) <= max(6.0, 2.2 * gt_med)
        and tp_p75 <= max(14.0, 5.0 * gt_med)
    )
    pass_lead = bool(len(lead_t2) and float(lead_t2.median()) > 0.0)
    decision_row = dict(
        n=len(val_df),
        GT_median=gt_med,
        T1_err_median=float(t1_err.median()) if len(t1_err) else np.nan,
        T1_err_mean=float(t1_err.mean()) if len(t1_err) else np.nan,
        T1_delay_median=float(t1_delay.median()) if len(t1_delay) else np.nan,
        T1_delay_mean=float(t1_delay.mean()) if len(t1_delay) else np.nan,
        T1_confirm_within_2GT_rate=within_confirm,
        T1_first_alert_delay_median=float(fa_delay.median()) if len(fa_delay) else np.nan,
        T1_first_alert_within_2GT_rate=within_first,
        lead_before_T2_median=float(lead_t2.median()) if len(lead_t2) else np.nan,
        T2_band_err_median=float(t2_band.median()) if len(t2_band) else np.nan,
        T2_band_err_p75=t2_p75,
        T2_band_err_max=float(t2_band.max()) if len(t2_band) else np.nan,
        Tp_err_median=float(tp_err.median()) if len(tp_err) else np.nan,
        Tp_err_p75=tp_p75,
        Tp_err_max=float(tp_err.max()) if len(tp_err) else np.nan,
        pass_T1_error=pass_t1_err,
        pass_T1_delay=pass_delay,
        pass_T2_band=pass_t2,
        pass_Tp=pass_tp,
        pass_lead_before_T2=pass_lead,
    )
    decision_row['quick_pass'] = all([
        decision_row['pass_T1_error'],
        decision_row['pass_T1_delay'],
        decision_row['pass_T2_band'],
        decision_row['pass_Tp'],
        decision_row['pass_lead_before_T2'],
    ])
    pd.DataFrame([decision_row]).to_csv(decision_csv, index=False, encoding='utf-8-sig')
    print(f"\n  Quick influenza validation saved:")
    print(f"    detail:  {out_csv}")
    print(f"    summary: {summary_csv}")
    print(f"    decision:{decision_csv}")
    print(
        "\n  Quick influenza decision: "
        f"{'PASS' if decision_row['quick_pass'] else 'NEEDS_TUNING'} | "
        f"T1_delay_med={decision_row['T1_delay_median']:.1f}d, "
        f"within2GT={decision_row['T1_confirm_within_2GT_rate']:.2f}, "
        f"T2_band_med={decision_row['T2_band_err_median']:.1f}d, "
        f"T2_p75={decision_row['T2_band_err_p75']:.1f}d, "
        f"Tp_med={decision_row['Tp_err_median']:.1f}d, "
        f"Tp_p75={decision_row['Tp_err_p75']:.1f}d"
    )
    return val_results, val_df


CORE_ARCHETYPE_NAMES = ('influenza_like', 'moderate_coronavirus')


def _select_archetype_scenarios(archetype_names: Tuple[str, ...]) -> List[Dict]:
    wanted = {str(x).lower() for x in archetype_names}
    selected = [
        copy.deepcopy(sc)
        for sc in SIRSimulator.ARCHETYPE_SCENARIOS
        if str(sc.get('archetype', '')).lower() in wanted
    ]
    found = {str(sc.get('archetype', '')).lower() for sc in selected}
    missing = sorted(wanted - found)
    if missing:
        raise RuntimeError(f"Required core archetype(s) missing from ARCHETYPE_SCENARIOS: {missing}")
    return selected


def _core_archetype_design_table(n_per_archetype: int) -> pd.DataFrame:
    rows = []
    for sc in _select_archetype_scenarios(CORE_ARCHETYPE_NAMES):
        rows.append(dict(
            role='primary_core',
            archetype=sc['archetype'],
            R=SIRSimulator._format_range(sc['R0_range']),
            GT=SIRSimulator._format_range(sc['GT_range']),
            SI=SIRSimulator._format_range(sc['SI_range']),
            report_delay=SIRSimulator._format_range(sc['report_delay_range']),
            noise=SIRSimulator._format_range(sc['noise_std_range']),
            structural_noise=sc['structural_noise_profile'],
            scenario_weight=float(sc['weight']),
            sample_n=int(n_per_archetype),
            rationale='respiratory core archetype for H1N1/Omicron-facing algorithm validation',
        ))
    return pd.DataFrame(rows)


def run_stage2_core_archetype_validation(ep_opt: EpiParams, n_per_archetype: int = 80,
                                         seed: int = 5151, eemd_trials_default: int = 12,
                                         output_label: str = 'core_archetype'):
    """Validate only the two prespecified respiratory core archetypes for the main paper table."""
    print(f"\n{'#'*72}")
    print("  CORE STAGE 2: primary respiratory archetype validation V31.1 adaptive")
    print("  Archetypes: influenza_like + moderate_coronavirus")
    print(f"  n_per_archetype={n_per_archetype}")
    print(f"{'#'*72}")

    scenario_settings = {
        'influenza_like': dict(days=180, min_Tp=24, window_size=70, step=1,
                               confirm_rounds=2, eemd_trials=8),
        'moderate_coronavirus': dict(days=300, min_Tp=35, window_size=100, step=2,
                                     confirm_rounds=2, eemd_trials=eemd_trials_default),
    }
    all_results = []
    all_frames = []
    design_df = _core_archetype_design_table(n_per_archetype)
    design_path = os.path.join(OUTPUT_STAGE2, f'{output_label}_design_v31_1.csv')
    design_df.to_csv(design_path, index=False, encoding='utf-8-sig')

    for idx, sc in enumerate(_select_archetype_scenarios(CORE_ARCHETYPE_NAMES)):
        arch = sc['archetype']
        cfg = scenario_settings.get(arch, {})
        sim = SIRSimulator(seed=seed + idx * 101)
        val_curves = sim.batch_generate_archetype(
            n_curves=n_per_archetype,
            days=int(cfg.get('days', 300)),
            add_structured_noise=True,
            min_Tp=int(cfg.get('min_Tp', 35)),
            scenario_set=[sc],
        )
        label = f'core_{arch}_validation'
        SIRSimulator.print_truth_distribution(val_curves, label)
        SignalDiagnostics.diagnose_signal_complexity(val_curves, label)
        validator = RollingWindowValidator(
            ep_opt,
            window_size=int(cfg.get('window_size', 100)),
            step=int(cfg.get('step', 2)),
            eemd_trials=int(cfg.get('eemd_trials', eemd_trials_default)),
            confirm_rounds=int(cfg.get('confirm_rounds', 2)),
            confirm_tol=max(3, int(np.ceil(1.0 * ep_opt.GT))),
        )
        val_results, val_df = validator.batch_validate(
            val_curves, label=f'{label} (V31.1 adaptive)', verbose=True,
        )
        val_df['validation_role'] = 'primary_core'
        val_df['core_archetype'] = arch
        val_df['base_window_size'] = int(cfg.get('window_size', 100))
        val_df['base_step'] = int(cfg.get('step', 2))
        val_df['base_confirm_rounds'] = int(cfg.get('confirm_rounds', 2))
        val_df['eemd_trials_used'] = int(cfg.get('eemd_trials', eemd_trials_default))
        detail_path = os.path.join(OUTPUT_STAGE2, f'{output_label}_validation_{arch}_v31_1.csv')
        val_df.to_csv(detail_path, index=False, encoding='utf-8-sig')
        all_results.extend(val_results)
        all_frames.append(val_df)

    core_df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    detail_csv = os.path.join(OUTPUT_STAGE2, f'{output_label}_validation_v31_1.csv')
    summary_csv = os.path.join(OUTPUT_STAGE2, f'{output_label}_validation_summary_v31_1.csv')
    main_table_xlsx = os.path.join(OUTPUT_STAGE2, f'{output_label}_main_table_v31_1.xlsx')
    main_table_csv = os.path.join(OUTPUT_STAGE2, f'{output_label}_main_table_v31_1.csv')
    core_df.to_csv(detail_csv, index=False, encoding='utf-8-sig')
    summary_df = _summarize_core_archetype_validation(core_df)
    summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    summary_df.to_csv(main_table_csv, index=False, encoding='utf-8-sig')
    try:
        with pd.ExcelWriter(main_table_xlsx) as writer:
            summary_df.to_excel(writer, sheet_name='core_performance', index=False)
            design_df.to_excel(writer, sheet_name='core_design', index=False)
    except Exception as exc:
        print(f"  Core Excel export failed: {exc}")
        main_table_xlsx = ''

    print("\n  Core archetype validation saved:")
    print(f"    design:  {design_path}")
    print(f"    detail:  {detail_csv}")
    print(f"    summary: {summary_csv}")
    if main_table_xlsx:
        print(f"    xlsx:    {main_table_xlsx}")
    if len(summary_df):
        cols = ['group', 'n', 'T1_err_median', 'T1_delay_median',
                'T1_confirm_within_2GT_rate', 'T2_band_err_median', 'Tp_err_median']
        print("\n  Core main metrics:")
        print(summary_df[[c for c in cols if c in summary_df.columns]].to_string(index=False))
    return all_results, core_df, summary_df


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if df is None or col not in df.columns:
        return pd.Series([], dtype=float)
    return pd.to_numeric(df[col], errors='coerce').dropna()


def _rate_from_boolish(df: pd.DataFrame, col: str) -> float:
    vals = _numeric_series(df, col)
    return float(vals.mean()) if len(vals) else np.nan


def _summarize_core_archetype_validation(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    def num(col: str, sub: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(sub[col], errors='coerce').dropna() if col in sub else pd.Series([], dtype=float)

    def med(col: str, sub: pd.DataFrame) -> float:
        vals = num(col, sub)
        return float(vals.median()) if len(vals) else np.nan

    def mean(col: str, sub: pd.DataFrame) -> float:
        vals = num(col, sub)
        return float(vals.mean()) if len(vals) else np.nan

    def p75(col: str, sub: pd.DataFrame) -> float:
        vals = num(col, sub)
        return float(vals.quantile(0.75)) if len(vals) else np.nan

    def rate(col: str, sub: pd.DataFrame) -> float:
        vals = num(col, sub)
        return float(vals.mean()) if len(vals) else np.nan

    work = df.copy()
    group_col = 'core_archetype' if 'core_archetype' in work.columns else 'archetype'
    groups = [('core_all', work)]
    if group_col in work.columns:
        groups.extend([(str(k), v) for k, v in work.groupby(group_col, observed=False)])

    rows = []
    for group, sub in groups:
        if len(sub) == 0:
            continue
        lead_t2 = num('lead_before_T2', sub)
        lead_peak = num('lead_by_confirm', sub)
        first_lead_t2 = num('lead_before_T2_by_first_alert', sub)
        t1_signed = num('T1_estimate_signed_error', sub)
        row = dict(
            validation_role='primary_core' if group != 'core_all' else 'primary_core_combined',
            group=group,
            n=int(len(sub)),
            GT_median=med('GT', sub),
            SI_median=med('SI', sub),
            report_delay_median=med('report_delay', sub),
            R0_median=med('R0', sub),
            T1_err_median=med('T1_err', sub),
            T1_err_mean=mean('T1_err', sub),
            T1_err_p75=p75('T1_err', sub),
            T1_band_err_median=med('T1_band_err', sub),
            T1_estimate_signed_error_median=float(t1_signed.median()) if len(t1_signed) else np.nan,
            T1_delay_median=med('T1_delay', sub),
            T1_delay_mean=mean('T1_delay', sub),
            T1_delay_GT_norm_median=med('T1_delay_GT_norm', sub),
            T1_confirm_within_2GT_rate=rate('T1_confirm_within_2GT', sub),
            T1_first_alert_err_median=med('T1_first_alert_err', sub),
            T1_first_alert_delay_median=med('T1_first_alert_delay', sub),
            T1_first_alert_delay_GT_norm_median=med('T1_first_alert_delay_GT_norm', sub),
            T1_first_alert_within_2GT_rate=rate('T1_first_alert_within_2GT', sub),
            lead_before_T2_median=float(lead_t2.median()) if len(lead_t2) else np.nan,
            confirm_before_T2_rate=float((lead_t2 > 0).mean()) if len(lead_t2) else np.nan,
            lead_by_confirm_median=float(lead_peak.median()) if len(lead_peak) else np.nan,
            confirm_before_peak_rate=float((lead_peak > 0).mean()) if len(lead_peak) else np.nan,
            first_alert_before_T2_rate=float((first_lead_t2 > 0).mean()) if len(first_lead_t2) else np.nan,
            T2_band_err_median=med('T2_band_err', sub),
            T2_band_err_mean=mean('T2_band_err', sub),
            T2_band_err_p75=p75('T2_band_err', sub),
            Tp_err_median=med('Tp_err', sub),
            Tp_err_mean=mean('Tp_err', sub),
            Tp_err_p75=p75('Tp_err', sub),
            adaptive_window_size_median=med('adaptive_window_size', sub),
            adaptive_confirm_tol_median=med('adaptive_confirm_tol', sub),
            eemd_trials_used_median=med('eemd_trials_used', sub),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _infer_gt_from_source(source) -> float:
    text = str(source).lower()
    if 'h1n1' in text or '甲流' in text:
        return 2.6
    if 'covid' in text or 'omicron' in text or '新冠' in text:
        return 3.0
    if 'dengue' in text or '登革' in text:
        return 7.0
    if 'chik' in text or '基孔' in text:
        return 5.0
    return np.nan


def _score_t1_focused_validation(df: pd.DataFrame) -> float:
    if df is None or len(df) == 0:
        return float('inf')
    score = 0.0
    t1_err = _numeric_series(df, 'T1_err')
    if len(t1_err):
        score += 2.8 * t1_err.mean()
        score += max(0.0, t1_err.median() - 6.0) * 6.0
    t1_band = _numeric_series(df, 'T1_band_err')
    if len(t1_band):
        score += 1.5 * t1_band.mean()

    signed = _numeric_series(df, 'T1_estimate_signed_error')
    if len(signed):
        score += max(0.0, abs(float(signed.median())) - 1.0) * 5.0
        if 'GT' in df:
            gt = pd.to_numeric(df.loc[signed.index, 'GT'], errors='coerce')
            ok = gt.notna() & (gt > 0)
            if ok.any():
                signed_norm = signed[ok] / gt[ok]
                score += (signed[ok] < -gt[ok]).mean() * 95.0
                score += (signed[ok] > 1.5 * gt[ok]).mean() * 45.0
                score += max(0.0, -float(signed_norm.median()) - 0.50) * 70.0
                score += max(0.0, float(signed_norm.median()) - 1.00) * 45.0

    delay = _numeric_series(df, 'T1_delay')
    if len(delay):
        score += 5.5 * delay.clip(lower=0).mean()
        target = 6.0
        if 'GT' in df:
            gt = pd.to_numeric(df.loc[delay.index, 'GT'], errors='coerce')
            ok = gt.notna() & (gt > 0)
            if ok.any():
                target = max(4.0, 1.5 * float(gt[ok].median()))
                score += (delay[ok] < -gt[ok]).mean() * 35.0
                score += (delay[ok] > 2.0 * gt[ok]).mean() * 60.0
                score += (delay[ok] > 3.0 * gt[ok]).mean() * 45.0
        score += max(0.0, float(delay.median()) - target) * 12.0

    within = _numeric_series(df, 'T1_confirm_within_2GT')
    if len(within):
        score += max(0.0, 0.88 - within.mean()) * 75.0
    lead_t2 = _numeric_series(df, 'lead_before_T2')
    if len(lead_t2):
        score += (lead_t2 <= 0).mean() * 60.0
        score += (lead_t2 < -3).mean() * 45.0

    first_err = _numeric_series(df, 'T1_first_alert_err')
    if len(first_err):
        score += 1.0 * first_err.mean()
    first_delay = _numeric_series(df, 'T1_first_alert_delay')
    if len(first_delay):
        score += 1.8 * first_delay.clip(lower=0).mean()
        if 'GT' in df:
            gt = pd.to_numeric(df.loc[first_delay.index, 'GT'], errors='coerce')
            ok = gt.notna() & (gt > 0)
            if ok.any():
                score += (first_delay[ok] < -2.0 * gt[ok]).mean() * 20.0
                score += (first_delay[ok] > 2.0 * gt[ok]).mean() * 25.0
    first_signed = _numeric_series(df, 'T1_first_alert_signed_error')
    if len(first_signed) and 'GT' in df:
        gt = pd.to_numeric(df.loc[first_signed.index, 'GT'], errors='coerce')
        ok = gt.notna() & (gt > 0)
        if ok.any():
            score += (first_signed[ok] < -1.5 * gt[ok]).mean() * 35.0
            score += (first_signed[ok] > 2.0 * gt[ok]).mean() * 20.0
    if len(delay) and len(first_delay):
        common = delay.index.intersection(first_delay.index)
        if len(common) > 0:
            gap = delay.loc[common] - first_delay.loc[common]
            gt_gap = pd.to_numeric(df.loc[common, 'GT'], errors='coerce') if 'GT' in df else pd.Series(np.nan, index=common)
            target_gap = max(1.0, float(gt_gap.dropna().median()) if gt_gap.notna().any() else 3.0)
            score += max(0.0, target_gap - float(gap.median())) * 4.0
    first_within = _numeric_series(df, 'T1_first_alert_within_2GT')
    if len(first_within):
        score += max(0.0, 0.92 - first_within.mean()) * 30.0
    first_lead_t2 = _numeric_series(df, 'lead_before_T2_by_first_alert')
    if len(first_lead_t2):
        score += (first_lead_t2 <= 0).mean() * 25.0

    t2_band = _numeric_series(df, 'T2_band_err')
    if len(t2_band):
        score += 1.2 * t2_band.mean()
        score += max(0.0, float(t2_band.median()) - 8.0) * 6.0
        score += max(0.0, float(t2_band.mean()) - 12.0) * 2.5
        score += max(0.0, float(t2_band.quantile(0.90)) - 28.0) * 2.0
    tp_err = _numeric_series(df, 'Tp_err')
    if len(tp_err):
        score += 0.9 * tp_err.mean()
        score += max(0.0, float(tp_err.median()) - 6.0) * 4.0
        score += max(0.0, float(tp_err.mean()) - 14.0) * 2.5
        score += max(0.0, float(tp_err.quantile(0.90)) - 35.0) * 2.0

    # The T1-focused tuner used to over-reward very early warnings even when
    # T2/Tp collapsed far too early in slow contact-network archetypes. Keep T1
    # primary, but explicitly penalize directional phase-order drift.
    t2_signed = _numeric_series(df, 'T2_estimate_signed_error')
    if len(t2_signed) and 'GT' in df:
        gt = pd.to_numeric(df.loc[t2_signed.index, 'GT'], errors='coerce')
        ok = gt.notna() & (gt > 0)
        if ok.any():
            t2_norm = t2_signed[ok] / gt[ok]
            score += (t2_signed[ok] < -2.0 * gt[ok]).mean() * 90.0
            score += (t2_signed[ok] < -3.0 * gt[ok]).mean() * 60.0
            score += (t2_signed[ok] > 2.5 * gt[ok]).mean() * 35.0
            score += max(0.0, -float(t2_norm.median()) - 0.75) * 55.0
    tp_signed = _numeric_series(df, 'Tp_estimate_signed_error')
    if len(tp_signed) and 'GT' in df:
        gt = pd.to_numeric(df.loc[tp_signed.index, 'GT'], errors='coerce')
        ok = gt.notna() & (gt > 0)
        if ok.any():
            tp_norm = tp_signed[ok] / gt[ok]
            score += (tp_signed[ok] < -2.0 * gt[ok]).mean() * 120.0
            score += (tp_signed[ok] < -4.0 * gt[ok]).mean() * 100.0
            score += (tp_signed[ok] > 3.0 * gt[ok]).mean() * 45.0
            score += max(0.0, -float(tp_norm.median()) - 0.75) * 80.0
            score += max(0.0, float(tp_norm.median()) - 2.0) * 30.0

    if {'T1_estimate', 'T2_estimate', 'GT'}.issubset(df.columns):
        t1 = pd.to_numeric(df['T1_estimate'], errors='coerce')
        t2 = pd.to_numeric(df['T2_estimate'], errors='coerce')
        gt = pd.to_numeric(df['GT'], errors='coerce')
        ok = t1.notna() & t2.notna() & gt.notna() & (gt > 0)
        if ok.any():
            gap_norm = (t2[ok] - t1[ok]) / gt[ok]
            score += (gap_norm < 1.0).mean() * 80.0
            score += (gap_norm > 10.0).mean() * 20.0
    if {'T2_estimate', 'Tp_estimate'}.issubset(df.columns):
        t2 = pd.to_numeric(df['T2_estimate'], errors='coerce')
        tp = pd.to_numeric(df['Tp_estimate'], errors='coerce')
        ok = t2.notna() & tp.notna()
        if ok.any():
            score += (tp[ok] <= t2[ok]).mean() * 120.0
    return float(score)


def _summarize_t1_validation(df: pd.DataFrame, label: str = '') -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    work = df.copy()
    if 'reference_alert_days' not in work and {'truth_T2_dom', 'truth_T1_growth'}.issubset(work.columns):
        work['reference_alert_days'] = pd.to_numeric(work['truth_T2_dom'], errors='coerce') - pd.to_numeric(work['truth_T1_growth'], errors='coerce')
    if 'reference_t1_to_peak_days' not in work and {'truth_Tp', 'truth_T1_growth'}.issubset(work.columns):
        work['reference_t1_to_peak_days'] = pd.to_numeric(work['truth_Tp'], errors='coerce') - pd.to_numeric(work['truth_T1_growth'], errors='coerce')
    if 'T1_delay_GT_norm' not in work and {'T1_delay', 'GT'}.issubset(work.columns):
        work['T1_delay_GT_norm'] = pd.to_numeric(work['T1_delay'], errors='coerce') / pd.to_numeric(work['GT'], errors='coerce').replace(0, np.nan)
    if 'T1_first_alert_delay_GT_norm' not in work and {'T1_first_alert_delay', 'GT'}.issubset(work.columns):
        work['T1_first_alert_delay_GT_norm'] = pd.to_numeric(work['T1_first_alert_delay'], errors='coerce') / pd.to_numeric(work['GT'], errors='coerce').replace(0, np.nan)
    work['GT_bin'] = pd.cut(work['GT'], bins=[0, 4, 5, 6, 10], labels=['<4', '4-5', '5-6', '>=6'], right=False)
    rows = []
    groups = [('all', work)] + [(str(k), v) for k, v in work.groupby('GT_bin', observed=False)]
    for key, sub in groups:
        if len(sub) == 0:
            continue
        row = dict(label=label, group=key, n=len(sub))
        for col in ['GT', 'T1_err', 'T1_band_err', 'T1_delay', 'T1_delay_GT_norm',
                    'T1_estimate_signed_error',
                    'T2_band_err', 'T2_estimate_signed_error',
                    'Tp_err', 'Tp_estimate_signed_error',
                    'T1_first_alert_err', 'T1_first_alert_delay',
                    'T1_first_alert_delay_GT_norm', 'T1_first_alert_signed_error',
                    'lead_by_first_alert', 'lead_before_T2_by_first_alert',
                    'lead_by_confirm', 'lead_before_T2', 'reference_alert_days',
                    'reference_t1_to_peak_days']:
            if col in sub:
                vals = pd.to_numeric(sub[col], errors='coerce').dropna()
                if len(vals) > 0:
                    row[f'{col}_median'] = float(vals.median())
                    row[f'{col}_mean'] = float(vals.mean())
        if 'T1_confirm_within_2GT' in sub:
            within = pd.to_numeric(sub['T1_confirm_within_2GT'], errors='coerce').dropna()
            row['T1_confirm_within_2GT_rate'] = float(within.mean()) if len(within) else np.nan
        if 'lead_before_T2' in sub:
            lead = pd.to_numeric(sub['lead_before_T2'], errors='coerce').dropna()
            row['confirm_not_before_T2_rate'] = float((lead <= 0).mean()) if len(lead) else np.nan
            row['confirm_before_T2_rate'] = float((lead > 0).mean()) if len(lead) else np.nan
        if 'lead_by_confirm' in sub:
            lead_peak = pd.to_numeric(sub['lead_by_confirm'], errors='coerce').dropna()
            row['confirm_before_peak_rate'] = float((lead_peak > 0).mean()) if len(lead_peak) else np.nan
            row['confirm_ge_7d_before_peak_rate'] = float((lead_peak >= 7).mean()) if len(lead_peak) else np.nan
        if 'T1_first_alert_within_2GT' in sub:
            within = pd.to_numeric(sub['T1_first_alert_within_2GT'], errors='coerce').dropna()
            row['T1_first_alert_within_2GT_rate'] = float(within.mean()) if len(within) else np.nan
        if 'lead_before_T2_by_first_alert' in sub:
            lead = pd.to_numeric(sub['lead_before_T2_by_first_alert'], errors='coerce').dropna()
            row['first_alert_not_before_T2_rate'] = float((lead <= 0).mean()) if len(lead) else np.nan
            row['first_alert_before_T2_rate'] = float((lead > 0).mean()) if len(lead) else np.nan
        if 'lead_by_first_alert' in sub:
            lead_peak = pd.to_numeric(sub['lead_by_first_alert'], errors='coerce').dropna()
            row['first_alert_before_peak_rate'] = float((lead_peak > 0).mean()) if len(lead_peak) else np.nan
            row['first_alert_ge_7d_before_peak_rate'] = float((lead_peak >= 7).mean()) if len(lead_peak) else np.nan
        if 'T1_delay' in sub and 'GT' in sub:
            delay = pd.to_numeric(sub['T1_delay'], errors='coerce')
            gt = pd.to_numeric(sub['GT'], errors='coerce')
            ok = delay.notna() & gt.notna() & (gt > 0)
            if ok.any():
                row['T1_delay_le_1GT_rate'] = float((delay[ok] <= gt[ok]).mean())
                row['T1_delay_le_2GT_rate'] = float((delay[ok] <= 2.0 * gt[ok]).mean())
        if 'T1_first_alert_delay' in sub and 'GT' in sub:
            delay = pd.to_numeric(sub['T1_first_alert_delay'], errors='coerce')
            gt = pd.to_numeric(sub['GT'], errors='coerce')
            ok = delay.notna() & gt.notna() & (gt > 0)
            if ok.any():
                row['T1_first_alert_delay_le_1GT_rate'] = float((delay[ok] <= gt[ok]).mean())
                row['T1_first_alert_delay_le_2GT_rate'] = float((delay[ok] <= 2.0 * gt[ok]).mean())
        if 'T1_estimate_signed_error' in sub and 'GT' in sub:
            signed = pd.to_numeric(sub['T1_estimate_signed_error'], errors='coerce')
            gt = pd.to_numeric(sub['GT'], errors='coerce')
            ok = signed.notna() & gt.notna() & (gt > 0)
            if ok.any():
                row['T1_estimate_too_early_gt_rate'] = float((signed[ok] < -gt[ok]).mean())
                row['T1_estimate_too_late_gt_rate'] = float((signed[ok] > gt[ok]).mean())
                row['T1_estimate_within_1GT_rate'] = float((signed[ok].abs() <= gt[ok]).mean())
        if 'T1_confirmed_detected' in sub:
            row['T1_confirmed_detected_rate'] = _rate_from_boolish(sub, 'T1_confirmed_detected')
        if 'T1_confirmed_source' in sub:
            src = sub['T1_confirmed_source'].dropna().astype(str)
            row['energy_assisted_confirm_rate'] = float((src == 'energy_assisted').mean()) if len(src) else np.nan
        if 'T1_first_alert_source' in sub:
            src = sub['T1_first_alert_source'].dropna().astype(str)
            row['energy_assisted_first_alert_rate'] = float((src == 'energy_assisted').mean()) if len(src) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run_t1_focused_optimization(ep_opt: EpiParams, n_tune: int = 36, n_val: int = 180,
                                fast_mode: bool = True):
    print(f"\n{'#'*72}\n  STAGE 2B: T1-focused optimization and sensitivity V31.1 adaptive\n{'#'*72}")
    sim_tune = SIRSimulator(seed=2026)
    tune_curves = sim_tune.batch_generate_for_prospective(n_curves=n_tune, days=300, add_structured_noise=True, min_Tp=40)
    confirm_variants = [
        dict(t1_confirm_ewma_k=0.38, t1_fast_confirm_strength=0.50, t1_single_confirm_strength=0.48,
             t1_valid_strength_min=0.18, t1_fast_local_margin=0.00, t1_confirm_span_gt=0.60,
             t1_early_confirm_strength=0.28, t1_early_confirm_min_persist_gt=0.35,
             t1_early_confirm_local_margin=-0.05,
             t1_confirm_single_min_persist_gt=0.25, t1_confirm_single_local_margin=0.03,
             t1_estimate_backtrack_gt=0.75, t1_estimate_max_backtrack_gt=0.75,
             t1_estimate_backtrack_min_persist_gt=0.90,
             t1_estimate_history_quantile=10, t1_aux_min_strength=0.12, t1_aux_max_lead_gt=4.0,
             t1_aux_stale_candidate_gt=2.0, t1_aux_stale_backcast_gt=0.35,
             t1_struct_stale_candidate_gt=3.5, t1_struct_stale_backcast_gt=0.60,
             t1_first_alert_strength=0.16, t1_first_alert_local_margin=-0.14,
             t1_first_alert_min_persist_gt=0.0, t1_first_alert_allow_energy=True,
             t1_raw_growth_z=1.8, t1_raw_growth_min_persist_gt=0.45,
             t1_raw_growth_confirm_strength=0.56, t1_raw_growth_min_abs_frac=0.03,
             t1_raw_growth_backcast_gt=1.20,
             prospective_min_window_gt=2.6, prospective_min_window_floor=10,
             prospective_t2_min_after_t1_gt=1.00, prospective_t2_strength_min=0.38,
             prospective_t2_confirm_rounds=2, prospective_t2_confirm_tol_gt=1.25,
             prospective_tp_maturity_gt=1.50, prospective_tp_decline_frac=0.06,
             prospective_tp_confirm_rounds=2, prospective_tp_confirm_tol_gt=2.00),
        dict(t1_confirm_ewma_k=0.42, t1_fast_confirm_strength=0.52, t1_single_confirm_strength=0.50,
             t1_valid_strength_min=0.20, t1_fast_local_margin=0.01, t1_confirm_span_gt=0.65,
             t1_early_confirm_strength=0.30, t1_early_confirm_min_persist_gt=0.45,
             t1_early_confirm_local_margin=-0.04,
             t1_confirm_single_min_persist_gt=0.35, t1_confirm_single_local_margin=0.05,
             t1_estimate_backtrack_gt=0.65, t1_estimate_max_backtrack_gt=0.65,
             t1_estimate_backtrack_min_persist_gt=1.00,
             t1_estimate_history_quantile=15, t1_aux_min_strength=0.13, t1_aux_max_lead_gt=3.75,
             t1_aux_stale_candidate_gt=2.25, t1_aux_stale_backcast_gt=0.40,
             t1_struct_stale_candidate_gt=4.0, t1_struct_stale_backcast_gt=0.70,
             t1_first_alert_strength=0.18, t1_first_alert_local_margin=-0.12,
             t1_first_alert_min_persist_gt=0.0, t1_first_alert_allow_energy=True,
             t1_raw_growth_z=2.0, t1_raw_growth_min_persist_gt=0.55,
             t1_raw_growth_confirm_strength=0.60, t1_raw_growth_min_abs_frac=0.04,
             t1_raw_growth_backcast_gt=1.10,
             prospective_min_window_gt=2.8, prospective_min_window_floor=11,
             prospective_t2_min_after_t1_gt=1.15, prospective_t2_strength_min=0.40,
             prospective_t2_confirm_rounds=2, prospective_t2_confirm_tol_gt=1.10,
             prospective_tp_maturity_gt=1.75, prospective_tp_decline_frac=0.08,
             prospective_tp_confirm_rounds=2, prospective_tp_confirm_tol_gt=1.75),
        dict(t1_confirm_ewma_k=0.46, t1_fast_confirm_strength=0.55, t1_single_confirm_strength=0.52,
             t1_valid_strength_min=0.22, t1_fast_local_margin=0.015, t1_confirm_span_gt=0.75,
             t1_early_confirm_strength=0.32, t1_early_confirm_min_persist_gt=0.50,
             t1_early_confirm_local_margin=-0.03,
             t1_confirm_single_min_persist_gt=0.45, t1_confirm_single_local_margin=0.07,
             t1_estimate_backtrack_gt=0.55, t1_estimate_max_backtrack_gt=0.55,
             t1_estimate_backtrack_min_persist_gt=1.10,
             t1_estimate_history_quantile=20, t1_aux_min_strength=0.14, t1_aux_max_lead_gt=3.5,
             t1_aux_stale_candidate_gt=2.5, t1_aux_stale_backcast_gt=0.45,
             t1_struct_stale_candidate_gt=4.0, t1_struct_stale_backcast_gt=0.80,
             t1_first_alert_strength=0.20, t1_first_alert_local_margin=-0.10,
             t1_first_alert_min_persist_gt=0.15, t1_first_alert_allow_energy=True,
             t1_raw_growth_z=2.2, t1_raw_growth_min_persist_gt=0.60,
             t1_raw_growth_confirm_strength=0.62, t1_raw_growth_min_abs_frac=0.04,
             t1_raw_growth_backcast_gt=1.00,
             prospective_min_window_gt=3.0, prospective_min_window_floor=12,
             prospective_t2_min_after_t1_gt=1.25, prospective_t2_strength_min=0.42,
             prospective_t2_confirm_rounds=3, prospective_t2_confirm_tol_gt=1.00,
             prospective_tp_maturity_gt=2.00, prospective_tp_decline_frac=0.10,
             prospective_tp_confirm_rounds=3, prospective_tp_confirm_tol_gt=1.50),
        dict(t1_confirm_ewma_k=0.50, t1_fast_confirm_strength=0.58, t1_single_confirm_strength=0.54,
             t1_valid_strength_min=0.24, t1_fast_local_margin=0.02, t1_confirm_span_gt=0.8,
             t1_early_confirm_strength=0.34, t1_early_confirm_min_persist_gt=0.55,
             t1_early_confirm_local_margin=-0.02,
             t1_confirm_single_min_persist_gt=0.55, t1_confirm_single_local_margin=0.09,
             t1_estimate_backtrack_gt=0.45, t1_estimate_max_backtrack_gt=0.45,
             t1_estimate_backtrack_min_persist_gt=1.20,
             t1_estimate_history_quantile=25, t1_aux_min_strength=0.15, t1_aux_max_lead_gt=3.25,
             t1_aux_stale_candidate_gt=3.0, t1_aux_stale_backcast_gt=0.50,
             t1_struct_stale_candidate_gt=4.5, t1_struct_stale_backcast_gt=0.90,
             t1_first_alert_strength=0.22, t1_first_alert_local_margin=-0.08,
             t1_first_alert_min_persist_gt=0.20, t1_first_alert_allow_energy=True,
             t1_raw_growth_z=2.4, t1_raw_growth_min_persist_gt=0.70,
             t1_raw_growth_confirm_strength=0.65, t1_raw_growth_min_abs_frac=0.05,
             t1_raw_growth_backcast_gt=0.90,
             prospective_min_window_gt=3.2, prospective_min_window_floor=12,
             prospective_t2_min_after_t1_gt=1.50, prospective_t2_strength_min=0.45,
             prospective_t2_confirm_rounds=3, prospective_t2_confirm_tol_gt=0.90,
             prospective_tp_maturity_gt=2.25, prospective_tp_decline_frac=0.12,
             prospective_tp_confirm_rounds=3, prospective_tp_confirm_tol_gt=1.25),
        dict(t1_confirm_ewma_k=0.62, t1_fast_confirm_strength=0.62, t1_single_confirm_strength=0.58,
             t1_valid_strength_min=0.28, t1_fast_local_margin=0.04, t1_confirm_span_gt=1.0,
             t1_confirm_single_min_persist_gt=0.60, t1_confirm_single_local_margin=0.12,
             t1_estimate_backtrack_gt=0.40, t1_estimate_max_backtrack_gt=0.40,
             t1_estimate_backtrack_min_persist_gt=1.25,
             t1_first_alert_strength=0.26, t1_first_alert_local_margin=-0.04,
             t1_first_alert_min_persist_gt=0.35, t1_first_alert_allow_energy=True,
             t1_raw_growth_z=2.6, t1_raw_growth_min_persist_gt=0.80,
             t1_raw_growth_confirm_strength=0.68, t1_raw_growth_min_abs_frac=0.06,
             t1_raw_growth_backcast_gt=0.80,
             prospective_min_window_gt=3.5, prospective_min_window_floor=14,
             prospective_t2_min_after_t1_gt=1.75, prospective_t2_strength_min=0.48,
             prospective_t2_confirm_rounds=3, prospective_t2_confirm_tol_gt=0.80,
             prospective_tp_maturity_gt=2.50, prospective_tp_decline_frac=0.14,
             prospective_tp_confirm_rounds=3, prospective_tp_confirm_tol_gt=1.25),
        dict(t1_confirm_ewma_k=0.70, t1_fast_confirm_strength=0.65, t1_single_confirm_strength=0.60,
             t1_valid_strength_min=0.30, t1_fast_local_margin=0.05, t1_confirm_span_gt=1.1,
             t1_confirm_single_min_persist_gt=0.70, t1_confirm_single_local_margin=0.14,
             t1_estimate_backtrack_gt=0.30, t1_estimate_max_backtrack_gt=0.30,
             t1_estimate_backtrack_min_persist_gt=1.35,
             t1_first_alert_strength=0.28, t1_first_alert_local_margin=-0.02,
             t1_first_alert_min_persist_gt=0.45, t1_first_alert_allow_energy=True,
             t1_raw_growth_z=2.8, t1_raw_growth_min_persist_gt=0.90,
             t1_raw_growth_confirm_strength=0.72, t1_raw_growth_min_abs_frac=0.08,
             t1_raw_growth_backcast_gt=0.70,
             prospective_min_window_gt=4.0, prospective_min_window_floor=14,
             prospective_t2_min_after_t1_gt=2.00, prospective_t2_strength_min=0.50,
             prospective_t2_confirm_rounds=4, prospective_t2_confirm_tol_gt=0.75,
             prospective_tp_maturity_gt=2.75, prospective_tp_decline_frac=0.16,
             prospective_tp_confirm_rounds=4, prospective_tp_confirm_tol_gt=1.00),
    ]
    if fast_mode:
        window_grid = [45, 55, 65, 75]
        step_grid = [1, 2]
        confirm_grid = [2, 3]
        variant_grid = confirm_variants[0:5]
        tune_trials = 6
        sens_trials = 8
    else:
        window_grid = [50, 60, 70, 80, 90, 100]
        step_grid = [1, 2, 3]
        confirm_grid = [2, 3]
        variant_grid = confirm_variants
        tune_trials = 10
        sens_trials = 12
    run_grid = []
    for window_size in window_grid:
        for step in step_grid:
            for confirm_rounds in confirm_grid:
                for vi, variant in enumerate(variant_grid, 1):
                    run_grid.append((window_size, step, confirm_rounds, vi, variant))
    records = []
    best_score = float('inf')
    best_record = None
    t0 = time.time()
    for idx, (window_size, step, confirm_rounds, vi, variant) in enumerate(run_grid, 1):
        print(f"  T1 tune [{idx:02d}/{len(run_grid)}] win={window_size} step={step} rounds={confirm_rounds} v{vi} ...", flush=True)
        ep_test = copy.deepcopy(ep_opt)
        ep_test.apply_override(variant)
        validator = RollingWindowValidator(
            ep_test, window_size=window_size, step=step, eemd_trials=tune_trials,
            confirm_rounds=confirm_rounds, confirm_tol=max(4, int(np.ceil(1.1 * ep_test.GT)))
        )
        _, df_tune = validator.batch_validate(tune_curves, label=f't1_focus_tune_{idx}', verbose=False)
        score = _score_t1_focused_validation(df_tune)
        def _med(col):
            vals = _numeric_series(df_tune, col)
            return float(vals.median()) if len(vals) else np.nan
        def _mean(col):
            vals = _numeric_series(df_tune, col)
            return float(vals.mean()) if len(vals) else np.nan
        lead_confirm = _numeric_series(df_tune, 'lead_before_T2')
        lead_first = _numeric_series(df_tune, 'lead_before_T2_by_first_alert')
        rec = dict(
            rank_input=idx, score=score, window_size=window_size, step=step,
            confirm_rounds=confirm_rounds, confirm_variant=vi,
            T1_err_median=_med('T1_err'),
            T1_signed_median=_med('T1_estimate_signed_error'),
            T1_delay_median=_med('T1_delay'),
            T1_delay_mean=_mean('T1_delay'),
            T1_confirm_within_2GT=_mean('T1_confirm_within_2GT'),
            T2_band_median=_med('T2_band_err'),
            T2_signed_median=_med('T2_estimate_signed_error'),
            Tp_err_median=_med('Tp_err'),
            Tp_signed_median=_med('Tp_estimate_signed_error'),
            lead_by_confirm_median=_med('lead_by_confirm'),
            lead_before_T2_median=_med('lead_before_T2'),
            confirm_not_before_T2_rate=float((lead_confirm <= 0).mean()) if len(lead_confirm) else np.nan,
            T1_first_alert_err_median=_med('T1_first_alert_err'),
            T1_first_alert_delay_median=_med('T1_first_alert_delay'),
            T1_first_alert_within_2GT=_mean('T1_first_alert_within_2GT'),
            lead_by_first_alert_median=_med('lead_by_first_alert'),
            lead_before_T2_by_first_alert_median=_med('lead_before_T2_by_first_alert'),
            first_alert_not_before_T2_rate=float((lead_first <= 0).mean()) if len(lead_first) else np.nan,
            **variant
        )
        records.append(rec)
        if score < best_score:
            best_score = score
            best_record = rec.copy()
            print(f"  * T1 best [{idx:02d}/{len(run_grid)}] score={score:.2f} delay_med={rec['T1_delay_median']:.1f}d within2GT={rec['T1_confirm_within_2GT']*100:.1f}% first_delay={rec['T1_first_alert_delay_median']:.1f}d first2GT={rec['T1_first_alert_within_2GT']*100:.1f}% win={window_size} step={step} rounds={confirm_rounds} v{vi}")
        else:
            print(f"    score={score:.2f} delay_med={rec['T1_delay_median']:.1f}d first_delay={rec['T1_first_alert_delay_median']:.1f}d elapsed={(time.time()-t0)/60:.1f}min", flush=True)
    search_df = pd.DataFrame(records).sort_values('score')
    search_df.to_csv(os.path.join(OUTPUT_STAGE2, 't1_focused_search_v31_1.csv'), index=False)
    if best_record is None:
        print('  T1-focused optimization failed; keep input params.')
        return ep_opt, {}, pd.DataFrame()
    best_params = {
        k: v for k, v in best_record.items()
        if k in ('window_size', 'step', 'confirm_rounds')
        or k.startswith('t1_') or k.startswith('prospective_')
    }
    with open(T1_FOCUSED_PARAMS_PATH, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)
    ep_t1 = copy.deepcopy(ep_opt)
    ep_t1.apply_override({
        k: v for k, v in best_params.items()
        if k.startswith('t1_') or k.startswith('prospective_')
    })
    sim_val = SIRSimulator(seed=2027)
    val_curves = sim_val.batch_generate_for_prospective(n_curves=n_val, days=300, add_structured_noise=True, min_Tp=40)
    validator = RollingWindowValidator(
        ep_t1, window_size=int(best_params['window_size']), step=int(best_params['step']),
        eemd_trials=20, confirm_rounds=int(best_params['confirm_rounds']),
        confirm_tol=max(4, int(np.ceil(1.1 * ep_t1.GT)))
    )
    _, val_df = validator.batch_validate(val_curves, label='T1-focused validation', verbose=True)
    val_df.to_csv(os.path.join(OUTPUT_STAGE2, 't1_focused_validation_v31_1.csv'), index=False)
    _summarize_t1_validation(val_df, 't1_focused_validation').to_csv(
        os.path.join(OUTPUT_STAGE2, 't1_focused_validation_summary_v31_1.csv'), index=False
    )
    sensitivity_rows = []
    base_win = int(best_params['window_size'])
    base_step = int(best_params['step'])
    base_rounds = int(best_params['confirm_rounds'])
    sens_configs = [
        ('best', base_win, base_step, base_rounds),
        ('shorter_window', max(50, base_win - 10), base_step, base_rounds),
        ('longer_window', base_win + 20, base_step, base_rounds),
        ('step_1', base_win, 1, base_rounds),
        ('step_3', base_win, 3, base_rounds),
        ('rounds_2_min', base_win, base_step, 2),
        ('rounds_3', base_win, base_step, 3),
    ]
    for label, window_size, step_s, confirm_rounds_s in sens_configs:
        validator_s = RollingWindowValidator(
            ep_t1, window_size=window_size, step=step_s, eemd_trials=sens_trials,
            confirm_rounds=confirm_rounds_s, confirm_tol=max(4, int(np.ceil(1.1 * ep_t1.GT)))
        )
        _, df_s = validator_s.batch_validate(val_curves[:min(30, len(val_curves))], label=f'sensitivity_{label}', verbose=False)
        lead_confirm = _numeric_series(df_s, 'lead_before_T2')
        lead_first = _numeric_series(df_s, 'lead_before_T2_by_first_alert')
        sensitivity_rows.append(dict(
            label=label, window_size=window_size, step=step_s, confirm_rounds=confirm_rounds_s,
            score=_score_t1_focused_validation(df_s),
            T1_err_median=float(_numeric_series(df_s, 'T1_err').median()) if len(_numeric_series(df_s, 'T1_err')) else np.nan,
            T1_delay_median=float(_numeric_series(df_s, 'T1_delay').median()) if len(_numeric_series(df_s, 'T1_delay')) else np.nan,
            T1_confirm_within_2GT=float(_numeric_series(df_s, 'T1_confirm_within_2GT').mean()) if len(_numeric_series(df_s, 'T1_confirm_within_2GT')) else np.nan,
            lead_by_confirm_median=float(_numeric_series(df_s, 'lead_by_confirm').median()) if len(_numeric_series(df_s, 'lead_by_confirm')) else np.nan,
            confirm_not_before_T2_rate=float((lead_confirm <= 0).mean()) if len(lead_confirm) else np.nan,
            T1_first_alert_err_median=float(_numeric_series(df_s, 'T1_first_alert_err').median()) if len(_numeric_series(df_s, 'T1_first_alert_err')) else np.nan,
            T1_first_alert_delay_median=float(_numeric_series(df_s, 'T1_first_alert_delay').median()) if len(_numeric_series(df_s, 'T1_first_alert_delay')) else np.nan,
            T1_first_alert_within_2GT=float(_numeric_series(df_s, 'T1_first_alert_within_2GT').mean()) if len(_numeric_series(df_s, 'T1_first_alert_within_2GT')) else np.nan,
            lead_by_first_alert_median=float(_numeric_series(df_s, 'lead_by_first_alert').median()) if len(_numeric_series(df_s, 'lead_by_first_alert')) else np.nan,
            first_alert_not_before_T2_rate=float((lead_first <= 0).mean()) if len(lead_first) else np.nan,
        ))
    pd.DataFrame(sensitivity_rows).to_csv(os.path.join(OUTPUT_STAGE2, 't1_focused_sensitivity_v31_1.csv'), index=False)
    return ep_t1, best_params, val_df


def _stage3_rolling_reference_row(signal: np.ndarray, ep: EpiParams, disease: str, city: str,
                                  T1_ref: int, T2_ref: int, Tp_ref: int, curve_id=None,
                                  window_size: int = 100, step: int = 3,
                                  eemd_trials: int = 8, confirm_rounds: int = 2,
                                  sig_valid_start: Optional[int] = None,
                                  bg_end: Optional[int] = None,
                                  T1_low: Optional[int] = None, T1_high: Optional[int] = None,
                                  candidate_max_early: Optional[int] = None) -> Dict:
    validator = RollingWindowValidator(
        ep, window_size=window_size, step=step, eemd_trials=eemd_trials,
        confirm_rounds=confirm_rounds, confirm_tol=max(4, int(np.ceil(1.1 * ep.GT)))
    )
    signal = np.maximum(np.nan_to_num(signal, nan=0), 0).astype(float)
    if candidate_max_early is None:
        candidate_max_early = max(2 * int(np.ceil(ep.GT)), 7)
    min_start_override = None
    if sig_valid_start is not None and pd.notna(sig_valid_start):
        min_start_override = max(0, int(sig_valid_start))
    elif bg_end is not None and pd.notna(bg_end):
        min_start_override = max(0, int(bg_end))
    candidate_min_day = max(0, int(T1_ref) - int(candidate_max_early))
    if min_start_override is not None:
        candidate_min_day = max(candidate_min_day, int(min_start_override) - int(np.ceil(ep.GT)))
    r = validator.detect_single(
        signal,
        truth_T1_growth=int(T1_ref), truth_T1_struct=int(T1_ref),
        truth_T2_dyn=int(T2_ref), truth_T2_dom=int(T2_ref),
        truth_Tp=int(Tp_ref), curve_id=curve_id, GT=ep.GT,
        min_start_override=min_start_override,
        candidate_min_day=candidate_min_day,
        candidate_max_early=candidate_max_early
    )
    t1_confirmed = r.get('T1_confirmed')
    t1_estimate = r.get('T1_estimate')
    t2_estimate = r.get('T2_estimate')
    tp_estimate = r.get('Tp_estimate')
    t2_confirmed = r.get('T2_confirmed')
    tp_confirmed = r.get('Tp_confirmed')
    lead_t2 = r.get('lead_before_T2')
    lead_peak = r.get('lead_by_confirm')
    t1_first_alert_day = r.get('T1_first_alert_day')
    t1_first_alert_estimate = r.get('T1_first_alert_estimate')
    lead_t2_first = r.get('lead_before_T2_by_first_alert')
    lead_peak_first = r.get('lead_by_first_alert')
    estimate_signed_error = (t1_estimate - int(T1_ref)) if t1_estimate is not None else np.nan
    t2_signed_error = (t2_estimate - int(T2_ref)) if t2_estimate is not None else np.nan
    tp_signed_error = (tp_estimate - int(Tp_ref)) if tp_estimate is not None else np.nan
    t2_confirmed_signed_error = (
        t2_estimate - int(T2_ref)
        if (t2_estimate is not None and t2_confirmed is not None) else np.nan
    )
    tp_confirmed_signed_error = (
        tp_estimate - int(Tp_ref)
        if (tp_estimate is not None and tp_confirmed is not None) else np.nan
    )
    t2_confirm_lag = (t2_confirmed - int(T2_ref)) if t2_confirmed is not None else np.nan
    tp_confirm_lag = (tp_confirmed - int(Tp_ref)) if tp_confirmed is not None else np.nan
    t2_confirm_before_peak = (
        bool(t2_confirmed <= int(Tp_ref))
        if t2_confirmed is not None and Tp_ref is not None and pd.notna(Tp_ref) else np.nan
    )
    tp_confirm_after_peak = (
        bool(tp_confirmed >= int(Tp_ref))
        if tp_confirmed is not None and Tp_ref is not None and pd.notna(Tp_ref) else np.nan
    )
    t2_confirm_within_2gt = (
        bool(abs(float(t2_confirm_lag)) <= 2.0 * float(ep.GT))
        if pd.notna(t2_confirm_lag) else np.nan
    )
    tp_confirm_within_2gt = (
        bool(abs(float(tp_confirm_lag)) <= 2.0 * float(ep.GT))
        if pd.notna(tp_confirm_lag) else np.nan
    )
    confirm_lag_vs_retro = (t1_confirmed - int(T1_ref)) if t1_confirmed is not None else np.nan
    first_alert_estimate_signed_error = (
        t1_first_alert_estimate - int(T1_ref)
        if t1_first_alert_estimate is not None else np.nan
    )
    first_alert_lag_vs_retro = (
        t1_first_alert_day - int(T1_ref)
        if t1_first_alert_day is not None else np.nan
    )
    retro_alert_days = int(T2_ref) - int(T1_ref)
    retro_t1_to_peak_days = int(Tp_ref) - int(T1_ref)
    delay_gt_norm = (float(confirm_lag_vs_retro) / max(float(ep.GT), 1e-10)) if pd.notna(confirm_lag_vs_retro) else np.nan
    first_alert_lag_gt_norm = (float(first_alert_lag_vs_retro) / max(float(ep.GT), 1e-10)) if pd.notna(first_alert_lag_vs_retro) else np.nan
    if T1_low is not None and T1_high is not None and pd.notna(T1_low) and pd.notna(T1_high):
        band_err = band_distance(t1_estimate, int(T1_low), int(T1_high))
        confirm_band_lag = band_distance(t1_confirmed, int(T1_low), int(T1_high))
    else:
        band_err = r.get('T1_err')
        confirm_band_lag = abs(confirm_lag_vs_retro) if pd.notna(confirm_lag_vs_retro) else np.nan
    return dict(
        disease=disease, city=city, curve_id=curve_id,
        T1_retro=int(T1_ref), T2_retro=int(T2_ref), T_peak_retro=int(Tp_ref),
        retro_alert_days=retro_alert_days, retro_T1_to_peak_days=retro_t1_to_peak_days,
        T1_retro_low=T1_low, T1_retro_high=T1_high,
        sig_valid_start=sig_valid_start, bg_end=bg_end, candidate_min_day=candidate_min_day,
        T1_estimate=t1_estimate, T1_confirmed_day=t1_confirmed,
        T2_estimate=t2_estimate, T2_confirmed_day=t2_confirmed,
        Tp_estimate=tp_estimate, Tp_confirmed_day=tp_confirmed,
        T1_err_vs_retro=r.get('T1_err'),
        T1_estimate_signed_error_vs_retro=estimate_signed_error,
        T2_signed_error_vs_retro=t2_signed_error,
        Tp_signed_error_vs_retro=tp_signed_error,
        T2_confirmed_estimate_signed_error_vs_retro=t2_confirmed_signed_error,
        Tp_confirmed_estimate_signed_error_vs_retro=tp_confirmed_signed_error,
        T2_confirm_lag_vs_retro=t2_confirm_lag,
        Tp_confirm_lag_vs_retro=tp_confirm_lag,
        T2_confirm_before_retro_peak=t2_confirm_before_peak,
        Tp_confirm_after_retro_peak=tp_confirm_after_peak,
        T2_confirm_within_2GT=t2_confirm_within_2gt,
        Tp_confirm_within_2GT=tp_confirm_within_2gt,
        T1_delay_vs_retro=r.get('T1_delay'),
        T1_confirm_lag_vs_retro=confirm_lag_vs_retro,
        T1_confirm_lag_GT_norm=delay_gt_norm,
        T1_first_alert_day=t1_first_alert_day,
        T1_first_alert_estimate=t1_first_alert_estimate,
        T1_first_alert_source=r.get('T1_first_alert_source'),
        T1_first_alert_strength=r.get('T1_first_alert_strength'),
        T1_first_alert_lag_vs_retro=first_alert_lag_vs_retro,
        T1_first_alert_estimate_signed_error_vs_retro=first_alert_estimate_signed_error,
        T1_first_alert_lag_GT_norm=first_alert_lag_gt_norm,
        T1_band_err_vs_retro=band_err,
        T1_confirm_band_lag_vs_retro=confirm_band_lag,
        T1_confirm_within_2GT=r.get('T1_confirm_within_2GT'),
        lead_to_T2_by_confirm=lead_t2, lead_to_peak_by_confirm=lead_peak,
        lead_to_T2_by_first_alert=lead_t2_first,
        lead_to_peak_by_first_alert=lead_peak_first,
        confirmed_before_T2=(float(lead_t2) > 0) if pd.notna(lead_t2) else np.nan,
        confirmed_before_peak=(float(lead_peak) > 0) if pd.notna(lead_peak) else np.nan,
        first_alert_before_T2=(float(lead_t2_first) > 0) if pd.notna(lead_t2_first) else np.nan,
        first_alert_before_peak=(float(lead_peak_first) > 0) if pd.notna(lead_peak_first) else np.nan,
        T1_ever_detected=r.get('T1_ever_detected'),
        T1_confirmed_detected=t1_confirmed is not None,
        T1_confirmed_source=r.get('T1_confirmed_source'),
        structural_failure=r.get('structural_failure'),
        GT=ep.GT, window_size_used=r.get('window_size_used'),
        min_start_used=r.get('min_start_used'),
        T1_hist_low=r.get('T1_hist_low'), T1_hist_high=r.get('T1_hist_high'),
    )


def _summarize_stage3_rolling_reference(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    rows = []
    for disease, sub in df.groupby('disease', dropna=False):
        row = dict(disease=disease, n=len(sub))
        for col in ['T1_err_vs_retro', 'T1_estimate_signed_error_vs_retro',
                    'T2_signed_error_vs_retro', 'Tp_signed_error_vs_retro',
                    'T2_confirmed_estimate_signed_error_vs_retro',
                    'Tp_confirmed_estimate_signed_error_vs_retro',
                    'T2_confirm_lag_vs_retro', 'Tp_confirm_lag_vs_retro',
                    'T1_delay_vs_retro', 'T1_confirm_lag_vs_retro',
                    'T1_first_alert_lag_vs_retro',
                    'T1_first_alert_estimate_signed_error_vs_retro',
                    'T1_confirm_lag_GT_norm', 'retro_alert_days', 'retro_T1_to_peak_days',
                    'T1_first_alert_lag_GT_norm',
                    'T1_band_err_vs_retro', 'T1_confirm_band_lag_vs_retro',
                    'lead_to_T2_by_confirm', 'lead_to_peak_by_confirm',
                    'lead_to_T2_by_first_alert', 'lead_to_peak_by_first_alert']:
            vals = pd.to_numeric(sub[col], errors='coerce').dropna() if col in sub else pd.Series([], dtype=float)
            if len(vals) > 0:
                row[f'{col}_median'] = float(vals.median())
                row[f'{col}_mean'] = float(vals.mean())
                row[f'{col}_q25'] = float(vals.quantile(0.25))
                row[f'{col}_q75'] = float(vals.quantile(0.75))
        for col in ['T1_confirm_within_2GT', 'confirmed_before_T2', 'confirmed_before_peak',
                    'first_alert_before_T2', 'first_alert_before_peak',
                    'T2_confirm_before_retro_peak', 'Tp_confirm_after_retro_peak',
                    'T2_confirm_within_2GT', 'Tp_confirm_within_2GT',
                    'T1_ever_detected', 'T1_confirmed_detected']:
            vals = pd.to_numeric(sub[col], errors='coerce').dropna() if col in sub else pd.Series([], dtype=float)
            row[f'{col}_rate'] = float(vals.mean()) if len(vals) else np.nan
        if 'lead_to_peak_by_confirm' in sub:
            lead_peak = pd.to_numeric(sub['lead_to_peak_by_confirm'], errors='coerce').dropna()
            row['confirmed_ge_7d_before_peak_rate'] = float((lead_peak >= 7).mean()) if len(lead_peak) else np.nan
        if 'lead_to_peak_by_first_alert' in sub:
            lead_peak = pd.to_numeric(sub['lead_to_peak_by_first_alert'], errors='coerce').dropna()
            row['first_alert_ge_7d_before_peak_rate'] = float((lead_peak >= 7).mean()) if len(lead_peak) else np.nan
        if 'T1_confirmed_source' in sub:
            src = sub['T1_confirmed_source'].dropna().astype(str)
            row['energy_assisted_confirm_rate'] = float((src == 'energy_assisted').mean()) if len(src) else np.nan
        if 'T1_first_alert_source' in sub:
            src = sub['T1_first_alert_source'].dropna().astype(str)
            row['energy_assisted_first_alert_rate'] = float((src == 'energy_assisted').mean()) if len(src) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _print_stage3_rolling_reference_summary(summary_df: pd.DataFrame):
    if summary_df is None or len(summary_df) == 0:
        return
    print("\n  Stage3 rolling-reference summary")
    for _, row in summary_df.iterrows():
        disease = row.get('disease', 'unknown')
        n = int(row.get('n', 0)) if pd.notna(row.get('n', np.nan)) else 0
        alert = row.get('retro_alert_days_median', np.nan)
        peak_window = row.get('retro_T1_to_peak_days_median', np.nan)
        fa_delay = row.get('T1_first_alert_lag_vs_retro_median', np.nan)
        fa_delay_gt = row.get('T1_first_alert_lag_GT_norm_median', np.nan)
        fa_lead_t2 = row.get('lead_to_T2_by_first_alert_median', np.nan)
        fa_lead_peak = row.get('lead_to_peak_by_first_alert_median', np.nan)
        fa_before_t2 = row.get('first_alert_before_T2_rate', np.nan)
        fa_before_peak = row.get('first_alert_before_peak_rate', np.nan)
        delay = row.get('T1_confirm_lag_vs_retro_median', np.nan)
        delay_gt = row.get('T1_confirm_lag_GT_norm_median', np.nan)
        t2_err = row.get(
            'T2_confirmed_estimate_signed_error_vs_retro_median',
            row.get('T2_signed_error_vs_retro_median', np.nan)
        )
        tp_err = row.get(
            'Tp_confirmed_estimate_signed_error_vs_retro_median',
            row.get('Tp_signed_error_vs_retro_median', np.nan)
        )
        t2_lag = row.get('T2_confirm_lag_vs_retro_median', np.nan)
        tp_lag = row.get('Tp_confirm_lag_vs_retro_median', np.nan)
        t2_before_peak = row.get('T2_confirm_before_retro_peak_rate', np.nan)
        t2_within2 = row.get('T2_confirm_within_2GT_rate', np.nan)
        tp_within2 = row.get('Tp_confirm_within_2GT_rate', np.nan)
        lead_t2 = row.get('lead_to_T2_by_confirm_median', np.nan)
        lead_peak = row.get('lead_to_peak_by_confirm_median', np.nan)
        before_t2 = row.get('confirmed_before_T2_rate', np.nan)
        before_peak = row.get('confirmed_before_peak_rate', np.nan)
        within2 = row.get('T1_confirm_within_2GT_rate', np.nan)
        print(
            f"    {disease} (N={n}): retrospective alert={alert:.1f}d, "
            f"T1-to-peak={peak_window:.1f}d"
        )
        print(
            f"      first alert: lag={fa_delay:.1f}d ({fa_delay_gt:.2f}GT), "
            f"lead_to_T2={fa_lead_t2:.1f}d, lead_to_peak={fa_lead_peak:.1f}d, "
            f"before_T2={fa_before_t2*100:.1f}%, before_peak={fa_before_peak*100:.1f}%"
        )
        print(
            f"      stable confirm: lag={delay:.1f}d ({delay_gt:.2f}GT), "
            f"lead_to_T2={lead_t2:.1f}d, lead_to_peak={lead_peak:.1f}d, "
            f"before_T2={before_t2*100:.1f}%, before_peak={before_peak*100:.1f}%, "
            f"within2GT={within2*100:.1f}%"
        )
        print(
            f"      secondary T2/Tp estimates: T2_signed={t2_err:.1f}d, T2_confirm_lag={t2_lag:.1f}d, "
            f"T2_before_peak={t2_before_peak*100:.1f}%, T2_within2GT={t2_within2*100:.1f}%, "
            f"Tp_signed={tp_err:.1f}d, Tp_confirm_lag={tp_lag:.1f}d, Tp_within2GT={tp_within2*100:.1f}%"
        )


def _build_method_comparison_table(df: pd.DataFrame, source_label: str = '',
                                   gt: Optional[float] = None) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    rows = []
    id_cols = [c for c in ['city', '城市', 'disease'] if c in df.columns]
    gt_default = gt if gt is not None else _infer_gt_from_source(source_label)

    def row_value(row: pd.Series, *names):
        for name in names:
            if name in row.index:
                value = row.get(name)
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    return value
        return np.nan

    def make_method_row(base: Dict, row: pd.Series, method: str,
                        t1_name: str, t2_name: str, tp_name: str,
                        confidence):
        t1 = row_value(row, t1_name)
        t2 = row_value(row, t2_name)
        tp = row_value(row, tp_name, 'Tp', 'T_peak')
        gt_val = row_value(row, 'GT')
        if pd.isna(gt_val):
            gt_val = gt_default
        alert_days = float(t2) - float(t1) if pd.notna(t1) and pd.notna(t2) else np.nan
        t1_to_peak = float(tp) - float(t1) if pd.notna(t1) and pd.notna(tp) else np.nan
        if pd.notna(t1) and pd.notna(t2) and pd.notna(tp) and pd.notna(gt_val):
            intervention = SignalDiagnostics.classify_intervention_window(
                int(t1), int(t2), int(tp), float(gt_val)
            )
        else:
            intervention = np.nan
        return dict(
            **base, method=method, T1=t1, T2=t2, Tp=tp, GT=gt_val,
            alert_days=alert_days,
            alert_days_GT_norm=alert_days / max(float(gt_val), 1e-10)
            if pd.notna(alert_days) and pd.notna(gt_val) else np.nan,
            T1_to_peak_days=t1_to_peak,
            confidence=confidence,
            short_alert_class=row.get('short_alert_class'),
            intervention_window=intervention,
        )

    for _, row in df.iterrows():
        base = {c: row.get(c) for c in id_cols}
        if source_label:
            base['source'] = source_label
        confidence = row_value(row, 'confidence', '整体置信')
        rows.append(make_method_row(
            base, row, VERSION_TAG,
            'T1', 'T2', 'Tp', confidence
        ))
        if pd.notna(row.get('T1_energy')) and pd.notna(row.get('T2_energy')):
            rows.append(make_method_row(
                base, row, 'V24_energy_threshold_baseline',
                'T1_energy', 'T2_energy', 'Tp_energy', 'baseline'
            ))
    return pd.DataFrame(rows)


def _augment_stage3_summary(df_summary: pd.DataFrame, dname: str, ep_d: EpiParams) -> pd.DataFrame:
    """Normalize Stage 3 summary columns used by exports and method comparisons."""
    if df_summary is None:
        return pd.DataFrame()
    work = df_summary.copy()
    work['disease'] = dname
    work['GT'] = ep_d.GT
    fill_map = {
        'confidence': ['整体置信'],
        'alert_days': ['警戒期天数', 'T1_to_T2_days'],
        'T1_lead_peak': ['T1_to_peak_days'],
        'Tp': ['T_peak'],
    }
    for target, candidates in fill_map.items():
        if target not in work.columns:
            work[target] = np.nan
        for col in candidates:
            if col in work.columns:
                work[target] = work[target].where(work[target].notna(), work[col])
    if 'SNR' in work.columns:
        snr = pd.to_numeric(work['SNR'], errors='coerce')
        work['SNR_capped'] = snr.clip(upper=1000)
        work['log1p_SNR'] = np.log1p(snr.clip(lower=0))
    return work


def _t1_runtime_overrides(ep_opt: EpiParams) -> Dict:
    keys = [
        't1_structure_score_thresh', 't1_mid_z_thresh', 't1_mid_high_ratio_z',
        't1_centroid_z_thresh', 't1_relaxed_score_thresh',
        't1_valid_strength_min', 't1_confirm_ewma_k',
        't1_fast_confirm_strength', 't1_single_confirm_strength',
        't1_fast_local_margin', 't1_confirm_span_gt',
        't1_confirm_allow_single_strong', 't1_confirm_single_min_persist_gt',
        't1_confirm_single_local_margin',
        't1_confirm_allow_same_window_as_first_alert',
        't1_early_confirm_strength', 't1_early_confirm_min_persist_gt',
        't1_early_confirm_local_margin',
        't1_first_alert_enable', 't1_first_alert_strength',
        't1_first_alert_local_margin', 't1_first_alert_min_persist_gt',
        't1_first_alert_allow_energy',
        't1_raw_growth_alert_enable', 't1_raw_growth_min_persist_gt',
        't1_raw_growth_confirm_strength', 't1_raw_growth_smooth_gt',
        't1_raw_growth_z', 't1_raw_growth_ratio', 't1_raw_growth_min_abs_frac',
        't1_raw_growth_backcast_gt',
        't1_estimate_backtrack_gt', 't1_estimate_backtrack_min_persist_gt',
        't1_estimate_max_backtrack_gt', 't1_estimate_history_quantile',
        't1_aux_min_strength', 't1_aux_max_lead_gt',
        't1_aux_stale_candidate_gt', 't1_aux_stale_backcast_gt',
        't1_struct_stale_candidate_gt', 't1_struct_stale_backcast_gt',
        'prospective_min_window_gt', 'prospective_min_window_floor',
        'prospective_t2_confirm_rounds', 'prospective_t2_confirm_tol_gt',
        'prospective_t2_min_after_t1_gt', 'prospective_t2_strength_min',
        'prospective_t2_dom_gap_min', 'prospective_t2_tail_quantile',
        'prospective_tp_confirm_rounds', 'prospective_tp_confirm_tol_gt',
        'prospective_tp_maturity_gt', 'prospective_tp_min_local_frac',
        'prospective_tp_decline_frac', 'prospective_tp_fallback_quantile',
        'stage3_rolling_window_frac', 'stage3_rolling_window_floor',
        'stage3_rolling_window_cap', 'stage3_rolling_confirm_rounds',
    ]
    return {k: getattr(ep_opt, k) for k in keys if hasattr(ep_opt, k)}


def _stage3_rolling_overrides(ep_opt: EpiParams) -> Dict:
    """Use a more operational rolling setup for real-data prospective checks."""
    return {
        'prospective_min_window_gt': min(float(getattr(ep_opt, 'prospective_min_window_gt', 3.0)), 2.5),
        'prospective_min_window_floor': min(int(getattr(ep_opt, 'prospective_min_window_floor', 12)), 10),
        'prospective_t2_confirm_rounds': 2,
        'prospective_t2_confirm_tol_gt': max(float(getattr(ep_opt, 'prospective_t2_confirm_tol_gt', 1.0)), 1.25),
        'prospective_t2_strength_min': min(float(getattr(ep_opt, 'prospective_t2_strength_min', 0.42)), 0.36),
        'prospective_t2_tail_quantile': min(float(getattr(ep_opt, 'prospective_t2_tail_quantile', 0.35)), 0.20),
        'prospective_tp_confirm_rounds': 2,
        'prospective_tp_confirm_tol_gt': max(float(getattr(ep_opt, 'prospective_tp_confirm_tol_gt', 1.5)), 2.0),
        'prospective_tp_maturity_gt': min(float(getattr(ep_opt, 'prospective_tp_maturity_gt', 2.0)), 1.25),
        'prospective_tp_decline_frac': min(float(getattr(ep_opt, 'prospective_tp_decline_frac', 0.10)), 0.05),
        't1_first_alert_strength': min(float(getattr(ep_opt, 't1_first_alert_strength', 0.24)), 0.16),
        't1_first_alert_local_margin': min(float(getattr(ep_opt, 't1_first_alert_local_margin', -0.08)), -0.12),
        't1_first_alert_min_persist_gt': min(float(getattr(ep_opt, 't1_first_alert_min_persist_gt', 0.20)), 0.0),
        't1_raw_growth_z': min(float(getattr(ep_opt, 't1_raw_growth_z', 2.2)), 1.9),
        't1_raw_growth_min_persist_gt': min(float(getattr(ep_opt, 't1_raw_growth_min_persist_gt', 0.6)), 0.45),
        't1_raw_growth_confirm_strength': min(float(getattr(ep_opt, 't1_raw_growth_confirm_strength', 0.62)), 0.58),
        't1_confirm_allow_same_window_as_first_alert': False,
    }


def _stage3_sim_rolling_window(ep_opt: EpiParams, series_len: int) -> int:
    """Short-GT simulation waves need materially shorter rolling windows than real data."""
    series_len = max(1, int(series_len))
    frac_cap = max(12, int(np.floor(series_len * float(getattr(ep_opt, 'sim_stage3_rolling_window_series_frac', 0.75)))))
    base = max(
        int(getattr(ep_opt, 'sim_stage3_rolling_window_floor', 18)),
        int(np.ceil(float(getattr(ep_opt, 'sim_stage3_rolling_window_gt', 6.0)) * float(ep_opt.GT)))
    )
    cap = min(
        int(getattr(ep_opt, 'sim_stage3_rolling_window_cap', 36)),
        frac_cap,
        series_len,
    )
    if cap < 12:
        return series_len
    return int(min(max(12, base), max(12, cap)))


def run_stage3_real_application(ep_opt: EpiParams, run_rolling_reference: bool = True,
                                rolling_eemd_trials: int = 8, rolling_step: int = 2):
    print(f"\n{'#'*72}\n  STAGE 3: real-world application and multi-disease comparison V31.1 adaptive\n  Full-series structural detection without onset-window truncation\n{'#'*72}")
    all_results = {}
    disease_tables = []
    method_tables = []
    stage3_rolling_rows = []

    for (path, sdate, preset, dname) in [
        (DATA_PATH_H1N1, DATA_START_H1N1, 'influenza_h1n1', 'H1N1 2009'),
        (DATA_PATH_COVID, DATA_START_COVID, 'omicron', 'COVID-19 Omicron'),
    ]:
        print(f"\n  {'-'*62}\n  [{dname}]\n  {'-'*62}")
        if not os.path.exists(path):
            print(f"  File not found: {path}; skipped")
            continue
        loader = RealDataLoader(path, sdate, dname)
        if loader.df is None or loader.n_days < 50 or len(loader.regions) < 10:
            print("  Data abnormal; skipped")
            continue
        ep_d = EpiParams(preset=preset)
        ep_d.apply_override({
            'energy_smooth_sigma': ep_opt.energy_smooth_sigma,
            'score_smooth_sigma': ep_opt.score_smooth_sigma,
            't1_structure_score_thresh': ep_opt.t1_structure_score_thresh,
            't1_mid_z_thresh': ep_opt.t1_mid_z_thresh,
            't1_mid_high_ratio_z': ep_opt.t1_mid_high_ratio_z,
            't1_centroid_z_thresh': ep_opt.t1_centroid_z_thresh,
            't1_relaxed_score_thresh': ep_opt.t1_relaxed_score_thresh,
            't1_valid_strength_min': ep_opt.t1_valid_strength_min,
            't1_confirm_ewma_k': ep_opt.t1_confirm_ewma_k,
            't1_fast_confirm_strength': ep_opt.t1_fast_confirm_strength,
            't1_single_confirm_strength': ep_opt.t1_single_confirm_strength,
            't1_fast_local_margin': ep_opt.t1_fast_local_margin,
            't1_confirm_span_gt': ep_opt.t1_confirm_span_gt,
            't2_dom_score_thresh': ep_opt.t2_dom_score_thresh,
            't2_low_share_thresh_short': ep_opt.t2_low_share_thresh_short,
            't2_low_share_thresh_medium': ep_opt.t2_low_share_thresh_medium,
            't2_low_share_thresh_long': ep_opt.t2_low_share_thresh_long,
        })
        ep_d.apply_override(_t1_runtime_overrides(ep_opt))
        detector = RealDataStructuralDetector(ep_d, eemd_trials=200, stability_runs=0)
        results, df_summary = detector.batch_segment_realdata(loader, verbose=True)
        df_summary = _augment_stage3_summary(df_summary, dname, ep_d)
        nat_result = detector.segment_realdata(loader.national, city_name=f'{dname} national', global_offset=0, do_ensemble=True)
        Visualizer.plot_realdata_segmentation(nat_result, loader, save_path=os.path.join(OUTPUT_STAGE3, f'{preset}_national_{VERSION_TAG}.png'), title=dname)
        df_summary.to_excel(os.path.join(OUTPUT_STAGE3, f'{preset}_phase_seg_{VERSION_TAG}.xlsx'), index=False)
        method_df = _build_method_comparison_table(df_summary, source_label=dname, gt=ep_d.GT)
        if len(method_df) > 0:
            method_df.to_csv(os.path.join(OUTPUT_STAGE3, f'{preset}_method_comparison_{VERSION_TAG}.csv'), index=False)
            method_tables.append(method_df)
        valid = [r for r in results if r['confidence'].get('overall') != 'none']
        for ci in range(min(3, len(valid))):
            r = valid[ci]
            fn = Visualizer._safe_filename(r['city'])
            Visualizer.plot_realdata_segmentation(r, loader, save_path=os.path.join(OUTPUT_STAGE3, f'{preset}_{fn}_{VERSION_TAG}.png'), title=f"{dname} - {r['city']}")
        disease_tables.append(df_summary.copy())
        all_results[dname] = dict(summary=df_summary, results=results, national=nat_result)
        if run_rolling_reference:
            print(f"  Stage3 rolling-reference validation: {dname}")
            ep_roll = copy.deepcopy(ep_d)
            ep_roll.apply_override(_stage3_rolling_overrides(ep_d))
            real_rolling_window = min(
                int(getattr(ep_roll, 'stage3_rolling_window_cap', 65)),
                max(
                    int(getattr(ep_roll, 'stage3_rolling_window_floor', 42)),
                    int(np.ceil(loader.n_days * float(getattr(ep_roll, 'stage3_rolling_window_frac', 0.45))))
                )
            )
            stage3_rounds = int(getattr(ep_roll, 'stage3_rolling_confirm_rounds', 1))
            print(f"    rolling window={real_rolling_window}d, step={rolling_step}d, rounds={stage3_rounds}")
            for ri, r in enumerate(results):
                try:
                    if r['confidence'].get('overall') == 'none':
                        continue
                    sig = loader.get_series(idx=ri)
                    stage3_rolling_rows.append(_stage3_rolling_reference_row(
                        sig, ep_roll, dname, r['city'], r['T1_global'], r['T2_global'], r['Tp_global'],
                        curve_id=ri, window_size=real_rolling_window, step=rolling_step,
                        eemd_trials=rolling_eemd_trials, confirm_rounds=stage3_rounds,
                        sig_valid_start=r.get('sig_valid_start'), bg_end=r.get('bg_end'),
                        T1_low=r.get('intervals', {}).get('T1', (np.nan, np.nan))[0],
                        T1_high=r.get('intervals', {}).get('T1', (np.nan, np.nan))[1]
                    ))
                except Exception as ex:
                    stage3_rolling_rows.append(dict(
                        disease=dname, city=r.get('city', str(ri)), curve_id=ri,
                        T1_retro=r.get('T1_global'), T2_retro=r.get('T2_global'),
                        T_peak_retro=r.get('Tp_global'), rolling_error=str(ex)
                    ))
                if (ri + 1) % 50 == 0:
                    print(f"    rolling [{ri+1}/{len(results)}]")

    sim = SIRSimulator(seed=2024)
    for (dkey, preset, dname, nc) in [
        ('dengue', 'dengue', 'Dengue(SIR)', 20),
        ('chikungunya', 'chikungunya', 'Chikungunya(SIR)', 20),
    ]:
        print(f"\n  {'-'*62}\n  [{dname}]\n  {'-'*62}")
        cits = sim.generate_disease_scenarios(disease=dkey, n_cities=nc, seed_offset=(100 if dkey == 'dengue' else 200))
        SIRSimulator.print_truth_distribution(cits, f'{dname}_simulation')
        SignalDiagnostics.diagnose_signal_complexity(cits, f'{dname}_simulation')
        ep_d = EpiParams(preset=preset)
        ep_d.apply_override({
            'energy_smooth_sigma': ep_opt.energy_smooth_sigma,
            'score_smooth_sigma': ep_opt.score_smooth_sigma,
            't1_structure_score_thresh': ep_opt.t1_structure_score_thresh,
            't1_mid_z_thresh': ep_opt.t1_mid_z_thresh,
            't1_mid_high_ratio_z': max(0.5, ep_opt.t1_mid_high_ratio_z),
            't1_centroid_z_thresh': ep_opt.t1_centroid_z_thresh,
            't1_relaxed_score_thresh': ep_opt.t1_relaxed_score_thresh,
            't1_valid_strength_min': ep_opt.t1_valid_strength_min,
            't1_confirm_ewma_k': ep_opt.t1_confirm_ewma_k,
            't1_fast_confirm_strength': ep_opt.t1_fast_confirm_strength,
            't1_single_confirm_strength': ep_opt.t1_single_confirm_strength,
            't1_fast_local_margin': ep_opt.t1_fast_local_margin,
            't1_confirm_span_gt': ep_opt.t1_confirm_span_gt,
            't2_dom_score_thresh': ep_opt.t2_dom_score_thresh,
            't2_low_share_thresh_short': ep_opt.t2_low_share_thresh_short,
            't2_low_share_thresh_medium': ep_opt.t2_low_share_thresh_medium,
            't2_low_share_thresh_long': ep_opt.t2_low_share_thresh_long,
        })
        ep_d.apply_override(_t1_runtime_overrides(ep_opt))
        ep_roll = copy.deepcopy(ep_d)
        ep_roll.apply_override(_stage3_rolling_overrides(ep_d))
        stage3_rounds = int(getattr(ep_roll, 'stage3_rolling_confirm_rounds', 1))
        sim_lengths = [len(np.asarray(curve['I'])) for curve in cits]
        sim_window_preview = _stage3_sim_rolling_window(ep_roll, int(np.median(sim_lengths))) if sim_lengths else np.nan
        sim_rolling_step = 1 if float(ep_roll.GT) <= 6.5 else rolling_step
        if run_rolling_reference:
            print(f"  Stage3 rolling-reference validation: {dname}")
            print(f"    rolling window~{sim_window_preview}d (adaptive), step={sim_rolling_step}d, rounds={stage3_rounds}")
        seg = StructuralPhaseSegmenter(ep_d, eemd_trials=200)
        rows = []
        for i, curve in enumerate(cits):
            sig = np.maximum(np.nan_to_num(curve['I'], nan=0), 0).astype(float)
            res = seg.segment(sig, curve['t'], city_name=curve['city_name'], prospective_mode=False, real_data=False)
            comparator = SignalDiagnostics.energy_threshold_baseline_from_result(res, ep_d)
            short_alert_class = SignalDiagnostics.classify_short_alert(sig, res['T1'], res['T2'], res['Tp'])
            intervention_window = SignalDiagnostics.classify_intervention_window(res['T1'], res['T2'], res['Tp'], curve['GT'])
            rows.append({
                'city': curve['city_name'], 'disease': dname,
                'T1': res['T1'], 'T2': res['T2'], 'Tp': res['Tp'], 'T3': res['T3'],
                'truth_T1_growth': curve['truth_T1_growth'], 'truth_T1_struct': curve['truth_T1_struct'],
                'truth_T2_dyn': curve['truth_T2_dyn'], 'truth_T2_dom': curve['truth_T2_dom'], 'truth_Tp': curve['truth_Tp'],
                'T1_growth_err': abs(res['T1'] - curve['truth_T1_growth']),
                'T1_struct_err': abs(res['T1'] - curve['truth_T1_struct']),
                'T1_band_err': band_distance(res['T1'], curve['truth_T1_growth'], curve['truth_T1_struct']),
                'T2_dyn_err': abs(res['T2'] - curve['truth_T2_dyn']),
                'T2_dom_err': abs(res['T2'] - curve['truth_T2_dom']),
                'T2_band_err': band_distance(res['T2'], curve['truth_T2_dyn'], curve['truth_T2_dom']),
                'Tp_err': abs(res['Tp'] - curve['truth_Tp']),
                'alert_days': res['T2'] - res['T1'],
                'alert_days_GT_norm': (res['T2'] - res['T1']) / max(curve['GT'], 1e-10),
                'T1_lead_peak': curve['truth_Tp'] - res['T1'],
                'short_alert_class': short_alert_class,
                'intervention_window': intervention_window,
                'method_structural': VERSION_TAG,
                'T1_energy': comparator.get('T1_energy'),
                'T2_energy': comparator.get('T2_energy'),
                'Tp_energy': comparator.get('Tp_energy'),
                'alert_days_energy': comparator.get('alert_days_energy'),
                'T1_energy_delta_vs_struct': comparator.get('T1_energy_delta_vs_struct'),
                'T2_energy_delta_vs_struct': comparator.get('T2_energy_delta_vs_struct'),
                'energy_baseline_ok': comparator.get('energy_baseline_ok'),
                'T1_energy_growth_err': abs(comparator.get('T1_energy', np.nan) - curve['truth_T1_growth']) if comparator.get('energy_baseline_ok') else np.nan,
                'T2_energy_band_err': band_distance(comparator.get('T2_energy'), curve['truth_T2_dyn'], curve['truth_T2_dom']) if comparator.get('energy_baseline_ok') else np.nan,
                'R0': curve['R0'], 'GT': curve['GT'], 'confidence': res['confidence']['overall'],
            })
            if run_rolling_reference:
                try:
                    sim_rolling_window = _stage3_sim_rolling_window(ep_roll, len(sig))
                    stage3_rolling_rows.append(_stage3_rolling_reference_row(
                        sig, ep_roll, dname, curve['city_name'], res['T1'], res['T2'], res['Tp'],
                        curve_id=curve.get('curve_id', i), window_size=sim_rolling_window, step=sim_rolling_step,
                        eemd_trials=rolling_eemd_trials, confirm_rounds=stage3_rounds
                    ))
                except Exception as ex:
                    stage3_rolling_rows.append(dict(
                        disease=dname, city=curve.get('city_name', str(i)), curve_id=curve.get('curve_id', i),
                        T1_retro=res.get('T1'), T2_retro=res.get('T2'), T_peak_retro=res.get('Tp'),
                        rolling_error=str(ex)
                    ))
            if (i + 1) % 5 == 0:
                print(f"  [{i+1}/{nc}] completed")
        err_df = pd.DataFrame(rows)
        err_df.to_csv(os.path.join(OUTPUT_STAGE3, f'{dkey}_sim_results_{VERSION_TAG}.csv'), index=False)
        method_df = _build_method_comparison_table(err_df, source_label=dname, gt=ep_d.GT)
        if len(method_df) > 0:
            method_df.to_csv(os.path.join(OUTPUT_STAGE3, f'{dkey}_method_comparison_{VERSION_TAG}.csv'), index=False)
            method_tables.append(method_df)
        print(f"\n  Accuracy for {dname}:")
        for m, nm in [('T1_growth_err', 'T1(growth)'), ('T2_band_err', 'T2(band)'), ('Tp_err', 'Tp')]:
            med, lo, hi = bootstrap_ci(err_df[m])
            print(f"    {nm}: median_MAE={med:.1f}d  CI=[{lo:.1f},{hi:.1f}]")
        print(f"    alert window: median={err_df['alert_days'].median():.0f}d  GT-normalized median={err_df['alert_days_GT_norm'].median():.2f}")
        print(f"    T1-to-peak: median={err_df['T1_lead_peak'].median():.0f}d  before_peak={(err_df['T1_lead_peak']>0).mean()*100:.0f}%")
        curve = cits[0]
        res = seg.segment(curve['I'], curve['t'], city_name=curve['city_name'])
        Visualizer.plot_global_segmentation(res, curve=curve, save_path=os.path.join(OUTPUT_STAGE3, f"{dkey}_{curve['city_name']}_{VERSION_TAG}.png"), title=f"{dname}: R0={curve['R0']:.2f}, GT={curve['GT']:.1f}d")
        disease_tables.append(err_df.copy())
        all_results[dname] = err_df

    if disease_tables:
        combined = pd.concat(disease_tables, axis=0, ignore_index=True, sort=False)
        combined.to_excel(os.path.join(OUTPUT_STAGE3, f'four_disease_comparison_{VERSION_TAG}.xlsx'), index=False)
    if method_tables:
        method_combined = pd.concat(method_tables, axis=0, ignore_index=True, sort=False)
        method_combined.to_csv(os.path.join(OUTPUT_STAGE3, f'method_comparison_long_{VERSION_TAG}.csv'), index=False)
        all_results['method_comparison_long'] = method_combined
    if stage3_rolling_rows:
        rolling_df = pd.DataFrame(stage3_rolling_rows)
        rolling_df.to_excel(os.path.join(OUTPUT_STAGE3, f'stage3_rolling_reference_validation_{VERSION_TAG}.xlsx'), index=False)
        rolling_summary = _summarize_stage3_rolling_reference(rolling_df)
        rolling_summary.to_csv(os.path.join(OUTPUT_STAGE3, f'stage3_rolling_reference_summary_{VERSION_TAG}.csv'), index=False)
        _print_stage3_rolling_reference_summary(rolling_summary)
        all_results['stage3_rolling_reference_validation'] = rolling_df
        all_results['stage3_rolling_reference_summary'] = rolling_summary
    return all_results


def _safe_read_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        if path.lower().endswith(('.xlsx', '.xls')):
            return pd.read_excel(path)
        return pd.read_csv(path)
    except Exception as exc:
        print(f"  [paper tables] read failed: {path} ({exc})")
        return pd.DataFrame()


def _fallback_output_path(path: str, tag: str = '') -> str:
    root, ext = os.path.splitext(path)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = f'_{tag}' if tag else ''
    return f'{root}{suffix}_{stamp}{ext}'


def _path_is_writable(path: str) -> bool:
    if not os.path.exists(path):
        parent = os.path.dirname(path) or '.'
        return os.access(parent, os.W_OK)
    try:
        with open(path, 'a+b'):
            return True
    except PermissionError:
        return False
    except OSError:
        return False


def _safe_table_path(path: str, tag: str = 'fallback') -> str:
    if _path_is_writable(path):
        return path
    alt = _fallback_output_path(path, tag=tag)
    print(f"  [paper tables] target is locked; writing fallback: {alt}")
    return alt


def _safe_output_dir(path: str, tag: str = 'fallback') -> str:
    target = path
    if os.path.exists(target):
        if not os.path.isdir(target) or not os.access(target, os.W_OK):
            target = _fallback_output_path(path, tag=tag)
    else:
        parent = os.path.dirname(target) or '.'
        if not os.access(parent, os.W_OK):
            target = _fallback_output_path(path, tag=tag)
    os.makedirs(target, exist_ok=True)
    return target


def _safe_to_csv(df: pd.DataFrame, path: str, index: bool = False) -> str:
    target = _safe_table_path(path, tag='unlocked')
    try:
        df.to_csv(target, index=index, encoding='utf-8-sig')
        return target
    except PermissionError:
        alt = _fallback_output_path(path, tag='unlocked')
        print(f"  [paper tables] CSV locked during write; writing fallback: {alt}")
        df.to_csv(alt, index=index, encoding='utf-8-sig')
        return alt


def _summarize_numeric_table(df: pd.DataFrame, label: str, metrics: List[str]) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        vals = pd.to_numeric(df[metric], errors='coerce').dropna()
        if len(vals) == 0:
            continue
        row = dict(
            dataset=label,
            metric=metric,
            n=int(len(vals)),
            median=float(vals.median()),
            mean=float(vals.mean()),
            q25=float(vals.quantile(0.25)),
            q75=float(vals.quantile(0.75)),
        )
        try:
            med, lo, hi = bootstrap_ci(vals.values)
            row.update(bootstrap_median=med, ci95_low=lo, ci95_high=hi)
        except Exception:
            pass
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_primary_t1_endpoint_block(df: pd.DataFrame, dataset: str, reference: str,
                                         colmap: Dict[str, str],
                                         group_col: Optional[str] = None) -> pd.DataFrame:
    """Manuscript-facing summary for the primary T1 early-warning endpoint."""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    groups = df.groupby(group_col, dropna=False) if group_col and group_col in df.columns else [('all', df)]
    rows = []
    for group_name, sub in groups:
        row = dict(dataset=dataset, group=str(group_name), reference=reference, n=int(len(sub)))
        numeric_items = {
            'first_alert_delay_median': colmap.get('first_delay'),
            'first_alert_delay_GT_norm_median': colmap.get('first_delay_gt'),
            'first_alert_T1_error_median': colmap.get('first_err'),
            'first_alert_T1_signed_error_median': colmap.get('first_signed'),
            'first_alert_lead_to_T2_median': colmap.get('first_lead_t2'),
            'first_alert_lead_to_peak_median': colmap.get('first_lead_peak'),
            'stable_confirm_delay_median': colmap.get('confirm_delay'),
            'stable_confirm_delay_GT_norm_median': colmap.get('confirm_delay_gt'),
            'stable_confirm_T1_error_median': colmap.get('confirm_err'),
            'stable_confirm_T1_signed_error_median': colmap.get('confirm_signed'),
            'stable_confirm_lead_to_T2_median': colmap.get('confirm_lead_t2'),
            'stable_confirm_lead_to_peak_median': colmap.get('confirm_lead_peak'),
            'reference_alert_window_median': colmap.get('reference_alert'),
            'reference_T1_to_peak_window_median': colmap.get('reference_peak'),
        }
        for out_col, src_col in numeric_items.items():
            if not src_col or src_col not in sub.columns:
                continue
            vals = pd.to_numeric(sub[src_col], errors='coerce').dropna()
            if len(vals):
                base = out_col.replace('_median', '')
                row[out_col] = float(vals.median())
                row[f'{base}_mean'] = float(vals.mean())
                row[f'{base}_q25'] = float(vals.quantile(0.25))
                row[f'{base}_q75'] = float(vals.quantile(0.75))
        rate_items = {
            'first_alert_within_2GT_rate': colmap.get('first_within_2gt'),
            'stable_confirm_within_2GT_rate': colmap.get('confirm_within_2gt'),
        }
        for out_col, src_col in rate_items.items():
            if src_col and src_col in sub.columns:
                vals = pd.to_numeric(sub[src_col], errors='coerce').dropna()
                if len(vals):
                    row[out_col] = float(vals.mean())
        for out_col, src_col in {
            'first_alert_before_T2_rate': colmap.get('first_lead_t2'),
            'first_alert_before_peak_rate': colmap.get('first_lead_peak'),
            'stable_confirm_before_T2_rate': colmap.get('confirm_lead_t2'),
            'stable_confirm_before_peak_rate': colmap.get('confirm_lead_peak'),
        }.items():
            if src_col and src_col in sub.columns:
                vals = pd.to_numeric(sub[src_col], errors='coerce').dropna()
                if len(vals):
                    row[out_col] = float((vals > 0).mean())
        for out_col, src_col in {
            'first_alert_ge_7d_before_peak_rate': colmap.get('first_lead_peak'),
            'stable_confirm_ge_7d_before_peak_rate': colmap.get('confirm_lead_peak'),
        }.items():
            if src_col and src_col in sub.columns:
                vals = pd.to_numeric(sub[src_col], errors='coerce').dropna()
                if len(vals):
                    row[out_col] = float((vals >= 7).mean())
        for out_col, src_col in {
            'energy_assisted_first_alert_rate': colmap.get('first_source'),
            'energy_assisted_stable_confirm_rate': colmap.get('confirm_source'),
        }.items():
            if src_col and src_col in sub.columns:
                vals = sub[src_col].dropna().astype(str)
                if len(vals):
                    row[out_col] = float((vals == 'energy_assisted').mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _build_primary_t1_endpoint_table(sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    stage2_colmap = {
        'first_delay': 'T1_first_alert_delay',
        'first_delay_gt': 'T1_first_alert_delay_GT_norm',
        'first_err': 'T1_first_alert_err',
        'first_signed': 'T1_first_alert_signed_error',
        'first_lead_t2': 'lead_before_T2_by_first_alert',
        'first_lead_peak': 'lead_by_first_alert',
        'first_within_2gt': 'T1_first_alert_within_2GT',
        'first_source': 'T1_first_alert_source',
        'confirm_delay': 'T1_delay',
        'confirm_delay_gt': 'T1_delay_GT_norm',
        'confirm_err': 'T1_err',
        'confirm_signed': 'T1_estimate_signed_error',
        'confirm_lead_t2': 'lead_before_T2',
        'confirm_lead_peak': 'lead_by_confirm',
        'confirm_within_2gt': 'T1_confirm_within_2GT',
        'confirm_source': 'T1_confirmed_source',
        'reference_alert': 'reference_alert_days',
        'reference_peak': 'reference_t1_to_peak_days',
    }
    for key, label in [
        ('stage2_rolling', 'SIR prospective validation'),
        ('stage2_t1_focused', 'T1-focused prospective validation'),
    ]:
        block = _summarize_primary_t1_endpoint_block(
            sources.get(key, pd.DataFrame()),
            dataset=label,
            reference='simulation truth',
            colmap=stage2_colmap,
        )
        if len(block):
            parts.append(block)

    stage3_colmap = {
        'first_delay': 'T1_first_alert_lag_vs_retro',
        'first_delay_gt': 'T1_first_alert_lag_GT_norm',
        'first_signed': 'T1_first_alert_estimate_signed_error_vs_retro',
        'first_lead_t2': 'lead_to_T2_by_first_alert',
        'first_lead_peak': 'lead_to_peak_by_first_alert',
        'first_source': 'T1_first_alert_source',
        'confirm_delay': 'T1_confirm_lag_vs_retro',
        'confirm_delay_gt': 'T1_confirm_lag_GT_norm',
        'confirm_err': 'T1_band_err_vs_retro',
        'confirm_signed': 'T1_estimate_signed_error_vs_retro',
        'confirm_lead_t2': 'lead_to_T2_by_confirm',
        'confirm_lead_peak': 'lead_to_peak_by_confirm',
        'confirm_within_2gt': 'T1_confirm_within_2GT',
        'confirm_source': 'T1_confirmed_source',
        'reference_alert': 'retro_alert_days',
        'reference_peak': 'retro_T1_to_peak_days',
    }
    block = _summarize_primary_t1_endpoint_block(
        sources.get('stage3_rolling_reference', pd.DataFrame()),
        dataset='Stage 3 rolling-reference validation',
        reference='full-series retrospective T1/T2/Tp',
        colmap=stage3_colmap,
        group_col='disease',
    )
    if len(block):
        parts.append(block)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def _build_reproducibility_design_table(sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for key, label, reference in [
        ('stage2_rolling', 'Stage 2 prospective validation', 'SIR truth'),
        ('stage2_t1_focused', 'Stage 2B T1-focused validation', 'SIR truth'),
    ]:
        df = sources.get(key, pd.DataFrame())
        if df is None or len(df) == 0:
            continue
        row = dict(stage=label, reference=reference, n=int(len(df)))
        for out_col, src_col in [
            ('GT_median', 'GT'),
            ('window_size_configured_median', 'window_size_configured'),
            ('window_size_actual_median', 'window_size_actual_median'),
            ('window_size_actual_min_median', 'window_size_actual_min'),
            ('window_size_actual_max_median', 'window_size_actual_max'),
            ('min_start_used_median', 'min_start_used'),
        ]:
            if src_col in df.columns:
                vals = pd.to_numeric(df[src_col], errors='coerce').dropna()
                if len(vals):
                    row[out_col] = float(vals.median())
        if 'structural_failure' in df.columns:
            row['structural_failure_rate'] = _rate_from_boolish(df, 'structural_failure')
        rows.append(row)

    df = sources.get('stage3_rolling_reference', pd.DataFrame())
    if df is not None and len(df):
        for disease, sub in df.groupby('disease', dropna=False):
            row = dict(
                stage='Stage 3 rolling-reference validation',
                disease=str(disease),
                reference='full-series retrospective segmentation',
                n=int(len(sub)),
            )
            for out_col, src_col in [
                ('GT_median', 'GT'),
                ('window_size_used_median', 'window_size_used'),
                ('min_start_used_median', 'min_start_used'),
                ('sig_valid_start_median', 'sig_valid_start'),
                ('bg_end_median', 'bg_end'),
                ('candidate_min_day_median', 'candidate_min_day'),
                ('retro_T1_median', 'T1_retro'),
                ('retro_T2_median', 'T2_retro'),
                ('retro_peak_median', 'T_peak_retro'),
            ]:
                if src_col in sub.columns:
                    vals = pd.to_numeric(sub[src_col], errors='coerce').dropna()
                    if len(vals):
                        row[out_col] = float(vals.median())
            if 'structural_failure' in sub.columns:
                row['structural_failure_rate'] = _rate_from_boolish(sub, 'structural_failure')
            rows.append(row)
    return pd.DataFrame(rows)


def _summarize_method_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0 or 'method' not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    if 'GT' not in work.columns:
        work['GT'] = np.nan
    if 'source' in work.columns:
        inferred = work['source'].map(_infer_gt_from_source)
        work['GT'] = pd.to_numeric(work['GT'], errors='coerce').fillna(inferred)
    if 'alert_days' not in work.columns and {'T1', 'T2'}.issubset(work.columns):
        work['alert_days'] = pd.to_numeric(work['T2'], errors='coerce') - pd.to_numeric(work['T1'], errors='coerce')
    if 'alert_days_GT_norm' not in work.columns and {'alert_days', 'GT'}.issubset(work.columns):
        work['alert_days_GT_norm'] = pd.to_numeric(work['alert_days'], errors='coerce') / pd.to_numeric(work['GT'], errors='coerce').replace(0, np.nan)
    if 'T1_to_peak_days' not in work.columns and {'T1', 'Tp'}.issubset(work.columns):
        work['T1_to_peak_days'] = pd.to_numeric(work['Tp'], errors='coerce') - pd.to_numeric(work['T1'], errors='coerce')
    if {'T1', 'T2', 'Tp', 'GT'}.issubset(work.columns):
        def _intervention_row(row):
            vals = [row.get('T1'), row.get('T2'), row.get('Tp'), row.get('GT')]
            if any(pd.isna(x) for x in vals):
                return np.nan
            return SignalDiagnostics.classify_intervention_window(
                int(row['T1']), int(row['T2']), int(row['Tp']), float(row['GT'])
            )
        work['intervention_window_recomputed'] = work.apply(_intervention_row, axis=1)
    group_cols = ['source', 'method'] if 'source' in df.columns else ['method']
    metric_cols = [
        'T1_growth_err', 'T1_band_err', 'T2_band_err', 'Tp_err',
        'T1', 'T2', 'Tp', 'GT', 'alert_days', 'alert_days_GT_norm',
        'T1_to_peak_days', 'T1_lead_peak',
    ]
    rows = []
    for keys, sub in work.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys), n=int(len(sub)))
        for col in metric_cols:
            if col in sub.columns:
                vals = pd.to_numeric(sub[col], errors='coerce').dropna()
                if len(vals):
                    row[f'{col}_median'] = float(vals.median())
                    row[f'{col}_mean'] = float(vals.mean())
        if 'intervention_window_recomputed' in sub.columns:
            cls = sub['intervention_window_recomputed'].dropna().astype(str)
            if len(cls):
                row['actionable_window_rate'] = float((cls == 'actionable_window').mean())
                row['moderate_or_actionable_rate'] = float(cls.isin(['moderate_window', 'actionable_window']).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _harmonize_stage3_disease_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    work = df.copy()
    fill_map = {
        'city': ['城市', 'region'],
        'Tp': ['T_peak'],
        'confidence': ['整体置信'],
        'alert_days': ['警戒期天数', 'T1_to_T2_days'],
        'alert_days_GT_norm': ['警戒期_GT标准化'],
        'T1_lead_peak': ['T1_to_peak_days'],
    }
    for target, candidates in fill_map.items():
        if target not in work.columns:
            work[target] = np.nan
        for src in candidates:
            if src in work.columns:
                work[target] = work[target].where(work[target].notna(), work[src])
    if 'GT' not in work.columns:
        work['GT'] = np.nan
    if 'disease' in work.columns:
        inferred = work['disease'].map(_infer_gt_from_source)
        work['GT'] = pd.to_numeric(work['GT'], errors='coerce').fillna(inferred)
    if 'alert_days' in work.columns and 'GT' in work.columns:
        gt = pd.to_numeric(work['GT'], errors='coerce').replace(0, np.nan)
        work['alert_days_GT_norm'] = pd.to_numeric(work['alert_days_GT_norm'], errors='coerce').where(
            pd.to_numeric(work['alert_days_GT_norm'], errors='coerce').notna(),
            pd.to_numeric(work['alert_days'], errors='coerce') / gt
        )
    if 'SNR' in work.columns:
        snr = pd.to_numeric(work['SNR'], errors='coerce')
        work['SNR_capped'] = snr.clip(upper=1000)
        work['log1p_SNR'] = np.log1p(snr.clip(lower=0))
    return work


def export_paper_ready_tables(output_path: str = PAPER_TABLES_PATH,
                              stage3_results: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
    """Collect the run outputs into compact, manuscript-oriented tables."""
    table_dir = _safe_output_dir(os.path.join(OUTPUT_STAGE3, f'paper_tables_{VERSION_TAG}'), tag='unlocked')

    sources = {
        'stage0_benchmark': _safe_read_table(os.path.join(OUTPUT_BENCH, 'benchmark_v31_1.csv')),
        'stage1_dev_eval': _safe_read_table(os.path.join(OUTPUT_STAGE1, 'dev_eval_v31_1.csv')),
        'stage1_global_val': _safe_read_table(os.path.join(OUTPUT_STAGE1, 'global_val_eval_v31_1.csv')),
        'stage2_rolling': _safe_read_table(os.path.join(OUTPUT_STAGE2, 'rolling_validation_v31_1.csv')),
        'stage2_t1_focused': _safe_read_table(os.path.join(OUTPUT_STAGE2, 't1_focused_validation_v31_1.csv')),
        'stage2_t1_summary': _safe_read_table(os.path.join(OUTPUT_STAGE2, 't1_focused_validation_summary_v31_1.csv')),
        'stage2_t1_sensitivity': _safe_read_table(os.path.join(OUTPUT_STAGE2, 't1_focused_sensitivity_v31_1.csv')),
        'stage3_disease_comparison': _safe_read_table(os.path.join(OUTPUT_STAGE3, f'four_disease_comparison_{VERSION_TAG}.xlsx')),
        'stage3_method_long': _safe_read_table(os.path.join(OUTPUT_STAGE3, f'method_comparison_long_{VERSION_TAG}.csv')),
        'stage3_rolling_reference': _safe_read_table(os.path.join(OUTPUT_STAGE3, f'stage3_rolling_reference_validation_{VERSION_TAG}.xlsx')),
        'stage3_rolling_summary': _safe_read_table(os.path.join(OUTPUT_STAGE3, f'stage3_rolling_reference_summary_{VERSION_TAG}.csv')),
    }
    if stage3_results:
        rolling_summary = stage3_results.get('stage3_rolling_reference_summary')
        if isinstance(rolling_summary, pd.DataFrame) and len(rolling_summary):
            sources['stage3_rolling_summary'] = rolling_summary.copy()
        method_long = stage3_results.get('method_comparison_long')
        if isinstance(method_long, pd.DataFrame) and len(method_long):
            sources['stage3_method_long'] = method_long.copy()

    tables = {}
    bench = sources['stage0_benchmark']
    if len(bench):
        tables['Table0_runtime_benchmark'] = bench.copy()

    global_parts = []
    for label, df in [
        ('development', sources['stage1_dev_eval']),
        ('global_validation', sources['stage1_global_val']),
    ]:
        global_parts.append(_summarize_numeric_table(
            df, label,
            ['T1_growth_err', 'T1_struct_err', 'T1_band_err', 'T2_band_err', 'Tp_err', 'T1_to_peak', 'alert_days'],
        ))
    global_summary = pd.concat([x for x in global_parts if len(x)], ignore_index=True) if any(len(x) for x in global_parts) else pd.DataFrame()
    if len(global_summary):
        tables['Table1_global_accuracy'] = global_summary

    prospective_parts = []
    for label, df in [
        ('rolling_validation', sources['stage2_rolling']),
        ('t1_focused_validation', sources['stage2_t1_focused']),
    ]:
        prospective_parts.append(_summarize_numeric_table(
            df, label,
            ['T1_err', 'T1_band_err', 'T1_delay', 'T1_delay_GT_norm',
             'T1_estimate_signed_error',
             'T1_first_alert_err', 'T1_first_alert_delay',
             'T1_first_alert_delay_GT_norm', 'T1_first_alert_signed_error',
             'lead_by_first_alert', 'lead_before_T2_by_first_alert',
             'reference_alert_days', 'reference_t1_to_peak_days',
             'lead_by_confirm', 'lead_before_T2', 'T2_band_err', 'Tp_err'],
        ))
    prospective_summary = pd.concat([x for x in prospective_parts if len(x)], ignore_index=True) if any(len(x) for x in prospective_parts) else pd.DataFrame()
    if len(prospective_summary):
        tables['Table2_prospective_accuracy'] = prospective_summary

    primary_t1 = _build_primary_t1_endpoint_table(sources)
    if len(primary_t1):
        tables['Table2_primary_T1_endpoint'] = primary_t1

    design_table = _build_reproducibility_design_table(sources)
    if len(design_table):
        tables['TableS1_reproducibility_design'] = design_table

    for name in ['stage2_t1_summary', 'stage2_t1_sensitivity', 'stage3_rolling_summary']:
        if len(sources[name]):
            tables[name] = sources[name]

    if len(sources['stage3_rolling_reference']):
        rolling_summary = _summarize_stage3_rolling_reference(sources['stage3_rolling_reference'])
        if len(rolling_summary):
            tables['stage3_rolling_summary'] = rolling_summary

    disease = _harmonize_stage3_disease_table(sources['stage3_disease_comparison'])
    if len(disease):
        disease = disease.copy()
        if 'confidence' not in disease.columns:
            for col in ['整体置信']:
                if col in disease.columns:
                    disease['confidence'] = disease[col]
                    break
        if 'alert_days' not in disease.columns:
            for col in ['警戒期天数', 'T1_to_T2_days']:
                if col in disease.columns:
                    disease['alert_days'] = disease[col]
                    break
        if 'T1_lead_peak' not in disease.columns:
            for col in ['T1_to_peak_days']:
                if col in disease.columns:
                    disease['T1_lead_peak'] = disease[col]
                    break
        if 'SNR' in disease.columns and 'SNR_capped' not in disease.columns:
            snr = pd.to_numeric(disease['SNR'], errors='coerce')
            disease['SNR_capped'] = snr.clip(upper=1000)
            disease['log1p_SNR'] = np.log1p(snr.clip(lower=0))
        group_col = 'disease' if 'disease' in disease.columns else None
        disease_rows = []
        groups = disease.groupby(group_col, dropna=False) if group_col else [('all', disease)]
        for disease_name, sub in groups:
            row = dict(disease=disease_name, n=int(len(sub)))
            for col in ['T1_growth_err', 'T1_band_err', 'T2_band_err', 'Tp_err',
                        'alert_days', 'alert_days_GT_norm', 'T1_lead_peak',
                        'SNR_capped', 'log1p_SNR']:
                if col in sub.columns:
                    vals = pd.to_numeric(sub[col], errors='coerce').dropna()
                    if len(vals):
                        row[f'{col}_median'] = float(vals.median())
                        row[f'{col}_mean'] = float(vals.mean())
            if 'confidence' in sub.columns:
                conf = sub['confidence'].dropna().astype(str)
                if len(conf):
                    row['confidence_high_medium_rate'] = float(conf.isin(['high', 'medium']).mean())
            disease_rows.append(row)
        tables['Table3_disease_application'] = pd.DataFrame(disease_rows)

    method_summary = _summarize_method_table(sources['stage3_method_long'])
    if len(method_summary):
        tables['Table4_method_comparison'] = method_summary

    tables['Table_notes_endpoint_roles'] = pd.DataFrame([
        dict(endpoint='T1_first_alert',
             role='primary operational early-warning endpoint',
             interpretation='first time a city-level warning could be raised from sequential data'),
        dict(endpoint='T1_confirmed',
             role='primary stable confirmation endpoint',
             interpretation='stable prospective confirmation of transition from inter-epidemic/background phase to alert phase'),
        dict(endpoint='T2',
             role='secondary acceleration endpoint',
             interpretation='onset of accelerated outbreak expansion; prospective confirmation may require additional post-T1 observations'),
        dict(endpoint='Tp',
             role='secondary maturity endpoint',
             interpretation='peak timing is a maturity/retrospective endpoint and should not be used as the main early-warning claim'),
    ])

    if not tables:
        tables['README'] = pd.DataFrame([dict(
            message='No stage output tables were found. Run main() or the individual stages before exporting paper tables.',
            output_dir=OUTPUT_STAGE3,
        )])

    excel_output_path = _safe_table_path(output_path, tag='unlocked')
    try:
        with pd.ExcelWriter(excel_output_path) as writer:
            for sheet, df in tables.items():
                clean_sheet = sheet[:31]
                df.to_excel(writer, sheet_name=clean_sheet, index=False)
    except PermissionError:
        excel_output_path = _fallback_output_path(output_path, tag='unlocked')
        print(f"  [paper tables] Excel workbook locked during write; writing fallback: {excel_output_path}")
        with pd.ExcelWriter(excel_output_path) as writer:
            for sheet, df in tables.items():
                clean_sheet = sheet[:31]
                df.to_excel(writer, sheet_name=clean_sheet, index=False)

    csv_outputs = {}
    for sheet, df in tables.items():
        csv_outputs[sheet] = _safe_to_csv(df, os.path.join(table_dir, f'{sheet}.csv'), index=False)

    manifest = pd.DataFrame([
        dict(sheet=k, rows=len(v), columns=len(v.columns),
             excel_output=excel_output_path, csv_output=csv_outputs.get(k, ''))
        for k, v in tables.items()
    ])
    manifest_path = _safe_to_csv(manifest, os.path.join(table_dir, 'manifest.csv'), index=False)
    print(f"  Paper-ready tables exported: {excel_output_path}")
    print(f"  Paper-ready CSV manifest: {manifest_path}")
    return tables


# ============================================================
#  Main program
# ============================================================

def main():
    print(f"\n{'#'*72}")
    print(f'  Epidemic phase segmentation {VERSION_TAG} Fusion')
    print('  V31.1 archetype-adaptive structural chain with rolling confirmation and reporting outputs')
    print('  T2 uses a dual-reference system: T2_dyn + T2_dom')
    print('  T1 keeps growth-truth and structural-reference sensitivity analyses')
    print('  Stage 3 uses full-series structural detection and exports paper-ready tables')
    print(f"{'#'*72}")
    t0 = time.time()
    args = {a.lower() for a in sys.argv[1:]}
    ep_base = EpiParams(preset='omicron')

    if any(a in args for a in ['--quick-influenza', '--influenza-quick', 'quick_influenza']):
        ep_quick = load_existing_v31_1_ep_for_partial_run()
        ep_quick.print_summary()
        run_stage2_influenza_quick_validation(ep_quick)
        print(f"\n{'#'*72}")
        print(f"  Quick influenza_like validation completed. Total time={(time.time()-t0)/60:.1f} min")
        print(f"  Output dir: {OUTPUT_STAGE2}/")
        print(f"{'#'*72}")
        return

    if any(a in args for a in ['--core-validation', '--core-archetype', 'core_validation']):
        ep_core = load_existing_v31_1_ep_for_partial_run()
        ep_core.print_summary()
        run_stage2_core_archetype_validation(
            ep_core, n_per_archetype=80, output_label='core_archetype'
        )
        print(f"\n{'#'*72}")
        print(f"  Core archetype validation completed. Total time={(time.time()-t0)/60:.1f} min")
        print(f"  Output dir: {OUTPUT_STAGE2}/")
        print(f"{'#'*72}")
        return

    if any(a in args for a in ['--core-validation-small', '--core-small', 'core_validation_small']):
        ep_core = load_existing_v31_1_ep_for_partial_run()
        ep_core.print_summary()
        run_stage2_core_archetype_validation(
            ep_core, n_per_archetype=12, eemd_trials_default=12,
            output_label='core_archetype_small'
        )
        print(f"\n{'#'*72}")
        print(f"  Small core archetype validation completed. Total time={(time.time()-t0)/60:.1f} min")
        print(f"  Output dir: {OUTPUT_STAGE2}/")
        print(f"{'#'*72}")
        return

    ep_base.print_summary()

    run_stage0_benchmark(ep_base, n_test=30)
    ep_opt, dev_curves, tune_curves, cache, grid_df = run_stage1_development(ep_base, n_dev=100, n_stage2_tune=18, force_cache=False, force_grid=True)
    ep_t1, t1_params, t1_val_df = run_t1_focused_optimization(ep_opt, n_tune=12, n_val=40, fast_mode=True)
    stage2_window = int(t1_params.get('window_size', 100)) if t1_params else 100
    stage2_step = int(t1_params.get('step', 2)) if t1_params else 2
    stage2_rounds = int(t1_params.get('confirm_rounds', 2)) if t1_params else 2
    run_stage2_validation(ep_t1, n_val=80, window_size=stage2_window, step=stage2_step,
                          confirm_rounds=stage2_rounds)
    stage3_results = run_stage3_real_application(ep_t1)
    export_paper_ready_tables(stage3_results=stage3_results)

    print(f"\n{'#'*72}")
    print(f"  {VERSION_TAG} workflow completed. Total time={(time.time()-t0)/60:.1f} min")
    print(f"  Output dirs: {OUTPUT_BENCH}/  {OUTPUT_STAGE1}/  {OUTPUT_STAGE2}/  {OUTPUT_STAGE3}/")
    print(f"  Paper tables: {PAPER_TABLES_PATH}")
    print(f"{'#'*72}")


if __name__ == '__main__':
    main()
