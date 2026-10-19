#!/usr/bin/env python3
"""
Four-campaign overview: indoor/outdoor × uniform/non-uniform layouts.

Campaigns
  1) indoor_coral      structure 2  — indoor, non-uniform (large baselines)
  2) indoor_coral_2    structure 3  — indoor, uniform
  3) outdoor_coral     structure 3  — outdoor, uniform
  4) outdoor_coral     structure 2  — outdoor, non-uniform (large baselines)

Pipeline (from coral_indoor_structure3_overview.py):
  equal-weight pair fusion is used for *uniform* layouts;
  dens-gated robust replacement vs per-round cloud;
  consecutive 15-round window localizations for map scatter.

Non-uniform (structure 2) extras (aligned with structure2 / cip):
  keep baseline^2 fusion weights (do not equalize);
  rely on multi-baseline overlap disambiguation in
  coral_indoor_result_process (ENABLE_OVERLAP_DISAMBIG) to resolve
  spatial aliasing when d > λ/2.

Figures (interactive, no export):
  - one 2×2 map grid (SEGMENT_N mean-of-SEGMENT_LEN localizations; xy ∈ [-1.5, 2])
  - one AoA residual figure: 2×2 pairs, each pair a tight 1×2 (X|Y)
  - one combined CDF (CDF_BOOT_N_OBS bootstrap errors) with mean/p90 markers;
    2×2 panel order: outdoor/indoor × uniform/non-uniform

Usage:
  python coral_four_campaigns_overview.py
"""

from __future__ import annotations

import glob
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D
except ImportError:
    print("Missing dependency: matplotlib  (pip install matplotlib)", file=sys.stderr)
    sys.exit(1)

import coral_indoor_result_process as cip

ROOT = os.path.dirname(os.path.abspath(__file__))

SEGMENT_LEN = 15
SEGMENT_N = 10
SEGMENT_SEED = 20010127          #42 is ok

# Map scatter: SEGMENT_N mean-of-SEGMENT_LEN draws (without replacement) per target.
# CDF: more draws for a smoother curve (same recipe).
CDF_BOOT_N_OBS = 20  # constructed samples per target (CDF only)
CDF_BOOT_SEED = 7
MAP_XY_LIM = (-1.5, 2.0)  # fixed axis range for the map figure

# AoA averaging:
#   "circ"   = atan2(mean(sin), mean(cos))  [preferred]
#   "sin"    = asin(mean(sin))
#   "arith"  = plain degree mean
#   "median" = plain degree median
AOA_MEAN_MODE = "circ"

# If False: aggregate window/session AoAs with plain mean (no MAD trim).
AOA_USE_MAD = False


# Fixed 2×2 panel order (row-major):
#   [outdoor uniform | indoor uniform]
#   [outdoor non-uniform | indoor non-uniform]
LAYOUT_KEYS: Tuple[str, ...] = (
    "outdoor_uniform",
    "indoor_uniform",
    "outdoor_nonuniform",
    "indoor_nonuniform",
)

_C = {
    "blue": "#4477AA",
    "cyan": "#66CCEE",
    "green": "#228833",
    "yellow": "#CCBB44",
    "red": "#EE6677",
    "purple": "#AA3377",
    "gray": "#BBBBBB",
    "ink": "#222222",
}

# Distinct colors for the four campaigns on the shared CDF.
CAMPAIGN_COLORS = {
    "indoor_nonuniform": _C["red"],
    "indoor_uniform": _C["blue"],
    "outdoor_uniform": _C["green"],
    "outdoor_nonuniform": _C["purple"],
}


@dataclass(frozen=True)
class Campaign:
    key: str
    title: str
    data_dir: str
    structure_id: int
    uniform: bool  # False → non-uniform / large-baseline path


CAMPAIGNS: Tuple[Campaign, ...] = (
    Campaign(
        key="indoor_nonuniform",
        title="Indoor non-uniform",
        data_dir=os.path.join(ROOT, "indoor_coral"),
        structure_id=2,
        uniform=False,
    ),
    Campaign(
        key="indoor_uniform",
        title="Indoor uniform",
        data_dir=os.path.join(ROOT, "indoor_coral_2"),
        structure_id=3,
        uniform=True,
    ),
    Campaign(
        key="outdoor_uniform",
        title="Outdoor uniform",
        data_dir=os.path.join(ROOT, "outdoor_coral"),
        structure_id=3,
        uniform=True,
    ),
    Campaign(
        key="outdoor_nonuniform",
        title="Outdoor non-uniform",
        data_dir=os.path.join(ROOT, "outdoor_coral"),
        structure_id=2,
        uniform=False,
    ),
)


def _campaigns_layout_order(
    campaigns: Sequence[Campaign] | None = None,
) -> List[Campaign]:
    """Return campaigns in the fixed 2×2 display order."""
    by_key = {c.key: c for c in (campaigns if campaigns is not None else CAMPAIGNS)}
    return [by_key[k] for k in LAYOUT_KEYS if k in by_key]


# ---------------------------------------------------------------------------
# Session discovery / helpers (structure3-style)
# ---------------------------------------------------------------------------


def find_structure_sessions(data_dir: str, structure_id: int) -> List[str]:
    pattern = os.path.join(data_dir, "*_reports.npz")
    out: List[str] = []
    for npz_path in sorted(glob.glob(pattern)):
        try:
            meta = cip.load_meta(npz_path)
        except FileNotFoundError:
            print(f"  skip (no meta): {os.path.basename(npz_path)}", file=sys.stderr)
            continue
        if int(meta.get("anchor_structure_id", -1)) != structure_id:
            continue
        out.append(npz_path)
    return out


def _equalize_pair_weights(pair_series: List[Dict[str, Any]]) -> None:
    for p in pair_series:
        p["fusion_weight"] = 1.0 if math.isfinite(float(p["aoa_mean_deg"])) else 0.0


def _round_positions(
    session: Dict[str, Any],
    per_round: List[List[Dict[str, Any]]],
    pair_series: List[Dict[str, Any]],
) -> List[Tuple[float, float]]:
    pair_weights = {
        (p["group"], int(p["id_a"]), int(p["id_b"])): float(p["fusion_weight"])
        for p in pair_series
    }
    out: List[Tuple[float, float]] = []
    for round_pairs in per_round:
        pos = cip.localize_from_pair_rays(
            cip.rays_from_round_pairs(round_pairs, pair_weights)
        )
        if pos is None:
            th_x = cip.fuse_aoa_one_round(round_pairs, pair_weights, "X")
            th_y = cip.fuse_aoa_one_round(round_pairs, pair_weights, "Y")
            pos = cip.position_from_two_aoas(
                th_x,
                th_y,
                session["x_ids"],
                session["y_ids"],
                session["node_pos"],
                session["target_pos"],
            )
        if pos is None:
            continue
        if not (math.isfinite(pos[0]) and math.isfinite(pos[1])):
            continue
        if abs(pos[0]) > 20.0 or abs(pos[1]) > 20.0:
            continue
        out.append((float(pos[0]), float(pos[1])))
    return out


def _classify_round_outliers(
    positions: List[Tuple[float, float]],
    *,
    hard_radius_m: float = 2.5,
    mad_k: float = 2.5,
    min_keep: int = 8,
) -> Tuple[np.ndarray, Optional[Tuple[float, float]], Tuple[float, float]]:
    n = len(positions)
    if n == 0:
        return np.zeros(0, dtype=bool), None, (float("nan"), float("nan"))
    px = np.asarray([p[0] for p in positions], dtype=np.float64)
    py = np.asarray([p[1] for p in positions], dtype=np.float64)
    med = (float(np.median(px)), float(np.median(py)))
    if n < min_keep:
        return np.ones(n, dtype=bool), None, med

    dist0 = np.hypot(px - med[0], py - med[1])
    hard = dist0 <= hard_radius_m
    if int(np.count_nonzero(hard)) < min_keep:
        hard = np.ones(n, dtype=bool)

    px_h, py_h = px[hard], py[hard]
    med_h = (float(np.median(px_h)), float(np.median(py_h)))
    dist_h = np.hypot(px_h - med_h[0], py_h - med_h[1])
    scale = 1.4826 * float(np.median(dist_h)) + 1e-9
    keep_h = dist_h <= mad_k * scale
    n_keep = int(np.count_nonzero(keep_h))
    inlier = np.zeros(n, dtype=bool)
    hard_idx = np.flatnonzero(hard)
    if n_keep < min_keep:
        order = np.argsort(dist_h)
        take = min(max(min_keep, int(0.6 * len(dist_h))), len(dist_h))
        kept_local = order[:take]
        inlier[hard_idx[kept_local]] = True
        center = (float(np.mean(px_h[kept_local])), float(np.mean(py_h[kept_local])))
    else:
        inlier[hard_idx[keep_h]] = True
        center = (float(np.mean(px_h[keep_h])), float(np.mean(py_h[keep_h])))
    return inlier, center, med_h


def _robust_round_center(
    positions: List[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
    _mask, center, _med = _classify_round_outliers(positions)
    return center


def _density(positions: List[Tuple[float, float]], center: Tuple[float, float], radius_m: float) -> float:
    if not positions:
        return 0.0
    px = np.asarray([p[0] for p in positions], dtype=np.float64)
    py = np.asarray([p[1] for p in positions], dtype=np.float64)
    return float(np.mean(np.hypot(px - center[0], py - center[1]) < radius_m))


def _maybe_replace_with_robust(
    base_pos: Optional[Tuple[float, float]],
    positions: List[Tuple[float, float]],
    *,
    dens_radius_m: float = 0.40,
    dens_ratio_max: float = 0.45,
    dens_rob_min: float = 0.15,
) -> Tuple[Optional[Tuple[float, float]], str]:
    if base_pos is None:
        return None, "none"
    rob = _robust_round_center(positions)
    if rob is None:
        return base_pos, "base"
    dens_b = _density(positions, base_pos, dens_radius_m)
    dens_r = _density(positions, rob, dens_radius_m)
    ratio = dens_b / dens_r if dens_r > 1e-12 else float("inf")
    if dens_r >= dens_rob_min and ratio < dens_ratio_max:
        return rob, f"robust_rounds(dens_r={dens_r:.2f},dens_b={dens_b:.2f},ratio={ratio:.2f})"
    return base_pos, f"base(dens_r={dens_r:.2f},dens_b={dens_b:.2f},ratio={ratio:.2f})"


def _ang_diff_deg(a: float, b: float) -> float:
    return float((a - b + 180.0) % 360.0 - 180.0)


def _sin_mean_aoa_deg(aoas_deg: Sequence[float]) -> float:
    vals = [float(a) for a in aoas_deg if math.isfinite(float(a))]
    if not vals:
        return float("nan")
    s = float(np.mean([math.sin(math.radians(a)) for a in vals]))
    s = max(-1.0, min(1.0, s))
    return float(math.degrees(math.asin(s)))


def _arith_mean_aoa_deg(aoas_deg: Sequence[float]) -> float:
    """Direct arithmetic mean of AoA degrees (valid for ~[-90, 90] ULA AoA)."""
    vals = [float(a) for a in aoas_deg if math.isfinite(float(a))]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _median_aoa_deg(aoas_deg: Sequence[float]) -> float:
    """Degree median of AoA samples (robust location for ~[-90, 90] ULA AoA)."""
    vals = [float(a) for a in aoas_deg if math.isfinite(float(a))]
    if not vals:
        return float("nan")
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def _circ_mean_aoa_deg(aoas_deg: Sequence[float]) -> float:
    """
    Circular mean: atan2(mean(sin), mean(cos)).
    For ULA AoA in ~[-90, 90], result is wrapped to (-180, 180].
    """
    vals = [float(a) for a in aoas_deg if math.isfinite(float(a))]
    if not vals:
        return float("nan")
    s = float(np.mean([math.sin(math.radians(a)) for a in vals]))
    c = float(np.mean([math.cos(math.radians(a)) for a in vals]))
    return float(math.degrees(math.atan2(s, c)))


def _mean_aoa_deg(aoas_deg: Sequence[float], *, mode: str = AOA_MEAN_MODE) -> float:
    if mode == "sin":
        return _sin_mean_aoa_deg(aoas_deg)
    if mode == "circ":
        return _circ_mean_aoa_deg(aoas_deg)
    if mode == "median":
        return _median_aoa_deg(aoas_deg)
    return _arith_mean_aoa_deg(aoas_deg)


def _robust_mean_aoa_deg(
    aoas_deg: Sequence[float],
    *,
    mad_k: float = 2.5,
    min_keep: int = 3,
    mode: str = AOA_MEAN_MODE,
) -> float:
    """
    Drop AoA outliers by MAD around the sample median, then aggregate inliers
    with the selected mode (default: circular mean).
    Residuals use wrapped angle difference (deg).
    """
    vals = np.asarray(
        [float(a) for a in aoas_deg if math.isfinite(float(a))],
        dtype=np.float64,
    )
    if vals.size == 0:
        return float("nan")
    if vals.size < min_keep:
        return _mean_aoa_deg(vals, mode=mode)

    # Provisional center = median (stable for MAD)
    mu0 = _median_aoa_deg(vals)
    if not math.isfinite(mu0):
        return float("nan")
    resid = np.asarray([_ang_diff_deg(float(a), mu0) for a in vals], dtype=np.float64)
    scale = 1.4826 * float(np.median(np.abs(resid))) + 1e-9
    keep = np.abs(resid) <= mad_k * scale
    n_keep = int(np.count_nonzero(keep))
    if n_keep < min_keep:
        order = np.argsort(np.abs(resid))
        keep = np.zeros(vals.size, dtype=bool)
        keep[order[: min(min_keep, vals.size)]] = True
    return _mean_aoa_deg(vals[keep], mode=mode)


def _localize_from_aoa_lists(
    r: Dict[str, Any],
    aoa_x: Sequence[float],
    aoa_y: Sequence[float],
    *,
    robust: Optional[bool] = None,
    mode: Optional[str] = None,
) -> Optional[Tuple[float, float]]:
    """Mean (optionally MAD-cleaned) X/Y AoAs → position. Reads AOA_* at call time."""
    use_mad = bool(AOA_USE_MAD) if robust is None else bool(robust)
    use_mode = AOA_MEAN_MODE if mode is None else mode
    if use_mad:
        ax = _robust_mean_aoa_deg(aoa_x, mode=use_mode)
        ay = _robust_mean_aoa_deg(aoa_y, mode=use_mode)
    else:
        ax = _mean_aoa_deg(aoa_x, mode=use_mode)
        ay = _mean_aoa_deg(aoa_y, mode=use_mode)
    if not (math.isfinite(ax) and math.isfinite(ay)):
        return None
    pos = cip.position_from_two_aoas(
        ax,
        ay,
        r["x_ids"],
        r["y_ids"],
        r["node_pos"],
        r["true_pos"],
    )
    if pos is None or not (math.isfinite(pos[0]) and math.isfinite(pos[1])):
        return None
    if abs(pos[0]) > 20.0 or abs(pos[1]) > 20.0:
        return None
    return (float(pos[0]), float(pos[1]))


def _target_label(true_pos: Sequence[float]) -> str:
    return f"({true_pos[0]:+.2f},{true_pos[1]:+.2f})"


def _axis_limits_equal(
    points: List[Tuple[float, float]],
    *,
    pad_frac: float = 0.12,
    min_span: float = 0.25,
) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    span = max(xmax - xmin, ymax - ymin, min_span)
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    half = 0.5 * span * (1.0 + pad_frac)
    return cx - half, cx + half, cy - half, cy + half


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "axes.linewidth": 0.9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": True,
            "legend.fancybox": False,
            "legend.framealpha": 0.92,
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFBFC",
            "grid.color": "0.82",
            "grid.linewidth": 0.6,
        }
    )


def _show(fig: plt.Figure, title: str) -> None:
    mgr = getattr(fig.canvas, "manager", None)
    if mgr is not None:
        try:
            mgr.set_window_title(title)
        except Exception:
            pass
    # tight_layout conflicts with constrained_layout once a colorbar exists
    use_constrained = bool(getattr(fig, "get_constrained_layout", lambda: False)())
    if not use_constrained:
        fig.tight_layout()
    print(f"Close window: {title}")
    plt.show(block=True)
    plt.close(fig)


def _draw_anchors(
    ax: plt.Axes,
    r0: Dict[str, Any],
    *,
    which: str = "both",
) -> None:
    """Draw REF and selected ULA(s). which: 'both' | 'X' | 'Y'."""
    node_pos = r0["node_pos"]
    ref_pos = r0["ref_pos"]
    x_ids, y_ids = r0["x_ids"], r0["y_ids"]
    if which in ("both", "X"):
        ax.plot(
            [node_pos[a][0] for a in x_ids],
            [node_pos[a][1] for a in x_ids],
            "o-",
            color=_C["gray"],
            ms=5,
            lw=1.0,
            zorder=2,
        )
    if which in ("both", "Y"):
        ax.plot(
            [node_pos[a][0] for a in y_ids],
            [node_pos[a][1] for a in y_ids],
            "s-",
            color=_C["gray"],
            ms=5,
            lw=1.0,
            zorder=2,
        )
    ax.plot(ref_pos[0], ref_pos[1], "^", color=_C["yellow"], ms=9, zorder=3)


def _ula_spacings_cm(
    ids: List[int],
    node_pos: Dict[int, Tuple[float, float]],
) -> List[float]:
    xs = [node_pos[a][0] for a in ids]
    ys = [node_pos[a][1] for a in ids]
    along_x = (max(xs) - min(xs)) >= (max(ys) - min(ys))
    if along_x:
        ordered = sorted(ids, key=lambda a: (node_pos[a][0], node_pos[a][1], a))
    else:
        ordered = sorted(ids, key=lambda a: (node_pos[a][1], node_pos[a][0], a))
    out: List[float] = []
    for i in range(len(ordered) - 1):
        out.append(cip.dist(node_pos[ordered[i]], node_pos[ordered[i + 1]]) * 100.0)
    return out


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------


def process_one_session(npz_path: str, *, uniform: bool) -> Optional[Dict[str, Any]]:
    """
    Structure-3 pipeline. For non-uniform layouts, keep baseline^2 weights so
    overlap-disambiguated long/short pairs keep physical reliability weighting.
    """
    session = cip.load_capture(npz_path)
    per_round, side = cip.process_session_pairs(
        session,
        half_plane_mode="true_aoa",
        infer_side=False,
    )
    pair_series = cip.collect_pair_series(per_round)
    if not pair_series:
        return None
    if uniform:
        _equalize_pair_weights(pair_series)

    loc = cip.fuse_session_localization(session, pair_series, per_round)
    round_pos = _round_positions(session, per_round, pair_series)
    inlier_mask, rob_center, _med = _classify_round_outliers(round_pos)
    true_pos = loc["true_pos"]

    pair_weights = {
        (p["group"], int(p["id_a"]), int(p["id_b"])): float(p["fusion_weight"])
        for p in pair_series
    }
    round_aoa_x: List[float] = []
    round_aoa_y: List[float] = []
    for round_pairs in per_round:
        round_aoa_x.append(cip.fuse_aoa_one_round(round_pairs, pair_weights, "X"))
        round_aoa_y.append(cip.fuse_aoa_one_round(round_pairs, pair_weights, "Y"))

    # Full-session estimate: aggregate round AoAs → position (MAD optional).
    if AOA_USE_MAD:
        fuse_aoa_x = _robust_mean_aoa_deg(round_aoa_x)
        fuse_aoa_y = _robust_mean_aoa_deg(round_aoa_y)
    else:
        fuse_aoa_x = _mean_aoa_deg(round_aoa_x)
        fuse_aoa_y = _mean_aoa_deg(round_aoa_y)
    base_pos = cip.position_from_two_aoas(
        fuse_aoa_x,
        fuse_aoa_y,
        session["x_ids"],
        session["y_ids"],
        session["node_pos"],
        true_pos,
    )
    if base_pos is not None:
        base_pos = (float(base_pos[0]), float(base_pos[1]))
        if abs(base_pos[0]) > 20.0 or abs(base_pos[1]) > 20.0:
            base_pos = None
    est_pos, loc_mode = _maybe_replace_with_robust(base_pos, round_pos)

    if est_pos is not None:
        pos_err_m = cip.dist(est_pos, true_pos)
    else:
        pos_err_m = float("nan")

    aliased_x = bool(side.get("aliased_x"))
    aliased_y = bool(side.get("aliased_y"))

    return {
        "npz_path": session["npz_path"],
        "name": os.path.basename(session["npz_path"]),
        "ref_pos": session["ref_pos"],
        "node_pos": session["node_pos"],
        "x_ids": session["x_ids"],
        "y_ids": session["y_ids"],
        "structure_name": session["structure_name"],
        "true_pos": true_pos,
        "est_pos": est_pos,
        "base_pos": base_pos,
        "robust_center": rob_center,
        "pos_err_m": pos_err_m,
        "fuse_aoa_x_deg": float(fuse_aoa_x),
        "fuse_aoa_y_deg": float(fuse_aoa_y),
        "true_aoa_x_deg": float(loc["true_aoa_x_deg"]),
        "true_aoa_y_deg": float(loc["true_aoa_y_deg"]),
        "n_rounds": session["n_rounds"],
        "loc_mode": loc_mode,
        "round_pos": round_pos,
        "round_aoa_x": round_aoa_x,
        "round_aoa_y": round_aoa_y,
        "round_inlier": inlier_mask,
        "n_outliers": int(np.count_nonzero(~inlier_mask)) if inlier_mask.size else 0,
        "n_round_pos": len(round_pos),
        "uniform": uniform,
        "aliased_x": aliased_x,
        "aliased_y": aliased_y,
        "overlap_enabled": bool(cip.ENABLE_OVERLAP_DISAMBIG),
        "side": side,
    }


def _window_mean_aoa_positions(
    r: Dict[str, Any],
    *,
    win_len: int = SEGMENT_LEN,
    n_windows: int = SEGMENT_N,
    rng: np.random.Generator,
) -> Tuple[List[Tuple[float, float]], List[int]]:
    rx = np.asarray(r.get("round_aoa_x") or [], dtype=np.float64)
    ry = np.asarray(r.get("round_aoa_y") or [], dtype=np.float64)
    n = int(min(rx.size, ry.size))
    if n < win_len:
        return [], []

    starts_ok: List[int] = []
    for s in range(0, n - win_len + 1):
        sl = slice(s, s + win_len)
        if np.all(np.isfinite(rx[sl])) and np.all(np.isfinite(ry[sl])):
            starts_ok.append(s)
    if not starts_ok:
        return [], []

    k = min(n_windows, len(starts_ok))
    chosen = [int(x) for x in rng.choice(starts_ok, size=k, replace=False)]
    chosen.sort()

    out: List[Tuple[float, float]] = []
    used: List[int] = []
    for s in chosen:
        sl = slice(s, s + win_len)
        pos = _localize_from_aoa_lists(r, rx[sl], ry[sl])
        if pos is None:
            continue
        out.append(pos)
        used.append(s)
    return out, used


def _attach_segment_samples(
    results: List[Dict[str, Any]],
    *,
    win_len: int = SEGMENT_LEN,
    n_windows: int = SEGMENT_N,
    seed: int = SEGMENT_SEED,
) -> None:
    for i, r in enumerate(results):
        rng = np.random.default_rng(seed + i)
        samples, starts = _window_mean_aoa_positions(
            r, win_len=win_len, n_windows=n_windows, rng=rng
        )
        true = r["true_pos"]
        r["segment_pos"] = samples
        r["segment_starts"] = starts
        r["segment_err_cm"] = [cip.dist(p, true) * 100.0 for p in samples]
        r["segment_win_len"] = win_len


def _bootstrap_window_positions(
    r: Dict[str, Any],
    *,
    n_obs: int = CDF_BOOT_N_OBS,
    n_draw: int = SEGMENT_LEN,
    rng: np.random.Generator,
) -> List[Tuple[float, float]]:
    """
    Construct many positions (same recipe as CDF):
    each sample draws n_draw round AoAs *without* replacement → aggregate
    (no MAD when AOA_USE_MAD is False) → localize.
    """
    rx = np.asarray(r.get("round_aoa_x") or [], dtype=np.float64)
    ry = np.asarray(r.get("round_aoa_y") or [], dtype=np.float64)
    n = int(min(rx.size, ry.size))
    if n < 2:
        return []
    valid = np.flatnonzero(np.isfinite(rx[:n]) & np.isfinite(ry[:n]))
    if valid.size < 2:
        return []

    k = min(n_draw, int(valid.size))
    out: List[Tuple[float, float]] = []
    for _ in range(n_obs):
        pick = rng.choice(valid, size=k, replace=False)
        pos = _localize_from_aoa_lists(r, rx[pick], ry[pick])
        if pos is None:
            continue
        out.append(pos)
    return out


def _attach_bootstrap_samples(
    results: List[Dict[str, Any]],
    *,
    n_obs: int = CDF_BOOT_N_OBS,
    n_draw: int = SEGMENT_LEN,
    seed: int = CDF_BOOT_SEED,
    seed_offset: int = 0,
    pos_key: str = "bootstrap_pos",
    err_key: str = "bootstrap_err_cm",
    n_obs_key: str = "bootstrap_n_obs",
    n_draw_key: str = "bootstrap_n_draw",
) -> None:
    """Fill constructed window positions/errors (map and/or CDF)."""
    for i, r in enumerate(results):
        rng = np.random.default_rng(seed + seed_offset + 17 * i)
        samples = _bootstrap_window_positions(r, n_obs=n_obs, n_draw=n_draw, rng=rng)
        true = r["true_pos"]
        r[pos_key] = samples
        r[err_key] = [cip.dist(p, true) * 100.0 for p in samples]
        r[n_obs_key] = len(samples)
        r[n_draw_key] = n_draw


def process_campaign(camp: Campaign) -> List[Dict[str, Any]]:
    if not os.path.isdir(camp.data_dir):
        print(f"[{camp.key}] missing dir: {camp.data_dir}", file=sys.stderr)
        return []
    npz_files = find_structure_sessions(camp.data_dir, camp.structure_id)
    print(
        f"==== {camp.title} ====\n"
        f"  dir={camp.data_dir}  structure_id={camp.structure_id}  "
        f"uniform={camp.uniform}  N_files={len(npz_files)}"
    )
    if not npz_files:
        return []

    try:
        first = cip.load_capture(npz_files[0])
        x_sp = _ula_spacings_cm(first["x_ids"], first["node_pos"])
        y_sp = _ula_spacings_cm(first["y_ids"], first["node_pos"])
        print(
            f"  layout={first.get('structure_name', '')}  "
            f"X d=[{', '.join(f'{s:.1f}' for s in x_sp)}] cm  "
            f"Y d=[{', '.join(f'{s:.1f}' for s in y_sp)}] cm  "
            f"λ/2={0.5 * cip.wavelength_m() * 100:.2f} cm  "
            f"overlap={cip.ENABLE_OVERLAP_DISAMBIG}"
        )
    except Exception as exc:
        print(f"  (layout print failed: {exc})", file=sys.stderr)

    results: List[Dict[str, Any]] = []
    for path in npz_files:
        name = os.path.basename(path)
        print(f"---- {name} ----")
        try:
            res = process_one_session(path, uniform=camp.uniform)
        except Exception as exc:
            print(f"  failed: {exc}", file=sys.stderr)
            continue
        if res is None:
            print("  skip: no pair estimates")
            continue
        results.append(res)
        if res["est_pos"] is None:
            print(f"  TARGET={res['true_pos']}  est=None")
        else:
            print(
                f"  TARGET=({res['true_pos'][0]:.3f},{res['true_pos'][1]:.3f})  "
                f"est=({res['est_pos'][0]:.3f},{res['est_pos'][1]:.3f})  "
                f"err={res['pos_err_m']*100:.1f} cm  mode={res['loc_mode']}"
            )

    # Deterministic offset per campaign; map uses SEGMENT_N, CDF uses CDF_BOOT_N_OBS.
    seed_offset = abs(hash(camp.key)) % 100_000
    _attach_bootstrap_samples(
        results,
        n_obs=SEGMENT_N,
        seed_offset=seed_offset,
        pos_key="map_pos",
        err_key="map_err_cm",
        n_obs_key="map_n_obs",
        n_draw_key="map_n_draw",
    )
    _attach_bootstrap_samples(
        results,
        n_obs=CDF_BOOT_N_OBS,
        seed_offset=seed_offset + 1_000_003,
    )
    errs = [
        r["pos_err_m"] * 100.0
        for r in results
        if r["est_pos"] is not None and math.isfinite(r["pos_err_m"])
    ]
    if errs:
        print(
            f"  summary N={len(errs)}  mean={np.mean(errs):.1f} cm  "
            f"median={np.median(errs):.1f} cm"
        )
    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _campaign_map_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Map rows: session angle-mean fuse (base_pos) + SEGMENT_N constructed
    positions (mean of SEGMENT_LEN rounds, without replacement).
    """
    rows: List[Dict[str, Any]] = []
    for r in results:
        est = r.get("base_pos")
        if est is None:
            est = r.get("est_pos")
        if est is None:
            continue
        true = r["true_pos"]
        err_cm = cip.dist(est, true) * 100.0
        if not math.isfinite(err_cm):
            continue
        rows.append(
            {
                "label": _target_label(true),
                "true": tuple(true),
                "est": tuple(est),
                "err_cm": float(err_cm),
                "samples": list(r.get("map_pos") or []),
                "loc_mode": r.get("loc_mode", ""),
            }
        )
    return rows


def _draw_campaign_map(
    ax: plt.Axes,
    camp: Campaign,
    results: List[Dict[str, Any]],
    *,
    legend: bool = True,
) -> None:
    rows = _campaign_map_rows(results)
    if not rows:
        ax.set_axis_off()
        ax.set_title(camp.title, fontsize=9)
        return

    r0 = results[0]
    lo, hi = MAP_XY_LIM
    tab = plt.cm.tab10  # type: ignore[attr-defined]
    group_colors = [tab(i % 10) for i in range(len(rows))]

    _draw_anchors(ax, r0)
    for i, row in enumerate(rows):
        tx, ty = row["true"]
        ex, ey = row["est"]
        gcolor = group_colors[i]
        if row["samples"]:
            ax.scatter(
                [p[0] for p in row["samples"]],
                [p[1] for p in row["samples"]],
                s=16,
                c=[gcolor],
                alpha=0.40,
                edgecolors="none",
                linewidths=0,
                zorder=5,
            )
        ax.plot([tx, ex], [ty, ey], "-", color=gcolor, lw=1.3, alpha=0.9, zorder=5)
        ax.plot(
            tx,
            ty,
            "*",
            color=gcolor,
            ms=10,
            markeredgecolor=_C["ink"],
            markeredgewidth=0.4,
            zorder=6,
        )
        ax.plot(
            ex,
            ey,
            "P",
            color=gcolor,
            ms=7.5,
            markeredgecolor=_C["ink"],
            markeredgewidth=0.4,
            zorder=7,
        )
        # Label near estimate, offset perpendicular to the link so it clears the legend.
        vx, vy = ex - tx, ey - ty
        link_len = math.hypot(vx, vy) or 1.0
        nx, ny = -vy / link_len, vx / link_len
        # prefer offset toward map exterior (away from origin)
        if nx * ex + ny * ey < 0.0:
            nx, ny = -nx, -ny
        lx = 0.72 * ex + 0.28 * tx + 0.055 * nx
        ly = 0.72 * ey + 0.28 * ty + 0.055 * ny
        ax.annotate(
            f"{row['err_cm']:.0f}cm",
            (lx, ly),
            fontsize=6.5,
            color=gcolor,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75),
            zorder=8,
        )

    n_map = max((len(row["samples"]) for row in rows), default=SEGMENT_N)
    if legend:
        handles = [
            Line2D([0], [0], marker="o", color=_C["gray"], ls="-", ms=4, label="X-ULA"),
            Line2D([0], [0], marker="s", color=_C["gray"], ls="-", ms=4, label="Y-ULA"),
            Line2D([0], [0], marker="^", color=_C["yellow"], ls="None", ms=6, label="REF"),
            Line2D([0], [0], marker="*", color=_C["ink"], ls="None", ms=8, label="True"),
            Line2D([0], [0], marker="P", color=_C["ink"], ls="None", ms=6, label="Angle-mean fuse"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="0.5",
                ls="None",
                ms=5,
                label=f"Samples ({SEGMENT_LEN}×{n_map})",
            ),
        ]
        # Fixed corner keeps labels from fighting auto-legend placement.
        ax.legend(handles=handles, loc="lower left", fontsize=5.5, ncol=2, framealpha=0.92)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.35)
    ax.set_title(camp.title, fontsize=9)


def fig_maps_2x2(
    campaigns: Sequence[Campaign],
    by_key: Dict[str, List[Dict[str, Any]]],
) -> None:
    """Four campaign map panels in fixed 2×2 layout order."""
    ordered = [c for c in _campaigns_layout_order(campaigns) if by_key.get(c.key)]
    if not ordered:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.5))
    axes_flat = np.asarray(axes).ravel()
    for i, ax in enumerate(axes_flat):
        if i >= len(ordered):
            ax.set_axis_off()
            continue
        camp = ordered[i]
        _draw_campaign_map(ax, camp, by_key.get(camp.key) or [], legend=(i == 0))
    fig.suptitle(
        f"Map + localizations "
        f"(mean of {SEGMENT_LEN} rounds × {SEGMENT_N}/target)",
        fontsize=12,
        y=0.995,
    )
    _show(fig, "Maps 2×2")


def fig_combined_cdf(
    campaigns: Sequence[Campaign],
    by_key: Dict[str, List[Dict[str, Any]]],
) -> None:
    """
    CDF from bootstrap-constructed window errors (many random mean-of-SEGMENT_LEN
    draws per target) so curves are smoother than the raw 5-window samples.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8.6, 5.8))
    markers = ["o", "s", "^", "D"]
    any_curve = False
    curve_handles: List[Any] = []
    mean_marks: List[Tuple[float, str]] = []
    p90_marks: List[Tuple[float, str]] = []
    ordered = _campaigns_layout_order(campaigns)
    for i, camp in enumerate(ordered):
        results = by_key.get(camp.key) or []
        err_list: List[float] = []
        for r in results:
            # Reuse the same bootstrap draws already attached for the map.
            for e_cm in r.get("bootstrap_err_cm") or []:
                if math.isfinite(float(e_cm)):
                    err_list.append(float(e_cm))
        errs = np.asarray(err_list, dtype=np.float64)
        if errs.size == 0:
            continue
        any_curve = True
        e = np.sort(errs)
        y = np.arange(1, e.size + 1, dtype=np.float64) / float(e.size)
        color = CAMPAIGN_COLORS.get(camp.key, _C["ink"])
        mean = float(np.mean(e))
        p90 = float(np.percentile(e, 90))
        y_mean = float(np.interp(mean, e, y, left=0.0, right=1.0))

        ax.plot(e, y, color=color, lw=1.9, zorder=4)
        step = max(1, e.size // 16)
        ax.plot(
            e[::step],
            y[::step],
            markers[i % len(markers)],
            color=color,
            ms=4.5,
            linestyle="None",
            zorder=5,
        )
        curve_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=1.9,
                marker=markers[i % len(markers)],
                ms=5,
                label=camp.title,
            )
        )

        ax.axvline(p90, color=color, ls="--", lw=1.0, alpha=0.45, zorder=3)
        ax.plot([p90], [0.90], marker="o", color=color, ms=6, zorder=6)
        ax.axvline(mean, color=color, ls="-.", lw=1.1, alpha=0.65, zorder=3)
        ax.plot([mean], [y_mean], marker="D", color=color, ms=6, zorder=7)
        mean_marks.append((mean, color))
        p90_marks.append((p90, color))

    if not any_curve:
        plt.close(fig)
        return

    # All numeric labels at the foot of each vline; left→right y = 0.05, 0.10, ...
    value_labels: List[Tuple[float, str, str]] = []
    for mean, color in mean_marks:
        value_labels.append((mean, color, f"μ={mean:.1f}"))
    for p90, color in p90_marks:
        value_labels.append((p90, color, f"p90={p90:.1f}"))
    value_labels.sort(key=lambda t: t[0])
    for rank, (x_val, color, text) in enumerate(value_labels):
        y_lab = 0.05 + rank * 0.05
        ax.text(
            x_val,
            y_lab,
            text,
            fontsize=8.0,
            color=color,
            ha="center",
            va="bottom",
            zorder=8,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75),
        )

    ax.axhline(0.90, color="0.55", ls=":", lw=0.9, zorder=2)
    ax.set_xlabel("position error [cm]")
    ax.set_ylabel("CDF")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(left=0.0)
    ax.grid(True, alpha=0.35)

    legend_handles = list(curve_handles) + [
        Line2D([0], [0], color="0.35", ls="-.", lw=1.2, marker="D", ms=6, label="Mean"),
        Line2D([0], [0], color="0.35", ls="--", lw=1.0, marker="o", ms=6, label="p90"),
        Line2D([0], [0], color="0.55", ls=":", lw=0.9, label="CDF = 0.90"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=7.0,
        framealpha=0.92,
    )
    ax.set_title(
        f"Bootstrap window-error CDF — mean of {SEGMENT_LEN} rounds × "
        f"{CDF_BOOT_N_OBS}/target"
    )
    _show(fig, "Combined CDF")


def fig_aoa_residuals_grid(
    campaigns: Sequence[Campaign],
    by_key: Dict[str, List[Dict[str, Any]]],
) -> None:
    """
    Outer 2×2 in LAYOUT_KEYS order; each cell is a tight 1×2 (X | Y).
    Small vertical gap between the two campaign rows.
    """
    camp_rows: List[Tuple[Campaign, List[Dict[str, Any]], Dict[str, Any]]] = []
    for camp in _campaigns_layout_order(campaigns):
        results = by_key.get(camp.key) or []
        rows: List[Dict[str, Any]] = []
        for r in results:
            if r.get("est_pos") is None:
                continue
            if not (
                math.isfinite(r.get("fuse_aoa_x_deg", float("nan")))
                and math.isfinite(r.get("fuse_aoa_y_deg", float("nan")))
                and math.isfinite(r.get("true_aoa_x_deg", float("nan")))
                and math.isfinite(r.get("true_aoa_y_deg", float("nan")))
            ):
                continue
            rows.append(
                {
                    "xy": tuple(r["true_pos"]),
                    "dx": _ang_diff_deg(r["fuse_aoa_x_deg"], r["true_aoa_x_deg"]),
                    "dy": _ang_diff_deg(r["fuse_aoa_y_deg"], r["true_aoa_y_deg"]),
                }
            )
        if rows:
            camp_rows.append((camp, rows, results[0]))

    if not camp_rows:
        return

    abs_all: List[float] = []
    for _c, rows, _r0 in camp_rows:
        abs_all.extend(abs(row["dx"]) for row in rows)
        abs_all.extend(abs(row["dy"]) for row in rows)
    vmax = max(abs_all + [1.0])
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.cm.RdBu_r  # type: ignore[attr-defined]

    # Manual GridSpec so hspace/wspace are honored (constrained_layout ignores them).
    fig = plt.figure(figsize=(13.8, 8.6))
    outer = fig.add_gridspec(
        2,
        2,
        left=0.05,
        right=0.90,
        top=0.90,
        bottom=0.07,
        hspace=0.05,
        wspace=0.18,
    )
    all_axes: List[plt.Axes] = []

    # Shared equal-aspect limits from full layout (both ULAs), even if one ULA is hidden.
    lim_pts: List[Tuple[float, float]] = []
    for _c, rows, r0 in camp_rows:
        lim_pts.extend(r0["node_pos"].values())
        lim_pts.append(r0["ref_pos"])
        lim_pts.extend(row["xy"] for row in rows)
    shared_xlim = _axis_limits_equal(lim_pts, pad_frac=0.22, min_span=0.5)

    def _panel(
        ax: plt.Axes,
        r0: Dict[str, Any],
        rows: List[Dict[str, Any]],
        key: str,
        title: str,
        *,
        ula: str,
    ) -> Any:
        # Draw only this ULA, but keep shared x/y limits & equal aspect.
        _draw_anchors(ax, r0, which=ula)
        xs = [row["xy"][0] for row in rows]
        ys = [row["xy"][1] for row in rows]
        vals = [row[key] for row in rows]
        sizes = [48.0 + 18.0 * abs(v) for v in vals]
        sc = ax.scatter(
            xs,
            ys,
            c=vals,
            s=sizes,
            cmap=cmap,
            norm=norm,
            edgecolors=_C["ink"],
            linewidths=0.4,
            zorder=5,
        )
        stalk = 0.010
        for row, v in zip(rows, vals):
            x0, y0 = row["xy"]
            ax.plot([x0, x0], [y0, y0 + stalk * v], color=_C["ink"], lw=0.65, alpha=0.40, zorder=4)
            ax.annotate(
                f"{v:+.1f}°",
                (x0, y0 + stalk * v),
                textcoords="offset points",
                xytext=(0, 3 if v >= 0 else -8),
                ha="center",
                fontsize=5.5,
                color=_C["ink"],
                zorder=6,
            )
        xmin, xmax, ymin, ymax = shared_xlim
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)
        ax.set_title(title, fontsize=8.0, pad=3)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.35)
        return sc

    last_sc = None
    for i, (camp, rows, r0) in enumerate(camp_rows[:4]):
        rr, cc = divmod(i, 2)
        # tight gap between X and Y within a campaign
        inner = outer[rr, cc].subgridspec(1, 2, wspace=0.06)
        ax_x = fig.add_subplot(inner[0, 0])
        ax_y = fig.add_subplot(inner[0, 1])
        all_axes.extend([ax_x, ax_y])
        last_sc = _panel(
            ax_x, r0, rows, "dx", f"{camp.title}  |  X-ULA Δθ", ula="X"
        )
        _panel(ax_y, r0, rows, "dy", f"{camp.title}  |  Y-ULA Δθ", ula="Y")

    if last_sc is not None and all_axes:
        cbar = fig.colorbar(last_sc, ax=all_axes, fraction=0.025, pad=0.03)
        cbar.set_label("AoA residual Δθ [deg]")

    fig.suptitle(
        "Fused AoA mean residual by target position (four campaigns)",
        fontsize=12,
        y=0.97,
    )
    _show(fig, "AoA residual grid (2×4 paired)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    _apply_plot_style()
    by_key: Dict[str, List[Dict[str, Any]]] = {}
    for camp in CAMPAIGNS:
        results = process_campaign(camp)
        by_key[camp.key] = results

    n_ok = sum(1 for v in by_key.values() if v)
    if n_ok == 0:
        print("No campaigns processed successfully.", file=sys.stderr)
        return 1

    fig_maps_2x2(CAMPAIGNS, by_key)
    fig_aoa_residuals_grid(CAMPAIGNS, by_key)
    fig_combined_cdf(CAMPAIGNS, by_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
