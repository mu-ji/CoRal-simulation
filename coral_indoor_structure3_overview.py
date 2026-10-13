#!/usr/bin/env python3
"""
Overview plots for indoor_coral_2 sessions with anchor_structure_id == 3.

Pipeline: equal-weight pair fusion (half_plane_mode="true_aoa"), then density
consistency check vs per-round positions (MAD center) when the fuse sits
outside the round cloud.

Three interactive figures (no file export):
    1) map + 5×15-round window localizations; window error box
    2) spatial X/Y-ULA AoA-mean residual fields
    3) CDF: bootstrap samples vs session angle-mean fuse
    4) within-session n_draw stability (optional)

Usage:
  python coral_indoor_structure3_overview.py
  python coral_indoor_structure3_overview.py path/to/indoor_coral_2
"""

from __future__ import annotations

import glob
import math
import os
import sys
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

INDOOR_CORAL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "indoor_coral_2",
)
STRUCTURE_ID = 3

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


def find_structure_sessions(
    data_dir: str,
    structure_id: int = STRUCTURE_ID,
) -> List[str]:
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
    *,
    hard_radius_m: float = 2.5,
    mad_k: float = 2.5,
    min_keep: int = 8,
) -> Optional[Tuple[float, float]]:
    _mask, center, _med = _classify_round_outliers(
        positions,
        hard_radius_m=hard_radius_m,
        mad_k=mad_k,
        min_keep=min_keep,
    )
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
    """
    Keep angle-mean fuse unless dens_r >= dens_rob_min and dens_b/dens_r < dens_ratio_max.
    """
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
    fig.tight_layout()
    print(f"Close window: {title}")
    plt.show(block=True)
    plt.close(fig)


def _draw_anchors(ax: plt.Axes, r0: Dict[str, Any]) -> None:
    node_pos = r0["node_pos"]
    ref_pos = r0["ref_pos"]
    x_ids, y_ids = r0["x_ids"], r0["y_ids"]
    ax.plot(
        [node_pos[a][0] for a in x_ids],
        [node_pos[a][1] for a in x_ids],
        "o-",
        color=_C["gray"],
        ms=5,
        lw=1.0,
        label="X-ULA",
        zorder=2,
    )
    ax.plot(
        [node_pos[a][0] for a in y_ids],
        [node_pos[a][1] for a in y_ids],
        "s-",
        color=_C["gray"],
        ms=5,
        lw=1.0,
        label="Y-ULA",
        zorder=2,
    )
    ax.plot(ref_pos[0], ref_pos[1], "^", color=_C["yellow"], ms=9, label="REF", zorder=3)


def process_one_session(npz_path: str) -> Optional[Dict[str, Any]]:
    session = cip.load_capture(npz_path)
    per_round, _side = cip.process_session_pairs(
        session,
        half_plane_mode="true_aoa",
        infer_side=False,
    )
    pair_series = cip.collect_pair_series(per_round)
    if not pair_series:
        return None
    _equalize_pair_weights(pair_series)
    loc = cip.fuse_session_localization(session, pair_series, per_round)

    round_pos = _round_positions(session, per_round, pair_series)
    inlier_mask, rob_center, _med = _classify_round_outliers(round_pos)
    base_pos = loc["est_pos"]
    est_pos, loc_mode = _maybe_replace_with_robust(base_pos, round_pos)
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

    if est_pos is not None:
        pos_err_m = cip.dist(est_pos, true_pos)
    else:
        pos_err_m = float("nan")

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
        "fuse_aoa_x_deg": float(loc["fuse_x"]["aoa_deg"]),
        "fuse_aoa_y_deg": float(loc["fuse_y"]["aoa_deg"]),
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
    }


def _plot_cdf(ax: plt.Axes, errs: np.ndarray, *, color: str, label: str, marker: str = "o") -> None:
    e = np.sort(errs[np.isfinite(errs)])
    if e.size == 0:
        return
    y = np.arange(1, e.size + 1, dtype=np.float64) / float(e.size)
    ax.step(e, y, where="post", color=color, lw=1.8, label=label)
    ax.plot(e, y, marker, color=color, ms=5, linestyle="None")


# ---------------------------------------------------------------------------
# Three figures
# ---------------------------------------------------------------------------


def _sin_mean_aoa_deg(aoas_deg: Sequence[float]) -> float:
    vals = [float(a) for a in aoas_deg if math.isfinite(float(a))]
    if not vals:
        return float("nan")
    s = float(np.mean([math.sin(math.radians(a)) for a in vals]))
    s = float(np.clip(s, -1.0, 1.0))
    return float(np.degrees(np.arcsin(s)))


# Resample existing ~50 round AoAs into a larger synthetic sample set (fig3/fig4).
BOOTSTRAP_N_OBS = 10  # how many constructed position samples per target
BOOTSTRAP_N_DRAW = 30  # rounds averaged into one constructed sample
BOOTSTRAP_SEED = 42

# Fig1: consecutive temporal windows as repeated localizations.
SEGMENT_LEN = 15  # consecutive rounds per localization
SEGMENT_N = 5  # number of random window starts per target
SEGMENT_SEED = 42


def _bootstrap_mean_aoa_positions(
    r: Dict[str, Any],
    *,
    n_obs: int = BOOTSTRAP_N_OBS,
    n_draw: int = BOOTSTRAP_N_DRAW,
    rng: np.random.Generator,
    replace: bool = True,
) -> List[Tuple[float, float]]:
    """
    Construct many position samples from existing round AoAs.

    Each sample: draw n_draw rounds (with replacement by default) from the
    ~50 angle estimates → sin-mean AoA_X/Y → localize. Repeat n_obs times.
    """
    rx = np.asarray(r.get("round_aoa_x") or [], dtype=np.float64)
    ry = np.asarray(r.get("round_aoa_y") or [], dtype=np.float64)
    n = int(min(rx.size, ry.size))
    if n < 2:
        return []
    valid = np.flatnonzero(np.isfinite(rx[:n]) & np.isfinite(ry[:n]))
    if valid.size < 2:
        return []

    out: List[Tuple[float, float]] = []
    k = min(n_draw, int(valid.size))
    side = r["true_pos"]
    for _ in range(n_obs):
        pick = rng.choice(valid, size=k, replace=replace)
        ax = _sin_mean_aoa_deg(rx[pick])
        ay = _sin_mean_aoa_deg(ry[pick])
        pos = cip.position_from_two_aoas(
            ax,
            ay,
            r["x_ids"],
            r["y_ids"],
            r["node_pos"],
            side,
        )
        if pos is None or not (math.isfinite(pos[0]) and math.isfinite(pos[1])):
            continue
        if abs(pos[0]) > 20.0 or abs(pos[1]) > 20.0:
            continue
        out.append((float(pos[0]), float(pos[1])))
    return out


def _attach_bootstrap_samples(
    results: List[Dict[str, Any]],
    *,
    n_obs: int = BOOTSTRAP_N_OBS,
    n_draw: int = BOOTSTRAP_N_DRAW,
    seed: int = BOOTSTRAP_SEED,
) -> None:
    """Fill r['bootstrap_pos'] / r['bootstrap_err_cm'] for each session."""
    rng = np.random.default_rng(seed)
    for r in results:
        samples = _bootstrap_mean_aoa_positions(r, n_obs=n_obs, n_draw=n_draw, rng=rng)
        true = r["true_pos"]
        r["bootstrap_pos"] = samples
        r["bootstrap_err_cm"] = [cip.dist(p, true) * 100.0 for p in samples]
        r["bootstrap_n_draw"] = n_draw
        r["bootstrap_n_obs"] = len(samples)


def _window_mean_aoa_positions(
    r: Dict[str, Any],
    *,
    win_len: int = SEGMENT_LEN,
    n_windows: int = SEGMENT_N,
    rng: np.random.Generator,
) -> Tuple[List[Tuple[float, float]], List[int]]:
    """
    Random start indices → consecutive win_len-round AoA mean → position.

    Returns (positions, start_indices_used).
    """
    rx = np.asarray(r.get("round_aoa_x") or [], dtype=np.float64)
    ry = np.asarray(r.get("round_aoa_y") or [], dtype=np.float64)
    n = int(min(rx.size, ry.size))
    if n < win_len:
        return [], []

    # starts where the whole window has finite X/Y AoA
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
    side = r["true_pos"]
    for s in chosen:
        sl = slice(s, s + win_len)
        ax = _sin_mean_aoa_deg(rx[sl])
        ay = _sin_mean_aoa_deg(ry[sl])
        pos = cip.position_from_two_aoas(
            ax,
            ay,
            r["x_ids"],
            r["y_ids"],
            r["node_pos"],
            side,
        )
        if pos is None or not (math.isfinite(pos[0]) and math.isfinite(pos[1])):
            continue
        if abs(pos[0]) > 20.0 or abs(pos[1]) > 20.0:
            continue
        out.append((float(pos[0]), float(pos[1])))
        used.append(s)
    return out, used


def _attach_segment_samples(
    results: List[Dict[str, Any]],
    *,
    win_len: int = SEGMENT_LEN,
    n_windows: int = SEGMENT_N,
    seed: int = SEGMENT_SEED,
) -> None:
    """Fill r['segment_pos'] / errors from random consecutive windows."""
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


def fig1_experiment_summary(results: List[Dict[str, Any]]) -> None:
    """
    Campaign summary:
      (a) true–est links + error labels + 5 consecutive-window localizations
      (b) error distribution of those window localizations

    Each window: SEGMENT_LEN consecutive round AoAs → sin-mean → position;
    SEGMENT_N random starts per target.
    """
    if any(r.get("segment_pos") is None for r in results):
        _attach_segment_samples(results)

    r0 = results[0]
    rows = []
    for r in results:
        if r["est_pos"] is None or not math.isfinite(r["pos_err_m"]):
            continue
        samples = list(r.get("segment_pos") or [])
        sample_err_cm = list(r.get("segment_err_cm") or [])
        rows.append(
            {
                "label": _target_label(r["true_pos"]),
                "true": tuple(r["true_pos"]),
                "est": tuple(r["est_pos"]),
                "err_cm": float(r["pos_err_m"]) * 100.0,
                "samples": samples,
                "sample_err_cm": sample_err_cm,
                "starts": list(r.get("segment_starts") or []),
                "n_rounds_aoa": len(r.get("round_aoa_x") or []),
            }
        )
    if not rows:
        return

    pts: List[Tuple[float, float]] = list(r0["node_pos"].values()) + [r0["ref_pos"]]
    for row in rows:
        pts.append(row["true"])
        pts.append(row["est"])
        pts.extend(row["samples"])
    xmin, xmax, ymin, ymax = _axis_limits_equal(pts, pad_frac=0.12, min_span=0.5)
    tab = plt.cm.tab10  # type: ignore[attr-defined]
    group_colors = [tab(i % 10) for i in range(len(rows))]

    fig, (ax_map, ax_dist) = plt.subplots(
        1,
        2,
        figsize=(13.2, 6.4),
        gridspec_kw={"width_ratios": [1.2, 1.05], "wspace": 0.28},
    )
    _draw_anchors(ax_map, r0)

    for i, row in enumerate(rows):
        tx, ty = row["true"]
        ex, ey = row["est"]
        gcolor = group_colors[i]
        if row["samples"]:
            sx = [p[0] for p in row["samples"]]
            sy = [p[1] for p in row["samples"]]
            ax_map.scatter(
                sx,
                sy,
                s=36,
                c=[gcolor],
                alpha=0.75,
                edgecolors=_C["ink"],
                linewidths=0.35,
                zorder=5,
                label="_nolegend_",
            )
        ax_map.plot([tx, ex], [ty, ey], "-", color=gcolor, lw=1.5, alpha=0.9, zorder=5)
        ax_map.plot(
            tx,
            ty,
            "*",
            color=gcolor,
            ms=12,
            markeredgecolor=_C["ink"],
            markeredgewidth=0.45,
            zorder=6,
        )
        ax_map.plot(
            ex,
            ey,
            "P",
            color=gcolor,
            ms=9,
            markeredgecolor=_C["ink"],
            markeredgewidth=0.45,
            zorder=7,
        )
        ax_map.annotate(
            f"{row['err_cm']:.0f}cm",
            (0.5 * (tx + ex), 0.5 * (ty + ey)),
            fontsize=7,
            color=gcolor,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
            zorder=8,
        )

    n_seg = max((len(row["samples"]) for row in rows), default=SEGMENT_N)
    handles = [
        Line2D([0], [0], marker="o", color=_C["gray"], ls="-", ms=5, label="X-ULA"),
        Line2D([0], [0], marker="s", color=_C["gray"], ls="-", ms=5, label="Y-ULA"),
        Line2D([0], [0], marker="^", color=_C["yellow"], ls="None", ms=8, label="REF"),
        Line2D([0], [0], marker="*", color=_C["ink"], ls="None", ms=10, label="True"),
        Line2D([0], [0], marker="P", color=_C["ink"], ls="None", ms=8, label="Full-session est."),
        Line2D([0], [0], color=_C["ink"], lw=1.2, label="True–est"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="0.5",
            ls="None",
            ms=6,
            label=f"Window est. ({SEGMENT_LEN} rounds × {n_seg})",
        ),
    ]
    for i, row in enumerate(rows):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color=group_colors[i],
                ls="None",
                ms=6,
                label=row["label"],
            )
        )
    ax_map.legend(handles=handles, loc="best", fontsize=6.5, ncol=2)
    ax_map.set_xlim(xmin, xmax)
    ax_map.set_ylim(ymin, ymax)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xlabel("x [m]")
    ax_map.set_ylabel("y [m]")
    ax_map.grid(True, alpha=0.35)
    ax_map.set_title(f"(a) Map + {SEGMENT_LEN}-round window localizations ({n_seg}/target)")

    labels = [row["label"] for row in rows]
    data = [row["sample_err_cm"] for row in rows]
    session_err = [row["err_cm"] for row in rows]
    positions = np.arange(1, len(rows) + 1)

    bp = ax_dist.boxplot(
        data,
        positions=positions,
        widths=0.55,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color=_C["ink"], lw=1.2),
        whiskerprops=dict(color="0.45"),
        capprops=dict(color="0.45"),
    )
    for patch, gcolor in zip(bp["boxes"], group_colors):
        patch.set_facecolor(gcolor)
        patch.set_alpha(0.35)
        patch.set_edgecolor(_C["ink"])
        patch.set_linewidth(0.6)
    for i, (errs, se) in enumerate(zip(data, session_err)):
        if not errs:
            continue
        gcolor = group_colors[i]
        xj = positions[i] + (np.linspace(-0.12, 0.12, len(errs)) if len(errs) > 1 else [0.0])
        ax_dist.scatter(
            xj,
            errs,
            s=28,
            c=[gcolor],
            alpha=0.85,
            edgecolors=_C["ink"],
            linewidths=0.3,
            zorder=4,
        )
        ax_dist.scatter(
            [positions[i]],
            [se],
            marker="P",
            s=70,
            c=[gcolor],
            edgecolors=_C["ink"],
            linewidths=0.5,
            zorder=5,
        )

    ax_dist.set_xticks(positions)
    ax_dist.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    ax_dist.set_ylabel("position error [cm]")
    ax_dist.set_title(f"(b) Window localization errors ({SEGMENT_LEN} rounds × {n_seg})")
    ax_dist.grid(True, axis="y", alpha=0.35)
    ax_dist.legend(
        handles=[
            Line2D([0], [0], marker="o", color="0.45", ls="None", ms=6, label="Window estimate"),
            Line2D([0], [0], marker="P", color=_C["ink"], ls="None", ms=8, label="Full-session estimate"),
        ],
        loc="upper right",
        fontsize=7,
    )

    n_pool = max((row["n_rounds_aoa"] for row in rows), default=0)
    fig.suptitle(
        f"Structure {STRUCTURE_ID}: consecutive windows from one capture  "
        f"({SEGMENT_N} random starts × {SEGMENT_LEN} rounds / ~{n_pool}; seed={SEGMENT_SEED})",
        fontsize=10,
        y=0.98,
    )
    _show(fig, "1 — Experiment summary")


def fig2_aoa_mean_by_position(results: List[Dict[str, Any]]) -> None:
    """Spatial residual fields for X-ULA and Y-ULA AoA means."""
    rows: List[Dict[str, Any]] = []
    for r in results:
        if r["est_pos"] is None:
            continue
        rows.append(
            {
                "label": _target_label(r["true_pos"]),
                "xy": tuple(r["true_pos"]),
                "dx": _ang_diff_deg(r["fuse_aoa_x_deg"], r["true_aoa_x_deg"]),
                "dy": _ang_diff_deg(r["fuse_aoa_y_deg"], r["true_aoa_y_deg"]),
            }
        )
    if not rows:
        return

    r0 = results[0]
    pts = list(r0["node_pos"].values()) + [r0["ref_pos"]] + [row["xy"] for row in rows]
    xmin, xmax, ymin, ymax = _axis_limits_equal(pts, pad_frac=0.22, min_span=0.5)

    abs_all = [abs(row["dx"]) for row in rows] + [abs(row["dy"]) for row in rows]
    vmax = max(abs_all + [1.0])
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.cm.RdBu_r  # type: ignore[attr-defined]

    fig, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(12.8, 6.2), sharex=True, sharey=True)

    def _panel(ax: plt.Axes, key: str, title: str, cbar_label: str) -> Any:
        _draw_anchors(ax, r0)
        xs = [row["xy"][0] for row in rows]
        ys = [row["xy"][1] for row in rows]
        vals = [row[key] for row in rows]
        sizes = [70.0 + 30.0 * abs(v) for v in vals]
        sc = ax.scatter(
            xs,
            ys,
            c=vals,
            s=sizes,
            cmap=cmap,
            norm=norm,
            edgecolors=_C["ink"],
            linewidths=0.55,
            zorder=5,
        )
        stalk = 0.012
        for row, v in zip(rows, vals):
            x0, y0 = row["xy"]
            ax.plot([x0, x0], [y0, y0 + stalk * v], color=_C["ink"], lw=0.8, alpha=0.45, zorder=4)
            ax.annotate(
                f"{v:+.1f}°",
                (x0, y0 + stalk * v),
                textcoords="offset points",
                xytext=(0, 5 if v >= 0 else -10),
                ha="center",
                fontsize=7,
                color=_C["ink"],
                zorder=6,
            )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(title)
        ax.grid(True, alpha=0.35)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label(cbar_label)
        return sc

    _panel(ax_x, "dx", "(a) X-ULA AoA mean residual", "Δθ_X [deg]")
    _panel(ax_y, "dy", "(b) Y-ULA AoA mean residual", "Δθ_Y [deg]")
    fig.suptitle("Fused AoA mean residual by target position", fontsize=12, y=0.98)
    _show(fig, "2 — X/Y AoA residual maps")


def fig3_position_error_cdf(results: List[Dict[str, Any]]) -> None:
    """CDF of bootstrap-constructed samples vs session angle-mean fuse."""
    if any(r.get("bootstrap_pos") is None for r in results):
        _attach_bootstrap_samples(results)

    boot_errs: List[float] = []
    mean_errs: List[float] = []
    mean_labels: List[str] = []

    for r in results:
        boot_errs.extend(float(e) for e in (r.get("bootstrap_err_cm") or []) if math.isfinite(float(e)))
        base = r.get("base_pos")
        if base is not None:
            mean_errs.append(cip.dist(base, r["true_pos"]) * 100.0)
            mean_labels.append(_target_label(r["true_pos"]))

    boot_a = np.asarray(boot_errs, dtype=np.float64)
    mean_a = np.asarray(mean_errs, dtype=np.float64)
    if boot_a.size == 0 and mean_a.size == 0:
        return

    fig, ax = plt.subplots(figsize=(7.4, 5.2))

    def _cdf_curve(errs: np.ndarray, *, color: str, label: str, marker: str, ms: float, lw: float) -> None:
        e = np.sort(errs[np.isfinite(errs)])
        if e.size == 0:
            return
        y = np.arange(1, e.size + 1, dtype=np.float64) / float(e.size)
        ax.fill_between(e, y, step="post", color=color, alpha=0.12, linewidth=0)
        ax.step(e, y, where="post", color=color, lw=lw, label=label)
        # avoid overplotting markers on large bootstrap N
        if e.size <= 40:
            ax.plot(e, y, marker, color=color, ms=ms, linestyle="None", markeredgecolor=_C["ink"], markeredgewidth=0.25)
        else:
            step = max(1, e.size // 25)
            ax.plot(e[::step], y[::step], marker, color=color, ms=ms, linestyle="None", markeredgecolor=_C["ink"], markeredgewidth=0.25)

    _cdf_curve(
        boot_a,
        color=_C["cyan"],
        label=f"Bootstrap samples  (N={np.sum(np.isfinite(boot_a))})",
        marker="o",
        ms=3.5,
        lw=1.6,
    )
    _cdf_curve(
        mean_a,
        color=_C["blue"],
        label=f"Angle-mean fuse  (N={np.sum(np.isfinite(mean_a))})",
        marker="D",
        ms=6.5,
        lw=2.2,
    )

    if mean_a.size:
        order = np.argsort(mean_a)
        e_sorted = mean_a[order]
        y_sorted = np.arange(1, e_sorted.size + 1) / float(e_sorted.size)
        labs = [mean_labels[i] for i in order]
        for lab, e, y in zip(labs, e_sorted, y_sorted):
            ax.annotate(lab, (e, y), textcoords="offset points", xytext=(5, -1), fontsize=6.5, color="0.3")

    if mean_a.size:
        ax.axvline(float(np.median(mean_a)), color=_C["red"], ls="--", lw=1.1, label=f"angle-mean P50={np.median(mean_a):.1f} cm")
    if boot_a.size:
        ax.axvline(float(np.median(boot_a)), color=_C["purple"], ls=":", lw=1.2, label=f"bootstrap P50={np.median(boot_a):.1f} cm")

    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("position error [cm]")
    ax.set_ylabel("CDF")
    ax.set_title(
        f"Position CDF: bootstrap "
        f"(each = mean of {BOOTSTRAP_N_DRAW} round AoAs × {BOOTSTRAP_N_OBS}/target) "
        f"vs angle-mean"
    )
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower right", fontsize=8)
    _show(fig, "3 — Position error CDF")


def fig4_error_vs_n_draw(results: List[Dict[str, Any]]) -> None:
    """
    Within-session resampling study (NOT independent experiments):

    At each n_draw, build n_obs subsample-mean localizations from the same
    session's round AoAs; record their mean error per target. Plot the
    across-target mean ± std vs n_draw (faint per-target curves in back).

    Interprets estimator stability / diminishing returns of averaging more
    rounds inside one capture — not cross-session reproducibility.
    """
    n_obs = 10
    draw_grid = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50]
    tab = plt.cm.tab10  # type: ignore[attr-defined]

    usable = [r for r in results if r.get("est_pos") is not None]
    per_target: List[List[float]] = []
    xs_final: List[int] = []

    for i, r in enumerate(usable):
        rng = np.random.default_rng(BOOTSTRAP_SEED + i)
        n_pool = len(r.get("round_aoa_x") or [])
        row_ys: List[float] = []
        row_xs: List[int] = []
        for n_draw in draw_grid:
            if n_draw > max(n_pool, 1):
                break
            samples = _bootstrap_mean_aoa_positions(
                r,
                n_obs=n_obs,
                n_draw=n_draw,
                rng=rng,
                replace=True,
            )
            if not samples:
                row_ys.append(float("nan"))
            else:
                errs = [cip.dist(p, r["true_pos"]) * 100.0 for p in samples]
                row_ys.append(float(np.mean(errs)))
            row_xs.append(n_draw)
        if not row_xs:
            continue
        xs_final = row_xs
        per_target.append(row_ys)

    if not per_target:
        return

    L = min(len(row) for row in per_target)
    xs_arr = np.asarray(xs_final[:L], dtype=np.float64)
    mat = np.asarray([row[:L] for row in per_target], dtype=np.float64)
    mean_y = np.nanmean(mat, axis=0)
    std_y = np.nanstd(mat, axis=0)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    # faint individual targets
    for i, row in enumerate(mat):
        ax.plot(
            xs_arr,
            row,
            "-",
            color=tab(i % 10),
            lw=0.9,
            alpha=0.25,
            zorder=2,
        )
    ax.fill_between(
        xs_arr,
        mean_y - std_y,
        mean_y + std_y,
        color=_C["blue"],
        alpha=0.18,
        linewidth=0,
        label="across-target ±1σ",
        zorder=3,
    )
    ax.plot(
        xs_arr,
        mean_y,
        "-o",
        color=_C["blue"],
        ms=6,
        lw=2.2,
        markeredgecolor=_C["ink"],
        markeredgewidth=0.35,
        label="across-target mean",
        zorder=4,
    )

    # annotate a few operating points
    for nd in (1, 10, 50):
        if nd in xs_arr:
            j = int(np.where(xs_arr == nd)[0][0])
            ax.annotate(
                f"n={nd}: {mean_y[j]:.0f} cm",
                (xs_arr[j], mean_y[j]),
                textcoords="offset points",
                xytext=(8, 8 if nd != 50 else -14),
                fontsize=8,
                color=_C["ink"],
            )

    ax.set_xlabel("rounds averaged into one localization  (n_draw)")
    ax.set_ylabel(f"mean position error of {n_obs} within-session resamples [cm]")
    ax.set_title(
        "Within-session stability vs #rounds averaged\n"
        "(resampling existing rounds — not independent experiments)"
    )
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(left=0.5)
    note = (
        "Each point: draw n_draw rounds from one capture, mean AoA → position; "
        f"repeat {n_obs}×; then average over targets."
    )
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=7, color="0.35", va="bottom")
    _show(fig, "4 — Error vs n_draw")


def visualize_all(results: List[Dict[str, Any]]) -> None:
    _apply_plot_style()
    _attach_segment_samples(results)
    _attach_bootstrap_samples(results)
    fig1_experiment_summary(results)
    fig2_aoa_mean_by_position(results)
    fig3_position_error_cdf(results)
    fig4_error_vs_n_draw(results)


def main(argv: Optional[Sequence[str]] = None) -> int:
    data_dir = argv[0] if argv else (sys.argv[1] if len(sys.argv) > 1 else INDOOR_CORAL_DIR)
    if not os.path.isdir(data_dir):
        print(f"Not a directory: {data_dir}", file=sys.stderr)
        return 1

    npz_files = find_structure_sessions(data_dir, STRUCTURE_ID)
    if not npz_files:
        print(f"No structure_id={STRUCTURE_ID} sessions under {data_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(npz_files)} session(s) with structure_id={STRUCTURE_ID}")
    results: List[Dict[str, Any]] = []
    for path in npz_files:
        name = os.path.basename(path)
        print(f"---- {name} ----")
        try:
            res = process_one_session(path)
        except Exception as exc:
            print(f"  failed: {exc}", file=sys.stderr)
            continue
        if res is None:
            print("  skip: no pair estimates")
            continue
        results.append(res)
        if res["est_pos"] is None:
            print(f"  TARGET={res['true_pos']}  est=None  mode={res.get('loc_mode')}")
        else:
            print(
                f"  TARGET=({res['true_pos'][0]:.3f},{res['true_pos'][1]:.3f})  "
                f"est=({res['est_pos'][0]:.3f},{res['est_pos'][1]:.3f})  "
                f"err={res['pos_err_m']*100:.1f} cm  "
                f"AoA X={res['fuse_aoa_x_deg']:+.1f}/{res['true_aoa_x_deg']:+.1f}  "
                f"Y={res['fuse_aoa_y_deg']:+.1f}/{res['true_aoa_y_deg']:+.1f}  "
                f"mode={res.get('loc_mode')}  "
                f"inliers={res['n_round_pos']-res['n_outliers']}/{res['n_round_pos']}"
            )

    if not results:
        print("No sessions processed successfully.", file=sys.stderr)
        return 1

    errs = [
        r["pos_err_m"] * 100.0
        for r in results
        if r["est_pos"] is not None and math.isfinite(r["pos_err_m"])
    ]
    if errs:
        print(f"---- overview ----  N={len(errs)}  mean={np.mean(errs):.1f} cm  median={np.median(errs):.1f} cm")

    visualize_all(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
