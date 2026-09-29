#!/usr/bin/env python3
"""
Offline processing for new_listener.py captures under indoor_coral/.

Input (one capture session):
  indoor_coral/{ref}_{tgt}_{structure}_reports.npz
  indoor_coral/{ref}_{tgt}_{structure}_meta.json

Pipeline per round:
  1) per-anchor cte_ref/cte_echo -> div_iq (spatial observable)
  2) for each anchor pair: slope = mean(
         LS slope of angle(cte_ref_a / cte_ref_b),
         LS slope of angle(cte_echo_a / cte_echo_b)
       )  used for CFO correction
  3) ESPRIT on corrected array snapshots (no half-plane prior)
  4) pair-fusion AoA with half-plane prior from target side of the ULA baseline
"""

from __future__ import annotations

import glob
import itertools
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Missing dependency: matplotlib  (pip install matplotlib)", file=sys.stderr)
    sys.exit(1)

import CoRal_listener as cl
from CoRal_listener import (
    C_LIGHT,
    F_CENTER_HZ,
    SAMPLE_INTERVAL_S,
    aligned_div_complex,
    compute_div_iq_and_ls,
    dist,
    div_iq_to_complex,
    esprit_aoa_deg,
    ref_to_anchor_phase,
    td_ticks_to_seconds,
    ula_element_spacing_m,
)

REPORT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indoor_coral")

# Skip π half-plane flip when |θ_raw| is below this (deg).
HALF_PLANE_DEADZONE_DEG = 12.0


# ---------------------------------------------------------------------------
# Session I/O
# ---------------------------------------------------------------------------

def meta_path_for_npz(npz_path: str) -> str:
    if npz_path.endswith("_reports.npz"):
        return npz_path[: -len("_reports.npz")] + "_meta.json"
    return npz_path + ".meta.json"


def load_meta(npz_path: str) -> Dict[str, Any]:
    path = meta_path_for_npz(npz_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing meta json next to npz: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_session_geometry(meta: Dict[str, Any]) -> Tuple[List[int], int, List[int]]:
    """
    Push capture geometry into CoRal_listener globals used by snapshot / phase helpers.
    Returns (anchor_list, array_ref_id, array_other_ids).
    """
    ref_pos = meta.get("ref_position") or meta.get("ref_pos")
    if ref_pos is None and "ref_position" in meta:
        ref_pos = meta["ref_position"]
    if ref_pos is None:
        raise KeyError("meta missing ref_position")
    cl.REF_POS = (float(ref_pos[0]), float(ref_pos[1]))

    structure = meta.get("anchor_structure") or {}
    raw_pos = structure.get("positions") or {}
    node_pos: Dict[int, Tuple[float, float]] = {
        int(k): (float(v[0]), float(v[1])) for k, v in raw_pos.items()
    }
    if not node_pos:
        raise KeyError("meta.anchor_structure.positions is empty")
    cl.NODE_POS = node_pos

    anchors = [int(a) for a in meta.get("anchor_list", sorted(node_pos.keys()))]
    # Array REF = leftmost / origin-most anchor (smallest x, then y).
    array_ref = min(
        anchors,
        key=lambda aid: (node_pos[aid][0], node_pos[aid][1], aid),
    )
    others = sorted(
        [a for a in anchors if a != array_ref],
        key=lambda aid: (node_pos[aid][0], node_pos[aid][1], aid),
    )
    cl.ARRAY_REF_ID = array_ref
    cl.ARRAY_OTHER_IDS = others
    cl.ANCHOR_LIST = list(anchors)
    return anchors, array_ref, others


def _iq_rows_to_pairs(arr: np.ndarray) -> List[Tuple[float, float]]:
    """(N,2) float array → [(I,Q),...], dropping NaN rows."""
    if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] == 0:
        return []
    out: List[Tuple[float, float]] = []
    for row in arr:
        if not (np.isfinite(row[0]) and np.isfinite(row[1])):
            continue
        out.append((float(row[0]), float(row[1])))
    return out


def load_session(npz_path: str) -> Tuple[Dict[str, Any], np.lib.npyio.NpzFile]:
    meta = load_meta(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    return meta, data


def n_rounds_in_npz(data: np.lib.npyio.NpzFile, anchors: List[int]) -> int:
    if "n_rounds" in data.files:
        return int(np.asarray(data["n_rounds"]).item())
    # Infer from first available CTE stack.
    for aid in anchors:
        key = f"id{aid}_cte_ref"
        if key in data.files:
            return int(np.asarray(data[key]).shape[0])
    raise ValueError("cannot determine n_rounds from npz")


def reports_for_round(
    data: np.lib.npyio.NpzFile,
    anchors: List[int],
    round_idx: int,
) -> Dict[int, Dict[str, Any]]:
    """
    Build one CoRal_listener-style reports dict for round_idx (0-based).

    Computes div_iq / slope / intercept from cte_ref & cte_echo.
    """
    reports: Dict[int, Dict[str, Any]] = {}
    for aid in anchors:
        prefix = f"id{aid}_"
        if prefix + "cte_ref" not in data.files or prefix + "cte_echo" not in data.files:
            continue

        cte_ref = _iq_rows_to_pairs(np.asarray(data[prefix + "cte_ref"][round_idx]))
        cte_echo = _iq_rows_to_pairs(np.asarray(data[prefix + "cte_echo"][round_idx]))
        if not cte_ref or not cte_echo:
            continue

        div_iq, intercept, slope = compute_div_iq_and_ls(cte_ref, cte_echo)
        if not div_iq:
            continue

        def _scalar(name: str, default: int = 0) -> int:
            key = prefix + name
            if key not in data.files:
                return default
            return int(np.asarray(data[key][round_idx]).item())

        reports[aid] = {
            "id": aid,
            "slot": _scalar("slot"),
            "x": _scalar("x"),
            "y": _scalar("y"),
            "time_diff": _scalar("time_diff"),
            "rssi": _scalar("rssi"),
            "nib": _scalar("nib", len(cte_ref)),
            "necho": _scalar("necho", len(cte_echo)),
            "frags": _scalar("frags"),
            "cte_ref": cte_ref,
            "cte_echo": cte_echo,
            "div_iq": div_iq,
            "intercept": float(intercept),
            "slope": float(slope),
            "ndiv": len(div_iq),
        }
    return reports


# ---------------------------------------------------------------------------
# Geometry / true AoA
# ---------------------------------------------------------------------------

def node_xy(node_id: int) -> Tuple[float, float]:
    return cl.node_xy(node_id)


def array_side_sign(target_xy: Tuple[float, float]) -> float:
    pos_ref = node_xy(cl.ARRAY_REF_ID)
    pos_end = node_xy(cl.ARRAY_OTHER_IDS[-1])
    bx = pos_end[0] - pos_ref[0]
    by = pos_end[1] - pos_ref[1]
    mid_x = 0.5 * (pos_ref[0] + pos_end[0])
    mid_y = 0.5 * (pos_ref[1] + pos_end[1])
    cross = bx * (target_xy[1] - mid_y) - by * (target_xy[0] - mid_x)
    if abs(cross) < 1e-15:
        return 0.0
    return 1.0 if cross > 0.0 else -1.0


def true_aoa_deg_from_target(target_xy: Tuple[float, float]) -> float:
    """
    Path-diff θ in ESPRIT convention; flip sign when target is on the CW
    side of the ULA baseline (front/back π ambiguity).
    """
    pos_ref = node_xy(cl.ARRAY_REF_ID)
    pos_end = node_xy(cl.ARRAY_OTHER_IDS[-1])
    baseline = dist(pos_end, pos_ref)
    if baseline < 1e-12:
        return float("nan")
    path_diff = dist(target_xy, pos_end) - dist(target_xy, pos_ref)
    sin_th = float(np.clip(-path_diff / baseline, -1.0, 1.0))
    theta_path = float(np.degrees(np.arcsin(sin_th)))
    if array_side_sign(target_xy) < 0.0:
        return -theta_path
    return theta_path


# ---------------------------------------------------------------------------
# Pair-fusion (+ half-plane prior)
# ---------------------------------------------------------------------------

def wrap_phase(phase_rad: float) -> float:
    return float(np.arctan2(np.sin(phase_rad), np.cos(phase_rad)))


def aoa_from_pair_phase(phase_rad: float, baseline_m: float) -> float:
    if baseline_m < 1e-12:
        return float("nan")
    sin_th = float(
        np.clip(
            -phase_rad * C_LIGHT / (2.0 * math.pi * F_CENTER_HZ * baseline_m),
            -1.0,
            1.0,
        )
    )
    return float(np.degrees(np.arcsin(sin_th)))


def aoa_from_pair_phase_half(
    phase_rad: float,
    baseline_m: float,
    prefer_positive: bool,
    deadzone_deg: float = HALF_PLANE_DEADZONE_DEG,
) -> Dict[str, Any]:
    """
    Half-plane prior from target true position side of the ULA.

    prefer_positive True  → force θ > 0
    prefer_positive False → force θ < 0

    If |θ_raw| < deadzone_deg, keep raw (do not flip).
    """
    th_raw = aoa_from_pair_phase(phase_rad, baseline_m)
    if not math.isfinite(th_raw):
        return {
            "aoa_raw_deg": float("nan"),
            "aoa_deg": float("nan"),
            "pi_corrected": False,
        }

    if abs(th_raw) < deadzone_deg:
        return {"aoa_raw_deg": th_raw, "aoa_deg": th_raw, "pi_corrected": False}

    on_preferred = th_raw > 0.0 if prefer_positive else th_raw < 0.0
    if on_preferred:
        return {"aoa_raw_deg": th_raw, "aoa_deg": th_raw, "pi_corrected": False}

    phase_fix = wrap_phase(phase_rad + math.pi)
    th_fix = aoa_from_pair_phase(phase_fix, baseline_m)
    if not math.isfinite(th_fix):
        return {"aoa_raw_deg": th_raw, "aoa_deg": th_raw, "pi_corrected": False}

    fix_ok = th_fix > 0.0 if prefer_positive else th_fix < 0.0
    closer = th_fix > th_raw if prefer_positive else th_fix < th_raw
    if fix_ok or closer:
        return {"aoa_raw_deg": th_raw, "aoa_deg": th_fix, "pi_corrected": True}
    return {"aoa_raw_deg": th_raw, "aoa_deg": th_raw, "pi_corrected": False}


def pair_slope_from_cte(
    rpt_a: Dict[str, Any],
    rpt_b: Dict[str, Any],
) -> Dict[str, float]:
    """
    Differential CFO slope for pair (a, b).

    slope_ref  = LS slope of unwrapped angle(cte_ref_a  / cte_ref_b)
    slope_echo = LS slope of unwrapped angle(cte_echo_a / cte_echo_b)
    slope      = 0.5 * (slope_ref + slope_echo)   # the two should be close
    """
    _, _, slope_ref = compute_div_iq_and_ls(
        rpt_a.get("cte_ref", []),
        rpt_b.get("cte_ref", []),
    )
    _, _, slope_echo = compute_div_iq_and_ls(
        rpt_a.get("cte_echo", []),
        rpt_b.get("cte_echo", []),
    )
    return {
        "slope_ref": float(slope_ref),
        "slope_echo": float(slope_echo),
        "slope": 0.5 * (float(slope_ref) + float(slope_echo)),
        "slope_diff": float(slope_ref) - float(slope_echo),
    }


def corrected_pair_ratio(
    reports: Dict[int, Dict[str, Any]],
    id_a: int,
    id_b: int,
) -> Optional[np.ndarray]:
    """
    Spatial ratio div_a/div_b after CFO + REF geometry correction.

    CFO uses pair CTE slopes:
      slope = mean( LS(ref_a/ref_b), LS(echo_a/echo_b) )
      phi_cfo = slope * (td / T_sample)
    """
    rpt_a = reports.get(id_a)
    rpt_b = reports.get(id_b)
    if rpt_a is None or rpt_b is None:
        return None
    ca = div_iq_to_complex(rpt_a.get("div_iq", []))
    cb = div_iq_to_complex(rpt_b.get("div_iq", []))
    ratio = aligned_div_complex(ca, cb)
    if ratio is None:
        return None
    finite = np.isfinite(ratio.real) & np.isfinite(ratio.imag)
    if not np.any(finite):
        return None
    ratio = np.where(finite, ratio, 0.0)

    slopes = pair_slope_from_cte(rpt_a, rpt_b)
    td_ticks = 0.5 * (float(rpt_a["time_diff"]) + float(rpt_b["time_diff"]))
    phi_cfo = slopes["slope"] * (td_ticks_to_seconds(td_ticks) / SAMPLE_INTERVAL_S)
    phi_m = ref_to_anchor_phase(id_a) - ref_to_anchor_phase(id_b)
    return ratio * np.exp(-1j * phi_cfo) * np.exp(-1j * phi_m)


def build_array_snapshots_pair_slope(
    reports: Dict[int, Dict[str, Any]],
    *,
    verbose: bool = False,
) -> Optional[np.ndarray]:
    """
    Same M x N ESPRIT snapshots as CoRal_listener.build_array_snapshots, but
    CFO correction uses pair_slope_from_cte(ref, other) instead of
    (slope_ref_anchor - slope_other_anchor).
    """
    ref_id = cl.ARRAY_REF_ID
    if ref_id not in reports:
        return None
    rpt_ref = reports[ref_id]
    ref_c = div_iq_to_complex(rpt_ref.get("div_iq", []))
    if len(ref_c) == 0:
        return None

    rows: List[np.ndarray] = []
    n_common: Optional[int] = None
    for oid in cl.ARRAY_OTHER_IDS:
        rpt = reports.get(oid)
        if rpt is None:
            return None
        other_c = div_iq_to_complex(rpt.get("div_iq", []))
        ratio = aligned_div_complex(ref_c, other_c)
        if ratio is None:
            return None
        finite = np.isfinite(ratio.real) & np.isfinite(ratio.imag)
        if not np.any(finite):
            return None
        ratio = np.where(finite, ratio, 0.0)

        slopes = pair_slope_from_cte(rpt_ref, rpt)
        td_ticks = 0.5 * (float(rpt_ref["time_diff"]) + float(rpt["time_diff"]))
        phi_cfo = slopes["slope"] * (td_ticks_to_seconds(td_ticks) / SAMPLE_INTERVAL_S)
        phi_m = ref_to_anchor_phase(oid) - ref_to_anchor_phase(ref_id)
        corrected = ratio * np.exp(-1j * phi_cfo) * np.exp(-1j * phi_m)

        if verbose:
            print(
                f"  corr {ref_id}/{oid}: "
                f"slope_ref={slopes['slope_ref']:.6g}  "
                f"slope_echo={slopes['slope_echo']:.6g}  "
                f"slope={slopes['slope']:.6g}  "
                f"dSlope={slopes['slope_diff']:.6g}  "
                f"phi_cfo={phi_cfo:+.6g}  "
                f"angM={phi_m:+.6g}"
            )

        rows.append(corrected)
        n_common = len(corrected) if n_common is None else min(n_common, len(corrected))

    assert n_common is not None and n_common > 0
    x = np.vstack([row[:n_common] for row in rows]).astype(np.complex128)
    col_ok = np.any(np.abs(x) > 1e-12, axis=0)
    x = x[:, col_ok]
    if x.shape[1] < 2:
        return None
    return x


def _fuse_sin_weighted(aoas_deg: List[float], baselines_m: List[float]) -> float:
    if not aoas_deg:
        return float("nan")
    sins = [math.sin(math.radians(th)) for th in aoas_deg]
    w = np.asarray([b * b for b in baselines_m], dtype=np.float64)
    w /= np.sum(w)
    sin_fused = float(np.clip(np.dot(w, np.asarray(sins, dtype=np.float64)), -1.0, 1.0))
    return float(np.degrees(np.arcsin(sin_fused)))


def estimate_aoa_all_pairs(
    reports: Dict[int, Dict[str, Any]],
    anchors: List[int],
    prefer_positive: bool,
) -> Dict[str, Any]:
    aoas_raw: List[float] = []
    aoas_fix: List[float] = []
    baselines_list: List[float] = []
    n_pi = 0

    for id_a, id_b in itertools.combinations(anchors, 2):
        pos_a = node_xy(id_a)
        pos_b = node_xy(id_b)
        if pos_a[0] < pos_b[0] or (pos_a[0] == pos_b[0] and pos_a[1] < pos_b[1]):
            id_a, id_b = id_b, id_a
            pos_a, pos_b = pos_b, pos_a

        baseline = dist(pos_a, pos_b)
        if baseline < 1e-9:
            continue
        ratio = corrected_pair_ratio(reports, id_a, id_b)
        if ratio is None:
            continue
        phase = float(np.angle(np.mean(ratio)))
        est = aoa_from_pair_phase_half(phase, baseline, prefer_positive)
        if not math.isfinite(est["aoa_raw_deg"]):
            continue
        aoas_raw.append(float(est["aoa_raw_deg"]))
        aoas_fix.append(float(est["aoa_deg"]))
        baselines_list.append(baseline)
        if est["pi_corrected"]:
            n_pi += 1

    if not baselines_list:
        return {
            "aoa_deg": float("nan"),
            "aoa_raw_deg": float("nan"),
            "n_pi_corrected": 0,
        }
    return {
        "aoa_deg": _fuse_sin_weighted(aoas_fix, baselines_list),
        "aoa_raw_deg": _fuse_sin_weighted(aoas_raw, baselines_list),
        "n_pi_corrected": n_pi,
    }


# ---------------------------------------------------------------------------
# Per-round / batch
# ---------------------------------------------------------------------------

def process_round(
    reports: Dict[int, Dict[str, Any]],
    anchors: List[int],
    d_sp: float,
    prefer_positive: bool,
    round_idx: int,
) -> Optional[Dict[str, Any]]:
    missing = [aid for aid in anchors if aid not in reports]
    if missing:
        print(f"  skip round {round_idx:03d}: missing anchors {missing}")
        return None

    x = build_array_snapshots_pair_slope(reports, verbose=False)
    if x is None:
        print(f"  skip round {round_idx:03d}: failed to build snapshots")
        return None

    try:
        theta_esprit = esprit_aoa_deg(x, d_sp, num_sources=1)
    except Exception as exc:
        print(f"  skip round {round_idx:03d}: ESPRIT error: {exc}")
        return None

    pair_est = estimate_aoa_all_pairs(reports, anchors, prefer_positive)
    return {
        "round_idx": round_idx,
        "n_snapshots": int(x.shape[1]),
        "aoa_deg": float(theta_esprit),
        "aoa_pair_deg": float(pair_est["aoa_deg"]),
        "aoa_pair_raw_deg": float(pair_est["aoa_raw_deg"]),
        "n_pi_corrected": int(pair_est["n_pi_corrected"]),
    }


def summarize(results: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    aoa = np.asarray([r[key] for r in results], dtype=np.float64)
    aoa = aoa[np.isfinite(aoa)]
    if len(aoa) == 0:
        return {
            "n": 0.0,
            "aoa_mean": float("nan"),
            "aoa_std": float("nan"),
            "aoa_min": float("nan"),
            "aoa_max": float("nan"),
            "aoa_median": float("nan"),
        }
    return {
        "n": float(len(aoa)),
        "aoa_mean": float(np.mean(aoa)),
        "aoa_std": float(np.std(aoa)),
        "aoa_min": float(np.min(aoa)),
        "aoa_max": float(np.max(aoa)),
        "aoa_median": float(np.median(aoa)),
    }


def _half_label(prefer_positive: bool) -> str:
    return "right (theta>0)" if prefer_positive else "left (theta<0)"


def visualize(
    results: List[Dict[str, Any]],
    stats_e: Dict[str, float],
    stats_p: Dict[str, float],
    stats_p_raw: Dict[str, float],
    true_aoa: float,
    prefer_positive: bool,
    session_name: str,
) -> None:
    half = _half_label(prefer_positive)
    aoa_e = np.asarray([r["aoa_deg"] for r in results], dtype=np.float64)
    aoa_p = np.asarray([r["aoa_pair_deg"] for r in results], dtype=np.float64)
    aoa_pr = np.asarray([r["aoa_pair_raw_deg"] for r in results], dtype=np.float64)
    rounds = np.asarray([r["round_idx"] for r in results], dtype=np.int32)
    n_bins = min(20, max(5, len(results) // 2))
    bin_edges = np.linspace(-90.0, 90.0, n_bins + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.hist(aoa_e, bins=bin_edges, color="C0", edgecolor="k", alpha=0.85, label="ESPRIT")
    ax.axvline(stats_e["aoa_mean"], color="C1", ls="--", lw=1.5,
               label=f"mean {stats_e['aoa_mean']:+.2f} deg")
    ax.axvline(true_aoa, color="C3", ls="-", lw=2.0, label="Target")
    err_e = stats_e["aoa_mean"] - true_aoa
    ax.set_title(f"ESPRIT  std={stats_e['aoa_std']:.2f} deg  mean-true={err_e:+.2f} deg")
    ax.set_xlim(-95, 95)
    ax.set_xlabel("AoA [deg]")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.hist(aoa_p, bins=bin_edges, color="C2", edgecolor="k", alpha=0.85,
            label=f"pair ({half})")
    ax.axvline(stats_p["aoa_mean"], color="C1", ls="--", lw=1.5,
               label=f"mean {stats_p['aoa_mean']:+.2f} deg")
    ax.axvline(true_aoa, color="C3", ls="-", lw=2.0, label="Target")
    err_p = stats_p["aoa_mean"] - true_aoa
    ax.set_title(f"Pair-fusion ({half})  std={stats_p['aoa_std']:.2f} deg  mean-true={err_p:+.2f} deg")
    ax.set_xlim(-95, 95)
    ax.set_xlabel("AoA [deg]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.hist(aoa_pr, bins=bin_edges, color="C3", edgecolor="k", alpha=0.55, label="pair raw")
    ax.hist(aoa_p, bins=bin_edges, color="C2", edgecolor="k", alpha=0.55, label="pair +pi")
    ax.axvline(true_aoa, color="k", ls="-", lw=2.0, label="Target")
    ax.axvline(0.0, color="gray", ls=":", lw=1.0)
    ax.set_xlim(-95, 95)
    ax.set_xlabel("AoA [deg]")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(
        f"Pair raw->fix  std {stats_p_raw['aoa_std']:.1f}->{stats_p['aoa_std']:.1f} deg"
    )

    ax = axes[1, 1]
    ax.plot(rounds, aoa_e, "o-", ms=3, lw=1.0, color="C0", label="ESPRIT")
    ax.plot(rounds, aoa_p, "s-", ms=3, lw=1.0, color="C2", label="pair")
    ax.axhline(true_aoa, color="C3", ls="-", lw=1.5, label="Target")
    ax.axhline(0.0, color="gray", ls=":", lw=1.0)
    ax.set_ylim(-95, 95)
    ax.set_xlabel("round")
    ax.set_ylabel("AoA [deg]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Per-round AoA")

    fig.suptitle(
        f"{session_name}  N={len(results)}  "
        f"|err|_E={abs(err_e):.2f} deg  |err|_P={abs(err_p):.2f} deg"
    )
    fig.tight_layout()
    print("Close figure to exit.")
    plt.show(block=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_report_npzs(path_or_dir: str) -> List[str]:
    if os.path.isfile(path_or_dir) and path_or_dir.endswith(".npz"):
        return [path_or_dir]
    pattern = os.path.join(path_or_dir, "*_reports.npz")
    return sorted(glob.glob(pattern))


def process_session(npz_path: str) -> int:
    meta, data = load_session(npz_path)
    anchors, array_ref, others = apply_session_geometry(meta)

    tgt = meta.get("target_position") or meta.get("target_pos")
    if tgt is None:
        raise KeyError("meta missing target_position")
    target_pos = (float(tgt[0]), float(tgt[1]))

    true_aoa = true_aoa_deg_from_target(target_pos)
    prefer_positive = bool(true_aoa > 0.0)
    half = _half_label(prefer_positive)
    side = array_side_sign(target_pos)
    side_name = (
        "+normal (CCW/+y)" if side > 0 else ("-normal (CW/-y)" if side < 0 else "on baseline")
    )

    d_sp = ula_element_spacing_m()
    n_rounds = n_rounds_in_npz(data, anchors)
    session_name = os.path.basename(npz_path)

    print(f"==== {session_name} ====")
    print(f"  anchors={anchors}  array_ref={array_ref}  others={others}")
    print(f"  structure={meta.get('anchor_structure', {}).get('name', '?')}")
    print(f"  REF_POS={cl.REF_POS}  TARGET={target_pos}")
    print(f"  ULA d={d_sp*100:.3f} cm  n_rounds={n_rounds}")
    print(f"  array side={side_name}  true AoA={true_aoa:+.3f} deg  -> pair prior {half}")

    results: List[Dict[str, Any]] = []
    for r in range(n_rounds):
        reports = reports_for_round(data, anchors, r)
        res = process_round(reports, anchors, d_sp, prefer_positive, round_idx=r + 1)
        if res is None:
            continue
        results.append(res)
        print(
            f"  round {res['round_idx']:03d}: "
            f"ESPRIT={res['aoa_deg']:+.2f}  "
            f"pair={res['aoa_pair_deg']:+.2f} "
            f"(raw {res['aoa_pair_raw_deg']:+.2f}, +pi x{res['n_pi_corrected']})  "
            f"(E err={res['aoa_deg']-true_aoa:+.2f}, "
            f"P err={res['aoa_pair_deg']-true_aoa:+.2f})"
        )

    if not results:
        print("  No rounds processed successfully.", file=sys.stderr)
        return 1

    stats_e = summarize(results, "aoa_deg")
    stats_p = summarize(results, "aoa_pair_deg")
    stats_p_raw = summarize(results, "aoa_pair_raw_deg")
    print("---- summary ----")
    print(f"  N rounds        : {len(results)}")
    print(f"  pair half prior : {half}")
    print(
        f"  ESPRIT          : mean={stats_e['aoa_mean']:+.3f}  "
        f"std={stats_e['aoa_std']:.3f}  "
        f"mean-true={stats_e['aoa_mean']-true_aoa:+.3f}"
    )
    print(
        f"  Pair raw        : mean={stats_p_raw['aoa_mean']:+.3f}  "
        f"std={stats_p_raw['aoa_std']:.3f}  "
        f"mean-true={stats_p_raw['aoa_mean']-true_aoa:+.3f}"
    )
    print(
        f"  Pair +pi        : mean={stats_p['aoa_mean']:+.3f}  "
        f"std={stats_p['aoa_std']:.3f}  "
        f"mean-true={stats_p['aoa_mean']-true_aoa:+.3f}"
    )
    print(f"  true AoA        : {true_aoa:+.3f} deg")

    visualize(
        results, stats_e, stats_p, stats_p_raw, true_aoa, prefer_positive, session_name
    )
    return 0


def main() -> int:
    path = REPORT_DATA_DIR
    if len(sys.argv) > 1:
        path = sys.argv[1]

    npz_files = find_report_npzs(path)
    if not npz_files:
        print(f"No *_reports.npz under {path}", file=sys.stderr)
        return 1

    print(f"Found {len(npz_files)} session(s)")
    rc = 0
    for npz_path in npz_files:
        try:
            rc = process_session(npz_path) or rc
        except Exception as exc:
            print(f"ERROR processing {npz_path}: {exc}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
