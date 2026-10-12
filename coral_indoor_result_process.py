#!/usr/bin/env python3
"""
Process one indoor_coral capture session (new_listener.py format).

Per-pair AoA without requiring known TARGET coordinates for calibration:

  corrected_pair_ratio → mean phase → optional half-plane

  Half-plane prior is only a left/right side bit (as from RSSI in the field).
  Offline we infer that bit from raw pair AoA votes; if the vote is weak
  (near broadside / ambiguous), half-plane is skipped.
  ULA element order is fixed (hardware axis), never flipped toward TARGET.

  TARGET in meta is used only for evaluation (true AoA / position error).

Usage:
  python coral_indoor_result_process.py indoor_coral/(0.00,0.00)_(0.50,0.00)_1_reports.npz
  python coral_indoor_result_process.py   # uses DEFAULT_NPZ below
"""

from __future__ import annotations

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

from CoRal_listener import compute_div_iq_and_ls

INDOOR_CORAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indoor_coral_2")

DEFAULT_NPZ = os.path.join(
    INDOOR_CORAL_DIR,
    "(0.00,0.00)_(1.00,1.00)_3_reports.npz",
)

# RF / timing (same as new_listener.py).
F_CENTER_HZ = 2.440e9
C_LIGHT = 2.99792458e8
TIMER_HZ = 16e6
SAMPLE_INTERVAL_S = 1e-6

# Half-plane without ground-truth angle magnitude.
# Side prior is a single left/right bit (RSSI in the field; offline mock below).
HALF_PLANE_DEADZONE_DEG = 12.0
# Skip half-plane when most raw samples sit near 0 (likely broadside / ambiguous).
BROADSIDE_RAW_FRAC = 0.55
BROADSIDE_RAW_DEG = 12.0
# Offline RSSI-mock: if |trueULA| below this, treat side cue as too weak (same
# situation as ambiguous RSSI near the array normal in the field).
SIDE_CONFIDENCE_DEG = 12.0

# Multi-baseline overlap disambiguation when d > λ/2 (spatial aliasing).
ENABLE_OVERLAP_DISAMBIG = True
OVERLAP_SIN_MATCH_TOL = 0.04  # |Δsinθ| to count as the same branch across pairs


def wavelength_m() -> float:
    return C_LIGHT / F_CENTER_HZ


def baseline_aliased(baseline_m: float) -> bool:
    """True if baseline exceeds λ/2 (pair AoA from wrapped phase is ambiguous)."""
    return baseline_m > 0.5 * wavelength_m() + 1e-12


def group_has_aliased_baseline(
    ids: List[int],
    node_pos: Dict[int, Tuple[float, float]],
) -> bool:
    for id_a, id_b in itertools.combinations(ids, 2):
        if baseline_aliased(dist(node_pos[id_a], node_pos[id_b])):
            return True
    return False


def sin_candidates_from_phase(
    phase_rad: float,
    baseline_m: float,
) -> List[float]:
    """
    All sinθ consistent with wrapped phase on baseline d:

      φ_meas = wrap(-2π d sinθ / λ)
      sinθ = -φ_meas·λ/(2πd) - n·(λ/d),  |sinθ| ≤ 1
    """
    if baseline_m < 1e-12:
        return []
    lam = wavelength_m()
    s0 = -float(phase_rad) * lam / (2.0 * math.pi * baseline_m)
    step = lam / baseline_m
    if step < 1e-15:
        return []
    # n such that |s0 - n·step| ≤ 1
    n_lo = int(math.floor((s0 - 1.0) / step)) - 2
    n_hi = int(math.ceil((s0 + 1.0) / step)) + 2
    out: List[float] = []
    for n in range(n_lo, n_hi + 1):
        s = s0 - float(n) * step
        if abs(s) <= 1.0 + 1e-9:
            out.append(float(np.clip(s, -1.0, 1.0)))
    # Deduplicate near-copies
    out.sort()
    uniq: List[float] = []
    for s in out:
        if not uniq or abs(s - uniq[-1]) > 1e-9:
            uniq.append(s)
    return uniq


def predicted_pair_phase(sin_theta: float, baseline_m: float) -> float:
    """φ = -2π d sinθ / λ (wrapped)."""
    lam = wavelength_m()
    return wrap_phase(
        -2.0 * math.pi * baseline_m * float(sin_theta) / lam
    )


def score_sin_against_pairs(
    sin_theta: float,
    pair_obs: List[Tuple[float, float, float]],
) -> float:
    """
    Robust multi-pair phase agreement for candidate sinθ.

    Per pair: w·cos(Δφ). Drop the worst pair (outlier), then sum — so one
    inconsistent short baseline cannot dominate overlap voting.
    """
    if not pair_obs:
        return float("-inf")
    terms: List[float] = []
    for phase, baseline, weight in pair_obs:
        pred = predicted_pair_phase(sin_theta, baseline)
        terms.append(float(weight) * math.cos(float(phase) - pred))
    if len(terms) >= 3:
        terms.remove(min(terms))
    return float(sum(terms))


def disambiguate_sin_by_overlap(
    pair_obs: List[Tuple[float, float, float]],
    prefer_positive: Optional[bool] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Pick sinθ by multi-baseline overlap: generate branch candidates from each
    pair, score every candidate by phase agreement on all pairs, keep the best.

    pair_obs entries: (phase_rad, baseline_m, weight).
    Returns (sinθ, info).
    """
    if not pair_obs:
        return float("nan"), {"n_candidates": 0, "score": float("nan")}

    candidates: List[float] = []
    for phase, baseline, _w in pair_obs:
        candidates.extend(sin_candidates_from_phase(phase, baseline))
    if not candidates:
        return float("nan"), {"n_candidates": 0, "score": float("nan")}

    # Cluster near-duplicates
    candidates.sort()
    seeds: List[float] = []
    for s in candidates:
        if not seeds or abs(s - seeds[-1]) > OVERLAP_SIN_MATCH_TOL * 0.5:
            seeds.append(s)

    best_s = float("nan")
    best_score = -1e300
    for s in seeds:
        if prefer_positive is not None:
            # Soft half-plane on θ = arcsin(s): drop clearly wrong half.
            th = float(np.degrees(np.arcsin(np.clip(s, -1.0, 1.0))))
            if abs(th) >= HALF_PLANE_DEADZONE_DEG:
                on_pref = th > 0.0 if prefer_positive else th < 0.0
                if not on_pref:
                    continue
        sc = score_sin_against_pairs(s, pair_obs)
        if sc > best_score:
            best_score = sc
            best_s = s

    # If half-plane filtered everything, retry without it.
    if not math.isfinite(best_s):
        for s in seeds:
            sc = score_sin_against_pairs(s, pair_obs)
            if sc > best_score:
                best_score = sc
                best_s = s

    return best_s, {
        "n_candidates": len(seeds),
        "score": float(best_score),
        "overlap": True,
    }


def aoa_deg_from_sin(sin_theta: float) -> float:
    if not math.isfinite(sin_theta):
        return float("nan")
    return float(np.degrees(np.arcsin(float(np.clip(sin_theta, -1.0, 1.0)))))


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


def load_session(npz_path: str) -> Tuple[Dict[str, Any], np.lib.npyio.NpzFile]:
    meta = load_meta(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    return meta, data


def _iq_rows_to_pairs(arr: np.ndarray) -> List[Tuple[float, float]]:
    if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] == 0:
        return []
    out: List[Tuple[float, float]] = []
    for row in arr:
        if not (np.isfinite(row[0]) and np.isfinite(row[1])):
            continue
        out.append((float(row[0]), float(row[1])))
    return out


def n_rounds_in_npz(data: np.lib.npyio.NpzFile, anchors: List[int]) -> int:
    if "n_rounds" in data.files:
        return int(np.asarray(data["n_rounds"]).item())
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


def parse_session_geometry(meta: Dict[str, Any]) -> Dict[str, Any]:
    ref_raw = meta.get("ref_position") or meta.get("ref_pos")
    tgt_raw = meta.get("target_position") or meta.get("target_pos")
    if ref_raw is None:
        raise KeyError("meta missing ref_position")
    if tgt_raw is None:
        raise KeyError("meta missing target_position")

    ref_pos = (float(ref_raw[0]), float(ref_raw[1]))
    target_pos = (float(tgt_raw[0]), float(tgt_raw[1]))

    structure = meta.get("anchor_structure") or {}
    raw_pos = structure.get("positions") or {}
    node_pos: Dict[int, Tuple[float, float]] = {
        int(k): (float(v[0]), float(v[1])) for k, v in raw_pos.items()
    }
    if not node_pos:
        raise KeyError("meta.anchor_structure.positions is empty")

    anchors = [int(a) for a in meta.get("anchor_list", sorted(node_pos.keys()))]
    x_ids = [int(a) for a in (meta.get("anchor_x_ids") or structure.get("x_ids") or [])]
    y_ids = [int(a) for a in (meta.get("anchor_y_ids") or structure.get("y_ids") or [])]
    if not x_ids or not y_ids:
        raise KeyError("meta missing anchor_x_ids / anchor_y_ids (or structure x_ids/y_ids)")

    return {
        "ref_pos": ref_pos,
        "target_pos": target_pos,
        "node_pos": node_pos,
        "anchors": anchors,
        "x_ids": x_ids,
        "y_ids": y_ids,
        "structure_id": int(meta.get("anchor_structure_id", -1)),
        "structure_name": str(structure.get("name", "")),
    }


def load_capture(npz_path: str) -> Dict[str, Any]:
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"npz not found: {npz_path}")
    meta, data = load_session(npz_path)
    geom = parse_session_geometry(meta)
    n_rounds = n_rounds_in_npz(data, geom["anchors"])
    return {
        "npz_path": os.path.abspath(npz_path),
        "meta": meta,
        "data": data,
        "n_rounds": n_rounds,
        **geom,
    }


def resolve_npz_arg(argv: List[str]) -> str:
    if len(argv) > 2:
        raise SystemExit(
            "Usage: python coral_indoor_result_process.py [one_reports.npz]\n"
            "Process exactly one file at a time."
        )
    if len(argv) == 2:
        path = argv[1]
        if os.path.isdir(path):
            raise SystemExit(
                f"Got a directory ({path}); pass a single *_reports.npz file."
            )
        return path
    return DEFAULT_NPZ


# ---------------------------------------------------------------------------
# Geometry helpers (session-local; same conventions as new_listener)
# ---------------------------------------------------------------------------

def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def node_xy(node_pos: Dict[int, Tuple[float, float]], node_id: int) -> Tuple[float, float]:
    return node_pos[node_id]


def orient_ula_ids(
    ids: List[int],
    node_pos: Dict[int, Tuple[float, float]],
    side_point: Optional[Tuple[float, float]] = None,
) -> Tuple[int, int, List[int]]:
    """
    Order ULA along principal axis (increasing x, else increasing y).

    If ``side_point`` is given (RSSI / coarse half-space cue), flip so that
    point lies on the +normal side (baseline +90° CCW). Exact TARGET
    coordinates are not required — any point on the correct side of the
    array line is enough. If None, keep the fixed hardware order.
    """
    xs = [node_pos[a][0] for a in ids]
    ys = [node_pos[a][1] for a in ids]
    along_x = (max(xs) - min(xs)) >= (max(ys) - min(ys))
    if along_x:
        ordered = sorted(ids, key=lambda a: (node_pos[a][0], node_pos[a][1], a))
    else:
        ordered = sorted(ids, key=lambda a: (node_pos[a][1], node_pos[a][0], a))

    if side_point is not None:
        pos_ref = node_pos[ordered[0]]
        pos_end = node_pos[ordered[-1]]
        bx = pos_end[0] - pos_ref[0]
        by = pos_end[1] - pos_ref[1]
        mid_x = 0.5 * (pos_ref[0] + pos_end[0])
        mid_y = 0.5 * (pos_ref[1] + pos_end[1])
        cross = bx * (side_point[1] - mid_y) - by * (side_point[0] - mid_x)
        if cross < 0.0:
            ordered = list(reversed(ordered))
    return ordered[0], ordered[-1], ordered


def theta_side_prior(
    point_xy: Tuple[float, float],
    ids: List[int],
    node_pos: Dict[int, Tuple[float, float]],
    side_point: Optional[Tuple[float, float]] = None,
) -> bool:
    """
    Left/right AoA bit after orienting with ``side_point``: sign(geometric AoA).

    Offline stand-in for a fine RSSI / sector cue. Only the boolean is used
    for half-plane — never |AoA|.
    """
    sp = side_point if side_point is not None else point_xy
    # true_aoa uses orient_ula_ids(..., side_point) via target_xy as geometry
    # point; pass sp for orientation consistency.
    _, _, ordered = orient_ula_ids(ids, node_pos, sp)
    pos_ref = node_pos[ordered[0]]
    pos_end = node_pos[ordered[-1]]
    return bool(true_aoa_deg_for_pair(point_xy, pos_end, pos_ref) > 0.0)


def true_aoa_deg_for_ula(
    target_xy: Tuple[float, float],
    ids: List[int],
    node_pos: Dict[int, Tuple[float, float]],
    side_point: Optional[Tuple[float, float]] = None,
) -> float:
    """Far-field geometric AoA; orientation from ``side_point`` or ``target_xy``."""
    sp = side_point if side_point is not None else target_xy
    _, _, ordered = orient_ula_ids(ids, node_pos, sp)
    pos_ref = node_pos[ordered[0]]
    pos_end = node_pos[ordered[-1]]
    return true_aoa_deg_for_pair(target_xy, pos_end, pos_ref)


def true_aoa_deg_for_pair(
    target_xy: Tuple[float, float],
    pos_end: Tuple[float, float],
    pos_ref: Tuple[float, float],
) -> float:
    """Geometric AoA for one directed pair (ref→end), same formula as ULA mid+axes."""
    mid_x = 0.5 * (pos_ref[0] + pos_end[0])
    mid_y = 0.5 * (pos_ref[1] + pos_end[1])
    bx = pos_end[0] - pos_ref[0]
    by = pos_end[1] - pos_ref[1]
    bl = math.atan2(by, bx)
    nx = math.cos(bl + 0.5 * math.pi)
    ny = math.sin(bl + 0.5 * math.pi)
    ax = math.cos(bl)
    ay = math.sin(bl)
    vx = target_xy[0] - mid_x
    vy = target_xy[1] - mid_y
    along = vx * ax + vy * ay
    normal_comp = vx * nx + vy * ny
    theta = math.degrees(math.atan2(-along, normal_comp))
    if theta > 90.0:
        theta = 180.0 - theta
    elif theta < -90.0:
        theta = -180.0 - theta
    return float(np.clip(theta, -90.0, 90.0))


def bearing_from_aoa(
    pos_a: Tuple[float, float],
    pos_b: Tuple[float, float],
    aoa_rad: float,
) -> float:
    bx = pos_a[0] - pos_b[0]
    by = pos_a[1] - pos_b[1]
    baseline_angle = math.atan2(by, bx)
    normal_angle = baseline_angle + 0.5 * math.pi
    return normal_angle + aoa_rad


# ---------------------------------------------------------------------------
# Pair AoA (new_listener pipeline, no half-plane)
# ---------------------------------------------------------------------------

def div_iq_to_complex(div_iq: List[Any]) -> np.ndarray:
    if not div_iq:
        return np.zeros(0, dtype=np.complex128)
    arr = np.asarray(div_iq, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros(0, dtype=np.complex128)
    return arr[:, 0] + 1j * arr[:, 1]


def aligned_div_complex(a: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    n = min(len(a), len(b))
    if n <= 0:
        return None
    return np.divide(
        a[:n],
        b[:n],
        out=np.full(n, np.nan, dtype=np.complex128),
        where=np.abs(b[:n]) > 0.0,
    )


def td_ticks_to_seconds(td_ticks: float) -> float:
    return float(td_ticks) / TIMER_HZ


def ref_to_anchor_phase(
    ref_pos: Tuple[float, float],
    node_pos: Dict[int, Tuple[float, float]],
    anchor_id: int,
) -> float:
    return -2.0 * math.pi * F_CENTER_HZ * dist(ref_pos, node_pos[anchor_id]) / C_LIGHT


def pair_slope_from_cte(rpt_a: Dict[str, Any], rpt_b: Dict[str, Any]) -> float:
    _, _, slope_ref = compute_div_iq_and_ls(rpt_a.get("cte_ref", []), rpt_b.get("cte_ref", []))
    _, _, slope_echo = compute_div_iq_and_ls(rpt_a.get("cte_echo", []), rpt_b.get("cte_echo", []))
    return 0.5 * (float(slope_ref) + float(slope_echo))


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


def wrap_phase(phase_rad: float) -> float:
    return float(np.arctan2(np.sin(phase_rad), np.cos(phase_rad)))


def aoa_from_pair_phase_half(
    phase_rad: float,
    baseline_m: float,
    prefer_positive: bool,
    deadzone_deg: float = HALF_PLANE_DEADZONE_DEG,
) -> Dict[str, Any]:
    """
    Half-plane prior: force θ>0 or θ<0 by optional phase + π.

    Same rule as new_listener.aoa_from_pair_phase_half:
    if |θ_raw| < deadzone_deg, keep raw (do not flip).
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


def corrected_pair_ratio(
    reports: Dict[int, Dict[str, Any]],
    id_a: int,
    id_b: int,
    ref_pos: Tuple[float, float],
    node_pos: Dict[int, Tuple[float, float]],
) -> Optional[np.ndarray]:
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
    slope = pair_slope_from_cte(rpt_a, rpt_b)
    td_ticks = 0.5 * (float(rpt_a["time_diff"]) + float(rpt_b["time_diff"]))
    phi_cfo = slope * (td_ticks_to_seconds(td_ticks) / SAMPLE_INTERVAL_S)
    phi_m = (
        ref_to_anchor_phase(ref_pos, node_pos, id_a)
        - ref_to_anchor_phase(ref_pos, node_pos, id_b)
    )
    return ratio * np.exp(-1j * phi_cfo) * np.exp(-1j * phi_m)


def raw_looks_broadside(
    raw_aoas: List[float],
    *,
    near_deg: float = BROADSIDE_RAW_DEG,
    frac: float = BROADSIDE_RAW_FRAC,
) -> bool:
    """True if a large fraction of raw AoAs cluster near 0 (skip half-plane)."""
    vals = [
        float(th)
        for th in raw_aoas
        if math.isfinite(th) and abs(abs(th) - 90.0) > 1e-6
    ]
    if len(vals) < 4:
        return True
    near = sum(1 for th in vals if abs(th) < near_deg)
    return (near / float(len(vals))) >= frac


def infer_side_from_raw_aoas(
    raw_aoas: List[float],
    *,
    evidence_deg: float = BROADSIDE_RAW_DEG,
    min_votes: int = 8,
    min_mean_sin: float = 0.12,
) -> Optional[bool]:
    """
    Blind side vote from raw AoAs (no TARGET). Often weak under front/back
    ambiguity; prefer RSSI / theta_side_prior in practice.
    """
    votes: List[float] = []
    for th in raw_aoas:
        if not math.isfinite(th):
            continue
        if abs(abs(th) - 90.0) <= 1e-6:
            continue
        if abs(th) < evidence_deg:
            continue
        votes.append(float(th))
    if len(votes) < min_votes:
        return None
    mean_sin = float(np.mean([math.sin(math.radians(th)) for th in votes]))
    if abs(mean_sin) < min_mean_sin:
        return None
    return bool(mean_sin > 0.0)


def estimate_pairs_in_group(
    reports: Dict[int, Dict[str, Any]],
    ids: List[int],
    ref_pos: Tuple[float, float],
    target_pos: Tuple[float, float],
    node_pos: Dict[int, Tuple[float, float]],
    group_name: str,
    prefer_positive: Optional[bool] = None,
    side_point: Optional[Tuple[float, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Per-pair AoA for one ULA group.

    side_point orients the ULA (RSSI half-space cue). Half-plane uses only
    prefer_positive (left/right bit). If None, keep raw. target_pos is for
    evaluation fields only.

    When any pair baseline exceeds λ/2, use multi-baseline overlap
    disambiguation (shared sinθ across pairs) instead of per-pair arcsin.
    """
    enable_half = prefer_positive is not None
    deadzone = HALF_PLANE_DEADZONE_DEG if enable_half else 0.0
    prefer_flag = bool(prefer_positive) if enable_half else False

    _, _, ordered = orient_ula_ids(ids, node_pos, side_point)
    rank = {aid: i for i, aid in enumerate(ordered)}
    true_ula = true_aoa_deg_for_ula(target_pos, ids, node_pos, side_point=side_point)

    # Collect oriented pairs + phases first.
    collected: List[Dict[str, Any]] = []
    for id_a, id_b in itertools.combinations(ids, 2):
        if rank[id_a] < rank[id_b]:
            id_a, id_b = id_b, id_a
        pos_a = node_pos[id_a]
        pos_b = node_pos[id_b]
        baseline = dist(pos_a, pos_b)
        if baseline < 1e-9:
            continue
        ratio = corrected_pair_ratio(reports, id_a, id_b, ref_pos, node_pos)
        if ratio is None:
            continue
        phase = float(np.angle(np.mean(ratio)))
        if not math.isfinite(phase):
            continue
        collected.append(
            {
                "id_a": id_a,
                "id_b": id_b,
                "pos_a": pos_a,
                "pos_b": pos_b,
                "baseline_m": baseline,
                "phase_rad": phase,
                "aliased": baseline_aliased(baseline),
            }
        )
    if not collected:
        return []

    use_overlap = ENABLE_OVERLAP_DISAMBIG and any(c["aliased"] for c in collected)
    overlap_sin = float("nan")
    overlap_info: Dict[str, Any] = {"overlap": False}
    if use_overlap:
        pair_obs = [
            (
                float(c["phase_rad"]),
                float(c["baseline_m"]),
                float(c["baseline_m"]) ** 2,
            )
            for c in collected
        ]
        overlap_sin, overlap_info = disambiguate_sin_by_overlap(
            pair_obs,
            prefer_positive=prefer_positive if enable_half else None,
        )

    out: List[Dict[str, Any]] = []
    for c in collected:
        baseline = float(c["baseline_m"])
        phase = float(c["phase_rad"])
        pos_a = c["pos_a"]
        pos_b = c["pos_b"]
        aoa_raw = aoa_from_pair_phase(phase, baseline)
        pi_corrected = False

        if use_overlap and math.isfinite(overlap_sin):
            # Pair estimate = consensus sinθ (overlap); report principal as raw.
            aoa_deg = aoa_deg_from_sin(overlap_sin)
            # Mark pi_corrected if consensus differs a lot from principal branch.
            if math.isfinite(aoa_raw) and abs(aoa_deg - aoa_raw) > 5.0:
                pi_corrected = True
        elif enable_half:
            est = aoa_from_pair_phase_half(
                phase, baseline, prefer_flag, deadzone_deg=deadzone
            )
            aoa_raw = float(est["aoa_raw_deg"])
            aoa_deg = float(est["aoa_deg"])
            pi_corrected = bool(est["pi_corrected"])
        else:
            aoa_deg = aoa_raw

        if not math.isfinite(aoa_raw) and not math.isfinite(aoa_deg):
            continue
        if not math.isfinite(aoa_deg):
            aoa_deg = aoa_raw
        if not math.isfinite(aoa_raw):
            aoa_raw = aoa_deg

        mid = (0.5 * (pos_a[0] + pos_b[0]), 0.5 * (pos_a[1] + pos_b[1]))
        true_deg = true_aoa_deg_for_pair(target_pos, pos_a, pos_b)
        bearing = bearing_from_aoa(pos_a, pos_b, math.radians(aoa_deg))
        out.append(
            {
                "group": group_name,
                "id_a": c["id_a"],
                "id_b": c["id_b"],
                "pos_a": pos_a,
                "pos_b": pos_b,
                "mid": mid,
                "baseline_m": baseline,
                "aoa_raw_deg": aoa_raw,
                "aoa_deg": aoa_deg,
                "pi_corrected": pi_corrected,
                "half_enabled": enable_half,
                "on_normal": not enable_half,
                "prefer_positive": prefer_flag,
                "sample_deadzone_deg": deadzone,
                "true_ula_aoa_deg": true_ula,
                "true_aoa_deg": true_deg,
                "bearing_rad": bearing,
                "direction": (math.cos(bearing), math.sin(bearing)),
                "overlap_disambig": bool(use_overlap),
                "overlap_score": float(overlap_info.get("score", float("nan"))),
                "aliased_baseline": bool(c["aliased"]),
            }
        )
    return out


def process_session_pairs(
    session: Dict[str, Any],
    *,
    prefer_x: Optional[bool] = None,
    prefer_y: Optional[bool] = None,
    side_point_x: Optional[Tuple[float, float]] = None,
    side_point_y: Optional[Tuple[float, float]] = None,
    infer_side: bool = True,
    half_plane_mode: str = "rssi_mock",
) -> Tuple[List[List[Dict[str, Any]]], Dict[str, Any]]:
    """
    Two-pass pair AoA.

    half_plane_mode:
      - \"rssi_mock\": side bit from TARGET sign (offline stand-in for RSSI);
        may broadside-skip on non-aliased ULAs.
      - \"true_aoa\": offline GT prior — orient with TARGET, prefer=sign(trueULA),
        skip half-plane when |trueULA| < SIDE_CONFIDENCE_DEG. No RSSI path.
    """
    use_rssi_mock = half_plane_mode == "rssi_mock"
    use_true_aoa = half_plane_mode == "true_aoa"

    # ULA orientation: TARGET as half-space cue (geometry), not an RSSI sensor.
    if use_rssi_mock or use_true_aoa:
        if side_point_x is None:
            side_point_x = session["target_pos"]
        if side_point_y is None:
            side_point_y = session["target_pos"]

    raw_rounds: List[List[Dict[str, Any]]] = []
    for r in range(session["n_rounds"]):
        reports = reports_for_round(session["data"], session["anchors"], r)
        raw_rounds.append(
            estimate_pairs_in_group(
                reports,
                session["x_ids"],
                session["ref_pos"],
                session["target_pos"],
                session["node_pos"],
                "X",
                prefer_positive=None,
                side_point=side_point_x,
            )
            + estimate_pairs_in_group(
                reports,
                session["y_ids"],
                session["ref_pos"],
                session["target_pos"],
                session["node_pos"],
                "Y",
                prefer_positive=None,
                side_point=side_point_y,
            )
        )

    raw_x = [
        float(p["aoa_raw_deg"])
        for pairs in raw_rounds
        for p in pairs
        if p["group"] == "X"
    ]
    raw_y = [
        float(p["aoa_raw_deg"])
        for pairs in raw_rounds
        for p in pairs
        if p["group"] == "Y"
    ]

    side: Dict[str, Any] = {
        "prefer_x": prefer_x,
        "prefer_y": prefer_y,
        "source_x": "given" if prefer_x is not None else None,
        "source_y": "given" if prefer_y is not None else None,
        "side_point_x": side_point_x,
        "side_point_y": side_point_y,
        "half_plane_mode": half_plane_mode,
    }

    def _assign_prefer_from_true(
        ids: List[int],
        side_point: Optional[Tuple[float, float]],
        uncertain_label: str,
        sign_label: str,
    ) -> Tuple[Optional[bool], str]:
        th = true_aoa_deg_for_ula(
            session["target_pos"],
            ids,
            session["node_pos"],
            side_point=side_point,
        )
        if abs(th) < SIDE_CONFIDENCE_DEG:
            return None, uncertain_label
        return bool(th > 0.0), sign_label

    if prefer_x is None:
        if use_true_aoa:
            prefer_x, side["source_x"] = _assign_prefer_from_true(
                session["x_ids"], side_point_x, "on_normal", "true_sign"
            )
        elif use_rssi_mock and side_point_x is not None:
            prefer_x, side["source_x"] = _assign_prefer_from_true(
                session["x_ids"],
                side_point_x,
                "rssi_mock_uncertain",
                "rssi_mock_theta_sign",
            )
        elif infer_side:
            prefer_x = infer_side_from_raw_aoas(raw_x)
            side["source_x"] = "inferred" if prefer_x is not None else "ambiguous"
        else:
            side["source_x"] = "none"
    if prefer_y is None:
        if use_true_aoa:
            prefer_y, side["source_y"] = _assign_prefer_from_true(
                session["y_ids"], side_point_y, "on_normal", "true_sign"
            )
        elif use_rssi_mock and side_point_y is not None:
            prefer_y, side["source_y"] = _assign_prefer_from_true(
                session["y_ids"],
                side_point_y,
                "rssi_mock_uncertain",
                "rssi_mock_theta_sign",
            )
        elif infer_side:
            prefer_y = infer_side_from_raw_aoas(raw_y)
            side["source_y"] = "inferred" if prefer_y is not None else "ambiguous"
        else:
            side["source_y"] = "none"

    aliased_x = group_has_aliased_baseline(session["x_ids"], session["node_pos"])
    aliased_y = group_has_aliased_baseline(session["y_ids"], session["node_pos"])
    side["aliased_x"] = aliased_x
    side["aliased_y"] = aliased_y

    # RSSI-mock only: raw broadside gate (not used for true_aoa / structure 1).
    if use_rssi_mock:
        if prefer_x is not None and (not aliased_x) and raw_looks_broadside(raw_x):
            prefer_x = None
            side["source_x"] = f"{side['source_x']}+broadside_skip"
        if prefer_y is not None and (not aliased_y) and raw_looks_broadside(raw_y):
            prefer_y = None
            side["source_y"] = f"{side['source_y']}+broadside_skip"

    side["prefer_x"] = prefer_x
    side["prefer_y"] = prefer_y

    per_round: List[List[Dict[str, Any]]] = []
    for r in range(session["n_rounds"]):
        reports = reports_for_round(session["data"], session["anchors"], r)
        per_round.append(
            estimate_pairs_in_group(
                reports,
                session["x_ids"],
                session["ref_pos"],
                session["target_pos"],
                session["node_pos"],
                "X",
                prefer_positive=prefer_x,
                side_point=side_point_x,
            )
            + estimate_pairs_in_group(
                reports,
                session["y_ids"],
                session["ref_pos"],
                session["target_pos"],
                session["node_pos"],
                "Y",
                prefer_positive=prefer_y,
                side_point=side_point_y,
            )
        )
    return per_round, side


def _side_label(prefer: Optional[bool], source: Optional[str]) -> str:
    if prefer is None:
        return f"off({source or 'none'})"
    half = "theta>0" if prefer else "theta<0"
    return f"{half}/{source or 'given'}"


def _finite_nonsat_aoas(aoas: np.ndarray, sat_eps_deg: float = 1e-6) -> np.ndarray:
    """Drop non-finite and arcsin-saturated samples at exactly ±90 deg."""
    a = np.asarray(aoas, dtype=np.float64)
    mask = np.isfinite(a) & (np.abs(np.abs(a) - 90.0) > sat_eps_deg)
    return a[mask]


def _mean_std_nonsat(aoas: np.ndarray) -> Tuple[float, float, int]:
    valid = _finite_nonsat_aoas(aoas)
    if valid.size == 0:
        return float("nan"), float("nan"), 0
    return float(np.mean(valid)), float(np.std(valid)), int(valid.size)


def collect_pair_series(
    per_round: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Gather all per-round AoA samples for each (group, id_a, id_b).

    Mean/std exclude samples saturated at ±90 deg (arcsin clip).
    """
    buckets: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = {}
    for round_pairs in per_round:
        for p in round_pairs:
            key = (p["group"], int(p["id_a"]), int(p["id_b"]))
            buckets.setdefault(key, []).append(p)

    series: List[Dict[str, Any]] = []
    for _key, samples in sorted(buckets.items()):
        aoas = np.asarray([s["aoa_deg"] for s in samples], dtype=np.float64)
        aoas_raw = np.asarray([s["aoa_raw_deg"] for s in samples], dtype=np.float64)
        pi_flags = np.asarray([s["pi_corrected"] for s in samples], dtype=bool)
        proto = samples[0]
        mean_deg, std_deg, n_mean = _mean_std_nonsat(aoas)
        n_sat = int(np.sum(np.isfinite(aoas) & (np.abs(np.abs(aoas) - 90.0) <= 1e-6)))
        n_rounds = int(aoas.size)
        sat_frac = float(n_sat) / float(max(n_rounds, 1))
        reliability = (float(proto["baseline_m"]) ** 2) * ((1.0 - sat_frac) ** 2)
        if math.isfinite(mean_deg):
            bearing = bearing_from_aoa(
                proto["pos_a"], proto["pos_b"], math.radians(mean_deg)
            )
        else:
            bearing = bearing_from_aoa(proto["pos_a"], proto["pos_b"], 0.0)
        series.append(
            {
                "group": proto["group"],
                "id_a": proto["id_a"],
                "id_b": proto["id_b"],
                "pos_a": proto["pos_a"],
                "pos_b": proto["pos_b"],
                "mid": proto["mid"],
                "baseline_m": proto["baseline_m"],
                "true_aoa_deg": proto["true_aoa_deg"],
                "true_ula_aoa_deg": float(proto["true_ula_aoa_deg"]),
                "half_enabled": bool(proto["half_enabled"]),
                "on_normal": bool(proto.get("on_normal", False)),
                "prefer_positive": bool(proto["prefer_positive"]),
                "sample_deadzone_deg": float(proto.get("sample_deadzone_deg", HALF_PLANE_DEADZONE_DEG)),
                "aoa_samples": aoas,
                "aoa_raw_samples": aoas_raw,
                "pi_corrected_flags": pi_flags,
                "aoa_mean_deg": mean_deg,
                "aoa_std_deg": std_deg,
                "n_mean": n_mean,
                "n_sat": n_sat,
                "sat_frac": sat_frac,
                "fusion_weight": reliability,
                "n_pi_corrected": int(np.sum(pi_flags)),
                "n_rounds": n_rounds,
                "bearing_rad": bearing,
                "direction": (math.cos(bearing), math.sin(bearing)),
            }
        )
    return series


def _fuse_sin_weighted(aoas_deg: List[float], weights: List[float]) -> float:
    if not aoas_deg or not weights or sum(weights) <= 0.0:
        return float("nan")
    sins = [math.sin(math.radians(th)) for th in aoas_deg]
    w = np.asarray(weights, dtype=np.float64)
    w = w / np.sum(w)
    sin_fused = float(np.clip(np.dot(w, np.asarray(sins, dtype=np.float64)), -1.0, 1.0))
    return float(np.degrees(np.arcsin(sin_fused)))


def fuse_group_aoa_from_pairs(
    pair_series: List[Dict[str, Any]],
    group: str,
) -> Dict[str, Any]:
    """
    Fuse pair means within one ULA group.

    Weight = baseline^2 * (1 - sat_frac)^2; pairs with no valid mean get 0.
    """
    aoas: List[float] = []
    weights: List[float] = []
    details: List[Dict[str, Any]] = []
    for p in pair_series:
        if p["group"] != group:
            continue
        th = float(p["aoa_mean_deg"])
        w = float(p["fusion_weight"])
        if not math.isfinite(th) or w <= 0.0:
            details.append({**p, "used": False, "w_norm": 0.0})
            continue
        aoas.append(th)
        weights.append(w)
        details.append({**p, "used": True, "w_norm": w})

    fused = _fuse_sin_weighted(aoas, weights)
    w_sum = float(sum(weights))
    for d in details:
        d["w_norm"] = (float(d["w_norm"]) / w_sum) if w_sum > 0.0 and d["used"] else 0.0

    true_vals = [
        float(p["true_aoa_deg"])
        for p in pair_series
        if p["group"] == group and math.isfinite(p["true_aoa_deg"])
    ]
    true_mean = float(np.mean(true_vals)) if true_vals else float("nan")
    return {
        "group": group,
        "aoa_deg": fused,
        "true_aoa_deg": true_mean,
        "n_pairs_used": len(aoas),
        "pair_details": details,
    }


def _is_saturated(aoa_deg: float, sat_eps_deg: float = 1e-6) -> bool:
    return math.isfinite(aoa_deg) and abs(abs(aoa_deg) - 90.0) <= sat_eps_deg


def fuse_aoa_one_round(
    round_pairs: List[Dict[str, Any]],
    pair_weights: Dict[Tuple[str, int, int], float],
    group: str,
) -> float:
    """Per-round group fusion; saturated samples get weight 0."""
    aoas: List[float] = []
    weights: List[float] = []
    for p in round_pairs:
        if p["group"] != group:
            continue
        key = (p["group"], int(p["id_a"]), int(p["id_b"]))
        w = float(pair_weights.get(key, 0.0))
        th = float(p["aoa_deg"])
        if not math.isfinite(th) or _is_saturated(th) or w <= 0.0:
            continue
        aoas.append(th)
        weights.append(w)
    return _fuse_sin_weighted(aoas, weights)


def ray_from_ula(
    ids: List[int],
    aoa_deg: float,
    node_pos: Dict[int, Tuple[float, float]],
    side_point: Optional[Tuple[float, float]] = None,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    _, _, ordered = orient_ula_ids(ids, node_pos, side_point)
    pos_ref = node_pos[ordered[0]]
    pos_end = node_pos[ordered[-1]]
    mid = (0.5 * (pos_ref[0] + pos_end[0]), 0.5 * (pos_ref[1] + pos_end[1]))
    bearing = bearing_from_aoa(pos_end, pos_ref, math.radians(aoa_deg))
    return mid, (math.cos(bearing), math.sin(bearing))


def intersect_rays(
    p1: Tuple[float, float],
    d1: Tuple[float, float],
    p2: Tuple[float, float],
    d2: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    a11, a12 = d1[0], -d2[0]
    a21, a22 = d1[1], -d2[1]
    det = a11 * a22 - a12 * a21
    if abs(det) < 1e-15:
        return None
    b1 = p2[0] - p1[0]
    b2 = p2[1] - p1[1]
    t = (b1 * a22 - b2 * a12) / det
    return (p1[0] + t * d1[0], p1[1] + t * d1[1])


def position_from_two_aoas(
    aoa_x_deg: float,
    aoa_y_deg: float,
    x_ids: List[int],
    y_ids: List[int],
    node_pos: Dict[int, Tuple[float, float]],
    side_point: Optional[Tuple[float, float]] = None,
    side_point_x: Optional[Tuple[float, float]] = None,
    side_point_y: Optional[Tuple[float, float]] = None,
) -> Optional[Tuple[float, float]]:
    if not (math.isfinite(aoa_x_deg) and math.isfinite(aoa_y_deg)):
        return None
    spx = side_point_x if side_point_x is not None else side_point
    spy = side_point_y if side_point_y is not None else side_point
    o_x, d_x = ray_from_ula(x_ids, aoa_x_deg, node_pos, spx)
    o_y, d_y = ray_from_ula(y_ids, aoa_y_deg, node_pos, spy)
    return intersect_rays(o_x, d_x, o_y, d_y)


def fuse_session_localization(
    session: Dict[str, Any],
    pair_series: List[Dict[str, Any]],
    per_round: List[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Fuse X/Y group AoAs and localize; also per-round positions."""
    fuse_x = fuse_group_aoa_from_pairs(pair_series, "X")
    fuse_y = fuse_group_aoa_from_pairs(pair_series, "Y")
    aoa_x = float(fuse_x["aoa_deg"])
    aoa_y = float(fuse_y["aoa_deg"])
    est_pos = position_from_two_aoas(
        aoa_x,
        aoa_y,
        session["x_ids"],
        session["y_ids"],
        session["node_pos"],
        session["target_pos"],
    )
    true_pos = session["target_pos"]
    err_m = dist(est_pos, true_pos) if est_pos is not None else float("nan")

    pair_weights = {
        (p["group"], int(p["id_a"]), int(p["id_b"])): float(p["fusion_weight"])
        for p in pair_series
    }
    round_pos: List[Optional[Tuple[float, float]]] = []
    round_aoa_x: List[float] = []
    round_aoa_y: List[float] = []
    for round_pairs in per_round:
        th_x = fuse_aoa_one_round(round_pairs, pair_weights, "X")
        th_y = fuse_aoa_one_round(round_pairs, pair_weights, "Y")
        round_aoa_x.append(th_x)
        round_aoa_y.append(th_y)
        round_pos.append(
            position_from_two_aoas(
                th_x,
                th_y,
                session["x_ids"],
                session["y_ids"],
                session["node_pos"],
                session["target_pos"],
            )
        )

    return {
        "fuse_x": fuse_x,
        "fuse_y": fuse_y,
        "aoa_x_deg": aoa_x,
        "aoa_y_deg": aoa_y,
        "true_aoa_x_deg": float(fuse_x["true_aoa_deg"]),
        "true_aoa_y_deg": float(fuse_y["true_aoa_deg"]),
        "est_pos": est_pos,
        "true_pos": true_pos,
        "pos_err_m": err_m,
        "round_pos": round_pos,
        "round_aoa_x": round_aoa_x,
        "round_aoa_y": round_aoa_y,
    }


# ---------------------------------------------------------------------------
# Visualization — one figure per anchor pair (50-round distribution)
# ---------------------------------------------------------------------------

def _scene_extent(
    node_pos: Dict[int, Tuple[float, float]],
    ref_pos: Tuple[float, float],
    target_pos: Tuple[float, float],
) -> Tuple[float, float, float, float, float]:
    xs = [p[0] for p in node_pos.values()] + [ref_pos[0], target_pos[0]]
    ys = [p[1] for p in node_pos.values()] + [ref_pos[1], target_pos[1]]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    span = max(xmax - xmin, ymax - ymin, 0.2)
    pad = 0.15 * span
    ray_len = 0.55 * span
    return xmin - pad, xmax + pad, ymin - pad, ymax + pad, ray_len


def _draw_scene_base(
    ax: Any,
    session: Dict[str, Any],
    pair: Dict[str, Any],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> None:
    node_pos = session["node_pos"]
    ref_pos = session["ref_pos"]
    target_pos = session["target_pos"]
    x_ids = session["x_ids"]
    y_ids = session["y_ids"]

    ax.plot(
        [node_pos[a][0] for a in x_ids],
        [node_pos[a][1] for a in x_ids],
        "o-",
        color="C0",
        ms=7,
        lw=1.2,
        alpha=0.55,
        label="X-ULA",
        zorder=2,
    )
    ax.plot(
        [node_pos[a][0] for a in y_ids],
        [node_pos[a][1] for a in y_ids],
        "s-",
        color="C2",
        ms=7,
        lw=1.2,
        alpha=0.55,
        label="Y-ULA",
        zorder=2,
    )
    for aid, (px, py) in node_pos.items():
        ax.annotate(str(aid), (px, py), textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax.plot(ref_pos[0], ref_pos[1], "^", color="gray", ms=11, label="REF", zorder=4)
    ax.plot(
        target_pos[0], target_pos[1], "*", color="C3", ms=15, label="TARGET", zorder=4
    )

    ma, mb = pair["pos_a"], pair["pos_b"]
    ax.plot([mb[0], ma[0]], [mb[1], ma[1]], "-", color="k", lw=2.4, zorder=3, label="pair")
    ax.plot([ma[0], mb[0]], [ma[1], mb[1]], "o", color="k", ms=9, zorder=5)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)


def visualize_one_pair(
    session: Dict[str, Any],
    pair: Dict[str, Any],
    extent: Tuple[float, float, float, float, float],
) -> None:
    """One figure: geometry + rays | histogram | time series."""
    xmin, xmax, ymin, ymax, ray_len = extent
    aoas = np.asarray(pair["aoa_samples"], dtype=np.float64)
    aoas_raw = np.asarray(pair["aoa_raw_samples"], dtype=np.float64)
    pi_flags = np.asarray(pair["pi_corrected_flags"], dtype=bool)
    true_deg = float(pair["true_aoa_deg"])
    half_on = bool(pair["half_enabled"])
    half_label = (
        "off(on-normal)"
        if pair.get("on_normal")
        else ("theta>0" if pair["prefer_positive"] else "theta<0")
    )
    color = "C0" if pair["group"] == "X" else "C2"
    pair_label = f"{pair['group']} {pair['id_b']}-{pair['id_a']}"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))

    ax = axes[0]
    _draw_scene_base(ax, session, pair, xmin, xmax, ymin, ymax)
    mid = pair["mid"]
    for th, used_pi in zip(aoas, pi_flags):
        br = bearing_from_aoa(pair["pos_a"], pair["pos_b"], math.radians(float(th)))
        dx, dy = math.cos(br), math.sin(br)
        ax.plot(
            [mid[0], mid[0] + ray_len * dx],
            [mid[1], mid[1] + ray_len * dy],
            "-",
            color=("C1" if used_pi else color),
            lw=0.8,
            alpha=0.18,
            zorder=1,
        )
    dx, dy = pair["direction"]
    ax.annotate(
        "",
        xy=(mid[0] + ray_len * dx, mid[1] + ray_len * dy),
        xytext=mid,
        arrowprops=dict(arrowstyle="->", color=color, lw=2.0),
        zorder=6,
    )
    if half_on:
        ax.plot([], [], color=color, lw=1.5, label="kept")
        ax.plot([], [], color="C1", lw=1.5, label="+pi flipped")
    ax.legend(loc="best", fontsize=7)
    ax.set_title(f"{pair_label}  half={half_label}")

    ax = axes[1]
    bins = np.linspace(-90.0, 90.0, 37)
    if half_on:
        ax.hist(
            aoas_raw,
            bins=bins,
            color="0.65",
            edgecolor="k",
            alpha=0.45,
            label="raw",
        )
    ax.hist(
        aoas,
        bins=bins,
        color=color,
        edgecolor="k",
        alpha=0.75,
        label=f"used  mean={pair['aoa_mean_deg']:+.1f} (n={pair['n_mean']})",
    )
    ax.axvline(true_deg, color="C3", ls="-", lw=2.0, label=f"true {true_deg:+.1f}")
    ax.axvline(0.0, color="gray", ls=":", lw=1.0)
    ax.axvline(90.0, color="C1", ls=":", lw=1.0, alpha=0.7)
    ax.axvline(-90.0, color="C1", ls=":", lw=1.0, alpha=0.7)
    ax.set_xlim(-95, 95)
    ax.set_xlabel("AoA [deg]")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title(
        f"excl +/-90  sat={pair['n_sat']}/{pair['n_rounds']}  "
        f"+pi x{pair['n_pi_corrected']}  err={pair['aoa_mean_deg']-true_deg:+.1f}"
    )

    ax = axes[2]
    rounds = np.arange(1, aoas.size + 1)
    if half_on:
        ax.plot(rounds, aoas_raw, "o-", color="0.55", ms=3, lw=0.8, label="raw", alpha=0.8)
    ax.plot(rounds, aoas, "s-", color=color, ms=4, lw=1.0, label="used")
    if half_on and np.any(pi_flags):
        ax.scatter(
            rounds[pi_flags],
            aoas[pi_flags],
            color="C1",
            s=28,
            zorder=5,
            label="+pi",
        )
    ax.axhline(true_deg, color="C3", ls="-", lw=1.5, label="true")
    if math.isfinite(pair["aoa_mean_deg"]):
        ax.axhline(pair["aoa_mean_deg"], color="k", ls="--", lw=1.2, label="mean")
    ax.axhline(0.0, color="gray", ls=":", lw=1.0)
    ax.set_ylim(-95, 95)
    ax.set_xlabel("round")
    ax.set_ylabel("AoA [deg]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title(f"per-round  ula_true={pair['true_ula_aoa_deg']:+.1f}")

    name = os.path.basename(session["npz_path"])
    fig.suptitle(
        f"{name}  |  pair {pair_label}  baseline={pair['baseline_m']*100:.1f} cm  "
        f"half={half_label}  sample_dz={pair.get('sample_deadzone_deg', HALF_PLANE_DEADZONE_DEG):.0f} deg"
    )
    fig.tight_layout()


def visualize_localization(
    session: Dict[str, Any],
    loc: Dict[str, Any],
    pair_series: List[Dict[str, Any]],
) -> None:
    """Final fused AoA / position vs true TARGET."""
    node_pos = session["node_pos"]
    ref_pos = session["ref_pos"]
    true_pos = loc["true_pos"]
    est_pos = loc["est_pos"]
    x_ids = session["x_ids"]
    y_ids = session["y_ids"]

    # Expand view to include estimated / per-round positions.
    xs = [p[0] for p in node_pos.values()] + [ref_pos[0], true_pos[0]]
    ys = [p[1] for p in node_pos.values()] + [ref_pos[1], true_pos[1]]
    if est_pos is not None:
        xs.append(est_pos[0])
        ys.append(est_pos[1])
    for p in loc["round_pos"]:
        if p is not None:
            xs.append(p[0])
            ys.append(p[1])
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    span = max(xmax - xmin, ymax - ymin, 0.2)
    pad = 0.18 * span
    xmin, xmax = xmin - pad, xmax + pad
    ymin, ymax = ymin - pad, ymax + pad
    ray_len = 0.85 * span

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0))

    # --- map: true vs fused AoA position ---
    ax = axes[0]
    ax.plot(
        [node_pos[a][0] for a in x_ids],
        [node_pos[a][1] for a in x_ids],
        "o-",
        color="C0",
        ms=7,
        lw=1.2,
        label="X-ULA",
    )
    ax.plot(
        [node_pos[a][0] for a in y_ids],
        [node_pos[a][1] for a in y_ids],
        "s-",
        color="C2",
        ms=7,
        lw=1.2,
        label="Y-ULA",
    )
    for aid, (px, py) in node_pos.items():
        ax.annotate(str(aid), (px, py), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.plot(ref_pos[0], ref_pos[1], "^", color="gray", ms=11, label="REF")
    ax.plot(true_pos[0], true_pos[1], "*", color="C3", ms=18, label="true TARGET", zorder=5)

    # per-round estimates (background)
    rx = [p[0] for p in loc["round_pos"] if p is not None]
    ry = [p[1] for p in loc["round_pos"] if p is not None]
    if rx:
        ax.plot(rx, ry, "x", color="0.6", ms=4, alpha=0.45, label="round est", zorder=2)

    if est_pos is not None:
        o_x, d_x = ray_from_ula(x_ids, loc["aoa_x_deg"], node_pos, true_pos)
        o_y, d_y = ray_from_ula(y_ids, loc["aoa_y_deg"], node_pos, true_pos)
        # Extend rays through the intersection so the fused position is obvious.
        ax.plot(
            [o_x[0] - 0.15 * ray_len * d_x[0], o_x[0] + ray_len * d_x[0]],
            [o_x[1] - 0.15 * ray_len * d_x[1], o_x[1] + ray_len * d_x[1]],
            "--",
            color="C0",
            lw=1.6,
            label=f"X ray ({loc['aoa_x_deg']:+.1f} deg)",
            zorder=3,
        )
        ax.plot(
            [o_y[0] - 0.15 * ray_len * d_y[0], o_y[0] + ray_len * d_y[0]],
            [o_y[1] - 0.15 * ray_len * d_y[1], o_y[1] + ray_len * d_y[1]],
            "--",
            color="C2",
            lw=1.6,
            label=f"Y ray ({loc['aoa_y_deg']:+.1f} deg)",
            zorder=3,
        )
        # Error segment true -> est
        ax.plot(
            [true_pos[0], est_pos[0]],
            [true_pos[1], est_pos[1]],
            "-",
            color="C3",
            lw=1.2,
            alpha=0.7,
            zorder=4,
        )
        ax.plot(
            est_pos[0],
            est_pos[1],
            "P",
            color="C1",
            ms=16,
            markeredgecolor="k",
            markeredgewidth=0.8,
            label="fused AoA position",
            zorder=7,
        )
        ax.annotate(
            f"est ({est_pos[0]:.3f},{est_pos[1]:.3f})\n"
            f"err={loc['pos_err_m']*100:.1f} cm\n"
            f"X={loc['aoa_x_deg']:+.1f} Y={loc['aoa_y_deg']:+.1f}",
            xy=est_pos,
            xytext=(12, 12),
            textcoords="offset points",
            fontsize=8,
            color="C1",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="C1", alpha=0.9),
            zorder=8,
        )
        ax.annotate(
            f"true ({true_pos[0]:.3f},{true_pos[1]:.3f})",
            xy=true_pos,
            xytext=(12, -18),
            textcoords="offset points",
            fontsize=8,
            color="C3",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="C3", alpha=0.85),
            zorder=8,
        )
        err_cm = loc["pos_err_m"] * 100.0
        ax.set_title(
            f"fused AoA position vs true  |  err={err_cm:.1f} cm"
        )
    else:
        ax.set_title("position (no intersection from fused AoA)")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")

    # --- pair weights + fused AoA ---
    ax = axes[1]
    used = [p for p in pair_series if p["fusion_weight"] > 0 and math.isfinite(p["aoa_mean_deg"])]
    labels = [f"{p['group']}:{p['id_b']}-{p['id_a']}" for p in used]
    xs_i = np.arange(len(used))
    weights = np.asarray([p["fusion_weight"] for p in used], dtype=np.float64)
    w_norm = weights / np.sum(weights) if np.sum(weights) > 0 else weights
    colors = ["C0" if p["group"] == "X" else "C2" for p in used]
    ax.bar(xs_i, w_norm, color=colors, edgecolor="k", alpha=0.8)
    ax.set_xticks(xs_i)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("normalized fusion weight")
    pos_str = (
        f"({est_pos[0]:.3f},{est_pos[1]:.3f})" if est_pos is not None else "None"
    )
    ax.set_title(
        f"pair weights  w=b^2*(1-sat)^2\n"
        f"fused X={loc['aoa_x_deg']:+.1f} (true {loc['true_aoa_x_deg']:+.1f})  "
        f"Y={loc['aoa_y_deg']:+.1f} (true {loc['true_aoa_y_deg']:+.1f})\n"
        f"est pos={pos_str}  true=({true_pos[0]:.3f},{true_pos[1]:.3f})"
    )
    ax.grid(True, axis="y", alpha=0.3)

    name = os.path.basename(session["npz_path"])
    fig.suptitle(f"{name}  |  final fused AoA localization")
    fig.tight_layout()


def visualize_all_pairs(
    session: Dict[str, Any],
    pair_series: List[Dict[str, Any]],
    loc: Optional[Dict[str, Any]] = None,
) -> None:
    extent = _scene_extent(session["node_pos"], session["ref_pos"], session["target_pos"])
    n_fig = len(pair_series) + (1 if loc is not None else 0)
    print(f"Opening {n_fig} figures (pairs + localization). Close windows to exit.")
    for pair in pair_series:
        visualize_one_pair(session, pair, extent)
    if loc is not None:
        visualize_localization(session, loc, pair_series)
    plt.show(block=True)
    plt.close("all")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    npz_path = resolve_npz_arg(sys.argv)
    try:
        session = load_capture(npz_path)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Load failed: {exc}", file=sys.stderr)
        return 1

    print(f"==== {os.path.basename(session['npz_path'])} ====")
    print(f"  REF={session['ref_pos']}  TARGET={session['target_pos']}")
    print(f"  X-ULA={session['x_ids']}  Y-ULA={session['y_ids']}")
    print(f"  n_rounds={session['n_rounds']}")
    true_x = true_aoa_deg_for_ula(
        session["target_pos"], session["x_ids"], session["node_pos"]
    )
    true_y = true_aoa_deg_for_ula(
        session["target_pos"], session["y_ids"], session["node_pos"]
    )
    per_round, side = process_session_pairs(session)
    print(
        f"  half-plane (no GT): X={_side_label(side['prefer_x'], side['source_x'])}  "
        f"Y={_side_label(side['prefer_y'], side['source_y'])}  "
        f"sample_dz={HALF_PLANE_DEADZONE_DEG:.0f} deg"
    )
    print(f"  trueULA (eval only): X={true_x:+.2f}  Y={true_y:+.2f}")

    pair_series = collect_pair_series(per_round)
    if not pair_series:
        print("No pair AoA estimates.", file=sys.stderr)
        return 1

    print("---- per-pair summary ----")
    for p in pair_series:
        err = p["aoa_mean_deg"] - p["true_aoa_deg"]
        half = (
            "off"
            if not p["half_enabled"]
            else ("theta>0" if p["prefer_positive"] else "theta<0")
        )
        print(
            f"  {p['group']} {p['id_b']}-{p['id_a']}: "
            f"est={p['aoa_mean_deg']:+.2f}+/-{p['aoa_std_deg']:.2f} "
            f"(n={p['n_mean']}, sat={p['n_sat']})  "
            f"true={p['true_aoa_deg']:+.2f}  err={err:+.2f}  "
            f"half={half}  +pi x{p['n_pi_corrected']}/{p['n_rounds']}  "
            f"w={p['fusion_weight']:.3e}  base={p['baseline_m']*100:.1f} cm"
        )

    loc = fuse_session_localization(session, pair_series, per_round)
    print("---- fused AoA / position ----")
    print(
        f"  AoA X={loc['aoa_x_deg']:+.2f} (true {loc['true_aoa_x_deg']:+.2f})  "
        f"Y={loc['aoa_y_deg']:+.2f} (true {loc['true_aoa_y_deg']:+.2f})"
    )
    if loc["est_pos"] is not None:
        print(
            f"  est pos=({loc['est_pos'][0]:.3f},{loc['est_pos'][1]:.3f})  "
            f"true=({loc['true_pos'][0]:.3f},{loc['true_pos'][1]:.3f})  "
            f"err={loc['pos_err_m']*100:.1f} cm"
        )
    else:
        print("  est pos=None (rays did not intersect)")
    n_ok = sum(1 for p in loc["round_pos"] if p is not None)
    print(f"  per-round positions available: {n_ok}/{len(loc['round_pos'])}")

    visualize_all_pairs(session, pair_series, loc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
