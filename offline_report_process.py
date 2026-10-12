#!/usr/bin/env python3
"""
Offline batch processing for captured CoRal listener rounds.

Supports two capture formats:

  1) New (indoor_coral / new_listener.py):
       indoor_coral/{ref}_{tgt}_{structure}_reports.npz
       indoor_coral/{ref}_{tgt}_{structure}_meta.json
     All rounds in one npz (cte_ref / cte_echo stacks). Geometry + TARGET
     come from meta. For CROSS_8 layouts, ULA_GROUP selects X or Y arm.

  2) Legacy (CoRal_listener.py):
       captured_reports/.../round_XXX.npz
     One file per round with precomputed div_iq.

Pipeline: snapshot correction + ESPRIT, plus pair-fusion AoA with half-plane
prior from TARGET_POS.
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
    bearing_from_aoa,
    build_array_snapshots,
    compute_div_iq_and_ls,
    dist,
    div_iq_to_complex,
    node_xy,
    ref_to_anchor_phase,
    td_ticks_to_seconds,
    ula_element_spacing_m,
)

# Default new-format session directory / file.
INDOOR_CORAL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "indoor_coral"
)
DEFAULT_SESSION_NPZ = os.path.join(
    INDOOR_CORAL_DIR,
    "(0.00,0.00)_(0.50,0.00)_1_reports.npz",
)

# For CROSS_8 meta: which linear arm to process ("x" or "y").
ULA_GROUP = "x"

# Fallback TARGET when meta has none (legacy captured_reports).
# TARGET_POS: Tuple[float, float] = (0.1, 0.1)       #test1
TARGET_POS: Tuple[float, float] = (0.1, 0.75)       #test2
# TARGET_POS: Tuple[float, float] = (0.1, -0.1)       #test3


# ---------------------------------------------------------------------------
# Session I/O — new indoor_coral format
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


def apply_session_geometry(
    meta: Dict[str, Any],
    ula_group: str = ULA_GROUP,
) -> Tuple[List[int], Tuple[float, float]]:
    """
    Push capture geometry into CoRal_listener globals.

    For CROSS layouts, only the selected arm (x/y) becomes ANCHOR_LIST so the
    existing linear-ULA ESPRIT / pair pipeline stays valid.

    Returns (anchor_list, target_pos).
    """
    global TARGET_POS

    ref_raw = meta.get("ref_position") or meta.get("ref_pos")
    if ref_raw is None:
        raise KeyError("meta missing ref_position")
    cl.REF_POS = (float(ref_raw[0]), float(ref_raw[1]))

    tgt_raw = meta.get("target_position") or meta.get("target_pos")
    if tgt_raw is None:
        raise KeyError("meta missing target_position")
    target_pos = (float(tgt_raw[0]), float(tgt_raw[1]))
    TARGET_POS = target_pos

    structure = meta.get("anchor_structure") or {}
    raw_pos = structure.get("positions") or {}
    node_pos: Dict[int, Tuple[float, float]] = {
        int(k): (float(v[0]), float(v[1])) for k, v in raw_pos.items()
    }
    if not node_pos:
        raise KeyError("meta.anchor_structure.positions is empty")
    cl.NODE_POS = dict(node_pos)

    x_ids = [int(a) for a in (meta.get("anchor_x_ids") or structure.get("x_ids") or [])]
    y_ids = [int(a) for a in (meta.get("anchor_y_ids") or structure.get("y_ids") or [])]
    all_ids = [int(a) for a in meta.get("anchor_list", sorted(node_pos.keys()))]

    group = ula_group.strip().lower()
    if group in ("x", "x_ula", "x-ula") and x_ids:
        anchors = list(x_ids)
    elif group in ("y", "y_ula", "y-ula") and y_ids:
        anchors = list(y_ids)
    else:
        # Linear / unknown: use full anchor_list ordered by (x, y).
        anchors = sorted(all_ids, key=lambda a: (node_pos[a][0], node_pos[a][1], a))

    # Direct ULA along principal axis; keep target on +normal side.
    xs = [node_pos[a][0] for a in anchors]
    ys = [node_pos[a][1] for a in anchors]
    along_x = (max(xs) - min(xs)) >= (max(ys) - min(ys))
    if along_x:
        ordered = sorted(anchors, key=lambda a: (node_pos[a][0], node_pos[a][1], a))
    else:
        ordered = sorted(anchors, key=lambda a: (node_pos[a][1], node_pos[a][0], a))

    pos_ref = node_pos[ordered[0]]
    pos_end = node_pos[ordered[-1]]
    bx = pos_end[0] - pos_ref[0]
    by = pos_end[1] - pos_ref[1]
    mid_x = 0.5 * (pos_ref[0] + pos_end[0])
    mid_y = 0.5 * (pos_ref[1] + pos_end[1])
    cross = bx * (target_pos[1] - mid_y) - by * (target_pos[0] - mid_x)
    if cross < 0.0:
        ordered = list(reversed(ordered))

    cl.ARRAY_REF_ID = ordered[0]
    cl.ARRAY_OTHER_IDS = list(ordered[1:])
    cl.ANCHOR_LIST = list(ordered)
    return list(ordered), target_pos


def reports_for_round_session(
    data: np.lib.npyio.NpzFile,
    anchors: List[int],
    round_idx: int,
) -> Dict[int, Dict[str, Any]]:
    """
    Build reports dict for one round from a multi-round session npz
    (cte_ref / cte_echo -> div_iq), same as new_offline_process.
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


def load_session(npz_path: str) -> Tuple[Dict[str, Any], np.lib.npyio.NpzFile]:
    meta = load_meta(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    return meta, data


# ---------------------------------------------------------------------------
# Legacy round_XXX.npz I/O
# ---------------------------------------------------------------------------

def load_reports_from_npz(path: str) -> Tuple[int, Dict[int, Dict[str, Any]]]:
    """Reconstruct listener-style reports dict from one legacy round_XXX.npz."""
    data = np.load(path, allow_pickle=True)
    round_idx = int(np.asarray(data["round_idx"]).item()) if "round_idx" in data.files else 0
    if "anchor_list" in data.files:
        anchors = [int(a) for a in np.asarray(data["anchor_list"]).tolist()]
    else:
        anchors = list(ANCHOR_LIST)

    reports: Dict[int, Dict[str, Any]] = {}
    for aid in anchors:
        prefix = f"id{aid}_"
        # New-style single-round with cte stacks (rare) — treat as round 0.
        if prefix + "div_iq" not in data.files:
            if prefix + "cte_ref" in data.files and prefix + "cte_echo" in data.files:
                cte_ref_arr = np.asarray(data[prefix + "cte_ref"])
                cte_echo_arr = np.asarray(data[prefix + "cte_echo"])
                if cte_ref_arr.ndim == 3:
                    cte_ref = _iq_rows_to_pairs(cte_ref_arr[0])
                    cte_echo = _iq_rows_to_pairs(cte_echo_arr[0])
                else:
                    cte_ref = _iq_rows_to_pairs(cte_ref_arr)
                    cte_echo = _iq_rows_to_pairs(cte_echo_arr)
                if not cte_ref or not cte_echo:
                    continue
                div_iq, intercept, slope = compute_div_iq_and_ls(cte_ref, cte_echo)
                if not div_iq:
                    continue
                def _scalar(name: str, default: int = 0) -> int:
                    key = prefix + name
                    if key not in data.files:
                        return default
                    arr = np.asarray(data[key])
                    return int(arr.flat[0])
                reports[aid] = {
                    "id": aid,
                    "slot": _scalar("slot"),
                    "x": _scalar("x"),
                    "y": _scalar("y"),
                    "time_diff": _scalar("time_diff"),
                    "rssi": _scalar("rssi"),
                    "intercept": float(intercept),
                    "slope": float(slope),
                    "ndiv": len(div_iq),
                    "div_iq": div_iq,
                    "cte_ref": cte_ref,
                    "cte_echo": cte_echo,
                }
            continue

        div_iq = np.asarray(data[prefix + "div_iq"], dtype=np.float64)
        reports[aid] = {
            "id": aid,
            "slot": int(np.asarray(data[prefix + "slot"]).item()),
            "x": int(np.asarray(data[prefix + "x"]).item()),
            "y": int(np.asarray(data[prefix + "y"]).item()),
            "intercept": float(np.asarray(data[prefix + "intercept"]).item()),
            "slope": float(np.asarray(data[prefix + "slope"]).item()),
            "time_diff": int(np.asarray(data[prefix + "time_diff"]).item()),
            "rssi": int(np.asarray(data[prefix + "rssi"]).item()),
            "ndiv": int(np.asarray(data[prefix + "ndiv"]).item()),
            "div_iq": [(float(row[0]), float(row[1])) for row in div_iq],
        }
    return round_idx, reports


def resolve_input(argv: List[str]) -> Tuple[str, str]:
    """
    Returns (mode, path) where mode is 'session' or 'legacy_dir'.

    CLI:
      python offline_report_process.py [session_reports.npz | legacy_dir]
    """
    if len(argv) > 2:
        raise ValueError(
            "Usage: python offline_report_process.py "
            "[indoor_coral/..._reports.npz | captured_reports/dir]"
        )
    path = argv[1] if len(argv) == 2 else DEFAULT_SESSION_NPZ
    if os.path.isfile(path) and path.endswith("_reports.npz"):
        return "session", path
    if os.path.isfile(path) and path.endswith(".npz") and "round_" in os.path.basename(path):
        return "legacy_dir", os.path.dirname(path) or "."
    if os.path.isdir(path):
        sessions = sorted(glob.glob(os.path.join(path, "*_reports.npz")))
        if len(sessions) == 1:
            return "session", sessions[0]
        if len(sessions) > 1:
            raise ValueError(
                f"Multiple *_reports.npz under {path}; pass one file explicitly."
            )
        return "legacy_dir", path
    raise FileNotFoundError(f"Input not found or unsupported: {path}")


def corrected_pair_ratio(
    reports: Dict[int, Dict[str, Any]],
    id_a: int,
    id_b: int,
) -> Optional[np.ndarray]:
    """
    Corrected complex ratio for pair (a, b): div_a/div_b after CFO + REF geometry.

      out = (div_a/div_b) · exp(-j φ_cfo) · exp(-j (φ_REF->a - φ_REF->b))
    matching the listener correction for array rows.
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

    slope_diff = float(rpt_a["slope"]) - float(rpt_b["slope"])
    td_ticks = 0.5 * (float(rpt_a["time_diff"]) + float(rpt_b["time_diff"]))
    phi_cfo = slope_diff * (td_ticks_to_seconds(td_ticks) / SAMPLE_INTERVAL_S)
    phi_m = ref_to_anchor_phase(id_a) - ref_to_anchor_phase(id_b)
    return ratio * np.exp(-1j * phi_cfo) * np.exp(-1j * phi_m)


def aoa_from_pair_phase(phase_rad: float, baseline_m: float) -> float:
    """
    φ = exp(-j 2pi f_c L sintheta / c)  ⇒  sintheta = -∠φ · c / (2pi f_c L).
    """
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
) -> Dict[str, Any]:
    """
    Half-plane prior from TARGET_POS side.

    prefer_positive=True  -> force theta > 0 (right half)
    prefer_positive=False -> force theta < 0 (left half)

    If raw theta is on the wrong half, add pi to the phase and re-estimate.
    """
    th_raw = aoa_from_pair_phase(phase_rad, baseline_m)
    if not math.isfinite(th_raw):
        return {
            "aoa_raw_deg": float("nan"),
            "aoa_deg": float("nan"),
            "phase_raw_rad": float(phase_rad),
            "phase_used_rad": float(phase_rad),
            "pi_corrected": False,
        }

    on_preferred = th_raw > 0.0 if prefer_positive else th_raw < 0.0
    if on_preferred:
        return {
            "aoa_raw_deg": th_raw,
            "aoa_deg": th_raw,
            "phase_raw_rad": float(phase_rad),
            "phase_used_rad": float(phase_rad),
            "pi_corrected": False,
        }

    phase_fix = wrap_phase(phase_rad + math.pi)
    th_fix = aoa_from_pair_phase(phase_fix, baseline_m)
    if not math.isfinite(th_fix):
        return {
            "aoa_raw_deg": th_raw,
            "aoa_deg": th_raw,
            "phase_raw_rad": float(phase_rad),
            "phase_used_rad": float(phase_rad),
            "pi_corrected": False,
        }

    fix_on_preferred = th_fix > 0.0 if prefer_positive else th_fix < 0.0
    # Prefer the corrected half; if still wrong, pick the one closer to the preferred side.
    closer_to_preferred = th_fix > th_raw if prefer_positive else th_fix < th_raw
    if fix_on_preferred or closer_to_preferred:
        return {
            "aoa_raw_deg": th_raw,
            "aoa_deg": th_fix,
            "phase_raw_rad": float(phase_rad),
            "phase_used_rad": phase_fix,
            "pi_corrected": True,
        }
    return {
        "aoa_raw_deg": th_raw,
        "aoa_deg": th_raw,
        "phase_raw_rad": float(phase_rad),
        "phase_used_rad": float(phase_rad),
        "pi_corrected": False,
    }


def _fuse_sin_weighted(
    aoas_deg: List[float],
    baselines_m: List[float],
) -> float:
    if not aoas_deg:
        return float("nan")
    sins = [math.sin(math.radians(th)) for th in aoas_deg]
    weights = [b * b for b in baselines_m]
    w = np.asarray(weights, dtype=np.float64)
    w /= np.sum(w)
    sin_fused = float(np.clip(np.dot(w, np.asarray(sins, dtype=np.float64)), -1.0, 1.0))
    return float(np.degrees(np.arcsin(sin_fused)))


def estimate_aoa_all_pairs(
    reports: Dict[int, Dict[str, Any]],
    prefer_positive: bool,
) -> Dict[str, Any]:
    """
    Estimate AoA from every unordered anchor pair, then fuse.

    For each pair: if raw theta is on the wrong half relative to prefer_positive,
    add pi to phase and re-map (TARGET_POS half-plane prior).

    Fusion: weighted average of sin(theta_pair) with weight = baseline².
    """
    pair_aoas_raw: Dict[str, float] = {}
    pair_aoas: Dict[str, float] = {}
    pair_baselines: Dict[str, float] = {}
    pair_pi_corrected: Dict[str, bool] = {}

    aoas_raw: List[float] = []
    aoas_fix: List[float] = []
    baselines_list: List[float] = []
    n_pi = 0

    anchors = list(cl.ANCHOR_LIST)
    for id_a, id_b in itertools.combinations(anchors, 2):
        # Orient pair along ULA order (ARRAY_REF -> end) via increasing rank.
        rank = {aid: i for i, aid in enumerate(anchors)}
        if rank.get(id_a, 0) < rank.get(id_b, 0):
            id_a, id_b = id_b, id_a
        pos_a = node_xy(id_a)
        pos_b = node_xy(id_b)

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

        key = f"{id_b}-{id_a}"
        pair_aoas_raw[key] = float(est["aoa_raw_deg"])
        pair_aoas[key] = float(est["aoa_deg"])
        pair_baselines[key] = baseline
        pair_pi_corrected[key] = bool(est["pi_corrected"])
        if est["pi_corrected"]:
            n_pi += 1

        aoas_raw.append(float(est["aoa_raw_deg"]))
        aoas_fix.append(float(est["aoa_deg"]))
        baselines_list.append(baseline)

    if not baselines_list:
        return {
            "aoa_deg": float("nan"),
            "aoa_raw_deg": float("nan"),
            "pair_aoas_deg": {},
            "pair_aoas_raw_deg": {},
            "pair_baselines_m": {},
            "pair_pi_corrected": {},
            "n_pi_corrected": 0,
        }

    return {
        "aoa_deg": _fuse_sin_weighted(aoas_fix, baselines_list),
        "aoa_raw_deg": _fuse_sin_weighted(aoas_raw, baselines_list),
        "pair_aoas_deg": pair_aoas,
        "pair_aoas_raw_deg": pair_aoas_raw,
        "pair_baselines_m": pair_baselines,
        "pair_pi_corrected": pair_pi_corrected,
        "n_pi_corrected": n_pi,
    }


def esprit_phi(x: np.ndarray, d_sp: float, num_sources: int = 1) -> complex:
    """Return the ESPRIT spatial shift φ (single-source)."""
    m, n = x.shape
    if num_sources < 1 or num_sources >= m:
        raise ValueError(f"need 1 <= num_sources < M={m}, got {num_sources}")

    r_xx = x @ x.conj().T / float(n)
    eigvals, eigvecs = np.linalg.eigh(r_xx)
    idx = np.argsort(eigvals)[::-1]
    es = eigvecs[:, idx[:num_sources]]
    es1 = es[:-1, :]
    es2 = es[1:, :]
    psi = np.linalg.pinv(es1) @ es2
    phi_vals = np.linalg.eigvals(psi)
    if len(phi_vals) == 1:
        return complex(phi_vals[0])
    # Pick φ whose |theta| is smallest among candidates.
    angles = [aoa_from_pair_phase(float(np.angle(p)), d_sp) for p in phi_vals]
    k = int(np.argmin(np.abs(angles)))
    return complex(phi_vals[k])


def esprit_aoa_deg(x: np.ndarray, d_sp: float, num_sources: int = 1) -> float:
    """
    ESPRIT AoA from spatial shift φ (no right-half correction).

      φ = ESPRIT eigenvalue ≈ exp(-j 2pi f_c d sintheta / c)
      sintheta = -∠φ · c / (2pi f_c d)
    """
    phi = esprit_phi(x, d_sp, num_sources=num_sources)
    mag = abs(phi)
    if mag < 1e-12:
        return float("nan")
    phi = phi / mag
    return aoa_from_pair_phase(float(np.angle(phi)), d_sp)


def process_reports(
    reports: Dict[int, Dict[str, Any]],
    d_sp: float,
    prefer_positive: bool,
    round_idx: int,
    path: str = "",
) -> Optional[Dict[str, Any]]:
    """ESPRIT (uncorrected) + per-pair fused AoA (half-plane prior from TARGET_POS)."""
    anchors = list(cl.ANCHOR_LIST)
    missing = [aid for aid in anchors if aid not in reports]
    if missing:
        label = path or f"round {round_idx:03d}"
        print(f"  skip {label}: missing anchors {missing}")
        return None

    x = build_array_snapshots(reports, verbose=False)
    if x is None:
        label = path or f"round {round_idx:03d}"
        print(f"  skip {label}: failed to build snapshots")
        return None

    try:
        theta_esprit = esprit_aoa_deg(x, d_sp, num_sources=1)
    except Exception as exc:
        label = path or f"round {round_idx:03d}"
        print(f"  skip {label}: ESPRIT error: {exc}")
        return None

    pair_est = estimate_aoa_all_pairs(reports, prefer_positive=prefer_positive)
    theta_pair = pair_est["aoa_deg"]
    theta_pair_raw = pair_est["aoa_raw_deg"]

    pos_ref = node_xy(cl.ARRAY_REF_ID)
    pos_end = node_xy(cl.ARRAY_OTHER_IDS[-1])
    mid = (
        0.5 * (pos_ref[0] + pos_end[0]),
        0.5 * (pos_ref[1] + pos_end[1]),
    )
    bearing_esprit = math.degrees(
        bearing_from_aoa(pos_end, pos_ref, math.radians(theta_esprit))
    )

    return {
        "path": path,
        "round_idx": round_idx,
        "n_snapshots": int(x.shape[1]),
        "aoa_deg": float(theta_esprit),
        "aoa_pair_deg": float(theta_pair),
        "aoa_pair_raw_deg": float(theta_pair_raw),
        "pair_aoas_deg": pair_est["pair_aoas_deg"],
        "pair_aoas_raw_deg": pair_est["pair_aoas_raw_deg"],
        "pair_pi_corrected": pair_est["pair_pi_corrected"],
        "n_pi_corrected": int(pair_est["n_pi_corrected"]),
        "bearing_deg": float(bearing_esprit),
        "mid": mid,
        "mean_phase_rad": np.angle(np.mean(x, axis=1)),
        "rssi": {aid: reports[aid]["rssi"] for aid in anchors},
        "td": {aid: reports[aid]["time_diff"] for aid in anchors},
    }


def process_round(
    path: str,
    d_sp: float,
    prefer_positive: bool,
) -> Optional[Dict[str, Any]]:
    """Legacy: load one round_XXX.npz then process."""
    round_idx, reports = load_reports_from_npz(path)
    return process_reports(
        reports, d_sp, prefer_positive, round_idx=round_idx, path=path
    )


def summarize(results: List[Dict[str, Any]], key: str = "aoa_deg") -> Dict[str, float]:
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


def array_side_sign(target_xy: Tuple[float, float]) -> float:
    """
    Which side of the ULA baseline the target lies on.

    Baseline directed ARRAY_REF -> last other. Positive = CCW side
    of that directed line; negative = CW side.
    """
    pos_ref = node_xy(cl.ARRAY_REF_ID)
    pos_end = node_xy(cl.ARRAY_OTHER_IDS[-1])
    bx = pos_end[0] - pos_ref[0]
    by = pos_end[1] - pos_ref[1]
    mid_x = 0.5 * (pos_ref[0] + pos_end[0])
    mid_y = 0.5 * (pos_ref[1] + pos_end[1])
    tx = target_xy[0] - mid_x
    ty = target_xy[1] - mid_y
    cross = bx * ty - by * tx
    if abs(cross) < 1e-15:
        return 0.0
    return 1.0 if cross > 0.0 else -1.0


def true_aoa_deg_from_target(
    target_xy: Tuple[float, float],
) -> float:
    """
    Ground-truth signed AoA (deg) for TARGET_POS vs the ULA baseline.
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


def _half_label(prefer_positive: bool) -> str:
    return "right (theta>0)" if prefer_positive else "left (theta<0)"


def visualize_aoa_comparison(
    results: List[Dict[str, Any]],
    stats_esprit: Dict[str, float],
    stats_pair: Dict[str, float],
    true_aoa_deg: float,
    prefer_positive: bool,
) -> None:
    """Two histograms: ESPRIT (raw) vs pair-fusion (half prior), with true AoA."""
    aoa_e = np.asarray([r["aoa_deg"] for r in results], dtype=np.float64)
    aoa_p = np.asarray([r["aoa_pair_deg"] for r in results], dtype=np.float64)
    half = _half_label(prefer_positive)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    n_bins = min(20, max(5, len(results) // 2))
    bin_edges = np.linspace(-90.0, 90.0, n_bins + 1)
    true_label = "Target"

    ax = axes[0]
    ax.hist(
        aoa_e,
        bins=bin_edges,
        color="C0",
        edgecolor="k",
        alpha=0.85,
        label="ESPRIT",
    )
    ax.axvline(
        stats_esprit["aoa_mean"],
        color="C1",
        ls="--",
        lw=1.5,
        label=f"mean {stats_esprit['aoa_mean']:+.2f}°",
    )
    ax.axvline(true_aoa_deg, color="C3", ls="-", lw=2.0, label=true_label)
    err_e = stats_esprit["aoa_mean"] - true_aoa_deg
    ax.set_title(
        f"ESPRIT  σ={stats_esprit['aoa_std']:.2f}°  "
        f"mean-true={err_e:+.2f}°"
    )
    ax.set_xlabel("AoA [deg]")
    ax.set_ylabel("count")
    ax.set_xlim(-95, 95)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    ax = axes[1]
    ax.hist(
        aoa_p,
        bins=bin_edges,
        color="C2",
        edgecolor="k",
        alpha=0.85,
        label=f"pair-fusion ({half})",
    )
    ax.axvline(
        stats_pair["aoa_mean"],
        color="C1",
        ls="--",
        lw=1.5,
        label=f"mean {stats_pair['aoa_mean']:+.2f}°",
    )
    ax.axvline(true_aoa_deg, color="C3", ls="-", lw=2.0, label=true_label)
    err_p = stats_pair["aoa_mean"] - true_aoa_deg
    ax.set_title(
        f"Pair-fusion ({half})  σ={stats_pair['aoa_std']:.2f}°  "
        f"mean-true={err_p:+.2f}°"
    )
    ax.set_xlabel("AoA [deg]")
    ax.set_xlim(-95, 95)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"AoA comparison (ESPRIT raw / pair ->{half})  N={len(results)}  "
        f"|err|_E={abs(err_e):.2f}°  |err|_P={abs(err_p):.2f}°"
    )
    fig.tight_layout()
    print("Close figure 1 (ESPRIT vs pair-fusion) to continue…")
    plt.show(block=True)
    plt.close(fig)


def visualize_pair_half_effect(
    results: List[Dict[str, Any]],
    stats_pair_raw: Dict[str, float],
    stats_pair_fix: Dict[str, float],
    true_aoa_deg: float,
    prefer_positive: bool,
) -> None:
    """Compare pair-fusion raw vs phase+pi half-plane prior."""
    rounds = np.asarray([r["round_idx"] for r in results], dtype=np.int32)
    p_raw = np.asarray([r["aoa_pair_raw_deg"] for r in results], dtype=np.float64)
    p_fix = np.asarray([r["aoa_pair_deg"] for r in results], dtype=np.float64)
    p_npi = np.asarray([r["n_pi_corrected"] for r in results], dtype=np.int32)
    half = _half_label(prefer_positive)
    fix_label = "+pi->right" if prefer_positive else "+pi->left"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    n_bins = min(20, max(5, len(results) // 2))
    bin_edges = np.linspace(-90.0, 90.0, n_bins + 1)

    ax = axes[0]
    ax.hist(p_raw, bins=bin_edges, color="C3", edgecolor="k", alpha=0.55, label="pair raw")
    ax.hist(p_fix, bins=bin_edges, color="C2", edgecolor="k", alpha=0.55, label=f"pair {fix_label}")
    ax.axvline(true_aoa_deg, color="k", ls="-", lw=2.0, label="Target")
    ax.axvline(0.0, color="gray", ls=":", lw=1.0)
    ax.axvline(stats_pair_raw["aoa_mean"], color="C3", ls="--", lw=1.2)
    ax.axvline(stats_pair_fix["aoa_mean"], color="C2", ls="--", lw=1.2)
    ax.set_xlim(-95, 95)
    ax.set_xlabel("AoA [deg]")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(
        f"Pair-fusion  σ {stats_pair_raw['aoa_std']:.1f}°->{stats_pair_fix['aoa_std']:.1f}°  "
        f"|err| {abs(stats_pair_raw['aoa_mean']-true_aoa_deg):.1f}°->"
        f"{abs(stats_pair_fix['aoa_mean']-true_aoa_deg):.1f}°"
    )

    ax = axes[1]
    ax.plot(rounds, p_raw, "o-", ms=4, lw=1.0, color="C3", label="raw")
    ax.plot(rounds, p_fix, "s-", ms=4, lw=1.0, color="C2", label=fix_label)
    ax.axhline(true_aoa_deg, color="k", ls="-", lw=1.5, label="Target")
    ax.axhline(0.0, color="gray", ls=":", lw=1.0)
    used_p = p_npi > 0
    if np.any(used_p):
        ax.scatter(
            rounds[used_p],
            p_fix[used_p],
            s=70,
            facecolors="none",
            edgecolors="C1",
            linewidths=1.5,
            label="+pi on ≥1 pair",
            zorder=5,
        )
    ax.set_ylim(-95, 95)
    ax.set_xlabel("round")
    ax.set_ylabel("AoA [deg]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(
        f"Pair-fusion per round  "
        f"+pi pairs={int(np.sum(p_npi))}  rounds={int(np.sum(used_p))}/{len(results)}"
    )

    fig.suptitle(f"Half-plane prior on pair phases -> {half} (from TARGET_POS)")
    fig.tight_layout()
    print("Close figure 2 (pair phase+pi effect) to exit.")
    plt.show(block=True)
    plt.close(fig)

    print(f"---- pair half-plane summary ({half}) ----")
    print(
        f"  Pair    : raw mean={stats_pair_raw['aoa_mean']:+.3f}°  "
        f"fixed={stats_pair_fix['aoa_mean']:+.3f}°  "
        f"+pi pairs={int(np.sum(p_npi))}"
    )
    print(f"  true AoA: {true_aoa_deg:+.3f}°")


def main() -> int:
    try:
        mode, path = resolve_input(sys.argv)
    except (ValueError, FileNotFoundError) as exc:
        print(exc, file=sys.stderr)
        return 1

    results: List[Dict[str, Any]] = []

    if mode == "session":
        try:
            meta, data = load_session(path)
            anchors, target_pos = apply_session_geometry(meta, ula_group=ULA_GROUP)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"Session load failed: {exc}", file=sys.stderr)
            return 1

        n_rounds = n_rounds_in_npz(data, anchors)
        d_sp = ula_element_spacing_m()
        true_aoa = true_aoa_deg_from_target(target_pos)
        side = array_side_sign(target_pos)
        prefer_positive = bool(true_aoa > 0.0)
        half = _half_label(prefer_positive)
        side_name = (
            "+normal (CCW/+y)"
            if side > 0
            else ("-normal (CW/-y)" if side < 0 else "on baseline")
        )
        structure = meta.get("anchor_structure") or {}
        print(f"==== session {os.path.basename(path)} ====")
        print(f"  structure={structure.get('name', '?')}  ULA_GROUP={ULA_GROUP}")
        print(f"  REF={cl.REF_POS}  TARGET={target_pos}")
        print(f"  anchors={list(cl.ANCHOR_LIST)}  "
              f"ref={cl.ARRAY_REF_ID}  others={list(cl.ARRAY_OTHER_IDS)}")
        print(f"  ULA spacing d={d_sp*100:.3f} cm  n_rounds={n_rounds}")
        print(
            f"  array side={side_name}  true AoA={true_aoa:+.3f} deg  -> pair prior {half}"
        )

        for r in range(n_rounds):
            reports = reports_for_round_session(data, anchors, r)
            res = process_reports(
                reports, d_sp, prefer_positive, round_idx=r + 1, path=path
            )
            if res is None:
                continue
            results.append(res)
            print(
                f"  round {res['round_idx']:03d}: "
                f"ESPRIT={res['aoa_deg']:+.2f}  "
                f"pair={res['aoa_pair_deg']:+.2f} "
                f"(raw {res['aoa_pair_raw_deg']:+.2f}, +pix{res['n_pi_corrected']})  "
                f"(E err={res['aoa_deg']-true_aoa:+.2f}, "
                f"P err={res['aoa_pair_deg']-true_aoa:+.2f})"
            )
    else:
        # Legacy: directory of round_*.npz
        capture_dir = path
        pattern = os.path.join(capture_dir, "round_*.npz")
        paths = sorted(glob.glob(pattern))
        if not paths:
            print(f"No files matching {pattern}", file=sys.stderr)
            return 1

        d_sp = ula_element_spacing_m()
        true_aoa = true_aoa_deg_from_target(TARGET_POS)
        side = array_side_sign(TARGET_POS)
        prefer_positive = bool(true_aoa > 0.0)
        half = _half_label(prefer_positive)
        side_name = (
            "+normal (CCW/+y)"
            if side > 0
            else ("-normal (CW/-y)" if side < 0 else "on baseline")
        )
        print(f"Found {len(paths)} rounds in {capture_dir}")
        print(f"ULA spacing d={d_sp*100:.3f} cm  anchors={list(cl.ANCHOR_LIST)}")
        print(
            f"TARGET_POS=({TARGET_POS[0]:.3f}, {TARGET_POS[1]:.3f}) m  "
            f"array side={side_name}  true AoA={true_aoa:+.3f} deg  -> pair prior {half}"
        )

        for p in paths:
            res = process_round(p, d_sp, prefer_positive=prefer_positive)
            if res is None:
                continue
            results.append(res)
            print(
                f"  round {res['round_idx']:03d}: "
                f"ESPRIT={res['aoa_deg']:+.2f}  "
                f"pair={res['aoa_pair_deg']:+.2f} "
                f"(raw {res['aoa_pair_raw_deg']:+.2f}, +pix{res['n_pi_corrected']})  "
                f"(E err={res['aoa_deg']-true_aoa:+.2f}, "
                f"P err={res['aoa_pair_deg']-true_aoa:+.2f})"
            )

    if not results:
        print("No rounds processed successfully.", file=sys.stderr)
        return 1

    results.sort(key=lambda r: r["round_idx"])
    stats_e = summarize(results, key="aoa_deg")
    stats_p = summarize(results, key="aoa_pair_deg")
    stats_p_raw = summarize(results, key="aoa_pair_raw_deg")
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
        f"  Pair +pi->{('R' if prefer_positive else 'L')}  : mean={stats_p['aoa_mean']:+.3f}  "
        f"std={stats_p['aoa_std']:.3f}  "
        f"mean-true={stats_p['aoa_mean']-true_aoa:+.3f}"
    )
    print(f"  true AoA        : {true_aoa:+.3f} deg")

    visualize_aoa_comparison(results, stats_e, stats_p, true_aoa, prefer_positive)
    visualize_pair_half_effect(results, stats_p_raw, stats_p, true_aoa, prefer_positive)
    return 0


if __name__ == "__main__":
    sys.exit(main())
