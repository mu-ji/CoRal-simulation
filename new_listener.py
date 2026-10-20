#!/usr/bin/env python3
"""
UART listener: collect CAPTURE_ROUNDS slotframes, save one npz.

Firmware:
  RPT id=2 slot=1 sf=3 x=0 y=0 td=8171 rssi=-27 nib=88 necho=88 frags=8
  CTE_REF n=88
    idx=0 I=... Q=...
  CTE_ECHO n=88
    idx=0 I=... Q=...

One round = one slotframe (`sf`). Reports in a round must share the same sf.
Missing anchors are marked lost (empty CTE / lost=1), not waited on.
"""

from __future__ import annotations

import datetime
import itertools
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import serial


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

#SERIAL_PORT = "COM4"
SERIAL_PORT = "COM9"
SERIAL_BAUD = 115200

ANCHOR_LIST = [2, 3, 4, 5, 6, 7, 8, 9]
ANCHOR_X_IDS = [2, 3, 4, 5]
ANCHOR_Y_IDS = [6, 7, 8, 9]

CAPTURE_ROUNDS = 50
HALF_PLANE_DEADZONE_DEG = 12.0

F_CENTER_HZ = 2.440e9
C_LIGHT = 2.99792458e8
TIMER_HZ = 16e6
SAMPLE_INTERVAL_S = 1e-6

REF_POSITION: Tuple[float, float] = (0.0, 0.0)
TARGET_POSITION: Tuple[float, float] = (0.50, 0.00)
ANCHOR_STRUCTURE_ID = 2

ANCHOR_STRUCTURES: Dict[int, Dict[str, Any]] = {
    3: {
        "name": "CROSS_8_d25mm",
        "positions": {
            2: (-0.09, -1.00), 3: (-0.03, -1.00), 4: (0.03, -1.00), 5: (0.09, -1.00),
            6: (-1.00, -0.09), 7: (-1.00, -0.03), 8: (-1.00, 0.03), 9: (-1.00, 0.09),
        },
        "x_ids": [2, 3, 4, 5],
        "y_ids": [6, 7, 8, 9],
    },
    2: {
        "name": "CROSS_8_d50mm",
        "positions": {
            2: (-0.1, -1), 3: (0.0, -1), 4: (0.15, -1), 5: (0.35, -1),
            6: (-1, -0.15), 7: (-1, 0.0), 8: (-1, 0.1), 9: (-1, 0.25),
        },
        "x_ids": [2, 3, 4, 5],
        "y_ids": [6, 7, 8, 9],
    },
}


def active_structure() -> Dict[str, Any]:
    return ANCHOR_STRUCTURES[ANCHOR_STRUCTURE_ID]


def anchor_positions() -> Dict[int, Tuple[float, float]]:
    raw = active_structure().get("positions", {})
    return {int(k): (float(v[0]), float(v[1])) for k, v in raw.items()}


def group_ids() -> Tuple[List[int], List[int]]:
    st = active_structure()
    return (
        [int(a) for a in st.get("x_ids", ANCHOR_X_IDS)],
        [int(a) for a in st.get("y_ids", ANCHOR_Y_IDS)],
    )


def node_xy(node_id: int) -> Tuple[float, float]:
    return anchor_positions()[node_id]


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _pos_token(pos: Tuple[float, float]) -> str:
    return f"({pos[0]:.2f},{pos[1]:.2f})"


def build_capture_npz_path() -> str:
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indoor_coral_4")
    os.makedirs(base, exist_ok=True)
    name = (
        f"{_pos_token(REF_POSITION)}_"
        f"{_pos_token(TARGET_POSITION)}_"
        f"{ANCHOR_STRUCTURE_ID}_towards_x_reports.npz"
    )
    return os.path.join(base, name)


CAPTURE_NPZ_PATH = build_capture_npz_path()
_round_results: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

IqSample = Tuple[float, float]


@dataclass
class AnchorReport:
    anchor_id: int
    slot: int
    slotframe: int
    x: int
    y: int
    time_diff: int
    rssi: int
    nib: int
    necho: int
    frags: int
    timestamp: str = ""
    cte_ref: List[IqSample] = field(default_factory=list)
    cte_echo: List[IqSample] = field(default_factory=list)
    _section: Optional[str] = field(default=None, repr=False)
    _cte_ref_raw: List[Dict[str, float]] = field(default_factory=list, repr=False)
    _cte_echo_raw: List[Dict[str, float]] = field(default_factory=list, repr=False)

    def is_complete(self) -> bool:
        return (
            len(self._cte_ref_raw) >= self.nib
            and len(self._cte_echo_raw) >= self.necho
        )

    def finalize(self) -> None:
        def to_pairs(raw: List[Dict[str, float]]) -> List[IqSample]:
            ordered = sorted(raw, key=lambda s: int(s["idx"]))
            return [(float(s["i"]), float(s["q"])) for s in ordered]

        self.cte_ref = to_pairs(self._cte_ref_raw)
        self.cte_echo = to_pairs(self._cte_echo_raw)


RPT_RE = re.compile(
    r"^RPT\s+"
    r"id=(?P<id>\d+)\s+"
    r"slot=(?P<slot>\d+)\s+"
    r"sf=(?P<sf>\d+)\s+"
    r"x=(?P<x>-?\d+)\s+"
    r"y=(?P<y>-?\d+)\s+"
    r"td=(?P<td>-?\d+)\s+"
    r"rssi=(?P<rssi>-?\d+)\s+"
    r"nib=(?P<nib>\d+)\s+"
    r"necho=(?P<necho>\d+)\s+"
    r"frags=(?P<frags>\d+)\s*$"
)
CTE_HDR_RE = re.compile(r"^(?P<tag>CTE_REF|CTE_ECHO)\s+n=(?P<n>\d+)\s*$")
CTE_SAMPLE_RE = re.compile(
    r"^\s*idx=(?P<idx>\d+)\s+I=(?P<I>-?\d+)\s+Q=(?P<Q>-?\d+)\s*$"
)


def parse_rpt_line(line: str) -> Optional[AnchorReport]:
    m = RPT_RE.match(line.strip())
    if not m:
        return None
    return AnchorReport(
        anchor_id=int(m.group("id")),
        slot=int(m.group("slot")),
        slotframe=int(m.group("sf")),
        x=int(m.group("x")),
        y=int(m.group("y")),
        time_diff=int(m.group("td")),
        rssi=int(m.group("rssi")),
        nib=int(m.group("nib")),
        necho=int(m.group("necho")),
        frags=int(m.group("frags")),
        timestamp=datetime.datetime.now().isoformat(timespec="milliseconds"),
    )


def parse_cte_hdr(line: str) -> Optional[Dict[str, object]]:
    m = CTE_HDR_RE.match(line.strip())
    if not m:
        return None
    return {"tag": m.group("tag"), "n": int(m.group("n"))}


def parse_cte_sample(line: str) -> Optional[Dict[str, float]]:
    m = CTE_SAMPLE_RE.match(line)
    if not m:
        return None
    return {"idx": float(m.group("idx")), "i": float(m.group("I")), "q": float(m.group("Q"))}


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_all_rounds(
    npz_path: str,
    captured_rounds: List[Dict[int, AnchorReport]],
    lost_rounds: List[List[int]],
) -> str:
    os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
    n_rounds = len(captured_rounds)
    payload: Dict[str, Any] = {
        "n_rounds": np.int32(n_rounds),
        "anchor_list": np.asarray(ANCHOR_LIST, dtype=np.int32),
        "ref_position": np.asarray(REF_POSITION, dtype=np.float64),
        "target_position": np.asarray(TARGET_POSITION, dtype=np.float64),
        "anchor_structure_id": np.int32(ANCHOR_STRUCTURE_ID),
        "saved_at": np.asarray(
            datetime.datetime.now().isoformat(timespec="milliseconds")
        ),
    }

    for aid in ANCHOR_LIST:
        slots, sfs, xs, ys, tds, rssis = [], [], [], [], [], []
        nibs, nechos, frags, timestamps, lost_flags = [], [], [], [], []
        cte_refs, cte_echos = [], []

        for reports, lost in zip(captured_rounds, lost_rounds):
            rpt = reports.get(aid)
            if rpt is None:
                slots.append(0)
                sfs.append(0)
                xs.append(0)
                ys.append(0)
                tds.append(0)
                rssis.append(0)
                nibs.append(0)
                nechos.append(0)
                frags.append(0)
                timestamps.append("")
                lost_flags.append(1)
                cte_refs.append(np.zeros((0, 2), dtype=np.float64))
                cte_echos.append(np.zeros((0, 2), dtype=np.float64))
                continue

            slots.append(rpt.slot)
            sfs.append(rpt.slotframe)
            xs.append(rpt.x)
            ys.append(rpt.y)
            tds.append(rpt.time_diff)
            rssis.append(rpt.rssi)
            nibs.append(rpt.nib)
            nechos.append(rpt.necho)
            frags.append(rpt.frags)
            timestamps.append(rpt.timestamp)
            lost_flags.append(1 if aid in lost else 0)
            cte_refs.append(
                np.asarray(rpt.cte_ref, dtype=np.float64)
                if rpt.cte_ref
                else np.zeros((0, 2), dtype=np.float64)
            )
            cte_echos.append(
                np.asarray(rpt.cte_echo, dtype=np.float64)
                if rpt.cte_echo
                else np.zeros((0, 2), dtype=np.float64)
            )

        prefix = f"id{aid}_"
        payload[prefix + "slot"] = np.asarray(slots, dtype=np.int32)
        payload[prefix + "slotframe"] = np.asarray(sfs, dtype=np.int32)
        payload[prefix + "x"] = np.asarray(xs, dtype=np.int32)
        payload[prefix + "y"] = np.asarray(ys, dtype=np.int32)
        payload[prefix + "time_diff"] = np.asarray(tds, dtype=np.int32)
        payload[prefix + "rssi"] = np.asarray(rssis, dtype=np.int32)
        payload[prefix + "nib"] = np.asarray(nibs, dtype=np.int32)
        payload[prefix + "necho"] = np.asarray(nechos, dtype=np.int32)
        payload[prefix + "frags"] = np.asarray(frags, dtype=np.int32)
        payload[prefix + "lost"] = np.asarray(lost_flags, dtype=np.int32)
        payload[prefix + "timestamp"] = np.asarray(timestamps)

        max_ref = max((a.shape[0] for a in cte_refs), default=0)
        max_echo = max((a.shape[0] for a in cte_echos), default=0)
        ref_stack = np.full((n_rounds, max_ref, 2), np.nan, dtype=np.float64)
        echo_stack = np.full((n_rounds, max_echo, 2), np.nan, dtype=np.float64)
        for r, arr in enumerate(cte_refs):
            if arr.shape[0] > 0:
                ref_stack[r, : arr.shape[0], :] = arr
        for r, arr in enumerate(cte_echos):
            if arr.shape[0] > 0:
                echo_stack[r, : arr.shape[0], :] = arr
        payload[prefix + "cte_ref"] = ref_stack
        payload[prefix + "cte_echo"] = echo_stack

    np.savez_compressed(npz_path, **payload)
    return npz_path


# ---------------------------------------------------------------------------
# AoA helpers
# ---------------------------------------------------------------------------

def iq_list_to_complex(samples: List[Any]) -> np.ndarray:
    if not samples:
        return np.zeros(0, dtype=np.complex128)
    if isinstance(samples[0], dict):
        ordered = sorted(samples, key=lambda s: int(s["idx"]))
        arr = np.asarray([(s["i"], s["q"]) for s in ordered], dtype=np.float64)
    else:
        arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros(0, dtype=np.complex128)
    return arr[:, 0] + 1j * arr[:, 1]


def compute_div_iq_and_ls(
    cte_a: List[Any],
    cte_b: List[Any],
) -> Tuple[List[Tuple[float, float]], float, float]:
    a_c = iq_list_to_complex(cte_a)
    b_c = iq_list_to_complex(cte_b)
    n = min(len(a_c), len(b_c))
    if n < 2:
        return [], 0.0, 0.0

    div_iq: List[Tuple[float, float]] = []
    phases: List[float] = []
    xs: List[float] = []
    prev = 0.0
    n_valid = 0
    for i in range(n):
        den = b_c[i]
        if abs(den) == 0.0:
            continue
        ratio = a_c[i] / den
        div_iq.append((float(ratio.real), float(ratio.imag)))
        phase = float(np.angle(ratio))
        if n_valid == 0:
            prev = phase
        else:
            delta = phase - prev
            while delta > math.pi:
                delta -= 2.0 * math.pi
            while delta < -math.pi:
                delta += 2.0 * math.pi
            prev = prev + delta
        xs.append(float(i))
        phases.append(prev)
        n_valid += 1

    if n_valid < 2:
        return div_iq, 0.0, 0.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(phases, dtype=np.float64)
    sum_x = float(np.sum(x))
    sum_x2 = float(np.sum(x * x))
    sum_y = float(np.sum(y))
    sum_xy = float(np.sum(x * y))
    denom = n_valid * sum_x2 - sum_x * sum_x
    if denom == 0.0:
        return div_iq, float(sum_y / n_valid), 0.0
    slope = (n_valid * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n_valid
    return div_iq, float(intercept), float(slope)


def report_to_dict(rpt: AnchorReport) -> Dict[str, Any]:
    div_iq, intercept, slope = compute_div_iq_and_ls(rpt.cte_ref, rpt.cte_echo)
    return {
        "id": rpt.anchor_id,
        "slot": rpt.slot,
        "slotframe": rpt.slotframe,
        "x": rpt.x,
        "y": rpt.y,
        "time_diff": rpt.time_diff,
        "rssi": rpt.rssi,
        "nib": rpt.nib,
        "necho": rpt.necho,
        "frags": rpt.frags,
        "cte_ref": list(rpt.cte_ref),
        "cte_echo": list(rpt.cte_echo),
        "div_iq": div_iq,
        "intercept": intercept,
        "slope": slope,
        "ndiv": len(div_iq),
    }


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


def ref_to_anchor_phase(anchor_id: int) -> float:
    return -2.0 * math.pi * F_CENTER_HZ * dist(REF_POSITION, node_xy(anchor_id)) / C_LIGHT


def pair_slope_from_cte(rpt_a: Dict[str, Any], rpt_b: Dict[str, Any]) -> float:
    _, _, slope_ref = compute_div_iq_and_ls(rpt_a.get("cte_ref", []), rpt_b.get("cte_ref", []))
    _, _, slope_echo = compute_div_iq_and_ls(rpt_a.get("cte_echo", []), rpt_b.get("cte_echo", []))
    return 0.5 * (float(slope_ref) + float(slope_echo))


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
    th_raw = aoa_from_pair_phase(phase_rad, baseline_m)
    if not math.isfinite(th_raw):
        return {"aoa_raw_deg": float("nan"), "aoa_deg": float("nan"), "pi_corrected": False}
    if abs(th_raw) < deadzone_deg:
        return {"aoa_raw_deg": th_raw, "aoa_deg": th_raw, "pi_corrected": False}
    on_preferred = th_raw > 0.0 if prefer_positive else th_raw < 0.0
    if on_preferred:
        return {"aoa_raw_deg": th_raw, "aoa_deg": th_raw, "pi_corrected": False}
    th_fix = aoa_from_pair_phase(wrap_phase(phase_rad + math.pi), baseline_m)
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
    phi_m = ref_to_anchor_phase(id_a) - ref_to_anchor_phase(id_b)
    return ratio * np.exp(-1j * phi_cfo) * np.exp(-1j * phi_m)


def _fuse_sin_weighted(aoas_deg: List[float], baselines_m: List[float]) -> float:
    if not aoas_deg:
        return float("nan")
    sins = [math.sin(math.radians(th)) for th in aoas_deg]
    w = np.asarray([b * b for b in baselines_m], dtype=np.float64)
    w /= np.sum(w)
    sin_fused = float(np.clip(np.dot(w, np.asarray(sins, dtype=np.float64)), -1.0, 1.0))
    return float(np.degrees(np.arcsin(sin_fused)))


def orient_ula_ids(ids: List[int]) -> Tuple[int, int, List[int]]:
    pts = [(aid, node_xy(aid)) for aid in ids]
    xs = [p[1][0] for p in pts]
    ys = [p[1][1] for p in pts]
    along_x = (max(xs) - min(xs)) >= (max(ys) - min(ys))
    if along_x:
        ordered = sorted(ids, key=lambda a: (node_xy(a)[0], node_xy(a)[1], a))
    else:
        ordered = sorted(ids, key=lambda a: (node_xy(a)[1], node_xy(a)[0], a))
    pos_ref = node_xy(ordered[0])
    pos_end = node_xy(ordered[-1])
    bx = pos_end[0] - pos_ref[0]
    by = pos_end[1] - pos_ref[1]
    mid_x = 0.5 * (pos_ref[0] + pos_end[0])
    mid_y = 0.5 * (pos_ref[1] + pos_end[1])
    cross = bx * (TARGET_POSITION[1] - mid_y) - by * (TARGET_POSITION[0] - mid_x)
    if cross < 0.0:
        ordered = list(reversed(ordered))
    return ordered[0], ordered[-1], ordered


def true_aoa_deg_for_ula(target_xy: Tuple[float, float], ids: List[int]) -> float:
    _, _, ordered = orient_ula_ids(ids)
    pos_ref = node_xy(ordered[0])
    pos_end = node_xy(ordered[-1])
    mid_x = 0.5 * (pos_ref[0] + pos_end[0])
    mid_y = 0.5 * (pos_ref[1] + pos_end[1])
    bx = pos_end[0] - pos_ref[0]
    by = pos_end[1] - pos_ref[1]
    bl = math.atan2(by, bx)
    nx, ny = math.cos(bl + 0.5 * math.pi), math.sin(bl + 0.5 * math.pi)
    ax, ay = math.cos(bl), math.sin(bl)
    vx, vy = target_xy[0] - mid_x, target_xy[1] - mid_y
    theta = math.degrees(math.atan2(-(vx * ax + vy * ay), vx * nx + vy * ny))
    if theta > 90.0:
        theta = 180.0 - theta
    elif theta < -90.0:
        theta = -180.0 - theta
    return float(np.clip(theta, -90.0, 90.0))


def estimate_group_aoa(
    reports: Dict[int, Dict[str, Any]],
    ids: List[int],
    prefer_positive: bool,
) -> Dict[str, Any]:
    empty = {"aoa_deg": float("nan"), "aoa_raw_deg": float("nan"), "n_pi_corrected": 0}
    if len(ids) < 2:
        return empty
    aoas_raw: List[float] = []
    aoas_fix: List[float] = []
    baselines: List[float] = []
    n_pi = 0
    _, _, ordered = orient_ula_ids(ids)
    rank = {aid: i for i, aid in enumerate(ordered)}
    for id_a, id_b in itertools.combinations(ids, 2):
        if rank[id_a] < rank[id_b]:
            id_a, id_b = id_b, id_a
        baseline = dist(node_xy(id_a), node_xy(id_b))
        if baseline < 1e-9:
            continue
        ratio = corrected_pair_ratio(reports, id_a, id_b)
        if ratio is None:
            continue
        est = aoa_from_pair_phase_half(float(np.angle(np.mean(ratio))), baseline, prefer_positive)
        if not math.isfinite(est["aoa_raw_deg"]):
            continue
        aoas_raw.append(float(est["aoa_raw_deg"]))
        aoas_fix.append(float(est["aoa_deg"]))
        baselines.append(baseline)
        if est["pi_corrected"]:
            n_pi += 1
    if not baselines:
        return empty
    return {
        "aoa_deg": _fuse_sin_weighted(aoas_fix, baselines),
        "aoa_raw_deg": _fuse_sin_weighted(aoas_raw, baselines),
        "n_pi_corrected": n_pi,
    }


def bearing_from_aoa(
    pos_a: Tuple[float, float],
    pos_b: Tuple[float, float],
    aoa_rad: float,
) -> float:
    bx, by = pos_a[0] - pos_b[0], pos_a[1] - pos_b[1]
    return math.atan2(by, bx) + 0.5 * math.pi + aoa_rad


def ray_from_ula(ids: List[int], aoa_deg: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    _, _, ordered = orient_ula_ids(ids)
    pos_ref, pos_end = node_xy(ordered[0]), node_xy(ordered[-1])
    mid = (0.5 * (pos_ref[0] + pos_end[0]), 0.5 * (pos_ref[1] + pos_end[1]))
    bearing = bearing_from_aoa(pos_end, pos_ref, math.radians(aoa_deg))
    return mid, (math.cos(bearing), math.sin(bearing))


def intersect_rays(
    p1: Tuple[float, float],
    d1: Tuple[float, float],
    p2: Tuple[float, float],
    d2: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    a11, a12, a21, a22 = d1[0], -d2[0], d1[1], -d2[1]
    det = a11 * a22 - a12 * a21
    if abs(det) < 1e-15:
        return None
    b1, b2 = p2[0] - p1[0], p2[1] - p1[1]
    t = (b1 * a22 - b2 * a12) / det
    return (p1[0] + t * d1[0], p1[1] + t * d1[1])


def position_from_two_aoas(
    aoa_x_deg: float,
    aoa_y_deg: float,
    x_ids: List[int],
    y_ids: List[int],
) -> Optional[Tuple[float, float]]:
    if not (math.isfinite(aoa_x_deg) and math.isfinite(aoa_y_deg)):
        return None
    o_x, d_x = ray_from_ula(x_ids, aoa_x_deg)
    o_y, d_y = ray_from_ula(y_ids, aoa_y_deg)
    return intersect_rays(o_x, d_x, o_y, d_y)


# ---------------------------------------------------------------------------
# Round processing / collection
# ---------------------------------------------------------------------------

def data_process(
    reports_in: Dict[int, AnchorReport],
    lost: List[int],
    round_idx: int,
) -> Optional[Dict[str, Any]]:
    x_ids, y_ids = group_ids()
    reports = {
        aid: report_to_dict(rpt)
        for aid, rpt in reports_in.items()
        if aid in ANCHOR_LIST and (rpt.cte_ref or rpt.cte_echo)
    }
    if lost:
        print(f"  lost anchors: {lost}")
    if len(reports) < 2:
        return None

    true_x = true_aoa_deg_for_ula(TARGET_POSITION, x_ids)
    true_y = true_aoa_deg_for_ula(TARGET_POSITION, y_ids)
    prefer_x = bool(true_x > 0.0)
    prefer_y = bool(true_y > 0.0)
    x_present = [aid for aid in x_ids if aid in reports]
    y_present = [aid for aid in y_ids if aid in reports]
    est_x = estimate_group_aoa(reports, x_present, prefer_x)
    est_y = estimate_group_aoa(reports, y_present, prefer_y)
    aoa_x = float(est_x["aoa_deg"])
    aoa_y = float(est_y["aoa_deg"])
    est_pos = position_from_two_aoas(aoa_x, aoa_y, x_ids, y_ids)
    err_pos = dist(est_pos, TARGET_POSITION) if est_pos is not None else float("nan")

    result: Dict[str, Any] = {
        "round_idx": round_idx,
        "aoa_x_deg": aoa_x,
        "aoa_y_deg": aoa_y,
        "true_aoa_x_deg": true_x,
        "true_aoa_y_deg": true_y,
        "est_pos": est_pos,
        "pos_err_m": err_pos,
        "lost": list(lost),
    }
    pos_str = f"({est_pos[0]:.3f},{est_pos[1]:.3f})" if est_pos is not None else "None"
    err_str = f"  err={err_pos*100:.1f} cm" if math.isfinite(err_pos) else ""
    print(f"  AoA X={aoa_x:+.2f} Y={aoa_y:+.2f}  pos={pos_str}{err_str}")
    _round_results.append(result)
    return result


def finalize_round(
    sf: Optional[int],
    reports: Dict[int, AnchorReport],
    captured_rounds: List[Dict[int, AnchorReport]],
    lost_rounds: List[List[int]],
) -> bool:
    """Close one slotframe. Missing anchors -> lost. Returns True if done."""
    if not reports:
        return False
    lost = [aid for aid in ANCHOR_LIST if aid not in reports]
    snap = {aid: reports[aid] for aid in sorted(reports)}
    captured_rounds.append(snap)
    lost_rounds.append(lost)
    round_idx = len(captured_rounds)
    print(
        f"round {round_idx}/{CAPTURE_ROUNDS}  sf={sf}  "
        f"got={sorted(snap)}"
        + (f"  lost={lost}" if lost else "")
    )
    data_process(snap, lost, round_idx)
    reports.clear()

    if round_idx >= CAPTURE_ROUNDS:
        path = save_all_rounds(CAPTURE_NPZ_PATH, captured_rounds, lost_rounds)
        print(f"saved {round_idx} rounds -> {path}")
        return True
    return False


def commit_report(
    rpt: AnchorReport,
    state: Dict[str, Any],
    captured_rounds: List[Dict[int, AnchorReport]],
    lost_rounds: List[List[int]],
) -> bool:
    """
    Accumulate reports for the current slotframe (`sf`).
    Finalize when sf changes, same anchor repeats, or all anchors arrived.
    """
    rpt.finalize()
    aid = rpt.anchor_id
    if aid not in ANCHOR_LIST:
        return False

    cur_sf: Optional[int] = state["sf"]
    bag: Dict[int, AnchorReport] = state["reports"]
    required = set(ANCHOR_LIST)

    def start(sf: int, first: AnchorReport) -> None:
        state["sf"] = sf
        bag.clear()
        bag[first.anchor_id] = first

    if cur_sf is None:
        start(rpt.slotframe, rpt)
    elif rpt.slotframe != cur_sf or aid in bag:
        done = finalize_round(cur_sf, bag, captured_rounds, lost_rounds)
        state["sf"] = None
        if done:
            return True
        start(rpt.slotframe, rpt)
    else:
        bag[aid] = rpt

    print(f"  sf={state['sf']} id={aid}  have={sorted(bag)}")
    if set(bag) >= required:
        done = finalize_round(state["sf"], bag, captured_rounds, lost_rounds)
        state["sf"] = None
        return done
    return False


# ---------------------------------------------------------------------------
# UART
# ---------------------------------------------------------------------------

def coral_listener(ser: serial.Serial) -> None:
    global _round_results
    _round_results = []

    raw = bytearray()
    state: Dict[str, Any] = {"sf": None, "reports": {}}
    pending: Optional[AnchorReport] = None
    captured_rounds: List[Dict[int, AnchorReport]] = []
    lost_rounds: List[List[int]] = []

    print(f"out={CAPTURE_NPZ_PATH}  rounds={CAPTURE_ROUNDS}")
    print(f"anchors={ANCHOR_LIST}")

    while True:
        b = ser.read(1)
        if not b:
            continue
        raw += b
        if len(raw) < 2 or raw[-2:] != b"\r\n":
            continue
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        raw = bytearray()
        if not line.strip():
            continue

        rpt = parse_rpt_line(line)
        if rpt is not None:
            pending = rpt
            if pending.nib == 0 and pending.necho == 0:
                if commit_report(pending, state, captured_rounds, lost_rounds):
                    return
                pending = None
            continue

        hdr = parse_cte_hdr(line)
        if hdr is not None:
            if pending is not None:
                pending._section = str(hdr["tag"])
            continue

        sample = parse_cte_sample(line)
        if sample is not None:
            if pending is None or pending._section is None:
                continue
            if pending._section == "CTE_REF":
                pending._cte_ref_raw.append(sample)
            elif pending._section == "CTE_ECHO":
                pending._cte_echo_raw.append(sample)
            if pending.is_complete():
                if commit_report(pending, state, captured_rounds, lost_rounds):
                    return
                pending = None


def main() -> int:
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
    print(f"listen {SERIAL_PORT}@{SERIAL_BAUD}")
    print(f"REF={_pos_token(REF_POSITION)} TGT={_pos_token(TARGET_POSITION)} "
          f"structure={ANCHOR_STRUCTURE_ID}")
    try:
        coral_listener(ser)
    except KeyboardInterrupt:
        print("stopped")
    finally:
        ser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
