#!/usr/bin/env python3
"""
two_anchor_listener.py — two-anchor CoRal UART listener (AoA only).

Firmware lines (same as new_listener.py):
  RPT id=2 slot=1 x=0 y=0 td=8171 rssi=-27 nib=88 necho=88 frags=8
  CTE_REF n=88
    idx=0 I=123 Q=-45
  CTE_ECHO n=88
    idx=0 I=... Q=...

Architecture:
  Collect one complete report per anchor in ANCHOR_LIST each round.
  data_process():
    - form a 2-element baseline (ids 2, 3)
    - estimate target AoA (pair phase + half-plane prior)
    - live-plot each round's estimated angle vs true angle
  After CAPTURE_ROUNDS complete rounds, save:
    CoRal-simulation/indoor_2anchor_coral/
      {ref_distance}_{ref_angle}_{target_distance}_{target_angle}_reports.npz
"""

from __future__ import annotations

import datetime
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import serial
except ImportError:
    print("Missing dependency: pyserial  (pip install pyserial)", file=sys.stderr)
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Missing dependency: numpy  (pip install numpy)", file=sys.stderr)
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Serial / capture config
# ---------------------------------------------------------------------------

SERIAL_PORT = "COM4"
SERIAL_BAUD = 115200

# Two firmware anchors forming one baseline.
ANCHOR_LIST = [2, 3]

PRINT_EACH_CTE = False
CAPTURE_ROUNDS = 50
ENABLE_VISUALIZATION = True

# Skip π half-plane flip when |θ_raw| is below this (deg).
HALF_PLANE_DEADZONE_DEG = 12.0

# RF / timing (CHANNEL=17 → NRF FREQUENCY=40 → 2440 MHz).
F_CENTER_HZ = 2.440e9
C_LIGHT = 2.99792458e8
TIMER_HZ = 16e6
SAMPLE_INTERVAL_S = 1e-6

# Experiment geometry in polar coords relative to the array midpoint.
# Angle is CCW from +x (deg). Positions = (r cos θ, r sin θ).
REF_DISTANCE_M = 0.30
REF_ANGLE_DEG = 0.0
TARGET_DISTANCE_M = 0.40
TARGET_ANGLE_DEG = 0.0

# Two-anchor baseline along +x, centered at origin (meters).
BASELINE_M = 0.06
ANCHOR_POSITIONS: Dict[int, Tuple[float, float]] = {
    2: (-0.5 * BASELINE_M, 0.0),
    3: (0.5 * BASELINE_M, 0.0),
}


def polar_to_xy(distance_m: float, angle_deg: float) -> Tuple[float, float]:
    th = math.radians(angle_deg)
    return (float(distance_m * math.cos(th)), float(distance_m * math.sin(th)))


def ref_position() -> Tuple[float, float]:
    return polar_to_xy(REF_DISTANCE_M, REF_ANGLE_DEG)


def target_position() -> Tuple[float, float]:
    return polar_to_xy(TARGET_DISTANCE_M, TARGET_ANGLE_DEG)


def node_xy(node_id: int) -> Tuple[float, float]:
    if node_id not in ANCHOR_POSITIONS:
        raise KeyError(f"Unknown anchor id={node_id}")
    return ANCHOR_POSITIONS[node_id]


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _geom_token(distance_m: float, angle_deg: float) -> str:
    """Format one (distance, angle) pair for filenames."""
    return f"{distance_m:.2f}_{angle_deg:.1f}"


def build_capture_npz_path() -> str:
    """
    indoor_2anchor_coral/{ref_d}_{ref_a}_{tgt_d}_{tgt_a}_reports.npz
    """
    base = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "indoor_2anchor_coral",
    )
    os.makedirs(base, exist_ok=True)
    filename = (
        f"{_geom_token(REF_DISTANCE_M, REF_ANGLE_DEG)}_"
        f"{_geom_token(TARGET_DISTANCE_M, TARGET_ANGLE_DEG)}_"
        f"reports.npz"
    )
    return os.path.join(base, filename)


CAPTURE_NPZ_PATH = build_capture_npz_path()

_round_results: List[Dict[str, Any]] = []
_live_fig = None
_live_axes = None


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------

IqSample = Tuple[float, float]


@dataclass
class AnchorReport:
    """One complete report from a single anchor."""

    anchor_id: int
    slot: int
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
    raw_rpt: str = ""

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

    def to_save_dict(self) -> Dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "slot": self.slot,
            "x": self.x,
            "y": self.y,
            "time_diff": self.time_diff,
            "rssi": self.rssi,
            "nib": self.nib,
            "necho": self.necho,
            "frags": self.frags,
            "timestamp": self.timestamp,
            "cte_ref": self.cte_ref,
            "cte_echo": self.cte_echo,
            "raw_rpt": self.raw_rpt,
        }


# ---------------------------------------------------------------------------
# Line parsers
# ---------------------------------------------------------------------------

RPT_RE = re.compile(
    r"^RPT\s+"
    r"id=(?P<id>\d+)\s+"
    r"slot=(?P<slot>\d+)\s+"
    r"x=(?P<x>-?\d+)\s+"
    r"y=(?P<y>-?\d+)\s+"
    r"td=(?P<td>-?\d+)\s+"
    r"rssi=(?P<rssi>-?\d+)\s+"
    r"nib=(?P<nib>\d+)\s+"
    r"necho=(?P<necho>\d+)\s+"
    r"frags=(?P<frags>\d+)\s*$"
)

CTE_HDR_RE = re.compile(
    r"^(?P<tag>CTE_REF|CTE_ECHO)\s+n=(?P<n>\d+)\s*$"
)

CTE_SAMPLE_RE = re.compile(
    r"^\s*idx=(?P<idx>\d+)\s+"
    r"I=(?P<I>-?\d+)\s+"
    r"Q=(?P<Q>-?\d+)\s*$"
)


def parse_rpt_line(line: str) -> Optional[AnchorReport]:
    m = RPT_RE.match(line.strip())
    if not m:
        return None
    return AnchorReport(
        anchor_id=int(m.group("id")),
        slot=int(m.group("slot")),
        x=int(m.group("x")),
        y=int(m.group("y")),
        time_diff=int(m.group("td")),
        rssi=int(m.group("rssi")),
        nib=int(m.group("nib")),
        necho=int(m.group("necho")),
        frags=int(m.group("frags")),
        timestamp=datetime.datetime.now().isoformat(timespec="milliseconds"),
        raw_rpt=line.strip(),
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
    return {
        "idx": float(m.group("idx")),
        "i": float(m.group("I")),
        "q": float(m.group("Q")),
    }


def format_report(rpt: AnchorReport) -> str:
    return (
        f"[{rpt.timestamp}] "
        f"id={rpt.anchor_id} slot={rpt.slot} "
        f"pos=({rpt.x},{rpt.y}) "
        f"td={rpt.time_diff} rssi={rpt.rssi} "
        f"nib={rpt.nib} necho={rpt.necho} frags={rpt.frags} "
        f"cte_ref={len(rpt.cte_ref)} cte_echo={len(rpt.cte_echo)}"
    )


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def write_session_meta(npz_path: str) -> None:
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    ref_xy = ref_position()
    tgt_xy = target_position()
    meta = {
        "ref_distance_m": REF_DISTANCE_M,
        "ref_angle_deg": REF_ANGLE_DEG,
        "target_distance_m": TARGET_DISTANCE_M,
        "target_angle_deg": TARGET_ANGLE_DEG,
        "ref_position": list(ref_xy),
        "target_position": list(tgt_xy),
        "baseline_m": BASELINE_M,
        "anchor_list": list(ANCHOR_LIST),
        "anchor_positions": {
            str(k): list(v) for k, v in ANCHOR_POSITIONS.items()
        },
        "capture_rounds": CAPTURE_ROUNDS,
        "f_center_hz": F_CENTER_HZ,
        "npz_path": npz_path,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = npz_path.replace("_reports.npz", "_meta.json")
    if meta_path == npz_path:
        meta_path = npz_path + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Session meta: {meta_path}")


def save_all_rounds(
    npz_path: str,
    captured_rounds: List[Dict[int, AnchorReport]],
) -> str:
    os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
    n_rounds = len(captured_rounds)
    ref_xy = ref_position()
    tgt_xy = target_position()

    payload: Dict[str, Any] = {
        "n_rounds": np.int32(n_rounds),
        "anchor_list": np.asarray(ANCHOR_LIST, dtype=np.int32),
        "ref_distance_m": np.float64(REF_DISTANCE_M),
        "ref_angle_deg": np.float64(REF_ANGLE_DEG),
        "target_distance_m": np.float64(TARGET_DISTANCE_M),
        "target_angle_deg": np.float64(TARGET_ANGLE_DEG),
        "ref_position": np.asarray(ref_xy, dtype=np.float64),
        "target_position": np.asarray(tgt_xy, dtype=np.float64),
        "baseline_m": np.float64(BASELINE_M),
        "saved_at": np.asarray(
            datetime.datetime.now().isoformat(timespec="milliseconds")
        ),
    }

    for aid in ANCHOR_LIST:
        slots: List[int] = []
        xs: List[int] = []
        ys: List[int] = []
        tds: List[int] = []
        rssis: List[int] = []
        nibs: List[int] = []
        nechos: List[int] = []
        frags: List[int] = []
        timestamps: List[str] = []
        cte_refs: List[np.ndarray] = []
        cte_echos: List[np.ndarray] = []

        for reports in captured_rounds:
            rpt = reports.get(aid)
            if rpt is None:
                slots.append(0)
                xs.append(0)
                ys.append(0)
                tds.append(0)
                rssis.append(0)
                nibs.append(0)
                nechos.append(0)
                frags.append(0)
                timestamps.append("")
                cte_refs.append(np.zeros((0, 2), dtype=np.float64))
                cte_echos.append(np.zeros((0, 2), dtype=np.float64))
                continue

            slots.append(rpt.slot)
            xs.append(rpt.x)
            ys.append(rpt.y)
            tds.append(rpt.time_diff)
            rssis.append(rpt.rssi)
            nibs.append(rpt.nib)
            nechos.append(rpt.necho)
            frags.append(rpt.frags)
            timestamps.append(rpt.timestamp)
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
        payload[prefix + "x"] = np.asarray(xs, dtype=np.int32)
        payload[prefix + "y"] = np.asarray(ys, dtype=np.int32)
        payload[prefix + "time_diff"] = np.asarray(tds, dtype=np.int32)
        payload[prefix + "rssi"] = np.asarray(rssis, dtype=np.int32)
        payload[prefix + "nib"] = np.asarray(nibs, dtype=np.int32)
        payload[prefix + "necho"] = np.asarray(nechos, dtype=np.int32)
        payload[prefix + "frags"] = np.asarray(frags, dtype=np.int32)
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
# Signal / AoA helpers
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
    return (
        -2.0
        * math.pi
        * F_CENTER_HZ
        * dist(ref_position(), node_xy(anchor_id))
        / C_LIGHT
    )


def pair_slope_from_cte(rpt_a: Dict[str, Any], rpt_b: Dict[str, Any]) -> float:
    _, _, slope_ref = compute_div_iq_and_ls(
        rpt_a.get("cte_ref", []), rpt_b.get("cte_ref", [])
    )
    _, _, slope_echo = compute_div_iq_and_ls(
        rpt_a.get("cte_echo", []), rpt_b.get("cte_echo", [])
    )
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


def orient_ula_ids(ids: List[int]) -> Tuple[int, int, List[int]]:
    """
    Order baseline along principal axis so TARGET lies on +normal
    (baseline +90 deg CCW). Returns (ref_id, end_id, ordered_ids).
    """
    pts = [(aid, node_xy(aid)) for aid in ids]
    xs = [p[1][0] for p in pts]
    ys = [p[1][1] for p in pts]
    along_x = (max(xs) - min(xs)) >= (max(ys) - min(ys))
    if along_x:
        ordered = sorted(ids, key=lambda a: (node_xy(a)[0], node_xy(a)[1], a))
    else:
        ordered = sorted(ids, key=lambda a: (node_xy(a)[1], node_xy(a)[0], a))

    tgt = target_position()
    pos_ref = node_xy(ordered[0])
    pos_end = node_xy(ordered[-1])
    bx = pos_end[0] - pos_ref[0]
    by = pos_end[1] - pos_ref[1]
    mid_x = 0.5 * (pos_ref[0] + pos_end[0])
    mid_y = 0.5 * (pos_ref[1] + pos_end[1])
    cross = bx * (tgt[1] - mid_y) - by * (tgt[0] - mid_x)
    if cross < 0.0:
        ordered = list(reversed(ordered))
    return ordered[0], ordered[-1], ordered


def true_aoa_deg(target_xy: Tuple[float, float], ids: List[int]) -> float:
    """Geometric ULA AoA in [-90, 90] matching bearing_from_aoa convention."""
    _, _, ordered = orient_ula_ids(ids)
    pos_ref = node_xy(ordered[0])
    pos_end = node_xy(ordered[-1])
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


def estimate_pair_aoa(
    reports: Dict[int, Dict[str, Any]],
    prefer_positive: bool,
) -> Dict[str, Any]:
    """Single-baseline AoA from the two anchors."""
    ids = list(ANCHOR_LIST)
    _, _, ordered = orient_ula_ids(ids)
    id_b, id_a = ordered[0], ordered[-1]  # ref → end
    pos_a = node_xy(id_a)
    pos_b = node_xy(id_b)
    baseline = dist(pos_a, pos_b)
    if baseline < 1e-9:
        return {
            "aoa_deg": float("nan"),
            "aoa_raw_deg": float("nan"),
            "pi_corrected": False,
            "baseline_m": baseline,
            "id_a": id_a,
            "id_b": id_b,
        }

    ratio = corrected_pair_ratio(reports, id_a, id_b)
    if ratio is None:
        return {
            "aoa_deg": float("nan"),
            "aoa_raw_deg": float("nan"),
            "pi_corrected": False,
            "baseline_m": baseline,
            "id_a": id_a,
            "id_b": id_b,
        }

    phase = float(np.angle(np.mean(ratio)))
    est = aoa_from_pair_phase_half(phase, baseline, prefer_positive)
    return {
        "aoa_deg": float(est["aoa_deg"]),
        "aoa_raw_deg": float(est["aoa_raw_deg"]),
        "pi_corrected": bool(est["pi_corrected"]),
        "baseline_m": baseline,
        "id_a": id_a,
        "id_b": id_b,
        "phase_rad": phase,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _ensure_live_figure() -> None:
    global _live_fig, _live_axes
    if plt is None:
        return
    if _live_fig is None or _live_axes is None:
        plt.ion()
        _live_fig, _live_axes = plt.subplots(1, 2, figsize=(11, 4.5))
        try:
            _live_fig.canvas.manager.set_window_title("Two-anchor AoA")  # type: ignore[union-attr]
        except Exception:
            pass


def visualize_round(result: Dict[str, Any]) -> None:
    """Live 1x2: AoA vs round, and geometry with current bearing ray."""
    if not ENABLE_VISUALIZATION or plt is None:
        return
    _ensure_live_figure()
    assert _live_fig is not None and _live_axes is not None
    axes = _live_axes

    aoa = result["aoa_deg"]
    aoa_raw = result["aoa_raw_deg"]
    true_aoa = result["true_aoa_deg"]
    hist = result.get("history") or []

    # --- AoA history ---
    ax = axes[0]
    ax.clear()
    if hist:
        rounds = [h["round_idx"] for h in hist]
        ax.plot(
            rounds,
            [h["aoa_deg"] for h in hist],
            "o-",
            ms=4,
            color="C0",
            label="est AoA",
        )
        ax.plot(
            rounds,
            [h["aoa_raw_deg"] for h in hist],
            "x--",
            ms=4,
            color="C1",
            alpha=0.7,
            label="raw AoA",
        )
    ax.axhline(
        true_aoa,
        color="C3",
        ls="-",
        lw=1.5,
        label=f"true {true_aoa:+.1f} deg",
    )
    ax.axhline(0.0, color="gray", ls=":", lw=1.0)
    ax.set_ylim(-95, 95)
    ax.set_xlabel("round")
    ax.set_ylabel("AoA [deg]")
    err = aoa - true_aoa if math.isfinite(aoa) else float("nan")
    ax.set_title(
        f"Target AoA  now={aoa:+.1f} deg  "
        f"(raw {aoa_raw:+.1f}, err {err:+.1f})"
        if math.isfinite(err)
        else f"Target AoA  now={aoa:+.1f} deg"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    # --- Geometry ---
    ax = axes[1]
    ax.clear()
    ids = list(ANCHOR_LIST)
    _, _, ordered = orient_ula_ids(ids)
    pos_ref = node_xy(ordered[0])
    pos_end = node_xy(ordered[-1])
    mid = (0.5 * (pos_ref[0] + pos_end[0]), 0.5 * (pos_ref[1] + pos_end[1]))
    ax.plot(
        [node_xy(a)[0] for a in ordered],
        [node_xy(a)[1] for a in ordered],
        "o-",
        color="C0",
        ms=8,
        label="baseline",
    )
    for aid in ordered:
        px, py = node_xy(aid)
        ax.annotate(str(aid), (px, py), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ref_xy = ref_position()
    tgt_xy = target_position()
    ax.plot(ref_xy[0], ref_xy[1], "^", color="gray", ms=10, label="REF")
    ax.plot(tgt_xy[0], tgt_xy[1], "*", color="C3", ms=14, label="true tgt")

    if math.isfinite(aoa):
        bearing = bearing_from_aoa(pos_end, pos_ref, math.radians(aoa))
        ray_len = max(TARGET_DISTANCE_M, REF_DISTANCE_M, 0.5) * 1.15
        ax.plot(
            [mid[0], mid[0] + ray_len * math.cos(bearing)],
            [mid[1], mid[1] + ray_len * math.sin(bearing)],
            "--",
            color="C0",
            lw=1.4,
            label="est ray",
        )
    # True ray
    true_bearing = bearing_from_aoa(pos_end, pos_ref, math.radians(true_aoa))
    ray_len = max(TARGET_DISTANCE_M, REF_DISTANCE_M, 0.5) * 1.15
    ax.plot(
        [mid[0], mid[0] + ray_len * math.cos(true_bearing)],
        [mid[1], mid[1] + ray_len * math.sin(true_bearing)],
        "-",
        color="C3",
        lw=1.0,
        alpha=0.7,
        label="true ray",
    )

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(
        f"Geometry  tgt=({TARGET_DISTANCE_M:.2f}m,{TARGET_ANGLE_DEG:.1f}°)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")

    _live_fig.suptitle(f"round {result['round_idx']} / {CAPTURE_ROUNDS}")
    _live_fig.tight_layout()
    _live_fig.canvas.draw_idle()
    plt.pause(0.05)


def visualize_summary(results: List[Dict[str, Any]]) -> None:
    if not results or plt is None:
        return
    aoas = np.asarray([r["aoa_deg"] for r in results], dtype=np.float64)
    raws = np.asarray([r["aoa_raw_deg"] for r in results], dtype=np.float64)
    true_aoa = float(results[0]["true_aoa_deg"])
    errs = aoas - true_aoa

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bins = np.linspace(-90, 90, 19)

    ax = axes[0]
    ax.hist(aoas[np.isfinite(aoas)], bins=bins, color="C0", edgecolor="k", alpha=0.85)
    ax.axvline(true_aoa, color="C3", lw=2.0, label="true")
    ax.set_title(
        f"AoA hist  mean={np.nanmean(aoas):+.1f} deg  "
        f"std={np.nanstd(aoas):.1f}"
    )
    ax.set_xlabel("AoA [deg]")
    ax.legend(fontsize=8)

    ax = axes[1]
    rounds = [r["round_idx"] for r in results]
    ax.plot(rounds, aoas, "o-", ms=4, color="C0", label="est")
    ax.plot(rounds, raws, "x--", ms=3, color="C1", alpha=0.6, label="raw")
    ax.axhline(true_aoa, color="C3", lw=1.5, label="true")
    ax.set_ylim(-95, 95)
    ax.set_xlabel("round")
    ax.set_ylabel("AoA [deg]")
    finite_err = errs[np.isfinite(errs)]
    mean_err = float(np.mean(finite_err)) if len(finite_err) else float("nan")
    ax.set_title(f"Per-round AoA  mean err={mean_err:+.1f} deg  N={len(results)}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(
        f"Summary  ref=({REF_DISTANCE_M:.2f}m,{REF_ANGLE_DEG:.1f}°)  "
        f"tgt=({TARGET_DISTANCE_M:.2f}m,{TARGET_ANGLE_DEG:.1f}°)"
    )
    fig.tight_layout()
    print("Close summary figure to exit.")
    plt.show(block=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def data_process(
    reports_in: Dict[int, AnchorReport],
    round_idx: int,
) -> Optional[Dict[str, Any]]:
    """Estimate target AoA from the two-anchor baseline; plot this round."""
    reports = {
        aid: report_to_dict(rpt)
        for aid, rpt in reports_in.items()
        if aid in ANCHOR_LIST
    }
    missing = [
        aid
        for aid in ANCHOR_LIST
        if aid not in reports or not reports[aid].get("div_iq")
    ]
    if missing:
        print(f"  data_process skip: missing/empty div_iq for {missing}")
        return None

    tgt = target_position()
    true_th = true_aoa_deg(tgt, ANCHOR_LIST)
    prefer_positive = bool(true_th > 0.0)
    print(
        f"  half-plane from TARGET: "
        f"{'theta>0' if prefer_positive else 'theta<0'} "
        f"(true {true_th:+.2f} deg)"
    )

    est = estimate_pair_aoa(reports, prefer_positive)
    aoa = float(est["aoa_deg"])
    aoa_raw = float(est["aoa_raw_deg"])
    err = aoa - true_th if math.isfinite(aoa) else float("nan")

    result: Dict[str, Any] = {
        "round_idx": round_idx,
        "aoa_deg": aoa,
        "aoa_raw_deg": aoa_raw,
        "true_aoa_deg": true_th,
        "aoa_err_deg": err,
        "pi_corrected": bool(est["pi_corrected"]),
        "baseline_m": float(est["baseline_m"]),
        "id_a": int(est["id_a"]),
        "id_b": int(est["id_b"]),
    }

    print(
        f"  AoA est={aoa:+.2f} deg  raw={aoa_raw:+.2f}  "
        f"true={true_th:+.2f}  err={err:+.2f}  "
        f"+pi={'yes' if result['pi_corrected'] else 'no'}  "
        f"pair={result['id_b']}→{result['id_a']}"
        if math.isfinite(err)
        else f"  AoA est={aoa:+.2f} raw={aoa_raw:+.2f} true={true_th:+.2f}"
    )

    _round_results.append(result)
    result["history"] = list(_round_results)
    visualize_round(result)
    return result


def commit_report(
    rpt: AnchorReport,
    latest_reports: Dict[int, AnchorReport],
    updated: Set[int],
    required: Set[int],
    captured_rounds: List[Dict[int, AnchorReport]],
) -> bool:
    """
    Commit one anchor report. When both anchors are ready:
      1) data_process() → AoA + live plot
      2) append round snapshot
      3) after CAPTURE_ROUNDS, write combined npz
    Returns True when capture is finished.
    """
    rpt.finalize()
    print(format_report(rpt))

    aid = rpt.anchor_id
    if aid not in required:
        return False

    latest_reports[aid] = rpt
    updated.add(aid)
    missing = sorted(required - updated)
    if missing:
        print(f"  collected {sorted(updated)}  waiting for {missing}")
        return False

    round_snapshot = {k: latest_reports[k] for k in ANCHOR_LIST if k in latest_reports}
    captured_rounds.append(round_snapshot)
    round_idx = len(captured_rounds)
    print(f"  round {round_idx}/{CAPTURE_ROUNDS} complete")
    data_process(latest_reports, round_idx=round_idx)

    updated.clear()

    if round_idx >= CAPTURE_ROUNDS:
        out_path = save_all_rounds(CAPTURE_NPZ_PATH, captured_rounds)
        print(f"  saved {round_idx} rounds -> {out_path}")
        if ENABLE_VISUALIZATION:
            visualize_summary(_round_results)
        return True
    return False


# ---------------------------------------------------------------------------
# UART loop
# ---------------------------------------------------------------------------

def coral_listener(ser: serial.Serial) -> None:
    global _round_results, _live_fig, _live_axes
    _round_results = []
    _live_fig = None
    _live_axes = None

    raw_line = bytearray()
    latest_reports: Dict[int, AnchorReport] = {}
    updated: Set[int] = set()
    required = set(ANCHOR_LIST)
    pending: Optional[AnchorReport] = None
    captured_rounds: List[Dict[int, AnchorReport]] = []

    write_session_meta(CAPTURE_NPZ_PATH)
    tgt = target_position()
    print(f"Output file: {CAPTURE_NPZ_PATH}")
    print(f"Will collect {CAPTURE_ROUNDS} rounds, then save one npz.")
    print(f"Anchors: {ANCHOR_LIST}  baseline={BASELINE_M*1000:.1f} mm")
    print(
        f"REF  r={REF_DISTANCE_M:.2f} m  θ={REF_ANGLE_DEG:.1f} deg  "
        f"xy=({ref_position()[0]:.3f},{ref_position()[1]:.3f})"
    )
    print(
        f"TGT  r={TARGET_DISTANCE_M:.2f} m  θ={TARGET_ANGLE_DEG:.1f} deg  "
        f"xy=({tgt[0]:.3f},{tgt[1]:.3f})"
    )
    print(f"true AoA = {true_aoa_deg(tgt, ANCHOR_LIST):+.2f} deg")

    while True:
        byte = ser.read(1)
        if not byte:
            continue

        raw_line += byte
        if len(raw_line) < 2 or raw_line[-2:] != b"\r\n":
            continue

        try:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        except Exception:
            raw_line = bytearray()
            continue

        raw_line = bytearray()
        if not line.strip():
            continue

        rpt = parse_rpt_line(line)
        if rpt is not None:
            if pending is not None:
                print(
                    f"[WARN] incomplete report id={pending.anchor_id} "
                    f"ref={len(pending._cte_ref_raw)}/{pending.nib} "
                    f"echo={len(pending._cte_echo_raw)}/{pending.necho}; "
                    f"discarding"
                )
            pending = rpt
            if pending.nib == 0 and pending.necho == 0:
                if commit_report(
                    pending, latest_reports, updated, required, captured_rounds
                ):
                    print(f"Captured {CAPTURE_ROUNDS} rounds; stopping.")
                    return
                pending = None
            continue

        hdr = parse_cte_hdr(line)
        if hdr is not None:
            if pending is None:
                print(f"[RAW] orphan {hdr['tag']}: {line.strip()}")
                continue
            pending._section = str(hdr["tag"])
            if PRINT_EACH_CTE:
                print(f"  {hdr['tag']} n={hdr['n']} (id={pending.anchor_id})")
            continue

        sample = parse_cte_sample(line)
        if sample is not None:
            if pending is None or pending._section is None:
                print(f"[RAW] orphan CTE sample: {line.strip()}")
                continue
            if pending._section == "CTE_REF":
                pending._cte_ref_raw.append(sample)
            elif pending._section == "CTE_ECHO":
                pending._cte_echo_raw.append(sample)
            else:
                print(f"[RAW] sample in unknown section: {line.strip()}")
                continue

            if PRINT_EACH_CTE:
                print(
                    f"  {pending._section} idx={int(sample['idx'])} "
                    f"I={sample['i']} Q={sample['q']}"
                )

            if pending.is_complete():
                if commit_report(
                    pending, latest_reports, updated, required, captured_rounds
                ):
                    print(f"Captured {CAPTURE_ROUNDS} rounds; stopping.")
                    return
                pending = None
            continue

        print(f"[RAW] {line.strip()}")


def main() -> int:
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
    except serial.SerialException as exc:
        print(f"Failed to open {SERIAL_PORT}: {exc}", file=sys.stderr)
        return 1

    print(f"Listening on {SERIAL_PORT} @ {SERIAL_BAUD}. Ctrl+C to stop.")
    print(f"Waiting for anchors: {ANCHOR_LIST}")
    print(
        f"REF=({REF_DISTANCE_M:.2f}m,{REF_ANGLE_DEG:.1f}°)  "
        f"TGT=({TARGET_DISTANCE_M:.2f}m,{TARGET_ANGLE_DEG:.1f}°)"
    )
    print(f"Output: {CAPTURE_NPZ_PATH}")
    print(f"Visualization: {'ON' if ENABLE_VISUALIZATION else 'OFF'}")
    try:
        coral_listener(ser)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
