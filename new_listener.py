#!/usr/bin/env python3
"""
new_listener.py — parse CoRal_listen_node UART (raw CTE report format).

Firmware lines:
  RPT id=2 slot=1 x=0 y=0 td=8171 rssi=-27 nib=88 necho=88 frags=8
  CTE_REF n=88
    idx=0 I=123 Q=-45
    ...
  CTE_ECHO n=88
    idx=0 I=... Q=...
    ...

Architecture:
  Collect one complete report per anchor in ANCHOR_LIST each round.
  data_process():
    - split anchors into X-ULA (ids 2..5) and Y-ULA (ids 6..9)
    - estimate AoA on each axis (pair-fusion + half-plane prior)
    - intersect the two bearing rays to get target position
  After CAPTURE_ROUNDS complete rounds, save all reports into one file:
    CoRal-simulation/indoor_coral/{ref_pos}_{tgt_pos}_{structure}_reports.npz
"""

from __future__ import annotations

import datetime
import itertools
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

SERIAL_PORT = "COM9"
SERIAL_BAUD = 115200

# Must match UART `id=` (firmware anchors are 2..9).
ANCHOR_LIST = [2, 3, 4, 5, 6, 7, 8, 9]
# X-axis ULA vs Y-axis ULA.
ANCHOR_X_IDS = [2, 3, 4, 5]
ANCHOR_Y_IDS = [6, 7, 8, 9]

PRINT_EACH_CTE = False

# Number of full rounds (all anchors) to save before exiting.
CAPTURE_ROUNDS = 50

# Show per-round figure (AoA + position). Set False for headless capture.
ENABLE_VISUALIZATION = True

# Skip π half-plane flip when |θ_raw| is below this (deg).
# Near-zero raw estimates are noise across broadside, not front/back ambiguity.
HALF_PLANE_DEADZONE_DEG = 12.0

# RF / timing (CHANNEL=17 → NRF FREQUENCY=40 → 2440 MHz).
F_CENTER_HZ = 2.440e9
C_LIGHT = 2.99792458e8
TIMER_HZ = 16e6
SAMPLE_INTERVAL_S = 1e-6

# Experiment geometry (meters). Used for naming / metadata / half-plane prior.
REF_POSITION: Tuple[float, float] = (0.0, 0.0)
TARGET_POSITION: Tuple[float, float] = (0.1, 0.1)

# Select which layout is deployed (key of ANCHOR_STRUCTURES).
ANCHOR_STRUCTURE_ID = 2

# Structure id → concrete anchor placement (meters).
# Default: 4 along +x, 4 along +y (L-shape, 25 mm spacing).
ANCHOR_STRUCTURES: Dict[int, Dict[str, Any]] = {
    1: {
        "name": "CROSS_8_d25mm",
        "description": "X-ULA ids 2..5 along +x; Y-ULA ids 6..9 along +y; d=25 mm",
        "positions": {
            2: (0.025, 0.00),
            3: (0.05, 0.00),
            4: (0.075, 0.00),
            5: (0.1, 0.00),
            6: (0.00, 0.025),
            7: (0.00, 0.05),
            8: (0.00, 0.075),
            9: (0.00, 0.10),
        },
        "x_ids": [2, 3, 4, 5],
        "y_ids": [6, 7, 8, 9],
    },
    2: {
        "name": "CROSS_8_d50mm",
        "description": "X-ULA ids 2..5 along +x; Y-ULA ids 6..9 along +y; d=50 mm",
        "positions": {
            2: (-0.0375, -0.1),
            3: (-0.0125, -0.1),
            4: (0.0125, -0.1),
            5: (0.0375, -0.1),
            6: (-0.1, -0.0375),
            7: (-0.1, -0.0125),
            8: (-0.1, 0.0125),
            9: (-0.1, 0.0375),
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
    x_ids = [int(a) for a in st.get("x_ids", ANCHOR_X_IDS)]
    y_ids = [int(a) for a in st.get("y_ids", ANCHOR_Y_IDS)]
    return x_ids, y_ids


def node_xy(node_id: int) -> Tuple[float, float]:
    pos = anchor_positions()
    if node_id not in pos:
        raise KeyError(f"Unknown anchor id={node_id} in structure {ANCHOR_STRUCTURE_ID}")
    return pos[node_id]


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _pos_token(pos: Tuple[float, float]) -> str:
    """Format (x, y) for folder names, e.g. (0.00,0.10)."""
    return f"({pos[0]:.2f},{pos[1]:.2f})"


def build_capture_npz_path() -> str:
    """
    CoRal-simulation/indoor_coral/{ref_position}_{target_position}_{structure}_reports.npz
    """
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indoor_coral")
    os.makedirs(base, exist_ok=True)
    filename = (
        f"{_pos_token(REF_POSITION)}_"
        f"{_pos_token(TARGET_POSITION)}_"
        f"{ANCHOR_STRUCTURE_ID}_reports.npz"
    )
    return os.path.join(base, filename)


CAPTURE_NPZ_PATH = build_capture_npz_path()

# Per-round estimate history (for end-of-capture summary plot).
_round_results: List[Dict[str, Any]] = []
_live_fig = None
_live_axes = None


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------

IqSample = Tuple[float, float]  # (I, Q)


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

    # Internal assembly state (not part of the committed snapshot).
    _section: Optional[str] = field(default=None, repr=False)
    _cte_ref_raw: List[Dict[str, float]] = field(default_factory=list, repr=False)
    _cte_echo_raw: List[Dict[str, float]] = field(default_factory=list, repr=False)

    def is_complete(self) -> bool:
        return (
            len(self._cte_ref_raw) >= self.nib
            and len(self._cte_echo_raw) >= self.necho
        )

    def finalize(self) -> None:
        """Sort raw CTE samples into (I, Q) lists."""

        def to_pairs(raw: List[Dict[str, float]]) -> List[IqSample]:
            ordered = sorted(raw, key=lambda s: int(s["idx"]))
            return [(float(s["i"]), float(s["q"])) for s in ordered]

        self.cte_ref = to_pairs(self._cte_ref_raw)
        self.cte_echo = to_pairs(self._cte_echo_raw)

    def to_save_dict(self) -> Dict[str, Any]:
        """Serializable snapshot (no private assembly fields)."""
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
    """Write metadata next to the npz (same stem + _meta.json)."""
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    structure = active_structure()
    x_ids, y_ids = group_ids()
    meta = {
        "ref_position": list(REF_POSITION),
        "target_position": list(TARGET_POSITION),
        "anchor_structure_id": ANCHOR_STRUCTURE_ID,
        "anchor_structure": {
            "name": structure.get("name", ""),
            "description": structure.get("description", ""),
            "positions": {
                str(k): list(v) for k, v in structure.get("positions", {}).items()
            },
            "x_ids": list(x_ids),
            "y_ids": list(y_ids),
        },
        "anchor_list": list(ANCHOR_LIST),
        "anchor_x_ids": list(x_ids),
        "anchor_y_ids": list(y_ids),
        "capture_rounds": CAPTURE_ROUNDS,
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
    """Save all rounds into a single compressed npz file."""
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
        slots = []
        xs = []
        ys = []
        tds = []
        rssis = []
        nibs = []
        nechos = []
        frags = []
        timestamps = []
        cte_refs = []
        cte_echos = []

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

        # Stack CTE: (n_rounds, n_samples, 2). Pad shorter runs with NaN.
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
    """Complex division a/b + LS slope/intercept on unwrapped phase."""
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
    return -2.0 * math.pi * F_CENTER_HZ * dist(REF_POSITION, node_xy(anchor_id)) / C_LIGHT


def pair_slope_from_cte(rpt_a: Dict[str, Any], rpt_b: Dict[str, Any]) -> float:
    """mean(LS(ref_a/ref_b), LS(echo_a/echo_b))."""
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
    """
    Half-plane prior from TARGET_POSITION side of the ULA.

    prefer_positive True  → force θ > 0
    prefer_positive False → force θ < 0

    If |θ_raw| < deadzone_deg, keep raw (do not flip).
    Otherwise, if raw θ is on the wrong half, add π and re-estimate.
    """
    th_raw = aoa_from_pair_phase(phase_rad, baseline_m)
    if not math.isfinite(th_raw):
        return {"aoa_raw_deg": float("nan"), "aoa_deg": float("nan"), "pi_corrected": False}

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


def _fuse_sin_weighted(aoas_deg: List[float], baselines_m: List[float]) -> float:
    if not aoas_deg:
        return float("nan")
    sins = [math.sin(math.radians(th)) for th in aoas_deg]
    w = np.asarray([b * b for b in baselines_m], dtype=np.float64)
    w /= np.sum(w)
    sin_fused = float(np.clip(np.dot(w, np.asarray(sins, dtype=np.float64)), -1.0, 1.0))
    return float(np.degrees(np.arcsin(sin_fused)))


def orient_ula_ids(ids: List[int]) -> Tuple[int, int, List[int]]:
    """
    Order a ULA group along its principal axis, directed so that
    TARGET_POSITION lies on the +normal side (baseline +90 deg CCW).

    Returns (ref_id, end_id, ordered_ids from ref→end).
    """
    pts = [(aid, node_xy(aid)) for aid in ids]
    xs = [p[1][0] for p in pts]
    ys = [p[1][1] for p in pts]
    along_x = (max(xs) - min(xs)) >= (max(ys) - min(ys))
    if along_x:
        ordered = sorted(ids, key=lambda a: (node_xy(a)[0], node_xy(a)[1], a))
    else:
        ordered = sorted(ids, key=lambda a: (node_xy(a)[1], node_xy(a)[0], a))

    # Flip direction if target is on the CW / -normal side of ref→end.
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


def array_side_sign(target_xy: Tuple[float, float], ids: List[int]) -> float:
    """
    +1 if target is on the +normal side of the ULA (CCW from ref→end),
    -1 if on the -normal side. Normal = baseline rotated +90 deg.
    """
    _, _, ordered = orient_ula_ids(ids)
    pos_ref = node_xy(ordered[0])
    pos_end = node_xy(ordered[-1])
    bx = pos_end[0] - pos_ref[0]
    by = pos_end[1] - pos_ref[1]
    mid_x = 0.5 * (pos_ref[0] + pos_end[0])
    mid_y = 0.5 * (pos_ref[1] + pos_end[1])
    cross = bx * (target_xy[1] - mid_y) - by * (target_xy[0] - mid_x)
    if abs(cross) < 1e-15:
        return 0.0
    return 1.0 if cross > 0.0 else -1.0


def true_aoa_deg_for_ula(
    target_xy: Tuple[float, float],
    ids: List[int],
) -> float:
    """
    Far-field geometric AoA (deg) matching bearing_from_aoa:

      bearing = normal(ref→end) + θ
      θ = atan2(-along, normal)   ∈ (-180, 180]

    Clamped to [-90, 90] for ULA use. Half-plane prior uses sign(θ).
    """
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
    # Clamp to ULA visible range [-90, 90].
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
    """Pair-fusion AoA over all pairs inside one ULA group."""
    aoas_raw: List[float] = []
    aoas_fix: List[float] = []
    baselines: List[float] = []
    n_pi = 0

    _, _, ordered = orient_ula_ids(ids)
    # Rank along the directed ULA (ref→end) so pair sign matches ESPRIT / bearing.
    rank = {aid: i for i, aid in enumerate(ordered)}

    for id_a, id_b in itertools.combinations(ids, 2):
        # id_a = higher rank (toward end), id_b = lower rank (toward ref).
        if rank[id_a] < rank[id_b]:
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
        aoas_raw.append(float(est["aoa_raw_deg"]))
        aoas_fix.append(float(est["aoa_deg"]))
        baselines.append(baseline)
        if est["pi_corrected"]:
            n_pi += 1

    if not baselines:
        return {
            "aoa_deg": float("nan"),
            "aoa_raw_deg": float("nan"),
            "n_pi_corrected": 0,
        }
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
    """Global bearing from +x: array normal (+90 deg CCW from b→a) + signed AoA."""
    bx = pos_a[0] - pos_b[0]
    by = pos_a[1] - pos_b[1]
    baseline_angle = math.atan2(by, bx)
    normal_angle = baseline_angle + 0.5 * math.pi
    return normal_angle + aoa_rad


def ray_from_ula(ids: List[int], aoa_deg: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return (origin, unit direction) for the AoA ray of one ULA."""
    _, _, ordered = orient_ula_ids(ids)
    pos_ref = node_xy(ordered[0])
    pos_end = node_xy(ordered[-1])
    mid = (0.5 * (pos_ref[0] + pos_end[0]), 0.5 * (pos_ref[1] + pos_end[1]))
    bearing = bearing_from_aoa(pos_end, pos_ref, math.radians(aoa_deg))
    direction = (math.cos(bearing), math.sin(bearing))
    return mid, direction


def intersect_rays(
    p1: Tuple[float, float],
    d1: Tuple[float, float],
    p2: Tuple[float, float],
    d2: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    """Intersect p1 + t d1 with p2 + s d2 (prefer t,s > 0 when possible)."""
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
) -> Optional[Tuple[float, float]]:
    """
    Far-field localization from two ULA AoAs.

    From each array midpoint cast a ray along
      bearing = normal(b→a) + θ
    and return their intersection.
    """
    if not (math.isfinite(aoa_x_deg) and math.isfinite(aoa_y_deg)):
        return None
    o_x, d_x = ray_from_ula(x_ids, aoa_x_deg)
    o_y, d_y = ray_from_ula(y_ids, aoa_y_deg)
    return intersect_rays(o_x, d_x, o_y, d_y)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _ensure_live_figure() -> None:
    global _live_fig, _live_axes
    if plt is None:
        return
    if _live_fig is None or _live_axes is None:
        plt.ion()
        _live_fig, _live_axes = plt.subplots(1, 3, figsize=(14, 4.5))
        try:
            _live_fig.canvas.manager.set_window_title("CoRal XY AoA / position")  # type: ignore[union-attr]
        except Exception:
            pass


def visualize_round(result: Dict[str, Any]) -> None:
    """Live 1x3 plot: X AoA, Y AoA, true vs estimated position."""
    if not ENABLE_VISUALIZATION or plt is None:
        return
    _ensure_live_figure()
    assert _live_fig is not None and _live_axes is not None
    axes = _live_axes

    aoa_x = result["aoa_x_deg"]
    aoa_y = result["aoa_y_deg"]
    true_x = result["true_aoa_x_deg"]
    true_y = result["true_aoa_y_deg"]
    est_pos = result.get("est_pos")
    hist = result.get("history") or []

    # --- X AoA ---
    ax = axes[0]
    ax.clear()
    if hist:
        ax.plot(
            [h["round_idx"] for h in hist],
            [h["aoa_x_deg"] for h in hist],
            "o-",
            ms=3,
            color="C0",
            label="est X",
        )
    ax.axhline(true_x, color="C3", ls="-", lw=1.5, label=f"true {true_x:+.1f} deg")
    ax.axhline(0.0, color="gray", ls=":", lw=1.0)
    ax.set_ylim(-95, 95)
    ax.set_xlabel("round")
    ax.set_ylabel("AoA [deg]")
    ax.set_title(f"X-ULA AoA  now={aoa_x:+.1f} deg")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    # --- Y AoA ---
    ax = axes[1]
    ax.clear()
    if hist:
        ax.plot(
            [h["round_idx"] for h in hist],
            [h["aoa_y_deg"] for h in hist],
            "s-",
            ms=3,
            color="C2",
            label="est Y",
        )
    ax.axhline(true_y, color="C3", ls="-", lw=1.5, label=f"true {true_y:+.1f} deg")
    ax.axhline(0.0, color="gray", ls=":", lw=1.0)
    ax.set_ylim(-95, 95)
    ax.set_xlabel("round")
    ax.set_ylabel("AoA [deg]")
    ax.set_title(f"Y-ULA AoA  now={aoa_y:+.1f} deg")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    # --- Positions ---
    ax = axes[2]
    ax.clear()
    pos_map = anchor_positions()
    x_ids, y_ids = group_ids()
    ax.plot(
        [pos_map[a][0] for a in x_ids],
        [pos_map[a][1] for a in x_ids],
        "o-",
        color="C0",
        label="X-ULA",
    )
    ax.plot(
        [pos_map[a][0] for a in y_ids],
        [pos_map[a][1] for a in y_ids],
        "s-",
        color="C2",
        label="Y-ULA",
    )
    ax.plot(REF_POSITION[0], REF_POSITION[1], "^", color="gray", ms=9, label="REF")
    ax.plot(
        TARGET_POSITION[0],
        TARGET_POSITION[1],
        "*",
        color="C3",
        ms=14,
        label="true tgt",
    )
    if hist:
        xs = [h["est_pos"][0] for h in hist if h.get("est_pos") is not None]
        ys = [h["est_pos"][1] for h in hist if h.get("est_pos") is not None]
        if xs:
            ax.plot(xs, ys, "x", color="C1", ms=6, alpha=0.7, label="est hist")
    if est_pos is not None:
        ax.plot(est_pos[0], est_pos[1], "P", color="C1", ms=12, label="est now")
        # Draw bearing rays for current estimate.
        o_x, d_x = ray_from_ula(x_ids, aoa_x)
        o_y, d_y = ray_from_ula(y_ids, aoa_y)
        ray_len = 0.35
        ax.plot(
            [o_x[0], o_x[0] + ray_len * d_x[0]],
            [o_x[1], o_x[1] + ray_len * d_x[1]],
            "--",
            color="C0",
            lw=1.0,
        )
        ax.plot(
            [o_y[0], o_y[0] + ray_len * d_y[0]],
            [o_y[1], o_y[1] + ray_len * d_y[1]],
            "--",
            color="C2",
            lw=1.0,
        )
        err = dist(est_pos, TARGET_POSITION)
        ax.set_title(f"Position  err={err*100:.1f} cm")
    else:
        ax.set_title("Position (no intersection)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")

    _live_fig.suptitle(f"round {result['round_idx']}")
    _live_fig.tight_layout()
    _live_fig.canvas.draw_idle()
    plt.pause(0.05)


def visualize_summary(results: List[Dict[str, Any]]) -> None:
    if not results or plt is None:
        return
    aoa_x = np.asarray([r["aoa_x_deg"] for r in results], dtype=np.float64)
    aoa_y = np.asarray([r["aoa_y_deg"] for r in results], dtype=np.float64)
    true_x = float(results[0]["true_aoa_x_deg"])
    true_y = float(results[0]["true_aoa_y_deg"])
    est = [r["est_pos"] for r in results if r.get("est_pos") is not None]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    bins = np.linspace(-90, 90, 19)

    ax = axes[0]
    ax.hist(aoa_x[np.isfinite(aoa_x)], bins=bins, color="C0", edgecolor="k", alpha=0.85)
    ax.axvline(true_x, color="C3", lw=2.0, label="true")
    ax.set_title(f"X AoA  mean={np.nanmean(aoa_x):+.1f} deg")
    ax.set_xlabel("AoA [deg]")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(aoa_y[np.isfinite(aoa_y)], bins=bins, color="C2", edgecolor="k", alpha=0.85)
    ax.axvline(true_y, color="C3", lw=2.0, label="true")
    ax.set_title(f"Y AoA  mean={np.nanmean(aoa_y):+.1f} deg")
    ax.set_xlabel("AoA [deg]")
    ax.legend(fontsize=8)

    ax = axes[2]
    pos_map = anchor_positions()
    x_ids, y_ids = group_ids()
    ax.plot([pos_map[a][0] for a in x_ids], [pos_map[a][1] for a in x_ids], "o-", color="C0", label="X-ULA")
    ax.plot([pos_map[a][0] for a in y_ids], [pos_map[a][1] for a in y_ids], "s-", color="C2", label="Y-ULA")
    ax.plot(TARGET_POSITION[0], TARGET_POSITION[1], "*", color="C3", ms=14, label="true")
    if est:
        ax.scatter([p[0] for p in est], [p[1] for p in est], c="C1", marker="x", s=40, label="est")
        mean_est = (float(np.mean([p[0] for p in est])), float(np.mean([p[1] for p in est])))
        ax.plot(mean_est[0], mean_est[1], "P", color="k", ms=10, label="mean est")
        err = dist(mean_est, TARGET_POSITION)
        ax.set_title(f"Position  mean err={err*100:.1f} cm  N={len(est)}")
    else:
        ax.set_title("Position (no estimates)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.suptitle(f"Summary N={len(results)}")
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
    """
    Split X/Y ULAs, estimate AoA on each, intersect bearings for target position.
    """
    x_ids, y_ids = group_ids()
    reports = {aid: report_to_dict(rpt) for aid, rpt in reports_in.items() if aid in ANCHOR_LIST}

    missing = [aid for aid in ANCHOR_LIST if aid not in reports or not reports[aid].get("div_iq")]
    if missing:
        print(f"  data_process skip: missing/empty div_iq for {missing}")
        return None

    true_x = true_aoa_deg_for_ula(TARGET_POSITION, x_ids)
    true_y = true_aoa_deg_for_ula(TARGET_POSITION, y_ids)
    # Half-plane prior from true target side of each ULA (sign of geometric AoA).
    prefer_x = bool(true_x > 0.0)
    prefer_y = bool(true_y > 0.0)
    print(
        f"  half-plane from TARGET: X={'theta>0' if prefer_x else 'theta<0'} "
        f"(true {true_x:+.2f})  Y={'theta>0' if prefer_y else 'theta<0'} "
        f"(true {true_y:+.2f})"
    )

    est_x = estimate_group_aoa(reports, x_ids, prefer_x)
    est_y = estimate_group_aoa(reports, y_ids, prefer_y)
    aoa_x = float(est_x["aoa_deg"])
    aoa_y = float(est_y["aoa_deg"])
    est_pos = position_from_two_aoas(aoa_x, aoa_y, x_ids, y_ids)

    err_pos = dist(est_pos, TARGET_POSITION) if est_pos is not None else float("nan")
    result: Dict[str, Any] = {
        "round_idx": round_idx,
        "aoa_x_deg": aoa_x,
        "aoa_y_deg": aoa_y,
        "aoa_x_raw_deg": float(est_x["aoa_raw_deg"]),
        "aoa_y_raw_deg": float(est_y["aoa_raw_deg"]),
        "true_aoa_x_deg": true_x,
        "true_aoa_y_deg": true_y,
        "est_pos": est_pos,
        "pos_err_m": err_pos,
        "n_pi_x": int(est_x["n_pi_corrected"]),
        "n_pi_y": int(est_y["n_pi_corrected"]),
    }

    pos_str = (
        f"({est_pos[0]:.3f},{est_pos[1]:.3f})" if est_pos is not None else "None"
    )
    print(
        f"  AoA X={aoa_x:+.2f} (true {true_x:+.2f}, +pi x{result['n_pi_x']})  "
        f"Y={aoa_y:+.2f} (true {true_y:+.2f}, +pi x{result['n_pi_y']})  "
        f"pos={pos_str}  err={err_pos*100:.1f} cm"
        if math.isfinite(err_pos)
        else f"  AoA X={aoa_x:+.2f} Y={aoa_y:+.2f}  pos={pos_str}"
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
    Commit one anchor report. When a full round is ready:
      1) data_process() → X/Y AoA + position
      2) append round snapshot
      3) after CAPTURE_ROUNDS, write one combined file
    Returns True when capture is finished (caller should stop).
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

    # Full round ready.
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
    x_ids, y_ids = group_ids()
    print(f"Output file: {CAPTURE_NPZ_PATH}")
    print(f"Will collect {CAPTURE_ROUNDS} rounds, then save one npz.")
    print(f"X-ULA ids={x_ids}  Y-ULA ids={y_ids}")
    print(
        f"true AoA X={true_aoa_deg_for_ula(TARGET_POSITION, x_ids):+.2f} deg  "
        f"Y={true_aoa_deg_for_ula(TARGET_POSITION, y_ids):+.2f} deg"
    )

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
    if ANCHOR_STRUCTURE_ID not in ANCHOR_STRUCTURES:
        print(
            f"Unknown ANCHOR_STRUCTURE_ID={ANCHOR_STRUCTURE_ID}; "
            f"valid ids: {sorted(ANCHOR_STRUCTURES)}",
            file=sys.stderr,
        )
        return 1

    structure = ANCHOR_STRUCTURES[ANCHOR_STRUCTURE_ID]
    x_ids, y_ids = group_ids()
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
    except serial.SerialException as exc:
        print(f"Failed to open {SERIAL_PORT}: {exc}", file=sys.stderr)
        return 1

    print(f"Listening on {SERIAL_PORT} @ {SERIAL_BAUD}. Ctrl+C to stop.")
    print(f"Waiting for anchors: {ANCHOR_LIST}")
    print(f"  X-ULA: {x_ids}")
    print(f"  Y-ULA: {y_ids}")
    print(f"REF={_pos_token(REF_POSITION)}  TGT={_pos_token(TARGET_POSITION)}")
    print(
        f"Structure id={ANCHOR_STRUCTURE_ID} "
        f"({structure.get('name', '')}): {structure.get('description', '')}"
    )
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
