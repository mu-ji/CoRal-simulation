#!/usr/bin/env python3
"""
Read ASCII report lines from CoRal_listen_node over UART.

Firmware line format (raw CTE IQ):
  RPT id=2 slot=1 x=0 y=0 td=8171 rssi=-27 nib=88 necho=88 frags=8
  CTE_REF n=88
    idx=0 I=123 Q=-45
    ...
  CTE_ECHO n=88
    idx=0 I=... Q=...
    ...

Host-side: build div_iq = CTE_REF / CTE_ECHO (sample-aligned complex division),
estimate slope from unwrapped phase, then run ESPRIT as before.
"""

from __future__ import annotations

import datetime
import math
import os
import re
import sys
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


# Explicit serial settings (change COM number here).
SERIAL_PORT = "COM9"
SERIAL_BAUD = 115200

# Anchor node ids that must all report before data_process() runs.
# Firmware: REF=0, TGT=1, anchors=2..9. Listen node only forwards id in [2,9].
ANCHOR_LIST = [2, 3, 4, 5]

# ULA reference + other sensors used for ESPRIT snapshots.
ARRAY_REF_ID = 2
ARRAY_OTHER_IDS = [3, 4, 5]

# RF / timing (CHANNEL=17 → NRF FREQUENCY=40 → 2440 MHz).
F_CENTER_HZ = 2.440e9
C_LIGHT = 2.99792458e8
TIMER_HZ = 16e6          # TIMER0 PRESCALER=0
SAMPLE_INTERVAL_S = 1e-6  # DF IQ spacing (TSAMPLESPACING=1 us)

# Known geometry in meters. Edit to match the deployed layout.
REF_POS: Tuple[float, float] = (0.00, 0.1)

NODE_POS: Dict[int, Tuple[float, float]] = {
    2: (0.00, 0.00),
    3: (0.025, 0.00),
    4: (0.05, 0.00),
    5: (0.075, 0.00),
}

# Print every CTE sample (verbose). If False, only print count summary.
PRINT_EACH_CTE = False

# Temporarily disable blocking matplotlib visualization.
ENABLE_VISUALIZATION = False

# Save each full anchor round to disk; stop after this many rounds.
CAPTURE_ROUNDS = 50
CAPTURE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "captured_reports", "test2"
)
_capture_count = 0

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


def parse_rpt_line(line: str) -> Optional[Dict[str, Any]]:
    m = RPT_RE.match(line.strip())
    if not m:
        return None

    nib = int(m.group("nib"))
    necho = int(m.group("necho"))
    return {
        "timestamp": datetime.datetime.now().isoformat(timespec="milliseconds"),
        "id": int(m.group("id")),
        "slot": int(m.group("slot")),
        "x": int(m.group("x")),
        "y": int(m.group("y")),
        "time_diff": int(m.group("td")),
        "rssi": int(m.group("rssi")),
        "nib": nib,
        "necho": necho,
        "frags": int(m.group("frags")),
        "cte_ref": [],   # list of dicts {idx,i,q} then (I,Q) tuples
        "cte_echo": [],
        "div_iq": [],
        "intercept": 0.0,
        "slope": 0.0,
        "section": None,       # "CTE_REF" | "CTE_ECHO" | None
        "section_n": 0,
        "raw": line.strip(),
    }


def parse_cte_hdr(line: str) -> Optional[Dict[str, Any]]:
    m = CTE_HDR_RE.match(line.strip())
    if not m:
        return None
    return {"tag": m.group("tag"), "n": int(m.group("n"))}


def parse_cte_sample(line: str) -> Optional[Dict[str, Any]]:
    m = CTE_SAMPLE_RE.match(line)
    if not m:
        return None
    return {
        "idx": int(m.group("idx")),
        "i": float(m.group("I")),
        "q": float(m.group("Q")),
    }


def iq_list_to_complex(samples: List[Any]) -> np.ndarray:
    """[(I,Q), ...] or [{i,q}, ...] → complex array."""
    if not samples:
        return np.zeros(0, dtype=np.complex128)
    if isinstance(samples[0], dict):
        ordered = sorted(samples, key=lambda s: s["idx"])
        return np.asarray(
            [s["i"] + 1j * s["q"] for s in ordered], dtype=np.complex128
        )
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros(0, dtype=np.complex128)
    return arr[:, 0] + 1j * arr[:, 1]


def compute_div_iq_and_ls(
    cte_ref: List[Any],
    cte_echo: List[Any],
) -> Tuple[List[Tuple[float, float]], float, float]:
    """
    Aligned complex division REF/ECHO, then LS slope/intercept on unwrapped phase.
    Returns (div_iq as [(I,Q),...], intercept, slope).
    """
    ref_c = iq_list_to_complex(cte_ref)
    echo_c = iq_list_to_complex(cte_echo)
    n = min(len(ref_c), len(echo_c))
    if n < 2:
        return [], 0.0, 0.0

    div_iq: List[Tuple[float, float]] = []
    phases: List[float] = []
    xs: List[float] = []
    prev = 0.0
    n_valid = 0

    for i in range(n):
        den = echo_c[i]
        if abs(den) == 0.0:
            continue
        ratio = ref_c[i] / den
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


def report_complete(rpt: Dict[str, Any]) -> bool:
    return (
        len(rpt["cte_ref"]) >= rpt["nib"]
        and len(rpt["cte_echo"]) >= rpt["necho"]
    )


def finalize_cte_fields(rpt: Dict[str, Any]) -> None:
    """Sort CTE samples, compute div_iq + LS slope/intercept."""
    for key in ("cte_ref", "cte_echo"):
        samples = rpt[key]
        if samples and isinstance(samples[0], dict):
            ordered = sorted(samples, key=lambda s: s["idx"])
            rpt[key] = [(s["i"], s["q"]) for s in ordered]

    div_iq, intercept, slope = compute_div_iq_and_ls(
        rpt["cte_ref"], rpt["cte_echo"]
    )
    rpt["div_iq"] = div_iq
    rpt["intercept"] = intercept
    rpt["slope"] = slope
    rpt["ndiv"] = len(div_iq)


def format_report(rpt: Dict[str, Any]) -> str:
    return (
        f"[{rpt['timestamp']}] "
        f"id={rpt['id']} slot={rpt['slot']} "
        f"pos=({rpt['x']},{rpt['y']}) "
        f"td={rpt['time_diff']} rssi={rpt['rssi']} "
        f"nib={rpt['nib']} necho={rpt['necho']} frags={rpt['frags']} "
        f"div={len(rpt.get('div_iq', []))} "
        f"intercept={rpt.get('intercept', 0):.6g} "
        f"slope={rpt.get('slope', 0):.6g}"
    )


def node_xy(node_id: int) -> Tuple[float, float]:
    if node_id == 0:
        return REF_POS
    if node_id not in NODE_POS:
        raise KeyError(f"Unknown node position for id={node_id}; set NODE_POS")
    return NODE_POS[node_id]


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def ula_element_spacing_m() -> float:
    return dist(node_xy(ARRAY_REF_ID), node_xy(ARRAY_OTHER_IDS[0]))


def ref_to_anchor_phase(anchor_id: int) -> float:
    return (
        -2.0
        * math.pi
        * F_CENTER_HZ
        * dist(REF_POS, node_xy(anchor_id))
        / C_LIGHT
    )


def m_phase_reference_correction(other_id: int) -> float:
    return ref_to_anchor_phase(other_id) - ref_to_anchor_phase(ARRAY_REF_ID)


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


def div_iq_to_complex(div_iq: List[Any]) -> np.ndarray:
    if not div_iq:
        return np.zeros(0, dtype=np.complex128)
    arr = np.asarray(div_iq, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros(0, dtype=np.complex128)
    return arr[:, 0] + 1j * arr[:, 1]


def aligned_div_complex(
    div_a: np.ndarray,
    div_b: np.ndarray,
) -> Optional[np.ndarray]:
    n = min(len(div_a), len(div_b))
    if n <= 0:
        return None
    a = div_a[:n]
    b = div_b[:n]
    return np.divide(
        a,
        b,
        out=np.full(n, np.nan, dtype=np.complex128),
        where=np.abs(b) > 0.0,
    )


def td_ticks_to_seconds(td_ticks: float) -> float:
    return float(td_ticks) / TIMER_HZ


def apply_snapshot_phase_corrections(
    ratio: np.ndarray,
    rpt_ref: Dict[str, Any],
    rpt_other: Dict[str, Any],
    other_id: int,
) -> Tuple[np.ndarray, float, float]:
    slope_diff = float(rpt_ref["slope"]) - float(rpt_other["slope"])
    td_ticks = 0.5 * (float(rpt_ref["time_diff"]) + float(rpt_other["time_diff"]))
    delta_t_s = td_ticks_to_seconds(td_ticks)
    phi_cfo = slope_diff * (delta_t_s / SAMPLE_INTERVAL_S)
    phi_m = m_phase_reference_correction(other_id)
    corrected = ratio * np.exp(-1j * phi_cfo) * np.exp(-1j * phi_m)
    return corrected, phi_cfo, phi_m


def build_array_snapshots(
    reports: Dict[int, Dict[str, Any]],
    *,
    verbose: bool = True,
) -> Optional[np.ndarray]:
    if ARRAY_REF_ID not in reports:
        return None
    rpt_ref = reports[ARRAY_REF_ID]
    ref_c = div_iq_to_complex(rpt_ref.get("div_iq", []))
    if len(ref_c) == 0:
        return None

    rows: List[np.ndarray] = []
    n_common: Optional[int] = None
    for oid in ARRAY_OTHER_IDS:
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

        corrected, phi_cfo, phi_m = apply_snapshot_phase_corrections(
            ratio, rpt_ref, rpt, oid
        )
        if verbose:
            td_ticks = 0.5 * (float(rpt_ref["time_diff"]) + float(rpt["time_diff"]))
            delta_t_s = td_ticks_to_seconds(td_ticks)
            slope_diff = float(rpt_ref["slope"]) - float(rpt["slope"])
            print(
                f"  corr 2/{oid}: slope_diff={slope_diff:.6g} rad/sample  "
                f"td={td_ticks:.0f} ticks ({delta_t_s*1e6:.3f} us)  "
                f"φ_cfo={phi_cfo:+.6g} rad  "
                f"∠M={phi_m:+.6g} rad ({math.degrees(phi_m):+.3f} deg)"
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


def esprit_aoa_deg(x: np.ndarray, d_sp: float, num_sources: int = 1) -> float:
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

    angles = []
    for phi in phi_vals:
        sin_theta = float(
            np.clip(
                -np.angle(phi) * C_LIGHT / (2.0 * np.pi * F_CENTER_HZ * d_sp),
                -1.0,
                1.0,
            )
        )
        angles.append(float(np.degrees(np.arcsin(sin_theta))))
    return float(angles[0]) if len(angles) == 1 else float(sorted(angles, key=abs)[0])


def visualize_div_iq_phases(
    reports: Dict[int, Dict[str, Any]],
    ratios: Optional[np.ndarray],
    theta_esprit_deg: Optional[float],
) -> None:
    if plt is None:
        print("  [viz] matplotlib not installed; skip plots")
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    title = "Corrected div_iq phases (2/k)"
    if theta_esprit_deg is not None:
        title += f"  |  ESPRIT AoA={theta_esprit_deg:+.2f}°"
    fig.suptitle(title)

    ref_id = ARRAY_REF_ID
    ref_c = div_iq_to_complex(reports.get(ref_id, {}).get("div_iq", []))

    for ax, oid, row_i in zip(axes, ARRAY_OTHER_IDS, range(3)):
        if ratios is not None and ratios.shape[0] > row_i:
            phase = np.angle(ratios[row_i])
            n = len(phase)
            t_us = np.arange(n, dtype=np.float64) * (SAMPLE_INTERVAL_S * 1e6)
            ax.plot(t_us, phase, lw=1.0)
            ax.set_title(f"∠(div_iq[{ref_id}] / div_iq[{oid}])  N={n}")
            ax.set_ylim(-math.pi, math.pi)
        elif len(ref_c) > 0 and oid in reports:
            other_c = div_iq_to_complex(reports[oid].get("div_iq", []))
            ratio = aligned_div_complex(ref_c, other_c)
            if ratio is not None:
                phase = np.angle(ratio)
                t_us = np.arange(len(phase), dtype=np.float64) * (
                    SAMPLE_INTERVAL_S * 1e6
                )
                ax.plot(t_us, phase, lw=1.0)
                ax.set_title(f"∠(div_iq[{ref_id}] / div_iq[{oid}])  N={len(phase)}")
                ax.set_ylim(-math.pi, math.pi)
            else:
                ax.set_title(f"2/{oid}: empty div_iq")
        else:
            ax.set_title(f"2/{oid}: missing")
        ax.set_ylabel("phase [rad]")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("sample time [µs]")
    fig.tight_layout()
    print("  [viz] showing figure (close window to continue)")
    plt.show(block=True)
    plt.close(fig)


def save_reports_round(
    reports: Dict[int, Dict[str, Any]],
    round_idx: int,
) -> str:
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    path = os.path.join(CAPTURE_DIR, f"round_{round_idx:03d}.npz")

    payload: Dict[str, Any] = {
        "round_idx": np.int32(round_idx),
        "saved_at": np.asarray(
            datetime.datetime.now().isoformat(timespec="milliseconds")
        ),
        "anchor_list": np.asarray(ANCHOR_LIST, dtype=np.int32),
        "ref_pos": np.asarray(REF_POS, dtype=np.float64),
        "f_center_hz": np.float64(F_CENTER_HZ),
    }
    for aid in ANCHOR_LIST:
        rpt = reports.get(aid)
        if rpt is None:
            continue
        prefix = f"id{aid}_"
        payload[prefix + "slot"] = np.int32(rpt["slot"])
        payload[prefix + "x"] = np.int32(rpt["x"])
        payload[prefix + "y"] = np.int32(rpt["y"])
        payload[prefix + "intercept"] = np.float64(rpt.get("intercept", 0.0))
        payload[prefix + "slope"] = np.float64(rpt.get("slope", 0.0))
        payload[prefix + "time_diff"] = np.int32(rpt["time_diff"])
        payload[prefix + "rssi"] = np.int32(rpt["rssi"])
        payload[prefix + "nib"] = np.int32(rpt.get("nib", 0))
        payload[prefix + "necho"] = np.int32(rpt.get("necho", 0))
        payload[prefix + "ndiv"] = np.int32(rpt.get("ndiv", 0))

        cte_ref = rpt.get("cte_ref", [])
        cte_echo = rpt.get("cte_echo", [])
        div_iq = rpt.get("div_iq", [])
        payload[prefix + "cte_ref"] = (
            np.asarray(cte_ref, dtype=np.float64)
            if cte_ref
            else np.zeros((0, 2), dtype=np.float64)
        )
        payload[prefix + "cte_echo"] = (
            np.asarray(cte_echo, dtype=np.float64)
            if cte_echo
            else np.zeros((0, 2), dtype=np.float64)
        )
        payload[prefix + "div_iq"] = (
            np.asarray(div_iq, dtype=np.float64)
            if div_iq
            else np.zeros((0, 2), dtype=np.float64)
        )
        if aid in NODE_POS:
            payload[prefix + "pos"] = np.asarray(NODE_POS[aid], dtype=np.float64)

    np.savez_compressed(path, **payload)
    return path


def data_process(reports: Dict[int, Dict[str, Any]]) -> bool:
    global _capture_count

    print("---- data_process ----")

    for aid in ANCHOR_LIST:
        rpt = reports.get(aid)
        if rpt is None:
            continue
        print(
            f"  id={aid}: nib={rpt.get('nib', 0)} necho={rpt.get('necho', 0)} "
            f"div={len(rpt.get('div_iq', []))} "
            f"slope={rpt.get('slope', 0):.6g}"
        )

    x = build_array_snapshots(reports)
    d_sp = ula_element_spacing_m()
    print(f"  ULA spacing d={d_sp*100:.3f} cm  fc={F_CENTER_HZ/1e9:.3f} GHz")

    theta_esprit: Optional[float] = None

    if x is None:
        print("  ESPRIT: failed to build 3×N snapshots from div_iq")
    else:
        print(f"  snapshot matrix X: {x.shape[0]}×{x.shape[1]}")
        try:
            theta_esprit = esprit_aoa_deg(x, d_sp, num_sources=1)
        except Exception as exc:
            print(f"  ESPRIT error: {exc}")
        else:
            print(f"  ESPRIT AoA: {theta_esprit:+.3f} deg")

            pos_ref = node_xy(ARRAY_REF_ID)
            pos_end = node_xy(ARRAY_OTHER_IDS[-1])
            mid = (
                0.5 * (pos_ref[0] + pos_end[0]),
                0.5 * (pos_ref[1] + pos_end[1]),
            )
            bearing = bearing_from_aoa(
                pos_end, pos_ref, math.radians(theta_esprit)
            )
            print(
                f"  ESPRIT bearing: {math.degrees(bearing):+.2f} deg  "
                f"(from array mid ({mid[0]:.3f},{mid[1]:.3f}))"
            )

    if ENABLE_VISUALIZATION:
        visualize_div_iq_phases(reports, x, theta_esprit)

    _capture_count += 1
    out_path = save_reports_round(reports, _capture_count)
    print(f"  saved round {_capture_count}/{CAPTURE_ROUNDS}: {out_path}")

    print("----------------------")
    return _capture_count >= CAPTURE_ROUNDS


def commit_report(
    rpt: Dict[str, Any],
    latest_reports: Dict[int, Dict[str, Any]],
    updated: Set[int],
    required: Set[int],
) -> bool:
    finalize_cte_fields(rpt)
    print(format_report(rpt))

    aid = rpt["id"]
    if aid not in required:
        return False

    latest_reports[aid] = rpt
    updated.add(aid)
    missing = sorted(required - updated)
    if missing:
        print(f"  collected {sorted(updated)}  waiting for {missing}")

    if updated >= required:
        done = data_process(latest_reports)
        updated.clear()
        return done
    return False


def coral_listener(ser: serial.Serial) -> None:
    """Assemble RPT + CTE_REF/CTE_ECHO sample blocks into full reports."""
    raw_line = bytearray()
    latest_reports: Dict[int, Dict[str, Any]] = {}
    updated: Set[int] = set()
    required = set(ANCHOR_LIST)
    pending: Optional[Dict[str, Any]] = None

    print(
        f"Capture: save {CAPTURE_ROUNDS} rounds to {CAPTURE_DIR} "
        f"(visualization={'on' if ENABLE_VISUALIZATION else 'off'})"
    )

    while True:
        byte = ser.read(1)
        if not byte:
            continue

        raw_line += byte

        if len(raw_line) >= 2 and raw_line[-2:] == b"\r\n":
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
                        f"[WARN] incomplete report id={pending['id']} "
                        f"ref={len(pending['cte_ref'])}/{pending['nib']} "
                        f"echo={len(pending['cte_echo'])}/{pending['necho']}; "
                        f"discarding"
                    )
                pending = rpt
                # No CTE expected.
                if rpt["nib"] == 0 and rpt["necho"] == 0:
                    if commit_report(pending, latest_reports, updated, required):
                        print(f"Captured {CAPTURE_ROUNDS} rounds; stopping.")
                        return
                    pending = None
                continue

            hdr = parse_cte_hdr(line)
            if hdr is not None:
                if pending is None:
                    print(f"[RAW] orphan {hdr['tag']}: {line.strip()}")
                    continue
                pending["section"] = hdr["tag"]
                pending["section_n"] = hdr["n"]
                if PRINT_EACH_CTE:
                    print(f"  {hdr['tag']} n={hdr['n']} (id={pending['id']})")
                continue

            sample = parse_cte_sample(line)
            if sample is not None:
                if pending is None or pending.get("section") is None:
                    print(f"[RAW] orphan CTE sample: {line.strip()}")
                    continue
                section = pending["section"]
                if section == "CTE_REF":
                    pending["cte_ref"].append(sample)
                elif section == "CTE_ECHO":
                    pending["cte_echo"].append(sample)
                else:
                    print(f"[RAW] sample in unknown section: {line.strip()}")
                    continue

                if PRINT_EACH_CTE:
                    print(
                        f"  {section} idx={sample['idx']} "
                        f"I={sample['i']} Q={sample['q']}"
                    )

                if report_complete(pending):
                    if commit_report(pending, latest_reports, updated, required):
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
    print(f"REF_POS={REF_POS}  F_c={F_CENTER_HZ/1e9:.3f} GHz")
    print(f"Will save {CAPTURE_ROUNDS} rounds then exit (viz off).")
    try:
        coral_listener(ser)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
