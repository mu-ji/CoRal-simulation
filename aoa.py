from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from communication import C_LIGHT, CHANNEL_CENTER_FREQ_HZ, adc_fs, path_phase
from node import L_ARM_SPACING_M, Node, Swarm, make_random_node

wavelength = C_LIGHT / CHANNEL_CENTER_FREQ_HZ

# MINIMUM_LOCAL_NODE_IDS = (0, 1, 2)
# MINIMUM_REMOTE_NODE_ID = 3

MINIMUM_DEFAULT_REMOTE = (4.0, 4.0)
MINIMUM_DEFAULT_T1_MS = 10.0
MINIMUM_DEFAULT_T2_MS = 10.15
MINIMUM_DEFAULT_TX1_ID = 1
DEFAULT_RANDOM_SEED = 20010127
MINIMUM_DEFAULT_PROPAGATION_SNR_DB = 30.0
MINIMUM_DEFAULT_RX_SNR_DB = 30.0
MINIMUM_DEFAULT_SAMPLE_PHASE_SNR_DB = 35.0
MINIMUM_MAP_XLIM = (-5.5, 5.5)
MINIMUM_MAP_YLIM = (-5.5, 5.5)

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]


def _require_matplotlib() -> None:
    if plt is None:
        raise ImportError("matplotlib is required for visualization functions")


def _burst_time_ms(t_start: float, n: int) -> np.ndarray:
    return (t_start + np.arange(n) / adc_fs) * 1e3


def order_rx_pair(rx_a: Node, rx_b: Node) -> tuple[Node, Node]:
    """Return RX pair as (right/upper, left/lower) by node position."""
    if rx_a.x > rx_b.x or (rx_a.x == rx_b.x and rx_a.y >= rx_b.y):
        return rx_a, rx_b
    return rx_b, rx_a


def _right_or_upper_receiver(rx_a: Node, rx_b: Node) -> tuple[Node, Node]:
    return order_rx_pair(rx_a, rx_b)


def anchor_phase(tx: Node, rx_a: Node, rx_b: Node) -> float:
    right_or_upper, left_or_lower = _right_or_upper_receiver(rx_a, rx_b)
    return path_phase(tx, right_or_upper) - path_phase(tx, left_or_lower)


def get_swarm_rx_ids(swarm: Swarm, tx_id: int) -> tuple[int, int]:
    """Return RX pair ids in swarm: (right/upper, left/lower)."""
    others = swarm.other_nodes(tx_id)
    right_or_upper, left_or_lower = _right_or_upper_receiver(
        others[0], others[1]
    )
    return right_or_upper.node_id, left_or_lower.node_id


def estimate_phase_diff_rate(phase_diff: np.ndarray) -> dict:
    phase_unwrapped = np.unwrap(phase_diff)
    t_local = np.arange(len(phase_unwrapped)) / adc_fs
    fit = np.polyfit(t_local, phase_unwrapped, 1)
    return {
        "phase_unwrapped": phase_unwrapped,
        "t_local": t_local,
        "fit": fit,
        "rate_rad_s": float(fit[0]),
    }


def calculate_aoa(
    rx_a: Node,
    rx_b: Node,
    compensated_phase: np.ndarray | float,
) -> float:
    antenna_interval = rx_a.distance_to(rx_b)
    if isinstance(compensated_phase, np.ndarray):
        phase = float(np.mean(compensated_phase))
    else:
        phase = float(compensated_phase)
    cos_aoa = np.clip(
        phase * wavelength / (2.0 * np.pi * antenna_interval),
        -1.0,
        1.0,
    )
    return float(np.arccos(cos_aoa))


def true_bearing_rad(rx_a: Node, rx_b: Node, tx: Node) -> float:
    mid_x = (rx_a.x + rx_b.x) / 2.0
    mid_y = (rx_a.y + rx_b.y) / 2.0
    return float(np.arctan2(tx.y - mid_y, tx.x - mid_x))


def true_aoa_from_baseline_rad(rx_a: Node, rx_b: Node, tx: Node) -> float:
    mid_x = (rx_a.x + rx_b.x) / 2.0
    mid_y = (rx_a.y + rx_b.y) / 2.0
    bx = rx_a.x - rx_b.x
    by = rx_a.y - rx_b.y
    tx_x = tx.x - mid_x
    tx_y = tx.y - mid_y
    baseline_norm = float(np.hypot(bx, by))
    to_tx_norm = float(np.hypot(tx_x, tx_y))
    if baseline_norm == 0.0 or to_tx_norm == 0.0:
        return 0.0
    cos_aoa = (bx * tx_x + by * tx_y) / (baseline_norm * to_tx_norm)
    return float(np.arccos(np.clip(cos_aoa, -1.0, 1.0)))


def bearing_from_aoa(
    rx_a: Node,
    rx_b: Node,
    aoa_rad: float,
    reference: Node,
) -> float:
    mid_x = (rx_a.x + rx_b.x) / 2.0
    mid_y = (rx_a.y + rx_b.y) / 2.0
    baseline_angle = float(np.arctan2(rx_a.y - rx_b.y, rx_a.x - rx_b.x))
    true_bearing = float(np.arctan2(reference.y - mid_y, reference.x - mid_x))
    candidates = (
        baseline_angle + aoa_rad,
        baseline_angle - aoa_rad,
    )
    return min(
        candidates,
        key=lambda bearing: abs(np.angle(np.exp(1j * (bearing - true_bearing)))),
    )


def analyze_aoa_from_phases(
    rx_a: Node,
    rx_b: Node,
    target_tx: Node,
    rx_phase_cte1: dict[int, np.ndarray],
    rx_phase_cte2: dict[int, np.ndarray],
    rx_a_id: int,
    rx_b_id: int,
    anchor_phase_value: float,
    t1: float,
    t2: float,
) -> dict:
    rx_a, rx_b = order_rx_pair(rx_a, rx_b)
    rx_a_id = rx_a.node_id
    rx_b_id = rx_b.node_id
    phase_diff1 = np.unwrap(rx_phase_cte1[rx_a_id] - rx_phase_cte1[rx_b_id])
    phase_diff2 = np.unwrap(rx_phase_cte2[rx_a_id] - rx_phase_cte2[rx_b_id])
    phase_rate = estimate_phase_diff_rate(phase_diff1)
    phase_diff_delta = (
        phase_diff2
        - phase_diff1
        - phase_rate["rate_rad_s"] * (t2 - t1)
        + anchor_phase_value
    )
    phase_diff_delta = np.arctan2(
        np.sin(phase_diff_delta), np.cos(phase_diff_delta)
    )
    aoa_rad = calculate_aoa(rx_a, rx_b, phase_diff_delta)
    true_aoa_rad = true_aoa_from_baseline_rad(rx_a, rx_b, target_tx)
    return {
        "phase_diff1": phase_diff1,
        "phase_diff2": phase_diff2,
        "anchor_phase": anchor_phase_value,
        "phase_diff_delta": phase_diff_delta,
        "aoa_rad": aoa_rad,
        "aoa_deg": float(np.degrees(aoa_rad)),
        "true_aoa_rad": true_aoa_rad,
        "true_aoa_deg": float(np.degrees(true_aoa_rad)),
        "aoa_error_deg": float(
            np.degrees(np.arccos(np.cos(aoa_rad - true_aoa_rad)))
        ),
    }

