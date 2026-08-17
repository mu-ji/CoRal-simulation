from __future__ import annotations

import math

import numpy as np

from aoa import (
    analyze_aoa_from_phases,
    anchor_phase,
    bearing_from_aoa,
    get_swarm_rx_ids,
    order_rx_pair,
    true_aoa_from_baseline_rad,
    true_bearing_rad,
)
from communication import cte_receive_phase
from node import Node, Swarm, swarm_centroid

def _validate_aoa_nodes(
    tx: Node,
    rx_a: Node,
    rx_b: Node,
    remote: Node,
) -> None:
    nodes = (tx, rx_a, rx_b, remote)
    ids = [node.node_id for node in nodes]
    if len(set(ids)) != len(ids):
        raise ValueError("tx, rx_a, rx_b and remote must be distinct nodes")
    if remote.node_id in {tx.node_id, rx_a.node_id, rx_b.node_id}:
        raise ValueError("remote must not be one of the local swarm nodes")


def aoa_estimation(
    tx: Node,
    rx_a: Node,
    rx_b: Node,
    remote: Node,
    t1: float,
    t2: float,
) -> dict:
    """Estimate AoA of remote relative to the RX baseline (rx_a, rx_b).

    CTE1 is transmitted by ``tx`` at ``t1``; CTE2 is transmitted by ``remote`` at
    ``t2``. Both ``rx_a`` and ``rx_b`` must belong to the same swarm as ``tx``.

    Returns a dict whose primary fields are ``aoa_rad`` and ``aoa_deg`` (angle
    between the RX baseline and the direction to remote). ``rx_a`` / ``rx_b`` are
    reordered internally to (right/upper, left/lower).
    """
    if t2 <= t1:
        raise ValueError("t2 must be greater than t1")

    _validate_aoa_nodes(tx, rx_a, rx_b, remote)

    rx_a, rx_b = order_rx_pair(rx_a, rx_b)

    rx_phase_cte1 = {
        rx_a.node_id: cte_receive_phase(tx, rx_a, t1),
        rx_b.node_id: cte_receive_phase(tx, rx_b, t1),
    }
    rx_phase_cte2 = {
        rx_a.node_id: cte_receive_phase(remote, rx_a, t2),
        rx_b.node_id: cte_receive_phase(remote, rx_b, t2),
    }
    anchor = anchor_phase(tx, rx_a, rx_b)

    result = analyze_aoa_from_phases(
        rx_a,
        rx_b,
        remote,
        rx_phase_cte1,
        rx_phase_cte2,
        rx_a.node_id,
        rx_b.node_id,
        anchor,
        t1,
        t2,
    )
    result.update(
        {
            "tx": tx,
            "rx_a": rx_a,
            "rx_b": rx_b,
            "remote": remote,
            "t1": t1,
            "t2": t2,
            "rx_phase_cte1": rx_phase_cte1,
            "rx_phase_cte2": rx_phase_cte2,
            "bearing_rad": bearing_from_aoa(
                rx_a, rx_b, result["aoa_rad"], remote
            ),
            "true_bearing_rad": true_bearing_rad(rx_a, rx_b, remote),
        }
    )
    return result


def _circular_mean_rad(angles: list[float] | np.ndarray) -> float:
    arr = np.asarray(angles, dtype=np.float64)
    return float(np.arctan2(np.mean(np.sin(arr)), np.mean(np.cos(arr))))


def _angle_between_rad(a: float, b: float) -> float:
    return float(np.arccos(np.cos(a - b)))


def _validate_swarms(
    swarm1_nodes: tuple[Node, Node, Node],
    swarm2_nodes: tuple[Node, Node, Node],
) -> None:
    all_ids = [n.node_id for n in (*swarm1_nodes, *swarm2_nodes)]
    if len(set(all_ids)) != 6:
        raise ValueError("swarm1 and swarm2 must contain 6 distinct nodes")


def _rx_pair_for_tx(swarm: Swarm, tx: Node) -> tuple[Node, Node]:
    rx_a_id, rx_b_id = get_swarm_rx_ids(swarm, tx.node_id)
    return swarm.get_node(rx_a_id), swarm.get_node(rx_b_id)


def multiple_aoa_estimation(
    swarm1_nodes: tuple[Node, Node, Node],
    swarm2_nodes: tuple[Node, Node, Node],
    t1: float,
    t2: float,
    *,
    slot_spacing_s: float = 0.5,
) -> dict:
    """Fuse cross-swarm AoA measurements: swarm2 direction relative to swarm1.

    For each ``tx`` in swarm1 and each ``remote`` in swarm2, the two remaining
    swarm1 nodes act as RX. All slot results are merged via circular mean of
    estimated bearings.

    Returns ``relative_angle_rad`` / ``relative_angle_deg``: estimated direction
    from swarm1 toward swarm2 (in the global frame).
    """
    if t2 <= t1:
        raise ValueError("t2 must be greater than t1")

    _validate_swarms(swarm1_nodes, swarm2_nodes)
    swarm1 = Swarm(list(swarm1_nodes))
    swarm2 = Swarm(list(swarm2_nodes))

    cte_gap_s = t2 - t1
    measurements: list[dict] = []
    bearings: list[float] = []

    slot_index = 0
    for tx in swarm1_nodes:
        rx_a, rx_b = _rx_pair_for_tx(swarm1, tx)
        for remote in swarm2_nodes:
            slot_t1 = t1 + slot_index * slot_spacing_s
            slot_t2 = slot_t1 + cte_gap_s
            est = aoa_estimation(tx, rx_a, rx_b, remote, slot_t1, slot_t2)
            est.update(
                {
                    "slot_index": slot_index,
                    "measuring_swarm": 1,
                    "tx_id": tx.node_id,
                    "rx_a_id": rx_a.node_id,
                    "rx_b_id": rx_b.node_id,
                    "remote_id": remote.node_id,
                }
            )
            measurements.append(est)
            bearings.append(est["bearing_rad"])
            slot_index += 1

    relative_angle_rad = _circular_mean_rad(bearings)

    c1 = swarm_centroid(swarm1)
    c2 = swarm_centroid(swarm2)
    true_relative_angle_rad = float(math.atan2(c2[1] - c1[1], c2[0] - c1[0]))

    return {
        "relative_angle_rad": relative_angle_rad,
        "relative_angle_deg": float(np.degrees(relative_angle_rad)),
        "true_relative_angle_rad": true_relative_angle_rad,
        "true_relative_angle_deg": float(np.degrees(true_relative_angle_rad)),
        "relative_angle_error_deg": float(
            np.degrees(_angle_between_rad(relative_angle_rad, true_relative_angle_rad))
        ),
        "num_slots": len(measurements),
        "measurements": measurements,
        "swarm1_centroid": c1,
        "swarm2_centroid": c2,
    }
