import matplotlib.pyplot as plt
import numpy as np

from node import C_LIGHT, CHANNEL_CENTER_FREQ_HZ, Node, adc_fs
import math

SLOT_DURATION_S = 0.5
NUM_SLOTS = 9

CTE_TX1_OFFSET_S = 10e-3
CTE_TX2_OFFSET_AFTER_RX1_S = 150e-6

wavelength = C_LIGHT / CHANNEL_CENTER_FREQ_HZ

np.random.seed(20010127)

def _make_node(x: float, y: float, node_id: int) -> Node:
    return Node(
        x,
        y,
        initial_phase=np.random.uniform(0, 2 * np.pi),
        frequency_offset=np.random.uniform(-100, 100),
        node_id=node_id,
    )


swarm1 = [
    _make_node(0.0, 0.0, 0),
    _make_node(0.065, 0.0, 1),
    _make_node(0.0, 0.065, 2),
]

swarm2 = [
    _make_node(4.0, 2.0, 3),
    _make_node(4.065, 2.0, 4),
    _make_node(4.0, 2.065, 5),
]

nodes = swarm1 + swarm2
nodes_by_id = {node.node_id: node for node in nodes}

slot_pairs = [
    (0, 3),
    (0, 4),
    (0, 5),
    (1, 3),
    (1, 4),
    (1, 5),
    (2, 3),
    (2, 4),
    (2, 5),
]


def run_slot(slot_index: int, slot_start: float) -> dict:
    tx1_id, tx2_id = slot_pairs[slot_index]
    t1 = slot_start + CTE_TX1_OFFSET_S
    t2 = t1 + CTE_TX2_OFFSET_AFTER_RX1_S

    tx1 = nodes_by_id[tx1_id]
    tx2 = nodes_by_id[tx2_id]

    bb1 = tx1.generate_cte_signal(t1)
    rf1 = tx1.upconvert_cte_to_ble(t1, bb1)
    bb1_adc = tx1.sample_baseband(bb1)

    rx_bb1 = {}
    for node in nodes:
        if node.node_id == tx1_id:
            continue
        rf = tx1.propagate_signal(rf1, node)
        rx_bb1[node.node_id] = node.sample_baseband(
            node.downconvert_received_signal(t1, rf)
        )

    bb2 = tx2.generate_cte_signal(t2)
    rf2 = tx2.upconvert_cte_to_ble(t2, bb2)
    bb2_adc = tx2.sample_baseband(bb2)

    rx_bb2 = {}
    for node in nodes:
        if node.node_id == tx2_id:
            continue
        rf = tx2.propagate_signal(rf2, node)
        rx_bb2[node.node_id] = node.sample_baseband(
            node.downconvert_received_signal(t2, rf)
        )

    return {
        "slot_index": slot_index,
        "tx1_id": tx1_id,
        "tx2_id": tx2_id,
        "t1": t1,
        "t2": t2,
        "tx_bb1_adc": bb1_adc,
        "tx_bb2_adc": bb2_adc,
        "rx_bb1": rx_bb1,
        "rx_bb2": rx_bb2,
    }


SWARM1_IDS = {0, 1, 2}

def calculate_anchor_phase(tx: Node, rx_a: Node, rx_b: Node) -> float:
    anchor_position = tx.get_position()
    rx_a_position = rx_a.get_position()
    rx_b_position = rx_b.get_position()
    distance_a = math.hypot(anchor_position[0] - rx_a_position[0], anchor_position[1] - rx_a_position[1])
    distance_b = math.hypot(anchor_position[0] - rx_b_position[0], anchor_position[1] - rx_b_position[1])
    phase = 2.0 * math.pi * (distance_a - distance_b) / wavelength
    return phase

def calculate_aoa(tx: Node, rx_a: Node, rx_b: Node, compensated_phase: float) -> float:
    rx_a_position = rx_a.get_position()
    rx_b_position = rx_b.get_position()
    antenna_interval = math.hypot(rx_a_position[0] - rx_b_position[0], rx_a_position[1] - rx_b_position[1])
    aoa = np.arcsin(compensated_phase * wavelength / (2.0 * np.pi * antenna_interval))
    return aoa


def _aoa_bearing_rad(
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
        baseline_angle + np.pi / 2.0 - aoa_rad,
        baseline_angle - np.pi / 2.0 + aoa_rad,
    )
    return min(
        candidates,
        key=lambda bearing: abs(np.angle(np.exp(1j * (bearing - true_bearing)))),
    )


def analyze_slot_baseband(run_slot_output: dict) -> dict:
    """
    分析某一 slot 内全部 baseband 信号。

    Parameters
    ----------
    run_slot_output : dict
        run_slot() 的返回值。
    """
    tx1_id = run_slot_output["tx1_id"]
    tx2_id = run_slot_output["tx2_id"]
    t1 = run_slot_output["t1"]
    t2 = run_slot_output["t2"]
    rx_bb1 = run_slot_output["rx_bb1"]
    rx_bb2 = run_slot_output["rx_bb2"]
    other_swarm1 = sorted(SWARM1_IDS - {tx1_id})
    node_a, node_b = other_swarm1

    swarm1_cte1_phase_diff = np.angle(
        rx_bb1[node_a] * np.conj(rx_bb1[node_b])
    )
    swarm1_cte2_phase_diff = np.angle(
        rx_bb2[node_a] * np.conj(rx_bb2[node_b])
    )
    swarm1_phase_diff_delta = np.angle(
        np.exp(1j * swarm1_cte1_phase_diff) * np.exp(-1j * swarm1_cte2_phase_diff)
    )

    t_local = np.arange(len(swarm1_cte1_phase_diff)) / adc_fs
    phase_diff_rate = float(
        np.polyfit(t_local, np.unwrap(swarm1_cte1_phase_diff), 1)[0]
    )

    delta_t = t2 - t1
    phase_diff_drift = phase_diff_rate * delta_t
    swarm1_phase_diff_delta_corrected = np.angle(
        np.exp(1j * (swarm1_phase_diff_delta - phase_diff_drift))
    )

    anchor_phase = calculate_anchor_phase(nodes_by_id[tx1_id], nodes_by_id[node_a], nodes_by_id[node_b])
    compensated_phase = anchor_phase - float(
        np.mean(swarm1_phase_diff_delta_corrected)
    )
    aoa_rad = calculate_aoa(
        nodes_by_id[tx2_id],
        nodes_by_id[node_a],
        nodes_by_id[node_b],
        compensated_phase,
    )

    burst1_diff = {
        "tx_id": tx1_id,
        "node_a": node_a,
        "node_b": node_b,
        "phase_diff": swarm1_cte1_phase_diff,
        "phase_diff_rate": phase_diff_rate,
    }
    burst2_diff = {
        "tx_id": run_slot_output["tx2_id"],
        "node_a": node_a,
        "node_b": node_b,
        "phase_diff": swarm1_cte2_phase_diff,
    }

    slot_index = run_slot_output["slot_index"]
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    label_ab = f"node {node_a} - node {node_b}"

    ax.plot(
        _burst_time_ms(t1, len(swarm1_cte1_phase_diff)),
        swarm1_cte1_phase_diff,
        marker="o",
        markersize=4,
        linewidth=0.8,
        label=f"CTE1 TX {tx1_id} ({label_ab})",
    )
    ax.plot(
        _burst_time_ms(t2, len(swarm1_cte2_phase_diff)),
        swarm1_cte2_phase_diff,
        marker="s",
        markersize=4,
        linewidth=0.8,
        label=f"CTE2 TX {burst2_diff['tx_id']} ({label_ab})",
    )
    ax.set_title(f"Slot {slot_index}: swarm1 RX phase diff")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Phase diff (rad)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig("slot_analysis_phase_diff.png", dpi=150)

    return {
        "slot_index": slot_index,
        "burst1_swarm1_rx_phase_diff": burst1_diff,
        "burst2_swarm1_rx_phase_diff": burst2_diff,
        "swarm1_phase_diff_delta": {
            "delta_t": delta_t,
            "phase_diff_drift": phase_diff_drift,
            "phase_diff": swarm1_phase_diff_delta,
            "phase_diff_corrected": swarm1_phase_diff_delta_corrected,
        },
        "aoa": {
            "tx2_id": tx2_id,
            "node_a": node_a,
            "node_b": node_b,
            "anchor_phase": anchor_phase,
            "compensated_phase": compensated_phase,
            "aoa_rad": float(aoa_rad),
            "aoa_deg": float(np.degrees(aoa_rad)),
        },
        "burst1": {
            "tx_id": tx1_id,
            "t": run_slot_output["t1"],
            "tx_bb_adc": run_slot_output["tx_bb1_adc"],
            "rx_bb_adc": rx_bb1,
        },
        "burst2": {
            "tx_id": run_slot_output["tx2_id"],
            "t": run_slot_output["t2"],
            "tx_bb_adc": run_slot_output["tx_bb2_adc"],
            "rx_bb_adc": run_slot_output["rx_bb2"],
        },
    }


def _burst_time_ms(t_start: float, n: int) -> np.ndarray:
    return (t_start + np.arange(n) / adc_fs) * 1e3


result = run_slot(4, 0.0)
analysis = analyze_slot_baseband(result)
tx1_id = result["tx1_id"]
tx2_id = result["tx2_id"]
t1 = result["t1"]
t2 = result["t2"]

fig, axes = plt.subplots(
    4,
    1,
    figsize=(10, 13),
    gridspec_kw={"height_ratios": [3, 1, 1, 1]},
    constrained_layout=True,
)

for node in swarm1:
    axes[0].scatter(node.x, node.y, s=80, marker="o")
    axes[0].text(node.x, node.y, f" {node.node_id}", fontsize=9)
for node in swarm2:
    axes[0].scatter(node.x, node.y, s=80, marker="s")
    axes[0].text(node.x, node.y, f" {node.node_id}", fontsize=9)

aoa_info = analysis["aoa"]
rx_a = nodes_by_id[aoa_info["node_a"]]
rx_b = nodes_by_id[aoa_info["node_b"]]
mid_x = (rx_a.x + rx_b.x) / 2.0
mid_y = (rx_a.y + rx_b.y) / 2.0
aoa_bearing = _aoa_bearing_rad(
    rx_a,
    rx_b,
    aoa_info["aoa_rad"],
    nodes_by_id[aoa_info["tx2_id"]],
)
ray_len = 0.8
axes[0].plot(
    [mid_x, mid_x + ray_len * np.cos(aoa_bearing)],
    [mid_y, mid_y + ray_len * np.sin(aoa_bearing)],
    color="C3",
    linewidth=2,
    linestyle="--",
    label=f"AoA estimate ({aoa_info['aoa_deg']:.1f} deg)",
)

axes[0].set_title("Node positions")
axes[0].set_xlabel("x (m)")
axes[0].set_ylabel("y (m)")
axes[0].legend(fontsize=8, loc="upper left")
axes[0].grid(True, alpha=0.3)
axes[0].set_aspect("equal")

axes[1].plot(
    _burst_time_ms(t1, len(result["tx_bb1_adc"])),
    np.angle(result["tx_bb1_adc"]),
    marker="o",
    markersize=4,
    linewidth=0.8,
    label=f"TX node {tx1_id}",
)
axes[1].plot(
    _burst_time_ms(t2, len(result["tx_bb2_adc"])),
    np.angle(result["tx_bb2_adc"]),
    marker="s",
    markersize=4,
    linewidth=0.8,
    label=f"TX node {tx2_id}",
)
axes[1].set_title(f"Slot {result['slot_index']}: TX baseband phase")
axes[1].set_xlabel("Time (ms)")
axes[1].set_ylabel("Phase (rad)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

for rid, bb_adc in result["rx_bb1"].items():
    axes[2].plot(
        _burst_time_ms(t1, len(bb_adc)),
        np.angle(bb_adc),
        marker="o",
        markersize=3,
        linewidth=0.8,
        label=f"RX node {rid} (from {tx1_id})",
    )
for rid, bb_adc in result["rx_bb2"].items():
    axes[2].plot(
        _burst_time_ms(t2, len(bb_adc)),
        np.angle(bb_adc),
        marker="s",
        markersize=3,
        linewidth=0.8,
        label=f"RX node {rid} (from {tx2_id})",
    )
axes[2].set_title(f"Slot {result['slot_index']}: RX baseband phase")
axes[2].set_xlabel("Time (ms)")
axes[2].set_ylabel("Phase (rad)")
axes[2].legend(fontsize=7)
axes[2].grid(True, alpha=0.3)

phase_diff_corrected = analysis["swarm1_phase_diff_delta"]["phase_diff_corrected"]
node_a = analysis["burst1_swarm1_rx_phase_diff"]["node_a"]
node_b = analysis["burst1_swarm1_rx_phase_diff"]["node_b"]
axes[3].plot(
    _burst_time_ms(t1, len(phase_diff_corrected)),
    phase_diff_corrected,
    marker="d",
    markersize=4,
    linewidth=0.8,
    color="C3",
    label=f"node {node_a} - node {node_b}",
)
axes[3].set_title(
    f"Slot {result['slot_index']}: swarm1 phase diff delta (corrected)"
)
axes[3].set_xlabel("Time (ms)")
axes[3].set_ylabel("Phase diff delta (rad)")
axes[3].legend()
axes[3].grid(True, alpha=0.3)

fig.savefig("test_single_link.png", dpi=150)
plt.show()
