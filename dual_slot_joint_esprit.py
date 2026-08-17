"""2-slot, 4-D CoRal steering + joint vs per-slot ESPRIT comparison.

Protocol per slot:
  t_s       : reference node 0 transmits
  t_s+Δt    : target node (5 or 6) transmits

Four virtual array dimensions per snapshot: [1, a_12, a_13, a_14]^T
  a_1k = calibrated ratio d_{target,1k} / d_{ref,1k} with CFO compensation.

Joint ESPRIT pools snapshots from both slots (rank-2 covariance).
Per-slot ESPRIT runs K=1 independently on each slot.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig, pinv

# ---------------------------------------------------------------------------
# Physical / protocol constants (aligned with array_simulator.py)
# ---------------------------------------------------------------------------
F_CENTER = 2.4e9
C_LIGHT = 3e8
DELTA_T = 150e-6
SAMPLE_INTERVAL_S = 1e-6
CTE_DURATION_S = 92e-6
N_SAMPLES = int(round(CTE_DURATION_S / SAMPLE_INTERVAL_S))
SLOT_SPACING_S = 0.5
FREQUENCY_OFFSET_RANGE_HZ = 5e4
BASELINE_D = 0.05  # |y_1 - y_2|, uniform RX spacing


def wrap_phase(phase: float) -> float:
    return float(np.arctan2(np.sin(phase), np.cos(phase)))


class Node:
    def __init__(
        self,
        node_id: int,
        x: float,
        y: float,
        initial_phase: float,
        frequency_offset: float,
    ) -> None:
        self.node_id = node_id
        self.x = x
        self.y = y
        self.initial_phase = wrap_phase(initial_phase)
        self.frequency_offset = frequency_offset

    def distance_to(self, other: Node) -> float:
        return float(np.hypot(self.x - other.x, self.y - other.y))


def make_node(rng: np.random.Generator, node_id: int, x: float, y: float) -> Node:
    return Node(
        node_id,
        x,
        y,
        wrap_phase(rng.uniform(0, 2 * np.pi)),
        float(rng.uniform(-FREQUENCY_OFFSET_RANGE_HZ, FREQUENCY_OFFSET_RANGE_HZ)),
    )


def build_nodes(rng: np.random.Generator) -> dict[int, Node]:
    return {
        0: make_node(rng, 0, 2.0, 2.0),
        1: make_node(rng, 1, 0.0, 0.15),
        2: make_node(rng, 2, 0.0, 0.10),
        3: make_node(rng, 3, 0.0, 0.05),
        4: make_node(rng, 4, 0.0, 0.00),
        # Separated in bearing so joint K=2 ESPRIT can resolve both (~13 deg apart).
        5: make_node(rng, 5, 5.0, 1.50),
        6: make_node(rng, 6, 5.0, 0.30),
    }


def fading_h(tx: Node, rx: Node) -> float:
    if tx.node_id == rx.node_id:
        return 0.0
    dist = rx.distance_to(tx)
    return 0.0 if dist < 1e-12 else 1.0 / dist


def signal_at_receiver(
    rx: Node,
    tx: Node,
    t0: float,
    fading: dict[tuple[int, int], float],
) -> np.ndarray:
    h = fading[(rx.node_id, tx.node_id)]
    tau = rx.distance_to(tx) / C_LIGHT
    t = t0 + np.arange(N_SAMPLES, dtype=np.float64) * SAMPLE_INTERVAL_S
    phase = (
        2 * np.pi * (tx.frequency_offset - rx.frequency_offset) * t
        + tx.initial_phase
        - rx.initial_phase
        - 2 * np.pi * (F_CENTER + tx.frequency_offset) * tau
    )
    return h * np.exp(1j * phase)


def add_awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.complex128)
    power = float(np.mean(np.abs(signal) ** 2))
    if power < 1e-20:
        return signal.copy()
    noise_power = power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal))
    )
    return signal + noise


def slope(signal: np.ndarray) -> float:
    phase = np.unwrap(np.angle(signal))
    t = np.arange(len(signal), dtype=np.float64) * SAMPLE_INTERVAL_S
    return float(np.polyfit(t, phase, 1)[0])


def normalize(signal: np.ndarray) -> np.ndarray:
    mag = np.abs(signal)
    safe = np.where(mag < 1e-12, 1.0, mag)
    return signal / safe


def delta_tau(rx_a: Node, rx_b: Node, tx: Node) -> float:
    return rx_a.distance_to(tx) / C_LIGHT - rx_b.distance_to(tx) / C_LIGHT


def true_theta_deg(rx_a: Node, rx_b: Node, tx: Node, d_sp: float) -> float:
    """Far-field AoA (deg) from TDOA on baseline rx_a–rx_b."""
    dt = delta_tau(rx_a, rx_b, tx)
    sin_theta = float(np.clip(dt * C_LIGHT / d_sp, -1.0, 1.0))
    return float(-np.degrees(np.arcsin(sin_theta)))


def compute_calibrated_steering(
    r_ref: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    r_tgt: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    m_factors: tuple[complex, complex, complex],
    freq_slopes: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r10, r20, r30, r40 = r_ref
    r1t, r2t, r3t, r4t = r_tgt

    d0_12, d0_13, d0_14 = r10 / r20, r10 / r30, r10 / r40
    d1_12, d1_13, d1_14 = r1t / r2t, r1t / r3t, r1t / r4t

    m1, m2, m3 = m_factors
    f12, f13, f14 = freq_slopes

    a12 = normalize(d1_12 / d0_12) * m1 * np.exp(-1j * f12 * DELTA_T)
    a13 = normalize(d1_13 / d0_13) * m2 * np.exp(-1j * f13 * DELTA_T)
    a14 = normalize(d1_14 / d0_14) * m3 * np.exp(-1j * f14 * DELTA_T)
    return a12, a13, a14


def build_4d_matrix(a12: np.ndarray, a13: np.ndarray, a14: np.ndarray) -> np.ndarray:
    """Stack calibrated steering snapshots as [1, φ, φ², φ³]^T (4 × N)."""
    n = len(a12)
    return np.vstack(
        [
            np.ones(n, dtype=np.complex128),
            np.asarray(a12, dtype=np.complex128),
            np.asarray(a13, dtype=np.complex128),
            np.asarray(a14, dtype=np.complex128),
        ]
    )


def esprit_angles_deg(
    x: np.ndarray,
    fc: float,
    d_sp: float,
    num_sources: int,
) -> np.ndarray:
    """ESPRIT on M×N data; returns num_sources angles (deg), sorted ascending."""
    m, n = x.shape
    if num_sources < 1 or num_sources >= m:
        raise ValueError(f"need 1 <= num_sources < M, got {num_sources}, M={m}")

    r_xx = x @ x.conj().T / n
    eigvals, eigvecs = eig(r_xx)
    idx = np.argsort(np.abs(eigvals))[::-1]
    es = eigvecs[:, idx[:num_sources]]

    es1 = es[:-1, :]
    es2 = es[1:, :]
    psi = pinv(es1) @ es2
    phi_vals = eig(psi)[0]

    angles = []
    for phi in phi_vals:
        sin_theta = float(np.clip(np.angle(phi) * C_LIGHT / (2 * np.pi * fc * d_sp), -1.0, 1.0))
        angles.append(float(np.degrees(np.arcsin(sin_theta))))
    return np.sort(np.asarray(angles, dtype=np.float64))


def match_angles_to_truth(
    est_deg: np.ndarray,
    true_deg: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Assign estimates to truths (2! permutations) and return errors + RMSE."""
    est = np.asarray(est_deg, dtype=np.float64)
    true = np.asarray(true_deg, dtype=np.float64)
    if len(est) != len(true):
        raise ValueError("estimate and truth must have same length")

    if len(true) == 1:
        err = np.array([est[0] - true[0]])
        return err, float(np.abs(err[0]))

    err_a = np.array([est[0] - true[0], est[1] - true[1]])
    err_b = np.array([est[0] - true[1], est[1] - true[0]])
    if np.sum(err_a**2) <= np.sum(err_b**2):
        return err_a, float(np.sqrt(np.mean(err_a**2)))
    return err_b, float(np.sqrt(np.mean(err_b**2)))


def simulate_slot(
    nodes: dict[int, Node],
    fading: dict[tuple[int, int], float],
    target_id: int,
    slot_start: float,
    snr_db: float | None,
    rng: np.random.Generator,
    m_factors: tuple[complex, complex, complex],
    freq_slopes: tuple[float, float, float],
) -> np.ndarray:
    """One slot: ref 0 at slot_start, target at slot_start+Δt → 4×N matrix."""
    ref = nodes[0]
    tgt = nodes[target_id]
    rx = [nodes[i] for i in (1, 2, 3, 4)]

    t_ref = slot_start
    t_tgt = slot_start + DELTA_T

    def rx_signals(tx: Node, t: float) -> tuple[np.ndarray, ...]:
        sigs = tuple(
            signal_at_receiver(r, tx, t, fading) for r in rx
        )
        if snr_db is not None:
            sigs = tuple(add_awgn(s, snr_db, rng) for s in sigs)
        return sigs

    r_ref = rx_signals(ref, t_ref)
    r_tgt = rx_signals(tgt, t_tgt)
    a12, a13, a14 = compute_calibrated_steering(
        r_ref, r_tgt, m_factors, freq_slopes
    )
    return build_4d_matrix(a12, a13, a14)


def calibration_from_ref_slot(
    nodes: dict[int, Node],
    fading: dict[tuple[int, int], float],
    slot_start: float,
    snr_db: float | None,
    rng: np.random.Generator,
) -> tuple[tuple[complex, complex, complex], tuple[float, float, float]]:
    """Estimate M_k and CFO slopes from reference-only ratios at slot_start."""
    ref = nodes[0]
    rx = [nodes[i] for i in (1, 2, 3, 4)]
    t_ref = slot_start

    def rx_signals(tx: Node, t: float) -> tuple[np.ndarray, ...]:
        sigs = tuple(signal_at_receiver(r, tx, t, fading) for r in rx)
        if snr_db is not None:
            sigs = tuple(add_awgn(s, snr_db, rng) for s in sigs)
        return sigs

    r10, r20, r30, r40 = rx_signals(ref, t_ref)
    d0_12, d0_13, d0_14 = r10 / r20, r10 / r30, r10 / r40
    freq_12, freq_13, freq_14 = slope(d0_12), slope(d0_13), slope(d0_14)

    rx1, rx2 = nodes[1], nodes[2]
    dt12 = delta_tau(rx1, rx2, ref)
    dt13 = delta_tau(rx1, nodes[3], ref)
    dt14 = delta_tau(rx1, nodes[4], ref)

    m1 = np.exp(-1j * 2 * np.pi * F_CENTER * dt12)
    m2 = np.exp(-1j * 2 * np.pi * F_CENTER * dt13)
    m3 = np.exp(-1j * 2 * np.pi * F_CENTER * dt14)
    return (complex(m1), complex(m2), complex(m3)), (freq_12, freq_13, freq_14)


def run_single_trial(
    nodes: dict[int, Node],
    fading: dict[tuple[int, int], float],
    snr_db: float,
    rng: np.random.Generator,
) -> dict:
    """One Monte-Carlo draw: joint K=2 vs per-slot K=1."""
    m_factors, freq_slopes = calibration_from_ref_slot(
        nodes, fading, slot_start=0.0, snr_db=snr_db, rng=rng
    )

    x5 = simulate_slot(
        nodes, fading, 5, slot_start=0.0, snr_db=snr_db, rng=rng,
        m_factors=m_factors, freq_slopes=freq_slopes,
    )
    x6 = simulate_slot(
        nodes, fading, 6, slot_start=SLOT_SPACING_S, snr_db=snr_db, rng=rng,
        m_factors=m_factors, freq_slopes=freq_slopes,
    )

    x_joint = np.hstack([x5, x6])
    est_joint = esprit_angles_deg(x_joint, F_CENTER, BASELINE_D, num_sources=2)
    est_slot5 = esprit_angles_deg(x5, F_CENTER, BASELINE_D, num_sources=1)
    est_slot6 = esprit_angles_deg(x6, F_CENTER, BASELINE_D, num_sources=1)
    est_sequential = np.sort(np.concatenate([est_slot5, est_slot6]))

    rx1, rx2 = nodes[1], nodes[2]
    true_angles = np.sort(
        np.array(
            [
                true_theta_deg(rx1, rx2, nodes[5], BASELINE_D),
                true_theta_deg(rx1, rx2, nodes[6], BASELINE_D),
            ]
        )
    )

    _, rmse_joint = match_angles_to_truth(est_joint, true_angles)
    _, rmse_seq = match_angles_to_truth(est_sequential, true_angles)
    err_joint, _ = match_angles_to_truth(est_joint, true_angles)
    err_seq, _ = match_angles_to_truth(est_sequential, true_angles)

    return {
        "true_angles": true_angles,
        "est_joint": est_joint,
        "est_sequential": est_sequential,
        "err_joint": err_joint,
        "err_sequential": err_seq,
        "rmse_joint": rmse_joint,
        "rmse_sequential": rmse_seq,
        "x5": x5,
        "x6": x6,
    }


def build_fading(nodes: dict[int, Node]) -> dict[tuple[int, int], float]:
    fading: dict[tuple[int, int], float] = {}
    for rx_id in nodes:
        for tx_id in nodes:
            fading[(rx_id, tx_id)] = fading_h(nodes[tx_id], nodes[rx_id])
    return fading


def run_monte_carlo(
    snr_db_range: np.ndarray,
    n_trials: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng_master = np.random.default_rng(seed)
    rmse_joint = np.zeros(len(snr_db_range))
    rmse_seq = np.zeros(len(snr_db_range))

    nodes = build_nodes(rng_master)
    fading = build_fading(nodes)

    for i, snr_db in enumerate(snr_db_range):
        trials_j: list[float] = []
        trials_s: list[float] = []
        for _ in range(n_trials):
            rng = np.random.default_rng(rng_master.integers(0, 2**31))
            out = run_single_trial(nodes, fading, float(snr_db), rng)
            trials_j.append(out["rmse_joint"])
            trials_s.append(out["rmse_sequential"])
        rmse_joint[i] = float(np.mean(trials_j))
        rmse_seq[i] = float(np.mean(trials_s))

    return snr_db_range, rmse_joint, rmse_seq


def match_est_to_truth(est_deg: np.ndarray, true_deg: np.ndarray) -> np.ndarray:
    """Reorder estimates to align with sorted true angles."""
    est = np.asarray(est_deg, dtype=np.float64)
    true = np.asarray(true_deg, dtype=np.float64)
    if len(true) == 1:
        return est
    err_a = abs(est[0] - true[0]) + abs(est[1] - true[1])
    err_b = abs(est[0] - true[1]) + abs(est[1] - true[0])
    if err_a <= err_b:
        return est
    return np.array([est[1], est[0]])


def aoa_bearing_rad(rx_a: Node, rx_b: Node, theta_deg: float, d_sp: float) -> float:
    """Global bearing (rad) from far-field θ and RX baseline 1–2."""
    dt = -np.sin(np.deg2rad(theta_deg)) * d_sp / C_LIGHT
    cos_aoa = float(np.clip(-dt * C_LIGHT / d_sp, -1.0, 1.0))
    aoa_rad = float(np.arccos(cos_aoa))
    bx = rx_a.x - rx_b.x
    by = rx_a.y - rx_b.y
    baseline_angle = float(np.arctan2(by, bx))
    if dt <= 0.0:
        return baseline_angle - aoa_rad
    return baseline_angle + aoa_rad


def ideal_phi(theta_deg: float) -> complex:
    sin_theta = float(np.sin(np.deg2rad(theta_deg)))
    return np.exp(1j * 2 * np.pi * F_CENTER * BASELINE_D * sin_theta / C_LIGHT)


def covariance_eigenvalues(x: np.ndarray) -> np.ndarray:
    r_xx = x @ x.conj().T / x.shape[1]
    return np.sort(np.real(np.linalg.eigvalsh(r_xx)))[::-1]


def plot_geometry_and_angles(
    ax: plt.Axes,
    nodes: dict[int, Node],
    true_by_id: dict[int, float],
    est_joint_by_id: dict[int, float],
    est_seq_by_id: dict[int, float],
    *,
    ray_len: float = 2.5,
) -> None:
    rx_nodes = [nodes[i] for i in (1, 2, 3, 4)]
    rx1, rx2 = nodes[1], nodes[2]
    mid_x = (rx1.x + rx2.x) / 2.0
    mid_y = (rx1.y + rx2.y) / 2.0

    ax.plot(
        [n.x for n in rx_nodes],
        [n.y for n in rx_nodes],
        "o-",
        color="C0",
        linewidth=2,
        markersize=9,
        label="RX 1–4",
        zorder=2,
    )
    ax.plot(nodes[0].x, nodes[0].y, "s", color="C2", markersize=10, label="Ref 0", zorder=3)
    ax.plot(nodes[5].x, nodes[5].y, "^", color="C3", markersize=11, label="TX 5", zorder=3)
    ax.plot(nodes[6].x, nodes[6].y, "v", color="C4", markersize=11, label="TX 6", zorder=3)

    for node, label in (
        (nodes[0], "0"),
        (nodes[1], "1"),
        (nodes[2], "2"),
        (nodes[3], "3"),
        (nodes[4], "4"),
        (nodes[5], "5"),
        (nodes[6], "6"),
    ):
        ax.annotate(label, (node.x, node.y), xytext=(5, 5), textcoords="offset points", fontsize=9)

    style_map = {
        "true": ("--", "black", 2.0),
        "joint": ("-", "C0", 1.8),
        "seq": ("-.", "C1", 1.8),
    }
    for tid, color_base in ((5, "C3"), (6, "C4")):
        th_true = true_by_id[tid]
        for kind, est_map in (("true", true_by_id), ("joint", est_joint_by_id), ("seq", est_seq_by_id)):
            th = est_map[tid] if kind != "true" else th_true
            bearing = aoa_bearing_rad(rx1, rx2, th, BASELINE_D)
            ls, color, lw = style_map[kind]
            label = None
            if tid == 5 and kind == "true":
                label = f"True θ_5={th_true:.1f}°"
            elif tid == 5 and kind == "joint":
                label = f"Joint θ_5={est_joint_by_id[5]:.1f}°"
            elif tid == 6 and kind == "seq":
                label = f"Per-slot θ_6={est_seq_by_id[6]:.1f}°"
            ax.plot(
                [mid_x, mid_x + ray_len * np.cos(bearing)],
                [mid_y, mid_y + ray_len * np.sin(bearing)],
                ls,
                color=color if kind != "true" else color_base,
                linewidth=lw,
                alpha=0.85 if kind == "true" else 0.75,
                label=label,
            )

    ax.plot(mid_x, mid_y, "k+", markersize=12, zorder=4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Node layout & AoA rays (baseline midpoint)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=7)


def plot_slot_timeline(ax: plt.Axes) -> None:
    slots = [
        ("Slot 0: ref 0", 0.0, DELTA_T, "C2"),
        ("Slot 0: TX 5", DELTA_T, CTE_DURATION_S, "C3"),
        ("Slot 1: ref 0", SLOT_SPACING_S, SLOT_SPACING_S + DELTA_T, "C2"),
        ("Slot 1: TX 6", SLOT_SPACING_S + DELTA_T, SLOT_SPACING_S + DELTA_T + CTE_DURATION_S, "C4"),
    ]
    for i, (label, t0, duration, color) in enumerate(slots):
        ax.barh(i, duration * 1e3, left=t0 * 1e3, height=0.55, color=color, alpha=0.75, edgecolor="k")
        ax.text(t0 * 1e3 + duration * 0.5e3, i, label, va="center", ha="center", fontsize=8, color="white", fontweight="bold")

    ax.set_yticks([])
    ax.set_xlabel("Time (ms)")
    ax.set_title(f"2-slot TDM protocol (Δt={DELTA_T*1e6:.0f} µs, spacing={SLOT_SPACING_S*1e3:.0f} ms)")
    ax.set_xlim(-5, (SLOT_SPACING_S + DELTA_T + CTE_DURATION_S) * 1e3 + 20)
    ax.grid(True, alpha=0.3, axis="x")


def plot_steering_phases(
    ax: plt.Axes,
    x_slot: np.ndarray,
    theta_true_deg: float,
    slot_label: str,
) -> None:
    """Compare mean measured phases vs ideal [φ, φ², φ³] for rows 1–3."""
    labels = ["a₁₂", "a₁₃", "a₁₄"]
    meas = np.degrees(np.angle([np.mean(x_slot[k, :]) for k in (1, 2, 3)]))
    phi = ideal_phi(theta_true_deg)
    ideal = np.degrees(np.angle([phi, phi**2, phi**3]))
    meas = (meas + 180) % 360 - 180
    ideal = (ideal + 180) % 360 - 180

    x_pos = np.arange(3)
    width = 0.35
    ax.bar(x_pos - width / 2, meas, width, label="Measured", color="C0")
    ax.bar(x_pos + width / 2, ideal, width, label="Ideal", color="C1", alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Phase (deg)")
    ax.set_title(f"{slot_label}: steering phases (θ_true={theta_true_deg:.1f}°)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="best", fontsize=7)


def plot_eigenvalues(
    ax: plt.Axes,
    x5: np.ndarray,
    x6: np.ndarray,
) -> None:
    evals_slot5 = covariance_eigenvalues(x5)
    evals_slot6 = covariance_eigenvalues(x6)
    evals_joint = covariance_eigenvalues(np.hstack([x5, x6]))

    x_pos = np.arange(4)
    width = 0.25
    ax.bar(x_pos - width, evals_joint, width, label="Joint [X₅,X₆]", color="C0")
    ax.bar(x_pos, evals_slot5, width, label="Slot 5 only", color="C3", alpha=0.85)
    ax.bar(x_pos + width, evals_slot6, width, label="Slot 6 only", color="C4", alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"λ{i+1}" for i in range(4)])
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Sample covariance eigenvalues (rank-1 per slot, rank-2 joint)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="best", fontsize=7)


def visualize_full(
    nodes: dict[int, Node],
    demo: dict,
    snr_axis: np.ndarray,
    rmse_joint: np.ndarray,
    rmse_seq: np.ndarray,
    *,
    demo_snr_db: float = 15.0,
) -> None:
    """Multi-panel overview: geometry, timeline, steering, eigenvalues, estimates, RMSE."""
    rx1, rx2 = nodes[1], nodes[2]
    true_by_id = {
        5: true_theta_deg(rx1, rx2, nodes[5], BASELINE_D),
        6: true_theta_deg(rx1, rx2, nodes[6], BASELINE_D),
    }
    true_sorted = np.sort([true_by_id[5], true_by_id[6]])
    est_joint_m = match_est_to_truth(demo["est_joint"], true_sorted)
    est_seq_m = match_est_to_truth(demo["est_sequential"], true_sorted)
    # Map sorted matched estimates back to node IDs by angle.
    ids_by_angle = sorted((5, 6), key=lambda tid: true_by_id[tid])
    est_joint_by_id = {ids_by_angle[i]: float(est_joint_m[i]) for i in range(2)}
    est_seq_by_id = {ids_by_angle[i]: float(est_seq_m[i]) for i in range(2)}

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1.0, 1.0], hspace=0.38, wspace=0.32)

    ax_geo = fig.add_subplot(gs[0, 0:2])
    plot_geometry_and_angles(ax_geo, nodes, true_by_id, est_joint_by_id, est_seq_by_id)

    ax_time = fig.add_subplot(gs[0, 2])
    plot_slot_timeline(ax_time)

    ax_ph5 = fig.add_subplot(gs[1, 0])
    plot_steering_phases(ax_ph5, demo["x5"], true_by_id[5], "Slot 0")

    ax_ph6 = fig.add_subplot(gs[1, 1])
    plot_steering_phases(ax_ph6, demo["x6"], true_by_id[6], "Slot 1")

    ax_eval = fig.add_subplot(gs[1, 2])
    plot_eigenvalues(ax_eval, demo["x5"], demo["x6"])

    ax_bar = fig.add_subplot(gs[2, 0])
    true_a = demo["true_angles"]
    x_pos = np.arange(2)
    width = 0.35
    ax_bar.bar(x_pos - width / 2, est_joint_m, width, label="Joint K=2", color="C0")
    ax_bar.bar(x_pos + width / 2, est_seq_m, width, label="Per-slot K=1", color="C1")
    for i, t in enumerate(true_a):
        ax_bar.axhline(t, color=f"C{i+2}", linestyle="--", linewidth=1.5, label=f"True {t:.1f}°")
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(["Lower θ", "Higher θ"])
    ax_bar.set_ylabel("θ (deg)")
    ax_bar.set_title(f"Matched estimates @ {demo_snr_db:.0f} dB")
    ax_bar.grid(True, alpha=0.3, axis="y")
    ax_bar.legend(loc="best", fontsize=7)

    ax_rmse = fig.add_subplot(gs[2, 1:])
    ax_rmse.plot(snr_axis, rmse_joint, "o-", label="Joint ESPRIT (K=2)", linewidth=2)
    ax_rmse.plot(snr_axis, rmse_seq, "s-", label="Per-slot ESPRIT (K=1×2)", linewidth=2)
    ax_rmse.set_xlabel("SNR (dB)")
    ax_rmse.set_ylabel("Mean matched RMSE (deg)")
    ax_rmse.set_title("Monte Carlo comparison")
    ax_rmse.grid(True, alpha=0.3)
    ax_rmse.legend(loc="best")

    sep = abs(true_by_id[5] - true_by_id[6])
    fig.suptitle(
        f"CoRal 2-slot dual-source ESPRIT | angular separation = {sep:.1f}° | "
        f"joint RMSE={demo['rmse_joint']:.2f}° per-slot RMSE={demo['rmse_sequential']:.2f}° @ {demo_snr_db:.0f} dB",
        fontsize=12,
    )
    fig.savefig("dual_slot_joint_esprit_overview.png", dpi=150, bbox_inches="tight")
    print("Saved figure: dual_slot_joint_esprit_overview.png")


def plot_results(
    nodes: dict[int, Node],
    demo: dict,
    snr_axis: np.ndarray,
    rmse_joint: np.ndarray,
    rmse_seq: np.ndarray,
    *,
    demo_snr_db: float = 15.0,
) -> None:
    visualize_full(nodes, demo, snr_axis, rmse_joint, rmse_seq, demo_snr_db=demo_snr_db)

    true_a = demo["true_angles"]
    est_joint_m = match_est_to_truth(demo["est_joint"], true_a)
    est_seq_m = match_est_to_truth(demo["est_sequential"], true_a)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    x_pos = np.arange(2)
    width = 0.35
    ax.bar(x_pos - width / 2, est_joint_m, width, label="Joint K=2", color="C0")
    ax.bar(x_pos + width / 2, est_seq_m, width, label="Per-slot K=1", color="C1")
    for i, t in enumerate(true_a):
        ax.axhline(t, color=f"C{i+2}", linestyle="--", linewidth=1.5, label=f"True {t:.2f}°")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Lower θ", "Higher θ"])
    ax.set_ylabel("Estimated θ (deg)")
    ax.set_title(f"Single trial @ SNR = {demo_snr_db:.0f} dB")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="best", fontsize=8)

    ax2 = axes[1]
    ax2.plot(snr_axis, rmse_joint, "o-", label="Joint ESPRIT (K=2)", linewidth=2)
    ax2.plot(snr_axis, rmse_seq, "s-", label="Per-slot ESPRIT (K=1 × 2)", linewidth=2)
    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel("Mean matched RMSE (deg)")
    ax2.set_title("2-slot Monte Carlo (200 trials per SNR)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    fig.suptitle("CoRal 4-D steering: joint vs per-slot dual-source ESPRIT")
    fig.tight_layout()
    fig.savefig("dual_slot_joint_esprit.png", dpi=150)
    print("Saved figure: dual_slot_joint_esprit.png")


def main() -> None:
    rng = np.random.default_rng(42)
    nodes = build_nodes(rng)
    fading = build_fading(nodes)

    print("=== 2-slot, 4-D dual-source ESPRIT simulation ===")
    rx1, rx2 = nodes[1], nodes[2]
    true_list = []
    for tid in (5, 6):
        th = true_theta_deg(rx1, rx2, nodes[tid], BASELINE_D)
        true_list.append(th)
        print(f"True θ_{tid} (deg): {th:.4f}")
    print(f"Angular separation: {abs(true_list[0] - true_list[1]):.2f} deg")

    demo = run_single_trial(nodes, fading, snr_db=15.0, rng=rng)
    print("\n--- Single trial @ 15 dB ---")
    print("True angles (sorted):     ", demo["true_angles"])
    print("Joint ESPRIT (K=2):       ", demo["est_joint"])
    print("Per-slot ESPRIT (K=1×2):  ", demo["est_sequential"])
    print("Errors joint (matched): ", demo["err_joint"])
    print("Errors per-slot:          ", demo["err_sequential"])
    print(f"RMSE joint:    {demo['rmse_joint']:.4f}°")
    print(f"RMSE per-slot: {demo['rmse_sequential']:.4f}°")

    snr_axis = np.arange(0, 41, 5)
    n_trials = 200
    print(f"\n--- Monte Carlo ({n_trials} trials/SNR) ---")
    snr_axis, rmse_joint, rmse_seq = run_monte_carlo(snr_axis, n_trials, seed=123)
    print(f"{'SNR':>4} | {'Joint':>8} | {'Per-slot':>8}")
    for snr, rj, rs in zip(snr_axis, rmse_joint, rmse_seq):
        print(f"{snr:4d} | {rj:8.3f}° | {rs:8.3f}°")

    plot_results(nodes, demo, snr_axis, rmse_joint, rmse_seq, demo_snr_db=15.0)


if __name__ == "__main__":
    main()
