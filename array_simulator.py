import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
FREQUENCY_OFFSET_RANGE_HZ = 5e4

f_center = 2.4e9  # center frequency
c = 3e8  # speed of light
delta_t = 150e-6  # transmit interval 150us
SAMPLE_INTERVAL_S = 1e-6  # 1 us per sample
CTE_DURATION_S = 92e-6  # 92 us total
N_SAMPLES = int(round(CTE_DURATION_S / SAMPLE_INTERVAL_S))

def wrap_phase(phase: float) -> float:
    """Wrap phase to [-pi, pi]."""
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

    def distance_to(self, other: "Node") -> float:
        return float(np.hypot(self.x - other.x, self.y - other.y))


def make_node(node_id: int, x: float, y: float) -> Node:
    initial_phase = wrap_phase(np.random.uniform(0, 2 * np.pi))
    frequency_offset = np.random.uniform(
        -FREQUENCY_OFFSET_RANGE_HZ,
        FREQUENCY_OFFSET_RANGE_HZ,
    )
    return Node(node_id, x, y, initial_phase, frequency_offset)


# reference node
node0 = make_node(0, 2, 2)

# receivers 1–4, one column
node1 = make_node(1, 0, 0.15)
node2 = make_node(2, 0, 0.10)
node3 = make_node(3, 0, 0.05)
node4 = make_node(4, 0, 0.00)

# transmitters 5–8, one column
node5 = make_node(5, 5, 1.15)
node6 = make_node(6, 5, 1.10)
node7 = make_node(7, 5, 1.05)
node8 = make_node(8, 5, 1.00)

nodes: dict[int, Node] = {
    0: node0,
    1: node1,
    2: node2,
    3: node3,
    4: node4,
    5: node5,
    6: node6,
    7: node7,
    8: node8,
}

NODE_IDS = tuple(sorted(nodes.keys()))


def amplitude_fading_coefficient(transmitter: Node, receiver: Node) -> float:
    """Amplitude fading from transmitter to receiver (1 / distance)."""
    if transmitter.node_id == receiver.node_id:
        return 0.0
    distance = receiver.distance_to(transmitter)
    if distance < 1e-12:
        return 0.0
    return 1.0 / distance


def build_amplitude_fading_matrix(
    node_dict: dict[int, Node],
    node_ids: tuple[int, ...] = NODE_IDS,
) -> np.ndarray:
    """Matrix H where H[i, j] is fading from node j (TX) to node i (RX)."""
    n = len(node_ids)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i, rx_id in enumerate(node_ids):
        for j, tx_id in enumerate(node_ids):
            matrix[i, j] = amplitude_fading_coefficient(
                node_dict[tx_id], node_dict[rx_id]
            )
    return matrix


amplitude_fading_matrix = build_amplitude_fading_matrix(nodes)


def signal_at_receiver(
    receiver_node: Node, transmitter_node: Node, t: float
) -> np.ndarray:
    """Received signal R_{j,i}(t) from transmitter i to receiver j.

    R_{j,i}(t) = h_{j,i} exp(j(2π(f_i - f_j)t + φ_i - φ_j - 2π(f_0 + f_i)τ_{j,i}))

    Samples from t, 1 us interval, 92 us total.
    """
    j = receiver_node
    i = transmitter_node

    h_ji = amplitude_fading_matrix[j.node_id, i.node_id]
    tau_ji = j.distance_to(i) / c
    t_samples = t + np.arange(N_SAMPLES, dtype=np.float64) * SAMPLE_INTERVAL_S

    f_i = i.frequency_offset
    f_j = j.frequency_offset
    phi_i = i.initial_phase
    phi_j = j.initial_phase

    # Baseband beat model: only (f_i - f_j) is time-varying; do not wrap f_0 term.
    phase = (
        2 * np.pi * (f_i - f_j) * t_samples
        + phi_i
        - phi_j
        - 2 * np.pi * (f_center + f_i) * tau_ji
    )
    return h_ji * np.exp(1j * phase)

def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Add complex AWGN; SNR = mean signal power / noise power (dB)."""
    signal = np.asarray(signal, dtype=np.complex128)
    power = float(np.mean(np.abs(signal) ** 2))
    if power < 1e-20:
        return signal.copy()
    noise_power = power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )
    return signal + noise


def compute_steering_samples(
    r10: np.ndarray,
    r20: np.ndarray,
    r30: np.ndarray,
    r40: np.ndarray,
    r15: np.ndarray,
    r25: np.ndarray,
    r35: np.ndarray,
    r45: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibrated steering snapshots [a_12, a_13, a_14] from received signals."""
    d0_12_l = r10 / r20
    d0_13_l = r10 / r30
    d0_14_l = r10 / r40
    d5_12_l = r15 / r25
    d5_13_l = r15 / r35
    d5_14_l = r15 / r45

    delta_50_l = d5_12_l / d0_12_l
    freq_12 = slope(d0_12_l)
    freq_13 = slope(d0_13_l)
    freq_14 = slope(d0_14_l)

    m1 = np.exp(-1j * 2 * np.pi * f_center * delta_tau_0_12)
    m2 = np.exp(-1j * 2 * np.pi * f_center * delta_tau_0_13)
    m3 = np.exp(-1j * 2 * np.pi * f_center * delta_tau_0_14)

    a12 = normalize(delta_50_l) * m1 * np.exp(-1j * freq_12 * delta_t)
    a13 = normalize(d5_13_l / d0_13_l) * m2 * np.exp(-1j * freq_13 * delta_t)
    a14 = normalize(d5_14_l / d0_14_l) * m3 * np.exp(-1j * freq_14 * delta_t)
    return a12, a13, a14


def estimate_theta_2rx(
    a12: np.ndarray,
    fc: float,
    d_sp: float,
    c_light: float,
    *,
    snapshot: int | None = None,
) -> float:
    """Single baseline (node 1–2): θ from a_12 phase.

    snapshot=None: coherent mean over all samples (best-case 2-RX).
    snapshot=k: use one snapshot (fair comparison with spatial ESPRIT under noise).
    """
    if snapshot is None:
        val = complex(np.mean(a12))
    else:
        val = complex(a12[snapshot])
    sin_theta = float(np.clip(np.angle(val) * c_light / (2 * np.pi * fc * d_sp), -1.0, 1.0))
    return float(np.degrees(np.arcsin(sin_theta)))


R10 = signal_at_receiver(node1, node0, 0)
R20 = signal_at_receiver(node2, node0, 0)
R30 = signal_at_receiver(node3, node0, 0)
R40 = signal_at_receiver(node4, node0, 0)

R15 = signal_at_receiver(node1, node5, delta_t)
R25 = signal_at_receiver(node2, node5, delta_t)
R35 = signal_at_receiver(node3, node5, delta_t)
R45 = signal_at_receiver(node4, node5, delta_t)


d5_12 = R15 / R25
d0_12 = R10 / R20


def slope(signal: np.ndarray) -> float:
    """Phase slope (rad/s) of complex signal samples vs time."""
    phase = np.unwrap(np.angle(signal))
    t = np.arange(len(signal), dtype=np.float64) * SAMPLE_INTERVAL_S
    return float(np.polyfit(t, phase, 1)[0])


frequency_offset_12 = slope(d0_12)

delta_50 = d5_12 / d0_12

# delta_tau_0 is known from geometry (reference calibration baseline).
delta_tau_0 = node1.distance_to(node0) / c - node2.distance_to(node0) / c
true_delta_tau_5 = node1.distance_to(node5) / c - node2.distance_to(node5) / c

# angle Δ_{y_{5,0}} = -2π(f_1-f_2)Δt - 2π f_center Δτ_5 + 2π f_center Δτ_0
# (ignore f_i << f_center in propagation terms)
#
# CFO term from R only: slope(d0_12) = -2π(f_1-f_2)  =>  -2π(f_1-f_2)Δt = slope(d0_12)*Δt
cfo_phase_from_r = frequency_offset_12 * delta_t
geo_phase = float(
    np.angle(np.mean(delta_50) * np.exp(-1j * cfo_phase_from_r))
)

# geo_phase ≈ -2π f_center Δτ_5 + 2π f_center Δτ_0
delta_tau_5 = (2 * np.pi * f_center * delta_tau_0 - geo_phase) / (
    2 * np.pi * f_center
)

print("estimated delta_tau_5:", delta_tau_5)
print("true delta_tau_5:     ", true_delta_tau_5)

wavelength = c / f_center
baseline_len_12 = node1.distance_to(node2)


def aoa_from_delta_tau(delta_tau: float, baseline_len: float) -> float:
    """Far-field AoA (rad): angle between baseline and source direction.

    (b · u) / c = Δτ  =>  cos(aoa) = -(Δτ c / |b|)
    """
    cos_aoa = float(np.clip(-delta_tau * c / baseline_len, -1.0, 1.0))
    return float(np.arccos(cos_aoa))


def bearing_from_aoa(
    rx_a: Node,
    rx_b: Node,
    aoa_rad: float,
    delta_tau: float,
) -> float:
    """Global bearing (from +x) from AoA and TDOA sign."""
    bx = rx_a.x - rx_b.x
    by = rx_a.y - rx_b.y
    baseline_angle = float(np.arctan2(by, bx))
    if delta_tau <= 0.0:
        return baseline_angle - aoa_rad
    return baseline_angle + aoa_rad


def true_bearing_rad(rx_a: Node, rx_b: Node, source: Node) -> float:
    mid_x = (rx_a.x + rx_b.x) / 2.0
    mid_y = (rx_a.y + rx_b.y) / 2.0
    return float(np.arctan2(source.y - mid_y, source.x - mid_x))


def true_aoa_rad(rx_a: Node, rx_b: Node, source: Node) -> float:
    mid_x = (rx_a.x + rx_b.x) / 2.0
    mid_y = (rx_a.y + rx_b.y) / 2.0
    bx = rx_a.x - rx_b.x
    by = rx_a.y - rx_b.y
    sx = source.x - mid_x
    sy = source.y - mid_y
    baseline_len = float(np.hypot(bx, by))
    source_len = float(np.hypot(sx, sy))
    if baseline_len < 1e-12 or source_len < 1e-12:
        return 0.0
    cos_aoa = (bx * sx + by * sy) / (baseline_len * source_len)
    return float(np.arccos(np.clip(cos_aoa, -1.0, 1.0)))


def visualize_aoa_from_tau(
    rx_a: Node,
    rx_b: Node,
    source: Node,
    ref_node: Node,
    bearing_est_rad: float,
    bearing_true_rad: float,
    *,
    ray_len: float = 2.0,
) -> None:
    mid_x = (rx_a.x + rx_b.x) / 2.0
    mid_y = (rx_a.y + rx_b.y) / 2.0

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(
        [rx_a.x, rx_b.x],
        [rx_a.y, rx_b.y],
        "o-",
        color="C0",
        linewidth=2,
        markersize=10,
        label="RX baseline (1–2)",
    )
    ax.plot(ref_node.x, ref_node.y, "s", color="C2", markersize=10, label="Ref TX 0")
    ax.plot(source.x, source.y, "^", color="C3", markersize=11, label="TX 5")

    for node, label in ((rx_a, "1"), (rx_b, "2"), (ref_node, "0"), (source, "5")):
        ax.annotate(label, (node.x, node.y), xytext=(5, 5), textcoords="offset points")

    ax.plot(
        [mid_x, mid_x + ray_len * np.cos(bearing_true_rad)],
        [mid_y, mid_y + ray_len * np.sin(bearing_true_rad)],
        "--",
        color="C2",
        linewidth=2,
        label=f"True {np.degrees(bearing_true_rad):.2f} deg",
    )
    ax.plot(
        [mid_x, mid_x + ray_len * np.cos(bearing_est_rad)],
        [mid_y, mid_y + ray_len * np.sin(bearing_est_rad)],
        "-",
        color="C1",
        linewidth=2,
        label=f"AoA from tau {np.degrees(bearing_est_rad):.2f} deg",
    )

    ax.plot(mid_x, mid_y, "k+", markersize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("AoA from true_delta_tau_5: node 5 relative to RX 1–2")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    plt.show()


# AoA from true_delta_tau_5 (far-field)
aoa_5_rad = aoa_from_delta_tau(true_delta_tau_5, baseline_len_12)
bearing_5_rad = bearing_from_aoa(node1, node2, aoa_5_rad, true_delta_tau_5)

true_aoa_5_rad = true_aoa_rad(node1, node2, node5)
true_bearing_5_rad = true_bearing_rad(node1, node2, node5)

print("--- AoA from true_delta_tau_5 ---")
print("AoA (deg):              ", np.degrees(aoa_5_rad))
print("true AoA (deg):         ", np.degrees(true_aoa_5_rad))
print("bearing (deg):          ", np.degrees(bearing_5_rad))
print("true bearing (deg):     ", np.degrees(true_bearing_5_rad))

visualize_aoa_from_tau(
    node1,
    node2,
    node5,
    node0,
    bearing_5_rad,
    true_bearing_5_rad,
)
d0_13 = R10 / R30
d0_14 = R10 / R40
d5_13 = R15 / R35
d5_14 = R15 / R45

frequency_offset_13 = slope(d0_13)
frequency_offset_14 = slope(d0_14)


def normalize(signal: np.ndarray) -> np.ndarray:
    """Normalize complex signal so each sample has unit magnitude (|z| = 1)."""
    signal = np.asarray(signal, dtype=np.complex128)
    magnitude = np.abs(signal)
    safe_mag = np.where(magnitude < 1e-12, 1.0, magnitude)
    return signal / safe_mag


d = baseline_len_12
delta_tau_0_12 = node1.distance_to(node0) / c - node2.distance_to(node0) / c
delta_tau_0_13 = node1.distance_to(node0) / c - node3.distance_to(node0) / c
delta_tau_0_14 = node1.distance_to(node0) / c - node4.distance_to(node0) / c

a_12, a_13, a_14 = compute_steering_samples(R10, R20, R30, R40, R15, R25, R35, R45)


def prepare_spatial_covariance(
    a12: np.ndarray,
    a13: np.ndarray,
    a14: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Align snapshots to [1, φ, φ²] and compute sample covariance."""
    x = np.vstack([np.asarray(a12), np.asarray(a13), np.asarray(a14)])
    ref = x[0:1, :]
    safe_ref = np.where(np.abs(ref) < 1e-12, 1.0 + 0j, ref)
    x_aligned = x / safe_ref
    r_xx = x_aligned @ x_aligned.conj().T / x_aligned.shape[1]
    return x_aligned, r_xx


def aligned_steering_vector(theta_deg: float, fc: float, d_sp: float, c_light: float) -> np.ndarray:
    """Steering [1, φ, φ²] with φ = exp(+j2π fc d sinθ / c).

    Matches calibrated data / ESPRIT: sinθ = angle(φ)·c/(2π fc d).
    """
    sin_theta = float(np.sin(np.deg2rad(theta_deg)))
    phi = np.exp(1j * 2 * np.pi * fc * d_sp * sin_theta / c_light)
    return np.array([1.0, phi, phi**2], dtype=np.complex128)


def esprit_steering(
    a12: np.ndarray,
    a13: np.ndarray,
    a14: np.ndarray,
    fc: float,
    d: float,
    c: float,
) -> tuple[float, complex, np.ndarray]:
    """ESPRIT for steering vector [a_12, a_13, a_14] = [φ, φ², φ³].

    φ = exp(+j2π fc d sinθ / c)
    Subarray 1: [a_12, a_13], subarray 2: [a_13, a_14].
    """
    _, r_xx = prepare_spatial_covariance(a12, a13, a14)

    eigvals, eigvecs = np.linalg.eigh(r_xx)
    es = eigvecs[:, np.argmax(eigvals)].reshape(-1, 1)

    es1 = es[0:2, :]  # [a_12, a_13]
    es2 = es[1:3, :]  # [a_13, a_14]

    psi = np.linalg.pinv(es1) @ es2
    phi = complex(psi[0, 0])

    # φ = exp(+j2π fc d sinθ/c)  =>  sinθ = +angle(φ)·c/(2π fc d)
    sin_theta = float(np.clip(np.angle(phi) * c / (2 * np.pi * fc * d), -1.0, 1.0))
    theta_deg = float(np.degrees(np.arcsin(sin_theta)))

    steering_vector = np.array([np.mean(a12), np.mean(a13), np.mean(a14)])
    return theta_deg, phi, steering_vector


def music_steering(
    a12: np.ndarray,
    a13: np.ndarray,
    a14: np.ndarray,
    fc: float,
    d_sp: float,
    c_light: float,
    *,
    num_sources: int = 1,
    scan_angles_deg: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """MUSIC for aligned steering vector [1, φ, φ²].

    P(θ) = 1 / (a(θ)^H En En^H a(θ))
    """
    if scan_angles_deg is None:
        scan_angles_deg = np.linspace(-90.0, 90.0, 721)

    _, r_xx = prepare_spatial_covariance(a12, a13, a14)
    eigvals, eigvecs = np.linalg.eigh(r_xx)
    idx = np.argsort(eigvals)
    en = eigvecs[:, idx[: -num_sources]]

    spectrum = np.zeros(len(scan_angles_deg))
    en_proj = en @ en.conj().T
    for i, theta_deg in enumerate(scan_angles_deg):
        a = aligned_steering_vector(theta_deg, fc, d_sp, c_light).reshape(-1, 1)
        denom = float(np.real(a.conj().T @ en_proj @ a)[0, 0])
        spectrum[i] = 1.0 / max(denom, 1e-20)

    theta_music_deg = float(scan_angles_deg[int(np.argmax(spectrum))])
    return theta_music_deg, scan_angles_deg, spectrum


def visualize_dof_algorithms(
    rx_nodes: tuple[Node, Node, Node, Node],
    source: Node,
    ref_node: Node,
    steering_vector: np.ndarray,
    phi_esprit: complex,
    theta_esprit_deg: float,
    theta_music_deg: float,
    scan_angles_deg: np.ndarray,
    music_spectrum: np.ndarray,
    theta_true_deg: float,
    *,
    ray_len: float = 2.0,
) -> None:
    """Visualize ESPRIT / MUSIC: geometry, steering phases, MUSIC spectrum."""
    rx1, rx2, rx3, rx4 = rx_nodes
    mid_x = (rx1.x + rx2.x) / 2.0
    mid_y = (rx1.y + rx2.y) / 2.0

    def bearing_from_theta(theta_deg: float) -> float:
        delta_tau = -np.sin(np.deg2rad(theta_deg)) * d / c
        aoa_rad = aoa_from_delta_tau(delta_tau, d)
        return bearing_from_aoa(rx1, rx2, aoa_rad, delta_tau)

    bearing_true_rad = bearing_from_theta(theta_true_deg)
    bearing_esprit_rad = bearing_from_theta(theta_esprit_deg)
    bearing_music_rad = bearing_from_theta(theta_music_deg)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.plot(
        [rx1.x, rx2.x, rx3.x, rx4.x],
        [rx1.y, rx2.y, rx3.y, rx4.y],
        "o-",
        color="C0",
        linewidth=2,
        markersize=9,
        label="RX 1–4",
    )
    ax.plot(ref_node.x, ref_node.y, "s", color="C2", markersize=10, label="Ref TX 0")
    ax.plot(source.x, source.y, "^", color="C3", markersize=11, label="TX 5")
    for node, label in ((rx1, "1"), (rx2, "2"), (rx3, "3"), (rx4, "4"), (ref_node, "0"), (source, "5")):
        ax.annotate(label, (node.x, node.y), xytext=(5, 5), textcoords="offset points")

    for bearing, style, color, label in (
        (bearing_true_rad, "--", "C2", f"True {theta_true_deg:.2f}°"),
        (bearing_esprit_rad, "-", "C1", f"ESPRIT {theta_esprit_deg:.2f}°"),
        (bearing_music_rad, "-.", "C4", f"MUSIC {theta_music_deg:.2f}°"),
    ):
        ax.plot(
            [mid_x, mid_x + ray_len * np.cos(bearing)],
            [mid_y, mid_y + ray_len * np.sin(bearing)],
            style,
            color=color,
            linewidth=2,
            label=label,
        )
    ax.plot(mid_x, mid_y, "k+", markersize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("AoA: node 5 relative to RX 1–2")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax2 = axes[1]
    labels = ["a_12", "a_13", "a_14"]
    meas_phase = np.degrees(np.angle(steering_vector))
    phi_angle = float(np.degrees(np.angle(phi_esprit)))
    ideal_phase = np.array([phi_angle, 2 * phi_angle, 3 * phi_angle])
    ideal_phase = (ideal_phase + 180) % 360 - 180
    x_pos = np.arange(len(labels))
    width = 0.35
    ax2.bar(x_pos - width / 2, meas_phase, width, label="Measured", color="C0")
    ax2.bar(x_pos + width / 2, ideal_phase, width, label="Ideal [φ, φ², φ³]", color="C1")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Phase (deg)")
    ax2.set_title("Steering vector phase")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend(loc="best", fontsize=9)

    ax3 = axes[2]
    spec_db = 10 * np.log10(music_spectrum / np.max(music_spectrum))
    ax3.plot(scan_angles_deg, spec_db, color="C4", linewidth=1.5)
    ax3.axvline(theta_true_deg, color="C2", linestyle="--", label=f"True {theta_true_deg:.2f}°")
    ax3.axvline(theta_music_deg, color="C4", linestyle="-.", label=f"MUSIC {theta_music_deg:.2f}°")
    ax3.axvline(theta_esprit_deg, color="C1", linestyle="-", label=f"ESPRIT {theta_esprit_deg:.2f}°")
    ax3.set_xlabel("θ (deg)")
    ax3.set_ylabel("Spectrum (dB)")
    ax3.set_title("MUSIC spatial spectrum")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"True={theta_true_deg:.2f}° | ESPRIT err={theta_esprit_deg - theta_true_deg:.3f}° "
        f"| MUSIC err={theta_music_deg - theta_true_deg:.3f}°"
    )
    fig.tight_layout()
    plt.show()


fc = f_center

theta_esprit_deg, phi_esprit, steering_vector = esprit_steering(
    a_12, a_13, a_14, fc, d, c
)
theta_music_deg, scan_angles_deg, music_spectrum = music_steering(
    a_12, a_13, a_14, fc, d, c
)
theta_2rx_deg = estimate_theta_2rx(a_12, fc, d, c)

sin_theta_5 = float(np.clip(true_delta_tau_5 * c / d, -1.0, 1.0))
theta_5_true_deg = float(-np.degrees(np.arcsin(sin_theta_5)))

phi_ratio_21 = complex(np.mean(a_13 / a_12))
phi_ratio_32 = complex(np.mean(a_14 / a_13))
theta_ratio_deg = float(
    np.degrees(
        np.arcsin(
            np.clip(np.angle(phi_ratio_21) * c / (2 * np.pi * fc * d), -1.0, 1.0)
        )
    )
)

print("--- DOA: 2-RX / ESPRIT / MUSIC ---")
print("steering_vector:        ", steering_vector)
print("phi (ESPRIT):           ", phi_esprit)
print("phi (a_13/a_12):        ", phi_ratio_21)
print("phi (a_14/a_13):        ", phi_ratio_32)
print("fc = f_0 + f_5 (Hz):    ", fc)
print("2-RX theta (deg):       ", theta_2rx_deg)
print("ESPRIT theta (deg):     ", theta_esprit_deg)
print("MUSIC theta (deg):      ", theta_music_deg)
print("ratio theta (deg):      ", theta_ratio_deg)
print("true theta (deg):       ", theta_5_true_deg)
print("2-RX error (deg):       ", theta_2rx_deg - theta_5_true_deg)
print("ESPRIT error (deg):     ", theta_esprit_deg - theta_5_true_deg)
print("MUSIC error (deg):      ", theta_music_deg - theta_5_true_deg)

visualize_dof_algorithms(
    (node1, node2, node3, node4),
    node5,
    node0,
    steering_vector,
    phi_esprit,
    theta_esprit_deg,
    theta_music_deg,
    scan_angles_deg,
    music_spectrum,
    theta_5_true_deg,
)


def run_noise_comparison(
    snr_db_range: np.ndarray,
    n_trials: int,
    theta_true_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Monte Carlo: RMSE of 2-RX vs ESPRIT vs MUSIC under AWGN."""
    rmse_2rx = np.zeros(len(snr_db_range))
    rmse_esprit = np.zeros(len(snr_db_range))
    rmse_music = np.zeros(len(snr_db_range))

    clean = (R10, R20, R30, R40, R15, R25, R35, R45)

    for i, snr_db in enumerate(snr_db_range):
        err_2rx = np.zeros(n_trials)
        err_esprit = np.zeros(n_trials)
        err_music = np.zeros(n_trials)
        for t in range(n_trials):
            noisy = [add_awgn(sig, snr_db) for sig in clean]
            a12, a13, a14 = compute_steering_samples(*noisy)
            snap = np.random.randint(N_SAMPLES)
            theta_2 = estimate_theta_2rx(a12, fc, d, c, snapshot=snap)
            theta_e, _, _ = esprit_steering(a12, a13, a14, fc, d, c)
            theta_m, _, _ = music_steering(a12, a13, a14, fc, d, c)
            err_2rx[t] = theta_2 - theta_true_deg
            err_esprit[t] = theta_e - theta_true_deg
            err_music[t] = theta_m - theta_true_deg
        rmse_2rx[i] = float(np.sqrt(np.mean(err_2rx**2)))
        rmse_esprit[i] = float(np.sqrt(np.mean(err_esprit**2)))
        rmse_music[i] = float(np.sqrt(np.mean(err_music**2)))

    return snr_db_range, rmse_2rx, rmse_esprit, rmse_music


def visualize_noise_comparison(
    snr_db_range: np.ndarray,
    rmse_2rx: np.ndarray,
    rmse_esprit: np.ndarray,
    rmse_music: np.ndarray,
    theta_true_deg: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(snr_db_range, rmse_2rx, "o-", label="2-RX single snapshot (node 1–2)", linewidth=2)
    ax.plot(snr_db_range, rmse_esprit, "s-", label="ESPRIT (node 1–4)", linewidth=2)
    ax.plot(snr_db_range, rmse_music, "d-", label="MUSIC (node 1–4)", linewidth=2)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("RMSE (deg)")
    ax.set_title(f"AoA estimation under AWGN (true θ = {theta_true_deg:.2f}°)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    plt.show()


SNR_DB_RANGE = np.arange(0, 41, 5)
N_MC_TRIALS = 200

print("--- Noise Monte Carlo ---")
snr_axis, rmse_2rx, rmse_esprit, rmse_music = run_noise_comparison(
    SNR_DB_RANGE, N_MC_TRIALS, theta_5_true_deg
)
for snr, e2, ee, em in zip(snr_axis, rmse_2rx, rmse_esprit, rmse_music):
    print(
        f"SNR {snr:3d} dB | 2-RX {e2:6.3f}° | ESPRIT {ee:6.3f}° | MUSIC {em:6.3f}°"
    )

visualize_noise_comparison(
    SNR_DB_RANGE, rmse_2rx, rmse_esprit, rmse_music, theta_5_true_deg
)
