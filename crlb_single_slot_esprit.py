"""CRLB for single-slot, single-source CoRal ESPRIT (aligned M=3).

Aligned observation model (matches array_simulator.prepare_spatial_covariance):
  x_n = a(theta) + w_n,   a = [1, phi, phi^2]^T,   phi = exp(j*2*pi*fc*d*sin(theta)/c)
  w_n ~ CN(0, sigma^2 I_3)

Three Monte Carlo tiers:
  1. ideal   : direct x_n = a(theta) + AWGN  (validates CRLB & estimators)
  2. genie   : full CoRal ratios, true CFO from node clocks, no magnitude normalize
  3. practical: estimated CFO slopes + per-sample normalize (deployed pipeline)

At high SNR, ideal+ML should converge to CRLB; ESPRIT is close; practical plateaus
due to ratio nonlinearity and CFO slope estimation.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig, pinv

F_CENTER = 2.4e9
C_LIGHT = 3e8
DELTA_T = 150e-6
SAMPLE_INTERVAL_S = 1e-6
CTE_DURATION_S = 92e-6
N_SAMPLES = int(round(CTE_DURATION_S / SAMPLE_INTERVAL_S))
FREQUENCY_OFFSET_RANGE_HZ = 5e4
BASELINE_D = 0.05
M_ALIGNED = 3  # [1, phi, phi^2] after a12 alignment


# =============================================================================
# Steering & CRLB (aligned M=3)
# =============================================================================
def psi_from_sin_theta(sin_theta: float, fc: float, d: float) -> float:
    return 2 * np.pi * fc * d * sin_theta / C_LIGHT


def steering_aligned(theta_deg: float, fc: float, d: float) -> np.ndarray:
    sin_t = float(np.sin(np.deg2rad(theta_deg)))
    phi = np.exp(1j * psi_from_sin_theta(sin_t, fc, d))
    return np.array([1.0, phi, phi**2], dtype=np.complex128)


def steering_derivative_aligned_rad(theta_deg: float, fc: float, d: float) -> np.ndarray:
    theta_rad = np.deg2rad(theta_deg)
    cos_t = np.cos(theta_rad)
    beta = 2 * np.pi * fc * d * cos_t / C_LIGHT
    a = steering_aligned(theta_deg, fc, d)
    k = np.arange(M_ALIGNED, dtype=np.float64)
    return (1j * k * beta * a).astype(np.complex128)


def gamma_projection(a: np.ndarray, a_theta: np.ndarray) -> float:
    a_col = a.reshape(-1, 1)
    a_th = a_theta.reshape(-1, 1)
    denom = float(np.real((a_col.conj().T @ a_col).item()))
    if denom < 1e-20:
        return 0.0
    p_perp = np.eye(len(a)) - (a_col @ a_col.conj().T) / denom
    proj = p_perp @ a_th
    return float(np.real((proj.conj().T @ proj).item()))


def noise_var_from_snr_db(snr_db: float) -> float:
    """Unit-amplitude signal; SNR = 1 / sigma^2."""
    return 10 ** (-snr_db / 10.0)


def crlb_theta_rad(
    theta_deg: float,
    fc: float,
    d: float,
    n_snapshots: int,
    noise_var: float,
) -> float:
    a = steering_aligned(theta_deg, fc, d)
    a_t = steering_derivative_aligned_rad(theta_deg, fc, d)
    gamma = gamma_projection(a, a_t)
    if gamma < 1e-20:
        return float("inf")
    return noise_var / (2.0 * n_snapshots * gamma)


def crlb_theta_deg_std(
    theta_deg: float, fc: float, d: float, n_snapshots: int, snr_db: float
) -> float:
    var_rad = crlb_theta_rad(theta_deg, fc, d, n_snapshots, noise_var_from_snr_db(snr_db))
    return float(np.rad2deg(np.sqrt(var_rad)))


def crlb_2rx_deg_std(theta_deg: float, fc: float, d: float, n_snapshots: int, snr_db: float) -> float:
    theta_rad = np.deg2rad(theta_deg)
    dpsi = 2 * np.pi * fc * d * np.cos(theta_rad) / C_LIGHT
    nv = noise_var_from_snr_db(snr_db)
    return float(np.rad2deg(np.sqrt(nv / (2.0 * n_snapshots * dpsi**2))))


# =============================================================================
# Estimators
# =============================================================================
def build_aligned_matrix(a12: np.ndarray, a13: np.ndarray, a14: np.ndarray) -> np.ndarray:
    """3 x N aligned snapshots [1, a13/a12, a14/a12] (array_simulator convention)."""
    x = np.vstack([np.asarray(a12), np.asarray(a13), np.asarray(a14)])
    ref = x[0:1, :]
    safe = np.where(np.abs(ref) < 1e-12, 1.0 + 0j, ref)
    return x / safe


def esprit_aligned_deg(x_aligned: np.ndarray, fc: float, d: float) -> float:
    """ESPRIT on 3xN aligned data (matches array_simulator.esprit_steering)."""
    r_xx = x_aligned @ x_aligned.conj().T / x_aligned.shape[1]
    eigvals, eigvecs = np.linalg.eigh(r_xx)
    es = eigvecs[:, int(np.argmax(eigvals))].reshape(-1, 1)
    es1, es2 = es[0:2, :], es[1:3, :]
    phi = complex((np.linalg.pinv(es1) @ es2)[0, 0])
    sin_theta = float(np.clip(np.angle(phi) * C_LIGHT / (2 * np.pi * fc * d), -1.0, 1.0))
    return float(np.degrees(np.arcsin(sin_theta)))


def ml_theta_deg(x_aligned: np.ndarray, fc: float, d: float, theta_grid: np.ndarray | None = None) -> float:
    """Coherent ML: coarse grid + fine local refinement (achieves CRLB at high SNR)."""
    x_bar = np.mean(x_aligned, axis=1)

    def score(th: float) -> float:
        a = steering_aligned(th, fc, d)
        return float(np.abs(np.vdot(a, x_bar)) ** 2)

    if theta_grid is None:
        theta_grid = np.linspace(-89.0, 89.0, 721)
    scores = np.array([score(float(th)) for th in theta_grid])
    coarse = float(theta_grid[int(np.argmax(scores))])
    fine = np.linspace(coarse - 0.5, coarse + 0.5, 2001)
    return float(fine[int(np.argmax([score(t) for t in fine]))])


def estimate_2rx_deg(a12: np.ndarray, fc: float, d: float) -> float:
    val = complex(np.mean(a12))
    sin_theta = float(np.clip(np.angle(val) * C_LIGHT / (2 * np.pi * fc * d), -1.0, 1.0))
    return float(np.degrees(np.arcsin(sin_theta)))


# =============================================================================
# Ideal snapshot generator (CRLB validation)
# =============================================================================
def simulate_ideal_aligned(
    theta_deg: float,
    fc: float,
    d: float,
    n_snapshots: int,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    a = steering_aligned(theta_deg, fc, d).reshape(-1, 1)
    sigma = np.sqrt(noise_var_from_snr_db(snr_db) / 2.0)
    noise = sigma * (rng.standard_normal((M_ALIGNED, n_snapshots))
                     + 1j * rng.standard_normal((M_ALIGNED, n_snapshots)))
    return a + noise


# =============================================================================
# CoRal physics chain
# =============================================================================
def wrap_phase(p: float) -> float:
    return float(np.arctan2(np.sin(p), np.cos(p)))


class Node:
    def __init__(self, node_id: int, x: float, y: float, phase: float, freq_off: float) -> None:
        self.node_id = node_id
        self.x, self.y = x, y
        self.initial_phase = wrap_phase(phase)
        self.frequency_offset = freq_off

    def distance_to(self, other: Node) -> float:
        return float(np.hypot(self.x - other.x, self.y - other.y))


def make_nodes(
    rng: np.random.Generator,
    *,
    zero_cfo: bool = False,
) -> dict[int, Node]:
    def fo() -> float:
        return 0.0 if zero_cfo else float(rng.uniform(-FREQUENCY_OFFSET_RANGE_HZ, FREQUENCY_OFFSET_RANGE_HZ))

    return {
        0: Node(0, 2.0, 2.0, wrap_phase(rng.uniform(0, 2 * np.pi)), fo()),
        1: Node(1, 0.0, 0.15, wrap_phase(rng.uniform(0, 2 * np.pi)), fo()),
        2: Node(2, 0.0, 0.10, wrap_phase(rng.uniform(0, 2 * np.pi)), fo()),
        3: Node(3, 0.0, 0.05, wrap_phase(rng.uniform(0, 2 * np.pi)), fo()),
        4: Node(4, 0.0, 0.00, wrap_phase(rng.uniform(0, 2 * np.pi)), fo()),
        5: Node(5, 5.0, 1.50, wrap_phase(rng.uniform(0, 2 * np.pi)), fo()),
    }


def build_fading(nodes: dict[int, Node]) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for ri in nodes:
        for ti in nodes:
            dist = nodes[ri].distance_to(nodes[ti])
            out[(ri, ti)] = 0.0 if ri == ti or dist < 1e-12 else 1.0 / dist
    return out


def signal_at_receiver(rx: Node, tx: Node, t0: float, fading: dict[tuple[int, int], float]) -> np.ndarray:
    h = fading[(rx.node_id, tx.node_id)]
    tau = rx.distance_to(tx) / C_LIGHT
    t = t0 + np.arange(N_SAMPLES, dtype=np.float64) * SAMPLE_INTERVAL_S
    ph = (
        2 * np.pi * (tx.frequency_offset - rx.frequency_offset) * t
        + tx.initial_phase - rx.initial_phase
        - 2 * np.pi * (F_CENTER + tx.frequency_offset) * tau
    )
    return h * np.exp(1j * ph)


def add_awgn(sig: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    p = float(np.mean(np.abs(sig) ** 2))
    if p < 1e-20:
        return sig.copy()
    npow = p / (10 ** (snr_db / 10))
    n = np.sqrt(npow / 2) * (rng.standard_normal(len(sig)) + 1j * rng.standard_normal(len(sig)))
    return sig + n


def slope(sig: np.ndarray) -> float:
    ph = np.unwrap(np.angle(sig))
    t = np.arange(len(sig), dtype=np.float64) * SAMPLE_INTERVAL_S
    return float(np.polyfit(t, ph, 1)[0])


def normalize(sig: np.ndarray) -> np.ndarray:
    m = np.abs(sig)
    return sig / np.where(m < 1e-12, 1.0, m)


def delta_tau(rx_a: Node, rx_b: Node, tx: Node) -> float:
    return rx_a.distance_to(tx) / C_LIGHT - rx_b.distance_to(tx) / C_LIGHT


def true_theta_deg(rx_a: Node, rx_b: Node, tx: Node, d_sp: float) -> float:
    dt = delta_tau(rx_a, rx_b, tx)
    return float(-np.degrees(np.arcsin(np.clip(dt * C_LIGHT / d_sp, -1.0, 1.0))))


def true_beat_slope_hz(rx_a: Node, rx_b: Node) -> float:
    """Phase slope (rad/s) of R_a/R_b ratio vs time for common TX."""
    return -2 * np.pi * (rx_a.frequency_offset - rx_b.frequency_offset)


def m_factors(nodes: dict[int, Node]) -> tuple[complex, complex, complex]:
    rx1, rx2, rx3, rx4, ref = nodes[1], nodes[2], nodes[3], nodes[4], nodes[0]
    return (
        complex(np.exp(-1j * 2 * np.pi * F_CENTER * delta_tau(rx1, rx2, ref))),
        complex(np.exp(-1j * 2 * np.pi * F_CENTER * delta_tau(rx1, rx3, ref))),
        complex(np.exp(-1j * 2 * np.pi * F_CENTER * delta_tau(rx1, rx4, ref))),
    )


def simulate_coral_slot(
    nodes: dict[int, Node],
    fading: dict[tuple[int, int], float],
    snr_db: float,
    rng: np.random.Generator,
    *,
    mode: str = "practical",
    cfo_slope_error_hz: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """mode: 'genie' | 'practical'."""
    rx = [nodes[i] for i in (1, 2, 3, 4)]
    ref, tgt = nodes[0], nodes[5]
    m1, m2, m3 = m_factors(nodes)

    def rx_sigs(tx: Node, t0: float) -> tuple[np.ndarray, ...]:
        return tuple(
            add_awgn(signal_at_receiver(r, tx, t0, fading), snr_db, rng) for r in rx
        )

    r_ref = rx_sigs(ref, 0.0)
    r_tgt = rx_sigs(tgt, DELTA_T)
    d0_12, d0_13, d0_14 = r_ref[0] / r_ref[1], r_ref[0] / r_ref[2], r_ref[0] / r_ref[3]
    d1_12 = r_tgt[0] / r_tgt[1]
    d1_13 = r_tgt[0] / r_tgt[2]
    d1_14 = r_tgt[0] / r_tgt[3]

    if mode == "genie":
        # True beat slopes from node clocks (no polyfit error).
        f12 = true_beat_slope_hz(nodes[1], nodes[2])
        f13 = true_beat_slope_hz(nodes[1], nodes[3])
        f14 = true_beat_slope_hz(nodes[1], nodes[4])
    else:
        f12 = slope(d0_12) + 2 * np.pi * cfo_slope_error_hz
        f13 = slope(d0_13) + 2 * np.pi * cfo_slope_error_hz
        f14 = slope(d0_14) + 2 * np.pi * cfo_slope_error_hz

    raw12 = (d1_12 / d0_12) * m1 * np.exp(-1j * f12 * DELTA_T)
    raw13 = (d1_13 / d0_13) * m2 * np.exp(-1j * f13 * DELTA_T)
    raw14 = (d1_14 / d0_14) * m3 * np.exp(-1j * f14 * DELTA_T)

    if mode == "practical":
        return normalize(raw12), normalize(raw13), normalize(raw14)
    return raw12, raw13, raw14


# =============================================================================
# Monte Carlo drivers
# =============================================================================
def run_mc(
    theta_true: float,
    snr_db: float,
    n_trials: int,
    rng: np.random.Generator,
    *,
    tier: str,
    nodes: dict[int, Node] | None = None,
    fading: dict[tuple[int, int], float] | None = None,
) -> tuple[float, float, float]:
    """Return (rmse_ml, rmse_esprit, rmse_2rx)."""
    err_ml = np.zeros(n_trials)
    err_esp = np.zeros(n_trials)
    err_2 = np.zeros(n_trials)
    for t in range(n_trials):
        tr = np.random.default_rng(rng.integers(0, 2**31))
        if tier == "ideal":
            x = simulate_ideal_aligned(theta_true, F_CENTER, BASELINE_D, N_SAMPLES, snr_db, tr)
        else:
            assert nodes is not None and fading is not None
            trial_nodes = make_nodes(tr, zero_cfo=(tier == "genie"))
            # Keep geometry; only randomize clocks/phases/noise per trial.
            for k, nid in enumerate(nodes):
                trial_nodes[nid].x = nodes[nid].x
                trial_nodes[nid].y = nodes[nid].y
            a12, a13, a14 = simulate_coral_slot(
                trial_nodes, fading, snr_db, tr, mode=tier,
            )
            x = build_aligned_matrix(a12, a13, a14)
            err_2[t] = estimate_2rx_deg(a12, F_CENTER, BASELINE_D) - theta_true

        err_ml[t] = ml_theta_deg(x, F_CENTER, BASELINE_D) - theta_true
        err_esp[t] = esprit_aligned_deg(x, F_CENTER, BASELINE_D) - theta_true
        if tier == "ideal":
            err_2[t] = err_ml[t]  # placeholder; 2-RX not separate in ideal tier

    return (
        float(np.sqrt(np.mean(err_ml**2))),
        float(np.sqrt(np.mean(err_esp**2))),
        float(np.sqrt(np.mean(err_2**2))),
    )


# =============================================================================
# Plots
# =============================================================================
def plot_snr_comparison(
    theta_true: float,
    snr_range: np.ndarray,
    crlb: np.ndarray,
    rmse_ideal_ml: np.ndarray,
    rmse_ideal_esp: np.ndarray,
    rmse_genie_esp: np.ndarray,
    rmse_prac_esp: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.semilogy(snr_range, crlb, "k--", linewidth=2.5, label="CRLB (aligned M=3)")
    ax.semilogy(snr_range, rmse_ideal_ml, "o-", linewidth=2, label="Ideal AWGN + ML")
    ax.semilogy(snr_range, rmse_ideal_esp, "s-", linewidth=2, label="Ideal AWGN + ESPRIT")
    ax.semilogy(snr_range, rmse_genie_esp, "^-", linewidth=2, label="CoRal genie CFO + ESPRIT")
    ax.semilogy(snr_range, rmse_prac_esp, "d-", linewidth=2, label="CoRal practical + ESPRIT")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("RMSE (deg)")
    ax.set_title(f"AoA estimators vs CRLB  (theta_true={theta_true:.1f} deg, N={N_SAMPLES})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig("crlb_vs_snr.png", dpi=150)
    print("Saved: crlb_vs_snr.png")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    rng = np.random.default_rng(42)
    nodes = make_nodes(rng, zero_cfo=False)
    fading = build_fading(nodes)
    theta_true = true_theta_deg(nodes[1], nodes[2], nodes[5], BASELINE_D)

    print("=== Single-slot CRLB (aligned M=3) ===")
    print(f"True theta = {theta_true:.4f} deg")

    # Quick CRLB self-test: ideal ML should match within ~1 dB at 30 dB.
    snr_range = np.arange(0, 41, 5)
    n_trials = 400
    crlb = np.array([crlb_theta_deg_std(theta_true, F_CENTER, BASELINE_D, N_SAMPLES, s) for s in snr_range])
    rmse_iml = np.zeros(len(snr_range))
    rmse_iesp = np.zeros(len(snr_range))
    rmse_gesp = np.zeros(len(snr_range))
    rmse_pesp = np.zeros(len(snr_range))

    print(f"\n{'SNR':>4} | {'CRLB':>7} | {'IdealML':>7} | {'IdESP':>7} | {'GnESP':>7} | {'PrESP':>7}")
    for i, snr in enumerate(snr_range):
        rmse_iml[i], rmse_iesp[i], _ = run_mc(theta_true, float(snr), n_trials, rng, tier="ideal")
        _, rmse_gesp[i], _ = run_mc(theta_true, float(snr), n_trials, rng, tier="genie", nodes=nodes, fading=fading)
        _, rmse_pesp[i], _ = run_mc(theta_true, float(snr), n_trials, rng, tier="practical", nodes=nodes, fading=fading)
        print(
            f"{snr:4.0f} | {crlb[i]:7.4f} | {rmse_iml[i]:7.4f} | {rmse_iesp[i]:7.4f} | "
            f"{rmse_gesp[i]:7.4f} | {rmse_pesp[i]:7.4f}"
        )

    plot_snr_comparison(theta_true, snr_range, crlb, rmse_iml, rmse_iesp, rmse_gesp, rmse_pesp)

    print("\n--- Why curves diverge ---")
    print("Ideal+ML  -> should track CRLB (validates bound & ML).")
    print("Ideal+ESPRIT -> near CRLB but slightly above (suboptimal vs ML).")
    print("Genie CoRal -> ratio-chain floor: nonlinear noise, no normalize in genie.")
    print("Practical -> adds slope-estimation error + magnitude normalize.")


if __name__ == "__main__":
    main()
