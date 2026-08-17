"""Compare geometry MUSIC (A) vs multi-baseline TDOA-LS (B) for CoRal AoA.

Works with arbitrary known RX positions (unequal spacing, baselines may exceed λ/2).
CoRal front-end (ref TX + target TX ratios + CFO compensation) is unchanged;
only the AoA mapper differs from classic ULA-ESPRIT.

Layouts:
  equal   : ULA along y, d = 0.05 m (< λ/2) — also runs classic ESPRIT
  unequal : irregular spacings including one baseline > λ/2
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
F_CENTER = 2.4e9
C_LIGHT = 3e8
WAVELENGTH = C_LIGHT / F_CENTER  # 0.125 m
DELTA_T = 150e-6
SAMPLE_INTERVAL_S = 1e-6
CTE_DURATION_S = 92e-6
N_SAMPLES = int(round(CTE_DURATION_S / SAMPLE_INTERVAL_S))
FREQUENCY_OFFSET_RANGE_HZ = 5e4

# CoRal calibrated a_1k ≈ exp(-j 2π fc Δτ_1k), Δτ_1k = τ_1 - τ_k


# =============================================================================
# Nodes & CoRal physics
# =============================================================================
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

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)


def make_node(rng: np.random.Generator, node_id: int, x: float, y: float) -> Node:
    return Node(
        node_id,
        x,
        y,
        wrap_phase(float(rng.uniform(0, 2 * np.pi))),
        float(rng.uniform(-FREQUENCY_OFFSET_RANGE_HZ, FREQUENCY_OFFSET_RANGE_HZ)),
    )


def build_scene(
    rng: np.random.Generator,
    rx_xy: list[tuple[float, float]],
    ref_xy: tuple[float, float] = (2.0, 2.0),
    tgt_xy: tuple[float, float] = (5.0, 1.50),
) -> dict[int, Node]:
    """RX ids 1..M, ref=0, target=5 (id kept for readability)."""
    nodes: dict[int, Node] = {
        0: make_node(rng, 0, ref_xy[0], ref_xy[1]),
        5: make_node(rng, 5, tgt_xy[0], tgt_xy[1]),
    }
    for i, (x, y) in enumerate(rx_xy, start=1):
        nodes[i] = make_node(rng, i, x, y)
    return nodes


def build_fading(nodes: dict[int, Node]) -> dict[tuple[int, int], float]:
    fading: dict[tuple[int, int], float] = {}
    for ri in nodes:
        for ti in nodes:
            if ri == ti:
                fading[(ri, ti)] = 0.0
                continue
            dist = nodes[ri].distance_to(nodes[ti])
            fading[(ri, ti)] = 0.0 if dist < 1e-12 else 1.0 / dist
    return fading


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


def add_awgn(sig: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    p = float(np.mean(np.abs(sig) ** 2))
    if p < 1e-20:
        return sig.copy()
    npow = p / (10 ** (snr_db / 10))
    n = np.sqrt(npow / 2) * (
        rng.standard_normal(len(sig)) + 1j * rng.standard_normal(len(sig))
    )
    return sig + n


def slope(sig: np.ndarray) -> float:
    ph = np.unwrap(np.angle(sig))
    t = np.arange(len(sig), dtype=np.float64) * SAMPLE_INTERVAL_S
    return float(np.polyfit(t, ph, 1)[0])


def normalize(sig: np.ndarray) -> np.ndarray:
    mag = np.abs(sig)
    return sig / np.where(mag < 1e-12, 1.0, mag)


def delta_tau_true(rx_a: Node, rx_b: Node, tx: Node) -> float:
    return rx_a.distance_to(tx) / C_LIGHT - rx_b.distance_to(tx) / C_LIGHT


def true_bearing_deg(rx_nodes: list[Node], tx: Node) -> float:
    mid = np.mean([n.pos for n in rx_nodes], axis=0)
    return float(np.degrees(np.arctan2(tx.y - mid[1], tx.x - mid[0])))


# =============================================================================
# CoRal calibration → per-baseline complex snapshots a_1k
# =============================================================================
def rx_list(nodes: dict[int, Node]) -> list[Node]:
    ids = sorted(i for i in nodes if i not in (0, 5))
    return [nodes[i] for i in ids]


def simulate_calibrated_baselines(
    nodes: dict[int, Node],
    fading: dict[tuple[int, int], float],
    snr_db: float,
    rng: np.random.Generator,
) -> dict[int, np.ndarray]:
    """Return {k: a_1k[n]} for each RX k != 1, CoRal-calibrated vs ref TX 0."""
    rxs = rx_list(nodes)
    ref, tgt = nodes[0], nodes[5]
    rx1 = rxs[0]

    def collect(tx: Node, t0: float) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        for rx in rxs:
            out[rx.node_id] = add_awgn(
                signal_at_receiver(rx, tx, t0, fading), snr_db, rng
            )
        return out

    r_ref = collect(ref, 0.0)
    r_tgt = collect(tgt, DELTA_T)

    a_by_k: dict[int, np.ndarray] = {}
    for rx in rxs[1:]:
        k = rx.node_id
        d0 = r_ref[rx1.node_id] / r_ref[k]
        d1 = r_tgt[rx1.node_id] / r_tgt[k]
        freq = slope(d0)
        m = np.exp(-1j * 2 * np.pi * F_CENTER * delta_tau_true(rx1, rx, ref))
        a_by_k[k] = normalize(d1 / d0) * m * np.exp(-1j * freq * DELTA_T)
    return a_by_k


def snapshots_matrix(a_by_k: dict[int, np.ndarray]) -> tuple[np.ndarray, list[int]]:
    """Stack calibrated baselines as M_b x N complex matrix (rows ordered by k)."""
    keys = sorted(a_by_k.keys())
    x = np.vstack([a_by_k[k] for k in keys])
    return x, keys


def estimate_delta_tau_from_a(a: np.ndarray) -> float:
    """Δτ̂ = -∠(mean a) / (2π fc); a ≈ exp(-j 2π fc Δτ)."""
    return float(-np.angle(np.mean(a)) / (2 * np.pi * F_CENTER))


# =============================================================================
# Far-field geometry model
# =============================================================================
def unit_vector(theta_deg: float) -> np.ndarray:
    """Bearing θ from +x axis (deg) → unit vector toward source."""
    th = np.deg2rad(theta_deg)
    return np.array([np.cos(th), np.sin(th)], dtype=np.float64)


def model_delta_tau(p_a: np.ndarray, p_b: np.ndarray, theta_deg: float) -> float:
    """Far-field Δτ = τ_a - τ_b = (p_b - p_a) · u / c.

    Path length ≈ |r| - p·û with û toward the source, so
    τ_a - τ_b = -(p_a - p_b)·û / c = (p_b - p_a)·û / c.
    """
    return float(np.dot(p_b - p_a, unit_vector(theta_deg)) / C_LIGHT)


def geometry_steering(
    p_ref: np.ndarray,
    p_others: list[np.ndarray],
    theta_deg: float,
) -> np.ndarray:
    """a_k(θ) = exp(-j 2π fc Δτ_1k), matching CoRal calibrated phase."""
    a = []
    for p_k in p_others:
        dtau = model_delta_tau(p_ref, p_k, theta_deg)
        a.append(np.exp(-1j * 2 * np.pi * F_CENTER * dtau))
    return np.asarray(a, dtype=np.complex128)


# =============================================================================
# Scheme A: Geometry MUSIC
# =============================================================================
def geometry_music(
    x: np.ndarray,
    p_ref: np.ndarray,
    p_others: list[np.ndarray],
    *,
    scan_deg: np.ndarray | None = None,
    num_sources: int = 1,
) -> tuple[float, np.ndarray, np.ndarray]:
    """MUSIC on CoRal baseline vector with geometry-aware steering."""
    if scan_deg is None:
        scan_deg = np.linspace(-90.0, 90.0, 721)

    m, n = x.shape
    r_xx = x @ x.conj().T / n
    eigvals, eigvecs = np.linalg.eigh(r_xx)
    idx = np.argsort(eigvals)
    en = eigvecs[:, idx[: m - num_sources]]
    en_proj = en @ en.conj().T

    spectrum = np.zeros(len(scan_deg))
    for i, th in enumerate(scan_deg):
        a = geometry_steering(p_ref, p_others, float(th)).reshape(-1, 1)
        denom = float(np.real((a.conj().T @ en_proj @ a).item()))
        spectrum[i] = 1.0 / max(denom, 1e-20)

    theta_hat = float(scan_deg[int(np.argmax(spectrum))])
    return theta_hat, scan_deg, spectrum


# =============================================================================
# Scheme B: Multi-baseline TDOA least squares (complex domain; wrap-safe)
# =============================================================================
def tdoa_ls(
    a_meas: list[complex],
    p_ref: np.ndarray,
    p_others: list[np.ndarray],
    *,
    scan_deg: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """θ̂ = argmin_θ Σ_k |â_k - exp(-j 2π fc Δτ_k(θ))|^2.

    Matching in complex domain avoids explicit Δτ unwrapping when baselines > λ/2.
    """
    if scan_deg is None:
        scan_deg = np.linspace(-90.0, 90.0, 3601)

    a_meas_arr = np.asarray(a_meas, dtype=np.complex128)
    cost = np.zeros(len(scan_deg))
    for i, th in enumerate(scan_deg):
        a_model = geometry_steering(p_ref, p_others, float(th))
        cost[i] = float(np.sum(np.abs(a_meas_arr - a_model) ** 2))

    theta_hat = float(scan_deg[int(np.argmin(cost))])
    return theta_hat, scan_deg, cost


# =============================================================================
# Classic ULA ESPRIT (equal spacing only, for reference)
# =============================================================================
def classic_esprit_ula(
    a_by_k: dict[int, np.ndarray],
    d_sp: float,
) -> float:
    keys = sorted(a_by_k.keys())
    rows = [a_by_k[k] for k in keys]
    x = np.vstack(rows)
    ref = x[0:1, :]
    x_al = x / np.where(np.abs(ref) < 1e-12, 1.0 + 0j, ref)
    r_xx = x_al @ x_al.conj().T / x_al.shape[1]
    eigvals, eigvecs = np.linalg.eigh(r_xx)
    es = eigvecs[:, int(np.argmax(eigvals))].reshape(-1, 1)
    if es.shape[0] < 2:
        return float("nan")
    es1, es2 = es[:-1, :], es[1:, :]
    phi = complex((np.linalg.pinv(es1) @ es2)[0, 0])
    # Match array_simulator: φ = exp(+j 2π fc d sinθ / c)
    sin_th = float(np.clip(np.angle(phi) * C_LIGHT / (2 * np.pi * F_CENTER * d_sp), -1.0, 1.0))
    return float(np.degrees(np.arcsin(sin_th)))


# =============================================================================
# Layouts
# =============================================================================
def layout_equal() -> list[tuple[float, float]]:
    """ULA along y, spacing 0.05 m < λ/2."""
    return [(0.0, 0.15), (0.0, 0.10), (0.0, 0.05), (0.0, 0.00)]


def layout_unequal() -> list[tuple[float, float]]:
    """Irregular y-spacings; longest baseline > λ/2 (0.0625 m)."""
    # positions: 0, 0.03, 0.09, 0.22 → spans 0.22 m ≈ 1.76 λ
    return [(0.0, 0.22), (0.0, 0.19), (0.0, 0.13), (0.0, 0.00)]


def layout_2d_irregular() -> list[tuple[float, float]]:
    """Non-collinear RX (2-D array)."""
    return [(0.00, 0.00), (0.08, 0.02), (0.03, 0.12), (-0.05, 0.07)]


def describe_baselines(rxs: list[Node]) -> None:
    p0 = rxs[0].pos
    print("  Baselines vs RX1:")
    for rx in rxs[1:]:
        d = float(np.linalg.norm(rx.pos - p0))
        print(f"    1–{rx.node_id}: {d*100:.1f} cm  ({d/WAVELENGTH:.2f} λ)  "
              f"{'> λ/2' if d > WAVELENGTH / 2 else '≤ λ/2'}")


# =============================================================================
# Single trial / Monte Carlo
# =============================================================================
def run_estimators(
    nodes: dict[int, Node],
    fading: dict[tuple[int, int], float],
    snr_db: float,
    rng: np.random.Generator,
    *,
    run_esprit: bool = False,
    d_ula: float | None = None,
) -> dict:
    a_by_k = simulate_calibrated_baselines(nodes, fading, snr_db, rng)
    rxs = rx_list(nodes)
    p_ref = rxs[0].pos
    p_others = [nodes[k].pos for k in sorted(a_by_k.keys())]
    x, keys = snapshots_matrix(a_by_k)

    th_music, scan_m, spec = geometry_music(x, p_ref, p_others)
    a_means = [complex(np.mean(a_by_k[k])) for k in keys]
    th_ls, scan_c, cost = tdoa_ls(a_means, p_ref, p_others)
    dtaus = [estimate_delta_tau_from_a(a_by_k[k]) for k in keys]

    true_bearing = true_bearing_deg(rxs, nodes[5])
    out = {
        "true_bearing_deg": true_bearing,
        "music_deg": th_music,
        "tdoa_ls_deg": th_ls,
        "scan_music": scan_m,
        "spectrum": spec,
        "scan_cost": scan_c,
        "cost": cost,
        "delta_tau_hat": dtaus,
        "delta_tau_true": [delta_tau_true(rxs[0], nodes[k], nodes[5]) for k in keys],
        "a_by_k": a_by_k,
        "keys": keys,
    }
    if run_esprit and d_ula is not None:
        out["esprit_deg"] = classic_esprit_ula(a_by_k, d_ula)
    return out


def monte_carlo_rmse(
    rx_xy: list[tuple[float, float]],
    snr_db_range: np.ndarray,
    n_trials: int,
    seed: int,
    *,
    run_esprit: bool = False,
    d_ula: float | None = None,
) -> dict[str, np.ndarray]:
    rng_master = np.random.default_rng(seed)
    rmse_m = np.zeros(len(snr_db_range))
    rmse_l = np.zeros(len(snr_db_range))
    rmse_e = np.zeros(len(snr_db_range)) if run_esprit else None

    # Fixed geometry across trials; clocks/noise vary.
    nodes0 = build_scene(rng_master, rx_xy)
    fading = build_fading(nodes0)

    for i, snr in enumerate(snr_db_range):
        err_m, err_l, err_e = [], [], []
        for _ in range(n_trials):
            rng = np.random.default_rng(rng_master.integers(0, 2**31))
            # refresh phases/CFO each trial, keep positions
            nodes = build_scene(rng, rx_xy)
            for nid in nodes0:
                nodes[nid].x = nodes0[nid].x
                nodes[nid].y = nodes0[nid].y
            fading = build_fading(nodes)
            res = run_estimators(
                nodes, fading, float(snr), rng,
                run_esprit=run_esprit, d_ula=d_ula,
            )
            true = res["true_bearing_deg"]
            err_m.append(res["music_deg"] - true)
            err_l.append(res["tdoa_ls_deg"] - true)
            if run_esprit:
                # ESPRIT returns sinθ-convention angle, not bearing — map carefully.
                # For ULA along y, compare against geometry MUSIC/LS on same trial
                # by converting ESPRIT θ to bearing via far-field ULA model is messy.
                # Instead compare ESPRIT to MUSIC when both use same θ definition:
                # we evaluate ESPRIT error vs true bearing after converting.
                err_e.append(esprit_to_bearing_error(res, nodes))
        rmse_m[i] = float(np.sqrt(np.mean(np.asarray(err_m) ** 2)))
        rmse_l[i] = float(np.sqrt(np.mean(np.asarray(err_l) ** 2)))
        if run_esprit and rmse_e is not None:
            rmse_e[i] = float(np.sqrt(np.mean(np.asarray(err_e) ** 2)))

    out = {"snr": snr_db_range, "rmse_music": rmse_m, "rmse_tdoa_ls": rmse_l}
    if rmse_e is not None:
        out["rmse_esprit"] = rmse_e
    return out


def esprit_to_bearing_error(res: dict, nodes: dict[int, Node]) -> float:
    """Convert ULA-ESPRIT sinθ angle to bearing error vs true bearing.

    For ULA along +y with reference at top, ESPRIT θ satisfies
    Δτ_12 ≈ -d sin(θ_esp)/c in their older sign, while bearing β from +x
    gives Δτ_12 = (p1-p2)·u(β)/c.  We map θ_esp → predicted Δτ → closest bearing.
    """
    if "esprit_deg" not in res:
        return float("nan")
    rxs = rx_list(nodes)
    d = float(np.linalg.norm(rxs[0].pos - rxs[1].pos))
    # ESPRIT: angle(φ) = 2π fc d sinθ / c  and a ≈ exp(+j ...) in classic code,
    # while CoRal a ≈ exp(-j 2π fc Δτ) ⇒ Δτ ≈ -d sin(θ_esp)/c
    dtau = -d * np.sin(np.deg2rad(res["esprit_deg"])) / C_LIGHT
    scan = np.linspace(-90.0, 90.0, 3601)
    cost = [
        (dtau - model_delta_tau(rxs[0].pos, rxs[1].pos, float(th))) ** 2
        for th in scan
    ]
    bearing_from_esprit = float(scan[int(np.argmin(cost))])
    return bearing_from_esprit - res["true_bearing_deg"]


# =============================================================================
# Visualization
# =============================================================================
def plot_comparison(
    layout_name: str,
    nodes: dict[int, Node],
    demo: dict,
    mc: dict,
    *,
    show_esprit: bool = False,
) -> None:
    rxs = rx_list(nodes)
    ref, tgt = nodes[0], nodes[5]
    true_b = demo["true_bearing_deg"]

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

    # --- Geometry ---
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot([n.x for n in rxs], [n.y for n in rxs], "o-", color="C0", markersize=9, label="RX")
    for n in rxs:
        ax0.annotate(str(n.node_id), (n.x, n.y), textcoords="offset points", xytext=(5, 5))
    ax0.plot(ref.x, ref.y, "s", color="C2", markersize=10, label="Ref 0")
    ax0.plot(tgt.x, tgt.y, "^", color="C3", markersize=11, label="TX 5")
    ax0.annotate("0", (ref.x, ref.y), textcoords="offset points", xytext=(5, 5))
    ax0.annotate("5", (tgt.x, tgt.y), textcoords="offset points", xytext=(5, 5))

    mid = np.mean([n.pos for n in rxs], axis=0)
    ray = 2.5
    for th, style, color, lab in (
        (true_b, "--", "C2", f"True {true_b:.1f}°"),
        (demo["music_deg"], "-", "C0", f"MUSIC {demo['music_deg']:.1f}°"),
        (demo["tdoa_ls_deg"], "-.", "C1", f"TDOA-LS {demo['tdoa_ls_deg']:.1f}°"),
    ):
        u = unit_vector(th)
        ax0.plot(
            [mid[0], mid[0] + ray * u[0]],
            [mid[1], mid[1] + ray * u[1]],
            style, color=color, linewidth=2, label=lab,
        )
    ax0.set_aspect("equal", adjustable="datalim")
    ax0.set_xlabel("x (m)")
    ax0.set_ylabel("y (m)")
    ax0.set_title(f"Layout: {layout_name}")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best", fontsize=7)

    # --- MUSIC spectrum ---
    ax1 = fig.add_subplot(gs[0, 1])
    spec_db = 10 * np.log10(demo["spectrum"] / np.max(demo["spectrum"]))
    ax1.plot(demo["scan_music"], spec_db, color="C0", linewidth=1.5, label="MUSIC")
    ax1.axvline(true_b, color="C2", linestyle="--", label=f"True {true_b:.1f}°")
    ax1.axvline(demo["music_deg"], color="C0", linestyle="-", alpha=0.7, label="MUSIC peak")
    ax1.axvline(demo["tdoa_ls_deg"], color="C1", linestyle="-.", label="TDOA-LS")
    ax1.set_xlabel("Bearing θ (deg)")
    ax1.set_ylabel("Spectrum (dB)")
    ax1.set_title("Scheme A: geometry MUSIC")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=7)

    # --- TDOA cost ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.semilogy(demo["scan_cost"], demo["cost"] + 1e-30, color="C1", linewidth=1.5)
    ax2.axvline(true_b, color="C2", linestyle="--", label=f"True {true_b:.1f}°")
    ax2.axvline(demo["tdoa_ls_deg"], color="C1", linestyle="-.", label="TDOA-LS min")
    ax2.axvline(demo["music_deg"], color="C0", linestyle="-", alpha=0.7, label="MUSIC")
    ax2.set_xlabel("Bearing θ (deg)")
    ax2.set_ylabel("Σ |a − a(θ)|²")
    ax2.set_title("Scheme B: multi-baseline LS cost (complex)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="best", fontsize=7)

    # --- RMSE vs SNR ---
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.semilogy(mc["snr"], mc["rmse_music"], "o-", label="A: geometry MUSIC", linewidth=2)
    ax3.semilogy(mc["snr"], mc["rmse_tdoa_ls"], "s-", label="B: TDOA-LS", linewidth=2)
    if show_esprit and "rmse_esprit" in mc:
        ax3.semilogy(mc["snr"], mc["rmse_esprit"], "d-", label="ULA-ESPRIT (ref)", linewidth=2)
    ax3.set_xlabel("SNR (dB)")
    ax3.set_ylabel("Bearing RMSE (deg)")
    ax3.set_title("Monte Carlo RMSE vs SNR")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend(loc="best", fontsize=8)

    err_m = demo["music_deg"] - true_b
    err_l = demo["tdoa_ls_deg"] - true_b
    fig.suptitle(
        f"CoRal arbitrary-geometry AoA | {layout_name} | "
        f"MUSIC err={err_m:+.2f}°  TDOA-LS err={err_l:+.2f}°  (demo @ 20 dB)",
        fontsize=12,
    )
    fname = f"geometry_aoa_{layout_name}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.close(fig)


def plot_delta_tau_table(layout_name: str, demo: dict) -> None:
    """Bar chart: true vs estimated Δτ per baseline."""
    keys = demo["keys"]
    true_v = np.asarray(demo["delta_tau_true"]) * 1e9
    hat_v = np.asarray(demo["delta_tau_hat"]) * 1e9
    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, true_v, width, label="True Δτ", color="C2")
    ax.bar(x + width / 2, hat_v, width, label="Estimated Δτ", color="C1")
    ax.set_xticks(x)
    ax.set_xticklabels([f"1–{k}" for k in keys])
    ax.set_ylabel("Δτ (ns)")
    ax.set_title(f"{layout_name}: CoRal Δτ estimates")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fname = f"geometry_aoa_{layout_name}_dtau.png"
    fig.savefig(fname, dpi=150)
    print(f"Saved: {fname}")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def run_layout(
    name: str,
    rx_xy: list[tuple[float, float]],
    *,
    run_esprit: bool = False,
    d_ula: float | None = None,
    snr_demo: float = 20.0,
    n_trials: int = 150,
) -> None:
    print(f"\n{'='*60}\nLayout: {name}\n{'='*60}")
    rng = np.random.default_rng(hash(name) % (2**31))
    nodes = build_scene(rng, rx_xy)
    fading = build_fading(nodes)
    rxs = rx_list(nodes)
    describe_baselines(rxs)
    true_b = true_bearing_deg(rxs, nodes[5])
    print(f"  True bearing: {true_b:.3f} deg")
    print(f"  λ/2 = {WAVELENGTH/2*100:.2f} cm")

    demo = run_estimators(
        nodes, fading, snr_demo, rng, run_esprit=run_esprit, d_ula=d_ula,
    )
    print(f"  Demo @ {snr_demo:.0f} dB:")
    print(f"    MUSIC     : {demo['music_deg']:.3f} deg  (err {demo['music_deg']-true_b:+.3f})")
    print(f"    TDOA-LS   : {demo['tdoa_ls_deg']:.3f} deg  (err {demo['tdoa_ls_deg']-true_b:+.3f})")
    if run_esprit:
        print(f"    ULA-ESPRIT: {demo.get('esprit_deg', float('nan')):.3f} deg (sinθ-convention)")

    print("  Δτ true vs hat (ns):")
    for k, t, h in zip(demo["keys"], demo["delta_tau_true"], demo["delta_tau_hat"]):
        print(f"    1–{k}: true={t*1e9:7.3f}  hat={h*1e9:7.3f}")

    snr_range = np.arange(0, 36, 5)
    print(f"  Monte Carlo ({n_trials} trials/SNR)...")
    mc = monte_carlo_rmse(
        rx_xy, snr_range, n_trials, seed=hash(name) % (2**31) + 1,
        run_esprit=run_esprit, d_ula=d_ula,
    )
    header = f"  {'SNR':>4} | {'MUSIC':>8} | {'TDOA-LS':>8}"
    if run_esprit:
        header += f" | {'ESPRIT':>8}"
    print(header)
    for i, snr in enumerate(snr_range):
        line = f"  {snr:4.0f} | {mc['rmse_music'][i]:8.3f} | {mc['rmse_tdoa_ls'][i]:8.3f}"
        if run_esprit:
            line += f" | {mc['rmse_esprit'][i]:8.3f}"
        print(line)

    plot_comparison(name, nodes, demo, mc, show_esprit=run_esprit)
    plot_delta_tau_table(name, demo)


def main() -> None:
    print("=== Geometry MUSIC (A) vs Multi-baseline TDOA-LS (B) ===")
    print(f"fc={F_CENTER/1e9:.2f} GHz, λ={WAVELENGTH*100:.2f} cm, λ/2={WAVELENGTH/2*100:.2f} cm")

    # Equal ULA — also compare classic ESPRIT
    run_layout(
        "equal_ula",
        layout_equal(),
        run_esprit=True,
        d_ula=0.05,
        n_trials=120,
    )

    # Unequal spacings with long baseline > λ/2
    run_layout(
        "unequal",
        layout_unequal(),
        run_esprit=False,
        n_trials=120,
    )

    # 2-D irregular array
    run_layout(
        "irregular_2d",
        layout_2d_irregular(),
        run_esprit=False,
        n_trials=120,
    )

    print("\nDone. Figures: geometry_aoa_*.png")


if __name__ == "__main__":
    main()
