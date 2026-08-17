"""Collinear non-uniform RX array: all adjacent spacings > λ/2.

Demonstrates geometry MUSIC (A) and complex-domain multi-baseline LS (B)
under phase-wrap conditions.  Compares against a mixed layout that includes
one short baseline (≤ λ/2) for reference.

Run:
    python collinear_long_baseline_aoa.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from geometry_music_vs_tdoa_ls import (
    C_LIGHT,
    F_CENTER,
    WAVELENGTH,
    Node,
    build_fading,
    build_scene,
    monte_carlo_rmse,
    run_estimators,
    rx_list,
    true_bearing_deg,
    unit_vector,
)

HALF_WAVELENGTH = WAVELENGTH / 2.0


# =============================================================================
# Collinear layouts (RX on x=0, increasing y)
# =============================================================================
def layout_all_long() -> list[tuple[float, float]]:
    """4 RX; every adjacent gap > λ/2 (8 cm, 9 cm, 10 cm)."""
    ys = [0.00, 0.18, 0.27, 0.37]
    return [(0.0, y) for y in ys]


def layout_mixed() -> list[tuple[float, float]]:
    """Same span but one short gap 5 cm ≤ λ/2 for comparison."""
    ys = [0.00, 0.05, 0.14, 0.27]
    return [(0.0, y) for y in ys]


def adjacent_gaps_m(rx_xy: list[tuple[float, float]]) -> list[float]:
    ys = sorted(y for _, y in rx_xy)
    return [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]


def describe_layout(name: str, rx_xy: list[tuple[float, float]]) -> None:
    gaps = adjacent_gaps_m(rx_xy)
    print(f"\n{name}:")
    print(f"  λ/2 = {HALF_WAVELENGTH*100:.2f} cm")
    for i, g in enumerate(gaps, start=1):
        tag = "≤ λ/2" if g <= HALF_WAVELENGTH + 1e-9 else "> λ/2"
        print(f"  gap RX{i}–RX{i+1}: {g*100:.1f} cm  ({g/WAVELENGTH:.2f} λ)  {tag}")
    rxs_y = [y for _, y in rx_xy]
    print(f"  aperture (RX1–RX{len(rx_xy)}): {(max(rxs_y)-min(rxs_y))*100:.1f} cm")


def target_xy_for_bearing(
    bearing_deg: float,
    range_m: float = 5.0,
    array_mid: tuple[float, float] = (0.0, 0.135),
) -> tuple[float, float]:
    u = unit_vector(bearing_deg)
    mx, my = array_mid
    return (mx + range_m * u[0], my + range_m * u[1])


# =============================================================================
# Angle sweep (ambiguity / grating-lobe check)
# =============================================================================
def sweep_true_bearing(
    rx_xy: list[tuple[float, float]],
    bearing_range_deg: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Estimate AoA while sweeping true source bearing."""
    err_music = np.zeros(len(bearing_range_deg))
    err_ls = np.zeros(len(bearing_range_deg))
    mid_y = float(np.mean([y for _, y in rx_xy]))

    for i, b_true in enumerate(bearing_range_deg):
        tgt = target_xy_for_bearing(float(b_true), array_mid=(0.0, mid_y))
        nodes = build_scene(rng, rx_xy, tgt_xy=tgt)
        fading = build_fading(nodes)
        res = run_estimators(nodes, fading, snr_db, rng)
        err_music[i] = res["music_deg"] - float(b_true)
        err_ls[i] = res["tdoa_ls_deg"] - float(b_true)

    return {
        "bearing_true": bearing_range_deg,
        "err_music": err_music,
        "err_ls": err_ls,
    }


# =============================================================================
# Visualization
# =============================================================================
def plot_overview(
    layout_name: str,
    nodes: dict[int, Node],
    demo: dict,
    mc: dict,
    rx_xy: list[tuple[float, float]],
) -> None:
    rxs = rx_list(nodes)
    true_b = demo["true_bearing_deg"]

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)

    # Geometry
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot([n.x for n in rxs], [n.y for n in rxs], "o-", color="C0", ms=9, lw=2)
    for n in rxs:
        ax0.annotate(str(n.node_id), (n.x, n.y), xytext=(6, 4), textcoords="offset points")
    ax0.plot(nodes[0].x, nodes[0].y, "s", color="C2", ms=10, label="Ref 0")
    ax0.plot(nodes[5].x, nodes[5].y, "^", color="C3", ms=11, label="TX")
    mid = np.mean([n.pos for n in rxs], axis=0)
    ray = 2.2
    for th, ls, c, lab in (
        (true_b, "--", "C2", f"True {true_b:.1f}°"),
        (demo["music_deg"], "-", "C0", f"MUSIC {demo['music_deg']:.1f}°"),
        (demo["tdoa_ls_deg"], "-.", "C1", f"LS {demo['tdoa_ls_deg']:.1f}°"),
    ):
        u = unit_vector(th)
        ax0.plot([mid[0], mid[0] + ray * u[0]], [mid[1], mid[1] + ray * u[1]], ls, c=c, lw=2, label=lab)
    gaps = adjacent_gaps_m(rx_xy)
    gap_txt = ", ".join(f"{g*100:.0f}cm" for g in gaps)
    ax0.set_title(f"{layout_name}\nadjacent gaps: {gap_txt} (all > λ/2)" if all(g > HALF_WAVELENGTH for g in gaps)
                  else f"{layout_name}\nadjacent gaps: {gap_txt}")
    ax0.set_aspect("equal", adjustable="datalim")
    ax0.set_xlabel("x (m)")
    ax0.set_ylabel("y (m)")
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=7, loc="best")

    # MUSIC spectrum (grating lobes)
    ax1 = fig.add_subplot(gs[0, 1])
    spec_db = 10 * np.log10(demo["spectrum"] / np.max(demo["spectrum"]))
    ax1.plot(demo["scan_music"], spec_db, "C0", lw=1.5)
    ax1.axvline(true_b, color="C2", ls="--", label=f"True {true_b:.1f}°")
    ax1.axvline(demo["music_deg"], color="C0", ls="-", alpha=0.8, label="MUSIC peak")
    ax1.axvline(demo["tdoa_ls_deg"], color="C1", ls="-.", label="LS min")
    ax1.set_xlabel("Bearing θ (deg)")
    ax1.set_ylabel("MUSIC (dB)")
    ax1.set_title("Scheme A: geometry MUSIC")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7)

    # LS cost
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.semilogy(demo["scan_cost"], demo["cost"] + 1e-30, "C1", lw=1.5)
    ax2.axvline(true_b, color="C2", ls="--")
    ax2.axvline(demo["tdoa_ls_deg"], color="C1", ls="-.")
    ax2.set_xlabel("Bearing θ (deg)")
    ax2.set_ylabel("Σ |a − a(θ)|²")
    ax2.set_title("Scheme B: complex LS")
    ax2.grid(True, which="both", alpha=0.3)

    # RMSE vs SNR
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.semilogy(mc["snr"], mc["rmse_music"], "o-", lw=2, label="MUSIC")
    ax3.semilogy(mc["snr"], mc["rmse_tdoa_ls"], "s-", lw=2, label="LS")
    ax3.set_xlabel("SNR (dB)")
    ax3.set_ylabel("Bearing RMSE (deg)")
    ax3.set_title("Monte Carlo RMSE")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend()

    fig.suptitle(
        f"Collinear long-baseline AoA | {layout_name} | "
        f"MUSIC err={demo['music_deg']-true_b:+.2f}°  LS err={demo['tdoa_ls_deg']-true_b:+.2f}°",
        fontsize=12,
    )
    fname = f"collinear_long_{layout_name}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.close(fig)


def plot_bearing_sweep(layout_name: str, sweep: dict, snr_db: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(sweep["bearing_true"], sweep["err_music"], "-", lw=2, label="MUSIC error")
    ax.plot(sweep["bearing_true"], sweep["err_ls"], "--", lw=2, label="LS error")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("True bearing (deg)")
    ax.set_ylabel("Estimate − true (deg)")
    ax.set_title(f"{layout_name}: estimation error vs true bearing @ {snr_db:.0f} dB")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fname = f"collinear_sweep_{layout_name}.png"
    fig.savefig(fname, dpi=150)
    print(f"Saved: {fname}")
    plt.close(fig)


def plot_layout_compare(
    mc_long: dict,
    mc_mixed: dict,
    label_long: str,
    label_mixed: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(mc_long["snr"], mc_long["rmse_music"], "o-", lw=2, label=f"{label_long} MUSIC")
    ax.semilogy(mc_long["snr"], mc_long["rmse_tdoa_ls"], "s--", lw=2, label=f"{label_long} LS")
    ax.semilogy(mc_mixed["snr"], mc_mixed["rmse_music"], "^-", lw=2, label=f"{label_mixed} MUSIC")
    ax.semilogy(mc_mixed["snr"], mc_mixed["rmse_tdoa_ls"], "v--", lw=2, label=f"{label_mixed} LS")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Bearing RMSE (deg)")
    ax.set_title("All gaps > λ/2 vs mixed (one short baseline)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("collinear_long_vs_mixed_rmse.png", dpi=150)
    print("Saved: collinear_long_vs_mixed_rmse.png")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def run_scenario(
    name: str,
    rx_xy: list[tuple[float, float]],
    *,
    snr_demo: float = 25.0,
    n_trials: int = 120,
    seed: int = 7,
) -> tuple[dict, dict]:
    describe_layout(name, rx_xy)
    rng = np.random.default_rng(seed)
    nodes = build_scene(rng, rx_xy)
    fading = build_fading(nodes)
    true_b = true_bearing_deg(rx_list(nodes), nodes[5])
    print(f"  Demo true bearing: {true_b:.2f} deg")

    demo = run_estimators(nodes, fading, snr_demo, rng)
    print(f"  @ {snr_demo:.0f} dB: MUSIC {demo['music_deg']:.2f}°  LS {demo['tdoa_ls_deg']:.2f}°")

    snr_range = np.arange(5, 36, 5)
    print(f"  Monte Carlo ({n_trials}/SNR)...")
    mc = monte_carlo_rmse(rx_xy, snr_range, n_trials, seed=seed + 1)
    for snr, rm, rl in zip(snr_range, mc["rmse_music"], mc["rmse_tdoa_ls"]):
        print(f"    SNR {snr:2.0f}: MUSIC {rm:6.3f}°  LS {rl:6.3f}°")

    bearing_sweep = np.linspace(-60, 60, 49)
    sweep = sweep_true_bearing(rx_xy, bearing_sweep, snr_demo, rng)

    plot_overview(name, nodes, demo, mc, rx_xy)
    plot_bearing_sweep(name, sweep, snr_demo)
    return demo, mc


def main() -> None:
    print("=== Collinear non-uniform array, all spacings > λ/2 ===")
    print(f"fc = {F_CENTER/1e9:.2f} GHz, λ = {WAVELENGTH*100:.2f} cm, λ/2 = {HALF_WAVELENGTH*100:.2f} cm")

    rx_long = layout_all_long()
    rx_mixed = layout_mixed()

    _, mc_long = run_scenario("all_long", rx_long, snr_demo=25.0, n_trials=100, seed=11)
    _, mc_mixed = run_scenario("mixed_short", rx_mixed, snr_demo=25.0, n_trials=100, seed=22)

    plot_layout_compare(mc_long, mc_mixed, "all_long", "mixed_short")

    print("\nDone. Figures:")
    print("  collinear_long_all_long.png")
    print("  collinear_sweep_all_long.png")
    print("  collinear_long_mixed_short.png")
    print("  collinear_sweep_mixed_short.png")
    print("  collinear_long_vs_mixed_rmse.png")


if __name__ == "__main__":
    main()
