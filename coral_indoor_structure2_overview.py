#!/usr/bin/env python3
"""
Overview plot for all indoor_coral sessions with anchor_structure_id == 2.

Same pipeline as coral_indoor_structure1_overview.py
(pair-fusion localization via coral_indoor_result_process.py), but for
structure 2 (CROSS_8 non-uniform / larger spacings; positions come from meta).

Usage:
  python coral_indoor_result_structure2_overview.py
  python coral_indoor_result_structure2_overview.py path/to/indoor_coral
"""

from __future__ import annotations

import glob
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Missing dependency: matplotlib  (pip install matplotlib)", file=sys.stderr)
    sys.exit(1)

import coral_indoor_result_process as cip

INDOOR_CORAL_DIR = cip.INDOOR_CORAL_DIR
STRUCTURE_ID = 2


def find_structure_sessions(
    data_dir: str,
    structure_id: int = STRUCTURE_ID,
) -> List[str]:
    """Return sorted paths to *_reports.npz whose meta has the given structure id."""
    pattern = os.path.join(data_dir, "*_reports.npz")
    out: List[str] = []
    for npz_path in sorted(glob.glob(pattern)):
        try:
            meta = cip.load_meta(npz_path)
        except FileNotFoundError:
            print(f"  skip (no meta): {os.path.basename(npz_path)}", file=sys.stderr)
            continue
        sid = int(meta.get("anchor_structure_id", -1))
        if sid != structure_id:
            continue
        out.append(npz_path)
    return out


def _ula_spacings_cm(
    ids: List[int],
    node_pos: Dict[int, Tuple[float, float]],
) -> List[float]:
    """Consecutive element spacings along a ULA (cm), order by coordinate."""
    xs = [node_pos[a][0] for a in ids]
    ys = [node_pos[a][1] for a in ids]
    along_x = (max(xs) - min(xs)) >= (max(ys) - min(ys))
    if along_x:
        ordered = sorted(ids, key=lambda a: (node_pos[a][0], node_pos[a][1], a))
    else:
        ordered = sorted(ids, key=lambda a: (node_pos[a][1], node_pos[a][0], a))
    out: List[float] = []
    for i in range(len(ordered) - 1):
        out.append(cip.dist(node_pos[ordered[i]], node_pos[ordered[i + 1]]) * 100.0)
    return out


def process_one_session(npz_path: str) -> Optional[Dict[str, Any]]:
    """Load one session and return fused localization summary."""
    session = cip.load_capture(npz_path)
    per_round, side = cip.process_session_pairs(session)
    pair_series = cip.collect_pair_series(per_round)
    if not pair_series:
        return None
    loc = cip.fuse_session_localization(session, pair_series, per_round)
    return {
        "npz_path": session["npz_path"],
        "name": os.path.basename(session["npz_path"]),
        "ref_pos": session["ref_pos"],
        "node_pos": session["node_pos"],
        "x_ids": session["x_ids"],
        "y_ids": session["y_ids"],
        "structure_name": session["structure_name"],
        "true_pos": loc["true_pos"],
        "est_pos": loc["est_pos"],
        "pos_err_m": loc["pos_err_m"],
        "aoa_x_deg": loc["aoa_x_deg"],
        "aoa_y_deg": loc["aoa_y_deg"],
        "true_aoa_x_deg": loc["true_aoa_x_deg"],
        "true_aoa_y_deg": loc["true_aoa_y_deg"],
        "n_rounds": session["n_rounds"],
        "side": side,
    }


def visualize_overview(results: List[Dict[str, Any]]) -> None:
    """One figure: all true TARGET vs estimate pairs, linked by dashed lines."""
    node_pos = results[0]["node_pos"]
    ref_pos = results[0]["ref_pos"]
    x_ids = results[0]["x_ids"]
    y_ids = results[0]["y_ids"]

    xs = [p[0] for p in node_pos.values()] + [ref_pos[0]]
    ys = [p[1] for p in node_pos.values()] + [ref_pos[1]]
    for r in results:
        xs.append(r["true_pos"][0])
        ys.append(r["true_pos"][1])
        if r["est_pos"] is not None:
            xs.append(r["est_pos"][0])
            ys.append(r["est_pos"][1])
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    span = max(xmax - xmin, ymax - ymin, 0.2)
    pad = 0.15 * span
    xmin, xmax = xmin - pad, xmax + pad
    ymin, ymax = ymin - pad, ymax + pad

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 8.0))

    ax.plot(
        [node_pos[a][0] for a in x_ids],
        [node_pos[a][1] for a in x_ids],
        "o-",
        color="0.55",
        ms=6,
        lw=1.0,
        label="X-ULA",
        zorder=2,
    )
    ax.plot(
        [node_pos[a][0] for a in y_ids],
        [node_pos[a][1] for a in y_ids],
        "s-",
        color="0.55",
        ms=6,
        lw=1.0,
        label="Y-ULA",
        zorder=2,
    )
    for aid, (px, py) in node_pos.items():
        ax.annotate(str(aid), (px, py), textcoords="offset points", xytext=(3, 3), fontsize=7)

    # Annotate non-uniform consecutive spacings (structure 2).
    x_sp = _ula_spacings_cm(x_ids, node_pos)
    y_sp = _ula_spacings_cm(y_ids, node_pos)
    x_ord = sorted(x_ids, key=lambda a: (node_pos[a][0], node_pos[a][1], a))
    y_ord = sorted(y_ids, key=lambda a: (node_pos[a][1], node_pos[a][0], a))
    for i, sp in enumerate(x_sp):
        a, b = x_ord[i], x_ord[i + 1]
        mx = 0.5 * (node_pos[a][0] + node_pos[b][0])
        my = 0.5 * (node_pos[a][1] + node_pos[b][1])
        ax.annotate(
            f"{sp:.0f}",
            (mx, my),
            textcoords="offset points",
            xytext=(0, -12),
            fontsize=6,
            color="0.35",
            ha="center",
        )
    for i, sp in enumerate(y_sp):
        a, b = y_ord[i], y_ord[i + 1]
        mx = 0.5 * (node_pos[a][0] + node_pos[b][0])
        my = 0.5 * (node_pos[a][1] + node_pos[b][1])
        ax.annotate(
            f"{sp:.0f}",
            (mx, my),
            textcoords="offset points",
            xytext=(10, 0),
            fontsize=6,
            color="0.35",
            ha="left",
            va="center",
        )

    ax.plot(ref_pos[0], ref_pos[1], "^", color="gray", ms=11, label="REF", zorder=3)

    cmap = plt.cm.tab10  # type: ignore[attr-defined]
    for i, r in enumerate(results):
        color = cmap(i % 10)
        tx, ty = r["true_pos"]
        ax.plot(tx, ty, "*", color=color, ms=14, zorder=5)
        if r["est_pos"] is None:
            ax.annotate(
                f"{i+1}: no est",
                (tx, ty),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=7,
                color=color,
            )
            continue
        ex, ey = r["est_pos"]
        ax.plot(ex, ey, "P", color=color, ms=11, markeredgecolor="k", markeredgewidth=0.5, zorder=6)
        ax.plot(
            [tx, ex],
            [ty, ey],
            "--",
            color=color,
            lw=1.3,
            alpha=0.85,
            zorder=4,
        )
        err_cm = r["pos_err_m"] * 100.0 if math.isfinite(r["pos_err_m"]) else float("nan")
        mid = (0.5 * (tx + ex), 0.5 * (ty + ey))
        ax.annotate(
            f"{err_cm:.0f}cm",
            mid,
            fontsize=7,
            color=color,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
            zorder=7,
        )

    ax.plot([], [], "*", color="k", ms=10, label="true TARGET")
    ax.plot([], [], "P", color="k", ms=9, label="fused estimate")
    ax.plot([], [], "--", color="k", lw=1.2, label="true–est pair")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    errs = [
        r["pos_err_m"] * 100.0
        for r in results
        if r["est_pos"] is not None and math.isfinite(r["pos_err_m"])
    ]
    mean_err = float(np.mean(errs)) if errs else float("nan")
    med_err = float(np.median(errs)) if errs else float("nan")
    struct_name = results[0].get("structure_name") or f"structure_{STRUCTURE_ID}"
    sp_note = (
        f"X spacings=[{', '.join(f'{s:.0f}' for s in x_sp)}] cm  "
        f"Y spacings=[{', '.join(f'{s:.0f}' for s in y_sp)}] cm"
    )
    ax.set_title(
        f"Structure {STRUCTURE_ID} ({struct_name})  N={len(results)} sessions\n"
        f"{sp_note}\n"
        f"mean err={mean_err:.1f} cm  median err={med_err:.1f} cm"
    )
    fig.tight_layout()
    print("Close figure to exit.")
    plt.show(block=True)
    plt.close(fig)


def main() -> int:
    data_dir = sys.argv[1] if len(sys.argv) > 1 else INDOOR_CORAL_DIR
    if not os.path.isdir(data_dir):
        print(f"Not a directory: {data_dir}", file=sys.stderr)
        return 1

    npz_files = find_structure_sessions(data_dir, STRUCTURE_ID)
    if not npz_files:
        print(f"No structure_id={STRUCTURE_ID} sessions under {data_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(npz_files)} session(s) with structure_id={STRUCTURE_ID}")
    # Print layout once from the first meta.
    try:
        first = cip.load_capture(npz_files[0])
        x_sp = _ula_spacings_cm(first["x_ids"], first["node_pos"])
        y_sp = _ula_spacings_cm(first["y_ids"], first["node_pos"])
        print(
            f"  layout={first.get('structure_name', '')}  "
            f"X d=[{', '.join(f'{s:.1f}' for s in x_sp)}] cm  "
            f"Y d=[{', '.join(f'{s:.1f}' for s in y_sp)}] cm"
        )
    except Exception as exc:
        print(f"  (could not print layout: {exc})", file=sys.stderr)

    results: List[Dict[str, Any]] = []
    for path in npz_files:
        name = os.path.basename(path)
        print(f"---- {name} ----")
        try:
            res = process_one_session(path)
        except Exception as exc:
            print(f"  failed: {exc}", file=sys.stderr)
            continue
        if res is None:
            print("  skip: no pair estimates")
            continue
        results.append(res)
        if res["est_pos"] is None:
            print(f"  TARGET={res['true_pos']}  est=None")
        else:
            side = res.get("side") or {}
            print(
                f"  TARGET=({res['true_pos'][0]:.3f},{res['true_pos'][1]:.3f})  "
                f"est=({res['est_pos'][0]:.3f},{res['est_pos'][1]:.3f})  "
                f"err={res['pos_err_m']*100:.1f} cm  "
                f"AoA X={res['aoa_x_deg']:+.1f} Y={res['aoa_y_deg']:+.1f}  "
                f"side X={cip._side_label(side.get('prefer_x'), side.get('source_x'))}  "
                f"Y={cip._side_label(side.get('prefer_y'), side.get('source_y'))}"
            )

    if not results:
        print("No sessions processed successfully.", file=sys.stderr)
        return 1

    print("---- overview ----")
    errs = [
        r["pos_err_m"] * 100.0
        for r in results
        if r["est_pos"] is not None and math.isfinite(r["pos_err_m"])
    ]
    if errs:
        print(f"  N={len(errs)}  mean={np.mean(errs):.1f} cm  median={np.median(errs):.1f} cm")

    visualize_overview(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
