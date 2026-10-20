#!/usr/bin/env python3
"""
Process captures saved by new_listener.py (slotframe rounds + per-anchor lost flags).

Same pair-fusion / localization pipeline as coral_indoor_result_process.py, but:
  - load geometry from npz (meta.json optional);
  - skip anchors marked lost in that round;
  - only form pairs among non-lost anchors present that round.

Usage:
  python coral_data_process.py path/to/*_reports.npz
  python coral_data_process.py   # DEFAULT_NPZ
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import coral_indoor_result_process as cip
from CoRal_listener import compute_div_iq_and_ls

# Layout table aligned with new_listener.ANCHOR_STRUCTURES (used when meta is absent).
ANCHOR_STRUCTURES: Dict[int, Dict[str, Any]] = {
    3: {
        "name": "CROSS_8_d25mm",
        "positions": {
            2: (-0.09, -1.00),
            3: (-0.03, -1.00),
            4: (0.03, -1.00),
            5: (0.09, -1.00),
            6: (-1.00, -0.09),
            7: (-1.00, -0.03),
            8: (-1.00, 0.03),
            9: (-1.00, 0.09),
        },
        "x_ids": [2, 3, 4, 5],
        "y_ids": [6, 7, 8, 9],
    },
    2: {
        "name": "CROSS_8_d50mm",
        "positions": {
            2: (-0.1, -1),
            3: (0.0, -1),
            4: (0.15, -1),
            5: (0.35, -1),
            6: (-1, -0.15),
            7: (-1, 0.0),
            8: (-1, 0.1),
            9: (-1, 0.25),
        },
        "x_ids": [2, 3, 4, 5],
        "y_ids": [6, 7, 8, 9],
    },
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indoor_coral_3")
DEFAULT_NPZ = os.path.join(DATA_DIR, "(0.00,0.00)_(0.25,0.00)_3_towards_x_reports.npz")


def _iq_rows_to_pairs(arr: np.ndarray) -> List[Tuple[float, float]]:
    if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] == 0:
        return []
    out: List[Tuple[float, float]] = []
    for row in arr:
        if not (np.isfinite(row[0]) and np.isfinite(row[1])):
            continue
        out.append((float(row[0]), float(row[1])))
    return out


def _scalar_at(
    data: np.lib.npyio.NpzFile,
    prefix: str,
    name: str,
    round_idx: int,
    default: int = 0,
) -> int:
    key = prefix + name
    if key not in data.files:
        return default
    return int(np.asarray(data[key][round_idx]).item())


def anchor_lost(
    data: np.lib.npyio.NpzFile,
    aid: int,
    round_idx: int,
) -> bool:
    """True if this anchor is marked lost (or has no usable CTE) in the round."""
    prefix = f"id{aid}_"
    lost_key = prefix + "lost"
    if lost_key in data.files:
        return int(np.asarray(data[lost_key][round_idx]).item()) != 0
    # Legacy npz without lost[]: empty CTE ⇒ treat as lost.
    if prefix + "cte_ref" not in data.files or prefix + "cte_echo" not in data.files:
        return True
    ref = _iq_rows_to_pairs(np.asarray(data[prefix + "cte_ref"][round_idx]))
    echo = _iq_rows_to_pairs(np.asarray(data[prefix + "cte_echo"][round_idx]))
    return not (ref and echo)


def reports_for_round_no_lost(
    data: np.lib.npyio.NpzFile,
    anchors: List[int],
    round_idx: int,
) -> Tuple[Dict[int, Dict[str, Any]], List[int]]:
    """
    Build per-anchor reports for one round.
    Lost anchors are omitted (not paired). Returns (reports, lost_ids).
    """
    reports: Dict[int, Dict[str, Any]] = {}
    lost_ids: List[int] = []
    for aid in anchors:
        prefix = f"id{aid}_"
        if anchor_lost(data, aid, round_idx):
            lost_ids.append(aid)
            continue
        if prefix + "cte_ref" not in data.files or prefix + "cte_echo" not in data.files:
            lost_ids.append(aid)
            continue

        cte_ref = _iq_rows_to_pairs(np.asarray(data[prefix + "cte_ref"][round_idx]))
        cte_echo = _iq_rows_to_pairs(np.asarray(data[prefix + "cte_echo"][round_idx]))
        if not cte_ref or not cte_echo:
            lost_ids.append(aid)
            continue

        div_iq, intercept, slope = compute_div_iq_and_ls(cte_ref, cte_echo)
        if not div_iq:
            lost_ids.append(aid)
            continue

        reports[aid] = {
            "id": aid,
            "slot": _scalar_at(data, prefix, "slot", round_idx),
            "slotframe": _scalar_at(data, prefix, "slotframe", round_idx),
            "x": _scalar_at(data, prefix, "x", round_idx),
            "y": _scalar_at(data, prefix, "y", round_idx),
            "time_diff": _scalar_at(data, prefix, "time_diff", round_idx),
            "rssi": _scalar_at(data, prefix, "rssi", round_idx),
            "nib": _scalar_at(data, prefix, "nib", round_idx, len(cte_ref)),
            "necho": _scalar_at(data, prefix, "necho", round_idx, len(cte_echo)),
            "frags": _scalar_at(data, prefix, "frags", round_idx),
            "lost": 0,
            "cte_ref": cte_ref,
            "cte_echo": cte_echo,
            "div_iq": div_iq,
            "intercept": float(intercept),
            "slope": float(slope),
            "ndiv": len(div_iq),
        }
    return reports, lost_ids


def _geometry_from_structure_id(
    structure_id: int,
    anchors: List[int],
) -> Dict[str, Any]:
    st = ANCHOR_STRUCTURES.get(int(structure_id))
    if st is None:
        raise KeyError(
            f"unknown anchor_structure_id={structure_id}; "
            f"known={sorted(ANCHOR_STRUCTURES)}"
        )
    node_pos = {int(k): (float(v[0]), float(v[1])) for k, v in st["positions"].items()}
    x_ids = [int(a) for a in st["x_ids"]]
    y_ids = [int(a) for a in st["y_ids"]]
    if anchors:
        # Keep only requested anchors that exist in the layout.
        node_pos = {a: node_pos[a] for a in anchors if a in node_pos}
    return {
        "node_pos": node_pos,
        "x_ids": x_ids,
        "y_ids": y_ids,
        "structure_id": int(structure_id),
        "structure_name": str(st.get("name", "")),
    }


def load_capture(npz_path: str) -> Dict[str, Any]:
    """
    Load new_listener npz. Prefer sibling *_meta.json when present; otherwise
    read REF/TARGET/structure_id/anchor_list from the npz itself.
    """
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"npz not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    meta: Dict[str, Any] = {}
    meta_path = cip.meta_path_for_npz(npz_path)
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    # Positions: meta first, then npz arrays.
    if meta.get("ref_position") is not None or meta.get("ref_pos") is not None:
        ref_raw = meta.get("ref_position") or meta.get("ref_pos")
        tgt_raw = meta.get("target_position") or meta.get("target_pos")
        ref_pos = (float(ref_raw[0]), float(ref_raw[1]))
        target_pos = (float(tgt_raw[0]), float(tgt_raw[1]))
    else:
        ref_pos = tuple(float(x) for x in np.asarray(data["ref_position"]).ravel()[:2])
        target_pos = tuple(
            float(x) for x in np.asarray(data["target_position"]).ravel()[:2]
        )

    if "anchor_list" in data.files:
        anchors = [int(a) for a in np.asarray(data["anchor_list"]).ravel()]
    elif meta.get("anchor_list"):
        anchors = [int(a) for a in meta["anchor_list"]]
    else:
        anchors = sorted(
            {
                int(k[2 : k.index("_")])
                for k in data.files
                if k.startswith("id") and k.endswith("_cte_ref")
            }
        )

    structure_id = int(
        meta.get(
            "anchor_structure_id",
            int(np.asarray(data["anchor_structure_id"]).item())
            if "anchor_structure_id" in data.files
            else -1,
        )
    )

    # Geometry: meta.anchor_structure if complete, else structure table.
    try:
        if meta.get("anchor_structure"):
            geom = cip.parse_session_geometry(
                {
                    **meta,
                    "ref_position": list(ref_pos),
                    "target_position": list(target_pos),
                    "anchor_list": anchors,
                    "anchor_structure_id": structure_id,
                }
            )
        else:
            raise KeyError("no meta structure")
    except (KeyError, TypeError, ValueError):
        layout = _geometry_from_structure_id(structure_id, anchors)
        geom = {
            "ref_pos": ref_pos,
            "target_pos": target_pos,
            "anchors": anchors,
            **layout,
        }

    n_rounds = cip.n_rounds_in_npz(data, geom["anchors"])
    return {
        "npz_path": os.path.abspath(npz_path),
        "meta": meta,
        "data": data,
        "n_rounds": n_rounds,
        **geom,
    }


def process_session_pairs_no_lost(
    session: Dict[str, Any],
    *,
    half_plane_mode: str = "true_aoa",
    max_baseline_m: Optional[float] = None,
) -> Tuple[List[List[Dict[str, Any]]], Dict[str, Any], List[List[int]]]:
    """
    Like cip.process_session_pairs, but each round only pairs non-lost anchors.
    Returns (per_round_pairs, side_info, lost_per_round).
    """
    side_point_x = session["target_pos"]
    side_point_y = session["target_pos"]

    # Pass-1: raw AoAs (no half-plane) to infer side, using only present anchors.
    raw_rounds: List[List[Dict[str, Any]]] = []
    lost_per_round: List[List[int]] = []
    for r in range(session["n_rounds"]):
        reports, lost_ids = reports_for_round_no_lost(
            session["data"], session["anchors"], r
        )
        lost_per_round.append(lost_ids)
        x_ids = [a for a in session["x_ids"] if a in reports]
        y_ids = [a for a in session["y_ids"] if a in reports]
        raw_rounds.append(
            cip.estimate_pairs_in_group(
                reports,
                x_ids,
                session["ref_pos"],
                session["target_pos"],
                session["node_pos"],
                "X",
                prefer_positive=None,
                side_point=side_point_x,
                max_baseline_m=max_baseline_m,
            )
            + cip.estimate_pairs_in_group(
                reports,
                y_ids,
                session["ref_pos"],
                session["target_pos"],
                session["node_pos"],
                "Y",
                prefer_positive=None,
                side_point=side_point_y,
                max_baseline_m=max_baseline_m,
            )
        )

    # Side prior: true_aoa (default) or blind vote from raw samples.
    aliased_x = cip.group_has_aliased_baseline(session["x_ids"], session["node_pos"])
    aliased_y = cip.group_has_aliased_baseline(session["y_ids"], session["node_pos"])
    if half_plane_mode == "true_aoa":
        true_x = cip.true_aoa_deg_for_ula(
            session["target_pos"], session["x_ids"], session["node_pos"]
        )
        true_y = cip.true_aoa_deg_for_ula(
            session["target_pos"], session["y_ids"], session["node_pos"]
        )
        prefer_x: Optional[bool] = (
            None if abs(true_x) < cip.SIDE_CONFIDENCE_DEG else bool(true_x > 0.0)
        )
        prefer_y: Optional[bool] = (
            None if abs(true_y) < cip.SIDE_CONFIDENCE_DEG else bool(true_y > 0.0)
        )
        src_x, src_y = "true_aoa", "true_aoa"
    else:
        raw_x = [
            float(p["aoa_raw_deg"])
            for pairs in raw_rounds
            for p in pairs
            if p["group"] == "X"
        ]
        raw_y = [
            float(p["aoa_raw_deg"])
            for pairs in raw_rounds
            for p in pairs
            if p["group"] == "Y"
        ]
        prefer_x = cip.infer_side_from_raw_aoas(raw_x)
        prefer_y = cip.infer_side_from_raw_aoas(raw_y)
        src_x = "inferred" if prefer_x is not None else "ambiguous"
        src_y = "inferred" if prefer_y is not None else "ambiguous"

    side = {
        "prefer_x": prefer_x,
        "prefer_y": prefer_y,
        "source_x": src_x,
        "source_y": src_y,
        "aliased_x": aliased_x,
        "aliased_y": aliased_y,
    }

    # Pass-2: apply half-plane with only non-lost anchors each round.
    per_round: List[List[Dict[str, Any]]] = []
    for r in range(session["n_rounds"]):
        reports, _lost = reports_for_round_no_lost(
            session["data"], session["anchors"], r
        )
        x_ids = [a for a in session["x_ids"] if a in reports]
        y_ids = [a for a in session["y_ids"] if a in reports]
        per_round.append(
            cip.estimate_pairs_in_group(
                reports,
                x_ids,
                session["ref_pos"],
                session["target_pos"],
                session["node_pos"],
                "X",
                prefer_positive=side["prefer_x"],
                side_point=side_point_x,
                max_baseline_m=max_baseline_m,
            )
            + cip.estimate_pairs_in_group(
                reports,
                y_ids,
                session["ref_pos"],
                session["target_pos"],
                session["node_pos"],
                "Y",
                prefer_positive=side["prefer_y"],
                side_point=side_point_y,
                max_baseline_m=max_baseline_m,
            )
        )
    return per_round, side, lost_per_round


def resolve_npz_arg(argv: List[str]) -> str:
    if len(argv) > 2:
        raise SystemExit("Usage: python coral_data_process.py [one_reports.npz]")
    if len(argv) == 2:
        path = argv[1]
        if os.path.isdir(path):
            raise SystemExit(f"Got a directory ({path}); pass a single *_reports.npz.")
        return path
    return DEFAULT_NPZ


def main() -> int:
    npz_path = resolve_npz_arg(sys.argv)
    try:
        session = load_capture(npz_path)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Load failed: {exc}", file=sys.stderr)
        return 1

    print(f"==== {os.path.basename(session['npz_path'])} ====")
    print(f"  REF={session['ref_pos']}  TARGET={session['target_pos']}")
    print(
        f"  structure={session.get('structure_id')} ({session.get('structure_name', '')})"
    )
    print(f"  X-ULA={session['x_ids']}  Y-ULA={session['y_ids']}")
    print(f"  n_rounds={session['n_rounds']}")

    true_x = cip.true_aoa_deg_for_ula(
        session["target_pos"], session["x_ids"], session["node_pos"]
    )
    true_y = cip.true_aoa_deg_for_ula(
        session["target_pos"], session["y_ids"], session["node_pos"]
    )

    per_round, side, lost_per_round = process_session_pairs_no_lost(
        session, half_plane_mode="true_aoa"
    )

    n_lost_rounds = sum(1 for lost in lost_per_round if lost)
    n_lost_events = sum(len(lost) for lost in lost_per_round)
    print(
        f"  lost: {n_lost_events} anchor-events across {n_lost_rounds}/{session['n_rounds']} rounds "
        f"(pairs only among non-lost anchors)"
    )
    print(
        f"  half-plane: X={cip._side_label(side['prefer_x'], side.get('source_x'))}  "
        f"Y={cip._side_label(side['prefer_y'], side.get('source_y'))}"
    )
    print(f"  trueULA: X={true_x:+.2f}  Y={true_y:+.2f}")

    pair_series = cip.collect_pair_series(per_round)
    if not pair_series:
        print("No pair AoA estimates (too many lost anchors?).", file=sys.stderr)
        return 1

    print("---- per-pair summary (non-lost rounds only) ----")
    for p in pair_series:
        err = p["aoa_mean_deg"] - p["true_aoa_deg"]
        half = (
            "off"
            if not p["half_enabled"]
            else ("theta>0" if p["prefer_positive"] else "theta<0")
        )
        print(
            f"  {p['group']} {p['id_b']}-{p['id_a']}: "
            f"est={p['aoa_mean_deg']:+.2f}+/-{p['aoa_std_deg']:.2f} "
            f"(n={p['n_mean']}, sat={p['n_sat']})  "
            f"true={p['true_aoa_deg']:+.2f}  err={err:+.2f}  "
            f"half={half}  +pi x{p['n_pi_corrected']}/{p['n_rounds']}  "
            f"base={p['baseline_m']*100:.1f} cm"
        )

    loc = cip.fuse_session_localization(session, pair_series, per_round)
    print("---- fused AoA / position ----")
    print(
        f"  AoA X={loc['aoa_x_deg']:+.2f} (true {loc['true_aoa_x_deg']:+.2f})  "
        f"Y={loc['aoa_y_deg']:+.2f} (true {loc['true_aoa_y_deg']:+.2f})"
    )
    if loc["est_pos"] is not None:
        print(
            f"  est pos=({loc['est_pos'][0]:.3f},{loc['est_pos'][1]:.3f})  "
            f"true=({loc['true_pos'][0]:.3f},{loc['true_pos'][1]:.3f})  "
            f"err={loc['pos_err_m']*100:.1f} cm"
        )
    else:
        print("  est pos=None")
    n_ok = sum(1 for p in loc["round_pos"] if p is not None)
    print(f"  per-round positions: {n_ok}/{len(loc['round_pos'])}")

    cip.visualize_all_pairs(session, pair_series, loc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
