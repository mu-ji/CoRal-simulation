from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from node import (
    DUAL_SWARM_ALL_NODE_IDS,
    DUAL_SWARM_A_IDS,
    DUAL_SWARM_B_IDS,
    DUAL_SWARM_DEFAULT_PROPAGATION_SNR_DB,
    DUAL_SWARM_DEFAULT_RANDOM_SEED,
    DUAL_SWARM_DEFAULT_RX_SNR_DB,
    DUAL_SWARM_DEFAULT_SAMPLE_PHASE_SNR_DB,
    DUAL_SWARM_DEFAULT_TX1_ID,
    DUAL_SWARM_DEFAULT_TX2_ID,
    DUAL_SWARM_NUM_SLOTS,
    DualSwarmParams,
    default_dual_swarm_params,
    simulate_dual_swarm,
    simulate_dual_swarm_all_slots,
    visualize_dual_swarm,
    visualize_dual_swarm_all_slots,
)

SWARM_A_IDS = DUAL_SWARM_A_IDS
SWARM_B_IDS = DUAL_SWARM_B_IDS
ALL_NODE_IDS = DUAL_SWARM_ALL_NODE_IDS
NUM_SLOTS = DUAL_SWARM_NUM_SLOTS


class DualSwarmSimulatorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("CoRal Dual Swarm Simulator")
        self.root.geometry("540x900")

        self.vars: dict[str, tk.StringVar] = {}
        main = ttk.Frame(root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        defaults = default_dual_swarm_params()

        ttk.Label(main, text="Swarm A positions (m)", font=("", 10, "bold")).pack(
            anchor=tk.W, pady=(0, 6)
        )
        swarm_a_frame = ttk.Frame(main)
        swarm_a_frame.pack(fill=tk.X, pady=(0, 8))
        for node_id in SWARM_A_IDS:
            row = ttk.Frame(swarm_a_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=f"Node {node_id}", width=8).pack(side=tk.LEFT)
            self._add_entry(row, f"swarm_a_{node_id}_x", "x", "0.0")
            self._add_entry(row, f"swarm_a_{node_id}_y", "y", "0.0")

        for node_id, (x, y) in zip(SWARM_A_IDS, defaults.swarm_a_positions):
            self.vars[f"swarm_a_{node_id}_x"].set(f"{x:.4f}")
            self.vars[f"swarm_a_{node_id}_y"].set(f"{y:.4f}")

        ttk.Separator(main).pack(fill=tk.X, pady=8)
        ttk.Label(main, text="Swarm B positions (m)", font=("", 10, "bold")).pack(
            anchor=tk.W, pady=(0, 6)
        )
        swarm_b_frame = ttk.Frame(main)
        swarm_b_frame.pack(fill=tk.X, pady=(0, 8))
        for node_id in SWARM_B_IDS:
            row = ttk.Frame(swarm_b_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=f"Node {node_id}", width=8).pack(side=tk.LEFT)
            self._add_entry(row, f"swarm_b_{node_id}_x", "x", "0.0")
            self._add_entry(row, f"swarm_b_{node_id}_y", "y", "0.0")

        for node_id, (x, y) in zip(SWARM_B_IDS, defaults.swarm_b_positions):
            self.vars[f"swarm_b_{node_id}_x"].set(f"{x:.4f}")
            self.vars[f"swarm_b_{node_id}_y"].set(f"{y:.4f}")

        ttk.Separator(main).pack(fill=tk.X, pady=8)
        ttk.Label(main, text="Slot timing", font=("", 10, "bold")).pack(
            anchor=tk.W, pady=(0, 6)
        )
        timing_frame = ttk.Frame(main)
        timing_frame.pack(fill=tk.X, pady=(0, 8))

        for label, key, value, width in (
            ("Slot duration (s)", "slot_duration_s", f"{defaults.slot_duration_s:.4f}", 12),
            ("CTE1 offset (ms)", "cte1_offset_ms", f"{defaults.cte1_offset_s * 1e3:.4f}", 12),
            ("CTE2 after CTE1 (us)", "cte2_offset_us", f"{defaults.cte2_offset_after_cte1_s * 1e6:.4f}", 12),
            ("Random seed", "random_seed", str(DUAL_SWARM_DEFAULT_RANDOM_SEED), 12),
        ):
            row = ttk.Frame(timing_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=18).pack(side=tk.LEFT)
            self._add_entry(row, key, "", value, width=width)

        slot_row = ttk.Frame(timing_frame)
        slot_row.pack(fill=tk.X, pady=2)
        ttk.Label(slot_row, text="CTE1 TX (tx1)", width=18).pack(side=tk.LEFT)
        self.vars["tx1_id"] = tk.StringVar(value=str(DUAL_SWARM_DEFAULT_TX1_ID))
        ttk.Combobox(
            slot_row,
            textvariable=self.vars["tx1_id"],
            values=[str(i) for i in ALL_NODE_IDS],
            width=10,
            state="readonly",
        ).pack(side=tk.LEFT, padx=6)

        tx2_row = ttk.Frame(timing_frame)
        tx2_row.pack(fill=tk.X, pady=2)
        ttk.Label(tx2_row, text="CTE2 TX (tx2)", width=18).pack(side=tk.LEFT)
        self.vars["tx2_id"] = tk.StringVar(value=str(DUAL_SWARM_DEFAULT_TX2_ID))
        ttk.Combobox(
            tx2_row,
            textvariable=self.vars["tx2_id"],
            values=[str(i) for i in ALL_NODE_IDS],
            width=10,
            state="readonly",
        ).pack(side=tk.LEFT, padx=6)
        ttk.Label(
            tx2_row,
            text="(RX = other nodes in tx1 swarm)",
            foreground="gray",
        ).pack(side=tk.LEFT, padx=4)

        ttk.Separator(main).pack(fill=tk.X, pady=8)
        ttk.Label(main, text="Noise", font=("", 10, "bold")).pack(
            anchor=tk.W, pady=(0, 6)
        )
        noise_frame = ttk.Frame(main)
        noise_frame.pack(fill=tk.X, pady=(0, 8))
        for label, key, default in (
            ("Propagation SNR (dB)", "propagation_snr_db", DUAL_SWARM_DEFAULT_PROPAGATION_SNR_DB),
            ("RX downconvert SNR (dB)", "rx_snr_db", DUAL_SWARM_DEFAULT_RX_SNR_DB),
            ("Sample phase SNR (dB)", "sample_phase_snr_db", DUAL_SWARM_DEFAULT_SAMPLE_PHASE_SNR_DB),
        ):
            row = ttk.Frame(noise_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=22).pack(side=tk.LEFT)
            self._add_entry(row, key, "", str(default), width=12)

        ttk.Separator(main).pack(fill=tk.X, pady=8)
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X)
        ttk.Button(
            btn_frame, text="Run selected slot", command=self._on_run_slot
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(
            btn_frame, text=f"Run all {NUM_SLOTS} slots", command=self._on_run_all
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        self.status = ttk.Label(main, text="Ready", foreground="gray")
        self.status.pack(anchor=tk.W, pady=(10, 0))

    def _add_entry(
        self,
        parent: ttk.Frame,
        key: str,
        label: str,
        default: str,
        width: int = 10,
    ) -> None:
        if label:
            ttk.Label(parent, text=label, width=3).pack(side=tk.LEFT, padx=(6, 0))
        self.vars[key] = tk.StringVar(value=default)
        ttk.Entry(parent, textvariable=self.vars[key], width=width).pack(
            side=tk.LEFT, padx=4
        )

    def _read_params(self) -> DualSwarmParams:
        swarm_a_positions = []
        for node_id in SWARM_A_IDS:
            swarm_a_positions.append(
                (
                    float(self.vars[f"swarm_a_{node_id}_x"].get()),
                    float(self.vars[f"swarm_a_{node_id}_y"].get()),
                )
            )
        swarm_b_positions = []
        for node_id in SWARM_B_IDS:
            swarm_b_positions.append(
                (
                    float(self.vars[f"swarm_b_{node_id}_x"].get()),
                    float(self.vars[f"swarm_b_{node_id}_y"].get()),
                )
            )

        tx1_id = int(self.vars["tx1_id"].get())
        tx2_id = int(self.vars["tx2_id"].get())
        cte2_offset_us = float(self.vars["cte2_offset_us"].get())

        return DualSwarmParams(
            swarm_a_positions=swarm_a_positions,
            swarm_b_positions=swarm_b_positions,
            slot_duration_s=float(self.vars["slot_duration_s"].get()),
            cte1_offset_s=float(self.vars["cte1_offset_ms"].get()) * 1e-3,
            cte2_offset_after_cte1_s=cte2_offset_us * 1e-6,
            tx1_id=tx1_id,
            tx2_id=tx2_id,
            random_seed=int(self.vars["random_seed"].get()),
            propagation_snr_db=float(self.vars["propagation_snr_db"].get()),
            rx_snr_db=float(self.vars["rx_snr_db"].get()),
            sample_phase_snr_db=float(self.vars["sample_phase_snr_db"].get()),
        )

    def _on_run_slot(self) -> None:
        try:
            params = self._read_params()
            result = simulate_dual_swarm(params)
            visualize_dual_swarm(result)
            tx1, tx2 = result["tx1_id"], result["tx2_id"]
            self.status.config(
                text=(
                    f"CTE1 TX {tx1} -> CTE2 TX {tx2}: "
                    f"AoA est = {result['aoa_deg']:.2f} deg, "
                    f"true = {result['true_aoa_deg']:.2f} deg, "
                    f"error = {result['aoa_error_deg']:.2f} deg"
                ),
                foreground="green",
            )
        except Exception as exc:
            messagebox.showerror("Simulation error", str(exc))
            self.status.config(text=str(exc), foreground="red")

    def _on_run_all(self) -> None:
        try:
            params = self._read_params()
            analyses = simulate_dual_swarm_all_slots(params)
            visualize_dual_swarm_all_slots(analyses)
            mean_error = float(np.mean([a["aoa_error_deg"] for a in analyses]))
            max_error = float(np.max([a["aoa_error_deg"] for a in analyses]))
            self.status.config(
                text=(
                    f"All {NUM_SLOTS} slots done. "
                    f"Mean error = {mean_error:.2f} deg, "
                    f"max error = {max_error:.2f} deg"
                ),
                foreground="green",
            )
        except Exception as exc:
            messagebox.showerror("Simulation error", str(exc))
            self.status.config(text=str(exc), foreground="red")


def main() -> None:
    root = tk.Tk()
    DualSwarmSimulatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
