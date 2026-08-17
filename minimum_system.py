from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from aoa import (
    MINIMUM_DEFAULT_PROPAGATION_SNR_DB,
    MINIMUM_DEFAULT_RANDOM_SEED,
    MINIMUM_DEFAULT_RX_SNR_DB,
    MINIMUM_DEFAULT_SAMPLE_PHASE_SNR_DB,
    MINIMUM_LOCAL_NODE_IDS,
    MinimumSystemParams,
    default_minimum_system_params,
    simulate_minimum_system,
    visualize_minimum_system,
)

LOCAL_NODE_IDS = MINIMUM_LOCAL_NODE_IDS


class MinimumSystemApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("CoRal Minimum System")
        self.root.geometry("520x820")

        self.vars: dict[str, tk.StringVar] = {}
        main = ttk.Frame(root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="Local swarm positions (m)", font=("", 10, "bold")).pack(
            anchor=tk.W, pady=(0, 6)
        )
        local_frame = ttk.Frame(main)
        local_frame.pack(fill=tk.X, pady=(0, 10))
        for node_id in LOCAL_NODE_IDS:
            row = ttk.Frame(local_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=f"Node {node_id}", width=8).pack(side=tk.LEFT)
            self._add_entry(row, f"local_{node_id}_x", "x", "0.0")
            self._add_entry(row, f"local_{node_id}_y", "y", "0.0")

        defaults = default_minimum_system_params()
        for node_id, (x, y) in zip(LOCAL_NODE_IDS, defaults.local_positions):
            self.vars[f"local_{node_id}_x"].set(f"{x:.4f}")
            self.vars[f"local_{node_id}_y"].set(f"{y:.4f}")

        ttk.Separator(main).pack(fill=tk.X, pady=8)
        ttk.Label(main, text="Remote node position (m)", font=("", 10, "bold")).pack(
            anchor=tk.W, pady=(0, 6)
        )
        remote_row = ttk.Frame(main)
        remote_row.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(remote_row, text="Node 3", width=8).pack(side=tk.LEFT)
        self._add_entry(
            remote_row, "remote_x", "x", f"{defaults.remote_position[0]:.4f}"
        )
        self._add_entry(
            remote_row, "remote_y", "y", f"{defaults.remote_position[1]:.4f}"
        )

        ttk.Separator(main).pack(fill=tk.X, pady=8)
        ttk.Label(main, text="CTE timing and transmitter", font=("", 10, "bold")).pack(
            anchor=tk.W, pady=(0, 6)
        )
        timing_frame = ttk.Frame(main)
        timing_frame.pack(fill=tk.X, pady=(0, 10))

        t1_row = ttk.Frame(timing_frame)
        t1_row.pack(fill=tk.X, pady=2)
        ttk.Label(t1_row, text="t1 (ms)", width=12).pack(side=tk.LEFT)
        self._add_entry(t1_row, "t1_ms", "", f"{defaults.t1_s * 1e3:.4f}", width=16)

        t2_row = ttk.Frame(timing_frame)
        t2_row.pack(fill=tk.X, pady=2)
        ttk.Label(t2_row, text="t2 (ms)", width=12).pack(side=tk.LEFT)
        self._add_entry(t2_row, "t2_ms", "", f"{defaults.t2_s * 1e3:.4f}", width=16)

        tx_row = ttk.Frame(timing_frame)
        tx_row.pack(fill=tk.X, pady=2)
        ttk.Label(tx_row, text="CTE1 TX node", width=12).pack(side=tk.LEFT)
        self.vars["tx1_id"] = tk.StringVar(value=str(defaults.tx1_id))
        ttk.Combobox(
            tx_row,
            textvariable=self.vars["tx1_id"],
            values=[str(i) for i in LOCAL_NODE_IDS],
            width=14,
            state="readonly",
        ).pack(side=tk.LEFT, padx=6)

        seed_row = ttk.Frame(timing_frame)
        seed_row.pack(fill=tk.X, pady=2)
        ttk.Label(seed_row, text="Random seed", width=12).pack(side=tk.LEFT)
        self._add_entry(
            seed_row,
            "random_seed",
            "",
            str(MINIMUM_DEFAULT_RANDOM_SEED),
            width=16,
        )

        ttk.Separator(main).pack(fill=tk.X, pady=8)
        ttk.Label(main, text="Noise", font=("", 10, "bold")).pack(
            anchor=tk.W, pady=(0, 6)
        )
        noise_frame = ttk.Frame(main)
        noise_frame.pack(fill=tk.X, pady=(0, 10))

        for label, key, default in (
            ("Propagation SNR (dB)", "propagation_snr_db", MINIMUM_DEFAULT_PROPAGATION_SNR_DB),
            ("RX downconvert SNR (dB)", "rx_snr_db", MINIMUM_DEFAULT_RX_SNR_DB),
            ("Sample phase SNR (dB)", "sample_phase_snr_db", MINIMUM_DEFAULT_SAMPLE_PHASE_SNR_DB),
        ):
            row = ttk.Frame(noise_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=18).pack(side=tk.LEFT)
            self._add_entry(row, key, "", str(default), width=12)

        ttk.Separator(main).pack(fill=tk.X, pady=8)
        ttk.Button(main, text="Run simulation", command=self._on_run).pack(fill=tk.X)
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

    def _read_params(self) -> MinimumSystemParams:
        local_positions = []
        for node_id in LOCAL_NODE_IDS:
            x = float(self.vars[f"local_{node_id}_x"].get())
            y = float(self.vars[f"local_{node_id}_y"].get())
            local_positions.append((x, y))

        return MinimumSystemParams(
            local_positions=local_positions,
            remote_position=(
                float(self.vars["remote_x"].get()),
                float(self.vars["remote_y"].get()),
            ),
            t1_s=float(self.vars["t1_ms"].get()) * 1e-3,
            t2_s=float(self.vars["t2_ms"].get()) * 1e-3,
            tx1_id=int(self.vars["tx1_id"].get()),
            random_seed=int(self.vars["random_seed"].get()),
            propagation_snr_db=float(self.vars["propagation_snr_db"].get()),
            rx_snr_db=float(self.vars["rx_snr_db"].get()),
            sample_phase_snr_db=float(self.vars["sample_phase_snr_db"].get()),
        )

    def _on_run(self) -> None:
        try:
            params = self._read_params()
            result = simulate_minimum_system(params)
            visualize_minimum_system(result)
            self.status.config(
                text=(
                    f"Done. AoA est = {result['aoa_deg']:.2f} deg, "
                    f"true = {result['true_aoa_deg']:.2f} deg, "
                    f"error = {result['aoa_error_deg']:.2f} deg"
                ),
                foreground="green",
            )
        except Exception as exc:
            messagebox.showerror("Simulation error", str(exc))
            self.status.config(text=str(exc), foreground="red")


def main() -> None:
    root = tk.Tk()
    MinimumSystemApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
