from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from node import (
    TRIPLE_SWARM_1_IDS,
    TRIPLE_SWARM_2_IDS,
    TRIPLE_SWARM_3_IDS,
    TRIPLE_SWARM_DEFAULT_CTE1_OFFSET_MS,
    TRIPLE_SWARM_DEFAULT_CTE2_OFFSET_US,
    TRIPLE_SWARM_DEFAULT_MOVE_ERROR_M,
    TRIPLE_SWARM_DEFAULT_PROPAGATION_SNR_DB,
    TRIPLE_SWARM_DEFAULT_RANDOM_SEED,
    TRIPLE_SWARM_DEFAULT_RX_SNR_DB,
    TRIPLE_SWARM_DEFAULT_SAMPLE_PHASE_SNR_DB,
    TRIPLE_SWARM_STEP_SEQUENCE,
    TripleSwarmParams,
    TripleSwarmSnapshot,
    TripleSwarmState,
    create_triple_swarm_state,
    default_triple_swarm_params,
    draw_triple_swarm_on_axes,
    draw_triple_swarm_centroid_error_history,
    simulate_triple_swarm_cycle,
    simulate_triple_swarm_step,
    snapshot_triple_swarm_state,
    triple_swarm_step_label,
)

SWARM_GROUPS = (
    ("Swarm 1", TRIPLE_SWARM_1_IDS, "swarm1"),
    ("Swarm 2", TRIPLE_SWARM_2_IDS, "swarm2"),
    ("Swarm 3", TRIPLE_SWARM_3_IDS, "swarm3"),
)
NUM_STEPS_PER_CYCLE = len(TRIPLE_SWARM_STEP_SEQUENCE)
OPERATION_SEQUENCE = "__sequence__"


class TripleSwarmSimulatorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("CoRal Triple Swarm Simulator")
        self.root.geometry("1600x820")
        self.root.minsize(1100, 700)

        self.vars: dict[str, tk.StringVar] = {}
        self.state: TripleSwarmState | None = None
        self.params: TripleSwarmParams | None = None
        self.last_result: dict | None = None
        self.before_snapshot: TripleSwarmSnapshot | None = None

        paned = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left_outer = ttk.Frame(paned, width=380)
        right_panel = ttk.Frame(paned)
        paned.add(left_outer, weight=1)
        paned.add(right_panel, weight=3)

        self._build_control_panel(left_outer)
        self._build_visual_panel(right_panel)

        self._on_reset()

    def _build_control_panel(self, parent: ttk.Frame) -> None:
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        scroll_frame = ttk.Frame(canvas, padding=10)
        scroll_frame.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        defaults = default_triple_swarm_params()
        position_lists = (
            defaults.swarm1_positions,
            defaults.swarm2_positions,
            defaults.swarm3_positions,
        )
        targets = (
            defaults.swarm1_target,
            defaults.swarm2_target,
            defaults.swarm3_target,
        )

        ttk.Label(
            scroll_frame, text="Simulator controls", font=("", 11, "bold")
        ).pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(scroll_frame, text="Initial positions (m)", font=("", 10, "bold")).pack(
            anchor=tk.W, pady=(0, 4)
        )
        for title, node_ids, key in SWARM_GROUPS:
            ttk.Label(scroll_frame, text=title).pack(anchor=tk.W, pady=(4, 2))
            for node_id in node_ids:
                row = ttk.Frame(scroll_frame)
                row.pack(fill=tk.X, pady=1)
                ttk.Label(row, text=f"Node {node_id}", width=8).pack(side=tk.LEFT)
                self._add_entry(row, f"{key}_{node_id}_x", "x", "0.0", width=8)
                self._add_entry(row, f"{key}_{node_id}_y", "y", "0.0", width=8)

        for (title, node_ids, key), positions in zip(SWARM_GROUPS, position_lists):
            for node_id, (x, y) in zip(node_ids, positions):
                self.vars[f"{key}_{node_id}_x"].set(f"{x:.4f}")
                self.vars[f"{key}_{node_id}_y"].set(f"{y:.4f}")

        ttk.Separator(scroll_frame).pack(fill=tk.X, pady=8)
        ttk.Label(scroll_frame, text="Operation", font=("", 10, "bold")).pack(
            anchor=tk.W, pady=(0, 4)
        )
        op_labels = ["Next in sequence"] + [
            triple_swarm_step_label(step) for step in TRIPLE_SWARM_STEP_SEQUENCE
        ]
        op_values = [OPERATION_SEQUENCE] + list(TRIPLE_SWARM_STEP_SEQUENCE)
        self.vars["operation"] = tk.StringVar(value=OPERATION_SEQUENCE)
        self._operation_map = dict(zip(op_values, op_labels))
        self._operation_reverse = {v: k for k, v in self._operation_map.items()}
        ttk.Combobox(
            scroll_frame,
            textvariable=self.vars["operation"],
            values=op_values,
            state="readonly",
            width=36,
        ).pack(fill=tk.X, pady=2)
        ttk.Label(
            scroll_frame,
            text="Sequence: S3 localize -> move -> L; S1 ...; S2 ...",
            foreground="gray",
            wraplength=340,
        ).pack(anchor=tk.W, pady=(0, 4))

        ttk.Separator(scroll_frame).pack(fill=tk.X, pady=8)
        ttk.Label(scroll_frame, text="Move targets (m)", font=("", 10, "bold")).pack(
            anchor=tk.W, pady=(0, 4)
        )
        for (title, _, key), (tx, ty) in zip(SWARM_GROUPS, targets):
            row = ttk.Frame(scroll_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=title, width=10).pack(side=tk.LEFT)
            self._add_entry(row, f"{key}_target_x", "x", f"{tx:.4f}", width=10)
            self._add_entry(row, f"{key}_target_y", "y", f"{ty:.4f}", width=10)

        ttk.Separator(scroll_frame).pack(fill=tk.X, pady=8)
        ttk.Label(
            scroll_frame, text="Timing, motion and noise", font=("", 10, "bold")
        ).pack(anchor=tk.W, pady=(0, 4))
        for label, key, value, width in (
            ("CTE1 offset (ms)", "cte1_offset_ms", f"{defaults.cte1_offset_s * 1e3:.4f}", 10),
            ("CTE2 after CTE1 (us)", "cte2_offset_us", f"{defaults.cte2_offset_after_cte1_s * 1e6:.4f}", 10),
            ("Move error std (m)", "move_error_std_m", f"{defaults.move_error_std_m:.4f}", 10),
            ("Random seed", "random_seed", str(TRIPLE_SWARM_DEFAULT_RANDOM_SEED), 10),
            ("Full cycles", "num_cycles", "1", 10),
        ):
            row = ttk.Frame(scroll_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=18).pack(side=tk.LEFT)
            self._add_entry(row, key, "", value, width=width)

        for label, key, default in (
            ("Propagation SNR (dB)", "propagation_snr_db", TRIPLE_SWARM_DEFAULT_PROPAGATION_SNR_DB),
            ("RX SNR (dB)", "rx_snr_db", TRIPLE_SWARM_DEFAULT_RX_SNR_DB),
            ("Sample phase SNR (dB)", "sample_phase_snr_db", TRIPLE_SWARM_DEFAULT_SAMPLE_PHASE_SNR_DB),
        ):
            row = ttk.Frame(scroll_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=18).pack(side=tk.LEFT)
            self._add_entry(row, key, "", str(default), width=10)

        ttk.Separator(scroll_frame).pack(fill=tk.X, pady=8)
        ttk.Button(scroll_frame, text="Reset / Initialize", command=self._on_reset).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(scroll_frame, text="Run operation", command=self._on_run_operation).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(scroll_frame, text="Run full cycle", command=self._on_run_cycle).pack(
            fill=tk.X, pady=2
        )

        self.step_label = ttk.Label(scroll_frame, text="Step: not started", foreground="gray")
        self.step_label.pack(anchor=tk.W, pady=(10, 2))
        self.status = ttk.Label(
            scroll_frame,
            text="Ready. Initialize to begin.",
            foreground="gray",
            wraplength=340,
        )
        self.status.pack(anchor=tk.W)

    def _build_visual_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text="Visualization: before | after | centroid error history",
            font=("", 10, "bold"),
        ).pack(anchor=tk.W, padx=8, pady=(8, 4))

        self.fig = Figure(figsize=(12, 5), dpi=100)
        self.ax_before = self.fig.add_subplot(1, 3, 1)
        self.ax_after = self.fig.add_subplot(1, 3, 2)
        self.ax_centroid = self.fig.add_subplot(1, 3, 3)
        self.fig.subplots_adjust(wspace=0.32, left=0.04, right=0.98, top=0.9, bottom=0.12)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    def _add_entry(
        self,
        parent: ttk.Frame,
        key: str,
        label: str,
        default: str,
        width: int = 10,
    ) -> None:
        if label:
            ttk.Label(parent, text=label, width=2).pack(side=tk.LEFT, padx=(4, 0))
        self.vars[key] = tk.StringVar(value=default)
        ttk.Entry(parent, textvariable=self.vars[key], width=width).pack(
            side=tk.LEFT, padx=3
        )

    def _read_positions(self, key: str, node_ids: tuple[int, ...]) -> list[tuple[float, float]]:
        return [
            (
                float(self.vars[f"{key}_{node_id}_x"].get()),
                float(self.vars[f"{key}_{node_id}_y"].get()),
            )
            for node_id in node_ids
        ]

    def _read_params(self) -> TripleSwarmParams:
        cte2_offset_us = float(self.vars["cte2_offset_us"].get())
        if cte2_offset_us <= 0.0:
            raise ValueError("CTE2 offset after CTE1 must be positive.")

        return TripleSwarmParams(
            swarm1_positions=self._read_positions("swarm1", TRIPLE_SWARM_1_IDS),
            swarm2_positions=self._read_positions("swarm2", TRIPLE_SWARM_2_IDS),
            swarm3_positions=self._read_positions("swarm3", TRIPLE_SWARM_3_IDS),
            swarm1_target=(
                float(self.vars["swarm1_target_x"].get()),
                float(self.vars["swarm1_target_y"].get()),
            ),
            swarm2_target=(
                float(self.vars["swarm2_target_x"].get()),
                float(self.vars["swarm2_target_y"].get()),
            ),
            swarm3_target=(
                float(self.vars["swarm3_target_x"].get()),
                float(self.vars["swarm3_target_y"].get()),
            ),
            cte1_offset_s=float(self.vars["cte1_offset_ms"].get()) * 1e-3,
            cte2_offset_after_cte1_s=cte2_offset_us * 1e-6,
            move_error_std_m=float(self.vars["move_error_std_m"].get()),
            random_seed=int(self.vars["random_seed"].get()),
            propagation_snr_db=float(self.vars["propagation_snr_db"].get()),
            rx_snr_db=float(self.vars["rx_snr_db"].get()),
            sample_phase_snr_db=float(self.vars["sample_phase_snr_db"].get()),
        )

    def _selected_operation(self) -> str | None:
        op = self.vars["operation"].get()
        if op == OPERATION_SEQUENCE:
            return None
        return op

    def _update_step_label(self) -> None:
        if self.state is None:
            self.step_label.config(text="Step: not started")
            return
        idx = self.state.step_index % NUM_STEPS_PER_CYCLE
        step_name = TRIPLE_SWARM_STEP_SEQUENCE[idx]
        self.step_label.config(
            text=(
                f"Cycle {self.state.cycle_index + 1}, "
                f"next in sequence ({idx + 1}/{NUM_STEPS_PER_CYCLE}): "
                f"{triple_swarm_step_label(step_name)}"
            ),
            foreground="black",
        )

    def _format_result(self, result: dict) -> str:
        if result.get("step_type") == "localize":
            return (
                f"{result['step_label']}: localization error = "
                f"{result['localization_error_m']:.3f} m"
            )
        if result.get("step_type") == "move":
            return (
                f"{result['step_label']}: move error = "
                f"{result['planned_error_m']:.3f} m"
            )
        if result.get("step_type") == "reform":
            reformed = "reformed to L" if result.get("reformed") else "already L-shaped"
            return f"{result['step_label']}: {reformed}"
        return result.get("step_label", "Operation complete")

    def _refresh_plot(
        self,
        before: TripleSwarmSnapshot | None,
        after_snapshot: TripleSwarmSnapshot | None = None,
        step_result: dict | None = None,
        before_title: str = "Current state",
        after_title: str = "After operation",
        centroid_title: str = "Centroid error history",
    ) -> None:
        if before is not None:
            draw_triple_swarm_on_axes(self.ax_before, snapshot=before, title=before_title)
        else:
            self.ax_before.clear()
            self.ax_before.set_title(before_title)

        if self.state is not None and (step_result is not None or after_snapshot is not None):
            draw_triple_swarm_on_axes(
                self.ax_after,
                swarms=self.state.swarms,
                believed=self.state.believed_centroids,
                step_result=step_result,
                title=after_title,
            )
        elif before is not None:
            draw_triple_swarm_on_axes(
                self.ax_after, snapshot=before, title=after_title
            )
        else:
            self.ax_after.clear()
            self.ax_after.set_title(after_title)

        history = self.state.history if self.state is not None else None
        draw_triple_swarm_centroid_error_history(
            self.ax_centroid,
            history,
            title=centroid_title,
        )

        self.canvas.draw_idle()

    def _show_initial_layout(self) -> None:
        if self.state is None:
            return
        snap = snapshot_triple_swarm_state(self.state)
        self.before_snapshot = snap
        self._refresh_plot(
            before=snap,
            before_title="Current state",
            after_title="Current state",
            centroid_title="Centroid error history",
        )

    def _on_reset(self) -> None:
        try:
            self.params = self._read_params()
            self.state = create_triple_swarm_state(self.params)
            self.last_result = None
            self._update_step_label()
            self._show_initial_layout()
            self.status.config(
                text="Initialized. Swarm 1/2 anchor Swarm 3; run localize Swarm 3 first.",
                foreground="green",
            )
        except Exception as exc:
            messagebox.showerror("Initialization error", str(exc))
            self.status.config(text=str(exc), foreground="red")

    def _on_run_operation(self) -> None:
        if self.state is None:
            messagebox.showwarning("Not initialized", "Click Reset / Initialize first.")
            return
        try:
            self.params = self._read_params()
            before = snapshot_triple_swarm_state(self.state)
            operation = self._selected_operation()
            advance = operation is None
            result = simulate_triple_swarm_step(
                self.state,
                self.params,
                operation=operation,
                advance_sequence=advance,
            )
            self.last_result = result
            self.before_snapshot = before
            self._refresh_plot(
                before=before,
                step_result=result,
                before_title="Before operation",
                after_title=result["step_label"],
                centroid_title="Centroid error history",
            )
            self._update_step_label()
            self.status.config(text=self._format_result(result), foreground="green")
        except Exception as exc:
            messagebox.showerror("Simulation error", str(exc))
            self.status.config(text=str(exc), foreground="red")

    def _on_run_cycle(self) -> None:
        if self.state is None:
            messagebox.showwarning("Not initialized", "Click Reset / Initialize first.")
            return
        try:
            self.params = self._read_params()
            before = snapshot_triple_swarm_state(self.state)
            num_cycles = max(1, int(self.vars["num_cycles"].get()))
            results = simulate_triple_swarm_cycle(
                self.state, self.params, num_cycles=num_cycles
            )
            self.last_result = results[-1] if results else None
            self.before_snapshot = before
            if self.last_result is not None:
                self._refresh_plot(
                    before=before,
                    step_result=self.last_result,
                    before_title=f"Before {num_cycles} cycle(s)",
                    after_title=self.last_result["step_label"],
                    centroid_title="Centroid error history",
                )
            self._update_step_label()
            self.status.config(
                text=f"Completed {num_cycles} cycle(s), {len(results)} operations.",
                foreground="green",
            )
        except Exception as exc:
            messagebox.showerror("Simulation error", str(exc))
            self.status.config(text=str(exc), foreground="red")


def main() -> None:
    root = tk.Tk()
    TripleSwarmSimulatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
