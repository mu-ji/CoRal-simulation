from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

from aoa import get_swarm_rx_ids
from communication import set_simulation_noise
from node import L_ARM_SPACING_M, Node, Swarm, make_random_node, swarm_centroid
from operation import aoa_estimation, multiple_aoa_estimation

DEFAULT_T1_MS = 10.0
DEFAULT_T2_MS = 10.15
DEFAULT_SLOT_SPACING_S = 0.5
DEFAULT_PROPAGATION_SNR_DB = 30.0
DEFAULT_RX_SNR_DB = 30.0
DEFAULT_SAMPLE_PHASE_SNR_DB = 35.0
DEFAULT_ZOOM_SPAN_M = 0.5
DEFAULT_RANDOM_SEED = 20010127

SWARM_MARKERS = ("o", "s", "^", "D", "v", "P")
SWARM_COLORS = ("C0", "C1", "C2", "C3", "C4", "C5")


class ScenarioState:
    def __init__(self) -> None:
        self.nodes: dict[int, Node] = {}
        self.swarms: dict[str, Swarm] = {}
        self.next_node_id = 0
        self.last_single_aoa: dict | None = None
        self.last_multiple_aoa: dict | None = None

    def add_node(self, node: Node) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"Node id {node.node_id} already exists")
        self.nodes[node.node_id] = node
        self.next_node_id = max(self.next_node_id, node.node_id + 1)

    def node_ids(self) -> list[str]:
        return [str(nid) for nid in sorted(self.nodes)]

    def swarm_names(self) -> list[str]:
        return list(self.swarms.keys())

    def get_node(self, node_id: int) -> Node:
        return self.nodes[node_id]

    def nodes_in_swarm(self, name: str) -> list[Node]:
        return list(self.swarms[name].nodes)

    def unassigned_node_ids(self) -> list[str]:
        assigned = {n.node_id for swarm in self.swarms.values() for n in swarm.nodes}
        return [str(nid) for nid in sorted(self.nodes) if nid not in assigned]


class VisualizationApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("CoRal GUI Visualization")
        self.root.geometry("1400x820")
        self.root.minsize(1100, 700)

        self.state = ScenarioState()
        self.vars: dict[str, tk.StringVar] = {}

        paned = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left_outer = ttk.Frame(paned, width=380)
        right_panel = ttk.Frame(paned)
        paned.add(left_outer, weight=1)
        paned.add(right_panel, weight=3)

        self._build_control_panel(left_outer)
        self._build_visual_panel(right_panel)
        self._load_demo_scenario()

    def _build_control_panel(self, parent: ttk.Frame) -> None:
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        scroll = ttk.Frame(canvas, padding=10)
        scroll.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=scroll, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(scroll, text="Operation panel", font=("", 11, "bold")).pack(
            anchor=tk.W, pady=(0, 8)
        )

        self._section(scroll, "Create node")
        row = ttk.Frame(scroll)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="x (m)", width=8).pack(side=tk.LEFT)
        self._var("new_node_x", "0.0")
        ttk.Entry(row, textvariable=self.vars["new_node_x"], width=10).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Label(row, text="y", width=3).pack(side=tk.LEFT)
        self._var("new_node_y", "0.0")
        ttk.Entry(row, textvariable=self.vars["new_node_y"], width=10).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(scroll, text="Add node", command=self._on_add_node).pack(
            fill=tk.X, pady=4
        )

        self._section(scroll, "Move node")
        self._node_combo(scroll, "move_node_id")
        row = ttk.Frame(scroll)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="New x", width=8).pack(side=tk.LEFT)
        self._var("move_x", "0.0")
        ttk.Entry(row, textvariable=self.vars["move_x"], width=10).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Label(row, text="y", width=3).pack(side=tk.LEFT)
        self._var("move_y", "0.0")
        ttk.Entry(row, textvariable=self.vars["move_y"], width=10).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(scroll, text="Move node", command=self._on_move_node).pack(
            fill=tk.X, pady=4
        )

        self._section(scroll, "Create swarm (3 nodes)")
        self._var("swarm_name", "Swarm A")
        row = ttk.Frame(scroll)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Name", width=8).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.vars["swarm_name"], width=18).pack(
            side=tk.LEFT, padx=3
        )
        for key, label in (
            ("swarm_n1", "Node 1"),
            ("swarm_n2", "Node 2"),
            ("swarm_n3", "Node 3"),
        ):
            self._node_combo(scroll, key, label)
        self._var("swarm_l_shape", "1")
        ttk.Checkbutton(
            scroll,
            text="Form L-shape after create",
            variable=self.vars["swarm_l_shape"],
            onvalue="1",
            offvalue="0",
        ).pack(anchor=tk.W, pady=2)
        ttk.Button(scroll, text="Create swarm", command=self._on_create_swarm).pack(
            fill=tk.X, pady=4
        )

        self._section(scroll, "Form L-shape")
        self._swarm_combo(scroll, "l_shape_swarm", "Swarm")
        row = ttk.Frame(scroll)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Arm spacing (m)", width=14).pack(side=tk.LEFT)
        self._var("l_arm_spacing", f"{L_ARM_SPACING_M:.4f}")
        ttk.Entry(row, textvariable=self.vars["l_arm_spacing"], width=10).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(scroll, text="Form L-shape", command=self._on_form_l_shape).pack(
            fill=tk.X, pady=4
        )

        self._section(scroll, "Single AoA estimation")
        for key, label in (
            ("aoa_tx", "TX"),
            ("aoa_rx_a", "RX A (R/U)"),
            ("aoa_rx_b", "RX B (L/D)"),
            ("aoa_remote", "Remote"),
        ):
            self._node_combo(scroll, key, label)
        ttk.Button(
            scroll,
            text="Auto-select RX (right/upper, left/lower)",
            command=self._on_auto_select_rx,
        ).pack(fill=tk.X, pady=2)
        self._timing_rows(scroll)
        ttk.Button(
            scroll, text="Run AoA estimation", command=self._on_single_aoa
        ).pack(fill=tk.X, pady=4)

        self._section(scroll, "Swarm AoA estimation")
        self._swarm_combo(scroll, "multi_swarm1", "Swarm 1 (measuring)")
        self._swarm_combo(scroll, "multi_swarm2", "Swarm 2 (target)")
        row = ttk.Frame(scroll)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Slot spacing (s)", width=14).pack(side=tk.LEFT)
        self._var("slot_spacing", str(DEFAULT_SLOT_SPACING_S))
        ttk.Entry(row, textvariable=self.vars["slot_spacing"], width=10).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(
            scroll, text="Run swarm AoA", command=self._on_multiple_aoa
        ).pack(fill=tk.X, pady=4)

        self._section(scroll, "Noise & seed")
        for label, key, default in (
            ("Propagation SNR", "prop_snr", DEFAULT_PROPAGATION_SNR_DB),
            ("RX SNR", "rx_snr", DEFAULT_RX_SNR_DB),
            ("Phase SNR", "phase_snr", DEFAULT_SAMPLE_PHASE_SNR_DB),
            ("Random seed", "seed", DEFAULT_RANDOM_SEED),
        ):
            row = ttk.Frame(scroll)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=14).pack(side=tk.LEFT)
            self._var(key, str(default))
            ttk.Entry(row, textvariable=self.vars[key], width=10).pack(
                side=tk.LEFT, padx=3
            )

        ttk.Separator(scroll).pack(fill=tk.X, pady=8)
        ttk.Button(scroll, text="Clear AoA overlays", command=self._clear_aoa).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(scroll, text="Reset scenario", command=self._reset_scenario).pack(
            fill=tk.X, pady=2
        )

        self.status = ttk.Label(
            scroll, text="Ready", foreground="gray", wraplength=340
        )
        self.status.pack(anchor=tk.W, pady=(10, 0))

    def _build_visual_panel(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent)
        header.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Label(header, text="Map visualization", font=("", 10, "bold")).pack(
            side=tk.LEFT
        )

        zoom_bar = ttk.Frame(parent)
        zoom_bar.pack(fill=tk.X, padx=8, pady=(0, 4))

        self._var("zoom_span", str(DEFAULT_ZOOM_SPAN_M))
        self._var("zoom_node_id", "")
        ttk.Label(zoom_bar, text="Local zoom span (m)").pack(side=tk.LEFT)
        ttk.Entry(zoom_bar, textvariable=self.vars["zoom_span"], width=8).pack(
            side=tk.LEFT, padx=(4, 10)
        )
        ttk.Label(zoom_bar, text="Center on node").pack(side=tk.LEFT)
        self._zoom_node_combo = ttk.Combobox(
            zoom_bar,
            textvariable=self.vars["zoom_node_id"],
            state="readonly",
            width=8,
        )
        self._zoom_node_combo.pack(side=tk.LEFT, padx=4)
        ttk.Button(zoom_bar, text="Zoom to node", command=self._zoom_to_node).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(zoom_bar, text="Reset view", command=self._reset_view).pack(
            side=tk.LEFT, padx=4
        )
        self._var("rect_zoom", "0")
        ttk.Checkbutton(
            zoom_bar,
            text="Box zoom (drag on map)",
            variable=self.vars["rect_zoom"],
            onvalue="1",
            offvalue="0",
            command=self._update_rect_zoom_mode,
        ).pack(side=tk.LEFT, padx=(12, 0))

        self._auto_fit_view = True
        self._default_xlim = (0.0, 1.0)
        self._default_ylim = (0.0, 1.0)
        self._rect_selector: RectangleSelector | None = None

        self.fig = Figure(figsize=(9, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))

        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        self.canvas.mpl_connect("draw_event", self._on_canvas_draw)

    def _section(self, parent: ttk.Frame, title: str) -> None:
        ttk.Separator(parent).pack(fill=tk.X, pady=(10, 6))
        ttk.Label(parent, text=title, font=("", 10, "bold")).pack(
            anchor=tk.W, pady=(0, 4)
        )

    def _var(self, key: str, default: str) -> None:
        self.vars[key] = tk.StringVar(value=default)

    def _node_combo(
        self, parent: ttk.Frame, key: str, label: str = "Node"
    ) -> ttk.Combobox:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=10).pack(side=tk.LEFT)
        self._var(key, "")
        combo = ttk.Combobox(
            row, textvariable=self.vars[key], state="readonly", width=16
        )
        combo.pack(side=tk.LEFT, padx=3)
        setattr(self, f"_{key}_combo", combo)
        return combo

    def _swarm_combo(
        self, parent: ttk.Frame, key: str, label: str
    ) -> ttk.Combobox:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=16).pack(side=tk.LEFT)
        self._var(key, "")
        combo = ttk.Combobox(
            row, textvariable=self.vars[key], state="readonly", width=14
        )
        combo.pack(side=tk.LEFT, padx=3)
        setattr(self, f"_{key}_combo", combo)
        return combo

    def _timing_rows(self, parent: ttk.Frame) -> None:
        for label, key, default in (
            ("t1 (ms)", "t1_ms", DEFAULT_T1_MS),
            ("t2 (ms)", "t2_ms", DEFAULT_T2_MS),
        ):
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=10).pack(side=tk.LEFT)
            self._var(key, str(default))
            ttk.Entry(row, textvariable=self.vars[key], width=12).pack(
                side=tk.LEFT, padx=3
            )

    def _refresh_combos(self) -> None:
        ids = self.state.node_ids()
        swarms = self.state.swarm_names()
        for key in (
            "move_node_id",
            "swarm_n1",
            "swarm_n2",
            "swarm_n3",
            "aoa_tx",
            "aoa_rx_a",
            "aoa_rx_b",
            "aoa_remote",
            "zoom_node_id",
        ):
            combo = getattr(self, f"_{key}_combo", None)
            if combo is not None:
                combo["values"] = ids
                if ids and self.vars[key].get() not in ids:
                    self.vars[key].set(ids[0])
        if hasattr(self, "_zoom_node_combo"):
            self._zoom_node_combo["values"] = ids
            if ids and self.vars["zoom_node_id"].get() not in ids:
                self.vars["zoom_node_id"].set(ids[0])
        for key in ("multi_swarm1", "multi_swarm2", "l_shape_swarm"):
            combo = getattr(self, f"_{key}_combo", None)
            if combo is not None:
                combo["values"] = swarms
                if swarms and self.vars[key].get() not in swarms:
                    self.vars[key].set(swarms[0])

    def _on_canvas_draw(self, _event) -> None:
        if self.toolbar.mode != "":
            self._auto_fit_view = False

    def _update_rect_zoom_mode(self) -> None:
        if self.vars["rect_zoom"].get() == "1":
            self._attach_rect_selector()
        elif self._rect_selector is not None:
            self._rect_selector.set_active(False)

    def _attach_rect_selector(self) -> None:
        if self._rect_selector is not None:
            self._rect_selector.set_active(True)
            return

        def on_select(eclick, erelease) -> None:
            if self.vars["rect_zoom"].get() != "1":
                return
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            if x1 is None or y1 is None or x2 is None or y2 is None:
                return
            if abs(x2 - x1) < 1e-6 or abs(y2 - y1) < 1e-6:
                return
            self.ax.set_xlim(min(x1, x2), max(x1, x2))
            self.ax.set_ylim(min(y1, y2), max(y1, y2))
            self._auto_fit_view = False
            self.canvas.draw_idle()

        self._rect_selector = RectangleSelector(
            self.ax,
            on_select,
            useblit=True,
            button=[1],
            minspanx=0.005,
            minspany=0.005,
            spancoords="data",
            interactive=False,
            props=dict(facecolor="C0", edgecolor="C0", alpha=0.25, fill=True),
        )

    def _reset_view(self) -> None:
        self._auto_fit_view = True
        self.ax.set_xlim(self._default_xlim)
        self.ax.set_ylim(self._default_ylim)
        self.canvas.draw_idle()

    def _zoom_to_node(self) -> None:
        try:
            node = self._read_node("zoom_node_id")
            span = float(self.vars["zoom_span"].get())
            if span <= 0.0:
                raise ValueError("Zoom span must be positive")
            half = span / 2.0
            self.ax.set_xlim(node.x - half, node.x + half)
            self.ax.set_ylim(node.y - half, node.y + half)
            self._auto_fit_view = False
            self.canvas.draw_idle()
            self.status.config(
                text=f"Zoomed to node {node.node_id} (span {span:.3f} m)",
                foreground="green",
            )
        except Exception as exc:
            messagebox.showerror("Zoom error", str(exc))
            self.status.config(text=str(exc), foreground="red")

    def _apply_noise(self) -> None:
        np.random.seed(int(self.vars["seed"].get()))
        set_simulation_noise(
            propagation_snr_db=float(self.vars["prop_snr"].get()),
            rx_snr_db=float(self.vars["rx_snr"].get()),
            sample_phase_snr_db=float(self.vars["phase_snr"].get()),
        )

    def _read_node(self, key: str) -> Node:
        node_id = int(self.vars[key].get())
        return self.state.get_node(node_id)

    def _on_add_node(self) -> None:
        try:
            x = float(self.vars["new_node_x"].get())
            y = float(self.vars["new_node_y"].get())
            node = make_random_node(x, y, self.state.next_node_id)
            self.state.add_node(node)
            self._refresh_combos()
            self._redraw()
            self.status.config(
                text=f"Added node {node.node_id} at ({x:.3f}, {y:.3f})",
                foreground="green",
            )
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            self.status.config(text=str(exc), foreground="red")

    def _on_move_node(self) -> None:
        try:
            node = self._read_node("move_node_id")
            x = float(self.vars["move_x"].get())
            y = float(self.vars["move_y"].get())
            node.move(x, y)
            self.vars["move_x"].set(f"{x:.4f}")
            self.vars["move_y"].set(f"{y:.4f}")
            self._redraw()
            self.status.config(
                text=f"Moved node {node.node_id} to ({x:.3f}, {y:.3f})",
                foreground="green",
            )
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            self.status.config(text=str(exc), foreground="red")

    def _on_create_swarm(self) -> None:
        try:
            name = self.vars["swarm_name"].get().strip()
            if not name:
                raise ValueError("Swarm name is required")
            if name in self.state.swarms:
                raise ValueError(f"Swarm '{name}' already exists")
            nodes = [
                self._read_node("swarm_n1"),
                self._read_node("swarm_n2"),
                self._read_node("swarm_n3"),
            ]
            if len({n.node_id for n in nodes}) != 3:
                raise ValueError("Swarm requires 3 distinct nodes")
            for swarm in self.state.swarms.values():
                swarm_ids = {n.node_id for n in swarm.nodes}
                if any(n.node_id in swarm_ids for n in nodes):
                    raise ValueError("A node can belong to only one swarm")

            swarm = Swarm(nodes)
            if self.vars["swarm_l_shape"].get() == "1":
                swarm.ensure_l_shape()
            self.state.swarms[name] = swarm
            self._refresh_combos()
            self._redraw()
            self.status.config(text=f"Created swarm '{name}'", foreground="green")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            self.status.config(text=str(exc), foreground="red")

    def _on_form_l_shape(self) -> None:
        try:
            name = self.vars["l_shape_swarm"].get().strip()
            if not name or name not in self.state.swarms:
                raise ValueError("Select an existing swarm")
            swarm = self.state.swarms[name]
            if len(swarm.nodes) != 3:
                raise ValueError("L-shape requires exactly 3 nodes in the swarm")

            arm_spacing = float(self.vars["l_arm_spacing"].get())
            if arm_spacing <= 0.0:
                raise ValueError("Arm spacing must be positive")

            a, b, c = swarm.nodes[0], swarm.nodes[1], swarm.nodes[2]
            was_l = Swarm.is_l_shaped(a, b, c, arm_spacing=arm_spacing)
            moved = swarm.ensure_l_shape(arm_spacing=arm_spacing)
            self._redraw()
            if was_l:
                self.status.config(
                    text=f"Swarm '{name}' was already L-shaped",
                    foreground="green",
                )
            elif moved:
                self.status.config(
                    text=f"Swarm '{name}' reformed to L-shape",
                    foreground="green",
                )
            else:
                self.status.config(
                    text=f"Swarm '{name}' L-shape unchanged",
                    foreground="green",
                )
        except Exception as exc:
            messagebox.showerror("L-shape error", str(exc))
            self.status.config(text=str(exc), foreground="red")

    def _swarm_containing_node(self, node_id: int) -> Swarm | None:
        for swarm in self.state.swarms.values():
            if node_id in {n.node_id for n in swarm.nodes}:
                return swarm
        return None

    def _on_auto_select_rx(self) -> None:
        try:
            tx_id = int(self.vars["aoa_tx"].get())
            swarm = self._swarm_containing_node(tx_id)
            if swarm is None:
                raise ValueError("TX must belong to a swarm")
            rx_a_id, rx_b_id = get_swarm_rx_ids(swarm, tx_id)
            self.vars["aoa_rx_a"].set(str(rx_a_id))
            self.vars["aoa_rx_b"].set(str(rx_b_id))
            self.status.config(
                text=f"RX set: {rx_a_id}=right/upper, {rx_b_id}=left/lower",
                foreground="green",
            )
        except Exception as exc:
            messagebox.showerror("RX selection error", str(exc))
            self.status.config(text=str(exc), foreground="red")

    def _on_single_aoa(self) -> None:
        try:
            self._apply_noise()
            t1 = float(self.vars["t1_ms"].get()) * 1e-3
            t2 = float(self.vars["t2_ms"].get()) * 1e-3
            tx = self._read_node("aoa_tx")
            rx_a = self._read_node("aoa_rx_a")
            rx_b = self._read_node("aoa_rx_b")
            remote = self._read_node("aoa_remote")
            result = aoa_estimation(tx, rx_a, rx_b, remote, t1, t2)
            self.state.last_single_aoa = result
            self.state.last_multiple_aoa = None
            self._redraw()
            self.status.config(
                text=(
                    f"AoA = {result['aoa_deg']:.2f} deg, "
                    f"true = {result['true_aoa_deg']:.2f} deg, "
                    f"error = {result['aoa_error_deg']:.2f} deg"
                ),
                foreground="green",
            )
        except Exception as exc:
            messagebox.showerror("AoA error", str(exc))
            self.status.config(text=str(exc), foreground="red")

    def _on_multiple_aoa(self) -> None:
        try:
            self._apply_noise()
            name1 = self.vars["multi_swarm1"].get()
            name2 = self.vars["multi_swarm2"].get()
            if name1 == name2:
                raise ValueError("Select two different swarms")
            s1 = self.state.swarms[name1]
            s2 = self.state.swarms[name2]
            if len(s1.nodes) != 3 or len(s2.nodes) != 3:
                raise ValueError("Each swarm must have exactly 3 nodes")

            t1 = float(self.vars["t1_ms"].get()) * 1e-3
            t2 = float(self.vars["t2_ms"].get()) * 1e-3
            spacing = float(self.vars["slot_spacing"].get())
            result = multiple_aoa_estimation(
                tuple(s1.nodes),
                tuple(s2.nodes),
                t1,
                t2,
                slot_spacing_s=spacing,
            )
            self.state.last_multiple_aoa = result
            self.state.last_single_aoa = None
            self._redraw()
            self.status.config(
                text=(
                    f"Swarm2 rel. angle = {result['relative_angle_deg']:.2f} deg, "
                    f"true = {result['true_relative_angle_deg']:.2f} deg, "
                    f"error = {result['relative_angle_error_deg']:.2f} deg "
                    f"({result['num_slots']} slots)"
                ),
                foreground="green",
            )
        except Exception as exc:
            messagebox.showerror("Swarm AoA error", str(exc))
            self.status.config(text=str(exc), foreground="red")

    def _clear_aoa(self) -> None:
        self.state.last_single_aoa = None
        self.state.last_multiple_aoa = None
        self._redraw()
        self.status.config(text="AoA overlays cleared", foreground="gray")

    def _reset_scenario(self) -> None:
        self.state = ScenarioState()
        self._load_demo_scenario()

    def _load_demo_scenario(self) -> None:
        self.state = ScenarioState()
        for i, (x, y) in enumerate(
            ((0.0, 0.0), (0.065, 0.0), (0.0, 0.065), (4.0, 2.0), (4.065, 2.0), (4.0, 2.065))
        ):
            self.state.add_node(make_random_node(x, y, i))
        self._apply_noise()
        swarm_a = Swarm(
            [self.state.get_node(i) for i in (0, 1, 2)]
        )
        swarm_a.ensure_l_shape()
        swarm_b = Swarm(
            [self.state.get_node(i) for i in (3, 4, 5)]
        )
        swarm_b.ensure_l_shape()
        self.state.swarms["Swarm A"] = swarm_a
        self.state.swarms["Swarm B"] = swarm_b
        self.vars["swarm_name"].set("Swarm C")
        self.vars["multi_swarm1"].set("Swarm A")
        self.vars["multi_swarm2"].set("Swarm B")
        self.vars["l_shape_swarm"].set("Swarm A")
        self._refresh_combos()
        if self.state.node_ids():
            for key in ("move_node_id", "aoa_tx", "aoa_rx_a", "aoa_rx_b"):
                self.vars[key].set("1")
            self.vars["aoa_remote"].set("3")
            self.vars["swarm_n1"].set("0")
            self.vars["swarm_n2"].set("1")
            self.vars["swarm_n3"].set("2")
        self._redraw()
        self.status.config(text="Demo scenario loaded (6 nodes, 2 swarms)", foreground="green")

    def _redraw(self) -> None:
        ax = self.ax
        saved_xlim = ax.get_xlim()
        saved_ylim = ax.get_ylim()
        ax.clear()

        assigned: dict[int, tuple[str, int]] = {}
        for si, (name, swarm) in enumerate(self.state.swarms.items()):
            for node in swarm.nodes:
                assigned[node.node_id] = (name, si)

        for node_id, node in sorted(self.state.nodes.items()):
            if node_id in assigned:
                name, si = assigned[node_id]
                color = SWARM_COLORS[si % len(SWARM_COLORS)]
                marker = SWARM_MARKERS[si % len(SWARM_MARKERS)]
                ax.scatter(node.x, node.y, s=90, marker=marker, color=color, zorder=4)
                ax.text(node.x, node.y, f" {node_id}", fontsize=9)
            else:
                ax.scatter(node.x, node.y, s=70, marker="x", color="gray", zorder=3)
                ax.text(node.x, node.y, f" {node_id}", fontsize=9, color="gray")

        for si, (name, swarm) in enumerate(self.state.swarms.items()):
            cx, cy = swarm_centroid(swarm)
            color = SWARM_COLORS[si % len(SWARM_COLORS)]
            ax.scatter([cx], [cy], s=120, marker="+", color=color, zorder=5)
            ax.text(cx, cy, f" {name}", fontsize=8, color=color)

        single = self.state.last_single_aoa
        if single is not None:
            rx_a = single["rx_a"]
            rx_b = single["rx_b"]
            remote = single["remote"]
            mid_x = (rx_a.x + rx_b.x) / 2.0
            mid_y = (rx_a.y + rx_b.y) / 2.0
            ray_len = 1.5
            ax.plot(
                [rx_b.x, rx_a.x], [rx_b.y, rx_a.y], color="gray", linewidth=1.2, label="RX baseline"
            )
            ax.plot(
                [mid_x, mid_x + ray_len * np.cos(single["true_bearing_rad"])],
                [mid_y, mid_y + ray_len * np.sin(single["true_bearing_rad"])],
                color="C2",
                linewidth=2,
                label="True bearing",
            )
            ax.plot(
                [mid_x, mid_x + ray_len * np.cos(single["bearing_rad"])],
                [mid_y, mid_y + ray_len * np.sin(single["bearing_rad"])],
                color="C3",
                linestyle="--",
                linewidth=2,
                label="Est. bearing",
            )
            ax.scatter([remote.x], [remote.y], s=100, facecolors="none", edgecolors="C3", linewidths=2)

        multi = self.state.last_multiple_aoa
        if multi is not None:
            c1 = multi["swarm1_centroid"]
            c2 = multi["swarm2_centroid"]
            ray_len = np.hypot(c2[0] - c1[0], c2[1] - c1[1]) * 0.9 or 1.0
            est = multi["relative_angle_rad"]
            truth = multi["true_relative_angle_rad"]
            ax.plot(
                [c1[0], c1[0] + ray_len * np.cos(truth)],
                [c1[1], c1[1] + ray_len * np.sin(truth)],
                color="C2",
                linewidth=2.5,
                label="True swarm2 direction",
            )
            ax.plot(
                [c1[0], c1[0] + ray_len * np.cos(est)],
                [c1[1], c1[1] + ray_len * np.sin(est)],
                color="C3",
                linestyle="--",
                linewidth=2.5,
                label="Est. swarm2 direction",
            )
            ax.plot([c1[0], c2[0]], [c1[1], c2[1]], color="gray", linestyle=":", linewidth=1)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Nodes, swarms and AoA overlays")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(fontsize=8, loc="upper left")

        if self.state.nodes:
            xs = [n.x for n in self.state.nodes.values()]
            ys = [n.y for n in self.state.nodes.values()]
            margin = 0.8
            self._default_xlim = (min(xs) - margin, max(xs) + margin)
            self._default_ylim = (min(ys) - margin, max(ys) + margin)
            if self._auto_fit_view:
                ax.set_xlim(self._default_xlim)
                ax.set_ylim(self._default_ylim)
            else:
                ax.set_xlim(saved_xlim)
                ax.set_ylim(saved_ylim)

        self._rect_selector = None
        if self.vars.get("rect_zoom") and self.vars["rect_zoom"].get() == "1":
            self._attach_rect_selector()

        self.canvas.draw_idle()


def main() -> None:
    root = tk.Tk()
    VisualizationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
