from __future__ import annotations

import math

import numpy as np


class Node:
    def __init__(
        self,
        x: float,
        y: float,
        initial_phase: float,
        frequency_offset: float,
        node_id: int,
    ) -> None:
        self.node_id = node_id
        self.x = float(x)
        self.y = float(y)
        self.initial_phase = float(initial_phase)
        self.frequency_offset = float(frequency_offset)

    def get_position(self) -> tuple[float, float]:
        return (self.x, self.y)

    def move(self, x: float, y: float) -> None:
        self.x = float(x)
        self.y = float(y)

    def move_by(self, dx: float, dy: float) -> None:
        self.x += float(dx)
        self.y += float(dy)

    def distance_to(self, other: Node) -> float:
        return math.hypot(other.x - self.x, other.y - self.y)


class Swarm:
    DEFAULT_L_ARM_SPACING_M = 0.065
    L_SHAPE_ANGLE_TOL_DEG = 5.0
    L_SHAPE_LENGTH_TOL_REL = 0.05

    def __init__(self, nodes: list[Node]) -> None:
        if len(nodes) < 3:
            raise ValueError("A swarm must contain at least 3 nodes")
        self.nodes = list(nodes)
        self.nodes_by_id = {node.node_id: node for node in self.nodes}

    def get_node(self, node_id: int) -> Node:
        return self.nodes_by_id[node_id]

    def other_nodes(self, node_id: int) -> list[Node]:
        return [node for node in self.nodes if node.node_id != node_id]

    @staticmethod
    def is_l_shaped(
        node_a: Node,
        node_b: Node,
        node_c: Node,
        arm_spacing: float = DEFAULT_L_ARM_SPACING_M,
        angle_tol_deg: float = L_SHAPE_ANGLE_TOL_DEG,
        length_tol_rel: float = L_SHAPE_LENGTH_TOL_REL,
    ) -> bool:
        nodes = (node_a, node_b, node_c)
        length_tol = arm_spacing * length_tol_rel
        angle_tol = math.radians(angle_tol_deg)

        for corner in nodes:
            legs = [n for n in nodes if n is not corner]
            leg_a, leg_b = legs[0], legs[1]
            va_x = leg_a.x - corner.x
            va_y = leg_a.y - corner.y
            vb_x = leg_b.x - corner.x
            vb_y = leg_b.y - corner.y
            len_a = math.hypot(va_x, va_y)
            len_b = math.hypot(vb_x, vb_y)
            if len_a < 1e-9 or len_b < 1e-9:
                continue

            cos_angle = (va_x * vb_x + va_y * vb_y) / (len_a * len_b)
            cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
            angle = math.acos(cos_angle)
            if abs(angle - math.pi / 2.0) > angle_tol:
                continue
            if abs(len_a - arm_spacing) > length_tol:
                continue
            if abs(len_b - arm_spacing) > length_tol:
                continue
            return True
        return False

    @staticmethod
    def form_l_shape(
        node_a: Node,
        node_b: Node,
        node_c: Node,
        arm_spacing: float = DEFAULT_L_ARM_SPACING_M,
    ) -> None:
        nodes = [node_a, node_b, node_c]
        cx = sum(node.x for node in nodes) / 3.0
        cy = sum(node.y for node in nodes) / 3.0
        d = float(arm_spacing)
        targets = (
            (cx - d / 3.0, cy - d / 3.0),
            (cx + 2.0 * d / 3.0, cy - d / 3.0),
            (cx - d / 3.0, cy + 2.0 * d / 3.0),
        )

        best_perm = min(
            (
                (0, 1, 2),
                (0, 2, 1),
                (1, 0, 2),
                (1, 2, 0),
                (2, 0, 1),
                (2, 1, 0),
            ),
            key=lambda perm: sum(
                math.hypot(
                    nodes[i].x - targets[perm[i]][0],
                    nodes[i].y - targets[perm[i]][1],
                )
                for i in range(3)
            ),
        )

        for node, target_idx in zip(nodes, best_perm):
            tx, ty = targets[target_idx]
            node.move(tx, ty)

    def ensure_l_shape(
        self,
        arm_spacing: float = DEFAULT_L_ARM_SPACING_M,
        angle_tol_deg: float = L_SHAPE_ANGLE_TOL_DEG,
        length_tol_rel: float = L_SHAPE_LENGTH_TOL_REL,
    ) -> bool:
        if len(self.nodes) != 3:
            raise ValueError("L-shape formation requires exactly 3 nodes in the swarm")
        a, b, c = self.nodes[0], self.nodes[1], self.nodes[2]
        if self.is_l_shaped(a, b, c, arm_spacing, angle_tol_deg, length_tol_rel):
            return False
        self.form_l_shape(a, b, c, arm_spacing)
        return True


L_ARM_SPACING_M = Swarm.DEFAULT_L_ARM_SPACING_M


def make_random_node(x: float, y: float, node_id: int) -> Node:
    return Node(
        x,
        y,
        initial_phase=np.random.uniform(0, 2 * np.pi),
        frequency_offset=np.random.uniform(-10000, 10000),
        node_id=node_id,
    )


def swarm_centroid(swarm: Swarm) -> tuple[float, float]:
    xs = [node.x for node in swarm.nodes]
    ys = [node.y for node in swarm.nodes]
    n = len(swarm.nodes)
    return (sum(xs) / n, sum(ys) / n)
