"""Placeholder for Dubins path planner."""

from typing import Any, List, Tuple

from .base_planner import PathPlanner


class DubinsPlanner(PathPlanner):
    """Curvature-constrained planner (not implemented)."""

    def plan(
        self, start: Tuple[int, int], goal: Tuple[int, int], map_data: Any
    ) -> List[Tuple[int, int]]:
        raise NotImplementedError("Dubins planner not implemented yet")

