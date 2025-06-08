"""Grid-based A* planner implementation."""

from __future__ import annotations

import heapq
from typing import Any, Dict, List, Tuple

import numpy as np

from .base_planner import PathPlanner


class AStarPlanner(PathPlanner):
    """A* path planner for 2D occupancy grids."""

    def __init__(self):
        pass

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return float(np.hypot(b[0] - a[0], b[1] - a[1]))

    def _neighbors(self, node: Tuple[int, int], grid: np.ndarray) -> List[Tuple[int, int]]:
        rows, cols = grid.shape
        dirs = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
        result = []
        for dr, dc in dirs:
            r, c = node[0] + dr, node[1] + dc
            if 0 <= r < rows and 0 <= c < cols and grid[r, c] == 0:
                result.append((r, c))
        return result

    def plan(
        self, start: Tuple[int, int], goal: Tuple[int, int], map_data: Any
    ) -> List[Tuple[int, int]]:
        grid = np.asarray(map_data)
        open_set = []
        heapq.heappush(open_set, (0 + self._heuristic(start, goal), 0, start))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        closed = set()

        while open_set:
            _, cost, current = heapq.heappop(open_set)
            if current in closed:
                continue
            if current == goal:
                # reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            closed.add(current)
            for neighbor in self._neighbors(current, grid):
                tentative_g = g_score[current] + self._heuristic(current, neighbor)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, tentative_g, neighbor))
                    came_from[neighbor] = current
        return []

