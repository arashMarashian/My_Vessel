"""Abstract interface for path planners."""

from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Any, List


class PathPlanner(ABC):
    """Abstract base class for path planners."""

    @abstractmethod
    def plan(
        self, start: Tuple[int, int], goal: Tuple[int, int], map_data: Any
    ) -> List[Tuple[int, int]]:
        """Plan a path from start to goal on the given map.

        Parameters
        ----------
        start : Tuple[int, int]
            Start position as (row, col).
        goal : Tuple[int, int]
            Goal position as (row, col).
        map_data : Any
            Map representation used by the planner. Typically a 2D occupancy grid.

        Returns
        -------
        List[Tuple[int, int]]
            Sequence of grid coordinates from start to goal inclusive. Empty if
            no path could be found.
        """
        pass

