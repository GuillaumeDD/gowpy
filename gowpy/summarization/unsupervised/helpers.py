import math
from typing import Sequence, Tuple


def elbow(points: Sequence[Tuple[int, float]]) -> int:
    """Elbow method for a sequence of 2-D point (sorted by the ascending 'x' value)"""
    selected_x = None

    if len(points) <= 2:
        x_max = 0
        y_max = -1
        for x, y in points:
            if y > y_max:
                x_max = x
                y_max = y

        selected_x = x_max
    else:
        # Computation of the line
        # y = a*x + b
        x_0, y_0 = points[0]
        x_n, y_n = points[-1]
        a = (y_n - y_0) / (x_n - x_0)
        b = y_n - a * x_n

        def distance_from_line(x: int, y: float) -> float:
            return abs(y - a * x - b) / math.sqrt(1 + a**2)

        x_distance_max = x_0
        distance_max = 0.0
        for x, y in points:
            distance = distance_from_line(x, y)
            if distance > distance_max:
                x_distance_max = x
                distance_max = distance

        selected_x = x_distance_max

    return selected_x
