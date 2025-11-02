"""
greedy_heuristic.py
Nearest-Neighbor (NN) TSP using src/problem_definition.py
"""

from __future__ import annotations
import math
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Tuple
from src import problem_definition as pd
from typing import List, Optional

# https://github.com/mahmoud-mohsen97/Travelling-salesman-problem-using-some-Random-Search-Algorithms.git
# --- Nearest Neighbor core
def get_nearest_neighbor(city: int, matrix: np.ndarray, visited: set[int]) -> Tuple[int, int]:
    n = matrix.shape[0]
    min_distance = 10**18
    nearest_neighbor = city
    for i in range(n):
        if i == city or i in visited:
            continue
        d = int(matrix[city, i])
        if d < min_distance:
            min_distance = d
            nearest_neighbor = i
    return min_distance, nearest_neighbor


def nearest_neighbor_path(distance_matrix: np.ndarray, start: int = 0) -> Tuple[List[int], int]:
    """
    Builds a NN path and returns (path, total_cost_including_return_to_start).
    """
    n = distance_matrix.shape[0]
    visited: set[int] = set([start])
    path: List[int] = [start]
    cur = start
    total_cost = 0

    while len(visited) < n:
        step_cost, nxt = get_nearest_neighbor(cur, distance_matrix, visited)
        visited.add(nxt)
        total_cost += step_cost
        path.append(nxt)
        cur = nxt

    # close tour
    total_cost += int(distance_matrix[cur, start])
    return path, total_cost

def visualize_tour(
    coords: np.ndarray,
    path: List[int],
    dmatrix: Optional[np.ndarray] = None,
    show_labels: bool = True
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))

    total_cost = 0

    n = len(path)
    for i in range(n):
        city1 = path[i]
        city2 = path[(i + 1) % n]  # closes the tour
        x1, y1 = coords[city1]
        x2, y2 = coords[city2]

        # edge weight (prefer provided distance matrix)
        if dmatrix is not None:
            w = int(dmatrix[city1, city2])
        else:
            # fallback: Euclidean rounded to nearest int
            w = int(round(math.hypot(x2 - x1, y2 - y1)))

        total_cost += w

        # draw edge
        ax.plot([int(x1), int(x2)], [int(y1), int(y2)],
                color='orange', linewidth=2, zorder=3)

        # edge-label at midpoint
        if show_labels:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            ax.text(int(mid_x), int(mid_y), str(w),
                    fontsize=7, color='darkgreen')

    # draw cities + ids
    ax.scatter(coords[:, 0], coords[:, 1], color='darkgreen', zorder=4)
    for idx, (px, py) in enumerate(coords):
        ax.text(int(px) + 1, int(py) + 1, f"{idx}", fontsize=10, color='black')

    ax.set_title(f'Best TSP Tour Path by GH Nearest Neighbor (Total Cost: {total_cost})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def gh_runner(all_coordinates: np.ndarray, k: int, n_cities: int = 30, start: int = 0) -> None:
    """
    Generates cities and distance matrix via src/problem_definition.py,
    then solves TSP with Nearest Neighbor.
    """
    coordinates_k = all_coordinates[:k]
    distance_matrix = pd.generate_distance_matrix(coordinates_k)
    distance_matrix = np.rint(distance_matrix).astype(int)

    start_time = time.perf_counter()
    path, total_cost = nearest_neighbor_path(distance_matrix, start=start)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    visualize_tour(coordinates_k, path)
    return elapsed_time, total_cost, path
