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

# https://github.com/mahmoud-mohsen97/Travelling-salesman-problem-using-some-Random-Search-Algorithms.git
# --- Distance helpers (kept for clarity; pd handles matrix creation) ---
def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    x1, y1 = p1
    x2, y2 = p2
    return math.hypot(x2 - x1, y2 - y1)


# --- Nearest Neighbor core
def get_nearest_neighbor(city: int, matrix: np.ndarray, visited: set[int]) -> Tuple[float, int]:
    n = matrix.shape[0]
    min_distance = float("inf")
    nearest_neighbor = city
    for i in range(n):
        if i == city or i in visited:
            continue
        d = float(matrix[city, i])
        if d < min_distance:
            min_distance = d
            nearest_neighbor = i
    return min_distance, nearest_neighbor


def nearest_neighbor_path(distance_matrix: np.ndarray, start: int = 0) -> Tuple[List[int], float]:
    """
    Builds a NN path and returns (path, total_cost_including_return_to_start).
    """
    n = distance_matrix.shape[0]
    visited: set[int] = set([start])
    path: List[int] = [start]
    cur = start
    total_cost = 0.0

    while len(visited) < n:
        step_cost, nxt = get_nearest_neighbor(cur, distance_matrix, visited)
        visited.add(nxt)
        total_cost += step_cost
        path.append(nxt)
        cur = nxt

    # close tour
    total_cost += float(distance_matrix[cur, start])
    return path, total_cost


# --- Plot (uses raw coordinates from pd) ---
def plot_path(coords: np.ndarray, path: List[int]) -> None:
    x = [coords[i, 0] for i in path] + [coords[path[0], 0]]
    y = [coords[i, 1] for i in path] + [coords[path[0], 1]]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(x, y, '-o', linewidth=2)
    for i, (px, py) in enumerate(coords):
        ax.text(px + 1, py + 1, f"{i}", fontsize=9)
    ax.set_title('Path by Nearest Neighbor')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def gh_runner(n_cities: int = 30, start: int = 0) -> None:
    """
    Generates cities and distance matrix via src/problem_definition.py,
    then solves TSP with Nearest Neighbor.
    """
    # From your problem definition (random generation lives there)

    start_time = time.perf_counter()

    coordinates: np.ndarray = pd.generate_coordinates(n_cities)

    distance_matrix: np.ndarray = pd.generate_distance_matrix(coordinates)

    end_time = time.perf_counter()

    elapsed_time = end_time - start_time

    path, total_cost = nearest_neighbor_path(distance_matrix, start=start)

    print("\n--- Nearest-Neighbor TSP ---")
    print(f"Number of cities: {n_cities}")
    print(f"Start city: {start}")
    print(f"Path (visit order): {path}")
    print(f"Total tour length (incl. return): {total_cost:.4f}")
    print(f"Execution Time: {elapsed_time:.6f} seconds")
    plot_path(coordinates, path)


if __name__ == "__main__":
    # Example run
    gh_runner(n_cities=30, start=0)
