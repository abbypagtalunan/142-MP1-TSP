import numpy as np
import matplotlib.pyplot as plt
import math

# STEP 1: GENERATE RANDOM COORDINATES
def generate_coordinates(n_cities=30, grid_size=100, seed=1):
    """
    Generates random (x, y) in range (-50, 50) coordinates for cities in 100x100 grid
    """
    np.random.seed(seed)
    coordinates = np.random.randint(-50, 50, size=(n_cities, 2))
    # print(f"\nGenerated {n_cities} cities in a {grid_size}x{grid_size} grid")
    # for i in range(n_cities):
    #     x, y = coordinates[i]
    #     print(f"  City {i}: x = {x:3d}, y = {y:3d}")   
    return coordinates

# STEP 2: CALCULATE EUCLIDEAN DISTANCE BETWEEN TWO CITIES
def calculate_euclidean_distance(city1, city2):
    """
    Calculates euclidean distance between two coordinate points.
    """
    x1, y1 = city1
    x2, y2 = city2
    dx = x1 - x2
    dy = y1 - y2
    distance = math.sqrt(dx**2 + dy**2)
    return distance

# STEP 3: GENERATE DISTANCE (COST) MATRIX
def generate_distance_matrix(coordinates):
    """
    Generates an nxn symmetric distance matrix rounded to the nearest integer.
    """
    n = len(coordinates)
    distance_matrix = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(i + 1, n):
            weight = round(calculate_euclidean_distance(coordinates[i], coordinates[j]))
            distance_matrix[i, j] = weight
            distance_matrix[j, i] = weight
    # print("\nDistance Matrix:")
    # print(distance_matrix)
    return distance_matrix

def visualize(coordinates, distance_matrix, k=None, show_labels=True):
    """
    Visualize cities and their weighted edges.
    """
    plt.figure(figsize=(8, 8))
    n = len(coordinates)

    # Plot and label edges
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[j]
            plt.plot([x1, x2], [y1, y2], 'lightgray', linewidth=0.5)
            if show_labels:
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                distance = distance_matrix[i, j]
                plt.text(mid_x, mid_y, str(distance), fontsize=7, color='darkgreen')

    # Plot and label city points
    plt.scatter(coordinates[:, 0], coordinates[:, 1], color='darkgreen', zorder=3)
    for i, (x, y) in enumerate(coordinates):
        plt.text(x + 1, y + 1, f"{i}", fontsize=10, color='black')
    
    plt.title(f"Visualization of {n} Cities and their Weighted Edges")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

def pd_runner(k, all_coordinates=None):
    """
    Uses the first k coordinates from a pregenerated n=30 cities.
    """
    if all_coordinates is None:
        all_coordinates = generate_coordinates(30)

    coordinates_k = all_coordinates[:k]
    distance_matrix = generate_distance_matrix(coordinates_k)
    # visualize(coordinates_k, distance_matrix, k=k)

    return all_coordinates