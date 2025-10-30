import math
import time
import matplotlib.pyplot as plt
from src import problem_definition as pd

#   EXHAUSTIVE SEARCH USING DECREASE-BY-ONE METHOD

# from Geeks4geeks
# Computes total cost of the given tour (including return to start).
def calculate_tour_cost(tour, distance_matrix):
    total_cost = 0
    for i in range(len(tour) - 1):
        total_cost += distance_matrix[tour[i]][tour[i + 1]]
    total_cost += distance_matrix[tour[-1]][tour[0]]  
    return total_cost

# Recursive function to generate all permutations using Decrease-by-One method
def generate_permutations(seq):
    if len(seq) == 1:
        return [seq]

    perms = []
    # Decrease by one: generate permutations of n-1 elements
    # Pick one element (start city) & generate permutations of the rest
    for i in range(len(seq)):
        remaining = seq[:i] + seq[i+1:]
        for p in generate_permutations(remaining):
            perms.append([seq[i]] + p)
    return perms

# Exhaustive TSP using Decrease-by-One permutation generation
def exhaustive_tsp_decrease_by_one(distance_matrix):
    n = len(distance_matrix)
    cities = list(range(n))
    start_city = cities[0]
    other_cities = cities[1:]

    best_tour = None
    best_cost = math.inf

    all_perms = generate_permutations(other_cities)

    for perm in all_perms:
        tour = [start_city] + perm
        cost = calculate_tour_cost(tour, distance_matrix)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour

    return best_tour, best_cost

# Visualization of the best tour
def visualize_tour(coordinates, distance_matrix, tour):
    plt.figure(figsize=(8, 8))
    n = len(coordinates)

    # Draw edges of best tour
    for i in range(len(tour)):
        city1 = tour[i]
        city2 = tour[(i + 1) % n]
        x1, y1 = coordinates[city1]
        x2, y2 = coordinates[city2]
        plt.plot([x1, x2], [y1, y2], 'orange', linewidth=2, zorder=2)

    # Plot city points
    plt.scatter(coordinates[:, 0], coordinates[:, 1], color='darkgreen', zorder=3)
    for i, (x, y) in enumerate(coordinates):
        plt.text(x + 1, y + 1, f"{i}", fontsize=10, color='black')

    plt.title(f"Best TSP Tour (Cost: {calculate_tour_cost(tour, distance_matrix)})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

# Runner function for Exhaustive Search
def es_runner(all_coordinates, k):
    # Use only the first k cities from the pregenerated set
    coordinates_k = all_coordinates[:k]
    distance_matrix = pd.generate_distance_matrix(coordinates_k)
    
    start_time = time.perf_counter()
    best_tour, best_cost = exhaustive_tsp_decrease_by_one(distance_matrix)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    return elapsed_time, best_tour, best_cost
    
    # print("\nBest Tour Found:", best_tour)
    # print ("Minimum Tour Cost:", best_cost)
    # print(f"Execution Time: {elapsed_time:.6f} seconds")
    # visualize_tour(coordinates_k, distance_matrix, best_tour)

# # Main execution
# if __name__ == "__main__" or __name__ == "exhaustive_search":
#     all_coordinates = pd.pd_runner(10)
#     # Run exhaustive search for smaller instance 
#     es_runner(all_coordinates, 10)

