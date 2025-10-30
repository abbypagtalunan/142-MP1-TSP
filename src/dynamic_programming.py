import time
import matplotlib.pyplot as plt
from src import problem_definition as pd


# DYNAMIC PROGRAMMING BELLMAN-HELD-KARP ALGORITHM (Top-Down Recursive with Memoization) 

# from Geeks for Geeks

# Dynamic Programming Bellman-Held-Karp  
# For solving Traveling Salesman Problem
def dp_bhk_solve(mask, curr, n, cost, memo, tour):
    # Base Case: If all cities are visited, 
    #               return the cost to return 
    #               to the starting city (0)
    if mask == (1 << n) - 1:
        return cost[curr][0]
    
    # If the value has been computed already,
    # return it from the memo table
    if memo[curr][mask] != -1: 
        return memo[curr][mask]
    
    ans = float('inf')
    next = -1

    # Visiting every city that has not been visited yet
    for i in range(n):
        if (mask & (1 << i)) == 0:   
            # If city has not been visited,
            # visit city i and update the mask
            ncost = cost[curr][i] + dp_bhk_solve(mask | (1 << i), i, n, cost, memo, tour)
            if ncost < ans:
                ans = ncost
                next = i

    memo[curr][mask] = ans
    tour[curr][mask] = next
    return ans

# Constructing best (optimal) tour from tour table
def btour(tour, n):
    mask = 1
    curr = 0
    path = [0]
    while True:
        i = tour[curr][mask]
        if i == -1 or i is None:
            break
        path.append(i)
        mask |= (1 << i)
        curr = i
    path.append(0)
    return path

# Runner function
def dp_bhk(dmatrix):
    n = len(dmatrix)
    # Initialize memoization table with -1 (uncomputed states)
    memo = [[-1] * (1 << n) for _ in range(n)]
    tour = [[-1] * (1 << n) for _ in range(n)]

    # Start from city 0, with only city 0 visited initially (mask = 1)
    best_cost = dp_bhk_solve(1, 0, n, dmatrix, memo, tour)
    best_tour = btour(tour, n)
    return best_tour, best_cost

def calculate_tour_cost(tour, dmatrix):
    total = 0
    for i in range(len(tour) - 1):
        total += dmatrix[tour[i]][tour[i + 1]]
    return total

def visualize_tour(coordinates, dmatrix, tour):
    plt.figure(figsize=(8, 8))
    n = len(coordinates)

    for i in range(len(tour) - 1):
        city1 = tour[i]
        city2 = tour[i + 1]
        x1, y1 = coordinates[city1]
        x2, y2 = coordinates[city2]
        plt.plot([x1, x2], [y1, y2], 'orange', linewidth=2, zorder=2)

    plt.scatter(coordinates[:, 0], coordinates[:, 1], color='darkgreen', zorder=3)
    for i, (x, y) in enumerate(coordinates):
        plt.text(x + 1, y + 1, f"{i}", fontsize=10, color='black')

    plt.title(f"Best TSP Tour (Cost: {calculate_tour_cost(tour, dmatrix)})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

def dp_runner(all_coordinates, k):
    coordinates_k = all_coordinates[:k]
    distance_matrix = pd.generate_distance_matrix(coordinates_k)

    start_time = time.perf_counter()
    best_tour, best_cost = dp_bhk(distance_matrix)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # print("\nBest Tour Found:", best_tour)
    # print ("Minimum Tour Cost:", best_cost)
    # print(f"Execution Time: {elapsed_time:.6f} seconds")
    # visualize_tour(all_coordinates, distance_matrix, best_tour)

    return elapsed_time, best_tour, best_cost


# Main execution
# if __name__ == "__main__" or __name__ == "src.dynamic_programming":
#     # Generate 30 cities
#     all_coordinates = pd.pd_runner(30)
#     # Run Dynamic Programming Bellman-Held-Karp for generated cities
#     dp_runner(all_coordinates, 30)

