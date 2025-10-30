import numpy as np
import src.problem_definition as pd
import src.exhaustive_search as es
import src.greedy_heuristic as gh
import src.dynamic_programming as dp

def run_algorithm(algorithm_name, runner, city_counts):
    """
    Runs a given TSP algorithm for different city counts and prints timing + results.
    """
    for n in city_counts:
        all_coordinates, distance_matrix = pd.pd_runner(n)
        if all_coordinates is None or len(all_coordinates) == 0:
            print(f"No coordinates returned for n={n}")
            continue

        print(f"\nDISTANCE MATRIX FOR {n} CITIES:")
        np.set_printoptions(linewidth=200, suppress=True, threshold=np.inf)
        print(distance_matrix)
        print() 

        print("-" * 130)
        print(f"{algorithm_name} WITH {n} CITIES")
        print("-" * 130)
        print(f"{'Run':<5} {'Time (s)':<12} {'Cost':<20} Path")
        print("-" * 130)

        times = []
        for run in range(3):
            elapsed_time, cost, path = runner(all_coordinates, n)
            times.append(elapsed_time)
            cost_str = f"{cost:.4f}" if isinstance(cost, (int, float, np.floating)) else str(cost)
            print(f"{run + 1:<5} {elapsed_time:<12.4f} {path} {cost_str:<20}")

        avg_time = np.mean(times)
        print("-" * 130)
        print(f"{'Average':<5} {avg_time:<12.4f}")
        print() 

        # visualize_tour(all_coordinates, distance_matrix, path)


def main():
    small_instance = [5, 8, 10, 11, 12, 13]
    large_instance = [15, 20, 22, 24, 25, 27]

    experiments = [
        ("EXHAUSTIVE SEARCH (DECREASE-BY-ONE)", es.es_runner, small_instance),
        ("GREEDY HEURISTIC", gh.gh_runner, large_instance),
        ("DYNAMIC PROGRAMMING (BELLMAN-HELD-KARP)", dp.dp_runner, large_instance),
    ]

    for name, runner, cities in experiments:
        run_algorithm(name, runner, cities)

if __name__ == "__main__":
    main()
