import argparse
from typing import Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial_population", type=int, default=300, help="Initial population size"
    )
    parser.add_argument(
        "--n_variables", type=int, default=2, help="Number of variables of the problem"
    )
    parser.add_argument(
        "--percentage",
        type=float,
        default=0.3,
        help="Percentage of solutions to keep after selection",
    )
    parser.add_argument(
        "--n_generations", type=int, default=80, help="Number of generations to run"
    )
    parser.add_argument(
        "--n_cross", type=int, default=10, help="Number of cross to perform"
    )
    parser.add_argument(
        "--n_new_solutions",
        type=int,
        default=50,
        help="Number of new solutions to generate",
    )
    return parser.parse_args()


def evaluate_objectives(solutions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluates and returns both objectives of the problem given a matrix of solutions
    with shape (n_solutions, n_variables). The objectives shape is (n_solutions,)."""

    first_objective = np.exp(-np.prod(solutions - 1 / np.sqrt(3), axis=1))
    second_objective = np.exp(-np.prod(solutions + 1 / np.sqrt(3), axis=1))

    return first_objective, second_objective


def evaluate_constraints(solutions: np.ndarray) -> np.ndarray:
    """Evaluates if the solutions violate the constraint of the problem given a matrix
    of solutions with shape (n_solutions, n_variables) and return a binary array of
    shape (n_solutions,). 1 means that the constraint is satisfied and 0 means that the
    constraint is violated. The constraint is violated if the any the variables of the
    solution is outside the interval [-1.5, 1.5]."""
    return np.prod(np.logical_and(solutions <= 1.5, solutions >= -1.5), axis=1)


def generate_population(n_solutions: int, n_variables: int) -> np.ndarray:
    """Generates a population of solutions with shape (n_solutions, n_variables) using a
    uniform distribution with range [-1.5, 1.5]."""
    return np.random.uniform(low=-1.5, high=1.5, size=(n_solutions, n_variables))


def rank_solutions(
    first_objective: np.ndarray, second_objective: np.ndarray
) -> np.ndarray:
    """Ranks the solutions given the first and second objective of each solution.
    The ranking corresponds to the number of solutions that dominate the solution. The
    smaller the ranking, the better the solution."""

    n_solutions = first_objective.shape[0]

    ranking_list = []

    for first_sol in range(n_solutions):
        # Iterate over every solution
        dominate_count = 0

        for second_sol in range(n_solutions):
            # Compare the solution with every other solution and evaluate if it is
            # dominated
            dominate_first = first_objective[second_sol] < first_objective[first_sol]
            dominate_second = second_objective[second_sol] < second_objective[first_sol]
            if dominate_first and dominate_second:
                # If the solution is dominated, add 1 to the dominate count
                dominate_count += 1

        # Append the dominate count to the ranking list
        ranking_list.append(dominate_count)
    return np.array(ranking_list)


def selections(
    solutions: np.ndarray, ranking: np.ndarray, percentage: float
) -> np.ndarray:
    if percentage > 1 or percentage < 0:
        raise ValueError("Percentage must be between 0 and 1")

    # Sort the solutions by ranking
    sorted_indices = np.argsort(ranking)

    # Get the sorted solutions
    sorted_solutions = solutions[sorted_indices]

    # Cut the solutions according to a percentage
    n_solutions_to_keep = int(percentage * solutions.shape[0])

    return sorted_solutions[:n_solutions_to_keep]


def cross(solutions: np.ndarray, n_cross: np.ndarray) -> np.ndarray:
    # Get the indices of the solutions to mutate
    first_permutation = np.random.permutation(solutions.shape[0])[:n_cross]
    second_permutation = np.random.permutation(solutions.shape[0])[:n_cross]

    solution_array = []

    for first_sol_ind, second_sol_ind in zip(first_permutation, second_permutation):
        # Iterate over the solutions to mutate

        # Get the solutions to mutate
        first_solution = solutions[first_sol_ind].reshape(1, -1)
        second_solution = solutions[second_sol_ind].reshape(1, -1)

        # Get the variables to mutate
        mutation_mask = np.random.choice(
            [True, False], size=solutions.shape[1], p=[0.5, 0.5]
        )

        # Swap the variables and append new solution
        new_solution = np.where(mutation_mask, first_solution, second_solution)
        solution_array.append(new_solution)

    # Convert the list to an array
    new_solutions = np.concatenate(solution_array, axis=0)

    # Concatenate the new solutions with the old ones
    new_population = np.concatenate([solutions, new_solutions], axis=0)

    return new_population


def maximum_spread(
    ranking: np.ndarray, first_objective: np.ndarray, second_objetive: np.ndarray
) -> float:
    # Set of non-dominated solutions
    non_dominated_solutions_mask = np.logical_not(ranking)

    acum = 0
    for objective in [first_objective, second_objetive]:
        # Get max and min solution
        maximum_solution = np.max(objective)
        minimum_solution = np.min(objective)

        # Get maz and min non-dominated solution
        maximum_non_dominated_solution = np.max(objective[non_dominated_solutions_mask])
        minimum_non_dominated_solution = np.min(objective[non_dominated_solutions_mask])

        # Compute the spread and add it to the acum
        acum += (maximum_non_dominated_solution - minimum_non_dominated_solution) / (
            maximum_solution - minimum_solution
        )

    # Take the average and compute square root
    return np.sqrt(acum / 2)


def main(args: argparse.Namespace):
    # Generate initial population
    population = generate_population(args.initial_population, args.n_variables)

    for generation_i in range(args.n_generations):
        # Iterate over the generations

        # Evaluate objectives and constraints
        first_objective, second_objective = evaluate_objectives(population)

        # Compute ranking
        ranking = rank_solutions(first_objective, second_objective)

        # Compute maximum spread
        maximum_spread_value = maximum_spread(
            ranking, first_objective, second_objective
        )

        # Print generation information
        print(f"Generation {generation_i} - Maximum spread: {maximum_spread_value:.2f}, f1 min: {np.min(first_objective):.4f}, f2 min: {np.min(second_objective):.4f}")

        # Selection
        selected_solutions = selections(population, ranking, percentage=args.percentage)

        # Cross
        new_population = cross(selected_solutions, n_cross=args.n_cross)

        # Generate new solutions
        new_solutions = generate_population(args.n_new_solutions, args.n_variables)

        # Concatenate new solutions with the old ones
        population = np.concatenate([new_population, new_solutions], axis=0)

    # Evaluate the final population
    first_objective, second_objective = evaluate_objectives(population)

    # Compute ranking
    ranking = rank_solutions(first_objective, second_objective)

    # Compute maximum spread
    maximum_spread_value = maximum_spread(ranking, first_objective, second_objective)

    # Print final information
    print(f"Final - Maximum spread: {maximum_spread_value:.2f}")

    non_dominated_solutions_mask = np.logical_not(ranking)

    non_dominated_solutions = population[non_dominated_solutions_mask]

    return non_dominated_solutions

def plot_solutions(solutions: np.ndarray):
    import matplotlib.pyplot as plt
    plt.plot([0, 1], [0, 1])
    # plt.scatter(solutions[:, 0], solutions[:, 1])
    plt.show()

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    solutions = main(args)

    plot_solutions(solutions)
