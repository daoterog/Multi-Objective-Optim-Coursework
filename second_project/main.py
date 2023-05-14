import argparse
from typing import Tuple, Dict, Union, Optional

import numpy as np


def parse_args() -> Dict[str, Union[int, float]]:
    """Parse the arguments of the program and return a dictionary of arguments."""
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
    return vars(parser.parse_args())


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

    ranking = np.array(ranking_list)

    return ranking


def selections(
    solutions: np.ndarray, ranking: np.ndarray, percentage: float
) -> np.ndarray:
    if percentage > 1 or percentage < 0:
        raise ValueError("Percentage must be between 0 and 1")

    # Cut the solutions according to a percentage
    n_solutions_to_keep = int(percentage * solutions.shape[0])

    # Sort the solutions by ranking
    best_solution_indices = np.argsort(ranking)[:n_solutions_to_keep]

    return solutions[best_solution_indices]


def cross(solutions: np.ndarray, n_cross: np.ndarray) -> np.ndarray:
    # Get the indices of the solutions to mutate
    first_permutation = np.random.permutation(solutions.shape[0])[:n_cross]
    second_permutation = np.random.permutation(solutions.shape[0])[:n_cross]

    solution_array = []
    n_variables = solutions.shape[1]

    for first_sol_ind, second_sol_ind in zip(first_permutation, second_permutation):
        # Iterate over the solutions to mutate

        # Get the solutions to mutate
        first_solution = solutions[first_sol_ind].reshape(1, -1)
        second_solution = solutions[second_sol_ind].reshape(1, -1)

        # Get the variables to mutate
        if n_variables > 2:
            mutation_mask = np.random.choice(
                [True, False], size=solutions.shape[1], p=[0.5, 0.5]
            )
        else:
            random_number = np.random.uniform()
            mutation_mask = np.array([random_number > 0.5, random_number <= 0.5])

        # Swap the variables and append new solution
        new_solution = np.where(mutation_mask, first_solution, second_solution)
        solution_array.append(new_solution)

    # Convert the list to an array
    new_solutions = np.concatenate(solution_array, axis=0)

    # Concatenate the new solutions with the old ones
    new_population = np.concatenate([solutions, new_solutions], axis=0)

    return new_population


def maximum_spread(first_objective: np.ndarray, second_objetive: np.ndarray) -> float:

    acum = 0
    for objective in [first_objective, second_objetive]:
        # Get max and min solution
        maximum_solution = np.max(objective)
        minimum_solution = np.min(objective)

        # Get maz and min non-dominated solution
        maximum_non_dominated_solution = np.max(objective)
        minimum_non_dominated_solution = np.min(objective)

        # Compute the spread and add it to the acum
        acum += (maximum_non_dominated_solution - minimum_non_dominated_solution) / (
            maximum_solution - minimum_solution
        )

    # Take the average and compute square root
    return np.sqrt(acum / 2)


def generational_distance(
    pareto_optimal: np.ndarray, non_dominated_solutions: np.ndarray
) -> float:
    """Calculate the non generational distance between the pareto optimal solutions and
    the non dominated solutions. The pareto optimal is a matrix of shape
    (n_pareto_solutions, 2) and the non dominated solutions is a matrix of shape
    (n_non_dominated_solutions, 2). The columns of the matrices are the first and second
    objective respectively."""

    def get_minimum_distance(
        pareto_optimal: np.ndarray, non_dominated_solution: np.ndarray
    ) -> float:
        """Calculates the minimum distance between a given non dominated solution and
        every pareto optimal solution."""

        min_distance = np.inf

        for pareto_solution in range(len(pareto_optimal)):
            difference = pareto_optimal[pareto_solution] - non_dominated_solution
            distance = np.sqrt(np.sum(difference**2))
            if distance < min_distance:
                min_distance = distance

        return min_distance

    # Compute the minimum distance for every non dominated solution
    minimum_distances = []
    for non_dominated_solution in range(len(non_dominated_solutions)):
        minimum_distance = get_minimum_distance(
            pareto_optimal, non_dominated_solutions[non_dominated_solution]
        )
        minimum_distances.append(minimum_distance**2)

    # Compute the average of the minimum distances
    return np.sqrt(np.sum(minimum_distances)) / len(non_dominated_solutions)


def hyper_volume(
    first_objective: np.ndarray, second_objective: np.ndarray, nadir_point: np.ndarray
) -> float:
    """Calculates the hyper volume of the pareto optimal solutions given the first and
    second objective of each solution. The hyper volume is the area under the curve
    formed by the pareto optimal solutions. The pareto optimal is a matrix of shape
    (n_pareto_solutions, 2). The columns of the matrix are the first and second
    objective respectively."""

    # Sort the pareto optimal solutions by the first objective
    sorted_indices = np.argsort(first_objective)
    sorted_first_objective = first_objective[sorted_indices]
    sorted_second_objective = second_objective[sorted_indices]

    # Compute the hyper volume
    acum = 0
    new_nadir_point = nadir_point.copy()
    for i in range(len(sorted_first_objective) - 1):
        acum += (new_nadir_point[0] - sorted_first_objective[i]) * (
            new_nadir_point[1] - sorted_second_objective[i]
        )
        new_nadir_point[1] = sorted_second_objective[i]

    return acum


def evaluate_optimality(
    population: np.ndarray, generation_i: int, return_mask: bool = False
) -> Optional[np.ndarray]:
    """Evaluates the optimality of the solutions in the population. The optimality is
    evaluated by computing the maximum spread and the hyper volume of the non dominated
    solutions. The non dominated solutions are the solutions that are not dominated by
    any other solution. The function returns the mask of the non dominated solutions if
    return_mask is True. Otherwise, it returns the ranking of the solutions."""

    # Evaluate objectives and constraints
    first_objective, second_objective = evaluate_objectives(population)

    # Compute ranking
    ranking = rank_solutions(first_objective, second_objective)

    # Get of non-dominated solutions and apply mask to objectives
    non_dominated_solutions_mask = np.logical_not(ranking)
    non_dom_first_objective = first_objective[non_dominated_solutions_mask]
    non_dom_second_objective = second_objective[non_dominated_solutions_mask]

    # Compute maximum spread
    maximum_spread_value = maximum_spread(
        non_dom_first_objective, non_dom_second_objective
    )

    # Compute Hyper volume
    hyper_volume_value = hyper_volume(
        non_dom_first_objective,
        non_dom_second_objective,
        np.array([1, 1]),
    )

    # Print generation information
    print(
        f"Generation {generation_i} - Maximum spread: {maximum_spread_value:.2f}"
        f" - HyperVolume: {hyper_volume_value}, "
        f"f1 min: {np.min(non_dom_first_objective):.4f}, "
        f"f2 min: {np.min(non_dom_second_objective):.4f}"
    )

    if return_mask:
        return non_dominated_solutions_mask

    return ranking


def main(
    initial_population: int,
    n_variables: int,
    percentage: float,
    n_generations: int,
    n_cross: int,
    n_new_solutions: int,
) -> np.ndarray:
    # Generate initial population
    population = generate_population(initial_population, n_variables)

    for generation_i in range(n_generations):
        # Iterate over the generations

        # Evaluate optimality
        ranking = evaluate_optimality(population, generation_i)

        # Selection
        selected_solutions = selections(population, ranking, percentage=percentage)

        # Cross
        new_population = cross(selected_solutions, n_cross=n_cross)

        # Remove repeated solutions
        new_population = np.unique(new_population, axis=0)

        # Generate new solutions
        if len(new_population) + n_new_solutions < initial_population:
            new_population_size = (
                initial_population - len(new_population) - n_new_solutions
            )
        else:
            new_population_size = n_new_solutions
        new_solutions = generate_population(new_population_size, n_variables)

        # Concatenate new solutions with the old ones
        population = np.concatenate([new_population, new_solutions], axis=0)

    # Evaluate optimality
    non_dominated_solutions_mask = evaluate_optimality(
        population, generation_i, return_mask=True
    )

    # Return non dominated solutions
    return population[non_dominated_solutions_mask]


def plot_solutions(solutions: np.ndarray):
    import matplotlib.pyplot as plt

    plt.plot([0, 1], [0, 1])
    # plt.scatter(solutions[:, 0], solutions[:, 1])
    plt.show()


def make_combinations(n_variables: int) -> np.ndarray:
    """Makes all the possible combinations of the vector [-1.5, 1.5] with n_variables
    using a 100 point discretization."""
    return np.array(np.meshgrid(*[np.linspace(-1.5, 1.5, 10)] * n_variables)).T.reshape(
        -1, n_variables
    )


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    solutions = main(args)

    print(np.unique(solutions, axis=0))

    plot_solutions(solutions)
