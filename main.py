import argparse
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

DATA_PATH = Path.cwd() / "data" / "SPP.csv"
VALID_COST_METHODS = {"weighted", "lexic", "mixture"}


@dataclass
class Costs:
    distance: int
    emission: int
    risk: int

    def __add__(self, other: "Costs") -> "Costs":
        return Costs(
            self.distance + other.distance,
            self.emission + other.emission,
            self.risk + other.risk,
        )


@dataclass
class Node:
    cur_node: int
    costs: Costs


@dataclass
class PathToNode:
    cur_node: int
    costs: Costs
    total_cost: float
    path: list

    def __lt__(self, other: "PathToNode") -> bool:
        return self.total_cost < other.total_cost


def arguments_santiy_check(args: dict) -> None:
    """Check arguments."""
    if args["cost_method"] not in VALID_COST_METHODS:
        raise ValueError(
            "Invalid cost method. Valid methods are: weighted, lexic, mixture."
        )
    if len(args["weights"]) != 3:
        raise ValueError("Weights must have 3 values.")

    # Normalize weights
    total_sum = sum(args["weights"])
    args["weights"] = [weight / total_sum for weight in args["weights"]]


def parse_arguments() -> dict:
    """Parse and return arguments from command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cost-method",
        type=str,
        nargs="+",
        default="weighted",
        help="Method to evaluate path costs.",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=[1, 1, 1],
        help="Weights to evaluate path costs.",
    )
    args = vars(parser.parse_args())
    arguments_santiy_check(args)
    return args


def parse_line(line: str) -> Tuple[Node, Node]:
    """Parse line from csv file and return two nodes."""
    node1, node2, distance, emission, risk = line.split(",")
    node1, node2 = int(node1), int(node2)
    distance, emission, risk = float(distance), float(emission), float(risk)
    costs = Costs(distance, emission, risk)
    node1 = Node(node1, costs)
    node2 = Node(node2, costs)
    return node1, node2


def read_data(path: Path) -> Tuple[int, Dict[int, Node]]:
    """Read data from csv file and return number of node, and path dictionary."""

    with open(path, "r") as f:
        lines = f.read().split("\n")

        n_nodes = int(lines[0].split(",")[0][-3:])

        paths = {}
        for line in lines[1:]:
            node1, node2 = parse_line(line)

            if node1.cur_node in paths:
                paths[node1.cur_node].append(node2)
            else:
                paths[node1.cur_node] = [node2]

            if node2.cur_node in paths:
                paths[node2.cur_node].append(node1)
            else:
                paths[node2.cur_node] = [node1]

    return n_nodes, paths


def create_new_path(
    cur_path: PathToNode, to_node: Node, cost_method: str, weights: List[float] = []
) -> PathToNode:
    """Create new path to node."""
    new_costs = cur_path.costs + to_node.costs
    if cost_method == "weighted":
        new_total_cost = (
            new_costs.distance * weights[0]
            + new_costs.emission * weights[1]
            + new_costs.risk * weights[2]
        )
    elif cost_method == "lexic":
        raise NotImplementedError("Pending to implement")
    else:
        raise NotImplementedError("Pending to implement")
    new_path = PathToNode(
        to_node.cur_node,
        new_costs,
        new_total_cost,
        cur_path.path + [to_node.cur_node],
    )
    return new_path


def dijkstra(end_node: int, paths: Dict[int, Node], cost_kwargs: dict) -> PathToNode:
    """Dijkstra algorithm to find shortest path."""
    visited = set()
    cur_path = PathToNode(1, Costs(0, 0, 0), 0, [1])
    priority_queue = [cur_path]
    heapq.heapify(priority_queue)

    while priority_queue:
        cur_path = heapq.heappop(priority_queue)
        cur_node = cur_path.cur_node
        if cur_node not in visited:
            visited.add(cur_node)
            for to_node in paths[cur_node]:
                if to_node.cur_node not in visited:
                    new_path = create_new_path(cur_path, to_node, **cost_kwargs)
                    heapq.heappush(priority_queue, new_path)
                    if new_path.cur_node == end_node:
                        return new_path


if __name__ == "__main__":
    args = parse_arguments()
    n_nodes, paths = read_data(DATA_PATH)
    shortest_path = dijkstra(n_nodes, paths, args)
    print(shortest_path)
