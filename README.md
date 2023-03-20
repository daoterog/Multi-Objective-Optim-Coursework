# Multi-Objective-Optim-Coursework

The file `first_project.py` contains the code to solve a Shortest Past Problem using a Multi-Objective Dijkstra Algorithm with three different cost methods. The data is contained on a .csv file in the `data` folder. The code is written in Python 3.10.7 and is self-contained. The last means that it does not require any external libraries to run.

### Cost Methods:
1. Weighted Average:

Evaluates the cost function as a weighted average of the three objectives. The weights are given by the `--weights` argument. The weights are always normalized to sum to 1. The default weights are 1, 1, 1. It is not valid to assign negative weights.

2. Lexicographic:

The lexicographic method works by prioritizing the objectives in a strict order using a sequential approach. The optimization problem is solved for the first objective function. The optimal value of the first objective function is then used as a constraint for the subsequent optimization problem, which aims to optimize the second objective function while ensuring that the optimal value of the first objective function is not compromised.

This process continues for all remaining objective functions, with the optimal values of the previous objective functions being used as constraints for each subsequent optimization. This ensures that the solutions found not only optimize each objective function individually but also ensure that the optimal values of previous objectives are preserved in subsequent optimizations.

The order in which the objectives are optimized is given by the `--order` argument. The default order is 1, 2, 3. The order must be a permutation of the numbers 1, 2, 3. The objective functions are aranged as follows: distance, emission, and risk. Hence the default order is to optimize the distance first, then the emission, and finally the risk. If the order is 2, 1, 3, then the emission is optimized first, followed by the distance, and finally the risk.

3. Mixture of the two above:

Our implementation of the mixture method preserves the sequential nature of the Lexicographic method but introducing a Weighted Average cost function. In each iteration, the prioritized objective function will receive a higher weight than the other two objectives. The remaining weights are going to be equally distributed between the other objectives such that the sum of the weights is 1. We hope that the "soft" evaluation of the cost function allows the algorithm to find a feasible optimal solution when the Lexicographic method fails to do so

As the Lexicographic method, the mixture uses the `--order` argument. It also uses the `--mixture-weight` argument to determine the weight of the prioritized objective function. The default value is 0.7. The value must be a float between 0.5 and 1.


### Arguments
- --cost-method: The cost method to use.
    - Default: mixture
    - Options: weighted, lexic, and mixture
- --weights: The weights to use with the `weighted` cost-method.
    - Default: 1, 1, 1
- --order: The order in which the objectives will be optimized. Only valid to use with `lexic` and `mixture` cost-methods.
    - Default: 1, 2, 3
- --mixture-weight: The weight of the prioritized objective function. Only valid to use with `mixture` cost-method.
    - Default: 0.7

### Usage

To run the code, simply run the following command:

    python first_project.py

The code will run with the default arguments. To change the arguments, simply add the argument name and the value to the command. For example, to run the code with the `lexic` cost-method and the order 2, 1, 3, run the following command:

    python first_project.py --cost-method lexic --order 2 1 3

### Output

The code will print the solution to the console. The solution will be printed in the following format:

    Path: [node1, node2, ..., nodeN]
    Distance: [distance]
    Emission: [emission]
    Risk: [risk]

It is possible that the code does not find a feasible solution when using `lexic` and `mixture` cost-methods. In this case, the code will print the following message:

    There is no feasible solution that preserves previous optimal solution.
    The previous optimal solution will be returned. It optimized the first n objectives.