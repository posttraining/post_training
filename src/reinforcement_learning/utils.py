"""Action selection utilities module for discretizing function-based actions."""

import itertools
import numpy as np


def discretize_actions(function_types, params_ranges, num_discretizations):
    """
    Generate a list of discrete actions by discretizing parameter ranges.

    Each action is represented as a tuple containing the function type and a
    dictionary of discretized parameters.

    Parameters:
        function_types (list of str): List of function types to apply.
        params_ranges (dict): Dictionary mapping each function type to a dictionary
            of parameter names and their (min, max) ranges.
        num_discretizations (int): Number of values to discretize each parameter range into.

    Returns:
        list of tuple: List of (function_type, parameter_dict) representing all discrete actions.
    """
    discrete_actions = []

    for function_type in function_types:
        param_ranges = params_ranges[function_type]

        # Create a grid of discretized values for each parameter
        params_grid = [
            np.linspace(start, end, num_discretizations)
            for start, end in param_ranges.values()
        ]

        # Iterate over all combinations in the grid
        for params in itertools.product(*params_grid):
            params_dict = dict(zip(param_ranges.keys(), params))
            discrete_actions.append((function_type, params_dict))

    return discrete_actions
