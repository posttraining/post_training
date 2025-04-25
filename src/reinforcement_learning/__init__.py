"""Action selection module for contextual bandit strategies."""

from typing import Any
from src.actions_definition import GenericFunction
from src.reinforcement_learning.random import contextual_bandit_random
from src.reinforcement_learning.bandit import (ContextualBanditSuccessiveRejects, ContextualBanditUniformBAI,
                                               ContextualBanditSuccessiveHalving, ContextualBanditLUCB)
from src.reinforcement_learning.rl import contextual_bandit_reinforcement
from src.reinforcement_learning.genetic import genetic_algorithm_reinforcement


FUNCTION_TYPES = [
    "scale_amplitude",
    "piecewise_scale_high",
    "piecewise_scale_low",
    "add_linear_trend_slope",
    "add_linear_trend_intercept",
    # "increase_minimum_factor",
    # "increase_maximum_factor",
]


def explore_instructions(
    exp: Any,
    args: Any,
    episodes: int = 10,
    action_budget: int = 10,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon:float = 0.1,
    streamlit: bool = False,
):
    """
    Runs an exploration strategy using a specified contextual bandit method.

    Parameters:
        exp: Experiment configuration or object.
        args: Argument object containing the 'method' attribute to select the strategy.
        episodes (int): Number of exploration episodes to run. Default is 10.
        action_budget (int): Maximum number of actions allowed. Default is 10.
        alpha (float): Learning rate parameter. Default is 0.1.
        gamma (float): Discount factor for future rewards. Default is 0.9.
        epsilon (float): Exploration rate for epsilon-greedy strategies. Default is 0.1.
        streamlit (bool): Flag to enable Streamlit-related behavior. Default is False.

    Returns:
        tuple: (best_pred, true, batch_x, best_function_set_per_channel)
            - best_pred: Best prediction values.
            - true: Ground truth values.
            - batch_x: Input features for the batch.
            - best_function_set_per_channel: Selected functions per channel.
    """
    N_ITERATIONS = 15
    MAX_ITER_HYPEROPT = 10

    if args.method == "random":
        rl_algorithm = contextual_bandit_random

    elif args.method == "SR-HPO":
        rl_algorithm = ContextualBanditSuccessiveRejects(n_function_types=len(FUNCTION_TYPES),
                                                         n_iterations=N_ITERATIONS,
                                                         max_iter_hyperopt=MAX_ITER_HYPEROPT,
                                                         n_jobs=args.n_jobs,
                                                         )

    elif args.method == "U-HPO":
        rl_algorithm = ContextualBanditUniformBAI(n_function_types=len(FUNCTION_TYPES),
                                                  n_iterations=N_ITERATIONS,
                                                  max_iter_hyperopt=MAX_ITER_HYPEROPT,
                                                  n_jobs=args.n_jobs,
                                                  )

    elif args.method == "SH-HPO":
        rl_algorithm = ContextualBanditSuccessiveHalving(n_function_types=len(FUNCTION_TYPES),
                                                         n_iterations=N_ITERATIONS,
                                                         max_iter_hyperopt=MAX_ITER_HYPEROPT,
                                                         n_jobs=args.n_jobs,
                                                         )

    elif args.method == "LUCB-HPO":
        rl_algorithm = ContextualBanditLUCB(n_function_types=len(FUNCTION_TYPES),
                                            n_iterations=N_ITERATIONS,
                                            max_iter_hyperopt=MAX_ITER_HYPEROPT,
                                            n_jobs=args.n_jobs,
                                            )

    elif args.method == "Genetic":
        rl_algorithm = genetic_algorithm_reinforcement

    elif args.method == "PPO":
        rl_algorithm = contextual_bandit_reinforcement

    else:
        raise ValueError(f"Unsupported method: '{args.method}'; valid method are 'random', 'SR-HPO',"
                         f" 'U-HPO', 'SH-HPO', 'Genetic' or 'PPO'")

    best_function_set_per_channel, best_mse, best_pred, true, batch_x = rl_algorithm(
        exp,
        args,
        episodes,
        alpha,
        gamma,
        epsilon,
        action_budget=action_budget,
        function_types=FUNCTION_TYPES,
        generic_function_class=GenericFunction,
        streamlit=streamlit
    )

    return best_pred, true, batch_x, best_function_set_per_channel
