"""RL-based action selection algorithm using PPO and a custom Gym environment."""

import numpy as np
from sklearn.metrics import mean_squared_error
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from gymnasium import spaces
from src.actions_definition import GenericFunction
from src.reinforcement_learning.utils import discretize_actions


def compute_reward(pred, true):
    """
    Compute the reward as the negative mean squared error between predictions and ground truth.

    Parameters:
        pred (np.ndarray): Predicted values.
        true (np.ndarray): Ground truth values.

    Returns:
        float: Negative mean squared error.
    """
    pred = np.nan_to_num(pred, nan=0.0, posinf=1e10, neginf=-1e10)
    true = np.nan_to_num(true, nan=0.0, posinf=1e10, neginf=-1e10)
    return -mean_squared_error(true.flatten(), pred.flatten())


class TimeSeriesEnv(gym.Env):
    """
    Custom Gym environment for time series transformation using discrete function-based actions.
    """

    def __init__(self, function_types, discrete_actions, params_ranges, true_values, initial_predictions):
        super(TimeSeriesEnv, self).__init__()
        self.function_types = function_types
        self.discrete_actions = discrete_actions
        self.params_ranges = params_ranges
        self.true_values = true_values
        self.initial_predictions = initial_predictions.copy().astype(np.float32)
        self.current_predictions = self.initial_predictions.copy()
        self.action_space = spaces.Discrete(len(discrete_actions))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=initial_predictions.shape, dtype=np.float32
        )
        self.action_sequence = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_predictions = self.initial_predictions.copy()
        self.action_sequence = []
        return self.current_predictions, {}

    def step(self, action):
        function_type, params = self.discrete_actions[action]
        self.action_sequence.append((function_type, params))
        generic_function = GenericFunction(function_type, params)
        self.current_predictions = generic_function.apply(
            self.current_predictions, self.true_values
        ).astype(np.float32)
        reward = float(compute_reward(self.current_predictions, self.true_values))
        return self.current_predictions, reward, False, False, {}

    def render(self, mode='human'):
        pass


def contextual_bandit_reinforcement(
    exp,
    args,
    episodes,
    alpha,
    gamma,
    epsilon,
    action_budget=20,
    improvement_threshold=0.0,
    generic_function_class=None,
    generate_random_parameters_function=None,
    function_types=None,
    save_dir='./results',
    batch_size=32,
    replay_buffer_size=10000,
    pred=None,
    true=None,
    batch_x=None,
    streamlit=False,
):
    """
    Explore and selction de best actions with a RL algorithm.

    Parameters:
        exp: Experiment object with validation method.
        args: Command-line arguments or configuration object.
        episodes (int): Number of RL training episodes.
        alpha (float): Learning rate (not used directly here).
        gamma (float): Discount factor (not used directly here).
        epsilon (float): Exploration factor (not used directly here).
        action_budget (int): Maximum number of actions to apply per channel.
        improvement_threshold (float): Threshold to consider improvements (not used here).
        generic_function_class (class): Placeholder, not used in this function.
        generate_random_parameters_function (callable): Placeholder, not used in this function.
        function_types (list): List of function types to use for transformations.
        save_dir (str): Directory path for saving results.
        batch_size (int): Batch size for evaluation (unused).
        replay_buffer_size (int): Replay buffer size (unused).
        pred (np.ndarray): Initial predictions, if available.
        true (np.ndarray): Ground truth values, if available.
        batch_x (np.ndarray): Input features for batch.
        streamlit (bool): Whether to enable Streamlit-compatible output (unused).

    Returns:
        tuple: (
            function_set_per_channel (list),
            best_mse (float),
            best_pred (np.ndarray),
            true (np.ndarray),
            batch_x (np.ndarray)
        )
    """
    num_discretizations = 10
    params_ranges = {
        "scale_amplitude": {"factor": (10, 20)},
        "piecewise_scale_high": {"threshold": (0, 100), "factor": (-5, 5)},
        "piecewise_scale_low": {"threshold": (0, 100), "factor": (-5, 5)},
        "add_linear_trend_slope": {"slope": (0.1, 0.4)},
        "add_linear_trend_intercept": {"intercept": (0.1, 0.4)},
        "increase_minimum_factor": {"factor": (0.1, 0.4)},
        "increase_maximum_factor": {"factor": (0.1, 0.4)}
    }

    discrete_actions = discretize_actions(function_types, params_ranges, num_discretizations)

    if pred is None:
        _, pred, true, batch_x = exp.validation(args.model_id, test=1)

    initial_predictions = pred
    best_pred = np.zeros_like(pred)
    best_mse = mean_squared_error(pred.flatten(), true.flatten())
    function_set_per_channel = []

    for channel_idx in range(initial_predictions.shape[2]):
        print(f"Training for Channel {channel_idx}...")
        true_channel = true[:, :, channel_idx:channel_idx + 1]
        pred_channel = initial_predictions[:, :, channel_idx:channel_idx + 1]

        env = TimeSeriesEnv(function_types, discrete_actions, params_ranges, true_channel, pred_channel)
        check_env(env)
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=100)
        model.save(f"{save_dir}/ppo_time_series_channel_{channel_idx}")

        # Re-run environment with trained policy
        print(f"Getting optimal action sequence for Channel {channel_idx}...")
        env = TimeSeriesEnv(function_types, discrete_actions, params_ranges, true_channel, pred_channel)
        state, _ = env.reset()

        optimal_actions = []
        function_optimal = []
        params_optimal = []

        for _ in range(100):
            action, _ = model.predict(state, deterministic=True)
            function_type, params = env.discrete_actions[action]
            optimal_actions.append((function_type, params))
            function_optimal.append(function_type)
            params_optimal.append(params)
            state, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        # Apply optimal sequence
        final_pred = pred_channel.copy()
        for func_type, params in optimal_actions:
            generic_func = GenericFunction(func_type, params)
            final_pred = generic_func.apply(final_pred, true_channel)

        best_pred[:, :, channel_idx] = final_pred.squeeze()

        for func, param in zip(function_optimal, params_optimal):
            function_set_per_channel.append((channel_idx, func, param))

    print("Final MSE:", mean_squared_error(best_pred.flatten(), true.flatten()))
    return function_set_per_channel, best_mse, best_pred, true, batch_x
