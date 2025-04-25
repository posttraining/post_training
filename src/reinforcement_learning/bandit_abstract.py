"""Bandit-based action selection algorithm using Upper Confidence Bound (UCB)."""

import time
from typing import Callable, List, Tuple, Optional, Any
from joblib import Parallel, delayed
import numpy as np
from scipy import optimize
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_random_state
from bayes_opt import BayesianOptimization
from .bandit_utils import get_logger, get_params_search_range


class ContextualBanditAbstract:

    """ContextualBanditAbstract abstract class for the post-training actions selection algorithm.
    """

    def __init__(
        self,
        n_function_types: int,
        n_iterations: Optional[int] = None,
        max_iter_hyperopt: int = 5,
        max_randint: int = 1000,
        n_jobs: int = 1,
        ):

        if n_iterations is None:
            self.n_iterations = int(1.5 *n_function_types)
        else:
            self.n_iterations = n_iterations

        self.max_iter_hyperopt = max_iter_hyperopt

        self.n_function_types = n_function_types
        self.bandit = None

        self.log_verbosity_level = 3
        self.joblib_verbose = 100

        self.n_jobs = n_jobs

        self.max_randint = max_randint

        random_state = None
        self.random_state = check_random_state(random_state)

    def _contextual_bandit_ucb_per_channel(self):
        pass

    def __call__(
        self,
        exp: Any,
        args: Any,
        episodes: int,
        alpha: float,
        gamma: float,
        epsilon: float,
        action_budget: int = 20,
        improvement_threshold: float = 0.0,
        generic_function_class: Optional[type] = None,
        generate_random_parameters_function: Optional[Callable] = None,
        function_types: Optional[List[str]] = None,
        save_dir: str = "./results",
        batch_size: int = 32,
        replay_buffer_size: int = 10000,
        population_size: int = 10,
        generations: int = 2,
        mutation_rate: float = 0.1,
        pred: Optional[np.ndarray] = None,
        true: Optional[np.ndarray] = None,
        batch_x: Optional[np.ndarray] = None,
        streamlit: bool = False,
        verbose: int = 100,
    ) -> Tuple[List[Tuple[int, str, dict]], float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Implements a contextual bandit algorithm using Upper Confidence Bound (UCB) to optimize actions
        and explore the best-performing transformations for time series prediction.

        Parameters:
            exp (Any): Experiment object with a `.validation()` method to get initial predictions and data.
            args (Any): Argument object containing relevant configuration parameters.
            episodes (int): Number of episodes to explore the actions. Default is 1.
            alpha (float): Exploration parameter for UCB, controlling exploration versus exploitation.
            gamma (float): Discount factor for future rewards (unused in this implementation).
            epsilon (float): Exploration rate for epsilon-greedy approaches (unused in this implementation).
            action_budget (int): Maximum number of actions to be selected per channel during each episode.
            improvement_threshold (float): Minimum required improvement in MSE to accept a transformation.
            generic_function_class (Optional[type]): Class used to instantiate transformations.
            generate_random_parameters_function (Optional[Callable]): Function to generate random parameters for actions.
            function_types (Optional[List[str]]): List of available function types (e.g., scaling, trend adding).
            save_dir (str): Directory to save results and predictions. Default is './results'.
            batch_size (int): Size of data batches used for evaluation. Default is 32.
            replay_buffer_size (int): Size of replay buffer for storing experiences. Default is 10000.
            population_size (int): Size of population if genetic strategies are used. Default is 10.
            generations (int): Number of generations for genetic search. Default is 2.
            mutation_rate (float): Mutation rate for evolutionary strategies. Default is 0.1.
            pred (Optional[np.ndarray]): Initial predictions from the experiment (optional).
            true (Optional[np.ndarray]): Ground truth values for evaluation (optional).
            batch_x (Optional[np.ndarray]): Input feature data for evaluation (optional).
            streamlit (bool): Whether to enable Streamlit for visualizations (unused in this implementation).
            verbose (int): verbose level of the parallelization of Joblib.

        Returns:
            Tuple: A tuple containing:
                - function_set_per_channel (List[Tuple[int, str, dict]]): Best actions and parameters per channel.
                - best_mse (float): Final MSE after applying the best actions.
                - best_pred (np.ndarray): Final predictions after transformations.
                - true (np.ndarray): Ground truth values.
                - pred (np.ndarray): Predictions before return.
        """
        # globals
        logger = get_logger(__name__)
        _id = self.random_state.randint(self.max_randint)

        best_pred = None
        function_set_per_channel: List[Tuple[int, str, dict]] = []

        if pred is None:
            _, pred, true, batch_x = exp.validation(args.model_id, test=1)

        n_channels = pred.shape[2]

        msg = (f"episodes={episodes} "
               f"x n_channels={n_channels} "
               f"x n_iterations={self.n_iterations} "
               f"x max_iter_hyperopt={self.max_iter_hyperopt} "
               f"= {episodes * n_channels * self.n_iterations * self.max_iter_hyperopt}")
        logger.info(f"[CHANNEL POST-PROCESS id {_id}] Starting searching improving action with total number samplings "
                    f"of: {msg}.", level=3, verbosity_level=self.log_verbosity_level)

        for episode in range(episodes):

            t0 = time.time()

            if episode == 0:
                best_pred = pred.copy()
            else:
                pred = best_pred.copy()

            # search improving action per channel
            results = Parallel(n_jobs=self.n_jobs, verbose=verbose)(
                delayed(self._contextual_bandit_ucb_per_channel)(
                    true=true,
                    pred=pred,
                    channel_idx=channel_idx,
                    batch_x=batch_x,
                    function_types=function_types,
                    n_iterations=self.n_iterations,
                    generic_function_class=generic_function_class,
                    max_iter_hyperopt=self.max_iter_hyperopt,
                    improvement_threshold=improvement_threshold
                )
                for channel_idx in range(n_channels)
            )

            # recover prediction
            for (channel_idx, best_action, best_params), channel_pred in results:
                function_set_per_channel.append((channel_idx, best_action, best_params))
                pred[:, :, channel_idx] = channel_pred.reshape(pred.shape[0], pred.shape[1])
                best_pred[:, :, channel_idx] = channel_pred.reshape(pred.shape[0], pred.shape[1])

            mse = mean_squared_error(pred.flatten(), true.flatten())

            logger.info(f"[Post-training id {_id}] Episode {episode + 1} / {episodes} done | mse = {mse:5.3f} |episode lasted {time.time() - t0:4.1f} s.",
                        level=3, verbosity_level=self.log_verbosity_level)

        return (
            function_set_per_channel,
            mean_squared_error(pred.flatten(), true.flatten()),
            best_pred,
            true,
            pred,
        )

    def _contextual_bandit_ucb_per_channel(
        self,
        true: np.ndarray,
        pred: np.ndarray,
        channel_idx: int,
        batch_x: np.ndarray,
        function_types: List[str],
        n_iterations: int,
        generic_function_class: Callable,
        max_iter_hyperopt: int,
        improvement_threshold: float,
    ) -> Tuple[Tuple[int, str, dict], np.ndarray]:
        """
        Applies UCB for a single channel to optimize the best transformation action based on the MSE improvement.

        Parameters:
            true (np.ndarray): Ground truth values for the current episode.
            pred (np.ndarray): Predictions for the current episode.
            channel_idx (int): Index for the current channel.
            batch_x (np.ndarray): Feature data for the current episode.
            function_types (List[str]): List of possible transformation functions to apply.
            n_iterations (int): Number of iterations to explore in UCB.
            generic_function_class (Callable): Class that applies transformations.
            max_iter_hyperopt (int): Maximum iterations for hyperparameter optimization.
            improvement_threshold (float): Minimum required improvement to consider a transformation valid.

        Returns:
            Tuple[Tuple[int, str, dict], np.ndarray]: Best action and parameters for the channel, and the transformed predictions.
        """
        if self.bandit is None:
            raise NotImplementedError('ContextualBanditUCBAbstract cannot be instantiated directly; please use an inherited class.')

        channel_true = true[:, :, channel_idx]
        channel_pred = pred[:, :, channel_idx]
        initial_mse = mean_squared_error(channel_true.flatten(), channel_pred.flatten())
        channel_batch = batch_x[:, :, channel_idx].reshape(batch_x.shape[0], batch_x.shape[1], 1).cpu().numpy()

        for t in range(1, n_iterations + 1):

            k = self.bandit.select(t)
            action = function_types[k]

            def _reward(action: str) -> Tuple[dict, float]:
                space_config = get_params_search_range(action)

                def __reward(params: dict) -> float:
                    return self._generic_reward(
                        generic_function_class,
                        action,
                        params,
                        pred,
                        channel_pred,
                        channel_batch,
                        channel_true
                    )

                best_params = self._hyperopt(__reward, space_config, max_iter_hyperopt)
                best_mse = __reward(best_params)

                return best_params, best_mse

            _, r = _reward(action)
            self.bandit.update(r, k)

        best_k = self.bandit.best_arm(t)
        best_action = function_types[best_k]
        best_params, best_mse_channel = _reward(best_action)

        # check if an improving action was found
        if best_mse_channel < initial_mse - improvement_threshold:
            improved_pred = generic_function_class(best_action, best_params).apply(
                channel_pred.reshape(pred.shape[0], pred.shape[1], 1), channel_batch)
            channel_pred = improved_pred

        return (channel_idx, best_action, best_params), channel_pred

    def _generic_reward(
        self,
        generic_function_class: Callable,
        action: str,
        params: dict,
        pred: np.ndarray,
        channel_pred: np.ndarray,
        channel_batch: np.ndarray,
        channel_true: np.ndarray
    ) -> float:
        """
        Calculates the mean squared error (MSE) as the reward for a transformation action.

        Parameters:
            generic_function_class (Callable): Class for applying the transformation.
            action (str): The action (transformation) being applied.
            params (dict): Parameters for the selected action.
            pred (np.ndarray): Predictions before transformation.
            channel_pred (np.ndarray): Predictions for the current channel.
            channel_batch (np.ndarray): Feature data for the current channel.
            channel_true (np.ndarray): True values for the current channel.

        Returns:
            float: The MSE as a reward measure.
        """
        new_pred = generic_function_class(action, params).apply(
            channel_pred.reshape(pred.shape[0], pred.shape[1], 1),
            channel_batch
        )
        return mean_squared_error(channel_true.flatten(), new_pred.flatten())

    def _hyperopt(
        self,
        reward_func: Callable[[dict], float],
        space_config: List[dict],
        max_iter: int = 100,
    ) -> dict:
        """
        Performs hyperparameter optimization.

        Parameters:
            reward_func (Callable[[dict], float]): The reward function to optimize.
            space_config (List[dict]): The parameter search space.
            max_iter (int): The number of iterations for optimization. Default is 100.

        Returns:
            dict: The best hyperparameters found during optimization.
        """
        if len(space_config) == 1:
            param_config = space_config[0]
            bounds = (param_config['lb'], param_config['ub'])
            _f = lambda x: reward_func({param_config['name']: x})

            result = optimize.minimize_scalar(_f,
                                              bounds=bounds,
                                              method='bounded',
                                              )
            x_min = result.x

            return {param_config['name']: x_min}

        else:
            pbounds = {param['name']: (param['lb'], param['ub']) for param in space_config}
            init_points = 5
            n_iter = max_iter - init_points
            if n_iter <= 0:
                raise ValueError(f"'max_iter' should be higher than {init_points}, got {max_iter}")

            optimizer = BayesianOptimization(f=lambda **p: -reward_func(p),
                                             pbounds=pbounds,
                                             random_state=self.random_state,
                                             verbose=0,
                                             )
            optimizer.maximize(init_points=init_points, n_iter=n_iter)

            return optimizer.max['params']
