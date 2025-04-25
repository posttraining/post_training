"""Bandit-based action selection algorithm using Upper Confidence Bound (UCB)."""

from .bandit_abstract import ContextualBanditAbstract
from .bandit_alg import SuccessiveRejects, UniformBAI, SuccessiveHalving, LUCB


class ContextualBanditSuccessiveRejects(ContextualBanditAbstract):

    """ContextualBanditSuccessiveRejects class for the post-training actions selection algorithm.
    """
    def __init__(
        self,
        n_function_types: int,
        n_iterations: int = None,
        max_iter_hyperopt: int = 5,
        max_randint: int = 1000,
        n_jobs: int = 1,
        ):

        super().__init__(n_function_types=n_function_types,
                         n_iterations=n_iterations,
                         max_iter_hyperopt=max_iter_hyperopt,
                         max_randint=max_randint,
                         n_jobs=n_jobs,
                         )

        self.bandit = SuccessiveRejects(n_arms=self.n_function_types, budget=n_iterations)


class ContextualBanditUniformBAI(ContextualBanditAbstract):

    """ContextualBanditUniformBAI class for the post-training actions selection algorithm.
    """
    def __init__(
        self,
        n_function_types: int,
        n_iterations: int = None,
        max_iter_hyperopt: int = 5,
        max_randint: int = 1000,
        n_jobs: int = 1,
        ):

        super().__init__(n_function_types=n_function_types,
                         n_iterations=n_iterations,
                         max_iter_hyperopt=max_iter_hyperopt,
                         max_randint=max_randint,
                         n_jobs=n_jobs,
                         )

        self.bandit = UniformBAI(n_arms=self.n_function_types, budget=n_iterations)


class ContextualBanditSuccessiveHalving(ContextualBanditAbstract):

    """ContextualBanditSuccessiveHalving class for the post-training actions selection algorithm.
    """
    def __init__(
        self,
        n_function_types: int,
        n_iterations: int = None,
        max_iter_hyperopt: int = 5,
        max_randint: int = 1000,
        n_jobs: int = 1,
        ):

        super().__init__(n_function_types=n_function_types,
                         n_iterations=n_iterations,
                         max_iter_hyperopt=max_iter_hyperopt,
                         max_randint=max_randint,
                         n_jobs=n_jobs,
                         )

        self.bandit = SuccessiveHalving(n_arms=self.n_function_types, budget=n_iterations)


class ContextualBanditLUCB(ContextualBanditAbstract):

    """ContextualBanditLUCB class for the post-training actions selection algorithm.
    """
    def __init__(
        self,
        n_function_types: int,
        n_iterations: int = None,
        max_iter_hyperopt: int = 5,
        max_randint: int = 1000,
        n_jobs: int = 1,
        ):

        super().__init__(n_function_types=n_function_types,
                         n_iterations=n_iterations,
                         max_iter_hyperopt=max_iter_hyperopt,
                         max_randint=max_randint,
                         n_jobs=n_jobs,
                         )

        self.bandit = LUCB(n_arms=self.n_function_types, budget=n_iterations)
