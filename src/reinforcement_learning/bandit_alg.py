"""Bandit-based action selection algorithm using Upper Confidence Bound (UCB)."""

from collections import defaultdict
import numpy as np


# regret setting


class ThompsonSampling:
    """Thompson Sampling bandit algorithm."""

    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def select(self, t):
        """Select an arm based on Thompson Sampling for minimal reward."""
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmin(samples)

    def update(self, r, k):
        """Update internal variables."""
        self.alpha[k] += r
        self.beta[k] += (1 - r)

    def best_arm(self, t):
        """Return the best arm based on minimal expected reward."""
        return np.argmin(self.alpha / (self.alpha + self.beta))


class EpsilonGreedy:
    """Epsilon-Greedy bandit algorithm."""

    def __init__(self, n_arms, eps=0.1):
        self.n_arms = n_arms
        self.eps = eps
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select(self, t):
        """Select an arm using epsilon-greedy strategy for minimal reward."""
        if np.random.rand() < self.eps:
            return np.random.randint(self.n_arms)
        return np.argmin(self.values)

    def update(self, r, k):
        """Update internal variables."""
        self.counts[k] += 1
        n = self.counts[k]
        value = self.values[k]
        self.values[k] += (r - value) / n

    def best_arm(self, t):
        """Return the arm with the lowest estimated value."""
        return np.argmin(self.values)


class UCB:
    """Upper Confidence Bound bandit algorithm."""

    def __init__(self, n_arms, alpha=1.0, eps=1e-6):
        self.n_arms = n_arms
        self.alpha = alpha
        self.eps = eps
        self.T_k = np.zeros(n_arms)
        self.mu_k = np.zeros(n_arms)

    def select(self, t):
        """Select an arm."""
        if t <= self.n_arms:
            return t - 1
        else:
            return np.argmin(self.mu_k - self.alpha * np.sqrt(np.log(t) / (self.T_k + self.eps)))

    def update(self, r, k):
        """Update internal variables."""
        self.T_k[k] += 1
        self.mu_k[k] += (r - self.mu_k[k]) / self.T_k[k]

    def best_arm(self, t):
        """Return the best arm (minimal average reward)."""
        return np.argmin(self.mu_k - self.alpha * np.sqrt(np.log(t) / (self.T_k + self.eps)))


# BAI setting


class SuccessiveRejects:
    """Successive Rejects algorithm for Best Arm Identification."""

    def __init__(self, n_arms, budget):
        self.n_arms = n_arms
        self.budget = budget
        self.counts = np.zeros(n_arms)
        self.sums = np.zeros(n_arms)
        self.active_arms = list(range(n_arms))
        self.current_phase = 0
        self.phase_lengths = self._compute_phase_lengths()
        self.phase_counter = 0

    def _compute_phase_lengths(self):
        logK = 0.5 + sum(1 / (k + 1) for k in range(self.n_arms - 1))
        return [int(self.budget / logK / (self.n_arms - k)) for k in range(self.n_arms - 1)]

    def select(self, t):
        """Select an arm."""
        if self.current_phase >= len(self.phase_lengths):
            return self.active_arms[0]  # Best arm remains
        idx = self.phase_counter % len(self.active_arms)
        return self.active_arms[idx]

    def update(self, r, k):
        """Update internal variables."""
        self.counts[k] += 1
        self.sums[k] += r
        self.phase_counter += 1

        if self.phase_counter >= self.phase_lengths[self.current_phase] * len(self.active_arms):
            avg_rewards = self.sums[self.active_arms] / (self.counts[self.active_arms] + 1e-8)
            worst_arm_idx = np.argmax(avg_rewards)
            del self.active_arms[worst_arm_idx]
            self.current_phase += 1
            self.phase_counter = 0

    def best_arm(self, t):
        """Return the best arm (minimal average reward)."""
        return self.active_arms[0] if self.active_arms else np.argmin(self.sums / (self.counts + 1e-8))


class UniformBAI:
    """Uniform allocation Best Arm Identification baseline."""

    def __init__(self, n_arms, budget):
        self.n_arms = n_arms
        self.budget = budget
        self.counts = np.zeros(n_arms)
        self.sums = np.zeros(n_arms)
        self.total_pulls = 0

    def select(self, t):
        """Select an arm."""
        return self.total_pulls % self.n_arms

    def update(self, r, k):
        """Update internal variables."""
        self.counts[k] += 1
        self.sums[k] += r
        self.total_pulls += 1

    def best_arm(self, t):
        """Return the best arm (minimal average reward)."""
        avg_rewards = self.sums / (self.counts + 1e-8)
        return np.argmin(avg_rewards)


class SuccessiveHalving:
    """Successive Halving algorithm for Best Arm Identification."""

    def __init__(self, n_arms, budget, eta=2):
        self.eta = eta
        self.rewards = defaultdict(list)
        self.rounds = []

        self.fallback_arm = 0
        arms = list(range(n_arms))

        while len(arms) > 1 and budget > 0:
            r = budget // (len(arms) * (int(np.log(n_arms) / np.log(eta)) + 1))
            if r == 0: break
            self.rounds.append((arms.copy(), r))
            budget -= len(arms) * r
            arms = arms[:max(1, len(arms) // eta)]
        if arms and budget > 0:
            self.rounds.append((arms.copy(), 1))

        self.current_round = 0
        self.round_index = 0
        self.finished = not self.rounds

        if not self.finished:
            self.active_arms, self.budget_per_arm = self.rounds[0]

    def select(self, t):
        """Select an arm."""
        if self.finished:
            return self.fallback_arm
        if self.round_index >= len(self.active_arms) * self.budget_per_arm:
            self._eliminate()
            self.current_round += 1
            if self.current_round >= len(self.rounds):
                self.finished = True
                return self.fallback_arm
            self.active_arms, self.budget_per_arm = self.rounds[self.current_round]
            self.round_index = 0
        arm = self.active_arms[self.round_index % len(self.active_arms)]
        self.round_index += 1
        return arm

    def update(self, r, k):
        """Update internal variables."""
        self.rewards[k].append(r)

    def _eliminate(self):
        means = {a: np.mean(self.rewards[a]) for a in self.active_arms if self.rewards[a]}
        self.active_arms = sorted(means, key=means.get)[:max(1, len(means) // self.eta)]

    def best_arm(self, t):
        """Return the best arm (minimal average reward)."""
        return min((a for a in self.rewards if self.rewards[a]), key=lambda a: np.mean(self.rewards[a]), default=None)


class LUCB:
    """LUCB algorithm for Best Arm Identification."""

    def __init__(self, n_arms, budget, delta=0.05):
        self.n_arms = n_arms
        self.budget = budget
        self.delta = delta
        self.rewards = defaultdict(list)

    def select(self, t):
        """Select an arm."""
        means = {a: np.mean(self.rewards[a]) if self.rewards[a] else float('inf') for a in range(self.n_arms)}
        counts = {a: len(self.rewards[a]) if self.rewards[a] else 1 for a in range(self.n_arms)}
        best = min(means, key=means.get)

        def ucb(a):
            return means[a] - np.sqrt(2 * np.log(max(1, t) / self.delta) / counts[a])

        second = min((a for a in range(self.n_arms) if a != best), key=ucb, default=best)
        return best if t % 2 == 0 else second

    def update(self, r, k):
        """Update internal variables."""
        self.rewards[k].append(r)

    def best_arm(self, t):
        """Return the best arm (minimal average reward)."""
        means = {a: np.mean(self.rewards[a]) if self.rewards[a] else float('inf') for a in range(self.n_arms)}
        return min(means, key=means.get)
