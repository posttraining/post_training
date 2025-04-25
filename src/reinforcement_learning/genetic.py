"""Genetic-based action selection algorithm for optimizing time series predictions."""

import random
import numpy as np
from sklearn.metrics import mean_squared_error
from src.actions_definition import GenericFunction
from src.reinforcement_learning.utils import discretize_actions


def genetic_algorithm_reinforcement(
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
    population_size=10,
    generations=2,
    mutation_rate=0.1,
    pred=None,
    true=None,
    batch_x=None,
    streamlit=False,
):
    """
    Explore and selction de best actions with a genetic algorithm.

    Parameters:
        exp: Experiment object containing the validation method.
        args: Argument object with the model ID.
        episodes (int): Number of episodes (not used).
        alpha, gamma, epsilon (float): RL-related params (not used).
        action_budget (int): Maximum number of actions per channel (not enforced here).
        improvement_threshold (float): Threshold to consider improvements (not used).
        generic_function_class (type): Placeholder (unused).
        generate_random_parameters_function (callable): Placeholder (unused).
        function_types (list): List of function names to explore.
        save_dir (str): Directory to save results.
        batch_size (int): Placeholder.
        replay_buffer_size (int): Placeholder.
        population_size (int): Number of individuals in the population.
        generations (int): Number of generations to evolve.
        mutation_rate (float): Probability of mutation per child.
        pred (np.ndarray): Initial predictions.
        true (np.ndarray): Ground truth values.
        batch_x (np.ndarray): Input batch data.
        streamlit (bool): Whether Streamlit mode is on (unused).

    Returns:
        tuple: (function_set_per_channel, best_mse, best_pred, true, batch_x)
    """
    num_discretizations = 20
    params_ranges = {
        "scale_amplitude": {"factor": (-1.5, 1.5)},
        "piecewise_scale_high": {"threshold": (0, 1), "factor": (-1.5, 1.5)},
        "piecewise_scale_low": {"threshold": (0, 1), "factor": (-1.5, 1.5)},
        "add_linear_trend_slope": {"slope": (-1.5, 1.5)},
        "add_linear_trend_intercept": {"intercept": (-1.5, 1.5)},
        "increase_minimum_factor": {"factor": (-1.5, 1.5)},
        "increase_maximum_factor": {"factor": (-1.5, 1.5)}
    }

    discrete_actions = discretize_actions(function_types, params_ranges, num_discretizations)

    if pred is None:
        _, pred, true, batch_x = exp.validation(args.model_id, test=1)

    initial_predictions = pred
    best_pred = np.zeros_like(pred)
    best_mse = mean_squared_error(pred.flatten(), true.flatten())
    function_set_per_channel = []

    for channel_idx in range(initial_predictions.shape[2]):
        print(f"Optimizing for Channel {channel_idx}...")

        true_channel = true[:, :, channel_idx:channel_idx + 1]
        pred_channel = initial_predictions[:, :, channel_idx:channel_idx + 1]

        population = initialize_population(population_size, discrete_actions, function_types)

        channel_best_mse = mean_squared_error(pred_channel.flatten(), true_channel.flatten())
        channel_best_pred = pred_channel.copy()
        improvement_found = False

        for generation in range(generations):
            print(f"Generation {generation + 1}/{generations} for Channel {channel_idx}")

            fitness_scores = evaluate_population(
                population,
                true_channel,
                pred_channel,
                discrete_actions,
                function_types,
                params_ranges
            )

            selected_parents = select_parents(population, fitness_scores)
            offspring = crossover(selected_parents, population_size - len(selected_parents))
            mutate(offspring, mutation_rate, discrete_actions, function_types, params_ranges)
            population = selected_parents + offspring

            best_individual, best_score = get_best_individual(population, fitness_scores)

            if best_score < channel_best_mse:
                improvement_found = True
                channel_best_mse = best_score
                channel_best_pred = apply_action_sequence(best_individual, pred_channel)

        if improvement_found:
            print("Improved:", best_score, channel_best_mse)
            for action in best_individual:
                for func_type, params in action:
                    function_set_per_channel.append((channel_idx, func_type, params))
            best_pred[:, :, channel_idx] = channel_best_pred.reshape(channel_best_pred.shape[:2])
        else:
            best_pred[:, :, channel_idx] = pred_channel.reshape(pred_channel.shape[:2])

    print("Final MSE:", mean_squared_error(best_pred.flatten(), true.flatten()))
    return function_set_per_channel, best_mse, best_pred, true, batch_x


def initialize_population(population_size, discrete_actions, function_types):
    """Initialize a population of random action sequences."""
    return [
        [generate_random_action_sequence(discrete_actions) for _ in function_types]
        for _ in range(population_size)
    ]


def generate_random_action_sequence(discrete_actions, length=10):
    """Generate a random action sequence of given length."""
    return [random.choice(discrete_actions) for _ in range(length)]


def evaluate_population(population, true, initial_predictions, discrete_actions, function_types, params_ranges):
    """Evaluate fitness (MSE) of each individual in the population."""
    return [
        mean_squared_error(
            apply_action_sequence(individual, initial_predictions).flatten(),
            true.flatten()
        )
        for individual in population
    ]


def apply_action_sequence(action_sequence, initial_predictions):
    """Apply a sequence of transformations to the predictions."""
    final_predictions = initial_predictions.copy()
    for channel_sequence in action_sequence:
        for function_type, params in channel_sequence:
            generic_function = GenericFunction(function_type, params)
            final_predictions = generic_function.apply(final_predictions, final_predictions)
    return final_predictions


def select_parents(population, fitness_scores, num_parents=None):
    """Select the top individuals as parents (elitism)."""
    if num_parents is None:
        num_parents = len(population) // 2
    sorted_population = [ind for _, ind in sorted(zip(fitness_scores, population), key=lambda x: x[0])]
    return sorted_population[:num_parents]


def crossover(parents, num_offspring):
    """Generate offspring from parent pairs using one-point crossover."""
    offspring = []
    while len(offspring) < num_offspring:
        parent1, parent2 = random.sample(parents, 2)
        child = [
            random.choice([p1_seq, p2_seq])
            for p1_seq, p2_seq in zip(parent1, parent2)
        ]
        offspring.append(child)
    return offspring


def mutate(offspring, mutation_rate, discrete_actions, function_types, params_ranges):
    """Randomly mutate some children in the population."""
    for individual in offspring:
        if random.random() < mutation_rate:
            mutation_idx = random.randint(0, len(individual) - 1)
            individual[mutation_idx] = generate_random_action_sequence(discrete_actions)


def get_best_individual(population, fitness_scores):
    """Return the best individual and its fitness score."""
    best_idx = int(np.argmin(fitness_scores))
    return population[best_idx], fitness_scores[best_idx]
