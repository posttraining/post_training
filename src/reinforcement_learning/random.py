"""Random action selection algorithm using contextual bandit with random transformations."""

import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.metrics import mean_squared_error
from src.utilities.plot_script import plot_rl_per_episode, final_plot, save_predictions
from src.actions_definition import generate_random_params_for_action


def contextual_bandit_random(
    exp,
    args,
    episodes,
    alpha,
    gamma,
    epsilon,
    action_budget=20,
    improvement_threshold=0.0,
    generic_function_class=None,
    generate_random_parameters_function=generate_random_params_for_action,
    function_types=None,
    save_dir='./results',
    pred=None,
    true=None,
    batch_x=None,
    streamlit=False
):
    """
    Explore and selction de best actions with a random sampling algorithm.

    Parameters:
        exp: Experiment object with a `.validation()` method.
        args: Argument object with model ID.
        episodes (int): Number of episodes to explore actions.
        alpha (float): Learning rate (not used here).
        gamma (float): Discount factor (not used here).
        epsilon (float): Exploration rate (not used here).
        action_budget (int): Number of transformation attempts per action.
        improvement_threshold (float): Required improvement in MSE to accept a transformation.
        generic_function_class (callable): Class that applies transformations.
        generate_random_parameters_function (callable): Function to generate random parameters for a given action.
        function_types (list): List of available transformation functions.
        save_dir (str): Directory to save plots and predictions.
        pred (np.ndarray): Initial predictions (optional).
        true (np.ndarray): Ground truth values (optional).
        batch_x (np.ndarray): Input features.
        streamlit (bool): Whether to enable Streamlit visualizations.

    Returns:
        tuple: (
            function_set_per_channel (list of tuples),
            best_mse (float),
            best_pred (np.ndarray),
            true (np.ndarray),
            pred (np.ndarray)
        )
    """
    best_mse = float('inf')
    best_pred = None
    mse_history = []
    episode_predictions = []
    function_set_per_channel = []

    if pred is None:
        _, pred, true, batch_x = exp.validation(args.model_id, test=1)

    init_pred = copy.deepcopy(pred)
    n_channels = pred.shape[2]
    sample_idx = 0  # for visualization
    mse_values = np.mean((true - pred) ** 2, axis=(1, 2))

    print('Initial MSE:', mean_squared_error(pred.flatten(), true.flatten()))

    # Prepare output directories
    plots_dir = os.path.join(save_dir, "plots")
    predictions_dir = os.path.join(save_dir, "predictions")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    # Setup plot style
    sns.set(style="whitegrid", palette="muted")
    ncols = 2
    nrows = (n_channels + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6 * nrows), sharex=True, sharey=True)
    axes = axes.flatten()

    for k in range(n_channels):
        ax = axes[k]
        ax.plot(batch_x[sample_idx, :, k].cpu(), color="gray", linestyle="--", linewidth=2, label="Input")
        ax.plot(
            np.arange(len(batch_x[sample_idx, :, k].cpu()), len(batch_x[sample_idx, :, k].cpu()) + len(true[sample_idx, :, k])),
            true[sample_idx, :, k],
            color="blue", linewidth=3, label="True values"
        )
        ax.axvline(x=len(batch_x[sample_idx, :, k].cpu()), color="black", linestyle="--", linewidth=2)

    for episode in range(episodes):
        if episode == 0:
            best_pred = pred.copy()
        else:
            pred = best_pred.copy()

        total_improvement = 0

        for channel_idx in range(n_channels):
            channel_true = true[:, :, channel_idx]
            channel_pred = pred[:, :, channel_idx]
            initial_mse = mean_squared_error(channel_true.flatten(), channel_pred.flatten())
            channel_batch = batch_x[:, :, channel_idx].reshape(batch_x.shape[0], batch_x.shape[1], 1).cpu().numpy()

            best_action = None
            best_params = None
            action_performance = []

            for action in function_types:
                if action in ['log_transform', 'normalize_predictions', 'align_distribution']:
                    params, _ = generate_random_parameters_function(action, batch_x)
                    new_pred = generic_function_class(action, params).apply(channel_pred.reshape(pred.shape[0], pred.shape[1], 1), channel_batch)
                    mse = mean_squared_error(channel_true.flatten(), new_pred.flatten())
                    action_performance.append((action, mse, params))
                else:
                    for _ in range(action_budget):
                        params = generate_random_parameters_function(action, batch_x)
                        new_pred = generic_function_class(action, params).apply(channel_pred.reshape(pred.shape[0], pred.shape[1], 1), channel_batch)
                        mse = mean_squared_error(channel_true.flatten(), new_pred.flatten())
                        action_performance.append((action, mse, params))

            action_performance.sort(key=lambda x: x[1])
            best_action, best_mse_channel, best_params = action_performance[0]

            if best_mse_channel < initial_mse - improvement_threshold:
                improved_pred = generic_function_class(best_action, best_params).apply(channel_pred.reshape(pred.shape[0], pred.shape[1], 1), channel_batch)
                channel_pred = improved_pred
                total_improvement += initial_mse - best_mse_channel
                function_set_per_channel.append((channel_idx, best_action, best_params))

            pred[:, :, channel_idx] = channel_pred.reshape(pred.shape[0], pred.shape[1])
            best_pred[:, :, channel_idx] = channel_pred.reshape(pred.shape[0], pred.shape[1])

        total_mse = mean_squared_error(pred.flatten(), true.flatten())

        if streamlit:
            mse_history, episode_predictions = plot_rl_per_episode(
                episode, mse_history, total_mse, episode_predictions,
                pred, batch_x, episodes, axes, sample_idx
            )
        else:
            mse_history.append((episode, total_mse))
            episode_predictions.append(np.array([pred[0, :, k] for k in range(n_channels)]))

            episode_color = "red" if episode == 0 else "green" if episode == episodes - 1 else "gray"
            linestyle = '-' if episode == 0 else '-.' if episode == episodes - 1 else ':'

            for k in range(n_channels):
                axes[k].plot(
                    np.arange(len(batch_x[sample_idx, :, k].cpu()), len(batch_x[sample_idx, :, k].cpu()) + len(episode_predictions[-1][k])),
                    episode_predictions[-1][k],
                    color=episode_color,
                    linestyle=linestyle,
                    alpha=0.7,
                    linewidth=3
                )

    print('Final MSE:', mean_squared_error(pred.flatten(), true.flatten()))

    if streamlit:
        final_plot(axes, fig, mse_history)
        save_predictions(batch_x, init_pred, true, best_pred, feedback=True)
    else:
        for ax in axes[:n_channels]:
            ax.set_title("Contextual Bandit Optimization", fontsize=16, fontweight='bold', color='darkgreen')
            ax.set_xlabel("Time Steps", fontsize=12)
            ax.set_ylabel("Prediction Value", fontsize=12)
            ax.grid(True, linestyle='-', alpha=0.3)

        custom_lines = [
            Line2D([0], [0], color="blue", lw=2, label="True values"),
            Line2D([0], [0], color="red", lw=2, linestyle='-', alpha=0.7, label="First Prediction"),
            Line2D([0], [0], color="orange", lw=2, linestyle='--', alpha=0.7, label="Middle Prediction"),
            Line2D([0], [0], color="green", lw=2, linestyle='-.', alpha=0.7, label="Last Prediction"),
        ]
        fig.legend(handles=custom_lines, loc="upper left", fontsize=10)

        plt.savefig(os.path.join(plots_dir, "final_prediction_plot.png"))

        fig_mse, ax_mse = plt.subplots(figsize=(12, 6))
        ax_mse.plot([e for e, _ in mse_history], [m for _, m in mse_history], color="blue", marker='o', label="Total MSE")
        ax_mse.set_title("Total MSE Across Episodes", fontsize=18, fontweight='bold', color='darkred')
        ax_mse.set_xlabel("Episode", fontsize=14)
        ax_mse.set_ylabel("Total MSE", fontsize=14)
        ax_mse.legend()
        fig_mse.savefig(os.path.join(plots_dir, "mse_history_plot.png"))

        np.savetxt(os.path.join(predictions_dir, "predictions.csv"), best_pred.reshape(-1, best_pred.shape[-1]), delimiter=",")
        print(f"Predictions saved to {os.path.join(predictions_dir, 'predictions.csv')}")

    return function_set_per_channel, best_mse, best_pred, true, pred
