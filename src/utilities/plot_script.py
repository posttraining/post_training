import numpy as np
import matplotlib.pylab as plt
import streamlit as st
from sklearn.metrics import mean_squared_error
from matplotlib.lines import Line2D  # Import Line2D to define custom legend lines

def plot_rl_per_episode(episode, mse_history, total_mse_for_episode, episode_predictions, pred, batch_x, episodes, axes, sample_idx):
    # Store total MSE for this episode
    mse_history.append((episode, total_mse_for_episode))
    episode_predictions.append(np.array([pred[0, :, k] for k in range(pred.shape[2])]))

    # Update plots after each episode
    if episode == 0:
        episode_color = "red"
        linestyle = '-'
    elif episode == episodes // 2:
        episode_color = "orange"
        linestyle = '--'
    elif episode == episodes - 1:
        episode_color = "green"
        linestyle = '-.'
    else:
        episode_color = "gray"  # Default color for other episodes
        linestyle = ':'  # Default linestyle for other episodes

    for channel_idx, _ in enumerate(range(pred.shape[2])):
        ax = axes[channel_idx]
        # Plot each episode line, but don't add them to the legend
        ax.plot(np.arange(len(batch_x[sample_idx, :, channel_idx]), len(batch_x[sample_idx, :, channel_idx]) + len(episode_predictions[-1][channel_idx])),
                episode_predictions[-1][channel_idx], color=episode_color, linestyle=linestyle, alpha=0.7, linewidth=3)
    return mse_history, episode_predictions

def final_plot(axes, fig, mse_history):
    # Final plot settings for titles and labels
    for ax in axes:
        ax.set_title("Contextual Bandit Optimization", fontsize=16, fontweight='bold', color='darkgreen')
        ax.set_xlabel("Time Steps", fontsize=12, color="black")
        ax.set_ylabel("Prediction Value", fontsize=12, color="black")
        ax.grid(True, linestyle='-', alpha=0.3)

    # Add custom legend
    custom_lines = [
        Line2D([0], [0], color="blue", lw=2, label="True values"),
        Line2D([0], [0], color="red", lw=2, linestyle='-', alpha=0.7, label="First Prediction"),
        Line2D([0], [0], color="orange", lw=2, linestyle='--', alpha=0.7, label="Middle Prediction"),
        Line2D([0], [0], color="green", lw=2, linestyle='-.', alpha=0.7, label="Last Prediction"),
    ]
    fig.legend(handles=custom_lines, loc="upper left", fontsize=10)

    # plt.tight_layout(pad=2)
    st.session_state.contextual_bandit_fig = fig  # Save to session state to avoid re-plotting

    # Display the main figure
    st.pyplot(fig)

    # Plot MSE history (total MSE across all channels for each episode)
    fig_mse, ax_mse = plt.subplots(figsize=(12, 6))
    ax_mse.plot([entry[0] for entry in mse_history], [entry[1] for entry in mse_history], color="blue", marker='o', label="Total MSE")
    ax_mse.set_title("Total MSE Across Episodes", fontsize=18, fontweight='bold', color='darkred')
    ax_mse.set_xlabel("Episode", fontsize=14)
    ax_mse.set_ylabel("Total MSE", fontsize=14)
    ax_mse.legend()
    # plt.tight_layout()
    st.pyplot(fig_mse)
def save_predictions(batch_x, init_pred, true, best_pred, feedback):
    if feedback:
        np.save('results/context_feedback.npy', batch_x.cpu().numpy())
        np.save('results/initial_feedback.npy', init_pred)
        np.save('results/ground_truth_feedback.npy', true)
        np.save('results/optimized_feedback.npy', best_pred)
    else:
        np.save('results/context.npy', batch_x.cpu().numpy())
        np.save('results/initial.npy', init_pred)
        np.save('results/ground_truth.npy', true)
        np.save('results/optimized.npy', best_pred)

def plot_results_with_episode_colors(pred, true, batch_x, args, title="Forecast Results", mse_history=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set a gradient color scheme for the episodes (from light to dark)
    cmap = plt.cm.viridis  # Using the viridis color map (you can choose other color maps)
    norm = plt.Normalize(vmin=0, vmax=len(mse_history)-1)  # Normalize based on number of episodes

    # Plot the initial context and ground truth once
    ax.plot(range(len(batch_x)), batch_x, label="Context", color="gray", linestyle="--", linewidth=2)
    ax.plot(range(len(batch_x), len(batch_x) + len(true)), true, label="Ground Truth", color="blue", linewidth=3)
    ax.axvline(x=len(batch_x), color="black", linestyle="--", label="Forecast Start", linewidth=2)

    for episode in range(len(mse_history)):
        # Use more intense color for later episodes
        color = cmap(norm(episode))
        
        # Plot predictions for each episode using a gradually changing color
        pred_step = pred[episode]
        ax.plot(range(len(batch_x), len(batch_x) + len(pred_step)), pred_step, 
                label=f"Prediction {episode+1} (MSE: {mse_history[episode][1]:.4f})", 
                color=color, linewidth=2)
        
        # Highlight the improvement in MSE with a different annotation color
        ax.annotate(f"Episode {episode + 1} - MSE: {mse_history[episode][1]:.4f}", 
                    xy=(len(batch_x) + len(pred_step), pred_step[-1]), 
                    xytext=(len(batch_x) + len(pred_step) + 10, pred_step[-1] + 0.1),
                    arrowprops=dict(facecolor=color, shrink=0.05), fontsize=10, color=color)

    # Beautify the plot
    initial_mse = mse_history[0][1]  # Initial MSE (before any transformations)
    final_mse = mse_history[-1][1]  # Final MSE (after transformations)
    
    # Set the title with both initial and final MSE
    ax.set_title(f"Contextual Bandit Optimization\nInitial MSE: {initial_mse:.4f} | Final MSE: {final_mse:.4f}",
                 fontsize=22, fontweight='bold', color='darkgreen')
    ax.set_xlabel("Time Steps", fontsize=16, color="black")
    ax.set_ylabel("Prediction Value", fontsize=16, color="black")
    ax.grid(True, linestyle='-', alpha=0.3)
    
    # Add a legend
    ax.legend(loc='upper left', fontsize=12, frameon=False)
    
    # Display the plot
    plt.tight_layout()
    st.pyplot(fig)

# Function to handle the test prediction visualization (show only final predictions)
def plot_final_test_predictions(best_pred, true, batch_x, args):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the final prediction after applying the best transformations
    ax.plot(range(len(batch_x)), batch_x, label="Context", color="gray", linestyle="--", linewidth=2)
    ax.plot(range(len(batch_x), len(batch_x) + len(true)), true, label="Ground Truth", color="blue", linewidth=3)
    
    # Plot the final prediction and highlight improvement
    ax.plot(range(len(batch_x), len(batch_x) + len(best_pred)), best_pred, 
            label="Final Prediction (Best MSE)", color="red", linewidth=3)

    # Compute the MSE for comparison
    initial_mse = mean_squared_error(true.flatten(), best_pred.flatten())
    ax.annotate(f"Final Test MSE: {initial_mse:.4f}", 
                xy=(len(batch_x) + len(best_pred), best_pred[-1]), 
                xytext=(len(batch_x) + len(best_pred) + 10, best_pred[-1] + 0.1),
                arrowprops=dict(facecolor='red', shrink=0.05), fontsize=10, color="red")
    
    # Beautify the plot
    ax.set_title(f"Final Test Prediction: MSE = {initial_mse:.4f}", 
                 fontsize=22, fontweight='bold', color='darkgreen')
    ax.set_xlabel("Time Steps", fontsize=16, color="black")
    ax.set_ylabel("Prediction Value", fontsize=16, color="black")
    ax.grid(True, linestyle='-', alpha=0.3)
    
    # Add a legend
    ax.legend(loc='upper left', fontsize=12, frameon=False)
    
    # Display the plot
    plt.tight_layout()
    st.pyplot(fig)


# Function for plotting results
def plot_results(pred, true, batch_x, args, title="Forecast Results"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(batch_x)), batch_x, label="Context", color="gray", linestyle="--")
    ax.plot(range(len(batch_x), len(batch_x) + len(true)), true, label="Ground Truth", color="blue")
    ax.plot(range(len(batch_x), len(batch_x) + len(pred)), pred, label="Predictions", color="red")
    ax.axvline(x=len(batch_x), color="black", linestyle="--", label="Forecast Start")
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

    
def plot_predictions_vs_ground_truth(pred, ground_truth, batch_x, num_samples=20, channel=0, start=0, save_path=None):
    """
    Plots predictions vs. ground truth for a given channel (e.g., channel 0) and for the first num_samples.
    Assumes the input shape is (N, T, D) where:
    - N: number of samples
    - T: number of time steps
    - D: number of channels
    """
    num_samples = min(num_samples, pred.shape[0] - start)  # Adjust num_samples based on the start index
    forecast_start = len(batch_x[0, :, channel])
    
    # Create a figure with 4 rows and 5 columns of subplots
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()  # Flatten the 2D axes array for easy indexing

    for i in range(start, start + num_samples):
        ax = axes[i - start]
        
        # Plot the historical (context) data
        ax.plot(np.arange(len(batch_x[i, :, channel])), batch_x[i, :, channel], color='g', linestyle='-', linewidth=2)
        
        # Plot the prediction and ground truth for the forecasted period
        ax.plot(np.arange(len(batch_x[i, :, channel]), len(batch_x[i, :, channel]) + len(pred[i, :, channel])), pred[i, :, channel], color='b', linestyle='--', linewidth=2)
        ax.plot(np.arange(len(batch_x[i, :, channel]), len(batch_x[i, :, channel]) + len(pred[i, :, channel])), ground_truth[i, :, channel], color='r', linewidth=2)
        
        # Add vertical line to mark the start of the forecast
        ax.axvline(forecast_start, color='k', linestyle=':')
        
        ax.set_title(f"Sample {i + 1}")
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend(loc='best')

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Show the plot
    st.pyplot(fig)