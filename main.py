import time
import argparse
import yaml
from src.data_extraction import load_data_sota
from src.reinforcement_learning import explore_instructions
from src.human_feedback import handle_feedback
from src.testing import visualize_improvements
from src.results_handler import load_previous_results, append_and_save_results
from src.utils import log, time_execution
from src.time_series_models.model_extraction import define_model
from src.actions_definition import GenericFunction


def load_config(config_path):
    """Load the YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def store_results(trial_results):
    """Stores and saves results."""
    append_and_save_results([trial_results])
    log("✅ Results saved successfully!")


def override_config_with_args(config, args):
    """Override config parameters with command-line arguments if provided."""
    if args.train_path:
        config["data_name"] = args.train_path

    if args.model:
        config["models"] = args.model

    if args.window_size:
        config["window_size"] = args.window_size

    if args.prediction_horizon:
        config["prediction_horizon"] = args.prediction_horizon

    if args.batch_size:
        config["batch_size"] = args.batch_size

    if args.method:
        config["method"] = args.method

    if args.n_jobs:
        config["n_jobs"] = args.n_jobs

    return config

def main():
    """Main function to run the experiment pipeline."""
    parser = argparse.ArgumentParser(description="Run experiments with RL and human feedback.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to configuration file")
    parser.add_argument("--training", type=bool, default=True, help="Enable model training")
    parser.add_argument("--human_feedback", action="store_true", help="Use human feedback")

    # New Arguments for Dynamic Configuration
    parser.add_argument("--train_path", type=str, help="Path to training dataset")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--method", type=str, help="RL type name")
    parser.add_argument("--window_size", type=int, help="Window size for training")
    parser.add_argument("--prediction_horizon", type=int, help="Prediction horizon")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--episodes", type=int, help="Nb of episodes")
    parser.add_argument("--n-jobs", type=int, help="Nb of CPUs")

    args = parser.parse_args()

    # Load config and override with command-line arguments
    config = load_config(args.config)
    config = override_config_with_args(config, args)

    results_list = load_previous_results()

    # Load dataset and model
    train_loader, val_loader, test_loader = load_data_sota(config)
    exp, exp_args = define_model(train_loader, val_loader, test_loader, config)

    # Run trials
    for trial in range(config["num_trials"]):
        log(f"🚀 Running Trial {trial + 1}/{config['num_trials']}")

        # Train if needed
        if args.training:
            log("🔧 Training model...")
            exp.train(args)

        print(exp_args)
        print(exp)

        # Explore instructions with RL
        log("🤖 Exploring instructions using RL framework...")

        t0 = time.time()
        best_pred, true, batch_x, best_function_set_per_channel = explore_instructions(exp=exp,
                                                                                       args=exp_args,
                                                                                       episodes=args.episodes,
                                                                                       )
        rl_duration = time.time() - t0

        log(f"⏳ RL Exploration Time: {rl_duration:.2f} seconds")

        # Apply human feedback if enabled
        best_function_set_per_channel_feedback = []
        if args.human_feedback:
            log("🧑‍🔬 Applying human feedback...")
            _, _, _, _, best_function_set_per_channel_feedback = handle_feedback(exp, args, pred=best_pred, true=true, batch_x=batch_x)

        # Evaluate performance
        log("📊 Evaluating improvements...")
        initial_mse, final_mse, improvement_percentage_mse, initial_mae, final_mae, improvement_percentage_mae = visualize_improvements(
            exp, exp_args, best_function_set_per_channel, best_function_set_per_channel_feedback, GenericFunction, GenericFunction
        )

        # Store results
        trial_results = {
            "Dataset": config["data_name"],
            "Window Size": config["window_size"],
            "Prediction Horizon": config["prediction_horizon"],
            "Model": config["models"],
            "method": config["method"],
            "Initial MSE": initial_mse,
            "Final MSE": final_mse,
            "Improvement % MSE": improvement_percentage_mse,
            "Initial MAE": initial_mae,
            "Final MAE": final_mae,
            "Improvement % MAE": improvement_percentage_mae,
            "RL Exploration Time (s)": rl_duration
        }
        store_results(results_list)

if __name__ == "__main__":
    main()
