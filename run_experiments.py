import argparse
import yaml
import itertools
import subprocess
import os
from datetime import datetime
from src.utils import log

def load_config(config_path):
    """Load the YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def run_experiment(args, dataset, model, window_size, pred_horizon, batch_size, method, trial):
    """Run a single experiment instance."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"results/logs/{model}_{os.path.basename(dataset).split('.')[0]}_ws{window_size}_ph{pred_horizon}_trial{trial}_{timestamp}.log"
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)
    
    command = [
        "python", "main.py",
        "--config", args.config,
        "--training", str(args.training),
        "--human_feedback" if args.human_feedback else "",
        "--train_path", dataset,
        "--model", model,
        "--window_size", str(window_size),
        "--prediction_horizon", str(pred_horizon),
        "--batch_size", str(batch_size),
        "--method", method
    ]
    command = [c for c in command if c]  # Remove empty strings
    
    log(f"🚀 Running Experiment: {command}")
    
    with open(log_dir, "w") as log_file:
        subprocess.run(command, stdout=log_file, stderr=log_file)

def main():
    """Main function to iterate over hyperparameters and datasets."""
    parser = argparse.ArgumentParser(description="Run batch experiments with varying hyperparameters.")
    parser.add_argument("--config", type=str, default="configs/experiments_config.yaml", help="Path to configuration file")
    parser.add_argument("--training", type=bool, default=True, help="Enable model training")
    parser.add_argument("--human_feedback", action="store_true", help="Use human feedback")
    args = parser.parse_args()
    
    config = load_config(args.config)
    experiments = list(itertools.product(
        config["data_name"],
        config["models"],
        config["window_sizes"],
        config["prediction_horizons"],
        config["batch_sizes"],
        config["methods"],
        range(1, config["num_trials"] + 1)
    ))
    
    log(f"🔬 Running {len(experiments)} experiments...")
    
    for dataset, model, window_size, pred_horizon, batch_size, method, trial in experiments:
        run_experiment(args, dataset, model, window_size, pred_horizon, batch_size, method, trial)

if __name__ == "__main__":
    main()
