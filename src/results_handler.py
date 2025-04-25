import pandas as pd
import os

RESULTS_CSV = "final_results.csv"

def load_previous_results():
    """Load existing results from CSV if available."""
    if os.path.exists(RESULTS_CSV):
        return pd.read_csv(RESULTS_CSV).to_dict(orient="records")
    return []

def append_and_save_results(results_list, filename=RESULTS_CSV):
    """
    Append new results to an existing CSV file, aggregate correctly, and avoid overwriting.
    """
    df_new = pd.DataFrame(results_list)

    if df_new.empty:
        print("No results yet.")
        return

    # Load existing results (if any)
    if os.path.exists(filename):
        df_existing = pd.read_csv(filename)
    else:
        df_existing = pd.DataFrame()

    # Concatenate new results with existing ones
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)

    # Ensure aggregation happens **only within the same Dataset, Model, and Method**
    df_grouped = df_combined.groupby(["Dataset", "Window Size", "Prediction Horizon", "Model", "method"], as_index=False).agg(
        {
            "Initial MSE": ["mean", "std"],
            "Final MSE": ["mean", "std"],
            "Improvement % MSE": ["mean", "std"],
            "Initial MAE": ["mean", "std"],
            "Final MAE": ["mean", "std"],
            "Improvement % MAE": ["mean", "std"],
            "RL Exploration Time (s)": ["mean", "std"]
        }
    )

    # Flatten MultiIndex columns
    df_grouped.columns = [' '.join(col).strip() for col in df_grouped.columns.values]

    # Save back to CSV
    df_grouped.to_csv(filename, index=False)
    print(f"\n[INFO] Aggregated results saved to {filename}\n")

    # Print the latest aggregated table
    print(df_grouped)
