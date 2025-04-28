import streamlit as st
import os
import numpy as np
import torch
from src.data_extraction import load_data_sota
from src.utilities.plot_script import plot_predictions_vs_ground_truth
from src.reinforcement_learning import explore_instructions
from src.human_feedback import handle_feedback
from src.time_series_models.model_extraction import define_model
from src.testing import test_and_visualize_on_new_data
from src.actions_definition import GenericFunction

os.environ["CURL_CA_BUNDLE"] = ""

def save_uploaded_file(uploaded_file, folder="temp_data"):
    if uploaded_file:
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None

def initialize_session_state():
    keys = ['feedback', 'phase', 'best_pred_rl', 'true_feedback', 'batch_x_feedback', 'best_function_set_per_channel', 'best_function_set_per_channel_feedback']
    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = None
    if 'phase' not in st.session_state or st.session_state.phase is None:
        st.session_state.phase = 1  # Ensure phase is initialized properly

def configure_sidebar():
    st.sidebar.header("Configuration")
    train_file = st.sidebar.file_uploader("Upload Training CSV", type=["csv"], key="train_file")
    model_name = st.sidebar.selectbox(
            "Model Name", [
                "Crossformer", "DLinear", "ETSformer", "FEDformer","FiLM", "FreTS", "Informer", "Koopa", 
                "LightTS", "MICN", "Nonstationnary_Transformer", "PatchTST", "Pyraformer", "Reformer", "SegRNN", 
                "TSMixer", "TiDE", "TimeMixer", "Transformer", "TimesNet", "iTransformer"
            ], index=0
        )   
    model_file = st.sidebar.file_uploader("Upload Custom Model (optional, model.py)", type=["py"], key="model_file")
 
    window_size = st.sidebar.number_input("Sliding Window Size (T)", min_value=1, max_value=100, value=96)
    prediction_horizon = st.sidebar.number_input("Prediction Horizon (T)", min_value=1, max_value=200, value=144)
    method = st.sidebar.selectbox("Method", ["random", "SH-HPO", "Genetic", "PPO"])    
    explore_button = st.sidebar.button("Explore Instructions")
    return train_file, model_name, window_size, prediction_horizon, method, model_file, explore_button

def display_results():
    if st.session_state.best_pred_rl is not None:
        best_pred = st.session_state.best_pred_rl
        ground_truth = st.session_state.true_feedback
        batch_x = st.session_state.batch_x_feedback

        channel_idx = st.selectbox("Select the channel", list(range(best_pred.shape[2])))
        start_idx = st.slider("Select start sample index", 0, best_pred.shape[0] - 1, 0, 1)

        plot_predictions_vs_ground_truth(best_pred, ground_truth, batch_x, num_samples=20, channel=channel_idx, start=start_idx)

def main():
    initialize_session_state()
    train_file, model_name, window_size, prediction_horizon, method, model_file, explore_button = configure_sidebar()
    
    if st.session_state.phase >= 1:
        if train_file and explore_button and st.session_state.best_pred_rl is None:
            config = {"data_name": save_uploaded_file(train_file), "models": model_name,
                      "num_trials": 1, "window_size": window_size, "prediction_horizon": prediction_horizon,
                      "label_len": 0, "n_samples": 100, "feature_size":7, "model_file": model_file,
                      "batch_size": 32, "method": method, "n_jobs": 7, "episodes": 5}
            train_loader, val_loader, test_loader = load_data_sota(config)
            exp, args = define_model(train_loader, val_loader, test_loader, config)
            exp.train(args)
            pred_rl, true, batch_x, best_function_set_per_channel = explore_instructions(exp, args, streamlit=True)

            st.session_state.update({
                "exp": exp, "args": args, "best_pred_rl": pred_rl,
                "true_feedback": true, "batch_x_feedback": batch_x,
                "best_function_set_per_channel": best_function_set_per_channel, "phase": 2
            })
        display_results()
    
    if st.session_state.phase >= 2:
        st.subheader("Provide Feedback")
        st.session_state.feedback = st.text_area("Enter your feedback", "")
        if st.button("Submit Feedback"):
            if st.session_state.feedback:
                new_class, best_pred, _, _, best_function_set_per_channel_feedback = handle_feedback(
                    st.session_state.exp, st.session_state.args, streamlit=True,
                    pred=st.session_state.best_pred_rl, true=st.session_state.true_feedback,
                    batch_x=torch.tensor(st.session_state.batch_x_feedback), feedback_text=st.session_state.feedback
                )
                st.session_state.update({
                    "new_class": new_class, "best_function_set_per_channel_feedback": best_function_set_per_channel_feedback,
                    "phase": 3
                })
            else:
                st.warning("Please enter feedback before submitting.")

    if st.session_state.phase >= 3:
        if st.button("Finalize Prediction"):
            test_and_visualize_on_new_data(
                st.session_state.exp, st.session_state.args,
                st.session_state.best_function_set_per_channel,
                st.session_state.best_function_set_per_channel_feedback,
                GenericFunction, st.session_state.new_class
            )
            st.success("Model tested with feedback applied!")
            st.session_state.phase = 1  # Reset to allow fresh runs

if __name__ == "__main__":
    main()
