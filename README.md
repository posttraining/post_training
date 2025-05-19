# Time Series Model Post-Training and Human Feedback Exploration

This repository provides a framework to apply post-training exploration on any existing time series model. It allows users to fine-tune model predictions through human feedback and a **reinforcement learning** or a **contextual bandit** approach. This dynamic feedback loop enhances the model’s adaptability and improves prediction accuracy over time.

## Project Goal

The primary objective of this project is to create an interactive environment for users to explore and refine the predictions of pre-trained time series models. The feedback mechanism enables model adjustments based on human-provided insights, enhancing the overall predictive power.

## Features

- **Post-Training Exploration**: Works with pre-trained time series models to refine predictions.
- **Contextual Bandit Framework**: Dynamically adjusts model predictions based on actions.
- **Human Feedback Integration**: Users can guide model improvements interactively.
- **Streamlit Interface**: A user-friendly web interface for testing adjustments and viewing results.

---

## Installation

### Clone the Repository

```bash
cd post_training_forecasting_official
```
### Set Up the Environment
Ensure you have Python 3.7+ and conda installed. Then, install the necessary dependencies:
```bash
conda env create -f environment.yml
```
one can also set-up the environment using:
```bash
pip install -r requirements.txt
```

### Activate the environment:
```bash
conda activate post_training_env
```

## Usage
Running the Post-Training Process
To execute the post-training script, use the following command:
```bash
python main.py --config configs/default_config.yaml --train_path <path_to_train_data> --model <model_name> --window_size <window_size> --prediction_horizon <prediction_horizon> --batch_size <batch_size> --method <rl_method>
```
We support SOTA time series models including PatchTST, SegRNN etc, for the full list of supporting models, please check the list under section "Supported Time Series Models". <br>
We support reinforcement learning and contextual bandit methods includes: random, SR-HPO, U-HPO, SH-HPO, PPO and Genetic.

## Running the Streamlit App
To launch the Streamlit interface:

```bash
streamlit run app_test.py
```
## Running Batch Experiments
To execute batch experiments using the predefined configurations:

```bash
python run_experiments.py --config configs/experiments_config.yaml
```

### Supported Time Series Models

Transformer-backbone: TimesNet; Autoformer; Transformer; Nonstationary_Transformer; FEDformer; Informer; Reformer; ETSformer; PatchTST; Pyraformer; Crossformer; iTransformer.<br>
MLP-backbone: Koopa, TiDE, FreTS, TimeMixer, TSMixer.<br>
Other: LightTS, MICN, FiLM, SegRNN.

## Folder Structure
```bash
post_training/
│
├── app_test.py             # Streamlit interface to test the method
├── environment.yml         # Conda environment setup
├── main.py                 # Main script for executing post-training
├── run_experiments.py      # Script for running batch experiments
│
├── configs/                # YAML files to reproduce paper results
├── experiments/            # Jupyter notebooks for experiment reproduction
├── data/                   # Dataset storage
├── results/                # Plots and results from experiments
│
└── src/                    # Core source code
    ├── data_extraction.py   # Data handling functions
    ├── model_extraction.py  # Model selection and initialization
    ├── reinforcement_learning.py  # Contextual bandit logic
    ├── human_feedback.py    # Functions for human feedback processing
    ├── testing.py           # Evaluation and visualization
    ├── results_handler.py   # Managing experiment results
    └── utils.py             # Helper functions
```
## Workflow Overview
### Model Initialization

Load data and initialize the selected time series model.
### Exploration Phase

Apply different post-training actions (e.g., adjusting amplitude, trends, etc.).
Use the contextual bandit algorithm to optimize predictions dynamically.
### Human Feedback Integration

Users provide corrective actions based on observed model performance.
Example feedback: "Increase the prediction amplitude by 5-10%."
### Evaluation and Visualization

Compare model performance before and after feedback.
Generate plots and metrics for analysis.
Contributions
Feel free to contribute by submitting pull requests or opening issues for bugs and feature suggestions.

## License
This project is released under the MIT License.

## Contact
For any inquiries or discussions, please contact the repository maintainer.
