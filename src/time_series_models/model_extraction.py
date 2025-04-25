from src.time_series_models.time_series_algorithms import Exp_Main


import os
import importlib.util
import tempfile

import torch

class Args:
    def __init__(self, **kwargs):
        # Set default values
        self.task_name = kwargs.get('task_name', "long_term_forecast")
        self.model = kwargs.get('model', 'default_model_name')
        self.is_training = kwargs.get('is_training', 1)
        self.model_id = kwargs.get('model_id', 'default_model_name')
        self.train_loader = kwargs.get('train_loader', None)
        self.train_dataset = kwargs.get('train_dataset', None)
        self.val_loader = kwargs.get('val_loader', None)
        self.val_dataset = kwargs.get('val_dataset', None)
        self.test_loader = kwargs.get('test_loader', None)
        self.test_dataset = kwargs.get('test_dataset', None)
        self.dim = kwargs.get('dim', None)
        self.train = kwargs.get('train', None)
        self.val = kwargs.get('val', None)
        self.test = kwargs.get('test', None)
        self.train_y = kwargs.get('train_y', None)
        self.val_y = kwargs.get('val_y', None)
        self.test_y = kwargs.get('test_y', None)
        self.data_path = kwargs.get('data_path', 'ETTh1.csv')
        self.dataset_name = kwargs.get('dataset_name', 'test')
        self.seq_len = kwargs.get('seq_len', 24)  # Default window_size, can be overwritten
        self.pred_len = kwargs.get('pred_len', 24)  # Default prediction length
        self.prediction_horizon = kwargs.get('predisction_horizon', 96)  # Default prediction length
        self.train_epochs = kwargs.get('train_epochs', 10)
        self.batch_size = kwargs.get('batch_size', 10)
        self.n_jobs = kwargs.get('n_jobs', 1)

        # Hardware related arguments
        self.use_gpu = kwargs.get('use_gpu', torch.cuda.is_available())  # Defaults to use GPU if available
        if self.use_gpu:
            self.gpu = kwargs.get('gpu', torch.cuda.current_device())  # Automatically selects the first available GPU
        else:
            self.gpu = -1  # No GPU selected

        self.seed = kwargs.get('seed', 2024)  # Random seed

        # Data related arguments
        self.data = kwargs.get('data', "custom")  # Default to custom data type
        self.root_path = kwargs.get('root_path', "/home/hcherkaoui/src/post_training_forecasting_official/data/")  # XXX
        self.features = kwargs.get('features', "M")
        self.target = kwargs.get('target', "OT")
        self.freq = kwargs.get('freq', "h")
        self.checkpoints = kwargs.get('checkpoints', "/home/hcherkaoui/src/post_training_forecasting_official/checkpoints/")  # XXX

        # Model-related parameters
        self.label_len = kwargs.get('label_len', 0)  # Default label length
        self.text_emb = kwargs.get('text_emb', 96)
        self.qwen_model_path = kwargs.get('qwen_model_path', "../../../../../../data/yumeng/qwen1.5-0.5B-chat/")
        self.seasonal_patterns = kwargs.get('seasonal_patterns', "Monthly")
        self.inverse = kwargs.get('inverse', False)
        self.mask_rate = kwargs.get('mask_rate', 0.25)
        self.anomaly_ratio = kwargs.get('anomaly_ratio', 0.25)
        self.expand = kwargs.get('expand', 2)
        self.d_conv = kwargs.get('d_conv', 4)
        self.top_k = kwargs.get('top_k', 5)
        self.num_kernels = kwargs.get('num_kernels', 6)
        self.enc_in = kwargs.get('enc_in', self.dim)
        self.dec_in = kwargs.get('dec_in', self.dim)
        self.c_out = kwargs.get('c_out', self.dim)
        self.d_model = kwargs.get('d_model', 512)
        self.n_heads = kwargs.get('n_heads', 8)
        self.e_layers = kwargs.get('e_layers', 2)
        self.d_layers = kwargs.get('d_layers', 1)
        self.d_ff = kwargs.get('d_ff', 2048)
        self.moving_avg = kwargs.get('moving_avg', 25)
        self.factor = kwargs.get('factor', 3)
        self.distil = kwargs.get('distil', False)
        self.dropout = kwargs.get('dropout', 0.1)
        self.embed = kwargs.get('embed', "timeF")
        self.activation = kwargs.get('activation', "gelu")
        self.output_attention = kwargs.get('output_attention', False)
        self.channel_independence = kwargs.get('channel_independence', 1)
        self.decomp_method = kwargs.get('decomp_method', "moving_avg")
        self.use_norm = kwargs.get('use_norm', True)
        self.down_sampling_layers = kwargs.get('down_sampling_layers', 0)
        self.down_sampling_window = kwargs.get('down_sampling_window', 1)
        self.down_sampling_method = kwargs.get('down_sampling_method', "avg")
        self.seg_len = kwargs.get('seg_len', self.seq_len)
        self.num_workers = kwargs.get('num_workers', 10)
        self.itr = kwargs.get('itr', 1)
        self.patience = kwargs.get('patience', 5)
        self.learning_rate = kwargs.get('learning_rate', 0.0001)
        self.des = kwargs.get('des', "test")
        self.loss = kwargs.get('loss', "MSE")
        self.lradj = kwargs.get('lradj', "type1")
        self.use_amp = kwargs.get('use_amp', False)
        self.huggingface_token = kwargs.get('huggingface_token', None)
        self.use_multi_gpu = kwargs.get('use_multi_gpu', False)
        self.text_path = kwargs.get('text_path', None)
        self.prompt_weight = kwargs.get('prompt_weight', 0.0)
        self.type_tag = kwargs.get('type_tag', '#F#')
        self.text_len = kwargs.get('text_len', 3)
        self.llm_model = kwargs.get('llm_model', 'Qwen')
        self.llm_dim = kwargs.get('llm_dim', 1024)
        self.llm_layers = kwargs.get('llm_layers', 6)
        self.learning_rate2 = kwargs.get('learning_rate2', 1e-2)
        self.learning_rate3 = kwargs.get('learning_rate3', 1e-3)
        self.pool_type = kwargs.get('pool_type', 'avg')
        self.date_name = kwargs.get('date_name', 'end_date')
        self.addHisRate = kwargs.get('addHisRate', 0.5)
        self.init_method = kwargs.get('init_method', 'normal')
        self.learning_rate_weight = kwargs.get('learning_rate_weight', 0.0001)
        self.save_name = kwargs.get('save_name', 'result_longterm_forecast')
        self.use_fullmodel = kwargs.get('use_fullmodel', 0)
        self.use_closedllm = kwargs.get('use_closedllm', 0)
        self.devices = kwargs.get('devices', '0,1,2,3')
        self.p_hidden_dims = kwargs.get('p_hidden_dims', [128, 128])
        self.p_hidden_layers = kwargs.get('p_hidden_layers', 2)
        self.method = kwargs.get('method', 'UCB')

def load_custom_model(model_file):
    # Use tempfile to create a temporary file for the uploaded model
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
        temp_model_filepath = temp_file.name  # Get the temporary file path

        # Write the contents of the uploaded model file to the temp file
        with open(model_file, 'r') as file:
            content = file.read()
            temp_file.write(content.encode())  # Write the content to the temp file

    # Load the model file dynamically using importlib
    spec = importlib.util.spec_from_file_location("custom_model", temp_model_filepath)
    custom_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_model)

    # Delete the temporary model file after loading
    os.remove(temp_model_filepath)

    # Ensure the model file contains the class 'exp'
    if not hasattr(custom_model, 'exp'):
        raise ValueError("The uploaded model file does not contain a class named 'exp'.")

    # Get the exp class
    exp_class = custom_model.exp

    # Debugging: Print the class and its methods
    print("Loaded exp class:", exp_class)
    print("Methods in exp class:", dir(exp_class))

    # Ensure the class has the required methods
    required_methods = ['train', 'validation', 'test']
    for method in required_methods:
        if not hasattr(exp_class, method):
            raise ValueError(f"The class 'exp' does not contain the required method '{method}'.")

    return exp_class


def define_model(train_loader, val_loader, test_loader, config):

    if config["model_file"]:
        exp_class = load_custom_model(config["model_file"])
        args = Args(
            task_name="short_term_forecast",
            model=config['model_name'],
            train=train,
            train_y=train_y,
            val=val,
            val_y=val_y,
            test=test,
            test_y=test_y,
            train_loader=None,
            train_dataset=None,
            val_loader=None,
            val_dataset=None,
            test_loader=None,
            test_dataset=None,
            batch_size=config['batch_size'],
            epochs=1,
            seq_len= config["window_size"],
            pred_len= config["prediction_horizon"],
            label_len=config['label_len'],
            n_samples=config['n_samples'],
            num_trials=config['num_trials'],
            method=config["method"],
            n_jobs=config['n_jobs'],
        )

        exp = exp_class(args)
        exp.train(args)

    else:
        for batch_idx, (X_batch, y_batch, seq_x_mark_batch, seq_y_mark_batch) in enumerate(train_loader):

            # Get the shape of one sample (first sample in the batch)
            dim = X_batch[0].shape[1]  # Get the shape of the first sample in the batch
            break  # Exit after the first batch (no need to iterate through all batches)

        args = Args(
            task_name="short_term_forecast",
            model=config["models"],
            train_loader=train_loader,
            train_dataset=None,
            val_loader=val_loader,
            val_dataset=None,
            test_loader=test_loader,
            test_dataset=None,
            batch_size=config["batch_size"],
            epochs=1,
            seq_len= config["window_size"],
            pred_len= config["prediction_horizon"],
            label_len=config["label_len"],
            n_samples=config["n_samples"],
            num_trials=config["num_trials"],
            dim=config["feature_size"],
            method=config["method"],
            n_jobs=config['n_jobs'],
        )
        exp = Exp_Main(args)

    return exp, args