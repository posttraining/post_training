import random
import numpy as np

class GenericFunction:
    def __init__(self, function_type, params):
        self.function_type = function_type
        self.params = params

    def apply(self, series, batch_x):
        if series.size == 0:
            raise ValueError("Input series is empty.")
        series = np.nan_to_num(series, nan=0.0, posinf=1e10, neginf=-1e10)
        if self.function_type == 'scale_amplitude':
            adjustment_factor = 1 + (self.params['factor'] / 100.0)
            return series * adjustment_factor
        elif self.function_type == 'swap_series':
            return self._swap_cos_sin(series)
        elif self.function_type == 'take_context':
            return self._apply_vertical_shift_on_context(series, batch_x, self.params['start'], self.params['vert_shift'])
        elif self.function_type == 'piecewise_scale_high':
            return self._apply_piecewise_scaling_high(series, self.params['threshold'], self.params['factor'])
        elif self.function_type == 'piecewise_scale_low':
            return self._apply_piecewise_scaling_low(series, self.params['threshold'], self.params['factor'])
        elif self.function_type == 'add_linear_trend_slope':
            return self._add_linear_trend_slope(series, self.params['slope'])
        elif self.function_type == 'add_linear_trend_intercept':
            return self._add_linear_trend_intercept(series, self.params['intercept'])
        elif self.function_type == 'increase_minimum_factor':
            return self._apply_piecewise_scaling_high(series, 10, self.params['factor'])
        elif self.function_type == 'increase_maximum_factor':
            return self._apply_piecewise_scaling_high(series, 90, self.params['factor'])
        elif self.function_type == 'add_noise':
            return self._add_noise(series, self.params['factor_increase_std'])

        else:
            raise ValueError(f"Unknown function type: {self.function_type}")

    def _add_noise(self, series, std_var):
        """
        Adds noise to the series based on a standard deviation
        which is a percentage of the series values, independently for each channel and sample.
        """
        # noise_std will be the factor (as a percentage) of the series values
        noise_std = std_var / 100.0

        # Generate noise for each sample, time step, and channel independently
        noise = np.random.normal(loc=0.0, scale=noise_std * np.abs(series), size=series.shape)

        # Return the series with the noise added
        return series + noise
    def _swap_cos_sin(self, series):
        """
        Transform a time series from cosine to sine (or vice versa),
        by centering, multiplying by -1, and then re-centering.

        :param series: The time series to transform.

        :return: The transformed time series (swapped phase).
        """

        # Option 1: Center the series by subtracting the mean
        mean = np.mean(series, axis=1, keepdims=True)
        series_centered = series - mean

        # Invert the signal (multiply by -1)
        transformed_series = -series_centered

        # Optionally, shift back to the original mean
        transformed_series += mean

        return transformed_series
    def _apply_vertical_shift_on_context(self, batch_x, series, start, vert_shift_max):
        """
        Apply vertical shift on a selected context segment of batch_x, based on 'start' index.

        :param batch_x: The full batch of time series data (N, T, D).
        :param series: The series to be transformed (N, d, D).
        :param start: The starting index from which to select a context (0 to T-d).
        :param vert_shift_max: The maximum vertical shift to apply to the context.

        :return: The transformed batch_x with applied vertical shift.
        """
        N, T, D = batch_x.shape
        d = series.shape[1]  # Length of the context segment

        # Ensure valid vertical shift range
        vert_shift = np.random.randint(-vert_shift_max, vert_shift_max + 1)

        # Initialize transformed batch as a copy of batch_x
        transformed_batch = np.copy(batch_x)

        for i in range(N):
            # Select the context segment from batch_x
            selected_context = batch_x[i, start:start+d, :]

            # Apply the vertical shift
            if vert_shift != 0:
                if vert_shift > 0:
                    if vert_shift < T:
                        transformed_batch[i, start+vert_shift:start+d+vert_shift, :] = selected_context
                        transformed_batch[i, start:start+vert_shift, :] = 0  # Fill the earlier part with zeros (or another strategy)
                elif vert_shift < 0:
                    if -vert_shift < T:
                        transformed_batch[i, start+vert_shift:start+d+vert_shift, :] = selected_context
                        transformed_batch[i, start+d+vert_shift:, :] = 0  # Fill the later part with zeros (or another strategy)

        return transformed_batch
    def _add_linear_trend_slope(self, series, slope_percentage):
        max_values = np.max(series, axis=1, keepdims=True)
        min_values = np.min(series, axis=1, keepdims=True)
        range_values = max_values - min_values
        slope = (slope_percentage / 100.0) * range_values
        time_indices = np.arange(series.shape[1])
        time_indices_expanded = time_indices.reshape(1, -1, 1)
        trend = slope * time_indices_expanded
        return series + trend

    def _add_linear_trend_intercept(self, series, intercept_percentage):
        mean_values = np.mean(series, axis=1, keepdims=True)
        intercept = (intercept_percentage / 100.0) * mean_values
        return series + intercept

    def _apply_piecewise_scaling_high(self, series, threshold, factor):
        threshold_value = np.percentile(series, threshold)
        below_quantile_mask = series <= threshold_value
        series[below_quantile_mask] *= (1 + factor / 100.0)
        return series

    def _apply_piecewise_scaling_low(self, series, threshold, factor):
        threshold_value = np.percentile(series, threshold)
        above_quantile_mask = series > threshold_value
        series[above_quantile_mask] *= (1 + factor / 100.0)
        return series

def generate_random_params_for_action(action, true):
    """
    Generate random parameters based on the action type.
    This function now returns both random values for parameters and the parameter ranges.
    """
    param_ranges = {}  # Dictionary to store ranges for each parameter

    if action == "scale_amplitude":
        param_ranges = {'factor': (-5, 5)}  # Range for the scale factor
        return {'factor': random.uniform(*param_ranges['factor'])}
    elif action == "take_context":
        param_ranges = {'start': (0, 5), 'vert_shift': (0, 1)}  # Range for the scale factor
        return {'start': np.random.randint(*param_ranges['start']),
                'vert_shift': 0}

    elif action == "piecewise_scale_high":
        param_ranges = {'threshold': (70, 100), 'factor': (-1, 10)}  # Ranges for threshold and factor
        return {'threshold': random.uniform(*param_ranges['threshold']),
                'factor': random.uniform(*param_ranges['factor'])}

    elif action == "increase_minimum_factor":
        param_ranges = {'factor': (-1, 10)}  # Range for the factor
        return {'factor': random.uniform(*param_ranges['factor'])}

    elif action == "increase_maximum_factor":
        param_ranges = {'factor': (-1, 10)}  # Range for the factor
        return {'factor': random.uniform(*param_ranges['factor'])}

    elif action == "piecewise_scale_low":
        param_ranges = {'threshold': (0.0, 30), 'factor': (-1, 10)}  # Ranges for threshold and factor
        return {'threshold': random.uniform(*param_ranges['threshold']),
                'factor': random.uniform(*param_ranges['factor'])}

    elif action == "add_linear_trend":
        param_ranges = {'slope': (-5, 5), 'intercept': (-1, 10)}  # Range for slope and intercept
        return {'slope': random.uniform(*param_ranges['slope']),
                'intercept': random.uniform(*param_ranges['intercept'])}

    elif action == "add_linear_trend_slope":
        param_ranges = {'slope': (-5, 5)}  # Range for slope
        return {'slope': random.uniform(*param_ranges['slope'])}

    elif action == "add_linear_trend_intercept":
        param_ranges = {'intercept': (-5, 5)}  # Range for intercept
        return {'intercept': random.uniform(*param_ranges['intercept'])}

    elif action == "add_seasonality":
        param_ranges = {'amplitude': (5, 5), 'period': (5, 100)}  # Range for amplitude and period
        return {'amplitude': random.uniform(*param_ranges['amplitude']),
                'period': random.randint(*param_ranges['period'])}

    elif action == "shift_series":
        param_ranges = {'shift_amount': (-200, 200)}  # Range for shift amount
        return {'shift_amount': random.randint(*param_ranges['shift_amount'])}

    elif action == "apply_smoothing":
        param_ranges = {'window_size': (2, 10)}  # Range for window size
        return {'window_size': random.randint(*param_ranges['window_size'])}

    elif action == "adjust_cyclic_pattern":
        param_ranges = {'amplitude': (0.1, 1.0), 'period': (10, 50)}  # Range for amplitude and period
        return {'amplitude': random.uniform(*param_ranges['amplitude']),
                'period': random.randint(*param_ranges['period'])}

    elif action == "outlier_replacement":
        param_ranges = {'threshold': (5, 15)}  # Range for threshold
        return {'threshold': random.uniform(*param_ranges['threshold'])}

    elif action == 'align_distribution':
        return true  # No parameters required for this transformation

    elif action == "normalize_predictions":
        return true  # No parameters for this transformation

    elif action == "quantile_adjustment":
        param_ranges = {'lower_quantile': (0.0, 0.25), 'upper_quantile': (0.75, 1.0)}  # Range for quantiles
        return {'lower_quantile': random.uniform(*param_ranges['lower_quantile']),
                'upper_quantile': random.uniform(*param_ranges['upper_quantile'])}

    elif action == "piecewise_correction":
        param_ranges = {'low_threshold': (-2, 0), 'high_threshold': (0, 2), 'low_factor': (0.5, 1.5), 'high_factor': (0.5, 1.5)}  # Ranges for thresholds and factors
        return {'low_threshold': random.uniform(*param_ranges['low_threshold']),
                'high_threshold': random.uniform(*param_ranges['high_threshold']),
                'low_factor': random.uniform(*param_ranges['low_factor']),
                'high_factor': random.uniform(*param_ranges['high_factor'])}

    elif action == 'shift_by_means':
        param_ranges = {'factor': (-5, 5)}  # Range for factor
        return {'factor': random.uniform(*param_ranges['factor'])}

    elif action == "add_noise":
        # Here, we generate a random 'factor_increase_std' which is a percentage (e.g., 0.1 to 10%)
        param_ranges = {'factor_increase_std': (10, 30)}  # The factor is a percentage of the series values
        return {'factor_increase_std': random.uniform(*param_ranges['factor_increase_std'])}

    else:
        return {}


def generate_random_params_for_action_rl(action, true):
    """
    Generate random parameters based on the action type.
    This function now returns both random values for parameters and the parameter ranges.
    """
    param_ranges = {}  # Dictionary to store ranges for each parameter

    if action == "scale_amplitude":
        param_ranges = {'factor': (-5, 5)}  # Range for the scale factor
        return {'factor': random.uniform(*param_ranges['factor'])}, param_ranges
    elif action == "take_context":
        param_ranges = {'start': (0, 5), 'vert_shift': (0, 1)}  # Range for the scale factor
        return {'start': np.random.randint(*param_ranges['start']),
                'vert_shift': 0}, param_ranges

    elif action == "piecewise_scale_high":
        param_ranges = {'threshold': (70, 100), 'factor': (-1, 10)}  # Ranges for threshold and factor
        return {'threshold': random.uniform(*param_ranges['threshold']),
                'factor': random.uniform(*param_ranges['factor'])}, param_ranges

    elif action == "increase_minimum_factor":
        param_ranges = {'factor': (-1, 10)}  # Range for the factor
        return {'factor': random.uniform(*param_ranges['factor'])}, param_ranges

    elif action == "increase_maximum_factor":
        param_ranges = {'factor': (-1, 10)}  # Range for the factor
        return {'factor': random.uniform(*param_ranges['factor'])}, param_ranges

    elif action == "piecewise_scale_low":
        param_ranges = {'threshold': (0.0, 30), 'factor': (-1, 10)}  # Ranges for threshold and factor
        return {'threshold': random.uniform(*param_ranges['threshold']),
                'factor': random.uniform(*param_ranges['factor'])}, param_ranges

    elif action == "add_linear_trend":
        param_ranges = {'slope': (-5, 5), 'intercept': (-1, 10)}  # Range for slope and intercept
        return {'slope': random.uniform(*param_ranges['slope']),
                'intercept': random.uniform(*param_ranges['intercept'])}, param_ranges

    elif action == "add_linear_trend_slope":
        param_ranges = {'slope': (-5, 5)}  # Range for slope
        return {'slope': random.uniform(*param_ranges['slope'])}, param_ranges

    elif action == "add_linear_trend_intercept":
        param_ranges = {'intercept': (-5, 5)}  # Range for intercept
        return {'intercept': random.uniform(*param_ranges['intercept'])}, param_ranges

    elif action == "add_seasonality":
        param_ranges = {'amplitude': (5, 5), 'period': (5, 100)}  # Range for amplitude and period
        return {'amplitude': random.uniform(*param_ranges['amplitude']),
                'period': random.randint(*param_ranges['period'])}, param_ranges

    elif action == "shift_series":
        param_ranges = {'shift_amount': (-200, 200)}  # Range for shift amount
        return {'shift_amount': random.randint(*param_ranges['shift_amount'])}, param_ranges

    elif action == "apply_smoothing":
        param_ranges = {'window_size': (2, 10)}  # Range for window size
        return {'window_size': random.randint(*param_ranges['window_size'])}, param_ranges

    elif action == "adjust_cyclic_pattern":
        param_ranges = {'amplitude': (0.1, 1.0), 'period': (10, 50)}  # Range for amplitude and period
        return {'amplitude': random.uniform(*param_ranges['amplitude']),
                'period': random.randint(*param_ranges['period'])}, param_ranges

    elif action == "outlier_replacement":
        param_ranges = {'threshold': (5, 15)}  # Range for threshold
        return {'threshold': random.uniform(*param_ranges['threshold'])}, param_ranges

    elif action == 'align_distribution':
        return true, {}  # No parameters required for this transformation

    elif action == "normalize_predictions":
        return true, {}  # No parameters for this transformation

    elif action == "quantile_adjustment":
        param_ranges = {'lower_quantile': (0.0, 0.25), 'upper_quantile': (0.75, 1.0)}  # Range for quantiles
        return {'lower_quantile': random.uniform(*param_ranges['lower_quantile']),
                'upper_quantile': random.uniform(*param_ranges['upper_quantile'])}, param_ranges

    elif action == "piecewise_correction":
        param_ranges = {'low_threshold': (-2, 0), 'high_threshold': (0, 2), 'low_factor': (0.5, 1.5), 'high_factor': (0.5, 1.5)}  # Ranges for thresholds and factors
        return {'low_threshold': random.uniform(*param_ranges['low_threshold']),
                'high_threshold': random.uniform(*param_ranges['high_threshold']),
                'low_factor': random.uniform(*param_ranges['low_factor']),
                'high_factor': random.uniform(*param_ranges['high_factor'])}, param_ranges

    elif action == 'shift_by_means':
        param_ranges = {'factor': (-5, 5)}  # Range for factor
        return {'factor': random.uniform(*param_ranges['factor'])}, param_ranges

    elif action == "add_noise":
        # Here, we generate a random 'factor_increase_std' which is a percentage (e.g., 0.1 to 10%)
        param_ranges = {'factor_increase_std': (10, 30)}  # The factor is a percentage of the series values
        return {'factor_increase_std': random.uniform(*param_ranges['factor_increase_std'])}, param_ranges

    else:
        return {}, {}