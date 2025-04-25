"""Bandit-based action selection algorithm using Upper Confidence Bound (UCB)."""

import sys
import logging
from colorama import Fore, Style


MAX_RANDINT = 1000
N_JOBS = 3


class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: Style.BRIGHT + Fore.CYAN,
        logging.WARNING: Style.BRIGHT + Fore.YELLOW,
        logging.ERROR: Style.BRIGHT + Fore.RED,
        logging.CRITICAL: Style.BRIGHT + Fore.MAGENTA,
    }

    INFO_COLORS = {
        1: Style.BRIGHT + Fore.GREEN,  # Info Level 1 -> Green
        2: Style.BRIGHT + Fore.BLUE,   # Info Level 2 -> Blue
        3: Style.BRIGHT + Fore.WHITE,  # Info Level 3 -> White
    }

    RESET = Style.RESET_ALL

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)

        if record.levelno == logging.INFO and hasattr(record, "level"):
            color = self.INFO_COLORS.get(record.level, self.RESET)

        return f"{color}{super().format(record)}{self.RESET}"


class CustomLogger(logging.Logger):
    def __init__(self, name):
        super().__init__(name)

    def info(self, msg, *args, level=3, verbosity_level=2, **kwargs):
        if verbosity_level >= level:
            extra = kwargs.get("extra", {})
            extra["level"] = level
            kwargs["extra"] = extra
            super().info(msg, *args, **kwargs)


def get_logger(name: str) -> logging.Logger:
    logging.setLoggerClass(CustomLogger)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_params_search_range(action):
    """ Generate action's parameters search range for the action type.

    Args:
        action (str): action type

    Returns:
        space_config: action space range
    """
    if action == "scale_amplitude":
        return [{'name': 'factor', 'type': 'num', 'lb': -5, 'ub': 5},
                ]

    elif action == "take_context":
        return [{'name': 'start', 'type': 'num', 'lb': 0, 'ub': 5},
                {'name': 'vert_shift', 'type': 'num', 'lb': 0, 'ub': 1},
                ]

    elif action == "piecewise_scale_high":
        return [{'name': 'threshold', 'type': 'num', 'lb': 70, 'ub': 100},
                {'name': 'factor', 'type': 'num', 'lb': -1, 'ub': 10},
                ]

    elif action == "increase_minimum_factor":
        return [{'name': 'factor', 'type': 'num', 'lb': -1, 'ub': 10},
                ]

    elif action == "increase_maximum_factor":
        return [{'name': 'factor', 'type': 'num', 'lb': -1, 'ub': 10},
                ]

    elif action == "piecewise_scale_low":
        return [{'name': 'threshold', 'type': 'num', 'lb': 0.0, 'ub': 30},
                {'name': 'factor', 'type': 'num', 'lb': -1, 'ub': 10},
                ]

    elif action == "add_linear_trend":
        return [{'name': 'slope', 'type': 'num', 'lb': -5, 'ub': 5},
                {'name': 'intercept', 'type': 'num', 'lb': -1, 'ub': 10},
                ]

    elif action == "add_linear_trend_slope":
        return [{'name': 'slope', 'type': 'num', 'lb': -5, 'ub': 5},
                ]

    elif action == "add_linear_trend_intercept":
        return [{'name': 'intercept', 'type': 'num', 'lb': -5, 'ub': 5},
                ]

    elif action == "add_seasonality":
        return [{'name': 'amplitude', 'type': 'num', 'lb': 1, 'ub': 5},
                {'name': 'period', 'type': 'num', 'lb': 5, 'ub': 100},
                ]

    elif action == "shift_series":
        return [{'name': 'shift_amount', 'type': 'num', 'lb': -200, 'ub': 200},
                ]

    elif action == "apply_smoothing":
        return [{'name': 'window_size', 'type': 'num', 'lb': 2, 'ub': 10},
                ]

    elif action == "adjust_cyclic_pattern":
        return [{'name': 'amplitude', 'type': 'num', 'lb': 0.1, 'ub': 1.0},
                {'name': 'period', 'type': 'num', 'lb': 10, 'ub': 50},
                ]

    elif action == "outlier_replacement":
        return [{'name': 'threshold', 'type': 'num', 'lb': 5, 'ub': 15},
                ]

    elif action == "quantile_adjustment":
        return [{'name': 'lower_quantile', 'type': 'num', 'lb': 0.0, 'ub': 0.25},
                {'name': 'upper_quantile', 'type': 'num', 'lb': 0.75, 'ub': 1.0},
                ]

    elif action == "piecewise_correction":
        return [{'name': 'low_threshold', 'type': 'num', 'lb': -2, 'ub': 0},
                {'name': 'high_threshold', 'type': 'num', 'lb': 0, 'ub': 2},
                {'name': 'low_factor', 'type': 'num', 'lb': 0.5, 'ub': 1.5},
                {'name': 'high_factor', 'type': 'num', 'lb': 0.5, 'ub': 1.5},
                ]

    elif action == 'shift_by_means':
        return [{'name': 'factor', 'type': 'num', 'lb': -5, 'ub': 5},
                ]

    elif action == "add_noise":
        return [{'name': 'factor_increase_std', 'type': 'num', 'lb': 10, 'ub': 30},
                ]

    else:
        raise ValueError(f"'action' name not understood, got {action}")
