import re
from typing import Optional, Any
import numpy as np
from openai import OpenAI
from src.reinforcement_learning import (ContextualBanditSuccessiveRejects, ContextualBanditUniformBAI, ContextualBanditSuccessiveHalving,
                                        ContextualBanditLUCB, genetic_algorithm_reinforcement, contextual_bandit_reinforcement,
                                        contextual_bandit_random)


def extract_apply_method(generated_class_code):
    """
    Extracts the `apply` method from the generated class code.
    This method will capture the entire logic of the `apply` function.
    """
    # Regular expression to match the apply method and its contents
    apply_method_pattern = re.compile(r'def\s+apply\(.*?\):\s*(.*?)\s*else\s+raise', re.DOTALL)
    match = apply_method_pattern.search(generated_class_code)

    if match:
        # Return the captured content inside the apply method (i.e., the method body)
        return match.group(1)
    return ''  # Return empty string if no match is found


# Function to build the prompt for the model based on user feedback
def build_feedback_prompt(feedback):
    """
    Build a prompt for the model to generate Python class and parameters based on the feedback.
    The prompt ensures only the class and the function are generated and are easy to extract.
    Additionally, it ensures that the `function_type` in the class is the same as the `action` in the `generate_random_params_for_action` function.
    """
    return f"""
    Given the following feedback about a time series prediction model:

    Feedback: "{feedback}"

    Please generate a Python class called `GenericFunction` that represents a transformation function based on the feedback. The class should include the following:

    1. A constructor (`__init__`) that accepts two parameters:
       - `function_type`: The type of transformation (the one put by the user). This should match the `action` parameter used in the `generate_random_params_for_action` function.
       - `params`: A dictionary containing parameters specific to the transformation.

    2. An `apply` method that takes two arguments:
       - (a) a time series (prediction) which should be modified of size N (sample size), T(time step), D(dimension)
       - (b) the context vector (batch_x) of size N(sample size), T(time step), D (dimension)
       The method should apply the transformation inferred from the user's feedback either on the prediction, the context vector, or both and output only the time series after prediction (only one output).
       The output of the transfrmation should be the same type, dtype and shape as the predictions in input.

    Additionally, please generate a function called `generate_random_params_for_action` that takes two parameters:
       - `action`: The type of transformation (the one put by the user). This should be the same as the `function_type` defined in the class.
       - `batch_x`: the time series of the context vector. Note that batch_x is a tensor of size N, T, D

    The function should return a dictionary of parameters based on the user's feedback.

    Please ensure that the generated class and function are valid Python code and clearly separated. The class and the function should be output as follows:

    --- START OF GENERATED CODE ---

    # Class Definition:
    class GenericFunction:
    <class-body>

    # Function Definition:
    def generate_random_params_for_action(action, batch_x):
    <function-body>

    --- END OF GENERATED CODE ---

    Please ensure that the generated code is valid, formatted correctly, and that the `function_type` in the `GenericFunction` class matches the `action` in the `generate_random_params_for_action` function.
    """

def get_generated_code(feedback):
    client = OpenAI(
    base_url='https://api.deepseek.com',
    api_key='sk-2054088250254f14bb701e2e45148af4',
)
    prompt = build_feedback_prompt(feedback)

    # Call OpenAI API to generate the code
    response = client.chat.completions.create(
        model="deepseek-chat",  # Update to the appropriate model
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.5,  # Adjust temperature for creativity level
        max_tokens=500,
    )

    # Extract and return the generated code
    return response.choices[0].message.content


def get_generic_function_class(generated_code):
    """
    Extracts the GenericFunction class from the generated code.
    Returns the class definition or None if the class is not found.
    """
    print(generated_code)  # Check the generated code

    # Regex to match 'class GenericFunction' and everything until the next marker
    class_pattern = re.compile(r'# Class Definition:.*?(class\s+GenericFunction.*?)(?=\n# Function Definition:)', re.DOTALL)
    match = class_pattern.search(generated_code)

    if match:
        return match.group(1)  # Returns the class definition including its body

    return None

def get_generate_random_parameters_function(generated_code):
    """
    Extracts the generate_random_params_for_action function from the generated code.
    Returns the function definition or None if the function is not found.
    """
    # Regex to extract everything between # Function Definition: and --- END OF GENERATED CODE ---
    # Allow for any content between the markers, including imports or other code.
    function_pattern = re.compile(r'# Function Definition:.*?(def\s+generate_random_params_for_action.*?)(?=\n--- END OF GENERATED CODE ---)', re.DOTALL)
    match = function_pattern.search(generated_code)

    # If no match is found, return None.
    if match:
        return match.group(1)  # Returns the function definition including its body

    return None  # Return None if the function is not found

def get_function_types_from_class(class_code):
    """
    This function extracts the transformation types from the class's `apply` method logic.
    We look for 'if self.function_type == <function_name>' statements in the code.
    """
    function_types = []

    # Regular expression to match function types after `if self.function_type ==`
    pattern = r"if\s+self\.function_type\s*==\s*['\"]([^'\"]+)['\"]"
    matches = re.findall(pattern, class_code)

    # Add all unique matches to the function types list
    function_types.extend(matches)

    return function_types
import re
def sanitize_generated_code(code):
    """Sanitize the generated code to remove unwanted characters like backticks or any syntax issues."""
    if code:
        # Remove backticks or unwanted characters (e.g., from markdown formatting)
        sanitized_code = code.replace('```', '').strip()
        return sanitized_code
    return code

def execute_generated_code(code):
    """Executes generated code while ensuring imports are handled separately."""
    sanitized_code = sanitize_generated_code(code)
    if not sanitized_code:
        return  # Avoid executing empty or invalid code blocks

    # Split code into lines
    lines = sanitized_code.split("\n")

    # Filter out import statements (they were already executed separately)
    non_import_code = "\n".join([line for line in lines if not line.strip().startswith(("import", "from"))])

    # Execute the remaining code
    exec(non_import_code, globals())
def extract_import_statements(*code_blocks):
    """Extracts import statements from multiple code blocks and returns them as a single string."""
    import_lines = set()  # Use a set to avoid duplicate imports
    for code in code_blocks:
        if code:
            lines = code.split("\n")
            for line in lines:
                if line.strip().startswith("import") or line.strip().startswith("from"):
                    import_lines.add(line.strip())  # Store unique import lines
    return "\n".join(import_lines)

def handle_feedback(
    exp: Any,
    args: Any,
    pred: Optional[np.ndarray] = None,
    true: Optional[np.ndarray] = None,
    batch_x: Optional[np.ndarray] = None,
    streamlit: bool = False,
    method: str = 'UCB',
    feedback_text: str = None,
    ):

    if feedback_text is None:
        feedback_text = input("Provide feedback on the model's predictions: ")

    # Generate code from feedback
    generated_code = get_generated_code(feedback_text)
    Generic_class_code = get_generic_function_class(generated_code)
    random_params_code = get_generate_random_parameters_function(generated_code)

    print(Generic_class_code)
    print(random_params_code)

    # Extract and execute imports first to make them available globally
    imports_code = extract_import_statements(Generic_class_code, random_params_code)
    exec(imports_code, globals())

    # Now execute the class and function definitions
    execute_generated_code(Generic_class_code)
    execute_generated_code(random_params_code)

    # Capture dynamically created GenericFunction class and generate_random_params_for_action function
    new_generic_function_class = globals().get('GenericFunction')
    generate_random_params_function = globals().get('generate_random_params_for_action')

    # Ensure both class and function exist before proceeding
    if new_generic_function_class is None:
        raise ValueError("The GenericFunction class was not defined correctly.")

    if generate_random_params_function is None:
        raise ValueError("The generate_random_params_for_action function was not defined correctly.")

    # Capture function types
    function_types = get_function_types_from_class(generated_code)

    N_ITERATIONS = 15
    MAX_ITER_HYPEROPT = 10

    # Choose reinforcement learning algorithm
    if method == 'random':
        rl_algorithm = contextual_bandit_random

    elif args.method == "SR-HPO":
        rl_algorithm = ContextualBanditSuccessiveRejects(n_function_types=len(function_types),
                                                         n_iterations=N_ITERATIONS,
                                                         max_iter_hyperopt=MAX_ITER_HYPEROPT,
                                                         n_jobs=args.n_jobs,
                                                         )

    elif args.method == "U-HPO":
        rl_algorithm = ContextualBanditUniformBAI(n_function_types=len(function_types),
                                                  n_iterations=N_ITERATIONS,
                                                  max_iter_hyperopt=MAX_ITER_HYPEROPT,
                                                  n_jobs=args.n_jobs,
                                                  )

    elif args.method == "SH-HPO":
        rl_algorithm = ContextualBanditSuccessiveHalving(n_function_types=len(function_types),
                                                         n_iterations=N_ITERATIONS,
                                                         max_iter_hyperopt=MAX_ITER_HYPEROPT,
                                                         n_jobs=args.n_jobs,
                                                         )

    elif args.method == "LUCB-HPO":
        rl_algorithm = ContextualBanditLUCB(n_function_types=len(function_types),
                                            n_iterations=N_ITERATIONS,
                                            max_iter_hyperopt=MAX_ITER_HYPEROPT,
                                            n_jobs=args.n_jobs,
                                            )
    elif method == 'Genetic':
        rl_algorithm = genetic_algorithm_reinforcement

    elif method == 'PPO':
        rl_algorithm = contextual_bandit_reinforcement

    else:
        raise ValueError(f"Unsupported method: '{method}'; valid method are 'random', 'SR-HPO',"
                         f" 'U-HPO', 'SH-HPO', 'Genetic' or 'PPO'")

    # Rerun RL algorithm with the newly defined function class and parameters function
    function_set_per_channel, best_mse, best_pred, true, batch_x = rl_algorithm(
        exp=exp,
        args=args,
        episodes=10,
        alpha=0,
        gamma=0,
        epsilon=0,
        action_budget=20,
        improvement_threshold=0.0,
        generic_function_class=new_generic_function_class,
        generate_random_parameters_function=generate_random_params_function,
        function_types=function_types,
        pred=pred,
        true=true,
        batch_x=batch_x,
        streamlit=True,
    )

    return new_generic_function_class, best_pred, best_mse, true, function_set_per_channel
