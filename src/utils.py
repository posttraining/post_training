import time

def time_execution(func, *args, **kwargs):
    """Measure the execution time of a function and return its result with the duration."""
    start_time = time.time()
    result = func(*args, **kwargs)  # Execute the function
    end_time = time.time()
    
    duration = end_time - start_time
    return result, duration

def log(message):
    """Prints a formatted log message."""
    print(f"\n📝 {message}\n" + "-" * 50)