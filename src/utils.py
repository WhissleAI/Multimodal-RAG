import functools

def log_execution(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f">>> Starting {func.__name__} ... ")
        result = func(*args, **kwargs)
        print(f">>> Successfully finished {func.__name__}!")
        return result
    return wrapper