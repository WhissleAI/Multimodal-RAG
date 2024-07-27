import functools
from datetime import datetime
import pytz
import os

def log_execution(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f">>> Starting {func.__name__} ... ")
        result = func(*args, **kwargs)
        print(f">>> Successfully finished {func.__name__}!")
        return result
    return wrapper

def experiment_init():
    # current_time = datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d-%H-%M-%S")
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    outpath = f"out/{current_time}"
    os.makedirs(f"out/{current_time}")
    os.system(f"cp src/config.yaml out/{current_time}/config.yaml")
    return outpath
