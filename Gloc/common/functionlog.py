import logging
import time
from functools import wraps

# Configure logging to write to a file
logging.basicConfig(filename='log.txt', level=logging.INFO)

def log_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        logging.info(f"Calling function: {func.__name__}")
        logging.info(f"Arguments: {args}, {kwargs}")
        logging.info(f"Return Value: {result}")
        logging.info(f"Execution Time: {start_time}")
        return result
    return wrapper