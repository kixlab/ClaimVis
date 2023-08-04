import logging
import time
from functools import wraps
from datetime import datetime
from typing import Any

# Configure logging to write to a file
logging.basicConfig(filename='../Gloc/log.txt', level=logging.INFO)

def log_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result = func(*args, **kwargs)

        logging.info("##############################################################################################################################")        
        logging.info(f"Calling function: {func.__name__}")
        logging.info(f"Arguments: {args}, {kwargs}")
        logging.info(f"Return Value: {result}")
        logging.info(f"Execution Time: {start_time}")
        logging.info(f"Token used: {TokenCount.token_count}")

        TokenCount.token_count = 0 # reset token count
        return result
    return wrapper

class TokenCount(object):
    token_count = 0

    def __init__(self, func) -> None:
        self.func = func

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        contents, tokens = self.func(*args, **kwds)
        # rack up token count
        TokenCount.token_count += tokens
        return contents