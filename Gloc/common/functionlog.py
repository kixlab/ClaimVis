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
        logging.info(f"Token used: {TokenCount.get_token_count()}")

        TokenCount.reset() 
        return result
    return wrapper

class TokenCount(object):
    token_count = 0
    
    def __init__(self, func) -> None:
        self.func = func

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        contents, tokens = self.func(*args, **kwds)
        TokenCount.add_token_count(tokens)
        return contents
    
    @staticmethod
    def reset():
        TokenCount.token_count = 0
    
    @staticmethod
    def get_token_count():
        return TokenCount.token_count
    
    @staticmethod
    def add_token_count(tokens):
        TokenCount.token_count += tokens

class AsyncTokenCount(object):
    token_count = 0
    
    def __init__(self, func) -> None:
        self.func = func

    async def __call__(self, *args: Any, **kwds: Any) -> Any:
        contents, tokens = await self.func(*args, **kwds)
        AsyncTokenCount.add_token_count(tokens)
        return contents
    
    @staticmethod
    def reset():
        AsyncTokenCount.token_count = 0
    
    @staticmethod
    def get_token_count():
        return AsyncTokenCount.token_count
    
    @staticmethod
    def add_token_count(tokens):
        AsyncTokenCount.token_count += tokens
