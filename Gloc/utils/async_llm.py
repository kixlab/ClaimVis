import enum
import openai
from Credentials.info import *
from common.functionlog import *
import asyncio
import logging

# Configure logging to write to a file
logging.basicConfig(filename='../Gloc/log.txt', level=logging.INFO)

class Model(str, enum.Enum):
    GPT3 = 'gpt-3.5-turbo'
    GPT3_4k = 'gpt-3.5-turbo'
    GPT3_16k = 'gpt-3.5-turbo-16k'
    GPT4 = 'gpt-4'

def retry(try_count=3, sleep_seconds=2):
    """Retry decorator for async functions."""

    def decorator(fn):
        async def newfn(*args, **kwargs):
            for idx in range(try_count):
                try:
                    return await fn(*args, **kwargs)
                except ValueError as e:  # rate limit hit
                    await asyncio.sleep(sleep_seconds * (2**idx))
                    if idx == try_count - 1:
                        raise ValueError('No more retries') from e
                except RuntimeError as e:  # context overshot
                    kwargs["engine"] = Model.GPT3_16k
                except KeyError as e:  # service unavailable
                    pass
        return newfn
    return decorator


@retry(try_count=3, sleep_seconds=1)
@AsyncTokenCount
async def _call_openai(
    prompt = [],
    engine = Model.GPT3,
    max_decode_steps = 500,
    temperature = 0,
    top_p = 1,
    frequency_penalty = 0,
    presence_penalty = 0,
    samples = 1,
    stop = ('Q:', 'A:', 'Summary:', '\n\n')):
    """Issues a completion request to the engine, while retrying on failure.

    Args:
        prompt: The prompt to send.
        engine: Model engine to use.
        max_decode_steps: The max_tokens parameter to send to the engine.
        temperature: Sampling temperature.
        top_p: Ratio of likelihood weighted token options to allow while sampling.
        frequency_penalty: Pentalty for the frequency of repeated tokens.
        presence_penalty: Penalty for the existence repeated tokens.
        samples: Number of outputs to generate.
        stop: Sequence of strings that elicit an end to decoding

    Returns:
        Text completion
    """
    try:
        reply = await openai.ChatCompletion.acreate(
						model=engine,
						messages=prompt,
						temperature=temperature,
						max_tokens=max_decode_steps,
						top_p=top_p,
						frequency_penalty=frequency_penalty,
						presence_penalty=presence_penalty,
						n=samples,
						stop=stop
                    )
        
        contents = [choice['message']['content'] for choice in reply['choices']] if reply else []
        return contents, reply['usage']['total_tokens']

    except openai.error.RateLimitError as e:
        print('Sleeping 10 secs.')
        await asyncio.sleep(10)
        raise ValueError('RateLimitError') from e
    except openai.error.InvalidRequestError as e:
        logging.info(f"Super long prompt: {'@'*100}\n{prompt}\n{'@'*100}")
        raise RuntimeError('InvalidRequestError') from e
    except openai.error.ServiceUnavailableError as e:
        await asyncio.sleep(10)
        raise KeyError('ServiceUnavailableError') from e

async def call_model(
    model,
    prompt,
    temperature,
    max_decode_steps,
    samples,
):
    """Calls model given a prompt."""
    results = []
    while len(results) < samples:
        if model in [Model.GPT3, Model.GPT3_16k, Model.GPT4, Model.GPT3_4k]:
            result = await _call_openai(
                prompt=prompt,
                engine=model,
                temperature=temperature,
                max_decode_steps=max_decode_steps,
                samples=samples
            )
            results.extend(result)
        else:
            raise ValueError(f'Unknown model_type={model}')
    return results[:samples]
