import collections
from collections.abc import Callable, Iterable, Mapping, Sequence
import enum
import json
import random
import time
from typing import TypeVar

import openai
import tensorflow as tf
from Credentials.info import *

from common.functionlog import *

openai.api_key = openai_api

class Model(str, enum.Enum):
  GPT3 = 'gpt-3.5-turbo'
  GPT3_4k = 'gpt-3.5-turbo'
  GPT3_16k = 'gpt-3.5-turbo-16k'
  GPT4 = 'gpt-4'

def retry(
    try_count = 3,
    sleep_seconds = 2,  # pylint: disable=unused-argument
):
  """Retry decorator."""

  def decorator(fn):

    def newfn(*args, **kwargs):
      for idx in range(try_count):
        try:
          return fn(*args, **kwargs)
        except ValueError as e: # rate limit hit
          time.sleep(sleep_seconds * (2**idx))
          if idx == try_count - 1:
            raise ValueError('No more retries') from e
        except RuntimeError as e: # context overshot
          kwargs["engine"] = Model.GPT3_16k
          


    return newfn

  return decorator


@retry(try_count=3, sleep_seconds=1)
@TokenCount
def _call_openai(
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
  # openai.api_key = random.choice(_OPENAI_CREDENTIALS.value)

  try:
    reply = openai.ChatCompletion.create(
        model=engine,
        messages=prompt,
        temperature=temperature,
        max_tokens=max_decode_steps,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        n=samples,
        stop=stop)
    
    contents = [choice['message']['content'] for choice in reply['choices']] if reply else []
    return contents, reply['usage']['total_tokens']

  except openai.error.RateLimitError as e:
    print('Sleeping 60 secs.')
    time.sleep(60)
    raise ValueError('RateLimitError') from e
  except openai.error.InvalidRequestError as e:
    raise RuntimeError('InvalidRequestError') from e

def call_model(
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
      results.extend(
          _call_openai(
              prompt=prompt,
              engine=model,
              temperature=temperature,
              max_decode_steps=max_decode_steps,
              samples=samples)
      )
    else:
      raise ValueError(f'Unknown model_type={model}')
  return results[:samples]


def chunks(
    generator,
    chunk_size,
    filter_fn):
  """Splits generator into chunks."""
  chunk = []
  idx = 0
  skipped = 0

  for item in generator:
    if not filter_fn(item):
      skipped += 1
      continue
    if len(chunk) >= chunk_size:
      yield idx, chunk
      idx += 1
      chunk = [item]
    else:
      chunk.append(item)

  if chunk:
    yield idx, chunk
  print('Total skipped', skipped)


def _majority(predictions):
  """Finds most frequent result among the first N predictions for each N."""
  result = []
  counter = collections.Counter()
  for prediction in predictions:
    if prediction:
      counter[prediction] += 1
    if counter:
      result.append(counter.most_common(1)[0][0])
    else:
      result.append('')
  return result


def _exec(code):
  """Executed model output and returns the `ans` variable."""

  def execute(x):
    try:
      exec(x)  # pylint: disable=exec-used
      answer = locals().get('ans', '')
      if isinstance(answer, str):
        return answer
      elif isinstance(answer, bool):
        return 'Yes' if answer else 'No'
      elif isinstance(answer, collections.abc.Sequence):
        return ', '.join(str(a) for a in answer)
      return str(answer)
    except Exception:  # pylint: disable=broad-except
      return ''

  return execute(code)


def _extract_answer(prediction):
  output = prediction.split('\n\n')[0]
  if output.lower().startswith('#python'):
    return _exec(output)
  return output.split('answer is')[-1].strip().rstrip('.').strip()


def _extract_answers(predictions):
  return [_extract_answer(output) for output in predictions]