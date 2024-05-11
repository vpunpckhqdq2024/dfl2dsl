"""Utilities for querying language models."""

import asyncio
import json
import logging
import os
from math import ceil
from random import random
from typing import List, Union, Optional

import openai
from openai.openai_object import OpenAIObject
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from src.llm_db import SQLiteCache


HISTORY_FILE = "cache/history.jsonl"  # File to store all queries.
CACHE_FILE = "cache/query_cache.db"   # File to store cache DB.
os.makedirs("cache", exist_ok=True)
EXP_CAP = 4     # log_2 of the maximum wait time for exponential backoff.

logger = logging.getLogger(__name__)


def get_batch_size_for_model(model_name: str) -> int:
    del model_name
    return 50


def is_gpt_model(model_name: str) -> bool:
    return "gpt" in model_name


def make_messages(prompt: str, system_msg: Optional[str], history: Optional[List[dict]] = None):
    """Creates a list of messages for OpenAI's API."""
    messages = []
    if system_msg is not None:
        messages += [{"role": "system", "content": system_msg}]
    if history is not None:
        messages += history
    messages += [{"role": "user", "content": prompt}]
    return messages


def make_messages_key(messages: List[dict]) -> str:
    """Creates a key for a list of messages."""
    key_vals = []
    for msg in messages:
        key_vals.append(f"{msg['role']} : {msg['content']}")
    return "\n".join(key_vals)


async def query_openai(messages: List[dict],
    model_name: str,
    n=1,
    max_tokens=None,
    temperature=0,
    retry=100,
    history_file=HISTORY_FILE,
    **kwargs,
):
    """Queries OpenAI's APIs.
    
    https://github.com/ekinakyurek/mylmapis/blob/b0adb192135898fba9e9dc88f09a18dc64c1f1a9/src/network_manager.py
    """

    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    kwargs["temperature"] = temperature
    kwargs["n"] = n

    for i in range(retry + 1):

        # Exponential backoff with jitter, and also increase the timeout.
        wait_time = (1 << min(i, EXP_CAP)) + random() / 10
        if "request_timeout" in kwargs:
            kwargs["request_timeout"] *= (2 if i > 0 else 1)

        try:
            response = await openai.ChatCompletion.acreate(
                model=model_name, messages=messages, **kwargs
            )
            assert isinstance(response, dict)
            with open(history_file, "a") as f:
                f.write(json.dumps((model_name, messages, kwargs, response)) + "\n")
            if any(choice["finish_reason"] != "stop" for choice in response["choices"]):
                print("Truncated response!")
                print(response)
            return response
        except (
            openai.error.APIError,  # type: ignore
            openai.error.TryAgain,  # type: ignore
            openai.error.Timeout,   # type: ignore
            openai.error.APIConnectionError,    # type: ignore
            openai.error.ServiceUnavailableError,   # type: ignore
            openai.error.RateLimitError,    # type: ignore
        ) as e:
            print(f"Failed with {e}: sleeping for {wait_time}...")
            if i == retry:
                raise e
            else:
                await asyncio.sleep(wait_time)
            print(f"Done sleeping {wait_time}")


def query_batch_wrapper(
    fn, messages, model_name, ns, *args, **kwargs
):
    """Queries a batch of prompts in parallel."""
    async def _query(messages_):
        async_responses = [
            fn(msgs, model_name, n_resps, *args, **kwargs)
            for msgs, n_resps in zip(messages_, ns)
        ]
        return await tqdm_asyncio.gather(*async_responses)

    all_results = asyncio.run(_query(messages))
    return all_results


completions_db: Union[SQLiteCache, None] = None
n_cache_consumed = {}   # Number of cached samples consumed for each query. To ensure we aren't resampling samples.

def query_batch(
    messages_batch: list,
    model_name: str,
    max_tokens: Optional[int] = None,
    temperature: float = 0.,
    retry: int = 100,
    stop_token: Optional[str] = None,
    n_samples: int = 1,

    cache_file: str = CACHE_FILE,
    history_file: str = HISTORY_FILE,

    use_cache: bool = True,  # If True, will use local cache.
    only_sample_once: bool = True,  # If True, will only use cached sample once each.

    **openai_kwargs,    # Arguments specific to openai models.
) -> List[OpenAIObject]:

    # only_sample_once = False

    if openai.api_key is None:
        openai.api_key = os.environ["OPENAI_API_KEY"]
    if temperature is None:
        temperature = 1.

    # TODO: handle this as argument.
    openai_kwargs["request_timeout"] = 30

    # Initialize the cache.
    global completions_db, n_cache_consumed
    if completions_db is None:
        completions_db = SQLiteCache(cache_file)

    # Determine which queries to make.
    unseen_prompts, n_queries, unseen_keys, n_consumed = [], [], [], []
    messages_keys = [make_messages_key(messages) for messages in messages_batch]
    for messages, messages_key in zip(messages_batch, messages_keys):
        key = (messages_key, model_name, temperature, max_tokens, stop_token)
        n_consumed_this_query = n_cache_consumed.get(key, 0) if only_sample_once else 0
        n_consumed.append(n_consumed_this_query)
        existing_completions = completions_db.n_entries(*key) - n_consumed_this_query
        n_to_make = max(n_samples - existing_completions, 0) if use_cache else n_samples

        # If we only want to sample once, mark how many we've already sampled.
        # (This only does something if there are duplicate queries in the batch).
        # We ignore temperature 0 because sampling is deterministic.
        if only_sample_once and existing_completions > 0 and temperature > 0:
            n_cache_consumed[key] = n_consumed_this_query + min(existing_completions, n_samples)

        if n_to_make > 0:
            unseen_prompts.append(messages)
            n_queries.append(n_to_make)
            unseen_keys.append(messages_key)

    # Make the queries.
    if len(unseen_prompts) > 0:

        batch_size = get_batch_size_for_model(model_name)
        total_batches = ceil(len(unseen_prompts) / batch_size)
        for start in tqdm(
            range(0, len(unseen_prompts), batch_size),
            desc="Querying batch",
            total=total_batches,
        ):
            unseen_prompts_batch = unseen_prompts[start : start + batch_size]
            n_queries_batch = n_queries[start : start + batch_size]
            unseen_keys_batch = unseen_keys[start : start + batch_size]

            if is_gpt_model(model_name):
                # closest_prompt, levenshtein_d = completions_db.find_closest_key(messages_key, model_name, temperature)
                responses = query_batch_wrapper(
                    query_openai,
                    unseen_prompts_batch,
                    model_name,
                    n_queries_batch,
                    max_tokens,
                    temperature,
                    retry,
                    history_file,
                    **openai_kwargs,
                )
            else:
                # TODO: Support for other LLMs.
                raise NotImplementedError
        
            # Update the cache.
            for messages, messages_key, n_resps, response in zip(
                unseen_prompts_batch, unseen_keys_batch, n_queries_batch, responses
            ):
                if is_gpt_model(model_name):
                    completions_db.extend(
                        [json.dumps(resp) for resp in response["choices"]],
                        messages_key,
                        model_name,
                        temperature,
                        max_tokens,
                        stop_token,
                    )
                else:
                    raise NotImplementedError

    # Return the results.
    batch_responses = []
    for messages, messages_key, n_consumed_message in zip(messages_batch, messages_keys, n_consumed):
        responses = completions_db.lookup(messages_key, model_name, temperature, max_tokens, stop_token, n_to_skip=n_consumed_message, n=n_samples)
        responses_for_batch = []
        for resp in responses:
            responses_for_batch.append(json.loads(resp))
        obj = OpenAIObject()
        obj.choices = responses_for_batch
        batch_responses.append(obj)

    n_resps = sum(len(r.choices) for r in batch_responses) if n_samples > 1 else len(batch_responses)
    assert n_resps == n_samples * len(messages_batch)

    return batch_responses


if __name__ == "__main__":
    prompts = ["What is a word that begins with a?", "What is a word that begins with a?"]
    messages = [make_messages(prompt, None) for prompt in prompts]
    resps = query_batch(
        messages,
        "gpt-3.5-turbo",
        n_samples=6,
        temperature=1.8,
        use_cache=True,
        logprobs=True,
        only_sample_once=True,
    )

    for prompt, responses in zip(prompts, resps):
        print(prompt)
        for resp in responses.choices:
            print("\t" + resp["message"]["content"])
    
