"""Utilities for querying embedding_models."""

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
from numpy import dot
from numpy.linalg import norm

from src.llm_db import SQLiteEmbeddingCache


HISTORY_FILE = "cache/embedding_history.jsonl"  # File to store all queries.
CACHE_FILE = "cache/embedding_cache.db"   # File to store cache DB.
os.makedirs("cache", exist_ok=True)
EXP_CAP = 4     # log_2 of the maximum wait time for exponential backoff.

logger = logging.getLogger(__name__)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculates the cosine similarity between two vectors."""
    return dot(a, b) / (norm(a) * norm(b))


def get_batch_size_for_model(model_name: str) -> int:
    del model_name
    return 50


def is_openai_model(model_name: str) -> bool:
    return model_name in ("text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002")


def query_openai_embedding(text: List[dict],
    model_name: str,
    retry=100,
    history_file=HISTORY_FILE,
    **kwargs,
):
    """Queries OpenAI's APIs.
    
    https://github.com/ekinakyurek/mylmapis/blob/b0adb192135898fba9e9dc88f09a18dc64c1f1a9/src/network_manager.py
    """

    for i in range(retry + 1):

        # Exponential backoff with jitter, and also increase the timeout.
        wait_time = (1 << min(i, EXP_CAP)) + random() / 10
        if "request_timeout" in kwargs:
            kwargs["request_timeout"] *= (2 if i > 0 else 1)

        try:
            response = openai.Embedding.create(input=text, model=model_name)
            assert isinstance(response, dict)
            with open(history_file, "a") as f:
                f.write(json.dumps((model_name, text, kwargs, response)) + "\n")
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
                import time
                time.sleep(wait_time)
            print(f"Done sleeping {wait_time}")


embeddings_db: Union[SQLiteEmbeddingCache, None] = None

def embed_batch(
    texts_batch: list,
    model_name: str,
    retry: int = 100,

    cache_file: str = CACHE_FILE,
    history_file: str = HISTORY_FILE,

    use_cache: bool = True,  # If True, will use local cache.

    **openai_kwargs,    # Arguments specific to openai models.
) -> list:

    if openai.api_key is None:
        openai.api_key = os.environ["OPENAI_API_KEY"]

    # TODO: handle this as argument.
    openai_kwargs["request_timeout"] = 30

    # Initialize the cache.
    global embeddings_db, n_cache_consumed
    if embeddings_db is None:
        embeddings_db = SQLiteEmbeddingCache(cache_file)

    # Determine which queries to make.
    unseen_texts = []
    for text in texts_batch:
        key = (text, model_name)
        existing_completions = embeddings_db.n_entries(*key)
        # assert existing_completions <= 1

        if existing_completions == 0:
            unseen_texts.append(text)

    # Make the queries.
    if len(unseen_texts) > 0:

        batch_size = get_batch_size_for_model(model_name)
        total_batches = ceil(len(unseen_texts) / batch_size)
        for start in tqdm(
            range(0, len(unseen_texts), batch_size),
            desc="Querying batch",
            total=total_batches,
        ):
            unseen_texts_batch = unseen_texts[start : start + batch_size]

            if is_openai_model(model_name):
                responses = query_openai_embedding(
                    unseen_texts_batch,
                    model_name,
                    retry,
                    history_file,
                    **openai_kwargs,
                )
                responses = responses.data
                assert len(responses) == len(unseen_texts_batch)
            else:
                raise NotImplementedError
        
            # Update the cache.
            for text, response in zip(unseen_texts_batch, responses):
                if is_openai_model(model_name):
                    embeddings_db.extend([json.dumps(response.embedding)], text, model_name)
                else:
                    raise NotImplementedError

    # Return the results.
    batch_responses = []
    for text in texts_batch:
        responses = embeddings_db.lookup(text, model_name)
        # assert len(responses) == 1
        batch_responses.append(json.loads(responses[0]))

    return batch_responses


if __name__ == "__main__":
    texts = ["What is a word that begins with a?", "What is a word that begins with a?", "What is a word that begins with b?", "fdshafudsafffs"]
    embeddings = embed_batch(
        texts,
        "text-embedding-3-small",
    )
    print(cosine_similarity(embeddings[0], embeddings[2]))
    print(cosine_similarity(embeddings[0], embeddings[3]))
    # messages = [make_messages(prompt, None) for prompt in prompts]
    # resps = embed_batch(
    #     messages,
    #     "gpt-3.5-turbo",
    #     n_samples=6,
    #     temperature=1.8,
    #     use_cache=True,
    #     logprobs=True,
    #     only_sample_once=True,
    # )

    # for prompt, responses in zip(prompts, resps):
    #     print(prompt)
    #     for resp in responses.choices:
    #         print("\t" + resp["message"]["content"])
    
