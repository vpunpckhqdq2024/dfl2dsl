"""Beta Feature: base interface for cache."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union

from sqlalchemy import Column, Integer, String, create_engine, select, Float
from sqlalchemy.sql.expression import func
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import Session
from Levenshtein import distance as levenshtein_distance

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base


# pylint: disable=redefined-outer-name


class BaseCache(ABC):
    """Base interface for cache."""

    @abstractmethod
    def lookup(self, prompt: str, llm_string: str):
        """Look up based on prompt and llm_string."""

    @abstractmethod
    def update(self, prompt: str, llm_string: str, return_val):
        """Update cache based on prompt and llm_string."""


class InMemoryCache(BaseCache):
    """Cache that stores things in memory."""

    def __init__(self) -> None:
        """Initialize with empty cache."""
        self._cache = {}

    def lookup(self, prompt: str, llm_string: str):
        """Look up based on prompt and llm_string."""
        return self._cache.get((prompt, llm_string), None)

    def update(self, prompt: str, llm_string: str, return_val) -> None:
        """Update cache based on prompt and llm_string."""
        self._cache[(prompt, llm_string)] = return_val


Base = declarative_base()


class FullLLMCache(Base):  # type: ignore
    """SQLite table for full LLM Cache (all generations)."""

    __tablename__ = "full_llm_cache"
    prompt = Column(String, primary_key=True)
    llm = Column(String, primary_key=True)
    temperature = Column(Float, primary_key=True)
    max_tokens = Column(Integer, primary_key=True)
    stop = Column(String, primary_key=True)
    idx = Column(Integer, primary_key=True)

    response = Column(String)


class SQLAlchemyCache(BaseCache):
    """Cache that uses SQAlchemy as a backend."""

    def __init__(self, engine: Engine, cache_schema: Any = FullLLMCache):
        """Initialize by creating all tables."""
        self.engine = engine
        self.cache_schema = cache_schema
        self.cache_schema.metadata.create_all(self.engine)

    def lookup(
        self, prompt: str, llm_string: str, temperature: float,
        max_tokens: Optional[int], stop: Optional[Union[str, List[str]]], n=None,
        enforce_max_tokens: bool = False,
        enforce_stop: bool = False, n_to_skip: int=0) -> List[str]:
        """Look up based on prompt and llm_string."""

        if isinstance(stop, list):
            stop = "|".join(stop)
        if max_tokens is None:
            max_tokens = -1
        if stop is None:
            stop = ""

        stmt = (
            select(self.cache_schema.response, self.cache_schema.idx)
            .where(self.cache_schema.prompt == prompt)
            .where(self.cache_schema.llm == llm_string)
            .where(self.cache_schema.temperature == temperature)
        )
        if enforce_max_tokens:
            stmt = stmt.where(self.cache_schema.max_tokens == max_tokens)
        if enforce_stop:
            stmt = stmt.where(self.cache_schema.stop == stop)

        stmt = stmt.order_by(self.cache_schema.idx)
        if n is not None:
            stmt = stmt.limit(n).offset(n_to_skip)

        with Session(self.engine) as session:
            generations = [row for row in session.execute(stmt)]
            generations.sort(key=lambda x: x[1])
            generations = [response for response, _ in generations]
            return generations

    def n_entries(
        self, prompt: str, llm_string: str, temperature: float,
        max_tokens: Optional[int], stop, session=None) -> int:
        """Look up based on prompt and llm_string."""

        if isinstance(stop, list):
            stop = "|".join(stop)
        if max_tokens is None:
            max_tokens = -1
        if stop is None:
            stop = ""

        stmt = (
            select(func.max(self.cache_schema.idx)) # pylint: disable=not-callable
            .where(self.cache_schema.prompt == prompt)
            .where(self.cache_schema.llm == llm_string)
            .where(self.cache_schema.temperature == temperature)
            .where(self.cache_schema.max_tokens == max_tokens)
            .where(self.cache_schema.stop == stop)
        )
        if session is None:
            with Session(self.engine) as session:
                generations = list(session.execute(stmt))
                if len(generations) > 0:
                    assert len(generations) == 1
                    g = generations[0][0]
                    if g is None: return 0
                    return g+1
        else:
            generations = list(session.execute(stmt))
            if len(generations) > 0:
                assert len(generations) == 1
                g = generations[0][0]
                if g is None: return 0
                return g+1

        return 0

    def update(
        self, return_val, prompt: str, llm_string: str, temperature: float,
        max_tokens: Optional[int], stop: Optional[str]) -> None:

        if isinstance(stop, list):
            stop = "|".join(stop)
        if max_tokens is None:
            max_tokens = -1
        if stop is None:
            stop = ""

        for i, response in enumerate(return_val):
            item = self.cache_schema(
                prompt=prompt, llm=llm_string, response=response, idx=i,
                temperature=temperature, max_tokens=max_tokens, stop=stop
            )
            with Session(self.engine) as session, session.begin(): # type: ignore
                session.merge(item)

    def extend(
        self, responses: List[str], prompt: str,
        llm_string: str, temperature: float,
        max_tokens: Optional[int], stop: Optional[Union[str, List[str]]],
        session=None) -> None:

        if isinstance(stop, list):
            stop = "|".join(stop)
        if max_tokens is None:
            max_tokens = -1
        if stop is None:
            stop = ""

        n = self.n_entries(
            prompt, llm_string, temperature, max_tokens, stop, session=session)

        new_items = []
        for i, response in enumerate(responses):
            item = self.cache_schema(
                prompt=prompt, llm=llm_string, temperature=temperature,
                max_tokens=max_tokens, stop=stop, response=response, idx=n+i
            )
            new_items.append(item)

        if session is None:
            with Session(self.engine) as session, session.begin(): # type: ignore
                session.add_all(new_items)
                session.commit()
        else:
            session.add_all(new_items)
            session.commit()

    def find_closest_key(self, prompt: str, llm_string: str, temperature: float) -> Tuple[str, Union[int, float]]:
        """
        Finds the closest key by string edit distance to the given prompt.

        :param prompt: The prompt string to compare.
        :param llm_string: The llm_string to compare.
        :return: A tuple containing the closest prompt and the edit distance.
        """

        with Session(self.engine) as session:
            # Retrieve all keys (prompts and llm_strings) from the database
            # that have the same llm / temperature
            stmt = (
                select(self.cache_schema.prompt)
                .where(self.cache_schema.temperature == temperature)
                .where(self.cache_schema.llm == llm_string)
            )
            all_keys = session.execute(stmt).fetchall()

        # Initialize variables to track the closest match
        closest_prompt = ""
        min_distance = float('inf')

        # Calculate the edit distance between the given key and each key in the database
        for db_prompt in all_keys:
            # Combine prompt and llm_string for a comprehensive comparison
            current_distance = levenshtein_distance(prompt, db_prompt[0])

            # Update the closest match if a closer one is found
            if current_distance < min_distance and current_distance != 0:
                min_distance = current_distance
                closest_prompt = db_prompt[0]

        return closest_prompt, min_distance


class SQLiteCache(SQLAlchemyCache):
    """Cache that uses SQLite as a backend."""

    def __init__(self, database_path: str = ".langchain.db"):
        """Initialize by creating the engine and all tables."""
        engine = create_engine(f"sqlite:///{database_path}")
        assert isinstance(engine, Engine)
        super().__init__(engine)


class EmbeddingCache(Base):  # type: ignore
    """SQLite table for embedding Cache."""

    __tablename__ = "embedding_cache"
    text = Column(String, primary_key=True)
    model = Column(String, primary_key=True)
    idx = Column(Integer, primary_key=True)

    response = Column(String)


class SQLAlchemyEmbeddingCache(BaseCache):
    """Cache that uses SQAlchemy as a backend."""

    def __init__(self, engine: Engine, cache_schema: Any = EmbeddingCache):
        """Initialize by creating all tables."""
        self.engine = engine
        self.cache_schema = cache_schema
        self.cache_schema.metadata.create_all(self.engine)

    def lookup(self, text: str, model: str) -> List[str]:
        """Look up based on text and model."""

        stmt = (
            select(self.cache_schema.response, self.cache_schema.idx)
            .where(self.cache_schema.text == text)
            .where(self.cache_schema.model == model)
        )

        stmt = stmt.order_by(self.cache_schema.idx)
        with Session(self.engine) as session:
            generations = [row for row in session.execute(stmt)]
            generations.sort(key=lambda x: x[1])
            generations = [response for response, _ in generations]
            return generations

    def n_entries(self, text: str, model: str, session=None) -> int:
        """Look up based on text and model."""

        stmt = (
            select(func.max(self.cache_schema.idx)) # pylint: disable=not-callable
            .where(self.cache_schema.text == text)
            .where(self.cache_schema.model == model)
        )
        if session is None:
            with Session(self.engine) as session:
                generations = list(session.execute(stmt))
                if len(generations) > 0:
                    assert len(generations) == 1
                    g = generations[0][0]
                    if g is None: return 0
                    return g+1
        else:
            generations = list(session.execute(stmt))
            if len(generations) > 0:
                assert len(generations) == 1
                g = generations[0][0]
                if g is None: return 0
                return g+1

        return 0

    def update(self, return_val, text: str, model: str) -> None:

        for i, response in enumerate(return_val):
            item = self.cache_schema(text=text, model=model, response=response, idx=i)
            with Session(self.engine) as session, session.begin(): # type: ignore
                session.merge(item)

    def extend(self, responses: List[str], text: str, model: str, session=None) -> None:

        n = self.n_entries(text, model, session=session)

        new_items = []
        for i, response in enumerate(responses):
            item = self.cache_schema(text=text, model=model, response=response, idx=n+i)
            new_items.append(item)

        if session is None:
            with Session(self.engine) as session, session.begin(): # type: ignore
                session.add_all(new_items)
                session.commit()
        else:
            session.add_all(new_items)
            session.commit()

class SQLiteEmbeddingCache(SQLAlchemyEmbeddingCache):
    """Cache that uses SQLite as a backend."""

    def __init__(self, database_path: str):
        """Initialize by creating the engine and all tables."""
        engine = create_engine(f"sqlite:///{database_path}")
        assert isinstance(engine, Engine)
        super().__init__(engine)
