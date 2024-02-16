"""Configures the pytests."""
import os
from collections.abc import Callable
import re
from time import sleep
from typing import Any
import pandas as pd
import pytest
import requests
import random
import yaml
from faker import Faker
import numpy as np
from dotenv import load_dotenv
from unittest.mock import MagicMock
from llm_workflow.hugging_face import llama_message_formatter
from llm_workflow.base import (
    ChatModel,
    Document,
    EmbeddingModel,
    EmbeddingRecord,
    ExchangeRecord,
    MemoryManager,
    PromptModel,
)

load_dotenv()


class MockPromptModel(PromptModel):
    """Used for unit tests to mock the behavior of an LLM."""

    def __init__(
            self,
            token_calculator: Callable[[str | list[str] | object], int],
            cost_calculator: Callable[[int, int], float] | None = None,
            return_prompt: str | None = None) -> None:
        """
        Used to test base classes.

        Args:
            token_calculator:
                custom token counter
            cost_calculator:
                callable to calculate costs
            return_prompt:
                if not None, prepends the string to the prompt and returns it as a response
        """
        super().__init__(
            token_calculator=token_calculator,
            cost_calculator=cost_calculator,
        )
        self.return_prompt = return_prompt

    def _run(self, prompt: str) -> tuple[str | list[str] | object, dict]:
        if self.return_prompt:
            response = self.return_prompt + prompt
        else:
            fake = Faker()
            response = ' '.join([fake.word() for _ in range(random.randint(10, 100))])

        return response, {'return_prompt': self.return_prompt}


class MockCostMemoryManager(MemoryManager):
    """
    Used to mock the behavior of an MemoryManager that has associated costs e.g. summariziation via
    LLM.
    """

    def __init__(self, cost: float | None) -> None:
        self._history = []
        self.cost = cost

    def history(self):  # noqa
        return self._history

    def __call__(
            self,
            system_message: str,
            history: list[ExchangeRecord],
            prompt: str,
            **kwargs: dict[str, Any]) -> str | list[str] | list[dict[str, str]]:
        """Mocks a call to a memory memanager."""
        message_formatter = kwargs['message_formatter']
        message = message_formatter(system_message, None, None)
        for record in history:
            record = record.model_copy()  # noqa
            message += message_formatter(None, [record], None)
        message += message_formatter(None, None, prompt)
        cost = self.cost
        if cost:
            cost *= (len(prompt) + len(message))
        self._history.append(ExchangeRecord(
            prompt=prompt,
            response=message,
            metadata={'model_name': 'memory'},
            input_tokens=len(prompt),
            response_tokens=len(message),
            total_tokens=len(prompt) + len(message),
            cost=cost,
        ))
        sleep(0.1)
        return message


class MockChatModel(ChatModel):
    """Used for unit tests to mock the behavior of an LLM."""

    def __init__(
            self,
            token_calculator: Callable[[str | list[str] | object], int],
            cost_calculator: Callable[[int, int], float] | None = None,
            return_prompt: str | None = None,
            message_formatter: Callable[[str, list[ExchangeRecord]], str] | None = None,
            memory_manager: MemoryManager | None = None) -> None:
        """
        Used to test base classes.

        Args:
            token_calculator:
                custom token counter to calculate total_tokens
            cost_calculator:
                callable to calculate costs
            return_prompt:
                if not None, prepends the string to the prompt and returns it as a response
            message_formatter:
                custom message formatter to format messages
            memory_manager:
                custom memory manager to manage memory
        """
        if message_formatter is None:
            message_formatter = llama_message_formatter
        super().__init__(
            system_message="This is a system message.",
            message_formatter=message_formatter,
            token_calculator=token_calculator,
            cost_calculator=cost_calculator,
            memory_manager=memory_manager,
        )
        self.return_prompt = return_prompt

    def _run(self, prompt: str) -> ExchangeRecord:
        if self.return_prompt:
            response = self.return_prompt + prompt
        else:
            fake = Faker()
            response = ' '.join([fake.word() for _ in range(random.randint(10, 100))])
        return response, {'model_name': 'mock'}


class MockRandomEmbeddings(EmbeddingModel):
    """Used for unit tests to mock the behavior of an LLM."""

    def __init__(
            self,
            token_counter: Callable[[str], int],
            cost_per_token: float | None = None) -> None:
        super().__init__()
        self.token_counter = token_counter
        self.cost_per_token = cost_per_token
        self.lookup = []

    def _run(self, docs: list[Document]) -> tuple[list[Document], EmbeddingRecord]:
        rng = np.random.default_rng()
        embeddings = [rng.uniform(low=-2, high=2, size=(50)).tolist() for _ in docs]
        total_tokens = sum(self.token_counter(x.content) for x in docs) \
            if self.token_counter else None
        cost = total_tokens * self.cost_per_token if self.cost_per_token else None
        self.lookup += embeddings
        len(self.lookup)
        len(embeddings)
        return embeddings, EmbeddingRecord(
            total_tokens=total_tokens,
            cost=cost,
        )


@pytest.fixture()
def fake_docs_abcd() -> list[Document]:
    """Meant to be used MockABCDEmbeddings model."""
    return [
        Document(content="Doc A", metadata={'id': 0}),
        Document(content="Doc B", metadata={'id': 1}),
        Document(content="Doc C", metadata={'id': 3}),
        Document(content="Doc D", metadata={'id': 4}),
    ]


class MockABCDEmbeddings(EmbeddingModel):
    """
    Used for unit tests to mock the behavior of an LLM.

    Used in conjunction with a specific document list `fake_docs_abcd`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.cost_per_token = 7
        self._next_lookup_index = None
        self.lookup = {
            0: [0.5, 0.5, 0.5, 0.5, 0.5],
            1: [1, 1, 1, 1, 1],
            3: [3, 3, 3, 3, 3],
            4: [4, 4, 4, 4, 4],
        }

    def _run(self, docs: list[Document]) -> tuple[list[Document], EmbeddingRecord]:
        if self._next_lookup_index:
            embeddings = [self.lookup[self._next_lookup_index]]
        else:
            embeddings = [self.lookup[x.metadata['id']] for x in docs]
        total_tokens = sum(len(x.content) for x in docs)
        cost = total_tokens * self.cost_per_token
        return embeddings, EmbeddingRecord(
            total_tokens=total_tokens,
            cost=cost,
            metadata={'content': [x.content for x in docs]},
        )


class FakeRetryHandler:
    """A fake retry handler used for unit tests."""

    def __call__(self, f, *args, **kwargs):  # noqa
        return f(*args, **kwargs)


@pytest.fixture()
def fake_retry_handler():  # noqa
    return FakeRetryHandler()


@pytest.fixture()
def fake_hugging_face_response_json():  # noqa
    fake = Faker()
    num_words = fake.random_int(min=8, max=10)
    if num_words == 8:
        return [{
            'generated_text': "",
        }]

    return [{
        'generated_text': " ".join(fake.words(nb=num_words)),
    }]


@pytest.fixture()
def fake_hugging_face_response(fake_hugging_face_response_json):  # noqa
    response = MagicMock()
    response.json.return_value = fake_hugging_face_response_json
    return response


def is_endpoint_available(url: str) -> bool:
    """Returns True if the endpoint is available."""
    available = False
    try:
        response = requests.head(url, timeout=5)  # You can use GET or HEAD method
        # checking if we get a response code of 2xx or 3xx
        available = 200 <= response.status_code <= 401
    except requests.RequestException:
        pass

    if not available:
        print(f'Endpoint Not available: {url}')

    return available


@pytest.fixture()
def conversation_mask_email() -> dict:
    """Returns a mock llm  conversation for masking emails."""
    with open('tests/test_data/compare/mock_conversation__mask_email_function.yml') as f:
        return yaml.safe_load(f)


@pytest.fixture()
def conversation_sum() -> dict:
    """Returns a mock llm  conversation for summing numbers."""
    with open('tests/test_data/compare/mock_conversation__sum_function.yml') as f:
        return yaml.safe_load(f)


@pytest.fixture()
def hugging_face_endpoint() -> str:
    """Returns the endpoint for the hugging face API."""
    return os.getenv('HUGGING_FACE_ENDPOINT_UNIT_TESTS')

@pytest.fixture()
def llama_cpp_endpoint() -> str:
    """Returns the endpoint for the hugging face API."""
    return os.getenv('LLAMA_CPP_ENDPOINT_UNIT_TESTS') or ''


def pattern_found(value: str, pattern: str) -> bool:
    """Returns True if the pattern is found in the value."""
    pattern = re.compile(pattern)
    return bool(pattern.match(value))


@pytest.fixture()
def credit_data() -> pd.DataFrame:
    """Returns a dataframe with credit data."""
    return pd.read_csv('tests/test_data/data/credit.csv')

@pytest.fixture()
def html_image_url_nl() -> str:
    """Returns a url to an image containing the northern lights."""
    return 'https://www.w3schools.com/w3css/img_lights.jpg'

@pytest.fixture()
def local_image_path_nl() -> str:
    """Returns a local path to an image containing the northern lights."""
    return 'tests/test_data/images/northern_lights.png'
