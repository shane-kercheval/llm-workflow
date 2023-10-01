"""Configures the pytests."""
from collections.abc import Callable
import pytest
import requests
import random
from faker import Faker
import numpy as np
from dotenv import load_dotenv
from unittest.mock import MagicMock


from llm_workflow.base import Document
from llm_workflow.models import EmbeddingModel, EmbeddingRecord, ExchangeRecord, PromptModel

load_dotenv()


class MockChat(PromptModel):
    """Used for unit tests to mock the behavior of an LLM."""

    def __init__(
            self,
            token_counter: Callable[[str], int] | None = None,
            cost_per_token: float | None = None,
            return_prompt: str | None = None ) -> None:
        """
        Used to test base classes.

        Args:
            token_counter:
                custom token counter to check total_tokens
            cost_per_token:
                custom costs to check costs
            return_prompt:
                if not None, prepends the string to the prompt and returns it as a response
        """
        super().__init__()
        self.token_counter = token_counter
        self.cost_per_token = cost_per_token
        self.return_prompt = return_prompt

    def _run(self, prompt: str) -> ExchangeRecord:
        if self.return_prompt:
            response = self.return_prompt + prompt
        else:
            fake = Faker()
            response = ' '.join([fake.word() for _ in range(random.randint(10, 100))])
        prompt_tokens = self.token_counter(prompt) if self.token_counter else None
        response_tokens = self.token_counter(response) if self.token_counter else None
        total_tokens = prompt_tokens + response_tokens if self.token_counter else None
        cost = total_tokens * self.cost_per_token if self.cost_per_token else None
        return ExchangeRecord(
            prompt=prompt,
            response=response,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            cost=cost,
            metadata={'model_name': 'mock'},
        )


class MockRandomEmbeddings(EmbeddingModel):
    """Used for unit tests to mock the behavior of an LLM."""

    def __init__(
            self,
            token_counter: Callable[[str], int],
            cost_per_token: float | None = None) -> None:
        super().__init__()
        self.token_counter = token_counter
        self.cost_per_token = cost_per_token

    def _run(self, docs: list[Document]) -> tuple[list[Document], EmbeddingRecord]:
        rng = np.random.default_rng()
        embeddings = [rng.uniform(low=-2, high=2, size=(50)).tolist() for _ in docs]
        total_tokens = sum(self.token_counter(x.content) for x in docs) \
            if self.token_counter else None
        cost = total_tokens * self.cost_per_token if self.cost_per_token else None
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
        print('Endpoint Not available.')

    return available
