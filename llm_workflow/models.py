"""Contains models."""

from abc import ABC, abstractmethod
from llm_workflow.base import CostRecord, Document, Record, RecordKeeper


class TokenUsageRecord(CostRecord):
    """Represents a record associated with token usage and/or costs."""

    total_tokens: int | None = None

    def __str__(self) -> str:
        return \
            f"timestamp: {self.timestamp}; " \
            f"cost: ${self.cost or 0:.6f}; " \
            f"total_tokens: {self.total_tokens or 0:,}; " \
            f"uuid: {self.uuid}"


class ExchangeRecord(TokenUsageRecord):
    """
    An ExchangeRecord represents a single exchange/transaction with an LLM, encompassing an input
    (prompt) and its corresponding output (response). For example, it could represent a single
    prompt and correpsonding response from within a larger conversation with ChatGPT. Its
    purpose is to record details of the interaction/exchange, including the token count and
    associated costs, if any.
    """

    prompt: str
    response: str
    prompt_tokens: int | None = None
    response_tokens: int | None = None

    def __str__(self) -> str:
        return \
            f"timestamp: {self.timestamp}; " \
            f"prompt: \"{self.prompt.strip()[0:50]}...\"; "\
            f"response: \"{self.response.strip()[0:50]}...\";  " \
            f"cost: ${self.cost or 0:.6f}; " \
            f"total_tokens: {self.total_tokens or 0:,}; " \
            f"prompt_tokens: {self.prompt_tokens or 0:,}; " \
            f"response_tokens: {self.response_tokens or 0:,}; " \
            f"uuid: {self.uuid}"


class EmbeddingRecord(TokenUsageRecord):
    """Record associated with an embedding request."""


class StreamingEvent(Record):
    """Contains the information from a streaming event."""

    response: str


class MemoryManager(ABC):
    """
    Class that has logic to handle the memory (i.e. total context) of the messages sent to an
    LLM.
    """

    @abstractmethod
    def __call__(self, history: list[ExchangeRecord]) -> list[ExchangeRecord]:
        """
        Takes the hisitory of messages and returns a modified/reduced list of messages based on the
        memory strategy.
        """


class LanguageModel(RecordKeeper):
    """
    A LanguageModel, such as ChatGPT-3 or text-embedding-ada-002 (an embedding model), is a
    class designed to be callable. Given specific inputs, such as prompts for chat-based models or
    documents for embedding models, it generates meaningful responses, which can be in the form of
    strings or documents.

    Additionally, a LanguageModel is equipped with helpful auxiliary methods that enable
    tracking and analysis of its usage history. These methods provide insights into
    metrics like token consumption and associated costs. It's worth noting that not all models
    incur direct costs, as is the case with ChatGPT; for example, offline models.
    """

    @abstractmethod
    def __call__(self, value: object) -> object:
        """Executes the chat request based on the value (e.g. message(s)) passed in."""

    @property
    def total_tokens(self) -> int | None:
        """
        Sums the `total_tokens` values across all Record objects (which contain that property)
        returned by this object's `history` property.
        """
        return self.sum(name='total_tokens')

    @property
    def cost(self) -> float | None:
        """
        Sums the `cost` values across all Record objects (which contain that property)
        returned by this object's `history` property.
        """
        return self.sum(name='cost')


class EmbeddingModel(LanguageModel):
    """A model that produces an embedding for any given text input."""

    def __init__(self) -> None:
        super().__init__()
        self._history: list[EmbeddingRecord] = []

    @abstractmethod
    def _run(self, docs: list[Document]) -> tuple[list[list[float]], EmbeddingRecord]:
        """
        Execute the embedding request.

        Returns a tuple. This tuple consists of two elements:
        1. The embedding, which are represented as a list where each item corresponds to a Document
        and contains the embedding (a list of floats).
        2. An `EmbeddingRecord` object, which track of costs and other relevant metadata. The
        record is added to the object's `history`. Only the embedding is returned to the user when
        the object is called.
        """

    def __call__(self, docs: list[Document] | list[str] | Document | str) -> list[list[float]]:
        """
        Executes the embedding request based on the document(s) provided. Returns a list of
        embeddings corresponding to the document(s). Adds a corresponding EmbeddingRecord record
        to the object's `history`.

        Args:
            docs:
                Either a list of Documents, single Document, or str. Returns the embedding that
                correspond to the doc(s).
        """
        if not docs:
            return []
        if isinstance(docs, list):
            if isinstance(docs[0], str):
                docs = [Document(content=x) for x in docs]
            else:
                assert isinstance(docs[0], Document)
        elif isinstance(docs, Document):
            docs = [docs]
        elif isinstance(docs, str):
            docs = [Document(content=docs)]
        else:
            raise TypeError("Invalid type.")

        embedding, metadata = self._run(docs=docs)
        self._history.append(metadata)
        return embedding

    def _get_history(self) -> list[EmbeddingRecord]:
        """A list of EmbeddingRecord that correspond to each embedding request."""
        return self._history


class PromptModel(LanguageModel):
    """
    The PromptModel class represents an LLM where each exchange (from the end-user's perspective)
    is a string input (user's prompt) and string output (model's response). For example, an
    exchange could represent a single prompt (input) and correpsonding response (output) from a
    ChatGPT or InstructGPT model. It provides auxiliary methods to monitor the usage history of an
    instantiated model, including metrics like tokens used.
    """

    def __init__(self):
        super().__init__()
        self._history: list[ExchangeRecord] = []

    @abstractmethod
    def _run(self, prompt: str) -> ExchangeRecord:
        """Subclasses should override this function and generate responses from the LLM."""


    def __call__(self, prompt: str) -> str:
        """
        Executes a chat request based on the given prompt and returns a response.

        Args:
            prompt: The prompt or question to be sent to the model.
        """
        response = self._run(prompt)
        self._history.append(response)
        return response.response

    def _get_history(self) -> list[ExchangeRecord]:
        """A list of ExchangeRecord objects for tracking chat messages (prompt/response)."""
        return self._history

    @property
    def previous_prompt(self) -> str | None:
        """Returns the last/previous prompt used in chat model."""
        return self.previous_record().prompt if self.previous_record() else None

    @property
    def previous_response(self) -> str | None:
        """Returns the last/previous response used in chat model."""
        return self.previous_record().response if self.previous_record() else None

    @property
    def prompt_tokens(self) -> int | None:
        """
        Sums the `prompt_tokens` values across all Record objects (which contain that property)
        returned by this object's `history` property.
        """
        return self.sum(name='prompt_tokens')

    @property
    def response_tokens(self) -> int | None:
        """
        Sums the `response_tokens` values across all Record objects (which contain that property)
        returned by this object's `history` property.
        """
        return self.sum(name='response_tokens')
