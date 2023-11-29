"""Contains all base and foundational classes."""

from abc import ABC, abstractmethod
from typing import Any
from collections.abc import Callable
from uuid import uuid4
from datetime import datetime
from pydantic import BaseModel, Field
from llm_workflow.internal_utilities import has_method, has_property


class Record(BaseModel):
    """Used to track the history of a task."""

    uuid: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
    )
    metadata: dict = Field(default_factory=dict)

    def __str__(self) -> str:
        return \
            f"timestamp: {self.timestamp}; " \
            f"uuid: {self.uuid}"


class CostRecord(Record):
    """A Record object that tracks cost."""

    cost: float | None = None


class SearchRecord(Record):
    """A Record object associated with a search query (e.g. web-search, Stack Overflow search)."""

    query: str
    results: list | None = None


class Document(BaseModel):
    """
    A Document comprises both content (text) and metadata, allowing it to represent a wide range of
    entities such as files, web pages, or even specific sections within a larger document.
    """

    content: str
    metadata: dict | None = None


class RecordKeeper(ABC):
    """A RecordKeeper is an object that tracks history i.e. `Record` objects."""

    @abstractmethod
    def _get_history(self) -> list[Record]:
        """
        Each inheriting class needs to implement the ability to track history.
        When an object doesn't have any history, it should return an empty list rather than `None`.
        """

    def history(self, types: type | tuple[type] | None = None) -> list[Record]:
        """
        Returns the history (list of Records objects) associated with an object, based on the
        `types` provided.

        Args:
            types:
                if None, all Record objects are returned
                if not None, either a single type or tuple of types indicating the type of Record
                objects to return (e.g. `TokenUsageRecord` or `(ExchangeRecord, EmbeddingRecord)`).
        """
        history = self._get_history() or []  # ensure empty list is returned instead of None
        if not types:
            return sorted(history, key=lambda x: x.timestamp)
        if isinstance(types, type | tuple):
            history = [x for x in history if isinstance(x, types)]
            return sorted(history, key=lambda x: x.timestamp)

        raise TypeError(f"types not a valid type ({type(types)}) ")

    def previous_record(self, types: type | tuple[type] | None = None) -> Record | None:
        """
        Returns the last/previous Record object in the history. If the object does not have any
        history, None is returned.
        """
        history = self.history(types=types)
        if history:
            return history[-1]
        return None

    def sum(  # noqa: A003
            self,
            name: str,
            types: type | tuple[type] | None = None) -> int | float:
        """
        For a given property `name` (e.g. `cost` or `total_tokens`), this function sums the values
        across all Record objects in the history, for any Record object that contains the property.

        Args:
            name:
                the name of the property on the Record object to aggregate
            types:
                if provided, only the Record objects in `history` with the corresponding type(s)
                are included in the aggregation
        """
        records = [
            x for x in self.history(types=types)
            if has_property(obj=x, property_name=name)
        ]
        if records:
            return sum(getattr(x, name) or 0 for x in records)
        return 0


class Value:
    """
    The Value class provides a convenient caching mechanism within the workflow.
    The `Value` object is callable, allowing it to cache and return the value when provided as an
    argument. When called without a value, it retrieves and returns the cached value.
    """

    def __init__(self):
        self.value = None

    def __call__(self, value: object | None = None) -> object:
        """
        When a `value` is provided, it gets cached and returned.
        If no `value` is provided, the previously cached value is returned (or None if no value has
        been cached).
        """
        if value:
            self.value = value
        return self.value


class Workflow(RecordKeeper):
    """
    A workflow object is a collection of `tasks`. Each task in the workflow is a callable, which
    can be either a function or an object that implements the `__call__` method.

    The output of one task serves as the input to the next task in the workflow.

    Additionally, each task can track its own history, including messages sent/received and token
    usage/costs, through a `history()` method that returns a list of `Record` objects. A workflow
    aggregates and propagates the history of any task that has a `history()` method, making it
    convenient to analyze costs or explore intermediate steps in the workflow.
    """

    def __init__(self, tasks: list[Callable[[Any], Any]]):
        self._tasks = tasks

    def __getitem__(self, index: int) -> Callable:
        return self._tasks[index]

    def __len__(self) -> int:
        return len(self._tasks)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """
        Executes the workflow by passing the provided value to the first task. The output of each
        task is passed as the input to the next task, creating a sequential execution of all tasks
        in the workflow.

        The execution continues until all tasks in the workflow have been executed. The final
        output from the last task is then returned.
        """
        if not self._tasks:
            return None
        result = self._tasks[0](*args, **kwargs)
        if len(self._tasks) > 1:
            for task in self._tasks[1:]:
                result = task(result)
        return result

    def _get_history(self) -> list[Record]:
        """
        Aggregates the `history` across all tasks in the workflow. This method ensures that if a
        task is added multiple times to the workflow (e.g. a chat model with multiple steps), the
        underlying Record objects associated with that task's `history` are not duplicated.
        """
        histories = [task.history() for task in self._tasks if _has_history(task)]
        # Edge-case: if the same model is used multiple times in the same workflow (e.g. embedding
        # model to embed documents and then embed query to search documents) then we can't loop
        # through the workflows because we'd be double-counting the history from those objects.
        # we have to build up a history and include the objects if they aren't already
        # to do this we'll use the uuid, and then sort by timestamp
        unique_records = []
        unique_uuids = set()
        for history in histories:
            for record in history:
                if record.uuid not in unique_uuids:
                    unique_records.append(record)
                    unique_uuids |= {record.uuid}
        return unique_records


class Session(RecordKeeper):
    """
    A Session is used to aggregate multiple workflow objects. It provides a way to track and manage
    multiple workflows within the same session. When calling a Session, it will execute the last
    workflow that was added to the session.
    """

    def __init__(self, workflows: list[Workflow] | None = None):
        self._workflows = workflows or []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """
        Calls/starts the workflow that was the last added to the session, passing in the
        corresponding arguments.
        """
        if self._workflows:
            return self._workflows[-1](*args, **kwargs)
        raise ValueError()

    def append(self, workflow: Workflow) -> None:
        """
        Add or append a new workflow object to the list of workflows in the session. If the session
        object is called (i.e. __call__), the session will forward the call to the new workflow
        object (i.e. the last workflow added in the list).
        """
        self._workflows.append(workflow)

    def __len__(self) -> int:
        return len(self._workflows)

    def _get_history(self) -> list[Record]:
        """
        Aggregates the `history` across all workflow objects in the Session. This method ensures
        that if a task is added multiple times to the Session, the underlying Record objects
        associated with that task's `history` are not duplicated.
        """
        # for each history in workflow, cycle through each task's history and add to the list of
        # records if it hasn't already been added.
        workflows = [workflow for workflow in self._workflows if workflow.history()]
        # Edge-case: if the same model is used multiple times in the same workflow or across
        # different tasks (e.g. embedding
        # model to embed documents and then embed query to search documents) then we can't loop
        # through the workflows because we'd be double-counting the history from those objects.
        # we have to build up a history and include the objects if they aren't already
        # to do this we'll use the uuid, and then sort by timestamp
        unique_records = []
        unique_uuids = set()
        for workflow in workflows:
            for record in workflow.history():
                if record.uuid not in unique_uuids:
                    unique_records.append(record)
                    unique_uuids |= {record.uuid}
        return sorted(unique_records, key=lambda x: x.timestamp)


def _has_history(obj: object) -> bool:
    """
    For a given object `obj`, return True if that object has a `history` method and if the
    history has any Record objects.
    """
    return has_method(obj, 'history') and \
        isinstance(obj.history(), list) and \
        len(obj.history()) > 0 and \
        isinstance(obj.history()[0], Record)


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

    prompt: str | list
    response: str
    input_tokens: int | None = None
    response_tokens: int | None = None

    def __str__(self) -> str:
        return \
            f"timestamp: {self.timestamp}; " \
            f"prompt: \"{self.prompt.strip()[0:50]}...\"; "\
            f"response: \"{self.response.strip()[0:50]}...\";  " \
            f"cost: ${self.cost or 0:.6f}; " \
            f"total_tokens: {self.total_tokens or 0:,}; " \
            f"input_tokens: {self.input_tokens or 0:,}; " \
            f"response_tokens: {self.response_tokens or 0:,}; " \
            f"uuid: {self.uuid}"


class EmbeddingRecord(TokenUsageRecord):
    """Record associated with an embedding request."""


class StreamingEvent(Record):
    """Contains the information from a streaming event."""

    response: str


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

    def __init__(
            self,
            token_calculator: Callable[[str | list[str] | object], int],
            cost_calculator: Callable[[int, int], float] | None = None) -> None:
        super().__init__()
        self._history: list[ExchangeRecord] = []
        self._token_calculator = token_calculator
        if cost_calculator:
            self._cost_calculator = cost_calculator
        else:
            self._cost_calculator = lambda _in, _out: None

    @abstractmethod
    def _run(self, prompt: str) -> tuple[str | list[str] | object, dict]:
        """
        Subclasses should override this function and generate responses from the LLM.
        This function should return a tuple. The first element of the tuple is the response from
        the LLM. The second element is a dictionary containing metadata associated with the
        response. This metadata is used to create an `ExchangeRecord` object, which is added to the
        `history` of the object.
        """

    def __call__(self, prompt: str) -> str:
        """
        Executes a chat request based on the given prompt and returns a response.

        Args:
            prompt: The prompt or question to be sent to the model.
        """
        response, metadata = self._run(prompt)
        input_tokens = self._token_calculator(prompt)
        response_tokens = self._token_calculator(response)
        total_tokens = input_tokens + response_tokens
        cost = self._cost_calculator(input_tokens, response_tokens)
        response = ExchangeRecord(
            prompt=prompt,
            response=response.strip(),
            metadata=metadata,
            input_tokens=input_tokens,
            response_tokens=response_tokens,
            total_tokens=total_tokens,
            cost=cost,
        )
        self._history.append(response)
        return response.response

    def _get_history(self) -> list[ExchangeRecord]:
        """A list of ExchangeRecord objects for tracking chat messages (prompt/response)."""
        return self._history.copy()

    @property
    def previous_prompt(self) -> str | None:
        """Returns the last/previous prompt used in chat model."""
        return self.previous_record().prompt if self.previous_record() else None

    @property
    def previous_response(self) -> str | None:
        """Returns the last/previous response used in chat model."""
        return self.previous_record().response if self.previous_record() else None

    @property
    def input_tokens(self) -> int | None:
        """
        Sums the `input_tokens` values across all Record objects (which contain that property)
        returned by this object's `history` property.
        """
        return self.sum(name='input_tokens')

    @property
    def response_tokens(self) -> int | None:
        """
        Sums the `response_tokens` values across all Record objects (which contain that property)
        returned by this object's `history` property.
        """
        return self.sum(name='response_tokens')


class MemoryManager(ABC):
    """
    Class that has logic to handle the memory (i.e. total context) of the messages sent to an
    LLM.
    """

    @abstractmethod
    def __call__(
        self,
        system_message: str,
        history: list[ExchangeRecord],
        prompt: str,
        **kwargs: dict[str, Any]) -> str | list[str] | list[dict[str, str]]:
        """
        Takes the hisitory of messages and returns a modified/reduced list of messages based on the
        memory strategy. Requires the system message and prompt to be passed in as well, since
        those will affect the memory. Only the list of messages to send to the model should be
        returned since those are the only messages that will be dynamically modified.
        """


class ChatMessageFormatter(ABC):
    """
    A ChatMessageFormatter is a class designed to be callable. Given specific inputs, (e.g. system
    message, history of messages, and prompt), it the final prompt to send to the model (e.g. list
    or string, depending on the type of model).
    """

    @abstractmethod
    def __call__(
            self,
            system_message: str,
            history: list[ExchangeRecord],
            prompt: str,
            ) -> list | str:
        """
        Takes the system message, history of messages, and prompt and returns a list of messages
        (or a single message) to send to the model. Some models, such as OpenAI's ChatGPT takes a
        list of dictionaries defining role/content pairs, while others, such as Llama, takes a
        single string.
        """


class ChatModel(PromptModel):
    """
    The ChatModel class represents an LLM where each exchange (from the end-user's perspective)
    is a string input (user's prompt) and string output (model's response). Unlike the PromptModel,
    it provides additional methods to separate the chat history (prompt/response) from the
    additional history that may be added (e.g. additional models for MemoryManager (e.g.
    summarization/embeddings)).

    A ChatModel is supplied with a `message_formatter` and `memory_manager` object. The
    `message_formatter` is used to generate the final prompt to send to the model. The
    `memory_manager` is used to modify the history of messages sent to the model, which is
    important for models that have a limited context. This is handled in the base class, because
    the memory manager depends on the message formatter and possibly other objects (e.g.
    token_calculator or cost_calculator).

    Any MemoryManager object that is passed in that has a `history()` function will be included in
    the `history()` of the underlying ChatModel object.
    """

    def __init__(
            self,
            system_message: str,
            message_formatter: Callable[[str | None, list[ExchangeRecord] | None, str | None], list | str],  # noqa
            token_calculator: Callable[[list[str]], int] | Callable[[str], int],
            cost_calculator: Callable[[int, int], float] | None = None,
            memory_manager: MemoryManager | None = None,
            ):
        """
        Args:
            system_message:
                The content of the message associated with the "system" `role`.
            message_formatter:
                A callable that takes the system message, the history of messages, and the prompt
                and returns a list of messages to send to the model.
            token_calculator:
                A callable that returns number of tokens in the message(s)
            cost_calculator:
                A callable that takes the number of tokens in the prompt and response and returns
                the cost.
            memory_manager:
                A callable that takes the history of messages and returns a list of messages to
                send to the model.
        """
        super().__init__(token_calculator=token_calculator, cost_calculator=cost_calculator)
        self._chat_history: list[ExchangeRecord] = []
        self.system_message = system_message
        self._message_formatter = message_formatter
        self._token_calculator = token_calculator
        self._memory_manager = memory_manager
        self._previous_messages = None

    def __call__(self, prompt: str) -> str:
        """
        Executes a chat request based on the given prompt and returns a response.

        Args:
            prompt: The prompt or question to be sent to the model.
        """
        if self._memory_manager:
            messages = self._memory_manager(
                system_message=self.system_message,
                history=self._chat_history,
                prompt=prompt,
                message_formatter=self._message_formatter,
                token_calculator=self._token_calculator,
                cost_calculator=self._cost_calculator,
            )
        else:
            messages = self._message_formatter(
                system_message=self.system_message,
                history=self._chat_history,
                prompt=prompt,
            )
        self._previous_messages = messages
        response = super().__call__(prompt=messages)
        self._history[-1].prompt = prompt
        self._history[-1].metadata['messages'] = messages
        # the reason to separate out the history from chat_history is so that subclasses can add
        # additional history (via from e.g. memory_manager / summarization)
        # without it being included in the chat history.
        self._chat_history.append(self._history[-1])
        return response

    @property
    def chat_history(self) -> list[ExchangeRecord]:
        """Returns the chat history (prompt, response, metadata, etc.) for the model."""
        return self._chat_history.copy()

    def _get_history(self) -> list[ExchangeRecord]:
        """A list of ExchangeRecord objects for tracking chat messages (prompt/response)."""
        history = self._history.copy()
        if self._memory_manager and _has_history(self._memory_manager):
            history += self._memory_manager.history()
        return history
