"""Contains models."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from llm_workflow.base import CostRecord, Document, Record, RecordKeeper
from llm_workflow.resources import MODEL_COST_PER_TOKEN
from llm_workflow.utilities import num_tokens, num_tokens_from_messages
from llm_workflow.internal_utilities import retry_handler


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


class OpenAIEmbedding(EmbeddingModel):
    """A wrapper around the OpenAI Embedding model that tracks token usage and costs."""

    def __init__(
            self,
            model_name: str,
            doc_prep: Callable[[str], str] = lambda x: x.strip().replace('\n', ' '),
            timeout: int = 10,
            ) -> None:
        """
        Args:
            model_name:
                e.g. 'text-embedding-ada-002'
            doc_prep:
                function that cleans the text of each doc before creating embedding.
            timeout:
                timeout value passed to OpenAI model.
        """
        super().__init__()
        self.model_name = model_name
        self.doc_prep = doc_prep
        self.timeout = timeout

    def _run(self, docs: list[Document]) -> tuple[list[list[float]], EmbeddingRecord]:
        import openai
        texts = [self.doc_prep(x.content) for x in docs]
        response = retry_handler()(
            openai.Embedding.create,
            input = texts,
            model=self.model_name,
            timeout=self.timeout,
        )
        total_tokens = response['usage']['total_tokens']
        embedding = [x['embedding'] for x in response['data']]
        metadata = EmbeddingRecord(
            metadata={'model_name': self.model_name},
            total_tokens=total_tokens,
            cost=self.cost_per_token * total_tokens,
        )
        return embedding, metadata

    @property
    def cost_per_token(self) -> float:
        """
        Returns a float corresponding to the cost-per-token for the corresponding model.
        We need to dynamically look this up since the model_name can change over the course of the
        object's lifetime.
        """
        return MODEL_COST_PER_TOKEN[self.model_name]


class OpenAIChat(PromptModel):
    """
    A wrapper around the OpenAI chat model (i.e. https://api.openai.com/v1/chat/completions
    endpoint). More info here: https://platform.openai.com/docs/api-reference/chat.

    This class manages the messages that are sent to OpenAI's model and, by default, sends all
    messages previously sent to the model in subsequent requests. Therefore, each object created
    represents a single conversation. The number of messages sent to the model can be controlled
    via `memory_manager`.
    """

    def __init__(
            self,
            model_name: str,
            temperature: float = 0,
            max_tokens: int = 2000,
            system_message: str = 'You are a helpful assistant.',
            streaming_callback: Callable[[StreamingEvent], None] | None = None,
            memory_manager: MemoryManager | \
                Callable[[list[ExchangeRecord]], list[ExchangeRecord]] | \
                None = None,
            timeout: int = 10,
            ) -> None:
        """
        Args:
            model_name:
                e.g. 'gpt-3.5-turbo'
            temperature:
                "What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
                make the output more random, while lower values like 0.2 will make it more focused
                and deterministic."
            max_tokens:
                The maximum number of tokens to generate in the chat completion.
                The total length of input tokens and generated tokens is limited by the model's
                context length.
            system_message:
                The content of the message associated with the "system" `role`.
            streaming_callback:
                Callable that takes a StreamingEvent object, which contains the streamed token (in
                the `response` property and perhaps other metadata.
            memory_manager:
                MemoryManager object (or callable that takes a list of ExchangeRecord objects and
                returns a list of ExchangeRecord objects. The underlying logic should return the
                messages sent to the OpenAI model.
            timeout:
                timeout value passed to OpenAI model.
        """
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.streaming_callback = streaming_callback
        self.timeout = timeout
        self._memory_manager = memory_manager
        self._system_message = {'role': 'system', 'content': system_message}
        self._previous_memory = None

    def _run(self, prompt: str) -> ExchangeRecord:
        """
        `openai.ChatCompletion.create` expects a list of messages with various roles (i.e. system,
        user, assistant). This function builds the list of messages based on the history of
        messages and based on an optional 'memory_manager' that filters the history based on
        it's own logic. The `system_message` is always the first message regardless if a
        `memory_manager` is passed in.

        The use of a streaming callback does not change the output returned from calling the object
        (i.e. a ExchangeRecord object).
        """
        import openai
        # build up messages from history
        memory = self.history().copy()
        if self._memory_manager:
            memory = self._memory_manager(history=memory)

        # initial message; always keep system message regardless of memory_manager
        messages = [self._system_message]
        for message in memory:
            messages += [
                {'role': 'user', 'content': message.prompt},
                {'role': 'assistant', 'content': message.response},
            ]
        # add latest prompt to messages
        messages += [{'role': 'user', 'content': prompt}]
        if self.streaming_callback:
            response = retry_handler()(
                openai.ChatCompletion.create,
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                stream=True,
            )
            # extract the content/token from the streaming response and send to the callback
            # build up the message so that we can calculate usage/costs and send back the same
            # ExchangeRecord response that we would return if we weren't streaming
            def get_delta(chunk):  # noqa
                delta = chunk['choices'][0]['delta']
                if 'content' in delta:
                    return delta['content']
                return None
            response_message = ''
            for chunk in response:
                delta = get_delta(chunk)
                if delta:
                    self.streaming_callback(StreamingEvent(response=delta))
                    response_message += delta

            prompt_tokens = num_tokens_from_messages(model_name=self.model_name, messages=messages)
            completion_tokens = num_tokens(model_name=self.model_name, value=response_message)
            total_tokens = prompt_tokens + completion_tokens
        else:
            response = retry_handler()(
                openai.ChatCompletion.create,
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
            response_message = response['choices'][0]['message'].content
            prompt_tokens = response['usage'].prompt_tokens
            completion_tokens = response['usage'].completion_tokens
            total_tokens = response['usage'].total_tokens

        self._previous_memory = messages
        cost = (prompt_tokens * self.cost_per_token['input']) + \
            (completion_tokens * self.cost_per_token['output'])

        return ExchangeRecord(
            prompt=prompt,
            response=response_message,
            metadata={'model_name': self.model_name},
            prompt_tokens=prompt_tokens,
            response_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
        )

    @property
    def cost_per_token(self) -> dict:
        """
        Returns a dictionary containing 'input' and 'output' keys each containing a float
        corresponding to the cost-per-token for the corresponding token type and model.
        We need to dynamically look this up since the model_name can change over the course of the
        object's lifetime.
        """
        return MODEL_COST_PER_TOKEN[self.model_name]


# class ModelHistoryMixin(RecordKeeper):
#     """
#     TODO: A ModelHistoryMixin is an object that aggregates the history across all associated
#     objects (e.g. across the tasks of a workflow object).
#     """

#     @property
#     def usage_history(self) -> list[TokenUsageRecord]:
#         """Returns all records of type UsageRecord."""
#         return self.history(TokenUsageRecord)

#     @property
#     def exchange_history(self) -> list[ExchangeRecord]:
#         """Returns all records of type ExchangeRecord."""
#         return self.history(ExchangeRecord)

#     @property
#     def embedding_history(self) -> list[EmbeddingRecord]:
#         """Returns all records of type ExchangeRecord."""
#         return self.history(EmbeddingRecord)

#     @property
#     def cost(self) -> int | None:
#         """The total cost summed across all Record objects."""
#         return self.sum(name='cost')

#     @property
#     def total_tokens(self) -> int | None:
#         """The total number of tokens summed across all Record objects."""
#         return self.sum(name='total_tokens')

#     @property
#     def prompt_tokens(self) -> int | None:
#         """The total number of prompt tokens summed across all Record objects."""
#         return self.sum(name='prompt_tokens')

#     @property
#     def response_tokens(self) -> int | None:
#         """The total number of response tokens summed across all Record objects."""
#         return self.sum(name='response_tokens')

#     @property
#     def embedding_tokens(self) -> int | None:
#         """The total number of embedding tokens summed across all EmbeddingRecord objects."""
#         return self.sum(name='total_tokens', types=EmbeddingRecord)
