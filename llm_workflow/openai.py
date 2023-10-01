"""Contains helper functions for interacting with OpenAI models."""


from typing import Callable
from functools import cache
import tiktoken
from tiktoken import Encoding
from llm_workflow.base import Document
from llm_workflow.internal_utilities import retry_handler
from llm_workflow.models import (
    EmbeddingModel,
    EmbeddingRecord,
    ExchangeRecord,
    MemoryManager,
    PromptModel,
    StreamingEvent,
)
from llm_workflow.resources import MODEL_COST_PER_TOKEN


@cache
def _get_encoding_for_model(model_name: str) -> Encoding:
    """Gets the encoding for a given model so that we can calculate the number of tokens."""
    return tiktoken.encoding_for_model(model_name)


def num_tokens(model_name: str, value: str) -> int:
    """For a given model, returns the number of tokens based on the str `value`."""
    return len(_get_encoding_for_model(model_name=model_name).encode(value))


def num_tokens_from_messages(model_name: str, messages: list[dict]) -> int:
    """
    Copied from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    Returns the number of tokens used by a list of messages.
    """
    if model_name in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model_name == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model_name:
        # Warning: gpt-3.5-turbo may update over time.
        # Returning num tokens assuming gpt-3.5-turbo-0613
        return num_tokens_from_messages(model_name="gpt-3.5-turbo-0613", messages=messages)
    elif "gpt-4" in model_name:
        # Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.
        return num_tokens_from_messages(model_name="gpt-4-0613", messages=messages)
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model_name}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")  # noqa
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(_get_encoding_for_model(model_name=model_name).encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


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
        self._previous_messages = None

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
        history = self.history().copy()
        if self._memory_manager:
            history = self._memory_manager(history=history)

        # initial message; always keep system message regardless of memory_manager
        messages = [self._system_message]
        for message in history:
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

        self._previous_messages = messages
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
