"""Contains helper functions for interacting with OpenAI models."""

from copy import deepcopy
from typing import Callable
from functools import cache
import tiktoken
from tiktoken import Encoding
import numpy as np
from llm_workflow.internal_utilities import encode_image, retry_handler
from llm_workflow.base import (
    ChatModel,
    Document,
    EmbeddingModel,
    EmbeddingRecord,
    MemoryManager,
    StreamingEvent,
)
from llm_workflow.message_formatters import openai_message_formatter


CHAT_MODEL_COST_PER_TOKEN = {
    # major versions
    'gpt-4o': {'input': 5.00 / 1_000_000, 'output': 15.00 / 1_000_000},
    'gpt-4o-mini':  {'input': 0.15 / 1_000_000, 'output': 0.60 / 1_000_000},
    'gpt-4-turbo': {'input': 10.00 / 1_000_000, 'output': 30.00 / 1_000_000},
    'gpt-3.5-turbo': {'input': 0.50 / 1_000_000, 'output': 1.50 / 1_000_000},
    # minor versions
    'gpt-4o-2024-05-13': {'input': 5.00 / 1_000_000, 'output': 15.00 / 1_000_000},
    'gpt-4o-mini-2024-07-18':  {'input': 0.15 / 1_000_000, 'output': 0.60 / 1_000_000},
    # GPT-4-Turbo 128K
    'gpt-4-turbo-2024-04-09': {'input': 10.00 / 1_000_000, 'output': 30.00 / 1_000_000},
    'gpt-4-0125-preview': {'input': 0.01 / 1_000, 'output': 0.03 / 1_000},
    # GPT-3.5 Turbo 16K
    'gpt-3.5-turbo-0125': {'input': 0.50 / 1_000_000, 'output': 1.50 / 1_000_000},

    # LEGACY MODELS
    # GPT-4-Turbo 128K
    # 'gpt-4-1106-preview': {'input': 0.01 / 1_000, 'output': 0.03 / 1_000},
    # GPT-3.5-Turbo 16K
    # 'gpt-3.5-turbo-1106': {'input': 0.001 / 1_000, 'output': 0.002 / 1_000},
    # GPT-4
    'gpt-4-0613': {'input': 0.03 / 1_000, 'output': 0.06 / 1_000},
    # GPT-4- 32K
    # 'gpt-4-32k-0613': {'input': 0.06 / 1_000, 'output': 0.12 / 1_000},
    # GPT-3.5-Turbo 4K
    # 'gpt-3.5-turbo-0613': {'input': 0.0015 / 1_000, 'output': 0.002 / 1_000},
    # GPT-3.5-Turbo 16K
    # 'gpt-3.5-turbo-16k-0613': {'input': 0.003 / 1_000, 'output': 0.004 / 1_000},
}

EMBEDDING_MODEL_COST_PER_TOKEN = {
    # "Prices are per 1,000 tokens. You can think of tokens as pieces of words, where 1,000 tokens
    # is about 750 words. This paragraph is 35 tokens."
    # https://openai.com/pricing
    # https://platform.openai.com/docs/models
    ####
    # Embedding models
    ####
    # LATEST MODELS
    # https://openai.com/blog/new-embedding-models-and-api-updates
    'text-embedding-3-small': 0.02 / 1_000_000,
    'text-embedding-3-large': 0.13 / 1_000_000,
    # LEGACY MODELS
    'text-embedding-ada-002': 0.1 / 1_000_000,
}

MODEL_COST_PER_TOKEN = CHAT_MODEL_COST_PER_TOKEN | EMBEDDING_MODEL_COST_PER_TOKEN


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
        # todo: verify once .ipynb is updated
        "gpt-4-1106-preview",
        "gpt-3.5-turbo-1106",
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
            model_name: str = 'text-embedding-3-small',
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
        from openai import OpenAI
        texts = [self.doc_prep(x.content) for x in docs]
        client = OpenAI()
        response = retry_handler()(
            client.embeddings.create,
            input = texts,
            model=self.model_name,
            timeout=self.timeout,
        )
        total_tokens = response.usage.total_tokens
        embedding = [x.embedding for x in response.data]
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


class OpenAIChat(ChatModel):
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
            model_name: str = 'gpt-4o-mini',
            system_message: str = 'You are a helpful AI assistant.',
            streaming_callback: Callable[[StreamingEvent], None] | None = None,
            memory_manager: MemoryManager | None = None,
            timeout: int = 90,
            seed: int | None = None,
            **model_kwargs: dict,
            ) -> None:
        """
        Args:
            model_name:
                e.g. 'gpt-4o-mini'
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
            seed:
                seed value passed to OpenAI model.
            model_kwargs:
                Additional keyword arguments that are forwarded to the OpenAI model. For example:
                ```
                **{
                    'temperature': 0.01,
                    'max_tokens': 4096,
                }
                ```
        """  # noqa
        super().__init__(
            system_message=system_message,
            message_formatter=openai_message_formatter,
            token_calculator=self._token_calc,
            cost_calculator=self._cost_calc,
            memory_manager=memory_manager,
        )
        self.model_name = model_name
        self.model_parameters = model_kwargs or {}
        self.streaming_callback = streaming_callback
        self.timeout = timeout
        self.seed = seed

    def _cost_calc(self, input_tokens: int, response_tokens: int) -> float:
        """
        _cost_calc needs to be an instance method rather than e.g. defining inside __init__ so
        it is picklable and can be used with multiprocessing.
        """
        model_costs = MODEL_COST_PER_TOKEN[self.model_name]  # TODO fixe
        return (input_tokens * model_costs['input']) + \
            (response_tokens * model_costs['output'])

    def _token_calc(self, messages: str | list[dict]) -> int:
        if isinstance(messages, str):
            return num_tokens(model_name=self.model_name, value=messages)
        if isinstance(messages, list):
            return num_tokens_from_messages(model_name=self.model_name, messages=messages)
        raise NotImplementedError(f"""token_calculator() is not implemented for messages of type {type(messages)}.""")  # noqa

    def _create_client(self) -> object:
        """
        _create_client is used to create the OpenAI client. We cannot define this in __init__
        because it is not picklable and cannot be used with multiprocessing. Additionally, this
        function lets OpenAIServerChat override the client creation and set the endpoint_url to use
        with a local server.
        """
        from openai import OpenAI
        return OpenAI()

    def _run(self, messages: list[dict]) -> tuple[str, dict]:
        """
        `client.chat.completions.create` expects a list of messages with various roles (i.e.
        system, user, assistant). This function builds the list of messages based on the history of
        messages and based on an optional 'memory_manager' that filters the history based on
        it's own logic. The `system_message` is always the first message regardless if a
        `memory_manager` is passed in.

        The use of a streaming callback does not change the output returned from calling the object
        (i.e. a ExchangeRecord object).
        """
        client = self._create_client()
        if self.streaming_callback:
            response = retry_handler()(
                client.chat.completions.create,
                model=self.model_name,
                messages=messages,
                timeout=self.timeout,
                stream=True,
                logprobs=True,
                seed=self.seed,
                **self.model_parameters,
            )
            # extract the content/token from the streaming response and send to the callback
            # build up the message so that we can calculate usage/costs and send back the same
            # ExchangeRecord response that we would return if we weren't streaming
            def get_delta(chunk) -> tuple[str, float]:  # noqa
                choice = chunk.choices[0]
                content = choice.delta.content
                if hasattr(choice, 'logprobs') and choice.logprobs is not None:
                    log_prob = choice.logprobs.content[0].logprob if content else np.nan
                else:
                    log_prob = np.nan
                return content, log_prob

            response_message = ''
            tokens = []
            log_probs = []
            for chunk in response:
                delta, log_prob = get_delta(chunk)
                if delta:
                    self.streaming_callback(StreamingEvent(response=delta, logprob=log_prob))
                    response_message += delta
                    tokens.append(delta)
                    log_probs.append(log_prob)
        else:
            response = retry_handler()(
                client.chat.completions.create,
                model=self.model_name,
                messages=messages,
                timeout=self.timeout,
                stream=False,
                logprobs=True,
                seed=self.seed,
                **self.model_parameters,
            )
            if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs is not None:  # noqa
                tokens = [x.token for x in response.choices[0].logprobs.content]
                log_probs = [x.logprob for x in response.choices[0].logprobs.content]
            else:
                tokens = []
                log_probs = []
            response_message = response.choices[0].message.content

        metadata = {
            'model_name': self.model_name,
            'model_parameters': self.model_parameters,
            'tokens': tokens,
            'log_probs': log_probs,
            'timeout': self.timeout,
        }
        return response_message, metadata

    @property
    def cost_per_token(self) -> dict:
        """
        Returns a dictionary containing 'input' and 'output' keys each containing a float
        corresponding to the cost-per-token for the corresponding token type and model.
        We need to dynamically look this up since the model_name can change over the course of the
        object's lifetime.
        """
        return MODEL_COST_PER_TOKEN[self.model_name]


class OpenAIImageChat(OpenAIChat):
    """A wrapper around the OpenAI chat model for vision/images."""

    def __init__(
            self,
            image_url: str,
            model_name: str = 'gpt-4-vision-preview',
            system_message: str = '',
            streaming_callback: Callable[[StreamingEvent], None] | None = None,
            memory_manager: MemoryManager | None = None,
            timeout: int = 90,
            seed: int | None = None,
            **model_kwargs: dict,
            ) -> None:
        """
        Args:
            image_url:
                URL of the image. Can either be a URL or a local file path.
            model_name:
                e.g. 'gpt-4-vision-preview'
            system_message:
                The content of the message associated with the "system" `role`.
            streaming_callback:
                Callable that takes a StreamingEvent object, which contains the streamed token (in
                the `response` property and perhaps other metadata.

                NOTE: When using the streaming callback, the cost of the messages is not
                calculated. This is because the cost is only returned from OpenAI when the
                response is not streamed.
            memory_manager:
                MemoryManager object (or callable that takes a list of ExchangeRecord objects and
                returns a list of ExchangeRecord objects. The underlying logic should return the
                messages sent to the OpenAI model.
            timeout:
                timeout value passed to OpenAI model.
            seed:
                seed value passed to OpenAI model.
            model_kwargs:
                Additional keyword arguments that are forwarded to the OpenAI model. For example:
                ```
                **{
                    'temperature': 0.01,
                    'max_tokens': 4096,
                }
                ```
        """
        super().__init__(
            model_name=model_name,
            system_message=system_message,
            streaming_callback=streaming_callback,
            memory_manager=memory_manager,
            timeout=timeout,
            seed=seed,
            **model_kwargs,
        )
        if 'http' not in image_url and 'www' not in image_url:
            image_url = f'data:image/jpeg;base64,{encode_image(image_url)}'
        self._image_message = {
            'role': 'user',
            'content': [
                {
                    'type': 'image_url',
                    'image_url': {'url': image_url},
                },
            ],
        }

    def _run(self, messages: list[dict]) -> tuple[str, dict]:
        """
        `client.chat.completions.create` expects a list of messages with various roles (i.e.
        system, user, assistant). This function builds the list of messages based on the history of
        messages and based on an optional 'memory_manager' that filters the history based on
        it's own logic. The `system_message` is always the first message regardless if a
        `memory_manager` is passed in.

        The use of a streaming callback does not change the output returned from calling the object
        (i.e. a ExchangeRecord object).
        """
        messages = deepcopy(messages)
        # insert the message with the image after system prompt and before everything else.
        messages.insert(1, self._image_message)
        return super()._run(messages=messages)

    @property
    def input_tokens(self) -> int | None:
        """
        Returns the number of tokens used in the input message. Only available for non-streaming
        requests. If the request is streaming, returns None, because the number of tokens is not
        given from OpenAI when the response is streamed.
        """
        if self.streaming_callback:
            return None
        return sum(h.metadata['input_tokens'] for h in self.history())

    @property
    def response_tokens(self) -> int | None:
        """
        Returns the number of tokens used in the response message. Only available for non-streaming
        requests. If the request is streaming, returns None, because the number of tokens is not
        given from OpenAI when the response is streamed.
        """
        if self.streaming_callback:
            return None
        return sum(h.metadata['output_tokens'] for h in self.history())

    @property
    def total_tokens(self) -> int | None:
        """
        Returns the total number of tokens used in the input and response messages. Only available
        for non-streaming requests. If the request is streaming, returns None, because the number
        of tokens is not given from OpenAI when the response is streamed.
        """
        if self.streaming_callback:
            return None
        return self.input_tokens + self.response_tokens

    @property
    def cost(self) -> float | None:
        """
        Returns the cost of the input and response messages. Only available for non-streaming
        requests. If the request is streaming, returns None, because the cost is not given from
        OpenAI when the response is streamed.
        """
        if self.streaming_callback:
            return None
        return self._cost_calc(
            input_tokens=self.input_tokens,
            response_tokens=self.response_tokens,
        )


class OpenAIServerChat(OpenAIChat):
    """
    Some servers (e.g. llama.cpp, hugging face endpoints) allow callers to use OpenAI's API for
    non-OpenAI models. This wrapper allows the caller to specify the endpoint_url of the server to
    connect to and uses OpenAI API to interact with the LLM.
    """

    def __init__(
            self,
            endpoint_url: str,
            api_key: str | None = None,
            system_message: str = 'You are a helpful AI assistant.',
            streaming_callback: Callable[[StreamingEvent], None] | None = None,
            memory_manager: MemoryManager | None = None,
            timeout: int = 90,
            seed: int | None = None,
            **model_kwargs: dict,
            ) -> None:
        """TODO document."""
        super().__init__(
            system_message=system_message,
            # token_calculator=len,
            # cost_calculator=None,
            memory_manager=memory_manager,
        )
        # override ChatGPT's token/cost calculator
        self._token_calculator = len
        self._cost_calculator = None
        # model name is not applicable
        self.model_name = 'local-model'
        self.model_parameters = model_kwargs or {}
        self.streaming_callback = streaming_callback
        self.timeout = timeout
        self.seed = seed
        self.endpoint_url = endpoint_url
        self.api_key = api_key

    def _create_client(self) -> object:
        from openai import OpenAI
        return OpenAI(base_url=self.endpoint_url, api_key=self.api_key)
