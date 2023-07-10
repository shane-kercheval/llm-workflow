"""Contains models."""

from collections.abc import Callable
import json
from llm_workflow.base import LanguageModel, PromptModel, Document, EmbeddingRecord, EmbeddingModel, \
    MemoryManager, ExchangeRecord, StreamingEvent
from llm_workflow.tools import Tool
from llm_workflow.resources import MODEL_COST_PER_TOKEN
from llm_workflow.utilities import num_tokens, num_tokens_from_messages
from llm_workflow.internal_utilities import retry_handler


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
        memory = self.history.copy()
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


class OpenAIToolAgent(LanguageModel):
    """
    Wrapper around OpenAI "functions" (https://platform.openai.com/docs/guides/gpt/function-calling).

    Decides which Tool to call.

    TODO.
    """

    def __init__(
            self,
            model_name: str,
            tools: list[Tool],
            system_message: str = "Decide which function to use. Only use the functions you have been provided with. Don't make assumptions about what values to plug into functions.",  # noqa
            timeout: int = 10,
        ) -> dict | None:
        """
        Args:
            model_name:
                e.g. 'gpt-3.5-turbo'
            tools:
                TODO.
            system_message:
                The content of the message associated with the "system" `role`.
            timeout:
                timeout value passed to OpenAI model.
        """
        super().__init__()
        self.model_name = model_name
        self._tools = tools
        self._system_message = system_message
        self._history = []
        self.timeout = timeout


    def __call__(self, prompt: object) -> object:
        """TODO."""
        import openai
        messages = [
            {"role": "system", "content": self._system_message},
            {"role": "user", "content": prompt},
        ]
        # we want to track to track costs/etc.; but we don't need the history to build up memory
        # essentially, for now, this class won't have any memory/context of previous questions;
        # it's only used to decide which tools/functions to call
        response = retry_handler()(
                openai.ChatCompletion.create,
                model=self.model_name,
                messages=messages,
                functions=[x.properties for x in self._tools],
                temperature=0,
                # max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
        prompt_tokens = response['usage'].prompt_tokens
        completion_tokens = response['usage'].completion_tokens
        total_tokens = response['usage'].total_tokens
        cost = (prompt_tokens * self.cost_per_token['input']) + \
            (completion_tokens * self.cost_per_token['output'])
        record = ToolRecord(
            prompt=prompt,
            response='',
            metadata={'model_name': self.model_name},
            prompt_tokens=prompt_tokens,
            response_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
        )
        self._history.append(record)
        function_call = response["choices"][0]["message"].get('function_call')
        if function_call:
            function_name = function_call['name']
            function_args = json.loads(function_call['arguments'])
            record.response = f"tool: '{function_name}' - {function_args}"
            record.metadata['tool_name'] = function_name
            record.metadata['tool_args'] = function_args
            for tool in self._tools:
                if function_name == tool.name:
                    return tool(**function_args)
        return None


    @property
    def cost_per_token(self) -> dict:
        """
        Returns a dictionary containing 'input' and 'output' keys each containing a float
        corresponding to the cost-per-token for the corresponding token type and model.
        We need to dynamically look this up since the model_name can change over the course of the
        object's lifetime.
        """
        return MODEL_COST_PER_TOKEN[self.model_name]


    @property
    def history(self) -> list[ExchangeRecord]:
        """A list of ExchangeRecord objects for tracking chat messages (prompt/response)."""
        return self._history

class ToolRecord(ExchangeRecord):
    """TODO."""
