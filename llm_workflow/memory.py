"""
Memory refers to the information sent to a chat model (i.e. how much memory/context the model is
given). Within the context of an OpenAI model, it is the list of messages (a list of dictionaries
where the `role` is `system`, `assistant`, or `user`). Sending the entire list of messages means
the model uses the entire history as context for the answer. However, we can't keep sending the
entire history indefinitely, since we will exceed the maximum context length.

The classes defined below are used to create different strategies for managing memory. The
MemoryManager object is called by passing it a list of ExchangeRecord objects. It returns a list of
ExchangeRecord objects based on its internal logic for filtering/managing the messages/memory. Why
pass a list of ExchangeRecords rather than the underlying list of messages? Because the
ExchangeRecord should be a common interface across all models as opposed to lists that may differ
in structure depending on the model. This makes these objects interchangable if you swap out the
underlying model.

These classes can be used with, for example, the OpenAIChat model by passing a MemoryManager object
to the memory_manager variable when initializing the model object.
"""

from typing import Any
from llm_workflow.base import MemoryManager, ExchangeRecord, PromptModel


class LastNExchangesManager(MemoryManager):
    """Returns the last `n` number of exchanges. An exchange is a prompt/response combination)."""

    def __init__(self, last_n_exchanges: int) -> None:
        super().__init__()
        self.last_n_exchanges = last_n_exchanges

    def __call__(
            self,
            system_message: str | None,
            history: list[ExchangeRecord],
            prompt: str | None,
            **kwargs: dict[str, Any],
            ) -> list[str]:
        """
        Args:
            system_message:
                The system message to be formatted and added to the memory.
            history:
                A list of ExchangeRecord objects that represent the history of the conversation.
            prompt:
                The prompt to be formatted and added to the memory.
            kwargs:
                The keyword arguments (e.g. message_formatter).
        """
        message_formatter = kwargs['message_formatter']
        history = None if self.last_n_exchanges == 0 else history[-self.last_n_exchanges:]
        return message_formatter(system_message, history, prompt)


class LastNTokensMemoryManager(MemoryManager):
    """
    For models that require a specific message format e.g. "[INST] ... [/INST], this class will
    keep as many recent messages that remain under the token limit. The system messages will always
    be retained.

    Returns a list of formatted messages that are under the token limit.
    """

    def __init__(self, last_n_tokens: int) -> None:
        """
        Args:
            last_n_tokens:
                The maximum number of tokens that can be used (and returned) in the messages.
        """
        super().__init__()
        self.last_n_tokens = last_n_tokens

    def __call__(  # noqa
            self,
            system_message: str | None,
            history: list[ExchangeRecord],
            prompt: str | None,
            **kwargs: dict[str, Any],
            ) -> list[str]:
        """
        Args:
            system_message:
                The system message to be formatted and added to the memory.
            history:
                A list of ExchangeRecord objects that represent the history of the conversation.
            prompt:
                The prompt to be formatted and added to the memory.
            kwargs:
                The keyword arguments (e.g. message_formatter).
        """
        # the output of message_formatter is either a single string a or list of messages
        # (e.g. strings, dicts, etc.)
        # if it's a string then we know that the output is a single message and we have to convert
        # it to a list so that we can iterate over it and reverse it
        message_formatter = kwargs['message_formatter']
        token_calculator = kwargs['token_calculator']
        test_formatting = message_formatter(system_message=None, history=None, prompt='test')
        if isinstance(test_formatting, str):
            is_list = False
        elif isinstance(test_formatting, list):
            is_list = True
        else:
            raise ValueError(
                "The output of message_formatter must be either a string or a list of strings.",
            )
        if system_message:
            system_message = message_formatter(system_message, [], None)
            system_tokens = token_calculator(system_message)
            if not is_list:
                system_message = [system_message]
        else:
            system_tokens = 0
        if prompt:
            memory = message_formatter(None, [], prompt)
            prompt_tokens = token_calculator(memory)
            if not is_list:
                memory = [memory]
        else:
            prompt_tokens = 0
            memory = []

        tokens_used = system_tokens + prompt_tokens
        assert tokens_used <= self.last_n_tokens, (
            "The system message and prompt are too long to be added to the memory.",
        )
        # start added the most recent messages to the memory
        history = reversed(history)
        for message in history:
            # if the message's tokens plus the tokens that are already used in the memory is more
            # than the threshold then we need to break and avoid adding more memory
            formatted_message = message_formatter(None, [message], None)
            message_tokens = token_calculator(formatted_message)
            if message_tokens + tokens_used > self.last_n_tokens:
                break
            if is_list:
                # if the output is a list, then we need to reverse it so that when we reverse at
                # the end of the function we get the correct order
                formatted_message = list(reversed(formatted_message))
            else:
                formatted_message = [formatted_message]
            memory += formatted_message
            tokens_used += message_tokens

        if system_message:
            memory += system_message

        memory = list(reversed(memory))
        return memory if is_list else ''.join(memory)


class MessageSummaryMemoryManager(MemoryManager):
    """
    The MessageSummaryMemoryManager class summarizes messages that exceed a certain threshold. The
    treshold is applied to both the prompt and the response, individually. The summaries are
    generated using a model provided to the __init__ method. The summaries are cached so that the
    same prompt/response pair is not summarized multiple times.

    NOTE: One improvement might be ot only summarize messages after the total number of tokens (or
    characters) across all messages exceeds a certain threshold. This would prevent summarizing
    messages when the overall number of tokens is relatively small. Another improvement might be
    to add logic to ensure the overall number of tokens (or characters) across all messages does
    not exceed a certain threshold (e.g. the context window threshold for the model).
    """

    def __init__(
            self,
            model: PromptModel,
            message_threshold: int = 150,
            summarize_instruction: str = 'Summarize the following while retaining the important information.',  # noqa
            ) -> None:
        """
        Args:
            model:
                The model to use for summarizing messages.
            message_threshold:
                The maximum number of characters that a message (the prompt or the response,
                individually) can be before it is summarized.
            summarize_instruction:
                The instruction to use when summarizing messages.
        """
        super().__init__()
        self._model = model
        self._model.system_message = None  # for any chat models; we don't need/want a system msg
        self._message_threshold = message_threshold
        self._summarize_instruction = summarize_instruction
        self._history = []
        self._cache = {}

    def __call__(
        self,
        system_message: str,
        history: list[ExchangeRecord],
        prompt: str,
        **kwargs: dict[str, Any]) -> str | list[str] | list[dict[str, str]]:
        """
        Initialize the object.

        Args:
            system_message:
                The system message passed to the `message_formatter` of the summarizer model.
            history:
                The history of messages to summarize.
            prompt:
                The prompt from the user, which is used to construct the final message sent to the
                final model.
            kwargs:
                A mechanism for passing additional arguments to MemoryManager objects specific to
                the MemoryManager implementation.
        """
        message_formatter = kwargs['message_formatter']
        # for each of the previous prompt/responses in history, we want to summarize the prompt
        # and response using the model; we want to cache the results so that we don't have to
        # summarize the same prompt/response pair multiple times
        summarized_records = []
        for record in history:
            if record.uuid not in self._cache:
                if len(record.prompt) > self._message_threshold:
                    summarized_prompt = self._model(self._summarize_format_message(record.prompt))
                else:
                    summarized_prompt = record.prompt
                if len(record.response) > self._message_threshold:
                    summarized_response = self._model(self._summarize_format_message(record.response))  # noqa
                else:
                    summarized_response = record.response
                self._cache[record.uuid] = {
                    'prompt': summarized_prompt,
                    'response': summarized_response,
                }
            summarized_records.append(ExchangeRecord(
                prompt=self._cache[record.uuid]['prompt'],
                response=self._cache[record.uuid]['response'],
            ))

        # we need to delegate to the message_formatter to format the system message and prompt
        # because this may change depending on the model
        return message_formatter(
            system_message,
            summarized_records,
            prompt,
        )

    def _summarize_format_message(self, message: str) -> str:
        """Formats the prompt sent to the summarizer model."""
        return f"{self._summarize_instruction}:\n\n```\n{message}\n```"

    def history(self) -> list[ExchangeRecord]:
        """
        Exposes the history of the underlying summarizer model. This is necesssary because Workflow
        objects build up history by checking for a `history` method on the the underlying tasks. By
        exposing the history of the underlying summarizer model (which is then exposed in the
        history method of the ChatModel object), we can ensure that the calls to summarize messages
        are included in the history of the primary model and workflow (if applicable), along with
        costs, tokens used, etc.).
        """
        return self._model.history()
