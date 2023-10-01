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

from typing import Callable
from llm_workflow.models import MemoryManager, ExchangeRecord


class LastNExchangesManager(MemoryManager):
    """Returns the last `n` number of exchanges. An exchange is a prompt/response combination)."""

    def __init__(self, last_n_exchanges: int) -> None:
        super().__init__()
        self.last_n_exchanges = last_n_exchanges

    def __call__(self, history: list[ExchangeRecord]) -> list[ExchangeRecord]:
        """
        Takes a list of `ExchangeRecord` objects and returns the last `n` messages based on the
        `last_n_message` variable set during initialization.
        """
        if self.last_n_exchanges == 0:
            return []
        return history[-self.last_n_exchanges:]


class TokenWindowManager(MemoryManager):
    """Returns the last x number of messages that are within a certain threshold of tokens."""

    def __init__(self, last_n_tokens: int) -> None:
        super().__init__()
        self.last_n_tokens = last_n_tokens

    def __call__(self, history: list[ExchangeRecord]) -> list[ExchangeRecord]:
        """
        Takes a list of `ExchangeRecord` objects and returns the last x messages where the
        aggregated number of tokens is less than the `last_n_message` variable set during
        initialization.
        """
        history = reversed(history)
        memory = []
        tokens_used = 0
        for message in history:
            # if the message's tokens plus the tokens that are already used in the memory is more
            # than the threshold then we need to break and avoid adding more memory
            if message.total_tokens + tokens_used > self.last_n_tokens:
                break
            memory.append(message)
            tokens_used += message.total_tokens
        return reversed(memory)


class MessageFormatterMaxTokensMemoryManager(MemoryManager):
    """
    For models that require a specific message format e.g. "[INST] ... [/INST], this class will
    keep as many recent messages that remain under the token limit. The system messages will always
    be retained.

    Returns a list of formatted messages that are under the token limit.
    """

    def __init__(
            self,
            last_n_tokens: int,
            calculate_num_tokens: Callable[[str], int],
            message_formatter: Callable[[str, list[ExchangeRecord]], str],
            ) -> None:
        """
        Args:
            last_n_tokens:
                The maximum number of tokens that can be used (and returned) in the messages.
            calculate_num_tokens:
                A function that takes a string and returns the number of tokens it contains.
            message_formatter:
                A function that takes a system message, a list of messages, and a prompt and
                returns a list of formatted messages.
        """
        super().__init__()
        self.last_n_tokens = last_n_tokens
        self._calculate_num_tokens = calculate_num_tokens
        self._message_formatter = message_formatter

    def __call__(
            self,
            system_message: str | None,
            history: list[ExchangeRecord],
            prompt: str | None,
            ) -> list[str]:
        """
        Args:
            system_message:
                The system message to be formatted and added to the memory.
            history:
                A list of ExchangeRecord objects that represent the history of the conversation.
            prompt:
                The prompt to be formatted and added to the memory.
        """
        if prompt:
            memory = self._message_formatter(None, [], prompt)
            prompt_tokens = self._calculate_num_tokens(memory[0])
        else:
            prompt_tokens = 0
            memory = []
        if system_message:
            system_message = self._message_formatter(system_message, [], None)[0]
            system_tokens = self._calculate_num_tokens(system_message)
        else:
            system_tokens = 0

        # start added the most recent messages to the memory
        history = reversed(history)
        tokens_used = system_tokens + prompt_tokens
        for message in history:
            # if the message's tokens plus the tokens that are already used in the memory is more
            # than the threshold then we need to break and avoid adding more memory
            formatted_message = self._message_formatter(None, [message], None)[0]
            message_tokens = self._calculate_num_tokens(formatted_message)
            if message_tokens + tokens_used > self.last_n_tokens:
                break
            memory.append(formatted_message)
            tokens_used += message_tokens

        if system_message:
            memory.append(system_message)

        return list(reversed(memory))
