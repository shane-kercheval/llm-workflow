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
