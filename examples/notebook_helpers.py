"""Helper methods for notebooks."""
from IPython.display import display, Markdown
from llm_workflow.base import ExchangeRecord


def usage_string(
        obj: object,
        cost_precision: int = 5) -> str:
    """Returns a friendly string containing cost/usage."""
    value = ""
    if getattr(obj, 'cost') and obj.cost:
        value += f"Cost:              ${obj.cost:.{cost_precision}f}\n"
    if getattr(obj, 'total_tokens') and obj.total_tokens:
        value += f"Total Tokens:       {obj.total_tokens:,}\n"
    if getattr(obj, 'prompt_tokens') and obj.prompt_tokens:
        value += f"Prompt Tokens:      {obj.prompt_tokens:,}\n"
    if getattr(obj, 'response_tokens') and obj.response_tokens:
        value += f"Response Tokens:    {obj.response_tokens:,}\n"
    if getattr(obj, 'embedding_tokens') and obj.embedding_tokens:
        value += f"Embedding Tokens:   {obj.embedding_tokens:,}\n"

    return value


def mprint(value: str) -> None:
    """Print/display the `value` as markdown in a notebook cell."""
    display(Markdown(value))


def messages_string(messages: list[ExchangeRecord], cost_precision: int = 5) -> str:
    """
    Returns a string containing the formatted messages; can be used with `mprint` to display the
    message history in a notebook cell.
    """
    value = '---\n\n'
    for index, message in enumerate(messages):
        value += f"### Message {index + 1}\n\n"
        value += f"{message.timestamp}; `${message.cost:.{cost_precision}f}`; `{message.total_tokens:,}` Tokens\n"  # noqa
        value += f"#### Prompt ({message.prompt_tokens} tokens):\n"
        value += f"\n{message.prompt}\n"
        value += f"#### Response ({message.response_tokens} tokens):\n"
        value += f"\n{message.response}\n"
        value += '\n---\n\n'
    return value
