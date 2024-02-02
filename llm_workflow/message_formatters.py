"""Contains helper functions for formatting messages for various models."""

from llm_workflow.base import ExchangeRecord


def openai_message_formatter(
        system_message: str | None,
        history: list[ExchangeRecord] | None,
        prompt: str | None) -> list[dict]:
    """
    A message formatter takes a system_message, list of messages (ExchangeRecord objects), and a
    prompt, and formats them according to the best practices for interacting with the model.
    """
    # initial message; always keep system message regardless of memory_manager
    messages = []
    if system_message:
        messages += [{'role': 'system', 'content': system_message}]
    if history:
        for message in history:
            messages += [
                {'role': 'user', 'content': message.prompt},
                {'role': 'assistant', 'content': message.response},
            ]
    if prompt:
        messages += [{'role': 'user', 'content': prompt}]
    return messages


def llama_message_formatter(
        system_message: str | None,
        history: list[ExchangeRecord] | None,
        prompt: str | None) -> str:
    """
    A message formatter takes a list of messages (ExchangeRecord objects) and formats them
    according to the best practices for interacting with the model.

    For example, for Lamma-2-7b, the messages should be formatted as follows:
        [INST] <<SYS>> You are a helpful assistant. <</SYS>> [/INST]
        [INST] Hello, how are you? [/INST]
        I am doing well. How are you?
        [INST] I am doing well. How's the weather? [/INST]
        It is sunny today.

    https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

    Args:
        system_message:
            The content of the message associated with the "system" `role`.
        history:
            A list of ExchangeRecord objects, containing the prompt/response pairs.
        prompt:
            The next prompt to be sent to the model.
    """
    formatted_messages = []
    if system_message:
        formatted_messages.append(f"[INST] <<SYS>> {system_message} <</SYS>> [/INST]\n")
    if history:
        for message in history:
            formatted_messages.append(
                f"[INST] {message.prompt} [/INST]\n" + f"{message.response}\n",
            )
    if prompt:
        formatted_messages.append(f"[INST] {prompt} [/INST]\n")
    return ''.join(formatted_messages)


def mistral_message_formatter(
        system_message: str | None,
        history: list[ExchangeRecord] | None,
        prompt: str | None) -> str:
    """
    A message formatter takes a list of messages (ExchangeRecord objects) and formats them
    according to the best practices for interacting with the model.

    For example, for a Mistral, the messages should be formatted as follows:
        You are a helpful assistant.[INST]Hello, how are you?[/INST]
        I am doing well. How are you?
        [INST]I am doing well. How's the weather?[/INST]
        It is sunny today.

    Args:
        system_message:
            The content of the message associated with the "system" `role`.
        history:
            A list of ExchangeRecord objects, containing the prompt/response pairs.
        prompt:
            The next prompt to be sent to the model.
    """
    formatted_messages = []
    if system_message:
        formatted_messages.append(f"{system_message}")
    if history:
        for message in history:
            formatted_messages.append(
                f"[INST]{message.prompt}[/INST] " + f"{message.response}\n\n",
            )
    if prompt:
        formatted_messages.append(f"[INST]{prompt}[/INST] ")
    return ''.join(formatted_messages)
