"""Contains helper functions for formatting LLM messages based on various types of models."""

from llm_workflow.base import ExchangeRecord


def openai_message_formatter(
        system_message: str | None,
        messages: list[ExchangeRecord | tuple | dict] | None,
        prompt: str | None) -> list[dict]:
    """
    A message formatter takes a system_message, list of messages (ExchangeRecord objects), and a
    prompt, and formats them according to the best practices for interacting with the model.

    Args:
        system_message:
            A system message to include at the beginning of the formatted messages.
        messages:
            A list of ExchangeRecord objects, a list of dictionaries, or a list of tuples.

            If a list of tuples, each tuple should have two elements: the prompt as the first
            element and the response as the second element.

            If a list of dictionaries is passed in, each dictionary contains a two key-value
            pairs, where one key is 'user' and the value is the user's message, and the other
            key is 'assistant' and the value is the assistant's response.
        prompt:
            A prompt to include at the end of the formatted messages.
    """
    # initial message; always keep system message regardless of memory_manager
    formatted_messages = []
    if system_message:
        formatted_messages += [{'role': 'system', 'content': system_message}]
    if messages:
        for message in messages:
            if isinstance(message, ExchangeRecord):
                formatted_messages += [
                    {'role': 'user', 'content': message.prompt},
                    {'role': 'assistant', 'content': message.response},
                ]
            elif isinstance(message, tuple):
                formatted_messages += [
                    {'role': 'user', 'content': message[0]},
                    {'role': 'assistant', 'content': message[1]},
                ]
            elif isinstance(message, dict):
                formatted_messages += [
                    {'role': 'user', 'content': message['user']},
                    {'role': 'assistant', 'content': message['assistant']},
                ]
            else:
                raise ValueError(
                    f"Invalid message format: {message}. ",
                    "Expected ExchangeRecord, tuple, or dict.",
                )
    if prompt:
        formatted_messages += [{'role': 'user', 'content': prompt}]
    return formatted_messages

# For example, for Lamma-2-7b, the messages should be formatted as follows:
#     [INST] <<SYS>> You are a helpful assistant. <</SYS>> [/INST]
#     [INST] Hello, how are you? [/INST]
#     I am doing well. How are you?
#     [INST] I am doing well. How's the weather? [/INST]
#     It is sunny today.
SYSTEM_FORMAT_LLAMA = "[INST] <<SYS>> {system_message} <</SYS>> [/INST]\n"
PROMPT_FORMAT_LLAMA = "[INST] {prompt} [/INST]"
RESPONSE_PREFIX_LLAMA = "\n"


# For example, for a Mistral, the messages should be formatted as follows:
#     You are a helpful assistant.[INST]Hello, how are you?[/INST]
#     I am doing well. How are you?
#     [INST]I am doing well. How's the weather?[/INST]
#     It is sunny today.
SYSTEM_FORMAT_MISTRAL = "{system_message}"
PROMPT_FORMAT_MISTRAL = "[INST]{prompt}[/INST]"
RESPONSE_PREFIX_MISTRAL = ""


# def _format_template(template: str, content: str) -> str:
#     """Formats a single message using a template."""
#     return template.format(**{'system_message': content, 'prompt': content, 'response': content})


class MessageFormatter:
    """Callable object that formats messages."""

    def __init__(  # noqa: D417
            self,
            system_format: str | None = '{system_message}',
            prompt_format: str | None = '{prompt}',
            response_prefix: str | None = '') -> None:
        """
        Initialize the message formatter.
        
        Args:
            system_format:
                The format string for system messages.

                Examples:
                    - Llama: "[INST] <<SYS>> {system_message} <</SYS>> [/INST]\n"
                    - Mistral: "{system_message}"
                    - Zephyr: "<|system|>\n{system_message}\n\n"

            prompt_format:
                The format string for prompt messages.

                Examples:
                    - Llama: "[INST] {prompt} [/INST]\n"
                    - Mistral: "[INST]{prompt}[/INST]"
                    - Zephyr: "<|user|>\n{prompt}\n"
            response_prefix:
                The string to prefix responses with (i.e. the format to prompt the model into
                responding).

                Examples:
                    - Llama: ""
                    - Mistral: ""
                    - Zephyr: "<|assistant|>\n"
        """  # noqa
        self.system_format = system_format
        self.prompt_format = prompt_format
        self.response_prefix = response_prefix

    def __call__(
            self,
            system_message: str | None,
            messages: list[ExchangeRecord | tuple | dict] | None,
            prompt: str | None) -> str:
        """
        Formats messages for interacting with an LLM.

        Args:
            system_message:
                A system message to include at the beginning of the formatted messages.
            messages:
                A list of ExchangeRecord objects, a list of dictionaries, or a list of tuples.

                If a list of tuples, each tuple should have two elements: the prompt as the first
                element and the response as the second element.

                If a list of dictionaries is passed in, each dictionary contains a two key-value
                pairs, where one key is 'user' and the value is the user's message, and the other
                key is 'assistant' and the value is the assistant's response.
            prompt:
                A prompt to include at the end of the formatted messages.
        """
        formatted_messages = []
        if system_message:
            formatted_messages.append(self.system_format.format(system_message=system_message))
        if messages:
            for message in messages:
                if isinstance(message, ExchangeRecord):
                    prompt_text = message.prompt
                    response_text = message.response
                elif isinstance(message, tuple):
                    prompt_text, response_text = message
                elif isinstance(message, dict):
                    prompt_text = message['user']
                    response_text = message['assistant']
                else:
                    raise ValueError(
                        f"Invalid message format: {message}. ",
                        "Expected ExchangeRecord, tuple, or dict.",
                    )
                formatted_prompt = self.prompt_format.format(prompt=prompt_text)
                formatted_response = self.response_prefix + response_text
                formatted_messages.append(formatted_prompt + formatted_response)
        if prompt:
            text = self.prompt_format.format(prompt=prompt) + self.response_prefix
            formatted_messages.append(text)
        return ''.join(formatted_messages)


class LlamaMessageFormatter(MessageFormatter):
    """Callable object that formats messages for Llama-based models."""

    def __init__(self) -> None:
        """Initialize the message formatter."""
        super().__init__(
            system_format=SYSTEM_FORMAT_LLAMA,
            prompt_format=PROMPT_FORMAT_LLAMA,
            response_prefix=RESPONSE_PREFIX_LLAMA,
        )


class MistralMessageFormatter(MessageFormatter):
    """Callable object that formats messages for Mistral models."""

    def __init__(self) -> None:
        """Initialize the message formatter."""
        super().__init__(
            system_format=SYSTEM_FORMAT_MISTRAL,
            prompt_format=PROMPT_FORMAT_MISTRAL,
            response_prefix=RESPONSE_PREFIX_MISTRAL,
        )
