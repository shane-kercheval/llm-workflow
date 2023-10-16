"""Contains helper functions for interacting with Hugging Face models."""

import os
import time
import requests
from typing import Callable
from transformers import PreTrainedTokenizer, AutoTokenizer
from llm_workflow.internal_utilities import retry_handler
from llm_workflow.base import ChatModel, ExchangeRecord, StreamingEvent


def query_hugging_face_endpoint(
        endpoint_url: str,
        payload: dict,
        ) -> dict:
    """
    Queries a Hugging Face endpoint and returns the response. Expects the environment variable
    HUGGING_FACE_API_KEY to be set. See https://ui.endpoints.huggingface.co/ for more info.

    Args:
        endpoint_url:
            The URL of the endpoint. Generated when you deploy a model to Hugging Face Endpoints.
        payload:
            The payload to send to the endpoint.

            For example:

            ```
            "parameters": {
                'max_new_tokens': self._max_streaming_tokens,  # controls chunk/delta size
                'temperature': self.temperature,
            }
            ```
    """
    headers = {
        "Authorization": f"Bearer {os.getenv('HUGGING_FACE_API_KEY')}",
        "Content-Type": "application/json",
    }
    repsonse = retry_handler()(
        requests.post,
        endpoint_url,
        headers=headers,
        json=payload,
    )
    return repsonse.json()


def get_tokenizer(model_path: str) -> PreTrainedTokenizer:
    """
    Returns a tokenizer for the given model path. Expects the environment variable
    HUGGING_FACE_API_KEY to be set. See https://ui.endpoints.huggingface.co/ for more info.

    Args:
        model_path:
            The path to the model. For example: 'meta-llama/Llama-2-7b-chat-hf'
    """
    return AutoTokenizer.from_pretrained(
        model_path,
        token=os.getenv('HUGGING_FACE_API_KEY'),
    )


def num_tokens(
        value: str,
        tokenizer: PreTrainedTokenizer,
        device: str = 'cpu',
        ) -> int:
    """
    Returns the number of tokens in the given string based on the `tokenizer`.

    Args:
        value:
            The string from which to calculate the number of tokens.
        tokenizer:
            The tokenizer to use to calculate the number of tokens. For example, the tokenizer
            returned by `get_tokenizer`.
        device:
            The device to use for the calculation. Valid values are 'cpu' and 'cuda'. Defaults to
            'cpu'.
    """
    tokens = tokenizer([value], return_tensors="pt").to(device)
    return len(tokens['input_ids'][0])


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


class HuggingFaceEndpointChat(ChatModel):
    """
    A wrapper around a model being served via Hugging Face Endpoints. More info here:
    https://ui.endpoints.huggingface.co/.

    NOTE: This class expects the environment variable HUGGING_FACE_API_KEY to be set.

    This class manages the messages that are sent to the model and, by default, sends all
    messages previously sent to the model in subsequent requests. Therefore, each object created
    represents a single conversation. The number of messages sent to the model can be controlled
    via `memory_manager` (e.g. `LastNTokensMemoryManager`).
    """

    def __init__(
            self,
            endpoint_url: str,
            system_message: str = 'You are a helpful assistant. Be concise and clear but give good explainations.',  # noqa
            message_formatter: Callable[[str, list[ExchangeRecord]], str] = llama_message_formatter,  # noqa
            temperature: float = 0.001,
            token_calculator: Callable[[str], int] = len,
            memory_manager: Callable[[list[ExchangeRecord]], list[str]] | None = None,
            streaming_callback: Callable[[StreamingEvent], None] | None = None,
            max_streaming_tokens: int = 10,
            timeout: int = 30,
            ) -> None:
        """
        Args:
            endpoint_url:
                The URL of the endpoint. Generated when you deploy a model to Hugging Face
                Endpoints.
            system_message:
                The content of the message associated with the "system" `role`.
            message_formatter:
                A callable that takes the system message, the history of messages, and the prompt
                and returns a list of messages to send to the model.
            temperature:
                The temperature to use when generating text. Defaults to 0.001 (must be > 0).
            token_calculator:
                A callable that takes a string and returns the number of tokens in the string.
                Defaults to `len` which returns the number of characters rather than "tokens".
            memory_manager:
                A callable that takes the history of messages and returns a list of messages to
                send to the model.
            streaming_callback:
                Callable that takes a StreamingEvent object, which contains the streamed token (in
                the `response` property and perhaps other metadata.
            max_streaming_tokens:
                The maximum number of tokens to return from the model in a single request when
                streaming.
            timeout:
                The maximum number of seconds to wait for a response from the model.
        """
        super().__init__(
            system_message=system_message,
            message_formatter=message_formatter,
            token_calculator=token_calculator,
            cost_calculator=None,
            memory_manager=memory_manager,
        )
        self.endpoint_url = endpoint_url
        self.streaming_callback = streaming_callback
        self.temperature = temperature
        self._max_streaming_tokens = max_streaming_tokens
        self._timeout = timeout

    def _run(self, messages: str) -> ExchangeRecord:
        """Runs the model based on the prompt and returns the response."""
        response = ""
        start = time.time()
        while True:
            if time.time() > start + self._timeout:
                break
            output = retry_handler()(
                query_hugging_face_endpoint,
                endpoint_url=self.endpoint_url,
                payload={
                    "inputs": messages + response,
                    "parameters": {
                        'max_new_tokens': self._max_streaming_tokens,  # controls chunk/delta size
                        'temperature': self.temperature,
                    },
                },
            )
            if isinstance(output, dict) and 'error' in output:
                response += f"\n\n{output['error']}"
                break
            if not output:
                break
            delta = output[0]['generated_text']
            if delta.strip() == '':
                break
            if self.streaming_callback:
                self.streaming_callback(StreamingEvent(response=delta))
            response += delta

        metadata = {
            'endpoint_url': self.endpoint_url,
            'temperature': self.temperature,
            'max_streaming_tokens': self._max_streaming_tokens,
            'timeout': self._timeout,
        }
        return response, metadata
