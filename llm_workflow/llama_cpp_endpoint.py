"""
Provides helper classes and functions for using models deployed via llama.cpp's server component.

Instructions for building and running the lamma.cpp server can be found here:

    https://github.com/ggerganov/llama.cpp/tree/master/examples/server
"""
import json
import os
import time
import requests
from typing import Callable
from llm_workflow.base import ChatModel, ExchangeRecord, StreamingEvent
from llm_workflow.internal_utilities import retry_handler


def query_lamma_cpp_endpoint(
        endpoint_url: str,
        payload: dict,
        stream: bool = False,
        ) -> dict:
    """
    Queries a llama.cpp server endpoint and returns the response. Expects the environment variable
    LLAMA_CPP_ENDPOINT_API_KEY to be set if required.

    Args:
        endpoint_url:
            The URL and port of the endpoint. For example: 'http://localhost:8080/completion'.
        payload:
            The payload to send to the endpoint.

            For example:

            ```
            "parameters": {
                'max_new_tokens': self._max_streaming_tokens,  # controls chunk/delta size
                'temperature': self.temperature,

            }
            ```
        stream:
            Whether to stream the response or not.
    """
    api_key = os.getenv('LLAMA_CPP_ENDPOINT_API_KEY')
    headers = {
        "Content-Type": "application/json",
    }
    if api_key :
        headers["Authorization"] = f"Bearer {api_key}"
    payload['stream'] = stream
    return retry_handler()(
        requests.post,
        endpoint_url,
        headers=headers,
        json=payload,
        stream=stream,
    )


class LlamaCppEndpointChat(ChatModel):
    """
    A wrapper around a model being served via a llama.cpp server. More info here:
    https://github.com/ggerganov/llama.cpp/tree/master/examples/server.

    NOTE: This class expects the environment variable LLAMA_CPP_ENDPOINT_API_KEY to be set if
    needed.

    This class manages the messages that are sent to the model and, by default, sends all
    messages previously sent to the model in subsequent requests. Therefore, each object created
    represents a single conversation. The number of messages sent to the model can be controlled
    via `memory_manager` (e.g. `LastNTokensMemoryManager`).
    """

    def __init__(  # noqa: D417
            self,
            endpoint_url: str,
            message_formatter: Callable[[str, list[ExchangeRecord]], str],
            parameters: dict | None = None,
            system_message: str = 'You are a helpful AI assistant.',
            token_calculator: Callable[[str], int] = len,
            memory_manager: Callable[[list[ExchangeRecord]], list[str]] | None = None,
            streaming_callback: Callable[[StreamingEvent], None] | None = None,
            timeout: int = 30,
            ) -> None:
        """
        Args:
            endpoint_url:
                The URL of the endpoint.
            system_message:
                The content of the message associated with the "system" `role`.
            message_formatter:
                A callable that takes the system message, the history of messages, and the prompt
                and returns a list of messages to send to the model.
            parameters:
                The parameters to send to the endpoint/model (e.g. `temperature`).
            token_calculator:
                A callable that takes a string and returns the number of tokens in the string.
                Defaults to `len` which returns the number of characters rather than "tokens".
            memory_manager:
                A callable that takes the history of messages and returns a list of messages to
                send to the model.
            streaming_callback:
                Callable that takes a StreamingEvent object, which contains the streamed token (in
                the `response` property and perhaps other metadata.
            timeout:
                The maximum number of seconds to wait for a response from the model.
        """  # noqa
        super().__init__(
            system_message=system_message,
            message_formatter=message_formatter,
            token_calculator=token_calculator,
            cost_calculator=None,
            memory_manager=memory_manager,
        )
        self.endpoint_url = endpoint_url
        self.streaming_callback = streaming_callback
        self._timeout = timeout
        self.parameters = parameters or {}

    def _run(self, messages: str) -> ExchangeRecord:
        """Runs the model based on the prompt and returns the response."""
        response = ""
        self.parameters['prompt'] = messages
        start = time.time()
        metadata = {
            'endpoint_url': self.endpoint_url,
            'parameters': self.parameters,
            'timeout': self._timeout,
        }
        if self.streaming_callback:
            output = retry_handler()(
                query_lamma_cpp_endpoint,
                endpoint_url=self.endpoint_url,
                payload=self.parameters,
                stream=True,
            )
            for line in output.iter_lines():
                if time.time() > start + self._timeout:
                    break
                if line:
                    json_str = line.decode('utf-8').split(":", 1)[1].strip()
                    parsed_json = json.loads(json_str)
                    content = parsed_json.get("content", "")
                    if content:
                        self.streaming_callback(StreamingEvent(response=content))
                        response += content
            parsed_json
        else:
            output = retry_handler()(
                query_lamma_cpp_endpoint,
                endpoint_url=self.endpoint_url,
                payload=self.parameters,
                stream=False,
            )
            output_json = output.json()
            metadata['generation_settings'] = output_json['generation_settings']
            response = output_json['content']
        metadata['response'] = response
        return response, metadata
