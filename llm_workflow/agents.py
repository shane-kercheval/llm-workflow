"""
TODO.
Contains helper functions and classes that can be used within a workflow.

A Tool is meant to be used by an LLM, so it returns a string (which can be fed back into

Any callable can be passed to a workflow, so a "tool" in this file can be a simple function, or it
ca be a `Tool` class. A `Tool` class has a `name` and `description` that is potentially sent to an
LLM (e.g. OpenAI "functions") to describe when the tool should be used.

Some classes can be both tasks and Tools. A `task` is simply a callable that tracks history. So if
it's useful for a tool to track history, then it can be both.
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
import json
from typing import Any
from collections.abc import Callable
import functools

from llm_workflow.base import Record, _has_history
from llm_workflow.internal_utilities import has_method, retry_handler
from llm_workflow.models import ExchangeRecord, LanguageModel
from llm_workflow.resources import MODEL_COST_PER_TOKEN


class ToolBase(ABC):
    """
    A tool is a callable object that has a name, description, and other properties that describe
    the tool. The name and description may be passed to an LLM (e.g. OpenAI "functions") and,
    therefore, should be a useful description for the LLM.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:  # noqa
        """A Tool object is callable, taking and returning any number of parameters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the tool. This value will be sent to an LLM."""

    @property
    @abstractmethod
    def description(self) -> str:
        """The description of the tool. This value will be sent to an LLM."""

    @property
    @abstractmethod
    def variables(self) -> dict:
        """
        For example.

            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                },
            },
            "required": ["location", "format"],
        }
        # {
        #     "type": "object",
        #     "properties": {
        #         "location": {
        #             "type": "string",
        #             "description": "The city and state, e.g. San Francisco, CA",
        #         },
        #         "format": {
        #             "type": "string",
        #             "enum": ["celsius", "fahrenheit"],
        #             "description": "The temperature unit to use. Infer this from the users location.",
        #         },
        #     },
        #     "required": ["location", "format"],
        # }
        """  # noqa: E501

    @property
    @abstractmethod
    def required(self) -> list:
        """
        For example.

        TODO.
        """

    def to_dict(self) -> str:
        """TODO. I don't love the dependency on OpenAI. Not sure how to deal with it."""
        return {
            'name': self.name,
            'description': self.description,
            'parameters': {
                # this is based on OpenAI requirement; i don't love this dependency
                "type": "object",
                "properties": self.variables,
                "required": self.required,
            },
        }


class Tool(ToolBase):
    """Class for wrapping an existing callable object."""

    def __init__(
            self,
            callable_obj: Callable,
            name: str,
            description: str,
            variables: dict,
            required: list[str] | None = None):
        self._callable_obj = callable_obj
        self._name = name
        self._description = description
        self._variables = variables
        self._required = required

    def __call__(self, *args, **kwargs) -> Any:  # noqa
        return self._callable_obj(*args, **kwargs)

    @property
    def name(self) -> str:
        """The name of the tool. This value will be sent to an LLM."""
        return self._name

    @property
    def description(self) -> str:
        """The description of the tool. This value will be sent to an LLM."""
        return self._description

    @property
    def variables(self) -> dict:
        """TODO."""
        return self._variables


    @property
    def required(self) -> list:
        """
        For example.

        TODO.
        """
        return self._required

    def history(self) -> list[Record]:
        """TODO. propegating history of underlying callable, if applicable."""
        if has_method(self._callable_obj, 'history'):
            return self._callable_obj.history()
        return None


def tool(name: str, description: str, variables: dict, required: list[str] | None = None) -> Tool:
    """
    TODO: Create a Tool from a function.
    We could theoretically build up parameters and required from docstrings and type-hints, but
    that seems unreliable and more confusing for the end user in terms of the function's
    requirements, and doesn't work for fuctions we don't define (e.g. we're importing).
    """
    def decorator(callable_obj: Callable):  # noqa: ANN202
        @functools.wraps(callable_obj)
        def wrapper(*args, **kwargs):  # noqa: ANN003, ANN002, ANN202
            return callable_obj(*args, **kwargs)
        return Tool(wrapper, name, description, variables, required)
    return decorator


class OpenAIFunctionAgent(LanguageModel):
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
                functions=[x.to_dict() for x in self._tools],
                temperature=0,
                # max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
        prompt_tokens = response['usage'].prompt_tokens
        completion_tokens = response['usage'].completion_tokens
        total_tokens = response['usage'].total_tokens
        cost = (prompt_tokens * self.cost_per_token['input']) + \
            (completion_tokens * self.cost_per_token['output'])
        record = ExchangeRecord(
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


    def _get_history(self) -> list[ExchangeRecord]:
        """TODO. A list of ExchangeRecord objects for tracking chat messages (prompt/response)."""
        histories = [tool.history() for tool in self._tools if _has_history(tool)]
        # Concatenate all the lists into a single list
        histories = [record for sublist in histories for record in sublist]
        histories += self._history
        unique_records = OrderedDict((record.uuid, record) for record in histories)
        unique_records = list(unique_records.values())
        return sorted(unique_records, key=lambda r: r.timestamp)



# class AgentTool(Tool):
#     """
#     An AgentTool is a specific type of Tool that is used by an Agent. Therefore, the inputs must
#     be a string (so that the agent can pass it )
#     """

# class DataChat()??? Data Agent?  has a dataset, and extracts context for

# class CustomTool(Tool):
#     """TODO."""

#     def __init__(self, name: str, description: str, func: Callable, properties: dict) -> None:
#         super().__init__()


# a Tool can have any inputs and any outputs

# "ChatTool" (AgentTool?) can only have string inputs and a single string output
# Why? Because a ChatTool is used by an Agent and so needs to be able to pass the tool inputs
# (strings) and must be able to use the output, which is a string

# class DuckDuckGoSearch(AgentTool):
#     """Does a web-search, and returns the the most relevant chunks as a string. TODO."""

#     def  __init__(self) -> None:
#         super().__init__()
