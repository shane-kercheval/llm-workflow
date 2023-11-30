"""
An Agent is an LLM that is given a set of tools and decides how to respond based on those tools.

Currently, the only agent in this library is the OpenAIFunctionAgent class, which wraps the logic
for OpenAI's "functions".
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
import json
from typing import Any
from collections.abc import Callable
import functools

from llm_workflow.base import Record, _has_history, ExchangeRecord, LanguageModel
from llm_workflow.internal_utilities import has_method, retry_handler
from llm_workflow.openai import MODEL_COST_PER_TOKEN


class ToolBase(ABC):
    """
    A tool is a callable object that has a name, description, and other properties that describe
    the tool. The name, description, etc., may be passed to an LLM (e.g. OpenAI "functions") and,
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
    def inputs(self) -> dict:
        """
        Property that describes the inputs of the tool.

        For example:

            {
                "variable_a": {
                    "type": "string",
                    "description": "This is a description of variable_a.",
                },
                "variable_b": {
                    "type": "string",
                    "enum": ["option_a", "option_b"],
                    "description": "This is a description of variable_b.",
                },
            }
        """

    @property
    @abstractmethod
    def required(self) -> list:
        """Returns a list of inputs that are required."""

    def to_dict(self) -> str:
        """
        Returns a dictionary with properties that describe the tool.
        Currently this dictinoary is in a formated expected by OpenAI "functions" API. The
        dependency to OpenAI is not ideal.
        """
        return {
            'name': self.name,
            'description': self.description,
            'parameters': {
                # this is based on OpenAI requirement; i don't love this dependency
                "type": "object",
                "properties": self.inputs,
                "required": self.required,
            },
        }


class Tool(ToolBase):
    """
    A tool is an object that has a name, description, and other properties that describe
    the tool. The name, description, etc., may be passed to an LLM (e.g. OpenAI "functions") and,
    therefore, should be a useful description for the LLM.

    A tool object is callable if a function is passed into callable_obj in the constructor.
    """

    def __init__(
            self,
            name: str,
            description: str,
            inputs: dict,
            required: list[str] | None = None,
            callable_obj: Callable | None = None):
        self._name = name
        self._description = description
        self._inputs = inputs
        self._required = required
        self._callable_obj = callable_obj

    def __call__(self, *args, **kwargs) -> Any:  # noqa
        return self._callable_obj(*args, **kwargs)

    @classmethod
    def from_dict(cls, dictionary):  # noqa
        """Returns a Tool object from a dictionary."""
        return cls(**dictionary)

    @property
    def name(self) -> str:
        """The name of the tool. This value will be sent to an LLM."""
        return self._name

    @property
    def description(self) -> str:
        """The description of the tool. This value will be sent to an LLM."""
        return self._description

    @property
    def inputs(self) -> dict:
        """
        Property that describes the inputs of the tool.

        For example:

            {
                "variable_a": {
                    "type": "string",
                    "description": "This is a description of variable_a.",
                },
                "variable_b": {
                    "type": "string",
                    "enum": ["option_a", "option_b"],
                    "description": "This is a description of variable_b.",
                },
            }
        """
        return self._inputs


    @property
    def required(self) -> list:
        """Returns a list of inputs that are required."""
        return self._required

    def history(self) -> list[Record]:
        """Returns the history of the underlying callable object, if applicable."""
        if has_method(self._callable_obj, 'history'):
            return self._callable_obj.history()
        return None


def tool(name: str, description: str, inputs: dict, required: list[str] | None = None) -> Tool:
    """
    A tool is a callable object that has a name, description, and other properties that describe
    the tool. The name, description, etc., may be passed to an LLM (e.g. OpenAI "functions") and,
    therefore, should be a useful description for the LLM.

    This decorator wraps a callable object.
    """
    def decorator(callable_obj: Callable):  # noqa: ANN202
        @functools.wraps(callable_obj)
        def wrapper(*args, **kwargs):  # noqa: ANN003, ANN002, ANN202
            return callable_obj(*args, **kwargs)
        return Tool(name, description, inputs, required, wrapper)
    return decorator


class OpenAIFunctions(LanguageModel):
    """
    Wrapper around OpenAI "functions" (https://platform.openai.com/docs/guides/gpt/function-calling).

    Calling the objec returns a list of tuples, where each tuple contains a Tool object and a
    dictionary of arguments (chosen by OpenAI) to pass to the tool.

    From OpenAI:

        "Developers can now describe functions to gpt-4-0613 and gpt-3.5-turbo-0613, and have the
        model intelligently choose to output a JSON object containing arguments to call those
        functions. This is a new way to more reliably connect GPT's capabilities with external
        tools and APIs.

    See this notebooks for an example: https://github.com/shane-kercheval/llm-workflow/blob/main/examples/agents.ipynb
    """

    def __init__(
            self,
            tools: list[Tool],
            model_name: str = 'gpt-3.5-turbo-1106',
            system_message: str = "Decide which function to use. Only use the functions you have been provided with. Don't make assumptions about what values to plug into functions.",  # noqa
            timeout: int = 10,
        ) -> dict | None:
        """
        Args:
            model_name:
                e.g. 'gpt-3.5-turbo-1106'
            tools:
                a list of Tool objects (created with the `Tool` class or `tool` decorator).
            system_message:
                The content of the message associated with the "system" `role`.
            timeout:
                timeout value passed to OpenAI model.
        """
        super().__init__()
        self.model_name = model_name
        self._tools = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"Tool name '{tool.name}' is already in use.")
            self._tools[tool.name] = tool
        self._system_message = system_message
        self._history = []
        self.timeout = timeout


    def __call__(self, prompt: object) -> list[tuple[Tool, dict]]:
        """
        Uses the OpenAI "functions" api to decide which tool to call based on the `prompt`. The
        selected tool (which is a callable) is called and passed the arguments determined by
        OpenAI. The response from the tool is retuned by the agent object.

        Returns a list of tuples, where each tuple contains a Tool object and a dictionary of
        arguments (chosen by OpenAI) to pass to the tool.
        """
        from openai import OpenAI
        messages = [
            {"role": "system", "content": self._system_message},
            {"role": "user", "content": prompt},
        ]
        # we want to track to track costs/etc.; but we don't need the history to build up memory
        # essentially, for now, this class won't have any memory/context of previous questions;
        # it's only used to decide which tools/functions to call
        client = OpenAI()
        tools = [{'type': 'function', 'function': x.to_dict()} for x in self._tools.values()]
        tools[0]
        response = retry_handler()(
                client.chat.completions.create,
                model=self.model_name,
                messages=messages,
                tools=tools,
                temperature=0,
                # max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
        input_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        cost = (input_tokens * self.cost_per_token['input']) + \
            (completion_tokens * self.cost_per_token['output'])
        record = ExchangeRecord(
            prompt=prompt,
            response='',
            metadata={'model_name': self.model_name},
            input_tokens=input_tokens,
            response_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
        )
        self._history.append(record)
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            tool_calls = [
                (self._tools[x.function.name], json.loads(x.function.arguments))
                for x in tool_calls
            ]
            tool_names = ' | '.join([x[0].name for x in tool_calls])
            record.response = f"tools: {tool_names}"
            record.metadata['tool_names'] = tool_names
            record.metadata['tool_args'] = ' | '.join([str(x[1]) for x in tool_calls])
            return tool_calls
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


    def _get_history(self) -> list[Record]:
        """Returns a list of Records corresponding to any OpenAI call."""
        return self._history


class OpenAIFunctionAgent(OpenAIFunctions):
    """
    Overrides OpenAIFunctions to return the response from the tool selected (rather than a list of
    tools/arguments).

    NOTE: This implementation only extracts and calls the first tool returned by OpenAI.
    This class uses the OpenAI "functions" api to decide which tool to use; the selected tool
    (which is a callable) is called and passed the arguments determined by OpenAI.
    The response from the tool is retuned by the agent object.
    """

    def __call__(self, prompt: object) -> str:
        """
        Uses the OpenAI "functions" api to decide which tool to call based on the `prompt`. The
        selected tool (which is a callable) is called and passed the arguments determined by
        OpenAI. The response from the tool is retuned by the agent object.
        """
        tool_calls = super().__call__(prompt)
        if tool_calls:
            tool, args = tool_calls[0]
            return tool(**args)
        return None

    def _get_history(self) -> list[Record]:
        """
        Returns a list of Records corresponding to any OpenAI call as well as any Record object
        associated with the underlying tools' history.

        NOTE: the entire history of each tool is included. If you pass the OpenAIFunctionAgent
        object a tool that was previously used (i.e. the tool "object" was instantiated and called
        and has resulting history), that history will be included, even though it is not directly
        related to the use of the Agent. As a best practice, you should only include tool objects
        that have not been previously instantiated/used.
        """
        histories = [tool.history() for tool in self._tools.values() if _has_history(tool)]
        # Concatenate all the lists into a single list
        histories = [record for sublist in histories for record in sublist]
        histories += self._history
        unique_records = OrderedDict((record.uuid, record) for record in histories)
        unique_records = list(unique_records.values())
        return sorted(unique_records, key=lambda r: r.timestamp)
