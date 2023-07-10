"""
Contains helper functions and classes that can be used within a Chain.



A Tool is meant to be used by an LLM, so it returns a string (which can be fed back into




Any callable can be passed to a chain, so a "tool" in this file can be a simple function, or it can
be a `Tool` class. A `Tool` class has a `name` and `description` that is potentially sent to an LLM
(e.g. OpenAI "functions") to describe when the tool should be used.

Some classes can be both Links and Tools. A `Link` is simply a callable that tracks history. So if
it's useful for a tool to track history, then it can be both.
"""

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
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
    def parameters(self) -> dict:
        """
        For example.

        {
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
        """  # noqa: E501

    @property
    def properties(self) -> str:
        """TODO."""
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
        }


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
