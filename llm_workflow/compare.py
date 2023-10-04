"""Utilities to compare the responses and performance of different models."""

from typing import Callable
import time

from llm_workflow.internal_utilities import execute_code_blocks, extract_code_blocks


class Scenario:
    """
    Represents a scenario where a model is provided one or more prompts. The model responds, and
    then subsequent prompts are provided. The model responds to each prompt in turn. The scenario
    ends when the model has responded to all prompts.

    NOTE: The model must track its memory and state internally so that it responses accordingly to
    each successive prompt.
    """

    def __init__(self, model: Callable[[str], str], description: str):
        """
        Initialize the scenario.

        Args:
            model:
                A callable model that accepts a single string argument and returns a string
                response.
                NOTE: The model must track its memory and state internally so that it responses
                accordingly to each successive prompt.
                The model may optionally have a cost attribute that represents the cost of running
                the model (e.g. cost per tokens).
            description:
                A description of the scenario (e.g. the name of the model).
        """
        self._model = model
        self.description = description
        self.responses = []
        self.prompts = None
        self.duration_seconds = None

    def __call__(self, prompts: list[str]) -> None:
        """
        Runs the trial.

        Args:
            prompts:
                A list of prompts to send to the chat model. The list represents consecutive
                turns/requests in a conversation (e.g. a follow up question or request).
        """
        if self.duration_seconds is not None:
            raise ValueError("Trial has already been run.")
        self.prompts = prompts
        # run the models
        start = time.time()
        for prompt in self.prompts:
            self.responses.append(self._model(prompt))
        end = time.time()
        self.duration_seconds = end - start
        # extract and execute the code blocks returned by the model
        local_namespace = {}
        self.code_blocks = [extract_code_blocks(response) for response in self.responses]
        self._code_block_results = [
            execute_code_blocks(code_blocks, local_namespace) for code_blocks in self.code_blocks
        ]

    def __str__(self) -> str:
        """Gives a summary of the the scenario's outcome."""
        results = f"{self.description}\n"
        results += f"Time: {self.duration_seconds:.2f} seconds\n"
        results += f"Response Characters: {self.num_response_chars:,}\n"
        results += f"Response Characters per second: {self.response_chars_per_second:.1f}\n"
        results += f"Num code blocks: {self.num_code_blocks}\n"
        results += f"Percent Passing Code blocks: {self.percent_successful_code_blocks:.1%}\n"
        if self.cost:
            results += f"Cost: ${self.cost:.5f}\n"
        return results

    @property
    def num_response_chars(self) -> int:
        """The total number of characters across all responses."""
        return sum(len(response) for response in self.responses)

    @property
    def response_chars_per_second(self) -> float:
        """The number of characters returned per second (across all responses)."""
        return self.num_response_chars / self.duration_seconds

    @property
    def num_code_blocks(self) -> int:
        """The total number of code blocks across all responses."""
        return sum(len(code_blocks) for code_blocks in self.code_blocks)

    @property
    def code_block_results(self) -> list[list[Exception]]:
        """
        Returns a list lists. Each outer list represents a response from the model. A response may
        have zero or more code blocks. Each inner list represents the results of executing each
        code block. If the code block executed successfully, None is returned. Otherwise,
        the exception is returned.
        """
        return self._code_block_results

    @property
    def num_successful_code_blocks(self) -> int:
        """The total number of code blocks that executed successfully."""
        return sum(
            sum(1 for result in results if result is None)
            for results in self.code_block_results
        )

    @property
    def percent_successful_code_blocks(self) -> float:
        """TODO."""
        return self.num_successful_code_blocks / self.num_code_blocks

    @property
    def cost(self) -> float:
        """TODO."""
        # if model has a cost attribute, use that
        if hasattr(self._model, 'cost'):
            return self._model.cost
        return None
