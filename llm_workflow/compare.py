"""Utilities to compare the responses and performance of different models."""

from typing import Callable, TypeVar
import time
from pydantic import BaseModel
from llm_workflow.internal_utilities import execute_code_blocks, extract_code_blocks

Model = TypeVar('Model', bound=Callable[[str], str])


class ModelCreation(BaseModel):
    """
    Used to define the requirements for comparing models.

    create:
        A callable function with no arguments that creates a model (e.g. probably just a lambda).
        The intent is that the model is created each time a trial is run so that the model is
        in a fresh state.

        The model needs to be a callable that accepts a single string argument and returns a string
        response.
        NOTE: The model must track its memory and state internally so that it responses
        accordingly to each successive prompt.
        The model may optionally have a cost attribute that represents the cost of running
        the model (e.g. cost per tokens).
    description:
        A description of the scenario (e.g. the name of the model).
    """

    create: Callable[[], Model]
    description: str


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
        """The percentage of code blocks that executed successfully."""
        return self.num_successful_code_blocks / self.num_code_blocks

    @property
    def cost(self) -> float:
        """
        The cost of running the model. If the model does not have a cost attribute, None is
        returned.
        """
        if hasattr(self._model, 'cost'):
            return self._model.cost
        return None


class CompareModels:
    """
    One requirements is that the underlying models need to maintain message history. They are
    passed a list of prompts. The second prompt is a follow up question to the first prompt so
    the model needs to be able to maintain the history of the conversation.
    """

    def __init__(
            self,
            prompts: list[list[str]],
            model_creations: list[ModelCreation],
        ):
        """
        Args:
            prompts:
                A nested list of prompts. Each outer list represents a single scenario.
                Each inner list is of prompts. The prompts represent consecutive turns/requests in
                a conversation (e.g. a follow up question or request).
            model_creations:
                A list of "model creations". Each model creation is a ModelCreation object that
                defines the requirements for comparing models.
                NOTE: The model must track its memory and state internally so that it responses
                accordingly to each successive prompt.
        """
        # ensure no model descriptions are duplicated
        model_descriptions = [model_creation.description for model_creation in model_creations]
        if len(model_descriptions) != len(set(model_descriptions)):
            raise ValueError("Model descriptions must be unique.")

        self.prompts = prompts
        self._model_creations = model_creations

    def __call__(self):
        """Runs the prompts/scenarios."""
        self.scenarios = []
        for prompts in self.prompts:
            runs = []
            for create_model in self._model_creations:
                scenario = Scenario(
                    model=create_model.create(),  # create a new model for each run
                    description=create_model.description,
                )
                scenario(prompts=prompts)
                runs.append(scenario)
            self.scenarios.append(runs)
        assert len(self.scenarios) == len(self.prompts)

    @property
    def num_scenarios(self) -> int:
        """
        The number of scenarios. self.scenarios is a list of lists (of prompts). The outer list
        represents the scenarios. The inner list represents the prompts for a single scenario.
        `num_scenarios` is the number of outer lists.
        """
        return len(self.scenarios)

    @property
    def num_models(self) -> int:
        """The number of models (used for each scenario)."""
        return len(self._model_creations)

    @property
    def model_descriptions(self) -> str:
        """A list of model descriptions."""
        return [model.description for model in self._model_creations]

    def _sum_property(self, model_description: str, property_name: str) -> float:
        """
        sums the property across all scenarios.

        Args:
            model_description:
                The description of the model. (The value passed to the ModelCreation object.)
            property_name:
                The name of the property to sum.
        """
        total = 0
        for use_case_trials in self.scenarios:
            for trial in use_case_trials:
                if trial.description == model_description:
                    value = getattr(trial, property_name)
                    if value:
                        total += value
        return total

    def duration_seconds(self, model_description: str) -> float:
        """
        The total duration (in seconds) across all scenarios for the model associated with
        model_description.
        """
        return self._sum_property(model_description, 'duration_seconds')

    def num_response_chars(self, model_description: str) -> int:
        """
        The total number of response characters across all scenarios for the model associated with
        model_description.
        """
        return self._sum_property(model_description, 'num_response_chars')

    def response_chars_per_second(self, model_description: str) -> float:
        """
        The number of response characters per second across all scenarios for the model associated
        with the model_description.
        """
        return self.num_response_chars(model_description) / self.duration_seconds(model_description)  # noqa

    def num_code_blocks(self, model_description: str) -> int:
        """
        The total number of code blocks across all scenarios for the model associated with
        model_description.
        """
        return self._sum_property(model_description, 'num_code_blocks')

    def num_successful_code_blocks(self, model_description: str) -> int:
        """
        The total number of code blocks that executed successfully across all scenarios for the
        model associated with model_description.
        """
        return self._sum_property(model_description, 'num_successful_code_blocks')

    def percent_successful_code_blocks(self, model_description: str) -> float:
        """
        The percentage of code blocks that executed successfully across all scenarios for the
        model associated with model_description.
        """
        return self.num_successful_code_blocks(model_description) / self.num_code_blocks(model_description)  # noqa

    def cost(self, model_description: str) -> float | None:
        """
        The total cost across all scenarios for the model associated with model_description. If the
        model does not have a cost attribute, None is returned.
        """
        return self._sum_property(model_description, 'cost')

    def __str__(self) -> str:
        """Returns a summary of the results."""
        results = ""
        for model_description in self.model_descriptions:
            results += f"{model_description}\n"
            results += f"Time: {self.duration_seconds(model_description):.2f} seconds\n"
            results += f"Response Characters: {self.num_response_chars(model_description):,}\n"
            results += f"Response Characters per second: {self.response_chars_per_second(model_description):.1f}\n"  # noqa
            results += f"Num code blocks: {self.num_code_blocks(model_description)}\n"
            results += f"Percent Passing Code blocks: {self.percent_successful_code_blocks(model_description):.1%}\n"  # noqa
            if self.cost(model_description):
                results += f"Cost: ${self.cost(model_description):.5f}\n"
            results += "\n"
        return results
