"""Utilities to compare the responses and performance of different models."""

from typing import TypeVar
from collections.abc import Callable
import time
import textwrap
from pydantic import BaseModel
import markdown
from llm_workflow.internal_utilities import execute_code_blocks, extract_code_blocks
from pygments.formatters import HtmlFormatter


Model = TypeVar('Model', bound=Callable[[str], str])


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
        results += f"Response Characters Per Pecond: {self.response_chars_per_second:.1f}\n"
        results += f"Num Code Blocks: {self.num_code_blocks}\n"
        results += f"Percent Passing Code Blocks: {self.percent_successful_code_blocks:.1%}\n"
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


class ModelDefinition(BaseModel):
    """
    Used to define the model creation function and description for a single model when comparing
    models.

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


class CompareModels:
    """
    One requirements is that the underlying models need to maintain message history. They are
    passed a list of prompts. The second prompt is a follow up question to the first prompt so
    the model needs to be able to maintain the history of the conversation.
    """

    def __init__(
            self,
            prompts: list[list[str]],
            model_definitions: list[ModelDefinition],
        ):
        """
        Args:
            prompts:
                A nested list of prompts. Each outer list represents a single scenario.
                Each inner list is of prompts. The prompts represent consecutive turns/requests in
                a conversation (e.g. a follow up question or request).
            model_definitions:
                A list of "model definitions". Each item is a ModelDefinition object that
                defines the requirements for comparing models.
                NOTE: The model must track its memory and state internally so that it responses
                accordingly to each successive prompt.
        """
        # ensure no model descriptions are duplicated
        model_descriptions = [model_creation.description for model_creation in model_definitions]
        if len(model_descriptions) != len(set(model_descriptions)):
            raise ValueError("Model descriptions must be unique.")

        self.prompts = prompts
        self._model_definitions = model_definitions

    def __call__(self):
        """Runs the prompts/scenarios."""
        self.scenarios = []
        for prompts in self.prompts:
            runs = []
            for create_model in self._model_definitions:
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
        return len(self._model_definitions)

    @property
    def model_descriptions(self) -> str:
        """A list of model descriptions."""
        return [model.description for model in self._model_definitions]

    def _sum_property(self, model_description: str, property_name: str) -> float:
        """
        sums the property across all scenarios.

        Args:
            model_description:
                The description of the model. (The value passed to the ModelDefinition object.)
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
            results += f"Response Characters Per Pecond: {self.response_chars_per_second(model_description):.1f}\n"  # noqa
            results += f"Num Code Blocks: {self.num_code_blocks(model_description)}\n"
            results += f"Percent Passing Code Blocks: {self.percent_successful_code_blocks(model_description):.1%}\n"  # noqa
            if self.cost(model_description):
                results += f"Cost: ${self.cost(model_description):.5f}\n"
            results += "\n"
        return results

    def to_html(self, file_path: str | None = None) -> str | None:
        """
        Returns an HTML string that summarizes the each scenario across all models defined in the
        comparison object.

        Args:
            file_path:
                If provided, the HTML string is written to the file path. Otherwise, the HTML
                string is returned.
        """
        html = _comparison_to_html(self)
        if file_path:
            with open(file_path, 'w') as f:
                f.write(html)
            return None
        return html


def _scenario_summary_to_html(scenario: Scenario) -> str:
    results = '<table style="border-collapse: collapse; width: auto;">\n'
    results += f'<tr><td style="border: none;">Time</td><td style="border: none;"><code>{scenario.duration_seconds:.2f} seconds</code></td></tr>\n'  # noqa
    results += f'<tr><td style="border: none;">Characters</td><td style="border: none;"><code>{scenario.num_response_chars:,}</code></td></tr>\n'  # noqa
    results += f'<tr><td style="border: none;">Characters per second</td><td style="border: none;"><code>{scenario.response_chars_per_second:.1f}</code></td></tr>\n'  # noqa
    results += f'<tr><td style="border: none;">Num code blocks</td><td style="border: none;"><code>{scenario.num_code_blocks}</code></td></tr>\n'  # noqa
    percent_successful_code_blocks = (
        scenario.num_successful_code_blocks / scenario.num_code_blocks
    )
    results += f'<tr><td style="border: none;">Percent Passing Code blocks</td><td style="border: none;"><code>{percent_successful_code_blocks:.1%}</code></td></tr>\n'  # noqa
    if scenario.cost:
        results += f'<tr><td style="border: none;">Cost</td><td style="border: none;"><code>${scenario.cost:.5f}</code></td></tr>\n'  # noqa
    results += '</table>\n'
    return results


def _comparison_summary_html(comparison: CompareModels) -> str:
    headers = [
        "Time",
        "Response Characters",
        "Response Characters per second",
        "Num code blocks",
        "Percent Passing Code blocks",
        "Cost",
    ]

    # Metrics for which an increase is considered good
    increase_is_good = [False, False, True, False, True, False]
    color_changes = [True, False, True, False, True, True]  # Whether to color the percent changes
    # Opening the HTML string and adding headers with CSS for minimal width and bold text for
    # headers
    html = (
        '<table border="1" style="width: fit-content; table-layout: fixed;">\n'
        '<tr><th style="width: 150px;"></th>'
    )

    for model_description in comparison.model_descriptions:
        html += f'<th style="width: 200px; font-weight: bold;">{model_description}</th>'
    html += '</tr>\n'

    # Storing the first model's metrics for percent change calculations
    first_model_metrics = []

    for idx, header in enumerate(headers):
        html += f'<tr><td style="font-weight: bold;">{header}</td>'

        for m_idx, model_description in enumerate(comparison.model_descriptions):
            if idx == 0:
                value = comparison.duration_seconds(model_description)
                display_value = f"{value:.2f} seconds"
            elif idx == 1:
                value = comparison.num_response_chars(model_description)
                display_value = f"{value:,}"
            elif idx == 2:
                value = comparison.response_chars_per_second(model_description)
                display_value = f"{value:.1f}"
            elif idx == 3:
                value = comparison.num_code_blocks(model_description)
                display_value = f"{value}"
            elif idx == 4:
                value = comparison.percent_successful_code_blocks(model_description)
                display_value = f"{value:.1%}"
            elif idx == 5:
                value = comparison.cost(model_description)
                display_value = f"${value:.5f}" if value is not None else ""
            else:
                value = None
                display_value = ""

            # Store the first model's metrics
            if m_idx == 0:
                first_model_metrics.append(value)

            # Calculate and display percentage change for other models
            if (
                m_idx > 0
                and value is not None
                and first_model_metrics[idx] is not None
                and first_model_metrics[idx] != 0
            ):
                percent_change = ((value - first_model_metrics[idx]) / first_model_metrics[idx]) * 100  # noqa

                # Determine color based on whether the change is good or bad
                color = "green" if (percent_change > 0) == increase_is_good[idx] else "red"
                display_value += (
                    f' (<span style="color: {color if color_changes[idx] else "black"};">'
                    f'{percent_change:.1f}% {"increase" if percent_change > 0 else "decrease"}'
                    '</span>)'
                )
            html += f'<td>{display_value}</td>'
        html += '</tr>\n'

    # Closing the HTML string
    html += '</table>'
    return html


def _comparison_to_html(comparison: CompareModels) -> str:
    """
    Returns an HTML string that summarizes the each scenario across all models defined in the
    comparison object.
    """
    scenarios = comparison.scenarios
    # Configure Markdown to HTML conversion
    md = markdown.Markdown(
        extensions=['fenced_code', 'codehilite'],
        extension_configs={
            'codehilite': {
                'css_class': 'highlight',
                'linenums': False,
                'use_pygments': True,
            },
        },
    )
    css = HtmlFormatter().get_style_defs('.highlight')

    horizontal_line = '<div class="centered-line"></div>'

    # Generate rows and columns
    column_names_html = ''
    for scenario in scenarios[0]:
        column_names_html += f'<th>{scenario.description}</th>'

    summary_html = _comparison_summary_html(comparison)
    scenario_tables = ''
    for model_scenarios in scenarios:
        # create row for summaries
        rows_html = '<tr>'
        for scenario in model_scenarios:
            rows_html += f'<td style="vertical-align: top;">{_scenario_summary_to_html(scenario)}</td>'  # noqa
        rows_html += '</tr>'

        num_prompts = len(comparison.prompts)
        for i in range(num_prompts):
            # create a row for the prompt
            rows_html += '<tr>'
            rows_html += f'<td colspan="{len(model_scenarios)}" style="vertical-align: top; text-align: center;" >'  # noqa
            rows_html += '<h3>Prompt</h3><br>'
            rows_html += f'{scenario.prompts[0]}<br><br>'
            rows_html += '</td>'
            rows_html += '</tr>'
            # create a row for the response
            rows_html += '<tr>'
            for scenario in model_scenarios:
                rows_html += '<td style="vertical-align: top;">'
                rows_html += '<h3>Response</h3><br>'
                html = md.convert(scenario.responses[i])
                rows_html += f'{html}</td>'
            rows_html += '</tr>'

        scenario_tables += textwrap.dedent(f"""
            <table border="1" style="width:100%; border-collapse: collapse;">
                <thead>
                    <tr>
                        {column_names_html}
                    </tr>
                </thead>
                {rows_html}
            </table>
            <br>
            {horizontal_line}
            <br>
            """)

    # Wrap the HTML and CSS in a complete HTML document with a table
    return textwrap.dedent(f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Document</title>
        <style>
        {css}
        .centered-line {{
            width: 50%;
            margin: auto;
            border-top: 1px solid #000;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;  # ensures equal column widths
        }}
        th, td {{
            border: 1px solid #B2BEB5;
            padding: 8px;
            text-align: left;
            word-wrap: break-word;  # ensures content doesn't overflow cell
        }}
        </style>
    </head>
    <body>
    <h1>Summary</h1>
    {summary_html}
    <br><br>
    <h1>Use Cases</h1>
    {scenario_tables}
    </body>
    </html>
    ''')
