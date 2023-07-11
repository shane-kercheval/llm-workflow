"""Contains all base and foundational classes."""
from abc import ABC, abstractmethod
from typing import Any
from collections.abc import Callable
from uuid import uuid4
from datetime import datetime
from pydantic import BaseModel, Field

from llm_workflow.internal_utilities import has_method, has_property


class Record(BaseModel):
    """Used to track the history of a task or task."""

    uuid: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
    )
    metadata: dict = Field(default_factory=dict)

    def __str__(self) -> str:
        return \
            f"timestamp: {self.timestamp}; " \
            f"uuid: {self.uuid}"


class CostRecord(Record):
    """TODO: different type makes it convenient to filter for specific types of records."""

    cost: float | None = None


class SearchRecord(Record):
    """TODO: different type makes it convenient to filter for specific types of records."""

    query: str
    results: list | None = None


class Document(BaseModel):
    """
    A Document comprises both content (text) and metadata, allowing it to represent a wide range of
    entities such as files, web pages, or even specific sections within a larger document.
    """

    content: str
    metadata: dict | None


class RecordKeeper(ABC):
    """A RecordKeeper is an object that tracks history i.e. `Record` objects."""

    @abstractmethod
    def _get_history(self) -> list[Record]:
        """
        Each inheriting class needs to implement the ability to track history.
        When an object doesn't have any history, it should return an empty list rather than `None`.
        """

    def history(self, types: type | tuple[type] | None = None) -> list[Record]:
        """
        Returns the history (list of Records objects) associated with an object, based on the
        `types` provided.

        Args:
            types:
                if None, all Record objects are returned
                if not None, either a single type or tuple of types indicating the type of Record
                objects to return (e.g. `TokenUsageRecord` or `(ExchangeRecord, EmbeddingRecord)`).
        """
        if not types:
            return self._get_history()
        if isinstance(types, type | tuple):
            return [x for x in self._get_history() if isinstance(x, types)]

        raise TypeError(f"types not a valid type ({type(types)}) ")

    def previous_record(self, types: type | tuple[type] | None = None) -> Record | None:
        """TODO."""
        history = self.history(types=types)
        if history:
            return history[-1]
        return None

    def sum(  # noqa: A003
            self,
            name: str,
            types: type | tuple[type] | None = None) -> int | float:
        """
        For a given property `name` (e.g. `cost` or `total_tokens`), this function sums the values
        across all Record objects in the history, for any Record object that contains the property.

        Args:
            name:
                the name of the property on the Record object to aggregate
            types:
                if provided, only the Record objects in `history` with the corresponding type(s)
                are included in the aggregation
        """
        records = [
            x for x in self.history(types=types)
            if has_property(obj=x, property_name=name)
        ]
        if records:
            return sum(getattr(x, name) or 0 for x in records)
        return 0


class Value:
    """
    The Value class provides a convenient caching mechanism within the workflow.
    The `Value` object is callable, allowing it to cache and return the value when provided as an
    argument. When called without a value, it retrieves and returns the cached value.
    """

    def __init__(self):
        self.value = None

    def __call__(self, value: object | None = None) -> object:
        """
        When a `value` is provided, it gets cached and returned.
        If no `value` is provided, the previously cached value is returned (or None if no value has
        been cached).
        """
        if value:
            self.value = value
        return self.value


class Workflow(RecordKeeper):
    """
    A workflow object is a collection of `tasks`. Each task in the workflow is a callable, which
    can be either a function or an object that implements the `__call__` method.

    The output of one task serves as the input to the next task in the workflow.

    Additionally, each task can track its own history, including messages sent/received and token
    usage/costs, through a `history` property that returns a list of `Record` objects. A workflow
    aggregates and propagates the history of any task that has a `history` property, making it
    convenient to analyze costs or explore intermediate steps in the workflow.
    """

    def __init__(self, tasks: list[Callable[[Any], Any]]):
        self._tasks = tasks

    def __getitem__(self, index: int) -> Callable:
        return self._tasks[index]

    def __len__(self) -> int:
        return len(self._tasks)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """
        Executes the workflow by passing the provided value to the first task. The output of each
        task is passed as the input to the next task, creating a sequential execution of all tasks
        in the workflow.

        The execution continues until all tasks in the workflow have been executed. The final
        output from the last task is then returned.
        """
        if not self._tasks:
            return None
        result = self._tasks[0](*args, **kwargs)
        if len(self._tasks) > 1:
            for task in self._tasks[1:]:
                result = task(result)
        return result

    def _get_history(self) -> list[Record]:
        """
        Aggregates the `history` across all tasks in the workflow. This method ensures that if a
        task is added multiple times to the workflow (e.g. a chat model with multiple steps), the
        underlying Record objects associated with that task's `history` are not duplicated.
        """
        histories = [task.history() for task in self._tasks if _has_history(task)]
        # Edge-case: if the same model is used multiple times in the same workflow (e.g. embedding
        # model to embed documents and then embed query to search documents) then we can't loop
        # through the workflows because we'd be double-counting the history from those objects.
        # we have to build up a history and include the objects if they aren't already
        # to do this we'll use the uuid, and then sort by timestamp
        unique_records = []
        unique_uuids = set()
        for history in histories:
            for record in history:
                if record.uuid not in unique_uuids:
                    unique_records.append(record)
                    unique_uuids |= {record.uuid}
        return sorted(unique_records, key=lambda x: x.timestamp)


class Session(RecordKeeper):
    """
    A Session is used to aggregate multiple workflow objects. It provides a way to track and manage
    multiple workflows within the same session. When calling a Session, it will execute the last
    workflow that was added to the session.
    """

    def __init__(self, workflows: list[Workflow] | None = None):
        self._workflows = workflows or []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """
        Calls/starts the workflow that was the last added to the session, passing in the
        corresponding arguments.
        """
        if self._workflows:
            return self._workflows[-1](*args, **kwargs)
        raise ValueError()

    def append(self, workflow: Workflow) -> None:
        """
        Add or append a new workflow object to the list of workflows in the session. If the session
        object is called (i.e. __call__), the session will forward the call to the new workflow
        object (i.e. the last workflow added in the list).
        """
        self._workflows.append(workflow)

    def __len__(self) -> int:
        return len(self._workflows)

    def _get_history(self) -> list[Record]:
        """
        Aggregates the `history` across all workflow objects in the Session. This method ensures
        that if a task is added multiple times to the Session, the underlying Record objects
        associated with that task's `history` are not duplicated.
        """
        """
        Aggregates the `history` across all workflows in the session. It ensures that if the same
        object (e.g. chat model) is added multiple times to the Session, that the underlying Record
        objects associated with that object's `history` are not duplicated.
        """
        # for each history in workflow, cycle through each task's history and add to the list of
        # records if it hasn't already been added.
        workflows = [workflow for workflow in self._workflows if workflow.history()]
        # Edge-case: if the same model is used multiple times in the same workflow or across
        # different tasks (e.g. embedding
        # model to embed documents and then embed query to search documents) then we can't loop
        # through the workflows because we'd be double-counting the history from those objects.
        # we have to build up a history and include the objects if they aren't already
        # to do this we'll use the uuid, and then sort by timestamp
        unique_records = []
        unique_uuids = set()
        for workflow in workflows:
            for record in workflow.history():
                if record.uuid not in unique_uuids:
                    unique_records.append(record)
                    unique_uuids |= {record.uuid}
        return sorted(unique_records, key=lambda x: x.timestamp)


def _has_history(obj: object) -> bool:
    """
    For a given object `obj`, return True if that object has a `history` method and if the
    history has any Record objects.
    """
    return has_method(obj, 'history') and \
        isinstance(obj.history(), list) and \
        len(obj.history()) > 0 and \
        isinstance(obj.history()[0], Record)
