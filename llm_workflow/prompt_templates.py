"""
A "prompt-template" is a callable object that takes a prompt (e.g. user query) as input and returns
a modified prompt. Each prompt-template class is instantiated with the necessary information it
requires. For instance, if a template's purpose is to search for relevant documents, it is provided
with the vector database during object creation rather than through the `__call__` method.

If you're prompt-template is simple, just use a function (or inline lambda) in the task.
"""

from abc import ABC, abstractmethod
from typing import Callable
import pandas as pd
from functools import singledispatch

from llm_workflow.base import Record, RecordKeeper
from llm_workflow.indexes import DocumentIndex
from llm_workflow.internal_utilities import extract_variables
from llm_workflow.resources import (
    PROMPT_TEMPLATE__INCLUDE_DOCUMENTS,
    PROMPT_TEMPLATE__PYTHON_METADATA,
)


class PromptTemplate(ABC):
    """
    A PromptTemplate is a callable object that takes a prompt (e.g. user query) as input and
    returns a modified prompt. Each PromptTemplate is provided with the necessary information
    during instantiation. For instance, if a template's purpose is to search for relevant
    documents, it is given the vector database when the object is created, rather than via the
    `__call__` method.
    """

    @abstractmethod
    def __call__(self, prompt: str) -> str:
        """Takes the original prompt (user inuput) and returns a modified prompt."""


class DocSearchTemplate(RecordKeeper, PromptTemplate):
    """
    `DocSearchTemplate` is a prompt-template that, when called (`__call__`) with a prompt, searches
    for the most similar documents using the provided `DocumentIndex` object. It then includes the
    content of all of the retrieved documents in the modified prompt. The Document objects included
    in the prompt can be retrieved via the `similar_docs` property.
    """

    def __init__(
            self,
            doc_index: DocumentIndex,
            template: str | None = None,
            n_docs: int = 3) -> None:
        """
        Args:
            doc_index:
                the document index used to search for relevant documents
            template:
                custom template (string value that must contain "{{documents}}" and "{{prompt}}"
                within the string); if None, then a default template is provided
            n_docs:
                the number of documents (returned by the doc_index) to include in the prompt
        """  # noqa
        super().__init__()
        self._doc_index = doc_index
        self.n_docs = n_docs
        self.template = template if template else PROMPT_TEMPLATE__INCLUDE_DOCUMENTS
        self.similar_docs = None

    def __call__(self, prompt: str) -> str:  # noqa
        super().__call__(prompt)
        self.similar_docs = self._doc_index.search(
            value=prompt,
            n_results=self.n_docs,
        )
        doc_string = '\n\n'.join([x.content for x in self.similar_docs])
        return self.template.\
            replace('{{documents}}', doc_string).\
            replace('{{prompt}}', prompt)

    def _get_history(self) -> list[Record]:
        """Propagate the history from the underlying DocumentIndex object."""
        return self._doc_index.history()

    @property
    def total_tokens(self) -> int | None:
        """
        Sums the `total_tokens` values across all Record objects (which contain that property)
        returned by this object's `history` property (e.g. underlying records from DocumentIndex
        objects; e.g. Embeddings).
        """
        return self.sum(name='total_tokens')

    @property
    def cost(self) -> float | None:
        """
        Sums the `cost` values across all Record objects (which contain that property)
        returned by this object's `history` property (e.g. underlying records from DocumentIndex
        objects; e.g. Embeddings).
        """
        return self.sum(name='cost')


@singledispatch
def extract_metadata(obj: object, obj_name: str | None = None) -> str:
    """Extract metadata from an object in python."""
    return f"{obj_name}: {obj!r}" if obj_name else str(obj)


@extract_metadata.register
def _(obj: dict, obj_name: str | None = None) -> str:
    """Extract metadata from a dictionary."""
    obj_name = f' `{obj_name}` ' if obj_name else ' '
    return f"A python dictionary{obj_name}with keys: {list(obj.keys())}"


@extract_metadata.register
def _(obj: list, obj_name: str | None = None) -> str:
    """Extract metadata from a list or tuple."""
    obj_name = f' `{obj_name}` ' if obj_name else ' '
    types = list({type(x) for x in obj})
    return f"A python list{obj_name}with length: {len(obj)} and types: {types}"


@extract_metadata.register
def _(obj: pd.DataFrame, obj_name: str | None = None) -> str:
    # extract data types
    obj_name = f' `{obj_name}` ' if obj_name else ' '
    metadata = f"A pd.DataFrame{obj_name}that contains the following numeric and non-numeric columns:\n\n"  # noqa
    # for column in obj.columns:
    #     metadata += f"`{column}`: {obj[column].apply(type).unique()}\n"

    # extract summary stats for numeric columns
    describe = obj.describe()
    metadata += "\nHere are the numeric columns and corresponding summary statistics:\n\n"
    metadata += str(describe.transpose())

    # extract summary stats for non-numeric columns
    metadata += "\n\nHere are the non-numeric columns and corresponding value counts:\n\n"
    for column in [x for x in obj.columns if x not in describe.columns]:
        unique_values = obj[column]\
            .value_counts(sort=True, ascending=False, normalize=False, dropna=False)
        top_10 = unique_values.head(10).to_dict()
        if len(unique_values) == len(top_10):
            metadata += f"`{column}`: {top_10}\n"
        else:
            metadata += f"`{column}` (top {len(top_10)} out of {len(unique_values)} unique values): {top_10}\n"  # noqa

    metadata += "\n\nUse both the numeric and non-numeric columns as appropriate."
    return metadata


class PythonObjectMetadataTemplate(PromptTemplate):
    """
    `PythonObjectMetadataTemplate` is a prompt-template that takes a list of Python objects and
    constructs a prompt from the "metadata" of those objects. For instance, if the list contains
    a Pandas DataFrame object, then the prompt will contain the column names and other information
    of that DataFrame.

    A PythonObjectMetadataTemplate object is instantiated with a list of Python objects as well as
    an optional list of functions that are used to extract the metadata from the objects. If no
    functions are provided, then the default functions are used.
    """

    def __init__(
            self,
            objects: dict[str, object | tuple[object, Callable]] | None = None,
            template: str | None = None,
            extract_variables: bool = True,
            raise_not_found_error: bool = True) -> None:
        """
        Initialize the object.

        NOTE: If metadatas contains multiple objects with the same object_name, then an error is
        raised.

        NOTE: If metadata is not used, then the prompt is not modified.

        Args:
            objects:
                A dictionary of objects that are used to construct the prompt. The keys of the
                dictionary are the names of the objects. The values of the dictionary are either
                the objects themselves or a tuple of the object and a function that is used to
                extract the metadata from the object. If the function is not provided, then the
                default function (extract_metadata) is used.
            template:
                The template that is used to construct the prompt. The template must contain
                "{{metadata}}" and "{{prompt}}" within the string.
            extract_variables:
                If True, extract variables in the prompt that start with @ (e.g. `@my_variable)
                and use the variables' metadata in the prompt. If False, then use the metadata of
                all of the objects in the prompt.
            raise_not_found_error:
                If True, raise an error if the @'d objects in the prompt don't have corresponding
                matches in the metadatas.
        """
        super().__init__()
        self.objects = objects if objects else {}
        self.template = template if template else PROMPT_TEMPLATE__PYTHON_METADATA
        self._extract_variables = extract_variables
        self._raise_not_found_error = raise_not_found_error
        self._extracted_variables_last_call = None

    def __call__(self, prompt: str) -> str:
        """Return a prompt that contains the metadata of the objects."""
        super().__call__(prompt)
        metadata = []
        extracted_variables = []
        if self._extract_variables:
            extracted_variables = extract_variables(prompt)
            self._extracted_variables_last_call = extracted_variables
            # if no variables are found, then no metadata is used
            for variable in extracted_variables:
                prompt = prompt.replace(f'@{variable}', f'`{variable}`')
                if variable in self.objects:
                    if isinstance(self.objects[variable], tuple):
                        obj, func = self.objects[variable]
                        metadata.append(func(obj, variable))
                    else:
                        metadata.append(extract_metadata(self.objects[variable], variable))
                elif self._raise_not_found_error:
                    assert variable in self.objects, \
                        f"Variable {variable} not found in list of objects."
        elif self.objects:
            # use metadata from all objects
            for object_name, obj in self.objects.items():
                if isinstance(obj, tuple):
                    obj, func = obj  # noqa
                    metadata.append(func(obj, object_name))
                else:
                    metadata.append(extract_metadata(obj, object_name))

        if metadata:
            return self.template.\
                replace('{{metadata}}', '\n\n---\n\n'.join(metadata)).\
                replace('{{prompt}}', prompt)

        return prompt


    def add_object(self, object_name: str, obj: object | tuple[object, Callable]) -> None:
        """
        Add an object to the list of objects used to construct the prompt.

        Args:
            object_name:
                The name of the object.
            obj:
                The object itself or a tuple of the object and a function that is used to extract
                the metadata from the object. If the function is not provided, then the default
                function (extract_metadata) is used.
        """
        assert object_name not in self.objects, \
            f"Object with name {object_name} already exists."
        self.objects[object_name] = obj
