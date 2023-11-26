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

from pydantic import BaseModel
from llm_workflow.base import Record, RecordKeeper
from llm_workflow.indexes import DocumentIndex
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
    metadata = f"A pd.DataFrame{obj_name}that contains the following columns with the following types of values:\n\n"  # noqa
    for column in obj.columns:
        metadata += f"`{column}`: {obj[column].apply(type).unique()}\n"

    # extract summary stats for numeric columns
    describe = obj.describe()
    metadata += "\nThe following numeric columns contain the following summary statistics:\n\n"
    metadata += str(describe.transpose())

    # extract summary stats for non-numeric columns
    metadata += "\n\nThe following non-numeric columns contain the following unique values and corresponding value counts:\n\n"  # noqa
    for column in [x for x in obj.columns if x not in describe.columns]:
        unique_values = obj[column]\
            .value_counts(sort=True, ascending=False, normalize=False, dropna=False)
        top_10 = unique_values.head(10).to_dict()
        if len(unique_values) == len(top_10):
            metadata += f"`{column}`: {top_10}\n"
        else:
            metadata += f"`{column}` (top {len(top_10)} out of {len(unique_values)} unique values): {top_10}\n"  # noqa
    return metadata


class MetadataMetadata(BaseModel):
    """Metadata about the metadata; used to extract/create the metadata."""

    obj: object
    object_name: str | None = None
    extract_func: Callable | None = None


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
            metadatas: list[MetadataMetadata],
            template: str | None = None) -> None:
        """
        Initialize the object.

        Args:
            metadatas:
                Information (metadata) about how to extract metadata from the objects that are
                provided.
            template:
                The template that is used to construct the prompt. The template must contain
                "{{metadata}}" and "{{prompt}}" within the string.
        """
        super().__init__()
        self.metadatas = metadatas
        self.template = template if template else PROMPT_TEMPLATE__PYTHON_METADATA

    def __call__(self, prompt: str) -> str:
        """Return a prompt that contains the metadata of the objects."""
        super().__call__(prompt)
        results = []
        for metadata in self.metadatas:
            func = metadata.extract_func if metadata.extract_func else extract_metadata
            results.append(func(metadata.obj, metadata.object_name))
        results = '\n\n---\n\n'.join(results)
        return self.template.\
            replace('{{metadata}}', results if results else 'No Metadata').\
            replace('{{prompt}}', prompt)


# class PythonObjectSelectorTemplate(PromptTemplate):
#     """Injects metadata based on matches to @object_name used in prompt."""

#     def __init__(
#             self,
#             metadatas: list[MetadataMetadata],
#             template: str | None = None,
#             error_if_not_found: bool = True) -> None:
#         """
#         Initialize the object.

#         Args:
#             metadatas:
#                 Information (metadata) about how to extract metadata from the objects that are
#                 provided.
#             template:
#                 The template that is used to construct the prompt. The template must contain
#                 "{{metadata}}" and "{{prompt}}" within the string.
#             error_if_not_found:
#                 If True, raise an error if the @'d objects in the prompt don't have corresponding
#                 matches in the metadatas.
#         """
#         super().__init__()
#         self.metadatas = metadatas
#         self.template = template if template else PROMPT_TEMPLATE__PYTHON_METADATA
#         self.erro4r_if_not_found = error_if_not_found
