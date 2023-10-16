"""
An index is a data structure or mechanism used to facilitate efficient retrieval of specific
information from a larger collection or database. One example of an index is a document index,
which stores/retrieves documents (i.e. text with metadata). One implementation of a document index
is ChromaDB which stores/retrieves documents based on embeddings, which allow for semantic search.
"""

from uuid import uuid4
from abc import ABC, abstractmethod
from typing import TypeVar
from llm_workflow.base import Document, RecordKeeper, EmbeddingModel, EmbeddingRecord
from llm_workflow.internal_utilities import create_hash


ChromaCollection = TypeVar('ChromaCollection')


class DocumentIndex(ABC):
    """
    A `DocumentIndex` is a mechanism for adding and searching for `Document` objects. It can be
    thought of as a wrapper around chromadb or any other similar index or vector database.

    A `DocumentIndex` object should propagate any `total_tokens` or `total_cost` used by the
    underlying models, such as an `EmbeddingModel`. If these metrics are not applicable, the
    `DocumentIndex` should return `None`.

    A `DocumentIndex` is callable and adds documents to the index when called with a list of
    Document objects or searches for documents when called with a single string or Document object.
    """

    def __init__(self, n_results: int = 3) -> None:
        """
        Args:
            n_results: the number of search-results (from the document index) to return.
        """
        super().__init__()
        self._n_results = n_results

    def __call__(
            self,
            value: Document | str | list[Document],
            n_results: int | None = None) -> list[Document] | None:
        """
        When the object is called, it can either invoke the `add` method (if the `value` passed in
        is a list) or the `search` method (if the `value` passed in is a string or Document). This
        flexible functionality allows the object to be seamlessly integrated into a workflow,
        enabling the addition of documents to the index or searching for documents, based on input.

        Args:
            value:
                The value used to determine and retrieve similar Documents.
                Please refer to the description above for more details.
            n_results:
                The maximum number of results to be returned. If provided, it overrides the
                `n_results` parameter specified during initialization (`__init__`).

        Returns:
            If `value` is a list (i.e. the `add` function is called), this method returns None.
            If `value` is a string or Document (i.e the `search` function is called), this method
            returns the search results.
        """
        if isinstance(value, list):
            return self.add(docs=value)
        if isinstance(value, Document | str):
            return self.search(value=value, n_results=n_results)
        raise TypeError("Invalid Type")

    @abstractmethod
    def add(self, docs: list[Document]) -> None:
        """Add documents to the underlying index/database."""

    @abstractmethod
    def _search(self, doc: Document, n_results: int) -> list[Document]:
        """Search for documents in the underlying index/database based on `doc."""

    def search(
            self,
            value: Document | str,
            n_results: int | None = None) -> list[Document]:
        """
        Search for documents in the underlying index/database.

        Args:
            value:
                The value used to determine and retrieve similar Documents.
            n_results:
                The maximum number of results to be returned. If provided, it overrides the
                `n_results` parameter specified during initialization (`__init__`).
        """
        if isinstance(value, str):
            value = Document(content=value)
        return self._search(doc=value, n_results=n_results or self._n_results)


class ChromaDocumentIndex(RecordKeeper, DocumentIndex):
    """
    Chroma is a document index (vector database) that which provides a way to store embeddings
    associated with Document objects and then retrieve the documents that are most similar to
    another set of embeddings (or corresponding string).
    embeddings.

    The `ChromaDocumentIndex` class is a wrapper around Chroma that makes it easy to work with in
    a workflow. When the `ChromaDocumentIndex` object is called (via __call__), either the `add`
    method will be called (if the `value` passed in is a list) or the `search` method will be
    called (if the `value` passed in is a string or Document).
    This functionality allows to object to be added to a workflow and either add documents to the
    index or search for document in the index based on input.
    """

    def __init__(
            self,
            embeddings_model: EmbeddingModel | None = None,
            collection: ChromaCollection | None = None,
            n_results: int = 3) -> None:
        import chromadb
        super().__init__(n_results=n_results)
        self._collection = collection or chromadb.Client().create_collection(str(uuid4()))
        self._emb_model = embeddings_model

    def add(self, docs: list[Document]) -> None:
        """Add documents to the underlying Chroma index/database."""
        if not docs:
            return
        existing_ids = set(self._collection.get(include=['documents'])['ids'])
        ids = []
        metadatas = []
        contents = []
        documents = []
        for doc in docs:
            doc_id = create_hash(doc.content)
            if doc_id not in existing_ids:
                ids.append(doc_id)
                # chromadb seems to have made is so we cannot pass empty dict
                metadatas.append(doc.metadata or {'': ''})
                contents.append(doc.content)
                documents.append(doc)

        embeddings = self._emb_model(docs=documents) if self._emb_model else None
        if documents:
            self._collection.add(
                embeddings=embeddings,
                metadatas=metadatas,
                documents=contents,
                ids=ids,
            )

    def _search(self, doc: Document, n_results: int) -> list[Document]:
        """Search for documents in the underlying Chroma index/database based on `doc`."""
        if self._emb_model:
            embeddings = self._emb_model(docs=doc)
            results = self._collection.query(
                query_embeddings=embeddings,
                n_results=n_results,
            )
        else:
            results = self._collection.query(
                query_texts=doc.content,
                n_results=n_results,
            )
        # index 0 because we are only searching against a single document
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        similar_docs = []
        for doc, meta, dist in zip(documents, metadatas, distances, strict=True):
            meta['distance'] = dist
            similar_docs.append(Document(
                content=doc,
                metadata=meta,
            ))
        return similar_docs

    def _get_history(self) -> list[EmbeddingRecord]:
        """Propagates the history of any underlying models (e.g. embeddings model)."""
        return self._emb_model.history() if self._emb_model else []

    @property
    def total_tokens(self) -> int | None:
        """
        Sums the `total_tokens` values across all EmbeddingRecord objects returned by this
        object's `history` property.
        """
        return self.sum(name='total_tokens')

    @property
    def cost(self) -> float | None:
        """
        Sums the `cost` values across all EmbeddingRecord objects returned by this object's
        `history` property.
        """
        return self.sum(name='cost')
