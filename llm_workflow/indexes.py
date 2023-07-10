"""
An index is a data structure or mechanism used to facilitate efficient retrieval of specific
information from a larger collection or database. One example of an index is a document index,
which stores/retrieves documents (i.e. text with metadata). One implementation of a document index
is ChromaDB which stores/retrieves documents based on embeddings, which allow for semantic search.
"""
from typing import TypeVar
from llm_workflow.base import Document, DocumentIndex, EmbeddingModel, EmbeddingRecord
from llm_workflow.internal_utilities import create_hash


ChromaCollection = TypeVar('ChromaCollection')


class ChromaDocumentIndex(DocumentIndex):
    """
    Chroma is a document index (vector database) that which provides a way to store embeddings
    associated with Document objects and then retrieve the documents that are most similar to
    another set of embeddings (or corresponding string).
    embeddings.

    The `ChromaDocumentIndex` class is a wrapper around Chroma that makes it easy to work with in
    a chain. When the `ChromaDocumentIndex` object is called (via __call__), either the `add`
    method will be called (if the `value` passed in is a list) or the `search` method will be
    called (if the `value` passed in is a string or Document).
    This functionality allows to object to be added to a chain and either add documents to the
    index or search for document in the index based on input.
    """

    def __init__(
            self,
            embeddings_model: EmbeddingModel | None = None,
            collection: ChromaCollection | None = None,
            n_results: int = 3) -> None:
        import chromadb
        super().__init__(n_results=n_results)
        self._collection = collection or chromadb.Client().create_collection('temp')
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
                metadatas.append(doc.metadata or {})
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

    @property
    def history(self) -> list[EmbeddingRecord]:
        """Propagates the history of any underlying models (e.g. embeddings model)."""
        return self._emb_model.history if self._emb_model else []

    @property
    def total_tokens(self) -> int | None:
        """
        Sums the `total_tokens` values across all EmbeddingRecord objects returned by this
        object's `history` property.
        """
        return self.calculate_historical(name='total_tokens')

    @property
    def cost(self) -> float | None:
        """
        Sums the `cost` values across all EmbeddingRecord objects returned by this object's
        `history` property.
        """
        return self.calculate_historical(name='cost')
