"""test llm_workflow/vector_db/chroma.db."""
import chromadb
import pytest
from uuid import uuid4
from llm_workflow.base import Document, Record
from llm_workflow.indexes import ChromaDocumentIndex, DocumentIndex
from tests.conftest import MockABCDEmbeddings, MockRandomEmbeddings


class MockIndex(DocumentIndex):  # noqa
    def __init__(self, n_results: int = 3) -> None:
        super().__init__(n_results=n_results)
        self.documents = []

    def add(self, docs: list[Document]) -> None:  # noqa
        self.documents += docs

    def _search(self, doc: Document, n_results: int = 3) -> list[Document]:  # noqa
        return self.documents[0:n_results]

    def history(self) -> list[Record]:  # noqa
        return [Record()]


def test_base_index():  # noqa
    mock_index = MockIndex()
    with pytest.raises(TypeError):
        mock_index(1)
    # test `call()` when passing list[Document] which should call `add_documents()`
    documents_to_add = [Document(content='Doc A'), Document(content='Doc B')]
    return_value = mock_index(documents_to_add)
    assert return_value is None
    assert mock_index.documents == documents_to_add
    # test `call()` when passing Document which should call `search_documents()`
    return_value = mock_index(Document(content='doc'))
    assert return_value == documents_to_add
    assert len(mock_index.history())

def test_base_index_n_results():  # noqa
    """Test n_results when passed into __init__ vs __call__/search."""
    default_n_results = 2
    mock_index = MockIndex(n_results=default_n_results)
    # test `call()` when passing list[Document] which should call `add_documents()`
    documents_to_add = [
        Document(content='Doc A'),
        Document(content='Doc B'),
        Document(content='Doc C'),
        Document(content='Doc D'),
    ]
    return_value = mock_index(documents_to_add)
    assert return_value is None
    assert mock_index.documents == documents_to_add

    return_value = mock_index(Document(content='this doc is ignored in mock'))
    assert return_value == documents_to_add[0:default_n_results]
    return_value = mock_index(Document(content='this doc is ignored in mock'), n_results=1)
    assert return_value == documents_to_add[0:1]
    return_value = mock_index(Document(content='this doc is ignored in mock'), n_results=2)
    assert return_value == documents_to_add[0:2]
    return_value = mock_index(Document(content='this doc is ignored in mock'), n_results=3)
    assert return_value == documents_to_add[0:3]

def test_chroma_add_search_documents(fake_docs_abcd):  # noqa
    embeddings_model = MockABCDEmbeddings()
    client = chromadb.Client()
    collection = client.create_collection(str(uuid4()))
    chroma_db = ChromaDocumentIndex(collection=collection, embeddings_model=embeddings_model)
    assert chroma_db.total_tokens == 0
    assert chroma_db.cost == 0
    chroma_db.add(docs=None)
    assert chroma_db.total_tokens == 0
    assert chroma_db.cost == 0
    chroma_db.add(docs=[])
    assert chroma_db.total_tokens == 0
    assert chroma_db.cost == 0

    chroma_db.add(docs=fake_docs_abcd)
    initial_expected_tokens = len("Doc X") * len(fake_docs_abcd)
    initial_expected_cost = initial_expected_tokens * embeddings_model.cost_per_token
    # test that usage reflects underlying usage in embeddings
    assert chroma_db.total_tokens == initial_expected_tokens
    assert chroma_db.cost == initial_expected_cost
    assert chroma_db.total_tokens == embeddings_model.total_tokens
    assert chroma_db.cost == embeddings_model.cost
    assert len(embeddings_model._history) == 1
    assert embeddings_model._history[0].total_tokens == initial_expected_tokens
    assert embeddings_model._history[0].cost == initial_expected_cost

    # verify documents and embeddings where added to collection
    collection_docs = collection.get(include = ['documents', 'metadatas', 'embeddings'])
    assert collection_docs['documents'] == [x.content for x in fake_docs_abcd]
    assert collection_docs['metadatas'] == [x.metadata for x in fake_docs_abcd]
    assert len(collection_docs['ids']) == 4
    assert collection_docs['embeddings'] == list(embeddings_model.lookup.values())

    # search based on first doc
    results = chroma_db.search(value=fake_docs_abcd[1], n_results=3)
    assert len(results) == 3
    # first/best result should match doc 1
    assert results[0].content == fake_docs_abcd[1].content
    assert results[0].metadata['id'] == fake_docs_abcd[1].metadata['id']
    assert results[0].metadata['distance'] == 0
    # second result should match doc 0
    assert results[1].content == fake_docs_abcd[0].content
    assert results[1].metadata['id'] == fake_docs_abcd[0].metadata['id']
    assert results[1].metadata['distance'] > 0
    # third result should match doc 3 (index 2)
    assert results[2].content == fake_docs_abcd[2].content
    assert results[2].metadata['id'] == fake_docs_abcd[2].metadata['id']
    assert results[2].metadata['distance'] > results[1].metadata['distance']

    new_expected_tokens = initial_expected_tokens + len('Doc X')
    new_expected_cost = new_expected_tokens * embeddings_model.cost_per_token
    # test that usage reflects underlying usage in embeddings
    assert chroma_db.total_tokens == new_expected_tokens
    assert chroma_db.cost == new_expected_cost
    assert chroma_db.total_tokens == embeddings_model.total_tokens
    assert chroma_db.cost == embeddings_model.cost
    assert len(embeddings_model._history) == 2
    assert embeddings_model._history[0].total_tokens == initial_expected_tokens
    assert embeddings_model._history[0].cost == initial_expected_cost
    assert embeddings_model._history[1].total_tokens == len("Doc X")
    assert embeddings_model._history[1].cost == len("Doc X") * embeddings_model.cost_per_token

    # search based on third doc
    results = chroma_db.search(value=fake_docs_abcd[2], n_results=1)
    assert len(results) == 1
    # first/best result should match doc 2
    assert results[0].content == fake_docs_abcd[2].content
    assert results[0].metadata['id'] == fake_docs_abcd[2].metadata['id']
    assert results[0].metadata['distance'] == 0

    new_expected_tokens += len('Doc X')
    new_expected_cost = new_expected_tokens * embeddings_model.cost_per_token
    # test that usage reflects underlying usage in embeddings
    assert chroma_db.total_tokens == new_expected_tokens
    assert chroma_db.cost == new_expected_cost
    assert chroma_db.total_tokens == embeddings_model.total_tokens
    assert chroma_db.cost == embeddings_model.cost
    assert len(embeddings_model._history) == 3
    assert embeddings_model._history[0].total_tokens == initial_expected_tokens
    assert embeddings_model._history[0].cost == initial_expected_cost
    assert embeddings_model._history[1].total_tokens == len("Doc X")
    assert embeddings_model._history[1].cost == len("Doc X") * embeddings_model.cost_per_token
    assert embeddings_model._history[1].total_tokens == len("Doc X")
    assert embeddings_model._history[1].cost == len("Doc X") * embeddings_model.cost_per_token

def test_chroma_add_document_without_metadata():  # noqa
    cost_per_token = 13
    embeddings_model = MockRandomEmbeddings(token_counter=len, cost_per_token=cost_per_token)
    # test without passing a collection
    doc_index = ChromaDocumentIndex(embeddings_model=embeddings_model)
    docs = [
        Document(content='This is a document'),
        Document(content='This is a another document'),
        Document(content='This is a another another document'),
    ]
    doc_index.add(docs=docs)
    collection_docs = doc_index._collection.get(include = ['documents', 'metadatas', 'embeddings'])
    assert collection_docs['documents'] == [x.content for x in docs]
    assert collection_docs['metadatas'] == [ {'': ''},  {'': ''},  {'': ''}]
    assert len(collection_docs['ids']) == len(docs)
    assert len(collection_docs['embeddings']) == len(docs)

    results = doc_index.search(value=docs[0], n_results=1)
    assert 'distance' in results[0].metadata

    # test adding same documents
    new_docs = [
        Document(content='New Doc 1', metadata={'id': 0}),
        Document(content='New Doc 2', metadata={'id': 1}),
    ]
    doc_index.add(docs=docs + new_docs)
    collection_docs = doc_index._collection.get(include = ['documents', 'metadatas', 'embeddings'])
    assert collection_docs['documents'] == [x.content for x in docs + new_docs]
    assert collection_docs['metadatas'] == [ {'': ''},  {'': ''},  {'': ''}, {'id': 0}, {'id': 1}]
    assert len(collection_docs['ids']) == len(docs + new_docs)
    assert len(collection_docs['embeddings']) == len(docs + new_docs)

    results = doc_index.search(value=docs[0], n_results=1)
    assert 'distance' in results[0].metadata

def test_chroma_search_with_document_and_str(fake_docs_abcd):  # noqa
    embeddings_model = MockABCDEmbeddings()
    client = chromadb.Client()
    collection = client.create_collection(str(uuid4()))
    chroma_db = ChromaDocumentIndex(collection=collection, embeddings_model=embeddings_model)
    chroma_db.add(docs=fake_docs_abcd)
    # we don't need to test that the results are in the correct order; that is done above
    # we just need to test that searching with a Document returns the same results as searching
    # with a str
    assert isinstance(fake_docs_abcd[1], Document)
    doc_results = chroma_db.search(value=fake_docs_abcd[1])  # pass document object
    chroma_db._emb_model._next_lookup_index = 1
    str_results = chroma_db.search(value=fake_docs_abcd[1].content)  # pass string
    assert doc_results == str_results
    assert chroma_db.history()[1].metadata == chroma_db.history()[2].metadata

def test_chroma_without_collection_or_embeddings_model():  # noqa
    chroma_db = ChromaDocumentIndex(collection=None, embeddings_model=None)
    assert chroma_db.total_tokens == 0
    assert chroma_db.cost == 0
    chroma_db.add(docs=None)
    assert chroma_db.total_tokens == 0
    assert chroma_db.cost == 0
    chroma_db.add(docs=[])
    assert chroma_db.total_tokens == 0
    assert chroma_db.cost == 0

    docs = [
        Document(content="This is a document about basketball.", metadata={'id': 0}),
        Document(content="This is a document about baseball.", metadata={'id': 1}),
        Document(content="This is a document about football.", metadata={'id': 2}),
    ]
    chroma_db.add(docs=docs)
    assert chroma_db.total_tokens == 0
    assert chroma_db.cost == 0
    assert chroma_db.history() == []

    # verify documents and embeddings where added to collection
    collection_docs = chroma_db._collection.get(include = ['documents', 'metadatas', 'embeddings'])
    assert collection_docs['documents'] == [x.content for x in docs]
    assert collection_docs['metadatas'] == [x.metadata for x in docs]
    assert len(collection_docs['ids']) == len(docs)
    assert len(collection_docs['embeddings']) == len(docs)

    # search based on first doc
    results = chroma_db.search(value="Give a document about baseball", n_results=1)
    assert len(results) == 1
    # first/best result should match doc 1
    assert results[0].content == docs[1].content
    assert results[0].metadata['id'] == docs[1].metadata['id']
    assert results[0].metadata['distance'] < 1
    assert chroma_db.total_tokens == 0
    assert chroma_db.cost == 0
    assert chroma_db.history() == []
