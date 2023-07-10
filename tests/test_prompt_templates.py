"""Test llm_workflow.indexes.py."""
import chromadb
from llm_workflow.base import Document
from llm_workflow.indexes import ChromaDocumentIndex
from llm_workflow.prompt_templates import DocSearchTemplate
from tests.conftest import MockRandomEmbeddings


def test_DocSearchTemplate():  # noqa
    question = "This is my question."
    cost_per_token = 13
    embeddings_model = MockRandomEmbeddings(token_counter=len, cost_per_token=cost_per_token)
    doc_index = ChromaDocumentIndex(
        collection=chromadb.Client().create_collection('test'),
        embeddings_model=embeddings_model,
    )
    docs=[
        Document(content='This is a document'),
        Document(content='This is a another document'),
        Document(content='This is a another another document'),
    ]
    doc_index.add(docs=docs)
    expected_initial_tokens = sum(len(x.content) for x in docs)
    assert embeddings_model._history[0].total_tokens == expected_initial_tokens
    assert embeddings_model._history[0].cost == expected_initial_tokens * cost_per_token

    prompt_template = DocSearchTemplate(doc_index=doc_index, n_docs=2)
    assert prompt_template.similar_docs is None
    # the prompt template is propagating the costs from the embeddings model; in this case the
    # embeddings model added documents during this session/instantiatin
    assert prompt_template.total_tokens == expected_initial_tokens
    assert prompt_template.cost == expected_initial_tokens * cost_per_token

    prompt = prompt_template(prompt=question)
    assert len(prompt) > 0
    assert len(prompt_template.similar_docs) == 2

    # there was the call to create the initial document embeddings and then the call to create
    # the embeddings for the question
    assert len(embeddings_model._history) == 2

    assert prompt_template.total_tokens == embeddings_model.total_tokens
    assert prompt_template.cost == embeddings_model.cost
