"""Test llm_workflow.indexes.py."""
import chromadb
import pandas as pd
from llm_workflow.base import Document
from llm_workflow.indexes import ChromaDocumentIndex
from llm_workflow.prompt_templates import (
    DocSearchTemplate,
    MetadataMetadata,
    PythonObjectMetadataTemplate,
    extract_metadata,
)
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


def test_extract_metadata__default():  # noqa
    assert '<lambda' in extract_metadata(lambda x: x)
    assert 'my_lambda' not in extract_metadata(lambda x: x)
    assert 'my_lambda' in extract_metadata(lambda x: x, 'my_lambda')


def test_extract_metadata__dict():  # noqa
    dict_metadata = extract_metadata({'a': 1, 'b': 2})
    assert 'python dictionary' in dict_metadata
    assert 'my_dict' not in dict_metadata
    assert "['a', 'b']" in dict_metadata
    assert 'my_dict' in extract_metadata({'a': 1, 'b': 2}, 'my_dict')


def test_extract_metadata__list():  # noqa
    assert 'python list' in extract_metadata(['a', 'b', 1, None])
    assert 'my_list' not in extract_metadata(['a', 'b', 1, None])
    assert 'my_list' in extract_metadata(['a', 'b', 1, None], 'my_list')


def test_extract_metadata__pandas_dataframe(credit_data: pd.DataFrame):  # noqa
    results = extract_metadata(credit_data)
    assert 'credit_data' not in results
    file_path = 'tests/test_data/prompt_templates/test_extract_metadata__dataframe__credit.txt'
    with open(file_path, 'w') as f:
        f.write(results)

    results = extract_metadata(credit_data, 'credit_data')
    assert 'credit_data' in results
    file_path = 'tests/test_data/prompt_templates/test_extract_metadata__dataframe__credit__name.txt'  # noqa
    with open(file_path, 'w') as f:
        f.write(results)


def test_PythonObjectMetadataTemplate(credit_data: pd.DataFrame):  # noqa
    template = PythonObjectMetadataTemplate(metadatas=[])
    assert 'This is my question.' in template(prompt='This is my question.')

    template = PythonObjectMetadataTemplate(metadatas=[
        MetadataMetadata(obj=credit_data),
    ])
    result = template(prompt='This is my question.')
    assert 'This is my question.' in result
    assert 'my_df' not in result
    assert 'checking_balance' in result
    assert 'months_loan_duration' in result

    template = PythonObjectMetadataTemplate(metadatas=[
        MetadataMetadata(obj=credit_data, object_name='my_df'),
    ])
    result = template(prompt='This is my question.')
    assert 'This is my question.' in result
    assert 'my_df' in result
    assert 'checking_balance' in result
    assert 'months_loan_duration' in result
    file_path = 'tests/test_data/prompt_templates/test_PythonObjectMetadataTemplate__dataframe__credit__name.txt'  # noqa
    with open(file_path, 'w') as f:
        f.write(result)


    template = PythonObjectMetadataTemplate(metadatas=[
        MetadataMetadata(obj=credit_data, object_name='my_df'),
        MetadataMetadata(obj={'a': 1, 'b': 2}),
    ])
    result = template(prompt='This is my question.')
    assert 'This is my question.' in result
    assert 'my_df' in result
    assert 'checking_balance' in result
    assert 'months_loan_duration' in result
    assert 'python dictionary' in result
    assert "['a', 'b']" in result
    assert 'my_dict' not in result
    file_path = 'tests/test_data/prompt_templates/test_PythonObjectMetadataTemplate__dataframe__dict.txt'  # noqa
    with open(file_path, 'w') as f:
        f.write(result)
