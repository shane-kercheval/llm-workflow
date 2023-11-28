"""Test llm_workflow.indexes.py."""
import chromadb
import pandas as pd
import pytest
from llm_workflow.base import Document
from llm_workflow.indexes import ChromaDocumentIndex
from llm_workflow.prompt_templates import (
    DocSearchTemplate,
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


def test_PythonObjectMetadataTemplate__extract_variables__false(credit_data: pd.DataFrame):  # noqa
    template = PythonObjectMetadataTemplate(extract_variables=False)
    # ensure prompt is not modified if no metadata is found
    assert template(prompt='This is my question.') == 'This is my question.'
    assert template._extracted_variables_last_call is None


    template = PythonObjectMetadataTemplate(objects={}, extract_variables=False)
    # ensure prompt is not modified if no metadata is found
    assert template(prompt='This is my question.') == 'This is my question.'
    assert template._extracted_variables_last_call is None

    template = PythonObjectMetadataTemplate(
        objects={'credit_data': credit_data},
        extract_variables=False,
    )
    result = template(prompt='This is my question.')
    assert template._extracted_variables_last_call is None
    assert 'This is my question.' in result
    assert 'credit_data' in result  # ensure object name is included
    assert 'checking_balance' in result  # ensure non-numeric columns are not included
    assert 'months_loan_duration' in result  # ensure numeric columns are not included

    template = PythonObjectMetadataTemplate(
        objects={'my_df': credit_data},
        extract_variables=False,
    )
    result = template(prompt='This is my question.')
    assert template._extracted_variables_last_call is None
    assert 'This is my question.' in result
    assert 'my_df' in result
    assert 'checking_balance' in result
    assert 'months_loan_duration' in result
    file_path = 'tests/test_data/prompt_templates/test_PythonObjectMetadataTemplate__dataframe__credit__name.txt'  # noqa
    with open(file_path, 'w') as f:
        f.write(result)

    template = PythonObjectMetadataTemplate(
        objects={'my_df': credit_data, 'my_dict': {'a': 1, 'b': 2}},
        extract_variables=False,
    )
    result = template(prompt='This is my question.')
    assert template._extracted_variables_last_call is None
    assert 'This is my question.' in result
    assert 'my_df' in result
    assert 'my_dict' in result
    assert 'checking_balance' in result
    assert 'months_loan_duration' in result
    assert 'python dictionary' in result
    assert "['a', 'b']" in result
    file_path = 'tests/test_data/prompt_templates/test_PythonObjectMetadataTemplate__dataframe__dict.txt'  # noqa
    with open(file_path, 'w') as f:
        f.write(result)


def test_PythonObjectMetadataTemplate__extract_variables__true__no_variables(credit_data: pd.DataFrame):  # noqa
    """Test scenario where no variables are used."""
    template = PythonObjectMetadataTemplate(extract_variables=True)
    # ensure prompt is not modified if no metadata is found
    assert template(prompt='This is my question.') == 'This is my question.'
    assert template._extracted_variables_last_call == set()

    template = PythonObjectMetadataTemplate(objects={}, extract_variables=True)
    # ensure prompt is not modified if no metadata is found
    assert template(prompt='This is my question.') == 'This is my question.'
    assert template._extracted_variables_last_call == set()

    template = PythonObjectMetadataTemplate(
        objects={'credit_data': credit_data},
        extract_variables=True,
    )
    result = template(prompt='This is my question.')
    assert template._extracted_variables_last_call == set()
    assert result == 'This is my question.'
    # `credit_data` is not used in the prompt, so it is not included in the metadata
    assert 'credit_data' not in result
    assert 'checking_balance' not in result
    assert 'months_loan_duration' not in result

def test_PythonObjectMetadataTemplate__extract_variables__true__variable_not_found():  # noqa
    """
    Test scenario where variable is used but not found in context with/without
    raise_not_found_error.
    """
    ####
    template = PythonObjectMetadataTemplate(
        objects=None,
        extract_variables=True,
        raise_not_found_error=False,  # do not raise error if variable is not found
    )
    result = template(prompt='This is a question about @credit_data dataset.')
    assert result == 'This is a question about `credit_data` dataset.'
    assert template._extracted_variables_last_call == {'credit_data'}

    template = PythonObjectMetadataTemplate(
        objects=None,
        extract_variables=True,
        raise_not_found_error=True,  # raise error if variable is not found
    )
    with pytest.raises(AssertionError):
        _ = template(prompt='This is a question about @non_existant_variable dataset.')
    assert template._extracted_variables_last_call == {'non_existant_variable'}


def test_PythonObjectMetadataTemplate__extract_variables__true(credit_data):  # noqa
    template = PythonObjectMetadataTemplate(
        objects={'credit_data': credit_data, 'my_dict': {'a': 1, 'b': 2}},
        extract_variables=True,
        raise_not_found_error=True,
    )
    result = template(prompt='This is a question about @credit_data dataset.')
    assert template._extracted_variables_last_call == {'credit_data'}
    assert 'This is a question about `credit_data` dataset.' in result
    assert 'credit_data' in result
    assert 'my_dict' not in result
    assert 'checking_balance' in result
    assert 'months_loan_duration' in result
    assert 'python dictionary' not in result
    assert "['a', 'b']" not in result
    file_path = 'tests/test_data/prompt_templates/test_PythonObjectMetadataTemplate__dataframe__extract_variables.txt'  # noqa
    with open(file_path, 'w') as f:
        f.write(result)

    result = template(prompt='This is a question about @my_dict.')
    assert template._extracted_variables_last_call == {'my_dict'}
    assert 'This is a question about `my_dict`.' in result
    assert 'credit_data' not in result
    assert 'my_dict' in result
    assert 'checking_balance' not in result
    assert 'months_loan_duration' not in result
    assert 'python dictionary' in result
    assert "['a', 'b']" in result
    file_path = 'tests/test_data/prompt_templates/test_PythonObjectMetadataTemplate__dict__extract_variables.txt'  # noqa
    with open(file_path, 'w') as f:
        f.write(result)

    # test adding objects after instantiation
    template = PythonObjectMetadataTemplate(
        objects=None,
        extract_variables=True,
        raise_not_found_error=True,
    )
    template.add_object('credit_data', credit_data)
    template.add_object('my_dict', {'a': 1, 'b': 2})
    result = template(prompt='This is a question about @my_dict and @credit_data.')
    assert template._extracted_variables_last_call == {'my_dict', 'credit_data'}
    assert 'This is a question about `my_dict` and `credit_data`.' in result
    assert 'credit_data' in result
    assert 'my_dict' in result
    assert 'checking_balance' in result
    assert 'months_loan_duration' in result
    assert 'python dictionary' in result
    assert "['a', 'b']" in result
    file_path = 'tests/test_data/prompt_templates/test_PythonObjectMetadataTemplate__dataframe_dict__extract_variables.txt'  # noqa
    with open(file_path, 'w') as f:
        f.write(result)


def test_PythonObjectMetadataTemplate__custom_functions(credit_data):  # noqa
    template = PythonObjectMetadataTemplate(
        objects={'credit_data': (credit_data, lambda obj, name: f'custom function: {name} - {obj.shape}')},  # noqa
        extract_variables=True,
        raise_not_found_error=True,
    )
    result = template(prompt='This is a question about @credit_data dataset.')
    assert template._extracted_variables_last_call == {'credit_data'}
    assert 'This is a question about `credit_data` dataset.' in result
    assert 'custom function: credit_data - (1000, 17)' in result
    assert 'credit_data' in result
    file_path = 'tests/test_data/prompt_templates/test_PythonObjectMetadataTemplate__dataframe__custom_functions.txt'  # noqa
    with open(file_path, 'w') as f:
        f.write(result)

    # test adding objects after instantiation
    template = PythonObjectMetadataTemplate(
        objects=None,
        extract_variables=True,
        raise_not_found_error=True,
    )
    template.add_object('credit_data', (credit_data, lambda obj, name: f'custom df function: {name} - {obj.shape}'))  # noqa
    template.add_object('my_dict', ({'a': 1, 'b': 2}, lambda obj, name: f'custom dict function: {name} - {len(obj)}'))  # noqa
    result = template(prompt='This is a question about @my_dict and @credit_data.')
    assert template._extracted_variables_last_call == {'my_dict', 'credit_data'}
    assert 'This is a question about `my_dict` and `credit_data`.' in result
    assert 'custom df function: credit_data - (1000, 17)' in result
    assert 'custom dict function: my_dict - 2' in result
    file_path = 'tests/test_data/prompt_templates/test_PythonObjectMetadataTemplate__dataframe_dict__custom_functions.txt'  # noqa
    with open(file_path, 'w') as f:
        f.write(result)



