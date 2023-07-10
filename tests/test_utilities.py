"""Tests the utilities.py file."""
from time import sleep
import re
import pytest
import openai
import requests
from llm_workflow.base import Document
from llm_workflow.exceptions import RequestError
from llm_workflow.internal_utilities import Timer, create_hash, has_method, has_property, \
    retry_handler
from llm_workflow.utilities import num_tokens, num_tokens_from_messages, scrape_url, split_documents

def test_timer_seconds():  # noqa
    with Timer() as timer:
        sleep(1.1)

    assert timer.interval
    assert re.match(pattern=r'1\.\d+ seconds', string=timer.formatted())
    assert str(timer) == timer.formatted()

    with pytest.raises(ValueError):  # noqa
        timer.formatted(units='days')

def test_create_hash():  # noqa
    value_a = create_hash('Test value 1')
    assert value_a
    value_b = create_hash('Test value 2')
    assert value_b
    assert value_a != value_b
    value_c = create_hash('Test value 1')
    assert value_c == value_a

def test_num_tokens():  # noqa
    assert num_tokens(model_name='gpt-3.5-turbo', value="This should be six tokens.") == 6

def test_num_tokens_from_messages():  # noqa
    # copied from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    example_messages = [
        {
            "role": "system",
            "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English.",  # noqa
        },
        {
            "role": "system",
            "name": "example_user",
            "content": "New synergies will help drive top-line growth.",
        },
        {
            "role": "system",
            "name": "example_assistant",
            "content": "Things working well together will increase revenue.",
        },
        {
            "role": "system",
            "name": "example_user",
            "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage.",  # noqa
        },
        {
            "role": "system",
            "name": "example_assistant",
            "content": "Let's talk later when we're less busy about how to do better.",
        },
        {
            "role": "user",
            "content": "This late pivot means we don't have time to boil the ocean for the client deliverable.",  # noqa
        },
    ]
    model_name = 'gpt-3.5-turbo'
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=example_messages,
        temperature=0,
        max_tokens=1,  # we're only counting input tokens here, so let's not waste tokens on output
    )
    expected_value = response["usage"]["prompt_tokens"]
    actual_value = num_tokens_from_messages(model_name=model_name, messages=example_messages)
    assert expected_value == actual_value

    # above we checked that the numbers match exactly from what OpenAI returns;
    # here, let's just check that the other models run and return >0 to avoid API calls
    assert num_tokens_from_messages(model_name='gpt-3.5-turbo-0301', messages=example_messages) > 0
    assert num_tokens_from_messages(model_name='gpt-4', messages=example_messages) > 0
    assert num_tokens_from_messages(model_name='gpt-4-0314', messages=example_messages) > 0
    with pytest.raises(NotImplementedError):
        num_tokens_from_messages(model_name='<not implemented>', messages=example_messages)

def test_retry_handler():  # noqa
    r = retry_handler()
    actual_value = r(
        lambda x, y: (x, y),
        x='A',
        y='B',
    )
    assert actual_value == ('A', 'B')

def test_has_method_has_property():  # noqa
    class Fake:
        def __init__(self) -> None:
            self.variable_c = 'c'

        def method_a(self) -> str:
            return 'a'

        @property
        def property_b(self) -> str:
            return 'b'

    assert has_method(Fake(), 'method_a')
    assert not has_method(Fake(), 'property_b')
    assert not has_method(Fake(), 'variable_c')
    assert not has_method(lambda x: x, 'test')

    assert not has_property(Fake(), 'method_a')
    assert has_property(Fake(), 'property_b')
    assert has_property(Fake(), 'variable_c')
    assert not has_property(lambda x: x, 'test')

def test_split_documents__preserve_words_false():  # noqa
    max_chunk_size = 10

    result = split_documents([], max_chars=max_chunk_size, preserve_words=False)
    assert result == []

    docs = [
        Document(content='', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=False)
    assert result == []

    docs = [
        Document(content=' ', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=False)
    assert result == []

    docs = [
        Document(content='  ', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=False)
    assert result == []

    docs = [
        Document(content='\n', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=False)
    assert result == []

    # test that space does not affect result
    docs = [
        Document(content='0123 5678', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=False)
    assert result == docs

    # test that leading and trailing space gets stripped
    docs = [
        Document(content=' 123 567 ', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=False)
    docs[0].content = docs[0].content.strip()
    assert result == docs


    docs = [
        Document(content=' 123 567  ', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=False)
    docs[0].content = docs[0].content.strip()
    assert result == docs

    docs = [
        Document(content=' 123 567  ', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=False)
    docs[0].content = docs[0].content.strip()
    assert result == docs

    docs = [
        Document(content='012345678', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=False)
    assert result == docs

    docs = [
        Document(content='0123456789', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=False)
    assert result == docs

    docs = [
        Document(content='0123456789a', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=False)
    expected_result = [
        Document(content='0123456789', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
        Document(content='a', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    assert result == expected_result

    docs = [
        Document(content='', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
        Document(content='0', metadata={'doc': 1}, embedding=[1, 1, 1, 1]),
        Document(content='0123 5678', metadata={'doc': 2}, embedding=[2, 2, 2, 2]),
        Document(content='0123 56789', metadata={'doc': 3}, embedding=[3, 3, 3, 3]),
        Document(content='0123 56789a', metadata={'doc': 4}, embedding=[4, 4, 4, 4]),
        Document(content='0123 56789abcdefghi', metadata={'doc': 5}, embedding=[5, 5, 5, 5]),
        Document(content='0123 56789abcdefghij', metadata={'doc': 6}, embedding=[6, 6, 6, 6]),
        Document(content='0123 56789abcdefghijk', metadata={'doc': 7}, embedding=[7, 7, 7, 7]),
    ]
    expected_result = [
        Document(content='0', metadata={'doc': 1}, embedding=[1, 1, 1, 1]),
        Document(content='0123 5678', metadata={'doc': 2}, embedding=[2, 2, 2, 2]),
        Document(content='0123 56789', metadata={'doc': 3}, embedding=[3, 3, 3, 3]),
        Document(content='0123 56789', metadata={'doc': 4}, embedding=[4, 4, 4, 4]),
        Document(content='a', metadata={'doc': 4}, embedding=[4, 4, 4, 4]),
        Document(content='0123 56789', metadata={'doc': 5}, embedding=[5, 5, 5, 5]),
        Document(content='abcdefghi', metadata={'doc': 5}, embedding=[5, 5, 5, 5]),
        Document(content='0123 56789', metadata={'doc': 6}, embedding=[6, 6, 6, 6]),
        Document(content='abcdefghij', metadata={'doc': 6}, embedding=[6, 6, 6, 6]),
        Document(content='0123 56789', metadata={'doc': 7}, embedding=[7, 7, 7, 7]),
        Document(content='abcdefghij', metadata={'doc': 7}, embedding=[7, 7, 7, 7]),
        Document(content='k', metadata={'doc': 7}, embedding=[7, 7, 7, 7]),
    ]
    result = split_documents(docs, max_chunk_size, preserve_words=False)
    assert result == expected_result

def test_split_documents__preserve_words_true():  # noqa
    max_chunk_size = 10

    result = split_documents([], max_chars=max_chunk_size, preserve_words=True)
    assert result == []

    docs = [
        Document(content='', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    assert result == []

    docs = [
        Document(content=' ', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    assert result == []

    docs = [
        Document(content='  ', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    assert result == []

    docs = [
        Document(content='\n', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    assert result == []

    # test that space does not affect result
    docs = [
        Document(content='0123 5678', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    assert result == docs

    # test that leading and trailing space gets stripped
    docs = [
        Document(content=' 123 567 ', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    docs[0].content = docs[0].content.strip()
    assert result == docs

    # test with 10 characters
    docs = [
        Document(content=' 123 567  ', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    docs[0].content = docs[0].content.strip()
    assert result == docs

    # test with 11 characters
    docs = [
        Document(content=' 123 567  ', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    docs[0].content = docs[0].content.strip()
    assert result == docs

    # test with 9 characaters
    docs = [
        Document(content='012345678', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    assert result == docs

    # test with 10 characters no space
    docs = [
        Document(content='0123456789', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    assert result == docs

    # test with 11 characters and no space; since no space was found, it splits up the word
    docs = [
        Document(content='0123456789a', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    expected_result = [
        Document(content='0123456789', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
        Document(content='a', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    assert result == expected_result

    # test with 11 characters and space; it should preserve whole words
    docs = [
        Document(content='0123 567 9a', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    expected_result = [
        Document(content='0123 567', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
        Document(content='9a', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    assert result == expected_result

    # test with 11 characters and space; it should preserve whole words
    docs = [
        Document(content='0123 56789\nabc', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    expected_result = [
        Document(content='0123 56789', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
        Document(content='abc', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    assert result == expected_result

    # test with 11 characters and space; it should preserve whole words
    docs = [
        Document(content='0123 56789\n abc', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    expected_result = [
        Document(content='0123 56789', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
        Document(content='abc', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    assert result == expected_result

    # test with 11 characters and space; it should preserve whole words
    docs = [
        Document(content='0123 5678\n 1234 67890 a', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    expected_result = [
        Document(content='0123 5678', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
        Document(content='1234 67890', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
        Document(content='a', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    assert result == expected_result

    docs = [
        Document(content='This is a normal sentence.', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),  # noqa
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    expected_result = [
        Document(content='This is a', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
        Document(content='normal', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
        Document(content='sentence.', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    assert result == expected_result

    docs = [
        Document(content='', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
        Document(content='0', metadata={'doc': 1}, embedding=[1, 1, 1, 1]),
        Document(content='012345678', metadata={'doc': 2}, embedding=[2, 2, 2, 2]),
        Document(content='0123456789', metadata={'doc': 3}, embedding=[3, 3, 3, 3]),
        Document(content='0123456789a', metadata={'doc': 4}, embedding=[4, 4, 4, 4]),
        Document(content=' 0123456789ab\ndefghijk', metadata={'doc': 5}, embedding=[5, 5, 5, 5]),
        Document(content=' 0 23456789abcd fghij', metadata={'doc': 6}, embedding=[6, 6, 6, 6]),
        Document(content='0123456789abcdefghijk ', metadata={'doc': 7}, embedding=[7, 7, 7, 7]),
    ]
    expected_result = [
        Document(content='0', metadata={'doc': 1}, embedding=[1, 1, 1, 1]),
        Document(content='012345678', metadata={'doc': 2}, embedding=[2, 2, 2, 2]),
        Document(content='0123456789', metadata={'doc': 3}, embedding=[3, 3, 3, 3]),
        Document(content='0123456789', metadata={'doc': 4}, embedding=[4, 4, 4, 4]),
        Document(content='a', metadata={'doc': 4}, embedding=[4, 4, 4, 4]),
        Document(content='0123456789', metadata={'doc': 5}, embedding=[5, 5, 5, 5]),
        Document(content='ab', metadata={'doc': 5}, embedding=[5, 5, 5, 5]),
        Document(content='defghijk', metadata={'doc': 5}, embedding=[5, 5, 5, 5]),
        Document(content='0', metadata={'doc': 6}, embedding=[6, 6, 6, 6]),
        Document(content='23456789ab', metadata={'doc': 6}, embedding=[6, 6, 6, 6]),
        Document(content='cd fghij', metadata={'doc': 6}, embedding=[6, 6, 6, 6]),
        Document(content='0123456789', metadata={'doc': 7}, embedding=[7, 7, 7, 7]),
        Document(content='abcdefghij', metadata={'doc': 7}, embedding=[7, 7, 7, 7]),
        Document(content='k', metadata={'doc': 7}, embedding=[7, 7, 7, 7]),
    ]
    result = split_documents(docs, max_chunk_size, preserve_words=True)
    assert result == expected_result

def test_scrape_url():  # noqa
    text = scrape_url(url='https://example.com/')
    assert 'example' in text.lower()

def test_scrape_url_404():  # noqa
    with pytest.raises(RequestError):
         scrape_url(url="https://example.com/asdf")

def test_RequestError():  # noqa
    response = requests.get("https://example.com/asdf")
    assert RequestError(status_code=response.status_code, reason=response.reason)
