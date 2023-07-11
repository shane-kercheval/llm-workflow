"""Tests the utilities.py file."""
from time import sleep
import re
import pytest
import openai
import requests
import os
from llm_workflow.exceptions import RequestError
from llm_workflow.base import Document
from llm_workflow.internal_utilities import (
    create_hash,
    has_method,
    has_property,
    retry_handler,
    Timer,
)
from llm_workflow.utilities import (
    DuckDuckGoSearch,
    StackOverflowSearchRecord,
    StackQuestion,
    _get_stack_overflow_answers,
    num_tokens,
    num_tokens_from_messages,
    scrape_url,
    SearchRecord,
    split_documents,
    StackOverflowSearch,
)


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
            """Not Needed."""

        @property
        def property_b(self) -> str:
            """Not Needed."""

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

def test_DuckDuckGoSearch():  # noqa
    query = "What is an agent in langworkflow?"
    search = DuckDuckGoSearch(top_n=1)
    results = search(query=query)
    assert len(results) == 1
    assert 'title' in results[0]
    assert 'href' in results[0]
    assert 'body' in results[0]
    assert len(search.history()) == 1
    assert search.history()[0].query == query
    assert search.history()[0].results == results

    query = "What is langworkflow?"
    results = search(query=query)
    assert len(results) == 1
    assert 'title' in results[0]
    assert 'href' in results[0]
    assert 'body' in results[0]
    assert len(search.history()) == 2
    assert search.history()[1].query == query
    assert search.history()[1].results == results

def test_DuckDuckGoSearch_caching():  # noqa
    """
    Test that searching DuckDuckGo based on same query returns same results with different uuid and
    timestamp.
    """
    query = "This is my fake query?"
    fake_results = [{'title': "fake results"}]
    search = DuckDuckGoSearch(top_n=1)
    # modify _history to mock a previous search based on a particular query
    search._history.append(SearchRecord(query=query, results=fake_results))
    response = search(query)
    assert response == fake_results
    assert len(search.history()) == 2
    assert search.history()[0].query == search.history()[1].query
    assert search.history()[0].results == search.history()[1].results
    assert search.history()[0].uuid != search.history()[1].uuid

def test_StackOverflowSearch():  # noqa
    # not sure how to test this in a way that won't break if the response from stack overflow
    # changes in the future
    # TODO: I don't want to make the tests fail when running on github workflows or someone is
    # building locally; but approach this will silently skip tests which is not ideal
    if os.getenv('STACK_OVERFLOW_KEY', None):
        # this question gets over 25K upvotes and has many answers; let's make sure we get the
        # expected number of questions/answers
        question = "Why is processing a sorted array faster than processing an unsorted array?"
        search = StackOverflowSearch(max_questions=1, max_answers=1)
        results = search(query=question)
        assert results
        assert len(results) == 1
        assert results[0].title == question
        assert results[0].answer_count > 1
        assert len(results[0].answers) == 1
        # check that the body of the question contains html but the text/markdown does not
        assert '<p>' in results[0].body
        assert len(results[0].body) > 100
        assert '<p>' not in results[0].text
        assert len(results[0].text) > 100
        assert '<p>' not in results[0].markdown
        assert len(results[0].markdown) > 100
        # check that the body of the answer contains html but the text/markdown does not
        assert '<p>' in results[0].answers[0].body
        assert len(results[0].answers[0].body) > 100
        assert '<p>' not in results[0].answers[0].text
        assert len(results[0].answers[0].text) > 100
        assert '<p>' not in results[0].answers[0].markdown
        assert len(results[0].answers[0].markdown) > 100

        question = "getting segmentation fault in linux"
        search = StackOverflowSearch(max_questions=2, max_answers=2)
        results = search(query=question)
        assert results
        assert len(results) > 1
        assert any(x for x in results if x.answer_count > 0)

        # make sure the function doesn't fail when there are no matches/results
        search = StackOverflowSearch(max_questions=2, max_answers=2)
        assert search(query="asdfasdfasdfasdflkasdfljsadlkfjasdlkfja") == []

def test_StackOverflowSearch_caching():  # noqa
    """
    Test that searching Stack Overflow based on same query returns same results with different uuid
    and timestamp.
    """
    query = "This is my fake query?"
    fake_results = [StackQuestion(question_id=1, score=1, creation_date=0, answer_count=1, title="fake", link="fake", body="<p>body</p>")]  # noqa
    search = StackOverflowSearch()
    # modify _history to mock a previous search based on a particular query
    search._history.append(StackOverflowSearchRecord(query=query, results=fake_results))
    response = search(query)
    assert response == fake_results
    assert len(search.history()) == 2
    assert search.history()[0].query == search.history()[1].query
    assert search.history()[0].results == search.history()[1].results
    assert search.history()[0].uuid != search.history()[1].uuid

def test__get_stack_overflow_answers_404():  # noqa
     if os.getenv('STACK_OVERFLOW_KEY', None):
        with pytest.raises(RequestError):
            _ = _get_stack_overflow_answers(question_id='asdf')
