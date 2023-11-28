"""Tests the utilities.py file."""
from time import sleep
import re
import pytest
import requests
import os
from textwrap import dedent
from llm_workflow.exceptions import RequestError
from llm_workflow.base import Document
from llm_workflow.internal_utilities import (
    create_hash,
    execute_code_blocks,
    extract_code_blocks,
    extract_variables,
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

def test__extract_code_blocks__conversation_sum(conversation_sum):  # noqa
    extracted_code_blocks = extract_code_blocks(conversation_sum['model_1']['responses'][0])
    assert len(extracted_code_blocks) == 2
    assert extracted_code_blocks[0] == dedent("""
        def sum_numbers(num1, num2):
            return num1 + num2
        """).strip()
    assert extracted_code_blocks[1] == dedent("""
        result = sum_numbers(5, 3)
        print(result)  # Output: 8
        """).strip()

    extracted_code_blocks = extract_code_blocks(conversation_sum['model_1']['responses'][1])
    assert len(extracted_code_blocks) == 1
    assert extracted_code_blocks[0] == dedent("""
        assert sum_numbers(5, 3) == 8
        assert sum_numbers(-10, 10) == 0
        """).strip()

    extracted_code_blocks = extract_code_blocks(conversation_sum['model_2']['responses'][0])
    assert len(extracted_code_blocks) == 2
    assert extracted_code_blocks[0] == dedent("""
        def sum_two_numbers(num1, num2):
            return num1 + num2
        """).strip()
    assert extracted_code_blocks[1] == dedent("""
        result = sum_two_numbers(5, 3)
        print(result)  # Outputs: 8
        """).strip()

    extracted_code_blocks = extract_code_blocks(conversation_sum['model_2']['responses'][1])
    assert len(extracted_code_blocks) == 1
    assert extracted_code_blocks[0] == dedent("""
        assert sum_two_numbers(5, 3) == 8, "Should be 8"
        assert sum_two_numbers(-1, 1) == 0, "Should be 0"
        assert sum_two_numbers(0, 0) == 0, "Should be 0"
        assert sum_two_numbers(100, 200) == 300, "Should be 300"
        """).strip()

def test__extract_code_blocks__conversation_mask_emails(conversation_mask_email):  # noqa
    extracted_code_blocks = extract_code_blocks(conversation_mask_email['model_1']['responses'][0])
    assert len(extracted_code_blocks) == 2
    assert extracted_code_blocks[0] == dedent("""
        def mask_email(email):
            local_part, domain = email.split('@')
            masked_local_part = '*' * len(local_part)
            masked_email = masked_local_part + '@' + domain
            return masked_email
        """).strip()
    assert extracted_code_blocks[1] == dedent("""
        email = 'example@example.com'
        masked_email = mask_email(email)
        print(masked_email)  # Output: ********@example.com
        """).strip()

    extracted_code_blocks = extract_code_blocks(conversation_mask_email['model_1']['responses'][1])
    assert len(extracted_code_blocks) == 1
    assert extracted_code_blocks[0] == dedent("""
        # Test case 1: Masking email with alphanumeric local part
        email1 = 'example123@example.com'
        assert mask_email(email1) == '***********@example.com'

        # Test case 2: Masking email with special characters in local part
        email2 = 'ex@mple@example.com'
        assert mask_email(email2) == '******@example.com'
        """).strip()

    extracted_code_blocks = extract_code_blocks(conversation_mask_email['model_2']['responses'][0])
    assert len(extracted_code_blocks) == 1
    assert extracted_code_blocks[0] == dedent("""
        def mask_email(email):
            try:
                email_parts = email.split('@')
                # Mask first part
                masked_part = email_parts[0][0] + "****" + email_parts[0][-1]
                # Combine masked part and domain
                masked_email = masked_part + '@' + email_parts[1]
                return masked_email
            except Exception as e:
                print("An error occurred: ", e)
                return None
        """).strip()

    extracted_code_blocks = extract_code_blocks(conversation_mask_email['model_2']['responses'][1])
    assert len(extracted_code_blocks) == 1
    assert extracted_code_blocks[0] == dedent("""
        assert mask_email("john.doe@example.com") == "j****e@example.com"
        assert mask_email("jane_doe@example.com") == "j****e@example.com"
        assert mask_email("test@test.com") == "t****t@test.com"
        """).strip()

def test__execute_code_blocks__without_global_namespace(conversation_sum):  # noqa
    code_blocks = extract_code_blocks(conversation_sum['model_1']['responses'][0])
    code_blocks.append('assert sum_numbers(5, 3) == 8')
    code_blocks.append('assert sum_numbers(5, 3) != 8')
    assert len(code_blocks) == 4
    results = execute_code_blocks(code_blocks)
    assert len(results) == 4
    assert results[0] is None
    assert results[1] is None
    assert results[2] is None
    assert isinstance(results[3], AssertionError)

    # this will fail because global_namespace was not reused so sum_numbers is not defined during
    # a subsequent call to execute_code_blocks
    results = execute_code_blocks(code_blocks=['assert sum_numbers(5, 3) == 8'])
    assert len(results) == 1
    assert isinstance(results[0], NameError)
    assert str(results[0]) == "name 'sum_numbers' is not defined"

def test__execute_code_blocks__with_global_namespace(conversation_sum):  # noqa
    code_blocks = extract_code_blocks(conversation_sum['model_1']['responses'][0])
    code_blocks.append('assert sum_numbers(5, 3) == 8')
    code_blocks.append('assert sum_numbers(5, 3) != 8')
    assert len(code_blocks) == 4
    global_namespace = {}
    results = execute_code_blocks(code_blocks, global_namespace)
    assert len(results) == 4
    assert results[0] is None
    assert results[1] is None
    assert results[2] is None
    assert isinstance(results[3], AssertionError)
    assert 'sum_numbers' in global_namespace
    assert 'result' in global_namespace

    # this will NOT fail because global_namespace was reused so the state is carried over to
    # a subsequent call to execute_code_blocks
    results = execute_code_blocks(
        code_blocks=['assert sum_numbers(5, 3) == 8', 'assert sum_numbers(5, 3) != 8'],
        global_namespace=global_namespace,
    )
    assert len(results) == 2
    assert results[0] is None
    assert isinstance(results[1], AssertionError)

def test__extract_variables():  # noqa
    assert extract_variables('') == set()
    assert extract_variables('This is an email shane@email.com not a variable.') == set()
    assert extract_variables('This is not an email shane@email and not a variable.') == set()
    assert extract_variables('shane@email.com') == set()
    assert extract_variables('.@email.com') == set()
    assert extract_variables('@email.com') == set()
    assert extract_variables('@email') == {'email'}
    text = 'This is a variable @my_variable and should be extracted'
    results = extract_variables(text)
    assert results == {'my_variable'}
    text = 'This variable is at the end of a sentence @my_variable!'
    results = extract_variables(text)
    assert results == {'my_variable'}
    text = 'This has two @my_variable and another @my_variable.'
    results = extract_variables(text)
    assert results == {'my_variable'}
    text = 'This has three @my_variable and another @my_variable and @my_variable.'
    results = extract_variables(text)
    assert results == {'my_variable'}
    text = '@_my_var_1 and @_my_var_2_.'
    results = extract_variables(text)
    assert results == {'_my_var_1', '_my_var_2_'}
    text = '@_my_var_1 and @_my_var_2_. This is another sentence'
    results = extract_variables(text)
    assert results == {'_my_var_1', '_my_var_2_'}
    text = '@_my_var_1 and @_my_var_2_; this is some more text.'
    results = extract_variables(text)
    assert results == {'_my_var_1', '_my_var_2_'}
    text = 'A variable with number @var1234 should match.'
    results = extract_variables(text)
    assert results == {'var1234'}
    text = 'A variable with underscore @var_name should match.'
    results = extract_variables(text)
    assert results == {'var_name'}
    text = 'Multiple @@ signs should not confuse @@var.'
    results = extract_variables(text)
    assert results == {'var'}
    text = 'Variable at the end of a line @end_of_line\n'
    results = extract_variables(text)
    assert results == {'end_of_line'}
    # text = 'Variables next to each other @var1@var2'
    # results = extract_variables(text)
    # assert results == {'var1', 'var2'}
    text = 'Variable in parentheses (@var_in_paren).'
    results = extract_variables(text)
    assert results == {'var_in_paren'}
    text = 'Variable in brackets [@var_in_brackets].'
    results = extract_variables(text)
    assert results == {'var_in_brackets'}
    text = 'Variable with punctuation @var_punc!'
    results = extract_variables(text)
    assert results == {'var_punc'}
    text = 'Variable with comma, @var_comma, should match.'
    results = extract_variables(text)
    assert results == {'var_comma'}
    text = 'Variables with leading underscores @_underscore_var should match.'
    results = extract_variables(text)
    assert results == {'_underscore_var'}
    text = 'Variable followed by a special character @special$ should match.'
    results = extract_variables(text)
    assert results == {'special'}
    text = 'Variable inside quotes "@quoted_var" should match.'
    results = extract_variables(text)
    assert results == {'quoted_var'}
    text = 'A tricky case with email-like pattern @not_an_email@domain.com'
    results = extract_variables(text)
    assert results == {'not_an_email'}
    text = 'Multiple variables separated by comma @var1, @var2, and @var3.'
    results = extract_variables(text)
    assert results == {'var1', 'var2', 'var3'}
    text = 'Multiple variables separated by comma and backtick `@var1`, `@var2`, and `@var3`.'
    results = extract_variables(text)
    assert results == {'var1', 'var2', 'var3'}
    text = 'Variable followed by a period and space @var_period. should match.'
    results = extract_variables(text)
    assert results == {'var_period'}
    text = 'Variable followed by other symbols @var_symbols?! should match.'
    results = extract_variables(text)
    assert results == {'var_symbols'}
