"""Test Link."""
import os
import pytest
from llm_workflow.base import EmbeddingRecord, ExchangeRecord, RecordKeeper, Record, UsageRecord
from llm_workflow.exceptions import RequestError
from llm_workflow.links import DuckDuckGoSearch, SearchRecord, StackOverflowSearch, \
    StackOverflowSearchRecord, StackQuestion, _get_stack_overflow_answers


class MockLink(RecordKeeper):
    """Mocks a Task object."""

    def __init__(self) -> None:
        self._history = []

    def __call__(self, record: Record) -> None:
        """Adds the record to the history."""
        return self._history.append(record)

    @property
    def history(self) -> list[Record]:
        """Returns history."""
        return self._history


def test_history_tracker():  # noqa
    tracker = MockLink()
    assert tracker.history == tracker.history_filter()
    assert tracker.history == []
    assert tracker.history_filter(Record) == []
    assert tracker.history_filter(UsageRecord) == []
    assert tracker.history_filter(ExchangeRecord) == []
    assert tracker.history_filter(EmbeddingRecord) == []
    assert tracker.calculate_historical(name='does_not_exist') == 0
    assert tracker.calculate_historical(name='cost') == 0
    assert tracker.calculate_historical(name='total_tokens') == 0
    assert tracker.calculate_historical(name='prompt_tokens') == 0
    assert tracker.calculate_historical(name='response_tokens') == 0

    record_a = Record(metadata={'id': 'a'})
    tracker(record_a)
    assert tracker.history == tracker.history_filter()
    assert tracker.history == [record_a]
    assert tracker.history_filter(Record) == [record_a]
    assert tracker.history_filter(UsageRecord) == []
    assert tracker.history_filter(ExchangeRecord) == []
    assert tracker.history_filter(EmbeddingRecord) == []
    assert tracker.calculate_historical(name='does_not_exist') == 0
    assert tracker.calculate_historical(name='cost') == 0
    assert tracker.calculate_historical(name='total_tokens') == 0
    assert tracker.calculate_historical(name='prompt_tokens') == 0
    assert tracker.calculate_historical(name='response_tokens') == 0

    record_b = UsageRecord(total_tokens=1, metadata={'id': 'b'})
    tracker(record_b)
    assert tracker.history == tracker.history_filter()
    assert tracker.history == [record_a, record_b]
    assert tracker.history_filter(Record) == [record_a, record_b]
    assert tracker.history_filter(UsageRecord) == [record_b]
    assert tracker.history_filter(ExchangeRecord) == []
    assert tracker.history_filter(EmbeddingRecord) == []
    assert tracker.calculate_historical(name='does_not_exist') == 0
    assert tracker.calculate_historical(name='cost') == 0
    assert tracker.calculate_historical(name='total_tokens') == 1
    assert tracker.calculate_historical(name='prompt_tokens') == 0
    assert tracker.calculate_historical(name='response_tokens') == 0

    record_c = UsageRecord(cost=3, metadata={'id': 'c'})
    tracker(record_c)
    assert tracker.history == tracker.history_filter()
    assert tracker.history == [record_a, record_b, record_c]
    assert tracker.history_filter(Record) == [record_a, record_b, record_c]
    assert tracker.history_filter(UsageRecord) == [record_b, record_c]
    assert tracker.history_filter(ExchangeRecord) == []
    assert tracker.history_filter(EmbeddingRecord) == []
    assert tracker.calculate_historical(name='does_not_exist') == 0
    assert tracker.calculate_historical(name='cost') == 3
    assert tracker.calculate_historical(name='total_tokens') == 1
    assert tracker.calculate_historical(name='prompt_tokens') == 0
    assert tracker.calculate_historical(name='response_tokens') == 0

    record_d = UsageRecord(total_tokens=7, cost=6, metadata={'id': 'd'})
    tracker(record_d)
    assert tracker.history == tracker.history_filter()
    assert tracker.history == [record_a, record_b, record_c, record_d]
    assert tracker.history_filter(Record) == [record_a, record_b, record_c, record_d]
    assert tracker.history_filter(UsageRecord) == [record_b, record_c, record_d]
    assert tracker.history_filter(ExchangeRecord) == []
    assert tracker.history_filter(EmbeddingRecord) == []
    assert tracker.calculate_historical(name='does_not_exist') == 0
    assert tracker.calculate_historical(name='cost') == 9
    assert tracker.calculate_historical(name='total_tokens') == 8
    assert tracker.calculate_historical(name='prompt_tokens') == 0
    assert tracker.calculate_historical(name='response_tokens') == 0

    record_e = ExchangeRecord(
        prompt="the prompt",
        response="the response",
        cost=20,
        total_tokens=10,
        metadata={'id': 'e'},
    )
    tracker(record_e)
    assert tracker.history == tracker.history_filter()
    assert tracker.history == [record_a, record_b, record_c, record_d, record_e]
    assert tracker.history_filter(Record) == [record_a, record_b, record_c, record_d, record_e]
    assert tracker.history_filter(UsageRecord) == [record_b, record_c, record_d, record_e]
    assert tracker.history_filter(ExchangeRecord) == [record_e]
    assert tracker.history_filter(EmbeddingRecord) == []
    assert tracker.calculate_historical(name='does_not_exist') == 0
    assert tracker.calculate_historical(name='cost') == 29
    assert tracker.calculate_historical(name='total_tokens') == 18
    assert tracker.calculate_historical(name='prompt_tokens') == 0
    assert tracker.calculate_historical(name='response_tokens') == 0
    # test calculating historical values based on ExchangeRecords
    assert tracker.calculate_historical(name='cost', record_types=ExchangeRecord) == 20
    assert tracker.calculate_historical(name='total_tokens', record_types=ExchangeRecord) == 10
    assert tracker.calculate_historical(name='prompt_tokens', record_types=ExchangeRecord) == 0
    assert tracker.calculate_historical(name='response_tokens', record_types=ExchangeRecord) == 0

    record_f = ExchangeRecord(
        prompt="the prompt",
        response="the response",
        cost=13,
        total_tokens=5,
        prompt_tokens=7,
        response_tokens=9,
        metadata={'id': 'f'},
    )
    tracker(record_f)
    assert tracker.history == tracker.history_filter()
    assert tracker.history == [record_a, record_b, record_c, record_d, record_e, record_f]
    assert tracker.history_filter(Record) == [record_a, record_b, record_c, record_d, record_e, record_f]  # noqa
    assert tracker.history_filter(UsageRecord) == [record_b, record_c, record_d, record_e, record_f]  # noqa
    assert tracker.history_filter(ExchangeRecord) == [record_e, record_f]
    assert tracker.history_filter(EmbeddingRecord) == []
    assert tracker.calculate_historical(name='does_not_exist') == 0
    assert tracker.calculate_historical(name='cost') == 42
    assert tracker.calculate_historical(name='total_tokens') == 23
    assert tracker.calculate_historical(name='prompt_tokens') == 7
    assert tracker.calculate_historical(name='response_tokens') == 9
    # test calculating historical values based on ExchangeRecords
    assert tracker.calculate_historical(name='cost', record_types=ExchangeRecord) == 33
    assert tracker.calculate_historical(name='total_tokens', record_types=ExchangeRecord) == 15
    assert tracker.calculate_historical(name='prompt_tokens', record_types=ExchangeRecord) == 7
    assert tracker.calculate_historical(name='response_tokens', record_types=ExchangeRecord) == 9

    record_g = EmbeddingRecord(
        cost=1,
        total_tokens=2,
        metadata={'id': 'g'},
    )
    tracker(record_g)
    assert tracker.history == tracker.history_filter()
    assert tracker.history == [record_a, record_b, record_c, record_d, record_e, record_f, record_g]  # noqa
    assert tracker.history_filter(Record) == [record_a, record_b, record_c, record_d, record_e, record_f, record_g]  # noqa
    assert tracker.history_filter(UsageRecord) == [record_b, record_c, record_d, record_e, record_f, record_g]  # noqa
    assert tracker.history_filter(ExchangeRecord) == [record_e, record_f]
    assert tracker.history_filter(EmbeddingRecord) == [record_g]
    assert tracker.calculate_historical(name='does_not_exist') == 0
    assert tracker.calculate_historical(name='cost') == 43
    assert tracker.calculate_historical(name='total_tokens') == 25
    assert tracker.calculate_historical(name='prompt_tokens') == 7
    assert tracker.calculate_historical(name='response_tokens') == 9
    # test calculating historical values based on EmbeddingRecords
    assert tracker.calculate_historical(name='cost', record_types=EmbeddingRecord) == 1
    assert tracker.calculate_historical(name='total_tokens', record_types=EmbeddingRecord) == 2
    assert tracker.calculate_historical(name='prompt_tokens', record_types=EmbeddingRecord) == 0
    assert tracker.calculate_historical(name='response_tokens', record_types=EmbeddingRecord) == 0
    # test calculating historical values based on ExchangeRecords or EmbeddingRecord
    assert tracker.calculate_historical(
            name='cost',
            record_types=(ExchangeRecord, EmbeddingRecord),
        ) == 34
    assert tracker.calculate_historical(
            name='total_tokens',
            record_types=(ExchangeRecord, EmbeddingRecord),
        ) == 17
    assert tracker.calculate_historical(
            name='prompt_tokens',
            record_types=(ExchangeRecord, EmbeddingRecord),
        ) == 7
    assert tracker.calculate_historical(
            name='response_tokens',
            record_types=(ExchangeRecord, EmbeddingRecord),
        ) == 9

def test_history_filter_invalid_value():  # noqa
    with pytest.raises(TypeError):
        MockLink().history_filter(1)

def test_DuckDuckGoSearch():  # noqa
    query = "What is an agent in langchain?"
    search = DuckDuckGoSearch(top_n=1)
    results = search(query=query)
    assert len(results) == 1
    assert 'title' in results[0]
    assert 'href' in results[0]
    assert 'body' in results[0]
    assert len(search.history) == 1
    assert search.history[0].query == query
    assert search.history[0].results == results

    query = "What is langchain?"
    results = search(query=query)
    assert len(results) == 1
    assert 'title' in results[0]
    assert 'href' in results[0]
    assert 'body' in results[0]
    assert len(search.history) == 2
    assert search.history[1].query == query
    assert search.history[1].results == results

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
    assert len(search.history) == 2
    assert search.history[0].query == search.history[1].query
    assert search.history[0].results == search.history[1].results
    assert search.history[0].uuid != search.history[1].uuid

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
    search._history.append(StackOverflowSearchRecord(query=query, questions=fake_results))
    response = search(query)
    assert response == fake_results
    assert len(search.history) == 2
    assert search.history[0].query == search.history[1].query
    assert search.history[0].questions == search.history[1].questions
    assert search.history[0].uuid != search.history[1].uuid

def test__get_stack_overflow_answers_404():  # noqa
     if os.getenv('STACK_OVERFLOW_KEY', None):
        with pytest.raises(RequestError):
            _ = _get_stack_overflow_answers(question_id='asdf')
