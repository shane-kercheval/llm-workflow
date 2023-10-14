"""Test history."""
import pytest
from llm_workflow.base import (
    RecordKeeper,
    Record,
    EmbeddingRecord,
    ExchangeRecord,
    TokenUsageRecord,
)


class Mocktask(RecordKeeper):
    """Mocks a Task object."""

    def __init__(self) -> None:
        self._history = []

    def __call__(self, record: Record) -> None:
        """Adds the record to the history."""
        return self._history.append(record)

    def _get_history(self) -> list[Record]:
        """Returns history."""
        return self._history


def test_history_tracker():  # noqa
    tracker = Mocktask()
    assert tracker.history() == []
    assert tracker.history(Record) == []
    assert tracker.history(TokenUsageRecord) == []
    assert tracker.history(ExchangeRecord) == []
    assert tracker.history(EmbeddingRecord) == []
    assert tracker.sum(name='does_not_exist') == 0
    assert tracker.sum(name='cost') == 0
    assert tracker.sum(name='total_tokens') == 0
    assert tracker.sum(name='input_tokens') == 0
    assert tracker.sum(name='response_tokens') == 0

    record_a = Record(metadata={'id': 'a'})
    tracker(record_a)
    assert tracker.history() == [record_a]
    assert tracker.history(Record) == [record_a]
    assert tracker.history(TokenUsageRecord) == []
    assert tracker.history(ExchangeRecord) == []
    assert tracker.history(EmbeddingRecord) == []
    assert tracker.sum(name='does_not_exist') == 0
    assert tracker.sum(name='cost') == 0
    assert tracker.sum(name='total_tokens') == 0
    assert tracker.sum(name='input_tokens') == 0
    assert tracker.sum(name='response_tokens') == 0

    record_b = TokenUsageRecord(total_tokens=1, metadata={'id': 'b'})
    tracker(record_b)
    assert tracker.history() == [record_a, record_b]
    assert tracker.history(Record) == [record_a, record_b]
    assert tracker.history(TokenUsageRecord) == [record_b]
    assert tracker.history(ExchangeRecord) == []
    assert tracker.history(EmbeddingRecord) == []
    assert tracker.sum(name='does_not_exist') == 0
    assert tracker.sum(name='cost') == 0
    assert tracker.sum(name='total_tokens') == 1
    assert tracker.sum(name='input_tokens') == 0
    assert tracker.sum(name='response_tokens') == 0

    record_c = TokenUsageRecord(cost=3, metadata={'id': 'c'})
    tracker(record_c)
    assert tracker.history() == [record_a, record_b, record_c]
    assert tracker.history(Record) == [record_a, record_b, record_c]
    assert tracker.history(TokenUsageRecord) == [record_b, record_c]
    assert tracker.history(ExchangeRecord) == []
    assert tracker.history(EmbeddingRecord) == []
    assert tracker.sum(name='does_not_exist') == 0
    assert tracker.sum(name='cost') == 3
    assert tracker.sum(name='total_tokens') == 1
    assert tracker.sum(name='input_tokens') == 0
    assert tracker.sum(name='response_tokens') == 0

    record_d = TokenUsageRecord(total_tokens=7, cost=6, metadata={'id': 'd'})
    tracker(record_d)
    assert tracker.history() == [record_a, record_b, record_c, record_d]
    assert tracker.history(Record) == [record_a, record_b, record_c, record_d]
    assert tracker.history(TokenUsageRecord) == [record_b, record_c, record_d]
    assert tracker.history(ExchangeRecord) == []
    assert tracker.history(EmbeddingRecord) == []
    assert tracker.sum(name='does_not_exist') == 0
    assert tracker.sum(name='cost') == 9
    assert tracker.sum(name='total_tokens') == 8
    assert tracker.sum(name='input_tokens') == 0
    assert tracker.sum(name='response_tokens') == 0

    record_e = ExchangeRecord(
        prompt="the prompt",
        response="the response",
        cost=20,
        total_tokens=10,
        metadata={'id': 'e'},
    )
    tracker(record_e)
    assert tracker.history() == [record_a, record_b, record_c, record_d, record_e]
    assert tracker.history(Record) == [record_a, record_b, record_c, record_d, record_e]
    assert tracker.history(TokenUsageRecord) == [record_b, record_c, record_d, record_e]
    assert tracker.history(ExchangeRecord) == [record_e]
    assert tracker.history(EmbeddingRecord) == []
    assert tracker.sum(name='does_not_exist') == 0
    assert tracker.sum(name='cost') == 29
    assert tracker.sum(name='total_tokens') == 18
    assert tracker.sum(name='input_tokens') == 0
    assert tracker.sum(name='response_tokens') == 0
    # test calculating historical values based on ExchangeRecords
    assert tracker.sum(name='cost', types=ExchangeRecord) == 20
    assert tracker.sum(name='total_tokens', types=ExchangeRecord) == 10
    assert tracker.sum(name='input_tokens', types=ExchangeRecord) == 0
    assert tracker.sum(name='response_tokens', types=ExchangeRecord) == 0

    record_f = ExchangeRecord(
        prompt="the prompt",
        response="the response",
        cost=13,
        total_tokens=5,
        input_tokens=7,
        response_tokens=9,
        metadata={'id': 'f'},
    )
    tracker(record_f)
    assert tracker.history() == [record_a, record_b, record_c, record_d, record_e, record_f]
    assert tracker.history(Record) == [record_a, record_b, record_c, record_d, record_e, record_f]
    assert tracker.history(TokenUsageRecord) == [record_b, record_c, record_d, record_e, record_f]
    assert tracker.history(ExchangeRecord) == [record_e, record_f]
    assert tracker.history(EmbeddingRecord) == []
    assert tracker.sum(name='does_not_exist') == 0
    assert tracker.sum(name='cost') == 42
    assert tracker.sum(name='total_tokens') == 23
    assert tracker.sum(name='input_tokens') == 7
    assert tracker.sum(name='response_tokens') == 9
    # test calculating historical values based on ExchangeRecords
    assert tracker.sum(name='cost', types=ExchangeRecord) == 33
    assert tracker.sum(name='total_tokens', types=ExchangeRecord) == 15
    assert tracker.sum(name='input_tokens', types=ExchangeRecord) == 7
    assert tracker.sum(name='response_tokens', types=ExchangeRecord) == 9

    record_g = EmbeddingRecord(
        cost=1,
        total_tokens=2,
        metadata={'id': 'g'},
    )
    tracker(record_g)
    assert tracker.history() == [record_a, record_b, record_c, record_d, record_e, record_f, record_g]  # noqa
    assert tracker.history(Record) == [record_a, record_b, record_c, record_d, record_e, record_f, record_g]  # noqa
    assert tracker.history(TokenUsageRecord) == [record_b, record_c, record_d, record_e, record_f, record_g]  # noqa
    assert tracker.history(ExchangeRecord) == [record_e, record_f]
    assert tracker.history(EmbeddingRecord) == [record_g]
    assert tracker.sum(name='does_not_exist') == 0
    assert tracker.sum(name='cost') == 43
    assert tracker.sum(name='total_tokens') == 25
    assert tracker.sum(name='input_tokens') == 7
    assert tracker.sum(name='response_tokens') == 9
    # test calculating historical values based on EmbeddingRecords
    assert tracker.sum(name='cost', types=EmbeddingRecord) == 1
    assert tracker.sum(name='total_tokens', types=EmbeddingRecord) == 2
    assert tracker.sum(name='input_tokens', types=EmbeddingRecord) == 0
    assert tracker.sum(name='response_tokens', types=EmbeddingRecord) == 0
    # test calculating historical values based on ExchangeRecords or EmbeddingRecord
    assert tracker.sum(
            name='cost',
            types=(ExchangeRecord, EmbeddingRecord),
        ) == 34
    assert tracker.sum(
            name='total_tokens',
            types=(ExchangeRecord, EmbeddingRecord),
        ) == 17
    assert tracker.sum(
            name='input_tokens',
            types=(ExchangeRecord, EmbeddingRecord),
        ) == 7
    assert tracker.sum(
            name='response_tokens',
            types=(ExchangeRecord, EmbeddingRecord),
        ) == 9

def test_history_filter_invalid_value():  # noqa
    with pytest.raises(TypeError):
        Mocktask().history(1)
