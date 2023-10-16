"""Test Session class."""
from time import sleep
import pytest
from llm_workflow.base import (
    EmbeddingRecord,
    ExchangeRecord,
    LanguageModel,
    Record,
    Session,
    TokenUsageRecord,
    Workflow,
)


class MockHistoricalUsageRecords(LanguageModel):
    """Object used to Mock a model used in a task."""

    def __init__(self, mock_id: str) -> None:
        super().__init__()
        # mock_id is used to tset that the correct workflow is called
        self.mock_id = mock_id
        self.records = []

    def __call__(self, record: Record) -> Record:  # noqa
        self.records.append(record)
        return record, self.mock_id

    def _get_history(self) -> list[TokenUsageRecord]:
        return self.records

def test_Session():  # noqa
    session = Session()
    with pytest.raises(ValueError):  # noqa: PT011
        session('test')
    assert session.history() == []
    assert session.history(TokenUsageRecord) == []
    assert session.history(ExchangeRecord) == []
    assert session.history(EmbeddingRecord) == []
    assert session.sum('cost') == 0
    assert session.sum('total_tokens') == 0
    assert session.sum('total_tokens', EmbeddingRecord) == 0
    assert session.sum('input_tokens') == 0
    assert session.sum('response_tokens') == 0
    assert len(session) == 0

    session.append(workflow=Workflow(tasks=[]))
    assert session('test') is None
    assert session.history() == []
    assert session.history(TokenUsageRecord) == []
    assert session.history(ExchangeRecord) == []
    assert session.history(EmbeddingRecord) == []
    assert session.sum('cost') == 0
    assert session.sum('total_tokens') == 0
    assert session.sum('total_tokens', EmbeddingRecord) == 0
    assert session.sum('input_tokens') == 0
    assert session.sum('response_tokens') == 0
    assert len(session) == 1

    # test workflow with a task that doesn't have a history property
    session.append(workflow=Workflow(tasks=[lambda x: x]))
    assert session('test') == 'test'
    assert session.history() == []
    assert session.history(TokenUsageRecord) == []
    assert session.history(ExchangeRecord) == []
    assert session.history(EmbeddingRecord) == []
    assert session.sum('cost') == 0
    assert session.sum('total_tokens') == 0
    assert session.sum('total_tokens', EmbeddingRecord) == 0
    assert session.sum('input_tokens') == 0
    assert session.sum('response_tokens') == 0
    assert len(session) == 2

    record_a = TokenUsageRecord(metadata={'id': 'record_a'}, total_tokens=None, cost=None)
    sleep(0.001)
    record_b = TokenUsageRecord(metadata={'id': 'record_b'}, total_tokens=100, cost=0.01)
    sleep(0.001)
    record_c = Record(metadata={'id': 'record_d'})
    sleep(0.001)
    record_d = Record(metadata={'id': 'record_e'})
    sleep(0.001)
    record_e = ExchangeRecord(
        metadata={'id': 'record_e'},
        prompt='prompt',
        response='response',
        cost=0.5,
        total_tokens=103,
        input_tokens=34,
        response_tokens=53,
    )
    record_f = EmbeddingRecord(
        metadata={'id': 'record_f'},
        cost=0.7,
        total_tokens=1_002,
    )

    session.append(workflow=Workflow(tasks=[MockHistoricalUsageRecords(mock_id='mock_a')]))
    return_value, mock_id = session(record_a)
    assert return_value == record_a
    assert mock_id == 'mock_a'
    assert session.history() == [record_a]
    assert session.history(TokenUsageRecord) == [record_a]
    assert session.history(ExchangeRecord) == []
    assert session.history(EmbeddingRecord) == []
    assert session.sum('cost') == 0
    assert session.sum('total_tokens') == 0
    assert session.sum('total_tokens', EmbeddingRecord) == 0
    assert session.sum('input_tokens') == 0
    assert session.sum('response_tokens') == 0
    assert len(session) == 3

    # if we add the same record it should be ignored
    return_value, mock_id = session(record_a)
    assert return_value == record_a
    assert mock_id == 'mock_a'
    assert session.history() == [record_a]
    assert session.history(TokenUsageRecord) == [record_a]
    assert session.history(ExchangeRecord) == []
    assert session.history(EmbeddingRecord) == []
    assert session.sum('cost') == 0
    assert session.sum('total_tokens') == 0
    assert session.sum('total_tokens', EmbeddingRecord) == 0
    assert session.sum('input_tokens') == 0
    assert session.sum('response_tokens') == 0
    assert len(session) == 3

    return_value, mock_id = session(record_b)
    assert return_value == record_b
    assert mock_id == 'mock_a'
    assert session.history() == [record_a, record_b]
    assert session.history(TokenUsageRecord) == [record_a, record_b]
    assert session.history(ExchangeRecord) == []
    assert session.history(EmbeddingRecord) == []
    assert session.sum('cost') == 0.01
    assert session.sum('total_tokens') == 100
    assert session.sum('total_tokens', EmbeddingRecord) == 0
    assert session.sum('input_tokens') == 0
    assert session.sum('response_tokens') == 0
    assert len(session) == 3

    # add record `e` out of order; later, ensure the correct order is returned
    session.append(workflow=Workflow(tasks=[MockHistoricalUsageRecords(mock_id='mock_b')]))
    return_value, mock_id = session(record_e)
    assert return_value == record_e
    assert mock_id == 'mock_b'
    assert session.history() == [record_a, record_b, record_e]
    assert session.history(TokenUsageRecord) == [record_a, record_b, record_e]
    assert session.history(ExchangeRecord) == [record_e]
    assert session.history(EmbeddingRecord) == []
    assert session.sum('cost') == 0.51
    assert session.sum('total_tokens') == 203
    assert session.sum('total_tokens', EmbeddingRecord) == 0
    assert session.sum('input_tokens') == 34
    assert session.sum('response_tokens') == 53
    assert len(session) == 4

    # adding the same record to a new task should not double-count
    return_value, mock_id = session(record_b)
    assert return_value == record_b
    assert mock_id == 'mock_b'
    assert session.history() == [record_a, record_b, record_e]
    assert session.history(TokenUsageRecord) == [record_a, record_b, record_e]
    assert session.history(ExchangeRecord) == [record_e]
    assert session.history(EmbeddingRecord) == []
    assert session.sum('cost') == 0.51
    assert session.sum('total_tokens') == 203
    assert session.sum('total_tokens', EmbeddingRecord) == 0
    assert session.sum('input_tokens') == 34
    assert session.sum('response_tokens') == 53
    assert len(session) == 4

    # add record `d` out of order; later, ensure the correct order is returned
    return_value, mock_id = session(record_d)
    assert return_value == record_d
    assert mock_id == 'mock_b'
    assert session.history() == [record_a, record_b, record_d, record_e]
    assert session.history(TokenUsageRecord) == [record_a, record_b, record_e]
    assert session.history(ExchangeRecord) == [record_e]
    assert session.history(EmbeddingRecord) == []
    assert session.sum('cost') == 0.51
    assert session.sum('total_tokens') == 203
    assert session.sum('total_tokens', EmbeddingRecord) == 0
    assert session.sum('input_tokens') == 34
    assert session.sum('response_tokens') == 53
    assert len(session) == 4

    # add record `c` out of order; c should be returned before d
    return_value, mock_id = session(record_c)
    assert return_value == record_c
    assert mock_id == 'mock_b'
    assert session.history() == [record_a, record_b, record_c, record_d, record_e]
    assert session.history(TokenUsageRecord) == [record_a, record_b, record_e]
    assert session.history(ExchangeRecord) == [record_e]
    assert session.history(EmbeddingRecord) == []
    assert session.sum('cost') == 0.51
    assert session.sum('total_tokens') == 203
    assert session.sum('total_tokens', EmbeddingRecord) == 0
    assert session.sum('input_tokens') == 34
    assert session.sum('response_tokens') == 53
    assert len(session) == 4

    # test EmbeddingRecord
    return_value, mock_id = session(record_f)
    assert return_value == record_f
    assert mock_id == 'mock_b'
    assert session.history() == [record_a, record_b, record_c, record_d, record_e, record_f]
    assert session.history(TokenUsageRecord) == [record_a, record_b, record_e, record_f]
    assert session.history(ExchangeRecord) == [record_e]
    assert session.history(EmbeddingRecord) == [record_f]
    assert session.sum('cost') == 0.51 + 0.7
    assert session.sum('total_tokens') == 203 + 1_002
    assert session.sum('total_tokens', EmbeddingRecord) == 1_002
    assert session.sum('input_tokens') == 34
    assert session.sum('response_tokens') == 53
    assert len(session) == 4
