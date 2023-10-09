"""tests llm_workflow/models.py."""
import pytest
from llm_workflow.base import (
    Document,
    Record,
    EmbeddingModel,
    EmbeddingRecord,
    ExchangeRecord,
    TokenUsageRecord,
)
from tests.conftest import MockChat, MockRandomEmbeddings, MockCostMemoryManager


def test_ChatModel__no_token_counter_or_costs():  # noqa
    model = MockChat(token_counter=None, cost_per_token=None)
    assert model.previous_record() is None
    assert model.previous_prompt is None
    assert model.previous_response is None
    assert model.cost == 0
    assert model.total_tokens == 0
    assert model.prompt_tokens == 0
    assert model.response_tokens == 0

    ####
    # first interaction
    ####
    prompt = "This is a question."
    response = model(prompt)
    assert isinstance(response, str)
    assert len(response) > 1

    assert len(model.history()) == 1
    message = model.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata == {'model_name': 'mock'}
    assert message.cost is None
    assert message.total_tokens is None
    assert message.prompt_tokens is None
    assert message.response_tokens is None
    assert message.uuid
    assert message.timestamp

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.cost_per_token is None
    assert model.token_counter is None
    assert model.cost == 0
    assert model.total_tokens == 0
    assert model.prompt_tokens == 0
    assert model.response_tokens == 0

    ####
    # second interaction
    ####
    previous_message = message
    prompt = "This is another question."
    response = model(prompt)
    assert isinstance(response, str)
    assert len(response) > 1

    assert len(model.history()) == 2
    message = model.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.metadata == {'model_name': 'mock'}
    assert message.response == response
    assert message.cost is None
    assert message.total_tokens is None
    assert message.prompt_tokens is None
    assert message.response_tokens is None
    assert message.uuid
    assert message.uuid != previous_message.uuid
    assert message.timestamp

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.cost_per_token is None
    assert model.token_counter is None
    assert model.cost == 0
    assert model.total_tokens == 0
    assert model.prompt_tokens == 0
    assert model.response_tokens == 0

def test_ChatModel__has_token_counter_and_costs():  # noqa
    token_counter = len
    cost_per_token = 3
    model = MockChat(token_counter=token_counter, cost_per_token=cost_per_token)
    assert model.previous_record() is None
    assert model.previous_prompt is None
    assert model.previous_response is None
    assert model.cost == 0
    assert model.total_tokens == 0
    assert model.prompt_tokens == 0
    assert model.response_tokens == 0

    ####
    # first interaction
    ####
    prompt = "This is a question."
    response = model(prompt)
    assert isinstance(response, str)
    assert len(response) > 1

    expected_prompt_tokens = token_counter(prompt)
    expected_response_tokens = token_counter(response)
    expected_tokens = expected_prompt_tokens + expected_response_tokens
    expected_costs = expected_tokens * cost_per_token

    assert len(model.history()) == 1
    message = model.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.cost == expected_costs
    assert message.total_tokens == expected_tokens
    assert message.prompt_tokens == expected_prompt_tokens
    assert message.response_tokens == expected_response_tokens
    assert message.uuid
    assert message.timestamp

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.token_counter is token_counter
    assert model.cost_per_token == cost_per_token
    assert model.cost == expected_costs
    assert model.total_tokens == expected_tokens
    assert model.prompt_tokens == expected_prompt_tokens
    assert model.response_tokens == expected_response_tokens

    previous_tokens = expected_tokens
    previous_prompt_tokens = expected_prompt_tokens
    previous_response_tokens = expected_response_tokens
    previous_costs = expected_costs
    previous_message = message

    ####
    # second interaction
    ####
    prompt = "This is another question."
    response = model(prompt)
    assert isinstance(response, str)
    assert len(response) > 1

    expected_prompt_tokens = token_counter(prompt)
    expected_response_tokens = token_counter(response)
    expected_tokens = expected_prompt_tokens + expected_response_tokens
    expected_costs = expected_tokens * cost_per_token

    assert len(model.history()) == 2
    message = model.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.cost == expected_costs
    assert message.total_tokens == expected_tokens
    assert message.prompt_tokens == expected_prompt_tokens
    assert message.response_tokens == expected_response_tokens
    assert message.uuid
    assert message.uuid != previous_message.uuid
    assert message.timestamp

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.token_counter is token_counter
    assert model.cost_per_token == cost_per_token
    assert model.cost == expected_costs + previous_costs
    assert model.total_tokens == expected_tokens + previous_tokens
    assert model.prompt_tokens == expected_prompt_tokens + previous_prompt_tokens
    assert model.response_tokens == expected_response_tokens + previous_response_tokens

def test_ChatModel_memory_manager__adds_history():  # noqa
    model = MockChat(
        token_counter=len,
        cost_per_token=3,
        memory_manager=MockCostMemoryManager(cost=6),
    )
    model('This is a question.')
    assert model.history()[0].metadata == {'model_name': 'memory'}
    assert model.history()[1].metadata == {'model_name': 'mock'}
    assert model.chat_history[0].uuid == model.history()[1].uuid

    expected_prompt = 'Memory Prompt: This is a question.'
    expected_response = 'Memory Response: This is a question.'
    assert model.history()[0].prompt == expected_prompt
    assert model.history()[0].response == expected_response
    assert model.history()[0].prompt_tokens == len(expected_prompt)
    assert model.history()[0].response_tokens == len(expected_response)
    assert model.history()[0].total_tokens == len(expected_prompt) + len(expected_response)
    assert model.history()[0].cost == 6 * (len(expected_prompt) + len(expected_response))

    assert model.history()[1].prompt == 'This is a question.'
    assert model.history()[1].response == model.previous_response
    assert model.history()[1].prompt_tokens == len('This is a question.')
    assert model.history()[1].response_tokens == len(model.previous_response)
    assert model.history()[1].total_tokens == len('This is a question.') + len(model.previous_response)  # noqa
    assert model.history()[1].cost == 3 * (len('This is a question.') + len(model.previous_response))  # noqa

    assert model.previous_prompt == 'This is a question.'
    assert model.previous_response == model.previous_response
    assert model.cost == model.history()[0].cost + model.history()[1].cost
    assert model.total_tokens == model.history()[0].total_tokens + model.history()[1].total_tokens
    assert model.prompt_tokens == model.history()[0].prompt_tokens + model.history()[1].prompt_tokens  # noqa
    assert model.response_tokens == model.history()[0].response_tokens + model.history()[1].response_tokens  # noqa
    previous_response = model.previous_response

    model('This is another question.')
    assert model.history()[0].metadata == {'model_name': 'memory'}
    assert model.history()[1].metadata == {'model_name': 'mock'}
    assert model.chat_history[0].uuid == model.history()[1].uuid
    assert model.history()[2].metadata == {'model_name': 'memory'}
    assert model.history()[3].metadata == {'model_name': 'mock'}
    assert model.chat_history[1].uuid == model.history()[3].uuid

    assert model.history()[0].prompt == expected_prompt
    assert model.history()[0].response == expected_response
    assert model.history()[0].prompt_tokens == len(expected_prompt)
    assert model.history()[0].response_tokens == len(expected_response)
    assert model.history()[0].total_tokens == len(expected_prompt) + len(expected_response)
    assert model.history()[0].cost == 6 * (len(expected_prompt) + len(expected_response))

    assert model.history()[1].prompt == 'This is a question.'
    assert model.history()[1].response == previous_response
    assert model.history()[1].prompt_tokens == len('This is a question.')
    assert model.history()[1].response_tokens == len(previous_response)
    assert model.history()[1].total_tokens == len('This is a question.') + len(previous_response)
    assert model.history()[1].cost == 3 * (len('This is a question.') + len(previous_response))

    expected_prompt = 'Memory Prompt: This is another question.'
    expected_response = 'Memory Response: This is another question.'
    assert model.history()[2].prompt == expected_prompt
    assert model.history()[2].response == expected_response
    assert model.history()[2].prompt_tokens == len(expected_prompt)
    assert model.history()[2].response_tokens == len(expected_response)
    assert model.history()[2].total_tokens == len(expected_prompt) + len(expected_response)
    assert model.history()[2].cost == 6 * (len(expected_prompt) + len(expected_response))

    assert model.history()[3].prompt == 'This is another question.'
    assert model.history()[3].response == model.previous_response
    assert model.history()[3].prompt_tokens == len('This is another question.')
    assert model.history()[3].response_tokens == len(model.previous_response)
    assert model.history()[3].total_tokens == len('This is another question.') + len(model.previous_response)  # noqa
    assert model.history()[3].cost == 3 * (len('This is another question.') + len(model.previous_response))  # noqa

    assert model.previous_prompt == 'This is another question.'
    assert model.previous_response == model.previous_response
    assert model.cost == model.history()[0].cost + model.history()[1].cost + model.history()[2].cost + model.history()[3].cost  # noqa
    assert model.total_tokens == model.history()[0].total_tokens + model.history()[1].total_tokens + model.history()[2].total_tokens + model.history()[3].total_tokens  # noqa
    assert model.prompt_tokens == model.history()[0].prompt_tokens + model.history()[1].prompt_tokens + model.history()[2].prompt_tokens + model.history()[3].prompt_tokens  # noqa
    assert model.response_tokens == model.history()[0].response_tokens + model.history()[1].response_tokens + model.history()[2].response_tokens + model.history()[3].response_tokens  # noqa

def test_EmbeddingModel__called_with_different_types():  # noqa
    class MockEmbeddings(EmbeddingModel):
        def _run(self, docs: list[Document]) -> tuple[list[list[float]], EmbeddingRecord]:
            return docs, EmbeddingRecord(metadata={'content': docs})

    embeddings = MockEmbeddings()
    assert embeddings(None) == []
    assert embeddings([]) == []
    with pytest.raises(TypeError):
        embeddings(1)

    value = 'string value'
    expected_value = [Document(content=value)]
    result = embeddings(docs=value)
    assert result == expected_value
    assert embeddings.history()[0].metadata == {'content': expected_value}

    value = 'Document value'
    expected_value = [Document(content=value)]
    result = embeddings(docs=Document(content=value))
    assert result == expected_value
    assert embeddings.history()[1].metadata == {'content': expected_value}

    value = ['string value 1', 'string value 2']
    expected_value = [Document(content=x) for x in value]
    result = embeddings(docs=value)
    assert result == expected_value
    assert embeddings.history()[2].metadata == {'content': expected_value}

    value = [Document(content='document value 1'), Document(content='document value 2')]
    expected_value = value
    result = embeddings(docs=value)
    assert result == expected_value
    assert embeddings.history()[3].metadata == {'content': expected_value}

def test_EmbeddingModel__no_costs():  # noqa
    model = MockRandomEmbeddings(token_counter=len, cost_per_token=None)
    assert model.cost == 0
    assert model.total_tokens == 0

    ####
    # first interaction
    ####
    doc_content_0 = "This is a doc."
    doc_content_1 = "This is a another doc."
    docs = [
        Document(content=doc_content_0),
        Document(content=doc_content_1),
    ]
    embeddings = model(docs)
    expected_tokens = sum(len(x.content) for x in docs)
    assert isinstance(embeddings, list)
    assert isinstance(embeddings[0][0], float)
    assert all(isinstance(x, list) for x in embeddings)
    assert len(embeddings) == len(docs)
    assert docs[0].content == doc_content_0
    assert docs[1].content == doc_content_1

    assert len(model.history()) == 1
    first_record = model.history()[0]
    assert isinstance(first_record, EmbeddingRecord)
    assert first_record.total_tokens == expected_tokens
    assert first_record.cost is None
    assert first_record.uuid
    assert first_record.timestamp

    assert model.total_tokens == expected_tokens
    assert model.cost == 0

    previous_tokens = model.total_tokens
    previous_record = first_record

    ####
    # second interaction
    ####
    doc_content_2 = "This is a doc for a second call."
    doc_content_3 = "This is a another doc for a second call."
    docs = [
        Document(content=doc_content_2),
        Document(content=doc_content_3),
    ]
    embeddings = model(docs)
    expected_tokens = sum(len(x.content) for x in docs)
    assert isinstance(embeddings, list)
    assert isinstance(embeddings[0][0], float)
    assert all(isinstance(x, list) for x in embeddings)
    assert len(embeddings) == len(docs)
    assert docs[0].content == doc_content_2
    assert docs[1].content == doc_content_3

    assert len(model.history()) == 2
    first_record = model.history()[0]
    assert isinstance(first_record, EmbeddingRecord)
    assert first_record.total_tokens == previous_tokens
    assert first_record.cost is None
    assert first_record.uuid
    assert first_record.uuid == previous_record.uuid
    assert first_record.timestamp

    second_record = model.history()[1]
    assert isinstance(second_record, EmbeddingRecord)
    assert second_record.total_tokens == expected_tokens
    assert second_record.cost is None
    assert second_record.uuid
    assert second_record.uuid != previous_record.uuid
    assert second_record.timestamp

    assert model.total_tokens == previous_tokens + expected_tokens
    assert model.cost == 0

def test_EmbeddingModel__with_costs():  # noqa
    cost_per_token = 3
    model = MockRandomEmbeddings(token_counter=len, cost_per_token=cost_per_token)
    assert model.cost == 0
    assert model.total_tokens == 0

    ####
    # first interaction
    ####
    doc_content_0 = "This is a doc."
    doc_content_1 = "This is a another doc."
    docs = [
        Document(content=doc_content_0),
        Document(content=doc_content_1),
    ]
    embeddings = model(docs)
    expected_tokens = sum(len(x.content) for x in docs)
    expected_cost = expected_tokens * cost_per_token
    assert isinstance(embeddings, list)
    assert isinstance(embeddings[0][0], float)
    assert all(isinstance(x, list) for x in embeddings)

    assert len(embeddings) == len(docs)
    assert docs[0].content == doc_content_0
    assert docs[1].content == doc_content_1

    assert len(model.history()) == 1
    first_record = model.history()[0]
    assert isinstance(first_record, EmbeddingRecord)
    assert first_record.total_tokens == expected_tokens
    assert first_record.cost == expected_cost
    assert first_record.uuid
    assert first_record.timestamp

    assert model.total_tokens == expected_tokens
    assert model.cost == expected_cost

    previous_tokens = model.total_tokens
    previous_cost = model.cost
    previous_record = first_record

    ####
    # second interaction
    ####
    doc_content_2 = "This is a doc for a second call."
    doc_content_3 = "This is a another doc for a second call."
    docs = [
        Document(content=doc_content_2),
        Document(content=doc_content_3),
    ]
    embeddings = model(docs)
    expected_tokens = sum(len(x.content) for x in docs)
    expected_cost = expected_tokens * cost_per_token
    assert isinstance(embeddings, list)
    assert isinstance(embeddings[0][0], float)
    assert all(isinstance(x, list) for x in embeddings)

    assert len(embeddings) == len(docs)
    assert docs[0].content == doc_content_2
    assert docs[1].content == doc_content_3

    assert len(model.history()) == 2
    first_record = model.history()[0]
    assert isinstance(first_record, EmbeddingRecord)
    assert first_record.total_tokens == previous_tokens
    assert first_record.cost == previous_cost
    assert first_record.uuid
    assert first_record.uuid == previous_record.uuid
    assert first_record.timestamp

    second_record = model.history()[1]
    assert isinstance(second_record, EmbeddingRecord)
    assert second_record.total_tokens == expected_tokens
    assert second_record.cost == expected_cost
    assert second_record.uuid
    assert second_record.uuid != previous_record.uuid
    assert second_record.timestamp

    assert model.total_tokens == previous_tokens + expected_tokens
    assert model.cost == previous_cost + expected_cost

def test_Records_to_string():  # noqa
    assert 'timestamp: ' in str(Record())
    assert 'uuid: ' in str(Record())

    assert 'timestamp: ' in str(TokenUsageRecord())
    assert 'uuid: ' in str(TokenUsageRecord())
    assert 'cost: ' in str(TokenUsageRecord())
    assert 'total_tokens: ' in str(TokenUsageRecord())

    assert 'timestamp: ' in str(TokenUsageRecord(total_tokens=1000, cost=1.5))
    assert 'uuid: ' in str(TokenUsageRecord(total_tokens=1000, cost=1.5))
    assert 'cost: $1.5' in str(TokenUsageRecord(total_tokens=1000, cost=1.5))
    assert 'total_tokens: 1,000' in str(TokenUsageRecord(total_tokens=1000, cost=1.5))

    assert 'timestamp: ' in str(ExchangeRecord(prompt='prompt', response='response'))
    assert 'prompt: "prompt' in str(ExchangeRecord(prompt='prompt', response='response'))
    assert 'response: "response' in str(ExchangeRecord(prompt='prompt', response='response'))
    assert 'cost: ' in str(ExchangeRecord(prompt='prompt', response='response'))
    assert 'total_tokens: ' in str(ExchangeRecord(prompt='prompt', response='response'))

    assert 'timestamp: ' in str(ExchangeRecord(prompt='prompt', response='response', total_tokens=1000, cost=1.5))  # noqa
    assert 'prompt: "prompt' in str(ExchangeRecord(prompt='prompt', response='response', total_tokens=1000, cost=1.5))  # noqa
    assert 'response: "response' in str(ExchangeRecord(prompt='prompt', response='response', total_tokens=1000, cost=1.5))  # noqa
    assert 'cost: $1.5' in str(ExchangeRecord(prompt='prompt', response='response', total_tokens=1000, cost=1.5))  # noqa
    assert 'total_tokens: 1,000' in str(ExchangeRecord(prompt='prompt', response='response', total_tokens=1000, cost=1.5))  # noqa
