"""tests llm_workflow/models.py."""
import pytest
from llm_workflow.base import Document, Record
from llm_workflow.models import (
    EmbeddingModel,
    EmbeddingRecord,
    ExchangeRecord,
    OpenAIChat,
    OpenAIEmbedding,
    StreamingEvent,
    TokenUsageRecord,
)
from llm_workflow.resources import MODEL_COST_PER_TOKEN
from tests.conftest import MockChat, MockRandomEmbeddings


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

def test_OpenAIChat():  # noqa
    model_name = 'gpt-3.5-turbo'
    openai_llm = OpenAIChat(model_name=model_name)
    assert openai_llm.previous_record() is None
    assert openai_llm.previous_prompt is None
    assert openai_llm.previous_response is None
    assert openai_llm.cost == 0
    assert openai_llm.total_tokens == 0
    assert openai_llm.prompt_tokens == 0
    assert openai_llm.response_tokens == 0

    ####
    # first interaction
    ####
    prompt = "This is a question."
    response = openai_llm(prompt)
    assert isinstance(response, str)
    assert len(response) > 1

    # previous memory is the input to ChatGPT
    assert openai_llm._previous_memory[0]['role'] == 'system'
    assert openai_llm._previous_memory[0]['content'] == 'You are a helpful assistant.'
    assert openai_llm._previous_memory[-1]['role'] == 'user'
    assert openai_llm._previous_memory[-1]['content'] == prompt

    assert len(openai_llm.history()) == 1
    message = openai_llm.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata == {'model_name': model_name}
    assert message.cost > 0
    assert message.prompt_tokens > 0
    assert message.response_tokens > 0
    assert message.total_tokens == message.prompt_tokens + message.response_tokens
    assert message.uuid
    assert message.timestamp

    assert openai_llm.previous_prompt == prompt
    assert openai_llm.previous_response == response
    assert openai_llm.cost_per_token == MODEL_COST_PER_TOKEN[model_name]
    assert openai_llm.cost == message.cost
    assert openai_llm.total_tokens == message.total_tokens
    assert openai_llm.prompt_tokens == message.prompt_tokens
    assert openai_llm.response_tokens == message.response_tokens

    previous_prompt = prompt
    previous_response = response
    previous_cost = message.cost
    previous_total_tokens = message.total_tokens
    previous_prompt_tokens = message.prompt_tokens
    previous_response_tokens = message.response_tokens
    previous_message = message

    ####
    # second interaction
    ####
    prompt = "This is another question."
    response = openai_llm(prompt)
    assert isinstance(response, str)
    assert len(response) > 1

    # previous memory is the input to ChatGPT
    assert openai_llm._previous_memory[0]['role'] == 'system'
    assert openai_llm._previous_memory[1]['role'] == 'user'
    assert openai_llm._previous_memory[1]['content'] == previous_prompt
    assert openai_llm._previous_memory[2]['role'] == 'assistant'
    assert openai_llm._previous_memory[2]['content'] == previous_response
    assert openai_llm._previous_memory[3]['role'] == 'user'
    assert openai_llm._previous_memory[3]['content'] == prompt

    assert len(openai_llm.history()) == 2
    message = openai_llm.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata == {'model_name': model_name}
    assert message.cost > 0
    assert message.prompt_tokens > 0
    assert message.response_tokens > 0
    assert message.total_tokens == message.prompt_tokens + message.response_tokens
    assert message.uuid
    assert message.uuid != previous_message.uuid
    assert message.timestamp

    assert openai_llm.previous_prompt == prompt
    assert openai_llm.previous_response == response
    assert openai_llm.cost_per_token == MODEL_COST_PER_TOKEN[model_name]
    assert openai_llm.cost == previous_cost + message.cost
    assert openai_llm.total_tokens == previous_total_tokens + message.total_tokens
    assert openai_llm.prompt_tokens == previous_prompt_tokens + message.prompt_tokens
    assert openai_llm.response_tokens == previous_response_tokens + message.response_tokens

def test_OpenAIChat_streaming():  # noqa
    """Test the same thing as above but for streaming. All usage and history should be the same."""
    callback_response = ''
    def streaming_callback(record: StreamingEvent) -> None:
        nonlocal callback_response
        callback_response += record.response

    model_name = 'gpt-3.5-turbo'
    openai_llm = OpenAIChat(model_name=model_name, streaming_callback=streaming_callback)
    assert openai_llm.previous_record() is None
    assert openai_llm.previous_prompt is None
    assert openai_llm.previous_response is None
    assert openai_llm.cost == 0
    assert openai_llm.total_tokens == 0
    assert openai_llm.prompt_tokens == 0
    assert openai_llm.response_tokens == 0

    ####
    # first interaction
    ####
    prompt = "This is a question."
    response = openai_llm(prompt)
    assert isinstance(response, str)
    assert len(response) > 1
    assert response == callback_response

    # previous memory is the input to ChatGPT
    assert openai_llm._previous_memory[0]['role'] == 'system'
    assert openai_llm._previous_memory[0]['content'] == 'You are a helpful assistant.'
    assert openai_llm._previous_memory[-1]['role'] == 'user'
    assert openai_llm._previous_memory[-1]['content'] == prompt

    assert len(openai_llm.history()) == 1
    message = openai_llm.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata == {'model_name': model_name}
    assert message.cost > 0
    assert message.prompt_tokens > 0
    assert message.response_tokens > 0
    assert message.total_tokens == message.prompt_tokens + message.response_tokens
    assert message.uuid
    assert message.timestamp

    assert openai_llm.previous_prompt == prompt
    assert openai_llm.previous_response == response
    assert openai_llm.cost_per_token == MODEL_COST_PER_TOKEN[model_name]
    assert openai_llm.cost == message.cost
    assert openai_llm.total_tokens == message.total_tokens
    assert openai_llm.prompt_tokens == message.prompt_tokens
    assert openai_llm.response_tokens == message.response_tokens

    previous_prompt = prompt
    previous_response = response
    previous_cost = message.cost
    previous_total_tokens = message.total_tokens
    previous_prompt_tokens = message.prompt_tokens
    previous_response_tokens = message.response_tokens
    previous_message = message

    ####
    # second interaction
    ####
    callback_response = ''
    prompt = "This is another question."
    response = openai_llm(prompt)
    assert isinstance(response, str)
    assert len(response) > 1
    assert response == callback_response

    # previous memory is the input to ChatGPT
    assert openai_llm._previous_memory[0]['role'] == 'system'
    assert openai_llm._previous_memory[1]['role'] == 'user'
    assert openai_llm._previous_memory[1]['content'] == previous_prompt
    assert openai_llm._previous_memory[2]['role'] == 'assistant'
    assert openai_llm._previous_memory[2]['content'] == previous_response
    assert openai_llm._previous_memory[3]['role'] == 'user'
    assert openai_llm._previous_memory[3]['content'] == prompt

    assert len(openai_llm.history()) == 2
    message = openai_llm.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata == {'model_name': model_name}
    assert message.cost > 0
    assert message.prompt_tokens > 0
    assert message.response_tokens > 0
    assert message.total_tokens == message.prompt_tokens + message.response_tokens
    assert message.uuid
    assert message.uuid != previous_message.uuid
    assert message.timestamp

    assert openai_llm.previous_prompt == prompt
    assert openai_llm.previous_response == response
    assert openai_llm.cost_per_token == MODEL_COST_PER_TOKEN[model_name]
    assert openai_llm.cost == previous_cost + message.cost
    assert openai_llm.total_tokens == previous_total_tokens + message.total_tokens
    assert openai_llm.prompt_tokens == previous_prompt_tokens + message.prompt_tokens
    assert openai_llm.response_tokens == previous_response_tokens + message.response_tokens

def test_OpenAIChat_streaming_response_matches_non_streaming():  # noqa
    """
    Test that we get the same final response and usage data when streaming vs not streaming.
    Additionally test that the response we get in the streaming callback matches the overall
    response.
    """
    question = "Explain what a large language model is in a single sentence."
    model_name = 'gpt-3.5-turbo'
    non_streaming_chat = OpenAIChat(
        model_name=model_name,
        temperature=0,
        )
    non_streaming_response = non_streaming_chat(question)

    callback_response = ''
    def streaming_callback(record: StreamingEvent) -> None:
        nonlocal callback_response
        callback_response += record.response

    streaming_chat = OpenAIChat(
        model_name=model_name,
        temperature=0,
        streaming_callback=streaming_callback,
    )
    streaming_response  = streaming_chat(question)
    assert non_streaming_response == streaming_response
    assert non_streaming_response == callback_response
    assert non_streaming_chat.prompt_tokens == streaming_chat.prompt_tokens
    assert non_streaming_chat.response_tokens == streaming_chat.response_tokens
    assert non_streaming_chat.total_tokens == streaming_chat.total_tokens
    assert non_streaming_chat.cost == streaming_chat.cost

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

def test_OpenAIEmbedding():  # noqa
    model = OpenAIEmbedding(model_name='text-embedding-ada-002')
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
    expected_cost = model.history()[0].total_tokens * model.cost_per_token
    assert isinstance(embeddings, list)
    assert isinstance(embeddings[0][0], float)
    assert len(embeddings) == len(docs)
    assert docs[0].content == doc_content_0
    assert docs[1].content == doc_content_1
    assert all(isinstance(x, list) for x in embeddings)
    assert all(len(x) > 100 for x in embeddings)

    assert len(model.history()) == 1
    previous_record = model.history()[0]
    assert isinstance(previous_record, EmbeddingRecord)
    assert previous_record.total_tokens > 0
    assert previous_record.cost == expected_cost
    assert previous_record.uuid
    assert previous_record.timestamp
    assert previous_record.metadata['model_name'] == 'text-embedding-ada-002'

    assert model.total_tokens == previous_record.total_tokens
    assert model.cost == expected_cost

    previous_tokens = model.total_tokens
    previous_cost = model.cost

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
    expected_cost = model.history()[1].total_tokens * model.cost_per_token
    assert isinstance(embeddings, list)
    assert isinstance(embeddings[0][0], float)
    assert len(embeddings) == len(docs)
    assert docs[0].content == doc_content_2
    assert docs[1].content == doc_content_3
    assert all(isinstance(x, list) for x in embeddings)

    assert len(model.history()) == 2
    first_record = model.history()[0]
    assert isinstance(first_record, EmbeddingRecord)
    assert first_record.total_tokens == previous_tokens
    assert first_record.cost == previous_cost
    assert first_record.uuid == previous_record.uuid
    assert first_record.timestamp == previous_record.timestamp
    assert first_record.metadata['model_name'] == 'text-embedding-ada-002'

    previous_record = model.history()[1]
    assert isinstance(previous_record, EmbeddingRecord)
    assert previous_record.total_tokens > 0
    assert previous_record.cost == expected_cost
    assert previous_record.uuid
    assert previous_record.uuid != first_record.uuid
    assert previous_record.timestamp
    assert previous_record.metadata['model_name'] == 'text-embedding-ada-002'

    assert model.total_tokens == previous_tokens + previous_record.total_tokens
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

def test_bug_where_costs_are_incorrect_after_changing_model_name_after_creation():  # noqa
    """We can't set cost_per_token during object creation, because the model might change."""
    model = OpenAIChat(model_name='gpt-3.5-turbo')
    assert model.cost_per_token == MODEL_COST_PER_TOKEN['gpt-3.5-turbo']
    model.model_name = 'gpt-4'
    assert model.cost_per_token == MODEL_COST_PER_TOKEN['gpt-4']

    model = OpenAIEmbedding(model_name='gpt-3.5-turbo')
    assert model.cost_per_token == MODEL_COST_PER_TOKEN['gpt-3.5-turbo']
    model.model_name = 'gpt-4'
    assert model.cost_per_token == MODEL_COST_PER_TOKEN['gpt-4']
