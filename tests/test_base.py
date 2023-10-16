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
from llm_workflow.hugging_face import llama_message_formatter
from tests.conftest import (
    MockChatModel,
    MockCostMemoryManager,
    MockRandomEmbeddings,
    MockPromptModel,
)


def test_MockPromptModel__no_costs():  # noqa
    model = MockPromptModel(
        token_calculator=len,
        cost_calculator=None,
    )
    assert model.previous_record() is None
    assert model.previous_prompt is None
    assert model.previous_response is None
    assert model.cost == 0
    assert model.total_tokens == 0
    assert model.input_tokens == 0
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
    assert message.metadata == {'return_prompt': None}
    assert message.cost is None
    assert message.input_tokens == len(prompt)
    assert message.response_tokens == len(response)
    assert message.total_tokens == len(prompt) + len(response)
    assert message.uuid
    assert message.timestamp

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.cost == 0
    assert model.input_tokens == len(prompt)
    assert model.response_tokens == len(response)
    assert model.total_tokens == len(prompt) + len(response)

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
    assert message.metadata == {'return_prompt': None}
    assert message.response == response
    assert message.cost is None
    assert message.input_tokens == len(prompt)
    assert message.response_tokens == len(response)
    assert message.total_tokens == len(prompt) + len(response)
    assert message.uuid
    assert message.uuid != previous_message.uuid
    assert message.timestamp

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.cost == 0
    assert model.input_tokens == len(prompt) + len(model.history()[0].prompt)
    assert model.response_tokens == len(response) + len(model.history()[0].response)
    assert model.total_tokens == model.input_tokens + model.response_tokens

def test_MockPromptModel__with_costs():  # noqa
    def cost_calc(input_tokens: int, response_tokens: int) -> float:
        return (input_tokens * 1.5) + (response_tokens * 2.5)

    model = MockPromptModel(
        token_calculator=lambda x: len(x) + 1,
        cost_calculator=cost_calc,
    )
    assert model.previous_record() is None
    assert model.previous_prompt is None
    assert model.previous_response is None
    assert model.cost == 0
    assert model.total_tokens == 0
    assert model.input_tokens == 0
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
    assert message.metadata == {'return_prompt': None}
    assert message.input_tokens == len(prompt) + 1
    assert message.response_tokens == len(response) + 1
    assert message.total_tokens == len(prompt) + 1 + len(response) + 1
    assert message.cost == cost_calc(len(prompt) + 1, len(response) + 1)
    assert message.uuid
    assert message.timestamp

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.input_tokens == len(prompt) + 1
    assert model.response_tokens == len(response) + 1
    assert model.total_tokens == len(prompt) + 1 + len(response) + 1
    assert model.cost == cost_calc(len(prompt) + 1, len(response) + 1)

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
    assert message.metadata == {'return_prompt': None}
    assert message.response == response
    assert message.cost == cost_calc(len(prompt) + 1, len(response) + 1)
    assert message.input_tokens == len(prompt) + 1
    assert message.response_tokens == len(response) + 1
    assert message.total_tokens == len(prompt) + 1 + len(response) + 1
    assert message.uuid
    assert message.uuid != previous_message.uuid
    assert message.timestamp

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.input_tokens == len(prompt) + 1 + len(model.history()[0].prompt) + 1
    assert model.response_tokens == len(response) + 1 + len(model.history()[0].response) + 1
    assert model.total_tokens == model.input_tokens + model.response_tokens
    assert model.cost == cost_calc(model.input_tokens, model.response_tokens)

def test_ChatModel__no_costs():  # noqa
    model = MockChatModel(token_calculator=len)
    assert model.previous_record() is None
    assert model.previous_prompt is None
    assert model.previous_response is None
    assert model.cost == 0
    assert model.total_tokens == 0
    assert model.input_tokens == 0
    assert model.response_tokens == 0

    ####
    # first interaction
    ####
    prompt = "This is a question."
    response = model(prompt)
    assert isinstance(response, str)
    assert len(response) > 1

    expected_message_0 = llama_message_formatter(
        system_message=model.system_message,
        history=None,
        prompt=prompt,
    )
    assert expected_message_0.count("<<SYS>>") == 1
    assert expected_message_0.count("<</SYS>>") == 1
    assert expected_message_0.count("[INST]") == 2
    assert expected_message_0.count("[/INST]") == 2
    assert expected_message_0.count(model.system_message) == 1
    assert expected_message_0.count(prompt) == 1
    assert len(model.history()) == 1
    assert len(model.chat_history) == 1
    assert model.history() == model.chat_history
    message = model.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata == {'model_name': 'mock', 'messages': expected_message_0}
    assert message.input_tokens == len(expected_message_0)
    assert message.response_tokens == len(response)
    assert message.total_tokens == len(expected_message_0) + len(response)
    assert message.cost is None
    assert message.uuid
    assert message.timestamp

    assert model._previous_messages == expected_message_0
    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.input_tokens == len(expected_message_0)
    assert model.response_tokens == len(response)
    assert model.total_tokens == len(expected_message_0) + len(response)
    assert model.cost == 0

    ####
    # second interaction
    ####
    previous_message = message
    prompt = "This is another question."
    response = model(prompt)
    assert isinstance(response, str)
    assert len(response) > 1

    expected_message_1 = llama_message_formatter(
        system_message=model.system_message,
        history=[model._history[0]],
        prompt=prompt,
    )
    assert expected_message_1.count("<<SYS>>") == 1
    assert expected_message_1.count("<</SYS>>") == 1
    assert expected_message_1.count("[INST]") == 3
    assert expected_message_1.count("[/INST]") == 3
    assert expected_message_1.count(model.system_message) == 1
    assert expected_message_1.count(model.chat_history[0].prompt) == 1
    assert expected_message_1.count(model.chat_history[0].response) == 1
    assert expected_message_1.count(prompt) == 1
    assert len(model.history()) == 2
    assert len(model.chat_history) == 2
    assert model.history() == model.chat_history
    message = model.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.metadata == {'model_name': 'mock', 'messages': expected_message_1}
    assert message.response == response
    assert message.input_tokens == len(expected_message_1)
    assert message.response_tokens == len(response)
    assert message.total_tokens == len(expected_message_1) + len(response)
    assert message.cost is None
    assert message.uuid
    assert message.uuid != previous_message.uuid
    assert message.timestamp

    assert model._previous_messages == expected_message_1
    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.input_tokens == len(expected_message_0) + len(expected_message_1)
    assert model.response_tokens == len(response) + len(model.history()[0].response)
    assert model.total_tokens == model.input_tokens + model.response_tokens
    assert model.cost == 0

def test_ChatModel__has_token_counter_and_costs():  # noqa
    def cost_calc(input_tokens: int, response_tokens: int) -> float:
        return (input_tokens * 1.5) + (response_tokens * 2.5)

    model = MockChatModel(
        token_calculator=lambda x: len(x),
        cost_calculator=cost_calc,
    )
    assert model.previous_record() is None
    assert model.previous_prompt is None
    assert model.previous_response is None
    assert model.cost == 0
    assert model.total_tokens == 0
    assert model.input_tokens == 0
    assert model.response_tokens == 0

    ####
    # first interaction
    ####
    prompt = "This is a question."
    response = model(prompt)
    assert isinstance(response, str)
    assert len(response) > 1

    expected_message_0 = llama_message_formatter(
        system_message=model.system_message,
        history=None,
        prompt=prompt,
    )

    expected_input_tokens = len(expected_message_0)
    expected_response_tokens = len(response)
    expected_tokens = expected_input_tokens + expected_response_tokens
    expected_costs = cost_calc(expected_input_tokens, expected_response_tokens)

    assert len(model.history()) == 1
    assert len(model.chat_history) == 1
    assert model.history() == model.chat_history
    message = model.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata == {'model_name': 'mock', 'messages': expected_message_0}
    assert message.input_tokens == expected_input_tokens
    assert message.response_tokens == expected_response_tokens
    assert message.total_tokens == expected_input_tokens + expected_response_tokens
    assert message.cost == expected_costs
    assert message.uuid
    assert message.timestamp

    assert model._previous_messages == expected_message_0
    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.input_tokens == expected_input_tokens
    assert model.response_tokens == expected_response_tokens
    assert model.total_tokens == expected_tokens
    assert model.cost == expected_costs

    previous_tokens = expected_tokens
    previous_input_tokens = expected_input_tokens
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

    expected_message_1 = llama_message_formatter(
        system_message=model.system_message,
        history=[model._history[0]],
        prompt=prompt,
    )
    expected_input_tokens = len(expected_message_1)
    expected_response_tokens = len(response)
    expected_tokens = expected_input_tokens + expected_response_tokens
    expected_costs = cost_calc(expected_input_tokens, expected_response_tokens)

    assert len(model.history()) == 2
    assert len(model.chat_history) == 2
    assert model.history() == model.chat_history
    message = model.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata == {'model_name': 'mock', 'messages': expected_message_1}
    assert message.cost == expected_costs
    assert message.total_tokens == expected_tokens
    assert message.input_tokens == expected_input_tokens
    assert message.response_tokens == expected_response_tokens
    assert message.uuid
    assert message.uuid != previous_message.uuid
    assert message.timestamp

    assert model._previous_messages == expected_message_1
    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.cost == expected_costs + previous_costs
    assert model.total_tokens == expected_tokens + previous_tokens
    assert model.input_tokens == expected_input_tokens + previous_input_tokens
    assert model.response_tokens == expected_response_tokens + previous_response_tokens

def test_ChatModel_memory_manager__adds_history():  # noqa
    def cost_calc(input_tokens: int, response_tokens: int) -> float:
        return (input_tokens * 1.5) + (response_tokens * 2.5)

    model = MockChatModel(
        token_calculator=len,
        cost_calculator=cost_calc,
        memory_manager=MockCostMemoryManager(cost=6),
        return_prompt='Model Message: ',
    )
    prompt = 'This is a question.'
    response = model(prompt)
    formatted_message_0 = llama_message_formatter(model.system_message, None, prompt)
    expected_response = f'Model Message: {formatted_message_0}'
    assert response == expected_response.strip()
    assert model.history()[0].metadata == {'model_name': 'memory'}
    assert model.history()[1].metadata['model_name'] == 'mock'
    assert model.history()[1].metadata['messages'] == formatted_message_0
    assert model.chat_history[0].uuid == model.history()[1].uuid

    expected_prompt = 'This is a question.'
    assert model.history()[0].prompt == expected_prompt
    assert model.history()[0].response == formatted_message_0
    assert model.history()[0].input_tokens == len(expected_prompt)
    assert model.history()[0].response_tokens == len(formatted_message_0)
    assert model.history()[0].total_tokens == len(expected_prompt) + len(formatted_message_0)
    assert model.history()[0].cost == 6 * (len(expected_prompt) + len(formatted_message_0))

    assert model.history()[1].prompt == 'This is a question.'
    assert model.history()[1].response == model.previous_response
    assert model.history()[1].response == expected_response.strip()
    assert model.history()[1].input_tokens == len(formatted_message_0)
    # add plus one from stripped \n
    assert model.history()[1].response_tokens == len(model.previous_response) + 1
    assert model.history()[1].total_tokens == len(formatted_message_0) + len(model.previous_response) + 1  # noqa
    assert model.history()[1].cost == cost_calc(len(formatted_message_0), len(model.previous_response) + 1)  # noqa

    assert model.previous_prompt == 'This is a question.'
    assert model.previous_response == model.previous_response
    assert model.cost == model.history()[0].cost + model.history()[1].cost
    assert model.total_tokens == model.history()[0].total_tokens + model.history()[1].total_tokens
    assert model.input_tokens == model.history()[0].input_tokens + model.history()[1].input_tokens
    assert model.response_tokens == model.history()[0].response_tokens + model.history()[1].response_tokens  # noqa
    previous_response = model.previous_response

    prompt = 'This is another question.'
    response = model(prompt)
    formatted_message_1 = llama_message_formatter(model.system_message, [model.chat_history[0]], prompt)  # noqa
    assert response == f'Model Message: {formatted_message_1}'.strip()
    assert model.history()[0].metadata == {'model_name': 'memory'}
    assert model.history()[1].metadata['model_name'] == 'mock'
    assert model.chat_history[0].uuid == model.history()[1].uuid
    assert model.history()[2].metadata == {'model_name': 'memory'}
    assert model.history()[3].metadata['model_name'] == 'mock'
    assert model.chat_history[1].uuid == model.history()[3].uuid

    assert model.history()[0].prompt == expected_prompt
    assert model.history()[0].response == formatted_message_0
    assert model.history()[0].input_tokens == len(expected_prompt)
    assert model.history()[0].response_tokens == len(formatted_message_0)
    assert model.history()[0].total_tokens == len(expected_prompt) + len(formatted_message_0)
    assert model.history()[0].cost == 6 * (len(expected_prompt) + len(formatted_message_0))

    assert model.history()[1].prompt == 'This is a question.'
    assert model.history()[1].response == previous_response
    assert model.history()[1].input_tokens == len(formatted_message_0)
    assert model.history()[1].response_tokens == len(previous_response) + 1
    assert model.history()[1].total_tokens == len(formatted_message_0) + len(previous_response) + 1
    assert model.history()[1].cost == cost_calc(len(formatted_message_0), len(previous_response) + 1)  # noqa

    expected_prompt = 'This is another question.'
    expected_response = 'This is another question.'
    assert model.history()[2].prompt == expected_prompt
    assert model.history()[2].response == formatted_message_1
    assert model.history()[2].input_tokens == len(expected_prompt)
    assert model.history()[2].response_tokens == len(formatted_message_1)
    assert model.history()[2].total_tokens == len(expected_prompt) + len(formatted_message_1)
    assert model.history()[2].cost == 6 * (len(expected_prompt) + len(formatted_message_1))

    assert model.history()[3].prompt == 'This is another question.'
    assert model.history()[3].response == model.previous_response
    assert model.history()[3].input_tokens == len(formatted_message_1)
    assert model.history()[3].response_tokens == len(model.previous_response) + 1
    assert model.history()[3].total_tokens == len(formatted_message_1) + len(model.previous_response) + 1  # noqa
    assert model.history()[3].cost == cost_calc(len(formatted_message_1), len(model.previous_response) + 1)  # noqa

    assert model.previous_prompt == 'This is another question.'
    assert model.previous_response == model.previous_response
    assert model.cost == model.history()[0].cost + model.history()[1].cost + model.history()[2].cost + model.history()[3].cost  # noqa
    assert model.total_tokens == model.history()[0].total_tokens + model.history()[1].total_tokens + model.history()[2].total_tokens + model.history()[3].total_tokens  # noqa
    assert model.input_tokens == model.history()[0].input_tokens + model.history()[1].input_tokens + model.history()[2].input_tokens + model.history()[3].input_tokens  # noqa
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
