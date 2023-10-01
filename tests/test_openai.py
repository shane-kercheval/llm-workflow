"""Test the OpenAI Model classes."""

import openai
import pytest
from llm_workflow.base import Document
from llm_workflow.models import EmbeddingRecord, ExchangeRecord, StreamingEvent
from llm_workflow.openai import OpenAIChat, OpenAIEmbedding, num_tokens, num_tokens_from_messages
from llm_workflow.resources import MODEL_COST_PER_TOKEN



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


def test_OpenAIChat():  # noqa
    model_name = 'gpt-3.5-turbo'
    model = OpenAIChat(model_name=model_name)
    assert len(model.history()) == 0
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

    # previous memory is the input to ChatGPT
    assert model._previous_messages[0]['role'] == 'system'
    assert model._previous_messages[0]['content'] == 'You are a helpful assistant.'
    assert model._previous_messages[-1]['role'] == 'user'
    assert model._previous_messages[-1]['content'] == prompt

    assert len(model.history()) == 1
    message = model.previous_record()
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

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.cost_per_token == MODEL_COST_PER_TOKEN[model_name]
    assert model.cost == message.cost
    assert model.total_tokens == message.total_tokens
    assert model.prompt_tokens == message.prompt_tokens
    assert model.response_tokens == message.response_tokens

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
    response = model(prompt)
    assert isinstance(response, str)
    assert len(response) > 1

    # previous memory is the input to ChatGPT
    assert model._previous_messages[0]['role'] == 'system'
    assert model._previous_messages[1]['role'] == 'user'
    assert model._previous_messages[1]['content'] == previous_prompt
    assert model._previous_messages[2]['role'] == 'assistant'
    assert model._previous_messages[2]['content'] == previous_response
    assert model._previous_messages[3]['role'] == 'user'
    assert model._previous_messages[3]['content'] == prompt

    assert len(model.history()) == 2
    message = model.previous_record()
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

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.cost_per_token == MODEL_COST_PER_TOKEN[model_name]
    assert model.cost == previous_cost + message.cost
    assert model.total_tokens == previous_total_tokens + message.total_tokens
    assert model.prompt_tokens == previous_prompt_tokens + message.prompt_tokens
    assert model.response_tokens == previous_response_tokens + message.response_tokens

def test_OpenAIChat_streaming():  # noqa
    """Test the same thing as above but for streaming. All usage and history should be the same."""
    callback_response = ''
    def streaming_callback(record: StreamingEvent) -> None:
        nonlocal callback_response
        callback_response += record.response

    model_name = 'gpt-3.5-turbo'
    model = OpenAIChat(model_name=model_name, streaming_callback=streaming_callback)
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
    assert response == callback_response

    # previous memory is the input to ChatGPT
    assert model._previous_messages[0]['role'] == 'system'
    assert model._previous_messages[0]['content'] == 'You are a helpful assistant.'
    assert model._previous_messages[-1]['role'] == 'user'
    assert model._previous_messages[-1]['content'] == prompt

    assert len(model.history()) == 1
    message = model.previous_record()
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

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.cost_per_token == MODEL_COST_PER_TOKEN[model_name]
    assert model.cost == message.cost
    assert model.total_tokens == message.total_tokens
    assert model.prompt_tokens == message.prompt_tokens
    assert model.response_tokens == message.response_tokens

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
    response = model(prompt)
    assert isinstance(response, str)
    assert len(response) > 1
    assert response == callback_response

    # previous memory is the input to ChatGPT
    assert model._previous_messages[0]['role'] == 'system'
    assert model._previous_messages[1]['role'] == 'user'
    assert model._previous_messages[1]['content'] == previous_prompt
    assert model._previous_messages[2]['role'] == 'assistant'
    assert model._previous_messages[2]['content'] == previous_response
    assert model._previous_messages[3]['role'] == 'user'
    assert model._previous_messages[3]['content'] == prompt

    assert len(model.history()) == 2
    message = model.previous_record()
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

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.cost_per_token == MODEL_COST_PER_TOKEN[model_name]
    assert model.cost == previous_cost + message.cost
    assert model.total_tokens == previous_total_tokens + message.total_tokens
    assert model.prompt_tokens == previous_prompt_tokens + message.prompt_tokens
    assert model.response_tokens == previous_response_tokens + message.response_tokens

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
