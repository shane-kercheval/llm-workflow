"""Test the OpenAI Model classes."""

from openai import OpenAI
import pytest
from llm_workflow.base import Document, EmbeddingRecord, ExchangeRecord, StreamingEvent
from llm_workflow.memory import (
    LastNExchangesManager,
    LastNTokensMemoryManager,
    MessageSummaryMemoryManager,
)
from llm_workflow.openai import (
    OpenAIChat,
    OpenAIEmbedding,
    message_formatter,
    num_tokens,
    num_tokens_from_messages,
    MODEL_COST_PER_TOKEN,
)


def test_num_tokens():  # noqa
    assert num_tokens(model_name='gpt-3.5-turbo-0613', value="This should be six tokens.") == 6

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
    model_name = 'gpt-3.5-turbo-0613'
    client = OpenAI()
    response = client.chat.completions.create(
        model=model_name,
        messages=example_messages,
        temperature=0,
        max_tokens=1,  # we're only counting input tokens here, so let's not waste tokens on output
    )
    expected_value = response.usage.prompt_tokens
    actual_value = num_tokens_from_messages(model_name=model_name, messages=example_messages)
    assert expected_value == actual_value

    # above we checked that the numbers match exactly from what OpenAI returns;
    # here, let's just check that the other models run and return >0 to avoid API calls
    assert num_tokens_from_messages(model_name='gpt-3.5-turbo-0301', messages=example_messages) > 0
    assert num_tokens_from_messages(model_name='gpt-4-1106-preview', messages=example_messages) > 0
    assert num_tokens_from_messages(model_name='gpt-4-0314', messages=example_messages) > 0
    with pytest.raises(NotImplementedError):
        num_tokens_from_messages(model_name='<not implemented>', messages=example_messages)

def test_message_formatter():  # noqa
    assert message_formatter(None, None, None) == []
    messages = message_formatter('System message', None, None)
    assert messages == [{'role': 'system', 'content': 'System message'}]
    messages = message_formatter(None, [ExchangeRecord(prompt='prompt', response='response')], None)  # noqa
    assert messages == [{'role': 'user', 'content': 'prompt'}, {'role': 'assistant', 'content': 'response'}]  # noqa
    messages = message_formatter(None, None, 'prompt')
    assert messages == [{'role': 'user', 'content': 'prompt'}]
    messages = message_formatter(
        'System message',
        [ExchangeRecord(prompt='prompt', response='response')],
        'New Prompt.',
    )
    assert messages == [
        {'role': 'system', 'content': 'System message'},
        {'role': 'user', 'content': 'prompt'}, {'role': 'assistant', 'content': 'response'},
        {'role': 'user', 'content': 'New Prompt.'},
    ]

def test_OpenAIChat():  # noqa
    model = OpenAIChat()
    assert len(model.history()) == 0
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

    # previous memory is the input to ChatGPT
    assert model._previous_messages[0]['role'] == 'system'
    assert model._previous_messages[0]['content'] == model.system_message
    assert model._previous_messages[-1]['role'] == 'user'
    assert model._previous_messages[-1]['content'] == prompt

    assert len(model.history()) == 1
    assert len(model.chat_history) == 1
    assert model.history() == model.chat_history
    assert model.chat_history[0].prompt == prompt
    assert model.chat_history[0].response == response

    message = model.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata['model_name'] == model.model_name
    assert message.metadata['messages'] == model._previous_messages
    assert message.cost == (MODEL_COST_PER_TOKEN[model.model_name]['input'] * message.input_tokens) + \
        (MODEL_COST_PER_TOKEN[model.model_name]['output'] * message.response_tokens)  # noqa: E501
    assert message.input_tokens == num_tokens_from_messages(
        model_name=model.model_name,
        messages=model._previous_messages,
    )
    assert message.response_tokens == num_tokens(model_name=model.model_name, value=response)
    assert message.total_tokens == message.input_tokens + message.response_tokens
    assert message.uuid
    assert message.timestamp

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.cost_per_token == MODEL_COST_PER_TOKEN[model.model_name]
    assert model.cost == message.cost
    assert model.total_tokens == message.total_tokens
    assert model.input_tokens == message.input_tokens
    assert model.response_tokens == message.response_tokens

    previous_prompt = prompt
    previous_response = response
    previous_cost = message.cost
    previous_total_tokens = message.total_tokens
    previous_input_tokens = message.input_tokens
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
    assert model._previous_messages[0]['content'] == model.system_message
    assert model._previous_messages[1]['role'] == 'user'
    assert model._previous_messages[1]['content'] == previous_prompt
    assert model._previous_messages[2]['role'] == 'assistant'
    assert model._previous_messages[2]['content'] == previous_response
    assert model._previous_messages[3]['role'] == 'user'
    assert model._previous_messages[3]['content'] == prompt

    assert len(model.history()) == 2
    assert len(model.chat_history) == 2
    assert model.history() == model.chat_history
    assert model.chat_history[0].prompt == previous_prompt
    assert model.chat_history[0].response == previous_response
    assert model.chat_history[1].prompt == prompt
    assert model.chat_history[1].response == response

    message = model.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata['model_name'] == model.model_name
    assert message.metadata['messages'] == model._previous_messages
    assert message.input_tokens == num_tokens_from_messages(
        model_name=model.model_name,
        messages=model._previous_messages,
    )
    assert message.response_tokens == num_tokens(model_name=model.model_name, value=response)
    assert message.cost == (MODEL_COST_PER_TOKEN[model.model_name]['input'] * message.input_tokens) + \
        (MODEL_COST_PER_TOKEN[model.model_name]['output'] * message.response_tokens)  # noqa: E501
    assert message.total_tokens == message.input_tokens + message.response_tokens
    assert message.uuid
    assert message.uuid != previous_message.uuid
    assert message.timestamp

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.cost_per_token == MODEL_COST_PER_TOKEN[model.model_name]
    assert model.cost == previous_cost + message.cost
    assert model.total_tokens == previous_total_tokens + message.total_tokens
    assert model.input_tokens == previous_input_tokens + message.input_tokens
    assert model.response_tokens == previous_response_tokens + message.response_tokens

def test_OpenAIChat_streaming():  # noqa
    """Test the same thing as above but for streaming. All usage and history should be the same."""
    callback_response = ''
    def streaming_callback(record: StreamingEvent) -> None:
        nonlocal callback_response
        callback_response += record.response

    model = OpenAIChat(streaming_callback=streaming_callback)
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
    assert response == callback_response

    # previous memory is the input to ChatGPT
    assert model._previous_messages[0]['role'] == 'system'
    assert model._previous_messages[0]['content'] == model.system_message
    assert model._previous_messages[-1]['role'] == 'user'
    assert model._previous_messages[-1]['content'] == prompt

    assert len(model.history()) == 1
    assert len(model.chat_history) == 1
    assert model.history() == model.chat_history
    assert model.chat_history[0].prompt == prompt
    assert model.chat_history[0].response == response

    message = model.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata['model_name'] == model.model_name
    assert message.metadata['messages'] == model._previous_messages
    assert message.input_tokens == num_tokens_from_messages(
        model_name=model.model_name,
        messages=model._previous_messages,
    )
    assert message.response_tokens == num_tokens(model_name=model.model_name, value=response)
    assert message.cost == (MODEL_COST_PER_TOKEN[model.model_name]['input'] * message.input_tokens) + \
        (MODEL_COST_PER_TOKEN[model.model_name]['output'] * message.response_tokens)  # noqa: E501
    assert message.total_tokens == message.input_tokens + message.response_tokens
    assert message.uuid
    assert message.timestamp

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.cost_per_token == MODEL_COST_PER_TOKEN[model.model_name]
    assert model.cost == message.cost
    assert model.total_tokens == message.total_tokens
    assert model.input_tokens == message.input_tokens
    assert model.response_tokens == message.response_tokens

    previous_prompt = prompt
    previous_response = response
    previous_cost = message.cost
    previous_total_tokens = message.total_tokens
    previous_input_tokens = message.input_tokens
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
    assert model._previous_messages[0]['content'] == model.system_message
    assert model._previous_messages[1]['role'] == 'user'
    assert model._previous_messages[1]['content'] == previous_prompt
    assert model._previous_messages[2]['role'] == 'assistant'
    assert model._previous_messages[2]['content'] == previous_response
    assert model._previous_messages[3]['role'] == 'user'
    assert model._previous_messages[3]['content'] == prompt

    assert len(model.history()) == 2
    assert len(model.chat_history) == 2
    assert model.history() == model.chat_history
    assert model.chat_history[0].prompt == previous_prompt
    assert model.chat_history[0].response == previous_response
    assert model.chat_history[1].prompt == prompt
    assert model.chat_history[1].response == response

    message = model.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata['model_name'] == model.model_name
    assert message.metadata['messages'] == model._previous_messages
    assert message.input_tokens == num_tokens_from_messages(
        model_name=model.model_name,
        messages=model._previous_messages,
    )
    assert message.response_tokens == num_tokens(model_name=model.model_name, value=response)
    assert message.cost == (MODEL_COST_PER_TOKEN[model.model_name]['input'] * message.input_tokens) + \
        (MODEL_COST_PER_TOKEN[model.model_name]['output'] * message.response_tokens)  # noqa: E501
    assert message.total_tokens == message.input_tokens + message.response_tokens
    assert message.uuid
    assert message.uuid != previous_message.uuid
    assert message.timestamp

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.cost_per_token == MODEL_COST_PER_TOKEN[model.model_name]
    assert model.cost == previous_cost + message.cost
    assert model.total_tokens == previous_total_tokens + message.total_tokens
    assert model.input_tokens == previous_input_tokens + message.input_tokens
    assert model.response_tokens == previous_response_tokens + message.response_tokens

def test_OpenAIChat_streaming_response_matches_non_streaming():  # noqa
    """
    Test that we get the same final response and usage data when streaming vs not streaming.
    Additionally test that the response we get in the streaming callback matches the overall
    response.
    """
    question = "What is the capital of France?"
    non_streaming_chat = OpenAIChat(seed=42)
    non_streaming_response = non_streaming_chat(question)

    callback_response = ''
    def streaming_callback(record: StreamingEvent) -> None:
        nonlocal callback_response
        callback_response += record.response

    streaming_chat = OpenAIChat(streaming_callback=streaming_callback, seed=42)
    streaming_response  = streaming_chat(question)
    assert non_streaming_response == streaming_response
    assert non_streaming_response == callback_response
    assert non_streaming_chat.input_tokens == streaming_chat.input_tokens
    assert non_streaming_chat.response_tokens == streaming_chat.response_tokens
    assert non_streaming_chat.total_tokens == streaming_chat.total_tokens
    assert non_streaming_chat.cost == streaming_chat.cost

def test_OpenAIChat__LastNExchangesManager0():  # noqa
    model = OpenAIChat(memory_manager=LastNExchangesManager(last_n_exchanges=0))
    assert model.previous_record() is None
    assert model.previous_prompt is None
    assert model.previous_response is None
    assert model.cost == 0
    assert model.total_tokens == 0
    assert model.input_tokens == 0
    assert model.response_tokens == 0

    ####
    # first interaction
    # this shouldn't be any different
    ####
    prompt = "Hi my name is shane. What is my name?"
    response = model(prompt)
    assert 'shane' in response.lower()
    assert isinstance(response, str)
    assert len(response) > 1

    # previous memory is the input to ChatGPT
    assert model._previous_messages[0]['role'] == 'system'
    assert model._previous_messages[-1]['role'] == 'user'
    assert model._previous_messages[-1]['content'] == prompt

    assert len(model.history()) == 1
    assert len(model.chat_history) == 1
    assert model.history() == model.chat_history
    assert model.chat_history[0].prompt == prompt
    assert model.chat_history[0].response == response

    message = model.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata['model_name'] == model.model_name
    assert message.input_tokens == num_tokens_from_messages(
        model_name=model.model_name,
        messages=model._previous_messages,
    )
    assert message.response_tokens == num_tokens(model_name=model.model_name, value=response)
    assert message.total_tokens == message.input_tokens + message.response_tokens
    assert message.cost == (MODEL_COST_PER_TOKEN[model.model_name]['input'] * message.input_tokens) + \
        (MODEL_COST_PER_TOKEN[model.model_name]['output'] * message.response_tokens)  # noqa: E501

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.cost_per_token == MODEL_COST_PER_TOKEN[model.model_name]
    assert model.cost == message.cost
    assert model.total_tokens == message.total_tokens
    assert model.input_tokens == message.input_tokens
    assert model.response_tokens == message.response_tokens

    previous_prompt = prompt
    previous_response = response
    previous_cost = message.cost
    previous_total_tokens = message.total_tokens
    previous_input_tokens = message.input_tokens
    previous_response_tokens = message.response_tokens

    ####
    # second interaction
    # this shouldn't be any different
    ####
    prompt = "What is my name?"
    response = model(prompt)
    assert 'shane' not in response.lower()
    assert isinstance(response, str)
    assert len(response) > 1

    # previous memory is the input to ChatGPT
    assert len(model._previous_messages) == 2  # system/user
    assert model._previous_messages[0]['role'] == 'system'
    assert model._previous_messages[0]['content'] == 'You are a helpful assistant.'
    assert model._previous_messages[1]['role'] == 'user'
    assert model._previous_messages[1]['content'] == prompt

    assert len(model.history()) == 2
    assert len(model.chat_history) == 2
    assert model.history() == model.chat_history
    assert model.chat_history[0].prompt == previous_prompt
    assert model.chat_history[0].response == previous_response
    assert model.chat_history[1].prompt == prompt
    assert model.chat_history[1].response == response

    message = model.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata['model_name'] == model.model_name
    assert message.input_tokens == num_tokens_from_messages(
        model_name=model.model_name,
        messages=model._previous_messages,
    )
    assert message.response_tokens == num_tokens(model_name=model.model_name, value=response)
    assert message.total_tokens == message.input_tokens + message.response_tokens
    assert message.cost == (MODEL_COST_PER_TOKEN[model.model_name]['input'] * message.input_tokens) + \
        (MODEL_COST_PER_TOKEN[model.model_name]['output'] * message.response_tokens)  # noqa: E501
    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.cost_per_token == MODEL_COST_PER_TOKEN[model.model_name]
    assert model.cost == previous_cost + message.cost
    assert model.total_tokens == previous_total_tokens + message.total_tokens
    assert model.input_tokens == previous_input_tokens + message.input_tokens
    assert model.response_tokens == previous_response_tokens + message.response_tokens

def test_OpenAIChat__LastNExchangesManager1():  # noqa
    model = OpenAIChat(memory_manager=LastNExchangesManager(last_n_exchanges=1))
    assert model.previous_record() is None
    assert model.previous_prompt is None
    assert model.previous_response is None
    assert model.cost == 0
    assert model.total_tokens == 0
    assert model.input_tokens == 0
    assert model.response_tokens == 0

    ####
    # first interaction
    # this shouldn't be any different
    ####
    prompt = "Hi my name is shane. What is my name?"
    response = model(prompt)
    assert 'shane' in response.lower()
    assert isinstance(response, str)
    assert len(response) > 1

    # previous memory is the input to ChatGPT
    assert model._previous_messages[0]['role'] == 'system'
    assert model._previous_messages[-1]['role'] == 'user'
    assert model._previous_messages[-1]['content'] == prompt

    assert len(model.history()) == 1
    assert len(model.chat_history) == 1
    assert model.history() == model.chat_history
    assert model.chat_history[0].prompt == prompt
    assert model.chat_history[0].response == response

    message = model.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata['model_name'] == model.model_name
    assert message.input_tokens == num_tokens_from_messages(
        model_name=model.model_name,
        messages=model._previous_messages,
    )
    assert message.response_tokens == num_tokens(model_name=model.model_name, value=response)
    assert message.total_tokens == message.input_tokens + message.response_tokens
    assert message.cost == (MODEL_COST_PER_TOKEN[model.model_name]['input'] * message.input_tokens) + \
        (MODEL_COST_PER_TOKEN[model.model_name]['output'] * message.response_tokens)  # noqa: E501

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.cost_per_token == MODEL_COST_PER_TOKEN[model.model_name]
    assert model.cost == message.cost
    assert model.total_tokens == message.total_tokens
    assert model.input_tokens == message.input_tokens
    assert model.response_tokens == message.response_tokens

    previous_prompt = prompt
    previous_response = response
    previous_cost = message.cost
    previous_total_tokens = message.total_tokens
    previous_input_tokens = message.input_tokens
    previous_response_tokens = message.response_tokens

    ####
    # second interaction
    # this shouldn't be any different
    ####
    prompt = "What is my name?"
    response = model(prompt)
    assert 'shane' in response.lower()
    assert isinstance(response, str)
    assert len(response) > 1

    # previous memory is the input to ChatGPT
    assert model._previous_messages[0]['role'] == 'system'
    assert model._previous_messages[0]['content'] == 'You are a helpful assistant.'
    assert model._previous_messages[1]['role'] == 'user'
    assert model._previous_messages[1]['content'] == previous_prompt
    assert model._previous_messages[2]['role'] == 'assistant'
    assert model._previous_messages[2]['content'] == previous_response
    assert model._previous_messages[3]['role'] == 'user'
    assert model._previous_messages[3]['content'] == prompt

    assert len(model.history()) == 2
    assert len(model.chat_history) == 2
    assert model.history() == model.chat_history
    assert model.chat_history[0].prompt == previous_prompt
    assert model.chat_history[0].response == previous_response
    assert model.chat_history[1].prompt == prompt
    assert model.chat_history[1].response == response

    message = model.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata['model_name'] == model.model_name
    assert message.input_tokens == num_tokens_from_messages(
        model_name=model.model_name,
        messages=model._previous_messages,
    )
    assert message.response_tokens == num_tokens(model_name=model.model_name, value=response)
    assert message.total_tokens == message.input_tokens + message.response_tokens
    assert message.cost == (MODEL_COST_PER_TOKEN[model.model_name]['input'] * message.input_tokens) + \
        (MODEL_COST_PER_TOKEN[model.model_name]['output'] * message.response_tokens)  # noqa: E501

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.cost_per_token == MODEL_COST_PER_TOKEN[model.model_name]
    assert model.cost == previous_cost + message.cost
    assert model.total_tokens == previous_total_tokens + message.total_tokens
    assert model.input_tokens == previous_input_tokens + message.input_tokens
    assert model.response_tokens == previous_response_tokens + message.response_tokens

    previous_prompt = prompt
    previous_response = response

    ####
    # third interaction
    # this shouldn't be any different
    ####
    prompt = "What is today's date?"
    response = model(prompt)
    assert 'shane' not in response.lower()
    assert isinstance(response, str)
    assert len(response) > 1

    # previous memory is the input to ChatGPT
    # The last message should contain shane, but not this one
    assert model._previous_messages[0]['role'] == 'system'
    assert model._previous_messages[0]['content'] == 'You are a helpful assistant.'
    assert model._previous_messages[1]['role'] == 'user'
    assert model._previous_messages[1]['content'] == previous_prompt
    assert model._previous_messages[2]['role'] == 'assistant'
    assert model._previous_messages[2]['content'] == previous_response
    assert 'shane' in model._previous_messages[2]['content'].lower()
    assert model._previous_messages[3]['role'] == 'user'
    assert model._previous_messages[3]['content'] == prompt
    assert len(model._previous_messages) == 4

    assert len(model.history()) == 3
    message = model.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata['model_name'] == model.model_name
    assert message.input_tokens == num_tokens_from_messages(
        model_name=model.model_name,
        messages=model._previous_messages,
    )
    assert message.response_tokens == num_tokens(model_name=model.model_name, value=response)
    assert message.total_tokens == message.input_tokens + message.response_tokens
    assert message.cost == (MODEL_COST_PER_TOKEN[model.model_name]['input'] * message.input_tokens) + \
        (MODEL_COST_PER_TOKEN[model.model_name]['output'] * message.response_tokens)  # noqa: E501

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.cost_per_token == MODEL_COST_PER_TOKEN[model.model_name]

    previous_prompt = prompt
    previous_response = response

    ####
    # 4th interaction
    # this shouldn't contain the name shane because the last interaction was the first that didn't
    ####
    prompt = "What is today's date?"
    response = model(prompt)
    assert 'shane' not in response.lower()
    assert isinstance(response, str)
    assert len(response) > 1

    # previous memory is the input to ChatGPT
    assert model._previous_messages[0]['role'] == 'system'
    assert model._previous_messages[0]['content'] == 'You are a helpful assistant.'
    assert model._previous_messages[1]['role'] == 'user'
    assert model._previous_messages[1]['content'] == previous_prompt
    assert model._previous_messages[2]['role'] == 'assistant'
    assert model._previous_messages[2]['content'] == previous_response
    assert model._previous_messages[3]['role'] == 'user'
    assert model._previous_messages[3]['content'] == prompt
    # still 4 because we are only keeping 1 message
    # (1)system + (2)previous question + (3)previous answer + (4)new question
    assert len(model._previous_messages) == 4

    assert len(model.history()) == 4
    message = model.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata['model_name'] == model.model_name
    assert message.input_tokens == num_tokens_from_messages(
        model_name=model.model_name,
        messages=model._previous_messages,
    )
    assert message.response_tokens == num_tokens(model_name=model.model_name, value=response)
    assert message.total_tokens == message.input_tokens + message.response_tokens
    assert message.cost == (MODEL_COST_PER_TOKEN[model.model_name]['input'] * message.input_tokens) + \
        (MODEL_COST_PER_TOKEN[model.model_name]['output'] * message.response_tokens)  # noqa: E501

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.cost_per_token == MODEL_COST_PER_TOKEN[model.model_name]

def test_OpenAIChat_with_LastNTokensMemoryManager_1000_tokens():  # noqa
    model = OpenAIChat(memory_manager=LastNTokensMemoryManager(last_n_tokens=1000))
    assert len(model.history()) == 0
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

    # previous memory is the input to ChatGPT
    assert model._previous_messages[0]['role'] == 'system'
    assert model._previous_messages[0]['content'] == model.system_message
    assert model._previous_messages[-1]['role'] == 'user'
    assert model._previous_messages[-1]['content'] == prompt

    assert len(model.history()) == 1
    assert len(model.chat_history) == 1
    assert model.history() == model.chat_history
    assert model.chat_history[0].prompt == prompt
    assert model.chat_history[0].response == response

    message = model.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata['model_name'] == model.model_name
    assert message.metadata['messages'] == model._previous_messages
    assert message.input_tokens == num_tokens_from_messages(
        model_name=model.model_name,
        messages=model._previous_messages,
    )
    assert message.response_tokens == num_tokens(model_name=model.model_name, value=response)
    assert message.total_tokens == message.input_tokens + message.response_tokens
    assert message.cost == (MODEL_COST_PER_TOKEN[model.model_name]['input'] * message.input_tokens) + \
        (MODEL_COST_PER_TOKEN[model.model_name]['output'] * message.response_tokens)  # noqa: E501
    assert message.uuid
    assert message.timestamp

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.cost_per_token == MODEL_COST_PER_TOKEN[model.model_name]
    assert model.cost == message.cost
    assert model.total_tokens == message.total_tokens
    assert model.input_tokens == message.input_tokens
    assert model.response_tokens == message.response_tokens

    previous_prompt = prompt
    previous_response = response
    previous_cost = message.cost
    previous_total_tokens = message.total_tokens
    previous_input_tokens = message.input_tokens
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
    assert model._previous_messages[0]['content'] == model.system_message
    assert model._previous_messages[1]['role'] == 'user'
    assert model._previous_messages[1]['content'] == previous_prompt
    assert model._previous_messages[2]['role'] == 'assistant'
    assert model._previous_messages[2]['content'] == previous_response
    assert model._previous_messages[3]['role'] == 'user'
    assert model._previous_messages[3]['content'] == prompt

    assert len(model.history()) == 2
    assert len(model.chat_history) == 2
    assert model.history() == model.chat_history
    assert model.chat_history[0].prompt == previous_prompt
    assert model.chat_history[0].response == previous_response
    assert model.chat_history[1].prompt == prompt
    assert model.chat_history[1].response == response

    message = model.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata['model_name'] == model.model_name
    assert message.metadata['messages'] == model._previous_messages
    assert message.input_tokens == num_tokens_from_messages(
        model_name=model.model_name,
        messages=model._previous_messages,
    )
    assert message.response_tokens == num_tokens(model_name=model.model_name, value=response)
    assert message.cost == (MODEL_COST_PER_TOKEN[model.model_name]['input'] * message.input_tokens) + \
        (MODEL_COST_PER_TOKEN[model.model_name]['output'] * message.response_tokens)  # noqa: E501
    assert message.total_tokens == message.input_tokens + message.response_tokens
    assert message.uuid
    assert message.uuid != previous_message.uuid
    assert message.timestamp

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.cost_per_token == MODEL_COST_PER_TOKEN[model.model_name]
    assert model.cost == previous_cost + message.cost
    assert model.total_tokens == previous_total_tokens + message.total_tokens
    assert model.input_tokens == previous_input_tokens + message.input_tokens
    assert model.response_tokens == previous_response_tokens + message.response_tokens

def test_OpenAIChat_with_LastNTokensMemoryManager_75_tokens():  # noqa
    model = OpenAIChat(memory_manager=LastNTokensMemoryManager(last_n_tokens=45))
    assert len(model.history()) == 0
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

    # previous memory is the input to ChatGPT
    assert model._previous_messages[0]['role'] == 'system'
    assert model._previous_messages[0]['content'] == model.system_message
    assert model._previous_messages[-1]['role'] == 'user'
    assert model._previous_messages[-1]['content'] == prompt

    assert len(model.history()) == 1
    assert len(model.chat_history) == 1
    assert model.history() == model.chat_history
    assert model.chat_history[0].prompt == prompt
    assert model.chat_history[0].response == response

    message = model.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata['model_name'] == model.model_name
    assert message.metadata['messages'] == model._previous_messages
    assert message.cost == (MODEL_COST_PER_TOKEN[model.model_name]['input'] * message.input_tokens) + \
        (MODEL_COST_PER_TOKEN[model.model_name]['output'] * message.response_tokens)  # noqa: E501
    assert message.input_tokens == num_tokens_from_messages(
        model_name=model.model_name,
        messages=model._previous_messages,
    )
    assert message.response_tokens == num_tokens(model_name=model.model_name, value=response)
    assert message.total_tokens == message.input_tokens + message.response_tokens
    assert message.uuid
    assert message.timestamp

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.cost_per_token == MODEL_COST_PER_TOKEN[model.model_name]
    assert model.cost == message.cost
    assert model.total_tokens == message.total_tokens
    assert model.input_tokens == message.input_tokens
    assert model.response_tokens == message.response_tokens

    previous_prompt = prompt
    previous_response = response
    previous_cost = message.cost
    previous_total_tokens = message.total_tokens
    previous_input_tokens = message.input_tokens
    previous_response_tokens = message.response_tokens
    previous_message = message

    ####
    # second interaction
    # this second call should not include the first question/response because there is not enough
    # tokens
    ####
    prompt = "This is another question."
    response = model(prompt)
    assert isinstance(response, str)
    assert len(response) > 1

    # previous memory is the input to ChatGPT
    assert model._previous_messages[0]['role'] == 'system'
    assert model._previous_messages[0]['content'] == model.system_message
    # assert model._previous_messages[1]['role'] == 'user'
    # assert model._previous_messages[1]['content'] == previous_prompt
    # assert model._previous_messages[2]['role'] == 'assistant'
    # assert model._previous_messages[2]['content'] == previous_response
    assert model._previous_messages[1]['role'] == 'user'
    assert model._previous_messages[1]['content'] == prompt

    assert len(model.history()) == 2
    assert len(model.chat_history) == 2
    assert model.history() == model.chat_history
    assert model.chat_history[0].prompt == previous_prompt
    assert model.chat_history[0].response == previous_response
    assert model.chat_history[1].prompt == prompt
    assert model.chat_history[1].response == response

    message = model.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata['model_name'] == model.model_name
    assert message.metadata['messages'] == model._previous_messages
    assert message.input_tokens == num_tokens_from_messages(
        model_name=model.model_name,
        messages=model._previous_messages,
    )
    assert message.response_tokens == num_tokens(model_name=model.model_name, value=response)
    assert message.cost == (MODEL_COST_PER_TOKEN[model.model_name]['input'] * message.input_tokens) + \
        (MODEL_COST_PER_TOKEN[model.model_name]['output'] * message.response_tokens)  # noqa: E501
    assert message.total_tokens == message.input_tokens + message.response_tokens
    assert message.uuid
    assert message.uuid != previous_message.uuid
    assert message.timestamp

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.cost_per_token == MODEL_COST_PER_TOKEN[model.model_name]
    assert model.cost == previous_cost + message.cost
    assert model.total_tokens == previous_total_tokens + message.total_tokens
    assert model.input_tokens == previous_input_tokens + message.input_tokens
    assert model.response_tokens == previous_response_tokens + message.response_tokens

def test_OpenAIChat_with_LastNTokensMemoryManager__1_tokens():  # noqa
    model = OpenAIChat(memory_manager=LastNTokensMemoryManager(last_n_tokens=1))
    prompt = "My name is Shane and my favorite color is blue. What's your name?"
    with pytest.raises(AssertionError):
        _ = model(prompt)

def test_OpenAIChat_with_MessageSummaryMemoryManager():  # noqa
    summarize_instruction = 'Summarize the following while retaining the important information.'
    model = OpenAIChat(
        memory_manager=MessageSummaryMemoryManager(
            model=OpenAIChat(),
            summarize_instruction=summarize_instruction,
        ),
    )
    assert len(model.history()) == 0
    assert model.previous_record() is None
    assert model.previous_prompt is None
    assert model.previous_response is None
    assert model.cost == 0
    assert model.total_tokens == 0
    assert model.input_tokens == 0
    assert model.response_tokens == 0

    ####
    # first interaction; no summarization yet
    ####
    prompt = "Hi my name is Shane. I'd like to study data science. Please form a curriculum for me."  # noqa
    response = model(prompt)
    assert isinstance(response, str)
    assert len(response) > 1

    # no summarization yet
    assert len(model._memory_manager._model.history()) == 0
    assert len(model._memory_manager.history()) == 0

    # previous memory is the input to ChatGPT
    assert model._previous_messages[0]['role'] == 'system'
    assert model._previous_messages[0]['content'] == model.system_message
    assert model._previous_messages[-1]['role'] == 'user'
    assert model._previous_messages[-1]['content'] == prompt

    assert len(model.history()) == 1
    assert len(model.chat_history) == 1
    assert model.history() == model.chat_history
    assert model.chat_history[0].prompt == prompt
    assert model.chat_history[0].response == response

    message = model.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata['model_name'] == model.model_name
    assert message.metadata['messages'] == model._previous_messages
    assert message.input_tokens == num_tokens_from_messages(
        model_name=model.model_name,
        messages=model._previous_messages,
    )
    assert message.response_tokens == num_tokens(model_name=model.model_name, value=response)
    assert message.total_tokens == message.input_tokens + message.response_tokens
    assert message.cost == (MODEL_COST_PER_TOKEN[model.model_name]['input'] * message.input_tokens) + \
        (MODEL_COST_PER_TOKEN[model.model_name]['output'] * message.response_tokens)  # noqa: E501
    assert message.uuid
    assert message.timestamp

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.cost_per_token == MODEL_COST_PER_TOKEN[model.model_name]
    assert model.cost == message.cost
    assert model.total_tokens == message.total_tokens
    assert model.input_tokens == message.input_tokens
    assert model.response_tokens == message.response_tokens

    previous_prompt_0 = prompt
    previous_response_0 = response
    previous_cost_0 = message.cost
    previous_total_tokens_0 = message.total_tokens
    previous_input_tokens_0 = message.input_tokens
    previous_response_tokens_0 = message.response_tokens
    previous_message_0 = message

    ####
    # second interaction
    ####
    prompt = "How long do you think it will take to complete the curriculum?"
    response = model(prompt)
    assert isinstance(response, str)
    assert len(response) > 1

    # summarized both prompt/response
    # only 1 message because we are summarizing only the response because the prompt doesn't
    # exceed the threshold
    assert len(model._memory_manager._model.history()) == 1
    assert len(model._memory_manager.history()) == 1
    assert model._memory_manager.history() == model._memory_manager._model.history()
    assert model._memory_manager.history()[0].prompt == model._memory_manager._summarize_format_message(previous_response_0)  # noqa
    summarized_response_0 = model._memory_manager.history()[0].response
    assert 'data science' in summarized_response_0
    assert summarized_response_0 != previous_response_0

    # previous memory is the input to ChatGPT
    assert model._previous_messages[0]['role'] == 'system'
    assert model._previous_messages[0]['content'] == model.system_message
    assert model._previous_messages[1]['role'] == 'user'
    assert model._previous_messages[1]['content'] == previous_prompt_0
    assert model._previous_messages[2]['role'] == 'assistant'
    assert model._previous_messages[2]['content'] == summarized_response_0
    assert model._previous_messages[3]['role'] == 'user'
    assert model._previous_messages[3]['content'] == prompt

    assert len(model.history()) == 3
    assert len(model.chat_history) == 2
    assert [model.history()[0], model.history()[2]] == model.chat_history
    assert model.chat_history[0].prompt == previous_prompt_0
    assert model.chat_history[0].response == previous_response_0
    assert model.chat_history[1].prompt == prompt
    assert model.chat_history[1].response == response

    message = model.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata['model_name'] == model.model_name
    assert message.metadata['messages'] == model._previous_messages
    assert message.input_tokens == num_tokens_from_messages(
        model_name=model.model_name,
        messages=model._previous_messages,
    )
    assert message.response_tokens == num_tokens(model_name=model.model_name, value=response)
    assert message.cost == (MODEL_COST_PER_TOKEN[model.model_name]['input'] * message.input_tokens) + \
        (MODEL_COST_PER_TOKEN[model.model_name]['output'] * message.response_tokens)  # noqa: E501
    assert message.total_tokens == message.input_tokens + message.response_tokens
    assert message.uuid
    assert message.uuid != previous_message_0.uuid
    assert message.timestamp

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.cost_per_token == MODEL_COST_PER_TOKEN[model.model_name]
    summarization_cost = sum(record.cost for record in model._memory_manager.history())
    summarization_input_tokens = sum(record.input_tokens for record in model._memory_manager.history())  # noqa
    summarization_response_tokens = sum(record.response_tokens for record in model._memory_manager.history())  # noqa
    assert model.input_tokens == previous_input_tokens_0 + message.input_tokens + summarization_input_tokens  # noqa
    assert model.response_tokens == previous_response_tokens_0 + message.response_tokens + summarization_response_tokens  # noqa
    assert model.total_tokens == previous_total_tokens_0 + message.total_tokens + summarization_input_tokens + summarization_response_tokens  # noqa
    assert round(model.cost, 4) == round(previous_cost_0 + message.cost + summarization_cost, 4)

    previous_prompt_1 = prompt
    previous_response_1 = response
    previous_cost_1 = message.cost
    previous_total_tokens_1 = message.total_tokens
    previous_input_tokens_1 = message.input_tokens
    previous_response_tokens_1 = message.response_tokens

    ####
    # third interaction
    ####
    prompt = "Do you remember my name?"
    response = model(prompt)
    assert isinstance(response, str)
    assert len(response) > 1
    assert 'Shane' in response

    # summarized only responses
    assert len(model._memory_manager._model.history()) == 2
    assert len(model._memory_manager.history()) == 2
    assert model._memory_manager.history() == model._memory_manager._model.history()
    summarized_response_1 = model._memory_manager.history()[1].response
    # ensure original summarized prompt/response are same
    assert model._memory_manager.history()[0].prompt == model._memory_manager._summarize_format_message(previous_response_0)  # noqa
    assert model._memory_manager.history()[0].response == summarized_response_0
    assert model._memory_manager.history()[1].prompt == model._memory_manager._summarize_format_message(previous_response_1)  # noqa
    assert model._memory_manager.history()[1].response == summarized_response_1
    assert summarized_response_1 != previous_response_1

    # previous memory is the input to ChatGPT
    assert model._previous_messages[0]['role'] == 'system'
    assert model._previous_messages[0]['content'] == model.system_message
    assert model._previous_messages[1]['role'] == 'user'
    assert model._previous_messages[1]['content'] == previous_prompt_0
    assert model._previous_messages[2]['role'] == 'assistant'
    assert model._previous_messages[2]['content'] == summarized_response_0
    assert model._previous_messages[3]['role'] == 'user'
    assert model._previous_messages[3]['content'] == previous_prompt_1
    assert model._previous_messages[4]['role'] == 'assistant'
    assert model._previous_messages[4]['content'] == summarized_response_1
    assert model._previous_messages[5]['role'] == 'user'
    assert model._previous_messages[5]['content'] == prompt

    assert len(model.history()) == 5
    assert len(model.chat_history) == 3
    assert [model.history()[0], model.history()[2], model.history()[4]] == model.chat_history
    assert model.chat_history[0].prompt == previous_prompt_0
    assert model.chat_history[0].response == previous_response_0
    assert model.chat_history[1].prompt == previous_prompt_1
    assert model.chat_history[1].response == previous_response_1
    assert model.chat_history[2].prompt == prompt
    assert model.chat_history[2].response == response

    message = model.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata['model_name'] == model.model_name
    assert message.metadata['messages'] == model._previous_messages
    assert message.input_tokens == num_tokens_from_messages(
        model_name=model.model_name,
        messages=model._previous_messages,
    )
    assert message.response_tokens == num_tokens(model_name=model.model_name, value=response)
    assert message.cost == (MODEL_COST_PER_TOKEN[model.model_name]['input'] * message.input_tokens) + \
        (MODEL_COST_PER_TOKEN[model.model_name]['output'] * message.response_tokens)  # noqa: E501
    assert message.total_tokens == message.input_tokens + message.response_tokens
    assert message.uuid
    assert message.uuid != previous_message_0.uuid
    assert message.timestamp

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.cost_per_token == MODEL_COST_PER_TOKEN[model.model_name]
    summarization_cost = sum(record.cost for record in model._memory_manager.history())
    summarization_input_tokens = sum(record.input_tokens for record in model._memory_manager.history())  # noqa
    summarization_response_tokens = sum(record.response_tokens for record in model._memory_manager.history())  # noqa
    assert model.input_tokens == previous_input_tokens_0 + previous_input_tokens_1 + message.input_tokens + summarization_input_tokens  # noqa
    assert model.response_tokens == previous_response_tokens_0 + previous_response_tokens_1 + message.response_tokens + summarization_response_tokens  # noqa
    assert model.total_tokens == previous_total_tokens_0 + previous_total_tokens_1 + message.total_tokens + summarization_input_tokens + summarization_response_tokens  # noqa
    assert round(model.cost, 4) == round(previous_cost_0 + previous_cost_1 + message.cost + summarization_cost, 4)  # noqa

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
    model = OpenAIChat()
    assert model.cost_per_token == MODEL_COST_PER_TOKEN[model.model_name]
    model.model_name = 'gpt-4-1106-preview'
    assert model.cost_per_token == MODEL_COST_PER_TOKEN['gpt-4-1106-preview']

    model = OpenAIEmbedding()
    assert model.cost_per_token == MODEL_COST_PER_TOKEN[model.model_name]
    model.model_name = 'gpt-4-1106-preview'
    assert model.cost_per_token == MODEL_COST_PER_TOKEN['gpt-4-1106-preview']
