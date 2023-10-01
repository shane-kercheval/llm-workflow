"""Test HuggingFace models and helpers."""

import os
from unittest.mock import patch
from llm_workflow.hugging_face import (
    HuggingFaceEndpointChat,
    get_tokenizer,
    llama_message_formatter,
    num_tokens,
    query_hugging_face_endpoint,
)
from llm_workflow.memory import MessageFormatterMaxTokensMemoryManager
from llm_workflow.models import ExchangeRecord, StreamingEvent
from tests.conftest import is_endpoint_available


def test_query_hugging_face_endpoint(fake_retry_handler, fake_hugging_face_response):  # noqa
    with patch('llm_workflow.hugging_face.requests.post', return_value=fake_hugging_face_response) as mock_post:  # noqa
        with patch('llm_workflow.hugging_face.retry_handler', return_value=fake_retry_handler) as mock_retry_handler:  # noqa
            endpoint_url = "https://fake.url"
            payload = {"text": "hello"}
            response = query_hugging_face_endpoint(endpoint_url, payload)
            assert isinstance(response, list)
            assert 'generated_text' in response[0]

def test_llama_message_formatter():  # noqa
    assert llama_message_formatter(system_message=None, messages=[]) == []

    messages = llama_message_formatter(
        system_message=None,
        messages=[
            ExchangeRecord(prompt='a', response='b'),
            ExchangeRecord(prompt='c', response='d'),
        ],
    )
    expected_value = ['[INST] a [/INST]\nb\n', '[INST] c [/INST]\nd\n']
    assert messages == expected_value

    messages = llama_message_formatter(
        system_message='system',
        messages=[
            ExchangeRecord(prompt='a', response='b'),
            ExchangeRecord(prompt='c', response='d'),
        ],
    )
    assert messages == ['[INST] <<SYS>> system <</SYS>> [/INST]\n', *expected_value]

    messages = llama_message_formatter(
        system_message='system',
        messages=[
            ExchangeRecord(prompt='a', response='b'),
            ExchangeRecord(prompt='c', response='d'),
        ],
        prompt='e',
    )
    assert messages == ['[INST] <<SYS>> system <</SYS>> [/INST]\n', *expected_value, '[INST] e [/INST]\n']  # noqa

def test_HuggingFaceEndpointChat__no_token_calculator():  # noqa
    endpoint = os.getenv('HUGGING_FACE_ENDPOINT_LLAMA2_7B')
    if is_endpoint_available(endpoint):
        callback_response = ''
        def streaming_callback(record: StreamingEvent) -> None:
            nonlocal callback_response
            callback_response += record.response

        model = HuggingFaceEndpointChat(
            endpoint_url=endpoint,
            streaming_callback=streaming_callback,
        )
        assert len(model.history()) == 0
        assert model.previous_record() is None
        assert model.previous_prompt is None
        assert model.previous_response is None
        assert model.cost == 0
        assert model.total_tokens == 0
        assert model.prompt_tokens == 0
        assert model.response_tokens == 0

        prompt = "My name is Shane. What's your name?"
        response = model(prompt)
        assert isinstance(response, str)
        assert len(response) > 0
        assert response == callback_response
        assert len(model.history()) == 1
        message = model.previous_record()
        assert isinstance(message, ExchangeRecord)
        assert message.prompt == prompt
        assert message.response == response
        history = model.history()
        assert history[0].prompt == prompt
        assert history[0].response == response
        assert message.metadata['endpoint_url'] == endpoint
        assert '[INST]' in message.metadata['messages']
        assert '[/INST]' in message.metadata['messages']
        assert '<<SYS>>' in message.metadata['messages']
        assert '<</SYS>>' in message.metadata['messages']
        assert prompt in message.metadata['messages']
        assert message.cost is None
        assert message.prompt_tokens is None
        assert message.response_tokens is None
        assert message.total_tokens is None
        assert message.uuid
        assert message.timestamp
        assert model.cost == 0
        assert model.sum(name='cost') == 0
        assert model.sum(name='total_tokens') == 0

def test_HuggingFaceEndpointChat():  # noqa
    endpoint = os.getenv('HUGGING_FACE_ENDPOINT_LLAMA2_7B')
    if is_endpoint_available(endpoint):
        tokenizer = get_tokenizer('meta-llama/Llama-2-7b-chat-hf')

        def calc_num_tokens(value: str):  # noqa
            return num_tokens(value, tokenizer)

        callback_response = ''
        def streaming_callback(record: StreamingEvent) -> None:
            nonlocal callback_response
            callback_response += record.response

        model = HuggingFaceEndpointChat(
            endpoint_url=endpoint,
            calculate_num_tokens=calc_num_tokens,
            streaming_callback=streaming_callback,
        )
        assert len(model.history()) == 0
        assert model.previous_record() is None
        assert model.previous_prompt is None
        assert model.previous_response is None
        assert model.cost == 0
        assert model.total_tokens == 0
        assert model.prompt_tokens == 0
        assert model.response_tokens == 0

        prompt = "My name is Shane. What's your name?"
        response = model(prompt)
        assert isinstance(response, str)
        assert len(response) > 0
        assert response == callback_response
        assert len(model.history()) == 1
        message = model.previous_record()
        assert isinstance(message, ExchangeRecord)
        assert message.prompt == prompt
        assert message.response == response
        history = model.history()
        assert history[0].prompt == prompt
        assert history[0].response == response
        assert message.metadata['endpoint_url'] == endpoint
        assert '[INST]' in message.metadata['messages']
        assert '[/INST]' in message.metadata['messages']
        assert '<<SYS>>' in message.metadata['messages']
        assert '<</SYS>>' in message.metadata['messages']
        assert prompt in message.metadata['messages']
        assert message.cost is None
        assert message.prompt_tokens > 0
        assert message.response_tokens > 0
        assert message.total_tokens == message.prompt_tokens + message.response_tokens
        assert message.uuid
        assert message.timestamp
        assert model.cost == 0
        assert model.sum(name='cost') == 0
        assert model.sum(name='total_tokens') == message.total_tokens

        previous_tokens = message.total_tokens
        previous_prompt = prompt
        previous_response = response
        callback_response = ''
        prompt = "What is my name?"
        response = model(prompt)
        assert isinstance(response, str)
        assert len(response) > 0
        assert 'Shane' in response
        assert response == callback_response
        assert len(model.history()) == 2
        message = model.previous_record()
        assert isinstance(message, ExchangeRecord)
        assert message.prompt == prompt
        assert message.response == response
        history = model.history()
        assert history[0].prompt == previous_prompt
        assert history[0].response == previous_response
        assert history[1].prompt == prompt
        assert history[1].response == response
        assert message.metadata['endpoint_url'] == endpoint
        assert '[INST]' in message.metadata['messages']
        assert '[/INST]' in message.metadata['messages']
        assert '<<SYS>>' in message.metadata['messages']
        assert '<</SYS>>' in message.metadata['messages']
        assert prompt in message.metadata['messages']
        assert message.cost is None
        assert message.prompt_tokens > 0
        assert message.response_tokens > 0
        assert message.total_tokens == message.prompt_tokens + message.response_tokens
        assert message.uuid
        assert message.timestamp
        assert model.cost == 0
        assert model.sum(name='cost') == 0
        assert model.sum(name='total_tokens') == message.total_tokens + previous_tokens

def test_HuggingFaceEndpointChat__timeout():  # noqa
    endpoint = os.getenv('HUGGING_FACE_ENDPOINT_LLAMA2_7B')
    if is_endpoint_available(endpoint):
        callback_response = ''
        callback_count = 0
        def streaming_callback(record: StreamingEvent) -> None:
            nonlocal callback_response, callback_count
            callback_response += record.response
            callback_count += 1

        model = HuggingFaceEndpointChat(
            endpoint_url=endpoint,
            streaming_callback=streaming_callback,
            max_streaming_tokens=30,
            # 1 second is only enough time for one call to the model and associated callback
            timeout=1,
        )
        assert len(model.history()) == 0
        assert model.previous_record() is None
        assert model.previous_prompt is None
        assert model.previous_response is None
        assert model.cost == 0
        assert model.total_tokens == 0
        assert model.prompt_tokens == 0
        assert model.response_tokens == 0

        prompt = "Write a poem about a dog."
        response = model(prompt)

        assert isinstance(response, str)
        assert len(response) > 0
        assert response == callback_response
        assert len(model.history()) == 1
        assert callback_count == 1  # there wouldn't have been enough time for a second callback

        message = model.previous_record()
        assert isinstance(message, ExchangeRecord)
        assert message.prompt == prompt
        assert message.response == response
        history = model.history()
        assert history[0].prompt == prompt
        assert history[0].response == response
        assert message.metadata['endpoint_url'] == endpoint
        assert '[INST]' in message.metadata['messages']
        assert '[/INST]' in message.metadata['messages']
        assert '<<SYS>>' in message.metadata['messages']
        assert '<</SYS>>' in message.metadata['messages']
        assert prompt in message.metadata['messages']
        assert message.cost is None

def test_HuggingFaceEndpointChat__memory_manager__1000_tokens():  # noqa
    endpoint = os.getenv('HUGGING_FACE_ENDPOINT_LLAMA2_7B')
    if is_endpoint_available(endpoint):
        tokenizer = get_tokenizer('meta-llama/Llama-2-7b-chat-hf')

        def calc_num_tokens(value: str):  # noqa
            return num_tokens(value, tokenizer)

        callback_response = ''
        def streaming_callback(record: StreamingEvent) -> None:
            nonlocal callback_response
            callback_response += record.response

        memory_manager = MessageFormatterMaxTokensMemoryManager(
            last_n_tokens=1000,
            calculate_num_tokens=calc_num_tokens,
            message_formatter=llama_message_formatter,
        )
        model = HuggingFaceEndpointChat(
            endpoint_url=endpoint,
            calculate_num_tokens=calc_num_tokens,
            streaming_callback=streaming_callback,
            message_formatter=llama_message_formatter,
            memory_manager=memory_manager,
        )
        assert len(model.history()) == 0
        assert model.previous_record() is None
        assert model.previous_prompt is None
        assert model.previous_response is None
        assert model.cost == 0
        assert model.total_tokens == 0
        assert model.prompt_tokens == 0
        assert model.response_tokens == 0

        prompt = "My name is Shane. What's your name?"
        response = model(prompt)
        assert isinstance(response, str)
        assert len(response) > 0
        assert response == callback_response
        assert len(model.history()) == 1
        message = model.previous_record()
        assert isinstance(message, ExchangeRecord)
        assert message.prompt == prompt
        assert message.response == response
        history = model.history()
        assert history[0].prompt == prompt
        assert history[0].response == response
        assert message.metadata['endpoint_url'] == endpoint
        assert '[INST]' in message.metadata['messages']
        assert '[/INST]' in message.metadata['messages']
        assert '<<SYS>>' in message.metadata['messages']
        assert '<</SYS>>' in message.metadata['messages']
        assert prompt in message.metadata['messages']
        assert message.cost is None
        assert message.prompt_tokens > 0
        assert message.response_tokens > 0
        assert message.total_tokens == message.prompt_tokens + message.response_tokens
        assert message.uuid
        assert message.timestamp
        assert model.cost == 0
        assert model.sum(name='cost') == 0
        assert model.sum(name='total_tokens') == message.total_tokens

        previous_tokens = message.total_tokens
        previous_prompt = prompt
        previous_response = response
        callback_response = ''
        prompt = "What is my name?"
        response = model(prompt)
        assert isinstance(response, str)
        assert len(response) > 0
        assert 'Shane' in response
        assert response == callback_response
        assert len(model.history()) == 2
        message = model.previous_record()
        assert isinstance(message, ExchangeRecord)
        assert message.prompt == prompt
        assert message.response == response
        history = model.history()
        assert history[0].prompt == previous_prompt
        assert history[0].response == previous_response
        assert history[1].prompt == prompt
        assert history[1].response == response
        assert message.metadata['endpoint_url'] == endpoint
        assert '[INST]' in message.metadata['messages']
        assert '[/INST]' in message.metadata['messages']
        assert '<<SYS>>' in message.metadata['messages']
        assert '<</SYS>>' in message.metadata['messages']
        assert prompt in message.metadata['messages']
        assert message.cost is None
        assert message.prompt_tokens > 0
        assert message.response_tokens > 0
        assert message.total_tokens == message.prompt_tokens + message.response_tokens
        assert message.uuid
        assert message.timestamp
        assert model.cost == 0
        assert model.sum(name='cost') == 0
        assert model.sum(name='total_tokens') == message.total_tokens + previous_tokens

def test_HuggingFaceEndpointChat__memory_manager__20_tokens():  # noqa
    endpoint = os.getenv('HUGGING_FACE_ENDPOINT_LLAMA2_7B')
    if is_endpoint_available(endpoint):
        tokenizer = get_tokenizer('meta-llama/Llama-2-7b-chat-hf')

        def calc_num_tokens(value: str):  # noqa
            return num_tokens(value, tokenizer)

        callback_response = ''
        def streaming_callback(record: StreamingEvent) -> None:
            nonlocal callback_response
            callback_response += record.response

        memory_manager = MessageFormatterMaxTokensMemoryManager(
            last_n_tokens=20,
            calculate_num_tokens=calc_num_tokens,
            message_formatter=llama_message_formatter,
        )
        model = HuggingFaceEndpointChat(
            endpoint_url=endpoint,
            calculate_num_tokens=calc_num_tokens,
            streaming_callback=streaming_callback,
            message_formatter=llama_message_formatter,
            memory_manager=memory_manager,
        )
        assert len(model.history()) == 0
        assert model.previous_record() is None
        assert model.previous_prompt is None
        assert model.previous_response is None
        assert model.cost == 0
        assert model.total_tokens == 0
        assert model.prompt_tokens == 0
        assert model.response_tokens == 0

        prompt = "My name is Shane and my favorite color is blue. What's your name?"
        response = model(prompt)
        assert isinstance(response, str)
        assert len(response) > 0
        assert response == callback_response
        assert len(model.history()) == 1
        message = model.previous_record()
        assert isinstance(message, ExchangeRecord)
        assert message.prompt == prompt
        assert message.response == response
        history = model.history()
        assert history[0].prompt == prompt
        assert history[0].response == response
        assert message.metadata['endpoint_url'] == endpoint
        assert '[INST]' in message.metadata['messages']
        assert '[/INST]' in message.metadata['messages']
        assert '<<SYS>>' in message.metadata['messages']
        assert '<</SYS>>' in message.metadata['messages']
        assert prompt in message.metadata['messages']
        assert message.cost is None
        assert message.prompt_tokens > 0
        assert message.response_tokens > 0
        assert message.total_tokens == message.prompt_tokens + message.response_tokens
        assert message.uuid
        assert message.timestamp
        assert model.cost == 0
        assert model.sum(name='cost') == 0
        assert model.sum(name='total_tokens') == message.total_tokens

        previous_tokens = message.total_tokens
        previous_prompt = prompt
        previous_response = response
        callback_response = ''
        prompt = "What is my name?"
        response = model(prompt)
        assert isinstance(response, str)
        assert len(response) > 0
        assert 'Shane' not in response
        assert prompt in ''.join(model._previous_messages)
        assert previous_prompt not in ''.join(model._previous_messages)
        assert response == callback_response
        assert len(model.history()) == 2
        message = model.previous_record()
        assert isinstance(message, ExchangeRecord)
        assert message.prompt == prompt
        assert message.response == response
        history = model.history()
        assert history[0].prompt == previous_prompt
        assert history[0].response == previous_response
        assert history[1].prompt == prompt
        assert history[1].response == response
        assert message.metadata['endpoint_url'] == endpoint
        assert '[INST]' in message.metadata['messages']
        assert '[/INST]' in message.metadata['messages']
        assert '<<SYS>>' in message.metadata['messages']
        assert '<</SYS>>' in message.metadata['messages']
        assert prompt in message.metadata['messages']
        assert message.cost is None
        assert message.prompt_tokens > 0
        assert message.response_tokens > 0
        assert message.total_tokens == message.prompt_tokens + message.response_tokens
        assert message.uuid
        assert message.timestamp
        assert model.cost == 0
        assert model.sum(name='cost') == 0
        assert model.sum(name='total_tokens') == message.total_tokens + previous_tokens
