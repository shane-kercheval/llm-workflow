"""Test HuggingFace models and helpers."""

from unittest.mock import patch

import pytest
from llm_workflow.hugging_face import (
    HuggingFaceEndpointChat,
    get_tokenizer,
    llama_message_formatter,
    num_tokens,
    query_hugging_face_endpoint,
)
from llm_workflow.base import ExchangeRecord, StreamingEvent
from llm_workflow.memory import LastNExchangesManager, LastNTokensMemoryManager
from tests.conftest import is_endpoint_available, pattern_found


def test_query_hugging_face_endpoint(fake_retry_handler, fake_hugging_face_response):  # noqa
    with patch('llm_workflow.hugging_face.requests.post', return_value=fake_hugging_face_response) as mock_post:  # noqa
        with patch('llm_workflow.hugging_face.retry_handler', return_value=fake_retry_handler) as mock_retry_handler:  # noqa
            endpoint_url = "https://fake.url"
            payload = {"text": "hello"}
            response = query_hugging_face_endpoint(endpoint_url, payload)
            assert isinstance(response, list)
            assert 'generated_text' in response[0]

def test_llama_message_formatter():  # noqa
    assert llama_message_formatter(system_message=None, history=[], prompt=None) == ''
    assert llama_message_formatter(system_message=None, history=None, prompt=None) == ''

    messages = llama_message_formatter(
        system_message=None,
        history=[
            ExchangeRecord(prompt='a', response='b'),
            ExchangeRecord(prompt='c', response='d'),
        ],
        prompt=None,
    )
    expected_value = '[INST] a [/INST]\nb\n[INST] c [/INST]\nd\n'
    assert messages == '[INST] a [/INST]\nb\n[INST] c [/INST]\nd\n'

    messages = llama_message_formatter(
        system_message='system',
        history=[
            ExchangeRecord(prompt='a', response='b'),
            ExchangeRecord(prompt='c', response='d'),
        ],
        prompt=None,
    )
    assert messages == '[INST] <<SYS>> system <</SYS>> [/INST]\n' + expected_value

    messages = llama_message_formatter(
        system_message='system',
        history=[
            ExchangeRecord(prompt='a', response='b'),
            ExchangeRecord(prompt='c', response='d'),
        ],
        prompt='e',
    )
    assert messages == f'[INST] <<SYS>> system <</SYS>> [/INST]\n{expected_value}[INST] e [/INST]\n'  # noqa

def test_HuggingFaceEndpointChat__no_token_calculator(hugging_face_endpoint):  # noqa
    if is_endpoint_available(hugging_face_endpoint):
        callback_response = ''
        def streaming_callback(record: StreamingEvent) -> None:
            nonlocal callback_response
            callback_response += record.response

        model = HuggingFaceEndpointChat(
            endpoint_url=hugging_face_endpoint,
            streaming_callback=streaming_callback,
        )
        assert len(model.history()) == 0
        assert model.previous_record() is None
        assert model.previous_prompt is None
        assert model.previous_response is None
        assert model.cost == 0
        assert model.total_tokens == 0
        assert model.input_tokens == 0
        assert model.response_tokens == 0

        prompt = "My name is Shane. What's your name?"
        response = model(prompt)
        assert isinstance(response, str)
        assert len(response) > 0
        assert response == callback_response
        assert len(model.history()) == 1
        assert len(model.chat_history) == 1
        assert model.history() == model.chat_history
        message = model.previous_record()
        assert isinstance(message, ExchangeRecord)
        assert message.prompt == prompt
        assert message.response == response
        history = model.history()
        assert history[0].prompt == prompt
        assert history[0].response == response
        assert message.metadata['endpoint_url'] == hugging_face_endpoint
        pattern = fr'^\[INST\].*?<<SYS>> {model.system_message} <</SYS>>.*?\[\/INST\]\n\[INST\].*?{prompt}.*?\[\/INST\]'  # noqa
        assert pattern_found(message.metadata['messages'], pattern)
        assert message.metadata['messages'].count('<<SYS>>') == 1
        assert message.metadata['messages'].count('<</SYS>>') == 1
        assert message.metadata['messages'].count('[INST]') == 2
        assert message.metadata['messages'].count('[/INST]') == 2
        assert message.metadata['messages'].count(prompt) == 1
        assert message.cost is None
        assert message.input_tokens == len(message.metadata['messages'])
        assert message.response_tokens == len(response)
        assert message.total_tokens == message.input_tokens + message.response_tokens
        assert message.uuid
        assert message.timestamp
        assert model.cost == 0
        assert model.sum(name='cost') == 0
        assert model.sum(name='total_tokens') == message.total_tokens

def test_HuggingFaceEndpointChat(hugging_face_endpoint):  # noqa
    if is_endpoint_available(hugging_face_endpoint):
        tokenizer = get_tokenizer('meta-llama/Llama-2-7b-chat-hf')

        def calc_num_tokens(value: str):  # noqa
            return num_tokens(value, tokenizer)

        callback_response = ''
        def streaming_callback(record: StreamingEvent) -> None:
            nonlocal callback_response
            callback_response += record.response

        model = HuggingFaceEndpointChat(
            endpoint_url=hugging_face_endpoint,
            token_calculator=calc_num_tokens,
            streaming_callback=streaming_callback,
        )
        assert len(model.history()) == 0
        assert model.previous_record() is None
        assert model.previous_prompt is None
        assert model.previous_response is None
        assert model.cost == 0
        assert model.total_tokens == 0
        assert model.input_tokens == 0
        assert model.response_tokens == 0

        prompt = "My name is Shane. What's your name?"
        response = model(prompt)
        assert isinstance(response, str)
        assert len(response) > 0
        assert response == callback_response
        assert len(model.history()) == 1
        assert len(model.chat_history) == 1
        assert model.history() == model.chat_history
        assert model.chat_history[0].prompt == prompt
        assert model.chat_history[0].response == response

        message = model.previous_record()
        assert isinstance(message, ExchangeRecord)
        assert message.prompt == prompt
        assert message.response == response
        history = model.history()
        assert history[0].prompt == prompt
        assert history[0].response == response
        assert message.metadata['endpoint_url'] == hugging_face_endpoint
        pattern = fr'^\[INST\].*?<<SYS>> {model.system_message} <</SYS>>.*?\[\/INST\]\n\[INST\].*?{prompt}.*?\[\/INST\]'  # noqa
        assert pattern_found(message.metadata['messages'], pattern)
        assert message.metadata['messages'].count('<<SYS>>') == 1
        assert message.metadata['messages'].count('<</SYS>>') == 1
        assert message.metadata['messages'].count('[INST]') == 2
        assert message.metadata['messages'].count('[/INST]') == 2
        assert message.metadata['messages'].count(prompt) == 1
        assert message.input_tokens == calc_num_tokens(message.metadata['messages'])
        assert message.response_tokens == calc_num_tokens(response)
        assert message.cost is None
        assert message.total_tokens == message.input_tokens + message.response_tokens
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
        history = model.history()
        assert history[0].prompt == previous_prompt
        assert history[0].response == previous_response
        assert history[1].prompt == prompt
        assert history[1].response == response
        assert message.metadata['endpoint_url'] == hugging_face_endpoint
        pattern = fr'^\[INST\].*?<<SYS>> {model.system_message} <</SYS>>.*?\[\/INST\]\n\[INST\].*?{previous_prompt}.*?\[\/INST\]\n.*?\n\[INST\].*?{prompt}.*?\[\/INST\]'  # noqa
        assert pattern_found(message.metadata['messages'], pattern)
        assert message.metadata['messages'].count('<<SYS>>') == 1
        assert message.metadata['messages'].count('<</SYS>>') == 1
        assert message.metadata['messages'].count('[INST]') == 3
        assert message.metadata['messages'].count('[/INST]') == 3
        assert message.metadata['messages'].count(previous_prompt) == 1
        assert message.metadata['messages'].count(previous_response) == 1
        assert message.metadata['messages'].count(prompt) == 1
        assert message.cost is None
        assert message.input_tokens == calc_num_tokens(message.metadata['messages'])
        assert message.response_tokens == calc_num_tokens(response)
        assert message.total_tokens == message.input_tokens + message.response_tokens
        assert message.uuid
        assert message.timestamp
        assert model.cost == 0
        assert model.sum(name='cost') == 0
        assert model.sum(name='total_tokens') == message.total_tokens + previous_tokens

def test_HuggingFaceEndpointChat__timeout(hugging_face_endpoint):  # noqa
    if is_endpoint_available(hugging_face_endpoint):
        callback_response = ''
        callback_count = 0
        def streaming_callback(record: StreamingEvent) -> None:
            nonlocal callback_response, callback_count
            callback_response += record.response
            callback_count += 1

        model = HuggingFaceEndpointChat(
            endpoint_url=hugging_face_endpoint,
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
        assert model.input_tokens == 0
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
        assert message.metadata['endpoint_url'] == hugging_face_endpoint
        assert '[INST]' in message.metadata['messages']
        assert '[/INST]' in message.metadata['messages']
        assert '<<SYS>>' in message.metadata['messages']
        assert '<</SYS>>' in message.metadata['messages']
        assert prompt in message.metadata['messages']
        assert message.cost is None

def test_HuggingFaceEndpointChat__memory_manager__1000_tokens(hugging_face_endpoint):  # noqa
    if is_endpoint_available(hugging_face_endpoint):
        tokenizer = get_tokenizer('meta-llama/Llama-2-7b-chat-hf')

        def calc_num_tokens(value: str):  # noqa
            return num_tokens(value, tokenizer)

        callback_response = ''
        def streaming_callback(record: StreamingEvent) -> None:
            nonlocal callback_response
            callback_response += record.response

        model = HuggingFaceEndpointChat(
            endpoint_url=hugging_face_endpoint,
            streaming_callback=streaming_callback,
            token_calculator=calc_num_tokens,
            message_formatter=llama_message_formatter,
            memory_manager=LastNTokensMemoryManager(last_n_tokens=1000),
        )
        assert len(model.history()) == 0
        assert model.previous_record() is None
        assert model.previous_prompt is None
        assert model.previous_response is None
        assert model.cost == 0
        assert model.total_tokens == 0
        assert model.input_tokens == 0
        assert model.response_tokens == 0

        prompt = "My name is Shane. What's your name?"
        response = model(prompt)
        assert isinstance(response, str)
        assert len(response) > 0
        assert response == callback_response
        assert len(model.history()) == 1
        assert len(model.chat_history) == 1
        assert model.history() == model.chat_history
        assert model.chat_history[0].prompt == prompt
        assert model.chat_history[0].response == response

        message = model.previous_record()
        assert isinstance(message, ExchangeRecord)
        assert message.prompt == prompt
        assert message.response == response
        history = model.history()
        assert history[0].prompt == prompt
        assert history[0].response == response
        assert message.metadata['endpoint_url'] == hugging_face_endpoint
        pattern = fr'^\[INST\].*?<<SYS>> {model.system_message} <</SYS>>.*?\[\/INST\]\n\[INST\].*?{prompt}.*?\[\/INST\]'  # noqa
        assert pattern_found(message.metadata['messages'], pattern)
        assert message.metadata['messages'].count('<<SYS>>') == 1
        assert message.metadata['messages'].count('<</SYS>>') == 1
        assert message.metadata['messages'].count('[INST]') == 2
        assert message.metadata['messages'].count('[/INST]') == 2
        assert message.metadata['messages'].count(prompt) == 1
        assert message.cost is None
        assert message.input_tokens == calc_num_tokens(message.metadata['messages'])
        assert message.response_tokens == calc_num_tokens(response)
        assert message.total_tokens == message.input_tokens + message.response_tokens
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
        history = model.history()
        assert history[0].prompt == previous_prompt
        assert history[0].response == previous_response
        assert history[1].prompt == prompt
        assert history[1].response == response
        assert message.metadata['endpoint_url'] == hugging_face_endpoint
        pattern = fr'^\[INST\].*?<<SYS>> {model.system_message} <</SYS>>.*?\[\/INST\]\n\[INST\].*?{previous_prompt}.*?\[\/INST\]\n.*?{previous_response}.*?\n\[INST\].*?{prompt}.*?\[\/INST\]'  # noqa
        assert pattern_found(message.metadata['messages'], pattern)
        assert message.metadata['messages'].count('<<SYS>>') == 1
        assert message.metadata['messages'].count('<</SYS>>') == 1
        assert message.metadata['messages'].count('[INST]') == 3
        assert message.metadata['messages'].count('[/INST]') == 3
        assert message.metadata['messages'].count(prompt) == 1
        assert message.cost is None
        assert message.input_tokens == calc_num_tokens(message.metadata['messages'])
        assert message.response_tokens == calc_num_tokens(response)
        assert message.total_tokens == message.input_tokens + message.response_tokens
        assert message.uuid
        assert message.timestamp
        assert model.cost == 0
        assert model.sum(name='cost') == 0
        assert model.sum(name='total_tokens') == message.total_tokens + previous_tokens

def test_HuggingFaceEndpointChat__memory_manager__1_tokens(hugging_face_endpoint):  # noqa
    if is_endpoint_available(hugging_face_endpoint):
        tokenizer = get_tokenizer('meta-llama/Llama-2-7b-chat-hf')

        def calc_num_tokens(value: str):  # noqa
            return num_tokens(value, tokenizer)

        model = HuggingFaceEndpointChat(
            endpoint_url=hugging_face_endpoint,
            token_calculator=calc_num_tokens,
            message_formatter=llama_message_formatter,
            memory_manager=LastNTokensMemoryManager(last_n_tokens=20),
        )
        assert len(model.history()) == 0
        assert model.previous_record() is None
        assert model.previous_prompt is None
        assert model.previous_response is None
        assert model.cost == 0
        assert model.total_tokens == 0
        assert model.input_tokens == 0
        assert model.response_tokens == 0

        prompt = "My name is Shane and my favorite color is blue. What's your name?"
        with pytest.raises(AssertionError):
            _ = model(prompt)

def test_HuggingFaceEndpointChat__memory_manager__100_tokens(hugging_face_endpoint):  # noqa
    if is_endpoint_available(hugging_face_endpoint):
        tokenizer = get_tokenizer('meta-llama/Llama-2-7b-chat-hf')

        def calc_num_tokens(value: str):  # noqa
            return num_tokens(value, tokenizer)

        callback_response = ''
        def streaming_callback(record: StreamingEvent) -> None:
            nonlocal callback_response
            callback_response += record.response

        model = HuggingFaceEndpointChat(
            endpoint_url=hugging_face_endpoint,
            token_calculator=calc_num_tokens,
            streaming_callback=streaming_callback,
            message_formatter=llama_message_formatter,
            memory_manager=LastNTokensMemoryManager(last_n_tokens=75),
        )
        assert len(model.history()) == 0
        assert model.previous_record() is None
        assert model.previous_prompt is None
        assert model.previous_response is None
        assert model.cost == 0
        assert model.total_tokens == 0
        assert model.input_tokens == 0
        assert model.response_tokens == 0

        prompt = "My name is Shane and my favorite color is blue. What's your name?"
        response = model(prompt)
        assert isinstance(response, str)
        assert len(response) > 0
        assert response == callback_response
        assert len(model.history()) == 1
        assert len(model.chat_history) == 1
        assert model.history() == model.chat_history
        assert model.chat_history[0].prompt == prompt
        assert model.chat_history[0].response == response

        message = model.previous_record()
        assert isinstance(message, ExchangeRecord)
        assert message.prompt == prompt
        assert message.response == response
        history = model.history()
        assert history[0].prompt == prompt
        assert history[0].response == response
        assert message.metadata['endpoint_url'] == hugging_face_endpoint
        pattern = fr'^\[INST\].*?<<SYS>> {model.system_message} <</SYS>>.*?\[\/INST\]\n\[INST\].*?{prompt}.*?\[\/INST\]'  # noqa
        assert pattern_found(message.metadata['messages'], pattern)
        assert message.metadata['messages'].count('<<SYS>>') == 1
        assert message.metadata['messages'].count('<</SYS>>') == 1
        assert message.metadata['messages'].count('[INST]') == 2
        assert message.metadata['messages'].count('[/INST]') == 2
        assert message.metadata['messages'].count(prompt) == 1
        assert message.cost is None
        assert message.input_tokens == calc_num_tokens(message.metadata['messages'])
        assert message.response_tokens == calc_num_tokens(response)
        assert message.total_tokens == message.input_tokens + message.response_tokens
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
        history = model.history()
        assert history[0].prompt == previous_prompt
        assert history[0].response == previous_response
        assert history[1].prompt == prompt
        assert history[1].response == response
        assert message.metadata['endpoint_url'] == hugging_face_endpoint
        pattern = fr'^\[INST\].*?<<SYS>> {model.system_message} <</SYS>>.*?\[\/INST\]\n\[INST\].*?{prompt}.*?\[\/INST\]'  # noqa
        assert pattern_found(message.metadata['messages'], pattern)
        assert message.metadata['messages'].count('<<SYS>>') == 1
        assert message.metadata['messages'].count('<</SYS>>') == 1
        assert message.metadata['messages'].count('[INST]') == 2
        assert message.metadata['messages'].count('[/INST]') == 2
        assert message.metadata['messages'].count(prompt) == 1
        assert message.cost is None
        assert message.input_tokens == calc_num_tokens(message.metadata['messages'])
        assert message.response_tokens == calc_num_tokens(response)
        assert message.total_tokens == message.input_tokens + message.response_tokens
        assert message.uuid
        assert message.timestamp
        assert model.cost == 0
        assert model.sum(name='cost') == 0
        assert model.sum(name='total_tokens') == message.total_tokens + previous_tokens

def test_HuggingFaceEndpointChat__memory_manager__LastNExchangesManager_1(hugging_face_endpoint):  # noqa
    if is_endpoint_available(hugging_face_endpoint):
        tokenizer = get_tokenizer('meta-llama/Llama-2-7b-chat-hf')

        def calc_num_tokens(value: str):  # noqa
            return num_tokens(value, tokenizer)

        callback_response = ''
        def streaming_callback(record: StreamingEvent) -> None:
            nonlocal callback_response
            callback_response += record.response

        model = HuggingFaceEndpointChat(
            endpoint_url=hugging_face_endpoint,
            streaming_callback=streaming_callback,
            token_calculator=calc_num_tokens,
            message_formatter=llama_message_formatter,
            memory_manager=LastNExchangesManager(last_n_exchanges=1),
        )
        assert len(model.history()) == 0
        assert model.previous_record() is None
        assert model.previous_prompt is None
        assert model.previous_response is None
        assert model.cost == 0
        assert model.total_tokens == 0
        assert model.input_tokens == 0
        assert model.response_tokens == 0

        prompt = "My name is Shane. What's your name?"
        response = model(prompt)
        assert isinstance(response, str)
        assert len(response) > 0
        assert response == callback_response
        assert len(model.history()) == 1
        assert len(model.chat_history) == 1
        assert model.history() == model.chat_history
        assert model.chat_history[0].prompt == prompt
        assert model.chat_history[0].response == response
        message = model.previous_record()
        assert isinstance(message, ExchangeRecord)
        assert message.prompt == prompt
        assert message.response == response
        history = model.history()
        assert history[0].prompt == prompt
        assert history[0].response == response
        assert message.metadata['endpoint_url'] == hugging_face_endpoint
        pattern = fr'^\[INST\].*?<<SYS>> {model.system_message} <</SYS>>.*?\[\/INST\]\n\[INST\].*?{prompt}.*?\[\/INST\]'  # noqa
        assert pattern_found(message.metadata['messages'], pattern)
        assert message.metadata['messages'].count('<<SYS>>') == 1
        assert message.metadata['messages'].count('<</SYS>>') == 1
        assert message.metadata['messages'].count('[INST]') == 2
        assert message.metadata['messages'].count('[/INST]') == 2
        assert message.metadata['messages'].count(prompt) == 1
        assert message.cost is None
        assert message.input_tokens == calc_num_tokens(message.metadata['messages'])
        assert message.response_tokens == calc_num_tokens(response)
        assert message.total_tokens == message.input_tokens + message.response_tokens
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
        history = model.history()
        assert history[0].prompt == previous_prompt
        assert history[0].response == previous_response
        assert history[1].prompt == prompt
        assert history[1].response == response
        assert message.metadata['endpoint_url'] == hugging_face_endpoint
        pattern = fr'^\[INST\].*?<<SYS>> {model.system_message} <</SYS>>.*?\[\/INST\]\n\[INST\].*?{previous_prompt}.*?\[\/INST\]\n.*?{previous_response}.*?\n\[INST\].*?{prompt}.*?\[\/INST\]'  # noqa
        assert pattern_found(message.metadata['messages'], pattern)
        assert message.metadata['messages'].count('<<SYS>>') == 1
        assert message.metadata['messages'].count('<</SYS>>') == 1
        assert message.metadata['messages'].count('[INST]') == 3
        assert message.metadata['messages'].count('[/INST]') == 3
        assert message.metadata['messages'].count(prompt) == 1
        assert message.cost is None
        assert message.input_tokens == calc_num_tokens(message.metadata['messages'])
        assert message.response_tokens == calc_num_tokens(response)
        assert message.total_tokens == message.input_tokens + message.response_tokens
        assert message.uuid
        assert message.timestamp
        assert model.cost == 0
        assert model.sum(name='cost') == 0
        assert model.sum(name='total_tokens') == message.total_tokens + previous_tokens

def test_HuggingFaceEndpointChat__memory_manager__LastNExchangesManager_0(hugging_face_endpoint):  # noqa
    if is_endpoint_available(hugging_face_endpoint):
        tokenizer = get_tokenizer('meta-llama/Llama-2-7b-chat-hf')

        def calc_num_tokens(value: str):  # noqa
            return num_tokens(value, tokenizer)

        callback_response = ''
        def streaming_callback(record: StreamingEvent) -> None:
            nonlocal callback_response
            callback_response += record.response

        model = HuggingFaceEndpointChat(
            endpoint_url=hugging_face_endpoint,
            streaming_callback=streaming_callback,
            token_calculator=calc_num_tokens,
            message_formatter=llama_message_formatter,
            memory_manager=LastNExchangesManager(last_n_exchanges=0),
        )
        assert len(model.history()) == 0
        assert model.previous_record() is None
        assert model.previous_prompt is None
        assert model.previous_response is None
        assert model.cost == 0
        assert model.total_tokens == 0
        assert model.input_tokens == 0
        assert model.response_tokens == 0

        prompt = "My name is Shane. What's your name?"
        response = model(prompt)
        assert isinstance(response, str)
        assert len(response) > 0
        assert response == callback_response
        assert len(model.history()) == 1
        assert len(model.chat_history) == 1
        assert model.history() == model.chat_history
        assert model.chat_history[0].prompt == prompt
        assert model.chat_history[0].response == response
        message = model.previous_record()
        assert isinstance(message, ExchangeRecord)
        assert message.prompt == prompt
        assert message.response == response
        history = model.history()
        assert history[0].prompt == prompt
        assert history[0].response == response
        assert message.metadata['endpoint_url'] == hugging_face_endpoint
        pattern = fr'^\[INST\].*?<<SYS>> {model.system_message} <</SYS>>.*?\[\/INST\]\n\[INST\].*?{prompt}.*?\[\/INST\]'  # noqa
        assert pattern_found(message.metadata['messages'], pattern)
        assert message.metadata['messages'].count('<<SYS>>') == 1
        assert message.metadata['messages'].count('<</SYS>>') == 1
        assert message.metadata['messages'].count('[INST]') == 2
        assert message.metadata['messages'].count('[/INST]') == 2
        assert message.metadata['messages'].count(prompt) == 1
        assert message.cost is None
        assert message.input_tokens == calc_num_tokens(message.metadata['messages'])
        assert message.response_tokens == calc_num_tokens(response)
        assert message.total_tokens == message.input_tokens + message.response_tokens
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
        assert 'Shane' not in response  # no history
        assert response == callback_response
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
        history = model.history()
        assert history[0].prompt == previous_prompt
        assert history[0].response == previous_response
        assert history[1].prompt == prompt
        assert history[1].response == response
        assert message.metadata['endpoint_url'] == hugging_face_endpoint
        # pattern = fr'^\[INST\].*?<<SYS>> {model.system_message} <</SYS>>.*?\[\/INST\]\n\[INST\].*?{previous_prompt}.*?\[\/INST\]\n.*?{previous_response}.*?\n\[INST\].*?{prompt}.*?\[\/INST\]'  # noqa
        # we should not see the previous prompt or response in the messages
        pattern = fr'^\[INST\].*?<<SYS>> {model.system_message} <</SYS>>.*?\[\/INST\]\n\[INST\].*?{prompt}.*?\[\/INST\]'  # noqa
        assert pattern_found(message.metadata['messages'], pattern)
        assert message.metadata['messages'].count('<<SYS>>') == 1
        assert message.metadata['messages'].count('<</SYS>>') == 1
        assert message.metadata['messages'].count('[INST]') == 2
        assert message.metadata['messages'].count('[/INST]') == 2
        assert message.metadata['messages'].count(prompt) == 1
        assert message.cost is None
        assert message.input_tokens == calc_num_tokens(message.metadata['messages'])
        assert message.response_tokens == calc_num_tokens(response)
        assert message.total_tokens == message.input_tokens + message.response_tokens
        assert message.uuid
        assert message.timestamp
        assert model.cost == 0
        assert model.sum(name='cost') == 0
        assert model.sum(name='total_tokens') == message.total_tokens + previous_tokens
