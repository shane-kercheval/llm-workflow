"""Test HuggingFace models and helpers."""

import os
import re
from unittest.mock import patch
from dotenv import load_dotenv


import pytest
from llm_workflow.hugging_face import (
    HuggingFaceEndpointChat,
    HuggingFaceRequestError,
    get_tokenizer,
    num_tokens,
    query_hugging_face_endpoint,
)
from llm_workflow.base import ExchangeRecord, StreamingEvent
from llm_workflow.memory import LastNExchangesManager, LastNTokensMemoryManager
from llm_workflow.message_formatters import llama_message_formatter
from tests.conftest import pattern_found


load_dotenv()


def test_query_hugging_face_endpoint(fake_retry_handler, fake_hugging_face_response):  # noqa
    with patch('llm_workflow.hugging_face.requests.post', return_value=fake_hugging_face_response) as mock_post:  # noqa
        with patch('llm_workflow.hugging_face.retry_handler', return_value=fake_retry_handler) as mock_retry_handler:  # noqa
            endpoint_url = "https://fake.url"
            payload = {"text": "hello"}
            response = query_hugging_face_endpoint(endpoint_url, payload)
            assert isinstance(response, list)
            assert 'generated_text' in response[0]


@pytest.mark.skipif(not os.environ.get('HUGGING_FACE_API_KEY'), reason="HUGGING_FACE_API_KEY is not set")  # noqa
@pytest.mark.skipif(not os.environ.get('HUGGING_FACE_ENDPOINT_UNIT_TESTS'), reason="HUGGING_FACE_ENDPOINT_UNIT_TESTS is not set")  # noqa
def test_HuggingFaceEndpointChat__no_token_calculator(hugging_face_endpoint):  # noqa
    callback_response = ''
    def streaming_callback(record: StreamingEvent) -> None:
        nonlocal callback_response
        callback_response += record.response

    model = HuggingFaceEndpointChat(
        endpoint_url=hugging_face_endpoint,
        message_formatter=llama_message_formatter,
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

@pytest.mark.skipif(not os.environ.get('HUGGING_FACE_API_KEY'), reason="HUGGING_FACE_API_KEY is not set")  # noqa
@pytest.mark.skipif(not os.environ.get('HUGGING_FACE_ENDPOINT_UNIT_TESTS'), reason="HUGGING_FACE_ENDPOINT_UNIT_TESTS is not set")  # noqa
def test_HuggingFaceEndpoint__with_parameters(hugging_face_endpoint):  # noqa
    # test valid parameters for non-streaming
    model_parameters = {'temperature': 0.01, 'max_tokens': 4096}
    model = HuggingFaceEndpointChat(
        endpoint_url=hugging_face_endpoint,
        message_formatter=llama_message_formatter,
        **model_parameters,
    )
    assert model.model_parameters == model_parameters
    response = model("What is the capital of France? Please respond in a single sentence.")
    assert 'Paris' in response
    assert model.history()[-1].metadata['model_parameters'] == model_parameters

    # test valid parameters for streaming
    callback_response = ''
    def streaming_callback(record: StreamingEvent) -> None:
        nonlocal callback_response
        callback_response += record.response

    model = HuggingFaceEndpointChat(
        endpoint_url=hugging_face_endpoint,
        message_formatter=llama_message_formatter,
        streaming_callback=streaming_callback,
        **model_parameters,
    )
    assert model.model_parameters == model_parameters
    response = model("What is the capital of France? Please respond in a single sentence.")
    assert 'Paris' in response
    assert response == callback_response
    assert model.history()[-1].metadata['model_parameters'] == model_parameters

    # test invalid parameters so that we know we're actually sending them
    model_parameters = {'temperature': -10}
    model = HuggingFaceEndpointChat(
        endpoint_url=hugging_face_endpoint,
        message_formatter=llama_message_formatter,
        **model_parameters,
    )
    assert model.model_parameters == model_parameters
    with pytest.raises(HuggingFaceRequestError) as exception:
        _ = model("What is the capital of France? Please respond in a single sentence.")
    exception = exception.value
    assert exception.error_type.lower() == 'validation'
    assert 'temperature' in exception.error_message

    # test invalid parameters for streaming
    model_parameters = {'temperature': -10}
    model = HuggingFaceEndpointChat(
        endpoint_url=hugging_face_endpoint,
        message_formatter=llama_message_formatter,
        streaming_callback=streaming_callback,
        **model_parameters,
    )
    assert model.model_parameters == model_parameters
    with pytest.raises(HuggingFaceRequestError) as exception:
        _ = model("What is the capital of France? Please respond in a single sentence.")
    exception = exception.value
    assert exception.error_type.lower() == 'validation'
    assert 'temperature' in exception.error_message

@pytest.mark.skipif(not os.environ.get('HUGGING_FACE_API_KEY'), reason="HUGGING_FACE_API_KEY is not set")  # noqa
@pytest.mark.skipif(not os.environ.get('HUGGING_FACE_ENDPOINT_UNIT_TESTS'), reason="HUGGING_FACE_ENDPOINT_UNIT_TESTS is not set")  # noqa
def test_HuggingFaceEndpointChat(hugging_face_endpoint):  # noqa
    tokenizer = get_tokenizer('meta-llama/Llama-2-7b-chat-hf')

    def calc_num_tokens(value: str):  # noqa
        return num_tokens(value, tokenizer)

    callback_response = ''
    def streaming_callback(record: StreamingEvent) -> None:
        nonlocal callback_response
        callback_response += record.response

    model = HuggingFaceEndpointChat(
        endpoint_url=hugging_face_endpoint,
        message_formatter=llama_message_formatter,
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

@pytest.mark.skipif(not os.environ.get('HUGGING_FACE_API_KEY'), reason="HUGGING_FACE_API_KEY is not set")  # noqa
@pytest.mark.skipif(not os.environ.get('HUGGING_FACE_ENDPOINT_UNIT_TESTS'), reason="HUGGING_FACE_ENDPOINT_UNIT_TESTS is not set")  # noqa
def test_HuggingFaceEndpointChat__timeout(hugging_face_endpoint):  # noqa
    callback_response = ''
    callback_count = 0
    def streaming_callback(record: StreamingEvent) -> None:
        nonlocal callback_response, callback_count
        callback_response += record.response
        callback_count += 1

    model = HuggingFaceEndpointChat(
        endpoint_url=hugging_face_endpoint,
        message_formatter=llama_message_formatter,
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

@pytest.mark.skipif(not os.environ.get('HUGGING_FACE_API_KEY'), reason="HUGGING_FACE_API_KEY is not set")  # noqa
@pytest.mark.skipif(not os.environ.get('HUGGING_FACE_ENDPOINT_UNIT_TESTS'), reason="HUGGING_FACE_ENDPOINT_UNIT_TESTS is not set")  # noqa
def test_HuggingFaceEndpointChat__memory_manager__1000_tokens(hugging_face_endpoint):  # noqa
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
    pattern = fr'^\[INST\].*?<<SYS>> {model.system_message} <</SYS>>.*?\[\/INST\]\n\[INST\].*?{previous_prompt}.*?\[\/INST\]\n.*?{re.escape(previous_response)}.*?\n\[INST\].*?{prompt}.*?\[\/INST\]'  # noqa
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

@pytest.mark.skipif(not os.environ.get('HUGGING_FACE_API_KEY'), reason="HUGGING_FACE_API_KEY is not set")  # noqa
@pytest.mark.skipif(not os.environ.get('HUGGING_FACE_ENDPOINT_UNIT_TESTS'), reason="HUGGING_FACE_ENDPOINT_UNIT_TESTS is not set")  # noqa
def test_HuggingFaceEndpointChat__memory_manager__1_tokens(hugging_face_endpoint):  # noqa
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

@pytest.mark.skipif(not os.environ.get('HUGGING_FACE_API_KEY'), reason="HUGGING_FACE_API_KEY is not set")  # noqa
@pytest.mark.skipif(not os.environ.get('HUGGING_FACE_ENDPOINT_UNIT_TESTS'), reason="HUGGING_FACE_ENDPOINT_UNIT_TESTS is not set")  # noqa
def test_HuggingFaceEndpointChat__memory_manager__100_tokens(hugging_face_endpoint):  # noqa
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

@pytest.mark.skipif(not os.environ.get('HUGGING_FACE_API_KEY'), reason="HUGGING_FACE_API_KEY is not set")  # noqa
@pytest.mark.skipif(not os.environ.get('HUGGING_FACE_ENDPOINT_UNIT_TESTS'), reason="HUGGING_FACE_ENDPOINT_UNIT_TESTS is not set")  # noqa
def test_HuggingFaceEndpointChat__memory_manager__LastNExchangesManager_1(hugging_face_endpoint):  # noqa
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
    pattern = fr'^\[INST\].*?<<SYS>> {model.system_message} <</SYS>>.*?\[\/INST\]\n\[INST\].*?{previous_prompt}.*?\[\/INST\]\n.*?{re.escape(previous_response)}.*?\n\[INST\].*?{prompt}.*?\[\/INST\]'  # noqa
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

@pytest.mark.skipif(not os.environ.get('HUGGING_FACE_API_KEY'), reason="HUGGING_FACE_API_KEY is not set")  # noqa
@pytest.mark.skipif(not os.environ.get('HUGGING_FACE_ENDPOINT_UNIT_TESTS'), reason="HUGGING_FACE_ENDPOINT_UNIT_TESTS is not set")  # noqa
def test_HuggingFaceEndpointChat__memory_manager__LastNExchangesManager_0(hugging_face_endpoint):  # noqa
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
    # pattern = fr'^\[INST\].*?<<SYS>> {model.system_message} <</SYS>>.*?\[\/INST\]\n\[INST\].*?{previous_prompt}.*?\[\/INST\]\n.*?{re.escape(previous_response)}.*?\n\[INST\].*?{prompt}.*?\[\/INST\]'  # noqa
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
