"""Test the LlamaCppEndpointChat class."""
import os
import pytest
from llm_workflow.base import ExchangeRecord, StreamingEvent
from llm_workflow.llama_cpp_endpoint import LlamaCppEndpointChat
from llm_workflow.message_formatters import mistral_message_formatter
from tests.conftest import pattern_found

@pytest.mark.skipif(not os.environ.get('LLAMA_CPP_ENDPOINT_UNIT_TESTS'), reason="LLAMA_CPP_ENDPOINT_UNIT_TESTS is not set")  # noqa
def test__LlamaCppEndpointChat__no_params_no_streaming(llama_cpp_endpoint):  # noqa
    system_message = "You are a helpful coding AI assistant."
    chat = LlamaCppEndpointChat(
        endpoint_url=llama_cpp_endpoint,
        system_message=system_message,
        message_formatter=mistral_message_formatter,
    )
    question = "Write a python function to mask emails called `mask_email`."
    response = chat(question)
    assert 'def mask_email(' in response
    prompt = chat.history()[-1].metadata['parameters']['prompt']
    assert prompt.count(system_message) == 1
    assert prompt.count(f'[INST]{question}[/INST]') == 1

    follow_up = "Write assert statements to test the function."
    response = chat(follow_up)
    prompt = chat.history()[-1].metadata['parameters']['prompt']
    # The system prompt, original questiona, original original response (i.e. function
    # definition) should only appear once in the prompt
    assert prompt.count(system_message) == 1
    assert prompt.count(f'[INST]{question}[/INST]') == 1
    assert prompt.count('def mask_email(') == 1
    assert prompt.count(f'[INST]{follow_up}[/INST]') == 1

@pytest.mark.skipif(not os.environ.get('LLAMA_CPP_ENDPOINT_UNIT_TESTS'), reason="LLAMA_CPP_ENDPOINT_UNIT_TESTS is not set")  # noqa
def test__LlamaCppEndpointChat__no_params_streaming(llama_cpp_endpoint):  # noqa
    callback_response = ''
    def streaming_callback(record: StreamingEvent) -> None:
        nonlocal callback_response
        callback_response += record.response

    system_message = "You are a helpful coding AI assistant."
    chat = LlamaCppEndpointChat(
        endpoint_url=llama_cpp_endpoint,
        system_message=system_message,
        message_formatter=mistral_message_formatter,
        streaming_callback=streaming_callback,
    )
    question = "Write a python function to mask emails called `mask_email`."
    response = chat(question)
    assert response == callback_response
    assert 'def mask_email(' in response
    prompt = chat.history()[-1].metadata['parameters']['prompt']
    assert prompt.count(system_message) == 1
    assert prompt.count(f'[INST]{question}[/INST]') == 1

    follow_up = "Write assert statements to test the function."
    callback_response = ''
    response = chat(follow_up)
    assert response == callback_response
    prompt = chat.history()[-1].metadata['parameters']['prompt']
    # The system prompt, original questiona, original original response (i.e. function
    # definition) should only appear once in the prompt
    assert prompt.count(system_message) == 1
    assert prompt.count(f'[INST]{question}[/INST]') == 1
    assert prompt.count('def mask_email(') == 1
    assert prompt.count(f'[INST]{follow_up}[/INST]') == 1

@pytest.mark.skipif(not os.environ.get('LLAMA_CPP_ENDPOINT_UNIT_TESTS'), reason="LLAMA_CPP_ENDPOINT_UNIT_TESTS is not set")  # noqa
def test__LlamaCppEndpointChat__params__no_streaming(llama_cpp_endpoint):  # noqa
    system_message = "You are a helpful coding AI assistant."
    question = "Write a python function to mask emails called `mask_email`."
    chat = LlamaCppEndpointChat(
        endpoint_url=llama_cpp_endpoint,
        system_message=system_message,
        # streaming_callback=streaming_callback,
        message_formatter=mistral_message_formatter,
        parameters={
            'temperature': 0.2,
            'n_predict': -1,
            'top_k': 40,
            'repeat_penalty': 1.1,
            'min_p': 0.05,
            'top_p': 0.95,
        },
    )
    response = chat(question)
    assert 'def mask_email(' in response
    prompt = chat.history()[-1].metadata['parameters']['prompt']
    assert prompt.count(system_message) == 1
    assert prompt.count(f'[INST]{question}[/INST]') == 1

    settings = chat.history()[-1].metadata['generation_settings']
    assert round(settings['temperature'], 5) == 0.2
    assert settings['n_predict'] == -1
    assert settings['top_k'] == 40
    assert round(settings['repeat_penalty'], 5) == 1.1
    assert round(settings['min_p'], 5) == 0.05
    assert round(settings['top_p'], 5) == 0.95

@pytest.mark.skipif(not os.environ.get('LLAMA_CPP_ENDPOINT_UNIT_TESTS'), reason="LLAMA_CPP_ENDPOINT_UNIT_TESTS is not set")  # noqa
def test_LlamaCppEndpointChat__no_token_calculator(llama_cpp_endpoint):  # noqa
    callback_response = ''
    def streaming_callback(record: StreamingEvent) -> None:
        nonlocal callback_response
        callback_response += record.response

    model = LlamaCppEndpointChat(
        endpoint_url=llama_cpp_endpoint,
        streaming_callback=streaming_callback,
        message_formatter=mistral_message_formatter,
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
    assert message.metadata['endpoint_url'] == llama_cpp_endpoint
    pattern = fr'{model.system_message}.*?\[INST\].*?{prompt}.*?\[\/INST\]'
    assert pattern_found(message.metadata['messages'], pattern)
    assert message.metadata['messages'].count('[INST]') == 1
    assert message.metadata['messages'].count('[/INST]') == 1
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

@pytest.mark.skipif(not os.environ.get('LLAMA_CPP_ENDPOINT_UNIT_TESTS'), reason="LLAMA_CPP_ENDPOINT_UNIT_TESTS is not set")  # noqa
def test_LlamaCppEndpointChat(llama_cpp_endpoint):  # noqa
    def calc_num_tokens(value: str):  # noqa
        return len(value)

    callback_response = ''
    def streaming_callback(record: StreamingEvent) -> None:
        nonlocal callback_response
        callback_response += record.response

    model = LlamaCppEndpointChat(
        endpoint_url=llama_cpp_endpoint,
        token_calculator=calc_num_tokens,
        message_formatter=mistral_message_formatter,
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
    assert message.metadata['endpoint_url'] == llama_cpp_endpoint
    pattern = fr'{model.system_message}.*?\[INST\].*?{prompt}.*?\[\/INST\]'
    assert pattern_found(message.metadata['messages'], pattern)
    assert message.metadata['messages'].count('[INST]') == 1
    assert message.metadata['messages'].count('[/INST]') == 1
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
    assert response == callback_response.strip()
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
    assert message.metadata['endpoint_url'] == llama_cpp_endpoint
    pattern = fr'^\[INST\].*?<<SYS>> {model.system_message} <</SYS>>.*?\[\/INST\]\n\[INST\].*?{previous_prompt}.*?\[\/INST\]\n.*?\n\[INST\].*?{prompt}.*?\[\/INST\]'  # noqa
    pattern = fr'{model.system_message}.*?\[INST\].*?{previous_prompt}.*?\[\/INST\].*?\n\n\[INST\].*?{prompt}.*?\[\/INST\]'  # noqa
    assert pattern_found(message.metadata['messages'], pattern)
    assert message.metadata['messages'].count('[INST]') == 2
    assert message.metadata['messages'].count('[/INST]') == 2
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
