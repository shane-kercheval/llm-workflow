
"""Tests the message formatters module."""
from textwrap import dedent
from llm_workflow.base import ExchangeRecord
from llm_workflow.message_formatters import (
    PROMPT_FORMAT_LLAMA,
    RESPONSE_PREFIX_LLAMA,
    SYSTEM_FORMAT_LLAMA,
    LlamaMessageFormatter,
    MessageFormatter,
)


def test__MessageFormatter__empty():  # noqa
    message_formatter = MessageFormatter(
        system_format=None,
        prompt_format=None,
        response_prefix=None,
    )
    assert message_formatter(None, None, None) == ''
    message_formatter = MessageFormatter(
        system_format=SYSTEM_FORMAT_LLAMA,
        prompt_format=PROMPT_FORMAT_LLAMA,
        response_prefix=RESPONSE_PREFIX_LLAMA,
    )
    assert message_formatter(None, None, None) == ''

def test__MessageFormatter__zephyr_syntax():  # noqa
    message_formatter = MessageFormatter(
        system_format='<|system|>\n{system_message}\n',
        prompt_format='\n<|user|>\n{prompt}\n',
        response_prefix='<|assistant|>\n',
    )
    system = 'You are a helpful AI assistant.'
    prompt = 'What is the capital of France?'
    response_1 = 'The capital of France is Paris.'
    prompt_2 = 'What is the capital of Spain?'

    assert message_formatter(system, [], None) == f'<|system|>\n{system}\n'
    expected_value = dedent(f"""
        <|system|>
        {system}\n
        <|user|>
        {prompt}
        <|assistant|>
        """).lstrip()
    assert message_formatter(system, [], prompt) == expected_value
    expected_value = dedent(f"""
        <|system|>
        {system}\n
        <|user|>
        {prompt}
        <|assistant|>
        {response_1}
        <|user|>
        {prompt_2}
        <|assistant|>
        """).lstrip()
    actual_value = message_formatter(
        system,
        [ExchangeRecord(prompt=prompt, response=response_1)],
        prompt_2,
    )
    assert actual_value == expected_value
    actual_value = message_formatter(
        system,
        [(prompt, response_1)],
        prompt_2,
    )
    assert actual_value == expected_value

def test__MessageFormatter():  # noqa
    message_formatter = MessageFormatter(
        system_format=SYSTEM_FORMAT_LLAMA,
        prompt_format=PROMPT_FORMAT_LLAMA,
        response_prefix=RESPONSE_PREFIX_LLAMA,
    )
    system = 'test system'
    message_prompt = 'test message prompt'
    message_response = 'test message response'
    prompt = 'test prompt'

    expected_value = SYSTEM_FORMAT_LLAMA.format(system_message=system)
    assert message_formatter(system, [], None) == expected_value
    assert message_formatter(system, None, None) == expected_value
    assert message_formatter(system, None, None) == expected_value

    expected_value = SYSTEM_FORMAT_LLAMA.format(system_message=system) \
        + PROMPT_FORMAT_LLAMA.format(prompt=prompt) + RESPONSE_PREFIX_LLAMA
    assert message_formatter(system, [], prompt) == expected_value
    assert message_formatter(system, None, prompt) == expected_value

    expected_value = PROMPT_FORMAT_LLAMA.format(prompt=prompt) + RESPONSE_PREFIX_LLAMA
    assert message_formatter(None, [], prompt) == expected_value
    assert message_formatter(None, None, prompt) == expected_value
    assert message_formatter('', None, prompt) == expected_value

    expected_value = SYSTEM_FORMAT_LLAMA.format(system_message=system) \
        + PROMPT_FORMAT_LLAMA.format(prompt=message_prompt) \
        + RESPONSE_PREFIX_LLAMA \
        + message_response
    actual_value = message_formatter(
        system,
        [ExchangeRecord(prompt=message_prompt, response=message_response)],
        None,
    )
    assert actual_value == expected_value
    actual_value = message_formatter(
        system,
        [(message_prompt, message_response)],
        None,
    )
    assert actual_value == expected_value
    actual_value = message_formatter(
        system,
        [
            {'user': message_prompt, 'assistant': message_response},
        ],
        None,
    )
    assert actual_value == expected_value

    expected_value = PROMPT_FORMAT_LLAMA.format(prompt=message_prompt) \
        + RESPONSE_PREFIX_LLAMA \
        + message_response
    actual_value = message_formatter(
        None,
        [ExchangeRecord(prompt=message_prompt, response=message_response)],
        None,
    )
    assert actual_value == expected_value
    actual_value = message_formatter(
        '',
        [(message_prompt, message_response)],
        None,
    )
    assert actual_value == expected_value

    expected_value = SYSTEM_FORMAT_LLAMA.format(system_message=system) \
        + PROMPT_FORMAT_LLAMA.format(prompt=message_prompt) \
        + RESPONSE_PREFIX_LLAMA \
        + message_response \
        + PROMPT_FORMAT_LLAMA.format(prompt=message_prompt + '2') \
        + RESPONSE_PREFIX_LLAMA \
        + message_response + '2' \
        + PROMPT_FORMAT_LLAMA.format(prompt=prompt) \
        + RESPONSE_PREFIX_LLAMA
    actual_value = message_formatter(
        system,
        [
            ExchangeRecord(prompt=message_prompt, response=message_response),
            ExchangeRecord(prompt=message_prompt + '2', response=message_response + '2'),
        ],
        prompt,
    )
    assert actual_value == expected_value
    actual_value = message_formatter(
        system,
        [
            (message_prompt, message_response),
            (message_prompt + '2', message_response + '2'),
        ],
        prompt,
    )
    assert actual_value == expected_value
    actual_value = message_formatter(
        system,
        [
            {'user': message_prompt, 'assistant': message_response},
            {'user': message_prompt + '2', 'assistant': message_response + '2'},
        ],
        prompt,
    )
    assert actual_value == expected_value

def test__LlamaMessageFormatter():  # noqa
    assert LlamaMessageFormatter()(system_message=None, messages=[], prompt=None) == ''
    assert LlamaMessageFormatter()(system_message=None, messages=None, prompt=None) == ''

    messages = LlamaMessageFormatter()(
        system_message=None,
        messages=[
            ExchangeRecord(prompt='a', response='b'),
            ExchangeRecord(prompt='c', response='d'),
        ],
        prompt=None,
    )
    expected_value = '[INST] a [/INST]\nb[INST] c [/INST]\nd'
    assert messages == expected_value

    messages = LlamaMessageFormatter()(
        system_message='system',
        messages=[
            ExchangeRecord(prompt='a', response='b'),
            ExchangeRecord(prompt='c', response='d'),
        ],
        prompt=None,
    )
    assert messages == '[INST] <<SYS>> system <</SYS>> [/INST]\n' + expected_value

    messages = LlamaMessageFormatter()(
        system_message='system',
        messages=[
            ExchangeRecord(prompt='a', response='b'),
            ExchangeRecord(prompt='c', response='d'),
        ],
        prompt='e',
    )
    assert messages == f'[INST] <<SYS>> system <</SYS>> [/INST]\n{expected_value}[INST] e [/INST]\n'  # noqa

