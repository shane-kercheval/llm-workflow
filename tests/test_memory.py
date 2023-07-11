"""tests llm_workflow/memory.py."""

from llm_workflow.memory import LastNExchangesManager, TokenWindowManager
from llm_workflow.models import OpenAIChat, ExchangeRecord
from llm_workflow.resources import MODEL_COST_PER_TOKEN


def test_OpenAIChat__MemoryManagerMessageWindow0():  # noqa
    model_name = 'gpt-3.5-turbo'
    openai_llm = OpenAIChat(
        model_name=model_name,
        memory_manager=LastNExchangesManager(last_n_exchanges=0),
    )
    assert openai_llm.previous_record() is None
    assert openai_llm.previous_prompt is None
    assert openai_llm.previous_response is None
    assert openai_llm.cost == 0
    assert openai_llm.total_tokens == 0
    assert openai_llm.prompt_tokens == 0
    assert openai_llm.response_tokens == 0

    ####
    # first interaction
    # this shouldn't be any different
    ####
    prompt = "Hi my name is shane. What is my name?"
    response = openai_llm(prompt)
    assert 'shane' in response.lower()
    assert isinstance(response, str)
    assert len(response) > 1

    # previous memory is the input to ChatGPT
    assert openai_llm._previous_memory[0]['role'] == 'system'
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

    assert openai_llm.previous_prompt == prompt
    assert openai_llm.previous_response == response
    assert openai_llm.cost_per_token == MODEL_COST_PER_TOKEN[model_name]
    assert openai_llm.cost == message.cost
    assert openai_llm.total_tokens == message.total_tokens
    assert openai_llm.prompt_tokens == message.prompt_tokens
    assert openai_llm.response_tokens == message.response_tokens

    # previous_prompt = prompt
    # previous_response = response
    previous_cost = message.cost
    previous_total_tokens = message.total_tokens
    previous_prompt_tokens = message.prompt_tokens
    previous_response_tokens = message.response_tokens

    ####
    # second interaction
    # this shouldn't be any different
    ####
    prompt = "What is my name?"
    response = openai_llm(prompt)
    assert 'shane' not in response.lower()
    assert isinstance(response, str)
    assert len(response) > 1

    # previous memory is the input to ChatGPT
    assert len(openai_llm._previous_memory) == 2  # system/user
    assert openai_llm._previous_memory[0]['role'] == 'system'
    assert openai_llm._previous_memory[0]['content'] == 'You are a helpful assistant.'
    assert openai_llm._previous_memory[1]['role'] == 'user'
    assert openai_llm._previous_memory[1]['content'] == prompt

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

    assert openai_llm.previous_prompt == prompt
    assert openai_llm.previous_response == response
    assert openai_llm.cost_per_token == MODEL_COST_PER_TOKEN[model_name]
    assert openai_llm.cost == previous_cost + message.cost
    assert openai_llm.total_tokens == previous_total_tokens + message.total_tokens
    assert openai_llm.prompt_tokens == previous_prompt_tokens + message.prompt_tokens
    assert openai_llm.response_tokens == previous_response_tokens + message.response_tokens

def test_OpenAIChat__MemoryManagerMessageWindow1():  # noqa
    model_name = 'gpt-3.5-turbo'
    openai_llm = OpenAIChat(
        model_name=model_name,
        memory_manager=LastNExchangesManager(last_n_exchanges=1),
    )
    assert openai_llm.previous_record() is None
    assert openai_llm.previous_prompt is None
    assert openai_llm.previous_response is None
    assert openai_llm.cost == 0
    assert openai_llm.total_tokens == 0
    assert openai_llm.prompt_tokens == 0
    assert openai_llm.response_tokens == 0

    ####
    # first interaction
    # this shouldn't be any different
    ####
    prompt = "Hi my name is shane. What is my name?"
    response = openai_llm(prompt)
    assert 'shane' in response.lower()
    assert isinstance(response, str)
    assert len(response) > 1

    # previous memory is the input to ChatGPT
    assert openai_llm._previous_memory[0]['role'] == 'system'
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

    ####
    # second interaction
    # this shouldn't be any different
    ####
    prompt = "What is my name?"
    response = openai_llm(prompt)
    assert 'shane' in response.lower()
    assert isinstance(response, str)
    assert len(response) > 1

    # previous memory is the input to ChatGPT
    assert openai_llm._previous_memory[0]['role'] == 'system'
    assert openai_llm._previous_memory[0]['content'] == 'You are a helpful assistant.'
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

    assert openai_llm.previous_prompt == prompt
    assert openai_llm.previous_response == response
    assert openai_llm.cost_per_token == MODEL_COST_PER_TOKEN[model_name]
    assert openai_llm.cost == previous_cost + message.cost
    assert openai_llm.total_tokens == previous_total_tokens + message.total_tokens
    assert openai_llm.prompt_tokens == previous_prompt_tokens + message.prompt_tokens
    assert openai_llm.response_tokens == previous_response_tokens + message.response_tokens

    previous_prompt = prompt
    previous_response = response

    ####
    # third interaction
    # this shouldn't be any different
    ####
    prompt = "What is today's date?"
    response = openai_llm(prompt)
    assert 'shane' not in response.lower()
    assert isinstance(response, str)
    assert len(response) > 1

    # previous memory is the input to ChatGPT
    # The last message should contain shane, but not this one
    assert openai_llm._previous_memory[0]['role'] == 'system'
    assert openai_llm._previous_memory[0]['content'] == 'You are a helpful assistant.'
    assert openai_llm._previous_memory[1]['role'] == 'user'
    assert openai_llm._previous_memory[1]['content'] == previous_prompt
    assert openai_llm._previous_memory[2]['role'] == 'assistant'
    assert openai_llm._previous_memory[2]['content'] == previous_response
    assert 'shane' in openai_llm._previous_memory[2]['content'].lower()
    assert openai_llm._previous_memory[3]['role'] == 'user'
    assert openai_llm._previous_memory[3]['content'] == prompt
    assert len(openai_llm._previous_memory) == 4

    assert len(openai_llm.history()) == 3
    message = openai_llm.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata == {'model_name': model_name}
    assert message.cost > 0
    assert message.prompt_tokens > 0
    assert message.response_tokens > 0
    assert message.total_tokens == message.prompt_tokens + message.response_tokens

    assert openai_llm.previous_prompt == prompt
    assert openai_llm.previous_response == response
    assert openai_llm.cost_per_token == MODEL_COST_PER_TOKEN[model_name]

    previous_prompt = prompt
    previous_response = response

    ####
    # 4th interaction
    # this shouldn't contain the name shane because the last interaction was the first that didn't
    ####
    prompt = "What is today's date?"
    response = openai_llm(prompt)
    assert 'shane' not in response.lower()
    assert isinstance(response, str)
    assert len(response) > 1

    # previous memory is the input to ChatGPT
    assert openai_llm._previous_memory[0]['role'] == 'system'
    assert openai_llm._previous_memory[0]['content'] == 'You are a helpful assistant.'
    assert openai_llm._previous_memory[1]['role'] == 'user'
    assert openai_llm._previous_memory[1]['content'] == previous_prompt
    assert openai_llm._previous_memory[2]['role'] == 'assistant'
    assert openai_llm._previous_memory[2]['content'] == previous_response
    assert openai_llm._previous_memory[3]['role'] == 'user'
    assert openai_llm._previous_memory[3]['content'] == prompt
    # still 4 because we are only keeping 1 message
    # (1)system + (2)previous question + (3)previous answer + (4)new question
    assert len(openai_llm._previous_memory) == 4

    assert len(openai_llm.history()) == 4
    message = openai_llm.previous_record()
    assert isinstance(message, ExchangeRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata == {'model_name': model_name}
    assert message.cost > 0
    assert message.prompt_tokens > 0
    assert message.response_tokens > 0
    assert message.total_tokens == message.prompt_tokens + message.response_tokens

    assert openai_llm.previous_prompt == prompt
    assert openai_llm.previous_response == response
    assert openai_llm.cost_per_token == MODEL_COST_PER_TOKEN[model_name]

def test_OpenAIChat__TokenWindowManager():  # noqa
    token_threshold = 100
    model_name = 'gpt-3.5-turbo'
    openai_llm = OpenAIChat(
        model_name=model_name,
        memory_manager=TokenWindowManager(last_n_tokens=token_threshold),
    )
    assert openai_llm.previous_record() is None
    assert openai_llm.previous_prompt is None
    assert openai_llm.previous_response is None
    assert openai_llm.cost == 0
    assert openai_llm.total_tokens == 0
    assert openai_llm.prompt_tokens == 0
    assert openai_llm.response_tokens == 0

    ####
    # first interaction
    # this shouldn't be any different
    ####
    prompt = "Hi my name is Shane and my favorite number is 42 and my favorite color is blue. What is my mame?"  # noqa
    response = openai_llm(prompt)
    assert 'shane' in response.lower()
    assert isinstance(response, str)
    assert len(openai_llm._previous_memory) == 2
    assert openai_llm.previous_record().total_tokens <= token_threshold
    assert openai_llm._previous_memory[0]['role'] == 'system'
    assert openai_llm._previous_memory[-1]['role'] == 'user'
    assert openai_llm._previous_memory[-1]['content'] == prompt
    assert len(openai_llm.history()) == 1

    ####
    # second interaction
    # The previous message/memory should be under the threshold, so it should know my favorite
    # number
    ####
    prompt = "What is my favorite number?"
    response = openai_llm(prompt)
    assert '42' in response.lower()
    assert len(openai_llm._previous_memory) == 4
    assert openai_llm.previous_record().total_tokens <= token_threshold
    assert openai_llm._previous_memory[0]['role'] == 'system'
    assert openai_llm._previous_memory[-1]['role'] == 'user'
    assert openai_llm._previous_memory[-1]['content'] == prompt
    assert len(openai_llm.history()) == 2

    ####
    # third interaction
    # The token threshold isn't high enough to send all of the messages so the first message will
    # be removed.
    ####
    prompt = "What is my favorite color?"
    response = openai_llm(prompt)

    assert 'blue' not in response.lower()
    # previous memory is still 4 since it only remembers last message + new prompt + system mesasge
    assert len(openai_llm._previous_memory) == 4
    assert openai_llm.previous_record().total_tokens <= token_threshold
    assert openai_llm._previous_memory[0]['role'] == 'system'
    assert openai_llm._previous_memory[-1]['role'] == 'user'
    assert openai_llm._previous_memory[-1]['content'] == prompt
    assert len(openai_llm.history()) == 3
