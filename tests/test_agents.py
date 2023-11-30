"""Test agents.py classes and functions."""

from time import sleep
from llm_workflow.agents import OpenAIFunctionAgent, OpenAIFunctions, Tool, tool
from llm_workflow.base import Record, ExchangeRecord


@tool(
    name="ask_weather",
    description="Use this function to answer questions about the weather for a particular city.",
    inputs={
        'location': {
            'type': 'string',
            'description': "The city and state, e.g. San Francisco, CA",
        },
        'unit': {
                'type': 'string',
                'enum': ['celsius', 'fahrenheit'],
                'description': "The temperature unit to use. The model needs to infer this from the `location`.",  # noqa
        },
    },
    required=['location', 'unit'],
)
def fake_weather(location: str, unit: str) -> str:
    """Fake function to lookup weather."""
    return f"The temperature of {location} is 1000 degrees {unit}."


@tool(
    name="ask_stock_price",
    description="Use this function to answer questions about the the stock price for a particular stock symbol.",  # noqa
    inputs={
        'symbol': {
            'type': 'string',
            'description': "The stock symbol, e.g. 'AAPL'",
        },
    },
    required= ['symbol'],
)
def fake_stock(symbol: str) -> str:
    """Fake function to lookup stock price."""
    return f"The stock price of {symbol} is $1000."


def test_OpenAIToolAgent__Tool_class():  # noqa
    class FakeWeatherAPI:
        def __init__(self) -> None:
            self._history = []

        def __call__(self, location: str, unit: str) -> str:
            result = f"The temperature of {location} is 1000 degrees {unit}."
            # need a slight delay so we sort records consistently for test
            # the ExchangeRecord is created before the function is called
            sleep(0.01)
            self._history.append(Record(metadata={'result': result}))
            return result

        def history(self) -> list[str]:
            return self._history

    class FakeStockAPI:
        def __init__(self) -> None:
            self._history = []

        def __call__(self, symbol: str) -> str:
            result = f"The stock price of {symbol} is $1000."
            # need a slight delay so we sort records consistently for test
            # the ExchangeRecord is created before the function is called
            sleep(0.01)
            self._history.append(Record(metadata={'result': result}))
            return result

        def history(self) -> list[str]:
            return self._history

    fake_weather_tool = Tool(
        callable_obj=FakeWeatherAPI(),
        name="ask_weather",
        description="Use this function to answer questions about the weather for a particular city.",  # noqa
        inputs={
            'location': {
                'type': 'string',
                'description': "The city and state, e.g. San Francisco, CA",
            },
            'unit': {
                    'type': 'string',
                    'enum': ['celsius', 'fahrenheit'],
                    'description': "The temperature unit to use. The model needs to infer this from the `location`.",  # noqa
            },
        },
        required= ['location', 'unit'],
    )

    fake_stock_tool = Tool(
        callable_obj=FakeStockAPI(),
        name="ask_stock_price",
        description="Use this function to answer questions about the the stock price for a particular stock symbol.",  # noqa
        inputs={
            'symbol': {
                'type': 'string',
                'description': "The stock symbol, e.g. 'AAPL'",
            },
        },
        required= ['symbol'],
    )

    assert fake_weather_tool.name == fake_weather.name
    assert fake_weather_tool.description == fake_weather.description
    assert fake_weather_tool.inputs == fake_weather.inputs
    assert fake_weather_tool.required == fake_weather.required

    assert fake_stock_tool.name == fake_stock.name
    assert fake_stock_tool.description == fake_stock.description
    assert fake_stock_tool.inputs == fake_stock.inputs
    assert fake_stock_tool.required == fake_stock.required

    agent = OpenAIFunctionAgent(tools=[fake_weather_tool, fake_stock_tool])

    question = "What is the temperature in Seattle WA."
    response = agent(question)
    assert 'Seattle' in response
    assert 'degrees' in response
    # assert 'fahrenheit' in response  # model does not correctly infer fahrenheight
    assert len(fake_weather_tool.history()) == 1
    assert fake_weather_tool.history()[0].metadata['result'] == response
    assert len(fake_stock_tool.history()) == 0
    # the first record is the ExchangeRecord associated with the OpenAIFunctionAgent
    # and the second record is from the tool we used
    assert len(agent.history()) == 2
    assert isinstance(agent.history()[0], ExchangeRecord)
    assert agent.history()[0].prompt == question
    assert fake_weather_tool.name in agent.history()[0].response
    assert agent.history()[0].metadata['tool_names'] == fake_weather_tool.name
    assert 'location' in agent.history()[0].metadata['tool_args']
    assert 'unit' in agent.history()[0].metadata['tool_args']
    assert agent.history()[0].input_tokens > 0
    assert agent.history()[0].response_tokens > 0
    assert agent.history()[0].total_tokens == agent.history()[0].input_tokens + agent.history()[0].response_tokens  # noqa
    assert agent.history()[0].total_tokens > 0
    assert agent.history()[0].cost > 0
    assert isinstance(agent.history()[1], Record)
    assert agent.history()[1].metadata['result'] == response

    question = "What is the stock price of Apple?"
    response = agent(question)
    assert 'AAPL' in response
    # the first record (in the second use) is the ExchangeRecord associated with the
    # OpenAIFunctionAgent and the second record is from the tool we used
    assert len(fake_weather_tool.history()) == 1
    assert len(fake_stock_tool.history()) == 1
    assert fake_stock_tool.history()[0].metadata['result'] == response

    assert len(agent.history()) == 4
    assert isinstance(agent.history()[2], ExchangeRecord)
    assert agent.history()[2].prompt == question
    assert fake_stock_tool.name in agent.history()[2].response
    assert agent.history()[2].metadata['tool_names'] == fake_stock_tool.name
    assert 'symbol' in agent.history()[2].metadata['tool_args']
    assert agent.history()[2].input_tokens > 0
    assert agent.history()[2].response_tokens > 0
    assert agent.history()[2].total_tokens == agent.history()[2].input_tokens + agent.history()[2].response_tokens  # noqa
    assert agent.history()[2].total_tokens > 0
    assert agent.history()[2].cost > 0
    assert isinstance(agent.history()[3], Record)
    assert agent.history()[3].metadata['result'] == response

    question = "No tool is applicable for this question."
    response = agent(question)
    assert response is None
    assert len(agent.history()) == 5
    assert agent.history()[4].prompt == question
    assert agent.history()[4].response == ''
    assert 'tool_name' not in agent.history()[4].metadata
    assert agent.history()[4].input_tokens > 0
    assert agent.history()[4].response_tokens > 0
    assert agent.history()[4].total_tokens == agent.history()[4].input_tokens + agent.history()[4].response_tokens  # noqa
    assert agent.history()[4].total_tokens > 0
    assert agent.history()[4].cost > 0

def test_OpenAIToolAgent__tool_decorator():  # noqa

    assert isinstance(fake_weather, Tool)
    assert isinstance(fake_stock, Tool)

    agent = OpenAIFunctionAgent(
        model_name='gpt-3.5-turbo-0613',
        tools=[fake_weather, fake_stock],
    )

    question = "What is the temperature in Seattle WA."
    response = agent(question)
    assert 'Seattle' in response
    assert 'degrees' in response
    # assert 'fahrenheit' in response  # model does not correctly infer fahrenheight
    assert len(agent.history()) == 1
    assert agent.history()[0].prompt == question
    assert fake_weather.name in agent.history()[0].response
    assert agent.history()[0].metadata['tool_names'] == fake_weather.name
    assert 'location' in agent.history()[0].metadata['tool_args']
    assert 'unit' in agent.history()[0].metadata['tool_args']
    assert agent.history()[0].input_tokens > 0
    assert agent.history()[0].response_tokens > 0
    assert agent.history()[0].total_tokens == agent.history()[0].input_tokens + agent.history()[0].response_tokens  # noqa
    assert agent.history()[0].total_tokens > 0
    assert agent.history()[0].cost > 0

    question = "What is the stock price of Apple?"
    response = agent(question)
    assert 'AAPL' in response
    assert len(agent.history()) == 2
    assert agent.history()[1].prompt == question
    assert fake_stock.name in agent.history()[1].response
    assert agent.history()[1].metadata['tool_names'] == fake_stock.name
    assert 'symbol' in agent.history()[1].metadata['tool_args']
    assert agent.history()[1].input_tokens > 0
    assert agent.history()[1].response_tokens > 0
    assert agent.history()[1].total_tokens == agent.history()[1].input_tokens + agent.history()[1].response_tokens  # noqa
    assert agent.history()[1].total_tokens > 0
    assert agent.history()[1].cost > 0

    question = "No tool is applicable for this question."
    response = agent(question)
    assert response is None
    assert len(agent.history()) == 3
    assert agent.history()[2].prompt == question
    assert agent.history()[2].response == ''
    assert 'tool_names' not in agent.history()[2].metadata
    assert agent.history()[2].input_tokens > 0
    assert agent.history()[2].response_tokens > 0
    assert agent.history()[2].total_tokens == agent.history()[2].input_tokens + agent.history()[2].response_tokens  # noqa
    assert agent.history()[2].total_tokens > 0
    assert agent.history()[2].cost > 0

def test_OpenAIToolAgent__tools_via_yaml():  # noqa
    # read in yaml file
    import yaml
    with open('tests/test_data/agents/mock_tools.yml') as f:
        yaml_data = yaml.safe_load(f)

    tools = {x['name']: Tool.from_dict(x) for x in yaml_data}
    assert tools['ask_weather'].name == fake_weather.name
    assert tools['ask_weather'].description == fake_weather.description
    assert tools['ask_weather'].inputs == fake_weather.inputs
    assert tools['ask_weather'].required == fake_weather.required

    assert tools['ask_stock_price'].name == fake_stock.name
    assert tools['ask_stock_price'].description == fake_stock.description
    assert tools['ask_stock_price'].inputs == fake_stock.inputs
    assert tools['ask_stock_price'].required == fake_stock.required

    tools = OpenAIFunctions(
        model_name='gpt-3.5-turbo-0613',
        tools=tools.values(),
    )

    question = "What is the temperature in Seattle WA."
    response = tools(question)
    assert len(response) == 1
    response_tool, response_arguments = response[0]
    assert 'Seattle' in response_arguments['location']
    # tool doesn't correctly infer fahrenheit
    assert 'degrees' in response_arguments['unit'] \
        or 'fahrenheit' in response_arguments['unit'] \
        or 'celsius' in response_arguments['unit']
    assert len(tools.history()) == 1
    assert tools.history()[0].prompt == question
    assert response_tool.name in tools.history()[0].response
    assert tools.history()[0].metadata['tool_names'] == response_tool.name
    assert 'location' in tools.history()[0].metadata['tool_args']
    assert 'unit' in tools.history()[0].metadata['tool_args']
    assert tools.history()[0].input_tokens > 0
    assert tools.history()[0].response_tokens > 0
    assert tools.history()[0].total_tokens == tools.history()[0].input_tokens + tools.history()[0].response_tokens  # noqa
    assert tools.history()[0].total_tokens > 0
    assert tools.history()[0].cost > 0

    question = "What is the stock price of Apple?"
    response = tools(question)
    response_tool, response_arguments = response[0]
    assert 'AAPL' in response_arguments['symbol']
    assert len(tools.history()) == 2
    assert tools.history()[1].prompt == question
    assert response_tool.name in tools.history()[1].response
    assert tools.history()[1].metadata['tool_names'] == response_tool.name
    assert 'symbol' in tools.history()[1].metadata['tool_args']
    assert tools.history()[1].input_tokens > 0
    assert tools.history()[1].response_tokens > 0
    assert tools.history()[1].total_tokens == tools.history()[1].input_tokens + tools.history()[1].response_tokens  # noqa
    assert tools.history()[1].total_tokens > 0
    assert tools.history()[1].cost > 0

    question = "No tool is applicable for this question."
    response = tools(question)
    assert response is None
    assert len(tools.history()) == 3
    assert tools.history()[2].prompt == question
    assert tools.history()[2].response == ''
    assert 'tool_names' not in tools.history()[2].metadata
    assert tools.history()[2].input_tokens > 0
    assert tools.history()[2].response_tokens > 0
    assert tools.history()[2].total_tokens == tools.history()[2].input_tokens + tools.history()[2].response_tokens  # noqa
    assert tools.history()[2].total_tokens > 0
    assert tools.history()[2].cost > 0
