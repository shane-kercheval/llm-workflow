"""Test agents.py classes and functions."""

from llm_workflow.agents import OpenAIFunctionAgent, Tool, ToolBase, tool


def test_tool():  # noqa
    pass  # TODO

def test_OpenAIToolAgent__ToolBase():  # noqa
    class FakeWeatherTool(ToolBase):
        @property
        def name(self) -> str:
            return "ask_weather"

        @property
        def description(self) -> str:
            return "Use this function to answer questions about the weather for a particular city."

        @property
        def variables(self) -> dict:
            return {
                'location': {
                    'type': 'string',
                    'description': "The city and state, e.g. San Francisco, CA",
                },
                'unit': {
                        'type': 'string',
                        'enum': ['celsius', 'fahrenheit'],
                        'description': "The temperature unit to use. The model needs to infer this from the `location`.",  # noqa
                },
            }

        @property
        def required(self) -> list:
            return ['location', 'unit']

        def __call__(self, location: str, unit: str) -> str:
            return f"The temperature of {location} is 1000 degrees {unit}."

    class FakeStockPriceTool(ToolBase):
        @property
        def name(self) -> str:
            return "ask_stock_price"

        @property
        def description(self) -> str:
            return "Use this function to answer questions about the the stock price for a particular stock symbol."  # noqa

        @property
        def variables(self) -> dict:
            return {
                'symbol': {
                    'type': 'string',
                    'description': "The stock symbol, e.g. 'AAPL'",
                },
            }

        @property
        def required(self) -> list:
            return ['symbol']


        def __call__(self, symbol: str) -> str:
            return f"The stock price of {symbol} is $1000."

    agent = OpenAIFunctionAgent(
        model_name='gpt-3.5-turbo',
        tools=[FakeWeatherTool(), FakeStockPriceTool()],
    )

    question = "What is the temperature in Seattle WA."
    response = agent(question)
    assert 'Seattle' in response
    assert 'degrees' in response
    # assert 'fahrenheit' in response  # model does not correctly infer fahrenheight
    assert len(agent.history()) == 1
    assert agent.history()[0].prompt == question
    assert FakeWeatherTool().name in agent.history()[0].response
    assert agent.history()[0].metadata['tool_name'] == FakeWeatherTool().name
    assert 'location' in agent.history()[0].metadata['tool_args']
    assert 'unit' in agent.history()[0].metadata['tool_args']
    assert agent.history()[0].prompt_tokens > 0
    assert agent.history()[0].response_tokens > 0
    assert agent.history()[0].total_tokens == agent.history()[0].prompt_tokens + agent.history()[0].response_tokens  # noqa
    assert agent.history()[0].total_tokens > 0
    assert agent.history()[0].cost > 0

    question = "What is the stock price of Apple?"
    response = agent(question)
    assert 'AAPL' in response
    assert len(agent.history()) == 2
    assert agent.history()[1].prompt == question
    assert FakeStockPriceTool().name in agent.history()[1].response
    assert agent.history()[1].metadata['tool_name'] == FakeStockPriceTool().name
    assert 'symbol' in agent.history()[1].metadata['tool_args']
    assert agent.history()[1].prompt_tokens > 0
    assert agent.history()[1].response_tokens > 0
    assert agent.history()[1].total_tokens == agent.history()[1].prompt_tokens + agent.history()[1].response_tokens  # noqa
    assert agent.history()[1].total_tokens > 0
    assert agent.history()[1].cost > 0

    question = "Should not exist."
    response = agent(question)
    assert response is None
    assert len(agent.history()) == 3
    assert agent.history()[2].prompt == question
    assert agent.history()[2].response == ''
    assert 'tool_name' not in agent.history()[2].metadata
    assert agent.history()[2].prompt_tokens > 0
    assert agent.history()[2].response_tokens > 0
    assert agent.history()[2].total_tokens == agent.history()[2].prompt_tokens + agent.history()[2].response_tokens  # noqa
    assert agent.history()[2].total_tokens > 0
    assert agent.history()[2].cost > 0


def test_OpenAIToolAgent__tool_decorator():  # noqa
    @tool(
        name="ask_weather",
        description="Use this function to answer questions about the weather for a particular city.",  # noqa
        variables={
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
    def fake_weather(location: str, unit: str) -> str:
        return f"The temperature of {location} is 1000 degrees {unit}."

    @tool(
        name="ask_stock_price",
        description="Use this function to answer questions about the the stock price for a particular stock symbol.",  # noqa
        variables={
            'symbol': {
                'type': 'string',
                'description': "The stock symbol, e.g. 'AAPL'",
            },
        },
        required= ['symbol'],
    )
    def fake_stock(symbol: str) -> str:
            return f"The stock price of {symbol} is $1000."

    assert isinstance(fake_weather, Tool)
    assert isinstance(fake_stock, Tool)

    agent = OpenAIFunctionAgent(
        model_name='gpt-3.5-turbo',
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
    assert agent.history()[0].metadata['tool_name'] == fake_weather.name
    assert 'location' in agent.history()[0].metadata['tool_args']
    assert 'unit' in agent.history()[0].metadata['tool_args']
    assert agent.history()[0].prompt_tokens > 0
    assert agent.history()[0].response_tokens > 0
    assert agent.history()[0].total_tokens == agent.history()[0].prompt_tokens + agent.history()[0].response_tokens  # noqa
    assert agent.history()[0].total_tokens > 0
    assert agent.history()[0].cost > 0

    question = "What is the stock price of Apple?"
    response = agent(question)
    assert 'AAPL' in response
    assert len(agent.history()) == 2
    assert agent.history()[1].prompt == question
    assert fake_stock.name in agent.history()[1].response
    assert agent.history()[1].metadata['tool_name'] == fake_stock.name
    assert 'symbol' in agent.history()[1].metadata['tool_args']
    assert agent.history()[1].prompt_tokens > 0
    assert agent.history()[1].response_tokens > 0
    assert agent.history()[1].total_tokens == agent.history()[1].prompt_tokens + agent.history()[1].response_tokens  # noqa
    assert agent.history()[1].total_tokens > 0
    assert agent.history()[1].cost > 0

    question = "Should not exist."
    response = agent(question)
    assert response is None
    assert len(agent.history()) == 3
    assert agent.history()[2].prompt == question
    assert agent.history()[2].response == ''
    assert 'tool_name' not in agent.history()[2].metadata
    assert agent.history()[2].prompt_tokens > 0
    assert agent.history()[2].response_tokens > 0
    assert agent.history()[2].total_tokens == agent.history()[2].prompt_tokens + agent.history()[2].response_tokens  # noqa
    assert agent.history()[2].total_tokens > 0
    assert agent.history()[2].cost > 0
