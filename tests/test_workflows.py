"""Tests workflow functionality."""
from time import sleep
from llm_workflow.base import (
    EmbeddingRecord,
    ExchangeRecord,
    Document,
    LanguageModel,
    Record,
    TokenUsageRecord,
    Workflow,
    Value,
)
from llm_workflow.internal_utilities import has_property
from tests.conftest import MockChatModel, MockRandomEmbeddings


class FakeLLM(LanguageModel):
    """Mock historical records to ensure we are not double-counting unique records."""

    def __init__(self) -> None:
        super().__init__()
        self.record_a = TokenUsageRecord(metadata={'id': 'record_a'}, total_tokens=1, cost=2)
        sleep(0.001)
        self.record_b = TokenUsageRecord(metadata={'id': 'record_b'}, total_tokens=3, cost=5)
        sleep(0.001)
        self.record_d = Record(metadata={'id': 'record_d'})
        sleep(0.001)
        self.record_f = ExchangeRecord(metadata={'id': 'record_f'}, prompt="p_f", response="r_f")
        sleep(0.001)
        self.record_c = TokenUsageRecord(metadata={'id': 'record_c'}, total_tokens=6, cost=8)
        sleep(0.001)
        self.record_e = Record(metadata={'id': 'record_e'})
        sleep(0.001)
        self.record_g = ExchangeRecord(metadata={'id': 'record_g'}, prompt="p_g", response="r_g")
        self.records = [
            self.record_a, self.record_b, self.record_d, self.record_f, self.record_c,
            self.record_e, self.record_g,
        ]

    def __call__(self, value: object) -> object:
        """Not needed."""

    def _get_history(self) -> list[Record]:
        """Return mock history."""
        return self.records


class FakeLLMNoUsage(LanguageModel):
    """Mock historical records to ensure we are not double-counting unique records."""

    def __init__(self) -> None:
        super().__init__()

        self.record_a = TokenUsageRecord(metadata={'id': 'record_a'}, total_tokens=None, cost=None)
        sleep(0.001)
        self.record_b = TokenUsageRecord(metadata={'id': 'record_b'}, total_tokens=None, cost=None)
        sleep(0.001)
        self.record_d = Record(metadata={'id': 'record_d'})
        sleep(0.001)
        self.record_c = TokenUsageRecord(metadata={'id': 'record_c'}, total_tokens=None, cost=None)
        sleep(0.001)
        self.record_e = Record(metadata={'id': 'record_e'})
        self.records = [self.record_a, self.record_b, self.record_d, self.record_c, self.record_e]

    def __call__(self, value: object) -> object:
        """Not needed."""

    def _get_history(self) -> list[Record]:
        """Return mock history."""
        return self.records


class MockHistoryWrapper:
    """Mock classes where the history is propagated up."""

    def __init__(self, hist_obj: LanguageModel) -> None:
        self._hist_obj = hist_obj

    def __call__(self, value: object) -> object:
        """Not needed."""

    def history(self) -> list[Record]:
        """Return mock history."""
        return self._hist_obj.history()


def test_Value():  # noqa
    value = Value()
    assert value() is None
    assert value('test') == 'test'
    assert value() == 'test'

def test_has_property():  # noqa
    chat = MockChatModel(token_calculator=len)
    lambda_func = lambda x: x  # noqa
    assert has_property(obj=chat, property_name='total_tokens')
    assert not has_property(obj=lambda_func, property_name='total_tokens')
    assert not has_property(obj=chat, property_name='does_not_have')
    assert not has_property(obj=lambda_func, property_name='does_not_have')

def test_history():  # noqa
    mock_records = FakeLLM()
    mock_wrapper = MockHistoryWrapper(hist_obj=mock_records)
    # make sure historical records are only counted once
    workflow = Workflow(tasks=[mock_records, mock_wrapper, mock_records])
    assert workflow.history() == mock_records.records

    mock_records = FakeLLMNoUsage()
    mock_wrapper = MockHistoryWrapper(hist_obj=mock_records)
    # make sure historical records are only counted once
    workflow = Workflow(tasks=[mock_records, mock_wrapper, mock_records])
    assert workflow.history() == mock_records.records

def test_usage_history():  # noqa
    mock_records = FakeLLM()
    mock_wrapper = MockHistoryWrapper(hist_obj=mock_records)
    # make sure historical records are only counted once
    workflow = Workflow(tasks=[mock_records, mock_wrapper, mock_records])
    records = workflow.history(types=TokenUsageRecord)
    assert len(records) == 5
    assert records[0] == mock_records.record_a
    assert records[1] == mock_records.record_b
    assert records[2] == mock_records.record_f
    assert records[3] == mock_records.record_c
    assert records[4] == mock_records.record_g

    assert workflow.sum('total_tokens') == mock_records.record_a.total_tokens + \
        mock_records.record_b.total_tokens + \
        mock_records.record_c.total_tokens
    assert workflow.sum('cost') == mock_records.record_a.cost + \
        mock_records.record_b.cost + \
        mock_records.record_c.cost

def test_usage_history_no_usage():  # noqa
    mock_records = FakeLLMNoUsage()
    mock_wrapper = MockHistoryWrapper(hist_obj=mock_records)
    # make sure historical records are only counted once
    workflow = Workflow(tasks=[mock_records, mock_wrapper, mock_records])
    records = workflow.history(TokenUsageRecord)
    assert len(records) == 3
    assert records[0] == mock_records.record_a
    assert records[1] == mock_records.record_b
    assert records[2] == mock_records.record_c

    assert workflow.sum('total_tokens') == 0
    assert workflow.sum('cost') == 0

def test_exchange_history():  # noqa
    mock_records = FakeLLM()
    mock_wrapper = MockHistoryWrapper(hist_obj=mock_records)
    # make sure historical records are only counted once
    workflow = Workflow(tasks=[mock_records, mock_wrapper, mock_records])
    records = workflow.history(ExchangeRecord)

    assert len(records) == 2
    assert records[0] == mock_records.record_f
    assert records[1] == mock_records.record_g

    assert workflow.sum('total_tokens') == mock_records.record_a.total_tokens + \
        mock_records.record_b.total_tokens + \
        mock_records.record_c.total_tokens
    assert workflow.sum('cost') == mock_records.record_a.cost + \
        mock_records.record_b.cost + \
        mock_records.record_c.cost

def test_workflow_propegation():  # noqa
    # test empty workflow
    workflow = Workflow(tasks=[])
    assert workflow() is None
    assert workflow('param') is None
    assert workflow(value=1) is None

    # test workflow with one task
    workflow = Workflow(tasks=[lambda x: x * 2])
    assert workflow(10) == 20
    assert workflow([1]) == [1, 1]

    # test workflow with two tasks and test args/kwargs
    def add_one(value: int) -> int:
        return value + 1

    workflow = Workflow(tasks=[add_one, lambda x: x * 2])
    assert workflow(10) == 22
    assert workflow(value=10) == 22

    # test with three tasks
    workflow = Workflow(tasks=[add_one, lambda x: x * 2, add_one])
    assert workflow(10) == 23
    assert workflow(value=10) == 23

def test_workflow_index_len():  # noqa
    workflow = Workflow(tasks=[])
    assert len(workflow) == 0
    workflow = Workflow(tasks=['test'])
    assert workflow[0] == 'test'

def test_workflow_with_MockChat():  # noqa
    # this Chat returns the "Response: " + prompt
    prompt = "Here's a question."
    first_response = "Response: " + prompt
    second_prompt = "Question: " + first_response
    second_response = "Response: " + second_prompt

    chat = MockChatModel(
        return_prompt="Response: ",
        token_calculator=len,
        message_formatter=lambda system_message, history, prompt: prompt,  # noqa
    )
    workflow = Workflow(tasks=[chat, lambda x: "Question: " + x, chat])
    result = workflow(prompt)
    # the final result should be the response returned by the second invokation of chat()
    assert result == second_response

    # check that the prompts/responses got propegated through the workflow
    assert len(chat.history()) == 2
    assert len(chat.chat_history) == 2
    assert chat.history() == chat.chat_history
    assert chat.history()[0].prompt == prompt
    assert chat.history()[0].response == first_response
    assert chat.history()[0].input_tokens == len(prompt)
    assert chat.history()[0].response_tokens == len(first_response)
    assert chat.history()[0].total_tokens == len(prompt) + len(first_response)
    assert chat.history()[0].cost is None
    assert chat.history()[1].prompt == second_prompt
    assert chat.history()[1].response == second_response
    assert chat.history()[1].input_tokens == len(second_prompt)
    assert chat.history()[1].response_tokens == len(second_response)
    assert chat.history()[1].total_tokens == len(second_prompt) + len(second_response)
    assert chat.history()[1].cost is None

    assert chat.input_tokens == chat.history()[0].input_tokens + chat.history()[1].input_tokens
    assert chat.response_tokens == chat.history()[0].response_tokens + chat.history()[1].response_tokens  # noqa
    assert chat.total_tokens == chat.history()[0].total_tokens + chat.history()[1].total_tokens
    assert chat.cost == 0

    assert workflow.sum('total_tokens', types=EmbeddingRecord) == 0
    assert workflow.sum('input_tokens') == chat.input_tokens
    assert workflow.sum('response_tokens') == chat.response_tokens
    assert workflow.sum('total_tokens') == chat.total_tokens
    assert workflow.sum('cost') == 0

def test_workflow_with_MockChat_tokens_costs():  # noqa
    prompt = "Here's a question."
    first_response = "Response: " + prompt
    second_prompt = "Question: " + first_response
    second_response = "Response: " + second_prompt

    cost_per_token = 15
    # this Chat returns the "Response: " + prompt
    chat = MockChatModel(
        return_prompt="Response: ",
        token_calculator=len,
        cost_calculator=lambda input_tokens, response_tokens: (input_tokens + response_tokens) * cost_per_token,  # noqa
        message_formatter=lambda system_message, history, prompt: prompt,  # noqa
    )
    workflow = Workflow(tasks=[chat, lambda x: "Question: " + x, chat])
    result = workflow(prompt)
    # the final result should be the response returned by the second invokation of chat()
    assert result == second_response

    # check that the prompts/responses got propegated through the workflow
    assert len(chat.history()) == 2
    assert chat.history()[0].prompt == prompt
    assert chat.history()[0].response == first_response
    assert chat.history()[0].input_tokens == len(prompt)
    assert chat.history()[0].response_tokens == len(first_response)
    assert chat.history()[0].total_tokens == len(prompt) + len(first_response)
    assert chat.history()[0].cost == chat.history()[0].total_tokens * cost_per_token
    assert chat.history()[1].prompt == second_prompt
    assert chat.history()[1].response == second_response
    assert chat.history()[1].input_tokens == len(second_prompt)
    assert chat.history()[1].response_tokens == len(second_response)
    assert chat.history()[1].total_tokens == len(second_prompt) + len(second_response)
    assert chat.history()[1].cost == chat.history()[1].total_tokens * cost_per_token

    assert chat.input_tokens == chat.history()[0].input_tokens + chat.history()[1].input_tokens
    assert chat.response_tokens == chat.history()[0].response_tokens + chat.history()[1].response_tokens  # noqa
    assert chat.total_tokens == chat.history()[0].total_tokens + chat.history()[1].total_tokens
    assert chat.cost == chat.history()[0].cost + chat.history()[1].cost

    # because the `chat` model is included twice in the workflow; this check ensures we are not
    # double-counting the totals
    assert workflow.sum('input_tokens') == chat.input_tokens
    assert workflow.sum('response_tokens') == chat.response_tokens
    assert workflow.sum('total_tokens') == chat.total_tokens
    assert workflow.sum('cost') == chat.cost

    new_prompt = "New Prompt"
    result = workflow(new_prompt)
    new_first_response = "Response: " + new_prompt
    new_second_prompt = "Question: " + new_first_response
    new_second_response = "Response: " + new_second_prompt
    # the final result should be the response returned by the second invokation of chat()
    assert result == new_second_response

    assert len(chat.history()) == 4
    assert chat.history()[0].prompt == prompt
    assert chat.history()[0].response == first_response
    assert chat.history()[0].input_tokens == len(prompt)
    assert chat.history()[0].response_tokens == len(first_response)
    assert chat.history()[0].total_tokens == len(prompt) + len(first_response)
    assert chat.history()[0].cost == chat.history()[0].total_tokens * cost_per_token
    assert chat.history()[1].prompt == second_prompt
    assert chat.history()[1].response == second_response
    assert chat.history()[1].input_tokens == len(second_prompt)
    assert chat.history()[1].response_tokens == len(second_response)
    assert chat.history()[1].total_tokens == len(second_prompt) + len(second_response)
    assert chat.history()[1].cost == chat.history()[1].total_tokens * cost_per_token
    assert chat.history()[2].prompt == new_prompt
    assert chat.history()[2].response == new_first_response
    assert chat.history()[2].input_tokens == len(new_prompt)
    assert chat.history()[2].response_tokens == len(new_first_response)
    assert chat.history()[2].total_tokens == len(new_prompt) + len(new_first_response)
    assert chat.history()[2].cost == chat.history()[2].total_tokens * cost_per_token
    assert chat.history()[3].prompt == new_second_prompt
    assert chat.history()[3].response == new_second_response
    assert chat.history()[3].input_tokens == len(new_second_prompt)
    assert chat.history()[3].response_tokens == len(new_second_response)
    assert chat.history()[3].total_tokens == len(new_second_prompt) + len(new_second_response)
    assert chat.history()[3].cost == chat.history()[3].total_tokens * cost_per_token

    assert chat.input_tokens == chat.history()[0].input_tokens + \
        chat.history()[1].input_tokens + \
        chat.history()[2].input_tokens + \
        chat.history()[3].input_tokens
    assert chat.response_tokens == chat.history()[0].response_tokens + \
        chat.history()[1].response_tokens + \
        chat.history()[2].response_tokens + \
        chat.history()[3].response_tokens
    assert chat.total_tokens == chat.history()[0].total_tokens + \
        chat.history()[1].total_tokens + \
        chat.history()[2].total_tokens + \
        chat.history()[3].total_tokens
    assert chat.cost == chat.history()[0].cost + \
        chat.history()[1].cost + \
        chat.history()[2].cost + \
        chat.history()[3].cost

    # because the `chat` model is included twice in the workflow; this check ensures we are not
    # double-counting the totals
    assert workflow.sum('input_tokens') == chat.input_tokens
    assert workflow.sum('response_tokens') == chat.response_tokens
    assert workflow.sum('total_tokens') == chat.total_tokens
    assert workflow.sum('cost') == chat.cost

def test_workflow_with_MockChat_MockEmbeddings():  # noqa
    """
    This is an unrealistic but useful test where we are using an embeddings model and a chat
    model in a workflow multiple times; helper functions are used to change the output of the
    embeddings model to the input of the chat model and vice versa; this is demonstrating/testing
    the extensibility of the workflow.
    """  # noqa: D404
    cost_per_token_chat = 15
    cost_per_token_embedding = 7
    # this Chat returns the "Response: " + prompt
    docs = [
        Document(content="Doc A"),
        Document(content="Doc B"),
    ]
    cache_list_docs = {'cache': docs}

    def list_docs_to_prompt(docs: list[Document]) -> str:
        """
        This isn't an example of how a workflow would actually be used, but simply a way to mimic
        fowarding information from one step to another.
        """  # noqa: D404
        return ' '.join([x.content for x in docs])

    def prompt_to_list_docs(prompt: str) -> list[Document]:
        """
        This isn't an example of how a workflow would actually be used, but simply a way to mimic
        fowarding information from one step to another.
        """  # noqa: D404
        docs = [Document(content=x) for x in prompt.split(' ')]
        cache_list_docs['cache'] = docs
        return docs

    def embeddings_to_docs(embeddings: list[list[float]]) -> list[Document]:
        """
        This isn't an example of how a workflow would actually be used, but simply a way to mimic
        fowarding information from one step to another.
        """  # noqa: D404
        temp_docs = cache_list_docs['cache']
        for doc, embedding in zip(temp_docs, embeddings, strict=True):
            doc.metadata = {'embedding': embedding}
        return temp_docs

    embeddings = MockRandomEmbeddings(token_counter=len, cost_per_token=cost_per_token_embedding)

    # we need to sleep a fraction of a second so that we can ensure the correct order of history
    def sleep_return(x):  # noqa
        sleep(0.001)
        return x

    chat = MockChatModel(
        return_prompt="Response: ",
        token_calculator=len,
        cost_calculator=lambda input_tokens, response_tokens: (input_tokens + response_tokens) * cost_per_token_chat,  # noqa
        message_formatter=lambda system_message, history, prompt: prompt,  # noqa
    )
    workflow = Workflow(tasks=[
        sleep_return,
        embeddings,
        sleep_return,
        embeddings_to_docs,
        list_docs_to_prompt,
        chat,
        sleep_return,
        prompt_to_list_docs,
        embeddings,
        sleep_return,
        embeddings_to_docs,
        list_docs_to_prompt,
        chat])
    result = workflow(docs)

    ####
    # Test chat model
    ####
    initial_prompt = "Doc A Doc B"
    first_response = "Response: " + initial_prompt
    second_prompt = first_response
    second_response = "Response: " + second_prompt


    # the final result should be the response returned by the second invokation of chat()
    assert result == second_response
    # check that the prompts/responses got propegated through the workflow
    assert len(chat.history()) == 2
    assert chat.history()[0].prompt == initial_prompt
    assert chat.history()[0].response == first_response
    assert chat.history()[0].input_tokens == len(initial_prompt)
    assert chat.history()[0].response_tokens == len(first_response)
    assert chat.history()[0].total_tokens == len(initial_prompt) + len(first_response)
    assert chat.history()[0].cost == chat.history()[0].total_tokens * cost_per_token_chat
    assert chat.history()[1].prompt == second_prompt
    assert chat.history()[1].response == second_response
    assert chat.history()[1].input_tokens == len(second_prompt)
    assert chat.history()[1].response_tokens == len(second_response)
    assert chat.history()[1].total_tokens == len(second_prompt) + len(second_response)
    assert chat.history()[1].cost == chat.history()[1].total_tokens * cost_per_token_chat

    assert chat.input_tokens == chat.history()[0].input_tokens + chat.history()[1].input_tokens
    assert chat.response_tokens == chat.history()[0].response_tokens + chat.history()[1].response_tokens  # noqa
    assert chat.total_tokens == chat.history()[0].total_tokens + chat.history()[1].total_tokens
    assert chat.cost == chat.history()[0].cost + chat.history()[1].cost

    ####
    # Test embeddings model
    ####
    assert len(embeddings.history()) == 2
    assert embeddings.history()[0].total_tokens == len(docs[0].content) + len(docs[1].content)
    assert embeddings.history()[0].cost == embeddings.history()[0].total_tokens * cost_per_token_embedding  # noqa
    assert embeddings.history()[1].total_tokens == len("Response:DocADocB")
    assert embeddings.history()[1].cost == embeddings.history()[1].total_tokens * cost_per_token_embedding  # noqa

    assert embeddings.total_tokens == embeddings.history()[0].total_tokens + embeddings.history()[1].total_tokens  # noqa
    assert embeddings.cost == embeddings.history()[0].cost + embeddings.history()[1].cost

    ####
    # Test workflow
    ####
    # because the `chat` model is included twice in the workflow; this check ensures we are not
    # double-counting the totals
    assert workflow.history() == [embeddings.history()[0], chat.history()[0], embeddings.history()[1], chat.history()[1]]  # noqa
    assert workflow.history(ExchangeRecord) == chat.history()
    assert workflow.history(EmbeddingRecord) == embeddings.history()
    assert workflow.sum('total_tokens') == chat.total_tokens + embeddings.total_tokens
    assert workflow.sum('total_tokens', types=EmbeddingRecord) == embeddings.total_tokens
    assert workflow.sum('cost') == chat.cost + embeddings.cost

    ####
    # Test using the same workflow again
    ####
    new_docs = [
        Document(content="Doc CC"),
        Document(content="Doc DD"),
    ]
    cache_list_docs = {'cache': new_docs}
    new_result = workflow(new_docs)
    ####
    # Test chat model
    ####
    new_initial_prompt = "Doc CC Doc DD"
    new_first_response = "Response: " + new_initial_prompt
    new_second_prompt = new_first_response
    new_second_response = "Response: " + new_second_prompt

    # the final result should be the response returned by the second invokation of chat()
    assert new_result == new_second_response
    # check that the prompts/responses got propegated through the workflow
    assert len(chat.history()) == 4
    # these should not have changed from last time
    assert chat.history()[0].prompt == initial_prompt
    assert chat.history()[0].response == first_response
    assert chat.history()[0].input_tokens == len(initial_prompt)
    assert chat.history()[0].response_tokens == len(first_response)
    assert chat.history()[0].total_tokens == len(initial_prompt) + len(first_response)
    assert chat.history()[0].cost == chat.history()[0].total_tokens * cost_per_token_chat
    assert chat.history()[1].prompt == second_prompt
    assert chat.history()[1].response == second_response
    assert chat.history()[1].input_tokens == len(second_prompt)
    assert chat.history()[1].response_tokens == len(second_response)
    assert chat.history()[1].total_tokens == len(second_prompt) + len(second_response)
    assert chat.history()[1].cost == chat.history()[1].total_tokens * cost_per_token_chat
    # test the new history
    assert chat.history()[2].prompt == new_initial_prompt
    assert chat.history()[2].response == new_first_response
    assert chat.history()[2].input_tokens == len(new_initial_prompt)
    assert chat.history()[2].response_tokens == len(new_first_response)
    assert chat.history()[2].total_tokens == len(new_initial_prompt) + len(new_first_response)
    assert chat.history()[2].cost == chat.history()[2].total_tokens * cost_per_token_chat
    assert chat.history()[3].prompt == new_second_prompt
    assert chat.history()[3].response == new_second_response
    assert chat.history()[3].input_tokens == len(new_second_prompt)
    assert chat.history()[3].response_tokens == len(new_second_response)
    assert chat.history()[3].total_tokens == len(new_second_prompt) + len(new_second_response)
    assert chat.history()[3].cost == chat.history()[3].total_tokens * cost_per_token_chat

    # test chat totals
    assert chat.input_tokens == chat.history()[0].input_tokens + \
        chat.history()[1].input_tokens + \
        chat.history()[2].input_tokens + \
        chat.history()[3].input_tokens
    assert chat.response_tokens == chat.history()[0].response_tokens + \
        chat.history()[1].response_tokens + \
        chat.history()[2].response_tokens + \
        chat.history()[3].response_tokens
    assert chat.total_tokens == chat.history()[0].total_tokens + \
        chat.history()[1].total_tokens + \
        chat.history()[2].total_tokens + \
        chat.history()[3].total_tokens
    assert chat.cost == chat.history()[0].cost + \
        chat.history()[1].cost + \
        chat.history()[2].cost + \
        chat.history()[3].cost

    ####
    # Test embeddings model
    ####
    assert len(embeddings.history()) == 4
    # original history should not have changed
    assert embeddings.history()[0].total_tokens == len(docs[0].content) + len(docs[1].content)
    assert embeddings.history()[0].cost == embeddings.history()[0].total_tokens * cost_per_token_embedding  # noqa
    assert embeddings.history()[1].total_tokens == len("Response:DocADocB")
    assert embeddings.history()[1].cost == embeddings.history()[1].total_tokens * cost_per_token_embedding  # noqa
    # test new history
    assert embeddings.history()[2].total_tokens == len(new_docs[0].content) + len(new_docs[1].content)  # noqa
    assert embeddings.history()[2].cost == embeddings.history()[2].total_tokens * cost_per_token_embedding  # noqa
    assert embeddings.history()[3].total_tokens == len("Response:DocCCDocDD")
    assert embeddings.history()[3].cost == embeddings.history()[3].total_tokens * cost_per_token_embedding  # noqa

    assert embeddings.total_tokens == embeddings.history()[0].total_tokens + \
        embeddings.history()[1].total_tokens + \
        embeddings.history()[2].total_tokens + \
        embeddings.history()[3].total_tokens
    assert embeddings.cost == embeddings.history()[0].cost + \
        embeddings.history()[1].cost + \
        embeddings.history()[2].cost + \
        embeddings.history()[3].cost

    ####
    # Test workflow
    ####
    # because the `chat` model is included twice in the workflow; this check ensures we are not
    # double-counting the totals
    assert workflow.history() == [
        embeddings.history()[0], chat.history()[0], embeddings.history()[1], chat.history()[1],
        embeddings.history()[2], chat.history()[2], embeddings.history()[3], chat.history()[3],
    ]
    assert workflow.history(ExchangeRecord) == chat.history()
    assert workflow.history(EmbeddingRecord) == embeddings.history()
    assert workflow.sum('total_tokens') == chat.total_tokens + embeddings.total_tokens
    assert workflow.sum('total_tokens', types=EmbeddingRecord) == embeddings.total_tokens
    assert workflow.sum('cost') == chat.cost + embeddings.cost

def test_workflow_with_empty_history():  # noqa
    class EmptyHistory:
        def history(self) -> list:
            return []

    class NoneHistory:
        def history(self) -> list:
            return None

    workflow = Workflow(tasks=[EmptyHistory()])
    assert workflow.history() == []
    assert workflow.history(ExchangeRecord) == []
    assert workflow.history(EmbeddingRecord) == []
    workflow = Workflow(tasks=[NoneHistory()])
    assert workflow.history() == []
    assert workflow.history(ExchangeRecord) == []
    assert workflow.history(EmbeddingRecord) == []
    workflow = Workflow(tasks=[EmptyHistory(), NoneHistory()])
    assert workflow.history() == []
    assert workflow.history(ExchangeRecord) == []
    assert workflow.history(EmbeddingRecord) == []
