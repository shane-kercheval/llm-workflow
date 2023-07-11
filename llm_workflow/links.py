"""
Built-in "Links" used for chains. Most of these classes will be simple wrappers that implement
history for common tasks.

They also serve as a pattern to build your own links. Remember, a chain only requires that links
are callable. You only need to create a Task class if you care about history for that particular
link. History helps track usage/costs, and also helps track intermediate steps in the chain.
"""
import os
from itertools import islice
import requests
from datetime import datetime
from pydantic import BaseModel, Field, validator
from llm_workflow.internal_utilities import retry_handler
from llm_workflow.base import RecordKeeper, Record
from llm_workflow.exceptions import RequestError


class SearchRecord(Record):
    """Contains the query and requests from a search request (e.g. from a web-search)."""

    query: str
    results: list[dict]


class DuckDuckGoSearch(RecordKeeper):
    """
    A wrapper around DuckDuckGo web search. The object is called with a query and returns a list of
    SearchRecord objects associated with the top search results.
    """

    def __init__(self, top_n: int = 3):
        self.top_n = top_n
        self._history = []

    def __call__(self, query: str) -> list[dict]:
        """
        The object is called with a query and returns a list of SearchRecord objects associated
        with the `top_n` search results. `top_n` is set during object initialization.
        """
        from duckduckgo_search import DDGS

        for search in self._history:
            # if we have previously searched based on the same query, return the results
            if query == search.query:
                # add a new SearchRecord object to the history so that we correctly account for all
                # searches; create a new object that has a different uuid and timestamp
                self._history.append(SearchRecord(
                    query=query,
                    results=search.results,
                ))
                return search.results

        with DDGS() as ddgs:
            ddgs_generator = ddgs.text(query, region='wt-wt', safesearch='Off', timelimit='y')
            results = list(islice(ddgs_generator, self.top_n))
        self._history.append(SearchRecord(
            query=query,
            results=results,
        ))
        return results

    @property
    def history(self) -> list[SearchRecord]:
        """A history of web-searches (list of SearchResult objects)."""
        return self._history


class StackAnswer(BaseModel):
    """
    An object that encapsulates the contents of a Stack Overflow answer.

    The body property holds the original answer in HTML format. The text property provides the
    same content with HTML tags removed, while the markdown property offers the content converted
    to Markdown format.
    """

    answer_id: int
    is_accepted: bool
    score: int
    body: str
    text: str | None
    markdown: str | None
    creation_date: int

    def __init__(self, **data):  # noqa
        from bs4 import BeautifulSoup
        from markdownify import markdownify as html_to_markdown
        super().__init__(**data)
        self.text = BeautifulSoup(self.body, 'html.parser').get_text(separator=' ')
        self.markdown = html_to_markdown(self.body)


    @validator('creation_date')
    def convert_to_datetime(cls, value: str) -> datetime:  # noqa: N805
        """Convert from string to datetime."""
        return datetime.fromtimestamp(value)


class StackQuestion(BaseModel):
    """
    An object that encapsulates the contents of a Stack Overflow question.

    The body property holds the original question in HTML format. The text property provides the
    same content with HTML tags removed, while the markdown property offers the content converted
    to Markdown format.
    """

    question_id: int
    score: int
    creation_date: int
    answer_count: int
    title: str
    link: str
    body: str
    text: str | None
    markdown: str | None
    content_license: str | None
    answers: list[StackAnswer] = Field(default_factory=list)

    def __init__(self, **data):  # noqa
        from bs4 import BeautifulSoup
        from markdownify import markdownify as html_to_markdown
        super().__init__(**data)
        self.text = BeautifulSoup(self.body, 'html.parser').get_text(separator=' ')
        self.markdown = html_to_markdown(self.body)

    @validator('creation_date')
    def convert_to_datetime(cls, value: str) -> datetime:  # noqa: N805
        """Convert from string to datetime."""
        return datetime.fromtimestamp(value)


class StackOverflowSearchRecord(Record):
    """
    Represents a single query to StackOverflowSearch.

    `query` is the original user-query.

    `questions` are the Stack Overflow questions that match the query.
    """

    query: str
    questions: list[StackQuestion]


class StackOverflowSearch(RecordKeeper):
    """
    Retrieves the top relevant Stack Overflow questions based on a given search query.
    The function assumes that the STACK_OVERFLOW_KEY environment variable is properly set and
    contains a valid Stack Overflow API key.

    To obtain an API key, please create an account and an app at [Stack Apps](https://stackapps.com/).
    Use the `key` generated for your app (not the `secret`).
    """

    def __init__(self, max_questions: int = 2, max_answers: int = 2):
        """
        Args:
            max_questions:
                The maximum number of questions to be returned. If the API doesn't find enough
                relevant questions, fewer questions may be returned.
            max_answers:
                The maximum number of answers to be returned with each question. If the number of
                answers is fewer than `max_answers`, fewer answers will be returned.
        """
        self.max_questions = max_questions
        self.max_answers = max_answers
        self._history = []

    def __call__(self, query: str) -> list[StackQuestion]:
        """
        Args:
            query:
                The search query used to find relevant Stack Overflow questions.
        """
        for search in self._history:
            # if we have previously searched based on the same query, return the results
            if query == search.query:
                # add a new StackOverflowSearchRecord object to the history so that we correctly
                # account for all searches; create a new object that has a different uuid and
                # timestamp
                self._history.append(StackOverflowSearchRecord(
                    query=query,
                    questions=search.questions.copy(),
                ))
                return search.questions

        params = {
            'site': 'stackoverflow',
            'key': os.getenv('STACK_OVERFLOW_KEY'),
            'q': query,
            'sort': 'relevance',
            'order': 'desc',
            'filter': 'withbody',  # Include the question body in the response
            'pagesize': self.max_questions,
            'page': 1,
        }
        response = retry_handler()(
            requests.get,
            'https://api.stackexchange.com/2.3/search/advanced',
            params=params,
        )
        assert response.status_code == 200
        questions = response.json().get('items', [])
        questions = [StackQuestion(**x) for x in questions]

        for question in questions:
            if question.answer_count > 0:
                question.answers = _get_stack_overflow_answers(
                    question_id=question.question_id,
                    max_answers=self.max_answers,
                )
        self._history.append(StackOverflowSearchRecord(
            query=query,
            questions=questions,
        ))
        return questions

    @property
    def history(self) -> list[StackOverflowSearchRecord]:
        """A history of StackOverflow searches (list of StackOverflowSearchRecord objects)."""
        return self._history


def _get_stack_overflow_answers(question_id: int, max_answers: int = 2) -> list[StackAnswer]:
    """For a given question_id on Stack Overflow, returns the top `num_answers`."""
    params = {
        'site': 'stackoverflow',
        'key': os.getenv('STACK_OVERFLOW_KEY'),
        'filter': 'withbody',  # Include the answer body in the response
        'sort': 'votes',  # Sort answers by votes (highest first)
        'order': 'desc',
        'pagesize': max_answers,  # Fetch only the top answers
    }
    response = retry_handler()(
        requests.get,
        f'https://api.stackexchange.com/2.3/questions/{question_id}/answers',
        params=params,
    )
    if response.status_code != 200:  # let client decide what to do
        raise RequestError(status_code=response.status_code, reason=response.reason)
    answers = response.json().get('items', [])
    return [StackAnswer(**x) for x in answers]
