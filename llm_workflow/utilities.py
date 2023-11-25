"""Misc helper functions and classes."""

from datetime import datetime
from itertools import islice
import os
import re
import requests
from pydantic import BaseModel, Field, field_validator

from llm_workflow.base import Document, RecordKeeper, SearchRecord
from llm_workflow.exceptions import RequestError
from llm_workflow.internal_utilities import retry_handler


def split_documents(
        docs: list[Document],
        max_chars: int = 500,
        preserve_words: bool = True) -> list[Document]:
    """
    split_documents divides a list of documents into smaller segments based on the specified
    maximum character limit for each individual Document object.

    Args:
        docs:
            A list of Document objects to be segmented.
        max_chars:
            The maximum allowable character count for each Document object.
        preserve_words:
            If set to True, guarantees that the document segmentation does not occur in the middle
            of a word, ensuring the integrity of the entire word.
    """
    new_docs = []
    for doc in docs:
        if doc.content:
            content = doc.content.strip()
            metadata = doc.metadata
            while len(content) > max_chars:
                # find the last space that is within the limit
                if preserve_words:
                    split_at = max_chars
                    # find the last whitespace that is within the limit
                    # if no whitespace found, take the whole chunk
                    # we are going 1 beyond the limit in case it is whitespace so that
                    # we can keep everything up until that point
                    for match in re.finditer(r'\s', content[:max_chars + 1]):
                        split_at = match.start()
                else:
                    split_at = max_chars
                new_doc = Document(
                    content=content[:split_at].strip(),
                    metadata=metadata,
                )
                new_docs.append(new_doc)
                content = content[split_at:].strip()  # remove the chunk we added
            # check for remaining/leftover content
            if content:
                new_docs.append(Document(
                    content=content,
                    metadata=metadata,
                ))
    return new_docs


def scrape_url(url: str) -> str:
    """
    Scrapes the content of a given URL and returns it as a string.

    Args:
        url: The URL to scrape.
    """
    from bs4 import BeautifulSoup
    response = requests.get(url)
    if response.status_code != 200:
        raise RequestError(status_code=response.status_code, reason=response.reason)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text().strip()


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

    def _get_history(self) -> list[SearchRecord]:
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
    creation_date: int
    text: str | None = None
    markdown: str | None = None

    def __init__(self, **data):  # noqa
        from bs4 import BeautifulSoup
        from markdownify import markdownify as html_to_markdown
        super().__init__(**data)
        self.text = BeautifulSoup(self.body, 'html.parser').get_text(separator=' ')
        self.markdown = html_to_markdown(self.body)

    @field_validator('creation_date')
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
    text: str | None = None
    markdown: str | None = None
    content_license: str | None = None
    answers: list[StackAnswer] = Field(default_factory=list)

    def __init__(self, **data):  # noqa
        from bs4 import BeautifulSoup
        from markdownify import markdownify as html_to_markdown
        super().__init__(**data)
        self.text = BeautifulSoup(self.body, 'html.parser').get_text(separator=' ')
        self.markdown = html_to_markdown(self.body)

    @field_validator('creation_date')
    def convert_to_datetime(cls, value: str) -> datetime:  # noqa: N805
        """Convert from string to datetime."""
        return datetime.fromtimestamp(value)


class StackOverflowSearchRecord(SearchRecord):
    """
    Represents a single query to StackOverflowSearch.

    `query` is the original user-query.

    `questions` are the Stack Overflow questions that match the query.
    """

    results: list[StackQuestion]


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
                    results=search.results.copy(),
                ))
                return search.results

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
            results=questions,
        ))
        return questions

    def _get_history(self) -> list[StackOverflowSearchRecord]:
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
