"""Misc helper functions and classes."""
from functools import cache
import re
import requests
import tiktoken
from tiktoken import Encoding
from llm_workflow.base import Document
from llm_workflow.exceptions import RequestError


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


@cache
def _get_encoding_for_model(model_name: str) -> Encoding:
    """Gets the encoding for a given model so that we can calculate the number of tokens."""
    return tiktoken.encoding_for_model(model_name)


def num_tokens(model_name: str, value: str) -> int:
    """For a given model, returns the number of tokens based on the str `value`."""
    return len(_get_encoding_for_model(model_name=model_name).encode(value))


def num_tokens_from_messages(model_name: str, messages: list[dict]) -> int:
    """
    Copied from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    Returns the number of tokens used by a list of messages.
    """
    if model_name in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model_name == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model_name:
        # Warning: gpt-3.5-turbo may update over time.
        # Returning num tokens assuming gpt-3.5-turbo-0613
        return num_tokens_from_messages(model_name="gpt-3.5-turbo-0613", messages=messages)
    elif "gpt-4" in model_name:
        # Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.
        return num_tokens_from_messages(model_name="gpt-4-0613", messages=messages)
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model_name}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")  # noqa
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(_get_encoding_for_model(model_name=model_name).encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
