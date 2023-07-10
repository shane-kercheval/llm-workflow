# `llm-chain`: simple and extensible LLM chaining

A `chain` is an object that executes a sequence of tasks referred to as `links`. Each `link` is a callable that can optionally track history. **The output of one `link` serves as the input to the next `link` in the chain.** Pretty simple.

The purpose of this library is to offer a simple pattern for developing LLM workflows (chains). First, it reduces the need for users to write repetitive/boilerplate code. Second, by establishing a standardized interface for links (e.g. specifying how a task tracks history), a chain can serve as a means of aggregating information from all links, such as token usage, costs, and more. Additionally, this approach enables us to examine each step within the chain and within a specific link, making the workflow transparent and facilitating debugging and comprehension.

---

Here's an example of a simple "prompt enhancer", where the first OpenAI model enhances the user's prompt and the second OpenAI model provides a response based on the enhanced prompt. This example is described in greater detail in the `Examples` section below and the corresponding notebook.

Here's the user's original prompt:

```python
prompt = "create a function to mask all emails from a string value"
```

The `prompt` variable is passed to the `chain` object, which gets passed to the first task (the `prompt_template` function).

```python
prompt_enhancer = OpenAIChat(...)
chat_assistant = OpenAIChat(...)

def prompt_template(user_prompt: str) -> str:
    return "Improve the user's request, below, by expanding the request " \
        "to describe the relevant python best practices and documentation " \
        f"requirements that should be followed:\n\n```{user_prompt}```"

def prompt_extract_code(_) -> str:
    # `_` signals that we are ignoring the input (from the previous link)
    return "Return only the primary code of interest from the previous answer, "\
        "including docstrings, but without any text/response."

chain = Chain(links=[
    prompt_template,      # modifies the user's prompt
    prompt_enhancer,      # returns an improved version of the user's prompt
    chat_assistant,       # returns the chat response based on the improved prompt
    prompt_extract_code,  # prompt to ask the model to extract only the relevant code
    chat_assistant,       # returns only the relevant code from the model's last response
])
response = chain(prompt)

print(response)               # ```python\n def mask_email_addresses(string): ...
print(chain.cost)             # 0.0034
print(chain.total_tokens)     # 1961
print(chain.prompt_tokens)    # 1104
print(chain.response_tokens)  # 857
print(chain.history)          # list of Record objects containing prompt/response/usage
```

See `Examples` section below for full output and explanation.

---

Note: Since the goal of this library is to provide a simple pattern for chaining, some of the classes provided in this library (e.g. `OpenAIEmbedding` or `ChromaDocumentIndex`) are nothing more than simple wrappers that implement the interface necessary to track history in a consistent way, allowing the chain to aggregate the history across all links. This makes it easy for people to create their own wrappers and workflows.

See examples below.

---

# Installing

**Note: This package is tested on Python versions `3.10` and `3.11`**

```commandline
pip install llm-chain
```

## API Keys

- **Any code in this library that uses OpenAI assumes that the `OPENAI_API_KEY` environment variable is set to a 
valid OpenAI API key. This includes many of the notebooks in the `examples` directory.**
- The `llm_workflow.utils.search_stack_overflow()` function assumes that the `STACK_OVERFLOW_KEY` environment variable is set. To use that function, you must create an account and app at [Stack Apps](https://stackapps.com/) and use the `key` that is generated (not the `secret`) to set the environment variable.

Here are two ways you can set environment variables directly in Python:

```python
import os
os.environ['OPENAI_API_KEY'] = 'sk-...'
```

Or you can create a file named `.env` (in your project/notebook directory) that contains the key(s) and value(s) in the format below, and use the `dotenv` library and `load_dotenv` function.

```
OPENAI_API_KEY=sk-...
STACK_OVERFLOW_KEY=...
```

```python
from dotenv import load_dotenv
load_dotenv()  # sets any environment variables from the .env file
```

---

# Examples

## Example 1

Here's the full example from the snippet above (alternatively, see [this notebook](https://github.com/shane-kercheval/llm-chain/tree/main/examples/chains.ipynb)). Note that this example is **not** meant to provide prompt-engineering best practices, it's simply to show how chaining works in this library.

- first link: defines a prompt-template that takes the user's prompt, and creates a new prompt asking a chat model to improve the prompt (within the context of creating python code)
- second link: the `prompt_enhancer` model takes the modified prompt and improves the prompt
- third link: the `chat_assistant` model takes the response from the last model (which is an improved prompt) and returns the request
- fourth link: ignores the response from the chat model; creates a new prompt asking the chat model to extract the relevant code created in the previous response
- fifth link: the chat model, which internally maintains the history of messages, returns only the relevant code from the previous response.

```python
from llm_workflow.base import Chain
from llm_workflow.models import OpenAIChat

prompt_enhancer = OpenAIChat(model_name='gpt-3.5-turbo')
# different model/object, therefore different message history (i.e. conversation)
chat_assistant = OpenAIChat(model_name='gpt-3.5-turbo')

def prompt_template(user_prompt: str) -> str:
    return "Improve the user's request, below, by expanding the request " \
        "to describe the relevant python best practices and documentation " \
        f"requirements that should be followed:\n\n```{user_prompt}```"

def prompt_extract_code(_) -> str:
    # `_` signals that we are ignoring the input (from the previous link)
    return "Return only the primary code of interest from the previous answer, "\
        "including docstrings, but without any text/response."

# The only requirement for the list of links is that each item/link is a
# callable where the output of one task matches the input of the next link.
# The input to the chain is passed to the first link;
# the output of the last task is returned by the chain.
chain = Chain(links=[
    prompt_template,      # modifies the user's prompt
    prompt_enhancer,      # returns an improved version of the user's prompt
    chat_assistant,       # returns the chat response based on the improved prompt
    prompt_extract_code,  # prompt to ask the model to extract only the relevant code
    chat_assistant,       # returns only the function from the model's last response
])
prompt = "create a function to mask all emails from a string value"
response = chain(prompt)
print(response)
```

The output of the chain (`response`):

```python
import re

def mask_email_addresses(string):
    """
    Mask email addresses within a given string.

    Args:
        string (str): The input string to be processed.

    Returns:
        str: The modified string with masked email addresses.

    Raises:
        ValueError: If the input is not a string.

    Dependencies:
        The function requires the 're' module to use regular expressions.
    """
    if not isinstance(string, str):
        raise ValueError("Input must be a string")

    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    masked_string = re.sub(pattern, '[email protected]', string)

    return masked_string
```

Total costs/tokens for all activity in the chain:

```python
print(f"Cost:             ${chain.cost:.4f}")
print(f"Total Tokens:      {chain.total_tokens:,}")
print(f"Prompt Tokens:     {chain.prompt_tokens:,}")
print(f"Response Tokens:   {chain.response_tokens:,}")
```

Output:

```
Cost:            $0.0034
Total Tokens:     1,961
Prompt Tokens:    1,104
Response Tokens:  857
```

We can view the history of the chain (i.e. the aggregated history across all links) with the `chain.history` property. 

In this example, the only class that tracks history is `OpenAIChat`. Therefore, both the `prompt_enhancer` and `chat_assistant` objects will contain history. `chain.history` will return a list of three `ExchangeRecord` objects. The first record corresponds to our request to the `prompt_enhancer`, and the second two records correspond to our `chat_assistant` requests. An ExchangeRecord represents a single exchange/transaction with an LLM, encompassing an input (`prompt`) and its corresponding output (`response`), along with other properties like `cost` and `token_tokens`.

We can view the response we received from the `prompt_enhancer` model by looking at the first record's `response` property (or the second record's `prompt` property since the chain passes the output of `prompt_enhancer` as the input to the `chat_assistant`):

```python
print(chain.history[0].response)
```

Output:

```
Create a Python function that adheres to best practices and follows proper documentation guidelines to mask all email addresses within a given string value. The function should take a string as input and return the modified string with masked email addresses.

To ensure code readability and maintainability, follow these best practices:

1. Use meaningful function and variable names that accurately describe their purpose.
2. Break down the problem into smaller, reusable functions if necessary.
3. Write clear and concise code with proper indentation and comments.
4. Handle exceptions and errors gracefully by using try-except blocks.
5. Use regular expressions to identify and mask email addresses within the string.
6. Avoid using global variables and prefer passing arguments to functions.
7. Write unit tests to verify the correctness of the function.

In terms of documentation, follow these guidelines:

1. Provide a clear and concise function description, including its purpose and expected behavior.
2. Specify the input parameters and their types, along with any default values.
3. Document the return value and its type.
4. Mention any exceptions that the function may raise and how to handle them.
5. Include examples of how to use the function with different inputs and expected outputs.
6. Document any dependencies or external libraries required for the function to work.

By following these best practices and documentation guidelines, you will create a well-structured and maintainable Python function to mask email addresses within a string value.
```

We could also view the original response from the `chat_assistant` model.


```python
mprint(chain.history[1].response)
```

Output:

```
Sure! Here's an example of a Python function that adheres to best practices and follows proper documentation guidelines to mask email addresses within a given string value:

import re

def mask_email_addresses(string):
    """
    Mask email addresses within a given string.

    Args:
        string (str): The input string to be processed.

    Returns:
        str: The modified string with masked email addresses.

    Raises:
        ValueError: If the input is not a string.

    Examples:
        >>> mask_email_addresses("Contact us at john.doe@example.com")
        'Contact us at [email protected]'

        >>> mask_email_addresses("Email me at john.doe@example.com or jane.doe@example.com")
        'Email me at [email protected] or [email protected]'

        >>> mask_email_addresses("No email addresses here")
        'No email addresses here'

    Dependencies:
        The function requires the 're' module to use regular expressions.
    """
    if not isinstance(string, str):
        raise ValueError("Input must be a string")

    # Regular expression pattern to match email addresses
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'

    # Replace email addresses with masked version
    masked_string = re.sub(pattern, '[email protected]', string)

    return masked_string

In this example, the mask_email_addresses function takes a string as input and returns the modified string with masked email addresses. It uses the re module to match email addresses using a regular expression pattern. The function raises a ValueError if the input is not a string.

The function is properly documented with a clear and concise description, input parameters with their types, return value and its type, exceptions that may be raised, examples of usage, and any dependencies required.

To use this function, you can call it with a string as an argument and it will return the modified string with masked email addresses.
```

The final response returned by the `chat_assistant` (and by the `chain` object) returns only the `mask_email_addresses` function. The `response` object should match the `response` value in the last record (`chain.history[-1].response`).

```python
assert response == chain.history[-1].response  # passes
```

---

## Example 2

Here's an example of using a chain to perform the following series of tasks:

- Ask a question.
- Perform a web search based on the question.
- Scrape the top_n web pages from the search results.
- Split the web pages into chunks (so that we can search for the most relevant chunks).
- Save the chunks to a document index (i.e. vector database).
- Create a prompt that includes the original question and the most relevant chunks.
- Send the prompt to the chat model.

**Again, the key concept of a chain is simply that the output of one task is the input of the next link.** So, in the code below, you can replace any step with your own implementation as long as the input/output matches the task you replace.

Something that may not be immediately obvious is the usage of the `Value` object, below. It serves as a convenient caching mechanism within the chain. The `Value` object is callable, allowing it to cache and return the value when provided as an argument. When called without a value, it retrieves and returns the cached value. In the example below, the `Value` object is used to cache the original question, pass it to the web search (i.e. the `duckduckgo_search` object), and subsequently reintroduce the question into the chain (i.e. into the `prompt_template` object).

See [this notebook](https://github.com/shane-kercheval/llm-chain/tree/main/examples/chains.ipynb) for an in-depth explanation.

```python
from llm_workflow.base import Document, Chain, Value
from llm_workflow.models import OpenAIEmbedding, OpenAIChat
from llm_workflow.tools import DuckDuckGoSearch, scrape_url, split_documents
from llm_workflow.indexes import ChromaDocumentIndex
from llm_workflow.prompt_templates import DocSearchTemplate

duckduckgo_search = DuckDuckGoSearch(top_n=3)
embeddings_model = OpenAIEmbedding(model_name='text-embedding-ada-002')
document_index = ChromaDocumentIndex(embeddings_model=embeddings_model, n_results=3)
prompt_template = DocSearchTemplate(doc_index=document_index, n_docs=3)
chat_model = OpenAIChat(model_name='gpt-3.5-turbo')

def scrape_urls(search_results):
    """
    For each url (i.e. `href` in `search_results`):
    - extracts text
    - replace new-lines with spaces
    - create a Document object
    """
    return [
        Document(content=re.sub(r'\s+', ' ', scrape_url(x['href'])))
        for x in search_results
    ]

question = Value()  # Value is a caching/reinjection mechanism; see note above

# each task is a callable where the output of one task is the input to the next link
chain = Chain(links=[
    question,
    duckduckgo_search,
    scrape_urls,
    split_documents,
    document_index,
    question,
    prompt_template,
    chat_model,
])
chain("What is ChatGPT?")
```

Response:

```
ChatGPT is an AI chatbot that is driven by AI technology and is a natural language processing tool. It allows users to have human-like conversations and can assist with tasks such as composing emails, essays, and code. It is built on a family of large language models known as GPT-3 and has now been upgraded to GPT-4 models. ChatGPT can understand and generate human-like answers to text prompts because it has been trained on large amounts of data.'
```

We can also track costs:

```python
print(f"Cost:            ${chain.cost:.4f}")
print(f"Total Tokens:     {chain.total_tokens:,}")
print(f"Prompt Tokens:    {chain.prompt_tokens:,}")
print(f"Response Tokens:  {chain.response_tokens:,}")
print(f"Embedding Tokens: {chain.embedding_tokens:,}")
```

Output:

```
Cost:            $0.0024
Total Tokens:     16,108
Prompt Tokens:    407
Response Tokens:  97
Embedding Tokens: 15,604
```

Additionally, we can track the history of the chain with the `chain.history` property. See [this notebook](https://github.com/shane-kercheval/llm-chain/tree/main/examples/chains.ipynb) for an example.

---

## Notebooks

- [chains.ipynb](https://github.com/shane-kercheval/llm-chain/tree/main/examples/chains.ipynb)
- [openai_chat.ipynb](https://github.com/shane-kercheval/llm-chain/tree/main/examples/openai_chat.ipynb)
- [indexes.ipynb](https://github.com/shane-kercheval/llm-chain/tree/main/examples/indexes.ipynb)
- [prompt_templates.ipynb](https://github.com/shane-kercheval/llm-chain/tree/main/examples/prompt_templates.ipynb)
- [memory.ipynb](https://github.com/shane-kercheval/llm-chain/tree/main/examples/memory.ipynb)
- [duckduckgo-web-search.ipynb](https://github.com/shane-kercheval/llm-chain/tree/main/examples/duckduckgo-web-search.ipynb)
- [scraping-urls.ipynb](https://github.com/shane-kercheval/llm-chain/tree/main/examples/scraping-urls.ipynb)
- [splitting-documents.ipynb](https://github.com/shane-kercheval/llm-chain/tree/main/examples/splitting-documents.ipynb)
- [search-stack-overflow.ipynb](https://github.com/shane-kercheval/llm-chain/tree/main/examples/search-stack-overflow.ipynb)
- [conversation-between-models.ipynb](https://github.com/shane-kercheval/llm-chain/tree/main/examples/conversation-between-models.ipynb)

---

# Possible Features

- [ ] Add `metadata` property to OpenAIChat which gets passed to the ExchangeRecord that are created
- [ ] Explore OpenAI functions
- [ ] Explore 'agents'
- [ ] Create PDF-Loader
- [ ] create additional prompt-templates

---

## Contributing

Contributions to this project are welcome. Please follow the coding standards, add appropriate unit tests, and ensure that all linting and tests pass before submitting pull requests.

### Coding Standards

- Coding standards should follow PEP 8 (Style Guide for Python Code).
    - https://peps.python.org/pep-0008/
    - Exceptions:
        - Use a maximum line length of 99 instead of the suggested 79.
- Document all files, classes, and functions following the existing documentation style.

### Docker

See the Makefile for all available commands.

To build the Docker container:

```commandline
make docker_build
```

To run the terminal inside the Docker container:

```commandline
make docker_zsh
```

To run the unit tests (including linting and doctests) from the command line inside the Docker container:

```commandline
make tests
```

To run the unit tests (including linting and doctests) from the command line outside the Docker container:

```commandline
make docker_tests
```

### Pre-Check-in

#### Unit Tests

The unit tests in this project are all found in the `/tests` directory.

In the terminal, in the project directory, run the following command to run linting and unit-tests:

```commandline
make tests
```
