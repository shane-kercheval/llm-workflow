"""Contains resources such as shared variables."""

MODEL_COST_PER_TOKEN = {
    'text-embedding-ada-002': 0.0001 / 1_000,
    'gpt-4': {'input': 0.03 / 1_000, 'output': 0.06 / 1_000},
    'gpt-4-32k': {'input': 0.06 / 1_000, 'output': 0.12 / 1_000},
    'gpt-3.5-turbo': {'input': 0.0015 / 1_000, 'output': 0.002 / 1_000},
    'pt-3.5-turbo-16k': {'input': 0.003 / 1_000, 'output': 0.004 / 1_000},
}

####
# Prompts
####
PROMPT_TEMPLATE__INCLUDE_DOCUMENTS = \
"""
Answer the question at the end of the text as truthfully and accurately as possible, based on the following information provided.

Here is the information:

```
{{documents}}
```

Here is the question:

```
{{prompt}}
```
"""  # noqa: E501
