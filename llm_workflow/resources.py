"""Contains resources such as shared variables."""

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
