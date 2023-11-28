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

PROMPT_TEMPLATE__PYTHON_METADATA = \
"""
Answer the question at the end of the text as truthfully and accurately as possible. Use the metadata of the python objects as appropriate. Tailor your response according to the most relevant objects. Don't use the metadata if they don't appear relevant.

Here is the metadata:

```
{{metadata}}
```

----

Here is the question:

```
{{prompt}}
```
"""  # noqa: E501
