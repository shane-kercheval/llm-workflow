"""Tests the compare module."""

import os
from textwrap import dedent

import pytest
from llm_workflow.compare import Scenario, ModelDefinition, CompareModels


class MockChatModel:
    """Mock chat model for testing."""

    def __init__(self, prompts: list[str], responses: list[str], cost: float | None = None):
        """
        Initialize the mock chat model. The model will respond to each prompt with the
        corresponding response (i.e. the first prompt will be responded to with the first
        response, the second prompt with the second response, etc.).
        """
        assert len(prompts) == len(responses)
        if cost is not None:
            self.cost = cost
        self.responses = dict(zip(prompts, responses))
        assert len(self.responses) == len(responses)

    def __call__(self, prompt: str) -> str:
        """Returns the response to the prompt."""
        return self.responses[prompt]


def test_scenario__conversation_sum__model_1__with_costs(conversation_sum):  # noqa
    prompts = conversation_sum['prompts']
    model_1_responses = conversation_sum['model_1']['responses']
    model = MockChatModel(
        prompts=prompts,
        responses=model_1_responses,
        cost=0.5,
    )
    scenario = Scenario(model=model, description="Mock Chat Model")
    scenario(prompts)
    assert str(scenario)
    assert scenario.description == "Mock Chat Model"
    assert scenario.prompts == prompts
    assert scenario.responses == model_1_responses
    expected_code_block_1 = dedent("""
        def sum_numbers(num1, num2):
            return num1 + num2
        """).strip()
    expected_code_block_2 = dedent("""
        result = sum_numbers(5, 3)
        print(result)  # Output: 8
        """).strip()
    expected_code_block_3 = dedent("""
        assert sum_numbers(5, 3) == 8
        assert sum_numbers(-10, 10) == 0
        """).strip()
    assert scenario.code_blocks == [
        [
            expected_code_block_1,
            expected_code_block_2,
        ],
        [
            expected_code_block_3,
        ],
    ]
    assert scenario.code_namespace['result'] == 8  # from `result = sum_numbers(5, 3)`
    assert scenario.duration_seconds is not None
    assert scenario.duration_seconds > 0
    assert scenario.num_response_chars == len("".join(model_1_responses))
    assert scenario.response_chars_per_second > 0
    assert scenario.num_code_blocks == 3
    assert scenario.code_block_errors == [[None, None], [None]]
    assert scenario.num_successful_code_blocks == 3
    assert scenario.percent_successful_code_blocks == 1.0
    assert scenario.cost == 0.5

def test_scenario__conversation_sum__model_2__with_costs(conversation_sum):  # noqa
    prompts = conversation_sum['prompts']
    model_2_responses = conversation_sum['model_2']['responses']
    model = MockChatModel(
        prompts=prompts,
        responses=model_2_responses,
        cost=None,
    )
    scenario = Scenario(model=model, description="Mock Chat Model")
    scenario(prompts)
    assert str(scenario)
    assert scenario.description == "Mock Chat Model"
    assert scenario.prompts == prompts
    assert scenario.responses == model_2_responses
    expected_code_block_1 = dedent("""
        def sum_two_numbers(num1, num2):
            return num1 + num2
        """).strip()
    expected_code_block_2 = dedent("""
        result = sum_two_numbers(5, 3)
        print(result)  # Outputs: 8
        """).strip()
    expected_code_block_3 = dedent("""
        assert sum_two_numbers(5, 3) == 8, "Should be 8"
        assert sum_two_numbers(-1, 1) == 0, "Should be 0"
        assert sum_two_numbers(0, 0) == 0, "Should be 0"
        assert sum_two_numbers(100, 200) == 300, "Should be 300"
        """).strip()
    assert scenario.code_blocks == [
        [
            expected_code_block_1,
            expected_code_block_2,
        ],
        [
            expected_code_block_3,
        ],
    ]
    assert scenario.code_namespace['result'] == 8  # from `result = sum_numbers(5, 3)`
    assert scenario.duration_seconds is not None
    assert scenario.duration_seconds > 0
    assert scenario.num_response_chars == len("".join(model_2_responses))
    assert scenario.response_chars_per_second > 0
    assert scenario.num_code_blocks == 3
    assert scenario.code_block_errors == [[None, None], [None]]
    assert scenario.num_successful_code_blocks == 3
    assert scenario.percent_successful_code_blocks == 1.0
    assert scenario.cost is None

def test_scenario__conversation_mask_email__model_1__with_costs(conversation_mask_email):  # noqa
    prompts = conversation_mask_email['prompts']
    model_1_responses = conversation_mask_email['model_1']['responses']
    model = MockChatModel(
        prompts=prompts,
        responses=model_1_responses,
        cost=0.5,
    )
    scenario = Scenario(model=model, description="Mock Chat Model")
    scenario(prompts)
    assert str(scenario)
    assert scenario.description == "Mock Chat Model"
    assert scenario.prompts == prompts
    assert scenario.responses == model_1_responses
    expected_code_block_1 = dedent("""
        def mask_email(email):
            local_part, domain = email.split('@')
            masked_local_part = '*' * len(local_part)
            masked_email = masked_local_part + '@' + domain
            return masked_email
        """).strip()
    expected_code_block_2 = dedent("""
        email = 'example@example.com'
        masked_email = mask_email(email)
        print(masked_email)  # Output: ********@example.com
        """).strip()
    expected_code_block_3 = dedent("""
        # Test case 1: Masking email with alphanumeric local part
        email1 = 'example123@example.com'
        assert mask_email(email1) == '***********@example.com'

        # Test case 2: Masking email with special characters in local part
        email2 = 'ex@mple@example.com'
        assert mask_email(email2) == '******@example.com'
        """).strip()
    assert scenario.code_blocks == [
        [
            expected_code_block_1,
            expected_code_block_2,
        ],
        [
            expected_code_block_3,
        ],
    ]
    assert scenario.duration_seconds is not None
    assert scenario.duration_seconds > 0
    assert scenario.num_response_chars == len("".join(model_1_responses))
    assert scenario.response_chars_per_second > 0
    assert scenario.num_code_blocks == 3
    assert scenario.code_block_errors[0] == [None, None]
    assert isinstance(scenario.code_block_errors[1][0], AssertionError)
    assert scenario.num_successful_code_blocks == 2
    assert scenario.percent_successful_code_blocks == 2 /3
    assert scenario.cost == 0.5

def test_scenario__conversation_mask_email__model_2__without_costs(conversation_mask_email):  # noqa
    prompts = conversation_mask_email['prompts']
    model_2_responses = conversation_mask_email['model_2']['responses']
    model = MockChatModel(
        prompts=prompts,
        responses=model_2_responses,
        cost=None,
    )
    scenario = Scenario(model=model, description="Mock Chat Model")
    scenario(prompts)
    assert str(scenario)
    assert scenario.description == "Mock Chat Model"
    assert scenario.prompts == prompts
    assert scenario.responses == model_2_responses
    expected_code_block_1 = dedent("""
        def mask_email(email):
            try:
                email_parts = email.split('@')
                # Mask first part
                masked_part = email_parts[0][0] + "****" + email_parts[0][-1]
                # Combine masked part and domain
                masked_email = masked_part + '@' + email_parts[1]
                return masked_email
            except Exception as e:
                print("An error occurred: ", e)
                return None
        """).strip()
    expected_code_block_2 = dedent("""
        assert mask_email("john.doe@example.com") == "j****e@example.com"
        assert mask_email("jane_doe@example.com") == "j****e@example.com"
        assert mask_email("test@test.com") == "t****t@test.com"
        """).strip()
    assert scenario.code_blocks == [
        [
            expected_code_block_1,
        ],
        [
            expected_code_block_2,
        ],
    ]
    assert scenario.duration_seconds is not None
    assert scenario.duration_seconds > 0
    assert scenario.num_response_chars == len("".join(model_2_responses))
    assert scenario.response_chars_per_second > 0
    assert scenario.num_code_blocks == 2
    assert scenario.code_block_errors == [[None], [None]]
    assert scenario.num_successful_code_blocks == 2
    assert scenario.percent_successful_code_blocks == 1
    assert scenario.cost is None

def test_scenario__conversation_sum__model_regex__code_setup(conversation_mask_email):  # noqa
    ####
    # Without setup code the code blocks should fail because there is no `re` module
    ####
    prompts = conversation_mask_email['prompts']
    model_regex_responses = conversation_mask_email['model_regex']['responses']
    model = MockChatModel(
        prompts=prompts,
        responses=model_regex_responses,
        cost=None,
    )
    scenario = Scenario(
        model=model,
        description="Mock Chat Model",
    )
    scenario(prompts)
    assert str(scenario)
    assert scenario.description == "Mock Chat Model"
    assert scenario.prompts == prompts
    assert scenario.responses == model_regex_responses
    expected_code_block_1 = dedent(r"""
        def mask_emails(text, mask="*****@*****"):
            # Define a regular expression pattern for matching email addresses
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
            # Use re.sub() to replace matched email addresses with the mask
            masked_text = re.sub(email_pattern, mask, text)
            return masked_text

        # Example usage:
        text_with_emails = "Please contact john.doe@example.com for more information."
        masked_text = mask_emails(text_with_emails)
        print(masked_text)
        """).strip()
    expected_code_block_2 = dedent("""
        # Example usage:
        text_with_emails = "Please contact john.doe@example.com for more information."
        masked_text = mask_emails(text_with_emails)

        # Assertion 1: Check if the email address is masked in the result
        assert "john.doe@example.com" not in masked_text

        # Assertion 2: Check if the masked text has the correct format
        assert masked_text == "Please contact *****@***** for more information."
        """).strip()
    assert expected_code_block_1 == scenario.code_blocks[0][0]
    assert expected_code_block_2 == scenario.code_blocks[1][0]
    assert scenario.duration_seconds is not None
    assert scenario.duration_seconds > 0
    assert scenario.num_response_chars == len("".join(model_regex_responses))
    assert scenario.response_chars_per_second > 0
    assert scenario.num_code_blocks == 2
    assert isinstance(scenario.code_block_errors[0][0], NameError)
    assert isinstance(scenario.code_block_errors[1][0], NameError)
    assert scenario.num_successful_code_blocks == 0
    assert scenario.percent_successful_code_blocks == 0
    assert scenario.cost is None

    ####
    # With setup code, the code blocks should pass
    ####
    scenario = Scenario(
        model=model,
        description="Mock Chat Model",
        code_setup="import re",
    )
    scenario(prompts)
    assert str(scenario)
    assert scenario.description == "Mock Chat Model"
    assert scenario.prompts == prompts
    assert scenario.responses == model_regex_responses
    assert expected_code_block_1 == scenario.code_blocks[0][0]
    assert expected_code_block_2 == scenario.code_blocks[1][0]
      # from code that was executed
    assert scenario.code_namespace['masked_text'] == 'Please contact *****@***** for more information.'  # noqa
    assert scenario.duration_seconds is not None
    assert scenario.duration_seconds > 0
    assert scenario.num_response_chars == len("".join(model_regex_responses))
    assert scenario.response_chars_per_second > 0
    assert scenario.num_code_blocks == 2
    # now the code blocks should pass
    scenario.code_block_errors == [[None], [None]]
    assert scenario.num_successful_code_blocks == 2
    assert scenario.percent_successful_code_blocks == 1
    assert scenario.cost is None

def test_scenario__cannot_rerun(conversation_sum):  # noqa
    prompts = conversation_sum['prompts']
    model_1_responses = conversation_sum['model_1']['responses']
    model = MockChatModel(
        prompts=prompts,
        responses=model_1_responses,
        cost=0.5,
    )
    scenario = Scenario(model=model, description="Mock Chat Model")
    scenario(prompts)
    with pytest.raises(ValueError, match="Trial has already been run."):
        scenario(prompts)

def test_compare_models(conversation_sum, conversation_mask_email):  # noqa
    scenario_1_prompts = conversation_sum['prompts']
    scenario_2_prompts = conversation_mask_email['prompts']
    all_prompts = scenario_1_prompts + scenario_2_prompts

    scenario_1_model_1_responses = conversation_sum['model_1']['responses']
    scenario_1_model_2_responses = conversation_sum['model_2']['responses']
    scenario_2_model_1_responses = conversation_mask_email['model_1']['responses']
    scenario_2_model_2_responses = conversation_mask_email['model_2']['responses']

    model_creations = [
        ModelDefinition(
            create=lambda: \
                MockChatModel(
                    prompts=all_prompts,
                    responses=scenario_1_model_1_responses + scenario_2_model_1_responses,
                    cost=0.5,
                ),
            description='Mock Model 1',
        ),
        ModelDefinition(
            create=lambda: \
                MockChatModel(
                    prompts=all_prompts,
                    responses=scenario_1_model_2_responses + scenario_2_model_2_responses,
                    cost=None,
                ),
            description='Mock Model 2',
        ),
    ]
    model_creations[0].create()(prompt=scenario_1_prompts[0]) == scenario_1_model_1_responses[0]
    model_creations[0].create()(prompt=scenario_1_prompts[1]) == scenario_1_model_1_responses[1]
    model_creations[0].create()(prompt=scenario_2_prompts[0]) == scenario_2_model_1_responses[0]
    model_creations[0].create()(prompt=scenario_2_prompts[1]) == scenario_2_model_1_responses[1]
    model_creations[1].create()(prompt=scenario_1_prompts[0]) == scenario_1_model_2_responses[0]
    model_creations[1].create()(prompt=scenario_1_prompts[1]) == scenario_1_model_2_responses[1]
    model_creations[1].create()(prompt=scenario_2_prompts[0]) == scenario_2_model_2_responses[0]
    model_creations[1].create()(prompt=scenario_2_prompts[1]) == scenario_2_model_2_responses[1]

    comparison = CompareModels(
        prompts=[scenario_1_prompts, scenario_2_prompts],
        model_definitions=model_creations,
    )
    comparison()
    assert 'Mock Model 1' in str(comparison)
    assert 'Mock Model 2' in str(comparison)
    assert comparison.prompts == [scenario_1_prompts, scenario_2_prompts]
    assert comparison.num_scenarios == 2
    assert comparison.num_models == 2
    assert comparison.model_descriptions == ['Mock Model 1', 'Mock Model 2']
    assert comparison.duration_seconds('Mock Model 1') is not None
    assert comparison.duration_seconds('Mock Model 2') is not None
    assert comparison.num_response_chars('Mock Model 1') == len("".join(scenario_1_model_1_responses + scenario_2_model_1_responses))  # noqa
    assert comparison.num_response_chars('Mock Model 2') == len("".join(scenario_1_model_2_responses + scenario_2_model_2_responses))  # noqa
    assert comparison.response_chars_per_second('Mock Model 1') > 0
    assert comparison.response_chars_per_second('Mock Model 2') > 0
    assert comparison.num_code_blocks('Mock Model 1') == 6
    assert comparison.num_code_blocks('Mock Model 2') == 5
    assert comparison.num_successful_code_blocks('Mock Model 1') == 5
    assert comparison.num_successful_code_blocks('Mock Model 2') == 5
    assert comparison.percent_successful_code_blocks('Mock Model 1') == 5 / 6
    assert comparison.percent_successful_code_blocks('Mock Model 2') == 1
    assert comparison.cost('Mock Model 1') == 0.5 * 2
    assert comparison.cost('Mock Model 2') == 0
    assert comparison.to_html()
    comparison.to_html('tests/test_data/compare/compare_models__2_models.html')
    assert os.path.exists('tests/test_data/compare/compare_models__2_models.html')

def test_compare_models__3_models(conversation_sum, conversation_mask_email):  # noqa
    scenario_1_prompts = conversation_sum['prompts']
    scenario_2_prompts = conversation_mask_email['prompts']
    all_prompts = scenario_1_prompts + scenario_2_prompts

    scenario_1_model_1_responses = conversation_sum['model_1']['responses']
    scenario_1_model_2_responses = conversation_sum['model_2']['responses']
    scenario_2_model_1_responses = conversation_mask_email['model_1']['responses']
    scenario_2_model_2_responses = conversation_mask_email['model_2']['responses']

    model_creations = [
        ModelDefinition(
            create=lambda: \
                MockChatModel(
                    prompts=all_prompts,
                    responses=scenario_1_model_1_responses + scenario_2_model_1_responses,
                    cost=0.5,
                ),
            description='Mock Model 1',
        ),
        ModelDefinition(
            create=lambda: \
                MockChatModel(
                    prompts=all_prompts,
                    responses=scenario_1_model_2_responses + scenario_2_model_2_responses,
                    cost=None,
                ),
            description='Mock Model 2',
        ),
        # same as model 1
        ModelDefinition(
            create=lambda: \
                MockChatModel(
                    prompts=all_prompts,
                    responses=scenario_1_model_1_responses + scenario_2_model_1_responses,
                    cost=0.1,
                ),
            description='Mock Model 1.5',
        ),
    ]
    model_creations[0].create()(prompt=scenario_1_prompts[0]) == scenario_1_model_1_responses[0]
    model_creations[0].create()(prompt=scenario_1_prompts[1]) == scenario_1_model_1_responses[1]
    model_creations[0].create()(prompt=scenario_2_prompts[0]) == scenario_2_model_1_responses[0]
    model_creations[0].create()(prompt=scenario_2_prompts[1]) == scenario_2_model_1_responses[1]
    model_creations[1].create()(prompt=scenario_1_prompts[0]) == scenario_1_model_2_responses[0]
    model_creations[1].create()(prompt=scenario_1_prompts[1]) == scenario_1_model_2_responses[1]
    model_creations[1].create()(prompt=scenario_2_prompts[0]) == scenario_2_model_2_responses[0]
    model_creations[1].create()(prompt=scenario_2_prompts[1]) == scenario_2_model_2_responses[1]
    model_creations[2].create()(prompt=scenario_1_prompts[0]) == scenario_1_model_1_responses[0]
    model_creations[2].create()(prompt=scenario_1_prompts[1]) == scenario_1_model_1_responses[1]
    model_creations[2].create()(prompt=scenario_2_prompts[0]) == scenario_2_model_1_responses[0]
    model_creations[2].create()(prompt=scenario_2_prompts[1]) == scenario_2_model_1_responses[1]

    comparison = CompareModels(
        prompts=[scenario_1_prompts, scenario_2_prompts],
        model_definitions=model_creations,
    )
    comparison()
    assert 'Mock Model 1' in str(comparison)
    assert 'Mock Model 2' in str(comparison)
    assert 'Mock Model 1.5' in str(comparison)
    assert comparison.prompts == [scenario_1_prompts, scenario_2_prompts]
    assert comparison.num_scenarios == 2
    assert comparison.num_models == 3
    assert comparison.model_descriptions == ['Mock Model 1', 'Mock Model 2', 'Mock Model 1.5']
    assert comparison.duration_seconds('Mock Model 1') is not None
    assert comparison.duration_seconds('Mock Model 2') is not None
    assert comparison.duration_seconds('Mock Model 1.5') is not None
    assert comparison.num_response_chars('Mock Model 1') == len("".join(scenario_1_model_1_responses + scenario_2_model_1_responses))  # noqa
    assert comparison.num_response_chars('Mock Model 2') == len("".join(scenario_1_model_2_responses + scenario_2_model_2_responses))  # noqa
    assert comparison.num_response_chars('Mock Model 1.5') == len("".join(scenario_1_model_1_responses + scenario_2_model_1_responses))  # noqa
    assert comparison.response_chars_per_second('Mock Model 1') > 0
    assert comparison.response_chars_per_second('Mock Model 2') > 0
    assert comparison.response_chars_per_second('Mock Model 1.5') > 0
    assert comparison.num_code_blocks('Mock Model 1') == 6
    assert comparison.num_code_blocks('Mock Model 2') == 5
    assert comparison.num_code_blocks('Mock Model 1.5') == 6
    assert comparison.num_successful_code_blocks('Mock Model 1') == 5
    assert comparison.num_successful_code_blocks('Mock Model 2') == 5
    assert comparison.num_successful_code_blocks('Mock Model 1.5') == 5
    assert comparison.percent_successful_code_blocks('Mock Model 1') == 5 / 6
    assert comparison.percent_successful_code_blocks('Mock Model 2') == 1
    assert comparison.percent_successful_code_blocks('Mock Model 1.5') == 5 / 6
    assert comparison.cost('Mock Model 1') == 0.5 * 2
    assert comparison.cost('Mock Model 2') == 0
    assert comparison.cost('Mock Model 1.5') == 0.1 * 2
    assert comparison.to_html()
    comparison.to_html('tests/test_data/compare/compare_models__3_models.html')
    assert os.path.exists('tests/test_data/compare/compare_models__3_models.html')

def test_compare_unique_descriptions(conversation_sum):  # noqa
    prompts = conversation_sum['prompts']
    model_definitions = [
        ModelDefinition(
            create=lambda: \
                MockChatModel(
                    prompts=prompts,
                    responses=["a", "b"],
                    cost=0.5,
                ),
            description='Mock Model 1',
        ),
        ModelDefinition(
            create=lambda: \
                MockChatModel(
                    prompts=prompts,
                    responses=["c", "d"],
                    cost=None,
                ),
            description='Mock Model 1',
        ),
    ]
    with pytest.raises(ValueError, match="Model descriptions must be unique."):
        _ = CompareModels(prompts=prompts, model_definitions=model_definitions)

def test_compare_models__code_setup(conversation_sum, conversation_mask_email):  # noqa
    # the first time CompareModels is run, the code blocks will fail because there is no `re`
    # module; the second time, the code blocks will pass because there is a `re` module (via
    # code_setup)
    scenario_1_prompts = conversation_sum['prompts']
    scenario_2_prompts = conversation_mask_email['prompts']
    all_prompts = scenario_1_prompts + scenario_2_prompts

    scenario_1_model_1_responses = conversation_sum['model_1']['responses']
    scenario_1_model_2_responses = conversation_sum['model_2']['responses']
    scenario_2_model_1_responses = conversation_mask_email['model_1']['responses']
    # the code blocks for this model will fail because there is no `re` module and no setup code
    scenario_2_model_regex_responses = conversation_mask_email['model_regex']['responses']

    model_creations = [
        ModelDefinition(
            create=lambda: \
                MockChatModel(
                    prompts=all_prompts,
                    responses=scenario_1_model_1_responses + scenario_2_model_1_responses,
                    cost=0.5,
                ),
            description='Mock Model 1',
        ),
        ModelDefinition(
            create=lambda: \
                MockChatModel(
                    prompts=all_prompts,
                    responses=scenario_1_model_2_responses + scenario_2_model_regex_responses,
                    cost=None,
                ),
            description='Mock Model 2',
        ),
        # same as model 1
        ModelDefinition(
            create=lambda: \
                MockChatModel(
                    prompts=all_prompts,
                    responses=scenario_1_model_1_responses + scenario_2_model_1_responses,
                    cost=0.1,
                ),
            description='Mock Model 1.5',
        ),
    ]
    model_creations[0].create()(prompt=scenario_1_prompts[0]) == scenario_1_model_1_responses[0]
    model_creations[0].create()(prompt=scenario_1_prompts[1]) == scenario_1_model_1_responses[1]
    model_creations[0].create()(prompt=scenario_2_prompts[0]) == scenario_2_model_1_responses[0]
    model_creations[0].create()(prompt=scenario_2_prompts[1]) == scenario_2_model_1_responses[1]
    model_creations[1].create()(prompt=scenario_1_prompts[0]) == scenario_1_model_2_responses[0]
    model_creations[1].create()(prompt=scenario_1_prompts[1]) == scenario_1_model_2_responses[1]
    model_creations[1].create()(prompt=scenario_2_prompts[0]) == scenario_2_model_regex_responses[0]  # noqa
    model_creations[1].create()(prompt=scenario_2_prompts[1]) == scenario_2_model_regex_responses[1]  # noqa
    model_creations[2].create()(prompt=scenario_1_prompts[0]) == scenario_1_model_1_responses[0]
    model_creations[2].create()(prompt=scenario_1_prompts[1]) == scenario_1_model_1_responses[1]
    model_creations[2].create()(prompt=scenario_2_prompts[0]) == scenario_2_model_1_responses[0]
    model_creations[2].create()(prompt=scenario_2_prompts[1]) == scenario_2_model_1_responses[1]

    comparison = CompareModels(
        prompts=[scenario_1_prompts, scenario_2_prompts],
        model_definitions=model_creations,
    )
    comparison()
    assert 'Mock Model 1' in str(comparison)
    assert 'Mock Model 2' in str(comparison)
    assert 'Mock Model 1.5' in str(comparison)
    assert comparison.prompts == [scenario_1_prompts, scenario_2_prompts]
    assert comparison.num_scenarios == 2
    assert comparison.num_models == 3
    assert comparison.model_descriptions == ['Mock Model 1', 'Mock Model 2', 'Mock Model 1.5']
    assert comparison.duration_seconds('Mock Model 1') is not None
    assert comparison.duration_seconds('Mock Model 2') is not None
    assert comparison.duration_seconds('Mock Model 1.5') is not None
    assert comparison.num_response_chars('Mock Model 1') == len("".join(scenario_1_model_1_responses + scenario_2_model_1_responses))  # noqa
    assert comparison.num_response_chars('Mock Model 2') == len("".join(scenario_1_model_2_responses + scenario_2_model_regex_responses))  # noqa
    assert comparison.num_response_chars('Mock Model 1.5') == len("".join(scenario_1_model_1_responses + scenario_2_model_1_responses))  # noqa
    assert comparison.response_chars_per_second('Mock Model 1') > 0
    assert comparison.response_chars_per_second('Mock Model 2') > 0
    assert comparison.response_chars_per_second('Mock Model 1.5') > 0
    assert comparison.num_code_blocks('Mock Model 1') == 6
    assert comparison.num_code_blocks('Mock Model 2') == 5
    assert comparison.num_code_blocks('Mock Model 1.5') == 6
    assert comparison.num_successful_code_blocks('Mock Model 1') == 5
    assert comparison.num_successful_code_blocks('Mock Model 2') == 3  # 2 code blocks fail
    assert comparison.num_successful_code_blocks('Mock Model 1.5') == 5
    assert comparison.percent_successful_code_blocks('Mock Model 1') == 5 / 6
    assert comparison.percent_successful_code_blocks('Mock Model 2') == 0.6  # 2 code blocks fail
    assert comparison.percent_successful_code_blocks('Mock Model 1.5') == 5 / 6
    assert comparison.cost('Mock Model 1') == 0.5 * 2
    assert comparison.cost('Mock Model 2') == 0
    assert comparison.cost('Mock Model 1.5') == 0.1 * 2

    ####
    # Now with code setup the code blocks should pass
    ####
    comparison = CompareModels(
        prompts=[scenario_1_prompts, scenario_2_prompts],
        model_definitions=model_creations,
        code_setup=[None, "import re"],
    )
    comparison()
    assert 'Mock Model 1' in str(comparison)
    assert 'Mock Model 2' in str(comparison)
    assert 'Mock Model 1.5' in str(comparison)
    assert comparison.prompts == [scenario_1_prompts, scenario_2_prompts]
    assert comparison.num_scenarios == 2
    assert comparison.num_models == 3
    assert comparison.model_descriptions == ['Mock Model 1', 'Mock Model 2', 'Mock Model 1.5']
    assert comparison.duration_seconds('Mock Model 1') is not None
    assert comparison.duration_seconds('Mock Model 2') is not None
    assert comparison.duration_seconds('Mock Model 1.5') is not None
    assert comparison.num_response_chars('Mock Model 1') == len("".join(scenario_1_model_1_responses + scenario_2_model_1_responses))  # noqa
    assert comparison.num_response_chars('Mock Model 2') == len("".join(scenario_1_model_2_responses + scenario_2_model_regex_responses))  # noqa
    assert comparison.num_response_chars('Mock Model 1.5') == len("".join(scenario_1_model_1_responses + scenario_2_model_1_responses))  # noqa
    assert comparison.response_chars_per_second('Mock Model 1') > 0
    assert comparison.response_chars_per_second('Mock Model 2') > 0
    assert comparison.response_chars_per_second('Mock Model 1.5') > 0
    assert comparison.num_code_blocks('Mock Model 1') == 6
    assert comparison.num_code_blocks('Mock Model 2') == 5
    assert comparison.num_code_blocks('Mock Model 1.5') == 6
    assert comparison.num_successful_code_blocks('Mock Model 1') == 5
    assert comparison.num_successful_code_blocks('Mock Model 2') == 5  # all code blocks fail
    assert comparison.num_successful_code_blocks('Mock Model 1.5') == 5
    assert comparison.percent_successful_code_blocks('Mock Model 1') == 5 / 6
    assert comparison.percent_successful_code_blocks('Mock Model 2') == 1  # all code blocks fail
    assert comparison.percent_successful_code_blocks('Mock Model 1.5') == 5 / 6
    assert comparison.cost('Mock Model 1') == 0.5 * 2
    assert comparison.cost('Mock Model 2') == 0
    assert comparison.cost('Mock Model 1.5') == 0.1 * 2
