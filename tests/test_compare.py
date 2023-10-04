"""Tests the compare module."""

from textwrap import dedent
from llm_workflow.compare import Scenario, ModelCreation, CompareModels


class MockChatModel:
    """Mock chat model for testing."""

    def __init__(self, prompts: list[str], responses: list[str], cost: float | None = None):
        """
        Initialize the mock chat model. The model will respond to each prompt with the
        corresponding response (i.e. the first prompt will be responded to with the first
        response, the second prompt with the second response, etc.).
        """
        assert len(prompts) == len(responses)
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
    assert scenario.duration_seconds is not None
    assert scenario.duration_seconds > 0
    assert scenario.num_response_chars == len("".join(model_1_responses))
    assert scenario.response_chars_per_second > 0
    assert scenario.num_code_blocks == 3
    assert scenario.code_block_results == [[None, None], [None]]
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
    assert scenario.duration_seconds is not None
    assert scenario.duration_seconds > 0
    assert scenario.num_response_chars == len("".join(model_2_responses))
    assert scenario.response_chars_per_second > 0
    assert scenario.num_code_blocks == 3
    assert scenario.code_block_results == [[None, None], [None]]
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
    assert scenario.code_block_results[0] == [None, None]
    assert isinstance(scenario.code_block_results[1][0], AssertionError)
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
    assert scenario.code_block_results == [[None], [None]]
    assert scenario.num_successful_code_blocks == 2
    assert scenario.percent_successful_code_blocks == 1
    assert scenario.cost is None

def test_compare_models(conversation_sum, conversation_mask_email):  # noqa
    scenario_1_prompts = conversation_sum['prompts']
    scenario_2_prompts = conversation_mask_email['prompts']
    all_prompts = scenario_1_prompts + scenario_2_prompts

    scenario_1_model_1_responses = conversation_sum['model_1']['responses']
    scenario_1_model_2_responses = conversation_sum['model_2']['responses']
    scenario_2_model_1_responses = conversation_mask_email['model_1']['responses']
    scenario_2_model_2_responses = conversation_mask_email['model_2']['responses']

    model_creations = [
        ModelCreation(
            create=lambda: \
                MockChatModel(
                    prompts=all_prompts,
                    responses=scenario_1_model_1_responses + scenario_2_model_1_responses,
                    cost=0.5,
                ),
            description='Mock Model 1',
        ),
        ModelCreation(
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
        model_creations=model_creations,
    )
    comparison()
    assert 'Mock Model 1' in str(comparison)
    assert 'Mock Model 2' in str(comparison)
    assert comparison.prompts == [scenario_1_prompts, scenario_2_prompts]
    assert comparison.num_scenarios == 2
