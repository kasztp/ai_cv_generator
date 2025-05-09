"""Unit tests for the CV generator module using pytest and unittest.mock."""

from unittest.mock import MagicMock

import google.generativeai as genai
import pytest

from cv_generator import generator


@pytest.fixture(autouse=True)
def mock_gemini_configure(mocker):
    """Prevent actual API configuration during tests."""
    mocker.patch("google.generativeai.configure", return_value=None)


@pytest.fixture
def mock_gemini_model(mocker):
    """Fixture to mock the Gemini model and its response."""
    mock_model = MagicMock(spec=genai.GenerativeModel)
    mock_response = MagicMock(spec=genai.types.GenerateContentResponse)

    # Simulate a successful response with text
    mock_response.text = "Mocked Gemini Response Text"
    # Simulate having content parts for non-blocked responses
    mock_part = MagicMock()
    mock_part.text = "Mocked Gemini Response Text"  # If response.parts[0].text is used
    mock_response.parts = [mock_part]
    mock_response.prompt_feedback = None  # Simulate no blocking by default

    # Set the return value of the mock model's method
    mock_model.generate_content.return_value = mock_response

    # Patch where 'GenerativeModel' is looked up *within the generator module*
    mocker.patch(
        "cv_generator.generator.genai.GenerativeModel", return_value=mock_model
    )
    mocker.patch("cv_generator.generator.model", new=mock_model)

    return mock_model


def test_send_prompt_to_gemini_success(mock_gemini_model):
    """Test successful interaction with the mocked Gemini API."""
    prompt = "Test prompt"
    expected_response = "Mocked Gemini Response Text"  # From mock_response.text
    response = generator.send_prompt_to_gemini(prompt)
    assert response == expected_response


def test_send_prompt_to_gemini_error(monkeypatch):
    """Test send_prompt_to_gemini returns None on exception."""

    def raise_exc(*args, **kwargs):
        raise Exception("API error")

    monkeypatch.setattr(generator.model, "generate_content", raise_exc)
    assert generator.send_prompt_to_gemini("prompt") is None


def test_process_cv_success(mock_gemini_model, mocker):
    """Test process_cv runs all steps and returns final text."""
    mocker.patch(
        "cv_generator.generator.send_prompt_to_gemini", return_value="Step1 Output"
    )
    steps = [
        {"prompt_template": "Prompt {cv_text}", "data": {}},
        {"prompt_template": "Prompt {cv_text}", "data": {}},
    ]
    result = generator.process_cv("Initial", steps)
    assert result == "Step1 Output"


def test_process_cv_empty_steps():
    """Test process_cv with no steps returns the initial text."""
    result = generator.process_cv("Initial CV", [])
    assert result == "Initial CV"


def test_process_cv_missing_prompt_template(mocker):
    """Test process_cv skips steps with missing prompt_template."""
    mocker.patch("cv_generator.generator.send_prompt_to_gemini", return_value="Output")
    steps = [
        {"data": {}},  # No prompt_template
        {"prompt_template": "Prompt {cv_text}", "data": {}},
    ]
    result = generator.process_cv("Initial", steps)
    assert result == "Output"


def test_process_cv_key_error(mocker):
    """Test process_cv returns None if step data is missing a required key."""
    mocker.patch("cv_generator.generator.send_prompt_to_gemini", return_value="Output")
    steps = [
        {"prompt_template": "Prompt {missing_key}", "data": {}},
    ]
    result = generator.process_cv("Initial", steps)
    assert result is None


def test_extract_text_from_pdf_success(mocker, tmp_path):
    """Test extract_text_from_pdf returns text from PDF."""
    fake_pdf = tmp_path / "test.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4\n...")
    mock_page = mocker.Mock()
    mock_page.extract_text.return_value = "Page text"
    mock_reader = mocker.Mock()
    mock_reader.pages = [mock_page, mock_page]
    mocker.patch("cv_generator.generator.PdfReader", return_value=mock_reader)
    text = generator.extract_text_from_pdf(fake_pdf)
    assert text == "Page text\nPage text\n"


def test_extract_text_from_pdf_error(mocker, tmp_path):
    """Test extract_text_from_pdf returns empty string on error."""
    fake_pdf = tmp_path / "bad.pdf"
    mocker.patch("cv_generator.generator.PdfReader", side_effect=Exception("fail"))
    text = generator.extract_text_from_pdf(fake_pdf)
    assert text == ""


def test_send_prompt_to_gemini_api_error(mock_gemini_model):
    """Test handling of an API error during generate_content call."""
    # Configure the mock to raise an exception when called
    mock_gemini_model.generate_content.side_effect = Exception("Simulated API Error")

    prompt = "Test prompt"
    response = generator.send_prompt_to_gemini(prompt)

    # Expect None because the exception should be caught
    assert response is None
    mock_gemini_model.generate_content.assert_called_once_with(
        prompt,
        generation_config=generator.GENERATION_CONFIG,
        safety_settings=generator.SAFETY_SETTINGS,
    )


def test_send_prompt_to_gemini_blocked(mock_gemini_model):
    """Test handling of a response blocked by safety settings."""
    # Configure the mock response to simulate blocking
    mock_response = MagicMock(spec=genai.types.GenerateContentResponse)
    mock_response.parts = []  # No content parts when blocked
    mock_response.text = ""  # Text might be empty
    # Simulate feedback indicating blocking
    mock_feedback = MagicMock()
    mock_feedback.block_reason = genai.types.BlockedReason.SAFETY
    mock_response.prompt_feedback = mock_feedback

    mock_gemini_model.generate_content.return_value = mock_response

    prompt = "Test prompt potentially blocked"
    response = generator.send_prompt_to_gemini(prompt)

    # Expect None because the response was blocked/empty
    assert response is None
    mock_gemini_model.generate_content.assert_called_once()


def test_process_cv_single_step(mock_gemini_model):
    """Test processing a CV through a single step using the mock."""
    initial_text = "Initial CV"
    step_prompt_template = "Format: {cv_text}"
    steps = [{"prompt_template": step_prompt_template, "data": {}}]

    # Configure the mock response for this specific test's call
    mock_response = MagicMock(spec=genai.types.GenerateContentResponse)
    mock_response.text = "Processed Step 1 Result"
    mock_response.parts = [MagicMock()]  # Ensure parts exist
    mock_response.prompt_feedback = None
    mock_gemini_model.generate_content.return_value = mock_response

    final_cv = generator.process_cv(initial_text, steps)

    assert final_cv == "Processed Step 1 Result"
    mock_gemini_model.generate_content.assert_called_once()
    call_args, _ = mock_gemini_model.generate_content.call_args
    assert call_args[0] == "Format: Initial CV"  # Check the prompt sent


def test_process_cv_multiple_steps(mock_gemini_model):
    """Test processing a CV through multiple steps with distinct mock responses."""
    initial_text = "Start"
    step1_prompt = "Step 1: {cv_text}"
    step2_prompt = "Step 2: {cv_text}"
    steps = [
        {"prompt_template": step1_prompt, "data": {}},
        {"prompt_template": step2_prompt, "data": {}},
    ]

    # Mock sequential responses using side_effect list
    mock_response1 = MagicMock(
        spec=genai.types.GenerateContentResponse,
        text="After Step 1",
        parts=[MagicMock()],
        prompt_feedback=None,
    )
    mock_response2 = MagicMock(
        spec=genai.types.GenerateContentResponse,
        text="Final Result",
        parts=[MagicMock()],
        prompt_feedback=None,
    )
    mock_gemini_model.generate_content.side_effect = [mock_response1, mock_response2]

    final_cv = generator.process_cv(initial_text, steps)

    assert final_cv == "Final Result"
    assert mock_gemini_model.generate_content.call_count == 2
    # Check prompts sent at each step
    call1_args, _ = mock_gemini_model.generate_content.call_args_list[0]
    call2_args, _ = mock_gemini_model.generate_content.call_args_list[1]
    assert call1_args[0] == "Step 1: Start"
    assert call2_args[0] == "Step 2: After Step 1"


def test_process_cv_step_failure(mock_gemini_model):
    """Test that processing stops if a Gemini call fails within a step."""
    initial_text = "Start"
    steps = [{"prompt_template": "Step: {cv_text}"}]
    # Simulate API failure during the generate_content call
    mock_gemini_model.generate_content.side_effect = Exception(
        "Simulated API Error during processing"
    )

    final_cv = generator.process_cv(initial_text, steps)
    assert final_cv is None  # Processing should abort and return None


def test_process_cv_missing_prompt_key(mock_gemini_model):
    """Test failure when prompt formatting key is missing (no API call)."""
    initial_text = "Start"
    steps = [{"prompt_template": "Step: {data_key}", "data": {}}]  # Missing 'data_key'
    final_cv = generator.process_cv(initial_text, steps)
    assert final_cv is None
    mock_gemini_model.generate_content.assert_not_called()
