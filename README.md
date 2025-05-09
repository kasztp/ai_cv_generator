# AI CV Generator

The AI CV Generator is a Python tool designed to process and refine resumes using Google's Gemini Large Language Model. It takes raw CV text or pdf resume as input and applies a user-defined sequence of AI-powered transformations to produce a polished editable Markdown document.

## Features

* **AI-Powered CV Processing**: Leverages Google's Gemini API for intelligent text generation and transformation.
* **Flexible Input**: Accepts raw CV text or PDF format resumes from a file.
* **Markdown Output**: Generates clean, readable, and professional CVs in Markdown format.
* **Customizable Processing Pipeline**:
  * Applies a sequence of processing steps, each driven by a specific prompt.
  * The sequence of steps is defined in `src/cv_generator/generator.py` (the `steps` list).
  * Prompts are defined and can be customized in `src/cv_generator/prompts.py`.
* **Configurable AI**: Allows configuration of the Gemini model, safety settings, and generation parameters.
* **Modern Python Tooling**:
  * Uses `uv` for fast dependency and virtual environment management.
  * Employs `ruff` for linting and formatting.
  * Includes `pytest` for automated testing.
* **Continuous Integration**: GitHub Actions workflow for automated linting and testing on pushes and pull requests.

## Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/kasztp/ai_cv_generator.git
    cd ai_cv_generator
    ```

2. **Install `uv` (if you haven't already):**
    Follow instructions at [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv) or use pipx:

    ```bash
    pipx install uv
    ```

3. **Create and activate a virtual environment:**

    ```bash
    uv venv
    ```

    Activate it:
    * macOS/Linux: `source .venv/bin/activate`
    * Windows: `.venv\Scripts\activate`

4. **Install dependencies:**

    ```bash
    uv pip install -e '.[dev]'
    ```

    (`-e` installs in editable mode, `[dev]` includes testing/linting tools)

5. **Set up Environment Variables:**
    * Copy the example environment file:

        ```bash
        cp .env.example .env
        ```

    * Edit the `.env` file and add your Google Gemini API key:

        ```env
        GEMINI_API_KEY="YOUR_ACTUAL_API_KEY_HERE"
        ```

        **Important:** Never commit your actual `.env` file to Git. It's already included in `.gitignore`.

## Usage

1. **Prepare Input:**
    * Have your raw CV text or PDF resume ready in a file (e.g., `input_cvs/my_cv.txt` or `input_cvs/my_resume.pdf`).

2. **(Optional) Customize Processing Steps & Prompts:**
    * The core logic of CV transformation lies in the sequence of prompts applied.
    * **Define Prompts**: Review and modify the prompts in `src/cv_generator/prompts.py` to suit your desired transformations (e.g., formatting, summarizing sections, extracting specific information, rephrasing).
    * **Define Processing Sequence**: Edit the `steps` list within the `main()` function of `src/cv_generator/generator.py`. Each element in this list typically defines a prompt to use and an optional description for logging. This allows you to chain multiple AI transformations. For example:

    ```python
        # In src/cv_generator/generator.py
        processing_steps = [
            {"prompt_name": "FORMAT_MARKDOWN_PROMPT", "description": "Formatting CV to Markdown"},
            {"prompt_name": "SUMMARIZE_EXPERIENCE_PROMPT", "description": "Summarizing key experiences"},
            # Add more steps as needed
        ]
    ```

3. **Run the Generator:**
    Provide the path to your input CV file as the main argument. You can optionally specify an output file path using `-o` or `--output`.

    * **Using python module:**

        ```bash
        # Process input_cvs/my_cv.txt and save to the default output/generated_cv.md
        python -m cv_generator.generator input_cvs/my_cv.txt

        # Process another_cv.txt and save to a specific file
        python -m cv_generator.generator input_cvs/another_cv.txt -o output/another_cv_processed.md
        ```

    * **Using the installed script (if virtual environment is active):**

        ```bash
        # Example:
        # generate-cv input_cvs/my_cv.txt --output output/custom_output.md
        ```

4. **Output:**
    The generated Markdown CV will be saved to the specified output path (defaulting to `output/generated_cv.md`).

## Development

The project uses `uv` to manage development tasks defined as scripts in `pyproject.toml` (or run directly).

* **Linting (with Ruff):**
    Check for code style and errors.

    ```bash
    uv run ruff check src tests
    ```

* **Formatting (with Ruff):**
    Automatically format the code.

    ```bash
    uv run ruff format src tests
    ```

* **Testing (with Pytest):**
    Run the test suite.

    ```bash
    uv run pytest
    ```

## GitHub Actions

The Continuous Integration (CI) workflow is defined in `.github/workflows/ci.yml`. It automatically runs linters (`ruff check`) and tests (`pytest`) on every push and pull request to the `main` branch to ensure code quality and correctness.
