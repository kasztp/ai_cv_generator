# AI CV Generator

Generate and refine CVs using the chosen LLM based on a series of prompts.

## Features

* Takes raw CV text as input.
* Applies a sequence of processing steps using Gemini.
* Uses prompts defined in `src/cv_generator/prompts.py`.
* Configurable Gemini model, safety settings, and generation parameters.
* Modern Python tooling: `uv`, `ruff`, `pytest`.
* GitHub Actions for Continuous Integration (Linting & Testing).

## Setup

1. **Clone the repository:**

    ```bash
    git clone <your-repo-url>
    cd gemini-cv-processor
    ```

2. **Install `uv`:**
    Follow instructions at [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv) or use pipx:

    ```bash
    pipx install uv
    ```

3. **Create a virtual environment:**

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

        **Important:** Never commit your actual `.env` file to Git. It's included in `.gitignore`.

## Usage

1. **Prepare Input:** Have your raw CV text ready in a file (e.g., `my_cv.txt`).

2. **Customize Prompts:** Edit `src/cv_generator/prompts.py` and the `steps` list in `src/cv_generator/generator.py` to define the processing workflow you need.

3. **Run the generator:**

    Provide the path to your input CV file as the main argument. You can optionally specify an output file path using `-o` or `--output`.

    ```bash
    # Example: Process my_cv.txt and save to the default output/generated_cv.md
    python -m cv_generator.generator path/to/your/my_cv.txt

    # Example: Process another_cv.txt and save to a specific file
    python -m cv_generator.generator path/to/another_cv.txt -o output/another_cv_processed.md

    # Using the installed script (if venv active)
    # generate-cv path/to/your/my_cv.txt --output custom_output.md
    ```

4. **Output:** The generated Markdown CV will be saved to the specified output path (defaulting to `output/generated_cv.md`).

## Development

* **Linting:**

    ```bash
    uv run ruff check src tests
    ```

* **Formatting:**

    ```bash
    uv run ruff format src tests
    ```

* **Testing:**

    ```bash
    uv run pytest
    ```

## GitHub Actions

The CI workflow in `.github/workflows/ci.yml` automatically runs linters and tests on every push and pull request to the `main` branch.
