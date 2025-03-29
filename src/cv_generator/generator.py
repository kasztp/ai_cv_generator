"""Module for generating and refining CVs using the Gemini API."""
import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
import google.generativeai as genai

from . import prompts

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    logging.error("GEMINI_API_KEY not found in environment variables.")
    exit(1)

try:
    genai.configure(api_key=API_KEY)
    # Choose model - gemini-1.5-flash is often faster/cheaper
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    logging.info("Gemini API configured successfully.")
except Exception as e:
    logging.error("Failed to configure Gemini API: %s", e)
    exit(1)

SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
]

GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 32,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# --- Core Functions ---


def send_prompt_to_gemini(prompt: str) -> str | None:
    """Sends a prompt to the configured Gemini model and returns the text response.

    Args:
    -----
        prompt (str): The prompt to send to the Gemini model.

    Returns:
    --------
        str | None: The generated text response from Gemini, or None if there was an error.
    """
    try:
        logging.debug(f"Sending prompt (length {len(prompt)} chars) to Gemini...")
        response = model.generate_content(
            prompt,
            generation_config=GENERATION_CONFIG,
            safety_settings=SAFETY_SETTINGS,
        )

        if response.parts:
            generated_text = response.text
            logging.debug(
                "Received response (length %s chars) from Gemini.",
                len(generated_text)
            )
            return generated_text

        feedback = response.prompt_feedback
        reason = feedback.block_reason if feedback else "Unknown"
        logging.warning(f"Gemini response was empty or blocked. Reason: {reason}")
        return None

    except Exception as e:
        logging.error(f"Error interacting with Gemini API: {e}")
        return None


def process_cv(initial_cv_text: str, processing_steps: list[dict]) -> str | None:
    """Processes the CV through a series of steps using Gemini.

    Args:
    -----
        initial_cv_text (str): The initial CV text to process.
        processing_steps (list[dict]): A list of dictionaries defining the processing steps.

    Returns:
    --------
        str | None: The final processed CV text, or None if processing failed.
    """
    current_cv_state = initial_cv_text
    logging.info(f"Starting CV processing with {len(processing_steps)} steps.")

    for i, step in enumerate(processing_steps, 1):
        prompt_template = step.get("prompt_template")
        step_data = step.get("data", {})

        if not prompt_template:
            logging.warning(f"Skipping step {i}: No prompt template provided.")
            continue

        step_data["cv_text"] = current_cv_state

        try:
            full_prompt = prompt_template.format(**step_data)
            logging.info(f"Processing step {i}/{len(processing_steps)}...")
            response_text = send_prompt_to_gemini(full_prompt)

            if response_text:
                current_cv_state = response_text
                logging.info(f"Step {i} completed successfully.")
            else:
                logging.error(f"Step {i} failed: No response from Gemini.")
                return None

        except KeyError as e:
            logging.error(
                "Step %d failed: Missing key '%s' for prompt formatting. "
                "Available keys: %s",
                i, e, list(step_data.keys())
            )
            return None

        except Exception as e:
            logging.error(f"Step {i} failed with unexpected error: {e}")
            return None

    logging.info("CV processing finished successfully.")
    return current_cv_state


# --- Argument Parsing and Main Execution ---


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate and refine CVs using the Gemini API."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the input CV text file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output/generated_cv.md"),
        help="Path to save the generated Markdown CV file (default: output/generated_cv.md).",
    )

    return parser.parse_args()


def main():
    """Main function to load CV, define steps, process, and save."""
    args = parse_arguments()
    logging.info("--- Starting Gemini CV Processor ---")

    # 1. Load Initial CV Text
    input_cv_path = args.input_file
    if not input_cv_path.exists():
        logging.error(f"Input CV file does not exist: {input_cv_path}")
        return
    if not input_cv_path.is_file():
        logging.error(f"Input CV path is not a file: {input_cv_path}")
        return
    try:
        initial_text = input_cv_path.read_text(encoding="utf-8")
        logging.info(f"Loaded initial CV text from {input_cv_path}")
    except FileNotFoundError:
        logging.error(f"Input CV file not found: {input_cv_path}")
        return
    except Exception as e:
        logging.error(f"Error reading input CV file {input_cv_path}: {e}")
        return

    # 2. Define Processing Steps
    certifications_to_add = """
    - Partner Training - Advantages of Data Intelligence & Interoperability with SAP (Databricks, Issued Feb 2025, Expires Feb 2027)
    - Gen AI & LLM on Databricks - Pre-sales (Databricks, Issued Oct 2024, Expires Oct 2026)
    # ... (rest of certifications) ...
    - Python 3 Tutorial Course (Sololearn, Issued Sep 2016)
    """
    badge_info_text = """
    - SAP Interoperability: https://raw.githubusercontent.com/kasztp/cv-assets/main/badges/SAP_Interoperability_badge.png
    - Gen AI & LLM Pre-sales: https://raw.githubusercontent.com/kasztp/cv-assets/main/badges/Gen_AI_LLM_on_Databricks_badge.png
    # ... (rest of badge info) ...
    """

    steps = [
        {
            "prompt_template": prompts.FORMAT_MARKDOWN_PROMPT,
            "data": {},
        },
        {
            "prompt_template": prompts.SUMMARIZE_HR_ROLES_PROMPT,
            "data": {},
        },
        {
            "prompt_template": prompts.ADD_CERTIFICATIONS_PROMPT,
            "data": {"new_certifications_text": certifications_to_add},
        },
        {
            "prompt_template": prompts.ADD_BADGES_PROMPT,
            "data": {"badge_info_text": badge_info_text},
        },
    ]

    # 3. Process the CV
    final_cv = process_cv(initial_text, steps)

    # 4. Save the Result
    if final_cv:
        output_cv_path = args.output
        try:
            # Ensure output directory exists
            output_cv_path.parent.mkdir(parents=True, exist_ok=True)
            output_cv_path.write_text(final_cv, encoding="utf-8")
            logging.info(f"Successfully generated and saved CV to {output_cv_path}")

        except Exception as e:
            logging.error(f"Error saving generated CV to {output_cv_path}: {e}")

    else:
        logging.error("CV generation failed. No output file created.")

    logging.info("--- Gemini CV Processor Finished ---")


if __name__ == "__main__":
    main()
