"""This module contains the prompts used for generating and formatting CVs.
It includes prompts for formatting, summarizing, and adding sections to the CV.
"""
# --- CV Generation and Formatting Prompts ---

# Initial formatting prompt
FORMAT_MARKDOWN_PROMPT = """
Please take the following raw CV text and format it into a clean, professional Markdown document.
Use standard Markdown syntax (headings, lists, bold text, etc.).
Ensure good readability and structure. Keep all original information.

Raw CV Text:
---
{cv_text}
---

Formatted Markdown CV:
"""  # noqa: E501

# Summarization prompt
SUMMARIZE_HR_ROLES_PROMPT = """
Given the following Markdown CV, locate the roles between IBM and Datapao.
These roles are related to HR, Talent Acquisition, and Recruitment.
Summarize these roles into a single concise section titled "Talent Acquisition Leadership & Technical Recruitment (Summary)".
This summary should capture the progression, core expertise, key responsibilities, and representative employers/engagements during that period (approx. July 2011 - Feb 2022).
Replace the individual job entries for this period with this summary section. Keep the rest of the CV structure intact.

Current Markdown CV:
---
{cv_text}
---

Updated Markdown CV with Summarized Section:
"""  # noqa: E501

# Certification addition prompt (Note - more robust templating might be needed)
ADD_CERTIFICATIONS_PROMPT = """
Please update the "Certifications" section of the following Markdown CV.
Add the certifications listed below, grouping them by issuing body (like Databricks, WorldQuant University, etc.) and including issue/expiry dates where provided.
Ensure the formatting is consistent with any existing certifications.

Current Markdown CV:
---
{cv_text}
---

Certifications to Add:
---
{new_certifications_text}
---

Updated Markdown CV with New Certifications:
"""  # noqa: E501

# Badge section prompt (Example)
ADD_BADGES_PROMPT = """
Update the "Summary" section of the following Markdown CV.
Add a new subsection titled "Key Certifications Showcase (Badges):" just below the main summary text.
This subsection should contain a Markdown table displaying certification badges.
Use the following image URLs and descriptions to create a table (e.g., 3 columns).
Ensure the image URLs point to the raw image content on GitHub.

Example Image Info (replace with actual list):
- SAP Interoperability: https://raw.githubusercontent.com/kasztp/cv-assets/main/badges/SAP_Interoperability_badge.png
- Gen AI & LLM Pre-sales: https://raw.githubusercontent.com/kasztp/cv-assets/main/badges/Gen_AI_LLM_on_Databricks_badge.png
- Gen AI Fundamentals: https://raw.githubusercontent.com/kasztp/cv-assets/main/badges/Generative_AI_Fundamentals.png
- Cloud DW Migration: https://raw.githubusercontent.com/kasztp/cv-assets/main/badges/Snowflake_Migration_Badge.png
# ... add more badge URLs and descriptions as needed

Current Markdown CV:
---
{cv_text}
---

Updated Markdown CV with Badge Section:
"""  # noqa: E501

# LinkedIn profile transformation prompt
CONCISE_MARKDOWN_CV_PROMPT = """
You are an expert CV editor. Given the raw extracted text from a LinkedIn profile PDF, reformat it into a concise, professional Markdown CV.
- Use clear Markdown headings (e.g., # Name, ## Experience, ## Education, ## Skills).
- Include all job experiences, listing each position and its details under the Experience section.
- Remove redundancy, filler, and irrelevant content.
- Include only the most relevant information for a technical job application.
- Make the document well-structured and easy to read.

Raw LinkedIn Profile Text:
---
{cv_text}
---

Concise Markdown CV:
"""  # noqa: E501
