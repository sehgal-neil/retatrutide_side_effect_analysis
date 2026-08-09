# Data Availability

Raw Reddit posts and comments are not redistributed in this release because of platform terms and privacy considerations. The public release is intended to share the analysis code, prompts, aggregate tables, figure data, and generated vector figures.

MedDRA terminology files are not redistributed. Researchers must obtain MedDRA MedAscii files under license from MedDRA MSSO and pass the directory to `04_generate_tables.py` and `05_generate_appetite_combined_tables.py`.

OpenAI prompts are included in `prompts/`. Batch job outputs are not included because they contain raw Reddit text or row-level derived annotations tied to raw text.
