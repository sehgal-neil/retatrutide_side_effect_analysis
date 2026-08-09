# Analysis Code: Self-Reported Side Effects Among Reddit Users Taking Unapproved Retatrutide

This repository contains release code for:

> Self-Reported Side Effects Among Reddit Users Taking Unapproved Retatrutide
>
> Neil K. R. Sehgal, ME; Jena Shaw Tronieri, PhD; Benjamin Rader, PhD; Lyle Ungar, PhD; Sharath Chandra Guntuku, PhD.

## Overview

The pipeline processes public Reddit posts mentioning retatrutide to identify and characterize self-reported symptoms. Posts are collected from retatrutide-specific and broader peptide, glucagon-like peptide 1, exercise, and weight-management communities; classified for author self-use; processed for author-attributed symptoms; mapped to MedDRA Preferred Terms; and summarized in aggregate tables and figures.

This release mirrors the structure of the prior GLP-1 side-effects analysis repository while using sanitized paths and environment variables rather than local credentials.

## Pipeline

| Step | Script | Description |
| --- | --- | --- |
| 1 | `01_data_collection.py` | Collect and filter Reddit posts/comments from a local Pushshift/Arctic Shift-style MySQL archive |
| 2 | `02_self_use_classification.py` | Classify posts for author self-disclosed retatrutide use with the OpenAI Batch API |
| 3 | `03_side_effect_extraction.py` | Extract author-attributed symptoms and map them to MedDRA concepts using a saved OpenAI prompt or equivalent RAG workflow |
| 4 | `04_generate_tables.py` | Generate MedDRA PT, HLT, HLGT, and co-occurrence frequency tables |
| 5 | `05_generate_appetite_combined_tables.py` | Regenerate tables after combining appetite-related PTs |
| 6 | `06_make_comparison_figures.R` | Generate descriptive comparison figures as PNG, PDF, and EPS |

## Requirements

Python:

```bash
pip install -r requirements.txt
```

R:

The figure script uses base R only.

External requirements:

- Access to a local Reddit archive with monthly `com_YYYY_MM` and `sub_YYYY_MM` tables, or adaptation of `01_data_collection.py` to another public Reddit data source.
- An OpenAI API key in `OPENAI_API_KEY`.
- MedDRA MedAscii files, available under license from MedDRA MSSO. Set `MEDDRA_MEDASCII_DIR` or pass `--meddra`.
- For side-effect extraction, a saved OpenAI prompt configured with the full prompt in `prompts/side_effect_system_prompt.txt` and access to MedDRA retrieval resources, or an equivalent local implementation.

## Data

Raw Reddit text is not included because of platform terms and privacy considerations. Licensed MedDRA files are not included. This release includes aggregate output tables and figure data that do not contain raw Reddit text.

Expected generated/intermediate files:

| File | Description | Included? |
| --- | --- | --- |
| `data/filtered_posts.csv` | Retatrutide-related Reddit posts/comments after filtering | No, raw text |
| `data/self_use_posts.csv` | Posts/comments classified as current retatrutide self-use | No, raw text |
| `data/posts_with_meddra.csv` | Self-use rows with extracted MedDRA concepts | No, raw text |
| `outputs/*.csv` | Aggregate tables and figure data | Yes |
| `figures/*.eps` | Editable vector figures for journal submission | Yes when generated |

## Reproducing Tables

After producing `data/posts_with_meddra.csv` and obtaining MedDRA MedAscii files:

```bash
python 04_generate_tables.py --posts data/posts_with_meddra.csv --meddra "$MEDDRA_MEDASCII_DIR" --output-dir outputs
python 05_generate_appetite_combined_tables.py --posts data/posts_with_meddra.csv --meddra "$MEDDRA_MEDASCII_DIR" --output-dir outputs
Rscript 06_make_comparison_figures.R
```

## Subreddits

Tier 1 retatrutide-specific communities:

`RetatrutideGBP`, `Retatrutide`, `RetatrutideWomen`, `RetatrutideTrial`, `retatrutide4obesity`, `retatrutidevendors`

Tier 2 broader communities:

`Biohackers`, `Zepbound`, `GymMotivation`, `WeightLossAdvice`, `BariatricSurgery`, `workout`, `fit`, `Mounjaro`, `Peptidesource`, `BodyHackGuide`, `Peptides`, `PeptideGuidesPH`, `compoundedtirzepatide`, `peptidess`, `PeptidePathways`, `USPeptides`, `PeptideGuide`, `PeptideDiscussion`, `PeptideForum`, `PeptideProgress`, `tirzepatidecompound`

These communities were selected purposively to enrich for retatrutide discussion and are not a probability sample of Reddit communities or retatrutide users.

## Notes

- Scripts use relative paths by default and do not include API keys or local credentials.
- `01_data_collection.py` uses MySQL option files or environment variables for database access.
- `02_self_use_classification.py` and `03_side_effect_extraction.py` use the OpenAI Batch API and require `OPENAI_API_KEY`.
- The descriptive semaglutide/tirzepatide comparison values in `06_make_comparison_figures.R` are reproduced from the prior published Reddit analysis for context only. No statistical comparisons are performed.
