"""
Step 3: Side-Effect Extraction and MedDRA Concept Assignment
============================================================

Runs side-effect extraction over `data/self_use_posts.csv` using the OpenAI
Responses Batch API. The original analysis used a saved prompt configured with
the full prompt in `prompts/side_effect_system_prompt.txt` and MedDRA retrieval
resources. This script expects researchers to provide that saved prompt ID.

Authentication:
    export OPENAI_API_KEY=...
    export OPENAI_RETA_SIDE_EFFECT_PROMPT_ID=pmpt_...

Submit:
    python 03_side_effect_extraction.py --submit-only

Download after batches complete:
    python 03_side_effect_extraction.py --download
"""

import argparse
import glob
import json
import math
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_CSV = PROJECT_DIR / "data" / "self_use_posts.csv"
BATCH_DIR = PROJECT_DIR / "batch_jobs" / "side_effects"
OUTPUT_CSV = PROJECT_DIR / "data" / "posts_with_meddra.csv"

DEFAULT_MODEL = "gpt-5.4-nano"
TASKS_PER_FILE = 5000
MAX_OUTPUT_TOKENS = 8753

SIDE_EFFECT_SCHEMA = {
    "type": "object",
    "properties": {
        "side_effect_list": {
            "type": "array",
            "description": "A list of side effects and corresponding MedDRA concepts.",
            "items": {
                "type": "object",
                "properties": {
                    "side_effect": {"type": "string"},
                    "medDRA_concept": {"type": "string"},
                },
                "required": ["side_effect", "medDRA_concept"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["side_effect_list"],
    "additionalProperties": False,
}


def load_and_filter_data(csv_path):
    df = pd.read_csv(csv_path)
    current = df.get("currently_taking_retatrutide", pd.Series("", index=df.index)).fillna("").str.lower().eq("yes")
    previous = df.get("previously_took_retatrutide", pd.Series("", index=df.index)).fillna("").str.lower().eq("yes")
    return df[current | previous].copy().reset_index(drop=True)


def create_batch_files(df, model, saved_prompt_id, batch_dir):
    batch_dir.mkdir(parents=True, exist_ok=True)
    file_paths = []
    n_files = math.ceil(len(df) / TASKS_PER_FILE)

    for file_idx in range(n_files):
        start = file_idx * TASKS_PER_FILE
        end = min(start + TASKS_PER_FILE, len(df))
        chunk = df.iloc[start:end]
        file_path = batch_dir / f"batch_input_{file_idx + 1}.jsonl"
        with file_path.open("w") as f:
            for df_idx, row in chunk.iterrows():
                task = {
                    "custom_id": f"task-{df_idx}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": model,
                        "prompt": {"id": saved_prompt_id},
                        "input": [
                            {
                                "role": "user",
                                "content": [{"type": "input_text", "text": row["message"]}],
                            }
                        ],
                        "text": {
                            "format": {
                                "type": "json_schema",
                                "name": "side_effects",
                                "strict": True,
                                "schema": SIDE_EFFECT_SCHEMA,
                            }
                        },
                        "max_output_tokens": MAX_OUTPUT_TOKENS,
                    },
                }
                f.write(json.dumps(task) + "\n")
        print(f"Wrote {file_path} ({end - start:,} tasks)")
        file_paths.append(file_path)
    return file_paths


def submit_batches(client, file_paths, batch_dir):
    batch_ids = []
    for file_path in file_paths:
        with file_path.open("rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/responses",
            completion_window="24h",
        )
        batch_ids.append(batch.id)
        print(f"Submitted {batch.id}")

    ids_file = batch_dir / "batch_ids.txt"
    ids_file.write_text("\n".join(batch_ids))
    print(f"Saved batch IDs to {ids_file}")
    return batch_ids


def read_batch_ids(batch_dir, batch_ids_arg=None):
    if batch_ids_arg:
        return [value.strip() for value in batch_ids_arg.split(",") if value.strip()]
    ids_file = batch_dir / "batch_ids.txt"
    if not ids_file.exists():
        raise SystemExit(f"{ids_file} not found. Run --submit-only first or pass --batch-ids.")
    ids = [line.strip() for line in ids_file.read_text().splitlines() if line.strip()]
    for retry_file in sorted(glob.glob(str(batch_dir / "retry_*" / "batch_ids.txt"))):
        ids.extend(line.strip() for line in Path(retry_file).read_text().splitlines() if line.strip())
    return ids


def download_results(client, batch_ids, batch_dir):
    output_paths = []
    for batch_id in batch_ids:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"{batch.id}: status={batch.status} "
            f"total={counts.total} completed={counts.completed} failed={counts.failed}"
        )
        if batch.status != "completed":
            continue
        output_path = batch_dir / f"batch_output_{batch.id}.jsonl"
        content = client.files.content(batch.output_file_id)
        output_path.write_bytes(content.read())
        output_paths.append(output_path)
    output_paths.extend(Path(p) for p in sorted(glob.glob(str(batch_dir / "retry_*" / "batch_output_*.jsonl"))))
    return output_paths


def parse_results(output_paths):
    rows = []
    for path in output_paths:
        with Path(path).open() as f:
            for line in f:
                result = json.loads(line)
                df_idx = int(result["custom_id"].replace("task-", ""))
                try:
                    body = result["response"]["body"]
                    parsed = None
                    for item in body.get("output", []):
                        if item.get("type") != "message":
                            continue
                        for content in item.get("content", []):
                            if content.get("type") == "output_text":
                                parsed = json.loads(content["text"])
                                break
                        if parsed is not None:
                            break
                    if parsed is None:
                        raise ValueError("no output_text found")
                    side_effects = parsed.get("side_effect_list", [])
                    concepts = [
                        entry["medDRA_concept"]
                        for entry in side_effects
                        if entry.get("medDRA_concept")
                    ]
                    rows.append(
                        {
                            "_task_index": df_idx,
                            "meddra_concepts": str(concepts) if concepts else "[]",
                            "side_effect_details": json.dumps(side_effects) if side_effects else "[]",
                            "n_side_effects": len(side_effects),
                        }
                    )
                except (KeyError, ValueError, json.JSONDecodeError) as exc:
                    rows.append(
                        {
                            "_task_index": df_idx,
                            "meddra_concepts": "[]",
                            "side_effect_details": "[]",
                            "n_side_effects": 0,
                            "_error": str(exc),
                        }
                    )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset="_task_index", keep="last").sort_values("_task_index").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Extract side effects and MedDRA concepts with OpenAI Batch API.")
    parser.add_argument("--csv", default=str(INPUT_CSV))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-dir", default=str(BATCH_DIR))
    parser.add_argument("--output", default=str(OUTPUT_CSV))
    parser.add_argument("--saved-prompt-id", default=os.environ.get("OPENAI_RETA_SIDE_EFFECT_PROMPT_ID"))
    parser.add_argument("--batch-ids", default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--submit-only", action="store_true")
    mode.add_argument("--download", action="store_true")
    args = parser.parse_args()

    if not args.saved_prompt_id:
        raise SystemExit("Provide --saved-prompt-id or set OPENAI_RETA_SIDE_EFFECT_PROMPT_ID.")

    client = OpenAI()
    batch_dir = Path(args.batch_dir)

    if args.submit_only:
        df = load_and_filter_data(args.csv)
        file_paths = create_batch_files(df, args.model, args.saved_prompt_id, batch_dir)
        submit_batches(client, file_paths, batch_dir)
        return

    batch_ids = read_batch_ids(batch_dir, args.batch_ids)
    output_paths = download_results(client, batch_ids, batch_dir)
    if not output_paths:
        print("No completed batch outputs found.")
        return

    source = load_and_filter_data(args.csv)
    results = parse_results(output_paths)
    merged = source.merge(results, left_index=True, right_on="_task_index", how="left")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    print(f"Saved {len(merged):,} rows to {output}")


if __name__ == "__main__":
    main()
