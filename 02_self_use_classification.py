"""
Step 2: Self-Use Classification
===============================

Classifies author self-disclosed retatrutide use from `data/filtered_posts.csv`
using the OpenAI Batch API. The script writes only rows classified as current
self-use to `data/self_use_posts.csv`.

Authentication:
    export OPENAI_API_KEY=...

Submit:
    python 02_self_use_classification.py --submit-only

Download after batches complete:
    python 02_self_use_classification.py --download
"""

import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI


PROJECT_DIR = Path(__file__).resolve().parent
PROMPT_FILE = PROJECT_DIR / "prompts" / "self_use_classification_prompt.txt"
INPUT_CSV = PROJECT_DIR / "data" / "filtered_posts.csv"
BATCH_DIR = PROJECT_DIR / "batch_jobs" / "self_use"
OUTPUT_CSV = PROJECT_DIR / "data" / "self_use_posts.csv"

DEFAULT_MODEL = "gpt-5.4-nano"
TASKS_PER_FILE = 16000
MAX_TOKENS = 3000


def load_system_prompt(path):
    raw = Path(path).read_text()
    if raw.startswith('SYSTEM_PROMPT = """'):
        raw = raw[len('SYSTEM_PROMPT = """'):]
        raw = raw.lstrip("\\").lstrip("\n")
        if raw.endswith('"""'):
            raw = raw[:-3]
    return raw.strip()


def build_user_message(row):
    date = pd.to_datetime(row["created_utc"]).date() if "created_utc" in row.index else ""
    return f"posted in r/{row['subreddit']}\nposted on {date}\n{row['message']}"


def create_batch_files(df, system_prompt, model, batch_dir):
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
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "max_completion_tokens": MAX_TOKENS,
                        "temperature": 0,
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": build_user_message(row)},
                        ],
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
            endpoint="/v1/chat/completions",
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
    return [line.strip() for line in ids_file.read_text().splitlines() if line.strip()]


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
    return output_paths


def parse_results(output_paths):
    rows = []
    for path in output_paths:
        with path.open() as f:
            for line in f:
                result = json.loads(line)
                df_idx = int(result["custom_id"].replace("task-", ""))
                try:
                    content = result["response"]["body"]["choices"][0]["message"]["content"]
                    parsed = json.loads(content)
                except (KeyError, json.JSONDecodeError) as exc:
                    parsed = {"_error": str(exc)}
                parsed["_task_index"] = df_idx
                rows.append(parsed)
    return pd.DataFrame(rows).sort_values("_task_index").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Classify retatrutide self-use with OpenAI Batch API.")
    parser.add_argument("--csv", default=str(INPUT_CSV))
    parser.add_argument("--prompt", default=str(PROMPT_FILE))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-dir", default=str(BATCH_DIR))
    parser.add_argument("--output", default=str(OUTPUT_CSV))
    parser.add_argument("--batch-ids", default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--submit-only", action="store_true")
    mode.add_argument("--download", action="store_true")
    args = parser.parse_args()

    client = OpenAI()
    batch_dir = Path(args.batch_dir)

    if args.submit_only:
        df = pd.read_csv(args.csv)
        prompt = load_system_prompt(args.prompt)
        file_paths = create_batch_files(df, prompt, args.model, batch_dir)
        submit_batches(client, file_paths, batch_dir)
        return

    batch_ids = read_batch_ids(batch_dir, args.batch_ids)
    output_paths = download_results(client, batch_ids, batch_dir)
    if not output_paths:
        print("No completed batch outputs found.")
        return

    df = pd.read_csv(args.csv)
    results = parse_results(output_paths)
    merged = df.merge(results, left_index=True, right_on="_task_index", how="left")

    if "currently_taking_retatrutide" not in merged.columns:
        raise SystemExit("Column currently_taking_retatrutide missing from model output.")

    self_use = merged[merged["currently_taking_retatrutide"].str.lower().eq("yes")].copy()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    self_use.to_csv(output, index=False)
    print(f"Saved {len(self_use):,} self-use rows to {output}")


if __name__ == "__main__":
    main()
