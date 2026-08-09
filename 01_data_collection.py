"""
Step 1: Retatrutide Reddit Data Collection
==========================================

Collects public Reddit posts and comments mentioning retatrutide from a local
Pushshift/Arctic Shift-style MySQL archive with monthly tables:

  comments:    com_YYYY_MM
  submissions: sub_YYYY_MM

Tier 1 retatrutide-specific subreddits are queried broadly first and then
filtered for retatrutide terms after quality control. Tier 2 broader
subreddits are keyword-filtered during SQL querying.

No database credentials are stored in this script. Use a MySQL option file or
environment variables.

Example:
    python 01_data_collection.py --read-default-file ~/.my.cnf --output data/filtered_posts.csv
"""

import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import sqlalchemy


TIER1_SUBREDDITS = [
    "RetatrutideGBP",
    "Retatrutide",
    "RetatrutideWomen",
    "RetatrutideTrial",
    "retatrutide4obesity",
    "retatrutidevendors",
]

TIER2_SUBREDDITS = [
    "Biohackers",
    "Zepbound",
    "GymMotivation",
    "WeightLossAdvice",
    "BariatricSurgery",
    "workout",
    "fit",
    "Mounjaro",
    "Peptidesource",
    "BodyHackGuide",
    "Peptides",
    "PeptideGuidesPH",
    "compoundedtirzepatide",
    "peptidess",
    "PeptidePathways",
    "USPeptides",
    "PeptideGuide",
    "PeptideDiscussion",
    "PeptideForum",
    "PeptideProgress",
    "tirzepatidecompound",
]

DRUG_TERMS = [
    "retatrutide",
    "reta",
    "retatrutid",
    "retaturtide",
    "retratrutide",
    "retatritide",
    "retatruide",
    "retartutide",
    "retatrutyde",
    "reterutide",
    "retatrudite",
    "retatruitde",
    "retratutide",
    "retatatrutide",
    "retatrutiede",
    "retatratide",
]

BOT_USER_IDS = [
    "AutoModerator",
    "AutoPoster",
    "ModeratorBot",
    "AdminBot",
    "SystemBot",
    "HelperBot",
    "AssistantBot",
    "ServiceBot",
    "ReplyBot",
    "NotifierBot",
    "AlertBot",
    "ManagerBot",
    "RobotUser",
    "FilterBot",
]

COMMENT_COLS = "user_id, message_id, message, created_utc, subreddit, permalink"
SUBMISSION_COLS = "user_id, message_id, title, message, created_utc, subreddit, permalink"


def sql_quote(value):
    return "'" + value.replace("'", "''") + "'"


def sql_in_list(values):
    return ", ".join(sql_quote(v) for v in values)


def build_drug_regexp():
    return "|".join(re.escape(term) for term in DRUG_TERMS)


DRUG_REGEXP = build_drug_regexp()
DRUG_PATTERN_PY = re.compile("|".join(re.escape(t) for t in DRUG_TERMS), re.IGNORECASE)
TIER1_IN = sql_in_list(TIER1_SUBREDDITS)
TIER2_IN = sql_in_list(TIER2_SUBREDDITS)
BOT_IN = sql_in_list(BOT_USER_IDS)


def generate_table_names(prefix, start_year, start_month, end_year, end_month):
    tables = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        tables.append(f"{prefix}_{year}_{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return tables


def make_engine(args):
    query = {"charset": "utf8mb4"}
    read_default_file = args.read_default_file or os.environ.get("MYSQL_DEFAULTS_FILE")
    if read_default_file:
        query["read_default_file"] = os.path.expanduser(read_default_file)

    url = sqlalchemy.engine.url.URL.create(
        drivername="mysql+pymysql",
        username=args.user or os.environ.get("MYSQL_USER"),
        password=args.password or os.environ.get("MYSQL_PASSWORD"),
        host=args.host or os.environ.get("MYSQL_HOST", "127.0.0.1"),
        database=args.database or os.environ.get("MYSQL_DATABASE", "reddit"),
        query=query,
    )
    return sqlalchemy.create_engine(url)


def build_comment_query_tier1(table):
    return f"""
        SELECT {COMMENT_COLS}
        FROM {table}
        WHERE subreddit IN ({TIER1_IN})
          AND message IS NOT NULL
          AND message NOT IN ('[removed]', '[deleted]')
          AND user_id NOT IN ({BOT_IN})
    """


def build_comment_query_tier2(table):
    return f"""
        SELECT {COMMENT_COLS}
        FROM {table}
        WHERE subreddit IN ({TIER2_IN})
          AND message IS NOT NULL
          AND message NOT IN ('[removed]', '[deleted]')
          AND user_id NOT IN ({BOT_IN})
          AND message REGEXP '{DRUG_REGEXP}'
    """


def build_submission_query_tier1(table):
    return f"""
        SELECT {SUBMISSION_COLS}
        FROM {table}
        WHERE subreddit IN ({TIER1_IN})
          AND title IS NOT NULL
          AND message IS NOT NULL
          AND message NOT IN ('[removed]', '[deleted]')
          AND user_id NOT IN ({BOT_IN})
    """


def build_submission_query_tier2(table):
    return f"""
        SELECT {SUBMISSION_COLS}
        FROM {table}
        WHERE subreddit IN ({TIER2_IN})
          AND title IS NOT NULL
          AND message IS NOT NULL
          AND message NOT IN ('[removed]', '[deleted]')
          AND user_id NOT IN ({BOT_IN})
          AND (message REGEXP '{DRUG_REGEXP}' OR title REGEXP '{DRUG_REGEXP}')
    """


def fetch_table(engine, table, query_builder, label):
    query = query_builder(table)
    start = time.time()
    try:
        df = pd.read_sql_query(query, engine)
    except Exception as exc:
        if "doesn't exist" in str(exc) or "does not exist" in str(exc):
            print(f"{label} {table}: skipped, table not found")
            return pd.DataFrame()
        raise
    print(f"{label} {table}: {len(df):,} rows ({time.time() - start:.1f}s)")
    return df


def collect_parallel(engine, tables, query_builder, label, max_workers):
    frames = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_table, engine, table, query_builder, label): table
            for table in tables
        }
        for future in as_completed(futures):
            df = future.result()
            if not df.empty:
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def alpha_ratio(text):
    text = str(text)
    return sum(char.isalpha() for char in text) / max(len(text), 1)


def quality_filter_comments(df, min_words, min_alpha_ratio):
    if df.empty:
        return df
    keep = df.copy()
    keep = keep[~keep["user_id"].isin(BOT_USER_IDS)]
    keep = keep[keep["message"].str.split().str.len() >= min_words]
    keep = keep[keep["message"].str[:2048].str.contains(" ", na=False)]
    keep = keep[keep["message"].apply(alpha_ratio) >= min_alpha_ratio]
    return keep


def quality_filter_submissions(df, min_words, min_alpha_ratio):
    if df.empty:
        return df
    keep = df.copy()
    keep = keep[~keep["user_id"].isin(BOT_USER_IDS)]
    keep = keep[keep["title"].str[:2048].str.contains(" ", na=False)]
    keep["message"] = keep["title"].fillna("") + " " + keep["message"].fillna("")
    keep = keep[keep["message"].str.split().str.len() >= min_words]
    keep = keep[keep["message"].apply(alpha_ratio) >= min_alpha_ratio]
    return keep


def filter_drug_mentions(df):
    if df.empty:
        return df
    return df[df["message"].str.contains(DRUG_PATTERN_PY, na=False)]


def export_combined(comments, submissions, output):
    cols = ["user_id", "subreddit", "message", "created_utc", "message_id", "permalink"]
    combined = pd.concat([comments[cols], submissions[cols]], ignore_index=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    return combined


def main():
    parser = argparse.ArgumentParser(description="Collect retatrutide-related Reddit posts/comments.")
    parser.add_argument("--host", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--user", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--read-default-file", default=None)
    parser.add_argument("--output", default="data/filtered_posts.csv")
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--start-month", type=int, default=5)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--end-month", type=int, default=12)
    parser.add_argument("--min-words", type=int, default=10)
    parser.add_argument("--min-alpha-ratio", type=float, default=0.50)
    parser.add_argument("--max-workers", type=int, default=6)
    args = parser.parse_args()

    engine = make_engine(args)
    comment_tables = generate_table_names("com", args.start_year, args.start_month, args.end_year, args.end_month)
    submission_tables = generate_table_names("sub", args.start_year, args.start_month, args.end_year, args.end_month)

    print("Collecting comments")
    comments_t1 = collect_parallel(engine, comment_tables, build_comment_query_tier1, "T1 comment", args.max_workers)
    comments_t1["tier"] = 1
    comments_t2 = collect_parallel(engine, comment_tables, build_comment_query_tier2, "T2 comment", args.max_workers)
    comments_t2["tier"] = 2

    print("Collecting submissions")
    submissions_t1 = collect_parallel(engine, submission_tables, build_submission_query_tier1, "T1 submission", args.max_workers)
    submissions_t1["tier"] = 1
    submissions_t2 = collect_parallel(engine, submission_tables, build_submission_query_tier2, "T2 submission", args.max_workers)
    submissions_t2["tier"] = 2

    comments = quality_filter_comments(pd.concat([comments_t1, comments_t2], ignore_index=True), args.min_words, args.min_alpha_ratio)
    submissions = quality_filter_submissions(pd.concat([submissions_t1, submissions_t2], ignore_index=True), args.min_words, args.min_alpha_ratio)

    comments_final = pd.concat(
        [
            filter_drug_mentions(comments[comments["tier"] == 1]),
            comments[comments["tier"] == 2],
        ],
        ignore_index=True,
    )
    submissions_final = pd.concat(
        [
            filter_drug_mentions(submissions[submissions["tier"] == 1]),
            submissions[submissions["tier"] == 2],
        ],
        ignore_index=True,
    )

    combined = export_combined(comments_final, submissions_final, args.output)
    print(f"Saved {len(combined):,} posts/comments to {args.output}")
    print(f"Unique users: {combined['user_id'].nunique():,}")


if __name__ == "__main__":
    main()
