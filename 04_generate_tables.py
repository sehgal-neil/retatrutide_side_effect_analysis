"""
Step 7: Generate Analysis Tables — MedDRA Frequency and Co-occurrence
======================================================================
Maps extracted MedDRA concept strings to official PT codes, excludes
weight-loss-related terms, aggregates to user level, and produces:

  - unmatched_concepts_for_review.csv  (optional: remaining strings with no PT; omitted from tables)
  - table_pt_by_soc.csv                Table 1: PTs grouped by SOC
  - table_hlt_counts.csv               HLT frequencies
  - table_hlgt_counts.csv              HLGT frequencies
  - table_cooccurrence.csv             Top co-occurring PT pairs

WORKFLOW:
  1. python 04_generate_tables.py          <- resolves via PT name, LLT->PT, then
                                              manual_meddra_mappings.csv if present
  2. Optional: add rows to manual_meddra_mappings.csv for stragglers, re-run
  3. Any concepts still unmatched are skipped in frequency tables (safe to ignore)

Run options:
    python 04_generate_tables.py
    python 04_generate_tables.py --top-pairs 50
    python 04_generate_tables.py --min-pct 0 --min-soc-pct 0 --min-hlgt-pct 0
"""

import argparse
import itertools
import os
from ast import literal_eval
from collections import Counter

import numpy as np
import pandas as pd

# ============================================================================
# Configuration
# ============================================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
POSTS_FILE = os.path.join(PROJECT_DIR, "data", "posts_with_meddra.csv")
MEDDRA_DIR = os.environ.get("MEDDRA_MEDASCII_DIR", "")
MANUAL_MAPPINGS_FILE = os.path.join(PROJECT_DIR, "manual_meddra_mappings.csv")

OUTPUT_DIR_DEFAULT = os.path.join(PROJECT_DIR, "outputs")

# Weight-related / appetite terms to exclude (lowercased for matching)
WEIGHT_LOSS_EXCLUSIONS = {
    "weight decreased",
    "abnormal loss of weight",
    "decreased appetite",
    "hypophagia",
    "appetite disorder",
    "early satiety",
    "fat tissue decreased",
    "weight loss poor",
    "waist circumference decreased",
    "weight control",
}


# ============================================================================
# Load MedDRA hierarchy
# ============================================================================


def load_meddra(meddra_dir):
    """Load MedDRA hierarchy files and return lookup tables."""

    def asc(filename, names, usecols):
        return pd.read_csv(
            os.path.join(meddra_dir, filename),
            sep="$", header=None, usecols=usecols,
            names=names, encoding="latin-1",
        )

    pt    = asc("pt.asc",      ["pt_code",   "pt_term"],   [0, 1])
    hlt   = asc("hlt.asc",     ["hlt_code",  "hlt_term"],  [0, 1])
    hlgt  = asc("hlgt.asc",    ["hlgt_code", "hlgt_term"], [0, 1])
    soc   = asc("soc.asc",     ["soc_code",  "soc_term"],  [0, 1])
    llt   = asc("llt.asc",     ["llt_code", "llt_term", "pt_code", "c3", "c4", "c5",
                                 "c6", "c7", "c8", "llt_currency"], range(10))

    mdhier = pd.read_csv(
        os.path.join(meddra_dir, "mdhier.asc"),
        sep="$", header=None, encoding="latin-1", index_col=False,
        names=[
            "pt_code", "hlt_code", "hlgt_code", "soc_code",
            "pt_name", "hlt_name", "hlgt_name", "soc_name",
            "soc_abbrev", "null_field", "pt_soc_code", "primary_soc_fg",
        ],
    )
    mdhier_primary = mdhier[mdhier["primary_soc_fg"] == "Y"].copy()

    # Build LLT name (lowercase) -> pt_code lookup (current terms only, prefer current)
    llt_current = llt[llt["llt_currency"].str.strip() == "Y"]
    llt_name_to_ptcode = dict(
        zip(llt_current["llt_term"].str.strip().str.lower(), llt_current["pt_code"])
    )
    # Also include non-current as fallback (will be overwritten by current if duplicate)
    llt_noncurrent = llt[llt["llt_currency"].str.strip() != "Y"]
    llt_noncurrent_map = dict(
        zip(llt_noncurrent["llt_term"].str.strip().str.lower(), llt_noncurrent["pt_code"])
    )
    llt_noncurrent_map.update(llt_name_to_ptcode)  # current takes precedence
    full_llt_map = llt_noncurrent_map

    return {
        "pt":       pt,
        "hlt":      hlt,
        "hlgt":     hlgt,
        "soc":      soc,
        "mdhier":   mdhier_primary,
        "llt_map":  full_llt_map,   # llt_term.lower() -> pt_code
    }


# ============================================================================
# Load and parse posts
# ============================================================================


def load_posts(posts_file):
    """Load posts_with_meddra.csv, handling the duplicate _task_index columns."""
    df = pd.read_csv(posts_file, low_memory=False)

    # Drop duplicate _task_index columns if present (merge artifact)
    dup_cols = [c for c in df.columns if c == "_task_index"]
    if len(dup_cols) > 1:
        # Keep only the first occurrence
        seen = set()
        keep = []
        for c in df.columns:
            if c not in seen:
                keep.append(c)
                seen.add(c)
            else:
                keep.append(None)
        df.columns = [c if c is not None else f"_drop_{i}" for i, c in enumerate(keep)]
        df = df[[c for c in df.columns if not c.startswith("_drop_")]]

    return df


def parse_concepts(x):
    """Parse a meddra_concepts string ('['Nausea', 'Fatigue']') into a set."""
    if pd.isna(x) or (isinstance(x, str) and x.strip() in ("", "[]")):
        return set()
    try:
        parsed = literal_eval(x)
        if isinstance(parsed, list):
            return {c.strip() for c in parsed if isinstance(c, str) and c.strip()}
    except (ValueError, SyntaxError):
        pass
    return set()


# ============================================================================
# Build concept → PT code mapping
# ============================================================================


def build_concept_mapping(df, meddra_tables, manual_mappings_file=None):
    """
    Resolves concept strings to PT codes via three steps (in priority order):
      1. Direct PT name match
      2. Manual mappings (manual_meddra_mappings.csv)
      3. LLT name match -> parent PT code

    Returns:
        concept_to_ptcode: dict  lowercased concept string -> pt_code (int)
        unmatched: Counter       original concept -> frequency for concepts with no match
    """
    pt_df = meddra_tables["pt"]
    pt_name_to_code = dict(zip(pt_df["pt_term"].str.strip().str.lower(), pt_df["pt_code"]))
    llt_map = meddra_tables["llt_map"]

    # Load manual mappings if file exists
    manual_map = {}
    if manual_mappings_file and os.path.exists(manual_mappings_file):
        manual_df = pd.read_csv(manual_mappings_file)
        manual_df = manual_df.dropna(subset=["pt_code"])
        manual_df["pt_code"] = manual_df["pt_code"].astype(int)
        manual_map = dict(
            zip(manual_df["meddra_concept"].str.strip().str.lower(), manual_df["pt_code"])
        )
        print(f"  Loaded {len(manual_map):,} manual mappings from {manual_mappings_file}")

    # Collect all unique concepts across the dataset
    all_concepts: Counter = Counter()
    for x in df["meddra_concepts"].dropna():
        for c in parse_concepts(x):
            all_concepts[c] += 1

    print(f"  Unique concept strings: {len(all_concepts):,}")

    concept_to_ptcode = {}
    unmatched: Counter = Counter()
    n_via_pt = n_via_manual = n_via_llt = 0

    for concept, freq in all_concepts.items():
        key = concept.lower().strip()
        if key in pt_name_to_code:
            concept_to_ptcode[key] = pt_name_to_code[key]
            n_via_pt += 1
        elif key in manual_map:
            concept_to_ptcode[key] = manual_map[key]
            n_via_manual += 1
        elif key in llt_map:
            concept_to_ptcode[key] = llt_map[key]
            n_via_llt += 1
        else:
            unmatched[concept] += freq

    n_matched = len(all_concepts) - len(unmatched)
    print(f"  Matched:        {n_matched:,}  ({100 * n_matched / len(all_concepts):.1f}%)")
    print(f"    via PT name:  {n_via_pt:,}")
    if n_via_manual:
        print(f"    via manual:   {n_via_manual:,}")
    print(f"    via LLT->PT:  {n_via_llt:,}")
    print(f"  Unmatched:      {len(unmatched):,}")

    return concept_to_ptcode, unmatched


def save_unmatched(unmatched: Counter, output_path):
    """Save unmatched concepts sorted by frequency (blank pt_code for optional follow-up)."""
    rows = [{"meddra_concept": c, "frequency": freq, "pt_code": ""}
            for c, freq in unmatched.most_common()]
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df):,} unmatched concepts to: {output_path}")


# ============================================================================
# Build user-level data
# ============================================================================


def build_user_level(df, concept_to_ptcode, meddra_tables):
    """
    Aggregate posts to user level:
      - Union all PT codes across all posts per user
      - Map PT codes -> HLT, HLGT, SOC via primary SOC hierarchy
      - Exclude weight-loss terms
      - Drop users with no remaining PTs
    """
    mdhier = meddra_tables["mdhier"]
    pt_df  = meddra_tables["pt"]

    # Build PT code -> name lookup for exclusion by name
    pt_code_to_name = dict(zip(pt_df["pt_code"], pt_df["pt_term"].str.strip().str.lower()))

    # Build hierarchy lookups (pt_code -> set of codes at each level)
    pt2soc  = mdhier.groupby("pt_code")["soc_code"].apply(set).to_dict()
    pt2hlt  = mdhier.groupby("pt_code")["hlt_code"].apply(set).to_dict()
    pt2hlgt = mdhier.groupby("pt_code")["hlgt_code"].apply(set).to_dict()

    # Resolve concept strings -> PT code sets per row
    def row_to_pt_codes(concept_str):
        """Map each concept string to a PT code; strings with no mapping are omitted."""
        concepts = parse_concepts(concept_str)
        codes = set()
        for c in concepts:
            code = concept_to_ptcode.get(c.lower().strip())
            if code is not None:
                codes.add(code)
        return codes

    df = df.copy()
    df["pt_codes"] = df["meddra_concepts"].apply(row_to_pt_codes)

    # Aggregate to user level (union across all their posts)
    user_level = (
        df.groupby("user_id")["pt_codes"]
        .apply(lambda col: set().union(*col))
        .reset_index()
    )

    # Exclude weight-loss terms
    excluded_names = WEIGHT_LOSS_EXCLUSIONS
    excluded_codes = {
        code for code, name in pt_code_to_name.items()
        if name in excluded_names
    }
    print(f"\n  Weight-related/appetite-suppression exclusion PT codes: {excluded_codes}")

    user_level["pt_codes"] = user_level["pt_codes"].apply(
        lambda s: s - excluded_codes
    )

    # Drop users whose only PTs were in the exclusion set
    user_level = user_level[user_level["pt_codes"].apply(len) > 0].reset_index(drop=True)

    # Map to hierarchy codes
    def map_codes(pt_set, lookup):
        result = set()
        for pt_code in pt_set:
            if pt_code in lookup:
                result.update(lookup[pt_code])
        return result

    user_level["soc_codes"]  = user_level["pt_codes"].apply(lambda s: map_codes(s, pt2soc))
    user_level["hlt_codes"]  = user_level["pt_codes"].apply(lambda s: map_codes(s, pt2hlt))
    user_level["hlgt_codes"] = user_level["pt_codes"].apply(lambda s: map_codes(s, pt2hlgt))

    return user_level, pt2soc, pt2hlt, pt2hlgt


# ============================================================================
# PT frequency table (Table 1, grouped by SOC)
# ============================================================================


def generate_pt_by_soc(user_level, meddra_tables, pt2soc, min_pct=0.5, min_soc_pct=1.0):
    """PT counts grouped by primary SOC, sorted by SOC count then PT count."""
    n_users = len(user_level)
    pt_df   = meddra_tables["pt"]
    soc_df  = meddra_tables["soc"]

    # Count users per PT
    pt_counts = (
        user_level.explode("pt_codes", ignore_index=True)
        .dropna(subset=["pt_codes"])
        .value_counts(subset=["pt_codes"])
        .rename("n_users")
        .reset_index()
        .rename(columns={"pt_codes": "pt_code"})
    )
    pt_counts = pt_counts.merge(pt_df, how="left", on="pt_code")
    pt_counts["percent_users"] = np.round(100 * pt_counts["n_users"] / n_users, 2)

    # Count users per SOC
    soc_counts = (
        user_level.explode("soc_codes", ignore_index=True)
        .dropna(subset=["soc_codes"])
        .value_counts(subset=["soc_codes"])
        .rename("n_users")
        .reset_index()
        .rename(columns={"soc_codes": "soc_code"})
    )
    soc_counts = soc_counts.merge(soc_df, how="left", on="soc_code")
    soc_counts["soc_percent"] = np.round(100 * soc_counts["n_users"] / n_users, 2)

    # Join PT -> primary SOC
    map_rows = pd.DataFrame([
        {"pt_code": pt, "soc_code": soc}
        for pt, soc_set in pt2soc.items()
        for soc in soc_set
    ])
    pt_with_soc = pt_counts.merge(map_rows, on="pt_code", how="left")
    soc_lookup = soc_counts[["soc_code", "soc_term", "n_users"]].rename(
        columns={"n_users": "soc_n_users"}
    )
    pt_with_soc = pt_with_soc.merge(soc_lookup, on="soc_code", how="left")

    # Apply prevalence filters using exact rates rather than rounded display columns.
    if min_pct > 0:
        pt_with_soc = pt_with_soc[
            (100 * pt_with_soc["n_users"] / n_users) >= min_pct
        ]
    if min_soc_pct > 0:
        pt_with_soc = pt_with_soc[
            (100 * pt_with_soc["soc_n_users"] / n_users) >= min_soc_pct
        ]

    # Sort: SOC by descending user count, then PT within SOC
    pt_with_soc = pt_with_soc.sort_values(
        ["soc_n_users", "n_users"], ascending=[False, False]
    ).reset_index(drop=True)

    return pt_with_soc, pt_counts, soc_counts


# ============================================================================
# HLT and HLGT frequency tables
# ============================================================================


def generate_hlt_hlgt(user_level, meddra_tables):
    n_users = len(user_level)

    hlt_counts = (
        user_level.explode("hlt_codes", ignore_index=True)
        .dropna(subset=["hlt_codes"])
        .value_counts(subset=["hlt_codes"])
        .rename("n_users")
        .reset_index()
        .rename(columns={"hlt_codes": "hlt_code"})
    )
    hlt_counts = hlt_counts.merge(meddra_tables["hlt"], how="left", on="hlt_code")
    hlt_counts["percent_users"] = np.round(100 * hlt_counts["n_users"] / n_users, 2)

    hlgt_counts = (
        user_level.explode("hlgt_codes", ignore_index=True)
        .dropna(subset=["hlgt_codes"])
        .value_counts(subset=["hlgt_codes"])
        .rename("n_users")
        .reset_index()
        .rename(columns={"hlgt_codes": "hlgt_code"})
    )
    hlgt_counts = hlgt_counts.merge(meddra_tables["hlgt"], how="left", on="hlgt_code")
    hlgt_counts["percent_users"] = np.round(100 * hlgt_counts["n_users"] / n_users, 2)

    return hlt_counts, hlgt_counts


# ============================================================================
# Co-occurrence table
# ============================================================================


def generate_cooccurrence(user_level, meddra_tables, top_n=20):
    """Top PT pairs by number of users who reported both."""
    pt_df = meddra_tables["pt"]
    code_to_name = dict(zip(pt_df["pt_code"], pt_df["pt_term"]))

    pair_counts: dict = {}
    for _, row in user_level.iterrows():
        codes = sorted(row["pt_codes"])
        for a, b in itertools.combinations(codes, 2):
            pair = (a, b)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    if not pair_counts:
        return pd.DataFrame(columns=["pt_code_1", "pt_code_2", "pt_term_1", "pt_term_2",
                                     "n_users", "percent_users"])

    pairs_df = pd.DataFrame([
        {
            "pt_code_1":  k[0],
            "pt_code_2":  k[1],
            "pt_term_1":  code_to_name.get(k[0], str(k[0])),
            "pt_term_2":  code_to_name.get(k[1], str(k[1])),
            "n_users":    v,
        }
        for k, v in pair_counts.items()
    ])
    n_users = len(user_level)
    pairs_df = pairs_df.sort_values("n_users", ascending=False).head(top_n)
    pairs_df["percent_users"] = np.round(100 * pairs_df["n_users"] / n_users, 2)
    pairs_df = pairs_df.reset_index(drop=True)
    return pairs_df


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Generate MedDRA frequency tables for Retatrutide project")
    parser.add_argument("--posts",     default=POSTS_FILE,    help="Path to posts_with_meddra.csv")
    parser.add_argument("--meddra",    default=MEDDRA_DIR,    help="Path to MedDRA MedAscii directory")
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT, help="Directory for output CSVs")
    parser.add_argument("--min-pct",   type=float, default=0.5,
                        help="Min %% users to include a PT in the PT/SOC table (default: 0.5)")
    parser.add_argument("--min-soc-pct", type=float, default=1.0,
                        help="Min %% users to include a SOC in the PT/SOC table (default: 1.0)")
    parser.add_argument("--min-hlgt-pct", type=float, default=0.5,
                        help="Min %% users to include a HLGT in the HLGT table (default: >0.5)")
    parser.add_argument("--top-pairs", type=int,   default=20,
                        help="Number of top co-occurring pairs to report (default: 20)")
    args = parser.parse_args()
    if not args.meddra:
        raise SystemExit("Provide --meddra or set MEDDRA_MEDASCII_DIR to the MedDRA MedAscii directory.")
    os.makedirs(args.output_dir, exist_ok=True)
    output_unmatched = os.path.join(args.output_dir, "unmatched_concepts_for_review.csv")
    output_pt_soc = os.path.join(args.output_dir, "table_pt_by_soc.csv")
    output_hlt = os.path.join(args.output_dir, "table_hlt_counts.csv")
    output_hlgt = os.path.join(args.output_dir, "table_hlgt_counts.csv")
    output_cooccurrence = os.path.join(args.output_dir, "table_cooccurrence.csv")

    print("=" * 65)
    print("STEP 7: MedDRA Analysis Tables")
    print("=" * 65)

    # ---- Load MedDRA hierarchy ----
    print("\nLoading MedDRA hierarchy...")
    meddra = load_meddra(args.meddra)
    print(f"  PT terms:   {len(meddra['pt']):,}")
    print(f"  HLT terms:  {len(meddra['hlt']):,}")
    print(f"  HLGT terms: {len(meddra['hlgt']):,}")
    print(f"  SOC terms:  {len(meddra['soc']):,}")

    # ---- Load posts ----
    print(f"\nLoading posts from: {args.posts}")
    df = load_posts(args.posts)
    total_rows = len(df)
    rows_with_results = df["meddra_concepts"].notna().sum()
    rows_missing = total_rows - rows_with_results
    print(f"  Total rows:         {total_rows:,}")
    print(f"  Rows with results:  {rows_with_results:,}  ({100 * rows_with_results / total_rows:.1f}%)")
    if rows_missing > 0:
        print(f"  Rows missing:       {rows_missing:,}  (rate-limit failures — re-run after retry batches complete)")

    # ---- Map concepts -> PT codes ----
    print("\nBuilding concept -> PT code mapping...")
    concept_to_ptcode, unmatched = build_concept_mapping(df, meddra, MANUAL_MAPPINGS_FILE)

    # Save remaining unmatched (for audit); they are omitted from user-level aggregation
    if unmatched:
        n_mentions = sum(unmatched.values())
        print(f"\nSaving remaining unmatched concepts (for reference)...")
        save_unmatched(unmatched, output_unmatched)
        print(f"  These {len(unmatched):,} unique strings ({n_mentions:,} total mentions) "
              f"have no PT - they are skipped in the tables.")
        print(f"  To map more: add rows to {MANUAL_MAPPINGS_FILE} and re-run. Otherwise OK to ignore.")
    else:
        print("\n  All concepts mapped to a PT.")

    # ---- Build user-level data ----
    print("\nAggregating to user level (union of PTs per user)...")
    user_level, pt2soc, pt2hlt, pt2hlgt = build_user_level(df, concept_to_ptcode, meddra)
    n_users = len(user_level)
    pt_counts_per_user = user_level["pt_codes"].apply(len)
    print(f"  Users with ≥1 side effect (after exclusions): {n_users:,}")
    print(f"  Mean PTs per user: {pt_counts_per_user.mean():.2f}  "
          f"(SD {pt_counts_per_user.std():.2f},  "
          f"median {pt_counts_per_user.median():.0f},  "
          f"max {pt_counts_per_user.max():.0f})")

    # ---- Table 1: PT by SOC ----
    print(f"\nGenerating PT frequency table (min_pct={args.min_pct}%)...")
    pt_by_soc, pt_counts_df, soc_counts_df = generate_pt_by_soc(
        user_level, meddra, pt2soc, min_pct=args.min_pct, min_soc_pct=args.min_soc_pct
    )
    pt_by_soc.to_csv(output_pt_soc, index=False)
    print(f"  {len(pt_by_soc):,} PTs across {pt_by_soc['soc_term'].nunique()} SOCs")
    print(f"  Saved: {output_pt_soc}")

    # Print top 15 PTs
    print("\n  Top 15 PTs:")
    for _, r in pt_counts_df.head(15).iterrows():
        print(f"    {r['n_users']:>5}  ({r['percent_users']:>5.1f}%)  {r['pt_term']}")

    # ---- HLT / HLGT ----
    print(f"\nGenerating HLT and HLGT frequency tables...")
    hlt_counts, hlgt_counts = generate_hlt_hlgt(user_level, meddra)
    if args.min_hlgt_pct > 0:
        hlgt_counts = hlgt_counts[
            (100 * hlgt_counts["n_users"] / n_users) > args.min_hlgt_pct
        ].reset_index(drop=True)
    hlt_counts.to_csv(output_hlt, index=False)
    hlgt_counts.to_csv(output_hlgt, index=False)
    print(f"  HLT table:  {len(hlt_counts):,} terms -> {output_hlt}")
    print(f"  HLGT table: {len(hlgt_counts):,} terms -> {output_hlgt}")

    # Print top 10 HLTs
    print("\n  Top 10 HLTs:")
    for _, r in hlt_counts.head(10).iterrows():
        print(f"    {r['n_users']:>5}  ({r['percent_users']:>5.1f}%)  {r['hlt_term']}")

    # ---- Co-occurrence ----
    print(f"\nGenerating co-occurrence table (top {args.top_pairs} pairs)...")
    cooccurrence = generate_cooccurrence(user_level, meddra, top_n=args.top_pairs)
    cooccurrence.to_csv(output_cooccurrence, index=False)
    print(f"  Saved: {output_cooccurrence}")
    if not cooccurrence.empty:
        top = cooccurrence.iloc[0]
        print(f"  Top pair: {top['pt_term_1']} & {top['pt_term_2']} "
              f"({top['n_users']:,} users, {top['percent_users']}%)")
        print("\n  Top 10 pairs:")
        for _, r in cooccurrence.head(10).iterrows():
            print(f"    {r['n_users']:>5}  ({r['percent_users']:>5.1f}%)  "
                  f"{r['pt_term_1']}  +  {r['pt_term_2']}")

    # ---- Final summary ----
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  Total posts processed:         {total_rows:,}")
    print(f"  Posts with extraction results: {rows_with_results:,}  ({100 * rows_with_results / total_rows:.1f}%)")
    print(f"  Users with ≥1 side effect:     {n_users:,}")
    print(f"  Unique PT concepts mapped:     {len(concept_to_ptcode):,}")
    if unmatched:
        n_mentions = sum(unmatched.values())
        print(f"  Unmatched concepts (skipped):  {len(unmatched):,} unique  ({n_mentions:,} mentions)")
    else:
        print(f"  Unmatched concepts:            0")
    print(f"  PT table rows (≥{args.min_pct}%):         {len(pt_by_soc):,}")
    print(f"  HLT table rows:                {len(hlt_counts):,}")
    print(f"  HLGT table rows:               {len(hlgt_counts):,}")
    print(f"  Co-occurrence pairs:           {len(cooccurrence):,}")
    print()


if __name__ == "__main__":
    main()
