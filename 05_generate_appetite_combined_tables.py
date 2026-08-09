"""
Step 7 variant: Generate tables after combining related appetite-increase PTs.

This script reuses the main 04_generate_tables.py pipeline, but collapses these
MedDRA Preferred Terms (PTs) into one composite row before making tables:

  - Hunger
  - Increased appetite
  - Food craving
  - Lack of satiety

The composite uses the existing MedDRA hierarchy for "Increased appetite"
(Metabolism and nutrition disorders -> Appetite disorders -> Appetite and
general nutritional disorders). This means users who only had "Hunger" are
moved from Hunger's original primary SOC into the appetite composite.

Outputs are written alongside the main tables with the suffix
appetite_combined, without overwriting the main analysis tables:

  - table_pt_by_soc_appetite_combined.csv
  - table_hlt_counts_appetite_combined.csv
  - table_hlgt_counts_appetite_combined.csv
  - table_cooccurrence_appetite_combined.csv
  - unmatched_concepts_for_review_appetite_combined.csv

Run:
    python 05_generate_appetite_combined_tables.py
"""

import argparse
import importlib.util
import os
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = PROJECT_DIR / "04_generate_tables.py"

COMPOSITE_PT_NAME = "Increased appetite / hunger / food craving / lack of satiety"
COMPOSITE_REPRESENTATIVE_PT = "increased appetite"
SOURCE_PT_TERMS = [
    "hunger",
    "increased appetite",
    "food craving",
    "lack of satiety",
]


def load_base_module():
    """Load 04_generate_tables.py despite the numeric module name."""
    spec = importlib.util.spec_from_file_location("base_generate_tables", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def output_path(output_dir, stem, suffix):
    return Path(output_dir) / f"{stem}_{suffix}.csv"


def get_pt_code_lookup(meddra):
    pt = meddra["pt"]
    return dict(zip(pt["pt_term"].str.strip().str.lower(), pt["pt_code"].astype(int)))


def map_codes(pt_set, lookup):
    result = set()
    for pt_code in pt_set:
        if pt_code in lookup:
            result.update(lookup[pt_code])
    return result


def combine_appetite_pts(user_level, meddra, pt2soc, pt2hlt, pt2hlgt):
    """
    Collapse selected appetite-related PTs into the Increased appetite PT code.

    Returns:
        modified user_level, modified meddra copy, and a component count table.
    """
    pt_name_to_code = get_pt_code_lookup(meddra)
    missing = [term for term in SOURCE_PT_TERMS if term not in pt_name_to_code]
    if missing:
        raise ValueError(f"Missing source PT terms in MedDRA lookup: {missing}")

    source_codes = {pt_name_to_code[term] for term in SOURCE_PT_TERMS}
    representative_code = pt_name_to_code[COMPOSITE_REPRESENTATIVE_PT]

    component_rows = []
    for term in SOURCE_PT_TERMS:
        code = pt_name_to_code[term]
        n_users = int(user_level["pt_codes"].apply(lambda codes: code in codes).sum())
        component_rows.append({"pt_term": term, "pt_code": code, "n_users": n_users})

    n_any = int(user_level["pt_codes"].apply(lambda codes: bool(codes & source_codes)).sum())
    component_rows.append({
        "pt_term": COMPOSITE_PT_NAME,
        "pt_code": representative_code,
        "n_users": n_any,
    })

    combined = user_level.copy()

    def collapse_codes(codes):
        codes = set(codes)
        if codes & source_codes:
            codes = (codes - source_codes) | {representative_code}
        return codes

    combined["pt_codes"] = combined["pt_codes"].apply(collapse_codes)
    combined["soc_codes"] = combined["pt_codes"].apply(lambda codes: map_codes(codes, pt2soc))
    combined["hlt_codes"] = combined["pt_codes"].apply(lambda codes: map_codes(codes, pt2hlt))
    combined["hlgt_codes"] = combined["pt_codes"].apply(lambda codes: map_codes(codes, pt2hlgt))

    meddra_mod = dict(meddra)
    pt_mod = meddra["pt"].copy()
    pt_mod.loc[pt_mod["pt_code"].astype(int).eq(representative_code), "pt_term"] = COMPOSITE_PT_NAME
    meddra_mod["pt"] = pt_mod

    return combined, meddra_mod, pd.DataFrame(component_rows)


def main():
    base = load_base_module()

    parser = argparse.ArgumentParser(
        description="Generate MedDRA tables after combining appetite-related PTs."
    )
    parser.add_argument("--posts", default=base.POSTS_FILE, help="Path to posts_with_meddra.csv")
    parser.add_argument("--meddra", default=base.MEDDRA_DIR, help="Path to MedDRA MedAscii directory")
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "outputs"), help="Directory for output CSVs")
    parser.add_argument("--suffix", default="appetite_combined", help="Output filename suffix")
    parser.add_argument("--min-pct", type=float, default=0.5,
                        help="Min %% users to include a PT in the PT/SOC table")
    parser.add_argument("--min-soc-pct", type=float, default=1.0,
                        help="Min %% users to include a SOC in the PT/SOC table")
    parser.add_argument("--min-hlgt-pct", type=float, default=0.5,
                        help="Min %% users to include a HLGT in the HLGT table")
    parser.add_argument("--top-pairs", type=int, default=20,
                        help="Number of top co-occurring pairs to report")
    args = parser.parse_args()
    if not args.meddra:
        raise SystemExit("Provide --meddra or set MEDDRA_MEDASCII_DIR to the MedDRA MedAscii directory.")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("STEP 7 VARIANT: Appetite-Combined MedDRA Analysis Tables")
    print("=" * 72)

    print("\nLoading MedDRA hierarchy...")
    meddra = base.load_meddra(args.meddra)

    print(f"\nLoading posts from: {args.posts}")
    df = base.load_posts(args.posts)
    print(f"  Total rows: {len(df):,}")
    print(f"  Rows with results: {df['meddra_concepts'].notna().sum():,}")

    print("\nBuilding concept -> PT code mapping...")
    concept_to_ptcode, unmatched = base.build_concept_mapping(
        df, meddra, base.MANUAL_MAPPINGS_FILE
    )

    unmatched_path = output_path(args.output_dir, "unmatched_concepts_for_review", args.suffix)
    if unmatched:
        base.save_unmatched(unmatched, unmatched_path)

    print("\nAggregating to user level...")
    user_level, pt2soc, pt2hlt, pt2hlgt = base.build_user_level(
        df, concept_to_ptcode, meddra
    )
    n_users = len(user_level)
    print(f"  Users with >=1 side effect after exclusions: {n_users:,}")

    print("\nCombining appetite-related PTs...")
    combined_user_level, combined_meddra, component_counts = combine_appetite_pts(
        user_level, meddra, pt2soc, pt2hlt, pt2hlgt
    )
    component_counts["percent_users"] = (100 * component_counts["n_users"] / n_users).round(2)
    print(component_counts.to_string(index=False))

    component_path = output_path(args.output_dir, "table_appetite_components", args.suffix)
    component_counts.to_csv(component_path, index=False)
    print(f"  Saved component counts: {component_path}")

    print("\nGenerating PT frequency table...")
    pt_by_soc, pt_counts_df, soc_counts_df = base.generate_pt_by_soc(
        combined_user_level,
        combined_meddra,
        pt2soc,
        min_pct=args.min_pct,
        min_soc_pct=args.min_soc_pct,
    )
    pt_path = output_path(args.output_dir, "table_pt_by_soc", args.suffix)
    pt_by_soc.to_csv(pt_path, index=False)
    print(f"  Saved: {pt_path} ({len(pt_by_soc):,} rows)")

    print("\nGenerating HLT and HLGT frequency tables...")
    hlt_counts, hlgt_counts = base.generate_hlt_hlgt(combined_user_level, combined_meddra)
    if args.min_hlgt_pct > 0:
        hlgt_counts = hlgt_counts[
            (100 * hlgt_counts["n_users"] / n_users) > args.min_hlgt_pct
        ].reset_index(drop=True)

    hlt_path = output_path(args.output_dir, "table_hlt_counts", args.suffix)
    hlgt_path = output_path(args.output_dir, "table_hlgt_counts", args.suffix)
    hlt_counts.to_csv(hlt_path, index=False)
    hlgt_counts.to_csv(hlgt_path, index=False)
    print(f"  Saved: {hlt_path} ({len(hlt_counts):,} rows)")
    print(f"  Saved: {hlgt_path} ({len(hlgt_counts):,} rows)")

    print("\nGenerating co-occurrence table...")
    cooccurrence = base.generate_cooccurrence(
        combined_user_level, combined_meddra, top_n=args.top_pairs
    )
    co_path = output_path(args.output_dir, "table_cooccurrence", args.suffix)
    cooccurrence.to_csv(co_path, index=False)
    print(f"  Saved: {co_path} ({len(cooccurrence):,} pairs)")

    print("\nTop 15 PTs after combining:")
    for _, row in pt_counts_df.head(15).iterrows():
        print(f"  {int(row['n_users']):>5} ({row['percent_users']:>5.1f}%)  {row['pt_term']}")

    print("\nTop 15 HLGTs after combining:")
    for _, row in hlgt_counts.head(15).iterrows():
        print(f"  {int(row['n_users']):>5} ({row['percent_users']:>5.1f}%)  {row['hlgt_term']}")


if __name__ == "__main__":
    main()
