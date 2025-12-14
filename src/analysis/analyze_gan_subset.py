import argparse
import os

import pandas as pd


DEFAULT_CSV = "data/train_gan_subset.csv"
DEFAULT_OUT_DIR = "outputs/analysis"


# Helper to be flexible with column names
def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def summarize_column(df, col_name, out_dir):
    """
    Print and save counts and percentages for a single categorical column.
    """
    counts = df[col_name].value_counts(dropna=False).sort_index()
    perc = df[col_name].value_counts(normalize=True, dropna=False).sort_index() * 100.0

    summary = pd.DataFrame({"count": counts, "percent": perc.round(2)})

    print(summary)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{col_name}_distribution.csv")
    summary.to_csv(out_path)
    print(f"[Saved] {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze demographic distributions in GAN training subset"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_CSV,
        help="Path to train_gan_subset.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Directory to save summary CSVs",
    )

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    print(f"[analyze_gan_subset] Loading {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"[analyze_gan_subset] Total rows: {len(df)}")
    print(f"[analyze_gan_subset] Columns: {list(df.columns)}")

    # Try to locate likely demographic columns
    gender_col = find_column(df, ["Gender", "gender", "Gender Label", "gender_label"])
    age_col = find_column(df, ["Age", "age", "Age Group", "age_group"])
    skin_col = find_column(df, ["Skin Tone", "SkinTone", "skin_tone", "Skin_Tone"])
    intersec_col = find_column(
        df, ["Intersection", "Intersection Label", "intersection", "intersec_label"]
    )
    target_col = find_column(df, ["Target", "target", "Label", "label"])

    # Summarize any column we actually found
    if target_col:
        summarize_column(df, target_col, args.out_dir)
    if gender_col:
        summarize_column(df, gender_col, args.out_dir)
    if age_col:
        summarize_column(df, age_col, args.out_dir)
    if skin_col:
        summarize_column(df, skin_col, args.out_dir)
    if intersec_col:
        summarize_column(df, intersec_col, args.out_dir)

    if not any([target_col, gender_col, age_col, skin_col, intersec_col]):
        print(
            "[analyze_gan_subset] Warning: did not recognize any demographic columns. "
            "Check column names printed above."
        )


if __name__ == "__main__":
    main()
