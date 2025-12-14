import argparse
import os

import pandas as pd


def summarize_column(df, col_name, out_dir):
    counts = df[col_name].value_counts(dropna=False).sort_index()
    perc = df[col_name].value_counts(normalize=True, dropna=False).sort_index() * 100.0

    summary = pd.DataFrame({"count": counts, "percent": perc.round(2)})

    print(summary)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"generated_{col_name}_distribution.csv")
    summary.to_csv(out_path)
    print(f"[Saved] {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze demographics of generated faces (from CLIP annotations)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="generated/dcgan_gan_subset_clip_labels.csv",
        help="CSV produced by annotate_generated_clip.py",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/analysis_generated",
        help="Where to save summary CSVs",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    print(f"[analyze_generated_demographics] Loading {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"[analyze_generated_demographics] Total rows: {len(df)}")
    print(f"[analyze_generated_demographics] Columns: {list(df.columns)}")

    # Summarize gender (by index or prompt)
    if "GenderIdx" in df.columns:
        summarize_column(df, "GenderIdx", args.out_dir)
    if "GenderPrompt" in df.columns:
        summarize_column(df, "GenderPrompt", args.out_dir)

    # Summarize predicted skin tone
    if "SkinTonePred" in df.columns:
        summarize_column(df, "SkinTonePred", args.out_dir)
    elif "SkinPrompt" in df.columns:
        summarize_column(df, "SkinPrompt", args.out_dir)
    else:
        print("[WARN] No SkinTonePred or SkinPrompt columns found.")


if __name__ == "__main__":
    main()
