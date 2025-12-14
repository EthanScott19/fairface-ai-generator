import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

TRAIN_CSV = "outputs/analysis/Skin Tone_distribution.csv"
GEN_CSV = "outputs/analysis_generated/generated_SkinTonePred_distribution.csv"
OUT_DIR = "outputs/figures"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    train_df = pd.read_csv(TRAIN_CSV, index_col=0)
    # The index is the skin tone ID (1..10), columns: count, percent
    train_df.index = train_df.index.astype(int)
    train_df = train_df.sort_index()

    gen_df = pd.read_csv(GEN_CSV, index_col=0)
    gen_df.index = gen_df.index.astype(int)
    gen_df = gen_df.sort_index()

    all_tones = sorted(set(train_df.index).union(set(gen_df.index)))
    train_perc = train_df.reindex(all_tones).fillna(0)["percent"].values
    gen_perc = gen_df.reindex(all_tones).fillna(0)["percent"].values

    tones = np.array(all_tones)

    x = np.arange(len(tones))
    width = 0.35 

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, train_perc, width, label="Training (GAN subset)")
    ax.bar(x + width/2, gen_perc, width, label="Generated (DCGAN)")

    ax.set_xlabel("Skin Tone ID")
    ax.set_ylabel("Percentage of images (%)")
    ax.set_title("Skin Tone Distribution: Training vs Generated")
    ax.set_xticks(x)
    ax.set_xticklabels(tones)
    ax.legend()

    ax.grid(axis="y", alpha=0.3)

    out_path = os.path.join(OUT_DIR, "skin_tone_train_vs_generated.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
