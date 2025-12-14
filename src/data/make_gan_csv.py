import os
import pandas as pd

# Path to the original CSV inside the author repo
ORIG_CSV = "external/AI_Face_FairnessBench/metadata/train_data.csv"

# Output CSV
OUT_CSV = "data/train_gan_subset.csv"


GAN_ROOT = "" #path to dataset of images for training

# Map from keywords in original CSV to local folders
MODEL_MAP = {
    "AttGAN": "AttGAN",
    "MMD_GAN": "MMD_GAN_CelebA",
    "MSG_STYLE_GAN": "MSG_STYLE_GAN",
    "STARGAN": "STARGAN",
    "STGAN": "STGAN_CelebA",
    "STYLEGAN2": "StyleGAN2_FFHQ",
    "StyleGAN2": "StyleGAN2_FFHQ",
    "stylegan3": "stylegan3",
    "STYLEGAN": "STYLEGAN",
    "VQGAN": "taming_transformer-VQGAN",
    "taming_transformer": "taming_transformer-VQGAN",
}

def map_path(row):
    orig_path = str(row["Image Path"])
    fname = os.path.basename(orig_path)

    for key, folder in MODEL_MAP.items():
        if key in orig_path:
            new = os.path.join(GAN_ROOT, folder, fname)
            if os.path.exists(new):
                return new
    return None

def main():
    print("Loading:", ORIG_CSV)
    df = pd.read_csv(ORIG_CSV)

    print("Total rows:", len(df))

    df["Image Path"] = df.apply(map_path, axis=1)
    df = df[df["Image Path"].notnull()]

    print("Rows kept (GAN only):", len(df))

    os.makedirs("data", exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV)

if __name__ == "__main__":
    main()
                                                                                                                        
