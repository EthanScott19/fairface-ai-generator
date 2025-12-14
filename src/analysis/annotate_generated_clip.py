import argparse
import os

from tqdm import tqdm
import pandas as pd
from PIL import Image

import torch
import clip  # from OpenAI CLIP repo


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(path: str) -> bool:
    _, ext = os.path.splitext(path.lower())
    return ext in IMG_EXTENSIONS


def list_images(root_dir: str):
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if is_image_file(fname):
                paths.append(os.path.join(dirpath, fname))
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(
        description="Annotate generated faces with rough demographics using CLIP"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing generated PNG/JPG images",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="generated/gan_generated_clip_labels.csv",
        help="Output CSV for CLIP-based demographic labels",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run CLIP on: cuda or cpu",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    print(f"[annotate_generated_clip] Using device: {device}")

    image_paths = list_images(args.input_dir)
    if not image_paths:
        raise RuntimeError(f"No images found under {args.input_dir}")
    print(f"[annotate_generated_clip] Found {len(image_paths)} images")

    # Load CLIP model + preprocess
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # Gender prompts
    gender_prompts = [
        "a photo of a man",
        "a photo of a woman",
    ]
    gender_tokens = clip.tokenize(gender_prompts).to(device)

    # Skin tone prompts (approximate mapping to 1–10)
    skin_prompts = [
        "a person with very light skin",  # tone 1
        "a person with light skin",       # 2
        "a person with light beige skin", # 3
        "a person with beige skin",       # 4
        "a person with light brown skin", # 5
        "a person with medium brown skin",# 6
        "a person with dark brown skin",  # 7
        "a person with very dark brown skin", # 8
        "a person with almost black skin",    # 9
        "a person with very dark black skin", # 10
    ]
    skin_tokens = clip.tokenize(skin_prompts).to(device)

    results = []

    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Annotating images"):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[WARN] Failed to open {img_path}: {e}")
                continue

            img_tensor = preprocess(img).unsqueeze(0).to(device)

            # Gender
            img_features = model.encode_image(img_tensor)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)

            gender_text_features = model.encode_text(gender_tokens)
            gender_text_features = gender_text_features / gender_text_features.norm(
                dim=-1, keepdim=True
            )

            gender_logits = (100.0 * img_features @ gender_text_features.T).softmax(dim=-1)
            gender_probs = gender_logits[0].cpu().tolist()
            gender_idx = int(torch.argmax(gender_logits, dim=-1).cpu())
            gender_label = gender_prompts[gender_idx]

            # Skin tone
            skin_text_features = model.encode_text(skin_tokens)
            skin_text_features = skin_text_features / skin_text_features.norm(
                dim=-1, keepdim=True
            )

            skin_logits = (100.0 * img_features @ skin_text_features.T).softmax(dim=-1)
            skin_probs = skin_logits[0].cpu().tolist()
            skin_idx = int(torch.argmax(skin_logits, dim=-1).cpu())
            skin_label = skin_prompts[skin_idx]
            skin_tone_id = skin_idx + 1

            results.append(
                {
                    "Image Path": img_path,
                    "GenderPrompt": gender_label,
                    "GenderIdx": gender_idx,
                    "GenderProb_man": gender_probs[0],
                    "GenderProb_woman": gender_probs[1],
                    "SkinPrompt": skin_label,
                    "SkinTonePred": skin_tone_id,
                }
            )

    if not results:
        print("[annotate_generated_clip] No successful annotations.")
        return

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False)
    print(f"[annotate_generated_clip] Saved {len(df)} annotations to {args.out_csv}")


if __name__ == "__main__":
    main()
