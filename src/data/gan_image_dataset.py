import os
from typing import List

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(path: str) -> bool:
    _, ext = os.path.splitext(path.lower())
    return ext in IMG_EXTENSIONS


def list_images_recursive(root_dir: str) -> List[str]:
    img_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if is_image_file(fname):
                img_paths.append(os.path.join(dirpath, fname))
    return sorted(img_paths)


class GanImageDataset(Dataset):
    """
    Simple dataset that loads all images under a root directory.
    """

    def __init__(self, root_dir: str, image_size: int = 64):
        self.root_dir = root_dir
        self.image_paths = list_images_recursive(root_dir)

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found under {root_dir}")

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            # Normalize to [-1, 1]
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]),
        ])

        print(f"[GanImageDataset] Found {len(self.image_paths)} images under {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img
