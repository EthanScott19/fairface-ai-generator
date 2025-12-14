import argparse
import os

import torch
from torchvision.utils import save_image

from src.models.dcgan import DCGANGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples from trained DCGAN")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint saved by train_gan.py")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-dir", type=str, default="generated_samples")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[generate_samples] Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    latent_dim = ckpt["args"]["latent_dim"]

    netG = DCGANGenerator(latent_dim=latent_dim).to(device)
    netG.load_state_dict(ckpt["netG_state_dict"])
    netG.eval()

    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    sample_idx = 0

    with torch.no_grad():
        for b in range(num_batches):
            cur_bs = min(args.batch_size, args.num_samples - sample_idx)
            noise = torch.randn(cur_bs, latent_dim, device=device)
            fake_imgs = netG(noise)  # in [-1, 1]
            fake_imgs = (fake_imgs + 1) / 2  # [0, 1]

            for i in range(cur_bs):
                img = fake_imgs[i:i+1]
                out_path = os.path.join(args.out_dir, f"sample_{sample_idx:05d}.png")
                save_image(img, out_path)
                sample_idx += 1

    print(f"[generate_samples] Saved {sample_idx} images to {args.out_dir}")


if __name__ == "__main__":
    main()
