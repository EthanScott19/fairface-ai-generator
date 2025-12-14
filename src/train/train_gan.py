import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.data.gan_image_dataset import GanImageDataset
from src.models.dcgan import DCGANGenerator, DCGANDiscriminator


def parse_args():
    parser = argparse.ArgumentParser(description="Train DCGAN on AI-Face GAN subset")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root directory containing GAN images")
    parser.add_argument("--out-dir", type=str, default="outputs/dcgan64",
                        help="Directory to save checkpoints and samples")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda or cpu")
    parser.add_argument("--sample-every", type=int, default=500,
                        help="Save sample grid every N iterations")
    parser.add_argument("--save-every-epoch", type=int, default=1,
                        help="Save checkpoint every N epochs")

    return parser.parse_args()


def make_out_dirs(out_dir: str):
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    sample_dir = os.path.join(out_dir, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    return ckpt_dir, sample_dir


def train():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[train_gan] Using device: {device}")

    ckpt_dir, sample_dir = make_out_dirs(args.out_dir)

    # Dataset & DataLoader
    dataset = GanImageDataset(args.data_root, image_size=args.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True,
    )

    # Models
    netG = DCGANGenerator(latent_dim=args.latent_dim).to(device)
    netD = DCGANDiscriminator().to(device)

    # Initialize weights
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    netG.apply(weights_init)
    netD.apply(weights_init)

    # Loss + optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr,
                            betas=(args.beta1, args.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr,
                            betas=(args.beta1, args.beta2))

    fixed_noise = torch.randn(64, args.latent_dim, device=device)

    global_step = 0

    for epoch in range(1, args.num_epochs + 1):
        for i, real_imgs in enumerate(dataloader):
            real_imgs = real_imgs.to(device)

            b_size = real_imgs.size(0)
            real_labels = torch.full((b_size,), 1.0, device=device)
            fake_labels = torch.full((b_size,), 0.0, device=device)
			#Discriminator
            netD.zero_grad()

            # Real
            output_real = netD(real_imgs)
            lossD_real = criterion(output_real, real_labels)

            # Fake
            noise = torch.randn(b_size, args.latent_dim, device=device)
            fake_imgs = netG(noise)
            output_fake = netD(fake_imgs.detach())
            lossD_fake = criterion(output_fake, fake_labels)

            lossD = lossD_real + lossD_fake
            lossD.backward()
            optimizerD.step()

			#Generator
            netG.zero_grad()
            output_for_G = netD(fake_imgs)
            lossG = criterion(output_for_G, real_labels)  # want fake -> real
            lossG.backward()
            optimizerG.step()

            if global_step % 50 == 0:
                print(
                    f"[Epoch {epoch}/{args.num_epochs}] "
                    f"[Batch {i}/{len(dataloader)}] "
                    f"[D loss: {lossD.item():.4f}] [G loss: {lossG.item():.4f}]"
                )

            # Save sample grid
            if global_step % args.sample_every == 0:
                with torch.no_grad():
                    fake_samples = netG(fixed_noise).detach().cpu()
                    # Un-normalize from [-1,1] ->  [0,1] 
                    fake_samples = (fake_samples + 1) / 2
                    sample_path = os.path.join(
                        sample_dir, f"epoch{epoch}_step{global_step}.png"
                    )
                    save_image(fake_samples, sample_path, nrow=8)
                    print(f"[Samples] Saved to {sample_path}")

            global_step += 1

        # Save checkpoint
        if epoch % args.save_every_epoch == 0:
            ckpt_path = os.path.join(
                ckpt_dir, f"dcgan_epoch{epoch}.pt"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "netG_state_dict": netG.state_dict(),
                    "netD_state_dict": netD.state_dict(),
                    "optimizerG_state_dict": optimizerG.state_dict(),
                    "optimizerD_state_dict": optimizerD.state_dict(),
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"[Checkpoint] Saved to {ckpt_path}")

    print("[train_gan] Training complete.")


if __name__ == "__main__":
    train()
