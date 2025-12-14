from torch.utils.data import DataLoader
from torchvision import transforms

from external.AI_Face_FairnessBench.training.dataset.datasets_train import ImageDataset_Train

CSV_PATH = "data/train_gan_subset.csv"

def main():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = ImageDataset_Train(csv_file=CSV_PATH, owntransforms=transform)
    print("Dataset size:", len(dataset))

    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    batch = next(iter(loader))

    imgs = batch["image"]
    labels = batch["label"]
    intersec = batch["intersec_label"]

    print("Batch image tensor shape:", imgs.shape)
    print("Example label:", labels[0])
    print("Intersection label example:", intersec[0])

if __name__ == "__main__":
    main()
