import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
from pytorch_metric_learning.losses import ArcFaceLoss
from pytorch_metric_learning.samplers import MPerClassSampler
from datetime import datetime
from utils import get_max_batch_size
from metadata_dataset import WildlifeMetadataDataset
from ReIDNet import ReIDNet
import logging
import time

# ======================
# LOGGING SETUP
# ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)

# ======================
# TRAINING FUNCTION
# ======================
def train(model, train_loader, optimizer, loss_fn, scaler, device, save_path, num_epochs):
    best_loss = float("inf")

    for epoch in range(num_epochs):
        start_time = time.time()
        logging.info(f"Started epoch {epoch + 1}")
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with autocast():
                embeddings = model(images)
                loss = loss_fn(embeddings, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logging.info(f"[Epoch {epoch+1}] ArcFace Loss: {avg_loss:.4f}")
        logging.info(f" Epoch time: {time.time() - start_time:.2f} sec")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            logging.info(f" Saved new best model at epoch {epoch+1}")


# ======================
# MAIN ENTRY POINT
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ArcFace ReID model")

    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--m", type=int, default=2, help="Images per class (M)")
    parser.add_argument("--resize", type=int, default=128, help="Image resize size (pixels)")
    parser.add_argument("--n", type=int, default=5000, help="Number of training samples to use")

    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"model_data/reid_model_best_{run_id}.pth"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_ROOT = 'C:/Users/trade/.cache/kagglehub/datasets/wildlifedatasets/wildlifereid-10k/versions/6'

    logging.info(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

    transform = transforms.Compose([
        transforms.Resize((args.resize, args.resize)),
        transforms.ToTensor(),
    ])

    logging.info("Loading dataset...")
    train_dataset = WildlifeMetadataDataset(
        root_data=DATA_ROOT,
        split="train",
        transform=transform,
        n=args.n
    )

    sampler = MPerClassSampler(train_dataset.labels, m=args.m, length_before_new_iter=len(train_dataset.labels))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    model = ReIDNet(backbone_name="mobilenetv3_small_100", embedding_dim=512, device=DEVICE)
    num_classes = len(set(train_dataset.labels))
    loss_fn = ArcFaceLoss(num_classes=num_classes, embedding_size=512, margin=0.5, scale=64)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    train(model, train_loader, optimizer, loss_fn, scaler, DEVICE, save_path, args.num_epochs)
