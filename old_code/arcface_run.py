import os
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
        logging.StreamHandler(),  # Console
        logging.FileHandler("training.log")  # Log file
    ]
)

# ======================
# CONFIG
# ======================
N = 5000
DATA_ROOT = 'C:/Users/trade/.cache/kagglehub/datasets/wildlifedatasets/wildlifereid-10k/versions/6'
BATCH_SIZE = 1024
EMBED_DIM = 512
NUM_EPOCHS = 5
LR = 1e-3 #learning rate
M = 2  # images per class
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_PATH = f"reid_model_best_{run_id}.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# TRAINING FUNCTION
# ======================
def train(model, train_loader, optimizer, loss_fn, scaler, device):
    best_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        logging.info("Started epoch %d", epoch + 1)
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with autocast():
                embeddings = model(images)  # (batch_size, embed_dim)
                loss = loss_fn(embeddings, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logging.info(f"[Epoch {epoch+1}] ArcFace Loss: {avg_loss:.4f}")
        end_time = time.time()
        logging.info(f" Epoch time: {end_time - start_time:.2f} sec")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), SAVE_PATH)
            logging.info(f" Saved new best model at epoch {epoch+1}")


# ======================
# MAIN ENTRY POINT
# ======================
if __name__ == "__main__":
    logging.info(f"MAX batch size  = {get_max_batch_size()}")

    logging.info(f" Using device: {DEVICE}")
    if torch.cuda.is_available():
        logging.info(f" GPU: {torch.cuda.get_device_name(0)}")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    logging.info(f"Loading dataset")
    train_dataset = WildlifeMetadataDataset(
        root_data=DATA_ROOT,
        split="train",
        transform=transform,
        n= N
    )

    sampler = MPerClassSampler(train_dataset.labels, m=M, length_before_new_iter=len(train_dataset.labels))
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    print(f"Num batches: {len(train_loader)}")

    model = ReIDNet(backbone_name="mobilenetv3_small_100", embedding_dim=EMBED_DIM, device=DEVICE)

    # ✅ ArcFaceLoss: make sure labels are integer indices (0 to N-1)
    num_classes = len(set(train_dataset.labels))  # assumes integer labels
    loss_fn = ArcFaceLoss(
        num_classes=num_classes,
        embedding_size=EMBED_DIM,
        margin=0.5,
        scale=64
    )

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()

    train(model, train_loader, optimizer, loss_fn, scaler, DEVICE)
