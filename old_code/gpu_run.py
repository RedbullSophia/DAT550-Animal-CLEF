import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.samplers import MPerClassSampler

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

DATA_ROOT = 'C:/Users/trade/.cache/kagglehub/datasets/wildlifedatasets/wildlifereid-10k/versions/6'
BATCH_SIZE = 2048
EMBED_DIM = 512
NUM_EPOCHS = 5
LR = 1e-4
M = 2  # images per class
SAVE_PATH = "reid_model_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, optimizer, loss_fn, scaler):
    # assert model.device.type == "cuda", " Model is not on GPU!"
    # for name, param in model.named_parameters():
    #     print(f"{name} → {param.device}")
    #     break 
    is_on_gpu = next(model.parameters()).is_cuda
    print("Model is on GPU:", is_on_gpu)

    best_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        
        start_time = time.time()
        logging.info("Started epoch %d", epoch + 1)
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            with autocast():
                embeddings = model(images)
                loss = loss_fn(embeddings, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
        print("Allocated VRAM (MB):", torch.cuda.memory_allocated() / 1024 / 1024)
        print("Max Allocated:", torch.cuda.max_memory_allocated() / 1024 / 1024)
        avg_loss = total_loss / len(train_loader)
        logging.info(f" Epoch [{epoch+1}] — Loss: {avg_loss:.4f}")
        end_time = time.time()
        logging.info(f" Epoch time: {end_time - start_time:.2f} sec")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), SAVE_PATH)
            logging.info(f" New best model saved at epoch {epoch+1} with loss {avg_loss:.4f}")
        

# ======================
# MAIN ENTRY POINT
# ======================
if __name__ == "__main__":
    logging.info(f"MAX batch size  = {get_max_batch_size()}")

    logging.info(f" Using device: {DEVICE}")
    if torch.cuda.is_available():
        logging.info(f" GPU: {torch.cuda.get_device_name(0)}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = WildlifeMetadataDataset(
        root_data=DATA_ROOT,
        split="train",
        transform=transform
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

    model = ReIDNet(device=DEVICE)
    loss_fn = TripletMarginLoss(margin=0.2)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()  # mixed precision

    train(model, train_loader, optimizer, loss_fn, scaler)
