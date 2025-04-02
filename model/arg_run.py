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
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ======================
# LOGGING SETUP
# ======================
def setup_logging(save_path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(save_path, "model_log.txt"))
        ]
    )

# ======================
# TRAINING FUNCTION
# ======================
def train(model, train_loader, val_loader, optimizer, loss_fn, scaler, device, save_path, num_epochs):
    best_loss = float("inf")
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    # For tracking metrics
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'epochs': []
    }

    for epoch in range(num_epochs):
        start_time = time.time()
        logging.info(f"Started epoch {epoch + 1}")
        
        # Training phase
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

        avg_train_loss = total_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                with autocast():
                    embeddings = model(images)
                    loss = loss_fn(embeddings, labels)
                
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Log metrics
        logging.info(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        logging.info(f" Epoch time: {time.time() - start_time:.2f} sec")
        
        # Store metrics for plotting
        metrics['train_loss'].append(avg_train_loss)
        metrics['val_loss'].append(avg_val_loss)
        metrics['epochs'].append(epoch + 1)
        
        # Save metrics to file
        import json
        with open(os.path.join(save_path, "training_metrics.json"), 'w') as f:
            json.dump(metrics, f)

        # Step the scheduler using validation loss
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_path, "trained_model.pth"))
            logging.info(f" Saved new best model at epoch {epoch+1}")

# ======================
# MAIN ENTRY POINT
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ArcFace ReID model")
    parser.add_argument("--remote", action="store_true", help="Use remote paths for data and model saving")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--m", type=int, default=2, help="Images per class (M)")
    parser.add_argument("--resize", type=int, default=160, help="Image resize size (pixels)")
    parser.add_argument("--n", type=int, default=1000, help="Number of training samples to use")
    parser.add_argument("--backbone", type=str, default="mobilenetv3_small_100", help="Backbone model name")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")

    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set paths based on remote flag
    if args.remote:
        save_path = f"/home/stud/aleks99/bhome/DAT550-Animal-CLEF/model_data/bb{args.backbone}_bz{args.batch_size}_e{args.num_epochs}_lr{args.lr}_m{args.m}_r{args.resize}_n{args.n}/"
        DATA_ROOT = '/home/stud/aleks99/.cache/kagglehub/datasets/wildlifedatasets/wildlifereid-10k/versions/6'
    else:
        save_path = f"model_data/bb{args.backbone}_bz{args.batch_size}_e{args.num_epochs}_lr{args.lr}_m{args.m}_r{args.resize}_n{args.n}/"
        DATA_ROOT = 'C:/Users/trade/.cache/kagglehub/datasets/wildlifedatasets/wildlifereid-10k/versions/6'
    
    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists

    setup_logging(save_path)  # Initialize logging with the dynamic path

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info(f"Using device: {DEVICE}")
    logging.info(f"Using {'remote' if args.remote else 'local'} paths")
    logging.info(f"Data root: {DATA_ROOT}")
    logging.info(f"Save path: {save_path}")
    
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

    transform = transforms.Compose([
        transforms.Resize((args.resize, args.resize)),
        transforms.ToTensor(),
    ])

    logging.info("Loading dataset...")
    # Load the full dataset
    full_dataset = WildlifeMetadataDataset(
        root_data=DATA_ROOT,
        split="train",
        transform=transform,
        n=args.n
    )
    
    # Split into train and validation
    from torch.utils.data import random_split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    logging.info(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    
    # Get labels for the training subset (needed for the sampler)
    train_indices = train_dataset.indices
    train_labels = [full_dataset.labels[i] for i in train_indices]
    
    # Create samplers and dataloaders
    train_sampler = MPerClassSampler(train_labels, m=args.m, length_before_new_iter=len(train_labels))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    model = ReIDNet(backbone_name=args.backbone, embedding_dim=512, device=DEVICE)
    num_classes = len(set(full_dataset.labels))
    loss_fn = ArcFaceLoss(num_classes=num_classes, embedding_size=512, margin=0.5, scale=64)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler()

    train(model, train_loader, val_loader, optimizer, loss_fn, scaler, DEVICE, save_path, args.num_epochs)
