import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_metric_learning.samplers import MPerClassSampler
from pytorch_metric_learning.losses import TripletMarginLoss

from metadata_dataset import WildlifeMetadataDataset
from ReIDNet import ReIDNet  # assuming you have this in your model folder

# ======================
# CONFIG
# ======================
DATA_ROOT = 'C:/Users/trade/.cache/kagglehub/datasets/wildlifedatasets/wildlifereid-10k/versions/6'
BATCH_SIZE = 32
EMBED_DIM = 512
NUM_EPOCHS = 10
LR = 1e-4
M = 4  # images per class
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# DATASET + DATALOADER
# ======================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset from metadata
train_dataset = WildlifeMetadataDataset(
    root_data=DATA_ROOT,
    split="train",
    transform=transform,
    species_filter=None,      # optional: "sea turtle"
    dataset_filter=None       # optional: "ZindiTurtleRecall"
)

# Use MPerClassSampler to get M samples per class in each batch
sampler = MPerClassSampler(train_dataset.labels, m=M, length_before_new_iter=len(train_dataset.labels))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=4,  # use 0 if debugging on Windows
    pin_memory=True,
    drop_last=True
)

# ======================
# MODEL, LOSS, OPTIMIZER
# ======================

model = ReIDNet(embedding_dim=EMBED_DIM).to(DEVICE)
loss_fn = TripletMarginLoss(margin=0.2)
optimizer = optim.Adam(model.parameters(), lr=LR)

# ======================
# TRAINING LOOP
# ======================
def train():
    for epoch in range(NUM_EPOCHS):
        print("Started epoch")
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            embeddings = model(images)
            loss = loss_fn(embeddings, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "reid_model.pth")
    print("✅ Model saved to reid_model.pth")

# ======================
# MAIN
# ======================
if __name__ == "__main__":
    train()
