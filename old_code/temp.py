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
print(DEVICE)