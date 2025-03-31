# Declarations
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from wildlife_datasets.datasets import WildlifeReID10k
from wildlife_datasets.splits import analyze_split
from wildlife_datasets.metrics import BAKS, BAUS


# Import function
# def importdata

# Define dataset path (update if necessary)

# Load the dataset
dataset = WildlifeReID10k('C:/Users/Morten/Documents/GitHub/DAT550-Animal-CLEF/Code/Dataset', check_files=False)
metadata = dataset.metadata


exit()


