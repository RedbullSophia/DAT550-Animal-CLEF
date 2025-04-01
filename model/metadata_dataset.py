import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class WildlifeMetadataDataset(Dataset):
    def __init__(self, root_data, transform=None, split="train", species_filter=None, dataset_filter=None, n=5000):
        """
        Args:
            root_data (str): Root path to dataset folder (should point to `versions/6`)
            transform (callable, optional): PyTorch transforms
            split (str): 'train', 'val', 'test' — filters metadata
            species_filter (str): e.g., "sea turtle"
            dataset_filter (str): e.g., "ZindiTurtleRecall"
        """
        from wildlife_datasets.datasets import WildlifeReID10k

        self.root_data = root_data
        self.transform = transform

        # Load metadata
        dataset = WildlifeReID10k(root_data, check_files=False)
        metadata = dataset.metadata
	if n < 120000
            metadata = metadata.sample(n=n, random_state=42)  # Use a smaller subset


        # Apply filters
        if split:
            metadata = metadata[metadata["split"] == split]
        if species_filter:
            metadata = metadata[metadata["species"] == species_filter]
        if dataset_filter:
            metadata = metadata[metadata["dataset"] == dataset_filter]

        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        label_counter = 0

        for _, row in metadata.iterrows():
            identity = row["identity"]
            rel_path = row["path"]
            full_path = os.path.join(root_data, rel_path)

            if identity not in self.class_to_idx:
                self.class_to_idx[identity] = label_counter
                label_counter += 1

            self.image_paths.append(full_path)
            self.labels.append(self.class_to_idx[identity])

        print(f"Loaded {len(self.image_paths)} images across {len(self.class_to_idx)} identities.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label
