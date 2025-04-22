import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.manifold import TSNE
import seaborn as sns
from tqdm import tqdm
import argparse
from PIL import Image
from torchvision import transforms
import random
import json
from datetime import datetime
from collections import defaultdict

# Import your existing modules
from metadata_dataset import WildlifeMetadataDataset
from ReIDNet import ReIDNet

# Define OpenSetWildlifeDataset class at module level
class OpenSetWildlifeDataset(WildlifeMetadataDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_identities = []
        
        # Reload metadata to get original identities
        from wildlife_datasets.datasets import WildlifeReID10k
        dataset = WildlifeReID10k(self.root_data, check_files=False)
        metadata = dataset.metadata
        
        # Apply the same filters as in the parent class
        if kwargs.get('split'):
            metadata = metadata[metadata["split"] == kwargs.get('split')]
        if kwargs.get('species_filter'):
            metadata = metadata[metadata["species"] == kwargs.get('species_filter')]
        if kwargs.get('dataset_filter'):
            metadata = metadata[metadata["dataset"] == kwargs.get('dataset_filter')]
        
        if kwargs.get('n', 100000) < 100000:
            metadata = metadata.sample(n=kwargs.get('n'), random_state=42)
        
        # Store original identities in the same order as image_paths
        for _, row in metadata.iterrows():
            self.original_identities.append(row["identity"])

# Import evaluation metrics
def BAKS(y_true, y_pred, identity_test_only):
    """Computes BAKS (balanced accuracy on known samples)."""
    # Need to keep the object type due to mixed arrays
    y_true = np.array(y_true, dtype=object)
    y_pred = np.array(y_pred, dtype=object)
    identity_test_only = np.array(identity_test_only, dtype=object)

    # Remove data in identity_test_only
    idx = np.where(~np.isin(y_true, identity_test_only))[0]
    y_true_idx = y_true[idx]
    y_pred_idx = y_pred[idx]
    if len(y_true_idx) == 0:
        return np.nan
    
    df = pd.DataFrame({'y_true': y_true_idx, 'y_pred': y_pred_idx})

    # Compute the balanced accuracy
    accuracy = 0
    for _, df_identity in df.groupby('y_true'):
        accuracy += 1 / df['y_true'].nunique() * np.mean(df_identity['y_pred'] == df_identity['y_true'])
    return round(float(accuracy), 4)

def BAUS(y_true, y_pred, identity_test_only, new_class):
    """Computes BAUS (balanced accuracy on unknown samples)."""
    # Need to keep the object type due to mixed arrays
    y_true = np.array(y_true, dtype=object)
    y_pred = np.array(y_pred, dtype=object)
    identity_test_only = np.array(identity_test_only, dtype=object)

    # Remove data not in identity_test_only
    idx = np.where(np.isin(y_true, identity_test_only))[0]
    y_true_idx = y_true[idx]
    y_pred_idx = y_pred[idx]
    if len(y_true_idx) == 0:
        return np.nan

    df = pd.DataFrame({'y_true': y_true_idx, 'y_pred': y_pred_idx})

    # Compute the balanced accuracy
    accuracy = 0
    for _, df_identity in df.groupby('y_true'):
        accuracy += 1 / df['y_true'].nunique() * np.mean(df_identity['y_pred'] == new_class)
    return round(float(accuracy), 4)

def extract_embeddings(model, dataloader, device):
    """Extract embeddings for all images in the dataloader"""
    model.eval()
    embeddings = []
    labels = []
    image_paths = []
    original_identities = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            images, batch_labels = batch[:2]  # Your dataset returns (image, label)
            images = images.to(device)
            batch_embeddings = model(images)
            embeddings.append(batch_embeddings.cpu())
            labels.append(batch_labels)
            
            # Get image paths and original identities if available
            if hasattr(dataloader.dataset, 'image_paths'):
                if isinstance(dataloader.dataset, Subset):
                    # Handle Subset case
                    indices = dataloader.dataset.indices
                    batch_paths = [dataloader.dataset.dataset.image_paths[i] for i in range(len(batch_labels))]
                    batch_identities = [dataloader.dataset.dataset.original_identities[i] for i in range(len(batch_labels))]
                else:
                    batch_paths = [dataloader.dataset.image_paths[i] for i in range(len(batch_labels))]
                    batch_identities = [dataloader.dataset.original_identities[i] for i in range(len(batch_labels))]
                image_paths.extend(batch_paths)
                original_identities.extend(batch_identities)
    
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    
    return embeddings, labels, image_paths, original_identities

def calculate_distances(query_embeddings, gallery_embeddings):
    """Calculate distance matrix between query and gallery embeddings"""
    # Normalize embeddings
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
    gallery_embeddings = torch.nn.functional.normalize(gallery_embeddings, p=2, dim=1)
    
    # Calculate cosine similarity (convert to distance)
    similarity = torch.mm(query_embeddings, gallery_embeddings.t())
    distances = 1 - similarity
    
    return distances.numpy()

def find_optimal_threshold(distances, query_original_ids, gallery_original_ids, identity_test_only):
    """Find the optimal threshold for separating known and unknown individuals"""
    thresholds = np.linspace(0.1, 0.9, 50)
    best_threshold = 0.5  # Default
    best_score = 0
    
    results = []
    
    for threshold in thresholds:
        # For each query, predict identity or "new_individual"
        predictions = []
        for i, dist_row in enumerate(distances):
            min_idx = np.argmin(dist_row)
            min_dist = dist_row[min_idx]
            
            if min_dist > threshold:
                # Distance is too large, predict as unknown
                predictions.append("new_individual")
            else:
                # Predict as the closest gallery identity
                predictions.append(gallery_original_ids[min_idx])
        
        # Calculate BAKS and BAUS
        baks_score = BAKS(query_original_ids, predictions, identity_test_only)
        baus_score = BAUS(query_original_ids, predictions, identity_test_only, "new_individual")
        
        # Handle NaN values
        if np.isnan(baks_score):
            baks_score = 0
        if np.isnan(baus_score):
            baus_score = 0
        
        # Calculate geometric mean
        geometric_mean = round(float(np.sqrt(baks_score * baus_score)), 4)
        
        results.append({
            'threshold': round(float(threshold), 4),
            'baks': baks_score,
            'baus': baus_score,
            'geometric_mean': geometric_mean
        })
        
        if geometric_mean > best_score:
            best_score = geometric_mean
            best_threshold = round(float(threshold), 4)
    
    # Create a plot of threshold vs. metrics
    results_df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['threshold'], results_df['baks'], label='BAKS')
    plt.plot(results_df['threshold'], results_df['baus'], label='BAUS')
    plt.plot(results_df['threshold'], results_df['geometric_mean'], label='Geometric Mean')
    plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best Threshold: {best_threshold:.4f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Threshold Optimization')
    plt.legend()
    plt.grid(True)
    
    return best_threshold, results_df

def visualize_tsne_with_known_unknown(embeddings, original_ids, identity_test_only, save_path, n_samples=1000):
    """Visualize embeddings using t-SNE, highlighting known vs unknown individuals"""
    # Subsample if too many points
    if len(embeddings) > n_samples:
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings_subset = embeddings[indices]
        ids_subset = [original_ids[i] for i in indices]
    else:
        embeddings_subset = embeddings
        ids_subset = original_ids
    
    # Create known/unknown labels
    known_unknown_labels = ["Unknown" if id in identity_test_only else "Known" for id in ids_subset]
    
    # Apply t-SNE
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_subset.numpy())
    
    # Plot with known/unknown coloring
    plt.figure(figsize=(12, 10))
    
    # Plot known and unknown points with different colors
    known_mask = np.array(known_unknown_labels) == "Known"
    unknown_mask = np.array(known_unknown_labels) == "Unknown"
    
    plt.scatter(embeddings_2d[known_mask, 0], embeddings_2d[known_mask, 1], 
               c='blue', label='Known Individuals', alpha=0.7, s=50)
    plt.scatter(embeddings_2d[unknown_mask, 0], embeddings_2d[unknown_mask, 1], 
               c='red', label='Unknown Individuals', alpha=0.7, s=50)
    
    plt.title('t-SNE Visualization of Embeddings (Known vs Unknown)')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'tsne_known_unknown.png'), dpi=200)
    plt.close()
    
    # Also create a plot colored by individual identity (limited to top 20 most frequent)
    # plt.figure(figsize=(12, 10))
    
    # # Count occurrences of each identity
    # id_counts = {}
    # for id in ids_subset:
    #     if id in id_counts:
    #         id_counts[id] += 1
    #     else:
    #         id_counts[id] = 1
    
    # # Get top 20 most frequent identities
    # top_ids = sorted(id_counts.keys(), key=lambda x: id_counts[x], reverse=True)[:20]
    
    # # Create a colormap
    # cmap = plt.cm.get_cmap('tab20', len(top_ids))
    
    # # Plot points for each of the top identities
    # for i, id in enumerate(top_ids):
    #     mask = np.array([x == id for x in ids_subset])
    #     plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
    #                c=[cmap(i)], label=f'ID: {id}', alpha=0.7, s=50)
    
    # plt.title('t-SNE Visualization of Embeddings (Top 20 Identities)')
    # plt.legend(loc='best', ncol=2, fontsize='small', bbox_to_anchor=(1.05, 1))
    # plt.savefig(os.path.join(save_path, 'tsne_top_identities.png'), dpi=200)
    # plt.close()

def visualize_distance_distributions(distances, query_original_ids, gallery_original_ids, identity_test_only, save_path):
    """Visualize the distribution of distances for known and unknown individuals"""
    known_distances = []
    unknown_distances = []
    
    for i, dist_row in enumerate(distances):
        min_idx = np.argmin(dist_row)
        min_dist = dist_row[min_idx]
        
        if query_original_ids[i] in identity_test_only:
            unknown_distances.append(min_dist)
        else:
            known_distances.append(min_dist)
    
    plt.figure(figsize=(10, 6))
    plt.hist(known_distances, bins=50, alpha=0.5, label='Known Individuals')
    plt.hist(unknown_distances, bins=50, alpha=0.5, label='Unknown Individuals')
    plt.xlabel('Minimum Distance')
    plt.ylabel('Count')
    plt.title('Distribution of Minimum Distances')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'distance_distributions.png'), dpi=200)
    plt.close()

def create_sample_submission(dataset_query, predictions, file_name='sample_submission.csv'):
    """Create a submission CSV file with image IDs and predicted identities"""
    # Extract image IDs from the paths
    image_ids = [os.path.basename(path).split('.')[0] for path in dataset_query.image_paths]

    df = pd.DataFrame({
        'image_id': image_ids,
        'identity': predictions
    })
    df.to_csv(file_name, index=False)

def evaluate_open_set(model_path, gallery_loader, query_loader, device, save_path, backbone_name, embedding_dim, threshold=None, loss_type="arcface"):
    """Evaluate ReID model for open-set recognition"""
    # Load model
    model = ReIDNet(backbone_name=backbone_name, embedding_dim=embedding_dim, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # Extract embeddings
    gallery_embeddings, gallery_labels, gallery_paths, gallery_original_ids = extract_embeddings(model, gallery_loader, device)
    query_embeddings, query_labels, query_paths, query_original_ids = extract_embeddings(model, query_loader, device)
    
    # Calculate distances
    distances = calculate_distances(query_embeddings, gallery_embeddings)
    
    # Identify test-only identities (unknown individuals)
    identity_test_only = list(set(query_original_ids) - set(gallery_original_ids))
    print(f"Found {len(identity_test_only)} unknown identities in the query set")
    
    # Visualize embeddings with t-SNE
    visualize_tsne_with_known_unknown(query_embeddings, query_original_ids, identity_test_only, save_path)
    
    # Visualize distance distributions
    visualize_distance_distributions(distances, query_original_ids, gallery_original_ids, identity_test_only, save_path)
    
    # Find optimal threshold if not provided
    if threshold is None:
        threshold, threshold_results = find_optimal_threshold(
            distances, query_original_ids, gallery_original_ids, identity_test_only)
        
        # Save threshold results
        threshold_results.to_csv(os.path.join(save_path, 'threshold_results.csv'), index=False)
    
    print(f"Using threshold: {threshold}")
    
    # Make predictions using the threshold
    predictions = []
    for i, dist_row in enumerate(distances):
        min_idx = np.argmin(dist_row)
        min_dist = dist_row[min_idx]
        
        if min_dist > threshold:
            # Distance is too large, predict as unknown
            predictions.append("new_individual")
        else:
            # Predict as the closest gallery identity
            predictions.append(gallery_original_ids[min_idx])
    
    # Create sample submission file
    create_sample_submission(query_loader.dataset, predictions, os.path.join(save_path, 'sample_submission.csv'))
    
    # Calculate BAKS and BAUS
    baks_score = BAKS(query_original_ids, predictions, identity_test_only)
    baus_score = BAUS(query_original_ids, predictions, identity_test_only, "new_individual")
    
    # Calculate geometric mean
    if np.isnan(baks_score):
        baks_score = 0
    if np.isnan(baus_score):
        baus_score = 0
    
    geometric_mean = round(float(np.sqrt(baks_score * baus_score)), 4)
    
    # Print metrics
    print(f"BAKS (Known Accuracy): {baks_score:.4f}")
    print(f"BAUS (Unknown Accuracy): {baus_score:.4f}")
    print(f"Geometric Mean: {geometric_mean:.4f}")
    
    # Create confusion matrix for known vs unknown
    y_true_known = ["Known" if id not in identity_test_only else "Unknown" for id in query_original_ids]
    y_pred_known = ["Known" if pred != "new_individual" else "Unknown" for pred in predictions]
    
    # Create confusion matrix
    cm = pd.crosstab(
        pd.Series(y_true_known, name='True'),
        pd.Series(y_pred_known, name='Predicted')
    )
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Known vs Unknown Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'known_unknown_confusion.png'))
    plt.close()
    
    # Load existing metrics CSV
    metrics_csv_path = os.path.join('model_data', 'all_model_metrics.csv')
    if not os.path.exists(metrics_csv_path):
        df = pd.DataFrame(columns=[
            'filename',
            'final_train_loss',
            'final_val_loss',
            'backbone',
            'batch_size',
            'num_epochs',
            'learning_rate',
            'm',
            'resize',
            'n',
            'val_split',
            'weight_decay',
            'dropout',
            'scheduler',
            'patience',
            'augmentation',
            'embedding_dim',
            'margin',
            'scale',
            'loss_type',
            'open_set_baks',
            'open_set_baus',
            'open_set_geometric_mean',
            'open_set_threshold',
            'open_set_evaluation_time',
            'diff_final_train_loss',
            'diff_final_val_loss',
            'diff_open_set_baks',
            'diff_open_set_baus',
            'diff_open_set_geometric_mean',
            'diff_open_set_threshold'
        ])
        df.to_csv(metrics_csv_path, index=False)
        print(f"Created new metrics CSV at {metrics_csv_path}")
    else:
        df = pd.read_csv(metrics_csv_path)
    
    # Find the row with matching filename
    model_dir = os.path.basename(os.path.dirname(save_path))
    row_idx = df[df['filename'] == model_dir].index
    
    if len(row_idx) > 0:
        # Update the row with open set evaluation metrics
        df.loc[row_idx[0], 'open_set_baks'] = baks_score
        df.loc[row_idx[0], 'open_set_baus'] = baus_score
        df.loc[row_idx[0], 'open_set_geometric_mean'] = geometric_mean
        df.loc[row_idx[0], 'open_set_threshold'] = threshold
        df.loc[row_idx[0], 'open_set_evaluation_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Update difference columns if reference model is provided
        if args.reference_model:
            ref_row = df[df['filename'] == args.reference_model]
            if not ref_row.empty:
                ref_metrics = ref_row.iloc[0]
                if not pd.isna(ref_metrics['open_set_baks']):
                    df.loc[row_idx[0], 'diff_open_set_baks'] = round(float(baks_score - ref_metrics['open_set_baks']), 4)
                    df.loc[row_idx[0], 'diff_open_set_baus'] = round(float(baus_score - ref_metrics['open_set_baus']), 4)
                    df.loc[row_idx[0], 'diff_open_set_geometric_mean'] = round(float(geometric_mean - ref_metrics['open_set_geometric_mean']), 4)
                    df.loc[row_idx[0], 'diff_open_set_threshold'] = round(float(threshold - ref_metrics['open_set_threshold']), 4)
    else:
        # Create new row if model doesn't exist in CSV
        new_row = {
            'filename': model_dir,
            'open_set_baks': baks_score,
            'open_set_baus': baus_score,
            'open_set_geometric_mean': geometric_mean,
            'open_set_threshold': threshold,
            'open_set_evaluation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add difference columns if reference model is provided
        if args.reference_model:
            ref_row = df[df['filename'] == args.reference_model]
            if not ref_row.empty:
                ref_metrics = ref_row.iloc[0]
                if not pd.isna(ref_metrics['open_set_baks']):
                    new_row['diff_open_set_baks'] = round(float(baks_score - ref_metrics['open_set_baks']), 4)
                    new_row['diff_open_set_baus'] = round(float(baus_score - ref_metrics['open_set_baus']), 4)
                    new_row['diff_open_set_geometric_mean'] = round(float(geometric_mean - ref_metrics['open_set_geometric_mean']), 4)
                    new_row['diff_open_set_threshold'] = round(float(threshold - ref_metrics['open_set_threshold']), 4)
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save updated metrics
    df.to_csv(metrics_csv_path, index=False)
    print(f"Updated metrics saved to {metrics_csv_path}")
    
    return baks_score, baus_score, geometric_mean

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ReID model for open-set recognition")
    parser.add_argument("--remote", action="store_true", help="Use remote paths for data and model saving")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Backbone model name")
    parser.add_argument("--embedding_dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--resize", type=int, default=224, help="Image resize size")
    parser.add_argument("--threshold", type=float, default=None, help="Distance threshold for unknown detection")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Directory to save results (defaults to model directory)")
    parser.add_argument("--loss_type", type=str, default="arcface", 
                        choices=["arcface", "triplet", "contrastive", "multisimilarity", "cosface"], 
                        help="Loss function used for training")
    parser.add_argument("--reference_model", type=str, help="Filename of the reference model to compare against")
    
    args = parser.parse_args()
    
    # Set paths based on remote flag
    if args.remote:
        DATA_ROOT = '/home/stud/aleks99/.cache/kagglehub/datasets/wildlifedatasets/wildlifereid-10k/versions/6'
    else:
        DATA_ROOT = 'C:/Users/trade/.cache/kagglehub/datasets/wildlifedatasets/wildlifereid-10k/versions/6'
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.model_path) + "/open_set_evaluation"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create transform - simplified for evaluation
    transform = transforms.Compose([
        # Basic preprocessing only
        transforms.Resize((args.resize, args.resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.486, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    # For gallery, use training set (known identities)
    gallery_dataset = OpenSetWildlifeDataset(
        root_data=DATA_ROOT,
        split="train",
        transform=transform
        # Remove n parameter to use all available training images
    )
    
    # For query, use test set (mix of known and unknown identities)
    query_dataset = OpenSetWildlifeDataset(
        root_data=DATA_ROOT,
        split="test",
        transform=transform
        # Remove n parameter to use all available test images
    )
    
    print(f"Gallery size: {len(gallery_dataset)}, Query size: {len(query_dataset)}")
    
    # Create dataloaders
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    query_loader = DataLoader(
        query_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    # Evaluate
    evaluate_open_set(args.model_path, gallery_loader, query_loader, device, 
                     args.output_dir, args.backbone, args.embedding_dim, args.threshold, args.loss_type) 