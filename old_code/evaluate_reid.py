import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import seaborn as sns
from tqdm import tqdm
import argparse
import cv2
from PIL import Image
from torchvision import transforms
import random
from torch.nn import functional as F
import json
from datetime import datetime

# Import your existing modules
from metadata_dataset import WildlifeMetadataDataset
from ReIDNet import ReIDNet

def extract_embeddings(model, dataloader, device):
    """Extract embeddings for all images in the dataloader"""
    model.eval()
    embeddings = []
    labels = []
    image_paths = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            images, batch_labels = batch[:2]  # Your dataset returns (image, label)
            images = images.to(device)
            batch_embeddings = model(images)
            embeddings.append(batch_embeddings.cpu())
            labels.append(batch_labels)
            
            # Get image paths if available (modify if needed)
            if hasattr(dataloader.dataset, 'image_paths'):
                if isinstance(dataloader.dataset, Subset):
                    # Handle Subset case
                    indices = dataloader.dataset.indices
                    batch_paths = [dataloader.dataset.dataset.image_paths[i] for i in indices]
                else:
                    batch_paths = [dataloader.dataset.image_paths[i] for i in range(len(batch_labels))]
                image_paths.extend(batch_paths)
    
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    
    return embeddings, labels, image_paths

def calculate_distances(query_embeddings, gallery_embeddings):
    """Calculate distance matrix between query and gallery embeddings"""
    # Normalize embeddings
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
    gallery_embeddings = torch.nn.functional.normalize(gallery_embeddings, p=2, dim=1)
    
    # Calculate cosine similarity (convert to distance)
    similarity = torch.mm(query_embeddings, gallery_embeddings.t())
    distances = 1 - similarity
    
    return distances.numpy()

def calculate_cmc(distances, query_labels, gallery_labels, k=10):
    """Calculate CMC curve"""
    num_queries = distances.shape[0]
    cmc = np.zeros(k)
    
    for i in range(num_queries):
        # Get distance and label for this query
        dist = distances[i]
        query_label = query_labels[i]
        
        # Sort gallery by distance
        indices = np.argsort(dist)
        matches = (gallery_labels[indices] == query_label).astype(np.int32)
        
        # Calculate CMC
        if np.sum(matches) > 0:
            cmc_i = matches.cumsum()
            cmc_i = cmc_i / cmc_i[-1]  # Normalize
            cmc[:k] += cmc_i[:k]
    
    cmc = cmc / num_queries
    return cmc

def calculate_map(distances, query_labels, gallery_labels):
    """Calculate mean Average Precision"""
    num_queries = distances.shape[0]
    aps = []
    
    for i in range(num_queries):
        # Get distance and label for this query
        dist = distances[i]
        query_label = query_labels[i]
        
        # Sort gallery by distance
        indices = np.argsort(dist)
        matches = (gallery_labels[indices] == query_label).astype(np.int32)
        
        if np.sum(matches) == 0:
            continue
        
        # Calculate average precision
        cumsum = np.cumsum(matches)
        precisions = cumsum / (np.arange(len(matches)) + 1)
        ap = np.sum(precisions * matches) / np.sum(matches)
        aps.append(ap)
    
    if len(aps) == 0:
        return 0
    return np.mean(aps)

def plot_cmc(cmc, save_path):
    """Plot CMC curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(cmc)) + 1, cmc, marker='o')
    plt.xlabel('Rank')
    plt.ylabel('Matching Rate')
    plt.title('Cumulative Matching Characteristic (CMC) Curve')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'cmc_curve.png'))
    plt.close()

def plot_confusion_matrix(query_labels, predictions, class_names, save_path):
    """Plot confusion matrix"""
    # If too many classes, sample a subset
    if len(class_names) > 20:
        # Sample 20 classes
        sampled_classes = random.sample(class_names, 20)
        mask_query = np.isin(query_labels, sampled_classes)
        mask_pred = np.isin(predictions, sampled_classes)
        mask = mask_query & mask_pred
        
        query_labels_subset = query_labels[mask]
        predictions_subset = predictions[mask]
        class_names_subset = sampled_classes
    else:
        query_labels_subset = query_labels
        predictions_subset = predictions
        class_names_subset = class_names
    
    cm = confusion_matrix(query_labels_subset, predictions_subset)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names_subset, 
                yticklabels=class_names_subset)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Sampled Classes)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()

def visualize_tsne(embeddings, labels, save_path, n_samples=1000):
    """Visualize embeddings using t-SNE"""
    # Subsample if too many points
    if len(embeddings) > n_samples:
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings_subset = embeddings[indices]
        labels_subset = labels[indices]
    else:
        embeddings_subset = embeddings
        labels_subset = labels
    
    # Apply t-SNE
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_subset.numpy())
    
    # Plot
    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(labels_subset)
    
    # If too many unique labels, limit to top 20 most frequent
    if len(unique_labels) > 20:
        label_counts = {}
        for label in labels_subset:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        
        # Get top 20 most frequent labels
        top_labels = sorted(label_counts.keys(), key=lambda x: label_counts[x], reverse=True)[:20]
        mask = np.isin(labels_subset, top_labels)
        embeddings_2d = embeddings_2d[mask]
        labels_subset = labels_subset[mask]
        unique_labels = np.unique(labels_subset)
    
    # Use a colormap that can handle many classes
    cmap = plt.cm.get_cmap('tab20', len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        mask = labels_subset == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=[cmap(i)], label=f'ID {label}', alpha=0.7, s=50)
    
    # If too many classes, limit the legend
    if len(unique_labels) > 10:
        plt.legend(loc='best', ncol=2, fontsize='small', bbox_to_anchor=(1.05, 1))
    else:
        plt.legend(loc='best')
    
    plt.title('t-SNE Visualization of Embeddings')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'tsne_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_grad_cam(model, image_path, save_path, device, layer_name='backbone.blocks'):
    """Generate Grad-CAM visualization for a single image"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Get the specified layer
    target_layer = None
    for name, module in model.named_modules():
        if layer_name in name:
            target_layer = module
            break
    
    if target_layer is None:
        print(f"Layer {layer_name} not found. Available layers:")
        for name, _ in model.named_modules():
            print(name)
        return None
    
    # Register hooks
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    # Register hooks
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)
    
    # Forward pass
    model.eval()
    output = model(input_tensor)
    
    # Backward pass
    model.zero_grad()
    output.backward(torch.ones_like(output))
    
    # Remove hooks
    handle_forward.remove()
    handle_backward.remove()
    
    # Get activations and gradients
    activation = activations[0].cpu().detach()
    gradient = gradients[0].cpu().detach()
    
    # Calculate weights
    weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
    
    # Generate CAM
    cam = torch.sum(weights * activation, dim=1).squeeze()
    cam = F.relu(cam)
    cam = cam / (torch.max(cam) + 1e-7)
    cam = cam.numpy()
    
    # Resize CAM to image size
    cam = cv2.resize(cam, (image.width, image.height))
    
    # Convert image to numpy array
    img_np = np.array(image)
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Combine original image and heatmap
    superimposed = heatmap * 0.4 + img_np * 0.6
    superimposed = np.uint8(superimposed)
    
    # Save images
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img_np)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    ax[1].imshow(cam, cmap='jet')
    ax[1].set_title('Activation Map')
    ax[1].axis('off')
    
    ax[2].imshow(superimposed)
    ax[2].set_title('Grad-CAM')
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    
    return cam

def visualize_attention_maps(model, image_paths, save_path, device, num_samples=5):
    """Generate attention maps for a few sample images"""
    os.makedirs(os.path.join(save_path, 'attention_maps'), exist_ok=True)
    
    # Randomly select images
    if len(image_paths) > num_samples:
        sample_paths = random.sample(image_paths, num_samples)
    else:
        sample_paths = image_paths
    
    # Generate Grad-CAM for each sample
    for i, path in enumerate(sample_paths):
        output_path = os.path.join(save_path, 'attention_maps', f'gradcam_{i}.png')
        try:
            generate_grad_cam(model, path, output_path, device)
        except Exception as e:
            print(f"Error generating Grad-CAM for {path}: {e}")

def visualize_retrieval_results(query_paths, gallery_paths, query_labels, gallery_labels, 
                               distances, save_path, num_queries=5, top_k=5):
    """Visualize retrieval results for a few queries"""
    os.makedirs(os.path.join(save_path, 'retrieval_results'), exist_ok=True)
    
    # Randomly select queries
    if len(query_paths) > num_queries:
        query_indices = random.sample(range(len(query_paths)), num_queries)
    else:
        query_indices = range(len(query_paths))
    
    for idx in query_indices:
        query_path = query_paths[idx]
        query_label = query_labels[idx]
        
        # Get top-k gallery images
        dist = distances[idx]
        top_indices = np.argsort(dist)[:top_k]
        
        # Load images
        query_img = Image.open(query_path).convert('RGB')
        gallery_imgs = [Image.open(gallery_paths[i]).convert('RGB') for i in top_indices]
        gallery_labels_selected = [gallery_labels[i] for i in top_indices]
        
        # Create figure
        fig, axes = plt.subplots(1, top_k + 1, figsize=(15, 3))
        
        # Plot query
        axes[0].imshow(query_img)
        axes[0].set_title(f'Query\nID: {query_label}')
        axes[0].axis('off')
        
        # Plot gallery matches
        for i in range(top_k):
            axes[i+1].imshow(gallery_imgs[i])
            match = "✓" if gallery_labels_selected[i] == query_label else "✗"
            axes[i+1].set_title(f'Rank {i+1}\nID: {gallery_labels_selected[i]}\n{match}')
            axes[i+1].axis('off')
            
            # Add green/red border based on correct/incorrect match
            if gallery_labels_selected[i] == query_label:
                for spine in axes[i+1].spines.values():
                    spine.set_edgecolor('green')
                    spine.set_linewidth(5)
            else:
                for spine in axes[i+1].spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'retrieval_results', f'query_{idx}.png'), dpi=200)
        plt.close()

def evaluate_reid(model_path, gallery_loader, query_loader, device, save_path, backbone_name):
    """Evaluate ReID model performance"""
    # Load model
    model = ReIDNet(backbone_name=backbone_name, embedding_dim=512, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # Extract embeddings
    gallery_embeddings, gallery_labels, gallery_paths = extract_embeddings(model, gallery_loader, device)
    query_embeddings, query_labels, query_paths = extract_embeddings(model, query_loader, device)
    
    # Calculate distances
    distances = calculate_distances(query_embeddings, gallery_embeddings)
    
    # Calculate metrics
    cmc = calculate_cmc(distances, query_labels, gallery_labels, k=20)
    mAP = calculate_map(distances, query_labels, gallery_labels)
    
    # Get top-1 predictions
    predictions = gallery_labels[np.argmin(distances, axis=1)]
    
    # Plot results
    plot_cmc(cmc, save_path)
    
    # Get class names (if available)
    class_names = list(set(query_labels.numpy()))
    plot_confusion_matrix(query_labels, predictions, class_names, save_path)
    
    # Visualize embeddings with t-SNE
    visualize_tsne(query_embeddings, query_labels, save_path)
    
    # Generate attention maps
    visualize_attention_maps(model, query_paths, save_path, device)
    
    # Visualize retrieval results
    visualize_retrieval_results(query_paths, gallery_paths, query_labels, gallery_labels, 
                               distances, save_path)
    
    # Print metrics
    print(f"Rank-1: {cmc[0]:.4f}")
    print(f"Rank-5: {cmc[4]:.4f}")
    print(f"Rank-10: {cmc[9]:.4f}")
    print(f"mAP: {mAP:.4f}")
    
    # Save metrics
    metrics = {
        "rank1": float(cmc[0]),
        "rank5": float(cmc[4]),
        "rank10": float(cmc[9]),
        "mAP": float(mAP),
        "backbone": backbone_name,
        "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(save_path, 'reid_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return cmc, mAP

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ReID model")
    parser.add_argument("--remote", action="store_true", help="Use remote paths for data and model saving")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Backbone model name")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--resize", type=int, default=224, help="Image resize size")
    parser.add_argument("--n_test", type=int, default=5000, help="Number of test samples to use")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Directory to save results (defaults to model directory)")
    
    args = parser.parse_args()
    
    # Set paths based on remote flag
    if args.remote:
        DATA_ROOT = '/home/stud/aleks99/.cache/kagglehub/datasets/wildlifedatasets/wildlifereid-10k/versions/6'
    else:
        DATA_ROOT = 'C:/Users/trade/.cache/kagglehub/datasets/wildlifedatasets/wildlifereid-10k/versions/6'
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.model_path) + "/evaluation_results"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create transform
    transform = transforms.Compose([
        transforms.Resize((args.resize, args.resize)),
        transforms.ToTensor(),
    ])
    
    # Load test dataset
    test_dataset = WildlifeMetadataDataset(
        root_data=DATA_ROOT,
        split="test",
        transform=transform,
        n=args.n_test
    )
    
    # Split into gallery and query
    # For testing purposes, we'll use 80% as gallery and 20% as query
    from torch.utils.data import random_split
    gallery_size = int(len(test_dataset) * 0.8)
    query_size = len(test_dataset) - gallery_size
    gallery_dataset, query_dataset = random_split(test_dataset, [gallery_size, query_size])
    
    print(f"Gallery size: {gallery_size}, Query size: {query_size}")
    
    # Create dataloaders
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    query_loader = DataLoader(
        query_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate
    evaluate_reid(args.model_path, gallery_loader, query_loader, device, args.output_dir, args.backbone) 