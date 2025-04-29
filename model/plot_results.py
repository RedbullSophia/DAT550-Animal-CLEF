import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
from sklearn.manifold import TSNE
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

def load_model_metrics():
    """Load the model metrics from the CSV file"""
    metrics_path = os.path.join('model_data', 'all_model_metrics.csv')
    return pd.read_csv(metrics_path)

def plot_performance_heatmap():
    """Create a heatmap of model performance metrics"""
    df = load_model_metrics()
    
    # Define readable model names mapping
    model_name_mapping = {
        'resnet18basemodel': 'ResNet18 Base',
        'resnet18m3': 'ResNet18 (m=3)',
        'resnet18m6': 'ResNet18 (m=6)',
        'r50base': 'ResNet50 Base',
        'r50marg07': 'ResNet50 (margin=0.7)',
        'r50dr02': 'ResNet50 (dropout=0.2)'
    }
    
    # Select models and metrics for the heatmap
    selected_models = list(model_name_mapping.keys())
    
    # Filter data
    df_selected = df[df['filename'].isin(selected_models)]
    
    # Select metrics for the heatmap
    metrics = ['open_set_baks', 'open_set_baus', 'open_set_geometric_mean', 
               'final_train_loss', 'final_val_loss']
    
    # Create the heatmap data
    heatmap_data = df_selected[metrics].values
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data,
                annot=True,
                fmt='.3f',
                cmap='YlGnBu',
                xticklabels=['BAKS', 'BAUS', 'Geometric Mean', 'Train Loss', 'Val Loss'],
                yticklabels=[model_name_mapping[model] for model in selected_models],
                annot_kws={'size': 14})
    
    plt.title('Model Performance Heatmap', fontsize=18, pad=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('model_data/joint_plots/performance_heatmap.png')
    plt.close()

def plot_open_set_performance():
    """Plot the open-set recognition performance comparison"""
    # Load metrics
    df = load_model_metrics()
    
    # Define readable model names mapping
    model_name_mapping = {
        'resnet18basemodel': 'ResNet18 Base',
        'resnet18m3': 'ResNet18 (m=3)',
        'resnet18m6': 'ResNet18 (m=6)',
        'r50base': 'ResNet50 Base',
        'r50marg07': 'ResNet50 (margin=0.7)',
        'r50dr02': 'ResNet50 (dropout=0.2)'
    }
    
    # Select models to plot
    selected_models = list(model_name_mapping.keys())
    
    # Filter data for selected models
    df_selected = df[df['filename'].isin(selected_models)]
    
    # Extract metrics
    baks_scores = df_selected['open_set_baks']
    baus_scores = df_selected['open_set_baus']
    geometric_means = df_selected['open_set_geometric_mean']
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(selected_models))
    width = 0.25
    
    # Create bars with different colors
    plt.bar(x - width, baks_scores, width, label='BAKS', color='#3498db')
    plt.bar(x, baus_scores, width, label='BAUS', color='#e74c3c')
    plt.bar(x + width, geometric_means, width, label='Geometric Mean', color='#2ecc71')
    
    # Add value labels on top of bars
    for i, (baks, baus, gm) in enumerate(zip(baks_scores, baus_scores, geometric_means)):
        plt.text(i - width, baks + 0.01, f'{baks:.3f}', ha='center', va='bottom', fontsize=14)
        plt.text(i, baus + 0.01, f'{baus:.3f}', ha='center', va='bottom', fontsize=14)
        plt.text(i + width, gm + 0.01, f'{gm:.3f}', ha='center', va='bottom', fontsize=14)
    
    # Set y-axis limits to accommodate the text labels
    y_max = max(max(baks_scores), max(baus_scores), max(geometric_means))
    plt.ylim(0, y_max * 1.2)  # Add 20% padding to the top
    
    plt.ylabel('Score', fontsize=16)
    plt.title('Open-Set Recognition Performance Comparison', fontsize=18, pad=20)
    plt.xticks(x, [model_name_mapping[model] for model in selected_models], rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('model_data/joint_plots/performance_comparison.png')
    plt.close()

def plot_training_metrics():
    """Plot training and validation losses"""
    # Load metrics
    df = load_model_metrics()
    
    # Define readable model names mapping
    model_name_mapping = {
        'resnet18basemodel': 'ResNet18 Base',
        'resnet18m3': 'ResNet18 (m=3)',
        'resnet18m6': 'ResNet18 (m=6)',
        'r50base': 'ResNet50 Base',
        'r50marg07': 'ResNet50 (margin=0.7)',
        'r50dr02': 'ResNet50 (dropout=0.2)'
    }
    
    # Select models to plot
    selected_models = list(model_name_mapping.keys())
    
    # Filter data
    df_selected = df[df['filename'].isin(selected_models)]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(selected_models))
    width = 0.35
    
    plt.bar(x - width/2, df_selected['final_train_loss'], width, label='Training Loss', color='#3498db')
    plt.bar(x + width/2, df_selected['final_val_loss'], width, label='Validation Loss', color='#e74c3c')
    
    # Add value labels on top of bars
    for i, (train_loss, val_loss) in enumerate(zip(df_selected['final_train_loss'], df_selected['final_val_loss'])):
        plt.text(i - width/2, train_loss + 0.1, f'{train_loss:.2f}', ha='center', va='bottom', fontsize=14)
        plt.text(i + width/2, val_loss + 0.1, f'{val_loss:.2f}', ha='center', va='bottom', fontsize=14)
    
    # Set y-axis limits to accommodate the text labels
    y_max = max(max(df_selected['final_train_loss']), max(df_selected['final_val_loss']))
    plt.ylim(0, y_max * 1.2)  # Add 20% padding to the top
    
    plt.ylabel('Loss', fontsize=16)
    plt.title('Training and Validation Loss Comparison', fontsize=18, pad=20)
    plt.xticks(x, [model_name_mapping[model] for model in selected_models], rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('model_data/joint_plots/loss_comparison.png')
    plt.close()

def plot_hyperparameter_impact():
    """Plot the impact of different hyperparameters"""
    df = load_model_metrics()
    
    # Plot margin impact
    margin_models = ['resnet18margin01', 'resnet18margin05', 'r50marg06', 'r50marg07']
    df_margin = df[df['filename'].isin(margin_models)]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df_margin['margin'], df_margin['open_set_geometric_mean'], 
               s=100, alpha=0.6, c='#3498db')
    
    # Add labels for each point
    for i, row in df_margin.iterrows():
        plt.annotate(f"{row['filename']}\n({row['margin']})", 
                    (row['margin'], row['open_set_geometric_mean']),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    fontsize=14)
    
    plt.xlabel('Margin Value', fontsize=16)
    plt.ylabel('Geometric Mean', fontsize=16)
    plt.title('Impact of Margin Parameter on Performance', fontsize=18, pad=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('model_data/joint_plots/margin_impact.png')
    plt.close()

def create_training_plot_from_json(json_path, output_path):
    """Create a new training plot from JSON data with updated font sizes"""
    # Load the training metrics
    with open(json_path, 'r') as f:
        metrics = json.load(f)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot training and validation loss
    plt.plot(metrics['epochs'], metrics['train_loss'], label='Training Loss', marker='o', linewidth=2)
    plt.plot(metrics['epochs'], metrics['val_loss'], label='Validation Loss', marker='x', linewidth=2)
    
    # Add labels and title with updated font sizes
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Training and Validation Loss for ResNet50 (dropout=0.2)', fontsize=18, pad=20)
    
    # Update tick labels
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Add legend with updated font size
    plt.legend(fontsize=16)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_tsne_comparison():
    """Create and compare t-SNE plots from two different models"""
    # Define paths
    paths = {
        'ResNet18 Base': 'model_data/resnet18basemodel/open_set_evaluation/tsne_known_unknown.png',
        'ECA-NFNet-L1': 'old_model_data/2025-04-17_eca_nfnet_l1_arcface_batchz_32_embdim_512_e_30_imgprcls_4_pixels_288/open_set_evaluation/tsne_known_unknown.png'
    }
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot first t-SNE
    img1 = plt.imread(paths['ResNet18 Base'])
    ax1.imshow(img1)
    ax1.set_title('ResNet18 Base Model', fontsize=18, pad=20)
    ax1.axis('off')
    
    # Plot second t-SNE
    img2 = plt.imread(paths['ECA-NFNet-L1'])
    ax2.imshow(img2)
    ax2.set_title('ECA-NFNet-L1 Model', fontsize=18, pad=20)
    ax2.axis('off')
    
    # Add overall title
    fig.suptitle('t-SNE Visualization Comparison', fontsize=20, y=1.05)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('model_data/joint_plots/tsne_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    # Create plots
    plot_performance_heatmap()
    plot_open_set_performance()
    plot_training_metrics()
    plot_hyperparameter_impact()
    
    # Create new training plot with updated font sizes
    create_training_plot_from_json(
        'model_data/r50dr02/training_metrics.json',
        'model_data/joint_plots/r50dr02_training.png'
    )
    
    # Create t-SNE comparison plot
    plot_tsne_comparison()
    
    print("Plots have been generated and saved in the model_data/joint_plots directory")