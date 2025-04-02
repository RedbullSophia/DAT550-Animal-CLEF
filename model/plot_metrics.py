import os
import json
import argparse
import matplotlib.pyplot as plt

def plot_training_metrics(metrics_file, output_dir=None):
    """
    Plot training and validation metrics from a JSON file.
    
    Args:
        metrics_file: Path to the JSON file containing training metrics
        output_dir: Directory to save the plot (defaults to same directory as metrics file)
    """
    # Load metrics from file
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot training and validation loss
    plt.plot(metrics['epochs'], metrics['train_loss'], label='Training Loss', marker='o')
    plt.plot(metrics['epochs'], metrics['val_loss'], label='Validation Loss', marker='x')
    
    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Determine output path
    if output_dir is None:
        output_dir = os.path.dirname(metrics_file)
    
    output_path = os.path.join(output_dir, 'training_plot.png')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training metrics")
    parser.add_argument("--metrics_file", type=str, required=True, 
                        help="Path to the training_metrics.json file")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save the plot (defaults to same directory as metrics file)")
    
    args = parser.parse_args()
    plot_training_metrics(args.metrics_file, args.output_dir) 