import os
import json
import pandas as pd
from datetime import datetime

def extract_training_metrics(model_dir):
    """Extract training metrics from training_metrics.json"""
    metrics_path = os.path.join('model_data', model_dir, 'training_metrics.json')
    if not os.path.exists(metrics_path):
        return None
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Get the last epoch's metrics and round to 4 decimal places
    return {
        'final_train_loss': round(float(metrics['train_loss'][-1]), 4),
        'final_val_loss': round(float(metrics['val_loss'][-1]), 4)
    }

def extract_evaluation_metrics(model_dir):
    """Extract open set evaluation metrics from the evaluation directory"""
    eval_dir = os.path.join('model_data', model_dir, 'open_set_evaluation')
    if not os.path.exists(eval_dir):
        return None
    
    # Look for the metrics in the evaluation directory
    for file in os.listdir(eval_dir):
        if file.endswith('_metrics.json'):
            metrics_path = os.path.join(eval_dir, file)
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return {
                'open_set_baks': float(metrics.get('BAKS', 0)),
                'open_set_baus': float(metrics.get('BAUS', 0)),
                'open_set_geometric_mean': float(metrics.get('geometric_mean', 0)),
                'open_set_threshold': float(metrics.get('threshold', 0)),
                'open_set_evaluation_time': metrics.get('evaluation_time', '')
            }
    
    return None

def extract_model_params(model_dir):
    """Extract model parameters from model_log.txt"""
    log_path = os.path.join('model_data', model_dir, 'model_log.txt')
    if not os.path.exists(log_path):
        return {'filename': model_dir}
    
    params = {'filename': model_dir}
    
    # Default values in case some parameters are not found
    defaults = {
        'backbone': 'resnet18',
        'batch_size': 32,
        'num_epochs': 15,
        'learning_rate': 0.0001,
        'm': 4,
        'resize': 288,
        'n': 140000,
        'val_split': 0.2,
        'weight_decay': 5e-05,
        'dropout': 0.3,
        'scheduler': 'cosine',
        'patience': 10,
        'augmentation': True,
        'embedding_dim': 512,
        'margin': 0.3,
        'scale': 64.0,
        'loss_type': 'arcface'
    }
    
    # Parameter mapping from log to CSV column names
    param_mapping = {
        'Backbone': 'backbone',
        'Batch size': 'batch_size',
        'Epochs': 'num_epochs',
        'Learning rate': 'learning_rate',
        'Images per class (M)': 'm',
        'Image size': 'resize',
        'Dataset size (n)': 'n',
        'Validation split': 'val_split',
        'Weight decay': 'weight_decay',
        'Dropout rate': 'dropout',
        'Scheduler': 'scheduler',
        'Early stopping patience': 'patience',
        'Data augmentation': 'augmentation',
        'Embedding dimension': 'embedding_dim',
        'Margin': 'margin',
        'Scale': 'scale',
        'Loss type': 'loss_type'
    }
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('===') or not line:
                    continue
                
                for log_key, csv_key in param_mapping.items():
                    if line.startswith(log_key):
                        value = line.split(': ')[1].strip()
                        
                        # Convert to appropriate type
                        if csv_key in ['batch_size', 'num_epochs', 'm', 'resize', 'n', 'patience', 'embedding_dim']:
                            value = int(value)
                        elif csv_key in ['learning_rate', 'val_split', 'weight_decay', 'dropout', 'margin', 'scale']:
                            value = round(float(value), 4)
                        elif csv_key == 'augmentation':
                            value = value.lower() == 'true'
                        
                        params[csv_key] = value
                        break
        
        # Fill in any missing parameters with defaults
        for key, default_value in defaults.items():
            if key not in params:
                params[key] = default_value
        
        # Special handling for resize parameter which might be in format "160x160"
        if 'resize' in params and isinstance(params['resize'], str) and 'x' in params['resize']:
            params['resize'] = int(params['resize'].split('x')[0])
        
    except Exception as e:
        print(f"Error reading parameters from {log_path}: {e}")
        # Use defaults if there's an error
        params.update(defaults)
    
    return params

def main():
    # Get all model directories starting with "2025-04-21"
    model_dirs = [d for d in os.listdir('model_data') 
                 if d.startswith('2025-04-21') and os.path.isdir(os.path.join('model_data', d))]
    
    # Initialize DataFrame with all required columns
    columns = [
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
        'diff_final_train_loss',
        'diff_final_val_loss'
    ]
    
    df = pd.DataFrame(columns=columns)
    
    # Process each model
    for model_dir in model_dirs:
        print(f"Processing {model_dir}...")
        
        # Extract metrics
        training_metrics = extract_training_metrics(model_dir)
        params = extract_model_params(model_dir)
        
        # Combine all metrics
        metrics = {**params}
        if training_metrics:
            metrics.update(training_metrics)
        
        # Add to DataFrame
        df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)
    
    # Calculate differences with base model if it exists
    base_model = df[df['filename'] == '2025-04-21_resnet18basemodel']
    if not base_model.empty:
        base_train_loss = base_model['final_train_loss'].iloc[0]
        base_val_loss = base_model['final_val_loss'].iloc[0]
        
        # Calculate differences
        df['diff_final_train_loss'] = round(df['final_train_loss'] - base_train_loss, 4)
        df['diff_final_val_loss'] = round(df['final_val_loss'] - base_val_loss, 4)
    
    # Save to CSV
    csv_path = os.path.join('model_data', 'all_model_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics to {csv_path}")

if __name__ == "__main__":
    main() 