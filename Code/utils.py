import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, List, Tuple, Callable
from wildlife_datasets import datasets, metrics

license_conversion = {
    'Missing': 'None',
    'Other': 'Other',
    'Attribution 4.0 International (CC BY 4.0)': 'CC BY 4.0',
    'Creative Commons Attribution 4.0 International': 'CC BY 4.0',
    'Attribution-NonCommercial-ShareAlike 4.0 International': 'CC BY-NC-SA 4.0',
    'Non-Commercial Government Licence for public sector information': 'NC-Government',
    'Community Data License Agreement – Permissive': 'CDLA-Permissive-1.0',
    'Community Data License Agreement – Permissive, Version 1.0': 'CDLA-Permissive-1.0',
    'MIT License': 'MIT',
    'Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)': 'CC BY-NC-SA 4.0',
    'Attribution-ShareAlike 3.0 Unported' : 'CC BY-SA 3.0',
    'Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)': 'CC BY-SA 4.0',
}

def get_summary_species(df):
    summary_species = {}
    for name, df_red in df.groupby('species'):
        summary_species[name] = {
            'images': len(df_red),
            'individuals': df_red['identity'].nunique(),
        }
    summary_species = pd.DataFrame(summary_species).T
    summary_species.loc['TOTAL'] = summary_species.sum()
    return summary_species

def get_summary_datasets(df):
    summary_datasets = {}
    for name, df_red in df.groupby('dataset'):
        metadata = eval(f'datasets.{name}.metadata')
        if 'licenses' in metadata:
            license = metadata['licenses']
        else:
            license = 'Missing'    
        summary_datasets[name] = {
            'images': len(df_red),
            'individuals': df_red['identity'].nunique(),
            'wild': metadata['wild'],
            'license': license_conversion[license],
        }

    summary_datasets = pd.DataFrame(summary_datasets).T.sort_index(key=lambda x: x.str.lower())
    summary_datasets.loc['TOTAL'] = summary_datasets.sum()
    summary_datasets.loc['TOTAL', ['license', 'wild']] = ''
    for col in ['images', 'individuals']:
        summary_datasets[col] = summary_datasets[col].astype(int)        
    return summary_datasets

def compute_predictions(
        features_query: np.ndarray,
        features_database: np.ndarray,
        ignore: Optional[List[List[int]]] = None,
        matcher: Callable = cosine_similarity,
        k: int = 4,
        return_score: bool = False
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Computes a closest match in the database for each vector in the query set.

    Args:
        features_query (np.ndarray): Query features of size n_query*n_feature. 
        features_database (np.ndarray): Database features of size n_database*n_feature
        ignore (Optional[List[List[int]]], optional): `ignore[i]` is a list of indices
            in the database ignores for i-th query.
        matcher (Callable, optional): function computing similarity.
        k (int, optional): Returned number of predictions.
        return_score (bool, optional): Whether the similalarity is returned.

    Returns:
        Vector of size (n_query,) and array of size (n_query,k). The latter are indices
            in the database for the closest matches (with ignored `ignore` indices).
            If `return_score`, it also returns an array of size (n_query,k) of scores.
    """

    # Create batch chunks
    n_query = len(features_query)
    # If ignore is not provided, initialize as empty
    if ignore is None:
        ignore = [[] for _ in range(n_query)]
    
    idx_true = np.array(range(n_query))
    idx_pred = np.zeros((n_query, k), dtype=np.int32)
    scores = np.zeros((n_query, k))
    # Compute the cosine similarity between the query and the database
    similarity = matcher(features_query, features_database)
    # Set -infinity for ignored indices
    for i in range(len(ignore)):
        similarity[i, ignore[i]] = -np.inf
    # Find the closest matches (k highest values)
    idx_pred = (-similarity).argsort(axis=-1)[:, :k]
    if return_score:
        scores = np.take_along_axis(similarity, idx_pred, axis=-1)
        return idx_true, idx_pred, scores
    else:
        return idx_true, idx_pred

def mean(x, idx=None):
    if idx is None:
        return np.mean(list(x.values()))
    else:
        return np.mean([x[i] for i in idx])

def compute_baks_baus(df, y_pred, new_individual='', split_col='split'):
    baks = {}
    baus = {}
    y_true = df['identity'].to_numpy()
    for dataset, df_dataset in df.groupby('dataset'):
        identity_train = df_dataset['identity'][df_dataset[split_col] == 'train'].to_numpy()
        identity_test = df_dataset['identity'][df_dataset[split_col] == 'test'].to_numpy()
        identity_test_only = list(set(identity_test) - set(identity_train))                
        idx = df.index.get_indexer(df_dataset.index[df_dataset[split_col] == 'test'])
        baks[dataset] = metrics.BAKS(y_true[idx], y_pred[idx], identity_test_only)
        baus[dataset] = metrics.BAUS(y_true[idx], y_pred[idx], identity_test_only, new_individual)
    return baks, baus

def predict(df, features, split_col='split'):
    y_pred = np.full(len(df), np.nan, dtype=object)
    similarity_pred = np.full(len(df), np.nan, dtype=object)
    for dataset_name in df['dataset'].unique():
        idx_train = np.where((df['dataset'] == dataset_name) * (df[split_col] == 'train'))[0]
        idx_test = np.where((df['dataset'] == dataset_name) * (df[split_col] == 'test'))[0]

        idx_true, idx_pred, similarity = compute_predictions(features[idx_test], features[idx_train], return_similarity=True)
        idx_true = idx_test[idx_true]
        idx_pred = idx_train[idx_pred]

        y_pred[idx_true] = df['identity'].iloc[idx_pred[:,0]].values
        similarity_pred[idx_true] = similarity[:,0]
    return y_pred, similarity_pred