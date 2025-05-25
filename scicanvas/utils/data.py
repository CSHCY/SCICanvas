"""
Shared data utilities for SCICanvas toolkit.
"""

import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import h5py
import pickle


class DataLoader:
    """
    Universal data loader for different scientific data types.
    """
    
    @staticmethod
    def load_csv(
        file_path: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """Load data from CSV file."""
        return pd.read_csv(file_path, **kwargs)
    
    @staticmethod
    def load_hdf5(
        file_path: Union[str, Path],
        key: str = 'data'
    ) -> np.ndarray:
        """Load data from HDF5 file."""
        with h5py.File(file_path, 'r') as f:
            return f[key][:]
    
    @staticmethod
    def load_pickle(
        file_path: Union[str, Path]
    ) -> Any:
        """Load data from pickle file."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def save_csv(
        data: pd.DataFrame,
        file_path: Union[str, Path],
        **kwargs
    ) -> None:
        """Save data to CSV file."""
        data.to_csv(file_path, **kwargs)
    
    @staticmethod
    def save_hdf5(
        data: np.ndarray,
        file_path: Union[str, Path],
        key: str = 'data'
    ) -> None:
        """Save data to HDF5 file."""
        with h5py.File(file_path, 'w') as f:
            f.create_dataset(key, data=data)
    
    @staticmethod
    def save_pickle(
        data: Any,
        file_path: Union[str, Path]
    ) -> None:
        """Save data to pickle file."""
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)


class UniversalDataset(Dataset):
    """
    Universal PyTorch dataset for scientific data.
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, torch.Tensor, pd.DataFrame],
        targets: Optional[Union[np.ndarray, torch.Tensor, pd.Series]] = None,
        transform: Optional[callable] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data: Input data
            targets: Target labels (optional)
            transform: Optional transform to apply
        """
        # Convert data to numpy array
        if isinstance(data, pd.DataFrame):
            self.data = data.values
        elif isinstance(data, torch.Tensor):
            self.data = data.numpy()
        else:
            self.data = data
            
        # Convert targets to numpy array
        if targets is not None:
            if isinstance(targets, pd.Series):
                self.targets = targets.values
            elif isinstance(targets, torch.Tensor):
                self.targets = targets.numpy()
            else:
                self.targets = targets
        else:
            self.targets = None
            
        self.transform = transform
        
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        sample = {
            'data': torch.FloatTensor(self.data[idx])
        }
        
        if self.targets is not None:
            sample['target'] = torch.LongTensor([self.targets[idx]])[0]
            
        if self.transform:
            sample = self.transform(sample)
            
        return sample


def create_data_splits(
    data: np.ndarray,
    targets: Optional[np.ndarray] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, ...]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        data: Input data
        targets: Target labels (optional)
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_state: Random seed
        
    Returns:
        Tuple of split data arrays
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    np.random.seed(random_state)
    n_samples = len(data)
    indices = np.random.permutation(n_samples)
    
    # Calculate split points
    train_end = int(train_ratio * n_samples)
    val_end = train_end + int(val_ratio * n_samples)
    
    # Split indices
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    # Split data
    train_data = data[train_idx]
    val_data = data[val_idx]
    test_data = data[test_idx]
    
    if targets is not None:
        train_targets = targets[train_idx]
        val_targets = targets[val_idx]
        test_targets = targets[test_idx]
        
        return (train_data, val_data, test_data,
                train_targets, val_targets, test_targets)
    else:
        return train_data, val_data, test_data


def normalize_data(
    data: np.ndarray,
    method: str = 'standard',
    axis: int = 0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalize data using various methods.
    
    Args:
        data: Input data
        method: Normalization method ('standard', 'minmax', 'robust')
        axis: Axis along which to normalize
        
    Returns:
        Tuple of (normalized_data, normalization_params)
    """
    if method == 'standard':
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        normalized_data = (data - mean) / (std + 1e-8)
        params = {'mean': mean, 'std': std}
        
    elif method == 'minmax':
        min_val = np.min(data, axis=axis, keepdims=True)
        max_val = np.max(data, axis=axis, keepdims=True)
        normalized_data = (data - min_val) / (max_val - min_val + 1e-8)
        params = {'min': min_val, 'max': max_val}
        
    elif method == 'robust':
        median = np.median(data, axis=axis, keepdims=True)
        mad = np.median(np.abs(data - median), axis=axis, keepdims=True)
        normalized_data = (data - median) / (mad + 1e-8)
        params = {'median': median, 'mad': mad}
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized_data, params


def apply_normalization(
    data: np.ndarray,
    params: Dict[str, Any],
    method: str = 'standard'
) -> np.ndarray:
    """
    Apply previously computed normalization parameters to new data.
    
    Args:
        data: Input data
        params: Normalization parameters
        method: Normalization method
        
    Returns:
        Normalized data
    """
    if method == 'standard':
        return (data - params['mean']) / (params['std'] + 1e-8)
    elif method == 'minmax':
        return (data - params['min']) / (params['max'] - params['min'] + 1e-8)
    elif method == 'robust':
        return (data - params['median']) / (params['mad'] + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def batch_generator(
    data: np.ndarray,
    batch_size: int,
    shuffle: bool = True
) -> np.ndarray:
    """
    Generate batches from data.
    
    Args:
        data: Input data
        batch_size: Size of each batch
        shuffle: Whether to shuffle the data
        
    Yields:
        Batches of data
    """
    n_samples = len(data)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield data[batch_indices] 