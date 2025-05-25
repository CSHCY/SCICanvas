"""
Data handling utilities for single-cell analysis.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import scanpy as sc
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path


class SingleCellDataset(Dataset):
    """
    PyTorch Dataset for single-cell gene expression data.
    """
    
    def __init__(
        self,
        adata: sc.AnnData,
        target_key: Optional[str] = None,
        transform: Optional[callable] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            adata: Annotated data matrix
            target_key: Key in adata.obs for target labels (optional)
            transform: Optional transform to apply to the data
        """
        self.adata = adata
        self.target_key = target_key
        self.transform = transform
        
        # Extract expression matrix
        self.X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        
        # Extract targets if provided
        if target_key is not None and target_key in adata.obs.columns:
            self.y = adata.obs[target_key].values
            # Encode categorical targets
            if self.y.dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                self.label_encoder = LabelEncoder()
                self.y = self.label_encoder.fit_transform(self.y)
        else:
            self.y = None
            
    def __len__(self) -> int:
        """Return the number of samples."""
        return self.X.shape[0]
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        sample = {
            'expression': torch.FloatTensor(self.X[idx])
        }
        
        if self.y is not None:
            sample['target'] = torch.LongTensor([self.y[idx]])[0]
            
        if self.transform:
            sample = self.transform(sample)
            
        return sample


class SingleCellDataLoader:
    """
    Data loader factory for single-cell datasets.
    """
    
    @staticmethod
    def create_dataloaders(
        adata: sc.AnnData,
        target_key: Optional[str] = None,
        batch_size: int = 32,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        shuffle: bool = True,
        num_workers: int = 0,
        random_state: int = 42
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test data loaders.
        
        Args:
            adata: Annotated data matrix
            target_key: Key in adata.obs for target labels
            batch_size: Batch size for data loaders
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Validate splits
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
            "Splits must sum to 1.0"
            
        # Set random seed
        np.random.seed(random_state)
        
        # Create indices for splitting
        n_samples = adata.n_obs
        indices = np.random.permutation(n_samples)
        
        # Calculate split points
        train_end = int(train_split * n_samples)
        val_end = train_end + int(val_split * n_samples)
        
        # Split indices
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # Create subset AnnData objects
        adata_train = adata[train_indices].copy()
        adata_val = adata[val_indices].copy()
        adata_test = adata[test_indices].copy()
        
        # Create datasets
        train_dataset = SingleCellDataset(adata_train, target_key)
        val_dataset = SingleCellDataset(adata_val, target_key)
        test_dataset = SingleCellDataset(adata_test, target_key)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        return train_loader, val_loader, test_loader


def load_example_data(dataset_name: str = "pbmc3k") -> sc.AnnData:
    """
    Load example single-cell datasets.
    
    Args:
        dataset_name: Name of the dataset to load
        
    Returns:
        Annotated data matrix
    """
    if dataset_name == "pbmc3k":
        # Load PBMC 3k dataset from scanpy
        adata = sc.datasets.pbmc3k_processed()
        return adata
    elif dataset_name == "paul15":
        # Load Paul et al. 2015 dataset
        adata = sc.datasets.paul15()
        return adata
    elif dataset_name == "blobs":
        # Generate synthetic blob data
        from sklearn.datasets import make_blobs
        X, y = make_blobs(n_samples=1000, centers=5, n_features=100, random_state=42)
        adata = sc.AnnData(X)
        adata.obs['cluster'] = y.astype(str)
        return adata
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def preprocess_data(
    adata: sc.AnnData,
    min_genes: int = 200,
    min_cells: int = 3,
    target_sum: float = 1e4,
    n_top_genes: int = 2000,
    log_transform: bool = True,
    scale: bool = True
) -> sc.AnnData:
    """
    Standard preprocessing pipeline for single-cell data.
    
    Args:
        adata: Annotated data matrix
        min_genes: Minimum number of genes per cell
        min_cells: Minimum number of cells per gene
        target_sum: Target sum for normalization
        n_top_genes: Number of highly variable genes to keep
        log_transform: Whether to apply log transformation
        scale: Whether to scale the data
        
    Returns:
        Preprocessed annotated data matrix
    """
    # Make a copy to avoid modifying the original
    adata = adata.copy()
    
    # Basic filtering
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    # Calculate QC metrics
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    
    # Save raw data
    adata.raw = adata
    
    # Normalization
    sc.pp.normalize_total(adata, target_sum=target_sum)
    
    # Log transformation
    if log_transform:
        sc.pp.log1p(adata)
        
    # Find highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var.highly_variable]
    
    # Scaling
    if scale:
        sc.pp.scale(adata, max_value=10)
        
    return adata 