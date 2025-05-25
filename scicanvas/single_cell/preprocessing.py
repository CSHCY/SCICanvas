"""
Preprocessing utilities for single-cell data.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from typing import Optional, List, Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def quality_control(
    adata: sc.AnnData,
    min_genes_per_cell: int = 200,
    max_genes_per_cell: int = 5000,
    min_cells_per_gene: int = 3,
    max_mito_percent: float = 20.0,
    max_ribo_percent: float = 50.0
) -> sc.AnnData:
    """
    Perform quality control filtering on single-cell data.
    
    Args:
        adata: Annotated data matrix
        min_genes_per_cell: Minimum number of genes per cell
        max_genes_per_cell: Maximum number of genes per cell
        min_cells_per_gene: Minimum number of cells per gene
        max_mito_percent: Maximum mitochondrial gene percentage
        max_ribo_percent: Maximum ribosomal gene percentage
        
    Returns:
        Filtered annotated data matrix
    """
    # Make a copy
    adata = adata.copy()
    
    # Calculate QC metrics
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    adata.var['ribo'] = adata.var_names.str.startswith(('RPS', 'RPL'))
    
    sc.pp.calculate_qc_metrics(
        adata, 
        percent_top=None, 
        log1p=False, 
        inplace=True
    )
    
    # Filter cells
    sc.pp.filter_cells(adata, min_genes=min_genes_per_cell)
    
    # Filter genes
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
    
    # Filter based on QC metrics
    adata = adata[adata.obs.n_genes_by_counts < max_genes_per_cell, :]
    adata = adata[adata.obs.pct_counts_mt < max_mito_percent, :]
    
    if 'pct_counts_ribo' in adata.obs.columns:
        adata = adata[adata.obs.pct_counts_ribo < max_ribo_percent, :]
    
    return adata


def normalize_data(
    adata: sc.AnnData,
    target_sum: float = 1e4,
    method: str = 'total_count',
    log_transform: bool = True
) -> sc.AnnData:
    """
    Normalize single-cell gene expression data.
    
    Args:
        adata: Annotated data matrix
        target_sum: Target sum for normalization
        method: Normalization method ('total_count', 'median', 'quantile')
        log_transform: Whether to apply log transformation
        
    Returns:
        Normalized annotated data matrix
    """
    # Make a copy
    adata = adata.copy()
    
    if method == 'total_count':
        sc.pp.normalize_total(adata, target_sum=target_sum)
    elif method == 'median':
        sc.pp.normalize_total(adata, target_sum=np.median(adata.obs.total_counts))
    elif method == 'quantile':
        # Quantile normalization
        from sklearn.preprocessing import quantile_transform
        X_norm = quantile_transform(adata.X.toarray(), axis=0, output_distribution='normal')
        adata.X = X_norm
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if log_transform:
        sc.pp.log1p(adata)
    
    return adata


def select_highly_variable_genes(
    adata: sc.AnnData,
    n_top_genes: int = 2000,
    method: str = 'seurat_v3',
    batch_key: Optional[str] = None
) -> sc.AnnData:
    """
    Select highly variable genes.
    
    Args:
        adata: Annotated data matrix
        n_top_genes: Number of top genes to select
        method: Method for HVG selection ('seurat_v3', 'cell_ranger', 'seurat')
        batch_key: Key for batch information (for batch-aware selection)
        
    Returns:
        Data with highly variable genes selected
    """
    # Make a copy
    adata = adata.copy()
    
    if batch_key is not None:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor=method,
            batch_key=batch_key
        )
    else:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor=method
        )
    
    # Keep only highly variable genes
    adata = adata[:, adata.var.highly_variable]
    
    return adata


def scale_data(
    adata: sc.AnnData,
    method: str = 'standard',
    max_value: Optional[float] = 10.0,
    zero_center: bool = True
) -> sc.AnnData:
    """
    Scale gene expression data.
    
    Args:
        adata: Annotated data matrix
        method: Scaling method ('standard', 'minmax', 'robust')
        max_value: Maximum value after scaling (for clipping)
        zero_center: Whether to center the data at zero
        
    Returns:
        Scaled annotated data matrix
    """
    # Make a copy
    adata = adata.copy()
    
    if method == 'standard':
        sc.pp.scale(adata, max_value=max_value, zero_center=zero_center)
    elif method == 'minmax':
        scaler = MinMaxScaler()
        adata.X = scaler.fit_transform(adata.X.toarray())
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        adata.X = scaler.fit_transform(adata.X.toarray())
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    return adata


def batch_correction(
    adata: sc.AnnData,
    batch_key: str,
    method: str = 'combat'
) -> sc.AnnData:
    """
    Perform batch correction.
    
    Args:
        adata: Annotated data matrix
        batch_key: Key in adata.obs containing batch information
        method: Batch correction method ('combat', 'scanorama', 'harmony')
        
    Returns:
        Batch-corrected annotated data matrix
    """
    # Make a copy
    adata = adata.copy()
    
    if method == 'combat':
        sc.pp.combat(adata, key=batch_key)
    elif method == 'scanorama':
        try:
            import scanorama
            # Split by batch
            batches = []
            batch_names = []
            for batch in adata.obs[batch_key].unique():
                batch_data = adata[adata.obs[batch_key] == batch]
                batches.append(batch_data.X.toarray())
                batch_names.append(batch)
            
            # Integrate
            integrated, genes = scanorama.integrate(batches, batch_names)
            
            # Reconstruct adata
            X_integrated = np.vstack(integrated)
            adata.X = X_integrated
            
        except ImportError:
            raise ImportError("scanorama not installed. Please install with: pip install scanorama")
            
    elif method == 'harmony':
        try:
            import harmonypy as hm
            # Run Harmony on PCA representation
            sc.tl.pca(adata)
            ho = hm.run_harmony(adata.obsm['X_pca'], adata.obs, batch_key)
            adata.obsm['X_pca_harmony'] = ho.Z_corr.T
        except ImportError:
            raise ImportError("harmonypy not installed. Please install with: pip install harmonypy")
    else:
        raise ValueError(f"Unknown batch correction method: {method}")
    
    return adata


def impute_missing_values(
    adata: sc.AnnData,
    method: str = 'magic',
    **kwargs
) -> sc.AnnData:
    """
    Impute missing values in single-cell data.
    
    Args:
        adata: Annotated data matrix
        method: Imputation method ('magic', 'scimpute', 'dca')
        **kwargs: Additional arguments for the imputation method
        
    Returns:
        Data with imputed values
    """
    # Make a copy
    adata = adata.copy()
    
    if method == 'magic':
        try:
            import magic
            magic_op = magic.MAGIC(**kwargs)
            adata.X = magic_op.fit_transform(adata.X.toarray())
        except ImportError:
            raise ImportError("magic-impute not installed. Please install with: pip install magic-impute")
            
    elif method == 'scimpute':
        # Placeholder for scImpute implementation
        raise NotImplementedError("scImpute integration not yet implemented")
        
    elif method == 'dca':
        try:
            from dca.api import dca
            adata = dca(adata, **kwargs)
        except ImportError:
            raise ImportError("dca not installed. Please install with: pip install dca")
    else:
        raise ValueError(f"Unknown imputation method: {method}")
    
    return adata 