"""
Visualization utilities for single-cell analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scanpy as sc
from typing import Optional, List, Dict, Any, Tuple, Union


def plot_qc_metrics(
    adata: sc.AnnData,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot quality control metrics.
    
    Args:
        adata: Annotated data matrix
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Number of genes per cell
    axes[0, 0].hist(adata.obs['n_genes_by_counts'], bins=50, alpha=0.7)
    axes[0, 0].set_xlabel('Number of genes')
    axes[0, 0].set_ylabel('Number of cells')
    axes[0, 0].set_title('Genes per cell')
    
    # Total counts per cell
    axes[0, 1].hist(adata.obs['total_counts'], bins=50, alpha=0.7)
    axes[0, 1].set_xlabel('Total counts')
    axes[0, 1].set_ylabel('Number of cells')
    axes[0, 1].set_title('Total counts per cell')
    
    # Mitochondrial gene percentage
    if 'pct_counts_mt' in adata.obs.columns:
        axes[0, 2].hist(adata.obs['pct_counts_mt'], bins=50, alpha=0.7)
        axes[0, 2].set_xlabel('Mitochondrial gene %')
        axes[0, 2].set_ylabel('Number of cells')
        axes[0, 2].set_title('Mitochondrial genes')
    
    # Scatter plots
    axes[1, 0].scatter(adata.obs['total_counts'], adata.obs['n_genes_by_counts'], alpha=0.5)
    axes[1, 0].set_xlabel('Total counts')
    axes[1, 0].set_ylabel('Number of genes')
    axes[1, 0].set_title('Counts vs Genes')
    
    if 'pct_counts_mt' in adata.obs.columns:
        axes[1, 1].scatter(adata.obs['total_counts'], adata.obs['pct_counts_mt'], alpha=0.5)
        axes[1, 1].set_xlabel('Total counts')
        axes[1, 1].set_ylabel('Mitochondrial gene %')
        axes[1, 1].set_title('Counts vs Mitochondrial %')
        
        axes[1, 2].scatter(adata.obs['n_genes_by_counts'], adata.obs['pct_counts_mt'], alpha=0.5)
        axes[1, 2].set_xlabel('Number of genes')
        axes[1, 2].set_ylabel('Mitochondrial gene %')
        axes[1, 2].set_title('Genes vs Mitochondrial %')
    
    plt.tight_layout()
    return fig


def plot_highly_variable_genes(
    adata: sc.AnnData,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot highly variable genes.
    
    Args:
        adata: Annotated data matrix
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Mean vs dispersion
    sc.pl.highly_variable_genes(adata, ax=axes[0], show=False)
    
    # Top highly variable genes
    if 'highly_variable_rank' in adata.var.columns:
        top_genes = adata.var.nlargest(20, 'dispersions_norm')
        axes[1].barh(range(len(top_genes)), top_genes['dispersions_norm'])
        axes[1].set_yticks(range(len(top_genes)))
        axes[1].set_yticklabels(top_genes.index)
        axes[1].set_xlabel('Normalized dispersion')
        axes[1].set_title('Top 20 highly variable genes')
    
    plt.tight_layout()
    return fig


def plot_dimensionality_reduction(
    adata: sc.AnnData,
    method: str = 'umap',
    color: Optional[Union[str, List[str]]] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot dimensionality reduction results.
    
    Args:
        adata: Annotated data matrix
        method: Dimensionality reduction method ('umap', 'tsne', 'pca')
        color: Variable(s) to color by
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if method == 'umap':
        sc.pl.umap(adata, color=color, ax=ax, show=False)
    elif method == 'tsne':
        sc.pl.tsne(adata, color=color, ax=ax, show=False)
    elif method == 'pca':
        sc.pl.pca(adata, color=color, ax=ax, show=False)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return fig


def plot_gene_expression(
    adata: sc.AnnData,
    genes: Union[str, List[str]],
    method: str = 'umap',
    ncols: int = 3,
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Plot gene expression on dimensionality reduction plot.
    
    Args:
        adata: Annotated data matrix
        genes: Gene(s) to plot
        method: Dimensionality reduction method
        ncols: Number of columns in subplot grid
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if isinstance(genes, str):
        genes = [genes]
    
    nrows = (len(genes) + ncols - 1) // ncols
    
    if figsize is None:
        figsize = (ncols * 4, nrows * 4)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for i, gene in enumerate(genes):
        if gene in adata.var_names:
            if method == 'umap':
                sc.pl.umap(adata, color=gene, ax=axes[i], show=False)
            elif method == 'tsne':
                sc.pl.tsne(adata, color=gene, ax=axes[i], show=False)
            elif method == 'pca':
                sc.pl.pca(adata, color=gene, ax=axes[i], show=False)
        else:
            axes[i].text(0.5, 0.5, f'Gene {gene}\nnot found', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
    
    # Hide empty subplots
    for i in range(len(genes), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_cell_type_composition(
    adata: sc.AnnData,
    cell_type_key: str,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot cell type composition.
    
    Args:
        adata: Annotated data matrix
        cell_type_key: Key in adata.obs containing cell type labels
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Cell type counts
    cell_counts = adata.obs[cell_type_key].value_counts()
    axes[0].bar(range(len(cell_counts)), cell_counts.values)
    axes[0].set_xticks(range(len(cell_counts)))
    axes[0].set_xticklabels(cell_counts.index, rotation=45, ha='right')
    axes[0].set_ylabel('Number of cells')
    axes[0].set_title('Cell type counts')
    
    # Cell type proportions (pie chart)
    axes[1].pie(cell_counts.values, labels=cell_counts.index, autopct='%1.1f%%')
    axes[1].set_title('Cell type proportions')
    
    plt.tight_layout()
    return fig


def plot_trajectory(
    adata: sc.AnnData,
    pseudotime_key: str = 'pseudotime',
    method: str = 'umap',
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot trajectory inference results.
    
    Args:
        adata: Annotated data matrix
        pseudotime_key: Key in adata.obs containing pseudotime values
        method: Dimensionality reduction method
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Pseudotime on dimensionality reduction
    if method == 'umap':
        sc.pl.umap(adata, color=pseudotime_key, ax=axes[0], show=False)
    elif method == 'tsne':
        sc.pl.tsne(adata, color=pseudotime_key, ax=axes[0], show=False)
    elif method == 'pca':
        sc.pl.pca(adata, color=pseudotime_key, ax=axes[0], show=False)
    
    # Pseudotime distribution
    axes[1].hist(adata.obs[pseudotime_key], bins=50, alpha=0.7)
    axes[1].set_xlabel('Pseudotime')
    axes[1].set_ylabel('Number of cells')
    axes[1].set_title('Pseudotime distribution')
    
    plt.tight_layout()
    return fig


def plot_gene_regulatory_network(
    adjacency_matrix: np.ndarray,
    gene_names: Optional[List[str]] = None,
    top_n: int = 50,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot gene regulatory network.
    
    Args:
        adjacency_matrix: Gene-gene adjacency matrix
        gene_names: Names of genes (optional)
        top_n: Number of top connections to show
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Adjacency matrix heatmap
    subset_size = min(top_n, adjacency_matrix.shape[0])
    adj_subset = adjacency_matrix[:subset_size, :subset_size]
    
    sns.heatmap(adj_subset, ax=axes[0], cmap='Reds', 
                xticklabels=False, yticklabels=False)
    axes[0].set_title(f'Gene Regulatory Network\n(Top {subset_size}x{subset_size} genes)')
    
    # Degree distribution
    degrees = np.sum(adjacency_matrix > 0, axis=1)
    axes[1].hist(degrees, bins=20, alpha=0.7, color='skyblue')
    axes[1].set_xlabel('Node Degree')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Degree Distribution')
    
    plt.tight_layout()
    return fig


def plot_classification_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot classification results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix, classification_report
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[0], cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Confusion Matrix')
    
    # Classification metrics
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Extract metrics for plotting
    classes = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
    precision = [report[cls]['precision'] for cls in classes]
    recall = [report[cls]['recall'] for cls in classes]
    f1_score = [report[cls]['f1-score'] for cls in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    axes[1].bar(x - width, precision, width, label='Precision', alpha=0.8)
    axes[1].bar(x, recall, width, label='Recall', alpha=0.8)
    axes[1].bar(x + width, f1_score, width, label='F1-score', alpha=0.8)
    
    axes[1].set_xlabel('Classes')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Classification Metrics')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(classes, rotation=45, ha='right')
    axes[1].legend()
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    return fig 