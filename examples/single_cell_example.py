"""
Example script demonstrating single-cell analysis with SCICanvas.

This script shows how to use the single-cell module for:
1. Cell type classification
2. Trajectory inference
3. Gene regulatory network inference
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import SCICanvas modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import scanpy as sc
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False
    print("Warning: scanpy not available, using synthetic data structures")

try:
    from scicanvas.single_cell import CellTypeClassifier, TrajectoryInference, GeneRegulatoryNetwork
    print("✓ Successfully imported SCICanvas single-cell module")
except ImportError as e:
    print(f"✗ Error importing SCICanvas single-cell module: {e}")
    sys.exit(1)

def generate_synthetic_data(n_cells=1000, n_genes=2000, n_cell_types=5):
    """Generate synthetic single-cell data for demonstration."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate cell type labels
    cell_types = [f"CellType_{i}" for i in range(n_cell_types)]
    cell_labels = np.random.choice(cell_types, n_cells)
    
    # Generate gene expression data with cell type-specific patterns
    X = np.random.negative_binomial(n=5, p=0.3, size=(n_cells, n_genes)).astype(float)
    
    # Add cell type-specific expression patterns
    for i, cell_type in enumerate(cell_types):
        mask = cell_labels == cell_type
        # Make certain genes more highly expressed in specific cell types
        marker_genes = slice(i * 100, (i + 1) * 100)
        X[mask, marker_genes] *= (2 + i)  # Different expression levels
        
    # Create gene names
    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    
    if SCANPY_AVAILABLE:
        # Create AnnData object
        adata = sc.AnnData(X)
        adata.obs['cell_type'] = cell_labels
        adata.var_names = gene_names
        adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]
    else:
        # Create simple data structure
        class SimpleAnnData:
            def __init__(self, X):
                self.X = X
                self.n_obs, self.n_vars = X.shape
                self.obs = pd.DataFrame(index=[f"Cell_{i}" for i in range(X.shape[0])])
                self.var = pd.DataFrame(index=gene_names)
                self.var_names = gene_names
                self.obs_names = self.obs.index
        
        adata = SimpleAnnData(X)
        adata.obs['cell_type'] = cell_labels
    
    return adata

def demonstrate_cell_type_classification():
    """Demonstrate cell type classification."""
    print("=" * 50)
    print("CELL TYPE CLASSIFICATION DEMO")
    print("=" * 50)
    
    # Generate synthetic data
    adata = generate_synthetic_data(n_cells=500, n_genes=1000, n_cell_types=3)
    print(f"Generated data: {adata.n_obs} cells, {adata.n_vars} genes")
    print(f"Cell types: {adata.obs['cell_type'].unique()}")
    
    # Basic preprocessing
    if SCANPY_AVAILABLE:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        # Simple normalization
        adata.X = adata.X / np.sum(adata.X, axis=1, keepdims=True) * 1e4
        adata.X = np.log1p(adata.X)
    
    # Initialize classifier
    classifier = CellTypeClassifier(
        model_type='transformer',
        model_params={
            'd_model': 256,
            'n_heads': 4,
            'n_layers': 3
        }
    )
    
    # Train the classifier
    print("\nTraining cell type classifier...")
    classifier.fit(
        adata,
        cell_type_key='cell_type',
        batch_size=32,
        num_epochs=20,  # Reduced for demo
        learning_rate=1e-3
    )
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = classifier.predict(adata)
    probabilities = classifier.predict_proba(adata)
    
    # Evaluate performance
    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(adata.obs['cell_type'], predictions)
    print(f"\nClassification Accuracy: {accuracy:.3f}")
    
    print("\nClassification Report:")
    print(classification_report(adata.obs['cell_type'], predictions))
    
    return adata, classifier

def demonstrate_trajectory_inference():
    """Demonstrate trajectory inference."""
    print("\n" + "=" * 50)
    print("TRAJECTORY INFERENCE DEMO")
    print("=" * 50)
    
    # Generate synthetic trajectory data
    adata = generate_synthetic_data(n_cells=300, n_genes=500, n_cell_types=3)
    
    # Basic preprocessing
    if SCANPY_AVAILABLE:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        # Simple normalization
        adata.X = adata.X / np.sum(adata.X, axis=1, keepdims=True) * 1e4
        adata.X = np.log1p(adata.X)
    
    # Initialize trajectory inference
    trajectory_model = TrajectoryInference(
        model_type='vae',
        latent_dim=32,
        model_params={
            'hidden_dims': [256, 128, 64]
        }
    )
    
    # Train the model
    print("\nTraining trajectory inference model...")
    trajectory_model.fit(
        adata,
        batch_size=32,
        num_epochs=20,  # Reduced for demo
        learning_rate=1e-3
    )
    
    # Predict trajectory and pseudotime
    print("\nInferring trajectory...")
    results = trajectory_model.predict(adata)
    
    # Add results to adata
    adata.obsm['X_latent'] = results['latent']
    adata.obs['pseudotime'] = results['pseudotime']
    
    print(f"Latent representation shape: {results['latent'].shape}")
    print(f"Pseudotime range: [{results['pseudotime'].min():.3f}, {results['pseudotime'].max():.3f}]")
    
    return adata, trajectory_model

def demonstrate_gene_regulatory_network():
    """Demonstrate gene regulatory network inference."""
    print("\n" + "=" * 50)
    print("GENE REGULATORY NETWORK DEMO")
    print("=" * 50)
    
    # Generate smaller dataset for GRN (computational efficiency)
    adata = generate_synthetic_data(n_cells=200, n_genes=100, n_cell_types=2)
    
    # Basic preprocessing
    if SCANPY_AVAILABLE:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        # Simple normalization
        adata.X = adata.X / np.sum(adata.X, axis=1, keepdims=True) * 1e4
        adata.X = np.log1p(adata.X)
    
    # Initialize GRN inference
    grn_model = GeneRegulatoryNetwork(
        model_type='gnn',
        hidden_dim=64,
        model_params={
            'n_layers': 2
        }
    )
    
    # Train the model
    print("\nTraining gene regulatory network model...")
    grn_model.fit(
        adata,
        batch_size=16,
        num_epochs=10,  # Reduced for demo
        learning_rate=1e-3
    )
    
    # Predict regulatory network
    print("\nInferring gene regulatory network...")
    results = grn_model.predict(adata)
    
    adjacency_matrix = results['adjacency_matrix']
    print(f"Adjacency matrix shape: {adjacency_matrix.shape}")
    print(f"Number of predicted edges: {np.sum(adjacency_matrix > 0)}")
    
    return adata, grn_model, adjacency_matrix

def visualize_results(adata_classification, adata_trajectory, adjacency_matrix):
    """Create visualizations of the results."""
    print("\n" + "=" * 50)
    print("CREATING VISUALIZATIONS")
    print("=" * 50)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Cell type classification confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(adata_classification.obs['cell_type'], 
                         adata_classification.obs.get('predicted_cell_type', 
                                                    adata_classification.obs['cell_type']))
    
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 0], cmap='Blues')
    axes[0, 0].set_title('Cell Type Classification\nConfusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('True')
    
    # 2. Trajectory pseudotime
    if 'pseudotime' in adata_trajectory.obs.columns:
        scatter = axes[0, 1].scatter(
            adata_trajectory.obsm['X_latent'][:, 0],
            adata_trajectory.obsm['X_latent'][:, 1],
            c=adata_trajectory.obs['pseudotime'],
            cmap='viridis'
        )
        axes[0, 1].set_title('Trajectory Inference\nPseudotime')
        axes[0, 1].set_xlabel('Latent Dimension 1')
        axes[0, 1].set_ylabel('Latent Dimension 2')
        plt.colorbar(scatter, ax=axes[0, 1])
    
    # 3. Gene regulatory network heatmap
    # Show top 20x20 genes for visualization
    top_genes = 20
    adj_subset = adjacency_matrix[:top_genes, :top_genes]
    
    sns.heatmap(adj_subset, ax=axes[1, 0], cmap='Reds', 
                xticklabels=False, yticklabels=False)
    axes[1, 0].set_title(f'Gene Regulatory Network\n(Top {top_genes}x{top_genes} genes)')
    
    # 4. Network degree distribution
    degrees = np.sum(adjacency_matrix > 0, axis=1)
    axes[1, 1].hist(degrees, bins=20, alpha=0.7, color='skyblue')
    axes[1, 1].set_title('Gene Regulatory Network\nDegree Distribution')
    axes[1, 1].set_xlabel('Node Degree')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('single_cell_analysis_results.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved as 'single_cell_analysis_results.png'")
    
    return fig

def main():
    """Main function to run all demonstrations."""
    print("SCICanvas Single-Cell Analysis Demo")
    print("===================================")
    
    try:
        # 1. Cell type classification
        adata_classification, classifier = demonstrate_cell_type_classification()
        
        # 2. Trajectory inference
        adata_trajectory, trajectory_model = demonstrate_trajectory_inference()
        
        # 3. Gene regulatory network
        adata_grn, grn_model, adjacency_matrix = demonstrate_gene_regulatory_network()
        
        # 4. Visualizations
        visualize_results(adata_classification, adata_trajectory, adjacency_matrix)
        
        print("\n" + "=" * 50)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("\nKey Results:")
        print(f"- Cell type classification completed")
        print(f"- Trajectory inference completed")
        print(f"- Gene regulatory network inference completed")
        print(f"- Visualizations saved")
        
    except Exception as e:
        print(f"\nError during demo: {str(e)}")
        print("This is expected if dependencies are not fully installed.")
        print("Please install all requirements and try again.")

if __name__ == "__main__":
    main() 