"""
Predictor classes for single-cell analysis tasks.
"""

import torch
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Optional, Tuple
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, adjusted_rand_score

from ..core.base import BasePredictor
from .models import SingleCellTransformer, VariationalAutoEncoder, GraphNeuralNetwork


class CellTypeClassifier(BasePredictor):
    """
    Cell type classification predictor for single-cell RNA-seq data.
    
    This predictor can classify cells into different cell types based on
    their gene expression profiles using various neural network architectures.
    """
    
    def __init__(
        self,
        model_type: str = 'transformer',
        n_genes: Optional[int] = None,
        n_cell_types: Optional[int] = None,
        model_params: Optional[Dict[str, Any]] = None,
        device: str = 'auto'
    ):
        super().__init__(device=device)
        
        self.model_type = model_type
        self.n_genes = n_genes
        self.n_cell_types = n_cell_types
        self.model_params = model_params or {}
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
        if n_genes is not None and n_cell_types is not None:
            self._build_model()
            
    def _build_model(self):
        """Build the neural network model."""
        try:
            if self.model_type == 'transformer':
                self.model = SingleCellTransformer(
                    n_genes=self.n_genes,
                    n_cell_types=self.n_cell_types,
                    **self.model_params
                )
            elif self.model_type == 'vae':
                # VAE with classification head
                self.model = VariationalAutoEncoder(
                    n_genes=self.n_genes,
                    **self.model_params
                )
                # Add classification head
                latent_dim = self.model_params.get('latent_dim', 64)
                self.classifier_head = torch.nn.Linear(latent_dim, self.n_cell_types)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
                
            if self.model is None:
                raise ValueError(f"Failed to create model of type: {self.model_type}")
                
            self.model.to(self.device)
            if hasattr(self, 'classifier_head'):
                self.classifier_head.to(self.device)
                
        except Exception as e:
            self.logger.error(f"Error building model: {e}")
            # Create a simple fallback model
            self.model = torch.nn.Sequential(
                torch.nn.Linear(self.n_genes, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(64, self.n_cell_types)
            )
            self.model.to(self.device)
            self.logger.warning("Using fallback simple neural network model")
            
    def fit(
        self,
        adata: sc.AnnData,
        cell_type_key: str = 'cell_type',
        batch_size: int = 32,
        num_epochs: int = 100,
        learning_rate: float = 1e-3,
        validation_split: float = 0.2
    ):
        """
        Fit the cell type classifier.
        
        Args:
            adata: Annotated data matrix with gene expression and cell type labels
            cell_type_key: Key in adata.obs containing cell type labels
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            validation_split: Fraction of data to use for validation
        """
        # Extract data
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        y = adata.obs[cell_type_key].values
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Set model parameters
        self.n_genes = X.shape[1]
        self.n_cell_types = len(np.unique(y_encoded))
        
        # Build model if not already built
        if not hasattr(self, 'model') or self.model is None:
            self._build_model()
            
        # Verify model was created successfully
        if self.model is None:
            raise ValueError("Model was not created successfully")
            
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y_encoded).to(self.device)
        
        # Split data
        n_samples = X.shape[0]
        n_val = int(n_samples * validation_split)
        indices = torch.randperm(n_samples)
        
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        X_train, y_train = X_tensor[train_indices], y_tensor[train_indices]
        X_val, y_val = X_tensor[val_indices], y_tensor[val_indices]
        
        # Setup training
        if hasattr(self, 'classifier_head'):
            # For VAE model, include both model and classifier head parameters
            all_params = list(self.model.parameters()) + list(self.classifier_head.parameters())
            optimizer = torch.optim.Adam(all_params, lr=learning_rate)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            # Shuffle training data
            perm = torch.randperm(X_train.size(0))
            
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, X_train.size(0), batch_size):
                batch_indices = perm[i:i + batch_size]
                batch_X = X_train[batch_indices]
                batch_y = y_train[batch_indices]
                
                optimizer.zero_grad()
                
                if self.model_type == 'transformer':
                    outputs = self.model(batch_X)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                elif self.model_type == 'vae':
                    outputs = self.model(batch_X)
                    z = outputs['z'] if isinstance(outputs, dict) else outputs
                    logits = self.classifier_head(z)
                else:
                    # Fallback simple model
                    logits = self.model(batch_X)
                    
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
                
            # Validation
            if epoch % 10 == 0:
                val_acc = self._evaluate(X_val, y_val)
                self.logger.info(f"Epoch {epoch}: Loss={epoch_loss/n_batches:.4f}, Val Acc={val_acc:.4f}")
                
        self.is_fitted = True
        self.logger.info("Training completed")
        
    def predict(self, adata: sc.AnnData) -> np.ndarray:
        """
        Predict cell types for new data.
        
        Args:
            adata: Annotated data matrix with gene expression
            
        Returns:
            Predicted cell type labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            if self.model_type == 'transformer':
                outputs = self.model(X_tensor)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            elif self.model_type == 'vae':
                outputs = self.model(X_tensor)
                z = outputs['z'] if isinstance(outputs, dict) else outputs
                logits = self.classifier_head(z)
            else:
                # Fallback simple model
                logits = self.model(X_tensor)
                
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            
        # Decode labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        return predicted_labels
        
    def predict_proba(self, adata: sc.AnnData) -> np.ndarray:
        """
        Predict cell type probabilities.
        
        Args:
            adata: Annotated data matrix with gene expression
            
        Returns:
            Predicted probabilities for each cell type
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            if self.model_type == 'transformer':
                outputs = self.model(X_tensor)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            elif self.model_type == 'vae':
                outputs = self.model(X_tensor)
                z = outputs['z'] if isinstance(outputs, dict) else outputs
                logits = self.classifier_head(z)
            else:
                # Fallback simple model
                logits = self.model(X_tensor)
                
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            
        return probabilities
        
    def _evaluate(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Evaluate model accuracy."""
        self.model.eval()
        with torch.no_grad():
            if self.model_type == 'transformer':
                outputs = self.model(X)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            elif self.model_type == 'vae':
                outputs = self.model(X)
                z = outputs['z'] if isinstance(outputs, dict) else outputs
                logits = self.classifier_head(z)
            else:
                # Fallback simple model
                logits = self.model(X)
                
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == y).float().mean().item()
            
        self.model.train()
        return accuracy


class TrajectoryInference(BasePredictor):
    """
    Trajectory inference predictor for single-cell data.
    
    This predictor infers developmental trajectories and pseudotime
    from single-cell gene expression data.
    """
    
    def __init__(
        self,
        model_type: str = 'vae',
        latent_dim: int = 64,
        model_params: Optional[Dict[str, Any]] = None,
        device: str = 'auto'
    ):
        super().__init__(device=device)
        
        self.model_type = model_type
        self.latent_dim = latent_dim
        self.model_params = model_params or {}
        self.is_fitted = False
        
    def fit(
        self,
        adata: sc.AnnData,
        batch_size: int = 32,
        num_epochs: int = 100,
        learning_rate: float = 1e-3
    ):
        """
        Fit the trajectory inference model.
        
        Args:
            adata: Annotated data matrix with gene expression
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
        """
        # Extract data
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        
        # Set model parameters
        self.n_genes = X.shape[1]
        
        # Build model
        if self.model_type == 'vae':
            self.model = VariationalAutoEncoder(
                n_genes=self.n_genes,
                latent_dim=self.latent_dim,
                **self.model_params
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        self.model.to(self.device)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Setup training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            perm = torch.randperm(X_tensor.size(0))
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, X_tensor.size(0), batch_size):
                batch_indices = perm[i:i + batch_size]
                batch_X = X_tensor[batch_indices]
                
                optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                
                # VAE loss
                recon_loss = torch.nn.functional.mse_loss(
                    outputs['reconstruction'], batch_X, reduction='sum'
                )
                kl_loss = -0.5 * torch.sum(
                    1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp()
                )
                
                loss = recon_loss + kl_loss
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
                
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Loss={epoch_loss/n_batches:.4f}")
                
        self.is_fitted = True
        self.logger.info("Training completed")
        
    def predict(self, adata: sc.AnnData) -> Dict[str, np.ndarray]:
        """
        Predict trajectory and pseudotime.
        
        Args:
            adata: Annotated data matrix with gene expression
            
        Returns:
            Dictionary containing latent representation and pseudotime
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            latent = outputs['z'].cpu().numpy()
            
        # Compute pseudotime (simple approach using first principal component)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        pseudotime = pca.fit_transform(latent).flatten()
        
        # Normalize pseudotime to [0, 1]
        pseudotime = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min())
        
        return {
            'latent': latent,
            'pseudotime': pseudotime
        }


class GeneRegulatoryNetwork(BasePredictor):
    """
    Gene regulatory network inference predictor.
    
    This predictor infers gene-gene regulatory relationships
    from single-cell gene expression data.
    """
    
    def __init__(
        self,
        model_type: str = 'gnn',
        hidden_dim: int = 128,
        model_params: Optional[Dict[str, Any]] = None,
        device: str = 'auto'
    ):
        super().__init__(device=device)
        
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.model_params = model_params or {}
        self.is_fitted = False
        
    def fit(
        self,
        adata: sc.AnnData,
        batch_size: int = 32,
        num_epochs: int = 100,
        learning_rate: float = 1e-3
    ):
        """
        Fit the gene regulatory network model.
        
        Args:
            adata: Annotated data matrix with gene expression
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
        """
        # Extract data
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        
        # Set model parameters
        self.n_genes = X.shape[1]
        
        # Build initial gene-gene graph (correlation-based)
        correlation_matrix = np.corrcoef(X.T)
        threshold = 0.3  # Lower correlation threshold to ensure some edges
        
        # Create edge index
        edges = []
        for i in range(self.n_genes):
            for j in range(i + 1, self.n_genes):
                if abs(correlation_matrix[i, j]) > threshold:
                    edges.append([i, j])
                    edges.append([j, i])  # Undirected graph
        
        # If no edges found, create a minimal graph (connect each gene to next)
        if len(edges) == 0:
            print(f"No edges found with threshold {threshold}, creating minimal graph")
            for i in range(self.n_genes - 1):
                edges.append([i, i + 1])
                edges.append([i + 1, i])
        
        print(f"Created {len(edges)} edges for GRN")
        self.edge_index = torch.LongTensor(edges).T.to(self.device)
        
        # Build model
        if self.model_type == 'gnn':
            self.model = GraphNeuralNetwork(
                input_dim=1,  # Single gene expression value
                hidden_dim=self.hidden_dim,
                output_dim=1,  # Predict gene expression
                **self.model_params
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        self.model.to(self.device)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Setup training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            perm = torch.randperm(X_tensor.size(0))
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, X_tensor.size(0), batch_size):
                batch_indices = perm[i:i + batch_size]
                batch_X = X_tensor[batch_indices]
                
                optimizer.zero_grad()
                
                # For each sample in batch
                batch_loss = 0.0
                for sample_idx in range(batch_X.size(0)):
                    sample = batch_X[sample_idx]
                    
                    # Node features (gene expression values)
                    node_features = sample.unsqueeze(-1)  # (n_genes, 1)
                    
                    # Forward pass
                    predictions = self.model(node_features, self.edge_index)
                    
                    # Loss (reconstruct gene expression)
                    loss = criterion(predictions.squeeze(), sample)
                    batch_loss += loss
                    
                batch_loss = batch_loss / batch_X.size(0)
                batch_loss.backward()
                optimizer.step()
                
                epoch_loss += batch_loss.item()
                n_batches += 1
                
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Loss={epoch_loss/n_batches:.4f}")
                
        self.is_fitted = True
        self.logger.info("Training completed")
        
    def predict(self, adata: sc.AnnData) -> Dict[str, np.ndarray]:
        """
        Predict gene regulatory network.
        
        Args:
            adata: Annotated data matrix with gene expression
            
        Returns:
            Dictionary containing regulatory network adjacency matrix
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        # Extract learned edge weights from the model
        # This is a simplified approach - in practice, you'd want more sophisticated
        # methods to extract regulatory relationships
        
        edge_weights = []
        with torch.no_grad():
            # Get edge weights from the first graph convolution layer
            first_conv = self.model.convs[0]
            weights = first_conv.linear.weight.data.cpu().numpy()
            
        # Create adjacency matrix
        adjacency_matrix = np.zeros((self.n_genes, self.n_genes))
        
        # Fill adjacency matrix based on edge index and learned weights
        edge_index_np = self.edge_index.cpu().numpy()
        for i, (source, target) in enumerate(edge_index_np.T):
            # Use magnitude of weight as edge strength
            weight = np.abs(weights[0, 0])  # Simplified
            adjacency_matrix[source, target] = weight
            
        return {
            'adjacency_matrix': adjacency_matrix,
            'edge_index': edge_index_np
        } 