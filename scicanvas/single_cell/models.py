"""
Neural network models for single-cell analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

from ..core.base import BaseModel


class SingleCellTransformer(BaseModel):
    """
    Transformer model for single-cell gene expression analysis.
    
    This model treats genes as tokens and cells as sequences,
    enabling attention-based analysis of gene expression patterns.
    """
    
    def __init__(
        self,
        n_genes: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 5000,
        n_cell_types: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        
        self.n_genes = n_genes
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_cell_types = n_cell_types
        
        # Gene embedding layer
        self.gene_embedding = nn.Linear(1, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Output heads
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        if n_cell_types is not None:
            self.classifier = nn.Linear(d_model, n_cell_types)
        else:
            self.classifier = None
            
        # Reconstruction head for autoencoder functionality
        self.reconstruction_head = nn.Linear(d_model, 1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the transformer.
        
        Args:
            x: Gene expression tensor of shape (batch_size, n_genes)
            mask: Optional attention mask
            
        Returns:
            Dictionary containing model outputs
        """
        batch_size, n_genes = x.shape
        
        # Reshape for transformer: (batch_size, n_genes, 1)
        x = x.unsqueeze(-1)
        
        # Gene embedding
        x = self.gene_embedding(x)  # (batch_size, n_genes, d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoding
        encoded = self.transformer(x, src_key_padding_mask=mask)
        
        # Global pooling for classification
        pooled = self.global_pool(encoded.transpose(1, 2)).squeeze(-1)  # (batch_size, d_model)
        
        outputs = {
            'encoded': encoded,
            'pooled': pooled
        }
        
        # Classification output
        if self.classifier is not None:
            outputs['logits'] = self.classifier(pooled)
            
        # Reconstruction output
        reconstruction = self.reconstruction_head(encoded).squeeze(-1)  # (batch_size, n_genes)
        outputs['reconstruction'] = reconstruction
        
        return outputs


class VariationalAutoEncoder(BaseModel):
    """
    Variational Autoencoder for single-cell data dimensionality reduction.
    
    This model learns a low-dimensional latent representation of single-cell
    gene expression data while preserving important biological variation.
    """
    
    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 64,
        hidden_dims: list = [512, 256, 128],
        dropout: float = 0.1,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = n_genes
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.mu_layer = nn.Linear(prev_dim, latent_dim)
        self.logvar_layer = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        decoder_layers.append(nn.Linear(prev_dim, n_genes))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters."""
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass of the VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        
        return {
            'reconstruction': reconstruction,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }


class GraphNeuralNetwork(BaseModel):
    """
    Graph Neural Network for single-cell analysis.
    
    This model operates on cell-cell or gene-gene graphs to capture
    relationships and perform tasks like cell type classification.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: Optional[int] = None,
        n_layers: int = 3,
        dropout: float = 0.1,
        aggregation: str = 'mean',
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.aggregation = aggregation
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GraphConvLayer(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(n_layers - 2):
            self.convs.append(GraphConvLayer(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
        # Output layer
        if output_dim is not None:
            self.convs.append(GraphConvLayer(hidden_dim, output_dim))
            self.batch_norms.append(nn.BatchNorm1d(output_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # Global pooling for graph-level predictions
        if aggregation == 'mean':
            self.global_pool = lambda x, batch: torch.mean(x, dim=0, keepdim=True)
        elif aggregation == 'max':
            self.global_pool = lambda x, batch: torch.max(x, dim=0, keepdim=True)[0]
        elif aggregation == 'sum':
            self.global_pool = lambda x, batch: torch.sum(x, dim=0, keepdim=True)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
            
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the GNN.
        
        Args:
            x: Node features (n_nodes, input_dim)
            edge_index: Edge indices (2, n_edges)
            batch: Batch assignment for nodes (optional)
            
        Returns:
            Node or graph-level representations
        """
        # Apply graph convolutions
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            
            if i < len(self.convs) - 1:  # No activation after last layer
                x = F.relu(x)
                x = self.dropout(x)
                
        # Global pooling if batch is provided (graph-level prediction)
        if batch is not None:
            x = self.global_pool(x, batch)
            
        return x


class GraphConvLayer(nn.Module):
    """Simple graph convolution layer."""
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Graph convolution operation.
        
        Args:
            x: Node features (n_nodes, in_dim)
            edge_index: Edge indices (2, n_edges)
            
        Returns:
            Updated node features (n_nodes, out_dim)
        """
        # Simple message passing: aggregate neighbor features
        row, col = edge_index
        
        # Transform features
        x = self.linear(x)
        
        # Aggregate messages from neighbors
        out = torch.zeros_like(x)
        out.index_add_(0, row, x[col])
        
        # Add self-loops (optional)
        out = out + x
        
        return out


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:x.size(1), :].transpose(0, 1) 