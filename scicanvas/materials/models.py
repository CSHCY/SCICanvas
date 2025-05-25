"""
Neural network models for materials science prediction.

This module implements state-of-the-art models for crystal structure prediction,
materials property prediction, and catalyst design.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math
import numpy as np

from ..core.base import BaseModel


class CrystalGraphConvNet(BaseModel):
    """
    Crystal Graph Convolutional Network for materials property prediction.
    
    This model uses graph neural networks to predict properties of crystalline materials
    from their atomic structure and bonding information.
    """
    
    def __init__(
        self,
        atom_features_dim: int = 92,  # Number of elements in periodic table
        bond_features_dim: int = 41,  # Bond feature dimension
        hidden_dim: int = 128,
        n_conv_layers: int = 6,
        n_hidden_layers: int = 3,
        output_dim: int = 1,
        pooling_method: str = "global_mean",
        dropout: float = 0.1,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        
        self.atom_features_dim = atom_features_dim
        self.bond_features_dim = bond_features_dim
        self.hidden_dim = hidden_dim
        self.n_conv_layers = n_conv_layers
        self.pooling_method = pooling_method
        
        # Atom embedding
        self.atom_embedding = nn.Linear(atom_features_dim, hidden_dim)
        
        # Bond embedding
        self.bond_embedding = nn.Linear(bond_features_dim, hidden_dim)
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList([
            CrystalGraphConvLayer(hidden_dim, hidden_dim)
            for _ in range(n_conv_layers)
        ])
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim)
            for _ in range(n_conv_layers)
        ])
        
        # Output layers
        output_layers = []
        for i in range(n_hidden_layers):
            if i == 0:
                output_layers.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                output_layers.append(nn.Linear(hidden_dim, hidden_dim))
            output_layers.append(nn.ReLU())
            output_layers.append(nn.Dropout(dropout))
        
        output_layers.append(nn.Linear(hidden_dim, output_dim))
        self.output_layers = nn.Sequential(*output_layers)
        
    def forward(
        self,
        atom_features: torch.Tensor,
        bond_features: torch.Tensor,
        bond_indices: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the crystal graph convolution network.
        
        Args:
            atom_features: Atom feature matrix (n_atoms, atom_features_dim)
            bond_features: Bond feature matrix (n_bonds, bond_features_dim)
            bond_indices: Bond connectivity (n_bonds, 2)
            batch_indices: Batch indices for atoms (n_atoms,)
            
        Returns:
            Dictionary containing model outputs
        """
        # Embed atom and bond features
        atom_repr = self.atom_embedding(atom_features)
        bond_repr = self.bond_embedding(bond_features)
        
        # Apply graph convolution layers
        for conv_layer, batch_norm in zip(self.conv_layers, self.batch_norms):
            atom_repr_new = conv_layer(atom_repr, bond_repr, bond_indices)
            atom_repr = batch_norm(atom_repr_new) + atom_repr  # Residual connection
            atom_repr = F.relu(atom_repr)
        
        # Global pooling
        if batch_indices is not None:
            # Batch processing
            graph_repr = self._global_pool(atom_repr, batch_indices)
        else:
            # Single graph
            if self.pooling_method == "global_mean":
                graph_repr = torch.mean(atom_repr, dim=0, keepdim=True)
            elif self.pooling_method == "global_max":
                graph_repr = torch.max(atom_repr, dim=0, keepdim=True)[0]
            elif self.pooling_method == "global_sum":
                graph_repr = torch.sum(atom_repr, dim=0, keepdim=True)
            else:
                raise ValueError(f"Unknown pooling method: {self.pooling_method}")
        
        # Output prediction
        output = self.output_layers(graph_repr)
        
        return {
            'predictions': output,
            'atom_representations': atom_repr,
            'graph_representation': graph_repr
        }
    
    def _global_pool(self, atom_repr: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
        """Global pooling for batched graphs."""
        batch_size = batch_indices.max().item() + 1
        pooled_repr = []
        
        for i in range(batch_size):
            mask = batch_indices == i
            atoms_in_graph = atom_repr[mask]
            
            if self.pooling_method == "global_mean":
                pooled = torch.mean(atoms_in_graph, dim=0)
            elif self.pooling_method == "global_max":
                pooled = torch.max(atoms_in_graph, dim=0)[0]
            elif self.pooling_method == "global_sum":
                pooled = torch.sum(atoms_in_graph, dim=0)
            else:
                raise ValueError(f"Unknown pooling method: {self.pooling_method}")
            
            pooled_repr.append(pooled)
        
        return torch.stack(pooled_repr)


class CrystalGraphConvLayer(nn.Module):
    """Single crystal graph convolution layer."""
    
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        # Linear transformations
        self.linear_node = nn.Linear(node_dim, node_dim)
        self.linear_edge = nn.Linear(edge_dim, node_dim)
        self.linear_out = nn.Linear(node_dim, node_dim)
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of crystal graph convolution layer.
        
        Args:
            node_features: Node features (n_nodes, node_dim)
            edge_features: Edge features (n_edges, edge_dim)
            edge_indices: Edge connectivity (n_edges, 2)
            
        Returns:
            Updated node features
        """
        n_nodes = node_features.size(0)
        
        # Transform node features
        node_transformed = self.linear_node(node_features)
        
        # Transform edge features
        edge_transformed = self.linear_edge(edge_features)
        
        # Aggregate messages
        aggregated = torch.zeros_like(node_transformed)
        
        for i in range(edge_indices.size(0)):
            src, dst = edge_indices[i]
            message = node_transformed[src] * edge_transformed[i]
            aggregated[dst] += message
        
        # Output transformation
        output = self.linear_out(aggregated)
        
        return output


class MaterialsTransformer(BaseModel):
    """
    Transformer model for materials property prediction.
    
    This model treats atoms as tokens and uses attention mechanisms
    to capture long-range interactions in materials.
    """
    
    def __init__(
        self,
        atom_vocab_size: int = 118,  # Number of elements
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        d_ff: int = 2048,
        max_atoms: int = 200,
        output_dim: int = 1,
        dropout: float = 0.1,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        
        self.atom_vocab_size = atom_vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_atoms = max_atoms
        
        # Atom embedding
        self.atom_embedding = nn.Embedding(atom_vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_atoms, d_model)
        
        # Coordinate embedding (for 3D position information)
        self.coord_embedding = nn.Linear(3, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Output layers
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        atom_types: torch.Tensor,
        coordinates: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the materials transformer.
        
        Args:
            atom_types: Atomic numbers (batch_size, n_atoms)
            coordinates: 3D coordinates (batch_size, n_atoms, 3)
            attention_mask: Attention mask (batch_size, n_atoms)
            
        Returns:
            Dictionary containing model outputs
        """
        batch_size, n_atoms = atom_types.shape
        
        # Create position indices
        position_ids = torch.arange(n_atoms, device=atom_types.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        atom_embeds = self.atom_embedding(atom_types)
        position_embeds = self.position_embedding(position_ids)
        coord_embeds = self.coord_embedding(coordinates)
        
        # Combine embeddings
        embeddings = atom_embeds + position_embeds + coord_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Create attention mask for transformer
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None
        
        # Transformer encoding
        encoded = self.transformer(embeddings, src_key_padding_mask=src_key_padding_mask)
        
        # Global pooling
        if attention_mask is not None:
            # Masked average pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(encoded)
            pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled = encoded.mean(dim=1)
        
        # Output prediction
        output = self.output_head(pooled)
        
        return {
            'predictions': output,
            'atom_representations': encoded,
            'pooled_representation': pooled
        }


class CatalystDesignNet(BaseModel):
    """
    Neural network for catalyst design and optimization.
    
    This model predicts catalytic activity and selectivity from
    catalyst structure and reaction conditions.
    """
    
    def __init__(
        self,
        atom_features_dim: int = 92,
        reaction_features_dim: int = 50,
        hidden_dim: int = 256,
        n_gnn_layers: int = 4,
        n_mlp_layers: int = 3,
        output_dim: int = 2,  # Activity and selectivity
        dropout: float = 0.1,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        
        # Catalyst structure encoder (GNN)
        self.atom_embedding = nn.Linear(atom_features_dim, hidden_dim)
        
        self.gnn_layers = nn.ModuleList([
            CatalystGNNLayer(hidden_dim)
            for _ in range(n_gnn_layers)
        ])
        
        # Reaction condition encoder
        self.reaction_encoder = nn.Sequential(
            nn.Linear(reaction_features_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layers
        output_layers = []
        for i in range(n_mlp_layers):
            if i == 0:
                output_layers.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                output_layers.append(nn.Linear(hidden_dim, hidden_dim))
            output_layers.append(nn.ReLU())
            output_layers.append(nn.Dropout(dropout))
        
        output_layers.append(nn.Linear(hidden_dim, output_dim))
        self.output_layers = nn.Sequential(*output_layers)
        
    def forward(
        self,
        atom_features: torch.Tensor,
        bond_indices: torch.Tensor,
        reaction_features: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of catalyst design network.
        
        Args:
            atom_features: Catalyst atom features
            bond_indices: Bond connectivity
            reaction_features: Reaction condition features
            batch_indices: Batch indices for atoms
            
        Returns:
            Dictionary containing predictions
        """
        # Encode catalyst structure
        atom_repr = self.atom_embedding(atom_features)
        
        for gnn_layer in self.gnn_layers:
            atom_repr = gnn_layer(atom_repr, bond_indices) + atom_repr  # Residual
            atom_repr = F.relu(atom_repr)
        
        # Global pooling for catalyst representation
        if batch_indices is not None:
            catalyst_repr = self._global_pool(atom_repr, batch_indices)
        else:
            catalyst_repr = torch.mean(atom_repr, dim=0, keepdim=True)
        
        # Encode reaction conditions
        reaction_repr = self.reaction_encoder(reaction_features)
        
        # Fuse catalyst and reaction representations
        fused_repr = self.fusion_layer(torch.cat([catalyst_repr, reaction_repr], dim=-1))
        
        # Predict activity and selectivity
        predictions = self.output_layers(fused_repr)
        
        return {
            'predictions': predictions,
            'catalyst_representation': catalyst_repr,
            'reaction_representation': reaction_repr,
            'fused_representation': fused_repr
        }
    
    def _global_pool(self, atom_repr: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
        """Global pooling for batched graphs."""
        batch_size = batch_indices.max().item() + 1
        pooled_repr = []
        
        for i in range(batch_size):
            mask = batch_indices == i
            atoms_in_graph = atom_repr[mask]
            pooled = torch.mean(atoms_in_graph, dim=0)
            pooled_repr.append(pooled)
        
        return torch.stack(pooled_repr)


class CatalystGNNLayer(nn.Module):
    """Graph neural network layer for catalyst representation."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, node_features: torch.Tensor, edge_indices: torch.Tensor) -> torch.Tensor:
        """Forward pass of catalyst GNN layer."""
        n_nodes = node_features.size(0)
        
        # Compute messages
        messages = torch.zeros_like(node_features)
        
        for i in range(edge_indices.size(0)):
            src, dst = edge_indices[i]
            edge_input = torch.cat([node_features[src], node_features[dst]], dim=-1)
            message = self.message_net(edge_input)
            messages[dst] += message
        
        # Update node features
        update_input = torch.cat([node_features, messages], dim=-1)
        updated_features = self.update_net(update_input)
        
        return updated_features


class PhasePredictor(BaseModel):
    """
    Neural network for predicting phase diagrams and phase transitions.
    
    This model predicts stable phases and phase boundaries from
    composition and thermodynamic conditions.
    """
    
    def __init__(
        self,
        composition_dim: int = 10,  # Number of elements in composition
        condition_dim: int = 3,    # Temperature, pressure, etc.
        hidden_dim: int = 256,
        n_layers: int = 6,
        n_phases: int = 5,         # Number of possible phases
        dropout: float = 0.1,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        
        self.composition_dim = composition_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.n_phases = n_phases
        
        # Composition encoder
        self.composition_encoder = nn.Sequential(
            nn.Linear(composition_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Condition encoder
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Fusion and prediction layers
        fusion_layers = []
        for i in range(n_layers):
            if i == 0:
                fusion_layers.append(nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim))
            else:
                fusion_layers.append(nn.Linear(hidden_dim, hidden_dim))
            fusion_layers.append(nn.ReLU())
            fusion_layers.append(nn.Dropout(dropout))
        
        self.fusion_layers = nn.Sequential(*fusion_layers)
        
        # Phase prediction head
        self.phase_head = nn.Linear(hidden_dim, n_phases)
        
        # Stability prediction head
        self.stability_head = nn.Linear(hidden_dim, 1)
        
    def forward(
        self,
        composition: torch.Tensor,
        conditions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of phase predictor.
        
        Args:
            composition: Material composition (batch_size, composition_dim)
            conditions: Thermodynamic conditions (batch_size, condition_dim)
            
        Returns:
            Dictionary containing phase predictions
        """
        # Encode composition and conditions
        comp_repr = self.composition_encoder(composition)
        cond_repr = self.condition_encoder(conditions)
        
        # Fuse representations
        fused_repr = torch.cat([comp_repr, cond_repr], dim=-1)
        fused_repr = self.fusion_layers(fused_repr)
        
        # Predict phases and stability
        phase_logits = self.phase_head(fused_repr)
        stability_score = self.stability_head(fused_repr)
        
        return {
            'phase_logits': phase_logits,
            'phase_probabilities': F.softmax(phase_logits, dim=-1),
            'stability_score': stability_score,
            'composition_representation': comp_repr,
            'condition_representation': cond_repr,
            'fused_representation': fused_repr
        }


class ElectronicStructureNet(BaseModel):
    """
    Neural network for predicting electronic structure properties.
    
    This model predicts band gaps, density of states, and other
    electronic properties from crystal structure.
    """
    
    def __init__(
        self,
        atom_features_dim: int = 92,
        hidden_dim: int = 256,
        n_conv_layers: int = 8,
        n_output_properties: int = 3,  # Band gap, DOS, etc.
        dropout: float = 0.1,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        
        self.hidden_dim = hidden_dim
        self.n_conv_layers = n_conv_layers
        
        # Atom embedding
        self.atom_embedding = nn.Linear(atom_features_dim, hidden_dim)
        
        # Graph convolution layers with attention
        self.conv_layers = nn.ModuleList([
            ElectronicStructureConvLayer(hidden_dim)
            for _ in range(n_conv_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(n_conv_layers)
        ])
        
        # Property-specific heads
        self.band_gap_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.dos_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 100)  # DOS spectrum
        )
        
        self.fermi_level_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(
        self,
        atom_features: torch.Tensor,
        bond_indices: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of electronic structure network.
        
        Args:
            atom_features: Atom features
            bond_indices: Bond connectivity
            batch_indices: Batch indices for atoms
            
        Returns:
            Dictionary containing electronic structure predictions
        """
        # Embed atom features
        atom_repr = self.atom_embedding(atom_features)
        
        # Apply graph convolution layers
        for conv_layer, layer_norm in zip(self.conv_layers, self.layer_norms):
            atom_repr_new = conv_layer(atom_repr, bond_indices)
            atom_repr = layer_norm(atom_repr_new + atom_repr)  # Residual + LayerNorm
        
        # Global pooling
        if batch_indices is not None:
            graph_repr = self._global_pool(atom_repr, batch_indices)
        else:
            graph_repr = torch.mean(atom_repr, dim=0, keepdim=True)
        
        # Predict electronic properties
        band_gap = self.band_gap_head(graph_repr)
        dos_spectrum = self.dos_head(graph_repr)
        fermi_level = self.fermi_level_head(graph_repr)
        
        return {
            'band_gap': band_gap,
            'dos_spectrum': dos_spectrum,
            'fermi_level': fermi_level,
            'atom_representations': atom_repr,
            'graph_representation': graph_repr
        }
    
    def _global_pool(self, atom_repr: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
        """Global pooling for batched graphs."""
        batch_size = batch_indices.max().item() + 1
        pooled_repr = []
        
        for i in range(batch_size):
            mask = batch_indices == i
            atoms_in_graph = atom_repr[mask]
            pooled = torch.mean(atoms_in_graph, dim=0)
            pooled_repr.append(pooled)
        
        return torch.stack(pooled_repr)


class ElectronicStructureConvLayer(nn.Module):
    """Graph convolution layer with attention for electronic structure."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Message passing
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update network
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, node_features: torch.Tensor, edge_indices: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention-based message passing."""
        n_nodes = node_features.size(0)
        
        # Self-attention
        node_features_attended, _ = self.attention(
            node_features.unsqueeze(0), 
            node_features.unsqueeze(0), 
            node_features.unsqueeze(0)
        )
        node_features_attended = node_features_attended.squeeze(0)
        
        # Message passing
        messages = torch.zeros_like(node_features)
        
        for i in range(edge_indices.size(0)):
            src, dst = edge_indices[i]
            edge_input = torch.cat([node_features_attended[src], node_features_attended[dst]], dim=-1)
            message = self.message_net(edge_input)
            messages[dst] += message
        
        # Update node features
        update_input = torch.cat([node_features_attended, messages], dim=-1)
        updated_features = self.update_net(update_input)
        
        return updated_features 