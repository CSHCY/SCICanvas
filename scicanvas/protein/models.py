"""
Neural network models for protein prediction.

This module implements state-of-the-art protein prediction models including
AlphaFold-inspired architectures for structure prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math
import numpy as np

from ..core.base import BaseModel


class ProteinTransformer(BaseModel):
    """
    Transformer model for protein sequence analysis.
    
    This model processes protein sequences using attention mechanisms
    to capture long-range dependencies in amino acid sequences.
    """
    
    def __init__(
        self,
        vocab_size: int = 21,  # 20 amino acids + unknown
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        d_ff: int = 2048,
        max_seq_length: int = 1024,
        dropout: float = 0.1,
        output_dim: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_length = max_seq_length
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Output projection
        if output_dim is not None:
            self.output_projection = nn.Linear(d_model, output_dim)
        else:
            self.output_projection = None
            
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the protein transformer.
        
        Args:
            input_ids: Amino acid sequence tokens (batch_size, seq_length)
            attention_mask: Attention mask (batch_size, seq_length)
            
        Returns:
            Dictionary containing model outputs
        """
        batch_size, seq_length = input_ids.shape
        
        # Create position indices
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        embeddings = token_embeds + position_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to boolean mask (True for positions to mask)
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None
            
        # Transformer encoding
        encoded = self.transformer(embeddings, src_key_padding_mask=src_key_padding_mask)
        
        outputs = {
            'last_hidden_state': encoded,
            'pooled_output': encoded.mean(dim=1)  # Global average pooling
        }
        
        # Output projection
        if self.output_projection is not None:
            outputs['logits'] = self.output_projection(encoded)
            
        return outputs


class ContactPredictor(BaseModel):
    """
    Contact map prediction model for protein structure.
    
    Predicts residue-residue contacts from protein sequences,
    which is a key component in structure prediction.
    """
    
    def __init__(
        self,
        vocab_size: int = 21,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        
        self.d_model = d_model
        
        # Protein sequence encoder
        self.sequence_encoder = ProteinTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout
        )
        
        # Contact prediction head
        self.contact_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for contact prediction.
        
        Args:
            input_ids: Amino acid sequence tokens
            attention_mask: Attention mask
            
        Returns:
            Dictionary containing contact predictions
        """
        # Encode sequence
        encoded_output = self.sequence_encoder(input_ids, attention_mask)
        sequence_repr = encoded_output['last_hidden_state']  # (batch, seq_len, d_model)
        
        batch_size, seq_len, d_model = sequence_repr.shape
        
        # Create pairwise representations
        # Expand to create all pairs
        repr_i = sequence_repr.unsqueeze(2).expand(-1, -1, seq_len, -1)  # (batch, seq_len, seq_len, d_model)
        repr_j = sequence_repr.unsqueeze(1).expand(-1, seq_len, -1, -1)  # (batch, seq_len, seq_len, d_model)
        
        # Concatenate pairwise features
        pairwise_repr = torch.cat([repr_i, repr_j], dim=-1)  # (batch, seq_len, seq_len, 2*d_model)
        
        # Predict contacts
        contact_logits = self.contact_head(pairwise_repr).squeeze(-1)  # (batch, seq_len, seq_len)
        
        # Make symmetric
        contact_probs = (contact_logits + contact_logits.transpose(-1, -2)) / 2
        
        return {
            'contact_map': contact_probs,
            'sequence_representation': sequence_repr
        }


class AlphaFoldModel(BaseModel):
    """
    AlphaFold-inspired model for protein structure prediction.
    
    This model combines multiple sequence alignment (MSA) processing,
    pairwise representations, and iterative refinement for structure prediction.
    """
    
    def __init__(
        self,
        vocab_size: int = 21,
        msa_dim: int = 256,
        pair_dim: int = 128,
        n_msa_layers: int = 4,
        n_pair_layers: int = 4,
        n_structure_layers: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_seq_length: int = 512,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        
        self.msa_dim = msa_dim
        self.pair_dim = pair_dim
        self.max_seq_length = max_seq_length
        
        # MSA embedding
        self.msa_embedding = nn.Embedding(vocab_size, msa_dim)
        self.msa_position_embedding = nn.Embedding(max_seq_length, msa_dim)
        
        # Pair representation initialization
        self.pair_embedding = nn.Linear(msa_dim * 2, pair_dim)
        
        # MSA processing stack
        self.msa_stack = nn.ModuleList([
            MSAAttentionBlock(msa_dim, n_heads, dropout)
            for _ in range(n_msa_layers)
        ])
        
        # Pair processing stack
        self.pair_stack = nn.ModuleList([
            PairAttentionBlock(pair_dim, n_heads, dropout)
            for _ in range(n_pair_layers)
        ])
        
        # Structure module
        self.structure_module = StructureModule(
            msa_dim=msa_dim,
            pair_dim=pair_dim,
            n_layers=n_structure_layers,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Output heads
        self.distance_head = nn.Sequential(
            nn.Linear(pair_dim, pair_dim // 2),
            nn.ReLU(),
            nn.Linear(pair_dim // 2, 64)  # Distance bins
        )
        
        self.angle_head = nn.Sequential(
            nn.Linear(msa_dim, msa_dim // 2),
            nn.ReLU(),
            nn.Linear(msa_dim // 2, 3)  # Phi, Psi, Omega angles
        )
        
    def forward(
        self,
        msa: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of AlphaFold model.
        
        Args:
            msa: Multiple sequence alignment (batch, n_seqs, seq_len)
            attention_mask: Attention mask
            
        Returns:
            Dictionary containing structure predictions
        """
        batch_size, n_seqs, seq_len = msa.shape
        
        # MSA embedding
        msa_repr = self.msa_embedding(msa)  # (batch, n_seqs, seq_len, msa_dim)
        
        # Add positional encoding
        pos_ids = torch.arange(seq_len, device=msa.device).unsqueeze(0).unsqueeze(0).expand(batch_size, n_seqs, -1)
        pos_embeds = self.msa_position_embedding(pos_ids)
        msa_repr = msa_repr + pos_embeds
        
        # Initialize pair representation
        # Use first sequence (target sequence) for pair initialization
        target_seq = msa_repr[:, 0]  # (batch, seq_len, msa_dim)
        pair_repr = self._init_pair_representation(target_seq)
        
        # Process MSA
        for msa_layer in self.msa_stack:
            msa_repr = msa_layer(msa_repr, attention_mask)
            
        # Process pairs
        for pair_layer in self.pair_stack:
            pair_repr = pair_layer(pair_repr, msa_repr)
            
        # Structure prediction
        structure_output = self.structure_module(msa_repr, pair_repr)
        
        # Predict distances and angles
        distance_logits = self.distance_head(pair_repr)
        angle_predictions = self.angle_head(msa_repr[:, 0])  # Use target sequence
        
        return {
            'msa_representation': msa_repr,
            'pair_representation': pair_repr,
            'distance_logits': distance_logits,
            'angle_predictions': angle_predictions,
            'coordinates': structure_output.get('coordinates'),
            'confidence': structure_output.get('confidence')
        }
        
    def _init_pair_representation(self, sequence_repr: torch.Tensor) -> torch.Tensor:
        """Initialize pairwise representation from sequence representation."""
        batch_size, seq_len, msa_dim = sequence_repr.shape
        
        # Create pairwise features
        repr_i = sequence_repr.unsqueeze(2).expand(-1, -1, seq_len, -1)
        repr_j = sequence_repr.unsqueeze(1).expand(-1, seq_len, -1, -1)
        
        # Concatenate and project
        pairwise_features = torch.cat([repr_i, repr_j], dim=-1)
        pair_repr = self.pair_embedding(pairwise_features)
        
        return pair_repr


class MSAAttentionBlock(nn.Module):
    """Multi-head attention block for MSA processing."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.row_attention = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.col_attention = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, msa: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of MSA attention block."""
        batch_size, n_seqs, seq_len, d_model = msa.shape
        
        # Row-wise attention (across sequences for each position)
        msa_reshaped = msa.transpose(1, 2).reshape(batch_size * seq_len, n_seqs, d_model)
        row_attn_out, _ = self.row_attention(msa_reshaped, msa_reshaped, msa_reshaped)
        row_attn_out = row_attn_out.reshape(batch_size, seq_len, n_seqs, d_model).transpose(1, 2)
        
        msa = self.layer_norm1(msa + self.dropout(row_attn_out))
        
        # Column-wise attention (across positions for each sequence)
        msa_reshaped = msa.reshape(batch_size * n_seqs, seq_len, d_model)
        col_attn_out, _ = self.col_attention(msa_reshaped, msa_reshaped, msa_reshaped)
        col_attn_out = col_attn_out.reshape(batch_size, n_seqs, seq_len, d_model)
        
        msa = self.layer_norm2(msa + self.dropout(col_attn_out))
        
        # Feed forward
        ff_out = self.feed_forward(msa)
        msa = self.layer_norm3(msa + self.dropout(ff_out))
        
        return msa


class PairAttentionBlock(nn.Module):
    """Attention block for pairwise representation processing."""
    
    def __init__(self, pair_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.triangle_attention_start = TriangleAttention(pair_dim, n_heads, dropout)
        self.triangle_attention_end = TriangleAttention(pair_dim, n_heads, dropout)
        self.triangle_multiplication_out = TriangleMultiplication(pair_dim, dropout)
        self.triangle_multiplication_in = TriangleMultiplication(pair_dim, dropout)
        
        self.layer_norms = nn.ModuleList([nn.LayerNorm(pair_dim) for _ in range(4)])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, pair_repr: torch.Tensor, msa_repr: torch.Tensor) -> torch.Tensor:
        """Forward pass of pair attention block."""
        # Triangle attention starting
        pair_repr = self.layer_norms[0](pair_repr + self.dropout(
            self.triangle_attention_start(pair_repr)
        ))
        
        # Triangle attention ending
        pair_repr = self.layer_norms[1](pair_repr + self.dropout(
            self.triangle_attention_end(pair_repr.transpose(-2, -3)).transpose(-2, -3)
        ))
        
        # Triangle multiplication outgoing
        pair_repr = self.layer_norms[2](pair_repr + self.dropout(
            self.triangle_multiplication_out(pair_repr)
        ))
        
        # Triangle multiplication incoming
        pair_repr = self.layer_norms[3](pair_repr + self.dropout(
            self.triangle_multiplication_in(pair_repr.transpose(-2, -3)).transpose(-2, -3)
        ))
        
        return pair_repr


class TriangleAttention(nn.Module):
    """Triangle attention mechanism for pair representations."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        
    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        """Apply triangle attention."""
        batch_size, seq_len, seq_len, d_model = pair_repr.shape
        
        # Reshape for attention
        pair_reshaped = pair_repr.reshape(batch_size * seq_len, seq_len, d_model)
        
        # Apply attention
        attn_out, _ = self.attention(pair_reshaped, pair_reshaped, pair_reshaped)
        
        # Reshape back
        return attn_out.reshape(batch_size, seq_len, seq_len, d_model)


class TriangleMultiplication(nn.Module):
    """Triangle multiplication for pair representations."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.linear_a = nn.Linear(d_model, d_model)
        self.linear_b = nn.Linear(d_model, d_model)
        self.linear_g = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        """Apply triangle multiplication."""
        a = self.linear_a(pair_repr)
        b = self.linear_b(pair_repr)
        g = torch.sigmoid(self.linear_g(pair_repr))
        
        # Triangle multiplication
        ab = torch.einsum('bijk,bikl->bijl', a, b)
        
        # Apply gating and output projection
        output = self.linear_out(g * ab)
        return self.dropout(output)


class StructureModule(nn.Module):
    """Structure module for 3D coordinate prediction."""
    
    def __init__(
        self,
        msa_dim: int,
        pair_dim: int,
        n_layers: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_layers = n_layers
        
        # Invariant point attention layers
        self.ipa_layers = nn.ModuleList([
            InvariantPointAttention(msa_dim, pair_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Backbone update
        self.backbone_update = nn.Sequential(
            nn.Linear(msa_dim, msa_dim),
            nn.ReLU(),
            nn.Linear(msa_dim, 6)  # 3 for translation, 3 for rotation
        )
        
        # Confidence prediction
        self.confidence_head = nn.Sequential(
            nn.Linear(msa_dim, msa_dim // 2),
            nn.ReLU(),
            nn.Linear(msa_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, msa_repr: torch.Tensor, pair_repr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass of structure module."""
        batch_size, n_seqs, seq_len, msa_dim = msa_repr.shape
        
        # Use target sequence (first in MSA)
        target_repr = msa_repr[:, 0]  # (batch, seq_len, msa_dim)
        
        # Initialize backbone frames (simplified)
        coordinates = torch.zeros(batch_size, seq_len, 3, device=msa_repr.device)
        
        # Iterative refinement
        for ipa_layer in self.ipa_layers:
            target_repr = ipa_layer(target_repr, pair_repr, coordinates)
            
            # Update coordinates
            backbone_update = self.backbone_update(target_repr)
            translation = backbone_update[..., :3]
            coordinates = coordinates + translation
            
        # Predict confidence
        confidence = self.confidence_head(target_repr).squeeze(-1)
        
        return {
            'coordinates': coordinates,
            'confidence': confidence
        }


class InvariantPointAttention(nn.Module):
    """Invariant Point Attention mechanism."""
    
    def __init__(self, msa_dim: int, pair_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.msa_dim = msa_dim
        self.n_heads = n_heads
        self.head_dim = msa_dim // n_heads
        
        # Query, key, value projections
        self.q_proj = nn.Linear(msa_dim, msa_dim)
        self.k_proj = nn.Linear(msa_dim, msa_dim)
        self.v_proj = nn.Linear(msa_dim, msa_dim)
        
        # Pair bias
        self.pair_bias = nn.Linear(pair_dim, n_heads)
        
        # Output projection
        self.out_proj = nn.Linear(msa_dim, msa_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        msa_repr: torch.Tensor, 
        pair_repr: torch.Tensor, 
        coordinates: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of IPA."""
        batch_size, seq_len, msa_dim = msa_repr.shape
        
        # Project to Q, K, V
        q = self.q_proj(msa_repr).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(msa_repr).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(msa_repr).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Compute attention scores
        scores = torch.einsum('bihd,bjhd->bhij', q, k) / math.sqrt(self.head_dim)
        
        # Add pair bias
        pair_bias = self.pair_bias(pair_repr).permute(0, 3, 1, 2)  # (batch, heads, seq, seq)
        scores = scores + pair_bias
        
        # Apply attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply to values
        out = torch.einsum('bhij,bjhd->bihd', attn_weights, v)
        out = out.reshape(batch_size, seq_len, msa_dim)
        
        # Output projection
        out = self.out_proj(out)
        
        return msa_repr + out 