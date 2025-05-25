"""
Visualization utilities for protein prediction.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches


def plot_sequence_features(
    features: Dict[str, float],
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot protein sequence features.
    
    Args:
        features: Dictionary of sequence features
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Basic properties
    basic_props = ['length', 'molecular_weight', 'net_charge']
    basic_values = [features.get(prop, 0) for prop in basic_props]
    
    axes[0, 0].bar(basic_props, basic_values)
    axes[0, 0].set_title('Basic Properties')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Physicochemical ratios
    ratios = ['hydrophobic_ratio', 'polar_ratio', 'charged_ratio', 'aromatic_ratio']
    ratio_values = [features.get(ratio, 0) for ratio in ratios]
    
    axes[0, 1].bar(ratios, ratio_values)
    axes[0, 1].set_title('Physicochemical Ratios')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Amino acid composition
    aa_composition = {}
    for aa in 'ACDEFGHIKLMNPQRSTVWY':
        aa_composition[aa] = features.get(f'composition_{aa}', 0)
    
    aa_names = list(aa_composition.keys())
    aa_values = list(aa_composition.values())
    
    axes[1, 0].bar(aa_names, aa_values)
    axes[1, 0].set_title('Amino Acid Composition')
    axes[1, 0].set_ylabel('Frequency')
    
    # Pie chart of major categories
    categories = {
        'Hydrophobic': features.get('hydrophobic_ratio', 0),
        'Polar': features.get('polar_ratio', 0),
        'Charged': features.get('charged_ratio', 0),
        'Other': 1 - sum([features.get('hydrophobic_ratio', 0),
                         features.get('polar_ratio', 0),
                         features.get('charged_ratio', 0)])
    }
    
    axes[1, 1].pie(categories.values(), labels=categories.keys(), autopct='%1.1f%%')
    axes[1, 1].set_title('Physicochemical Categories')
    
    plt.tight_layout()
    return fig


def plot_secondary_structure(
    sequence: str,
    ss_prediction: str,
    confidence: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (15, 6)
) -> plt.Figure:
    """
    Plot secondary structure prediction.
    
    Args:
        sequence: Protein sequence
        ss_prediction: Secondary structure prediction string
        confidence: Confidence scores (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    
    # Color mapping for secondary structures
    ss_colors = {'H': 'red', 'E': 'blue', 'C': 'gray'}
    
    # Plot secondary structure
    for i, ss in enumerate(ss_prediction):
        color = ss_colors.get(ss, 'black')
        axes[0].bar(i, 1, color=color, alpha=0.7, width=1.0)
    
    axes[0].set_xlim(-0.5, len(sequence) - 0.5)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Secondary Structure')
    axes[0].set_title('Secondary Structure Prediction')
    
    # Create legend
    legend_elements = [
        mpatches.Patch(color='red', label='Helix (H)'),
        mpatches.Patch(color='blue', label='Sheet (E)'),
        mpatches.Patch(color='gray', label='Coil (C)')
    ]
    axes[0].legend(handles=legend_elements, loc='upper right')
    
    # Plot confidence if available
    if confidence is not None:
        axes[1].plot(range(len(confidence)), confidence, 'k-', linewidth=1)
        axes[1].fill_between(range(len(confidence)), confidence, alpha=0.3)
        axes[1].set_ylabel('Confidence')
        axes[1].set_ylim(0, 1)
    else:
        axes[1].text(0.5, 0.5, 'No confidence data available', 
                    ha='center', va='center', transform=axes[1].transAxes)
    
    axes[1].set_xlabel('Residue Position')
    axes[1].set_xlim(-0.5, len(sequence) - 0.5)
    
    plt.tight_layout()
    return fig


def plot_contact_map(
    contact_map: np.ndarray,
    sequence: Optional[str] = None,
    threshold: float = 0.5,
    figsize: Tuple[int, int] = (10, 10)
) -> plt.Figure:
    """
    Plot protein contact map.
    
    Args:
        contact_map: Contact map matrix
        sequence: Protein sequence (optional)
        threshold: Contact threshold for visualization
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot contact map
    im = ax.imshow(contact_map, cmap='Reds', vmin=0, vmax=1)
    
    # Add threshold contour
    if threshold > 0:
        ax.contour(contact_map, levels=[threshold], colors='blue', linewidths=1)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Contact Probability')
    
    # Labels
    ax.set_xlabel('Residue Index')
    ax.set_ylabel('Residue Index')
    ax.set_title('Protein Contact Map')
    
    # Add sequence information if available
    if sequence is not None:
        # Add amino acid labels (for small sequences)
        if len(sequence) <= 50:
            ax.set_xticks(range(len(sequence)))
            ax.set_yticks(range(len(sequence)))
            ax.set_xticklabels(list(sequence))
            ax.set_yticklabels(list(sequence))
        else:
            # Show every 10th residue for longer sequences
            step = max(1, len(sequence) // 20)
            ticks = range(0, len(sequence), step)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
    
    return fig


def plot_3d_structure(
    coordinates: np.ndarray,
    sequence: Optional[str] = None,
    secondary_structure: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Plot 3D protein structure.
    
    Args:
        coordinates: 3D coordinates array (n_residues, 3)
        sequence: Protein sequence (optional)
        secondary_structure: Secondary structure string (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
    
    # Color by secondary structure if available
    if secondary_structure is not None:
        colors = []
        color_map = {'H': 'red', 'E': 'blue', 'C': 'gray'}
        for ss in secondary_structure:
            colors.append(color_map.get(ss, 'black'))
        
        # Plot backbone
        ax.plot(x, y, z, 'k-', alpha=0.3, linewidth=1)
        
        # Plot residues colored by secondary structure
        scatter = ax.scatter(x, y, z, c=colors, s=50, alpha=0.8)
        
        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=8, label='Helix'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                      markersize=8, label='Sheet'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                      markersize=8, label='Coil')
        ]
        ax.legend(handles=legend_elements)
        
    else:
        # Plot backbone
        ax.plot(x, y, z, 'b-', alpha=0.7, linewidth=2)
        ax.scatter(x, y, z, c='red', s=30, alpha=0.8)
    
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title('3D Protein Structure')
    
    return fig


def plot_distance_matrix(
    coordinates: np.ndarray,
    figsize: Tuple[int, int] = (10, 10)
) -> plt.Figure:
    """
    Plot distance matrix from 3D coordinates.
    
    Args:
        coordinates: 3D coordinates array
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Compute distance matrix
    n_residues = coordinates.shape[0]
    distance_matrix = np.zeros((n_residues, n_residues))
    
    for i in range(n_residues):
        for j in range(n_residues):
            distance_matrix[i, j] = np.linalg.norm(coordinates[i] - coordinates[j])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot distance matrix
    im = ax.imshow(distance_matrix, cmap='viridis')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Distance (Å)')
    
    ax.set_xlabel('Residue Index')
    ax.set_ylabel('Residue Index')
    ax.set_title('Residue Distance Matrix')
    
    return fig


def plot_function_predictions(
    predictions: Dict[str, float],
    threshold: float = 0.5,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot protein function predictions.
    
    Args:
        predictions: Dictionary of function predictions
        threshold: Prediction threshold
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    functions = list(predictions.keys())
    scores = list(predictions.values())
    
    # Bar plot of all predictions
    colors = ['green' if score > threshold else 'red' for score in scores]
    bars = axes[0].bar(range(len(functions)), scores, color=colors, alpha=0.7)
    
    axes[0].axhline(y=threshold, color='black', linestyle='--', alpha=0.5)
    axes[0].set_xticks(range(len(functions)))
    axes[0].set_xticklabels(functions, rotation=45, ha='right')
    axes[0].set_ylabel('Prediction Score')
    axes[0].set_title('Function Predictions')
    axes[0].set_ylim(0, 1)
    
    # Add score labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Pie chart of top predictions
    top_predictions = {k: v for k, v in predictions.items() if v > threshold}
    
    if top_predictions:
        axes[1].pie(top_predictions.values(), labels=top_predictions.keys(), 
                   autopct='%1.2f', startangle=90)
        axes[1].set_title(f'Functions Above Threshold ({threshold})')
    else:
        axes[1].text(0.5, 0.5, f'No predictions\nabove threshold\n({threshold})', 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('No Significant Predictions')
    
    plt.tight_layout()
    return fig


def plot_drug_target_interactions(
    interactions: Dict[str, float],
    drug_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot drug-target interaction predictions.
    
    Args:
        interactions: Dictionary of interaction scores
        drug_names: List of drug names (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    if drug_names is None:
        drug_names = list(interactions.keys())
    
    scores = [interactions.get(drug, 0) for drug in drug_names]
    
    # Bar plot of interaction scores
    colors = plt.cm.viridis(np.array(scores))
    bars = axes[0].bar(range(len(drug_names)), scores, color=colors)
    
    axes[0].set_xticks(range(len(drug_names)))
    axes[0].set_xticklabels(drug_names, rotation=45, ha='right')
    axes[0].set_ylabel('Interaction Score')
    axes[0].set_title('Drug-Target Interactions')
    axes[0].set_ylim(0, 1)
    
    # Add score labels
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Histogram of interaction scores
    axes[1].hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1].set_xlabel('Interaction Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Interaction Scores')
    
    plt.tight_layout()
    return fig


def plot_msa_conservation(
    msa: List[str],
    conservation_scores: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (15, 8)
) -> plt.Figure:
    """
    Plot multiple sequence alignment and conservation.
    
    Args:
        msa: Multiple sequence alignment
        conservation_scores: Conservation scores for each position
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if not msa:
        raise ValueError("Empty MSA")
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    
    # Create MSA matrix
    seq_length = len(msa[0])
    n_sequences = len(msa)
    
    # Amino acid to number mapping for coloring
    aa_to_num = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY-X')}
    
    msa_matrix = np.zeros((n_sequences, seq_length))
    for i, seq in enumerate(msa):
        for j, aa in enumerate(seq):
            msa_matrix[i, j] = aa_to_num.get(aa, 21)  # 21 for unknown
    
    # Plot MSA
    im = axes[0].imshow(msa_matrix, cmap='tab20', aspect='auto')
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Sequence')
    axes[0].set_title('Multiple Sequence Alignment')
    
    # Limit y-axis labels for readability
    if n_sequences <= 20:
        axes[0].set_yticks(range(n_sequences))
        axes[0].set_yticklabels([f'Seq_{i}' for i in range(n_sequences)])
    
    # Plot conservation scores
    if conservation_scores is not None:
        axes[1].plot(range(seq_length), conservation_scores, 'b-', linewidth=2)
        axes[1].fill_between(range(seq_length), conservation_scores, alpha=0.3)
        axes[1].set_ylabel('Conservation')
        axes[1].set_ylim(0, 1)
    else:
        axes[1].text(0.5, 0.5, 'No conservation data available', 
                    ha='center', va='center', transform=axes[1].transAxes)
    
    axes[1].set_xlabel('Position')
    axes[1].set_xlim(0, seq_length - 1)
    
    plt.tight_layout()
    return fig


def plot_domain_architecture(
    sequence: str,
    domains: List[Dict[str, Any]],
    figsize: Tuple[int, int] = (15, 6)
) -> plt.Figure:
    """
    Plot protein domain architecture.
    
    Args:
        sequence: Protein sequence
        domains: List of domain predictions
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    seq_length = len(sequence)
    
    # Draw sequence backbone
    ax.plot([0, seq_length], [0, 0], 'k-', linewidth=3, alpha=0.5)
    
    # Color map for different domains
    domain_colors = plt.cm.Set3(np.linspace(0, 1, len(set(d['name'] for d in domains))))
    domain_color_map = {}
    
    # Plot domains
    y_offset = 0
    for i, domain in enumerate(domains):
        domain_name = domain['name']
        
        if domain_name not in domain_color_map:
            domain_color_map[domain_name] = domain_colors[len(domain_color_map)]
        
        color = domain_color_map[domain_name]
        
        # Draw domain rectangle
        start = domain['start']
        end = domain['end']
        width = end - start
        
        rect = Rectangle((start, y_offset - 0.2), width, 0.4, 
                        facecolor=color, alpha=0.7, edgecolor='black')
        ax.add_patch(rect)
        
        # Add domain label
        ax.text(start + width/2, y_offset, domain_name, 
               ha='center', va='center', fontsize=8, fontweight='bold')
        
        y_offset += 0.6  # Stack domains vertically if they overlap
    
    # Set limits and labels
    ax.set_xlim(-10, seq_length + 10)
    ax.set_ylim(-1, max(1, len(domains) * 0.6))
    ax.set_xlabel('Residue Position')
    ax.set_title('Protein Domain Architecture')
    
    # Remove y-axis ticks
    ax.set_yticks([])
    
    # Add sequence length annotation
    ax.text(seq_length/2, -0.8, f'Length: {seq_length} residues', 
           ha='center', va='center', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_prediction_confidence(
    predictions: np.ndarray,
    confidence: np.ndarray,
    labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot predictions vs confidence scores.
    
    Args:
        predictions: Prediction values
        confidence: Confidence scores
        labels: Labels for data points (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Scatter plot of predictions vs confidence
    scatter = axes[0].scatter(predictions, confidence, alpha=0.6, s=50)
    axes[0].set_xlabel('Prediction Score')
    axes[0].set_ylabel('Confidence Score')
    axes[0].set_title('Predictions vs Confidence')
    axes[0].grid(True, alpha=0.3)
    
    # Add diagonal line (perfect confidence)
    min_val = min(np.min(predictions), np.min(confidence))
    max_val = max(np.max(predictions), np.max(confidence))
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    
    # Histogram of confidence scores
    axes[1].hist(confidence, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    axes[1].set_xlabel('Confidence Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Confidence Scores')
    
    # Add statistics
    mean_conf = np.mean(confidence)
    std_conf = np.std(confidence)
    axes[1].axvline(mean_conf, color='red', linestyle='--', 
                   label=f'Mean: {mean_conf:.3f}')
    axes[1].axvline(mean_conf + std_conf, color='orange', linestyle='--', alpha=0.7,
                   label=f'±1 STD: {std_conf:.3f}')
    axes[1].axvline(mean_conf - std_conf, color='orange', linestyle='--', alpha=0.7)
    axes[1].legend()
    
    plt.tight_layout()
    return fig 