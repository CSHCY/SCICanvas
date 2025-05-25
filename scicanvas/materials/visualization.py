"""
Visualization utilities for materials science.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def plot_crystal_structure(
    structure: Dict[str, Any],
    save_path: Optional[str] = None,
    show_bonds: bool = True,
    show_labels: bool = False,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot 3D crystal structure.
    
    Args:
        structure: Crystal structure dictionary
        save_path: Path to save the plot
        show_bonds: Whether to show bonds
        show_labels: Whether to show atom labels
        figsize: Figure size
    """
    coordinates = np.array(structure['coordinates'])
    atom_types = np.array(structure['atom_types'])
    
    # Element colors (simplified)
    element_colors = {
        1: 'white',      # H
        6: 'gray',       # C
        8: 'red',        # O
        11: 'purple',    # Na
        12: 'green',     # Mg
        13: 'pink',      # Al
        14: 'yellow',    # Si
        16: 'orange',    # S
        20: 'lightgreen', # Ca
        22: 'silver',    # Ti
        26: 'brown',     # Fe
        29: 'orange',    # Cu
        30: 'blue',      # Zn
        31: 'cyan',      # Ga
        33: 'magenta'    # As
    }
    
    # Element sizes (relative)
    element_sizes = {
        1: 30,   # H
        6: 50,   # C
        8: 60,   # O
        11: 80,  # Na
        12: 70,  # Mg
        13: 75,  # Al
        14: 65,  # Si
        16: 65,  # S
        20: 90,  # Ca
        22: 75,  # Ti
        26: 70,  # Fe
        29: 70,  # Cu
        30: 70,  # Zn
        31: 75,  # Ga
        33: 70   # As
    }
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot atoms
    for i, (coord, atom_type) in enumerate(zip(coordinates, atom_types)):
        color = element_colors.get(atom_type, 'gray')
        size = element_sizes.get(atom_type, 50)
        
        ax.scatter(coord[0], coord[1], coord[2], 
                  c=color, s=size, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        if show_labels:
            ax.text(coord[0], coord[1], coord[2], f'{atom_type}', fontsize=8)
    
    # Plot bonds
    if show_bonds and 'bond_indices' in structure:
        bond_indices = structure['bond_indices']
        for bond in bond_indices:
            i, j = bond
            if i < len(coordinates) and j < len(coordinates):
                coord1 = coordinates[i]
                coord2 = coordinates[j]
                ax.plot([coord1[0], coord2[0]], 
                       [coord1[1], coord2[1]], 
                       [coord1[2], coord2[2]], 
                       'k-', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title('Crystal Structure')
    
    # Make axes equal
    max_range = np.array([coordinates[:,0].max()-coordinates[:,0].min(),
                         coordinates[:,1].max()-coordinates[:,1].min(),
                         coordinates[:,2].max()-coordinates[:,2].min()]).max() / 2.0
    mid_x = (coordinates[:,0].max()+coordinates[:,0].min()) * 0.5
    mid_y = (coordinates[:,1].max()+coordinates[:,1].min()) * 0.5
    mid_z = (coordinates[:,2].max()+coordinates[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_property_distribution(
    properties: List[float],
    property_name: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot distribution of material properties.
    
    Args:
        properties: List of property values
        property_name: Name of the property
        save_path: Path to save the plot
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    ax1.hist(properties, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel(property_name)
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Distribution of {property_name}')
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(properties, vert=True)
    ax2.set_ylabel(property_name)
    ax2.set_title(f'Box Plot of {property_name}')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    mean_val = np.mean(properties)
    std_val = np.std(properties)
    median_val = np.median(properties)
    
    stats_text = f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nMedian: {median_val:.3f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_property_correlation(
    properties_dict: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> None:
    """
    Plot correlation matrix of material properties.
    
    Args:
        properties_dict: Dictionary of property names and values
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Create DataFrame
    df = pd.DataFrame(properties_dict)
    
    # Compute correlation matrix
    corr_matrix = df.corr()
    
    # Create heatmap
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    
    plt.title('Material Properties Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_phase_diagram(
    compositions: List[List[float]],
    phases: List[int],
    conditions: Optional[List[List[float]]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot phase diagram.
    
    Args:
        compositions: List of material compositions
        phases: List of phase indices
        conditions: List of thermodynamic conditions (optional)
        save_path: Path to save the plot
        figsize: Figure size
    """
    compositions = np.array(compositions)
    phases = np.array(phases)
    
    if compositions.shape[1] == 2:
        # Binary phase diagram
        plt.figure(figsize=figsize)
        
        unique_phases = np.unique(phases)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_phases)))
        
        for i, phase in enumerate(unique_phases):
            mask = phases == phase
            plt.scatter(compositions[mask, 0], compositions[mask, 1], 
                       c=[colors[i]], label=f'Phase {phase}', alpha=0.7, s=50)
        
        plt.xlabel('Component A Fraction')
        plt.ylabel('Component B Fraction')
        plt.title('Binary Phase Diagram')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    elif compositions.shape[1] >= 3:
        # Ternary phase diagram (project to 2D)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        unique_phases = np.unique(phases)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_phases)))
        
        for i, phase in enumerate(unique_phases):
            mask = phases == phase
            ax.scatter(compositions[mask, 0], compositions[mask, 1], compositions[mask, 2],
                      c=[colors[i]], label=f'Phase {phase}', alpha=0.7, s=50)
        
        ax.set_xlabel('Component A Fraction')
        ax.set_ylabel('Component B Fraction')
        ax.set_zlabel('Component C Fraction')
        ax.set_title('Ternary Phase Diagram')
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_band_structure(
    band_gaps: List[float],
    dos_spectra: Optional[List[np.ndarray]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot electronic band structure and density of states.
    
    Args:
        band_gaps: List of band gap values
        dos_spectra: List of density of states spectra (optional)
        save_path: Path to save the plot
        figsize: Figure size
    """
    if dos_spectra is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0]//2, figsize[1]))
    
    # Band gap distribution
    ax1.hist(band_gaps, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
    ax1.set_xlabel('Band Gap (eV)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Band Gap Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Add vertical lines for material types
    ax1.axvline(x=0.1, color='red', linestyle='--', alpha=0.7, label='Metal/Semiconductor')
    ax1.axvline(x=3.0, color='blue', linestyle='--', alpha=0.7, label='Semiconductor/Insulator')
    ax1.legend()
    
    # Density of states
    if dos_spectra is not None:
        energy_range = np.linspace(-10, 10, len(dos_spectra[0]))
        
        # Plot average DOS
        avg_dos = np.mean(dos_spectra, axis=0)
        std_dos = np.std(dos_spectra, axis=0)
        
        ax2.plot(energy_range, avg_dos, 'b-', linewidth=2, label='Average DOS')
        ax2.fill_between(energy_range, avg_dos - std_dos, avg_dos + std_dos, 
                        alpha=0.3, color='blue', label='±1 std')
        
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Fermi Level')
        ax2.set_xlabel('Energy (eV)')
        ax2.set_ylabel('Density of States')
        ax2.set_title('Average Density of States')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_catalyst_performance(
    activities: List[float],
    selectivities: List[float],
    catalyst_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot catalyst activity vs selectivity.
    
    Args:
        activities: List of catalytic activities
        selectivities: List of selectivities
        catalyst_names: List of catalyst names (optional)
        save_path: Path to save the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Scatter plot
    scatter = plt.scatter(activities, selectivities, alpha=0.7, s=100, 
                         c=range(len(activities)), cmap='viridis', edgecolors='black')
    
    # Add labels if provided
    if catalyst_names:
        for i, name in enumerate(catalyst_names):
            plt.annotate(name, (activities[i], selectivities[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Activity')
    plt.ylabel('Selectivity')
    plt.title('Catalyst Performance: Activity vs Selectivity')
    plt.grid(True, alpha=0.3)
    
    # Add ideal region
    plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='High Selectivity')
    plt.axvline(x=0.8, color='blue', linestyle='--', alpha=0.5, label='High Activity')
    plt.legend()
    
    # Color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Catalyst Index')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_structure_features(
    features_dict: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Plot structural features analysis.
    
    Args:
        features_dict: Dictionary of feature names and values
        save_path: Path to save the plot
        figsize: Figure size
    """
    n_features = len(features_dict)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, (feature_name, values) in enumerate(features_dict.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Histogram
        ax.hist(values, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax.set_xlabel(feature_name)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {feature_name}')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
        ax.legend()
    
    # Hide unused subplots
    for i in range(len(features_dict), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_radial_distribution_function(
    rdf_data: List[Tuple[np.ndarray, np.ndarray]],
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot radial distribution functions.
    
    Args:
        rdf_data: List of (r_values, rdf_values) tuples
        labels: List of labels for each RDF
        save_path: Path to save the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(rdf_data)))
    
    for i, (r_values, rdf_values) in enumerate(rdf_data):
        label = labels[i] if labels else f'Structure {i+1}'
        plt.plot(r_values, rdf_values, color=colors[i], linewidth=2, label=label)
    
    plt.xlabel('Distance (Å)')
    plt.ylabel('g(r)')
    plt.title('Radial Distribution Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max(r_values))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_formation_energy_vs_composition(
    compositions: List[List[float]],
    formation_energies: List[float],
    element_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot formation energy vs composition.
    
    Args:
        compositions: List of material compositions
        formation_energies: List of formation energies
        element_names: List of element names
        save_path: Path to save the plot
        figsize: Figure size
    """
    compositions = np.array(compositions)
    
    if compositions.shape[1] == 2:
        # Binary system
        plt.figure(figsize=figsize)
        
        # Sort by composition for smooth line
        sort_idx = np.argsort(compositions[:, 0])
        sorted_comp = compositions[sort_idx, 0]
        sorted_energy = np.array(formation_energies)[sort_idx]
        
        plt.scatter(sorted_comp, sorted_energy, alpha=0.7, s=50, color='blue')
        plt.plot(sorted_comp, sorted_energy, 'r-', alpha=0.5, linewidth=1)
        
        element_a = element_names[0] if element_names else 'A'
        element_b = element_names[1] if element_names else 'B'
        
        plt.xlabel(f'{element_a} Fraction')
        plt.ylabel('Formation Energy (eV/atom)')
        plt.title(f'Formation Energy vs Composition ({element_a}-{element_b})')
        plt.grid(True, alpha=0.3)
        
    else:
        # Multi-component system - use PCA for visualization
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2)
        comp_2d = pca.fit_transform(compositions)
        
        plt.figure(figsize=figsize)
        scatter = plt.scatter(comp_2d[:, 0], comp_2d[:, 1], 
                            c=formation_energies, cmap='viridis', s=50, alpha=0.7)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
        plt.title('Formation Energy vs Composition (PCA)')
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Formation Energy (eV/atom)')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_prediction_results(
    true_values: List[float],
    predicted_values: List[float],
    property_name: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot prediction results vs true values.
    
    Args:
        true_values: True property values
        predicted_values: Predicted property values
        property_name: Name of the property
        save_path: Path to save the plot
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Scatter plot
    ax1.scatter(true_values, predicted_values, alpha=0.6, s=50)
    
    # Perfect prediction line
    min_val = min(min(true_values), min(predicted_values))
    max_val = max(max(true_values), max(predicted_values))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax1.set_xlabel(f'True {property_name}')
    ax1.set_ylabel(f'Predicted {property_name}')
    ax1.set_title(f'Prediction Results: {property_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = np.array(predicted_values) - np.array(true_values)
    ax2.scatter(true_values, residuals, alpha=0.6, s=50)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    
    ax2.set_xlabel(f'True {property_name}')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals Plot')
    ax2.grid(True, alpha=0.3)
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)
    
    metrics_text = f'MSE: {mse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}'
    ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_materials_database_overview(
    database: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 12)
) -> None:
    """
    Plot overview of materials database.
    
    Args:
        database: Materials database DataFrame
        save_path: Path to save the plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    # 1. Number of atoms distribution
    if 'n_atoms' in database.columns:
        axes[0].hist(database['n_atoms'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_xlabel('Number of Atoms')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of System Sizes')
        axes[0].grid(True, alpha=0.3)
    
    # 2. Formation energy distribution
    if 'formation_energy' in database.columns:
        axes[1].hist(database['formation_energy'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1].set_xlabel('Formation Energy (eV/atom)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Formation Energy Distribution')
        axes[1].grid(True, alpha=0.3)
    
    # 3. Band gap distribution
    if 'band_gap' in database.columns:
        axes[2].hist(database['band_gap'], bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[2].set_xlabel('Band Gap (eV)')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Band Gap Distribution')
        axes[2].grid(True, alpha=0.3)
    
    # 4. Density vs formation energy
    if 'density' in database.columns and 'formation_energy' in database.columns:
        axes[3].scatter(database['density'], database['formation_energy'], alpha=0.6, s=30)
        axes[3].set_xlabel('Density (g/cm³)')
        axes[3].set_ylabel('Formation Energy (eV/atom)')
        axes[3].set_title('Density vs Formation Energy')
        axes[3].grid(True, alpha=0.3)
    
    # 5. Element frequency
    if 'composition' in database.columns:
        # Count element occurrences
        element_counts = {}
        for comp in database['composition']:
            if isinstance(comp, list):
                for element in comp:
                    element_counts[element] = element_counts.get(element, 0) + 1
        
        if element_counts:
            elements = list(element_counts.keys())[:10]  # Top 10 elements
            counts = [element_counts[elem] for elem in elements]
            
            axes[4].bar(range(len(elements)), counts, alpha=0.7, color='purple')
            axes[4].set_xlabel('Element')
            axes[4].set_ylabel('Frequency')
            axes[4].set_title('Most Common Elements')
            axes[4].set_xticks(range(len(elements)))
            axes[4].set_xticklabels(elements)
            axes[4].grid(True, alpha=0.3)
    
    # 6. Property correlation (if multiple properties exist)
    numeric_cols = database.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = database[numeric_cols].corr()
        im = axes[5].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[5].set_xticks(range(len(numeric_cols)))
        axes[5].set_yticks(range(len(numeric_cols)))
        axes[5].set_xticklabels(numeric_cols, rotation=45, ha='right')
        axes[5].set_yticklabels(numeric_cols)
        axes[5].set_title('Property Correlations')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[5])
        cbar.set_label('Correlation')
    
    # Hide unused subplots
    for i in range(len(axes)):
        if not axes[i].has_data():
            axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_materials_report(
    structures: List[Dict[str, Any]],
    properties: List[Dict[str, Any]],
    predictions: Optional[Dict[str, List[float]]] = None,
    save_dir: str = "materials_report"
) -> None:
    """
    Create comprehensive materials analysis report.
    
    Args:
        structures: List of crystal structures
        properties: List of property dictionaries
        predictions: Dictionary of predictions (optional)
        save_dir: Directory to save the report
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    print(f"Creating materials analysis report in {save_dir}/")
    
    # 1. Structure overview
    if structures:
        n_atoms_list = [len(s['atom_types']) for s in structures]
        plot_property_distribution(n_atoms_list, 'Number of Atoms', 
                                 save_path=save_path / 'atoms_distribution.png')
    
    # 2. Property distributions
    if properties:
        for prop_name in ['formation_energy', 'band_gap', 'density', 'bulk_modulus']:
            if prop_name in properties[0]:
                prop_values = [p[prop_name] for p in properties]
                plot_property_distribution(prop_values, prop_name.replace('_', ' ').title(),
                                         save_path=save_path / f'{prop_name}_distribution.png')
    
    # 3. Property correlations
    if properties:
        properties_dict = {}
        for prop_name in properties[0].keys():
            properties_dict[prop_name] = [p[prop_name] for p in properties]
        
        plot_property_correlation(properties_dict, 
                                save_path=save_path / 'property_correlations.png')
    
    # 4. Prediction results
    if predictions and properties:
        for prop_name, pred_values in predictions.items():
            if prop_name in properties[0]:
                true_values = [p[prop_name] for p in properties]
                plot_prediction_results(true_values, pred_values, prop_name.replace('_', ' ').title(),
                                      save_path=save_path / f'{prop_name}_predictions.png')
    
    # 5. Sample structures
    if structures:
        for i, structure in enumerate(structures[:3]):  # Plot first 3 structures
            plot_crystal_structure(structure, save_path=save_path / f'structure_{i+1}.png')
    
    print(f"Report generated successfully in {save_dir}/")


def plot_defect_analysis(
    defects_data: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot defect analysis results.
    
    Args:
        defects_data: Defect analysis data
        save_path: Path to save the plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # 1. Defect type distribution
    defect_types = ['vacancies', 'interstitials', 'substitutions', 'dislocations']
    defect_counts = [len(defects_data.get(dt, [])) for dt in defect_types]
    
    axes[0].bar(defect_types, defect_counts, alpha=0.7, color=['red', 'blue', 'green', 'orange'])
    axes[0].set_ylabel('Count')
    axes[0].set_title('Defect Type Distribution')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Defect positions (if available)
    if 'vacancies' in defects_data and defects_data['vacancies']:
        positions = [v['position'] for v in defects_data['vacancies']]
        positions = np.array(positions)
        
        axes[1].scatter(positions[:, 0], positions[:, 1], alpha=0.7, color='red', label='Vacancies')
        axes[1].set_xlabel('X Position')
        axes[1].set_ylabel('Y Position')
        axes[1].set_title('Vacancy Positions')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # 3. Coordination number distribution for defects
    if 'vacancies' in defects_data and defects_data['vacancies']:
        coord_nums = [v['coordination'] for v in defects_data['vacancies']]
        axes[2].hist(coord_nums, bins=10, alpha=0.7, color='red', edgecolor='black')
        axes[2].set_xlabel('Coordination Number')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Coordination Numbers (Vacancies)')
        axes[2].grid(True, alpha=0.3)
    
    # 4. Summary statistics
    summary = defects_data.get('summary', {})
    if summary:
        labels = list(summary.keys())
        values = list(summary.values())
        
        axes[3].bar(labels, values, alpha=0.7, color='purple')
        axes[3].set_ylabel('Value')
        axes[3].set_title('Defect Summary Statistics')
        axes[3].tick_params(axis='x', rotation=45)
        axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show() 