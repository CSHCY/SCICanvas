"""
Preprocessing utilities for materials science data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings


def normalize_structure(structure: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize crystal structure data.
    
    Args:
        structure: Crystal structure dictionary
        
    Returns:
        Normalized structure dictionary
    """
    normalized = structure.copy()
    
    # Normalize coordinates to unit cell
    coordinates = np.array(structure['coordinates'])
    
    if 'lattice_params' in structure:
        lattice_params = np.array(structure['lattice_params'])
        # Simple normalization (assumes orthogonal cell)
        a, b, c = lattice_params[:3]
        coordinates[:, 0] = coordinates[:, 0] % a
        coordinates[:, 1] = coordinates[:, 1] % b
        coordinates[:, 2] = coordinates[:, 2] % c
        
        # Convert to fractional coordinates
        coordinates[:, 0] /= a
        coordinates[:, 1] /= b
        coordinates[:, 2] /= c
    else:
        # Center coordinates around origin
        coordinates = coordinates - np.mean(coordinates, axis=0)
        
        # Scale to unit sphere
        max_dist = np.max(np.linalg.norm(coordinates, axis=1))
        if max_dist > 0:
            coordinates = coordinates / max_dist
    
    normalized['coordinates'] = coordinates
    normalized['is_normalized'] = True
    
    return normalized


def compute_structural_features(structure: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute structural features from crystal structure.
    
    Args:
        structure: Crystal structure dictionary
        
    Returns:
        Dictionary of structural features
    """
    coordinates = np.array(structure['coordinates'])
    atom_types = np.array(structure['atom_types'])
    n_atoms = len(atom_types)
    
    features = {}
    
    # Basic features
    features['n_atoms'] = n_atoms
    features['n_unique_elements'] = len(np.unique(atom_types))
    
    # Density features
    if n_atoms > 1:
        distances = pdist(coordinates)
        features['mean_distance'] = np.mean(distances)
        features['std_distance'] = np.std(distances)
        features['min_distance'] = np.min(distances)
        features['max_distance'] = np.max(distances)
        features['density'] = n_atoms / (4/3 * np.pi * features['max_distance']**3)
    else:
        features['mean_distance'] = 0.0
        features['std_distance'] = 0.0
        features['min_distance'] = 0.0
        features['max_distance'] = 0.0
        features['density'] = 0.0
    
    # Coordination features
    coordination_numbers = compute_coordination_numbers(coordinates)
    features['mean_coordination'] = np.mean(coordination_numbers)
    features['std_coordination'] = np.std(coordination_numbers)
    features['max_coordination'] = np.max(coordination_numbers)
    
    # Lattice features
    if 'lattice_params' in structure:
        lattice_params = structure['lattice_params']
        features['lattice_a'] = lattice_params[0]
        features['lattice_b'] = lattice_params[1]
        features['lattice_c'] = lattice_params[2]
        features['lattice_alpha'] = lattice_params[3]
        features['lattice_beta'] = lattice_params[4]
        features['lattice_gamma'] = lattice_params[5]
        
        # Volume
        a, b, c = lattice_params[:3]
        alpha, beta, gamma = np.radians(lattice_params[3:6])
        
        volume = a * b * c * np.sqrt(
            1 + 2*np.cos(alpha)*np.cos(beta)*np.cos(gamma) - 
            np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2
        )
        features['volume'] = volume
        features['packing_fraction'] = n_atoms / volume
    else:
        # Estimate volume from coordinates
        if n_atoms > 3:
            hull_volume = compute_convex_hull_volume(coordinates)
            features['volume'] = hull_volume
            features['packing_fraction'] = n_atoms / hull_volume if hull_volume > 0 else 0
        else:
            features['volume'] = 0.0
            features['packing_fraction'] = 0.0
    
    # Symmetry features (simplified)
    features['symmetry_score'] = compute_symmetry_score(coordinates, atom_types)
    
    return features


def compute_coordination_numbers(
    coordinates: np.ndarray, 
    cutoff: float = 3.5
) -> np.ndarray:
    """
    Compute coordination numbers for each atom.
    
    Args:
        coordinates: Atomic coordinates
        cutoff: Distance cutoff for coordination
        
    Returns:
        Array of coordination numbers
    """
    n_atoms = len(coordinates)
    coordination_numbers = np.zeros(n_atoms)
    
    for i in range(n_atoms):
        distances = np.linalg.norm(coordinates - coordinates[i], axis=1)
        # Exclude self (distance = 0)
        neighbors = np.sum((distances > 0) & (distances < cutoff))
        coordination_numbers[i] = neighbors
    
    return coordination_numbers


def compute_convex_hull_volume(coordinates: np.ndarray) -> float:
    """
    Compute volume of convex hull of coordinates.
    
    Args:
        coordinates: Atomic coordinates
        
    Returns:
        Volume of convex hull
    """
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(coordinates)
        return hull.volume
    except:
        # Fallback: use bounding box volume
        ranges = np.ptp(coordinates, axis=0)
        return np.prod(ranges)


def compute_symmetry_score(
    coordinates: np.ndarray, 
    atom_types: np.ndarray
) -> float:
    """
    Compute a simple symmetry score for the structure.
    
    Args:
        coordinates: Atomic coordinates
        atom_types: Atomic types
        
    Returns:
        Symmetry score (0-1, higher = more symmetric)
    """
    n_atoms = len(coordinates)
    if n_atoms < 2:
        return 1.0
    
    # Center coordinates
    centered_coords = coordinates - np.mean(coordinates, axis=0)
    
    # Check for inversion symmetry
    inversion_score = 0.0
    for i in range(n_atoms):
        inverted_coord = -centered_coords[i]
        # Find closest atom to inverted position
        distances = np.linalg.norm(centered_coords - inverted_coord, axis=1)
        closest_idx = np.argmin(distances)
        
        if distances[closest_idx] < 0.1 and atom_types[i] == atom_types[closest_idx]:
            inversion_score += 1.0
    
    inversion_score /= n_atoms
    
    # Check for mirror symmetry (simplified)
    mirror_score = 0.0
    for axis in range(3):
        mirrored_coords = centered_coords.copy()
        mirrored_coords[:, axis] *= -1
        
        axis_score = 0.0
        for i in range(n_atoms):
            distances = np.linalg.norm(centered_coords - mirrored_coords[i], axis=1)
            closest_idx = np.argmin(distances)
            
            if distances[closest_idx] < 0.1 and atom_types[i] == atom_types[closest_idx]:
                axis_score += 1.0
        
        mirror_score += axis_score / n_atoms
    
    mirror_score /= 3  # Average over three axes
    
    # Combine scores
    symmetry_score = (inversion_score + mirror_score) / 2
    return min(symmetry_score, 1.0)


def compute_elemental_features(atom_types: List[int]) -> Dict[str, float]:
    """
    Compute elemental composition and property features.
    
    Args:
        atom_types: List of atomic numbers
        
    Returns:
        Dictionary of elemental features
    """
    # Simplified periodic table properties
    element_properties = {
        1: {'period': 1, 'group': 1, 'electronegativity': 2.20, 'atomic_radius': 0.37, 'mass': 1.008},
        6: {'period': 2, 'group': 14, 'electronegativity': 2.55, 'atomic_radius': 0.77, 'mass': 12.011},
        8: {'period': 2, 'group': 16, 'electronegativity': 3.44, 'atomic_radius': 0.73, 'mass': 15.999},
        11: {'period': 3, 'group': 1, 'electronegativity': 0.93, 'atomic_radius': 1.54, 'mass': 22.990},
        12: {'period': 3, 'group': 2, 'electronegativity': 1.31, 'atomic_radius': 1.36, 'mass': 24.305},
        13: {'period': 3, 'group': 13, 'electronegativity': 1.61, 'atomic_radius': 1.18, 'mass': 26.982},
        14: {'period': 3, 'group': 14, 'electronegativity': 1.90, 'atomic_radius': 1.11, 'mass': 28.085},
        16: {'period': 3, 'group': 16, 'electronegativity': 2.58, 'atomic_radius': 1.03, 'mass': 32.065},
        20: {'period': 4, 'group': 2, 'electronegativity': 1.00, 'atomic_radius': 1.74, 'mass': 40.078},
        22: {'period': 4, 'group': 4, 'electronegativity': 1.54, 'atomic_radius': 1.32, 'mass': 47.867},
        26: {'period': 4, 'group': 8, 'electronegativity': 1.83, 'atomic_radius': 1.24, 'mass': 55.845},
        29: {'period': 4, 'group': 11, 'electronegativity': 1.90, 'atomic_radius': 1.28, 'mass': 63.546},
        30: {'period': 4, 'group': 12, 'electronegativity': 1.65, 'atomic_radius': 1.33, 'mass': 65.38},
        31: {'period': 4, 'group': 13, 'electronegativity': 1.81, 'atomic_radius': 1.22, 'mass': 69.723},
        33: {'period': 4, 'group': 15, 'electronegativity': 2.18, 'atomic_radius': 1.21, 'mass': 74.922}
    }
    
    atom_types = np.array(atom_types)
    unique_elements, counts = np.unique(atom_types, return_counts=True)
    total_atoms = len(atom_types)
    
    features = {}
    
    # Composition features
    features['n_elements'] = len(unique_elements)
    features['composition_entropy'] = -np.sum((counts/total_atoms) * np.log(counts/total_atoms))
    
    # Average properties
    avg_properties = {
        'avg_period': 0.0,
        'avg_group': 0.0,
        'avg_electronegativity': 0.0,
        'avg_atomic_radius': 0.0,
        'avg_mass': 0.0
    }
    
    total_weight = 0.0
    for element, count in zip(unique_elements, counts):
        if element in element_properties:
            props = element_properties[element]
            weight = count / total_atoms
            total_weight += weight
            
            avg_properties['avg_period'] += props['period'] * weight
            avg_properties['avg_group'] += props['group'] * weight
            avg_properties['avg_electronegativity'] += props['electronegativity'] * weight
            avg_properties['avg_atomic_radius'] += props['atomic_radius'] * weight
            avg_properties['avg_mass'] += props['mass'] * weight
    
    if total_weight > 0:
        for key in avg_properties:
            avg_properties[key] /= total_weight
    
    features.update(avg_properties)
    
    # Property variance
    var_properties = {
        'var_electronegativity': 0.0,
        'var_atomic_radius': 0.0,
        'var_mass': 0.0
    }
    
    if len(unique_elements) > 1:
        electronegativities = []
        radii = []
        masses = []
        
        for element in unique_elements:
            if element in element_properties:
                props = element_properties[element]
                electronegativities.append(props['electronegativity'])
                radii.append(props['atomic_radius'])
                masses.append(props['mass'])
        
        if electronegativities:
            var_properties['var_electronegativity'] = np.var(electronegativities)
            var_properties['var_atomic_radius'] = np.var(radii)
            var_properties['var_mass'] = np.var(masses)
    
    features.update(var_properties)
    
    return features


def augment_structure(
    structure: Dict[str, Any], 
    augmentation_type: str = 'rotation',
    **kwargs
) -> Dict[str, Any]:
    """
    Apply data augmentation to crystal structure.
    
    Args:
        structure: Crystal structure dictionary
        augmentation_type: Type of augmentation ('rotation', 'noise', 'scaling')
        **kwargs: Additional parameters for augmentation
        
    Returns:
        Augmented structure
    """
    augmented = structure.copy()
    coordinates = np.array(structure['coordinates'])
    
    if augmentation_type == 'rotation':
        # Random rotation
        angle = kwargs.get('angle', np.random.uniform(0, 2*np.pi))
        axis = kwargs.get('axis', np.random.randint(0, 3))
        
        rotation_matrix = create_rotation_matrix(angle, axis)
        augmented['coordinates'] = coordinates @ rotation_matrix.T
        
    elif augmentation_type == 'noise':
        # Add Gaussian noise
        noise_std = kwargs.get('noise_std', 0.1)
        noise = np.random.normal(0, noise_std, coordinates.shape)
        augmented['coordinates'] = coordinates + noise
        
    elif augmentation_type == 'scaling':
        # Uniform scaling
        scale_factor = kwargs.get('scale_factor', np.random.uniform(0.9, 1.1))
        augmented['coordinates'] = coordinates * scale_factor
        
        # Update lattice parameters if present
        if 'lattice_params' in structure:
            lattice_params = np.array(structure['lattice_params'])
            lattice_params[:3] *= scale_factor  # Scale a, b, c
            augmented['lattice_params'] = lattice_params
            
    elif augmentation_type == 'supercell':
        # Create supercell
        multipliers = kwargs.get('multipliers', [2, 1, 1])
        augmented = create_supercell(structure, multipliers)
        
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")
    
    return augmented


def create_rotation_matrix(angle: float, axis: int) -> np.ndarray:
    """
    Create rotation matrix for given angle and axis.
    
    Args:
        angle: Rotation angle in radians
        axis: Rotation axis (0=x, 1=y, 2=z)
        
    Returns:
        3x3 rotation matrix
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    if axis == 0:  # x-axis
        return np.array([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ])
    elif axis == 1:  # y-axis
        return np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
    else:  # z-axis
        return np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])


def create_supercell(
    structure: Dict[str, Any], 
    multipliers: List[int]
) -> Dict[str, Any]:
    """
    Create supercell from unit cell.
    
    Args:
        structure: Crystal structure dictionary
        multipliers: Multipliers for each lattice direction [nx, ny, nz]
        
    Returns:
        Supercell structure
    """
    coordinates = np.array(structure['coordinates'])
    atom_types = np.array(structure['atom_types'])
    
    nx, ny, nz = multipliers
    
    # Generate supercell coordinates
    supercell_coords = []
    supercell_types = []
    
    if 'lattice_params' in structure:
        lattice_params = structure['lattice_params']
        a, b, c = lattice_params[:3]
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Translate coordinates
                    translation = np.array([i*a, j*b, k*c])
                    translated_coords = coordinates + translation
                    
                    supercell_coords.extend(translated_coords)
                    supercell_types.extend(atom_types)
    else:
        # For non-periodic structures, just replicate in space
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if i == 0 and j == 0 and k == 0:
                        continue  # Skip original
                    
                    # Simple translation
                    translation = np.array([i*5.0, j*5.0, k*5.0])
                    translated_coords = coordinates + translation
                    
                    supercell_coords.extend(translated_coords)
                    supercell_types.extend(atom_types)
        
        # Add original coordinates
        supercell_coords = list(coordinates) + supercell_coords
        supercell_types = list(atom_types) + supercell_types
    
    supercell = structure.copy()
    supercell['coordinates'] = np.array(supercell_coords)
    supercell['atom_types'] = supercell_types
    supercell['n_atoms'] = len(supercell_types)
    
    # Update lattice parameters
    if 'lattice_params' in structure:
        new_lattice = structure['lattice_params'].copy()
        new_lattice[0] *= nx  # a
        new_lattice[1] *= ny  # b
        new_lattice[2] *= nz  # c
        supercell['lattice_params'] = new_lattice
    
    return supercell


def filter_structures(
    structures: List[Dict[str, Any]],
    min_atoms: int = 2,
    max_atoms: int = 200,
    min_distance: float = 0.5,
    max_distance: float = 20.0
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Filter structures based on quality criteria.
    
    Args:
        structures: List of crystal structures
        min_atoms: Minimum number of atoms
        max_atoms: Maximum number of atoms
        min_distance: Minimum interatomic distance
        max_distance: Maximum interatomic distance
        
    Returns:
        Tuple of (filtered_structures, valid_indices)
    """
    filtered_structures = []
    valid_indices = []
    
    for i, structure in enumerate(structures):
        coordinates = np.array(structure['coordinates'])
        n_atoms = len(coordinates)
        
        # Check atom count
        if n_atoms < min_atoms or n_atoms > max_atoms:
            continue
        
        # Check distances
        if n_atoms > 1:
            distances = pdist(coordinates)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if min_dist < min_distance or max_dist > max_distance:
                continue
        
        # Check for NaN or infinite coordinates
        if np.any(~np.isfinite(coordinates)):
            continue
        
        filtered_structures.append(structure)
        valid_indices.append(i)
    
    return filtered_structures, valid_indices


def standardize_features(
    features_list: List[Dict[str, float]],
    scaler_type: str = 'standard'
) -> Tuple[List[Dict[str, float]], object]:
    """
    Standardize feature values across structures.
    
    Args:
        features_list: List of feature dictionaries
        scaler_type: Type of scaler ('standard', 'minmax')
        
    Returns:
        Tuple of (standardized_features, fitted_scaler)
    """
    if not features_list:
        return [], None
    
    # Convert to DataFrame
    df = pd.DataFrame(features_list)
    
    # Handle missing values
    df = df.fillna(0.0)
    
    # Select scaler
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    # Fit and transform
    scaled_values = scaler.fit_transform(df.values)
    
    # Convert back to list of dictionaries
    standardized_features = []
    for i, row in enumerate(scaled_values):
        feature_dict = {col: row[j] for j, col in enumerate(df.columns)}
        standardized_features.append(feature_dict)
    
    return standardized_features, scaler


def compute_radial_distribution_function(
    coordinates: np.ndarray,
    r_max: float = 10.0,
    n_bins: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute radial distribution function (RDF) for structure.
    
    Args:
        coordinates: Atomic coordinates
        r_max: Maximum radius for RDF
        n_bins: Number of bins for RDF
        
    Returns:
        Tuple of (r_values, rdf_values)
    """
    n_atoms = len(coordinates)
    
    if n_atoms < 2:
        r_values = np.linspace(0, r_max, n_bins)
        return r_values, np.zeros(n_bins)
    
    # Compute all pairwise distances
    distances = pdist(coordinates)
    
    # Create histogram
    r_values = np.linspace(0, r_max, n_bins + 1)
    hist, _ = np.histogram(distances, bins=r_values)
    
    # Normalize by shell volume and density
    r_centers = (r_values[:-1] + r_values[1:]) / 2
    dr = r_values[1] - r_values[0]
    
    # Shell volumes
    shell_volumes = 4 * np.pi * r_centers**2 * dr
    
    # Number density
    volume = np.prod(np.ptp(coordinates, axis=0))  # Bounding box volume
    density = n_atoms / volume if volume > 0 else 1.0
    
    # Normalize RDF
    rdf = hist / (shell_volumes * density * n_atoms)
    
    return r_centers, rdf


def compute_bond_angles(
    coordinates: np.ndarray,
    bond_indices: np.ndarray,
    cutoff: float = 3.5
) -> List[float]:
    """
    Compute bond angles in structure.
    
    Args:
        coordinates: Atomic coordinates
        bond_indices: Bond connectivity
        cutoff: Distance cutoff for bonds
        
    Returns:
        List of bond angles in degrees
    """
    if len(bond_indices) == 0:
        return []
    
    # Build adjacency list
    adjacency = {}
    for bond in bond_indices:
        i, j = bond
        if i not in adjacency:
            adjacency[i] = []
        if j not in adjacency:
            adjacency[j] = []
        adjacency[i].append(j)
        adjacency[j].append(i)
    
    angles = []
    
    # For each atom, compute angles between its bonds
    for center_atom in adjacency:
        neighbors = adjacency[center_atom]
        
        if len(neighbors) < 2:
            continue
        
        center_pos = coordinates[center_atom]
        
        # Compute all pairwise angles
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                atom1 = neighbors[i]
                atom2 = neighbors[j]
                
                vec1 = coordinates[atom1] - center_pos
                vec2 = coordinates[atom2] - center_pos
                
                # Compute angle
                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
                angle = np.arccos(cos_angle) * 180 / np.pi
                
                angles.append(angle)
    
    return angles


def detect_defects(
    structure: Dict[str, Any],
    reference_structure: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Detect structural defects in crystal structure.
    
    Args:
        structure: Crystal structure to analyze
        reference_structure: Reference perfect structure (optional)
        
    Returns:
        Dictionary containing defect information
    """
    coordinates = np.array(structure['coordinates'])
    atom_types = np.array(structure['atom_types'])
    
    defects = {
        'vacancies': [],
        'interstitials': [],
        'substitutions': [],
        'dislocations': []
    }
    
    # Simple defect detection based on coordination analysis
    coordination_numbers = compute_coordination_numbers(coordinates)
    
    # Detect under-coordinated atoms (potential vacancies nearby)
    mean_coordination = np.mean(coordination_numbers)
    std_coordination = np.std(coordination_numbers)
    
    for i, coord_num in enumerate(coordination_numbers):
        if coord_num < mean_coordination - 2 * std_coordination:
            defects['vacancies'].append({
                'atom_index': i,
                'coordination': coord_num,
                'position': coordinates[i].tolist()
            })
        elif coord_num > mean_coordination + 2 * std_coordination:
            defects['interstitials'].append({
                'atom_index': i,
                'coordination': coord_num,
                'position': coordinates[i].tolist()
            })
    
    # Detect unusual distances (potential dislocations)
    if len(coordinates) > 1:
        distances = pdist(coordinates)
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        distance_matrix = squareform(distances)
        
        for i in range(len(coordinates)):
            for j in range(i + 1, len(coordinates)):
                dist = distance_matrix[i, j]
                if dist > mean_distance + 3 * std_distance:
                    defects['dislocations'].append({
                        'atom_pair': [i, j],
                        'distance': dist,
                        'positions': [coordinates[i].tolist(), coordinates[j].tolist()]
                    })
    
    # Summary statistics
    defects['summary'] = {
        'total_defects': len(defects['vacancies']) + len(defects['interstitials']) + 
                        len(defects['substitutions']) + len(defects['dislocations']),
        'defect_density': (len(defects['vacancies']) + len(defects['interstitials'])) / len(coordinates)
    }
    
    return defects 