"""
Data handling utilities for materials science.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List, Union
from pathlib import Path
import pickle
import json
from collections import defaultdict


class MaterialsDataset(Dataset):
    """
    PyTorch Dataset for materials data.
    """
    
    def __init__(
        self,
        structures: List[Dict[str, Any]],
        properties: Optional[List[Any]] = None,
        compositions: Optional[List[List[float]]] = None,
        conditions: Optional[List[List[float]]] = None,
        transform: Optional[callable] = None
    ):
        """
        Initialize the materials dataset.
        
        Args:
            structures: List of crystal structure dictionaries
            properties: Target properties (optional)
            compositions: Material compositions (optional)
            conditions: Thermodynamic conditions (optional)
            transform: Optional transform to apply to the data
        """
        self.structures = structures
        self.properties = properties
        self.compositions = compositions
        self.conditions = conditions
        self.transform = transform
        
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.structures)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset."""
        sample = {
            'structure': self.structures[idx]
        }
        
        if self.properties is not None:
            sample['property'] = self.properties[idx]
            
        if self.compositions is not None:
            sample['composition'] = self.compositions[idx]
            
        if self.conditions is not None:
            sample['conditions'] = self.conditions[idx]
            
        if self.transform:
            sample = self.transform(sample)
            
        return sample


class MaterialsDataLoader:
    """
    Data loader factory for materials datasets.
    """
    
    @staticmethod
    def create_dataloaders(
        structures: List[Dict[str, Any]],
        properties: Optional[List[Any]] = None,
        compositions: Optional[List[List[float]]] = None,
        conditions: Optional[List[List[float]]] = None,
        batch_size: int = 32,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        shuffle: bool = True,
        num_workers: int = 0,
        random_state: int = 42
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test data loaders.
        
        Args:
            structures: List of crystal structures
            properties: Target properties
            compositions: Material compositions
            conditions: Thermodynamic conditions
            batch_size: Batch size for data loaders
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Validate splits
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
            "Splits must sum to 1.0"
            
        # Set random seed
        np.random.seed(random_state)
        
        # Create indices for splitting
        n_samples = len(structures)
        indices = np.random.permutation(n_samples)
        
        # Calculate split points
        train_end = int(train_split * n_samples)
        val_end = train_end + int(val_split * n_samples)
        
        # Split indices
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # Split data
        train_structures = [structures[i] for i in train_indices]
        val_structures = [structures[i] for i in val_indices]
        test_structures = [structures[i] for i in test_indices]
        
        train_properties = [properties[i] for i in train_indices] if properties else None
        val_properties = [properties[i] for i in val_indices] if properties else None
        test_properties = [properties[i] for i in test_indices] if properties else None
        
        train_compositions = [compositions[i] for i in train_indices] if compositions else None
        val_compositions = [compositions[i] for i in val_indices] if compositions else None
        test_compositions = [compositions[i] for i in test_indices] if compositions else None
        
        train_conditions = [conditions[i] for i in train_indices] if conditions else None
        val_conditions = [conditions[i] for i in val_indices] if conditions else None
        test_conditions = [conditions[i] for i in test_indices] if conditions else None
        
        # Create datasets
        train_dataset = MaterialsDataset(train_structures, train_properties, train_compositions, train_conditions)
        val_dataset = MaterialsDataset(val_structures, val_properties, val_compositions, val_conditions)
        test_dataset = MaterialsDataset(test_structures, test_properties, test_compositions, test_conditions)
        
        # Custom collate function
        def collate_fn(batch):
            """Custom collate function for materials data."""
            structures = [item['structure'] for item in batch]
            
            result = {'structures': structures}
            
            # Handle properties
            if 'property' in batch[0]:
                properties = [item['property'] for item in batch]
                if isinstance(properties[0], (int, float)):
                    result['properties'] = torch.tensor(properties)
                else:
                    result['properties'] = properties
                    
            # Handle compositions
            if 'composition' in batch[0]:
                compositions = [item['composition'] for item in batch]
                result['compositions'] = torch.tensor(compositions)
                
            # Handle conditions
            if 'conditions' in batch[0]:
                conditions = [item['conditions'] for item in batch]
                result['conditions'] = torch.tensor(conditions)
                
            return result
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        
        return train_loader, val_loader, test_loader


def load_example_data(dataset_name: str = "synthetic") -> Dict[str, Any]:
    """
    Load example materials datasets.
    
    Args:
        dataset_name: Name of the dataset to load
        
    Returns:
        Dictionary containing materials data
    """
    if dataset_name == "synthetic":
        return generate_synthetic_materials_data()
    elif dataset_name == "mp_sample":
        return load_materials_project_sample()
    elif dataset_name == "oqmd_sample":
        return load_oqmd_sample()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def generate_synthetic_materials_data(
    n_materials: int = 100,
    max_atoms: int = 50,
    n_elements: int = 10
) -> Dict[str, Any]:
    """
    Generate synthetic materials data for testing.
    
    Args:
        n_materials: Number of materials to generate
        max_atoms: Maximum number of atoms per material
        n_elements: Number of different elements to use
        
    Returns:
        Dictionary containing synthetic materials data
    """
    np.random.seed(42)
    
    # Element properties (simplified)
    element_properties = {
        1: {'symbol': 'H', 'atomic_mass': 1.008, 'electronegativity': 2.20},
        6: {'symbol': 'C', 'atomic_mass': 12.011, 'electronegativity': 2.55},
        8: {'symbol': 'O', 'atomic_mass': 15.999, 'electronegativity': 3.44},
        11: {'symbol': 'Na', 'atomic_mass': 22.990, 'electronegativity': 0.93},
        12: {'symbol': 'Mg', 'atomic_mass': 24.305, 'electronegativity': 1.31},
        13: {'symbol': 'Al', 'atomic_mass': 26.982, 'electronegativity': 1.61},
        14: {'symbol': 'Si', 'atomic_mass': 28.085, 'electronegativity': 1.90},
        16: {'symbol': 'S', 'atomic_mass': 32.065, 'electronegativity': 2.58},
        26: {'symbol': 'Fe', 'atomic_mass': 55.845, 'electronegativity': 1.83},
        29: {'symbol': 'Cu', 'atomic_mass': 63.546, 'electronegativity': 1.90}
    }
    
    elements = list(element_properties.keys())
    
    structures = []
    properties = []
    compositions = []
    
    for i in range(n_materials):
        # Generate random composition
        n_atoms = np.random.randint(5, max_atoms + 1)
        atom_types = np.random.choice(elements, n_atoms, replace=True)
        
        # Generate random coordinates
        coordinates = np.random.randn(n_atoms, 3) * 5.0
        
        # Generate lattice parameters (for crystal structure)
        lattice_params = np.random.uniform(3.0, 10.0, 6)  # a, b, c, alpha, beta, gamma
        
        # Create atom features
        atom_features = []
        for atom_type in atom_types:
            props = element_properties[atom_type]
            features = [
                atom_type,  # Atomic number
                props['atomic_mass'],
                props['electronegativity'],
                np.random.random(),  # Random feature 1
                np.random.random(),  # Random feature 2
            ]
            # Pad to fixed size (92 features for all elements)
            features.extend([0.0] * (92 - len(features)))
            atom_features.append(features)
        
        atom_features = np.array(atom_features)
        
        # Generate bond information
        bond_indices = []
        bond_features = []
        
        # Simple bonding: connect nearby atoms
        for j in range(n_atoms):
            for k in range(j + 1, n_atoms):
                distance = np.linalg.norm(coordinates[j] - coordinates[k])
                if distance < 3.0:  # Arbitrary cutoff
                    bond_indices.append([j, k])
                    bond_indices.append([k, j])  # Bidirectional
                    
                    # Bond features
                    bond_feat = [
                        distance,
                        1.0 / distance,  # Inverse distance
                        np.exp(-distance),  # Exponential decay
                    ]
                    # Pad to fixed size (41 features)
                    bond_feat.extend([0.0] * (41 - len(bond_feat)))
                    bond_features.extend([bond_feat, bond_feat])  # For both directions
        
        if not bond_indices:
            # If no bonds, create at least one dummy bond
            bond_indices = [[0, 0]]
            bond_features = [[0.0] * 41]
        
        structure = {
            'atom_types': atom_types.tolist(),
            'coordinates': coordinates,
            'atom_features': atom_features,
            'bond_indices': np.array(bond_indices),
            'bond_features': np.array(bond_features),
            'lattice_params': lattice_params,
            'n_atoms': n_atoms
        }
        
        structures.append(structure)
        
        # Generate synthetic properties
        # Formation energy (eV/atom)
        formation_energy = np.random.normal(-2.0, 1.0)
        
        # Band gap (eV)
        band_gap = np.random.exponential(1.5)
        
        # Bulk modulus (GPa)
        bulk_modulus = np.random.normal(100, 50)
        
        # Density (g/cm³)
        total_mass = sum(element_properties[at]['atomic_mass'] for at in atom_types)
        volume = np.prod(lattice_params[:3])  # Simplified volume
        density = total_mass / volume * 1.66  # Conversion factor
        
        material_properties = {
            'formation_energy': formation_energy,
            'band_gap': max(0, band_gap),  # Band gap can't be negative
            'bulk_modulus': max(0, bulk_modulus),
            'density': density
        }
        
        properties.append(material_properties)
        
        # Composition (normalized)
        composition = np.zeros(len(elements))
        for atom_type in atom_types:
            idx = elements.index(atom_type)
            composition[idx] += 1
        composition = composition / np.sum(composition)
        compositions.append(composition.tolist())
    
    # Generate thermodynamic conditions
    conditions = []
    for _ in range(n_materials):
        temp = np.random.uniform(300, 2000)  # Temperature (K)
        pressure = np.random.uniform(0.1, 100)  # Pressure (GPa)
        chemical_potential = np.random.normal(0, 1)  # Chemical potential (eV)
        conditions.append([temp, pressure, chemical_potential])
    
    # Generate phase information
    phases = np.random.randint(0, 5, n_materials).tolist()  # 5 possible phases
    stabilities = np.random.uniform(0, 1, n_materials).tolist()
    
    # Generate catalyst data
    catalyst_structures = []
    reaction_conditions = []
    activities = []
    selectivities = []
    
    for i in range(min(20, n_materials)):  # Generate fewer catalyst examples
        # Use existing structure as catalyst
        catalyst_structures.append(structures[i])
        
        # Reaction conditions
        reaction_temp = np.random.uniform(400, 800)  # K
        reaction_pressure = np.random.uniform(1, 50)  # atm
        reactant_conc = np.random.uniform(0.1, 2.0)  # mol/L
        
        reaction_feat = [reaction_temp, reaction_pressure, reactant_conc]
        reaction_feat.extend([0.0] * (50 - len(reaction_feat)))  # Pad to 50 features
        
        reaction_conditions.append({'features': reaction_feat})
        
        # Activity and selectivity
        activities.append(np.random.uniform(0, 1))
        selectivities.append(np.random.uniform(0, 1))
    
    # Generate electronic structure data
    electronic_data = []
    for i in range(min(30, n_materials)):
        dos_spectrum = np.random.exponential(1, 100)  # Density of states
        fermi_level = np.random.normal(0, 2)  # Fermi level (eV)
        
        electronic_data.append({
            'dos_spectrum': dos_spectrum,
            'fermi_level': fermi_level
        })
    
    return {
        'structures': structures,
        'properties': properties,
        'compositions': compositions,
        'conditions': conditions,
        'phases': phases,
        'stabilities': stabilities,
        'catalyst_structures': catalyst_structures,
        'reaction_conditions': reaction_conditions,
        'activities': activities,
        'selectivities': selectivities,
        'electronic_data': electronic_data,
        'element_properties': element_properties
    }


def load_materials_project_sample() -> Dict[str, Any]:
    """
    Load a sample of Materials Project data.
    
    Returns:
        Dictionary containing Materials Project sample data
    """
    # This is a simplified example - in practice you'd use the Materials Project API
    
    sample_data = {
        'mp_ids': ['mp-1', 'mp-2', 'mp-3', 'mp-4', 'mp-5'],
        'formulas': ['Si', 'GaAs', 'TiO2', 'Fe2O3', 'CaTiO3'],
        'structures': [
            {
                'atom_types': [14, 14, 14, 14, 14, 14, 14, 14],  # Silicon
                'coordinates': np.random.randn(8, 3) * 2.7,
                'lattice_params': [5.43, 5.43, 5.43, 90, 90, 90],
                'space_group': 'Fd-3m'
            },
            {
                'atom_types': [31, 33, 31, 33],  # GaAs
                'coordinates': np.random.randn(4, 3) * 2.8,
                'lattice_params': [5.65, 5.65, 5.65, 90, 90, 90],
                'space_group': 'F-43m'
            },
            {
                'atom_types': [22, 8, 8],  # TiO2
                'coordinates': np.random.randn(3, 3) * 2.3,
                'lattice_params': [4.59, 4.59, 2.96, 90, 90, 90],
                'space_group': 'P42/mnm'
            },
            {
                'atom_types': [26, 26, 8, 8, 8],  # Fe2O3
                'coordinates': np.random.randn(5, 3) * 2.5,
                'lattice_params': [5.04, 5.04, 13.75, 90, 90, 120],
                'space_group': 'R-3c'
            },
            {
                'atom_types': [20, 22, 8, 8, 8],  # CaTiO3
                'coordinates': np.random.randn(5, 3) * 2.4,
                'lattice_params': [5.38, 5.44, 7.64, 90, 90, 90],
                'space_group': 'Pbnm'
            }
        ],
        'properties': {
            'formation_energy': [-10.85, -0.74, -9.75, -8.24, -15.67],  # eV/atom
            'band_gap': [1.17, 1.42, 3.20, 2.20, 3.40],  # eV
            'density': [2.33, 5.32, 4.23, 5.24, 4.04],  # g/cm³
            'bulk_modulus': [97.8, 75.5, 210.0, 220.0, 165.0]  # GPa
        }
    }
    
    return sample_data


def load_oqmd_sample() -> Dict[str, Any]:
    """
    Load a sample of OQMD (Open Quantum Materials Database) data.
    
    Returns:
        Dictionary containing OQMD sample data
    """
    # This is a simplified example - in practice you'd use the OQMD API
    
    sample_data = {
        'oqmd_ids': [1, 2, 3, 4, 5],
        'compositions': ['Al', 'NaCl', 'MgO', 'SiC', 'ZnS'],
        'structures': [
            {
                'atom_types': [13, 13, 13, 13],  # Aluminum
                'coordinates': np.random.randn(4, 3) * 2.0,
                'lattice_params': [4.05, 4.05, 4.05, 90, 90, 90]
            },
            {
                'atom_types': [11, 17, 11, 17, 11, 17, 11, 17],  # NaCl
                'coordinates': np.random.randn(8, 3) * 2.8,
                'lattice_params': [5.64, 5.64, 5.64, 90, 90, 90]
            },
            {
                'atom_types': [12, 8, 12, 8],  # MgO
                'coordinates': np.random.randn(4, 3) * 2.1,
                'lattice_params': [4.21, 4.21, 4.21, 90, 90, 90]
            },
            {
                'atom_types': [14, 6, 14, 6],  # SiC
                'coordinates': np.random.randn(4, 3) * 2.2,
                'lattice_params': [4.36, 4.36, 4.36, 90, 90, 90]
            },
            {
                'atom_types': [30, 16, 30, 16],  # ZnS
                'coordinates': np.random.randn(4, 3) * 2.7,
                'lattice_params': [5.41, 5.41, 5.41, 90, 90, 90]
            }
        ],
        'formation_energies': [-3.74, -4.24, -6.08, -0.73, -1.10],  # eV/atom
        'stability': [0.0, 0.0, 0.0, 0.0, 0.0]  # eV/atom above hull
    }
    
    return sample_data


def create_crystal_graph(structure: Dict[str, Any], cutoff: float = 5.0) -> Dict[str, Any]:
    """
    Create a crystal graph from structure data.
    
    Args:
        structure: Crystal structure dictionary
        cutoff: Distance cutoff for bonds
        
    Returns:
        Dictionary containing graph representation
    """
    coordinates = np.array(structure['coordinates'])
    atom_types = np.array(structure['atom_types'])
    n_atoms = len(atom_types)
    
    # Compute pairwise distances
    distances = np.zeros((n_atoms, n_atoms))
    for i in range(n_atoms):
        for j in range(n_atoms):
            distances[i, j] = np.linalg.norm(coordinates[i] - coordinates[j])
    
    # Find bonds within cutoff
    bond_indices = []
    bond_distances = []
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if distances[i, j] < cutoff:
                bond_indices.extend([[i, j], [j, i]])  # Bidirectional
                bond_distances.extend([distances[i, j], distances[i, j]])
    
    # Create bond features
    bond_features = []
    for dist in bond_distances:
        features = [
            dist,
            1.0 / dist if dist > 0 else 0.0,
            np.exp(-dist),
            np.cos(dist),
            np.sin(dist)
        ]
        # Pad to desired size
        features.extend([0.0] * (41 - len(features)))
        bond_features.append(features)
    
    return {
        'atom_types': atom_types,
        'coordinates': coordinates,
        'bond_indices': np.array(bond_indices) if bond_indices else np.array([[0, 0]]),
        'bond_features': np.array(bond_features) if bond_features else np.array([[0.0] * 41]),
        'distances': distances
    }


def save_materials_data(
    data: Dict[str, Any],
    file_path: Union[str, Path],
    format: str = 'pickle'
) -> None:
    """
    Save materials data to file.
    
    Args:
        data: Materials data dictionary
        file_path: Path to save the data
        format: File format ('pickle', 'json')
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'pickle':
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    elif format == 'json':
        # Convert numpy arrays to lists for JSON serialization
        json_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                json_data[key] = [v.tolist() if isinstance(v, np.ndarray) else v for v in value]
            else:
                json_data[key] = value
        
        with open(file_path, 'w') as f:
            json.dump(json_data, f, indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_materials_data(
    file_path: Union[str, Path],
    format: str = 'pickle'
) -> Dict[str, Any]:
    """
    Load materials data from file.
    
    Args:
        file_path: Path to the data file
        format: File format ('pickle', 'json')
        
    Returns:
        Materials data dictionary
    """
    file_path = Path(file_path)
    
    if format == 'pickle':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif format == 'json':
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unknown format: {format}")


def compute_material_fingerprint(structure: Dict[str, Any]) -> np.ndarray:
    """
    Compute a fingerprint representation of a material.
    
    Args:
        structure: Crystal structure dictionary
        
    Returns:
        Fingerprint vector
    """
    atom_types = np.array(structure['atom_types'])
    coordinates = np.array(structure['coordinates'])
    
    # Composition features
    unique_elements, counts = np.unique(atom_types, return_counts=True)
    composition = np.zeros(118)  # For all elements
    for elem, count in zip(unique_elements, counts):
        composition[elem - 1] = count / len(atom_types)
    
    # Structural features
    n_atoms = len(atom_types)
    
    # Pairwise distances
    distances = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(coordinates[i] - coordinates[j])
            distances.append(dist)
    
    if distances:
        distance_features = [
            np.mean(distances),
            np.std(distances),
            np.min(distances),
            np.max(distances)
        ]
    else:
        distance_features = [0.0, 0.0, 0.0, 0.0]
    
    # Lattice features
    if 'lattice_params' in structure:
        lattice_features = structure['lattice_params'][:6]  # a, b, c, alpha, beta, gamma
    else:
        lattice_features = [0.0] * 6
    
    # Combine all features
    fingerprint = np.concatenate([
        composition,
        distance_features,
        lattice_features,
        [n_atoms]  # Number of atoms
    ])
    
    return fingerprint


def create_materials_database(
    structures: List[Dict[str, Any]],
    properties: List[Dict[str, Any]],
    save_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Create a materials database from structures and properties.
    
    Args:
        structures: List of crystal structures
        properties: List of property dictionaries
        save_path: Optional path to save the database
        
    Returns:
        Pandas DataFrame containing the database
    """
    database_entries = []
    
    for i, (structure, props) in enumerate(zip(structures, properties)):
        entry = {
            'material_id': f'mat_{i:06d}',
            'n_atoms': structure.get('n_atoms', len(structure['atom_types'])),
            'composition': structure['atom_types'],
        }
        
        # Add fingerprint
        fingerprint = compute_material_fingerprint(structure)
        for j, fp_val in enumerate(fingerprint):
            entry[f'fingerprint_{j:03d}'] = fp_val
        
        # Add properties
        entry.update(props)
        
        database_entries.append(entry)
    
    df = pd.DataFrame(database_entries)
    
    if save_path:
        df.to_csv(save_path, index=False)
    
    return df 