"""
Data handling utilities for protein prediction.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List, Union
from pathlib import Path
import pickle
import gzip
import requests
from io import StringIO


class ProteinDataset(Dataset):
    """
    PyTorch Dataset for protein data.
    """
    
    def __init__(
        self,
        sequences: List[str],
        targets: Optional[Union[List[Any], np.ndarray]] = None,
        msas: Optional[List[List[str]]] = None,
        structures: Optional[List[Dict[str, Any]]] = None,
        transform: Optional[callable] = None
    ):
        """
        Initialize the protein dataset.
        
        Args:
            sequences: List of protein sequences
            targets: Target labels or values (optional)
            msas: Multiple sequence alignments (optional)
            structures: Protein structures (optional)
            transform: Optional transform to apply to the data
        """
        self.sequences = sequences
        self.targets = targets
        self.msas = msas
        self.structures = structures
        self.transform = transform
        
        # Amino acid vocabulary
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.amino_acids)}
        self.aa_to_idx['X'] = 20  # Unknown amino acid
        
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.sequences)
        
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """Encode protein sequence to tensor."""
        encoded = [self.aa_to_idx.get(aa, 20) for aa in sequence.upper()]
        return torch.LongTensor(encoded)
        
    def encode_msa(self, msa: List[str]) -> torch.Tensor:
        """Encode multiple sequence alignment to tensor."""
        encoded_msa = []
        for seq in msa:
            encoded_seq = [self.aa_to_idx.get(aa, 20) for aa in seq.upper()]
            encoded_msa.append(encoded_seq)
        return torch.LongTensor(encoded_msa)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset."""
        sample = {
            'sequence': self.sequences[idx],
            'encoded_sequence': self.encode_sequence(self.sequences[idx])
        }
        
        if self.targets is not None:
            sample['target'] = self.targets[idx]
            
        if self.msas is not None and idx < len(self.msas):
            sample['msa'] = self.msas[idx]
            sample['encoded_msa'] = self.encode_msa(self.msas[idx])
        else:
            # Create dummy MSA with just the target sequence
            sample['msa'] = [self.sequences[idx]]
            sample['encoded_msa'] = self.encode_sequence(self.sequences[idx]).unsqueeze(0)
            
        if self.structures is not None and idx < len(self.structures):
            sample['structure'] = self.structures[idx]
            
        if self.transform:
            sample = self.transform(sample)
            
        return sample


class ProteinDataLoader:
    """
    Data loader factory for protein datasets.
    """
    
    @staticmethod
    def create_dataloaders(
        sequences: List[str],
        targets: Optional[List[Any]] = None,
        msas: Optional[List[List[str]]] = None,
        structures: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 16,
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
            sequences: List of protein sequences
            targets: Target labels or values
            msas: Multiple sequence alignments
            structures: Protein structures
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
        n_samples = len(sequences)
        indices = np.random.permutation(n_samples)
        
        # Calculate split points
        train_end = int(train_split * n_samples)
        val_end = train_end + int(val_split * n_samples)
        
        # Split indices
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # Split data
        train_sequences = [sequences[i] for i in train_indices]
        val_sequences = [sequences[i] for i in val_indices]
        test_sequences = [sequences[i] for i in test_indices]
        
        train_targets = [targets[i] for i in train_indices] if targets else None
        val_targets = [targets[i] for i in val_indices] if targets else None
        test_targets = [targets[i] for i in test_indices] if targets else None
        
        train_msas = [msas[i] for i in train_indices] if msas else None
        val_msas = [msas[i] for i in val_indices] if msas else None
        test_msas = [msas[i] for i in test_indices] if msas else None
        
        train_structures = [structures[i] for i in train_indices] if structures else None
        val_structures = [structures[i] for i in val_indices] if structures else None
        test_structures = [structures[i] for i in test_indices] if structures else None
        
        # Create datasets
        train_dataset = ProteinDataset(train_sequences, train_targets, train_msas, train_structures)
        val_dataset = ProteinDataset(val_sequences, val_targets, val_msas, val_structures)
        test_dataset = ProteinDataset(test_sequences, test_targets, test_msas, test_structures)
        
        # Custom collate function for variable length sequences
        def collate_fn(batch):
            """Custom collate function for protein data."""
            sequences = [item['sequence'] for item in batch]
            encoded_sequences = [item['encoded_sequence'] for item in batch]
            
            # Pad sequences to same length
            max_len = max(len(seq) for seq in encoded_sequences)
            padded_sequences = []
            attention_masks = []
            
            for seq in encoded_sequences:
                padded_seq = torch.zeros(max_len, dtype=torch.long)
                padded_seq[:len(seq)] = seq
                padded_sequences.append(padded_seq)
                
                mask = torch.zeros(max_len, dtype=torch.bool)
                mask[:len(seq)] = True
                attention_masks.append(mask)
                
            result = {
                'sequences': sequences,
                'input_ids': torch.stack(padded_sequences),
                'attention_mask': torch.stack(attention_masks)
            }
            
            # Handle targets
            if 'target' in batch[0]:
                targets = [item['target'] for item in batch]
                if isinstance(targets[0], (int, float)):
                    result['targets'] = torch.tensor(targets)
                else:
                    result['targets'] = targets
                    
            # Handle MSAs
            if 'encoded_msa' in batch[0]:
                msas = [item['encoded_msa'] for item in batch]
                # For simplicity, we'll handle MSAs individually in the model
                result['msas'] = msas
                
            # Handle structures
            if 'structure' in batch[0]:
                structures = [item['structure'] for item in batch]
                result['structures'] = structures
                
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
    Load example protein datasets.
    
    Args:
        dataset_name: Name of the dataset to load
        
    Returns:
        Dictionary containing protein data
    """
    if dataset_name == "synthetic":
        return generate_synthetic_protein_data()
    elif dataset_name == "uniprot_sample":
        return load_uniprot_sample()
    elif dataset_name == "pdb_sample":
        return load_pdb_sample()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def generate_synthetic_protein_data(
    n_proteins: int = 100,
    min_length: int = 50,
    max_length: int = 300,
    n_functions: int = 10
) -> Dict[str, Any]:
    """
    Generate synthetic protein data for testing.
    
    Args:
        n_proteins: Number of proteins to generate
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        n_functions: Number of different functions
        
    Returns:
        Dictionary containing synthetic protein data
    """
    np.random.seed(42)
    
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    
    # Generate sequences
    sequences = []
    for _ in range(n_proteins):
        length = np.random.randint(min_length, max_length + 1)
        sequence = ''.join(np.random.choice(list(amino_acids), length))
        sequences.append(sequence)
    
    # Generate function labels
    function_names = [f"Function_{i}" for i in range(n_functions)]
    functions = np.random.choice(function_names, n_proteins)
    
    # Generate synthetic structures (contact maps)
    structures = []
    for seq in sequences:
        length = len(seq)
        # Generate random contact map
        contact_map = np.random.random((length, length)) > 0.8
        # Make symmetric
        contact_map = contact_map | contact_map.T
        # Remove diagonal
        np.fill_diagonal(contact_map, False)
        
        # Generate random coordinates
        coordinates = np.random.randn(length, 3) * 10
        
        structure = {
            'contact_map': contact_map.astype(float),
            'coordinates': coordinates
        }
        structures.append(structure)
    
    # Generate synthetic MSAs (simplified)
    msas = []
    for seq in sequences:
        # Create MSA with mutations
        msa = [seq]  # Original sequence
        for _ in range(5):  # Add 5 homologous sequences
            mutated_seq = list(seq)
            # Introduce random mutations
            n_mutations = max(1, len(seq) // 20)
            positions = np.random.choice(len(seq), n_mutations, replace=False)
            for pos in positions:
                mutated_seq[pos] = np.random.choice(list(amino_acids))
            msa.append(''.join(mutated_seq))
        msas.append(msa)
    
    return {
        'sequences': sequences,
        'functions': functions.tolist(),
        'structures': structures,
        'msas': msas
    }


def load_uniprot_sample() -> Dict[str, Any]:
    """
    Load a sample of UniProt protein data.
    
    Returns:
        Dictionary containing UniProt protein data
    """
    # This is a simplified example - in practice you'd download from UniProt
    # For demonstration, we'll create some realistic-looking data
    
    sample_data = {
        'sequences': [
            'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
            'MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVK',
            'MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEK',
            'MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVAD',
            'MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFPTSREJ'
        ],
        'functions': [
            'GO:0003677',  # DNA binding
            'GO:0005515',  # protein binding
            'GO:0016787',  # hydrolase activity
            'GO:0003824',  # catalytic activity
            'GO:0005515'   # protein binding
        ],
        'descriptions': [
            'DNA-binding protein',
            'Serum albumin',
            'Metabolic enzyme',
            'Hemoglobin subunit',
            'Structural protein'
        ]
    }
    
    return sample_data


def load_pdb_sample() -> Dict[str, Any]:
    """
    Load a sample of PDB structure data.
    
    Returns:
        Dictionary containing PDB structure data
    """
    # This is a simplified example - in practice you'd parse PDB files
    
    sample_data = {
        'pdb_ids': ['1ABC', '2DEF', '3GHI'],
        'sequences': [
            'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
            'MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVK',
            'MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEK'
        ],
        'structures': [
            {
                'coordinates': np.random.randn(64, 3) * 10,
                'secondary_structure': 'H' * 20 + 'E' * 24 + 'C' * 20,
                'resolution': 2.1
            },
            {
                'coordinates': np.random.randn(65, 3) * 10,
                'secondary_structure': 'E' * 30 + 'H' * 15 + 'C' * 20,
                'resolution': 1.8
            },
            {
                'coordinates': np.random.randn(66, 3) * 10,
                'secondary_structure': 'C' * 20 + 'H' * 26 + 'E' * 20,
                'resolution': 2.5
            }
        ]
    }
    
    return sample_data


def save_protein_data(
    data: Dict[str, Any],
    file_path: Union[str, Path],
    format: str = 'pickle'
) -> None:
    """
    Save protein data to file.
    
    Args:
        data: Protein data dictionary
        file_path: Path to save the data
        format: File format ('pickle', 'fasta')
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'pickle':
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    elif format == 'fasta':
        with open(file_path, 'w') as f:
            for i, seq in enumerate(data['sequences']):
                f.write(f">protein_{i}\n{seq}\n")
    else:
        raise ValueError(f"Unknown format: {format}")


def load_protein_data(
    file_path: Union[str, Path],
    format: str = 'pickle'
) -> Dict[str, Any]:
    """
    Load protein data from file.
    
    Args:
        file_path: Path to the data file
        format: File format ('pickle', 'fasta')
        
    Returns:
        Protein data dictionary
    """
    file_path = Path(file_path)
    
    if format == 'pickle':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif format == 'fasta':
        sequences = []
        with open(file_path, 'r') as f:
            sequence = ""
            for line in f:
                if line.startswith('>'):
                    if sequence:
                        sequences.append(sequence)
                        sequence = ""
                else:
                    sequence += line.strip()
            if sequence:
                sequences.append(sequence)
        return {'sequences': sequences}
    else:
        raise ValueError(f"Unknown format: {format}")


def parse_fasta(fasta_string: str) -> List[Tuple[str, str]]:
    """
    Parse FASTA format string.
    
    Args:
        fasta_string: FASTA format string
        
    Returns:
        List of (header, sequence) tuples
    """
    sequences = []
    lines = fasta_string.strip().split('\n')
    
    current_header = None
    current_sequence = ""
    
    for line in lines:
        if line.startswith('>'):
            if current_header is not None:
                sequences.append((current_header, current_sequence))
            current_header = line[1:]
            current_sequence = ""
        else:
            current_sequence += line.strip()
    
    if current_header is not None:
        sequences.append((current_header, current_sequence))
    
    return sequences


def create_msa_from_sequences(
    target_sequence: str,
    homologous_sequences: List[str],
    max_sequences: int = 100
) -> List[str]:
    """
    Create a multiple sequence alignment from homologous sequences.
    
    Args:
        target_sequence: Target protein sequence
        homologous_sequences: List of homologous sequences
        max_sequences: Maximum number of sequences in MSA
        
    Returns:
        List of aligned sequences
    """
    # This is a simplified MSA creation - in practice you'd use proper alignment tools
    msa = [target_sequence]
    
    # Add homologous sequences (up to max_sequences)
    for seq in homologous_sequences[:max_sequences-1]:
        # Simple alignment by padding/truncating to target length
        if len(seq) < len(target_sequence):
            aligned_seq = seq + '-' * (len(target_sequence) - len(seq))
        elif len(seq) > len(target_sequence):
            aligned_seq = seq[:len(target_sequence)]
        else:
            aligned_seq = seq
        msa.append(aligned_seq)
    
    return msa 