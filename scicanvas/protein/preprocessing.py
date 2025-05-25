"""
Preprocessing utilities for protein data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
import re
from collections import Counter


def clean_sequence(sequence: str, remove_gaps: bool = True) -> str:
    """
    Clean protein sequence by removing invalid characters.
    
    Args:
        sequence: Protein sequence
        remove_gaps: Whether to remove gap characters
        
    Returns:
        Cleaned sequence
    """
    # Standard amino acids
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    
    # Convert to uppercase
    sequence = sequence.upper()
    
    # Remove gaps if requested
    if remove_gaps:
        sequence = sequence.replace('-', '').replace('.', '')
    
    # Replace invalid characters with X
    cleaned = ''.join(aa if aa in valid_aa or (not remove_gaps and aa in '-.')
                     else 'X' for aa in sequence)
    
    return cleaned


def filter_sequences(
    sequences: List[str],
    min_length: int = 10,
    max_length: int = 5000,
    max_unknown_ratio: float = 0.1
) -> Tuple[List[str], List[int]]:
    """
    Filter protein sequences based on quality criteria.
    
    Args:
        sequences: List of protein sequences
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        max_unknown_ratio: Maximum ratio of unknown amino acids (X)
        
    Returns:
        Tuple of (filtered_sequences, valid_indices)
    """
    filtered_sequences = []
    valid_indices = []
    
    for i, seq in enumerate(sequences):
        # Check length
        if len(seq) < min_length or len(seq) > max_length:
            continue
            
        # Check unknown amino acid ratio
        unknown_count = seq.count('X')
        unknown_ratio = unknown_count / len(seq)
        
        if unknown_ratio > max_unknown_ratio:
            continue
            
        filtered_sequences.append(seq)
        valid_indices.append(i)
    
    return filtered_sequences, valid_indices


def compute_sequence_features(sequence: str) -> Dict[str, float]:
    """
    Compute basic features from protein sequence.
    
    Args:
        sequence: Protein sequence
        
    Returns:
        Dictionary of sequence features
    """
    # Amino acid properties
    aa_properties = {
        'A': {'hydrophobic': 1, 'polar': 0, 'charged': 0, 'aromatic': 0, 'mw': 89.1},
        'C': {'hydrophobic': 1, 'polar': 0, 'charged': 0, 'aromatic': 0, 'mw': 121.0},
        'D': {'hydrophobic': 0, 'polar': 1, 'charged': -1, 'aromatic': 0, 'mw': 133.1},
        'E': {'hydrophobic': 0, 'polar': 1, 'charged': -1, 'aromatic': 0, 'mw': 147.1},
        'F': {'hydrophobic': 1, 'polar': 0, 'charged': 0, 'aromatic': 1, 'mw': 165.2},
        'G': {'hydrophobic': 0, 'polar': 0, 'charged': 0, 'aromatic': 0, 'mw': 75.1},
        'H': {'hydrophobic': 0, 'polar': 1, 'charged': 1, 'aromatic': 1, 'mw': 155.2},
        'I': {'hydrophobic': 1, 'polar': 0, 'charged': 0, 'aromatic': 0, 'mw': 131.2},
        'K': {'hydrophobic': 0, 'polar': 1, 'charged': 1, 'aromatic': 0, 'mw': 146.2},
        'L': {'hydrophobic': 1, 'polar': 0, 'charged': 0, 'aromatic': 0, 'mw': 131.2},
        'M': {'hydrophobic': 1, 'polar': 0, 'charged': 0, 'aromatic': 0, 'mw': 149.2},
        'N': {'hydrophobic': 0, 'polar': 1, 'charged': 0, 'aromatic': 0, 'mw': 132.1},
        'P': {'hydrophobic': 0, 'polar': 0, 'charged': 0, 'aromatic': 0, 'mw': 115.1},
        'Q': {'hydrophobic': 0, 'polar': 1, 'charged': 0, 'aromatic': 0, 'mw': 146.2},
        'R': {'hydrophobic': 0, 'polar': 1, 'charged': 1, 'aromatic': 0, 'mw': 174.2},
        'S': {'hydrophobic': 0, 'polar': 1, 'charged': 0, 'aromatic': 0, 'mw': 105.1},
        'T': {'hydrophobic': 0, 'polar': 1, 'charged': 0, 'aromatic': 0, 'mw': 119.1},
        'V': {'hydrophobic': 1, 'polar': 0, 'charged': 0, 'aromatic': 0, 'mw': 117.1},
        'W': {'hydrophobic': 1, 'polar': 0, 'charged': 0, 'aromatic': 1, 'mw': 204.2},
        'Y': {'hydrophobic': 1, 'polar': 1, 'charged': 0, 'aromatic': 1, 'mw': 181.2},
        'X': {'hydrophobic': 0, 'polar': 0, 'charged': 0, 'aromatic': 0, 'mw': 110.0}  # Unknown
    }
    
    # Basic features
    length = len(sequence)
    aa_counts = Counter(sequence)
    
    # Amino acid composition
    composition = {aa: count / length for aa, count in aa_counts.items()}
    
    # Physicochemical properties
    hydrophobic_count = sum(aa_properties.get(aa, aa_properties['X'])['hydrophobic'] 
                           for aa in sequence)
    polar_count = sum(aa_properties.get(aa, aa_properties['X'])['polar'] 
                     for aa in sequence)
    charged_count = sum(abs(aa_properties.get(aa, aa_properties['X'])['charged']) 
                       for aa in sequence)
    aromatic_count = sum(aa_properties.get(aa, aa_properties['X'])['aromatic'] 
                        for aa in sequence)
    
    # Molecular weight
    molecular_weight = sum(aa_properties.get(aa, aa_properties['X'])['mw'] 
                          for aa in sequence)
    
    # Net charge
    net_charge = sum(aa_properties.get(aa, aa_properties['X'])['charged'] 
                    for aa in sequence)
    
    features = {
        'length': length,
        'molecular_weight': molecular_weight,
        'net_charge': net_charge,
        'hydrophobic_ratio': hydrophobic_count / length,
        'polar_ratio': polar_count / length,
        'charged_ratio': charged_count / length,
        'aromatic_ratio': aromatic_count / length,
        'unknown_ratio': composition.get('X', 0)
    }
    
    # Add amino acid composition
    for aa in 'ACDEFGHIKLMNPQRSTVWY':
        features[f'composition_{aa}'] = composition.get(aa, 0)
    
    return features


def compute_dipeptide_composition(sequence: str) -> Dict[str, float]:
    """
    Compute dipeptide composition features.
    
    Args:
        sequence: Protein sequence
        
    Returns:
        Dictionary of dipeptide composition features
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    dipeptides = [aa1 + aa2 for aa1 in amino_acids for aa2 in amino_acids]
    
    # Count dipeptides
    dipeptide_counts = {dp: 0 for dp in dipeptides}
    
    for i in range(len(sequence) - 1):
        dipeptide = sequence[i:i+2]
        if dipeptide in dipeptide_counts:
            dipeptide_counts[dipeptide] += 1
    
    # Normalize by total number of dipeptides
    total_dipeptides = len(sequence) - 1
    if total_dipeptides > 0:
        dipeptide_composition = {dp: count / total_dipeptides 
                               for dp, count in dipeptide_counts.items()}
    else:
        dipeptide_composition = {dp: 0 for dp in dipeptides}
    
    return dipeptide_composition


def predict_secondary_structure(sequence: str) -> Dict[str, Any]:
    """
    Simple secondary structure prediction (placeholder).
    
    Args:
        sequence: Protein sequence
        
    Returns:
        Dictionary containing secondary structure predictions
    """
    # This is a simplified prediction - in practice you'd use tools like DSSP, PSIPRED, etc.
    
    # Simple rules based on amino acid propensities
    helix_propensity = {'A': 1.42, 'E': 1.51, 'L': 1.21, 'M': 1.45, 'Q': 1.11, 'K': 1.16}
    sheet_propensity = {'V': 1.70, 'I': 1.60, 'Y': 1.47, 'F': 1.38, 'T': 1.19, 'C': 1.19}
    
    helix_score = sum(helix_propensity.get(aa, 1.0) for aa in sequence) / len(sequence)
    sheet_score = sum(sheet_propensity.get(aa, 1.0) for aa in sequence) / len(sequence)
    
    # Predict secondary structure (simplified)
    ss_prediction = []
    for aa in sequence:
        h_prop = helix_propensity.get(aa, 1.0)
        s_prop = sheet_propensity.get(aa, 1.0)
        
        if h_prop > s_prop and h_prop > 1.2:
            ss_prediction.append('H')  # Helix
        elif s_prop > h_prop and s_prop > 1.2:
            ss_prediction.append('E')  # Sheet
        else:
            ss_prediction.append('C')  # Coil
    
    return {
        'secondary_structure': ''.join(ss_prediction),
        'helix_content': ss_prediction.count('H') / len(sequence),
        'sheet_content': ss_prediction.count('E') / len(sequence),
        'coil_content': ss_prediction.count('C') / len(sequence),
        'helix_score': helix_score,
        'sheet_score': sheet_score
    }


def extract_domains(sequence: str, domain_patterns: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """
    Extract protein domains using pattern matching.
    
    Args:
        sequence: Protein sequence
        domain_patterns: Dictionary of domain patterns (regex)
        
    Returns:
        List of domain predictions
    """
    if domain_patterns is None:
        # Some common domain patterns (simplified)
        domain_patterns = {
            'zinc_finger': r'C.{2,4}C.{3,12}H.{3,5}H',
            'leucine_zipper': r'L.{6}L.{6}L.{6}L',
            'helix_turn_helix': r'[RK].{15,25}[RK]',
            'signal_peptide': r'^M[^DEKR]{10,30}[AILV]'
        }
    
    domains = []
    
    for domain_name, pattern in domain_patterns.items():
        matches = re.finditer(pattern, sequence, re.IGNORECASE)
        
        for match in matches:
            domain = {
                'name': domain_name,
                'start': match.start(),
                'end': match.end(),
                'sequence': match.group(),
                'confidence': 0.5  # Simplified confidence score
            }
            domains.append(domain)
    
    return domains


def compute_disorder_propensity(sequence: str) -> Dict[str, float]:
    """
    Compute intrinsic disorder propensity.
    
    Args:
        sequence: Protein sequence
        
    Returns:
        Dictionary containing disorder predictions
    """
    # Disorder-promoting amino acids
    disorder_promoting = set('RQSYGKPEND')
    order_promoting = set('WFYLIV')
    
    disorder_score = sum(1 for aa in sequence if aa in disorder_promoting) / len(sequence)
    order_score = sum(1 for aa in sequence if aa in order_promoting) / len(sequence)
    
    # Simple disorder prediction
    disorder_regions = []
    window_size = 10
    
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i + window_size]
        window_disorder = sum(1 for aa in window if aa in disorder_promoting) / window_size
        
        if window_disorder > 0.6:
            disorder_regions.append((i, i + window_size))
    
    return {
        'disorder_score': disorder_score,
        'order_score': order_score,
        'disorder_ratio': disorder_score / (disorder_score + order_score + 1e-8),
        'disorder_regions': disorder_regions
    }


def normalize_contact_map(contact_map: np.ndarray, method: str = 'binary') -> np.ndarray:
    """
    Normalize contact map.
    
    Args:
        contact_map: Contact map matrix
        method: Normalization method ('binary', 'distance', 'probability')
        
    Returns:
        Normalized contact map
    """
    if method == 'binary':
        # Convert to binary (0 or 1)
        return (contact_map > 0).astype(float)
    elif method == 'distance':
        # Normalize by maximum distance
        max_dist = np.max(contact_map)
        if max_dist > 0:
            return 1.0 - (contact_map / max_dist)
        else:
            return contact_map
    elif method == 'probability':
        # Convert to probabilities (0 to 1)
        min_val = np.min(contact_map)
        max_val = np.max(contact_map)
        if max_val > min_val:
            return (contact_map - min_val) / (max_val - min_val)
        else:
            return contact_map
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def extract_structural_features(coordinates: np.ndarray) -> Dict[str, Any]:
    """
    Extract structural features from 3D coordinates.
    
    Args:
        coordinates: 3D coordinates array (n_residues, 3)
        
    Returns:
        Dictionary of structural features
    """
    n_residues = coordinates.shape[0]
    
    # Compute pairwise distances
    distances = np.zeros((n_residues, n_residues))
    for i in range(n_residues):
        for j in range(i + 1, n_residues):
            dist = np.linalg.norm(coordinates[i] - coordinates[j])
            distances[i, j] = dist
            distances[j, i] = dist
    
    # Radius of gyration
    center_of_mass = np.mean(coordinates, axis=0)
    rg = np.sqrt(np.mean(np.sum((coordinates - center_of_mass) ** 2, axis=1)))
    
    # Compactness
    max_distance = np.max(distances)
    compactness = rg / max_distance if max_distance > 0 else 0
    
    # Contact density (residues within 8 Å)
    contact_threshold = 8.0
    contacts = np.sum(distances < contact_threshold) - n_residues  # Exclude diagonal
    contact_density = contacts / (n_residues * (n_residues - 1))
    
    return {
        'radius_of_gyration': rg,
        'max_distance': max_distance,
        'compactness': compactness,
        'contact_density': contact_density,
        'n_residues': n_residues
    }


def align_sequences(sequences: List[str], method: str = 'simple') -> List[str]:
    """
    Align multiple protein sequences.
    
    Args:
        sequences: List of protein sequences
        method: Alignment method ('simple', 'clustal')
        
    Returns:
        List of aligned sequences
    """
    if method == 'simple':
        # Simple alignment by padding to maximum length
        max_length = max(len(seq) for seq in sequences)
        aligned = []
        
        for seq in sequences:
            if len(seq) < max_length:
                # Pad with gaps at the end
                aligned_seq = seq + '-' * (max_length - len(seq))
            else:
                aligned_seq = seq
            aligned.append(aligned_seq)
        
        return aligned
    
    elif method == 'clustal':
        # Placeholder for ClustalW/ClustalO integration
        # In practice, you'd use external tools or libraries like Biopython
        raise NotImplementedError("ClustalW integration not implemented")
    
    else:
        raise ValueError(f"Unknown alignment method: {method}")


def create_position_weight_matrix(msa: List[str]) -> np.ndarray:
    """
    Create position weight matrix from multiple sequence alignment.
    
    Args:
        msa: Multiple sequence alignment
        
    Returns:
        Position weight matrix (length x 21) for 20 amino acids + gap
    """
    if not msa:
        raise ValueError("Empty MSA")
    
    seq_length = len(msa[0])
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY-'  # Include gap character
    n_aa = len(amino_acids)
    
    # Initialize PWM
    pwm = np.zeros((seq_length, n_aa))
    
    # Count amino acids at each position
    for pos in range(seq_length):
        aa_counts = Counter(seq[pos] for seq in msa if pos < len(seq))
        
        for i, aa in enumerate(amino_acids):
            pwm[pos, i] = aa_counts.get(aa, 0)
    
    # Normalize to frequencies
    row_sums = np.sum(pwm, axis=1, keepdims=True)
    pwm = np.divide(pwm, row_sums, out=np.zeros_like(pwm), where=row_sums != 0)
    
    return pwm


def compute_conservation_scores(msa: List[str]) -> np.ndarray:
    """
    Compute conservation scores for each position in MSA.
    
    Args:
        msa: Multiple sequence alignment
        
    Returns:
        Conservation scores for each position
    """
    if not msa:
        raise ValueError("Empty MSA")
    
    seq_length = len(msa[0])
    conservation_scores = np.zeros(seq_length)
    
    for pos in range(seq_length):
        # Get amino acids at this position
        aa_at_pos = [seq[pos] for seq in msa if pos < len(seq) and seq[pos] != '-']
        
        if aa_at_pos:
            # Calculate Shannon entropy
            aa_counts = Counter(aa_at_pos)
            total = len(aa_at_pos)
            
            entropy = 0
            for count in aa_counts.values():
                if count > 0:
                    p = count / total
                    entropy -= p * np.log2(p)
            
            # Convert entropy to conservation score (higher = more conserved)
            max_entropy = np.log2(20)  # Maximum possible entropy for 20 amino acids
            conservation_scores[pos] = 1 - (entropy / max_entropy)
        else:
            conservation_scores[pos] = 0
    
    return conservation_scores 