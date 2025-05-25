"""
Predictor classes for protein prediction tasks.
"""

import torch
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Optional, Tuple
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
import logging

from ..core.base import BasePredictor
from .models import AlphaFoldModel, ProteinTransformer, ContactPredictor


class StructurePredictor(BasePredictor):
    """
    Protein structure prediction predictor using AlphaFold-inspired architecture.
    
    This predictor can predict 3D protein structures from sequences and MSAs,
    including contact maps, distance matrices, and atomic coordinates.
    """
    
    def __init__(
        self,
        model_type: str = 'alphafold',
        vocab_size: int = 21,
        model_params: Optional[Dict[str, Any]] = None,
        device: str = 'auto'
    ):
        super().__init__(device=device)
        
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.model_params = model_params or {}
        self.is_fitted = False
        
        # Amino acid vocabulary
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.amino_acids)}
        self.aa_to_idx['X'] = 20  # Unknown amino acid
        
        if model_type in ['alphafold', 'contact']:
            self._build_model()
            
    def _build_model(self):
        """Build the neural network model."""
        if self.model_type == 'alphafold':
            self.model = AlphaFoldModel(
                vocab_size=self.vocab_size,
                **self.model_params
            )
        elif self.model_type == 'contact':
            self.model = ContactPredictor(
                vocab_size=self.vocab_size,
                **self.model_params
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        self.model.to(self.device)
        
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
        
    def fit(
        self,
        sequences: List[str],
        structures: Optional[List[Dict[str, Any]]] = None,
        msas: Optional[List[List[str]]] = None,
        batch_size: int = 8,
        num_epochs: int = 50,
        learning_rate: float = 1e-4,
        validation_split: float = 0.2
    ):
        """
        Fit the structure prediction model.
        
        Args:
            sequences: List of protein sequences
            structures: List of structure dictionaries (coordinates, contacts, etc.)
            msas: List of MSAs for each sequence (optional)
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            validation_split: Fraction of data for validation
        """
        if structures is None:
            raise ValueError("Structure data is required for training")
            
        # Prepare data
        encoded_sequences = []
        encoded_msas = []
        
        for i, seq in enumerate(sequences):
            encoded_seq = self.encode_sequence(seq)
            encoded_sequences.append(encoded_seq)
            
            if msas is not None and i < len(msas):
                # Use provided MSA
                encoded_msa = self.encode_msa(msas[i])
            else:
                # Create dummy MSA with just the target sequence
                encoded_msa = self.encode_sequence(seq).unsqueeze(0)
            encoded_msas.append(encoded_msa)
            
        # Setup training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            # Simple batching (in practice, you'd want more sophisticated batching)
            for i in range(0, len(sequences), batch_size):
                batch_end = min(i + batch_size, len(sequences))
                
                optimizer.zero_grad()
                batch_loss = 0.0
                
                for j in range(i, batch_end):
                    if self.model_type == 'alphafold':
                        # Prepare MSA input
                        msa_input = encoded_msas[j].unsqueeze(0).to(self.device)
                        
                        # Forward pass
                        outputs = self.model(msa_input)
                        
                        # Compute loss (simplified - in practice you'd have multiple loss terms)
                        if 'coordinates' in structures[j]:
                            target_coords = torch.FloatTensor(structures[j]['coordinates']).to(self.device)
                            pred_coords = outputs['coordinates'].squeeze(0)
                            
                            # Coordinate loss (MSE)
                            coord_loss = torch.nn.functional.mse_loss(pred_coords, target_coords)
                            batch_loss += coord_loss
                            
                    elif self.model_type == 'contact':
                        # Prepare sequence input
                        seq_input = encoded_sequences[j].unsqueeze(0).to(self.device)
                        
                        # Forward pass
                        outputs = self.model(seq_input)
                        
                        # Contact map loss
                        if 'contact_map' in structures[j]:
                            target_contacts = torch.FloatTensor(structures[j]['contact_map']).to(self.device)
                            pred_contacts = outputs['contact_map'].squeeze(0)
                            
                            # Binary cross-entropy loss
                            contact_loss = torch.nn.functional.binary_cross_entropy(pred_contacts, target_contacts)
                            batch_loss += contact_loss
                            
                batch_loss = batch_loss / (batch_end - i)
                batch_loss.backward()
                optimizer.step()
                
                epoch_loss += batch_loss.item()
                n_batches += 1
                
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Loss={epoch_loss/n_batches:.4f}")
                
        self.is_fitted = True
        self.logger.info("Training completed")
        
    def predict(
        self, 
        sequence: str, 
        msa: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Predict protein structure from sequence.
        
        Args:
            sequence: Protein sequence
            msa: Multiple sequence alignment (optional)
            
        Returns:
            Dictionary containing structure predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        self.model.eval()
        
        with torch.no_grad():
            if self.model_type == 'alphafold':
                # Prepare MSA input
                if msa is not None:
                    encoded_msa = self.encode_msa(msa)
                else:
                    # Create dummy MSA with just target sequence
                    encoded_msa = self.encode_sequence(sequence).unsqueeze(0)
                    
                msa_input = encoded_msa.unsqueeze(0).to(self.device)
                
                # Forward pass
                outputs = self.model(msa_input)
                
                results = {
                    'coordinates': outputs['coordinates'].cpu().numpy(),
                    'confidence': outputs['confidence'].cpu().numpy(),
                    'distance_logits': outputs['distance_logits'].cpu().numpy(),
                    'angle_predictions': outputs['angle_predictions'].cpu().numpy()
                }
                
            elif self.model_type == 'contact':
                # Prepare sequence input
                seq_input = self.encode_sequence(sequence).unsqueeze(0).to(self.device)
                
                # Forward pass
                outputs = self.model(seq_input)
                
                results = {
                    'contact_map': outputs['contact_map'].cpu().numpy(),
                    'sequence_representation': outputs['sequence_representation'].cpu().numpy()
                }
                
        return results
        
    def predict_batch(
        self, 
        sequences: List[str], 
        msas: Optional[List[List[str]]] = None
    ) -> List[Dict[str, np.ndarray]]:
        """Predict structures for a batch of sequences."""
        results = []
        
        for i, seq in enumerate(sequences):
            msa = msas[i] if msas is not None and i < len(msas) else None
            result = self.predict(seq, msa)
            results.append(result)
            
        return results


class FunctionAnnotator(BasePredictor):
    """
    Protein function annotation predictor.
    
    This predictor can annotate protein functions including GO terms,
    enzyme classification, and other functional annotations.
    """
    
    def __init__(
        self,
        model_type: str = 'transformer',
        vocab_size: int = 21,
        n_functions: Optional[int] = None,
        model_params: Optional[Dict[str, Any]] = None,
        device: str = 'auto'
    ):
        super().__init__(device=device)
        
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.n_functions = n_functions
        self.model_params = model_params or {}
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
        # Amino acid vocabulary
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.amino_acids)}
        self.aa_to_idx['X'] = 20
        
    def _build_model(self):
        """Build the neural network model."""
        if self.model_type == 'transformer':
            self.model = ProteinTransformer(
                vocab_size=self.vocab_size,
                output_dim=self.n_functions,
                **self.model_params
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        self.model.to(self.device)
        
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """Encode protein sequence to tensor."""
        encoded = [self.aa_to_idx.get(aa, 20) for aa in sequence.upper()]
        return torch.LongTensor(encoded)
        
    def fit(
        self,
        sequences: List[str],
        functions: List[Union[str, List[str]]],
        batch_size: int = 16,
        num_epochs: int = 50,
        learning_rate: float = 1e-4,
        validation_split: float = 0.2
    ):
        """
        Fit the function annotation model.
        
        Args:
            sequences: List of protein sequences
            functions: List of function labels (GO terms, EC numbers, etc.)
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            validation_split: Fraction of data for validation
        """
        # Process function labels
        if isinstance(functions[0], str):
            # Single-label classification
            encoded_functions = self.label_encoder.fit_transform(functions)
            self.n_functions = len(self.label_encoder.classes_)
            self.multi_label = False
        else:
            # Multi-label classification
            all_functions = set()
            for func_list in functions:
                all_functions.update(func_list)
            
            self.function_to_idx = {func: i for i, func in enumerate(sorted(all_functions))}
            self.n_functions = len(all_functions)
            self.multi_label = True
            
            # Create binary encoding
            encoded_functions = []
            for func_list in functions:
                binary_encoding = np.zeros(self.n_functions)
                for func in func_list:
                    if func in self.function_to_idx:
                        binary_encoding[self.function_to_idx[func]] = 1
                encoded_functions.append(binary_encoding)
            encoded_functions = np.array(encoded_functions)
            
        # Build model
        self._build_model()
        
        # Prepare data
        encoded_sequences = [self.encode_sequence(seq) for seq in sequences]
        
        # Setup training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        if self.multi_label:
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()
            
        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            # Simple batching
            for i in range(0, len(sequences), batch_size):
                batch_end = min(i + batch_size, len(sequences))
                
                # Prepare batch
                batch_sequences = []
                batch_functions = []
                max_len = 0
                
                for j in range(i, batch_end):
                    seq = encoded_sequences[j]
                    batch_sequences.append(seq)
                    max_len = max(max_len, len(seq))
                    
                    if self.multi_label:
                        batch_functions.append(encoded_functions[j])
                    else:
                        batch_functions.append(encoded_functions[j])
                        
                # Pad sequences
                padded_sequences = []
                attention_masks = []
                
                for seq in batch_sequences:
                    padded_seq = torch.zeros(max_len, dtype=torch.long)
                    padded_seq[:len(seq)] = seq
                    padded_sequences.append(padded_seq)
                    
                    mask = torch.zeros(max_len, dtype=torch.bool)
                    mask[:len(seq)] = True
                    attention_masks.append(mask)
                    
                batch_input = torch.stack(padded_sequences).to(self.device)
                batch_mask = torch.stack(attention_masks).to(self.device)
                
                if self.multi_label:
                    batch_targets = torch.FloatTensor(batch_functions).to(self.device)
                else:
                    batch_targets = torch.LongTensor(batch_functions).to(self.device)
                    
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_input, batch_mask)
                
                if self.multi_label:
                    # Use pooled output for multi-label classification
                    logits = outputs['pooled_output']
                    # Add final classification layer if not present
                    if not hasattr(self.model, 'classification_head'):
                        self.model.classification_head = torch.nn.Linear(
                            logits.size(-1), self.n_functions
                        ).to(self.device)
                    logits = self.model.classification_head(logits)
                else:
                    logits = outputs['logits'].mean(dim=1)  # Global average pooling
                    
                loss = criterion(logits, batch_targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
                
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Loss={epoch_loss/n_batches:.4f}")
                
        self.is_fitted = True
        self.logger.info("Training completed")
        
    def predict(self, sequence: str) -> Union[str, List[str]]:
        """
        Predict protein function from sequence.
        
        Args:
            sequence: Protein sequence
            
        Returns:
            Predicted function(s)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        encoded_seq = self.encode_sequence(sequence).unsqueeze(0).to(self.device)
        attention_mask = torch.ones(encoded_seq.size(), dtype=torch.bool).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(encoded_seq, attention_mask)
            
            if self.multi_label:
                logits = self.model.classification_head(outputs['pooled_output'])
                predictions = torch.sigmoid(logits) > 0.5
                predicted_functions = []
                
                for i, pred in enumerate(predictions[0]):
                    if pred:
                        func = list(self.function_to_idx.keys())[i]
                        predicted_functions.append(func)
                        
                return predicted_functions
            else:
                logits = outputs['logits'].mean(dim=1)
                prediction = torch.argmax(logits, dim=-1).cpu().numpy()[0]
                return self.label_encoder.inverse_transform([prediction])[0]
                
    def predict_proba(self, sequence: str) -> np.ndarray:
        """Predict function probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        encoded_seq = self.encode_sequence(sequence).unsqueeze(0).to(self.device)
        attention_mask = torch.ones(encoded_seq.size(), dtype=torch.bool).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(encoded_seq, attention_mask)
            
            if self.multi_label:
                logits = self.model.classification_head(outputs['pooled_output'])
                probabilities = torch.sigmoid(logits).cpu().numpy()[0]
            else:
                logits = outputs['logits'].mean(dim=1)
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                
        return probabilities


class DrugTargetPredictor(BasePredictor):
    """
    Drug-target interaction predictor.
    
    This predictor can predict interactions between drugs (molecules)
    and protein targets, including binding affinity prediction.
    """
    
    def __init__(
        self,
        model_type: str = 'interaction',
        protein_vocab_size: int = 21,
        drug_vocab_size: int = 100,  # For molecular tokens
        model_params: Optional[Dict[str, Any]] = None,
        device: str = 'auto'
    ):
        super().__init__(device=device)
        
        self.model_type = model_type
        self.protein_vocab_size = protein_vocab_size
        self.drug_vocab_size = drug_vocab_size
        self.model_params = model_params or {}
        self.is_fitted = False
        
        # Amino acid vocabulary
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.amino_acids)}
        self.aa_to_idx['X'] = 20
        
    def _build_model(self):
        """Build the drug-target interaction model."""
        # Protein encoder
        self.protein_encoder = ProteinTransformer(
            vocab_size=self.protein_vocab_size,
            **self.model_params.get('protein_params', {})
        )
        
        # Drug encoder (simplified - in practice you'd use molecular transformers)
        drug_params = self.model_params.get('drug_params', {})
        self.drug_encoder = ProteinTransformer(  # Reusing transformer architecture
            vocab_size=self.drug_vocab_size,
            d_model=drug_params.get('d_model', 256),
            n_layers=drug_params.get('n_layers', 6)
        )
        
        # Interaction prediction head
        protein_dim = self.model_params.get('protein_params', {}).get('d_model', 512)
        drug_dim = drug_params.get('d_model', 256)
        
        self.interaction_head = torch.nn.Sequential(
            torch.nn.Linear(protein_dim + drug_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        )
        
        # Move to device
        self.protein_encoder.to(self.device)
        self.drug_encoder.to(self.device)
        self.interaction_head.to(self.device)
        
    def encode_protein(self, sequence: str) -> torch.Tensor:
        """Encode protein sequence."""
        encoded = [self.aa_to_idx.get(aa, 20) for aa in sequence.upper()]
        return torch.LongTensor(encoded)
        
    def encode_drug(self, smiles: str) -> torch.Tensor:
        """Encode drug SMILES string (simplified)."""
        # This is a simplified encoding - in practice you'd use proper molecular tokenization
        chars = list(smiles)
        encoded = [ord(c) % self.drug_vocab_size for c in chars]
        return torch.LongTensor(encoded)
        
    def fit(
        self,
        protein_sequences: List[str],
        drug_smiles: List[str],
        interactions: List[float],  # Binding affinities or binary interactions
        batch_size: int = 16,
        num_epochs: int = 50,
        learning_rate: float = 1e-4
    ):
        """
        Fit the drug-target interaction model.
        
        Args:
            protein_sequences: List of protein sequences
            drug_smiles: List of drug SMILES strings
            interactions: List of interaction values (affinities or binary)
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
        """
        # Build model
        self._build_model()
        
        # Prepare data
        encoded_proteins = [self.encode_protein(seq) for seq in protein_sequences]
        encoded_drugs = [self.encode_drug(smiles) for smiles in drug_smiles]
        
        # Setup training
        all_params = list(self.protein_encoder.parameters()) + \
                    list(self.drug_encoder.parameters()) + \
                    list(self.interaction_head.parameters())
        optimizer = torch.optim.Adam(all_params, lr=learning_rate)
        criterion = torch.nn.MSELoss()  # For regression, use BCELoss for binary classification
        
        # Training loop
        self.protein_encoder.train()
        self.drug_encoder.train()
        self.interaction_head.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            # Simple batching
            for i in range(0, len(protein_sequences), batch_size):
                batch_end = min(i + batch_size, len(protein_sequences))
                
                # Prepare batch
                batch_proteins = []
                batch_drugs = []
                batch_interactions = []
                
                max_protein_len = 0
                max_drug_len = 0
                
                for j in range(i, batch_end):
                    protein = encoded_proteins[j]
                    drug = encoded_drugs[j]
                    
                    batch_proteins.append(protein)
                    batch_drugs.append(drug)
                    batch_interactions.append(interactions[j])
                    
                    max_protein_len = max(max_protein_len, len(protein))
                    max_drug_len = max(max_drug_len, len(drug))
                    
                # Pad sequences
                padded_proteins = []
                padded_drugs = []
                protein_masks = []
                drug_masks = []
                
                for protein, drug in zip(batch_proteins, batch_drugs):
                    # Pad protein
                    padded_protein = torch.zeros(max_protein_len, dtype=torch.long)
                    padded_protein[:len(protein)] = protein
                    padded_proteins.append(padded_protein)
                    
                    protein_mask = torch.zeros(max_protein_len, dtype=torch.bool)
                    protein_mask[:len(protein)] = True
                    protein_masks.append(protein_mask)
                    
                    # Pad drug
                    padded_drug = torch.zeros(max_drug_len, dtype=torch.long)
                    padded_drug[:len(drug)] = drug
                    padded_drugs.append(padded_drug)
                    
                    drug_mask = torch.zeros(max_drug_len, dtype=torch.bool)
                    drug_mask[:len(drug)] = True
                    drug_masks.append(drug_mask)
                    
                # Convert to tensors
                protein_input = torch.stack(padded_proteins).to(self.device)
                drug_input = torch.stack(padded_drugs).to(self.device)
                protein_mask = torch.stack(protein_masks).to(self.device)
                drug_mask = torch.stack(drug_masks).to(self.device)
                interaction_targets = torch.FloatTensor(batch_interactions).to(self.device)
                
                optimizer.zero_grad()
                
                # Encode protein and drug
                protein_output = self.protein_encoder(protein_input, protein_mask)
                drug_output = self.drug_encoder(drug_input, drug_mask)
                
                # Get pooled representations
                protein_repr = protein_output['pooled_output']
                drug_repr = drug_output['pooled_output']
                
                # Concatenate and predict interaction
                combined_repr = torch.cat([protein_repr, drug_repr], dim=-1)
                interaction_pred = self.interaction_head(combined_repr).squeeze(-1)
                
                # Compute loss
                loss = criterion(interaction_pred, interaction_targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
                
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Loss={epoch_loss/n_batches:.4f}")
                
        self.is_fitted = True
        self.logger.info("Training completed")
        
    def predict(
        self, 
        protein_sequence: str, 
        drug_smiles: str
    ) -> float:
        """
        Predict drug-target interaction.
        
        Args:
            protein_sequence: Protein sequence
            drug_smiles: Drug SMILES string
            
        Returns:
            Predicted interaction score
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        # Encode inputs
        protein_input = self.encode_protein(protein_sequence).unsqueeze(0).to(self.device)
        drug_input = self.encode_drug(drug_smiles).unsqueeze(0).to(self.device)
        
        protein_mask = torch.ones(protein_input.size(), dtype=torch.bool).to(self.device)
        drug_mask = torch.ones(drug_input.size(), dtype=torch.bool).to(self.device)
        
        self.protein_encoder.eval()
        self.drug_encoder.eval()
        self.interaction_head.eval()
        
        with torch.no_grad():
            # Encode protein and drug
            protein_output = self.protein_encoder(protein_input, protein_mask)
            drug_output = self.drug_encoder(drug_input, drug_mask)
            
            # Get pooled representations
            protein_repr = protein_output['pooled_output']
            drug_repr = drug_output['pooled_output']
            
            # Predict interaction
            combined_repr = torch.cat([protein_repr, drug_repr], dim=-1)
            interaction_score = self.interaction_head(combined_repr).item()
            
        return interaction_score
        
    def predict_batch(
        self, 
        protein_sequences: List[str], 
        drug_smiles: List[str]
    ) -> List[float]:
        """Predict interactions for a batch of protein-drug pairs."""
        predictions = []
        
        for protein, drug in zip(protein_sequences, drug_smiles):
            pred = self.predict(protein, drug)
            predictions.append(pred)
            
        return predictions 