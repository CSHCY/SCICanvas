"""
Predictor classes for materials science tasks.
"""

import torch
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

from ..core.base import BasePredictor
from .models import (
    CrystalGraphConvNet, MaterialsTransformer, CatalystDesignNet,
    PhasePredictor, ElectronicStructureNet
)


class PropertyPredictor(BasePredictor):
    """
    Materials property prediction predictor.
    
    This predictor can predict various materials properties such as
    formation energy, band gap, bulk modulus, etc. from crystal structure.
    """
    
    def __init__(
        self,
        model_type: str = 'cgcnn',
        property_type: str = 'formation_energy',
        model_params: Optional[Dict[str, Any]] = None,
        device: str = 'auto'
    ):
        super().__init__(device=device)
        
        self.model_type = model_type
        self.property_type = property_type
        self.model_params = model_params or {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Property-specific configurations
        self.property_configs = {
            'formation_energy': {'output_dim': 1, 'scale': True},
            'band_gap': {'output_dim': 1, 'scale': True},
            'bulk_modulus': {'output_dim': 1, 'scale': True},
            'shear_modulus': {'output_dim': 1, 'scale': True},
            'density': {'output_dim': 1, 'scale': True},
            'thermal_conductivity': {'output_dim': 1, 'scale': True}
        }
        
        if model_type in ['cgcnn', 'transformer']:
            self._build_model()
            
    def _build_model(self):
        """Build the neural network model."""
        config = self.property_configs.get(self.property_type, {'output_dim': 1, 'scale': True})
        
        if self.model_type == 'cgcnn':
            self.model = CrystalGraphConvNet(
                output_dim=config['output_dim'],
                **self.model_params
            )
        elif self.model_type == 'transformer':
            self.model = MaterialsTransformer(
                output_dim=config['output_dim'],
                **self.model_params
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        self.model.to(self.device)
        
    def fit(
        self,
        structures: List[Dict[str, Any]],
        properties: List[float],
        batch_size: int = 32,
        num_epochs: int = 100,
        learning_rate: float = 1e-3,
        validation_split: float = 0.2
    ):
        """
        Fit the property prediction model.
        
        Args:
            structures: List of crystal structure dictionaries
            properties: List of property values
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            validation_split: Fraction of data for validation
        """
        # Scale properties if needed
        config = self.property_configs.get(self.property_type, {'scale': True})
        if config['scale']:
            properties_scaled = self.scaler.fit_transform(np.array(properties).reshape(-1, 1)).flatten()
        else:
            properties_scaled = np.array(properties)
        
        # Split data
        n_samples = len(structures)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)
        
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        train_structures = [structures[i] for i in train_indices]
        train_properties = properties_scaled[train_indices]
        val_structures = [structures[i] for i in val_indices]
        val_properties = properties_scaled[val_indices]
        
        # Setup training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            # Simple batching (in practice, you'd want more sophisticated batching)
            for i in range(0, len(train_structures), batch_size):
                batch_end = min(i + batch_size, len(train_structures))
                
                optimizer.zero_grad()
                batch_loss = 0.0
                
                for j in range(i, batch_end):
                    structure = train_structures[j - i]
                    target = torch.FloatTensor([train_properties[j - i]]).to(self.device)
                    
                    if self.model_type == 'cgcnn':
                        # Prepare CGCNN input
                        atom_features = torch.FloatTensor(structure['atom_features']).to(self.device)
                        bond_features = torch.FloatTensor(structure['bond_features']).to(self.device)
                        bond_indices = torch.LongTensor(structure['bond_indices']).to(self.device)
                        
                        outputs = self.model(atom_features, bond_features, bond_indices)
                        prediction = outputs['predictions']
                        
                    elif self.model_type == 'transformer':
                        # Prepare transformer input
                        atom_types = torch.LongTensor(structure['atom_types']).unsqueeze(0).to(self.device)
                        coordinates = torch.FloatTensor(structure['coordinates']).unsqueeze(0).to(self.device)
                        
                        outputs = self.model(atom_types, coordinates)
                        prediction = outputs['predictions']
                        
                    loss = criterion(prediction.squeeze(), target.squeeze())
                    batch_loss += loss
                    
                batch_loss = batch_loss / (batch_end - i)
                batch_loss.backward()
                optimizer.step()
                
                epoch_loss += batch_loss.item()
                n_batches += 1
                
            # Validation
            if epoch % 10 == 0:
                val_loss = self._evaluate(val_structures, val_properties)
                self.logger.info(f"Epoch {epoch}: Train Loss={epoch_loss/n_batches:.4f}, Val Loss={val_loss:.4f}")
                
        self.is_fitted = True
        self.logger.info("Training completed")
        
    def _evaluate(self, structures: List[Dict[str, Any]], properties: np.ndarray) -> float:
        """Evaluate model on validation data."""
        self.model.eval()
        total_loss = 0.0
        n_samples = 0
        
        with torch.no_grad():
            for i, structure in enumerate(structures):
                target = torch.FloatTensor([properties[i]]).to(self.device)
                
                if self.model_type == 'cgcnn':
                    atom_features = torch.FloatTensor(structure['atom_features']).to(self.device)
                    bond_features = torch.FloatTensor(structure['bond_features']).to(self.device)
                    bond_indices = torch.LongTensor(structure['bond_indices']).to(self.device)
                    
                    outputs = self.model(atom_features, bond_features, bond_indices)
                    prediction = outputs['predictions']
                    
                elif self.model_type == 'transformer':
                    atom_types = torch.LongTensor(structure['atom_types']).unsqueeze(0).to(self.device)
                    coordinates = torch.FloatTensor(structure['coordinates']).unsqueeze(0).to(self.device)
                    
                    outputs = self.model(atom_types, coordinates)
                    prediction = outputs['predictions']
                
                loss = torch.nn.functional.mse_loss(prediction.squeeze(), target.squeeze())
                total_loss += loss.item()
                n_samples += 1
        
        self.model.train()
        return total_loss / n_samples
        
    def predict(self, structure: Dict[str, Any]) -> float:
        """
        Predict property from crystal structure.
        
        Args:
            structure: Crystal structure dictionary
            
        Returns:
            Predicted property value
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        self.model.eval()
        
        with torch.no_grad():
            if self.model_type == 'cgcnn':
                atom_features = torch.FloatTensor(structure['atom_features']).to(self.device)
                bond_features = torch.FloatTensor(structure['bond_features']).to(self.device)
                bond_indices = torch.LongTensor(structure['bond_indices']).to(self.device)
                
                outputs = self.model(atom_features, bond_features, bond_indices)
                prediction = outputs['predictions'].cpu().numpy()
                
            elif self.model_type == 'transformer':
                atom_types = torch.LongTensor(structure['atom_types']).unsqueeze(0).to(self.device)
                coordinates = torch.FloatTensor(structure['coordinates']).unsqueeze(0).to(self.device)
                
                outputs = self.model(atom_types, coordinates)
                prediction = outputs['predictions'].cpu().numpy()
        
        # Inverse transform if scaling was applied
        config = self.property_configs.get(self.property_type, {'scale': True})
        if config['scale'] and hasattr(self.scaler, 'scale_'):
            prediction = self.scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()
        
        return float(prediction[0])
        
    def predict_batch(self, structures: List[Dict[str, Any]]) -> List[float]:
        """Predict properties for a batch of structures."""
        predictions = []
        for structure in structures:
            pred = self.predict(structure)
            predictions.append(pred)
        return predictions


class StructurePredictor(BasePredictor):
    """
    Crystal structure prediction predictor.
    
    This predictor can predict stable crystal structures from
    composition and thermodynamic conditions.
    """
    
    def __init__(
        self,
        model_type: str = 'transformer',
        max_atoms: int = 200,
        model_params: Optional[Dict[str, Any]] = None,
        device: str = 'auto'
    ):
        super().__init__(device=device)
        
        self.model_type = model_type
        self.max_atoms = max_atoms
        self.model_params = model_params or {}
        self.is_fitted = False
        
        if model_type == 'transformer':
            self._build_model()
            
    def _build_model(self):
        """Build the structure prediction model."""
        if self.model_type == 'transformer':
            self.model = MaterialsTransformer(
                max_atoms=self.max_atoms,
                output_dim=self.max_atoms * 3,  # Predict coordinates
                **self.model_params
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        self.model.to(self.device)
        
    def fit(
        self,
        compositions: List[List[int]],
        structures: List[Dict[str, Any]],
        batch_size: int = 16,
        num_epochs: int = 100,
        learning_rate: float = 1e-4
    ):
        """
        Fit the structure prediction model.
        
        Args:
            compositions: List of atomic compositions
            structures: List of target crystal structures
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
        """
        # Setup training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, len(compositions), batch_size):
                batch_end = min(i + batch_size, len(compositions))
                
                optimizer.zero_grad()
                batch_loss = 0.0
                
                for j in range(i, batch_end):
                    composition = compositions[j - i]
                    target_structure = structures[j - i]
                    
                    # Prepare input
                    atom_types = torch.LongTensor(composition).unsqueeze(0).to(self.device)
                    # Use random initial coordinates
                    init_coords = torch.randn(1, len(composition), 3).to(self.device)
                    
                    # Target coordinates
                    target_coords = torch.FloatTensor(target_structure['coordinates']).to(self.device)
                    target_coords = target_coords.flatten()
                    
                    # Forward pass
                    outputs = self.model(atom_types, init_coords)
                    prediction = outputs['predictions'].squeeze()
                    
                    # Pad or truncate to match target
                    if len(prediction) > len(target_coords):
                        prediction = prediction[:len(target_coords)]
                    elif len(prediction) < len(target_coords):
                        padding = torch.zeros(len(target_coords) - len(prediction)).to(self.device)
                        prediction = torch.cat([prediction, padding])
                    
                    loss = criterion(prediction, target_coords)
                    batch_loss += loss
                    
                batch_loss = batch_loss / (batch_end - i)
                batch_loss.backward()
                optimizer.step()
                
                epoch_loss += batch_loss.item()
                n_batches += 1
                
            if epoch % 20 == 0:
                self.logger.info(f"Epoch {epoch}: Loss={epoch_loss/n_batches:.4f}")
                
        self.is_fitted = True
        self.logger.info("Training completed")
        
    def predict(self, composition: List[int]) -> Dict[str, np.ndarray]:
        """
        Predict crystal structure from composition.
        
        Args:
            composition: Atomic composition
            
        Returns:
            Dictionary containing predicted structure
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        self.model.eval()
        
        with torch.no_grad():
            atom_types = torch.LongTensor(composition).unsqueeze(0).to(self.device)
            # Use random initial coordinates
            init_coords = torch.randn(1, len(composition), 3).to(self.device)
            
            outputs = self.model(atom_types, init_coords)
            prediction = outputs['predictions'].cpu().numpy()
            
            # Reshape to coordinates
            n_atoms = len(composition)
            coordinates = prediction.reshape(n_atoms, 3)
        
        return {
            'atom_types': composition,
            'coordinates': coordinates,
            'n_atoms': n_atoms
        }


class CatalystDesigner(BasePredictor):
    """
    Catalyst design and optimization predictor.
    
    This predictor can design catalysts for specific reactions and
    predict their activity and selectivity.
    """
    
    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        device: str = 'auto'
    ):
        super().__init__(device=device)
        
        self.model_params = model_params or {}
        self.is_fitted = False
        
        self._build_model()
        
    def _build_model(self):
        """Build the catalyst design model."""
        self.model = CatalystDesignNet(**self.model_params)
        self.model.to(self.device)
        
    def fit(
        self,
        catalyst_structures: List[Dict[str, Any]],
        reaction_conditions: List[Dict[str, Any]],
        activities: List[float],
        selectivities: List[float],
        batch_size: int = 16,
        num_epochs: int = 100,
        learning_rate: float = 1e-3
    ):
        """
        Fit the catalyst design model.
        
        Args:
            catalyst_structures: List of catalyst structure dictionaries
            reaction_conditions: List of reaction condition dictionaries
            activities: List of catalytic activities
            selectivities: List of selectivities
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
        """
        # Combine activities and selectivities
        targets = np.column_stack([activities, selectivities])
        
        # Setup training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, len(catalyst_structures), batch_size):
                batch_end = min(i + batch_size, len(catalyst_structures))
                
                optimizer.zero_grad()
                batch_loss = 0.0
                
                for j in range(i, batch_end):
                    catalyst = catalyst_structures[j - i]
                    reaction = reaction_conditions[j - i]
                    target = torch.FloatTensor(targets[j - i]).to(self.device)
                    
                    # Prepare input
                    atom_features = torch.FloatTensor(catalyst['atom_features']).to(self.device)
                    bond_indices = torch.LongTensor(catalyst['bond_indices']).to(self.device)
                    reaction_features = torch.FloatTensor(reaction['features']).unsqueeze(0).to(self.device)
                    
                    # Forward pass
                    outputs = self.model(atom_features, bond_indices, reaction_features)
                    prediction = outputs['predictions'].squeeze()
                    
                    loss = criterion(prediction, target)
                    batch_loss += loss
                    
                batch_loss = batch_loss / (batch_end - i)
                batch_loss.backward()
                optimizer.step()
                
                epoch_loss += batch_loss.item()
                n_batches += 1
                
            if epoch % 20 == 0:
                self.logger.info(f"Epoch {epoch}: Loss={epoch_loss/n_batches:.4f}")
                
        self.is_fitted = True
        self.logger.info("Training completed")
        
    def predict(
        self,
        catalyst_structure: Dict[str, Any],
        reaction_conditions: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Predict catalytic activity and selectivity.
        
        Args:
            catalyst_structure: Catalyst structure dictionary
            reaction_conditions: Reaction conditions dictionary
            
        Returns:
            Dictionary containing activity and selectivity predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        self.model.eval()
        
        with torch.no_grad():
            atom_features = torch.FloatTensor(catalyst_structure['atom_features']).to(self.device)
            bond_indices = torch.LongTensor(catalyst_structure['bond_indices']).to(self.device)
            reaction_features = torch.FloatTensor(reaction_conditions['features']).unsqueeze(0).to(self.device)
            
            outputs = self.model(atom_features, bond_indices, reaction_features)
            prediction = outputs['predictions'].cpu().numpy().flatten()
        
        return {
            'activity': float(prediction[0]),
            'selectivity': float(prediction[1])
        }


class PhaseAnalyzer(BasePredictor):
    """
    Phase diagram and phase transition predictor.
    
    This predictor can predict stable phases and phase boundaries
    from composition and thermodynamic conditions.
    """
    
    def __init__(
        self,
        n_phases: int = 5,
        model_params: Optional[Dict[str, Any]] = None,
        device: str = 'auto'
    ):
        super().__init__(device=device)
        
        self.n_phases = n_phases
        self.model_params = model_params or {}
        self.is_fitted = False
        
        self._build_model()
        
    def _build_model(self):
        """Build the phase prediction model."""
        self.model = PhasePredictor(
            n_phases=self.n_phases,
            **self.model_params
        )
        self.model.to(self.device)
        
    def fit(
        self,
        compositions: List[List[float]],
        conditions: List[List[float]],
        phases: List[int],
        stabilities: List[float],
        batch_size: int = 32,
        num_epochs: int = 100,
        learning_rate: float = 1e-3
    ):
        """
        Fit the phase prediction model.
        
        Args:
            compositions: List of material compositions
            conditions: List of thermodynamic conditions
            phases: List of stable phase indices
            stabilities: List of stability scores
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
        """
        # Setup training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        phase_criterion = torch.nn.CrossEntropyLoss()
        stability_criterion = torch.nn.MSELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, len(compositions), batch_size):
                batch_end = min(i + batch_size, len(compositions))
                
                # Prepare batch
                batch_compositions = torch.FloatTensor(compositions[i:batch_end]).to(self.device)
                batch_conditions = torch.FloatTensor(conditions[i:batch_end]).to(self.device)
                batch_phases = torch.LongTensor(phases[i:batch_end]).to(self.device)
                batch_stabilities = torch.FloatTensor(stabilities[i:batch_end]).to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_compositions, batch_conditions)
                
                # Compute losses
                phase_loss = phase_criterion(outputs['phase_logits'], batch_phases)
                stability_loss = stability_criterion(outputs['stability_score'].squeeze(), batch_stabilities)
                
                total_loss = phase_loss + stability_loss
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                n_batches += 1
                
            if epoch % 20 == 0:
                self.logger.info(f"Epoch {epoch}: Loss={epoch_loss/n_batches:.4f}")
                
        self.is_fitted = True
        self.logger.info("Training completed")
        
    def predict(
        self,
        composition: List[float],
        conditions: List[float]
    ) -> Dict[str, Any]:
        """
        Predict stable phase and stability.
        
        Args:
            composition: Material composition
            conditions: Thermodynamic conditions
            
        Returns:
            Dictionary containing phase predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        self.model.eval()
        
        with torch.no_grad():
            comp_tensor = torch.FloatTensor(composition).unsqueeze(0).to(self.device)
            cond_tensor = torch.FloatTensor(conditions).unsqueeze(0).to(self.device)
            
            outputs = self.model(comp_tensor, cond_tensor)
            
            phase_probs = outputs['phase_probabilities'].cpu().numpy().flatten()
            stability = outputs['stability_score'].cpu().numpy().flatten()[0]
            predicted_phase = np.argmax(phase_probs)
        
        return {
            'predicted_phase': int(predicted_phase),
            'phase_probabilities': phase_probs.tolist(),
            'stability_score': float(stability),
            'confidence': float(phase_probs[predicted_phase])
        }


class ElectronicStructureAnalyzer(BasePredictor):
    """
    Electronic structure property predictor.
    
    This predictor can predict band gaps, density of states,
    and other electronic properties from crystal structure.
    """
    
    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        device: str = 'auto'
    ):
        super().__init__(device=device)
        
        self.model_params = model_params or {}
        self.is_fitted = False
        
        self._build_model()
        
    def _build_model(self):
        """Build the electronic structure model."""
        self.model = ElectronicStructureNet(**self.model_params)
        self.model.to(self.device)
        
    def fit(
        self,
        structures: List[Dict[str, Any]],
        band_gaps: List[float],
        dos_spectra: List[np.ndarray],
        fermi_levels: List[float],
        batch_size: int = 16,
        num_epochs: int = 100,
        learning_rate: float = 1e-3
    ):
        """
        Fit the electronic structure model.
        
        Args:
            structures: List of crystal structures
            band_gaps: List of band gap values
            dos_spectra: List of density of states spectra
            fermi_levels: List of Fermi level values
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
        """
        # Setup training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        mse_criterion = torch.nn.MSELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, len(structures), batch_size):
                batch_end = min(i + batch_size, len(structures))
                
                optimizer.zero_grad()
                batch_loss = 0.0
                
                for j in range(i, batch_end):
                    structure = structures[j - i]
                    
                    # Prepare input
                    atom_features = torch.FloatTensor(structure['atom_features']).to(self.device)
                    bond_indices = torch.LongTensor(structure['bond_indices']).to(self.device)
                    
                    # Targets
                    target_bg = torch.FloatTensor([band_gaps[j - i]]).to(self.device)
                    target_dos = torch.FloatTensor(dos_spectra[j - i]).to(self.device)
                    target_fermi = torch.FloatTensor([fermi_levels[j - i]]).to(self.device)
                    
                    # Forward pass
                    outputs = self.model(atom_features, bond_indices)
                    
                    # Compute losses
                    bg_loss = mse_criterion(outputs['band_gap'].squeeze(), target_bg)
                    dos_loss = mse_criterion(outputs['dos_spectrum'].squeeze(), target_dos)
                    fermi_loss = mse_criterion(outputs['fermi_level'].squeeze(), target_fermi)
                    
                    total_loss = bg_loss + dos_loss + fermi_loss
                    batch_loss += total_loss
                    
                batch_loss = batch_loss / (batch_end - i)
                batch_loss.backward()
                optimizer.step()
                
                epoch_loss += batch_loss.item()
                n_batches += 1
                
            if epoch % 20 == 0:
                self.logger.info(f"Epoch {epoch}: Loss={epoch_loss/n_batches:.4f}")
                
        self.is_fitted = True
        self.logger.info("Training completed")
        
    def predict(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict electronic structure properties.
        
        Args:
            structure: Crystal structure dictionary
            
        Returns:
            Dictionary containing electronic structure predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        self.model.eval()
        
        with torch.no_grad():
            atom_features = torch.FloatTensor(structure['atom_features']).to(self.device)
            bond_indices = torch.LongTensor(structure['bond_indices']).to(self.device)
            
            outputs = self.model(atom_features, bond_indices)
            
            band_gap = outputs['band_gap'].cpu().numpy().flatten()[0]
            dos_spectrum = outputs['dos_spectrum'].cpu().numpy().flatten()
            fermi_level = outputs['fermi_level'].cpu().numpy().flatten()[0]
        
        return {
            'band_gap': float(band_gap),
            'dos_spectrum': dos_spectrum.tolist(),
            'fermi_level': float(fermi_level),
            'is_metal': float(band_gap) < 0.1,
            'is_semiconductor': 0.1 <= float(band_gap) <= 3.0,
            'is_insulator': float(band_gap) > 3.0
        } 