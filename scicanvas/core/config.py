"""
Configuration management for SCICanvas toolkit.
"""

import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """
    Base configuration class for SCICanvas models and experiments.
    """
    
    # Model configuration
    model_name: str = "base_model"
    model_type: str = "neural_network"
    
    # Training configuration
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    optimizer: str = "adam"
    scheduler: str = "cosine"
    
    # Data configuration
    data_path: Optional[str] = None
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Hardware configuration
    device: str = "auto"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Logging and checkpointing
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 10
    log_every: int = 100
    
    # Experiment tracking
    experiment_name: Optional[str] = None
    tags: Optional[list] = None
    notes: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.tags is None:
            self.tags = []
            
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        Create a Config instance from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            Config instance
        """
        # Filter out keys that are not in the dataclass fields
        valid_keys = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        return cls(**filtered_dict)
        
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'Config':
        """
        Load configuration from a YAML file.
        
        Args:
            yaml_path: Path to the YAML configuration file
            
        Returns:
            Config instance
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
            
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        return cls.from_dict(config_dict)
        
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'Config':
        """
        Load configuration from a JSON file.
        
        Args:
            json_path: Path to the JSON configuration file
            
        Returns:
            Config instance
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {json_path}")
            
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
            
        return cls.from_dict(config_dict)
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return asdict(self)
        
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            yaml_path: Path to save the YAML configuration file
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
            
        logger.info(f"Configuration saved to {yaml_path}")
        
    def to_json(self, json_path: Union[str, Path]) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            json_path: Path to save the JSON configuration file
        """
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
        logger.info(f"Configuration saved to {json_path}")
        
    def update(self, **kwargs) -> None:
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
                
    def merge(self, other_config: 'Config') -> 'Config':
        """
        Merge with another configuration.
        
        Args:
            other_config: Another Config instance to merge with
            
        Returns:
            New Config instance with merged parameters
        """
        merged_dict = self.to_dict()
        merged_dict.update(other_config.to_dict())
        return self.from_dict(merged_dict)
        
    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"Config({self.to_dict()})"
        
    def __repr__(self) -> str:
        """Detailed string representation of the configuration."""
        return self.__str__()


# Domain-specific configuration classes

@dataclass
class SingleCellConfig(Config):
    """Configuration for single-cell analysis models."""
    
    # Single-cell specific parameters
    n_genes: Optional[int] = None
    n_cells: Optional[int] = None
    cell_types: Optional[list] = None
    
    # Preprocessing parameters
    min_genes_per_cell: int = 200
    min_cells_per_gene: int = 3
    max_genes_per_cell: int = 5000
    max_mito_percent: float = 20.0
    
    # Normalization parameters
    normalize_total: bool = True
    target_sum: float = 1e4
    log_transform: bool = True
    
    # Feature selection
    n_top_genes: int = 2000
    
    def __post_init__(self):
        super().__post_init__()
        if self.cell_types is None:
            self.cell_types = []


@dataclass 
class ProteinConfig(Config):
    """Configuration for protein prediction models."""
    
    # Protein specific parameters
    max_sequence_length: int = 1024
    amino_acid_vocab_size: int = 21  # 20 amino acids + unknown
    
    # Structure prediction parameters
    predict_structure: bool = True
    predict_function: bool = False
    predict_interactions: bool = False
    
    # Model architecture
    embedding_dim: int = 512
    num_layers: int = 12
    num_heads: int = 8
    dropout: float = 0.1


@dataclass
class MaterialsConfig(Config):
    """Configuration for materials science models."""
    
    # Materials specific parameters
    max_atoms: int = 200
    atom_features_dim: int = 92  # Number of elements in periodic table
    
    # Crystal structure parameters
    predict_properties: bool = True
    predict_stability: bool = False
    predict_bandgap: bool = False
    
    # Graph neural network parameters
    node_features_dim: int = 128
    edge_features_dim: int = 64
    num_conv_layers: int = 6
    pooling_method: str = "global_mean" 