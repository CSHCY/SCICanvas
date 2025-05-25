"""
Base classes for all models and predictors in SCICanvas.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all neural network models in SCICanvas.
    
    This class provides common functionality for model initialization,
    saving/loading, and basic forward pass structure.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for the model."""
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor
            **kwargs: Additional keyword arguments
            
        Returns:
            Output tensor
        """
        pass
    
    def save_model(self, path: Union[str, Path]) -> None:
        """
        Save model state dict and configuration.
        
        Args:
            path: Path to save the model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'state_dict': self.state_dict(),
            'config': self.config,
            'model_class': self.__class__.__name__
        }
        
        torch.save(save_dict, path)
        self.logger.info(f"Model saved to {path}")
        
    def load_model(self, path: Union[str, Path]) -> None:
        """
        Load model state dict and configuration.
        
        Args:
            path: Path to load the model from
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
            
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['state_dict'])
        self.config.update(checkpoint.get('config', {}))
        self.logger.info(f"Model loaded from {path}")
        
    def get_num_parameters(self) -> int:
        """Get the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
        
    def get_num_trainable_parameters(self) -> int:
        """Get the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BasePredictor(ABC):
    """
    Abstract base class for all predictors in SCICanvas.
    
    This class provides a common interface for prediction tasks
    across different scientific domains.
    """
    
    def __init__(self, model: Optional[BaseModel] = None, device: str = 'auto'):
        self.model = model
        self.device = self._setup_device(device)
        self._setup_logging()
        
        if self.model is not None:
            self.model.to(self.device)
            
    def _setup_device(self, device: str) -> torch.device:
        """Setup the computation device."""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        return torch.device(device)
        
    def _setup_logging(self):
        """Setup logging for the predictor."""
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def predict(self, data: Any, **kwargs) -> Any:
        """
        Make predictions on input data.
        
        Args:
            data: Input data for prediction
            **kwargs: Additional keyword arguments
            
        Returns:
            Predictions
        """
        pass
        
    def predict_batch(self, data_list: List[Any], batch_size: int = 32, **kwargs) -> List[Any]:
        """
        Make predictions on a batch of data.
        
        Args:
            data_list: List of input data
            batch_size: Batch size for processing
            **kwargs: Additional keyword arguments
            
        Returns:
            List of predictions
        """
        predictions = []
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]
            batch_predictions = [self.predict(data, **kwargs) for data in batch]
            predictions.extend(batch_predictions)
        return predictions
        
    def save_predictor(self, path: Union[str, Path]) -> None:
        """
        Save the predictor including model and configuration.
        
        Args:
            path: Path to save the predictor
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'predictor_class': self.__class__.__name__,
            'device': str(self.device),
        }
        
        if self.model is not None:
            model_path = path.parent / f"{path.stem}_model.pt"
            self.model.save_model(model_path)
            save_dict['model_path'] = str(model_path)
            
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
            
        self.logger.info(f"Predictor saved to {path}")
        
    @classmethod
    def load_predictor(cls, path: Union[str, Path]) -> 'BasePredictor':
        """
        Load a predictor from file.
        
        Args:
            path: Path to load the predictor from
            
        Returns:
            Loaded predictor instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Predictor file not found: {path}")
            
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
            
        # Create predictor instance
        predictor = cls()
        predictor.device = torch.device(save_dict['device'])
        
        # Load model if available
        if 'model_path' in save_dict:
            model_path = Path(save_dict['model_path'])
            if model_path.exists():
                # This would need to be implemented by subclasses
                # as they know their specific model types
                pass
                
        return predictor 