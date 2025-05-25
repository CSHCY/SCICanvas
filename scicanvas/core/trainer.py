"""
Training framework for SCICanvas models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, List, Callable
from pathlib import Path
import json
import time
from tqdm import tqdm
import wandb

from .base import BaseModel
from .config import Config

logger = logging.getLogger(__name__)


class Trainer:
    """
    Universal trainer for all SCICanvas models.
    
    Provides a unified interface for training models across different domains
    with support for various optimizers, schedulers, and logging backends.
    """
    
    def __init__(
        self,
        model: BaseModel,
        config: Config,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = 'auto'
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Setup device
        self.device = self._setup_device(device)
        self.model.to(self.device)
        
        # Setup training components
        self.criterion = criterion or self._get_default_criterion()
        self.optimizer = optimizer or self._get_optimizer()
        self.scheduler = scheduler or self._get_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Setup logging
        self._setup_logging()
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        return torch.device(device)
        
    def _setup_logging(self):
        """Setup logging and experiment tracking."""
        # Create directories
        log_dir = Path(self.config.log_dir)
        checkpoint_dir = Path(self.config.checkpoint_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup tensorboard
        self.writer = SummaryWriter(log_dir / 'tensorboard')
        
        # Setup wandb if experiment name is provided
        if self.config.experiment_name:
            wandb.init(
                project="scicanvas",
                name=self.config.experiment_name,
                config=self.config.to_dict(),
                tags=self.config.tags,
                notes=self.config.notes
            )
            
    def _get_default_criterion(self) -> nn.Module:
        """Get default loss function based on model type."""
        # This can be overridden by domain-specific trainers
        return nn.MSELoss()
        
    def _get_optimizer(self) -> optim.Optimizer:
        """Get optimizer based on configuration."""
        optimizer_name = self.config.optimizer.lower()
        lr = self.config.learning_rate
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        elif optimizer_name == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
            
    def _get_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Get learning rate scheduler based on configuration."""
        scheduler_name = self.config.scheduler.lower()
        
        if scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.num_epochs
            )
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        elif scheduler_name == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=10
            )
        elif scheduler_name == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
            
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = self._compute_loss(outputs, batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Log batch metrics
            if batch_idx % self.config.log_every == 0:
                step = self.current_epoch * num_batches + batch_idx
                self.writer.add_scalar('train/batch_loss', loss.item(), step)
                
                if self.config.experiment_name:
                    wandb.log({'train/batch_loss': loss.item()}, step=step)
                    
        return total_loss / num_batches
        
    def validate_epoch(self) -> float:
        """Validate for one epoch."""
        if self.val_loader is None:
            return float('inf')
            
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch = self._move_batch_to_device(batch)
                outputs = self.model(batch)
                loss = self._compute_loss(outputs, batch)
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)
        
    def test(self) -> Dict[str, float]:
        """Test the model."""
        if self.test_loader is None:
            logger.warning("No test loader provided")
            return {}
            
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                batch = self._move_batch_to_device(batch)
                outputs = self.model(batch)
                loss = self._compute_loss(outputs, batch)
                total_loss += loss.item()
                
        test_loss = total_loss / len(self.test_loader)
        
        results = {'test_loss': test_loss}
        logger.info(f"Test results: {results}")
        
        return results
        
    def train(self) -> Dict[str, List[float]]:
        """Train the model for the specified number of epochs."""
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Model has {self.model.get_num_trainable_parameters():,} trainable parameters")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate epoch
            val_loss = self.validate_epoch()
            self.val_losses.append(val_loss)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                    
            # Log epoch metrics
            self.writer.add_scalar('train/epoch_loss', train_loss, epoch)
            self.writer.add_scalar('val/epoch_loss', val_loss, epoch)
            self.writer.add_scalar('train/learning_rate', 
                                 self.optimizer.param_groups[0]['lr'], epoch)
            
            if self.config.experiment_name:
                wandb.log({
                    'train/epoch_loss': train_loss,
                    'val/epoch_loss': val_loss,
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch
                })
                
            # Save checkpoint
            if epoch % self.config.save_every == 0 or val_loss < self.best_val_loss:
                self.save_checkpoint(epoch, val_loss < self.best_val_loss)
                
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                
            # Log progress
            logger.info(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, lr={self.optimizer.param_groups[0]['lr']:.6f}"
            )
            
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Final test
        test_results = self.test()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'test_results': test_results,
            'training_time': training_time
        }
        
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict()
        }
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with val_loss={self.best_val_loss:.4f}")
            
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        
    def _move_batch_to_device(self, batch):
        """Move batch to the appropriate device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                   for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [item.to(self.device) if isinstance(item, torch.Tensor) else item 
                   for item in batch]
        else:
            return batch
            
    def _compute_loss(self, outputs, batch):
        """Compute loss - to be overridden by domain-specific trainers."""
        # This is a basic implementation that assumes outputs and targets
        # are in the expected format. Domain-specific trainers should override this.
        if isinstance(batch, dict) and 'targets' in batch:
            targets = batch['targets']
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            targets = batch[1]
        else:
            raise ValueError("Cannot determine targets from batch format")
            
        return self.criterion(outputs, targets) 