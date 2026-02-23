"""Base classes for model trainers."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging
import numpy as np
from tqdm import tqdm

from ..utils import setup_logger, calculate_comprehensive_metrics


class BaseTrainer(ABC):
    """
    Base class for all model trainers.
    
    Provides common functionality for training different types of models
    in the fingerprint robustness evaluation framework.
    """
    
    def __init__(self, model_name: str, device: str = "cuda", num_epochs: int = 15, 
                 learning_rate: float = 1e-3, batch_size: int = 32, patience: int = 999999):
        """
        Initialize the base trainer.
        
        Args:
            model_name: Name of the model being trained
            device: Device to train on ("cuda", "mps", or "cpu")
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            patience: Patience for early stopping (set to very high to disable)
        """
        self.trainer_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.patience = patience
        
        self.logger = setup_logger(f"Trainer_{model_name}")
        
        # Training state
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
    
    @abstractmethod
    def build_model(self, input_dim: int, output_dim: int, **kwargs) -> nn.Module:
        """
        Build the model architecture.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            **kwargs: Additional model parameters
            
        Returns:
            PyTorch model
        """
        pass
    
    @abstractmethod
    def get_model_save_path(self, **kwargs) -> Path:
        """
        Get the path to save the trained model.
        
        Returns:
            Path to save the model
        """
        pass
    
    def prepare_data(self, X: torch.Tensor, y: torch.Tensor, 
                    val_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare training and validation data loaders.
        
        Args:
            X: Input features
            y: Target labels
            val_split: Validation split ratio
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Shuffle data
        indices = torch.randperm(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Split into train and validation
        val_size = int(len(X) * val_split)
        train_size = len(X) - val_size
        
        X_train, X_val = X_shuffled[:train_size], X_shuffled[train_size:]
        y_train, y_val = y_shuffled[:train_size], y_shuffled[train_size:]
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        return train_loader, val_loader
    
    def setup_training(self, model: nn.Module):
        """
        Setup optimizer, criterion, and scheduler.
        
        Args:
            model: Model to train
        """
        self.model = model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate
        )
        
        # Setup criterion (will be overridden by subclasses if needed)
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, metrics)
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Collect predictions for metrics
            _, predicted = torch.max(output.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        metrics = calculate_comprehensive_metrics(
            np.array(all_targets), np.array(all_predictions)
        )
        
        return avg_loss, metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, metrics)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                # Collect predictions for metrics
                _, predicted = torch.max(output.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        metrics = calculate_comprehensive_metrics(
            np.array(all_targets), np.array(all_predictions)
        )
        
        return avg_loss, metrics
    
    def train(self, X: torch.Tensor, y: torch.Tensor, 
              val_split: float = 0.2, **model_kwargs) -> nn.Module:
        """
        Train the model.
        
        Args:
            X: Input features
            y: Target labels
            val_split: Validation split ratio
            **model_kwargs: Additional model parameters
            
        Returns:
            Trained model
        """
        self.logger.info(f"Starting training for {self.trainer_name}")
        self.logger.info(f"Training data: {X.shape}, Labels: {y.shape}")
        
        # Build model
        input_dim = X.shape[1] if X.dim() > 1 else X.shape[0]
        output_dim = len(torch.unique(y))
        model = self.build_model(input_dim, output_dim, **model_kwargs)
        
        # Setup training
        self.setup_training(model)
        
        # Prepare data
        train_loader, val_loader = self.prepare_data(X, y, val_split)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_metrics.append(train_metrics)
            
            # Validate
            val_loss, val_metrics = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            # Log progress
            if (epoch + 1) % 1 == 0:  # Log every epoch
                self.logger.info(
                    f"Epoch {epoch+1:3d}/{self.num_epochs} - "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f} | "
                    f"Train Acc: {train_metrics['accuracy']:.6f}, Val Acc: {val_metrics['accuracy']:.6f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )
            
            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(epoch, val_loss)
                self.logger.info(f"  ✓ New best validation loss: {val_loss:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs (patience: {self.patience})")
                    break
        
        # Load best model
        self._load_checkpoint()
        
        self.logger.info("Training completed")
        self.logger.info(f"Best validation loss: {best_val_loss:.6f}")
        self.logger.info(f"Final training loss: {self.train_losses[-1]:.6f}")
        return self.model
    
    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
        
        checkpoint_path = self.get_model_save_path().with_suffix('.checkpoint')
        torch.save(checkpoint, checkpoint_path)
    
    def _load_checkpoint(self):
        """Load best model checkpoint."""
        checkpoint_path = self.get_model_save_path().with_suffix('.checkpoint')
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    def save_model(self, **kwargs):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_path = self.get_model_save_path(**kwargs)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'model_architecture': self.model,
            'trainer_name': self.trainer_name,
            'device': self.device,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
        
        torch.save(model_data, model_path)
        self.logger.info(f"Model saved to {model_path}")
    
    def load_model(self, **kwargs) -> nn.Module:
        """Load a previously trained model."""
        model_path = self.get_model_save_path(**kwargs)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_data = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Load model architecture and weights
        self.model = model_data['model_architecture']
        self.model.load_state_dict(model_data['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load training history
        self.train_losses = model_data.get('train_losses', [])
        self.val_losses = model_data.get('val_losses', [])
        self.train_metrics = model_data.get('train_metrics', [])
        self.val_metrics = model_data.get('val_metrics', [])
        
        self.logger.info(f"Model loaded from {model_path}")
        return self.model