"""
Surrogate Extractor Trainer.

Trains a surrogate fingerprint extractor φ_s that approximates the behavior
of non-differentiable fingerprint extractors, used for W2 attacks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Any, List

from .base import BaseTrainer
from ..fingerprints import FingerprintExtractor
from ..utils import load_images


class SurrogateExtractorTrainer(BaseTrainer):
    """
    Trainer for surrogate fingerprint extractors φ_s.
    
    This trainer builds and trains a neural network that learns to approximate
    the output of non-differentiable fingerprint extractors. Used for W2 attacks
    where we need gradient access to the fingerprint extraction process.
    """
    
    def __init__(self, fingerprint_method: str, dataset: str, image_size: int, 
                 device: str = "cuda", num_epochs: int = 15, learning_rate: float = 1e-4, 
                 batch_size: int = 32, patience: int = 999999):
        """
        Initialize the surrogate extractor trainer.
        
        Args:
            fingerprint_method: Fingerprint extraction method
            dataset: Dataset name
            image_size: Image size
            device: Device to train on
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            patience: Patience for early stopping (set to very high to disable)
        """
        super().__init__(
            model_name="surrogate_extractor",
            device=device,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            patience=patience
        )
        
        self.fingerprint_method = fingerprint_method
        self.dataset = dataset
        self.image_size = image_size
        
        # Note: We don't initialize original_extractor here because:
        # 1. During training, we receive pre-extracted fingerprints as input
        # 2. This avoids unnecessary re-download of FFHQ data
        # 3. The extractor is only needed for validation/comparison
        self.original_extractor = None
        
        # Override criterion for regression
        self.criterion = nn.MSELoss()
    
    def setup_training(self, model: nn.Module):
        """
        Setup optimizer, criterion, and scheduler (override for regression).
        
        Args:
            model: Model to train
        """
        self.model = model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate
        )
        
        # Setup criterion for regression
        self.criterion = nn.MSELoss()
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
    
    def build_model(self, input_dim: int = None, output_dim: int = None, 
                   hidden_dims: List[int] = [1024, 512, 256]) -> nn.Module:
        """
        Build CNN architecture for image-to-fingerprint regression.
        
        Args:
            input_dim: Input dimension (not used for CNN, kept for compatibility)
            output_dim: Fingerprint feature dimension
            hidden_dims: Hidden layer dimensions
            
        Returns:
            CNN model for fingerprint extraction
        """
        if output_dim is None:
            raise ValueError("output_dim must be specified")
        
        class SurrogateExtractorCNN(nn.Module):
            def __init__(self, image_size: int, feature_dim: int, hidden_dims: List[int]):
                super().__init__()
                
                self.image_size = image_size
                self.feature_dim = feature_dim  # Store feature dimension for validation
                
                # Convolutional feature extractor
                self.conv_layers = nn.Sequential(
                    # First conv block
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    
                    # Second conv block
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    
                    # Third conv block
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    
                    # Fourth conv block
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((4, 4))  # Adjusted to 4x4 output
                )
                
                # Calculate flattened size
                conv_output_size = 256 * 4 * 4  # From the last conv layer: 256 channels, 4x4 spatial size
                
                # Fully connected layers for regression
                self.fc_layers = nn.Sequential(
                    nn.Linear(conv_output_size, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(128, feature_dim)  # Output layer for regression
                )
            
            def forward(self, x):
                # Ensure input is in correct range [0, 1]
                if x.min() < 0:
                    x = (x + 1) / 2  # Convert from [-1, 1] to [0, 1]
                
                # Pass through convolutional layers
                x = self.conv_layers(x)  # Output: [batch_size, 256, 4, 4]
                
                # Flatten the output
                x = x.reshape(x.size(0), -1)  # Output: [batch_size, 256 * 4 * 4]
                
                # Pass through fully connected layers
                x = self.fc_layers(x)  # Output: [batch_size, feature_dim]
                
                # Ensure output has the correct shape
                if x.shape[-1] != self.feature_dim:
                    x = x.view(-1, self.feature_dim)
                    if x.shape[-1] != self.feature_dim:
                        raise ValueError(f"Model output dimension {x.shape[-1]} does not match expected {self.feature_dim}")
                
                return x
        
        # Ensure the model is built with the correct output dimension
        model = SurrogateExtractorCNN(self.image_size, output_dim, hidden_dims)
        
        # Verify the model output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, self.image_size, self.image_size)
            dummy_output = model(dummy_input)
            if dummy_output.shape[1] != output_dim:
                raise ValueError(f"Model output dimension {dummy_output.shape[1]} does not match expected {output_dim}")
        
        return model
    
    def train(self, X: torch.Tensor, y: torch.Tensor, 
              val_split: float = 0.2, **model_kwargs) -> nn.Module:
        """
        Train the surrogate extractor (regression task).
        
        Overrides BaseTrainer.train to avoid inferring output_dim from labels
        (which is classification-specific) and instead uses the feature
        dimension of the regression targets.
        """
        self.logger.info(f"Starting training for {self.trainer_name}")
        self.logger.info(f"Training data: {X.shape}, Labels: {y.shape}")
        self.logger.info(f"Training for {self.num_epochs} epochs with patience {self.patience}")
        
        # Determine output dimension from regression targets
        if y.dim() != 2:
            raise ValueError("SurrogateExtractorTrainer expects y to be a 2D tensor of regression targets [N, feature_dim]")
        feature_dim = y.shape[1]
        
        # Build model with correct output dimension
        model = self.build_model(output_dim=feature_dim, **model_kwargs)
        
        # Setup training
        self.setup_training(model)
        
        # Prepare data
        train_loader, val_loader = self.prepare_data(X, y, val_split)
        
        # Training loop (same structure as BaseTrainer but using regression epochs)
        best_val_loss = float('inf')
        
        self.logger.info("Starting training loop...")
        for epoch in range(self.num_epochs):
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_metrics.append(train_metrics)
            
            # Validate
            val_loss, val_metrics = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            # Log detailed progress for each epoch
            self.logger.info(
                f"Epoch {epoch+1:3d}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f} | "
                f"Train MSE: {train_metrics['mse']:.6f}, Val MSE: {val_metrics['mse']:.6f} | "
                f"Train MAE: {train_metrics['mae']:.6f}, Val MAE: {val_metrics['mae']:.6f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss)
                self.logger.info(f"  ✓ New best validation loss: {val_loss:.6f}")
        
        # Load best model
        self._load_checkpoint()
        
        self.logger.info("Training completed")
        self.logger.info(f"Best validation loss: {best_val_loss:.6f}")
        self.logger.info(f"Final training loss: {self.train_losses[-1]:.6f}")
        self.logger.info(f"Final validation loss: {self.val_losses[-1]:.6f}")
        return self.model
    
    def get_model_save_path(self, **kwargs) -> Path:
        """Get path to save the surrogate extractor model."""
        # Note: This method is overridden by the entry scripts (train_models.py, run_attacks.py)
        # with lambda functions that return the correct paths for the new directory structure.
        # This implementation is kept for backward compatibility but is not used by the main workflow.
        raise NotImplementedError("Use the new entry scripts (train_models.py, run_attacks.py) for model path management")
    
    def prepare_training_data(self, model_names: List[str], 
                            num_images_per_model: int = 1000) -> tuple:
        """
        Prepare training data by extracting fingerprints from images.
        
        Args:
            model_names: List of model names to include
            num_images_per_model: Number of images to load per model
            
        Returns:
            Tuple of (images, fingerprints)
        """
        all_images = []
        all_fingerprints = []
        
        for model_name in model_names:
            self.logger.info(f"Processing images from {model_name}")
            
            # Load images
            images_path = self.data_paths.get_images_path(
                model_name, self.dataset, self.image_size, num_images_per_model
            )
            
            if not images_path.exists():
                self.logger.warning(f"Images not found for {model_name}, skipping")
                continue
            
            images, metadata = load_images(images_path)
            
            # Take only the requested number of images
            if len(images) > num_images_per_model:
                images = images[:num_images_per_model]
            
            # Extract fingerprints using the original method
            self.logger.info(f"Extracting {self.fingerprint_method} fingerprints")
            
            # Initialize original_extractor only when needed
            if self.original_extractor is None:
                from ..fingerprints import FingerprintExtractor
                self.original_extractor = FingerprintExtractor.create(self.fingerprint_method, device=self.device)
            
            fingerprints = self.original_extractor.extract_fingerprint(images)
            
            all_images.append(images)
            all_fingerprints.append(fingerprints)
            
            self.logger.info(f"Processed {len(images)} images from {model_name}")
        
        if not all_images:
            raise ValueError("No images loaded. Check that image files exist.")
        
        # Concatenate all data
        X = torch.cat(all_images, dim=0)
        y = torch.cat(all_fingerprints, dim=0)
        
        self.logger.info(f"Total training data: Images {X.shape}, Fingerprints {y.shape}")
        
        return X, y
    
    def train_from_models(self, model_names: List[str], 
                         num_images_per_model: int = 1000,
                         val_split: float = 0.2, **model_kwargs) -> nn.Module:
        """
        Train surrogate extractor from multiple generative models.
        
        Args:
            model_names: List of model names to include in training
            num_images_per_model: Number of images per model
            val_split: Validation split ratio
            **model_kwargs: Additional model parameters
            
        Returns:
            Trained surrogate extractor model
        """
        # Prepare data
        X, y = self.prepare_training_data(model_names, num_images_per_model)
        
        # Train model
        model = self.train(X, y, val_split, **model_kwargs)
        
        # Save model
        self.save_model()
        
        return model
    
    def train_epoch(self, train_loader) -> tuple:
        """
        Train for one epoch (override for regression).
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, metrics)
        """
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        num_batches = 0
        
        # Show batch progress for small datasets
        total_batches = len(train_loader)
        if total_batches <= 10:  # Show progress for small datasets
            self.logger.info(f"  Training epoch with {total_batches} batches...")
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Ensure data is in the correct shape
            if data.dim() == 2:
                data = data.view(-1, 3, self.image_size, self.image_size)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # Ensure output and target have the same shape
            if output.shape != target.shape:
                batch_size = target.shape[0]
                output = output.reshape(batch_size, -1)
                if output.shape[1] != target.shape[1]:
                    raise ValueError(f"Model output dimension {output.shape[1]} does not match target dimension {target.shape[1]}")
            
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate regression metrics
            with torch.no_grad():
                mse = torch.mean((output - target) ** 2).item()
                mae = torch.mean(torch.abs(output - target)).item()
                total_mse += mse
                total_mae += mae
                num_batches += 1
                
                # Show batch progress for small datasets
                if total_batches <= 10:
                    self.logger.info(f"    Batch {batch_idx+1}/{total_batches}: Loss={loss.item():.6f}, MSE={mse:.6f}, MAE={mae:.6f}")
        
        avg_loss = total_loss / len(train_loader)
        metrics = {
            'mse': total_mse / num_batches,
            'mae': total_mae / num_batches,
            'rmse': (total_mse / num_batches) ** 0.5
        }
        
        return avg_loss, metrics
    
    def validate_epoch(self, val_loader) -> tuple:
        """
        Validate for one epoch (override for regression).
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, metrics)
        """
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        num_batches = 0
        
        # Show batch progress for small datasets
        total_batches = len(val_loader)
        if total_batches <= 10:  # Show progress for small datasets
            self.logger.info(f"  Validating epoch with {total_batches} batches...")
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Ensure data is in the correct shape
                if data.dim() == 2:
                    data = data.view(-1, 3, self.image_size, self.image_size)
                
                output = self.model(data)
                
                # Ensure output and target have the same shape
                if output.shape != target.shape:
                    batch_size = target.shape[0]
                    output = output.reshape(batch_size, -1)
                    if output.shape[1] != target.shape[1]:
                        raise ValueError(f"Model output dimension {output.shape[1]} does not match target dimension {target.shape[1]}")
                
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                # Calculate regression metrics
                mse = torch.mean((output - target) ** 2).item()
                mae = torch.mean(torch.abs(output - target)).item()
                total_mse += mse
                total_mae += mae
                num_batches += 1
                
                # Show batch progress for small datasets
                if total_batches <= 10:
                    self.logger.info(f"    Val Batch {batch_idx+1}/{total_batches}: Loss={loss.item():.6f}, MSE={mse:.6f}, MAE={mae:.6f}")
        
        avg_loss = total_loss / len(val_loader)
        metrics = {
            'mse': total_mse / num_batches,
            'mae': total_mae / num_batches,
            'rmse': (total_mse / num_batches) ** 0.5
        }
        
        return avg_loss, metrics
    
    def extract_fingerprint(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract fingerprints using the trained surrogate model.
        
        Args:
            images: Input images tensor
            
        Returns:
            Extracted fingerprint features
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        self.model.eval()
        with torch.no_grad():
            images = images.to(self.device)
            fingerprints = self.model(images)
        
        return fingerprints.cpu()
    
    def validate_approximation(self, test_images: torch.Tensor) -> Dict[str, float]:
        """
        Validate how well the surrogate approximates the original extractor.
        
        Args:
            test_images: Test images to validate on
            
        Returns:
            Validation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Extract fingerprints using both methods
        # Initialize original_extractor only when needed
        if self.original_extractor is None:
            from ..fingerprints import FingerprintExtractor
            self.original_extractor = FingerprintExtractor.create(self.fingerprint_method, device=self.device)
        
        original_features = self.original_extractor.extract_fingerprint(test_images)
        surrogate_features = self.extract_fingerprint(test_images)
        
        # Calculate approximation quality metrics
        mse = torch.mean((original_features - surrogate_features) ** 2).item()
        mae = torch.mean(torch.abs(original_features - surrogate_features)).item()
        max_error = torch.max(torch.abs(original_features - surrogate_features)).item()
        
        # Correlation coefficient
        original_flat = original_features.flatten()
        surrogate_flat = surrogate_features.flatten()
        correlation = torch.corrcoef(torch.stack([original_flat, surrogate_flat]))[0, 1].item()
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': mse ** 0.5,
            'max_error': max_error,
            'correlation': correlation
        }
        
        self.logger.info(f"Surrogate approximation quality: MSE={mse:.6f}, "
                        f"MAE={mae:.6f}, Correlation={correlation:.4f}")
        
        return metrics
    
    def save_model(self, **kwargs):
        """Save model with additional metadata."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_path = self.get_model_save_path()
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'model_architecture': self.model,
            'trainer_name': self.trainer_name,
            'fingerprint_method': self.fingerprint_method,
            'dataset': self.dataset,
            'image_size': self.image_size,
            'device': self.device,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'original_extractor_info': self.original_extractor.get_method_info() if self.original_extractor is not None else None,
            'input_dim': self.model.input_dim if hasattr(self.model, 'input_dim') else None,
            'output_dim': self.model.output_dim if hasattr(self.model, 'output_dim') else None
        }
        
        torch.save(model_data, model_path)
        self.logger.info(f"Surrogate extractor model saved to {model_path}")
    
    def load_model(self, **kwargs) -> nn.Module:
        """Load model with additional metadata."""
        model_path = self.get_model_save_path()
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_data = torch.load(model_path, map_location=self.device)
        
        # Load model
        self.model = model_data['model_architecture']
        self.model.load_state_dict(model_data['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load metadata
        self.fingerprint_method = model_data.get('fingerprint_method', self.fingerprint_method)
        self.dataset = model_data.get('dataset', self.dataset)
        self.image_size = model_data.get('image_size', self.image_size)
        
        # Load training history
        self.train_losses = model_data.get('train_losses', [])
        self.val_losses = model_data.get('val_losses', [])
        self.train_metrics = model_data.get('train_metrics', [])
        self.val_metrics = model_data.get('val_metrics', [])
        
        self.logger.info(f"Surrogate extractor model loaded from {model_path}")
        return self.model