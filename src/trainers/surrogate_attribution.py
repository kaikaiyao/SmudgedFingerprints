"""
Surrogate Attribution Model Trainer.

Trains a model h_s that directly maps images to model source labels,
used for B1 (black-box surrogate classifier) attacks.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, List

from .base import BaseTrainer
from ..utils import load_images


class SurrogateAttributionModelTrainer(BaseTrainer):
    """
    Trainer for surrogate attribution models h_s.
    
    This trainer builds and trains a CNN that directly maps from images
    to model source labels (e.g., "stylegan2", "stylegan3", etc.).
    Used for B1 attacks where we don't have access to the fingerprint extractor.
    """
    
    def __init__(self, fingerprint_method: str, dataset: str, image_size: int,
                 device: str = "cuda", batch_size: int = 32, learning_rate: float = 1e-4,
                 num_epochs: int = 15, patience: int = 999999):
        """
        Initialize the surrogate attribution model trainer.
        
        Args:
            fingerprint_method: Name of the fingerprint method to use
            dataset: Dataset name
            image_size: Size of input images
            device: Device to train on
            batch_size: Training batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            patience: Patience for early stopping (set to very high to disable)
        """
        super().__init__(
            model_name="surrogate_attribution",
            device=device,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            patience=patience
        )
        
        self.dataset = dataset
        self.image_size = image_size
    
    def build_model(self, input_dim: int, output_dim: int, 
                   hidden_dims: List[int] = [512, 256, 128]) -> nn.Module:
        """
        Build CNN architecture for image-to-label classification.
        
        Args:
            input_dim: Input dimension (not used for CNN)
            output_dim: Number of model classes
            hidden_dims: Hidden layer dimensions
            
        Returns:
            CNN model
        """
        class SurrogateAttributionCNN(nn.Module):
            def __init__(self, image_size: int, num_classes: int, hidden_dims: List[int]):
                super().__init__()
                
                self.image_size = image_size
                self.num_classes = num_classes
                
                # Convolutional layers
                self.conv_layers = nn.Sequential(
                    # First conv block
                    nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    
                    # Second conv block
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    
                    # Third conv block
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    
                    # Fourth conv block
                    nn.Conv2d(256, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((4, 4))
                )
                
                # Calculate flattened size
                conv_output_size = 512 * 4 * 4
                
                # Fully connected layers
                fc_layers = []
                prev_dim = conv_output_size
                
                for hidden_dim in hidden_dims:
                    fc_layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.5)
                    ])
                    prev_dim = hidden_dim
                
                # Output layer
                fc_layers.append(nn.Linear(prev_dim, num_classes))
                
                self.fc_layers = nn.Sequential(*fc_layers)
            
            def forward(self, x):
                # Ensure input is in correct range [0, 1]
                if x.min() < 0:
                    x = (x + 1) / 2  # Convert from [-1, 1] to [0, 1]
                
                x = self.conv_layers(x)
                x = x.view(x.size(0), -1)  # Flatten
                x = self.fc_layers(x)
                return x
        
        return SurrogateAttributionCNN(self.image_size, output_dim, hidden_dims)
    
    def get_model_save_path(self, **kwargs) -> Path:
        """Get path to save the surrogate attribution model."""
        # Note: This method is overridden by the entry scripts (train_models.py, run_attacks.py)
        # with lambda functions that return the correct paths for the new directory structure.
        # This implementation is kept for backward compatibility but is not used by the main workflow.
        raise NotImplementedError("Use the new entry scripts (train_models.py, run_attacks.py) for model path management")
    
    def prepare_training_data(self, model_names: List[str], 
                            num_images_per_model: int = 1000) -> tuple:
        """
        Prepare training data by loading images from multiple models.
        
        Args:
            model_names: List of model names to include
            num_images_per_model: Number of images to load per model
            
        Returns:
            Tuple of (images, labels)
        """
        all_images = []
        all_labels = []
        
        # Create label mapping
        label_map = {model_name: idx for idx, model_name in enumerate(model_names)}
        
        for model_name in model_names:
            self.logger.info(f"Loading images from {model_name}")
            
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
            
            # Create labels
            labels = torch.full((len(images),), label_map[model_name], dtype=torch.long)
            
            all_images.append(images)
            all_labels.append(labels)
            
            self.logger.info(f"Loaded {len(images)} images from {model_name}")
        
        if not all_images:
            raise ValueError("No images loaded. Check that image files exist.")
        
        # Concatenate all data
        X = torch.cat(all_images, dim=0)
        y = torch.cat(all_labels, dim=0)
        
        self.logger.info(f"Total training data: {X.shape}, Labels: {y.shape}")
        self.logger.info(f"Label distribution: {torch.bincount(y)}")
        
        return X, y, label_map
    
    def train_from_models(self, model_names: List[str], 
                         num_images_per_model: int = 1000,
                         val_split: float = 0.2, **model_kwargs) -> nn.Module:
        """
        Train surrogate attribution model from multiple generative models.
        
        Args:
            model_names: List of model names to include in training
            num_images_per_model: Number of images per model
            val_split: Validation split ratio
            **model_kwargs: Additional model parameters
            
        Returns:
            Trained surrogate attribution model
        """
        # Prepare data
        X, y, label_map = self.prepare_training_data(model_names, num_images_per_model)
        
        # Store label mapping for later use
        self.label_map = label_map
        self.reverse_label_map = {v: k for k, v in label_map.items()}
        
        # Train model
        model = self.train(X, y, val_split, **model_kwargs)
        
        # Save label mapping along with model
        self.save_model(label_map=label_map)
        
        return model
    
    def predict(self, images: torch.Tensor) -> tuple:
        """
        Predict model labels for images.
        
        Args:
            images: Input images tensor
            
        Returns:
            Tuple of (predicted_labels, confidence_scores)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        self.model.eval()
        with torch.no_grad():
            images = images.to(self.device)
            outputs = self.model(images)
            
            # Get predictions and confidence scores
            confidence_scores, predicted_indices = torch.max(torch.softmax(outputs, dim=1), 1)
            
            # Convert indices to model names if label map is available
            if hasattr(self, 'reverse_label_map'):
                predicted_labels = [self.reverse_label_map[idx.item()] for idx in predicted_indices]
            else:
                predicted_labels = predicted_indices.cpu().numpy().tolist()
        
        return predicted_labels, confidence_scores.cpu().numpy()
    
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
            'dataset': self.dataset,
            'image_size': self.image_size,
            'device': self.device,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'label_map': kwargs.get('label_map', getattr(self, 'label_map', {})),
            'input_dim': self.model.input_dim if hasattr(self.model, 'input_dim') else None,
            'output_dim': self.model.num_classes if hasattr(self.model, 'num_classes') else None
        }
        
        torch.save(model_data, model_path)
        self.logger.info(f"Surrogate attribution model saved to {model_path}")
    
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
        self.dataset = model_data.get('dataset', self.dataset)
        self.image_size = model_data.get('image_size', self.image_size)
        self.label_map = model_data.get('label_map', {})
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        # Load training history
        self.train_losses = model_data.get('train_losses', [])
        self.val_losses = model_data.get('val_losses', [])
        self.train_metrics = model_data.get('train_metrics', [])
        self.val_metrics = model_data.get('val_metrics', [])
        
        self.logger.info(f"Surrogate attribution model loaded from {model_path}")
        return self.model