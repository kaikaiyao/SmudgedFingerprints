"""
True Attribution Model Trainer.

Trains the true attribution model h that maps fingerprint features 
to model source labels, used as the target model for attacks.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, List

from .base import BaseTrainer
from ..fingerprints import FingerprintExtractor


class TrueAttributionModelTrainer(BaseTrainer):
    """
    Trainer for true attribution models h.
    
    This trainer builds and trains a classifier that maps from fingerprint features
    to model source labels. This is the target model that attacks try to fool.
    """
    
    def __init__(self, fingerprint_method: str, dataset: str, image_size: int,
                 device: str = "cuda", batch_size: int = 32, learning_rate: float = 1e-6,
                 num_epochs: int = 15, patience: int = 999999):
        """
        Initialize the true attribution model trainer.
        
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
            model_name="true_attribution",
            device=device,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            patience=patience
        )
        
        self.fingerprint_method = fingerprint_method
        self.dataset = dataset
        self.image_size = image_size
        
        # Note: We don't initialize fingerprint_extractor here because:
        # 1. During training, we receive pre-extracted fingerprints as input
        # 2. This avoids unnecessary re-download of FFHQ data
        # 3. The extractor is only needed for prediction from raw images
        self.fingerprint_extractor = None
    
    def build_model(self, input_dim: int, output_dim: int, 
                   hidden_dims: List[int] = [512, 256, 128]) -> nn.Module:
        """
        Build MLP architecture for fingerprint-to-label classification.
        
        Args:
            input_dim: Fingerprint feature dimension
            output_dim: Number of model classes
            hidden_dims: Hidden layer dimensions
            
        Returns:
            MLP classifier model
        """
        class TrueAttributionMLP(nn.Module):
            def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int]):
                super().__init__()
                
                self.input_dim = input_dim
                self.num_classes = num_classes
                
                # Build MLP layers
                layers = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.3),
                        nn.BatchNorm1d(hidden_dim)
                    ])
                    prev_dim = hidden_dim
                
                # Output layer
                layers.append(nn.Linear(prev_dim, num_classes))
                
                self.layers = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.layers(x)
        
        return TrueAttributionMLP(input_dim, output_dim, hidden_dims)
    
    def get_model_save_path(self, **kwargs) -> Path:
        """Get path to save the true attribution model."""
        # Note: This method is overridden by the entry scripts (train_models.py, run_attacks.py)
        # with lambda functions that return the correct paths for the new directory structure.
        # This implementation is kept for backward compatibility but is not used by the main workflow.
        raise NotImplementedError("Use the new entry scripts (train_models.py, run_attacks.py) for model path management")
    
    def prepare_training_data(self, model_names: List[str], 
                            num_images_per_model: int = 1000) -> tuple:
        """
        Prepare training data by extracting fingerprints and creating labels.
        
        Args:
            model_names: List of model names to include
            num_images_per_model: Number of images to load per model
            
        Returns:
            Tuple of (fingerprints, labels, label_map)
        """
        # Note: This method is overridden by the entry scripts (train_models.py, run_attacks.py)
        # with lambda functions that handle data preparation directly.
        # This implementation is kept for backward compatibility but is not used by the main workflow.
        raise NotImplementedError("Use the new entry scripts (train_models.py, run_attacks.py) for data preparation")
    
    def train_from_models(self, model_names: List[str], 
                         num_images_per_model: int = 1000,
                         val_split: float = 0.2, **model_kwargs) -> nn.Module:
        """
        Train true attribution model from multiple generative models.
        
        Args:
            model_names: List of model names to include in training
            num_images_per_model: Number of images per model
            val_split: Validation split ratio
            **model_kwargs: Additional model parameters
            
        Returns:
            Trained true attribution model
        """
        # Prepare data
        X, y, label_map = self.prepare_training_data(model_names, num_images_per_model)
        
        # Store label mapping for later use
        self.label_map = label_map
        self.reverse_label_map = {v: k for k, v in label_map.items()}
        
        # Train model
        model = self.train(X, y, val_split, **model_kwargs)
        
        # Save model with label mapping
        self.save_model(label_map=label_map)
        
        return model
    
    def predict_from_images(self, images: torch.Tensor) -> tuple:
        """
        Predict model labels from images (extract fingerprints first).
        
        Args:
            images: Input images tensor
            
        Returns:
            Tuple of (predicted_labels, confidence_scores)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Initialize fingerprint extractor only when needed for prediction
        if self.fingerprint_extractor is None:
            from ..fingerprints import FingerprintExtractor
            self.fingerprint_extractor = FingerprintExtractor.create(self.fingerprint_method, device=self.device)
        
        # Extract fingerprints
        fingerprints = self.fingerprint_extractor.extract_fingerprint(images)
        
        # Predict from fingerprints
        return self.predict_from_fingerprints(fingerprints)
    
    def predict_from_fingerprints(self, fingerprints: torch.Tensor) -> tuple:
        """
        Predict model labels from fingerprint features.
        
        Args:
            fingerprints: Fingerprint features tensor
            
        Returns:
            Tuple of (predicted_labels, confidence_scores)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Flatten fingerprints if they are multi-dimensional (e.g., Nataraj19 produces 3x256x256)
        if fingerprints.dim() > 2:
            original_shape = fingerprints.shape
            fingerprints = fingerprints.view(fingerprints.shape[0], -1)
            self.logger.info(f"Flattened fingerprints from {original_shape} to {fingerprints.shape}")
        
        self.model.eval()
        with torch.no_grad():
            fingerprints = fingerprints.to(self.device)
            outputs = self.model(fingerprints)
            
            # Get predictions and confidence scores
            confidence_scores, predicted_indices = torch.max(torch.softmax(outputs, dim=1), 1)
            
            # Convert indices to model names if label map is available
            if hasattr(self, 'reverse_label_map'):
                predicted_labels = [self.reverse_label_map[idx.item()] for idx in predicted_indices]
            else:
                predicted_labels = predicted_indices.cpu().numpy().tolist()
        
        return predicted_labels, confidence_scores.cpu().numpy()
    
    def get_prediction_probabilities(self, fingerprints: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities for all classes.
        
        Args:
            fingerprints: Fingerprint features tensor
            
        Returns:
            Probability tensor of shape (N, num_classes)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Flatten fingerprints if they are multi-dimensional (e.g., Nataraj19 produces 3x256x256)
        if fingerprints.dim() > 2:
            original_shape = fingerprints.shape
            fingerprints = fingerprints.view(fingerprints.shape[0], -1)
            self.logger.info(f"Flattened fingerprints from {original_shape} to {fingerprints.shape}")
        
        self.model.eval()
        with torch.no_grad():
            fingerprints = fingerprints.to(self.device)
            outputs = self.model(fingerprints)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu()
    
    def evaluate_on_test_set(self, test_model_names: List[str], 
                            num_test_images: int = 200) -> Dict[str, float]:
        """
        Evaluate the model on a test set.
        
        Args:
            test_model_names: List of model names for testing
            num_test_images: Number of test images per model
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Prepare test data
        X_test, y_test, _ = self.prepare_training_data(test_model_names, num_test_images)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            X_test = X_test.to(self.device)
            outputs = self.model(X_test)
            _, predicted = torch.max(outputs, 1)
        
        # Calculate metrics
        from ..utils import calculate_comprehensive_metrics
        metrics = calculate_comprehensive_metrics(y_test.numpy(), predicted.cpu().numpy())
        
        self.logger.info(f"Test evaluation results: {metrics}")
        return metrics
    
    def save_model(self, **kwargs):
        """Save model with additional metadata."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_path = self.get_model_save_path()
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get model configuration for reliable loading
        model_config = {}
        if hasattr(self.model, 'input_dim'):
            model_config['input_dim'] = self.model.input_dim
        if hasattr(self.model, 'num_classes'):
            model_config['output_dim'] = self.model.num_classes
        if hasattr(self.model, 'layers'):
            # Extract hidden dimensions from the model
            hidden_dims = []
            for i, layer in enumerate(self.model.layers):
                if isinstance(layer, nn.Linear) and i < len(self.model.layers) - 1:  # Skip output layer
                    hidden_dims.append(layer.out_features)
            model_config['hidden_dims'] = hidden_dims
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'model_config': model_config,
            'trainer_name': self.trainer_name,
            'fingerprint_method': self.fingerprint_method,
            'dataset': self.dataset,
            'image_size': self.image_size,
            'device': self.device,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'label_map': kwargs.get('label_map', getattr(self, 'label_map', {})),
            'extractor_info': self.fingerprint_extractor.get_method_info() if self.fingerprint_extractor is not None else None,
            'input_dim': self.model.input_dim if hasattr(self.model, 'input_dim') else None,
            'output_dim': self.model.num_classes if hasattr(self.model, 'num_classes') else None,
            'model_class': self.model.__class__.__name__,
            'hidden_dims': self.model.layers[1].out_features if hasattr(self.model, 'layers') and len(self.model.layers) > 1 else None
        }
        
        torch.save(model_data, model_path)
        self.logger.info(f"True attribution model saved to {model_path}")
    
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
        self.label_map = model_data.get('label_map', {})
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        # Load training history
        self.train_losses = model_data.get('train_losses', [])
        self.val_losses = model_data.get('val_losses', [])
        self.train_metrics = model_data.get('train_metrics', [])
        self.val_metrics = model_data.get('val_metrics', [])
        
        self.logger.info(f"True attribution model loaded from {model_path}")
        return self.model