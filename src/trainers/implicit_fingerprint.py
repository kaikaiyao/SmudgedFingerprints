"""
Implicit Fingerprint Model Trainer.

Trains implicit fingerprint methods where the fingerprinting network itself
is the attribution model (e.g., F3-Net for Qian20).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .base import BaseTrainer
from ..fingerprints import FingerprintExtractor


class ImplicitFingerprintTrainer(BaseTrainer):
    """
    Trainer for implicit fingerprint methods.
    
    This trainer trains the fingerprinting network itself as the attribution model.
    Examples include F3-Net (Qian20) where the network learns to classify
    images directly to model sources.
    """
    
    def __init__(self, fingerprint_method: str, dataset: str, image_size: int,
                 device: str = "cuda", batch_size: int = 32, learning_rate: float = 1e-4,
                 num_epochs: int = 50, patience: int = 10):
        """
        Initialize the implicit fingerprint trainer.
        
        Args:
            fingerprint_method: Name of the fingerprint method to use
            dataset: Dataset name
            image_size: Size of input images
            device: Device to train on
            batch_size: Training batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            patience: Patience for early stopping
        """
        super().__init__(
            model_name="implicit_fingerprint",
            device=device,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            patience=patience
        )
        
        self.fingerprint_method = fingerprint_method
        self.dataset = dataset
        self.image_size = image_size
        
        # We'll initialize the model after we know the number of classes
        self.fingerprint_method = fingerprint_method
        self.attribution_model = None
        self.fingerprint_extractor = None
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # We'll initialize optimizer and scheduler after model creation
        self.optimizer = None
        self.scheduler = None
    
    def build_model(self, input_dim: int, output_dim: int, **kwargs) -> nn.Module:
        """
        Build the model architecture.
        
        For implicit fingerprint methods, the model is already built when the
        fingerprint extractor is created. This method returns the existing model.
        
        Args:
            input_dim: Input dimension (not used for implicit methods)
            output_dim: Output dimension (number of classes)
            **kwargs: Additional model parameters
            
        Returns:
            PyTorch model (the implicit fingerprint model)
        """
        # For implicit methods, the model is already built
        # We just need to ensure it has the right number of output classes
        if hasattr(self.attribution_model, 'classification_head'):
            # Update the classification head if needed
            if hasattr(self.attribution_model.classification_head, 'out_features'):
                if self.attribution_model.classification_head.out_features != output_dim:
                    # Rebuild the classification head with the correct number of classes
                    import torch.nn as nn
                    input_features = self.attribution_model.classification_head.in_features
                    self.attribution_model.classification_head = nn.Linear(input_features, output_dim)
        
        return self.attribution_model
    
    def _initialize_model(self, num_classes: int):
        """
        Initialize the fingerprint extractor and attribution model with the correct number of classes.
        
        Args:
            num_classes: Number of classes for the attribution model
        """
        print(f"🔄 Initializing {self.fingerprint_method} model...")
        print(f"   - Image size: {self.image_size}")
        print(f"   - Number of classes: {num_classes}")
        
        # Initialize the fingerprint extractor with the correct number of classes and image size
        print("   - Creating fingerprint extractor...")
        self.fingerprint_extractor = FingerprintExtractor.create(
            self.fingerprint_method, 
            num_classes=num_classes,
            img_size=self.image_size,
            device=self.device
        )
        print("   ✅ Fingerprint extractor created")
        
        if not self.fingerprint_extractor.is_implicit_fingerprint:
            raise ValueError(f"Method {self.fingerprint_method} is not an implicit fingerprint method")
        
        # Get the attribution model
        print("   - Getting attribution model...")
        self.attribution_model = self.fingerprint_extractor.get_attribution_model()
        if self.attribution_model is None:
            raise ValueError(f"Method {self.fingerprint_method} does not provide an attribution model")
        print("   ✅ Attribution model retrieved")
        
        # Move model to device
        print(f"   - Moving model to device: {self.device}")
        self.attribution_model.to(self.device)
        print("   ✅ Model moved to device")
        
        # Set the model attribute that the base class expects
        self.model = self.attribution_model
        
        # Initialize optimizer
        print("   - Initializing optimizer...")
        self.optimizer = optim.Adam(self.attribution_model.parameters(), lr=self.learning_rate)
        print("   ✅ Optimizer initialized")
        
        # Initialize learning rate scheduler
        print("   - Initializing learning rate scheduler...")
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        print("   ✅ Learning rate scheduler initialized")
        
        # Test forward pass with dummy data
        print("   - Testing forward pass with dummy data...")
        try:
            with torch.no_grad():
                # Use a larger batch size to avoid BatchNorm issues
                dummy_input = torch.randn(2, 3, self.image_size, self.image_size).to(self.device)
                dummy_output = self.attribution_model(dummy_input)
                print(f"   ✅ Forward pass successful - Output shape: {dummy_output.shape}")
        except Exception as e:
            print(f"   ❌ Forward pass failed: {e}")
            raise
        
        print(f"✅ Initialized {self.fingerprint_method} model with {num_classes} classes")
    
    def get_model_save_path(self, **kwargs) -> Path:
        """Get path for saving the trained model."""
        output_dir = Path("data") / f"models_{self.fingerprint_method}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / "true_attribution_model.pth"
    
    def _prepare_data(self, data_dir: Path) -> tuple:
        """
        Prepare training data by loading images from the data directory.
        
        Args:
            data_dir: Directory containing images organized by model
            
        Returns:
            Tuple of (train_loader, val_loader, num_classes, model_names)
        """
        # Load images and labels
        images_dir = data_dir / "images"
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        # Use the shared model discovery utility to ensure label ordering matches evaluation
        try:
            from src.data_loader.model_discovery import ModelDiscovery
            model_names = ModelDiscovery.discover_available_models(data_dir)
            print(f"Models sorted by load order: {model_names}")
        except Exception:
            # Fallback: collect model names from directory if discovery utility isn't available
            model_names = []
            for model_dir in images_dir.iterdir():
                if model_dir.is_dir():
                    model_names.append(model_dir.name)
            model_names.sort()  # Sort for consistency
            print(f"Models collected from directory and sorted: {model_names}")
        
        # Create label mapping
        label_map = {model_name: idx for idx, model_name in enumerate(model_names)}
        
        # Load all data
        all_images = []
        all_labels = []
        
        # Now load images with correct labels
        for model_name in model_names:
            model_dir = images_dir / model_name
            if not model_dir.exists():
                print(f"Warning: Model directory {model_name} not found, skipping")
                continue
                
            model_label = label_map[model_name]
            
            # Load images for this model
            image_files = list(model_dir.glob("*.png"))
            for image_file in image_files:
                from PIL import Image
                from torchvision import transforms
                
                # Load and preprocess image
                img_pil = Image.open(image_file)
                img_tensor = transforms.ToTensor()(img_pil)
                all_images.append(img_tensor)
                all_labels.append(model_label)
        
        if not all_images:
            raise ValueError(f"No images found in {images_dir}")
        
        # Stack all images
        all_images = torch.stack(all_images, dim=0)
        all_labels = torch.tensor(all_labels, dtype=torch.long)
        
        print(f"Loaded {len(all_images)} samples from {len(model_names)} models")
        print(f"Image shape: {all_images.shape}")
        print(f"Model names: {model_names}")
        print(f"Label range: {all_labels.min().item()} to {all_labels.max().item()}")
        print(f"Label distribution: {torch.bincount(all_labels).tolist()}")
        
        # Split into train/val
        num_samples = len(all_images)
        indices = torch.randperm(num_samples)
        
        train_size = int(0.8 * num_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(
            all_images[train_indices], all_labels[train_indices]
        )
        val_dataset = torch.utils.data.TensorDataset(
            all_images[val_indices], all_labels[val_indices]
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        return train_loader, val_loader, len(model_names), model_names
    
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.attribution_model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.attribution_model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def validate(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.attribution_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.attribution_model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(val_loader)
        
        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, data_dir: Path, **kwargs) -> Dict[str, Any]:
        """
        Train the implicit fingerprint model.
        
        Args:
            data_dir: Directory containing the data
            **kwargs: Additional arguments
            
        Returns:
            Training results
        """
        print(f"Training implicit fingerprint model: {self.fingerprint_method}")
        
        # Load data
        train_loader, val_loader, num_classes, model_names = self._prepare_data(data_dir)
        
        # Initialize the model with the correct number of classes
        if self.attribution_model is None:
            self._initialize_model(num_classes)
        else:
            # Update model if needed (in case num_classes changed)
            if hasattr(self.attribution_model, 'num_classes') and self.attribution_model.num_classes != num_classes:
                print(f"Updating model from {self.attribution_model.num_classes} to {num_classes} classes")
                # Update the model's num_classes attribute
                self.attribution_model.num_classes = num_classes
                
                # Rebuild the classification head with the correct number of classes
                if hasattr(self.attribution_model, 'classification_head'):
                    # Get the input dimension from the current classification head
                    input_dim = self.attribution_model.classification_head[0].in_features
                    
                    # Rebuild the classification head
                    self.attribution_model.classification_head = self.attribution_model._build_classification_head(
                        input_dim, num_classes, hidden_dims=[512, 256]  # Default hidden dims
                    )
                    
                    # Move the new classification head to the correct device
                    self.attribution_model.classification_head.to(self.device)
                    
                    # Reinitialize optimizer with the updated model
                    self.optimizer = optim.Adam(self.attribution_model.parameters(), lr=self.learning_rate)
                    
                    print(f"Successfully updated classification head to {num_classes} classes")
            else:
                print(f"Model already has correct number of classes ({num_classes})")
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        train_history = []
        val_history = []
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            train_history.append(train_metrics)
            
            # Validate
            val_metrics = self.validate(val_loader)
            val_history.append(val_metrics)
            
            # Print metrics
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['accuracy'])
            
            # Early stopping
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                patience_counter = 0
                
                # Save best model
                self.save_model()
                print(f"New best validation accuracy: {best_val_acc:.2f}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
        
        # Load best model
        self.load_model()
        
        # Final evaluation
        final_val_metrics = self.validate(val_loader)
        
        # Save training history
        self.save_training_history(train_history, val_history, model_names)
        
        results = {
            'best_val_accuracy': best_val_acc,
            'final_val_metrics': final_val_metrics,
            'num_epochs_trained': epoch + 1,
            'model_names': model_names
        }
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {best_val_acc:.2f}")
        print(f"Final validation metrics: {final_val_metrics}")
        
        return results
    
    def save_model(self):
        """Save the trained model."""
        save_path = self.get_model_save_path()
        
        # Get number of classes from the classification head if available
        num_classes = None
        if hasattr(self.attribution_model, 'num_classes'):
            num_classes = self.attribution_model.num_classes
        elif hasattr(self.attribution_model, 'classification_head'):
            if hasattr(self.attribution_model.classification_head, 'out_features'):
                num_classes = self.attribution_model.classification_head.out_features
        
        torch.save({
            'model_state_dict': self.attribution_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'fingerprint_method': self.fingerprint_method,
            'num_classes': num_classes,
            'model_config': {
                'img_size': getattr(self.attribution_model, 'img_size', None),
                'LFS_window_size': getattr(self.attribution_model, 'LFS_window_size', None),
                'LFS_M': getattr(self.attribution_model, 'LFS_M', None),
            }
        }, save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self):
        """Load the trained model."""
        save_path = self.get_model_save_path()
        if save_path.exists():
            checkpoint = torch.load(save_path, map_location=self.device)
            self.attribution_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {save_path}")
        else:
            print(f"No saved model found at {save_path}")
    
    def save_training_history(self, train_history: List[Dict], val_history: List[Dict], 
                            model_names: List[str]):
        """Save training history and plots."""
        # Import matplotlib only when needed to avoid import conflicts
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not available, skipping plot generation")
            return
            
        output_dir = Path("data") / f"models_{self.fingerprint_method}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save history as numpy arrays
        history_path = output_dir / "training_history.npz"
        np.savez(history_path,
                 train_loss=[h['loss'] for h in train_history],
                 train_acc=[h['accuracy'] for h in train_history],
                 val_loss=[h['loss'] for h in val_history],
                 val_acc=[h['accuracy'] for h in val_history],
                 val_precision=[h.get('precision', 0) for h in val_history],
                 val_recall=[h.get('recall', 0) for h in val_history],
                 val_f1=[h.get('f1', 0) for h in val_history],
                 model_names=model_names)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs = range(1, len(train_history) + 1)
        
        # Loss plot
        axes[0, 0].plot(epochs, [h['loss'] for h in train_history], 'b-', label='Train')
        axes[0, 0].plot(epochs, [h['loss'] for h in val_history], 'r-', label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(epochs, [h['accuracy'] for h in train_history], 'b-', label='Train')
        axes[0, 1].plot(epochs, [h['accuracy'] for h in val_history], 'r-', label='Validation')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision plot
        axes[1, 0].plot(epochs, [h.get('precision', 0) for h in val_history], 'g-', label='Validation')
        axes[1, 0].set_title('Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # F1 plot
        axes[1, 1].plot(epochs, [h.get('f1', 0) for h in val_history], 'g-', label='Validation')
        axes[1, 1].set_title('F1 Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / "training_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training history saved to {history_path}")
        print(f"Training plots saved to {output_dir / 'training_plots.png'}")
