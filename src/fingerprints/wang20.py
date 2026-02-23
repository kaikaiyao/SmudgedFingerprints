"""
Wang20 fingerprint extraction method implementation.

Based on "CNN-Generated Images Are Surprisingly Easy to Spot... for Now" by Wang et al. (2020).

This method uses a ResNet50 backbone for CNN-generated image detection,
making it more computationally efficient while maintaining effectiveness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional

from .base import FingerprintExtractor, AnalyticApproximation


@FingerprintExtractor.register("wang20")
class Wang20(FingerprintExtractor, nn.Module):
    """
    Wang20 attribution model using ResNet50 backbone.
    
    This is an implicit fingerprint method where the ResNet50-based network itself 
    is the attribution model. The network uses a pre-trained ResNet50 backbone
    with a custom classification head for model attribution.
    
    This method provides a simpler alternative to complex architectures while
    maintaining effectiveness for CNN-generated image detection.
    """
    
    def __init__(self, img_size: int = 256, num_classes: int = 2, 
                 pretrained: bool = True, hidden_dims: list = [512, 256], device: str = None):
        """
        Initialize Wang20 ResNet50-based attribution model.
        
        Args:
            img_size: Input image size (assumed square)
            num_classes: Number of model classes for attribution
            pretrained: Whether to use pretrained ResNet50 weights
            hidden_dims: Hidden layer dimensions for classification head
            device: Device to run on ("cuda", "cpu", "mps", or None for auto)
        """
        nn.Module.__init__(self)
        
        self.img_size = img_size
        self.num_classes = num_classes
        
        # Setup device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        
        # Calculate feature dimension (ResNet50 output is 2048)
        feature_dim = 2048
        
        FingerprintExtractor.__init__(
            self,
            method_name="wang20",
            is_differentiable=True,  # ResNet50 is fully differentiable
            has_analytic_approx=False,  # No approximation needed as it's already differentiable
            feature_dim=feature_dim,
            is_implicit_fingerprint=True  # This is an implicit fingerprint method
        )
        
        # Load pretrained ResNet50 backbone
        if pretrained:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Remove the original classification head
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Build custom classification head for attribution
        self.classification_head = self._build_classification_head(feature_dim, num_classes, hidden_dims)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Move the model to the specified device
        self.to(self.device)
    
    def _build_classification_head(self, input_dim: int, num_classes: int, hidden_dims: list) -> nn.Module:
        """
        Build classification head for attribution.
        
        Args:
            input_dim: Input feature dimension (2048 for ResNet50)
            num_classes: Number of model classes
            hidden_dims: Hidden layer dimensions
            
        Returns:
            Classification head module
        """
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                # Use GroupNorm instead of BatchNorm to avoid issues with small batch sizes
                nn.GroupNorm(min(32, hidden_dim // 4), hidden_dim) if hidden_dim >= 4 else nn.Identity()
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        return nn.Sequential(*layers)
    
    def extract_fingerprint(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract ResNet50 features from images.
        
        Args:
            images: Input images tensor of shape (N, C, H, W)
            
        Returns:
            ResNet50 features tensor of shape (N, feature_dim)
        """
        batch_size = images.shape[0]
        
        # Ensure images are in correct range [0, 1] and size
        if images.min() < 0 or images.max() > 1:
            # Normalize to [0, 1] range
            images = (images - images.min()) / (images.max() - images.min())
        
        # Move images to the correct device
        images = images.to(self.device)
        
        # Resize images if needed (ResNet50 expects 224x224)
        if images.shape[2] != 224 or images.shape[3] != 224:
            images = F.interpolate(images, size=(224, 224), 
                                 mode='bilinear', align_corners=False)
        
        # Extract features using ResNet50 backbone
        features = self.backbone(images)  # [N, 2048, 1, 1]
        
        # Flatten features
        features = features.view(batch_size, -1)  # [N, 2048]
        
        return features
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for attribution prediction.
        
        Args:
            images: Input images tensor of shape (N, C, H, W)
            
        Returns:
            Attribution logits tensor of shape (N, num_classes)
        """
        # Extract ResNet50 features
        features = self.extract_fingerprint(images)
        
        # Apply dropout
        features = self.dropout(features)
        
        # Pass through classification head
        logits = self.classification_head(features)
        
        return logits
    
    def predict_attribution(self, images: torch.Tensor) -> torch.Tensor:
        """
        Predict model attribution directly.
        
        Args:
            images: Input images tensor of shape (N, C, H, W)
            
        Returns:
            Attribution logits tensor of shape (N, num_classes)
        """
        # Use eval mode to disable dropout
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
            logits = self.forward(images)
        
        # Restore training state
        if was_training:
            self.train()
        
        return logits
    
    def get_backbone_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get intermediate features from ResNet50 backbone for analysis.
        
        Args:
            images: Input images tensor of shape (N, C, H, W)
            
        Returns:
            Backbone features tensor of shape (N, 2048)
        """
        return self.extract_fingerprint(images)


class Wang20_Approx(AnalyticApproximation):
    """
    Analytic approximation for Wang20 fingerprint extraction.
    
    Since Wang20 is already differentiable and is an implicit fingerprint method,
    this approximation is not needed for W1 attacks (which can be done directly).
    This class is kept for compatibility with the framework.
    """
    
    def __init__(self, original_extractor: Wang20):
        """
        Initialize Wang20 approximation.
        
        Args:
            original_extractor: Original Wang20 extractor
        """
        super().__init__(original_extractor)
        # For implicit fingerprint methods, we can access the model directly
        self.attribution_model = original_extractor.get_attribution_model()
    
    def extract_fingerprint_approx(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract approximate ResNet50 features.
        
        For implicit fingerprint methods, this returns the attribution logits.
        
        Args:
            images: Input images tensor of shape (N, C, H, W)
            
        Returns:
            Attribution logits tensor of shape (N, num_classes)
        """
        # For implicit fingerprint methods, return attribution predictions
        return self.original_extractor.predict_attribution(images)
