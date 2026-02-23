"""
B1 Attack: Surrogate Classifier Attack.

This black-box attack uses a surrogate classifier h_s trained on raw images
to model source labels, relying on transferability for attacks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base import Attacker


@Attacker.register("b1")
class Attacker_B1(Attacker):
    """
    B1 Attack using surrogate classifier.
    
    This black-box attack trains a surrogate classifier directly on images
    and relies on adversarial transferability to fool the target system.
    """
    
    def __init__(self, surrogate_classifier: nn.Module,
                 attack_type: str = "removal", epsilon: float = 0.1,
                 num_steps: int = 100, step_size: float = 0.01,
                 targeted: bool = False, device: str = "cuda"):
        """
        Initialize B1 attacker.
        
        Args:
            surrogate_classifier: Trained surrogate classifier h_s
            attack_type: Type of attack ("removal" or "forgery")
            epsilon: Maximum perturbation magnitude
            num_steps: Number of attack iterations
            step_size: Step size for iterative attacks
            targeted: Whether to perform targeted attack
            device: Device to run attacks on
        """
        super().__init__(
            attack_name="b1",
            attack_type=attack_type,
            epsilon=epsilon,
            num_steps=num_steps,
            step_size=step_size,
            targeted=targeted,
            device=device
        )
        
        self.surrogate_classifier = surrogate_classifier.to(device)
        self.surrogate_classifier.eval()
    
    def attack(self, images: torch.Tensor, targets: Optional[torch.Tensor] = None,
               **kwargs) -> torch.Tensor:
        """
        Perform B1 surrogate classifier attack.
        
        Args:
            images: Input images tensor
            targets: For removal: original labels. For forgery: per-sample target labels.
            **kwargs: Additional parameters
            
        Returns:
            Adversarially perturbed images
        """
        images = images.to(self.device)
        
        # For simplified forgery logic, per-sample targets must be provided at runtime
        
        def loss_fn(adv_images, targets=None):
            # Get predictions directly from images
            logits = self.surrogate_classifier(adv_images)
            
            if self.attack_type == "removal":
                if targets is None:
                    raise ValueError("Removal requires original labels provided as 'targets'.")
                return F.cross_entropy(logits, targets)
            elif self.attack_type == "forgery":
                if targets is None:
                    raise ValueError("Forgery requires per-sample target labels provided as 'targets'.")
                return -F.cross_entropy(logits, targets)
            else:
                if targets is None:
                    raise ValueError("Targets must be provided for attack loss computation.")
                return F.cross_entropy(logits, targets)
        
        # Perform PGD attack
        adversarial_images = self.pgd_attack(images, loss_fn, targets)
        
        return adversarial_images