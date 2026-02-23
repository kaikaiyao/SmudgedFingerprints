"""
W1 Attack: Direct Gradient Attack on Implicit Fingerprint Methods.

This attack has full gradient access to the implicit fingerprint model
(where the fingerprinting network itself is the attribution model),
allowing for direct optimization via PGD.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base import Attacker


@Attacker.register("w1")
class Attacker_W1(Attacker):
    """
    W1 Attack with direct gradient access to implicit fingerprint models.
    
    This is the strongest white-box attack for implicit fingerprint methods
    where the attacker has full access to the differentiable fingerprinting
    network that serves as the attribution model.
    """
    
    def __init__(self, implicit_fingerprint_model: nn.Module,
                 attack_type: str = "removal", epsilon: float = 0.1,
                 num_steps: int = 100, step_size: float = 0.01,
                 targeted: bool = False, device: str = "cuda"):
        """
        Initialize W1 attacker for implicit fingerprint methods.
        
        Args:
            implicit_fingerprint_model: The complete fingerprinting network that serves as attribution model
            attack_type: Type of attack ("removal" or "forgery")
            epsilon: Maximum perturbation magnitude
            num_steps: Number of attack iterations
            step_size: Step size for iterative attacks
            targeted: Whether to perform targeted attack
            device: Device to run attacks on
        """
        super().__init__(
            attack_name="w1",
            attack_type=attack_type,
            epsilon=epsilon,
            num_steps=num_steps,
            step_size=step_size,
            targeted=targeted,
            device=device
        )
        
        self.implicit_fingerprint_model = implicit_fingerprint_model.to(device)
        
        # Set model to evaluation mode
        self.implicit_fingerprint_model.eval()
    
    def attack(self, images: torch.Tensor, targets: Optional[torch.Tensor] = None,
               **kwargs) -> torch.Tensor:
        """
        Perform W1 direct gradient attack on implicit fingerprint model.
        
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
            # Direct forward pass through the implicit fingerprint model
            logits = self.implicit_fingerprint_model(adv_images)
            
            if self.attack_type == "removal":
                # Maximize CE(logits, true) to reduce confidence in the original class
                if targets is None:
                    raise ValueError("Removal requires original labels provided as 'targets'.")
                return F.cross_entropy(logits, targets)
            elif self.attack_type == "forgery":
                # Maximize probability of per-sample target via -CE(logits, target)
                if targets is None:
                    raise ValueError("Forgery requires per-sample target labels provided as 'targets'.")
                return -F.cross_entropy(logits, targets)
            else:
                # Default safety (should not hit)
                if targets is None:
                    raise ValueError("Targets must be provided for attack loss computation.")
                return F.cross_entropy(logits, targets)
        
        # Perform PGD attack
        adversarial_images = self.pgd_attack(images, loss_fn, targets)
        
        return adversarial_images