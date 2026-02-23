"""
W3 Attack: Surrogate Extractor Attack.

This attack uses a trained surrogate fingerprint extractor φ_s 
to approximate the behavior of non-differentiable extractors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional

from .base import Attacker


@Attacker.register("w3")
class Attacker_W3(Attacker):
    """
    W3 Attack using surrogate fingerprint extractor.
    
    This attack trains a neural network to approximate the output
    of non-differentiable fingerprint extractors, then uses gradients
    through this surrogate for adversarial optimization.
    """
    
    def __init__(self, surrogate_extractor: nn.Module, attribution_model: nn.Module,
                 attack_type: str = "removal", epsilon: float = 0.1,
                 num_steps: int = 100, step_size: float = 0.01,
                 targeted: bool = False, device: str = "cuda"):
        """
        Initialize W3 attacker.
        
        Args:
            surrogate_extractor: Trained surrogate extractor φ_s
            attribution_model: Attribution model h
            attack_type: Type of attack ("removal" or "forgery")
            epsilon: Maximum perturbation magnitude
            num_steps: Number of attack iterations
            step_size: Step size for iterative attacks
            targeted: Whether to perform targeted attack
            device: Device to run attacks on
        """
        super().__init__(
            attack_name="w3",
            attack_type=attack_type,
            epsilon=epsilon,
            num_steps=num_steps,
            step_size=step_size,
            targeted=targeted,
            device=device
        )
        
        self.surrogate_extractor = surrogate_extractor.to(device)
        self.attribution_model = attribution_model.to(device)
        
        # Set models to evaluation mode
        self.surrogate_extractor.eval()
        self.attribution_model.eval()
    
    def attack(self, images: torch.Tensor, targets: Optional[torch.Tensor] = None,
               attack_batch_size: int = 10, **kwargs) -> torch.Tensor:
        """
        Perform W3 surrogate extractor attack.
        
        Args:
            images: Input images tensor
            targets: For removal: original labels. For forgery: per-sample target labels.
            attack_batch_size: Batch size for PGD attack to save memory (smaller = less memory)
            **kwargs: Additional parameters
            
        Returns:
            Adversarially perturbed images
            
        Note:
            This method processes images in batches to save memory. The actual batch size
            used might be smaller than attack_batch_size if there's not enough memory.
            Memory is freed after processing each batch.
        """
        images = images.to(self.device)
        
        # Simplified forgery logic handled via per-sample targets passed in loss
        
        # Ensure models are in training mode for gradient computation
        self.surrogate_extractor.train()
        self.attribution_model.train()
        
        def loss_fn(adv_images, targets=None):
            try:
                # Ensure input requires gradients
                if not adv_images.requires_grad:
                    adv_images.requires_grad_(True)
                
                # Extract fingerprints using surrogate
                fingerprints = self.surrogate_extractor(adv_images)
                
                # Get attribution predictions
                logits = self.attribution_model(fingerprints)
                del fingerprints  # Free memory early
                
                # Compute loss based on attack type (use CE on logits for stronger gradients)
                if self.attack_type == "removal":
                    if targets is None:
                        raise ValueError("Removal requires original labels provided as 'targets'.")
                    loss = F.cross_entropy(logits, targets)
                elif self.attack_type == "forgery":
                    if targets is None:
                        raise ValueError("Forgery requires per-sample target labels provided as 'targets'.")
                    loss = -F.cross_entropy(logits, targets)
                else:
                    if targets is None:
                        raise ValueError("Targets must be provided for attack loss computation.")
                    loss = F.cross_entropy(logits, targets)
                
                return loss
                
            except Exception as e:
                self.logger.error(f"Error in loss computation: {e}")
                # Clean up on error
                if 'fingerprints' in locals(): del fingerprints
                if 'logits' in locals(): del logits
                if 'probabilities' in locals(): del probabilities
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                torch.mps.empty_cache() if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else None
                raise
        
        # Define progress callback
        def progress_callback(iteration: int, current_images: torch.Tensor, batch_progress: dict = None):
            """Callback to log progress during PGD attack"""
            if iteration % max(1, self.num_steps // 10) == 0:  # Log every 10% progress
                if batch_progress:
                    batch_idx = batch_progress['batch_idx']
                    num_batches = batch_progress['num_batches']
                    step = batch_progress['step']
                    num_steps = batch_progress['num_steps']
                    overall_progress = batch_progress['overall_progress']
                    
                    self.logger.info(
                        f"PGD progress: Batch {batch_idx + 1}/{num_batches}, "
                        f"Step {step + 1}/{num_steps} "
                        f"(Overall: {overall_progress*100:.1f}%)"
                    )
                else:
                    # Fallback for old progress format
                    total_steps = (len(images) + attack_batch_size - 1) // attack_batch_size * self.num_steps
                    self.logger.info(f"PGD progress: iteration {iteration}/{total_steps} ({(iteration/total_steps)*100:.1f}%)")
                
                # Log memory usage if in debug mode
                if self.logger.isEnabledFor(logging.DEBUG):
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1024**2
                        reserved = torch.cuda.memory_reserved() / 1024**2
                        self.logger.debug(
                            f"GPU Memory - Allocated: {allocated:.1f}MB, "
                            f"Reserved: {reserved:.1f}MB"
                        )
        
        try:
            # Perform PGD attack with batching
            self.logger.info(f"Starting PGD attack iterations (total steps: {self.num_steps}, batch size: {attack_batch_size})...")
            adversarial_images = self.pgd_attack(
                images, loss_fn, targets, random_start=True,
                progress_callback=progress_callback,
                batch_size=attack_batch_size
            )
            return adversarial_images
            
        except Exception as e:
            self.logger.error(f"Error during PGD attack: {e}")
            # Clean up
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            torch.mps.empty_cache() if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else None
            raise
            
        finally:
            # Always restore models to eval mode and clean up
            self.surrogate_extractor.eval()
            self.attribution_model.eval()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            torch.mps.empty_cache() if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else None