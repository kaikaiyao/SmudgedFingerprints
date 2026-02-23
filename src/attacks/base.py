"""Base class for adversarial attacks on fingerprinting methods."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple
import logging

from ..utils import setup_logger, clip_images


class Attacker(ABC):
    """
    Base class for adversarial attacks on fingerprinting methods.
    
    This class provides a common interface for different attack strategies
    targeting fingerprint-based model attribution systems.
    """
    
    # Registry of available attack methods
    _registry = {}
    
    def __init__(self, attack_name: str, attack_type: str = "removal",
                 epsilon: float = 0.1, num_steps: int = 100, step_size: float = 0.01,
                 targeted: bool = False, device: str = "cuda", momentum: float = 0.9):
        """
        Initialize the attacker.
        
        Args:
            attack_name: Name of the attack strategy (w1, w2, w3, b1, b2)
            attack_type: Type of attack ("removal" or "forgery")
            epsilon: Maximum perturbation magnitude (L∞ norm)
            num_steps: Number of attack iterations
            step_size: Step size for iterative attacks
            targeted: Whether to perform targeted attack
            device: Device to run attacks on
            momentum: Momentum coefficient for gradient updates (default: 0.9)
        """
        self.attack_name = attack_name
        self.attack_type = attack_type
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.targeted = targeted
        self.device = device
        self.momentum = momentum
        
        self.logger = setup_logger(f"Attacker_{attack_name}")
        
        # Attack statistics
        self.attack_stats = {
            'total_attacks': 0,
            'successful_attacks': 0,
            'average_perturbation': 0.0,
            'average_iterations': 0.0
        }
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register attack methods."""
        def decorator(attacker_class):
            cls._registry[name] = attacker_class
            return attacker_class
        return decorator
    
    @classmethod
    def create(cls, attack_name: str, **kwargs) -> 'Attacker':
        """
        Factory method to create appropriate attacker.
        
        Args:
            attack_name: Name of the attack strategy
            **kwargs: Additional parameters for the attack
            
        Returns:
            Appropriate Attacker instance
        """
        if attack_name not in cls._registry:
            raise ValueError(f"Unknown attack: {attack_name}. "
                           f"Available attacks: {list(cls._registry.keys())}")
        
        attacker_class = cls._registry[attack_name]
        return attacker_class(attack_name=attack_name, **kwargs)
    
    @abstractmethod
    def attack(self, images: torch.Tensor, targets: Optional[torch.Tensor] = None,
               **kwargs) -> torch.Tensor:
        """
        Perform adversarial attack on images.
        
        Args:
            images: Input images tensor of shape (N, C, H, W)
            targets: Target labels for targeted attacks (optional)
            **kwargs: Additional attack-specific parameters
            
        Returns:
            Adversarially perturbed images tensor
        """
        pass
    
    def pgd_attack(self, images: torch.Tensor, loss_fn: callable,
                   targets: Optional[torch.Tensor] = None,
                   random_start: bool = True,
                   progress_callback: Optional[callable] = None,
                   batch_size: int = 10) -> torch.Tensor:
        """
        Projected Gradient Descent (PGD) attack implementation with batching.
        
        Args:
            images: Input images tensor
            loss_fn: Loss function that takes (images, targets) and returns loss
            targets: Target labels (for targeted attacks)
            random_start: Whether to start from random perturbation
            progress_callback: Optional callback for progress tracking
            batch_size: Size of batches to process at once (to save memory)
            
        Returns:
            Adversarially perturbed images
            
        Note:
            This implementation processes images in batches to save memory.
            Each batch is processed independently to allow for large datasets.
            Memory is freed after processing each batch.
            Progress is tracked both for batches and overall progress.
        """
        images = images.to(self.device)
        if targets is not None:
            targets = targets.to(self.device)
        
        total_images = len(images)
        num_batches = (total_images + batch_size - 1) // batch_size
        all_adv_images = []
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_images)
            
            batch_images = images[start_idx:end_idx]
            batch_targets = targets[start_idx:end_idx] if targets is not None else None
            
            self.logger.info(f"Processing batch {batch_idx + 1}/{num_batches} (images {start_idx} to {end_idx})")
            
            # Initialize perturbation for this batch
            if random_start:
                delta = torch.empty_like(batch_images).uniform_(-self.epsilon, self.epsilon)
            else:
                delta = torch.zeros_like(batch_images)
            
            delta.requires_grad_(True)
            
            # Initialize momentum accumulator
            momentum_accumulator = torch.zeros_like(delta)
            
            for step in range(self.num_steps):
                # Clear GPU cache periodically
                if step % 10 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                
                # Compute adversarial images
                adv_images = batch_images + delta
                adv_images = clip_images(adv_images, 0.0, 1.0)
                
                # Compute loss
                try:
                    loss = loss_fn(adv_images, batch_targets)
                    self.logger.debug(f"Batch {batch_idx + 1}, Step {step}: Loss = {loss.item():.6f}")
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        self.logger.error(f"Batch {batch_idx + 1}, Step {step}: Critical memory error - stopping attack: {e}")
                        raise  # Re-raise to stop the attack
                    else:
                        self.logger.error(f"Batch {batch_idx + 1}, Step {step}: Loss computation failed: {e}")
                        continue
                except Exception as e:
                    self.logger.error(f"Batch {batch_idx + 1}, Step {step}: Loss computation failed: {e}")
                    continue
                
                # For targeted attacks, minimize loss; for untargeted, maximize
                if self.targeted:
                    loss = -loss
                
                # Compute gradients
                try:
                    loss.backward()
                except Exception as e:
                    self.logger.error(f"Batch {batch_idx + 1}, Step {step}: Backward pass failed: {e}")
                    continue
                
                # Check if gradients exist
                if delta.grad is None:
                    self.logger.warning(f"Batch {batch_idx + 1}, Step {step}: No gradients")
                    continue
                
                # Update perturbation with momentum
                with torch.no_grad():
                    # Update momentum accumulator
                    grad = delta.grad.sign()
                    momentum_accumulator = self.momentum * momentum_accumulator + grad
                    
                    # Update delta using momentum
                    new_delta = delta + self.step_size * momentum_accumulator.sign()
                    new_delta = torch.clamp(new_delta, -self.epsilon, self.epsilon)
                    new_delta = torch.clamp(batch_images + new_delta, 0.0, 1.0) - batch_images
                    delta.copy_(new_delta)
                    delta.grad.zero_()
                
                # Call progress callback if provided
                if progress_callback is not None:
                    overall_step = batch_idx * self.num_steps + step
                    total_steps = num_batches * self.num_steps
                    batch_progress = {
                        'batch_idx': batch_idx,
                        'num_batches': num_batches,
                        'step': step,
                        'num_steps': self.num_steps,
                        'overall_progress': (overall_step + 1) / total_steps
                    }
                    progress_callback(overall_step + 1, adv_images.detach(), batch_progress)
                    
                    # Log memory usage in debug mode
                    if self.logger.isEnabledFor(logging.DEBUG):
                        if torch.cuda.is_available():
                            allocated = torch.cuda.memory_allocated() / 1024**2
                            reserved = torch.cuda.memory_reserved() / 1024**2
                            self.logger.debug(
                                f"Batch {batch_idx + 1}/{num_batches}, Step {step + 1}/{self.num_steps} - "
                                f"GPU Memory: Allocated={allocated:.1f}MB, Reserved={reserved:.1f}MB"
                            )
            
            # Final adversarial images for this batch
            with torch.no_grad():
                final_adv_images = batch_images + delta
                final_adv_images = clip_images(final_adv_images, 0.0, 1.0)
                all_adv_images.append(final_adv_images.cpu())  # Move to CPU to save GPU memory
            
            # Clean up batch tensors
            del delta, adv_images
            if 'loss' in locals():
                del loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            torch.mps.empty_cache() if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else None
        
        # Combine all batches
        try:
            combined_adv_images = torch.cat(all_adv_images, dim=0)
            # Clean up individual batch results
            del all_adv_images
            # Move combined results to device
            combined_adv_images = combined_adv_images.to(self.device)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            torch.mps.empty_cache() if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else None
            return combined_adv_images
        except Exception as e:
            self.logger.error(f"Error combining batch results: {e}")
            # Clean up on error
            del all_adv_images
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            torch.mps.empty_cache() if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else None
            raise
    
    def get_attack_info(self) -> Dict[str, Any]:
        """Get information about the attack configuration."""
        return {
            'attack_name': self.attack_name,
            'attack_type': self.attack_type,
            'epsilon': self.epsilon,
            'num_steps': self.num_steps,
            'step_size': self.step_size,
            'targeted': self.targeted,
            'device': self.device,
            'momentum': self.momentum,
            'class_name': self.__class__.__name__
        }
    
    def get_attack_stats(self) -> Dict[str, float]:
        """Get attack statistics."""
        stats = self.attack_stats.copy()
        if stats['total_attacks'] > 0:
            stats['overall_success_rate'] = stats['successful_attacks'] / stats['total_attacks']
        else:
            stats['overall_success_rate'] = 0.0
        return stats
    
    def reset_stats(self):
        """Reset attack statistics."""
        self.attack_stats = {
            'total_attacks': 0,
            'successful_attacks': 0,
            'average_perturbation': 0.0,
            'average_iterations': 0.0
        }
    
    def save_attack_results(self, original_images: torch.Tensor,
                           adversarial_images: torch.Tensor,
                           metrics: Dict[str, float],
                           filepath: str):
        """
        Save attack results to file.
        
        Args:
            original_images: Original images
            adversarial_images: Adversarial images
            metrics: Attack metrics
            filepath: Path to save results
        """
        results = {
            'original_images': original_images.cpu(),
            'adversarial_images': adversarial_images.cpu(),
            'perturbations': (adversarial_images - original_images).cpu(),
            'metrics': metrics,
            'attack_info': self.get_attack_info(),
            'attack_stats': self.get_attack_stats()
        }
        
        torch.save(results, filepath)
        self.logger.info(f"Attack results saved to {filepath}")
    
    def load_attack_results(self, filepath: str) -> Dict[str, Any]:
        """
        Load previously saved attack results.
        
        Args:
            filepath: Path to results file
            
        Returns:
            Dictionary containing attack results
        """
        results = torch.load(filepath, map_location='cpu')
        self.logger.info(f"Attack results loaded from {filepath}")
        return results