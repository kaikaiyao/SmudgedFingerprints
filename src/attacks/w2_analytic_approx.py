"""
W2 Attack: Analytic Approximation Attack.

This attack uses analytic approximations of non-differentiable fingerprint extractors
to generate adversarial examples. It's particularly effective against methods like
Nataraj19 that use discrete operations like histograms or co-occurrence matrices.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any

from .base import Attacker
from ..fingerprints import FingerprintExtractor
from ..fingerprints.nataraj19 import Nataraj19_Approx
from ..fingerprints.mccloskey18 import McCloskey18_Approx
from ..fingerprints.durall20 import Durall20_Approx
from ..fingerprints.dzanic20 import Dzanic20_Approx
from ..fingerprints.marra19a import Marra19a_Approx
from ..fingerprints.nowroozi22 import Nowroozi22_Approx
from ..fingerprints.corvi23 import Corvi23R_Approx, Corvi23S_Approx
from ..fingerprints.giudice21 import Giudice21_Approx


def create_fingerprint_extractor(fingerprint_method: str, device: str):
    """Create fingerprint extractor with proper parameter handling for different methods."""
    # List of fingerprint methods that accept device parameter
    device_accepting_methods = {
        'wang20', 'nataraj19', 'song24', 'giudice21', 'corvi23r', 'corvi23s'
    }
    
    if fingerprint_method in device_accepting_methods:
        # For methods that accept device parameter
        return FingerprintExtractor.create(fingerprint_method, device=device)
    else:
        # For methods that don't accept device parameter (like durall20, nowroozi22, etc.)
        return FingerprintExtractor.create(fingerprint_method)


@Attacker.register("w2")
class Attacker_W2(Attacker):
    """
    W2 Attack using analytic approximations.
    
    This attack leverages differentiable approximations of originally 
    non-differentiable fingerprint extractors to compute gradients
    and generate adversarial perturbations.
    """
    
    def __init__(self, fingerprint_method: str, attribution_model: nn.Module,
                 attack_type: str = "removal", epsilon: float = 0.1,
                 num_steps: int = 100, step_size: float = 0.01,
                 targeted: bool = False, device: str = "cuda",
                 temperature: float = 0.1, confidence_penalty: float = 0.0):
        """
        Initialize W2 attacker.
        
        Args:
            fingerprint_method: Name of the fingerprint method to attack
            attribution_model: Trained attribution model h
            attack_type: Type of attack ("removal" or "forgery")
            epsilon: Maximum perturbation magnitude
            num_steps: Number of attack iterations
            step_size: Step size for iterative attacks
            targeted: Whether to perform targeted attack
            device: Device to run attacks on
            temperature: Temperature for soft approximations
            confidence_penalty: Penalty term for overconfident predictions
        """
        super().__init__(
            attack_name="w2",
            attack_type=attack_type,
            epsilon=epsilon,
            num_steps=num_steps,
            step_size=step_size,
            targeted=targeted,
            device=device
        )
        
        self.fingerprint_method = fingerprint_method
        self.attribution_model = attribution_model.to(device)
        self.attribution_model.eval()
        self.temperature = temperature
        self.confidence_penalty = confidence_penalty
        
        # Initialize original and approximate extractors
        self._setup_extractors()
        
        # Model compatibility is now verified during extractor setup
    
    def _setup_extractors(self):
        """Setup original and approximate fingerprint extractors."""
        # Initialize original extractor
        self.original_extractor = create_fingerprint_extractor(self.fingerprint_method, self.device)
        
        # Initialize analytic approximation
        if self.fingerprint_method == "nataraj19":
            self.approx_extractor = Nataraj19_Approx(
                self.original_extractor, temperature=self.temperature
            )
        elif self.fingerprint_method == "mccloskey18":
            self.approx_extractor = McCloskey18_Approx(
                self.original_extractor, temperature=self.temperature
            )
        elif self.fingerprint_method == "durall20":
            self.approx_extractor = Durall20_Approx(self.original_extractor)
        elif self.fingerprint_method == "dzanic20":
            self.approx_extractor = Dzanic20_Approx(self.original_extractor)
        elif self.fingerprint_method == "marra19a":
            self.approx_extractor = Marra19a_Approx(self.original_extractor)
        elif self.fingerprint_method == "qian20":
            # Qian20 is an implicit fingerprint method, no analytic approximation needed
            self.approx_extractor = None
            self.logger.info("Qian20 is an implicit fingerprint method - no analytic approximation needed")
        elif self.fingerprint_method == "wang20":
            # Wang20 is an implicit fingerprint method, no analytic approximation needed
            self.approx_extractor = None
            self.logger.info("Wang20 is an implicit fingerprint method - no analytic approximation needed")
        elif self.fingerprint_method == "nowroozi22":
            # Nowroozi22 has very large feature dimensions (393,216 for full, 24,576 for reduced)
            # Check what dimension the attribution model expects and use the appropriate one
            expected_dim = None
            
            # Try multiple ways to detect the expected input dimension
            if hasattr(self.attribution_model, 'fc') and hasattr(self.attribution_model.fc, 'in_features'):
                expected_dim = self.attribution_model.fc.in_features
                self.logger.info(f"Detected input dimension from fc.in_features: {expected_dim}")
            elif hasattr(self.attribution_model, 'classifier') and hasattr(self.attribution_model.classifier, 'in_features'):
                expected_dim = self.attribution_model.classifier.in_features
                self.logger.info(f"Detected input dimension from classifier.in_features: {expected_dim}")
            elif hasattr(self.attribution_model, 'linear') and hasattr(self.attribution_model.linear, 'in_features'):
                expected_dim = self.attribution_model.linear.in_features
                self.logger.info(f"Detected input dimension from linear.in_features: {expected_dim}")
            else:
                # Try to detect by testing with dummy inputs
                self.logger.info("Could not detect input dimension from model attributes, testing with dummy inputs...")
                try:
                    # Test with full dimension first
                    dummy_full = torch.randn(1, 393216, device=self.device)
                    with torch.no_grad():
                        _ = self.attribution_model(dummy_full)
                    expected_dim = 393216
                    self.logger.info("Model accepts full dimension (393,216 features)")
                except Exception as e1:
                    try:
                        # Test with reduced dimension
                        dummy_reduced = torch.randn(1, 24576, device=self.device)
                        with torch.no_grad():
                            _ = self.attribution_model(dummy_reduced)
                        expected_dim = 24576
                        self.logger.info("Model accepts reduced dimension (24,576 features)")
                    except Exception as e2:
                        self.logger.error(f"Could not determine model input dimension. Full dim error: {e1}")
                        self.logger.error(f"Reduced dim error: {e2}")
                        # Default to full dimension as it's more common
                        expected_dim = 393216
                        self.logger.warning("Defaulting to full dimension (393,216 features)")
            
            # Determine which dimension to use based on model expectation
            if expected_dim == 393216:
                # Model expects full dimension
                use_reduced_dim = False
                self.logger.info("Using full dimension Nowroozi22 approximation (393,216 features) to match model")
            elif expected_dim == 24576:
                # Model expects reduced dimension
                use_reduced_dim = True
                self.logger.info("Using reduced dimension Nowroozi22 approximation (24,576 features) to match model")
            else:
                # Unknown dimension, try reduced first for memory efficiency
                use_reduced_dim = True
                self.logger.warning(f"Unknown expected dimension {expected_dim}, using reduced dimension for memory efficiency")
            
            try:
                self.approx_extractor = Nowroozi22_Approx(
                    self.original_extractor, temperature=self.temperature,
                    reduced_dim=use_reduced_dim, feature_reduction_factor=4
                )
                feature_dim = self.approx_extractor.effective_feature_dim
                self.logger.info(f"Initialized Nowroozi22 approximation with {feature_dim} features")
                
                # Verify that the extractor produces the expected dimension
                try:
                    # Create a dummy image to test the extractor
                    dummy_image = torch.randn(1, 3, 256, 256, device=self.device)
                    dummy_fingerprint = self.approx_extractor.extract_fingerprint_approx(dummy_image)
                    
                    # Flatten if needed
                    if dummy_fingerprint.dim() > 2:
                        dummy_fingerprint = dummy_fingerprint.view(dummy_fingerprint.shape[0], -1)
                    
                    actual_dim = dummy_fingerprint.shape[1]
                    self.logger.info(f"Extractor produces fingerprints with dimension: {actual_dim}")
                    
                    # Test if the model accepts this dimension
                    with torch.no_grad():
                        _ = self.attribution_model(dummy_fingerprint)
                    
                    self.logger.info(f"✓ Model compatibility verified: extractor ({actual_dim}) matches model expectations")
                    
                except Exception as e:
                    self.logger.error(f"✗ Model compatibility test failed: {e}")
                    self.logger.error(f"Extractor produces {actual_dim if 'actual_dim' in locals() else 'unknown'} features")
                    self.logger.error(f"Model expects {expected_dim} features")
                    raise ValueError(f"Model compatibility test failed: {e}")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize Nowroozi22 approximation: {e}")
                raise ValueError(f"Failed to initialize Nowroozi22 approximation: {e}")
        elif self.fingerprint_method == "corvi23r":
            # Corvi23-R is now differentiable, no analytic approximation needed
            self.approx_extractor = None
            self.logger.info("Corvi23-R is now differentiable - no analytic approximation needed")
        elif self.fingerprint_method == "corvi23s":
            # Corvi23-S is now differentiable, no analytic approximation needed
            self.approx_extractor = None
            self.logger.info("Corvi23-S is now differentiable - no analytic approximation needed")
        elif self.fingerprint_method == "giudice21":
            # Giudice21 is already differentiable, no analytic approximation needed
            self.approx_extractor = None
            self.logger.info("Giudice21 is already differentiable - no analytic approximation needed")
        else:
            raise ValueError(f"Analytic approximation not available for {self.fingerprint_method}")
        
        self.logger.info(f"Initialized W2 attack for {self.fingerprint_method} with analytic approximation")
    
    def attack(self, images: torch.Tensor, targets: Optional[torch.Tensor] = None,
               attack_batch_size: int = 10, **kwargs) -> torch.Tensor:
        """
        Perform W2 attack using analytic approximation.
        
        Args:
            images: Input images tensor of shape (N, C, H, W)
            targets: For removal: original labels. For forgery: per-sample target labels.
            attack_batch_size: Batch size for PGD attack to save memory (smaller = less memory)
            **kwargs: Additional attack parameters
            
        Returns:
            Adversarially perturbed images
            
        Note:
            This method processes images in batches to save memory. The actual batch size
            used might be smaller than attack_batch_size if there's not enough memory.
            Memory is freed after processing each batch.
        """
        # Log initial attack parameters
        self.logger.info(f"Starting W2 attack with parameters:")
        self.logger.info(f"  Attack type: {self.attack_type}")
        self.logger.info(f"  Epsilon: {self.epsilon}")
        self.logger.info(f"  Number of steps: {self.num_steps}")
        self.logger.info(f"  Step size: {self.step_size}")
        self.logger.info(f"  Temperature: {self.temperature}")
        self.logger.info(f"  Confidence penalty: {self.confidence_penalty}")
        self.logger.info(f"  Attack batch size: {attack_batch_size}")
        
        # Ensure images are on the correct device
        images = images.to(self.device)
        total_images = images.shape[0]
        self.logger.info(f"Total images to process: {total_images} with shape {images.shape}")
        
        # For Nowroozi22, reduce batch size to prevent memory issues
        if self.fingerprint_method == "nowroozi22":
            original_batch_size = attack_batch_size
            # Check if we're using full dimension (which requires more memory)
            if hasattr(self.approx_extractor, 'effective_feature_dim') and self.approx_extractor.effective_feature_dim > 100000:
                attack_batch_size = min(attack_batch_size, 1)  # Full dimension: limit to 1 image per batch
                self.logger.info(f"Reduced attack batch size from {original_batch_size} to {attack_batch_size} for Nowroozi22 full dimension")
            else:
                attack_batch_size = min(attack_batch_size, 2)  # Reduced dimension: limit to 2 images per batch
                if original_batch_size != attack_batch_size:
                    self.logger.info(f"Reduced attack batch size from {original_batch_size} to {attack_batch_size} for Nowroozi22 reduced dimension")
        
        # Store target class for forgery attacks
        # Simplified forgery logic handled via per-sample targets passed in loss
        
        # Define loss function using analytic approximation
        def loss_fn(adv_images, targets=None):
            try:
                # Ensure the input requires gradients
                if not adv_images.requires_grad:
                    adv_images.requires_grad_(True)
                
                # Compute fingerprints and free memory of intermediate results
                try:
                    # For Nowroozi22, use smaller batch size if needed
                    if self.fingerprint_method == "nowroozi22" and adv_images.shape[0] > 1:
                        # Process in smaller sub-batches for memory efficiency (full dimension is very large)
                        # Use even smaller batches for Nowroozi22 due to its high memory requirements
                        # For full dimension (393,216 features), process one image at a time
                        # For reduced dimension (24,576 features), can process 2 images at a time
                        if hasattr(self.approx_extractor, 'effective_feature_dim') and self.approx_extractor.effective_feature_dim > 100000:
                            sub_batch_size = 1  # Full dimension: process one image at a time
                        else:
                            sub_batch_size = 2  # Reduced dimension: process two images at a time
                        
                        all_fingerprints = []
                        for i in range(0, adv_images.shape[0], sub_batch_size):
                            sub_batch = adv_images[i:i+sub_batch_size]
                            sub_fingerprints = self.approx_extractor.extract_fingerprint_approx(sub_batch)
                            all_fingerprints.append(sub_fingerprints)
                            # Clear cache after each sub-batch
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                            torch.mps.empty_cache() if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else None
                            # Additional cleanup for Nowroozi22
                            del sub_batch, sub_fingerprints
                        approx_fingerprints = torch.cat(all_fingerprints, dim=0)
                        # Clean up the list after concatenation
                        del all_fingerprints
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        torch.mps.empty_cache() if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else None
                    else:
                        approx_fingerprints = self.approx_extractor.extract_fingerprint_approx(adv_images)
                    
                    approx_fingerprints = approx_fingerprints.to(self.device)
                    
                    # Flatten fingerprints if they are multi-dimensional
                    if approx_fingerprints.dim() > 2:
                        batch_size = approx_fingerprints.shape[0]
                        approx_fingerprints = approx_fingerprints.view(batch_size, -1)
                        self.logger.debug(f"Flattened fingerprints to shape {approx_fingerprints.shape}")
                    
                    # Verify fingerprint dimension matches model expectation
                    if hasattr(self.attribution_model, 'fc') and hasattr(self.attribution_model.fc, 'in_features'):
                        expected_dim = self.attribution_model.fc.in_features
                        actual_dim = approx_fingerprints.shape[1]
                        if expected_dim != actual_dim:
                            self.logger.error(f"Fingerprint dimension mismatch: expected {expected_dim}, got {actual_dim}")
                            raise ValueError(f"Fingerprint dimension mismatch: expected {expected_dim}, got {actual_dim}")
                except Exception as e:
                    self.logger.error(f"Error in fingerprint extraction: {e}")
                    raise
                
                # Get model predictions
                try:
                    self.logger.debug(f"Input fingerprint shape: {approx_fingerprints.shape}")
                    logits = self.attribution_model(approx_fingerprints)
                    del approx_fingerprints  # Free memory early
                except Exception as e:
                    self.logger.error(f"Error in model prediction: {e}")
                    self.logger.error(f"Fingerprint shape: {approx_fingerprints.shape if 'approx_fingerprints' in locals() else 'unknown'}")
                    if hasattr(self.attribution_model, 'fc') and hasattr(self.attribution_model.fc, 'in_features'):
                        self.logger.error(f"Model expects input dimension: {self.attribution_model.fc.in_features}")
                    if 'approx_fingerprints' in locals(): del approx_fingerprints
                    if 'logits' in locals(): del logits
                    raise
                
                # Compute loss based on attack type
                try:
                    if self.attack_type == "removal":
                        if targets is None:
                            raise ValueError("Removal requires original labels provided as 'targets'.")
                        loss = torch.nn.functional.cross_entropy(logits, targets)
                    elif self.attack_type == "forgery":
                        if targets is None:
                            raise ValueError("Forgery requires per-sample target labels provided as 'targets'.")
                        loss = -torch.nn.functional.cross_entropy(logits, targets)
                    else:
                        if targets is None:
                            raise ValueError("Targets must be provided for attack loss computation.")
                        loss = torch.nn.functional.cross_entropy(logits, targets)
                    
                    # Log stats if in debug mode
                    # Optional: prediction stats (skipped to save memory)
                    return loss
                    
                except Exception as e:
                    self.logger.error(f"Error in loss computation: {e}")
                    if 'probabilities' in locals(): del probabilities
                    raise
                
            except Exception as e:
                self.logger.error(f"Error in loss function: {e}")
                # Clean up on error
                if 'approx_fingerprints' in locals(): del approx_fingerprints
                if 'logits' in locals(): del logits
                if 'probabilities' in locals(): del probabilities
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                torch.mps.empty_cache() if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else None
                raise
        
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
        
        # Perform PGD attack with batching
        self.logger.info(f"Starting PGD attack iterations (total steps: {self.num_steps}, batch size: {attack_batch_size})...")
        try:
            # Perform PGD attack
            adversarial_images = self.pgd_attack(
                images, loss_fn, targets, random_start=True,
                progress_callback=progress_callback,
                batch_size=attack_batch_size
            )
            
            # Calculate and log final attack statistics
            with torch.no_grad():
                try:
                    # Process statistics in batches to save memory
                    batch_size = min(attack_batch_size, len(adversarial_images))
                    total_l2 = 0
                    total_linf = 0
                    total_conf = 0
                    total_success = None  # per-sample forgery success is evaluated outside
                    num_batches = (len(adversarial_images) + batch_size - 1) // batch_size
                    
                    for i in range(0, len(adversarial_images), batch_size):
                        batch_end = min(i + batch_size, len(adversarial_images))
                        batch_adv = adversarial_images[i:batch_end].to(self.device)
                        batch_orig = images[i:batch_end].to(self.device)
                        
                        # Compute distances
                        l2_dist = torch.norm(batch_adv - batch_orig, p=2, dim=(1,2,3)).mean().item()
                        linf_dist = torch.norm(batch_adv - batch_orig, p=float('inf'), dim=(1,2,3)).mean().item()
                        
                        # Get predictions
                        batch_fp = self.approx_extractor.extract_fingerprint_approx(batch_adv)
                        if batch_fp.dim() > 2:
                            batch_fp = batch_fp.view(batch_fp.shape[0], -1)
                        batch_logits = self.attribution_model(batch_fp)
                        batch_probs = torch.softmax(batch_logits, dim=1)
                        batch_conf, batch_preds = torch.max(batch_probs, dim=1)
                        
                        # Accumulate statistics
                        total_l2 += l2_dist * (batch_end - i)
                        total_linf += linf_dist * (batch_end - i)
                        total_conf += batch_conf.sum().item()
                        
                        # per-sample forgery success not aggregated here
                        
                        # Clean up batch data
                        del batch_adv, batch_orig, batch_fp, batch_logits, batch_probs, batch_conf, batch_preds
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        torch.mps.empty_cache() if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else None
                    
                    # Calculate averages
                    avg_l2 = total_l2 / len(adversarial_images)
                    avg_linf = total_linf / len(adversarial_images)
                    avg_conf = total_conf / len(adversarial_images)
                    
                    # Log final statistics
                    self.logger.info(f"Attack completed on {len(adversarial_images)} images")
                    self.logger.info(f"Final statistics:")
                    self.logger.info(f"  Average L2 distance: {avg_l2:.4f}")
                    self.logger.info(f"  Average L∞ distance: {avg_linf:.4f}")
                    self.logger.info(f"  Average confidence: {avg_conf:.4f}")
                    
                    if total_success is not None:
                        success_rate = total_success / len(adversarial_images)
                        self.logger.info(f"  Forgery success rate: {success_rate:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Error computing final statistics: {e}")
                    # Error in statistics doesn't affect the attack result
                    pass
                finally:
                    # Clean up
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    torch.mps.empty_cache() if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else None
            
            return adversarial_images
            
        except Exception as e:
            self.logger.error(f"Error during PGD attack: {e}")
            # Clean up
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            torch.mps.empty_cache() if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else None
            raise
    
    def _compute_attack_loss(self, images: torch.Tensor, 
                            targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute attack loss using analytic approximation.
        
        Args:
            images: Input images
            
        Returns:
            Attack loss tensor
        """
        # Ensure input images require gradients for proper backpropagation
        if not images.requires_grad:
            images.requires_grad_(True)
        
        # Extract approximate fingerprints (differentiable)
        approx_fingerprints = self.approx_extractor.extract_fingerprint_approx(images)
        
        # Ensure fingerprints are on the correct device and maintain gradient connection
        approx_fingerprints = approx_fingerprints.to(self.device)
        
        # Flatten fingerprints if they are multi-dimensional (e.g., Nataraj19's 3x256x256)
        if approx_fingerprints.dim() > 2:
            approx_fingerprints = approx_fingerprints.view(approx_fingerprints.shape[0], -1)
            self.logger.debug(f"Flattened fingerprints to shape {approx_fingerprints.shape}")
        
        # Get model predictions
        logits = self.attribution_model(approx_fingerprints)
        probabilities = torch.softmax(logits, dim=1)
        
        if self.attack_type == "removal":
            # For removal (untargeted), ascend a loss that DECREASES original class probability
            if targets is not None:
                original_class_probs = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
                loss = -original_class_probs.mean()
            else:
                # Fallback: decrease the maximum class probability
                max_probs, _ = torch.max(probabilities, dim=1)
                loss = -max_probs.mean()
            
        elif self.attack_type == "forgery":
            # For forgery attacks, maximize probability of per-sample target class
            if targets is None:
                raise ValueError("Forgery requires per-sample target labels provided as 'targets'.")
            target_probs = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
            loss = -target_probs.mean()
        
        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")
        
        # Add confidence penalty if specified
        if self.confidence_penalty > 0:
            # Penalize overconfident predictions
            confidence_penalty = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1).mean()
            loss += self.confidence_penalty * confidence_penalty
        
        return loss
    
    def validate_approximation(self, test_images: torch.Tensor, 
                              tolerance: float = 0.1) -> Dict[str, float]:
        """
        Validate the quality of the analytic approximation.
        
        Args:
            test_images: Test images for validation
            tolerance: Acceptable approximation error
            
        Returns:
            Validation metrics
        """
        self.logger.info("Validating analytic approximation quality")
        
        # Ensure test images are on the correct device
        test_images = test_images.to(self.device)
        
        with torch.no_grad():
            # Extract features using both methods
            original_features = self.original_extractor.extract_fingerprint(test_images)
            approx_features = self.approx_extractor.extract_fingerprint_approx(test_images)
            
            # Ensure features are on the correct device
            original_features = original_features.to(self.device)
            approx_features = approx_features.to(self.device)
            
            # Get predictions using both feature sets
            original_logits = self.attribution_model(original_features)
            approx_logits = self.attribution_model(approx_features)
            
            original_probs = torch.softmax(original_logits, dim=1)
            approx_probs = torch.softmax(approx_logits, dim=1)
            
            # Calculate feature-level metrics
            feature_mse = torch.mean((original_features - approx_features) ** 2).item()
            feature_mae = torch.mean(torch.abs(original_features - approx_features)).item()
            feature_max_error = torch.max(torch.abs(original_features - approx_features)).item()
            
            # Calculate prediction-level metrics
            pred_mse = torch.mean((original_probs - approx_probs) ** 2).item()
            pred_mae = torch.mean(torch.abs(original_probs - approx_probs)).item()
            
            # Check prediction agreement
            _, orig_preds = torch.max(original_logits, 1)
            _, approx_preds = torch.max(approx_logits, 1)
            prediction_agreement = (orig_preds == approx_preds).float().mean().item()
            
            metrics = {
                'feature_mse': feature_mse,
                'feature_mae': feature_mae,
                'feature_max_error': feature_max_error,
                'prediction_mse': pred_mse,
                'prediction_mae': pred_mae,
                'prediction_agreement': prediction_agreement,
                'within_tolerance': feature_max_error <= tolerance,
                'tolerance': tolerance
            }
            
            self.logger.info(f"Approximation validation results:")
            self.logger.info(f"  Feature MSE: {feature_mse:.6f}")
            self.logger.info(f"  Feature MAE: {feature_mae:.6f}")
            self.logger.info(f"  Prediction Agreement: {prediction_agreement:.4f}")
            self.logger.info(f"  Within Tolerance: {metrics['within_tolerance']}")
            
            return metrics
    
    def attack_with_validation(self, images: torch.Tensor, 
                              targets: Optional[torch.Tensor] = None,
                              validate_approx: bool = True,
                              **kwargs) -> tuple:
        """
        Perform attack with optional approximation validation.
        
        Args:
            images: Input images
            targets: Original labels (removal) or per-sample targets (forgery)
            validate_approx: Whether to validate approximation quality
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (adversarial_images, validation_metrics)
        """
        validation_metrics = None
        
        if validate_approx:
            # Use a subset of images for validation to save computation
            val_images = images[:min(100, len(images))]
            # Ensure validation images are on the correct device
            val_images = val_images.to(self.device)
            validation_metrics = self.validate_approximation(val_images)
            
            # Warn if approximation quality is poor
            if not validation_metrics['within_tolerance']:
                self.logger.warning(
                    f"Approximation quality may be poor. "
                    f"Max error: {validation_metrics['feature_max_error']:.6f}, "
                    f"Tolerance: {validation_metrics['tolerance']:.6f}"
                )
        
        # Perform attack
        adversarial_images = self.attack(images, targets, **kwargs)
        
        return adversarial_images, validation_metrics
    
    def adaptive_temperature_attack(self, images: torch.Tensor,
                                   temperatures: list = [0.01, 0.05, 0.1, 0.2],
                                   targets: Optional[torch.Tensor] = None) -> tuple:
        """
        Perform attack with adaptive temperature selection.
        
        This method tries different temperature values for the analytic approximation
        and selects the one that provides the best attack performance.
        
        Args:
            images: Input images
            temperatures: List of temperature values to try
            targets: Original labels (removal) or per-sample targets (forgery)
            
        Returns:
            Tuple of (best_adversarial_images, best_temperature, results)
        """
        best_success_rate = -1
        best_adversarial_images = None
        best_temperature = None
        all_results = {}
        
        original_temp = self.temperature
        
        for temp in temperatures:
            self.logger.info(f"Trying temperature: {temp}")
            
            # Update temperature
            self.temperature = temp
            self._setup_extractors()  # Reinitialize with new temperature
            
            # Perform attack
            adversarial_images = self.attack(images, targets)
            
            # Evaluate success (simplified - using original extractor for ground truth)
            # Ensure images are on the correct device for feature extraction
            images_device = images.to(self.device)
            adversarial_images_device = adversarial_images.to(self.device)
            
            original_features = self.original_extractor.extract_fingerprint(images_device)
            adv_features = self.original_extractor.extract_fingerprint(adversarial_images_device)
            
            # Ensure features are on the correct device
            original_features = original_features.to(self.device)
            adv_features = adv_features.to(self.device)
            
            original_preds = self.attribution_model(original_features)
            adv_preds = self.attribution_model(adv_features)
            
            _, orig_labels = torch.max(original_preds, 1)
            _, adv_labels = torch.max(adv_preds, 1)
            
            if self.attack_type == "removal":
                success_rate = (orig_labels != adv_labels).float().mean().item()
            elif self.attack_type == "forgery":
                # Compare against provided per-sample targets
                if targets is None:
                    success_rate = 0.0
                else:
                    # Restrict to processed batch window
                    comp_targets = targets[:len(adv_labels)].to(adv_labels.device)
                    comparison_result = (adv_labels == comp_targets)
                    success_rate = comparison_result.float().mean().item()
            else:
                success_rate = (orig_labels != adv_labels).float().mean().item()
            
            all_results[temp] = {
                'adversarial_images': adversarial_images.clone(),
                'success_rate': success_rate
            }
            
            self.logger.info(f"Temperature {temp}: Success rate = {success_rate:.4f}")
            
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_adversarial_images = adversarial_images.clone()
                best_temperature = temp
        
        # Restore original temperature
        self.temperature = original_temp
        self._setup_extractors()
        
        self.logger.info(f"Best temperature: {best_temperature} (Success rate: {best_success_rate:.4f})")
        
        return best_adversarial_images, best_temperature, all_results