"""
Individual attack runners for different attack types.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod

from ..attack_evaluation.metrics import AttackMetrics
from ..attack_evaluation.target_selection import TargetSelector
from ..data_loader.fingerprint_extractor_factory import FingerprintExtractorFactory


class BaseAttackRunner(ABC):
    """Base class for individual attack runners."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.metrics = AttackMetrics(device)
        self.target_selector = TargetSelector()
    
    @abstractmethod
    def run_attack(
        self,
        X_images: torch.Tensor,
        y_labels: torch.Tensor,
        fingerprint_method: str,
        true_attribution_model: torch.nn.Module,
        attack_params: Dict[str, Any],
        attack_batch_size: int = 10,
        real_data_path: Optional[str] = None,
        cache_dir: str = "cache",
        data_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run the specific attack type."""
        pass


class W1AttackRunner(BaseAttackRunner):
    """Runner for W1 attacks (direct gradient access)."""
    
    def run_attack(
        self,
        X_images: torch.Tensor,
        y_labels: torch.Tensor,
        fingerprint_method: str,
        true_attribution_model: torch.nn.Module,
        attack_params: Dict[str, Any],
        attack_batch_size: int = 10,
        real_data_path: Optional[str] = None,
        cache_dir: str = "cache",
        data_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run W1 attack using direct gradient access to fingerprint model with multiple step sizes."""
        
        print("\n--- W1 Attack (Multi-Step-Size) ---")
        
        # Check fingerprint method properties first
        fingerprint_extractor = FingerprintExtractorFactory.create(
            fingerprint_method,
            self.device,
            real_data_path=real_data_path,
            cache_dir=cache_dir,
            data_dir=data_dir
        )
        is_implicit = fingerprint_extractor.is_implicit_fingerprint
        is_differentiable = fingerprint_extractor.is_differentiable
        
        # Get original predictions first
        with torch.no_grad():
            if is_implicit:
                # For implicit methods, directly use the model on images
                orig_logits = true_attribution_model(X_images.to(self.device))
                _, original_preds = torch.max(orig_logits, 1)
            else:
                # For explicit methods, extract fingerprints first
                orig_fingerprints = fingerprint_extractor.extract_fingerprint(X_images)
                if orig_fingerprints.dim() > 2:
                    orig_fingerprints = orig_fingerprints.view(orig_fingerprints.shape[0], -1)
                orig_logits = true_attribution_model(orig_fingerprints.to(self.device))
                _, original_preds = torch.max(orig_logits, 1)
        
        # Define correct_mask once for all step sizes
        y_labels = y_labels.to(original_preds.device)
        correct_mask = (original_preds == y_labels)
        
        # W1 attack is supported for:
        # 1. Implicit fingerprint methods (where the network itself is the attribution model, like qian20 and wang20)
        # 2. Fully differentiable explicit fingerprint methods (like giudice21)
        if not (is_implicit or is_differentiable):
            raise ValueError(f"W1 attack is only supported for implicit fingerprint methods or "
                            f"fully differentiable explicit methods. Method {fingerprint_method} "
                            f"is not implicit (is_implicit_fingerprint={is_implicit}) and "
                            f"not differentiable (is_differentiable={is_differentiable}).")
        
        # Get step sizes to test
        step_sizes = attack_params.get('step_size_list', [attack_params['step_size']])
        print(f"Testing {len(step_sizes)} step sizes: {step_sizes}")
        
        # Create W1 attacker
        attack_type = attack_params['attack_goal']
        # For forgery we always use per-sample targets; do not set targeted=True (loss handles targeting)
        targeted = False
        
        print(f"Attack goal: {attack_type}")
        
        # Store results for each step size
        all_adversarial_images = []
        all_success_masks = []
        
        for i, step_size in enumerate(step_sizes):
            print(f"\n--- W1 Attack Step Size {i+1}/{len(step_sizes)}: {step_size} ---")
            
            try:
                from src.attacks.w1_direct import Attacker_W1
            except ImportError as e:
                raise ImportError(f"Failed to import W1 attack modules: {e}")
            
            if is_implicit:
                # For implicit methods, the true_attribution_model is the fingerprint model itself
                w1_attacker = Attacker_W1(
                    implicit_fingerprint_model=true_attribution_model,  # The model itself is the implicit fingerprint model
                    attack_type=attack_type,
                    epsilon=attack_params['epsilon'],
                    num_steps=attack_params['num_steps'],
                    step_size=step_size,
                    targeted=targeted,
                    device=self.device
                )
            else:
                # For differentiable explicit methods, we need to create a combined model
                # that includes both the fingerprint extractor and the attribution model
                w1_attacker = Attacker_W1(
                    implicit_fingerprint_model=CombinedFingerprintModel(fingerprint_extractor, true_attribution_model),
                    attack_type=attack_type,
                    epsilon=attack_params['epsilon'],
                    num_steps=attack_params['num_steps'],
                    step_size=step_size,
                    targeted=targeted,
                    device=self.device
                )
            
            # Run attack with this step size
            # For forgery, always pass per-sample targets generated at runtime
            use_targets = attack_params.get('generated_random_targets') if attack_type == 'forgery' else y_labels
            adversarial_images = w1_attacker.attack(
                X_images,
                targets=use_targets,
                attack_batch_size=attack_batch_size
            )
            all_adversarial_images.append(adversarial_images)
            
            # Evaluate success for this step size
            with torch.no_grad():
                if is_implicit:
                    # For implicit methods, directly use the model on images
                    adv_logits = true_attribution_model(adversarial_images)
                    _, adv_preds = torch.max(adv_logits, 1)
                else:
                    # For explicit methods, extract fingerprints first
                    adv_fingerprints = fingerprint_extractor.extract_fingerprint(adversarial_images)
                    if adv_fingerprints.dim() > 2:
                        adv_fingerprints = adv_fingerprints.view(adv_fingerprints.shape[0], -1)
                    adv_logits = true_attribution_model(adv_fingerprints.to(self.device))
                    _, adv_preds = torch.max(adv_logits, 1)
                
                # Calculate success mask for this step size, but only consider S_correct
                if attack_type == 'removal':
                    # Success: prediction changed from original, only count for correctly classified
                    success_mask = (adv_preds != y_labels) & correct_mask
                else:  # forgery
                    # Forgery success against per-sample targets
                    gen_targets = attack_params.get('generated_random_targets')
                    if gen_targets is not None:
                        gen_targets = gen_targets.to(adv_preds.device)
                        if len(gen_targets) > len(adv_preds):
                            gen_targets = gen_targets[:len(adv_preds)]
                        elif len(gen_targets) < len(adv_preds):
                            pad = torch.zeros(len(adv_preds) - len(gen_targets), dtype=gen_targets.dtype, device=gen_targets.device)
                            gen_targets = torch.cat([gen_targets, pad])
                        success_mask = (adv_preds == gen_targets) & correct_mask
                    else:
                        success_mask = torch.zeros_like(correct_mask, dtype=torch.bool)
                
                all_success_masks.append(success_mask)
                # Calculate success rate only on S_correct
                success_rate = success_mask[correct_mask].float().mean().item() if correct_mask.any() else 0.0
                total_correct = int(correct_mask.sum().item())
                successes = int(success_mask.sum().item())
                print(f"  Step size {step_size}: Success rate on S_correct = {success_rate:.4f} ({successes}/{total_correct})")
                # Optional brief breakdown of successful target hits (top-3)
                if attack_type == 'forgery' and successes > 0:
                    try:
                        succ_classes = adv_preds[success_mask]
                        counts = torch.bincount(succ_classes).cpu()
                        topk = torch.topk(counts, k=min(3, len(counts)))
                        breakdown = [f"{int(idx)}:{int(cnt)}" for cnt, idx in zip(topk.values, topk.indices)]
                        print(f"    Top target hits among successes: {', '.join(breakdown)}")
                    except Exception:
                        pass
        
        # Aggregate success across all step sizes
        # Compute attack success metrics using common function
        success_metrics = self.metrics.compute_attack_success_metrics(original_preds, y_labels, all_success_masks, step_sizes)
        
        # Get best adversarial images
        best_adversarial_images = all_adversarial_images[success_metrics['best_step_idx']]
        
        # Return both adversarial images and combined success information
        return {
            'adversarial_images': best_adversarial_images,
            'combined_success_mask': success_metrics['combined_success_mask'],
            'overall_success_rate': success_metrics['overall_success_rate'],
            'individual_success_rates': [mask[correct_mask].float().mean().item() if correct_mask.any() else 0.0 for mask in all_success_masks],
            'step_sizes': step_sizes,
            'best_step_idx': success_metrics['best_step_idx'],
            'original_predictions': original_preds,
            'combined_success_info': {
                'combined_success_mask': success_metrics['combined_success_mask'],
                'step_sizes': step_sizes,
                'individual_success_rates': [mask[correct_mask].float().mean().item() if correct_mask.any() else 0.0 for mask in all_success_masks],
                'best_step_idx': success_metrics['best_step_idx'],
                'overall_success_rate': success_metrics['overall_success_rate']
            }
        }


class W2AttackRunner(BaseAttackRunner):
    """Runner for W2 attacks (analytic approximation)."""
    
    def run_attack(
        self,
        X_images: torch.Tensor,
        y_labels: torch.Tensor,
        fingerprint_method: str,
        true_attribution_model: torch.nn.Module,
        attack_params: Dict[str, Any],
        attack_batch_size: int = 10,
        real_data_path: Optional[str] = None,
        cache_dir: str = "cache",
        data_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run W2 attack using analytic approximation with multiple step sizes."""
        
        print("\n--- W2 Attack (Multi-Step-Size) ---")
        
        # Create fingerprint extractor for true model
        fingerprint_extractor = FingerprintExtractorFactory.create(
            fingerprint_method,
            self.device,
            real_data_path=real_data_path,
            cache_dir=cache_dir,
            data_dir=data_dir
        )
        is_implicit = fingerprint_extractor.is_implicit_fingerprint
        
        # W2 attacks are not supported for implicit fingerprint methods
        if is_implicit:
            raise ValueError(f"W2 attack is not supported for implicit fingerprint methods like {fingerprint_method}. "
                           f"Use W1 attacks instead for implicit methods.")
        
        # Get original predictions first
        with torch.no_grad():
            # For explicit methods, extract fingerprints first
            orig_fingerprints = fingerprint_extractor.extract_fingerprint(X_images)
            if orig_fingerprints.dim() > 2:
                orig_fingerprints = orig_fingerprints.view(orig_fingerprints.shape[0], -1)
            orig_logits = true_attribution_model(orig_fingerprints.to(self.device))
            _, original_preds = torch.max(orig_logits, 1)
        
        # Define correct_mask once for all step sizes
        y_labels = y_labels.to(original_preds.device)
        correct_mask = (original_preds == y_labels)
        
        # Get step sizes to test
        step_sizes = attack_params.get('step_size_list', [attack_params['step_size']])
        print(f"Testing {len(step_sizes)} step sizes: {step_sizes}")
        
        # Create W2 attacker
        attack_type = attack_params['attack_goal']
        targeted = False
        
        print(f"Attack goal: {attack_type}")
        
        # Store results for each step size
        all_adversarial_images = []
        all_success_masks = []
        
        try:
            from src.attacks.w2_analytic_approx import Attacker_W2
        except ImportError as e:
            raise ImportError(f"Failed to import W2 attack modules: {e}")
        
        for i, step_size in enumerate(step_sizes):
            print(f"\n--- W2 Attack Step Size {i+1}/{len(step_sizes)}: {step_size} ---")
            
            w2_attacker = Attacker_W2(
                fingerprint_method=fingerprint_method,
                attribution_model=true_attribution_model,
                attack_type=attack_type,
                epsilon=attack_params['epsilon'],
                num_steps=attack_params['num_steps'],
                step_size=step_size,
                targeted=targeted,
                device=self.device
            )
            
            # Run attack with this step size
            use_targets = attack_params.get('generated_random_targets') if attack_type == 'forgery' else y_labels
            adversarial_images = w2_attacker.attack(
                X_images,
                targets=use_targets,
                attack_batch_size=attack_batch_size
            )
            all_adversarial_images.append(adversarial_images)
            
            # Evaluate success for this step size
            with torch.no_grad():
                # For explicit methods, extract fingerprints first
                adv_fingerprints = fingerprint_extractor.extract_fingerprint(adversarial_images)
                if adv_fingerprints.dim() > 2:
                    adv_fingerprints = adv_fingerprints.view(adv_fingerprints.shape[0], -1)
                adv_logits = true_attribution_model(adv_fingerprints.to(self.device))
                _, adv_preds = torch.max(adv_logits, 1)
                
                # Calculate success mask for this step size, but only consider S_correct
                if attack_type == 'removal':
                    # Success: prediction changed from original, only count for correctly classified
                    success_mask = (adv_preds != y_labels) & correct_mask
                else:  # forgery
                    gen_targets = attack_params.get('generated_random_targets')
                    if gen_targets is not None:
                        gen_targets = gen_targets.to(adv_preds.device)
                        if len(gen_targets) > len(adv_preds):
                            gen_targets = gen_targets[:len(adv_preds)]
                        elif len(gen_targets) < len(adv_preds):
                            pad = torch.zeros(len(adv_preds) - len(gen_targets), dtype=gen_targets.dtype, device=gen_targets.device)
                            gen_targets = torch.cat([gen_targets, pad])
                        success_mask = (adv_preds == gen_targets) & correct_mask
                    else:
                        success_mask = torch.zeros_like(correct_mask, dtype=torch.bool)
                
                all_success_masks.append(success_mask)
                # Calculate success rate only on S_correct
                success_rate = success_mask[correct_mask].float().mean().item() if correct_mask.any() else 0.0
                total_correct = int(correct_mask.sum().item())
                successes = int(success_mask.sum().item())
                print(f"  Step size {step_size}: Success rate on S_correct = {success_rate:.4f} ({successes}/{total_correct})")
                if attack_type == 'forgery' and successes > 0:
                    try:
                        succ_classes = adv_preds[success_mask]
                        counts = torch.bincount(succ_classes).cpu()
                        topk = torch.topk(counts, k=min(3, len(counts)))
                        breakdown = [f"{int(idx)}:{int(cnt)}" for cnt, idx in zip(topk.values, topk.indices)]
                        print(f"    Top target hits among successes: {', '.join(breakdown)}")
                    except Exception:
                        pass
        
        # Aggregate success across all step sizes
        # Compute attack success metrics using common function
        success_metrics = self.metrics.compute_attack_success_metrics(original_preds, y_labels, all_success_masks, step_sizes)
        
        # Get best adversarial images
        best_adversarial_images = all_adversarial_images[success_metrics['best_step_idx']]
        
        # Return both adversarial images and combined success information
        return {
            'adversarial_images': best_adversarial_images,
            'combined_success_mask': success_metrics['combined_success_mask'],
            'overall_success_rate': success_metrics['overall_success_rate'],
            'individual_success_rates': [mask[correct_mask].float().mean().item() if correct_mask.any() else 0.0 for mask in all_success_masks],
            'step_sizes': step_sizes,
            'best_step_idx': success_metrics['best_step_idx'],
            'original_predictions': original_preds,
            'combined_success_info': {
                'combined_success_mask': success_metrics['combined_success_mask'],
                'step_sizes': step_sizes,
                'individual_success_rates': [mask[correct_mask].float().mean().item() if correct_mask.any() else 0.0 for mask in all_success_masks],
                'best_step_idx': success_metrics['best_step_idx'],
                'overall_success_rate': success_metrics['overall_success_rate']
            }
        }


class W3AttackRunner(BaseAttackRunner):
    """Runner for W3 attacks (surrogate extractor)."""
    
    def run_attack(
        self,
        X_images: torch.Tensor,
        y_labels: torch.Tensor,
        fingerprint_method: str,
        true_attribution_model: torch.nn.Module,
        attack_params: Dict[str, Any],
        attack_batch_size: int = 10,
        real_data_path: Optional[str] = None,
        cache_dir: str = "cache",
        data_dir: Optional[str] = None,
        surrogate_extractor: Optional[torch.nn.Module] = None
    ) -> Dict[str, Any]:
        """Run W3 attack using trained surrogate extractor with multiple step sizes."""
        
        print("\n--- W3 Attack (Multi-Step-Size) ---")
        
        # Create fingerprint extractor for true model
        fingerprint_extractor = FingerprintExtractorFactory.create(
            fingerprint_method,
            self.device,
            real_data_path=real_data_path,
            cache_dir=cache_dir,
            data_dir=data_dir
        )
        is_implicit = fingerprint_extractor.is_implicit_fingerprint
        
        # W3 attacks are not supported for implicit fingerprint methods
        if is_implicit:
            raise ValueError(f"W3 attack is not supported for implicit fingerprint methods like {fingerprint_method}. "
                           f"Use W1 attacks instead for implicit methods.")
        
        # Get original predictions first
        with torch.no_grad():
            # For explicit methods, extract fingerprints first
            orig_fingerprints = fingerprint_extractor.extract_fingerprint(X_images)
            if orig_fingerprints.dim() > 2:
                orig_fingerprints = orig_fingerprints.view(orig_fingerprints.shape[0], -1)
            orig_logits = true_attribution_model(orig_fingerprints.to(self.device))
            _, original_preds = torch.max(orig_logits, 1)
        
        # Define correct_mask once for all step sizes
        y_labels = y_labels.to(original_preds.device)
        correct_mask = (original_preds == y_labels)
        
        # Get step sizes to test
        step_sizes = attack_params.get('step_size_list', [attack_params['step_size']])
        print(f"Testing {len(step_sizes)} step sizes: {step_sizes}")
        
        # Create W3 attacker
        attack_type = attack_params['attack_goal']
        targeted = False
        
        print(f"Attack goal: {attack_type}")
        
        # Store results for each step size
        all_adversarial_images = []
        all_success_masks = []
        
        try:
            from src.attacks.w3_surrogate_extractor import Attacker_W3
        except ImportError as e:
            raise ImportError(f"Failed to import W3 attack modules: {e}")
        
        for i, step_size in enumerate(step_sizes):
            print(f"\n--- W3 Attack Step Size {i+1}/{len(step_sizes)}: {step_size} ---")
            
            w3_attacker = Attacker_W3(
                surrogate_extractor=surrogate_extractor,
                attribution_model=true_attribution_model,
                attack_type=attack_type,
                epsilon=attack_params['epsilon'],
                num_steps=attack_params['num_steps'],
                step_size=step_size,
                targeted=targeted,
                device=self.device
            )
            
            # Run attack with this step size
            use_targets = attack_params.get('generated_random_targets') if attack_type == 'forgery' else y_labels
            adversarial_images = w3_attacker.attack(
                X_images,
                targets=use_targets,
                attack_batch_size=attack_batch_size
            )
            all_adversarial_images.append(adversarial_images)
            
            # Evaluate success for this step size
            with torch.no_grad():
                # For explicit methods, extract fingerprints first
                adv_fingerprints = fingerprint_extractor.extract_fingerprint(adversarial_images)
                if adv_fingerprints.dim() > 2:
                    adv_fingerprints = adv_fingerprints.view(adv_fingerprints.shape[0], -1)
                adv_logits = true_attribution_model(adv_fingerprints.to(self.device))
                _, adv_preds = torch.max(adv_logits, 1)
                
                # Calculate success mask for this step size, but only consider S_correct
                if attack_type == 'removal':
                    # Success: prediction changed from original, only count for correctly classified
                    success_mask = (adv_preds != y_labels) & correct_mask
                else:  # forgery
                    gen_targets = attack_params.get('generated_random_targets')
                    if gen_targets is not None:
                        gen_targets = gen_targets.to(adv_preds.device)
                        if len(gen_targets) > len(adv_preds):
                            gen_targets = gen_targets[:len(adv_preds)]
                        elif len(gen_targets) < len(adv_preds):
                            pad = torch.zeros(len(adv_preds) - len(gen_targets), dtype=gen_targets.dtype, device=gen_targets.device)
                            gen_targets = torch.cat([gen_targets, pad])
                        success_mask = (adv_preds == gen_targets) & correct_mask
                    else:
                        success_mask = torch.zeros_like(correct_mask, dtype=torch.bool)
                
                all_success_masks.append(success_mask)
                # Calculate success rate only on S_correct
                success_rate = success_mask[correct_mask].float().mean().item() if correct_mask.any() else 0.0
                total_correct = int(correct_mask.sum().item())
                successes = int(success_mask.sum().item())
                print(f"  Step size {step_size}: Success rate on S_correct = {success_rate:.4f} ({successes}/{total_correct})")
                if attack_type == 'forgery' and successes > 0:
                    try:
                        succ_classes = adv_preds[success_mask]
                        counts = torch.bincount(succ_classes).cpu()
                        topk = torch.topk(counts, k=min(3, len(counts)))
                        breakdown = [f"{int(idx)}:{int(cnt)}" for cnt, idx in zip(topk.values, topk.indices)]
                        print(f"    Top target hits among successes: {', '.join(breakdown)}")
                    except Exception:
                        pass
        
        # Aggregate success across all step sizes
        # Compute attack success metrics using common function
        success_metrics = self.metrics.compute_attack_success_metrics(original_preds, y_labels, all_success_masks, step_sizes)
        
        # Get best adversarial images
        best_adversarial_images = all_adversarial_images[success_metrics['best_step_idx']]
        
        # Return both adversarial images and combined success information
        return {
            'adversarial_images': best_adversarial_images,
            'combined_success_mask': success_metrics['combined_success_mask'],
            'overall_success_rate': success_metrics['overall_success_rate'],
            'individual_success_rates': [mask[correct_mask].float().mean().item() if correct_mask.any() else 0.0 for mask in all_success_masks],
            'step_sizes': step_sizes,
            'best_step_idx': success_metrics['best_step_idx'],
            'original_predictions': original_preds,
            'combined_success_info': {
                'combined_success_mask': success_metrics['combined_success_mask'],
                'step_sizes': step_sizes,
                'individual_success_rates': [mask[correct_mask].float().mean().item() if correct_mask.any() else 0.0 for mask in all_success_masks],
                'best_step_idx': success_metrics['best_step_idx'],
                'overall_success_rate': success_metrics['overall_success_rate']
            }
        }


class B1AttackRunner(BaseAttackRunner):
    """Runner for B1 attacks (surrogate classifier)."""
    
    def run_attack(
        self,
        X_images: torch.Tensor,
        y_labels: torch.Tensor,
        fingerprint_method: str,
        true_attribution_model: torch.nn.Module,
        attack_params: Dict[str, Any],
        attack_batch_size: int = 10,
        real_data_path: Optional[str] = None,
        cache_dir: str = "cache",
        data_dir: Optional[str] = None,
        surrogate_attribution_model: Optional[torch.nn.Module] = None
    ) -> Dict[str, Any]:
        """Run B1 attack using surrogate classifier with multiple step sizes."""
        
        print("\n--- B1 Attack (Multi-Step-Size) ---")
        
        # Get original predictions from TRUE attribution model h for defining S_correct
        fingerprint_extractor = FingerprintExtractorFactory.create(
            fingerprint_method,
            self.device,
            real_data_path=real_data_path,
            cache_dir=cache_dir,
            data_dir=data_dir
        )
        is_implicit = fingerprint_extractor.is_implicit_fingerprint
        
        with torch.no_grad():
            if is_implicit:
                # For implicit methods (e.g., qian20, wang20), use h directly on images
                orig_logits = true_attribution_model(X_images.to(self.device))
                _, original_preds = torch.max(orig_logits, 1)
            else:
                # For explicit methods, extract fingerprints then classify with h
                orig_fingerprints = fingerprint_extractor.extract_fingerprint(X_images)
                if orig_fingerprints.dim() > 2:
                    orig_fingerprints = orig_fingerprints.view(orig_fingerprints.shape[0], -1)
                orig_logits = true_attribution_model(orig_fingerprints.to(self.device))
                _, original_preds = torch.max(orig_logits, 1)
        
        # Define correct_mask once for all step sizes
        y_labels = y_labels.to(original_preds.device)
        correct_mask = (original_preds == y_labels)
        
        # Get step sizes to test
        step_sizes = attack_params.get('step_size_list', [attack_params['step_size']])
        print(f"Testing {len(step_sizes)} step sizes: {step_sizes}")
        
        # Create B1 attacker
        attack_type = attack_params['attack_goal']
        targeted = False
        
        print(f"Attack goal: {attack_type}")
        
        # Store results for each step size
        all_adversarial_images = []
        all_success_masks = []
        
        try:
            from src.attacks.b1_surrogate_classifier import Attacker_B1
        except ImportError as e:
            raise ImportError(f"Failed to import B1 attack modules: {e}")
        
        for i, step_size in enumerate(step_sizes):
            print(f"\n--- B1 Attack Step Size {i+1}/{len(step_sizes)}: {step_size} ---")
            
            b1_attacker = Attacker_B1(
                surrogate_classifier=surrogate_attribution_model,
                attack_type=attack_type,
                epsilon=attack_params['epsilon'],
                num_steps=attack_params['num_steps'],
                step_size=step_size,
                targeted=targeted,
                device=self.device
            )
            
            # Run attack with this step size
            use_targets = attack_params.get('generated_random_targets') if attack_type == 'forgery' else y_labels
            adversarial_images = b1_attacker.attack(
                X_images,
                targets=use_targets,
                attack_batch_size=attack_batch_size
            )
            all_adversarial_images.append(adversarial_images)
            
            # Evaluate success for this step size using true attribution model
            with torch.no_grad():
                # B1 uses surrogate for attack but true model for evaluation
                if fingerprint_method in ['qian20', 'wang20']:  # Implicit methods
                    adv_logits = true_attribution_model(adversarial_images)
                    _, adv_preds = torch.max(adv_logits, 1)
                else:  # Explicit methods
                    fingerprint_extractor = FingerprintExtractorFactory.create(
                        fingerprint_method,
                        self.device,
                        real_data_path=real_data_path,
                        cache_dir=cache_dir,
                        data_dir=data_dir
                    )
                    adv_fingerprints = fingerprint_extractor.extract_fingerprint(adversarial_images)
                    if adv_fingerprints.dim() > 2:
                        adv_fingerprints = adv_fingerprints.view(adv_fingerprints.shape[0], -1)
                    adv_logits = true_attribution_model(adv_fingerprints.to(self.device))
                    _, adv_preds = torch.max(adv_logits, 1)
                
                # Calculate success mask for this step size, but only consider S_correct
                if attack_type == 'removal':
                    # Success: prediction changed from original, only count for correctly classified
                    success_mask = (adv_preds != y_labels) & correct_mask
                else:  # forgery
                    gen_targets = attack_params.get('generated_random_targets')
                    if gen_targets is not None:
                        gen_targets = gen_targets.to(adv_preds.device)
                        if len(gen_targets) > len(adv_preds):
                            gen_targets = gen_targets[:len(adv_preds)]
                        elif len(gen_targets) < len(adv_preds):
                            pad = torch.zeros(len(adv_preds) - len(gen_targets), dtype=gen_targets.dtype, device=gen_targets.device)
                            gen_targets = torch.cat([gen_targets, pad])
                        success_mask = (adv_preds == gen_targets) & correct_mask
                    else:
                        success_mask = torch.zeros_like(correct_mask, dtype=torch.bool)
                
                all_success_masks.append(success_mask)
                # Calculate success rate only on S_correct
                success_rate = success_mask[correct_mask].float().mean().item() if correct_mask.any() else 0.0
                total_correct = int(correct_mask.sum().item())
                successes = int(success_mask.sum().item())
                print(f"  Step size {step_size}: Success rate on S_correct = {success_rate:.4f} ({successes}/{total_correct})")
                if attack_type == 'forgery' and successes > 0:
                    try:
                        succ_classes = adv_preds[success_mask]
                        counts = torch.bincount(succ_classes).cpu()
                        topk = torch.topk(counts, k=min(3, len(counts)))
                        breakdown = [f"{int(idx)}:{int(cnt)}" for cnt, idx in zip(topk.values, topk.indices)]
                        print(f"    Top target hits among successes: {', '.join(breakdown)}")
                    except Exception:
                        pass
        
        # Aggregate success across all step sizes
        # Compute attack success metrics using common function
        success_metrics = self.metrics.compute_attack_success_metrics(original_preds, y_labels, all_success_masks, step_sizes)
        
        # Get best adversarial images
        best_adversarial_images = all_adversarial_images[success_metrics['best_step_idx']]
        
        # Return both adversarial images and combined success information
        return {
            'adversarial_images': best_adversarial_images,
            'combined_success_mask': success_metrics['combined_success_mask'],
            'overall_success_rate': success_metrics['overall_success_rate'],
            'individual_success_rates': [mask[correct_mask].float().mean().item() if correct_mask.any() else 0.0 for mask in all_success_masks],
            'step_sizes': step_sizes,
            'best_step_idx': success_metrics['best_step_idx'],
            'original_predictions': original_preds,
            'combined_success_info': {
                'combined_success_mask': success_metrics['combined_success_mask'],
                'step_sizes': step_sizes,
                'individual_success_rates': [mask[correct_mask].float().mean().item() if correct_mask.any() else 0.0 for mask in all_success_masks],
                'best_step_idx': success_metrics['best_step_idx'],
                'overall_success_rate': success_metrics['overall_success_rate']
            }
        }


class B2AttackRunner(BaseAttackRunner):
    """Runner for B2 attacks (image perturbations)."""
    
    def run_attack(
        self,
        X_images: torch.Tensor,
        y_labels: torch.Tensor,
        fingerprint_method: str,
        true_attribution_model: torch.nn.Module,
        attack_params: Dict[str, Any],
        attack_batch_size: int = 10,
        real_data_path: Optional[str] = None,
        cache_dir: str = "cache",
        data_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run B2 attack using basic image perturbations."""
        
        print("\n--- B2 Attack ---")
        
        # Create fingerprint extractor for true model
        fingerprint_extractor = FingerprintExtractorFactory.create(
            fingerprint_method, self.device, real_data_path, cache_dir, data_dir
        )
        is_implicit = fingerprint_extractor.is_implicit_fingerprint
        
        # Get original predictions first
        with torch.no_grad():
            if is_implicit:
                # For implicit methods, directly use the model on images
                orig_logits = true_attribution_model(X_images.to(self.device))
                _, original_preds = torch.max(orig_logits, 1)
            else:
                # For explicit methods, extract fingerprints first
                orig_fingerprints = fingerprint_extractor.extract_fingerprint(X_images)
                if orig_fingerprints.dim() > 2:
                    orig_fingerprints = orig_fingerprints.view(orig_fingerprints.shape[0], -1)
                orig_logits = true_attribution_model(orig_fingerprints.to(self.device))
                _, original_preds = torch.max(orig_logits, 1)
        
        # Define correct_mask - ensure tensors on same device
        y_labels = y_labels.to(original_preds.device)
        correct_mask = (original_preds == y_labels)
        
        # Create B2 attacker
        attack_type = attack_params['attack_goal']
        targeted = attack_params['attack_goal'] == 'forgery'
        
        print(f"Attack goal: {attack_type}")
        
        try:
            from src.attacks.b2_image_perturbations import Attacker_B2
        except ImportError as e:
            raise ImportError(f"Failed to import B2 attack modules: {e}")
        
        # Create B2 attacker (no auto-tuning; use fixed perturbation params)
        perturbation_types = attack_params.get('perturbation_types', ['gaussian_noise', 'blur', 'jpeg', 'resize'])
        b2_attacker = Attacker_B2(
            attack_type=attack_type,
            epsilon_linf=attack_params.get('epsilon_linf', 0.025),
            perturbation_types=perturbation_types,
            device=self.device
        )
        
        # Fixed perturbation parameters (overwrite with user-provided values if any)
        perturbation_params = {
            'gaussian_noise': {'std': 0.005},
            'blur': {'sigma': 0.5, 'kernel_size': 3},
            'jpeg': {'quality': 95},
            'resize': {'scale_factor': 0.9},
        }
        user_params = attack_params.get('perturbation_params')
        if isinstance(user_params, dict):
            # shallow merge: user overrides per-perturbation dict
            for k, v in user_params.items():
                perturbation_params[k] = v
        
        adversarial_results = b2_attacker.attack(
            X_images, 
                targets=y_labels, 
            attack_batch_size=attack_batch_size,
            perturbation_params=perturbation_params
        )
        
        # B2 returns individual perturbation results
        individual_results = adversarial_results['individual']
        all_attack_results = {}
        
        # Evaluate each perturbation type
        for pert_type, perturbed_images in individual_results.items():
            # Get predictions for this perturbation
            with torch.no_grad():
                if is_implicit:
                    adv_logits = true_attribution_model(perturbed_images)
                    _, adv_preds = torch.max(adv_logits, 1)
                else:
                    adv_fingerprints = fingerprint_extractor.extract_fingerprint(perturbed_images)
                    if adv_fingerprints.dim() > 2:
                        adv_fingerprints = adv_fingerprints.view(adv_fingerprints.shape[0], -1)
                    adv_logits = true_attribution_model(adv_fingerprints.to(self.device))
                    _, adv_preds = torch.max(adv_logits, 1)
                
                # Calculate success rate
                if attack_type == 'removal':
                    success_mask = (adv_preds != y_labels) & correct_mask
                else:  # forgery
                    # Success definition handled centrally; B2 uses runner/evaluator pairing
                    success_mask = torch.zeros_like(correct_mask, dtype=torch.bool)
                
                success_rate = success_mask[correct_mask].float().mean().item() if correct_mask.any() else 0.0
                print(f"  {pert_type}: Success rate on S_correct = {success_rate:.4f}")

                # Compute Avg L_inf over successful samples (relative to original images)
                try:
                    diffs = (perturbed_images - X_images.to(perturbed_images.device)).abs().amax(dim=(1,2,3))
                    succ_count = int(success_mask.sum().item())
                    if succ_count > 0:
                        avg_linf_successful = diffs[success_mask].mean().item()
                        print(f"    {pert_type}: computed avg_linf_successful = {avg_linf_successful:.6f} over {succ_count} successes")
                    else:
                        avg_linf_successful = float('nan')
                        print(f"    {pert_type}: no successes, avg_linf_successful = NaN")
                except Exception as e:
                    avg_linf_successful = float('nan')
                    print(f"    {pert_type}: exception computing avg_linf_successful: {e}")
                
                # Store results for this perturbation type
                all_attack_results[pert_type] = {
                    'adversarial_images': perturbed_images,
                    'success_mask': success_mask,
                    'success_rate': success_rate,
                    'predictions': adv_preds,
                    'avg_linf_successful': avg_linf_successful
                }
        
        # Compute combined success across perturbations (ANY success on S_correct)
        perturbation_types = list(all_attack_results.keys())
        success_masks = [all_attack_results[p]['success_mask'] for p in perturbation_types]
        if len(success_masks) > 0:
            combined_success_mask = success_masks[0].clone()
            for m in success_masks[1:]:
                combined_success_mask = combined_success_mask | m
        else:
            combined_success_mask = torch.zeros_like(correct_mask, dtype=torch.bool)

        # Individual success rates on S_correct
        individual_success_rates = []
        for p in perturbation_types:
            sm = all_attack_results[p]['success_mask']
            sr = sm[correct_mask].float().mean().item() if correct_mask.any() else 0.0
            individual_success_rates.append(sr)

        # Best perturbation idx (for fallback image selection)
        best_idx = int(max(range(len(individual_success_rates)), key=lambda i: individual_success_rates[i])) if individual_success_rates else 0
        best_type = perturbation_types[best_idx] if perturbation_types else None

        # Build per-image final adversarial images: prefer any successful perturbation; otherwise fallback to best_type
        # Stack perturbations for easy indexing
        stacked = torch.stack([all_attack_results[p]['adversarial_images'] for p in perturbation_types], dim=0) if perturbation_types else None
        B = X_images.shape[0]
        if stacked is not None and best_type is not None:
            final_adv_images = stacked[best_idx].clone()
            for i in range(B):
                chosen = None
                for p_idx, p in enumerate(perturbation_types):
                    if bool(all_attack_results[p]['success_mask'][i].item()):
                        chosen = p_idx
                        break
                if chosen is not None:
                    final_adv_images[i] = stacked[chosen, i]
        else:
            # Fallback: if no perturbations, keep originals
            final_adv_images = X_images.clone()

        # Return results with combined success info for evaluator to compute overall ASR (union)
        return {
            'adversarial_images': final_adv_images.to(self.device),
            'individual_results': {
                'perturbation_results': {
                    pert_type: {
                        'success_rate': results['success_rate'],
                        'adversarial_images': results['adversarial_images'].to(self.device),
                        'predictions': results['predictions'].to(self.device),
                        'success_mask': results['success_mask'].to(self.device),
                        'avg_linf_successful': results.get('avg_linf_successful', float('nan'))
                    }
                    for pert_type, results in all_attack_results.items()
                }
            },
            'combined_success_info': {
                'combined_success_mask': combined_success_mask.to(self.device),
                'step_sizes': perturbation_types,  # using perturbation names for display
                'individual_success_rates': individual_success_rates,
                'best_step_idx': best_idx,
                'overall_success_rate': combined_success_mask[correct_mask].float().mean().item() if correct_mask.any() else 0.0
            },
            'original_predictions': original_preds.to(self.device)
        }

    def _tune_b2_perturbation_params(
        self,
        images: torch.Tensor,
        epsilon_linf: float,
        perturbation_types: List[str],
        device: str = 'cpu',
        max_iters: int = 20,
        tol_ratio: float = 0.1
    ) -> Dict[str, Dict[str, Any]]:
        """Tune B2 perturbation parameters to match target epsilon_linf on average.
        Returns a dict mapping perturbation type -> params.
        """
        from src.attacks.b2_image_perturbations import Attacker_B2
        tuned: Dict[str, Dict[str, Any]] = {}
        # Create a helper attacker to reuse perturbation implementations
        helper = Attacker_B2(
            attack_type='removal', epsilon_linf=epsilon_linf,
            perturbation_types=perturbation_types, device=device
        )
        # Use at most 50 random images for tuning for speed
        images = images.to(device)
        num_images = images.shape[0]
        max_tune = min(50, num_images)
        if num_images > max_tune:
            idx = torch.randperm(num_images, device=device)[:max_tune]
            images = images.index_select(0, idx)
        with torch.no_grad():
            for p in perturbation_types:
                if p == 'gaussian_noise':
                    # Parameter: std
                    std_min, std_max = 1e-4, 0.5
                    std = max(epsilon_linf / 2.0, std_min)
                    print(f"    [tune] gaussian_noise: target={epsilon_linf:.6f}, init std={std:.6f}")
                    for _ in range(max_iters):
                        out = helper._gaussian_noise(images, std=std)
                        linf = (out - images).abs().amax(dim=(1,2,3)).mean().item()
                        rel_err = abs(linf - epsilon_linf) / max(epsilon_linf, 1e-12)
                        print(f"      iter: std={std:.6f}, mean_linf={linf:.6f}, rel_err={rel_err:.2%}")
                        if abs(linf - epsilon_linf) <= tol_ratio * epsilon_linf:
                            break
                        # multiplicative adjustment
                        if linf > 0:
                            scale = epsilon_linf / (linf + 1e-12)
                            std = std * scale
                        else:
                            std = std * 1.5
                        std = float(min(max(std, std_min), std_max))
                    print(f"      tuned std={std:.6f}")
                    tuned[p] = {'std': std}
                elif p == 'blur':
                    # Parameter: sigma (kernel_size fixed small odd number)
                    sigma_min, sigma_max = 0.2, 5.0
                    sigma = 0.8
                    kernel_size = 3
                    print(f"    [tune] blur: target={epsilon_linf:.6f}, init sigma={sigma:.6f}, k={kernel_size}")
                    for _ in range(max_iters):
                        out = helper._gaussian_blur(images, kernel_size=kernel_size, sigma=sigma)
                        linf = (out - images).abs().amax(dim=(1,2,3)).mean().item()
                        rel_err = abs(linf - epsilon_linf) / max(epsilon_linf, 1e-12)
                        print(f"      iter: sigma={sigma:.6f}, mean_linf={linf:.6f}, rel_err={rel_err:.2%}")
                        if abs(linf - epsilon_linf) <= tol_ratio * epsilon_linf:
                            break
                        # proportional update: increase sigma if linf < target; decrease if linf > target
                        if linf < epsilon_linf:
                            sigma = sigma * 1.25
                        else:
                            sigma = sigma * 0.75
                        sigma = float(min(max(sigma, sigma_min), sigma_max))
                    print(f"      tuned sigma={sigma:.6f}, k={kernel_size}")
                    tuned[p] = {'kernel_size': kernel_size, 'sigma': sigma}
                elif p == 'jpeg':
                    # Parameter: quality in [5, 99]; lower quality => higher distortion
                    q_low, q_high = 5, 99
                    # Binary search on quality
                    best_q, best_diff = 75, float('inf')
                    print(f"    [tune] jpeg: target={epsilon_linf:.6f}, search quality in [{q_low},{q_high}]")
                    for _ in range(max_iters):
                        q_mid = (q_low + q_high) // 2
                        out = helper._jpeg_compression(images, quality=int(q_mid))
                        linf = (out - images).abs().amax(dim=(1,2,3)).mean().item()
                        diff = abs(linf - epsilon_linf)
                        rel_err = diff / max(epsilon_linf, 1e-12)
                        print(f"      iter: q_mid={int(q_mid)}, mean_linf={linf:.6f}, rel_err={rel_err:.2%}, range=[{q_low},{q_high}]")
                        if diff < best_diff:
                            best_diff = diff
                            best_q = int(q_mid)
                        if linf > epsilon_linf:
                            # too strong => increase quality
                            q_low = max(q_low, int(q_mid) + 1)
                        else:
                            # too weak => decrease quality
                            q_high = min(q_high, int(q_mid) - 1)
                        if diff <= tol_ratio * epsilon_linf:
                            break
                        if q_low > q_high:
                            break
                    print(f"      tuned quality={best_q}, best_rel_err={(best_diff / max(epsilon_linf, 1e-12)):.2%}")
                    tuned[p] = {'quality': int(best_q)}
                elif p == 'resize':
                    # Parameter: scale_factor in (0,1]; smaller => stronger distortion
                    s_min, s_max = 0.5, 0.9999
                    low, high = s_min, s_max
                    best_s, best_diff = 0.75, float('inf')
                    print(f"    [tune] resize: target={epsilon_linf:.6f}, search scale in [{s_min:.2f},{s_max:.2f}]")
                    for _ in range(max_iters):
                        s_mid = (low + high) / 2.0
                        out = helper._resize_perturbation(images, scale_factor=float(s_mid))
                        linf = (out - images).abs().amax(dim=(1,2,3)).mean().item()
                        diff = abs(linf - epsilon_linf)
                        rel_err = diff / max(epsilon_linf, 1e-12)
                        print(f"      iter: s_mid={s_mid:.4f}, mean_linf={linf:.6f}, rel_err={rel_err:.2%}, range=[{low:.4f},{high:.4f}]")
                        if diff < best_diff:
                            best_diff = diff
                            best_s = float(s_mid)
                        if linf > epsilon_linf:
                            # too strong => increase scale (less downsample)
                            low = max(low, s_mid)
                        else:
                            # too weak => decrease scale
                            high = min(high, s_mid)
                        if diff <= tol_ratio * epsilon_linf:
                            break
                        if (high - low) < 1e-3:
                            break
                    print(f"      tuned scale_factor={best_s:.6f}, best_rel_err={(best_diff / max(epsilon_linf, 1e-12)):.2%}")
                    tuned[p] = {'scale_factor': float(best_s)}
                else:
                    # Unknown perturbation, skip
                    continue
        return tuned


class CombinedFingerprintModel(nn.Module):
    """
    Combined model that includes both fingerprint extractor and attribution model.
    Used for W1 attacks on differentiable explicit fingerprint methods.
    """
    
    def __init__(self, fingerprint_extractor, attribution_model):
        super().__init__()
        self.fingerprint_extractor = fingerprint_extractor
        self.attribution_model = attribution_model
    
    def forward(self, images):
        # Extract fingerprints
        fingerprints = self.fingerprint_extractor.extract_fingerprint(images)
        # Pass through attribution model
        logits = self.attribution_model(fingerprints)
        return logits
