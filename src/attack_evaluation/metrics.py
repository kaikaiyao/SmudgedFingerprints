"""
Attack metrics computation and analysis.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import lpips


class AttackMetrics:
    """Computes and stores attack effectiveness metrics."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.lpips_model = None
        
    def _get_lpips_model(self):
        """Lazy load LPIPS model."""
        if self.lpips_model is None:
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
        return self.lpips_model
    
    def compute_attack_success_metrics(
        self,
        original_preds: torch.Tensor,
        y_labels: torch.Tensor,
        all_success_masks: List[torch.Tensor],
        step_sizes: List[float]
    ) -> Dict[str, Any]:
        """Compute multi-step-size attack success metrics and selection.

        Args:
            original_preds: Predictions before attack
            y_labels: True labels
            all_success_masks: List of boolean tensors, one per step size
            step_sizes: List of step sizes corresponding to the masks

        Returns:
            Dict with keys:
              - individual_success_rates: list[float]
              - combined_success_mask: torch.BoolTensor
              - overall_success_rate: float (ANY step size success on S_correct)
              - best_step_idx: int
              - step_sizes: list[float]
        """

        # Ensure inputs are aligned on device
        y_labels = y_labels.to(original_preds.device)
        correct_before = (original_preds == y_labels)

        # Compute individual success rates on S_correct
        individual_success_rates: List[float] = []
        for mask in all_success_masks:
            if correct_before.any():
                rate = mask[correct_before].float().mean().item()
            else:
                rate = 0.0
            individual_success_rates.append(rate)

        # Determine best step index (max success rate)
        if len(individual_success_rates) > 0:
            best_step_idx = int(max(range(len(individual_success_rates)), key=lambda i: individual_success_rates[i]))
        else:
            best_step_idx = 0

        # Combine successes across all step sizes (ANY success)
        if len(all_success_masks) > 0:
            combined_success_mask = all_success_masks[0].clone().detach()
            for mask in all_success_masks[1:]:
                combined_success_mask = combined_success_mask | mask
        else:
            combined_success_mask = torch.zeros_like(correct_before, dtype=torch.bool)

        # Overall success rate on S_correct for ANY step size
        if correct_before.any():
            overall_success_rate = combined_success_mask[correct_before].float().mean().item()
        else:
            overall_success_rate = 0.0

        return {
            'individual_success_rates': individual_success_rates,
            'combined_success_mask': combined_success_mask,
            'overall_success_rate': overall_success_rate,
            'best_step_idx': best_step_idx,
            'step_sizes': step_sizes,
        }
    
    def compute_image_quality_metrics(
        self, 
        original_images: torch.Tensor, 
        adversarial_images: torch.Tensor
    ) -> Dict[str, float]:
        """Compute image quality metrics between original and adversarial images."""
        
        # Ensure images are in [0, 1] range
        orig = torch.clamp(original_images, 0, 1)
        adv = torch.clamp(adversarial_images, 0, 1)
        
        # L2 distance
        l2_dist = torch.norm(orig - adv, p=2, dim=[1, 2, 3]).mean().item()
        
        # L∞ distance
        linf_dist = torch.norm(orig - adv, p=float('inf'), dim=[1, 2, 3]).mean().item()
        
        # PSNR
        mse = torch.mean((orig - adv) ** 2, dim=[1, 2, 3])
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8)).mean().item()
        
        # LPIPS
        try:
            lpips_model = self._get_lpips_model()
            lpips_dist = lpips_model(orig, adv).mean().item()
        except Exception as e:
            print(f"Warning: LPIPS computation failed: {e}")
            lpips_dist = float('nan')
        
        return {
            'l2_distance': l2_dist,
            'linf_distance': linf_dist,
            'psnr': psnr if not np.isnan(psnr) else float('nan'),
            'lpips': lpips_dist if not np.isnan(lpips_dist) else float('nan')
        }
    
    def compute_removal_metrics(
        self, 
        original_preds: torch.Tensor, 
        adversarial_preds: torch.Tensor, 
        y_labels: torch.Tensor
    ) -> Dict[str, float]:
        """Compute removal attack metrics."""
        
        # Only consider images that were correctly classified before attack
        correct_before = (original_preds == y_labels)
        
        if correct_before.sum() == 0:
            return {
                'asr_remove': 0.0,
                'total_correct_before': 0,
                'successful_removals': 0
            }
        
        # Success: originally correct prediction becomes incorrect
        successful_removals = correct_before & (adversarial_preds != y_labels)
        
        asr_remove = successful_removals.sum().float() / correct_before.sum().float()
        
        return {
            'asr_remove': asr_remove.item(),
            'total_correct_before': correct_before.sum().item(),
            'successful_removals': successful_removals.sum().item()
        }
    
    def compute_forgery_metrics(
        self, 
        original_preds: torch.Tensor, 
        adversarial_preds: torch.Tensor, 
        y_labels: torch.Tensor,
        target_predictions: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Compute forgery attack metrics."""
        
        # Only consider images that were correctly classified before attack
        correct_before = (original_preds == y_labels)
        
        if correct_before.sum() == 0:
            return {
                'tasr_rand': 0.0,
                'total_correct_before': 0,
                'successful_forgeries': 0,
                'target_type': 'random' if target_predictions is not None else 'random'
            }
        
        # Determine target predictions
        if target_predictions is not None:
            # Use provided target predictions
            targets = target_predictions
            target_type = 'random'
            # Ensure tensors are on same device and have same shape
            targets = targets.to(correct_before.device)
            if targets.shape != correct_before.shape:
                # Adjust targets shape to match correct_before
                if len(targets) > len(correct_before):
                    targets = targets[:len(correct_before)]
                else:
                    # Pad targets with zeros (will be masked out anyway)
                    targets = torch.cat([targets, torch.zeros(len(correct_before) - len(targets), dtype=targets.dtype, device=targets.device)])
            target_distribution = torch.bincount(targets[correct_before]).tolist()
        else:
            # Default to random targets (different from original)
            targets = y_labels.clone()
            while (targets == y_labels).any():
                mask = targets == y_labels
                targets[mask] = torch.randint(0, y_labels.max() + 1, (mask.sum(),), device=targets.device)
            target_type = 'random'
            # Ensure tensors are on same device and have same shape
            targets = targets.to(correct_before.device)
            if targets.shape != correct_before.shape:
                # Adjust targets shape to match correct_before
                if len(targets) > len(correct_before):
                    targets = targets[:len(correct_before)]
                else:
                    # Pad targets with zeros (will be masked out anyway)
                    targets = torch.cat([targets, torch.zeros(len(correct_before) - len(targets), dtype=targets.dtype, device=targets.device)])
            target_distribution = torch.bincount(targets[correct_before]).tolist()
        
        # Success: adversarial prediction matches target
        successful_forgeries = correct_before & (adversarial_preds == targets)
        
        tasr_rand = successful_forgeries.sum().float() / correct_before.sum().float()
        
        result = {
            'tasr_rand': tasr_rand.item(),
            'total_correct_before': correct_before.sum().item(),
            'successful_forgeries': successful_forgeries.sum().item(),
            'target_type': target_type
        }
        
        if target_distribution is not None:
            result['target_distribution'] = target_distribution
        
        return result
    
    def format_metric(self, value: float, format_type: str = 'default') -> str:
        """Format metric values for display."""
        if format_type == 'percentage':
            return f"{value:.4f}"
        elif format_type == 'distance':
            return f"{value:.6f}"
        elif format_type == 'psnr':
            return f"{value:.2f}" if not np.isnan(value) else "N/A"
        elif format_type == 'lpips':
            return f"{value:.4f}" if not np.isnan(value) else "N/A"
        else:
            return f"{value:.6f}"
