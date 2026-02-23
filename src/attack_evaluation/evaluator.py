"""
Main attack evaluation class that orchestrates the evaluation process.
"""

import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import lpips

from .metrics import AttackMetrics
from .target_selection import TargetSelector


class AttackEvaluator:
    """Main class for evaluating attack effectiveness."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.metrics = AttackMetrics(device)
        self.target_selector = TargetSelector()
    
    def evaluate_attack_effectiveness(
        self,
        original_images: torch.Tensor,
        adversarial_images: torch.Tensor,
        original_predictions: torch.Tensor,
        adversarial_predictions: torch.Tensor,
        y_labels: torch.Tensor,
        attack_type: str,
        attack_params: Dict[str, Any],
        fingerprint_method: str,
        true_attribution_model,
        attack_batch_size: int = 10,
        individual_results: Optional[Dict] = None,
        cumulative_results: Optional[Dict] = None,
        real_data_path: Optional[str] = None,
        cache_dir: str = "cache",
        data_dir: Optional[str] = None,
        model_names: Optional[List[str]] = None,
        combined_success_info: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of an attack.
        
        Args:
            original_images: Original images
            adversarial_images: Adversarial images
            original_predictions: Original model predictions
            adversarial_predictions: Adversarial model predictions
            y_labels: True labels
            attack_type: Type of attack (w1, w2, w3, b1, b2)
            attack_params: Attack parameters
            fingerprint_method: Fingerprint method name
            true_attribution_model: True attribution model
            attack_batch_size: Batch size for evaluation
            individual_results: Individual step results (for multi-step attacks)
            cumulative_results: Cumulative results (for multi-step attacks)
            real_data_path: Path to real data (for some methods)
            cache_dir: Cache directory
            data_dir: Data directory
            model_names: List of model names
            combined_success_info: Combined success info (for multi-step attacks)
            
        Returns:
            Dictionary containing evaluation results
        """
        
        print("\n--- Attack Effectiveness Evaluation ---")
        
        attack_goal = attack_params.get('attack_goal')
        
        # Calculate number of models from unique labels and ensure tensors on same device
        y_labels = y_labels.to(original_predictions.device)
        num_models = len(torch.unique(y_labels))
        
        # Define S_correct: images correctly classified before attack
        S_correct_mask = (original_predictions == y_labels)
        S_correct_indices = torch.where(S_correct_mask)[0]
        
        print(f"📊 EVALUATION SETUP:")
        print(f"   Total images: {len(original_images)}")
        print(f"   S_correct (correctly classified before attack): {S_correct_mask.sum().item()}")
        print(f"   S_incorrect (incorrectly classified before attack): {(~S_correct_mask).sum().item()}")
        
        if S_correct_mask.sum() == 0:
            print("❌ WARNING: No images were correctly classified before attack!")
            print("   Evaluation will be performed on all images, but results may be misleading.")
            S_correct_mask = torch.ones_like(y_labels, dtype=torch.bool)
            S_correct_indices = torch.arange(len(original_images))
        
        # Move tensors to device and extract S_correct data
        original_images = original_images.to(self.device)
        adversarial_images = adversarial_images.to(self.device)
        S_correct_original_preds = original_predictions[S_correct_mask].to(self.device)
        S_correct_y_labels = y_labels[S_correct_mask].to(self.device)
        S_correct_original_images = original_images[S_correct_mask].to(self.device)
        S_correct_adversarial_images = adversarial_images[S_correct_mask].to(self.device)
        
        # Handle multi-step-size evaluation
        if combined_success_info is not None:
            return self._evaluate_with_combined_success(
                S_correct_mask, S_correct_original_preds, S_correct_y_labels,
                S_correct_original_images, S_correct_adversarial_images,
                adversarial_predictions, attack_type, attack_params,
                fingerprint_method, true_attribution_model, attack_batch_size,
                real_data_path, cache_dir, data_dir, model_names,
                combined_success_info, num_models, individual_results
            )
        else:
            return self._evaluate_single_step(
                S_correct_mask, S_correct_original_preds, S_correct_y_labels,
                S_correct_original_images, S_correct_adversarial_images,
                adversarial_predictions, attack_type, attack_params,
                fingerprint_method, true_attribution_model, attack_batch_size,
                real_data_path, cache_dir, data_dir, model_names, num_models,
                individual_results
            )
    
    def _evaluate_with_combined_success(
        self,
        S_correct_mask: torch.Tensor,
        S_correct_original_preds: torch.Tensor,
        S_correct_y_labels: torch.Tensor,
        S_correct_original_images: torch.Tensor,
        S_correct_adversarial_images: torch.Tensor,
        adversarial_predictions: torch.Tensor,
        attack_type: str,
        attack_params: Dict[str, Any],
        fingerprint_method: str,
        true_attribution_model,
        attack_batch_size: int,
        real_data_path: Optional[str],
        cache_dir: str,
        data_dir: Optional[str],
        model_names: Optional[List[str]],
        combined_success_info: Dict,
        num_models: int,
        individual_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate with combined success across multiple step sizes."""
        
        print(f"   Evaluating on S_correct: {S_correct_mask.sum().item()} images")
        print(f"   Multi-step-size evaluation: {len(combined_success_info['step_sizes'])} step sizes tested")
        print(f"   Step sizes: {combined_success_info['step_sizes']}")
        print(f"   Individual step size success rates: {combined_success_info['individual_success_rates']}")
        # Fallback if overall_success_rate not passed
        overall_sr = combined_success_info.get('overall_success_rate')
        if overall_sr is None:
            comb_mask = combined_success_info['combined_success_mask']
            # Recompute on S_correct
            overall_sr = comb_mask.float().mean().item()
        print(f"   Combined success rate (any step size): {overall_sr:.4f}")
        
        # Use combined success mask from multi-step-size evaluation
        S_correct_combined_success_mask = combined_success_info['combined_success_mask'][S_correct_mask]
        
        attack_goal = attack_params.get('attack_goal')
        
        # Calculate metrics based on combined success across step sizes
        if attack_goal == 'removal':
            removal_metrics = self._compute_removal_metrics_combined(
                S_correct_combined_success_mask, S_correct_mask, combined_success_info
            )
            forgery_metrics = None
            attack_success_metric = removal_metrics['asr_remove']
        else:  # forgery
            forgery_metrics = self._compute_forgery_metrics_combined(
                S_correct_combined_success_mask, S_correct_mask, S_correct_original_preds,
                S_correct_y_labels, S_correct_adversarial_images, adversarial_predictions,
                attack_params, fingerprint_method, true_attribution_model, attack_batch_size,
                real_data_path, cache_dir, data_dir, num_models, combined_success_info
            )
            removal_metrics = None
            attack_success_metric = forgery_metrics['tasr_rand']
        
        # Compute image quality metrics
        quality_metrics = self.metrics.compute_image_quality_metrics(
            S_correct_original_images, S_correct_adversarial_images
        )
        
        # Model-specific analysis
        model_metrics = self._compute_model_specific_metrics(
            S_correct_y_labels, S_correct_original_preds, adversarial_predictions,
            S_correct_mask, S_correct_combined_success_mask, attack_goal,
            attack_params, model_names
        )

        # Target model forgery vulnerability (per-target success over assigned)
        target_vulnerability = None
        if attack_goal == 'forgery':
            target_vulnerability = self._compute_target_model_vulnerability(
                S_correct_mask, adversarial_predictions, S_correct_y_labels,
                attack_params, num_models, model_names, S_correct_combined_success_mask
            )
        
        # Compile results
        results = {
            'attack_type': attack_type,
            'attack_parameters': attack_params,
            'overall_metrics': {
                'attack_success_rate': attack_success_metric,
                'attack_success_description': f"{'ASR_remove' if attack_goal == 'removal' else 'tASR_rand'} (higher = more effective, ANY step size success)",
                'l2_distance': quality_metrics['l2_distance'],
                'linf_distance': quality_metrics['linf_distance'],
                'psnr': quality_metrics['psnr'],
                'lpips': quality_metrics['lpips'],
                'scorrect_size': S_correct_mask.sum().item(),
                'total_images': len(S_correct_original_images),
                'multi_step_size_evaluation': True,
                'step_sizes_tested': combined_success_info['step_sizes'],
                'individual_step_success_rates': combined_success_info['individual_success_rates'],
                'best_step_size_idx': combined_success_info['best_step_idx']
            },
            'model_specific_metrics': model_metrics,
            'evaluation_timestamp': datetime.now().isoformat()
        }

        # Add per-perturbation metrics for B2 if available
        if attack_type.lower() == 'b2' and individual_results and 'perturbation_results' in individual_results:
            perturbation_metrics = {}
            for pert_type, pert_results in individual_results.get('perturbation_results', {}).items():
                pert_preds = pert_results['predictions']
                if attack_goal == 'removal':
                    metrics = self.metrics.compute_removal_metrics(
                        S_correct_original_preds, pert_preds[S_correct_mask], S_correct_y_labels
                    )
                    attack_sr = metrics['asr_remove']
                    perturbation_metrics[pert_type] = {
                        'removal_metrics': metrics,
                        'attack_success_rate': attack_sr,
                        'avg_linf_successful': pert_results.get('avg_linf_successful', float('nan'))
                    }
                else:
                    metrics = self._compute_forgery_metrics_single_step(
                        S_correct_original_preds, pert_preds[S_correct_mask],
                        S_correct_y_labels, attack_params, num_models
                    )
                    attack_sr = metrics['tasr_rand']
                    perturbation_metrics[pert_type] = {
                        'forgery_metrics': metrics,
                        'attack_success_rate': attack_sr,
                        'avg_linf_successful': pert_results.get('avg_linf_successful', float('nan'))
                    }
            results['perturbation_metrics'] = perturbation_metrics
        
        # Add attack-specific metrics
        if attack_goal == 'removal':
            results['removal_metrics'] = removal_metrics
        elif attack_goal == 'forgery':
            results['forgery_metrics'] = forgery_metrics
            if target_vulnerability is not None:
                results['target_model_vulnerability'] = target_vulnerability
        
        return results
    
    def _evaluate_single_step(
        self,
        S_correct_mask: torch.Tensor,
        S_correct_original_preds: torch.Tensor,
        S_correct_y_labels: torch.Tensor,
        S_correct_original_images: torch.Tensor,
        S_correct_adversarial_images: torch.Tensor,
        adversarial_predictions: torch.Tensor,
        attack_type: str,
        attack_params: Dict[str, Any],
        fingerprint_method: str,
        true_attribution_model,
        attack_batch_size: int,
        real_data_path: Optional[str],
        cache_dir: str,
        data_dir: Optional[str],
        model_names: Optional[List[str]],
        num_models: int,
        individual_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate single-step attack."""
        
        print(f"   Evaluating on S_correct: {S_correct_mask.sum().item()} images")
        
        attack_goal = attack_params.get('attack_goal')
        
        # Initialize results dictionary
        results = {}
        
        # Calculate attack-specific metrics
        if attack_type.lower() == 'b2' and individual_results and 'perturbation_results' in individual_results:
            # Handle B2 attack's multiple perturbation types
            perturbation_metrics = {}
            for pert_type, pert_results in individual_results.get('perturbation_results', {}).items():
                pert_preds = pert_results['predictions']
                pert_images = pert_results['adversarial_images']
                # Restrict to S_correct for images
                pert_images_sc = pert_images[S_correct_mask].to(self.device)
                # Per-image Linf on S_correct
                linf_per_image = (S_correct_original_images - pert_images_sc).abs().amax(dim=(1,2,3))
                if attack_goal == 'removal':
                    metrics = self.metrics.compute_removal_metrics(
                        S_correct_original_preds, pert_preds[S_correct_mask], S_correct_y_labels
                    )
                    # Success mask for removal on S_correct (prefer runner-provided mask; union with derived)
                    derived_success_sc = (pert_preds[S_correct_mask] != S_correct_y_labels)
                    if 'success_mask' in pert_results and pert_results['success_mask'] is not None:
                        provided_success_sc = pert_results['success_mask'][S_correct_mask]
                        success_mask_sc = provided_success_sc | derived_success_sc
                    else:
                        success_mask_sc = derived_success_sc
                    # Ensure mask is boolean and on same device as linf_per_image
                    success_mask_sc = success_mask_sc.to(dtype=torch.bool, device=linf_per_image.device)
                    if int(success_mask_sc.sum().item()) > 0:
                        avg_linf_success = linf_per_image[success_mask_sc].mean().item()
                    else:
                        avg_linf_success = float('nan')
                    perturbation_metrics[pert_type] = {
                        'removal_metrics': metrics,
                        'attack_success_rate': metrics['asr_remove'],
                        'avg_linf_successful': avg_linf_success
                    }
                else:  # forgery
                    metrics = self._compute_forgery_metrics_single_step(
                        S_correct_original_preds, pert_preds[S_correct_mask],
                        S_correct_y_labels, attack_params, num_models
                    )
                    # Success mask for forgery on S_correct (prefer runner-provided mask; union with derived)
                    gen_targets = attack_params.get('generated_random_targets')
                    if gen_targets is not None:
                        gen_targets = gen_targets.to(pert_preds.device)
                        if len(gen_targets) > len(S_correct_mask):
                            gen_targets = gen_targets[:len(S_correct_mask)]
                        elif len(gen_targets) < len(S_correct_mask):
                            padding = torch.zeros(len(S_correct_mask) - len(gen_targets), dtype=gen_targets.dtype, device=gen_targets.device)
                            gen_targets = torch.cat([gen_targets, padding])
                        derived_success_sc = (pert_preds[S_correct_mask] == gen_targets)
                    else:
                        derived_success_sc = torch.zeros_like(S_correct_y_labels, dtype=torch.bool)
                    if 'success_mask' in pert_results and pert_results['success_mask'] is not None:
                        provided_success_sc = pert_results['success_mask'][S_correct_mask]
                        success_mask_sc = provided_success_sc | derived_success_sc
                    else:
                        success_mask_sc = derived_success_sc
                    success_mask_sc = success_mask_sc.to(dtype=torch.bool, device=linf_per_image.device)
                    if int(success_mask_sc.sum().item()) > 0:
                        avg_linf_success = linf_per_image[success_mask_sc].mean().item()
                    else:
                        avg_linf_success = float('nan')
                    perturbation_metrics[pert_type] = {
                        'forgery_metrics': metrics,
                        'attack_success_rate': metrics['tasr_rand'],
                        'avg_linf_successful': avg_linf_success
                    }
            
            # Use best perturbation's metrics for overall results
            best_type = max(perturbation_metrics.keys(), 
                          key=lambda k: perturbation_metrics[k]['attack_success_rate'])
            if attack_goal == 'removal':
                removal_metrics = perturbation_metrics[best_type]['removal_metrics']
                forgery_metrics = None
                attack_success_metric = removal_metrics['asr_remove']
            else:  # forgery
                forgery_metrics = perturbation_metrics[best_type]['forgery_metrics']
                removal_metrics = None
                attack_success_metric = forgery_metrics['tasr_rand']
            
            # Store per-perturbation metrics
            results['perturbation_metrics'] = perturbation_metrics
        else:
            # Handle other attacks normally
            if attack_goal == 'removal':
                removal_metrics = self.metrics.compute_removal_metrics(
                    S_correct_original_preds, adversarial_predictions[S_correct_mask], S_correct_y_labels
                )
                forgery_metrics = None
                attack_success_metric = removal_metrics['asr_remove']
            else:  # forgery
                forgery_metrics = self._compute_forgery_metrics_single_step(
                    S_correct_original_preds, adversarial_predictions[S_correct_mask],
                    S_correct_y_labels, attack_params, num_models
                )
                removal_metrics = None
                attack_success_metric = forgery_metrics['tasr_rand']
        
        # Compute image quality metrics
        quality_metrics = self.metrics.compute_image_quality_metrics(
            S_correct_original_images, S_correct_adversarial_images
        )
        
        # Model-specific analysis
        model_metrics = self._compute_model_specific_metrics_single_step(
            S_correct_y_labels, S_correct_original_preds, adversarial_predictions,
            S_correct_mask, attack_goal, attack_params, model_names
        )

        # Target model forgery vulnerability (per-target success over assigned)
        target_vulnerability = None
        if attack_goal == 'forgery':
            target_vulnerability = self._compute_target_model_vulnerability(
                S_correct_mask, adversarial_predictions, S_correct_y_labels,
                attack_params, num_models, model_names
            )
        
        # Compile results
        results = {
            'attack_type': attack_type,
            'attack_parameters': attack_params,
            'overall_metrics': {
                'attack_success_rate': attack_success_metric,
                'attack_success_description': f"{'ASR_remove' if attack_goal == 'removal' else 'tASR_rand'} (higher = more effective)",
                'l2_distance': quality_metrics['l2_distance'],
                'linf_distance': quality_metrics['linf_distance'],
                'psnr': quality_metrics['psnr'],
                'lpips': quality_metrics['lpips'],
                'scorrect_size': S_correct_mask.sum().item(),
                'total_images': len(S_correct_original_images),
                'multi_step_size_evaluation': False
            },
            'model_specific_metrics': model_metrics,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # Add attack-specific metrics
        if attack_goal == 'removal':
            results['removal_metrics'] = removal_metrics
        elif attack_goal == 'forgery':
            results['forgery_metrics'] = forgery_metrics
            if target_vulnerability is not None:
                results['target_model_vulnerability'] = target_vulnerability
        
        # Add perturbation metrics if available (for B2 attacks)
        if 'perturbation_metrics' in locals():
            results['perturbation_metrics'] = perturbation_metrics
        
        return results

    def _compute_target_model_vulnerability(
        self,
        S_correct_mask: torch.Tensor,
        adversarial_predictions: torch.Tensor,
        S_correct_y_labels: torch.Tensor,
        attack_params: Dict[str, Any],
        num_models: int,
        model_names: Optional[List[str]],
        S_correct_combined_success_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Compute target model forgery vulnerability per user's definition.

        For forgery: for each model t, consider images in S_correct that were assigned t
        as their target (fixed target class or per-sample random targets). Vulnerability
        for model t is successes_to_t / assigned_to_t.
        """

        # Ensure on same device
        device = adversarial_predictions.device
        S_correct_mask = S_correct_mask.to(device)

        # Determine per-sample targets over all images, then restrict to S_correct
        if 'generated_random_targets' in attack_params and attack_params['generated_random_targets'] is not None:
            # attack_params targets are for full dataset; restrict to S_correct
            full_targets = attack_params['generated_random_targets'].to(device)
            if len(full_targets) > len(S_correct_mask):
                full_targets = full_targets[:len(S_correct_mask)]
            elif len(full_targets) < len(S_correct_mask):
                padding = torch.zeros(len(S_correct_mask) - len(full_targets), dtype=full_targets.dtype, device=device)
                full_targets = torch.cat([full_targets, padding])
            targets = full_targets[S_correct_mask]
            target_type = 'random'
        else:
            # If unavailable, cannot compute per-target assignment reliably
            return None

        adv_preds_sc = adversarial_predictions[S_correct_mask]
        label_to_model_name = self._create_label_to_model_mapping(model_names, num_models)

        vulnerability = {}
        assigned_counts = torch.bincount(targets, minlength=num_models)
        success_counts = torch.zeros(num_models, dtype=torch.long, device=device)

        if S_correct_combined_success_mask is not None:
            success_source_mask = S_correct_combined_success_mask.to(device)
            for t in range(num_models):
                assigned_mask = (targets == t)
                assigned_t = int(assigned_mask.sum().item())
                if assigned_t == 0:
                    vul = 0.0
                    succ_t = 0
                else:
                    succ_t = int(success_source_mask[assigned_mask].sum().item())
                    vul = succ_t / assigned_t
                vulnerability[t] = {
                    'model_name': label_to_model_name.get(t, f'model_{t}'),
                    'assigned': assigned_t,
                    'successful': succ_t,
                    'forgery_success_rate': vul,
                }
        else:
            for t in range(num_models):
                assigned_mask = (targets == t)
                assigned_t = int(assigned_mask.sum().item())
                if assigned_t == 0:
                    vul = 0.0
                    succ_t = 0
                else:
                    succ_t = int((adv_preds_sc[assigned_mask] == t).sum().item())
                    vul = succ_t / assigned_t
                vulnerability[t] = {
                    'model_name': label_to_model_name.get(t, f'model_{t}'),
                    'assigned': assigned_t,
                    'successful': succ_t,
                    'forgery_success_rate': vul,
                }
        
        # Print concise vulnerability table
        print("\n📊 TARGET MODEL FORGERY VULNERABILITY (per assigned target in S_correct):")
        for t in range(num_models):
            info = vulnerability[t]
            print(f"   Target {t} ({info['model_name']}): {info['successful']}/{info['assigned']} => {info['forgery_success_rate']:.4f}")
        
        return {
            'target_type': target_type,
            'by_model': vulnerability,
        }
    
    def _compute_removal_metrics_combined(
        self,
        S_correct_combined_success_mask: torch.Tensor,
        S_correct_mask: torch.Tensor,
        combined_success_info: Dict
    ) -> Dict[str, Any]:
        """Compute removal metrics for combined multi-step evaluation."""
        
        print("\n📊 REMOVAL ATTACK EVALUATION (Multi-Step-Size):")
        print("   Goal: Modify images in S_correct such that h(x_adv) != y(x)")
        print("   Metric: Attack Success Rate (ASR_remove) - ANY step size success")
        
        # ASR_remove = number of x in S_correct where ANY step size succeeded / total number in S_correct
        ASR_remove = S_correct_combined_success_mask.float().mean().item()
        num_successful_removals = S_correct_combined_success_mask.sum().item()
        
        print(f"   Successful removals (any step size): {num_successful_removals}/{S_correct_mask.sum().item()}")
        print(f"   ASR_remove (combined): {ASR_remove:.4f}")
        
        return {
            'asr_remove': ASR_remove,
            'num_successful_removals': num_successful_removals,
            'total_in_scorrect': S_correct_mask.sum().item(),
            'removal_success_rate': ASR_remove,
            'multi_step_size_info': {
                'step_sizes': combined_success_info['step_sizes'],
                'individual_success_rates': combined_success_info['individual_success_rates'],
                'best_step_idx': combined_success_info['best_step_idx']
            }
        }
    
    def _compute_forgery_metrics_combined(
        self,
        S_correct_combined_success_mask: torch.Tensor,
        S_correct_mask: torch.Tensor,
        S_correct_original_preds: torch.Tensor,
        S_correct_y_labels: torch.Tensor,
        S_correct_adversarial_images: torch.Tensor,
        adversarial_predictions: torch.Tensor,
        attack_params: Dict[str, Any],
        fingerprint_method: str,
        true_attribution_model,
        attack_batch_size: int,
        real_data_path: Optional[str],
        cache_dir: str,
        data_dir: Optional[str],
        num_models: int,
        combined_success_info: Dict
    ) -> Dict[str, Any]:
        """Compute forgery metrics for combined multi-step evaluation."""
        
        print("\n📊 TARGETED FORGERY ATTACK EVALUATION (Multi-Step-Size):")
        print("   Goal: Modify images in S_correct to be classified as per-sample random target")
        
        # Random targets - use original assignment and combined success mask
        device = adversarial_predictions.device
        gen_targets = attack_params.get('generated_random_targets')
        if gen_targets is not None:
            gen_targets = gen_targets.to(device)
            if len(gen_targets) > len(S_correct_mask):
                gen_targets = gen_targets[:len(S_correct_mask)]
            elif len(gen_targets) < len(S_correct_mask):
                padding = torch.zeros(len(S_correct_mask) - len(gen_targets), dtype=gen_targets.dtype, device=device)
                gen_targets = torch.cat([gen_targets, padding])
            gen_targets = gen_targets[S_correct_mask]
            print(f"   Target distribution: {torch.bincount(gen_targets, minlength=num_models).tolist()}")
            target_distribution = torch.bincount(gen_targets, minlength=num_models).tolist()
        else:
            print("   Warning: generated_random_targets missing; target distribution unavailable")
            target_distribution = []

        # Use combined success (any step size) on S_correct
        tASR_rand = S_correct_combined_success_mask.float().mean().item()
        num_successful_forgeries = S_correct_combined_success_mask.sum().item()

        print(f"   Successful forgeries (any step size): {num_successful_forgeries}/{S_correct_mask.sum().item()}")
        print(f"   tASR_rand (combined): {tASR_rand:.4f}")

        return {
            'tasr_rand': tASR_rand,
            'num_successful_forgeries': num_successful_forgeries,
            'total_in_scorrect': S_correct_mask.sum().item(),
            'target_distribution': target_distribution,
            'forgery_success_rate': tASR_rand,
            'target_type': 'random',
            'multi_step_size_info': {
                'step_sizes': combined_success_info['step_sizes'],
                'individual_success_rates': combined_success_info['individual_success_rates'],
                'best_step_idx': combined_success_info['best_step_idx']
            }
        }
    
    def _compute_forgery_metrics_single_step(
        self,
        S_correct_original_preds: torch.Tensor,
        S_correct_adversarial_preds: torch.Tensor,
        S_correct_y_labels: torch.Tensor,
        attack_params: Dict[str, Any],
        num_models: int
    ) -> Dict[str, Any]:
        """Compute forgery metrics for single-step evaluation."""
        
        return self.metrics.compute_forgery_metrics(
            S_correct_original_preds, S_correct_adversarial_preds, S_correct_y_labels,
            target_predictions=attack_params.get('generated_random_targets')
        )
    
    def _compute_model_specific_metrics(
        self,
        S_correct_y_labels: torch.Tensor,
        S_correct_original_preds: torch.Tensor,
        adversarial_predictions: torch.Tensor,
        S_correct_mask: torch.Tensor,
        S_correct_combined_success_mask: torch.Tensor,
        attack_goal: str,
        attack_params: Dict[str, Any],
        model_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Compute model-specific metrics."""
        
        # Removed verbose per-model analysis print to avoid duplication with summary table
        model_metrics = {}
        unique_labels = torch.unique(S_correct_y_labels)
        
        # Create label-to-model name mapping
        label_to_model_name = self._create_label_to_model_mapping(model_names, len(unique_labels))
        
        for label in unique_labels:
            model_mask = (S_correct_y_labels == label)
            if model_mask.sum() > 0:
                model_orig_acc = (S_correct_original_preds[model_mask] == S_correct_y_labels[model_mask]).float().mean().item()
                model_adv_acc = (adversarial_predictions[S_correct_mask][model_mask] == S_correct_y_labels[model_mask]).float().mean().item()
                
                if attack_goal == 'removal':
                    # Use combined success mask for this model
                    model_combined_success = S_correct_combined_success_mask[model_mask]
                    model_asr = model_combined_success.float().mean().item()
                else:  # forgery
                    # For forgery, use combined success across step sizes
                    model_combined_success = S_correct_combined_success_mask[model_mask]
                    model_asr = model_combined_success.float().mean().item()
                
                model_metrics[int(label)] = {
                    'original_accuracy': model_orig_acc,
                    'adversarial_accuracy': model_adv_acc,
                    'attack_success_rate': model_asr,
                    'accuracy_drop': model_orig_acc - model_adv_acc,
                    'model_name': label_to_model_name.get(int(label), f"unknown_model_{label}")
                }
                
                # Suppress per-model verbose logging here; shown in summary table instead
        
        return model_metrics
    
    def _compute_model_specific_metrics_single_step(
        self,
        S_correct_y_labels: torch.Tensor,
        S_correct_original_preds: torch.Tensor,
        adversarial_predictions: torch.Tensor,
        S_correct_mask: torch.Tensor,
        attack_goal: str,
        attack_params: Dict[str, Any],
        model_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Compute model-specific metrics for single-step evaluation."""
        
        # Removed verbose per-model analysis print to avoid duplication with summary table
        model_metrics = {}
        unique_labels = torch.unique(S_correct_y_labels)
        
        # Create label-to-model name mapping
        label_to_model_name = self._create_label_to_model_mapping(model_names, len(unique_labels))
        
        for label in unique_labels:
            model_mask = (S_correct_y_labels == label)
            if model_mask.sum() > 0:
                model_orig_acc = (S_correct_original_preds[model_mask] == S_correct_y_labels[model_mask]).float().mean().item()
                model_adv_acc = (adversarial_predictions[S_correct_mask][model_mask] == S_correct_y_labels[model_mask]).float().mean().item()
                
                if attack_goal == 'removal':
                    # For removal, success is when adversarial prediction != true label
                    model_success = (adversarial_predictions[S_correct_mask][model_mask] != S_correct_y_labels[model_mask])
                    model_asr = model_success.float().mean().item()
                else:  # forgery
                    model_asr = self._compute_model_forgery_asr(
                        model_mask, S_correct_mask, adversarial_predictions,
                        attack_params, S_correct_y_labels
                    )
                
                model_metrics[int(label)] = {
                    'original_accuracy': model_orig_acc,
                    'adversarial_accuracy': model_adv_acc,
                    'attack_success_rate': model_asr,
                    'accuracy_drop': model_orig_acc - model_adv_acc,
                    'model_name': label_to_model_name.get(int(label), f"unknown_model_{label}")
                }
                
                # Suppress per-model verbose logging here; shown in summary table instead
        
        return model_metrics
    
    def _compute_model_forgery_asr(
        self,
        model_mask: torch.Tensor,
        S_correct_mask: torch.Tensor,
        adversarial_predictions: torch.Tensor,
        attack_params: Dict[str, Any],
        S_correct_y_labels: torch.Tensor
    ) -> float:
        """Compute forgery ASR for a specific model."""
        
        # Ensure all tensors are on the same device as model_mask
        device = model_mask.device
        S_correct_mask = S_correct_mask.to(device)
        adversarial_predictions = adversarial_predictions.to(device)
        
        if 'generated_random_targets' in attack_params:
            # Random targets - need to handle shape mismatch
            S_correct_random_targets = attack_params['generated_random_targets'].to(device)
            
            # Get predictions for this model's samples
            model_adv_preds = adversarial_predictions[S_correct_mask][model_mask]
            
            # Adjust random targets to match S_correct shape first
            if len(S_correct_random_targets) > len(S_correct_mask):
                # Truncate to match S_correct size
                S_correct_random_targets = S_correct_random_targets[:len(S_correct_mask)]
            else:
                # Pad with zeros (will be masked out anyway)
                padding = torch.zeros(len(S_correct_mask) - len(S_correct_random_targets), 
                                   dtype=S_correct_random_targets.dtype,
                                   device=device)
                S_correct_random_targets = torch.cat([S_correct_random_targets, padding])
            
            # Now get targets for this model's samples
            model_random_targets = S_correct_random_targets[S_correct_mask][model_mask]
            model_asr = (model_adv_preds == model_random_targets).float().mean().item()
        else:
            model_asr = 0.0
        
        return model_asr
    
    def _create_label_to_model_mapping(
        self, 
        model_names: Optional[List[str]], 
        num_labels: int
    ) -> Dict[int, str]:
        """Create mapping from label indices to model names."""
        
        if model_names is None:
            return {i: f"model_{i}" for i in range(num_labels)}
        
        try:
            from src.utils.model_isolation import sort_models_by_load_order
            sorted_model_names = sort_models_by_load_order(model_names)
            return {i: model_name for i, model_name in enumerate(sorted_model_names)}
        except ImportError:
            return {i: model_name for i, model_name in enumerate(model_names)}
