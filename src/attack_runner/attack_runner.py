"""
Main attack runner orchestrator.
"""

import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from .individual_attack_runners import (
    W1AttackRunner, W2AttackRunner, W3AttackRunner, 
    B1AttackRunner, B2AttackRunner
)
from ..attack_evaluation.evaluator import AttackEvaluator
from ..attack_evaluation.target_selection import TargetSelector
from ..data_loader.fingerprint_extractor_factory import FingerprintExtractorFactory
from ..results.image_saver import ImageSaver


class AttackRunner:
    """Main class for orchestrating and running attacks."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.evaluator = AttackEvaluator(device)
        self.target_selector = TargetSelector()
        
        # Initialize individual attack runners
        self.w1_runner = W1AttackRunner(device)
        self.w2_runner = W2AttackRunner(device)
        self.w3_runner = W3AttackRunner(device)
        self.b1_runner = B1AttackRunner(device)
        self.b2_runner = B2AttackRunner(device)
        self._cached_context: Optional[Dict[str, Any]] = None
        self._cached_context_key: Optional[Tuple[Any, ...]] = None
    
    def run_multiple_attacks(
        self,
        data_dir: str,
        attack_data_dir: Optional[str],
        fingerprint_method: str,
        attack_types: List[str],
        num_test_images_per_model: int,
        image_size: int,
        attack_params: Dict[str, Any],
        attack_batch_size: int = 10,
        real_data_path: Optional[str] = None,
        cache_dir: str = "cache",
        save_attacked_images_flag: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Run multiple attacks and return comprehensive results.
        
        Args:
            data_dir: Base data directory
            fingerprint_method: Fingerprint method name
            attack_types: List of attack types to run
            num_test_images_per_model: Number of test images per model
            image_size: Image size
            attack_params: Attack parameters
            attack_batch_size: Batch size for attacks
            real_data_path: Path to real data (for some methods)
            cache_dir: Cache directory
            save_attacked_images_flag: Whether to save attacked images
            
        Returns:
            List of attack results
        """
        
        print("🎯 FINGERPRINT ROBUSTNESS ATTACK EVALUATION")
        print("=" * 60)
        print(f"Fingerprint Method: {fingerprint_method}")
        print(f"Attack Types: {', '.join(attack_types).upper()}")
        print(f"Attack Goal: {attack_params.get('attack_goal', 'removal').upper()}")
        print("=" * 60)
        
        context = self._get_or_prepare_context(
            data_dir, attack_data_dir, fingerprint_method, num_test_images_per_model,
            image_size, real_data_path, cache_dir
        )
        data_dir_path = context["data_dir_path"]
        attack_data_dir_path = context["attack_data_dir_path"]
        model_names = context["model_names"]
        X_features = context["X_features"].clone()
        X_images = context["X_images"].clone()
        y = context["y"].clone()
        num_models = context["num_models"]
        true_attribution_model = context["true_attribution_model"]
        surrogate_attribution_model = context["surrogate_attribution_model"]
        surrogate_extractor = context["surrogate_extractor"]
        is_implicit = context["is_implicit"]
        
        # Handle target selection for forgery attacks (simplified: always random per-sample targets)
        if attack_params.get('attack_goal') == 'forgery':
            # Ignore any fixed target specification and always generate per-sample random targets
            if 'target_class' in attack_params:
                del attack_params['target_class']
            attack_params['use_random_targets'] = True
            print("ℹ️  Forgery: generating per-sample random targets at runtime.")
        
        # Run attacks
        all_attack_results = []
        
        for attack_type in attack_types:
            print(f"\n{'='*80}")
            print(f"RUNNING {attack_type.upper()} ATTACK")
            print(f"{'='*80}")
            
            try:
                # Prepare attack parameters
                current_attack_params = attack_params.copy()
                
                # Handle target generation for forgery attacks
                if current_attack_params.get('attack_goal') == 'forgery' and current_attack_params.get('use_random_targets', False):
                    current_attack_params['generated_random_targets'] = self.target_selector.select_random_targets(
                        y, num_models
                    )
                
                # Run the specific attack
                if attack_type.lower() == 'w1':
                    attack_result = self.w1_runner.run_attack(
                        X_images, y, fingerprint_method, true_attribution_model,
                        current_attack_params, attack_batch_size, real_data_path, cache_dir, str(data_dir_path)
                    )
                elif attack_type.lower() == 'w2':
                    attack_result = self.w2_runner.run_attack(
                        X_images, y, fingerprint_method, true_attribution_model,
                        current_attack_params, attack_batch_size, real_data_path, cache_dir, str(data_dir_path)
                    )
                elif attack_type.lower() == 'w3':
                    attack_result = self.w3_runner.run_attack(
                        X_images, y, fingerprint_method, true_attribution_model,
                        current_attack_params, attack_batch_size, real_data_path, cache_dir, str(data_dir_path),
                        surrogate_extractor
                    )
                elif attack_type.lower() == 'b1':
                    attack_result = self.b1_runner.run_attack(
                        X_images, y, fingerprint_method, true_attribution_model,
                        current_attack_params, attack_batch_size, real_data_path, cache_dir, str(data_dir_path),
                        surrogate_attribution_model
                    )
                elif attack_type.lower() == 'b2':
                    attack_result = self.b2_runner.run_attack(
                        X_images, y, fingerprint_method, true_attribution_model,
                        current_attack_params, attack_batch_size, real_data_path, cache_dir, str(data_dir_path)
                    )
                else:
                    raise ValueError(f"Unsupported attack type: {attack_type}")
                
                # Handle different return types based on attack type
                if isinstance(attack_result.get('adversarial_images'), dict):
                    # B2 attack returns a dictionary
                    adversarial_images = attack_result['adversarial_images']['final']
                    individual_perturbed = attack_result['adversarial_images'].get('individual')
                    cumulative_perturbed = attack_result['adversarial_images'].get('cumulative')
                else:
                    adversarial_images = attack_result['adversarial_images']
                    individual_perturbed = attack_result.get('individual_results')
                    cumulative_perturbed = attack_result.get('cumulative_results')
                
                # Evaluate adversarial performance
                original_preds = attack_result.get('original_predictions')
                if original_preds is None:
                    # Get original predictions if not provided
                    original_preds = self._get_original_predictions(
                        X_images, fingerprint_method, true_attribution_model, is_implicit,
                        real_data_path, cache_dir, str(data_dir_path) if data_dir_path is not None else None
                    )
                
                # Get adversarial predictions
                adversarial_preds = self._get_adversarial_predictions(
                    adversarial_images, fingerprint_method, true_attribution_model, is_implicit,
                    real_data_path, cache_dir, str(data_dir_path) if data_dir_path is not None else None
                )
                
                # Evaluate attack effectiveness
                if 'combined_success_info' in attack_result and attack_result['combined_success_info'] is not None:
                    # Multi-step-size evaluation
                    combined_success_info = {
                        'combined_success_mask': attack_result['combined_success_info'].get('combined_success_mask', attack_result.get('combined_success_mask')),
                        'step_sizes': attack_result['combined_success_info'].get('step_sizes', attack_result.get('step_sizes')),
                        'individual_success_rates': attack_result['combined_success_info'].get('individual_success_rates', attack_result.get('individual_success_rates')),
                        'best_step_idx': attack_result['combined_success_info'].get('best_step_idx', attack_result.get('best_step_idx')),
                        'overall_success_rate': attack_result['combined_success_info'].get('overall_success_rate', attack_result.get('overall_success_rate'))
                    }
                    
                    evaluation_result = self.evaluator.evaluate_attack_effectiveness(
                        X_images, adversarial_images, original_preds, adversarial_preds,
                        y, attack_type, current_attack_params, fingerprint_method,
                        true_attribution_model, attack_batch_size, individual_perturbed,
                        cumulative_perturbed, real_data_path, cache_dir, data_dir,
                        model_names, combined_success_info
                    )
                else:
                    # Single-step evaluation
                    evaluation_result = self.evaluator.evaluate_attack_effectiveness(
                        X_images, adversarial_images, original_preds, adversarial_preds,
                        y, attack_type, current_attack_params, fingerprint_method,
                        true_attribution_model, attack_batch_size, individual_perturbed,
                        cumulative_perturbed, real_data_path, cache_dir, data_dir,
                        model_names
                    )
                
                # Save attacked images if enabled
                if save_attacked_images_flag and adversarial_images is not None:
                    self._save_attacked_images(
                        X_images, adversarial_images, y, original_preds, adversarial_preds,
                        attack_type, current_attack_params, data_dir_path, model_names
                    )
                
                # Add attack type to results
                evaluation_result['attack_type'] = attack_type.upper()
                all_attack_results.append(evaluation_result)
                
                print(f"✅ {attack_type.upper()} attack completed successfully!")
                
            except Exception as e:
                print(f"❌ {attack_type.upper()} attack failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Generate comprehensive summary
        self._generate_summary(all_attack_results, attack_params)
        
        return all_attack_results
    
    def _get_original_predictions(
        self,
        X_images: torch.Tensor,
        fingerprint_method: str,
        true_attribution_model: torch.nn.Module,
        is_implicit: bool,
        real_data_path: Optional[str] = None,
        cache_dir: str = "cache",
        data_dir: Optional[str] = None
    ) -> torch.Tensor:
        """Get original predictions for images."""
        
        with torch.no_grad():
            if is_implicit:
                # For implicit methods, directly use the model on images
                orig_logits = true_attribution_model(X_images)
                _, original_preds = torch.max(orig_logits, 1)
            else:
                # For explicit methods, extract fingerprints first
                fingerprint_extractor = FingerprintExtractorFactory.create(
                    fingerprint_method,
                    self.device,
                    real_data_path=real_data_path,
                    cache_dir=cache_dir,
                    data_dir=data_dir
                )
                orig_fingerprints = fingerprint_extractor.extract_fingerprint(X_images)
                if orig_fingerprints.dim() > 2:
                    orig_fingerprints = orig_fingerprints.view(orig_fingerprints.shape[0], -1)
                orig_logits = true_attribution_model(orig_fingerprints.to(self.device))
                _, original_preds = torch.max(orig_logits, 1)
        
        return original_preds
    
    def _get_adversarial_predictions(
        self,
        adversarial_images: torch.Tensor,
        fingerprint_method: str,
        true_attribution_model: torch.nn.Module,
        is_implicit: bool,
        real_data_path: Optional[str] = None,
        cache_dir: str = "cache",
        data_dir: Optional[str] = None
    ) -> torch.Tensor:
        """Get adversarial predictions for images."""
        
        with torch.no_grad():
            if is_implicit:
                # For implicit methods, directly use the model on images
                adv_logits = true_attribution_model(adversarial_images)
                _, adv_preds = torch.max(adv_logits, 1)
            else:
                # For explicit methods, extract fingerprints first
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
        
        return adv_preds
    
    def _save_attacked_images(
        self,
        original_images: torch.Tensor,
        adversarial_images: torch.Tensor,
        y_labels: torch.Tensor,
        original_preds: torch.Tensor,
        adversarial_preds: torch.Tensor,
        attack_type: str,
        attack_params: Dict[str, Any],
        data_dir: Path,
        model_names: List[str]
    ) -> None:
        """Save successfully attacked images."""
        
        try:
            attack_goal = attack_params.get('attack_goal', 'removal')
            
            # Move all tensors to same device (use original_preds device)
            device = original_preds.device
            y_labels = y_labels.to(device)
            adversarial_preds = adversarial_preds.to(device)
            
            # Determine success based on attack goal
            if attack_goal == 'removal':
                # Success: originally correct prediction becomes incorrect
                success_mask = (original_preds == y_labels) & (adversarial_preds != y_labels)
            elif attack_goal == 'forgery':
                # Success: adversarial prediction matches per-sample target
                if 'generated_random_targets' in attack_params:
                    target_predictions = attack_params['generated_random_targets'].to(device)
                    success_mask = (adversarial_preds == target_predictions)
                else:
                    success_mask = (original_preds == y_labels) & (adversarial_preds != y_labels)
            else:
                print(f"Warning: Unknown attack goal '{attack_goal}', skipping image saving")
                return
            
            if not success_mask.any():
                print("No successful attacks found, skipping image saving")
                return
            
            # Create base directory for attacked images
            attacked_images_dir = data_dir / "attacked_images"
            attacked_images_dir.mkdir(parents=True, exist_ok=True)
            
            # Get successful indices
            successful_indices = torch.where(success_mask)[0]
            print(f"💾 Saving {len(successful_indices)} successfully attacked images...")
            
            # Move images to same device for saving
            original_images = original_images.to(device)
            adversarial_images = adversarial_images.to(device)
            
            # Create model name mapping
            label_to_model_name = {i: model_name for i, model_name in enumerate(model_names)}
            
            # Use ImageSaver to save images
            image_saver = ImageSaver(data_dir)
            image_saver.save_attacked_images(
                original_images=original_images,
                adversarial_images=adversarial_images,
                y_labels=y_labels,
                original_predictions=original_preds,
                adversarial_predictions=adversarial_preds,
                attack_type=attack_type,
                attack_goal=attack_goal,
                model_names=model_names,
                success_mask=success_mask,
                target_predictions=target_predictions if attack_goal == 'forgery' else None
            )
            
        except Exception as e:
            print(f"⚠️  Warning: Failed to save attacked images: {e}")

    def _get_or_prepare_context(
        self,
        data_dir: str,
        attack_data_dir: Optional[str],
        fingerprint_method: str,
        num_test_images_per_model: int,
        image_size: int,
        real_data_path: Optional[str],
        cache_dir: str,
    ) -> Dict[str, Any]:
        data_dir_path = Path(data_dir) if isinstance(data_dir, str) else data_dir
        attack_data_dir_path = Path(attack_data_dir) if attack_data_dir else data_dir_path
        cache_key = (
            str(data_dir_path),
            str(attack_data_dir_path),
            fingerprint_method,
            num_test_images_per_model,
            image_size,
            real_data_path,
            cache_dir,
        )
        if self._cached_context_key == cache_key and self._cached_context is not None:
            return self._cached_context
        
        context = self._prepare_context(
            data_dir_path,
            attack_data_dir_path,
            fingerprint_method,
            num_test_images_per_model,
            image_size,
            real_data_path,
            cache_dir,
        )
        self._cached_context_key = cache_key
        self._cached_context = context
        return context

    def _prepare_context(
        self,
        data_dir_path: Path,
        attack_data_dir_path: Path,
        fingerprint_method: str,
        num_test_images_per_model: int,
        image_size: int,
        real_data_path: Optional[str],
        cache_dir: str,
    ) -> Dict[str, Any]:
        from ..data_loader.model_discovery import ModelDiscovery
        from ..data_loader.data_preparer import DataPreparer
        from ..model_loader.model_loader import ModelLoader

        model_names = ModelDiscovery.discover_available_models(attack_data_dir_path)

        # Legacy handling: earlier implicit trainings (qian20/wang20) used alphabetical
        # directory order when assigning labels. To evaluate those checkpoints without
        # retraining, we must reproduce the same ordering here instead of the canonical
        # load order provided by ModelDiscovery.
        legacy_alpha_methods = {"qian20", "wang20"}
        if fingerprint_method in legacy_alpha_methods:
            alpha_order = sorted(model_names)
            print(
                f"⚠️  Using legacy alphabetical model order for {fingerprint_method} "
                "to match training label mapping"
            )
            print(f"   Alphabetical order: {alpha_order}")
            model_names = alpha_order

        data_preparer = DataPreparer(self.device)
        X_features, X_images, y, num_models = data_preparer.prepare_test_data(
            attack_data_dir_path,
            fingerprint_method,
            model_names,
            num_test_images_per_model,
            image_size,
            real_data_path,
            cache_dir,
        )
        model_loader = ModelLoader(self.device)
        models_dir = data_dir_path / f"models_{fingerprint_method}"
        h_model_path = models_dir / "true_attribution_model.pth"
        h_s_model_path = models_dir / "surrogate_attribution_model.pth"
        phi_s_model_path = models_dir / "surrogate_extractor_model.pth"
        feature_dim = X_features.numel() // X_features.shape[0]
        true_attribution_model, surrogate_attribution_model, surrogate_extractor = model_loader.load_all_models(
            str(h_model_path),
            str(h_s_model_path),
            str(phi_s_model_path),
            fingerprint_method,
            num_models,
            feature_dim,
            real_data_path,
            cache_dir,
            str(data_dir_path),
        )
        is_implicit = FingerprintExtractorFactory.is_implicit_method(
            fingerprint_method,
            self.device,
            real_data_path=real_data_path,
            cache_dir=cache_dir,
            data_dir=str(data_dir_path) if data_dir_path is not None else None,
        )
        return {
            "data_dir_path": data_dir_path,
            "attack_data_dir_path": attack_data_dir_path,
            "model_names": model_names,
            "X_features": X_features,
            "X_images": X_images,
            "y": y,
            "num_models": num_models,
            "true_attribution_model": true_attribution_model,
            "surrogate_attribution_model": surrogate_attribution_model,
            "surrogate_extractor": surrogate_extractor,
            "is_implicit": is_implicit,
        }
    
    def _generate_summary(self, all_attack_results: List[Dict[str, Any]], attack_params: Dict[str, Any]) -> None:
        """Generate comprehensive summary of attack results."""
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE ATTACK RESULTS SUMMARY")
        print(f"{'='*80}")
        
        if not all_attack_results:
            print("❌ No attacks completed successfully!")
            return
        
        # Create summary table
        summary_data = []
        
        for result in all_attack_results:
            attack_type = result['attack_type']
            overall_metrics = result['overall_metrics']
            
            # Extract key metrics
            attack_success_rate = overall_metrics['attack_success_rate']
            l2_dist = overall_metrics['l2_distance']
            linf_dist = overall_metrics['linf_distance']
            psnr = overall_metrics['psnr']
            lpips = overall_metrics['lpips']
            
            # Add attack-specific metrics based on attack goal
            if attack_params.get('attack_goal') == 'removal' and 'removal_metrics' in result:
                asr_remove = result['removal_metrics']['asr_remove']
                summary_data.append([
                    attack_type,
                    f"{asr_remove:.4f}",
                    f"{l2_dist:.6f}",
                    f"{linf_dist:.6f}",
                    f"{psnr:.2f}" if psnr != "N/A" else "N/A",
                    f"{lpips:.4f}" if lpips != "N/A" else "N/A"
                ])
            elif attack_params.get('attack_goal') == 'forgery' and 'forgery_metrics' in result:
                forgery_metrics = result['forgery_metrics']
                tasr_rand = forgery_metrics['tasr_rand']
                target_type = forgery_metrics.get('target_type', 'unknown')
                target_display = f"random ({len(forgery_metrics.get('target_distribution', []))} classes)" if target_type == 'random' else target_type
                
                summary_data.append([
                    attack_type,
                    f"{tasr_rand:.4f}",
                    target_display,
                    f"{l2_dist:.6f}",
                    f"{linf_dist:.6f}",
                    f"{psnr:.2f}" if psnr != "N/A" else "N/A",
                    f"{lpips:.4f}" if lpips != "N/A" else "N/A"
                ])
        
        # Create headers based on attack goal
        if attack_params.get('attack_goal') == 'removal':
            headers = [
                "Attack Type", "ASR_remove", "L2 Distance", "L∞ Distance", "PSNR", "LPIPS"
            ]
        else:  # forgery
            headers = [
                "Attack Type", "tASR_rand", "Target Dist.", "L2 Distance", "L∞ Distance", "PSNR", "LPIPS"
            ]
        
        # Print summary table
        try:
            from tabulate import tabulate
            print(tabulate(summary_data, headers=headers, tablefmt="grid"))
            
            # Print B2 perturbation-specific results if available
            for result in all_attack_results:
                if result['attack_type'].lower() == 'b2' and 'perturbation_metrics' in result:
                    print("\nB2 Attack Perturbation Results:")
                    pert_data = []
                    for pert_type, metrics in result['perturbation_metrics'].items():
                        if attack_params.get('attack_goal') == 'removal':
                            succ = metrics['removal_metrics'].get('successful_removals', 0)
                            total = metrics['removal_metrics'].get('total_correct_before', 0)
                            asr_val = metrics['removal_metrics'].get('asr_remove', 0.0)
                            # Prefer runner-computed avg_linf_successful if available
                            runner_avg = result.get('individual_results', {}).get('perturbation_results', {}).get(pert_type, {}).get('avg_linf_successful', None)
                            avg_linf_success = runner_avg if runner_avg is not None else metrics.get('avg_linf_successful', float('nan'))
                            # Debug: print what we got
                            print(f"    [DEBUG] {pert_type}: runner_avg={runner_avg}, metrics_avg={metrics.get('avg_linf_successful')}, final={avg_linf_success}")
                            # Format: check for NaN properly
                            import math
                            if isinstance(avg_linf_success, float) and math.isnan(avg_linf_success):
                                linf_str = "N/A"
                            else:
                                linf_str = f"{avg_linf_success:.6f}"
                            pert_data.append([
                                pert_type,
                                f"{asr_val:.4f}",
                                f"{succ}/{total}",
                                f"{asr_val:.4f}",
                                linf_str
                            ])
                    if pert_data:
                        print(tabulate(pert_data, 
                                    headers=['Perturbation', 'ASR_remove', 'Successful/Total', 'ASR', 'Avg L∞ (success)'],
                                    tablefmt="grid"))
        except ImportError:
            # Fallback if tabulate is not available
            print("Summary Table:")
            print(" | ".join(headers))
            print("-" * (len(" | ".join(headers))))
            for row in summary_data:
                print(" | ".join(row))
        
        # Per-model ASR table for removal attacks
        if attack_params.get('attack_goal') == 'removal':
            for result in all_attack_results:
                model_metrics = result.get('model_specific_metrics')
                if not model_metrics:
                    continue
                rows = []
                for label, metrics in model_metrics.items():
                    rows.append([
                        label,
                        metrics.get('model_name', f'model_{label}'),
                        f"{metrics['attack_success_rate']:.4f}",
                        f"{metrics['original_accuracy']:.4f}",
                        f"{metrics['adversarial_accuracy']:.4f}",
                        f"{metrics['accuracy_drop']:.4f}",
                    ])
                if rows:
                    print("\nPer-model removal metrics (ASR=any-step; Acc=best-step):")
                    try:
                        from tabulate import tabulate
                        print(tabulate(rows, headers=["Model Id", "Model Name", "ASR_remove", "Orig Acc", "Adv Acc", "Acc Drop"], tablefmt="grid"))
                    except Exception:
                        print("Model Id | Model Name | ASR_remove | Orig Acc | Adv Acc | Acc Drop")
                        for r in rows:
                            print(" | ".join(map(str, r)))

        # Print attack goal explanation
        print(f"\n📊 ATTACK GOAL: {attack_params.get('attack_goal').upper()}")
        if attack_params.get('attack_goal') == 'removal':
            print("   • ASR_remove: Higher values indicate more effective removal attacks")
            print("   • Goal: Modify images in S_correct such that h(x_adv) != y(x)")
            print("   • S_correct: Images correctly classified before attack")
        else:
            print("   • tASR_rand: Higher values indicate more successful forgery attacks")
            print("   • Goal: Modify images in S_correct to be classified as per-sample random target")
            print("   • S_correct: Images correctly classified before attack")
        
        # Print target model forgery vulnerability table if present
        if attack_params.get('attack_goal') == 'forgery':
            for result in all_attack_results:
                if 'target_model_vulnerability' in result:
                    vul = result['target_model_vulnerability']
                    print("\n📊 TARGET MODEL FORGERY VULNERABILITY (summary):")
                    rows = []
                    for t, info in vul['by_model'].items():
                        rows.append([t, info['model_name'], f"{info['successful']}/{info['assigned']}", f"{info['forgery_success_rate']:.4f}"])
                    try:
                        from tabulate import tabulate
                        print(tabulate(rows, headers=["Target", "Model", "Successful/Assigned", "Vulnerability"], tablefmt="grid"))
                    except Exception:
                        print("Target | Model | Successful/Assigned | Vulnerability")
                        for r in rows:
                            print(" | ".join(map(str, r)))

        # Print best performing attack
        if summary_data:
            if attack_params.get('attack_goal') == 'removal':
                # For removal, higher ASR_remove is better
                best_attack_idx = max(range(len(summary_data)), 
                                    key=lambda i: float(summary_data[i][1]))
            else:
                # For forgery, higher tASR_rand is better
                best_attack_idx = max(range(len(summary_data)), 
                                    key=lambda i: float(summary_data[i][1]))
            
            best_attack = summary_data[best_attack_idx][0]
            best_metric = summary_data[best_attack_idx][1]
            metric_name = "ASR_remove" if attack_params.get('attack_goal') == 'removal' else "tASR_rand"
            
            print(f"\n🏆 BEST PERFORMING ATTACK: {best_attack}")
            print(f"   {metric_name}: {best_metric}")
        
        # Show which attacks were actually run
        successful_attacks = [result['attack_type'] for result in all_attack_results]
        
        print(f"\n✅ All attacks completed! Processed {len(all_attack_results)} attack types.")
        print(f"📊 Evaluation was performed on S_correct (images correctly classified before attack)")
        
        if successful_attacks:
            print(f"   ✅ Successful attacks: {', '.join(successful_attacks)}")
