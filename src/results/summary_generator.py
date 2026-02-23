"""
Summary generation functionality for attack results.
"""

from typing import List, Dict, Any, Optional
from tabulate import tabulate


class SummaryGenerator:
    """Handles generation of comprehensive attack result summaries."""
    
    def __init__(self):
        pass
    
    def generate_comprehensive_summary(
        self,
        all_attack_results: List[Dict[str, Any]],
        attack_params: Dict[str, Any]
    ) -> None:
        """
        Generate comprehensive summary of attack results.
        
        Args:
            all_attack_results: List of attack result dictionaries
            attack_params: Attack parameters dictionary
        """
        
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
                
                if target_type == 'random':
                    target_display = f"random ({len(forgery_metrics.get('target_distribution', []))} classes)"
                else:
                    target_display = target_type
                
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
                "Attack Type", "tASR_rand", "Target Class", "L2 Distance", "L∞ Distance", "PSNR", "LPIPS"
            ]
        
        # Print summary table
        try:
            from tabulate import tabulate
            print(tabulate(summary_data, headers=headers, tablefmt="grid"))
        except ImportError:
            # Fallback if tabulate is not available
            print("Summary Table:")
            print(" | ".join(headers))
            print("-" * (len(" | ".join(headers))))
            for row in summary_data:
                print(" | ".join(row))
        
        # Print attack goal explanation
        self._print_attack_goal_explanation(attack_params)
        
        # Print best performing attack
        if summary_data:
            self._print_best_attack(summary_data, attack_params)
        
        # Print completion summary
        self._print_completion_summary(all_attack_results)
    
    def _print_attack_goal_explanation(self, attack_params: Dict[str, Any]) -> None:
        """Print explanation of attack goal and metrics."""
        
        print(f"\n📊 ATTACK GOAL: {attack_params.get('attack_goal').upper()}")
        
        if attack_params.get('attack_goal') == 'removal':
            print("   • ASR_remove: Higher values indicate more effective removal attacks")
            print("   • Goal: Modify images in S_correct such that h(x_adv) != y(x)")
            print("   • S_correct: Images correctly classified before attack")
        else:
            print("   • tASR_rand: Higher values indicate more successful forgery attacks")
            print("   • Goal: Modify images in S_correct to be classified as per-sample random target")
            print("   • S_correct: Images correctly classified before attack")
    
    def _print_best_attack(self, summary_data: List[List[str]], attack_params: Dict[str, Any]) -> None:
        """Print best performing attack."""
        
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
    
    def _print_completion_summary(self, all_attack_results: List[Dict[str, Any]]) -> None:
        """Print completion summary."""
        
        # Show which attacks were actually run
        successful_attacks = [result['attack_type'] for result in all_attack_results]
        
        print(f"\n✅ All attacks completed! Processed {len(all_attack_results)} attack types.")
        print(f"📊 Evaluation was performed on S_correct (images correctly classified before attack)")
        
        if successful_attacks:
            print(f"   ✅ Successful attacks: {', '.join(successful_attacks)}")
    
    def generate_attack_summary(
        self,
        attack_type: str,
        success_metrics: Dict[str, Any],
        step_sizes: List[float]
    ) -> None:
        """
        Generate summary for a single attack type.
        
        Args:
            attack_type: Type of attack
            success_metrics: Success metrics dictionary
            step_sizes: List of step sizes tested
        """
        
        print(f"\n--- {attack_type.upper()} Attack Summary ---")
        print(f"Step sizes tested: {step_sizes}")
        
        for step_size in step_sizes:
            step_key = f'step_size_{step_size}'
            if step_key in success_metrics:
                metrics = success_metrics[step_key]
                print(f"  Step size {step_size}:")
                print(f"    Success rate: {metrics['success_rate']:.4f}")
                print(f"    Total correct before: {metrics['total_correct_before']}")
                print(f"    Successful attacks: {metrics['successful_attacks']}")
    
    def print_attack_effectiveness(
        self,
        attack_goal: str,
        attack_success_metric: float,
        removal_metrics: Optional[Dict[str, Any]] = None,
        forgery_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Print attack effectiveness information.
        
        Args:
            attack_goal: Goal of the attack ('removal' or 'forgery')
            attack_success_metric: Success metric value
            removal_metrics: Removal-specific metrics (optional)
            forgery_metrics: Forgery-specific metrics (optional)
        """
        
        print(f"\n📊 ATTACK EFFECTIVENESS:")
        
        if attack_goal == 'removal':
            print(f"   Attack Goal: REMOVAL")
            print(f"   ASR_remove: {attack_success_metric:.4f}")
            print(f"   Interpretation: {attack_success_metric*100:.1f}% of correctly classified images were successfully attacked")
            
            if removal_metrics:
                print(f"   Successful removals: {removal_metrics.get('num_successful_removals', 'N/A')}")
                print(f"   Total in S_correct: {removal_metrics.get('total_in_scorrect', 'N/A')}")
        else:
            print(f"   Attack Goal: FORGERY")
            print(f"   tASR_rand: {attack_success_metric:.4f}")
            print(f"   Interpretation: {attack_success_metric*100:.1f}% of correctly classified images were successfully forged")
            
            if forgery_metrics:
                print(f"   Successful forgeries: {forgery_metrics.get('num_successful_forgeries', 'N/A')}")
                print(f"   Total in S_correct: {forgery_metrics.get('total_in_scorrect', 'N/A')}")
                print(f"   Target type: {forgery_metrics.get('target_type', 'unknown')}")
                
                if forgery_metrics.get('target_type') == 'random':
                    target_dist = forgery_metrics.get('target_distribution', [])
                    print(f"   Target distribution: {target_dist}")
