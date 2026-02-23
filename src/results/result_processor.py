"""
Result processing functionality for attack evaluation.
"""

import torch
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


class ResultProcessor:
    """Handles processing and analysis of attack results."""
    
    def __init__(self):
        pass
    
    def process_attack_results(
        self,
        attack_results: List[Dict[str, Any]],
        attack_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process and analyze attack results.
        
        Args:
            attack_results: List of attack result dictionaries
            attack_params: Attack parameters dictionary
            
        Returns:
            Processed results dictionary
        """
        
        processed_results = {
            'timestamp': datetime.now().isoformat(),
            'attack_params': attack_params,
            'total_attacks': len(attack_results),
            'successful_attacks': len([r for r in attack_results if 'attack_type' in r]),
            'results': attack_results
        }
        
        # Add summary statistics
        if attack_results:
            processed_results.update(self._compute_summary_statistics(attack_results, attack_params))
        
        return processed_results
    
    def _compute_summary_statistics(
        self,
        attack_results: List[Dict[str, Any]],
        attack_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute summary statistics from attack results."""
        
        attack_goal = attack_params.get('attack_goal', 'removal')
        
        # Extract success rates
        success_rates = []
        l2_distances = []
        linf_distances = []
        psnr_values = []
        lpips_values = []
        
        for result in attack_results:
            overall_metrics = result.get('overall_metrics', {})
            
            # Success rate
            success_rate = overall_metrics.get('attack_success_rate', 0.0)
            success_rates.append(success_rate)
            
            # Distance metrics
            l2_dist = overall_metrics.get('l2_distance', 0.0)
            l2_distances.append(l2_dist)
            
            linf_dist = overall_metrics.get('linf_distance', 0.0)
            linf_distances.append(linf_dist)
            
            # Quality metrics
            psnr = overall_metrics.get('psnr', float('nan'))
            if psnr != "N/A" and not torch.isnan(torch.tensor(psnr)):
                psnr_values.append(psnr)
            
            lpips = overall_metrics.get('lpips', float('nan'))
            if lpips != "N/A" and not torch.isnan(torch.tensor(lpips)):
                lpips_values.append(lpips)
        
        # Compute statistics
        summary_stats = {
            'success_rate_stats': self._compute_stats(success_rates, 'Success Rate'),
            'l2_distance_stats': self._compute_stats(l2_distances, 'L2 Distance'),
            'linf_distance_stats': self._compute_stats(linf_distances, 'L∞ Distance'),
        }
        
        if psnr_values:
            summary_stats['psnr_stats'] = self._compute_stats(psnr_values, 'PSNR')
        
        if lpips_values:
            summary_stats['lpips_stats'] = self._compute_stats(lpips_values, 'LPIPS')
        
        # Find best and worst performing attacks
        if success_rates:
            best_idx = max(range(len(success_rates)), key=lambda i: success_rates[i])
            worst_idx = min(range(len(success_rates)), key=lambda i: success_rates[i])
            
            summary_stats['best_attack'] = {
                'attack_type': attack_results[best_idx].get('attack_type', 'unknown'),
                'success_rate': success_rates[best_idx]
            }
            
            summary_stats['worst_attack'] = {
                'attack_type': attack_results[worst_idx].get('attack_type', 'unknown'),
                'success_rate': success_rates[worst_idx]
            }
        
        return summary_stats
    
    def _compute_stats(self, values: List[float], metric_name: str) -> Dict[str, Any]:
        """Compute basic statistics for a list of values."""
        
        if not values:
            return {
                'metric_name': metric_name,
                'count': 0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            }
        
        values_tensor = torch.tensor(values)
        
        return {
            'metric_name': metric_name,
            'count': len(values),
            'mean': values_tensor.mean().item(),
            'std': values_tensor.std().item(),
            'min': values_tensor.min().item(),
            'max': values_tensor.max().item()
        }
    
    def compare_attack_types(
        self,
        attack_results: List[Dict[str, Any]],
        metric_name: str = 'attack_success_rate'
    ) -> Dict[str, Any]:
        """
        Compare different attack types based on a specific metric.
        
        Args:
            attack_results: List of attack result dictionaries
            metric_name: Name of the metric to compare
            
        Returns:
            Comparison results dictionary
        """
        
        comparison = {}
        
        for result in attack_results:
            attack_type = result.get('attack_type', 'unknown')
            overall_metrics = result.get('overall_metrics', {})
            
            if metric_name in overall_metrics:
                comparison[attack_type] = {
                    'value': overall_metrics[metric_name],
                    'rank': 0  # Will be filled later
                }
        
        # Rank attacks by metric value (higher is better for success rates)
        ranked_attacks = sorted(
            comparison.items(),
            key=lambda x: x[1]['value'],
            reverse=True
        )
        
        # Assign ranks
        for rank, (attack_type, data) in enumerate(ranked_attacks, 1):
            comparison[attack_type]['rank'] = rank
        
        return {
            'metric_name': metric_name,
            'comparison': comparison,
            'ranking': ranked_attacks
        }
    
    def analyze_model_performance(
        self,
        attack_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze model-specific performance across attacks.
        
        Args:
            attack_results: List of attack result dictionaries
            
        Returns:
            Model performance analysis dictionary
        """
        
        model_analysis = {}
        
        for result in attack_results:
            attack_type = result.get('attack_type', 'unknown')
            model_metrics = result.get('model_specific_metrics', {})
            
            for model_id, metrics in model_metrics.items():
                if model_id not in model_analysis:
                    model_analysis[model_id] = {
                        'model_id': model_id,
                        'attacks': {},
                        'average_metrics': {}
                    }
                
                model_analysis[model_id]['attacks'][attack_type] = {
                    'original_accuracy': metrics.get('original_accuracy', 0.0),
                    'adversarial_accuracy': metrics.get('adversarial_accuracy', 0.0),
                    'attack_success_rate': metrics.get('attack_success_rate', 0.0),
                    'accuracy_drop': metrics.get('accuracy_drop', 0.0)
                }
        
        # Compute average metrics across attacks for each model
        for model_id, data in model_analysis.items():
            attacks = data['attacks']
            
            if attacks:
                avg_original_acc = sum(a['original_accuracy'] for a in attacks.values()) / len(attacks)
                avg_adversarial_acc = sum(a['adversarial_accuracy'] for a in attacks.values()) / len(attacks)
                avg_attack_success = sum(a['attack_success_rate'] for a in attacks.values()) / len(attacks)
                avg_accuracy_drop = sum(a['accuracy_drop'] for a in attacks.values()) / len(attacks)
                
                data['average_metrics'] = {
                    'average_original_accuracy': avg_original_acc,
                    'average_adversarial_accuracy': avg_adversarial_acc,
                    'average_attack_success_rate': avg_attack_success,
                    'average_accuracy_drop': avg_accuracy_drop
                }
        
        return model_analysis
    
    def generate_performance_report(
        self,
        attack_results: List[Dict[str, Any]],
        attack_params: Dict[str, Any]
    ) -> str:
        """
        Generate a comprehensive performance report.
        
        Args:
            attack_results: List of attack result dictionaries
            attack_params: Attack parameters dictionary
            
        Returns:
            Formatted performance report string
        """
        
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("FINGERPRINT ROBUSTNESS ATTACK EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Attack Goal: {attack_params.get('attack_goal', 'unknown').upper()}")
        report_lines.append("")
        
        # Summary
        report_lines.append("SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Attacks Run: {len(attack_results)}")
        report_lines.append(f"Successful Attacks: {len([r for r in attack_results if 'attack_type' in r])}")
        report_lines.append("")
        
        # Individual attack results
        for result in attack_results:
            attack_type = result.get('attack_type', 'unknown')
            overall_metrics = result.get('overall_metrics', {})
            
            report_lines.append(f"{attack_type.upper()} ATTACK")
            report_lines.append("-" * 40)
            report_lines.append(f"Success Rate: {overall_metrics.get('attack_success_rate', 0.0):.4f}")
            report_lines.append(f"L2 Distance: {overall_metrics.get('l2_distance', 0.0):.6f}")
            report_lines.append(f"L∞ Distance: {overall_metrics.get('linf_distance', 0.0):.6f}")
            
            psnr = overall_metrics.get('psnr', 'N/A')
            if psnr != 'N/A':
                report_lines.append(f"PSNR: {psnr:.2f} dB")
            else:
                report_lines.append("PSNR: N/A")
            
            lpips = overall_metrics.get('lpips', 'N/A')
            if lpips != 'N/A':
                report_lines.append(f"LPIPS: {lpips:.4f}")
            else:
                report_lines.append("LPIPS: N/A")
            
            report_lines.append("")
        
        # Comparison
        comparison = self.compare_attack_types(attack_results)
        if comparison['ranking']:
            report_lines.append("ATTACK RANKING")
            report_lines.append("-" * 40)
            for rank, (attack_type, data) in enumerate(comparison['ranking'], 1):
                report_lines.append(f"{rank}. {attack_type}: {data['value']:.4f}")
            report_lines.append("")
        
        # Model analysis
        model_analysis = self.analyze_model_performance(attack_results)
        if model_analysis:
            report_lines.append("MODEL-SPECIFIC ANALYSIS")
            report_lines.append("-" * 40)
            for model_id, data in model_analysis.items():
                report_lines.append(f"Model {model_id}:")
                avg_metrics = data.get('average_metrics', {})
                if avg_metrics:
                    report_lines.append(f"  Average Original Accuracy: {avg_metrics.get('average_original_accuracy', 0.0):.4f}")
                    report_lines.append(f"  Average Adversarial Accuracy: {avg_metrics.get('average_adversarial_accuracy', 0.0):.4f}")
                    report_lines.append(f"  Average Attack Success Rate: {avg_metrics.get('average_attack_success_rate', 0.0):.4f}")
                    report_lines.append(f"  Average Accuracy Drop: {avg_metrics.get('average_accuracy_drop', 0.0):.4f}")
                report_lines.append("")
        
        return "\n".join(report_lines)
