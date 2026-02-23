#!/usr/bin/env python3
"""
Clean CLI wrapper for fingerprint robustness attack evaluation.

This script provides a command-line interface to the modular attack evaluation system.
All core logic has been moved to appropriate modules in ./src/.
"""

import argparse
import copy
import logging
import sys
from pathlib import Path
import traceback
from typing import Any, Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.attack_runner import AttackRunner
from src.utils.device_utils import setup_device

PGD_ABLATION_ATTACKS = {"w1", "w2", "w3", "b1"}
B2_JPEG_QUALITIES = [95, 90, 85, 80, 75, 70, 65, 50]
B2_RESIZE_SCALES = [0.9, 0.8, 0.7, 0.6, 0.5]
B2_BLUR_SIGMAS = [0.2, 0.5, 1.0, 1.5]
B2_NOISE_STDS = [0.001, 0.0025, 0.005, 0.01, 0.02]


def parse_arguments():
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Run fingerprint robustness attacks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single W2 removal attack
  python run_attacks.py --data-dir ./data --fingerprint-method nataraj19 --attack-types w2 --attack-goal removal

  # Run multiple attacks in one run
  python run_attacks.py --data-dir ./data --fingerprint-method nataraj19 --attack-types w1,w2,w3,b1,b2 --attack-goal removal

  # Run multiple forgery attacks with fixed target class 1
  python run_attacks.py --data-dir ./data --fingerprint-method nataraj19 --attack-types w1,w2,w3,b1,b2 --attack-goal forgery --target-class 1

  # Run multiple forgery attacks with random targets for each image
  python run_attacks.py --data-dir ./data --fingerprint-method nataraj19 --attack-types w1,w2,w3,b1,b2 --attack-goal forgery

  # Custom attack parameters for multiple attacks
  python run_attacks.py --data-dir ./data --fingerprint-method nataraj19 --attack-types w1,w2,w3,b1,b2 --attack-goal removal \\
    --epsilon 0.05 --num-steps 30 --step-size 0.005 --test-images-per-model 50 --attack-batch-size 10

  # B2 attack with custom epsilon_linf constraint
  python run_attacks.py --data-dir ./data --fingerprint-method nataraj19 --attack-types b2 --attack-goal removal \\
    --epsilon-linf 0.05 --perturbation-types gaussian_noise blur

  # Multi-step-size evaluation for PGD-based attacks (W1, W2, W3, B1)
  python run_attacks.py --data-dir ./data --fingerprint-method nataraj19 --attack-types w1,w2,w3,b1 --attack-goal removal \\
    --step-size-list "0.005,0.01,0.02,0.05" --epsilon 0.05 --num-steps 20
        """
    )
    
    # Required arguments
    parser.add_argument("--data-dir", required=True, help="Directory containing models and default data (images/features) if --attack-data-dir is not provided")
    parser.add_argument("--attack-data-dir", required=False, default=None,
                       help="Optional directory to load input images and precomputed fingerprints from. If provided, model loading still uses --data-dir.")
    parser.add_argument("--fingerprint-method", required=True, help="Fingerprint method used")
    parser.add_argument("--attack-types", required=True, help="Comma-separated list of attack types (w1, w2, w3, b1, b2)")
    parser.add_argument("--attack-goal", required=True, choices=['removal', 'forgery'], 
                       help="Attack goal: removal (make uncertain) or forgery (make every image classified as target class)")
    
    # Optional arguments
    parser.add_argument("--target-class", type=int, 
                       help="Target class for forgery attack (0 to num_models-1). If not specified, random targets will be used for each image")
    parser.add_argument("--test-images-per-model", type=int, default=100, help="Number of test images per model")
    parser.add_argument("--image-size", type=int, default=256, help="Image size")
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, cpu, mps)")
    
    # Attack parameters
    parser.add_argument("--epsilon", type=float, default=0.05, help="Attack perturbation budget")
    parser.add_argument("--num-steps", type=int, default=20, help="Number of attack steps")
    parser.add_argument("--step-size", type=float, default=0.01, help="Attack step size (single value)")
    parser.add_argument("--step-size-list", type=str, default=None, 
                       help="Comma-separated list of step sizes to test (e.g., '0.005,0.01,0.02,0.05'). If provided, overrides --step-size")
    parser.add_argument("--attack-batch-size", type=int, default=10, help="Batch size for attack to save memory (smaller = less memory)")
    
    # Song24 specific arguments
    parser.add_argument("--real-data-path", type=str, default=None,
                       help="Path to real data for Song24 manifold estimation (optional - defaults to {data_dir}/ffhq_dataset/ffhq_real_data.pkl if not provided)")
    parser.add_argument("--cache-dir", type=str, default="cache",
                       help="Directory to cache manifold estimators for Song24 methods")
    
    # B2 attack specific arguments
    parser.add_argument("--perturbation-types", nargs="+", default=["gaussian_noise", "blur", "jpeg", "resize"],
                      choices=["gaussian_noise", "blur", "jpeg", "resize"],
                      help="Types of perturbations to use for B2 attack (only 4 types supported)")
    parser.add_argument("--perturbation-params", type=str, default="{}",
                      help='JSON string of perturbation parameters, e.g., \'{"gaussian_noise": {"std": 0.1}, "blur": {"kernel_size": 5}}\'')
    parser.add_argument("--epsilon-linf", type=float, default=0.025,
                      help="Maximum L-infinity distance for pixel changes in B2 attacks (default: 0.025)")
    
    # Image saving arguments
    parser.add_argument("--save-attacked-images", action="store_true", default=False,
                      help="Save successfully attacked images to data-dir/attacked_images/ (default: False)")
    
    # Ablation settings
    parser.add_argument("--enable-ablation", action="store_true", default=False,
                        help="Run built-in ablation sweeps over attack hyperparameters.")
    parser.add_argument("--ablation-epsilons", type=str,
                        default="0.01,0.025,0.05,0.1,0.25",
                        help="Comma-separated epsilon values for PGD-style attack ablations.")
    parser.add_argument("--ablation-output", type=str, default="logs/analysis/ablation_summary.csv",
                        help="Path to save ablation summary CSV.")
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments."""
    
    # Determine attack types
    attack_types = [at.strip().lower() for at in args.attack_types.split(',')]
    
    # Validate attack types
    valid_attack_types = ['w1', 'w2', 'w3', 'b1', 'b2']
    invalid_types = [at for at in attack_types if at not in valid_attack_types]
    if invalid_types:
        print(f"❌ Error: Invalid attack types: {invalid_types}")
        print(f"Valid types: {valid_attack_types}")
        sys.exit(1)
    
    # Parse step sizes
    if args.step_size_list:
        try:
            step_size_list = [float(x.strip()) for x in args.step_size_list.split(',')]
        except ValueError as e:
            print(f"❌ Error parsing step-size-list: {e}")
            print("Expected format: '0.005,0.01,0.02,0.05'")
            sys.exit(1)
    else:
        step_size_list = [args.step_size]
    
    return attack_types, step_size_list


def _parse_float_list_arg(text: str, flag_name: str) -> List[float]:
    values: List[float] = []
    try:
        for item in text.split(','):
            item = item.strip()
            if not item:
                continue
            values.append(float(item))
    except ValueError as e:
        print(f"❌ Error parsing {flag_name}: {e}")
        sys.exit(1)
    if not values:
        print(f"❌ Error: {flag_name} produced no numeric values.")
        sys.exit(1)
    return values


def print_configuration(args, attack_types, step_size_list):
    """Print configuration summary."""
    
    print("🎯 FINGERPRINT ROBUSTNESS ATTACK EVALUATION")
    print("=" * 60)
    print(f"Fingerprint Method: {args.fingerprint_method}")
    print(f"Attack Types: {', '.join(attack_types).upper()}")
    print(f"Attack Goal: {args.attack_goal.upper()}")
    if args.attack_goal == 'forgery':
        print("Target Class: Random (assigned per-sample at runtime)")
    print(f"Test Images per Model: {args.test_images_per_model}")
    print(f"Device: {args.device}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Number of Steps: {args.num_steps}")
    print(f"Step Sizes: {step_size_list}")
    if args.attack_data_dir:
        print(f"Attack Data Dir (images/features): {args.attack_data_dir}")
    print(f"Model/Data Dir (models, default save/load): {args.data_dir}")
    print("=" * 60)


def build_attack_params(args, step_size_list):
    """Build attack parameters dictionary."""
    
    attack_params = {
        'epsilon': args.epsilon,
        'num_steps': args.num_steps,
        'step_size': args.step_size,  # Keep for backward compatibility
        'step_size_list': step_size_list,
        'attack_goal': args.attack_goal,
        # For forgery, per-sample random targets are generated at runtime; fixed target removed
    }
    
    # Add B2-specific parameters if needed
    if 'b2' in args.attack_types.lower():
        import json
        try:
            perturbation_params = json.loads(args.perturbation_params)
        except json.JSONDecodeError:
            print(f"⚠️  Warning: Invalid perturbation_params JSON, using empty dict")
            perturbation_params = {}
        
        attack_params.update({
            'perturbation_types': args.perturbation_types,
            'perturbation_params': perturbation_params,
            'epsilon_linf': args.epsilon_linf
        })
    
    return attack_params


def build_ablation_specs(args, attack_type: str) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    attack_type = attack_type.lower()
    specs: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    if attack_type == "b2":
        if args.attack_goal != "removal":
            print("❌ Error: B2 ablation is only supported for removal attacks.")
            sys.exit(1)
        for q in B2_JPEG_QUALITIES:
            specs.append((
                f"b2-jpeg-q{q}",
                {
                    "perturbation_types": ["jpeg"],
                    "perturbation_params": {"jpeg": {"quality": q}},
                },
                {"param_name": "jpeg_quality", "param_value": q},
            ))
        for scale in B2_RESIZE_SCALES:
            specs.append((
                f"b2-resize-s{scale}",
                {
                    "perturbation_types": ["resize"],
                    "perturbation_params": {"resize": {"scale_factor": scale}},
                },
                {"param_name": "resize_scale", "param_value": scale},
            ))
        for sigma in B2_BLUR_SIGMAS:
            specs.append((
                f"b2-blur-sigma{sigma}",
                {
                    "perturbation_types": ["blur"],
                    "perturbation_params": {"blur": {"sigma": sigma, "kernel_size": 3}},
                },
                {"param_name": "blur_sigma", "param_value": sigma},
            ))
        for std in B2_NOISE_STDS:
            specs.append((
                f"b2-noise-std{std}",
                {
                    "perturbation_types": ["gaussian_noise"],
                    "perturbation_params": {"gaussian_noise": {"std": std}},
                },
                {"param_name": "gaussian_noise_std", "param_value": std},
            ))
    elif attack_type in PGD_ABLATION_ATTACKS:
        eps_list = _parse_float_list_arg(args.ablation_epsilons, "--ablation-epsilons")
        for eps in eps_list:
            specs.append((
                f"{attack_type}-eps-{eps}",
                {
                    "epsilon": eps,
                    "epsilon_linf": eps,
                },
                {"param_name": "epsilon", "param_value": eps},
            ))
    else:
        print(f"❌ Ablation not supported for attack type '{attack_type}'.")
        sys.exit(1)
    return specs


def run_ablation_mode(
    args,
    attack_runner: AttackRunner,
    attack_type: str,
    attack_params: Dict[str, Any],
    data_dir: str,
    attack_data_dir: Optional[str],
    real_data_path: Optional[str],
    cache_dir: str,
    attack_batch_size: int,
    save_attacked_images: bool,
) -> None:
    specs = build_ablation_specs(args, attack_type)
    if not specs:
        print("❌ No ablation specifications generated.")
        sys.exit(1)
    summary_rows: List[Dict[str, Any]] = []
    display_rows: List[Dict[str, Any]] = []
    for tag, overrides, meta in specs:
        print(f"\n{'-'*80}")
        print(f"ABlation configuration: {tag}")
        current_params = copy.deepcopy(attack_params)
        for key, value in overrides.items():
            if key == "perturbation_params":
                current_params[key] = value
            else:
                current_params[key] = value
        results = attack_runner.run_multiple_attacks(
            data_dir=data_dir,
            attack_data_dir=attack_data_dir,
            fingerprint_method=args.fingerprint_method,
            attack_types=[attack_type],
            num_test_images_per_model=args.test_images_per_model,
            image_size=args.image_size,
            attack_params=current_params,
            attack_batch_size=attack_batch_size,
            real_data_path=real_data_path,
            cache_dir=cache_dir,
            save_attacked_images_flag=save_attacked_images,
        )
        success_rate = None
        lpips_val = None
        psnr_val = None
        if results:
            overall = results[0].get("overall_metrics", {})
            success_rate = overall.get("attack_success_rate")
            lpips_val = overall.get("lpips")
            psnr_val = overall.get("psnr")
            sr_display = f"{success_rate:.4f}" if success_rate is not None else "N/A"
            lpips_display = "N/A" if lpips_val is None else (lpips_val if isinstance(lpips_val, str) else f"{lpips_val:.4f}")
            psnr_display = "N/A" if psnr_val is None else (psnr_val if isinstance(psnr_val, str) else f"{psnr_val:.2f}")
            print(f"➡️  Success rate for {tag}: {sr_display}")
            print(f"   LPIPS: {lpips_display}, PSNR: {psnr_display}")
        else:
            print(f"⚠️  No results for {tag}")
        row = {
            "tag": tag,
            "attack_type": attack_type,
            "attack_goal": args.attack_goal,
            "param_name": meta.get("param_name"),
            "param_value": meta.get("param_value"),
            "success_rate": success_rate,
            "lpips": lpips_val,
            "psnr": psnr_val,
        }
        summary_rows.append(row)
        display_rows.append(row)

    _write_ablation_summary(args.ablation_output, summary_rows)

    # Print a concise ablation summary table to stdout for kubectl logs
    headers = ["Tag", "Param", "Value", "ASR", "LPIPS", "PSNR"]
    table_rows = []
    for row in display_rows:
        sr = "-" if row["success_rate"] is None else f"{row['success_rate']:.4f}"
        lp = "-" if row.get("lpips") is None else (row["lpips"] if isinstance(row["lpips"], str) else f"{row['lpips']:.4f}")
        psnr = "-" if row.get("psnr") is None else (row["psnr"] if isinstance(row["psnr"], str) else f"{row['psnr']:.2f}")
        table_rows.append([
            row["tag"],
            row.get("param_name", ""),
            row.get("param_value", ""),
            sr,
            lp,
            psnr,
        ])
    print("\n📊 Ablation summary (stdout table):")
    try:
        from tabulate import tabulate
        print(tabulate(table_rows, headers=headers, tablefmt="grid"))
    except Exception:
        print(" | ".join(headers))
        print("-" * (len(" | ".join(headers))))
        for r in table_rows:
            print(" | ".join(map(str, r)))


def _write_ablation_summary(summary_path: str, rows: List[Dict[str, Any]]) -> None:
    path = Path(summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("tag,attack_type,attack_goal,param_name,param_value,success_rate,lpips,psnr\n")
        for row in rows:
            sr = "-" if row["success_rate"] is None else f"{row['success_rate']:.6f}"
            lp = "-" if row.get("lpips") is None else (row["lpips"] if isinstance(row["lpips"], str) else f"{row['lpips']:.6f}")
            psnr = "-" if row.get("psnr") is None else (row["psnr"] if isinstance(row["psnr"], str) else f"{row['psnr']:.6f}")
            f.write(
                f"{row['tag']},{row['attack_type']},{row['attack_goal']},"
                f"{row.get('param_name','')},{row.get('param_value','')},{sr},{lp},{psnr}\n"
            )
    print(f"\nAblation summary saved to {path}")


def main():
    """Main function."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Validate arguments
    attack_types, step_size_list = validate_arguments(args)
    
    # Print configuration
    print_configuration(args, attack_types, step_size_list)
    
    # Validate attack goal
    if args.attack_goal == 'forgery':
        print("ℹ️  Info: Using per-sample random targets generated at runtime.")
    
    # Setup device
    device = setup_device(args.device)
    print(f"Using device: {device}")
    
    # Build attack parameters
    attack_params = build_attack_params(args, step_size_list)
    
    # Auto-derive real_data_path from data_dir if not provided
    if args.real_data_path is None:
        real_data_path = f"{args.data_dir}/ffhq_dataset/ffhq_real_data.pkl"
        print(f"ℹ️  Auto-derived real data path: {real_data_path}")
    else:
        real_data_path = args.real_data_path
        print(f"ℹ️  Using provided real data path: {real_data_path}")
    
    try:
        attack_runner = AttackRunner(device)
        if args.enable_ablation:
            if len(attack_types) != 1:
                print("❌ Ablation mode requires specifying a single attack type.")
                sys.exit(1)
            run_ablation_mode(
                args,
                attack_runner,
                attack_types[0],
                attack_params,
                data_dir=args.data_dir,
                attack_data_dir=args.attack_data_dir,
                real_data_path=real_data_path,
                cache_dir=args.cache_dir,
                attack_batch_size=args.attack_batch_size,
                save_attacked_images=args.save_attacked_images,
            )
            return
        
        results = attack_runner.run_multiple_attacks(
            data_dir=args.data_dir,
            attack_data_dir=args.attack_data_dir,
            fingerprint_method=args.fingerprint_method,
            attack_types=attack_types,
            num_test_images_per_model=args.test_images_per_model,
            image_size=args.image_size,
            attack_params=attack_params,
            attack_batch_size=args.attack_batch_size,
            real_data_path=real_data_path,
            cache_dir=args.cache_dir,
            save_attacked_images_flag=args.save_attacked_images
        )
        
        if results:
            print("\n🎉 Attack evaluation completed successfully!")
            print(f"✅ Completed fingerprint robustness evaluation using method: {args.fingerprint_method}")
        else:
            print("\n❌ No attacks completed successfully!")
            print(f"❌ Failed fingerprint robustness evaluation using method: {args.fingerprint_method}")
            sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Attack evaluation failed: {e}")
        print(f"❌ Failed fingerprint robustness evaluation using method: {args.fingerprint_method}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
