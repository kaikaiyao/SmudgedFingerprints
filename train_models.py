#!/usr/bin/env python3
"""
Standalone script to train the three core models for fingerprint robustness framework.
Generates data on-the-fly from available generative models in src/models.

Directory Structure:
/data/
├── images/                    # Raw images (shared across fingerprint methods)
│   ├── model1/
│   ├── model2/
│   └── ...
├── features_{fingerprint_method}/  # Fingerprints for specific method
│   ├── model1/
│   ├── model2/
│   └── ...
└── models_{fingerprint_method}/    # Trained models for specific method
    ├── true_attribution_model.pth
    ├── surrogate_attribution_model.pth
    └── surrogate_extractor_model.pth

This structure allows reusing images across different fingerprint methods while keeping
fingerprints and models separate for each method.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import concurrent.futures
import multiprocessing
from functools import partial
import time

# Force-disable Hugging Face Xet backend early so all processes use standard HTTP downloads.
os.environ["HF_HUB_DISABLE_XET"] = "1"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def setup_device(device_preference: str):
    """Setup device based on preference."""
    if device_preference == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_preference

def discover_available_models():
    """Discover available generative models from src/models."""
    try:
        from src.models import ModelSampler
        available_models = [name for name in ModelSampler._registry.keys()]
        return available_models
    except ImportError as e:
        raise ImportError(f"Failed to import ModelSampler: {e}")


def create_fingerprint_extractor(fingerprint_method: str, device: str, real_data_path: str = None, cache_dir: str = "cache", data_dir: str = None):
    """Create fingerprint extractor with proper parameter handling for different methods."""
    from src.fingerprints import FingerprintExtractor
    
    # List of fingerprint methods that accept device parameter
    device_accepting_methods = {
        'wang20', 'nataraj19', 'song24', 'giudice21', 'corvi23r', 'corvi23s'
    }
    
    # Handle Song24 methods that require real_data_path
    if fingerprint_method.startswith('song24'):
        # For Song24 methods, real_data_path is optional (will auto-download if not provided)
        return FingerprintExtractor.create(
            fingerprint_method, 
            device=device,
            real_data_path=real_data_path,
            cache_dir=cache_dir,
            data_dir=data_dir
        )
    elif fingerprint_method in device_accepting_methods:
        # For methods that accept device parameter
        return FingerprintExtractor.create(fingerprint_method, device=device)
    else:
        # For methods that don't accept device parameter (like durall20, nowroozi22, etc.)
        return FingerprintExtractor.create(fingerprint_method)

def generate_data_on_the_fly(
    model_names: list,
    num_images_per_model: int,
    image_size: int,
    device: str,
    fingerprint_method: str,
    output_dir: Path,
    real_data_path: str = None,
    cache_dir: str = "cache",
    data_dir: str = None
):
    """Generate images and extract fingerprints on-the-fly."""
    # Check if this is an implicit fingerprint method
    try:
        fingerprint_extractor = create_fingerprint_extractor(fingerprint_method, device, real_data_path, cache_dir, data_dir)
        is_implicit = fingerprint_extractor.is_implicit_fingerprint
    except ImportError:
        # Fallback: assume it's not implicit if we can't check
        is_implicit = False
    
    if is_implicit:
        print(f"Generating {num_images_per_model} images from each of {len(model_names)} models (implicit fingerprint method)...")
    else:
        print(f"Generating {num_images_per_model} images from each of {len(model_names)} models...")
    
    try:
        from src.models import ModelSampler
        from src.fingerprints import FingerprintExtractor
        from src.utils.model_isolation import sort_models_by_load_order, cleanup_all_models
    except ImportError as e:
        raise ImportError(f"Failed to import required modules: {e}")
    
    # Sort models by recommended loading order to minimize conflicts
    sorted_model_names = sort_models_by_load_order(model_names)
    print(f"Loading models in recommended order: {sorted_model_names}")
    
    # Print explicit model-to-label mapping for data generation
    print(f"\n📋 MODEL-TO-LABEL MAPPING (Data Generation):")
    for i, model_name in enumerate(sorted_model_names):
        print(f"   Model {i}: {model_name}")
    print()
    
    # Create output directories
    images_dir = output_dir / "images"
    features_dir = output_dir / f"features_{fingerprint_method}"
    models_dir = output_dir / f"models_{fingerprint_method}"
    images_dir.mkdir(parents=True, exist_ok=True)
    if not is_implicit:
        features_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    all_features = []
    all_labels = []
    all_images = []
    
    for i, model_name in enumerate(sorted_model_names):
        print(f"\n--- Generating data for {model_name} ---")
        
        # Create model-specific directories
        model_images_dir = images_dir / model_name
        model_images_dir.mkdir(exist_ok=True)
        if not is_implicit:
            model_features_dir = features_dir / model_name
            model_features_dir.mkdir(exist_ok=True)
        
        try:
            # Clean up any previous model resources, except for models with persistent module dependencies
            if model_name not in ["ncsnpp-ffhq-256", "cips-ffhq-256"]:
                cleanup_all_models()
            else:
                print(f"    Skipping cleanup for {model_name} to avoid module re-initialization issues")
            
            # Create sampler
            sampler = ModelSampler.create(
                model_name=model_name,
                dataset="ffhq",
                image_size=image_size,
                device=device,
                batch_size=1
            )
            
            # Load model with isolation
            sampler.load_model()
            
            # Generate images in batches
            batch_size = min(10, num_images_per_model)  # Small batches to avoid memory issues
            total_generated = 0
            
            for batch_start in range(0, num_images_per_model, batch_size):
                current_batch_size = min(batch_size, num_images_per_model - batch_start)
                print(f"    Generating batch {batch_start//batch_size + 1}/{(num_images_per_model + batch_size - 1)//batch_size} ({current_batch_size} images)...")
                
                try:
                    # Generate batch
                    batch_imgs = sampler.sample(num_images=current_batch_size, save_to_file=False)
                    
                    # Save images
                    for j in range(current_batch_size):
                        img_idx = total_generated + j
                        img_path = model_images_dir / f"{img_idx:06d}.png"
                        img_pil = transforms.ToPILImage()(batch_imgs[j].cpu())
                        img_pil.save(img_path)
                    
                    if is_implicit:
                        # For implicit methods, we don't extract fingerprints
                        # Store for training
                        all_images.extend([img.cpu() for img in batch_imgs])
                        # Create dummy features (will be replaced by the implicit fingerprint model during training)
                        dummy_features = [torch.zeros(1) for _ in range(current_batch_size)]
                        all_features.extend(dummy_features)
                        all_labels.extend([i] * current_batch_size)
                        
                        total_generated += current_batch_size
                        print(f"      ✅ Generated and saved {current_batch_size} images (implicit fingerprint method)")
                        
                        # Clean up memory
                        del batch_imgs
                    else:
                        # Extract fingerprints for explicit methods
                        fingerprint_extractor = create_fingerprint_extractor(fingerprint_method, device, real_data_path, cache_dir, data_dir)
                        batch_fp = fingerprint_extractor.extract_fingerprint(batch_imgs)
                        
                        # Save fingerprints
                        for j in range(current_batch_size):
                            fp_idx = total_generated + j
                            fp_path = model_features_dir / f"{fp_idx:06d}.npy"
                            np.save(fp_path, batch_fp[j].cpu().numpy())
                        
                        # Store for training
                        all_images.extend([img.cpu() for img in batch_imgs])
                        all_features.extend([fp.cpu() for fp in batch_fp])
                        all_labels.extend([i] * current_batch_size)
                        
                        total_generated += current_batch_size
                        print(f"      ✅ Generated and saved {current_batch_size} images and fingerprints")
                        
                        # Clean up memory
                        del batch_imgs, batch_fp, fingerprint_extractor
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                        
                except Exception as e:
                    print(f"      ❌ Error in batch: {e}")
                    continue
            
            print(f"  ✅ Generated {total_generated} images for {model_name}")
            
        except Exception as e:
            print(f"  ❌ Failed to generate data for {model_name}: {e}")
            continue
    
    if len(all_features) == 0:
        raise RuntimeError("No data was generated successfully")
    
    # Convert to tensors
    X_features = torch.tensor(np.array(all_features), dtype=torch.float32)
    X_images = torch.stack(all_images, dim=0)
    y = torch.tensor(all_labels, dtype=torch.long)
    
    print(f"✅ Total data generated: {len(all_features)} samples")
    print(f"  Features shape: {X_features.shape}")
    print(f"  Images shape: {X_images.shape}")
    print(f"  Labels shape: {y.shape}")
    
    return X_features, X_images, y, len(sorted_model_names)

def generate_additional_data(
    model_names: list,
    num_images_per_model: int,
    image_size: int,
    device: str,
    fingerprint_method: str,
    output_dir: Path,
    existing_data: dict = None,
    real_data_path: str = None,
    cache_dir: str = "cache",
    data_dir: str = None
):
    """Generate additional images and fingerprints to reach the target count per model, or extract fingerprints from existing images."""
    # Check if this is an implicit fingerprint method
    try:
        fingerprint_extractor = create_fingerprint_extractor(fingerprint_method, device, real_data_path, cache_dir, data_dir)
        is_implicit = fingerprint_extractor.is_implicit_fingerprint
    except ImportError:
        # Fallback: assume it's not implicit if we can't check
        is_implicit = False
    
    # Check if we need to generate more images or just extract fingerprints
    need_more_images = False
    need_extract_features = False
    
    images_dir = output_dir / "images"
    features_dir = output_dir / f"features_{fingerprint_method}"
    
    # Sort models by the same order used in attacks to ensure label consistency
    try:
        from src.utils.model_isolation import sort_models_by_load_order
        sorted_model_names = sort_models_by_load_order(model_names)
        print(f"Models sorted by load order for consistent labeling: {sorted_model_names}")
    except ImportError:
        # Fallback to original order if sorting function not available
        sorted_model_names = model_names
        print(f"Using original model order for labeling: {sorted_model_names}")
    
    # Print explicit model-to-label mapping for additional data generation
    print(f"\n📋 MODEL-TO-LABEL MAPPING (Additional Data Generation):")
    for i, model_name in enumerate(sorted_model_names):
        print(f"   Model {i}: {model_name}")
    print()
    
    for model_name in sorted_model_names:
        model_images_dir = images_dir / model_name
        
        if model_images_dir.exists():
            existing_images = list(model_images_dir.glob("*.png"))
            existing_count = len(existing_images)
            
            if existing_count < num_images_per_model:
                need_more_images = True
            elif not is_implicit:
                # Only check for features if it's not an implicit method
                model_features_dir = features_dir / model_name
                if not model_features_dir.exists() or len(list(model_features_dir.glob("*.npy"))) != existing_count:
                    need_extract_features = True
    
    if need_more_images:
        print(f"Generating additional images to reach {num_images_per_model} per model...")
    elif need_extract_features:
        print(f"Extracting {fingerprint_method} fingerprints from existing images...")
    else:
        print("Processing existing data...")
    
    try:
        from src.models import ModelSampler
        from src.fingerprints import FingerprintExtractor
        from src.utils.model_isolation import sort_models_by_load_order, cleanup_all_models
    except ImportError as e:
        raise ImportError(f"Failed to import required modules: {e}")
    
    # Sort models by recommended loading order to minimize conflicts
    sorted_model_names = sort_models_by_load_order(model_names)
    print(f"Loading models in recommended order: {sorted_model_names}")
    
    # Create output directories
    images_dir = output_dir / "images"
    features_dir = output_dir / f"features_{fingerprint_method}"
    models_dir = output_dir / f"models_{fingerprint_method}"
    images_dir.mkdir(parents=True, exist_ok=True)
    if not is_implicit:
        features_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    all_features = []
    all_labels = []
    all_images = []
    
    for i, model_name in enumerate(sorted_model_names):
        print(f"\n--- Processing {model_name} ---")
        
        # Create model-specific directories
        model_images_dir = images_dir / model_name
        model_images_dir.mkdir(exist_ok=True)
        if not is_implicit:
            model_features_dir = features_dir / model_name
            model_features_dir.mkdir(exist_ok=True)
        
        # Count existing images and features
        existing_images = list(model_images_dir.glob("*.png"))
        existing_count = len(existing_images)
        
        if is_implicit:
            # For implicit methods, we don't need to check for features
            feature_count = existing_count  # Assume features exist if we have images
            print(f"  Found {existing_count} existing images (implicit fingerprint method)")
        else:
            # For explicit methods, check for features
            existing_features = list(model_features_dir.glob("*.npy"))
            feature_count = len(existing_features)
            print(f"  Found {existing_count} existing images, {feature_count} existing features")
        
        # Check if we have sufficient images
        if existing_count >= num_images_per_model:
            print(f"  ✅ Already have {existing_count} images")
            
            if is_implicit:
                # For implicit methods, just load the images directly
                print(f"  🔄 Loading existing images for implicit fingerprint method")
                model_images = []
                
                # Load existing images
                print(f"    Loading {existing_count} existing images...")
                for img_path in sorted(existing_images):
                    img_pil = Image.open(img_path)
                    img_tensor = transforms.ToTensor()(img_pil)
                    model_images.append(img_tensor)
                
                # For implicit methods, we don't extract features - we use images directly
                all_images.extend(model_images)
                # Create dummy features (will be replaced by the implicit fingerprint model during training)
                dummy_features = [torch.zeros(1) for _ in range(existing_count)]
                all_features.extend(dummy_features)
                all_labels.extend([i] * existing_count)
                print(f"  ✅ Loaded {existing_count} images for implicit fingerprint method")
            else:
                # For explicit methods, check if we need to extract fingerprints
                if feature_count == existing_count and existing_data and model_name in existing_data:
                    print(f"  ✅ Using existing features from memory")
                    all_images.extend(existing_data[model_name]['images'])
                    all_features.extend(existing_data[model_name]['features'])
                    all_labels.extend([i] * existing_count)
                else:
                    print(f"  🔄 Need to extract fingerprints from existing images")
                    # Load images and extract fingerprints
                    model_images = []
                    model_features = []
                    
                    # Load existing images
                    print(f"    Loading {existing_count} existing images...")
                    for img_path in sorted(existing_images):
                        img_pil = Image.open(img_path)
                        img_tensor = transforms.ToTensor()(img_pil)
                        model_images.append(img_tensor)
                    
                    # Extract fingerprints
                    print(f"    Extracting {fingerprint_method} fingerprints...")
                    fingerprint_extractor = create_fingerprint_extractor(fingerprint_method, device, real_data_path, cache_dir, data_dir)
                    images_tensor = torch.stack(model_images, dim=0).to(device)
                    
                    # Process in batches to avoid memory issues
                    batch_size = 10
                    for batch_start in range(0, existing_count, batch_size):
                        batch_end = min(batch_start + batch_size, existing_count)
                        batch_imgs = images_tensor[batch_start:batch_end]
                        batch_fp = fingerprint_extractor.extract_fingerprint(batch_imgs)
                        
                        # Save fingerprints
                        for j, fp in enumerate(batch_fp):
                            fp_idx = batch_start + j
                            fp_path = model_features_dir / f"{fp_idx:06d}.npy"
                            np.save(fp_path, fp.cpu().numpy())
                        
                        model_features.extend([fp.cpu() for fp in batch_fp])
                        print(f"      ✅ Processed batch {batch_start//batch_size + 1}/{(existing_count + batch_size - 1)//batch_size}")
                    
                    all_images.extend(model_images)
                    all_features.extend(model_features)
                    all_labels.extend([i] * existing_count)
                    print(f"  ✅ Extracted fingerprints for {existing_count} images")
            
            continue
        
        # Calculate how many more images we need
        additional_needed = num_images_per_model - existing_count
        print(f"  Need to generate {additional_needed} more images")
        
        # Start with existing data if available
        model_images = []
        model_features = []
        model_labels = []
        
        if existing_data and model_name in existing_data:
            print(f"  🔄 Starting with {existing_count} existing images")
            model_images.extend(existing_data[model_name]['images'])
            if is_implicit:
                # For implicit methods, create dummy features
                dummy_features = [torch.zeros(1) for _ in range(existing_count)]
                model_features.extend(dummy_features)
            else:
                model_features.extend(existing_data[model_name]['features'])
            model_labels.extend(existing_data[model_name]['labels'])
        
        try:
            # Clean up any previous model resources, except for models with persistent module dependencies
            if model_name not in ["ncsnpp-ffhq-256", "cips-ffhq-256"]:
                cleanup_all_models()
            else:
                print(f"    Skipping cleanup for {model_name} to avoid module re-initialization issues")
            
            # Create sampler
            sampler = ModelSampler.create(
                model_name=model_name,
                dataset="ffhq",
                image_size=image_size,
                device=device,
                batch_size=1
            )
            
            # Load model with isolation
            sampler.load_model()
            
            # Generate additional images in batches
            batch_size = min(10, additional_needed)  # Small batches to avoid memory issues
            total_generated = 0
            
            for batch_start in range(0, additional_needed, batch_size):
                current_batch_size = min(batch_size, additional_needed - batch_start)
                print(f"    Generating batch {batch_start//batch_size + 1}/{(additional_needed + batch_size - 1)//batch_size} ({current_batch_size} images)...")
                
                try:
                    # Generate batch
                    batch_imgs = sampler.sample(num_images=current_batch_size, save_to_file=False)
                    
                    # Save images (continue numbering from existing images)
                    for j in range(current_batch_size):
                        img_idx = existing_count + total_generated + j
                        img_path = model_images_dir / f"{img_idx:06d}.png"
                        img_pil = transforms.ToPILImage()(batch_imgs[j].cpu())
                        img_pil.save(img_path)
                    
                    if is_implicit:
                        # For implicit methods, we don't extract fingerprints
                        # Store for training (add to model-specific lists)
                        model_images.extend([img.cpu() for img in batch_imgs])
                        # Create dummy features (will be replaced by the implicit fingerprint model during training)
                        dummy_features = [torch.zeros(1) for _ in range(current_batch_size)]
                        model_features.extend(dummy_features)
                        model_labels.extend([i] * current_batch_size)
                        
                        total_generated += current_batch_size
                        print(f"      ✅ Generated and saved {current_batch_size} additional images (implicit fingerprint method)")
                        
                        # Clean up memory
                        del batch_imgs
                    else:
                        # Extract fingerprints for explicit methods
                        fingerprint_extractor = create_fingerprint_extractor(fingerprint_method, device, real_data_path, cache_dir, data_dir)
                        batch_fp = fingerprint_extractor.extract_fingerprint(batch_imgs)
                        
                        # Save fingerprints (continue numbering from existing features)
                        for j in range(current_batch_size):
                            fp_idx = existing_count + total_generated + j
                            fp_path = model_features_dir / f"{fp_idx:06d}.npy"
                            np.save(fp_path, batch_fp[j].cpu().numpy())
                        
                        # Store for training (add to model-specific lists)
                        model_images.extend([img.cpu() for img in batch_imgs])
                        model_features.extend([fp.cpu() for fp in batch_fp])
                        model_labels.extend([i] * current_batch_size)
                        
                        total_generated += current_batch_size
                        print(f"      ✅ Generated and saved {current_batch_size} additional images and fingerprints")
                        
                        # Clean up memory
                        del batch_imgs, batch_fp, fingerprint_extractor
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                        
                except Exception as e:
                    print(f"      ❌ Error in batch: {e}")
                    continue
            
            print(f"  ✅ Generated {total_generated} additional images for {model_name}")
            print(f"  📊 Total for {model_name}: {len(model_images)} images ({existing_count} existing + {total_generated} new)")
            
        except Exception as e:
            print(f"  ❌ Failed to generate additional data for {model_name}: {e}")
            # If generation failed, try to use existing data
            if existing_data and model_name in existing_data:
                print(f"  🔄 Falling back to existing data: {existing_count} images")
                model_images = existing_data[model_name]['images']
                if is_implicit:
                    # For implicit methods, create dummy features
                    dummy_features = [torch.zeros(1) for _ in range(existing_count)]
                    model_features = dummy_features
                else:
                    model_features = existing_data[model_name]['features']
                model_labels = existing_data[model_name]['labels']
            else:
                print(f"  ⚠️  No data available for {model_name}, skipping")
                continue
        
        # Add this model's data to the overall lists
        all_images.extend(model_images)
        all_features.extend(model_features)
        all_labels.extend(model_labels)
    
    if len(all_features) == 0:
        raise RuntimeError("No data was processed successfully")
    
    # Convert to tensors
    X_features = torch.tensor(np.array(all_features), dtype=torch.float32)
    X_images = torch.stack(all_images, dim=0)
    y = torch.tensor(all_labels, dtype=torch.long)
    
    print(f"✅ Total data processed: {len(all_features)} samples")
    print(f"  Features shape: {X_features.shape}")
    print(f"  Images shape: {X_images.shape}")
    print(f"  Labels shape: {y.shape}")
    
    return X_features, X_images, y, len(sorted_model_names)

def load_existing_data(
    images_dir: Path,
    features_dir: Path,
    model_names: list,
    device: str,
    num_images_per_model: int,
    max_workers: int = None,
    use_memory_mapping: bool = False,
    is_implicit: bool = False
):
    """Load existing images and fingerprints from disk with optimized performance."""
    print("Loading existing data from disk (optimized)...")
    start_time = time.time()
    
    all_features = []
    all_labels = []
    all_images = []
    
    # Pre-allocate lists with known sizes for better memory management
    total_files = 0
    for model_name in model_names:
        model_images_dir = images_dir / model_name
        if not model_images_dir.exists():
            continue

        image_files = list(model_images_dir.glob("*.png"))
        images_to_load = min(num_images_per_model, len(image_files))
        if images_to_load == 0:
            continue

        if not is_implicit:
            model_features_dir = features_dir / model_name
            if not model_features_dir.exists():
                continue
            feature_files = list(model_features_dir.glob("*.npy"))
            features_to_load = min(len(feature_files), images_to_load)
            if features_to_load < images_to_load:
                continue

        total_files += images_to_load
    
    print(f"Total files to load: {total_files}")
    
    # Use ThreadPoolExecutor for I/O bound operations (file reading)
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 8)  # Cap at 8 to avoid overwhelming the system
    print(f"Using {max_workers} parallel workers for file loading")
    
    if use_memory_mapping:
        print("Using memory mapping for faster numpy file loading")
    
    def load_single_image(img_path):
        """Load a single image and convert to tensor."""
        try:
            img_pil = Image.open(img_path)
            img_tensor = transforms.ToTensor()(img_pil)
            return img_tensor
        except Exception as e:
            print(f"Warning: Failed to load image {img_path}: {e}")
            return None
    
    def load_single_feature(feature_path):
        """Load a single feature file."""
        try:
            if use_memory_mapping:
                # Use memory mapping for faster loading of large numpy files
                feature_array = np.load(feature_path, mmap_mode='r')
                # Copy to memory to avoid issues with memory mapping
                feature_array = np.array(feature_array, copy=True)
            else:
                feature_array = np.load(feature_path)
            feature_tensor = torch.tensor(feature_array, dtype=torch.float32)
            return feature_tensor
        except Exception as e:
            print(f"Warning: Failed to load feature {feature_path}: {e}")
            return None
    
    # Progress tracking
    files_loaded = 0
    
    # Sort models by the same order used in attacks to ensure label consistency
    try:
        from src.utils.model_isolation import sort_models_by_load_order
        sorted_model_names = sort_models_by_load_order(model_names)
        print(f"Models sorted by load order for consistent labeling: {sorted_model_names}")
    except ImportError:
        # Fallback to original order if sorting function not available
        sorted_model_names = model_names
        print(f"Using original model order for labeling: {sorted_model_names}")
    
    # Print explicit model-to-label mapping for training data loading
    print(f"\n📋 MODEL-TO-LABEL MAPPING (Training Data Loading):")
    for i, model_name in enumerate(sorted_model_names):
        print(f"   Model {i}: {model_name}")
    print()
    
    for i, model_name in enumerate(sorted_model_names):
        model_start_time = time.time()
        model_images_dir = images_dir / model_name
        
        if not model_images_dir.exists():
            print(f"  ⚠️  Missing images for {model_name} in {images_dir}, skipping")
            continue
            
        # Get file lists more efficiently
        image_files = sorted(list(model_images_dir.glob("*.png")))
        
        if len(image_files) == 0:
            print(f"  ⚠️  No images found for model {model_name} in {model_images_dir}, skipping")
            continue
        
        # Limit to the requested number of images per model
        images_to_load = min(num_images_per_model, len(image_files))
        image_files = image_files[:images_to_load]
        
        if is_implicit:
            # For implicit methods, we only need images
            print(f"  Loading {images_to_load} images for {model_name} (implicit fingerprint method, {len(image_files)} available)...")
            
            # Load images in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all image loading tasks
                image_futures = [executor.submit(load_single_image, img_path) for img_path in image_files]
            
            # Collect results as they complete
            model_images = []
            
            # Process images
            for future in concurrent.futures.as_completed(image_futures):
                result = future.result()
                if result is not None:
                    model_images.append(result)
            
            # For implicit methods, we don't need features - the model works directly on images
            # We'll create placeholder features that won't be used
            model_features = []
        else:
            # For explicit methods, we need both images and features
            model_features_dir = features_dir / model_name
            
            if not model_features_dir.exists():
                print(f"  ⚠️  No features found for model {model_name} in {features_dir}, skipping")
                continue
                
            feature_files = sorted(list(model_features_dir.glob("*.npy")))
            
            # Limit feature files to match the number of image files we're loading
            features_to_load = min(len(feature_files), len(image_files))
            feature_files = feature_files[:features_to_load]
            
            if len(image_files) != len(feature_files):
                print(f"  ⚠️  Mismatch for {model_name}: {len(image_files)} images vs {len(feature_files)} features, skipping")
                continue
            
            print(f"  Loading {len(image_files)} files for {model_name} (explicit fingerprint method, {len(image_files)} images, {len(feature_files)} features)...")
            
            # Load images and features in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all image loading tasks
                image_futures = [executor.submit(load_single_image, img_path) for img_path in image_files]
                
                # Submit all feature loading tasks
                feature_futures = [executor.submit(load_single_feature, feature_path) for feature_path in feature_files]
            
            # Collect results as they complete
            model_images = []
            model_features = []
            
            # Process images
            for future in concurrent.futures.as_completed(image_futures):
                result = future.result()
                if result is not None:
                    model_images.append(result)
            
            # Process features for explicit methods
            for future in concurrent.futures.as_completed(feature_futures):
                result = future.result()
                if result is not None:
                    model_features.append(result)
        
        # Verify we got all the data
        if is_implicit:
            if len(model_images) != images_to_load:
                print(f"Warning: Only loaded {len(model_images)}/{images_to_load} images for {model_name}")
        else:
            if len(model_images) != images_to_load or len(model_features) != images_to_load:
                print(f"Warning: Only loaded {len(model_images)}/{images_to_load} images and {len(model_features)}/{images_to_load} features for {model_name}")
        
        # Extend the main lists
        all_images.extend(model_images)
        all_features.extend(model_features)
        all_labels.extend([i] * len(model_images))
        
        # Update progress
        files_loaded += len(model_images)
        progress = (files_loaded / total_files) * 100 if total_files > 0 else 0
        
        model_time = time.time() - model_start_time
        print(f"  ✅ Loaded {len(model_images)} images and {len(model_features)} features for {model_name} in {model_time:.2f}s")
        print(f"  📊 Progress: {files_loaded}/{total_files} files ({progress:.1f}%)")
    
    if len(all_images) == 0:
        raise RuntimeError("No existing data available to load")

    # Convert to tensors more efficiently
    print("Converting to tensors...")
    tensor_start_time = time.time()
    
    # Use torch.stack for better memory efficiency
    X_images = torch.stack(all_images, dim=0)
    y = torch.tensor(all_labels, dtype=torch.long)
    
    # Handle features based on whether this is an implicit method
    if is_implicit:
        # For implicit methods, we don't need features - create a placeholder
        X_features = torch.zeros(len(all_images), 1)  # Placeholder that won't be used
    else:
        # For explicit methods, stack the actual features
        X_features = torch.stack(all_features, dim=0)
    
    tensor_time = time.time() - tensor_start_time
    total_time = time.time() - start_time
    
    print(f"✅ Total data loaded: {len(all_features)} samples in {total_time:.2f}s")
    print(f"  Features shape: {X_features.shape}")
    print(f"  Images shape: {X_images.shape}")
    print(f"  Labels shape: {y.shape}")
    print(f"  Tensor conversion time: {tensor_time:.2f}s")
    print(f"  Average loading speed: {total_files/total_time:.1f} files/second")
    
    # Print label statistics
    print(f"  Label range: {y.min().item()} to {y.max().item()}")
    print(f"  Label distribution: {torch.bincount(y).tolist()}")
    print(f"  Number of models: {len(sorted_model_names)}")
    
    return X_features, X_images, y, len(sorted_model_names)

def train_models(
    fingerprint_method: str,
    output_path: str,
    # Separate hyperparameters for each model
    true_attribution_epochs: int = 30,
    true_attribution_batch_size: int = 32,
    true_attribution_lr: float = 1e-4,
    
    surrogate_attribution_epochs: int = 30,
    surrogate_attribution_batch_size: int = 32,
    surrogate_attribution_lr: float = 1e-4,
    
    surrogate_extractor_epochs: int = 30,
    surrogate_extractor_batch_size: int = 32,
    surrogate_extractor_lr: float = 1e-4,
    
    learning_rate: float = 1e-4,  # Keep for backward compatibility
    image_size: int = 256,
    num_images_per_model: int = 100,
    device_preference: str = "auto",
    max_workers: int = None,
    use_memory_mapping: bool = False,
    # Song24 specific parameters
    real_data_path: str = None,
    cache_dir: str = "cache",
    data_dir: str = None
):
    """Train all three models and save them."""
    
    # Setup device
    device = setup_device(device_preference)
    print(f"Using device: {device}")
    
    # Discover available models
    model_names = discover_available_models()
    if not model_names:
        raise RuntimeError("No generative models found in src/models")
    
    print(f"Found {len(model_names)} models: {model_names}")
    
    # Create output directory
    output_dir = Path(output_path)
    os.environ["DATA_DIR"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if data already exists
    images_dir = output_dir / "images"
    features_dir = output_dir / f"features_{fingerprint_method}"
    models_dir = output_dir / f"models_{fingerprint_method}"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if this is an implicit fingerprint method
    try:
        fingerprint_extractor = create_fingerprint_extractor(fingerprint_method, device, real_data_path, cache_dir, data_dir)
        is_implicit = fingerprint_extractor.is_implicit_fingerprint
    except ImportError:
        # Fallback: assume it's not implicit if we can't check
        is_implicit = False
    
    # Check if raw images exist (can be reused across fingerprint methods)
    if images_dir.exists():
        # Check existing data and determine what needs to be generated
        print("📁 Found existing data directories, checking contents...")
        
        existing_data_info = {}
        needs_additional_generation = False
        
        # Sort models by the same order used in attacks to ensure label consistency
        try:
            from src.utils.model_isolation import sort_models_by_load_order
            sorted_model_names = sort_models_by_load_order(model_names)
            print(f"Models sorted by load order for consistent labeling: {sorted_model_names}")
        except ImportError:
            # Fallback to original order if sorting function not available
            sorted_model_names = model_names
            print(f"Using original model order for labeling: {sorted_model_names}")
        
        # Print explicit model-to-label mapping for training data checking
        print(f"\n📋 MODEL-TO-LABEL MAPPING (Training Data Checking):")
        for i, model_name in enumerate(sorted_model_names):
            print(f"   Model {i}: {model_name}")
        print()
        
        for model_name in sorted_model_names:
            model_images_dir = images_dir / model_name
            
            # Check if images exist for this model
            if not model_images_dir.exists():
                print(f"  ⚠️  Missing image directory for {model_name}, will generate all data")
                needs_additional_generation = True
                continue
                
            # Count existing image files
            image_files = list(model_images_dir.glob("*.png"))
            existing_count = len(image_files)
            
            # For implicit fingerprint methods, we don't need to check for features
            if is_implicit:
                features_exist = True  # Always true for implicit methods since we don't extract features
            else:
                # Check if features exist for this model and fingerprint method
                model_features_dir = features_dir / model_name
                features_exist = model_features_dir.exists()
                if features_exist:
                    feature_files = list(model_features_dir.glob("*.npy"))
                    feature_count = len(feature_files)
                    if feature_count != existing_count:
                        print(f"  ⚠️  Mismatch in {model_name}: {existing_count} images vs {feature_count} features, will regenerate features")
                        features_exist = False
            
            existing_data_info[model_name] = {
                'count': existing_count,
                'target': num_images_per_model,
                'needs_more': existing_count < num_images_per_model,
                'features_exist': features_exist
            }
            
            if existing_count < num_images_per_model:
                needs_additional_generation = True
                print(f"  📊 {model_name}: {existing_count}/{num_images_per_model} images (need {num_images_per_model - existing_count} more)")
            elif not features_exist and not is_implicit:
                print(f"  🔄 {model_name}: {existing_count} images exist, need to extract {fingerprint_method} fingerprints")
                needs_additional_generation = True
            else:
                print(f"  ✅ {model_name}: {existing_count}/{num_images_per_model} images (sufficient)")
        
        if needs_additional_generation:
            # Check if we need to generate more images or just extract fingerprints
            need_more_images = any(info['needs_more'] for info in existing_data_info.values())
            need_extract_features = any(not info['features_exist'] for info in existing_data_info.values()) and not is_implicit
            
            if need_more_images:
                print(f"\n🔄 Some models need additional images to reach {num_images_per_model} per model")
                print("Generating additional data while preserving existing data...")
            elif need_extract_features:
                print(f"\n🔄 Need to extract {fingerprint_method} fingerprints from existing images")
                print("Loading existing images and extracting fingerprints...")
            else:
                print(f"\n🔄 Processing existing data...")
            
            # Load existing data first (if features exist or if it's an implicit method)
            existing_data = {}
            try:
                # Only try to load existing data if we have features or if it's an implicit method
                if is_implicit or any(info['features_exist'] for info in existing_data_info.values()):
                    X_features, X_images, y, num_models = load_existing_data(
                        images_dir, features_dir, sorted_model_names, device, num_images_per_model,
                        max_workers=max_workers, use_memory_mapping=use_memory_mapping,
                        is_implicit=is_implicit
                    )
                    
                    # Organize existing data by model for combination
                    start_idx = 0
                    for model_name in sorted_model_names:
                        if model_name in existing_data_info:
                            count = existing_data_info[model_name]['count']
                            features_exist = existing_data_info[model_name]['features_exist']
                            if count > 0 and (features_exist or is_implicit):
                                existing_data[model_name] = {
                                    'images': X_images[start_idx:start_idx + count],
                                    'features': X_features[start_idx:start_idx + count],
                                    'labels': y[start_idx:start_idx + count]
                                }
                            start_idx += count
                
                # Generate additional data and combine (this will extract fingerprints from existing images if needed)
                X_features, X_images, y, num_models = generate_additional_data(
                    sorted_model_names, num_images_per_model, image_size, device,
                    fingerprint_method, output_dir, existing_data, real_data_path, cache_dir, data_dir
                )
                
            except Exception as e:
                print(f"⚠️  Failed to load existing data: {e}")
                print("Falling back to generating all data from scratch...")
                X_features, X_images, y, num_models = generate_data_on_the_fly(
                    sorted_model_names, num_images_per_model, image_size, device,
                    fingerprint_method, output_dir, real_data_path, cache_dir, data_dir
                )
        else:
            print(f"✅ All models have sufficient data: {num_images_per_model} images per model")
            
            # Check if we need to extract fingerprints from existing images
            need_extract_features = any(not info['features_exist'] for info in existing_data_info.values()) and not is_implicit
            
            if need_extract_features:
                print("🔄 Need to extract fingerprints from existing images")
                print("Loading existing images and extracting fingerprints...")
                
                # Use generate_additional_data to extract fingerprints from existing images
                X_features, X_images, y, num_models = generate_additional_data(
                    sorted_model_names, num_images_per_model, image_size, device,
                    fingerprint_method, output_dir, {}, real_data_path, cache_dir, data_dir
                )
            else:
                print("Loading existing data...")
                
                # Load existing data
                X_features, X_images, y, num_models = load_existing_data(
                    images_dir, features_dir, sorted_model_names, device, num_images_per_model,
                    max_workers=max_workers, use_memory_mapping=use_memory_mapping,
                    is_implicit=is_implicit
                )
    else:
        print("📁 No existing data found. Generating new data...")
        # Sort models by the same order used in attacks to ensure label consistency
        try:
            from src.utils.model_isolation import sort_models_by_load_order
            sorted_model_names = sort_models_by_load_order(model_names)
            print(f"Models sorted by load order for consistent labeling: {sorted_model_names}")
        except ImportError:
            # Fallback to original order if sorting function not available
            sorted_model_names = model_names
            print(f"Using original model order for labeling: {sorted_model_names}")
        
        # Print explicit model-to-label mapping for fresh data generation
        print(f"\n📋 MODEL-TO-LABEL MAPPING (Fresh Data Generation):")
        for i, model_name in enumerate(sorted_model_names):
            print(f"   Model {i}: {model_name}")
        print()
        
        X_features, X_images, y, num_models = generate_data_on_the_fly(
            sorted_model_names, num_images_per_model, image_size, device, 
            fingerprint_method, output_dir, real_data_path, cache_dir, data_dir
        )
    
    # Move data to device
    X_features = X_features.to(device)
    X_images = X_images.to(device)
    y = y.to(device)
    
    # Get feature dimension from actual data
    if is_implicit:
        # For implicit methods, we don't use features - the model works directly on images
        feature_dim = 3 * image_size * image_size  # RGB images
        print(f"Feature dimension: {feature_dim} (RGB images for implicit method)")
    else:
        # For multi-dimensional features (e.g., Nataraj19's 3x256x256), flatten them
        feature_dim = X_features.numel() // X_features.shape[0]  # Total elements per sample
        print(f"Feature dimension: {feature_dim} (flattened from {tuple(X_features.shape[1:])})")
    
    print(f"Number of classes (models): {num_models}")
    
    # Train models
    results = {}
    
    try:
        from src.trainers.true_attribution import TrueAttributionModelTrainer
        from src.trainers.surrogate_attribution import SurrogateAttributionModelTrainer
        from src.trainers.surrogate_extractor import SurrogateExtractorTrainer
        from src.trainers.implicit_fingerprint import ImplicitFingerprintTrainer
        
        # Check if this is an implicit fingerprint method
        fingerprint_extractor = create_fingerprint_extractor(fingerprint_method, device, real_data_path, cache_dir, data_dir)
        is_implicit = fingerprint_extractor.is_implicit_fingerprint
        
        if is_implicit:
            print(f"\n--- {fingerprint_method} is an implicit fingerprint method ---")
            print("Training the fingerprinting network itself as the attribution model")
            
            # For implicit fingerprint methods, we train the network directly
            print(f"\n--- Training Implicit Fingerprint Model ---")
            print(f"  Epochs: {true_attribution_epochs}, Batch Size: {true_attribution_batch_size}, LR: {true_attribution_lr}")
            
            implicit_trainer = ImplicitFingerprintTrainer(
                fingerprint_method=fingerprint_method,
                dataset="ffhq",
                image_size=image_size,
                device=device,
                batch_size=true_attribution_batch_size,
                num_epochs=true_attribution_epochs,
                learning_rate=true_attribution_lr,
                patience=999999  # Disable early stopping
            )
            
            # Override the save path to use our custom output directory
            true_attribution_model_path = models_dir / "true_attribution_model.pth"
            implicit_trainer.get_model_save_path = lambda **kwargs: true_attribution_model_path
            
            # Train the implicit fingerprint model
            implicit_results = implicit_trainer.train(output_dir)
            
            results['implicit_fingerprint'] = {
                'model': f'Implicit Fingerprint Model ({fingerprint_method})',
                'best_val_accuracy': implicit_results['best_val_accuracy'],
                'final_val_metrics': implicit_results['final_val_metrics'],
                'num_epochs_trained': implicit_results['num_epochs_trained'],
                'model_names': implicit_results['model_names'],
                'config': f"E:{true_attribution_epochs}, BS:{true_attribution_batch_size}, LR:{true_attribution_lr}"
            }
            
            print(f"✅ Implicit fingerprint model training completed!")
            print(f"Best validation accuracy: {implicit_results['best_val_accuracy']:.2f}")
            
            # For implicit methods, we still need a surrogate attribution model for B1 attacks
            # The implicit fingerprint model serves as the true attribution model (h)
            # But we need a separate surrogate classifier (h_s) for black-box attacks
            print(f"\n--- Training Surrogate Attribution Model (h_s) for implicit fingerprint method ---")
            print(f"  Epochs: {surrogate_attribution_epochs}, Batch Size: {surrogate_attribution_batch_size}, LR: {surrogate_attribution_lr}")
            
            surrogate_attr_trainer = SurrogateAttributionModelTrainer(
                fingerprint_method=fingerprint_method,
                dataset="ffhq",
                image_size=image_size,
                device=device,
                batch_size=surrogate_attribution_batch_size,
                num_epochs=surrogate_attribution_epochs,
                learning_rate=surrogate_attribution_lr,
                patience=999999
            )
            
            # Override the save path to use our custom output directory
            surrogate_attr_model_path = models_dir / "surrogate_attribution_model.pth"
            surrogate_attr_trainer.get_model_save_path = lambda **kwargs: surrogate_attr_model_path
            
            # For implicit fingerprint methods, the surrogate attribution model should also use images directly
            # We'll create a simple data loader from the existing data
            from torch.utils.data import TensorDataset, DataLoader
            
            # Create dataset and dataloader
            dataset = TensorDataset(X_images, y)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=surrogate_attribution_batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=surrogate_attribution_batch_size, shuffle=False)
            
            # Train surrogate attribution model using the base trainer's training loop
            surrogate_attr_trainer.train_loader = train_loader
            surrogate_attr_trainer.val_loader = val_loader
            surrogate_attr_trainer.num_classes = num_models
            
            # Build model with correct number of classes
            surrogate_attr_trainer.model = surrogate_attr_trainer.build_model(
                input_dim=3 * image_size * image_size,  # RGB images
                output_dim=num_models
            )
            surrogate_attr_trainer.model.to(device)
            
            # Train the model
            surrogate_attr_trainer.train(X_images, y)
            
            # Save model manually to avoid pickling issues
            model_data = {
                'model_state_dict': surrogate_attr_trainer.model.state_dict(),
                'trainer_name': surrogate_attr_trainer.trainer_name,
                'dataset': 'ffhq',
                'image_size': image_size,
                'device': device,
                'train_losses': surrogate_attr_trainer.train_losses,
                'val_losses': surrogate_attr_trainer.val_losses,
                'train_metrics': surrogate_attr_trainer.train_metrics,
                'val_metrics': surrogate_attr_trainer.val_metrics,
                'num_classes': num_models,
                'training_config': {
                    'epochs': surrogate_attribution_epochs,
                    'batch_size': surrogate_attribution_batch_size,
                    'learning_rate': surrogate_attribution_lr
                }
            }
            torch.save(model_data, surrogate_attr_model_path)
            print(f"✅ Surrogate attribution model saved to {surrogate_attr_model_path}")
            
            # Store results
            # Find epoch with best validation loss (this is our checkpoint epoch)
            checkpoint_epoch = surrogate_attr_trainer.val_losses.index(min(surrogate_attr_trainer.val_losses)) if surrogate_attr_trainer.val_losses else 0
            
            results['surrogate_attribution'] = {
                'model': 'Surrogate Attribution Model (h_s)',
                'checkpoint_epoch': checkpoint_epoch + 1,  # +1 for 1-based epoch numbering
                'val_loss': surrogate_attr_trainer.val_losses[checkpoint_epoch] if surrogate_attr_trainer.val_losses else 'N/A',
                'val_acc': surrogate_attr_trainer.val_metrics[checkpoint_epoch]['accuracy'] if surrogate_attr_trainer.val_metrics else 'N/A',
                'train_loss': surrogate_attr_trainer.train_losses[checkpoint_epoch] if surrogate_attr_trainer.train_losses else 'N/A',
                'train_acc': surrogate_attr_trainer.train_metrics[checkpoint_epoch]['accuracy'] if surrogate_attr_trainer.train_metrics else 'N/A',
                'total_epochs': len(surrogate_attr_trainer.train_losses),
                'config': f"E:{surrogate_attribution_epochs}, BS:{surrogate_attribution_batch_size}, LR:{surrogate_attribution_lr}"
            }
            
            print(f"✅ Surrogate attribution model training completed!")
            print(f"Best validation accuracy: {surrogate_attr_trainer.val_metrics[checkpoint_epoch]['accuracy']:.2f}")
            
            # For implicit methods, we don't need a surrogate extractor (φ_s)
            # The implicit fingerprint model itself serves as the true attribution model
            print("\n--- Skipping surrogate extractor for implicit fingerprint method ---")
            print("The implicit fingerprint model itself serves as the true attribution model")
            
            return results
        
        # 1. Train True Attribution Model (h)
        print("\n--- Training True Attribution Model (h) ---")
        print(f"  Epochs: {true_attribution_epochs}, Batch Size: {true_attribution_batch_size}, LR: {true_attribution_lr}")
        
        true_trainer = TrueAttributionModelTrainer(
            fingerprint_method=fingerprint_method,
            dataset="ffhq",
            image_size=image_size,
            device=device,
            batch_size=true_attribution_batch_size,
            num_epochs=true_attribution_epochs,
            learning_rate=true_attribution_lr,
            patience=999999  # Disable early stopping
        )
        
        # Override the save path to use our custom output directory
        true_model_path = models_dir / "true_attribution_model.pth"
        true_trainer.get_model_save_path = lambda **kwargs: true_model_path
        
        # Mock prepare_training_data method
        true_trainer.prepare_training_data = lambda *args, **kwargs: (X_features, y, {f"model_{i}": i for i in range(num_models)})
        
        # Flatten features if they are multi-dimensional (e.g., Nataraj19's 3x256x256)
        if not is_implicit and X_features.dim() > 2:
            X_features = X_features.view(X_features.shape[0], -1)
            print(f"Flattened features from {X_features.shape} to {X_features.shape}")

        # Train
        true_trainer.train(X_features, y)
        
        # Save model manually to avoid pickling issues
        model_data = {
            'model_state_dict': true_trainer.model.state_dict(),
            'trainer_name': true_trainer.trainer_name,
            'fingerprint_method': fingerprint_method,
            'dataset': 'ffhq',
            'image_size': image_size,
            'device': device,
            'train_losses': true_trainer.train_losses,
            'val_losses': true_trainer.val_losses,
            'train_metrics': true_trainer.train_metrics,
            'val_metrics': true_trainer.val_metrics,
            'input_dim': feature_dim,
            'num_classes': num_models,
            'training_config': {
                'epochs': true_attribution_epochs,
                'batch_size': true_attribution_batch_size,
                'learning_rate': true_attribution_lr
            }
        }
        torch.save(model_data, true_model_path)
        print(f"✅ True attribution model saved to {true_model_path}")
        
        # Store results
        # Find epoch with best validation loss (this is our checkpoint epoch)
        checkpoint_epoch = true_trainer.val_losses.index(min(true_trainer.val_losses)) if true_trainer.val_losses else 0
        
        results['true_attribution'] = {
            'model': 'True Attribution Model (h)',
            'checkpoint_epoch': checkpoint_epoch + 1,  # +1 for 1-based epoch numbering
            'val_loss': true_trainer.val_losses[checkpoint_epoch] if true_trainer.val_losses else 'N/A',
            'val_acc': true_trainer.val_metrics[checkpoint_epoch]['accuracy'] if true_trainer.val_metrics else 'N/A',
            'train_loss': true_trainer.train_losses[checkpoint_epoch] if true_trainer.train_losses else 'N/A',
            'train_acc': true_trainer.train_metrics[checkpoint_epoch]['accuracy'] if true_trainer.train_metrics else 'N/A',
            'total_epochs': len(true_trainer.train_losses),
            'config': f"E:{true_attribution_epochs}, BS:{true_attribution_batch_size}, LR:{true_attribution_lr}"
        }
        
        # 2. Train Surrogate Attribution Model (h_s)
        print("\n--- Training Surrogate Attribution Model (h_s) ---")
        print(f"  Epochs: {surrogate_attribution_epochs}, Batch Size: {surrogate_attribution_batch_size}, LR: {surrogate_attribution_lr}")
        
        surrogate_attr_trainer = SurrogateAttributionModelTrainer(
            fingerprint_method=fingerprint_method,
            dataset="ffhq",
            image_size=image_size,
            device=device,
            batch_size=surrogate_attribution_batch_size,
            num_epochs=surrogate_attribution_epochs,
            learning_rate=surrogate_attribution_lr,
            patience=999999
        )
        
        # Override the save path to use our custom output directory
        surrogate_attr_model_path = models_dir / "surrogate_attribution_model.pth"
        surrogate_attr_trainer.get_model_save_path = lambda **kwargs: surrogate_attr_model_path
        
        # Mock prepare_training_data method
        surrogate_attr_trainer.prepare_training_data = lambda *args, **kwargs: (X_images, y, {f"model_{i}": i for i in range(num_models)})
        
        # Train
        surrogate_attr_trainer.train(X_images, y)
        
        # Save model manually to avoid pickling issues
        model_data = {
            'model_state_dict': surrogate_attr_trainer.model.state_dict(),
            'trainer_name': surrogate_attr_trainer.trainer_name,
            'dataset': 'ffhq',
            'image_size': image_size,
            'device': device,
            'train_losses': surrogate_attr_trainer.train_losses,
            'val_losses': surrogate_attr_trainer.val_losses,
            'train_metrics': surrogate_attr_trainer.train_metrics,
            'val_metrics': surrogate_attr_trainer.val_metrics,
            'num_classes': num_models,
            'training_config': {
                'epochs': surrogate_attribution_epochs,
                'batch_size': surrogate_attribution_batch_size,
                'learning_rate': surrogate_attribution_lr
            }
        }
        torch.save(model_data, surrogate_attr_model_path)
        print(f"✅ Surrogate attribution model saved to {surrogate_attr_model_path}")
        
        # Store results
        # Find epoch with best validation loss (this is our checkpoint epoch)
        checkpoint_epoch = surrogate_attr_trainer.val_losses.index(min(surrogate_attr_trainer.val_losses)) if surrogate_attr_trainer.val_losses else 0
        
        results['surrogate_attribution'] = {
            'model': 'Surrogate Attribution Model (h_s)',
            'checkpoint_epoch': checkpoint_epoch + 1,  # +1 for 1-based epoch numbering
            'val_loss': surrogate_attr_trainer.val_losses[checkpoint_epoch] if surrogate_attr_trainer.val_losses else 'N/A',
            'val_acc': surrogate_attr_trainer.val_metrics[checkpoint_epoch]['accuracy'] if surrogate_attr_trainer.val_metrics else 'N/A',
            'train_loss': surrogate_attr_trainer.train_losses[checkpoint_epoch] if surrogate_attr_trainer.train_losses else 'N/A',
            'train_acc': surrogate_attr_trainer.train_metrics[checkpoint_epoch]['accuracy'] if surrogate_attr_trainer.train_metrics else 'N/A',
            'total_epochs': len(surrogate_attr_trainer.train_losses),
            'config': f"E:{surrogate_attribution_epochs}, BS:{surrogate_attribution_batch_size}, LR:{surrogate_attribution_lr}"
        }
        
        # 3. Train Surrogate Extractor (φ_s)
        if is_implicit:
            print("\n--- Skipping Surrogate Extractor (φ_s) for implicit fingerprint method ---")
            print("For implicit fingerprint methods, the model itself serves as the attribution model")
            print("No separate surrogate extractor is needed")
            
            # Create a placeholder result for the surrogate extractor
            results['surrogate_extractor'] = {
                'model': 'Surrogate Extractor (φ_s) - SKIPPED',
                'checkpoint_epoch': 'N/A',
                'val_loss': 'N/A',
                'val_acc': 'N/A (implicit method)',
                'train_loss': 'N/A',
                'train_acc': 'N/A (implicit method)',
                'total_epochs': 0,
                'config': 'SKIPPED - implicit fingerprint method'
            }
        else:
            print(f"\n--- Training Surrogate Extractor (φ_s) ---")
            print(f"  Epochs: {surrogate_extractor_epochs}, Batch Size: {surrogate_extractor_batch_size}, LR: {surrogate_extractor_lr}")
            
            surrogate_extractor_trainer = SurrogateExtractorTrainer(
                fingerprint_method=fingerprint_method,
                dataset="ffhq",
                image_size=image_size,
                device=device,
                num_epochs=surrogate_extractor_epochs,
                batch_size=surrogate_extractor_batch_size,
                learning_rate=surrogate_extractor_lr,
                patience=999999
            )
            
            # Override the save path to use our custom output directory
            surrogate_extractor_model_path = models_dir / "surrogate_extractor_model.pth"
            surrogate_extractor_trainer.get_model_save_path = lambda **kwargs: surrogate_extractor_model_path
            
            # Mock prepare_training_data method
            surrogate_extractor_trainer.prepare_training_data = lambda *args, **kwargs: (X_images, X_features, {f"model_{i}": i for i in range(num_models)})
            
            # Train
            surrogate_extractor_trainer.train(X_images, X_features)
            
            # Save model manually to avoid pickling issues
            model_data = {
                'model_state_dict': surrogate_extractor_trainer.model.state_dict(),
                'trainer_name': surrogate_extractor_trainer.trainer_name,
                'fingerprint_method': fingerprint_method,
                'dataset': 'ffhq',
                'image_size': image_size,
                'device': device,
                'train_losses': surrogate_extractor_trainer.train_losses,
                'val_losses': surrogate_extractor_trainer.val_losses,
                'train_metrics': surrogate_extractor_trainer.train_metrics,
                'val_metrics': surrogate_extractor_trainer.val_metrics,
                'input_channels': 3,
                'output_features': feature_dim,
                'training_config': {
                    'epochs': surrogate_extractor_epochs,
                    'batch_size': surrogate_extractor_batch_size,
                    'learning_rate': surrogate_extractor_lr
                }
            }
            torch.save(model_data, surrogate_extractor_model_path)
            print(f"✅ Surrogate extractor model saved to {surrogate_extractor_model_path}")
            
            # Store results
            # Find epoch with best validation loss (this is our checkpoint epoch)
            checkpoint_epoch = surrogate_extractor_trainer.val_losses.index(min(surrogate_extractor_trainer.val_losses)) if surrogate_extractor_trainer.val_losses else 0
            
            results['surrogate_extractor'] = {
                'model': 'Surrogate Extractor (φ_s)',
                'checkpoint_epoch': checkpoint_epoch + 1,  # +1 for 1-based epoch numbering
                'val_loss': surrogate_extractor_trainer.val_losses[checkpoint_epoch] if surrogate_extractor_trainer.val_losses else 'N/A',
                'val_acc': f"MSE: {surrogate_extractor_trainer.val_metrics[checkpoint_epoch]['mse']:.6f}" if surrogate_extractor_trainer.val_metrics else 'N/A',
                'train_loss': surrogate_extractor_trainer.train_losses[checkpoint_epoch] if surrogate_extractor_trainer.train_losses else 'N/A',
                'train_acc': f"MSE: {surrogate_extractor_trainer.train_metrics[checkpoint_epoch]['mse']:.6f}" if surrogate_extractor_trainer.train_metrics else 'N/A',
                'total_epochs': len(surrogate_extractor_trainer.train_losses),
                'config': f"E:{surrogate_extractor_epochs}, BS:{surrogate_extractor_batch_size}, LR:{surrogate_extractor_lr}"
            }
        

        
        # Print summary table
        print("\n" + "="*160)
        print("TRAINING SUMMARY TABLE")
        print("="*160)
        
        summary_data = [
            results['true_attribution'],
            results['surrogate_attribution'],
            results['surrogate_extractor']
        ]
        
        # Print table header
        header = (
            f"{'Model':<35} {'Config':<25} "
            f"{'Checkpoint':<10} {'Val Loss':<15} {'Val Metric':<20} "
            f"{'Train Loss':<15} {'Train Metric':<20} "
            f"{'Total Epochs':<12}"
        )
        print(header)
        print("-" * 160)
        
        # Print each row
        for result in summary_data:
            model_name = result['model']
            config = result['config']
            
            # Get metrics from checkpoint epoch
            checkpoint_epoch = str(result['checkpoint_epoch'])
            val_loss = f"{result['val_loss']:.6f}" if isinstance(result['val_loss'], float) else str(result['val_loss'])
            train_loss = f"{result['train_loss']:.6f}" if isinstance(result['train_loss'], float) else str(result['train_loss'])
            total_epochs = result['total_epochs']
            
            # Handle metrics differently for surrogate extractor vs classification models
            if 'Surrogate Extractor' in model_name:
                val_metric = result['val_acc']  # Already formatted as "MSE: X.XXXXXX"
                train_metric = result['train_acc']  # Already formatted as "MSE: X.XXXXXX"
            else:
                val_metric = f"Acc: {result['val_acc']:.4f}" if isinstance(result['val_acc'], float) else str(result['val_acc'])
                train_metric = f"Acc: {result['train_acc']:.4f}" if isinstance(result['train_acc'], float) else str(result['train_acc'])
            
            row = (
                f"{model_name:<35} {config:<25} "
                f"Epoch {checkpoint_epoch:<4} {val_loss:<15} {val_metric:<20} "
                f"{train_loss:<15} {train_metric:<20} "
                f"{total_epochs:<12}"
            )
            print(row)
        
        print("-" * 160)
        print("✅ All models trained and saved successfully!")
        

        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Train fingerprint robustness models with separate hyperparameters for each model type",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models with default hyperparameters
  python train_models.py --fingerprint-method nataraj19 --data-dir ./data
  
  # Train with different epochs for each model
  python train_models.py --fingerprint-method nataraj19 --data-dir ./data \\
    --true-attribution-epochs 50 --surrogate-attribution-epochs 40 --surrogate-extractor-epochs 60
  
  # Train with different batch sizes and learning rates
  python train_models.py --fingerprint-method nataraj19 --data-dir ./data \\
    --true-attribution-epochs 30 --true-attribution-batch-size 64 --true-attribution-lr 1e-3 \\
    --surrogate-attribution-epochs 25 --surrogate-attribution-batch-size 32 --surrogate-attribution-lr 5e-4 \\
    --surrogate-extractor-epochs 35 --surrogate-extractor-batch-size 16 --surrogate-extractor-lr 2e-4
  
  # Use existing data (if available) with custom hyperparameters
  python train_models.py --fingerprint-method nataraj19 --data-dir ./data \\
    --true-attribution-epochs 20 --surrogate-attribution-epochs 20 --surrogate-extractor-epochs 20
  
  # Generate additional images to reach target count (e.g., 1000 per model)
  python train_models.py --fingerprint-method nataraj19 --data-dir ./data \\
    --images-per-model 1000  # Will generate additional images if existing < 1000
  
  # Train with a different fingerprint method (will reuse existing images)
  python train_models.py --fingerprint-method durall20 --data-dir ./data \\
    --images-per-model 100  # Will use existing images from /data/images/
  
  # Optimize file loading speed with parallel workers and memory mapping
  python train_models.py --fingerprint-method nataraj19 --data-dir ./data \\
    --max-workers 16 --use-memory-mapping
  
  # Use custom number of parallel workers for file loading
  python train_models.py --fingerprint-method nataraj19 --data-dir ./data \\
    --max-workers 4  # Use only 4 parallel workers
        """
    )
    parser.add_argument("--fingerprint-method", required=True, help="Fingerprint method (e.g., nataraj19)")
    parser.add_argument("--data-dir", required=True, help="Path to save trained models and data")
    
    # Separate hyperparameters for each model
    parser.add_argument("--true-attribution-epochs", type=int, default=300, help="Epochs for True Attribution Model (h)")
    parser.add_argument("--true-attribution-batch-size", type=int, default=32, help="Batch size for True Attribution Model (h)")
    parser.add_argument("--true-attribution-lr", type=float, default=1e-4, help="Learning rate for True Attribution Model (h)")
    
    parser.add_argument("--surrogate-attribution-epochs", type=int, default=50, help="Epochs for Surrogate Attribution Model (h_s)")
    parser.add_argument("--surrogate-attribution-batch-size", type=int, default=32, help="Batch size for Surrogate Attribution Model (h_s)")
    parser.add_argument("--surrogate-attribution-lr", type=float, default=1e-4, help="Learning rate for Surrogate Attribution Model (h_s)")
    
    parser.add_argument("--surrogate-extractor-epochs", type=int, default=50, help="Epochs for Surrogate Extractor (φ_s)")
    parser.add_argument("--surrogate-extractor-batch-size", type=int, default=32, help="Batch size for Surrogate Extractor (φ_s)")
    parser.add_argument("--surrogate-extractor-lr", type=float, default=1e-4, help="Learning rate for Surrogate Extractor (φ_s)")
    
    # Legacy parameters for backward compatibility
    parser.add_argument("--num-epochs", type=int, default=30, help="Number of training epochs (legacy, use specific model args instead)")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size (legacy, use specific model args instead)")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate (legacy, use specific model args instead)")
    
    parser.add_argument("--image-size", type=int, default=256, help="Image size")
    parser.add_argument("--images-per-model", type=int, default=100, 
                       help="Target number of images per model. If existing data has fewer images, additional images will be generated to reach this target.")
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, cpu, mps)")
    parser.add_argument("--max-workers", type=int, default=None, 
                       help="Number of parallel workers for loading existing data (default: cpu_count() or 8, max: 8)")
    parser.add_argument("--use-memory-mapping", action="store_true", 
                       help="Use memory mapping for faster loading of numpy files (may use more RAM but faster I/O)")
    
    # Song24 specific arguments
    parser.add_argument("--real-data-path", type=str, default=None,
                       help="Path to real data for Song24 manifold estimation (optional - will auto-download FFHQ if not provided)")
    parser.add_argument("--cache-dir", type=str, default="cache",
                       help="Directory to cache manifold estimators for Song24 methods")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🧪 FINGERPRINT ROBUSTNESS MODEL TRAINING")
    print("=" * 60)
    print("📋 Training Configuration:")
    print(f"  True Attribution Model (h):     {args.true_attribution_epochs} epochs, BS {args.true_attribution_batch_size}, LR {args.true_attribution_lr}")
    print(f"  Surrogate Attribution (h_s):    {args.surrogate_attribution_epochs} epochs, BS {args.surrogate_attribution_batch_size}, LR {args.surrogate_attribution_lr}")
    print(f"  Surrogate Extractor (φ_s):      {args.surrogate_extractor_epochs} epochs, BS {args.surrogate_extractor_batch_size}, LR {args.surrogate_extractor_lr}")
    print(f"  Target Images per Model:        {args.images_per_model} (will generate additional images if needed)")
    
    # Display optimization settings
    max_workers_display = args.max_workers if args.max_workers else f"auto (max {min(multiprocessing.cpu_count(), 8)})"
    print(f"  File Loading Optimization:      {max_workers_display} workers, Memory mapping: {'Yes' if args.use_memory_mapping else 'No'}")
    print("=" * 60)
    
    # Train models
    success = train_models(
        fingerprint_method=args.fingerprint_method,
        output_path=args.data_dir,
        # True Attribution Model (h)
        true_attribution_epochs=args.true_attribution_epochs,
        true_attribution_batch_size=args.true_attribution_batch_size,
        true_attribution_lr=args.true_attribution_lr,
        # Surrogate Attribution Model (h_s)
        surrogate_attribution_epochs=args.surrogate_attribution_epochs,
        surrogate_attribution_batch_size=args.surrogate_attribution_batch_size,
        surrogate_attribution_lr=args.surrogate_attribution_lr,
        # Surrogate Extractor (φ_s)
        surrogate_extractor_epochs=args.surrogate_extractor_epochs,
        surrogate_extractor_batch_size=args.surrogate_extractor_batch_size,
        surrogate_extractor_lr=args.surrogate_extractor_lr,
        # Other parameters
        image_size=args.image_size,
        num_images_per_model=args.images_per_model,
        device_preference=args.device,
        max_workers=args.max_workers,
        use_memory_mapping=args.use_memory_mapping,
        # Song24 specific parameters
        real_data_path=args.real_data_path,
        cache_dir=args.cache_dir,
        data_dir=args.data_dir
    )
    
    if success:
        print("\n🎉 Training completed successfully!")
        print(f"Models and data saved to: {args.data_dir}")
    else:
        print("\n❌ Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
