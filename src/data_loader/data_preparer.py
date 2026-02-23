"""
Data preparation and loading functionality for fingerprint robustness testing.
"""

import torch
import numpy as np
import random
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from PIL import Image
from torchvision import transforms

from .model_discovery import ModelDiscovery
from .fingerprint_extractor_factory import FingerprintExtractorFactory


class DataPreparer:
    """Handles data preparation and loading for attack evaluation."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    def prepare_test_data(
        self,
        data_dir: Optional[Path],
        fingerprint_method: str,
        model_names: List[str],
        num_images_per_model: int,
        image_size: int,
        real_data_path: Optional[str] = None,
        cache_dir: str = "cache"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Prepare test data by either loading existing data or generating new data.
        
        Args:
            data_dir: Base data directory (optional)
            fingerprint_method: Fingerprint method name
            model_names: List of model names
            num_images_per_model: Number of images per model
            image_size: Image size
            real_data_path: Path to real data (for some methods)
            cache_dir: Cache directory
            
        Returns:
            Tuple of (features, images, labels, num_models)
        """
        
        # Check if this is an implicit fingerprint method
        try:
            is_implicit = FingerprintExtractorFactory.is_implicit_method(
                fingerprint_method,
                self.device,
                real_data_path=real_data_path,
                cache_dir=cache_dir,
                data_dir=str(data_dir) if data_dir is not None else None
            )
        except ImportError:
            # Fallback: assume it's not implicit if we can't check
            is_implicit = False
        
        if data_dir is not None and data_dir.exists():
            # Check if we have data for all models
            images_dir = data_dir / "images"
            features_dir = data_dir / f"features_{fingerprint_method}"
            
            # For implicit fingerprint methods, we only need images (features are computed on-the-fly)
            if is_implicit:
                print(f"📁 Checking for existing images (implicit fingerprint method: {fingerprint_method})...")
                
                if images_dir.exists():
                    all_models_have_data = True
                    total_images = 0
                    
                    for model_name in model_names:
                        model_dir = images_dir / model_name
                        if model_dir.exists():
                            image_count = len(list(model_dir.glob("*.png")))
                            total_images += image_count
                            if image_count < num_images_per_model:
                                all_models_have_data = False
                                break
                        else:
                            all_models_have_data = False
                            break
                    
                    if all_models_have_data:
                        print(f"✅ Found existing images: {total_images} images")
                        print("Loading existing images for implicit fingerprint method...")
                        
                        # For implicit methods, we load images and create dummy features
                        # (features will be computed on-the-fly during the attack)
                        dummy_features, X_images, y, num_models = self._load_existing_data_implicit(
                            images_dir, model_names, num_images_per_model, self.device
                        )
                        return dummy_features, X_images, y, num_models
                    else:
                        print("📁 Not all models have sufficient images. Generating new data...")
                else:
                    print("📁 No existing images directory found. Generating new data...")
            else:
                # For explicit fingerprint methods, prefer using existing images
                print(f"📁 Checking for existing images and features (explicit fingerprint method: {fingerprint_method})...")
                
                images_exist = images_dir.exists()
                features_exist = features_dir.exists()
                
                if images_exist:
                    # Check if each model has enough images
                    all_models_have_enough_images = True
                    for model_name in model_names:
                        model_images_dir = images_dir / model_name
                        if not model_images_dir.exists() or len(list(model_images_dir.glob("*.png"))) < num_images_per_model:
                            all_models_have_enough_images = False
                            break
                    
                    if features_exist:
                        # If features directory exists, check full availability
                        all_models_have_full_data = True
                        total_images = 0
                        total_features = 0
                        for model_name in model_names:
                            model_images_dir = images_dir / model_name
                            model_features_dir = features_dir / model_name
                            if model_images_dir.exists() and model_features_dir.exists():
                                image_count = len(list(model_images_dir.glob("*.png")))
                                feature_count = len(list(model_features_dir.glob("*.npy")))
                                total_images += image_count
                                total_features += feature_count
                                if image_count < num_images_per_model or feature_count < num_images_per_model:
                                    all_models_have_full_data = False
                                    break
                            else:
                                all_models_have_full_data = False
                                break
                        if all_models_have_full_data:
                            print(f"✅ Found existing data: {total_images} images, {total_features} features")
                            print("Loading existing data from disk...")
                            X_features, X_images, y, num_models = self._load_existing_data(
                                images_dir, features_dir, model_names, num_images_per_model, self.device
                            )
                            return X_features, X_images, y, num_models
                        else:
                            # If images are sufficient but features are missing/insufficient, extract from existing images
                            if all_models_have_enough_images:
                                print("🔄 Features missing or insufficient. Extracting fingerprints from existing images...")
                                X_features, X_images, y, num_models = self._extract_features_from_existing_images(
                                    images_dir, model_names, num_images_per_model, image_size, self.device,
                                    fingerprint_method, real_data_path, cache_dir, str(data_dir) if data_dir is not None else None
                                )
                                return X_features, X_images, y, num_models
                            else:
                                print("📁 Not all models have sufficient images. Generating new data...")
                    else:
                        # No features directory: if images are sufficient, extract fingerprints from them
                        if all_models_have_enough_images:
                            print("📁 Images found but no features directory. Extracting fingerprints from existing images...")
                            X_features, X_images, y, num_models = self._extract_features_from_existing_images(
                                images_dir, model_names, num_images_per_model, image_size, self.device,
                                fingerprint_method, real_data_path, cache_dir, str(data_dir) if data_dir is not None else None
                            )
                            return X_features, X_images, y, num_models
                        else:
                            print("📁 Images directory found but insufficient images. Generating new data...")
                else:
                    print("📁 No existing images directory found. Generating new data...")
        
        # Generate new data on-the-fly
        print("🔄 Generating new test data on-the-fly...")
        X_images, X_features, y, num_models = self._generate_test_data_on_the_fly(
            model_names, num_images_per_model, image_size, self.device, fingerprint_method,
            real_data_path, cache_dir, data_dir
        )
        
        return X_features, X_images, y, num_models
    
    def _load_existing_data_implicit(
        self,
        images_dir: Path,
        model_names: List[str],
        num_images_per_model: int,
        device: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Load existing images for implicit fingerprint methods (features computed on-the-fly)."""
        
        print(f"Loading existing images for implicit fingerprint method (requesting {num_images_per_model} images per model)...")
        
        # Respect the provided model order for implicit methods. AttackRunner already
        # applies legacy alphabetical ordering for qian20/wang20 checkpoints, so
        # re-sorting here would reintroduce the mismatch.
        sorted_model_names = model_names
        print(f"Models sorted by load order: {sorted_model_names}")
        
        # Print explicit model-to-label mapping for implicit attack data loading
        print(f"\n📋 MODEL-TO-LABEL MAPPING (Implicit Attack Data Loading):")
        for i, model_name in enumerate(sorted_model_names):
            print(f"   Model {i}: {model_name}")
        print()
        
        all_labels = []
        all_images = []
        
        for i, model_name in enumerate(sorted_model_names):
            model_images_dir = images_dir / model_name
            
            if not model_images_dir.exists():
                raise FileNotFoundError(f"Images for model {model_name} not found in {images_dir}")
                
            # Load images
            image_files = sorted(list(model_images_dir.glob("*.png")))
            if len(image_files) == 0:
                raise FileNotFoundError(f"No images found for model {model_name} in {model_images_dir}")
            
            # Determine how many images to load (either requested amount or available amount)
            available_images = len(image_files)
            images_to_load = min(num_images_per_model, available_images)
            
            print(f"  Model {model_name}: {available_images} images available, randomly loading {images_to_load}")
            
            # Randomly select image indices (seeded by current time for variability between runs)
            random.seed(int(time.time()))
            indices = list(range(available_images))
            selected_indices = random.sample(indices, images_to_load)
            
            # Load only the randomly selected images
            for idx in selected_indices:
                img_path = image_files[idx]
                
                # Load image using PIL and convert to tensor
                img_pil = Image.open(img_path)
                img_tensor = transforms.ToTensor()(img_pil)
                all_images.append(img_tensor)
                
            all_labels.extend([i] * images_to_load)
            
            print(f"  ✅ Loaded {images_to_load} images for {model_name}")
        
        # Convert to tensors
        X_images = torch.stack(all_images, dim=0)
        y = torch.tensor(all_labels, dtype=torch.long)
        
        # For implicit methods, create dummy features (will be replaced during attack)
        # The actual features will be computed on-the-fly by the implicit fingerprint model
        dummy_features = torch.zeros(len(all_images), 1)  # Placeholder features
        
        print(f"✅ Total images loaded: {len(all_images)} samples")
        print(f"  Images shape: {X_images.shape}")
        print(f"  Labels shape: {y.shape}")
        print(f"  Label range: {y.min().item()} to {y.max().item()}")
        print(f"  Label distribution: {torch.bincount(y).tolist()}")
        print(f"  Number of models: {len(model_names)}")
        print(f"  Note: Features will be computed on-the-fly by the implicit fingerprint model")
        
        return dummy_features, X_images, y, len(model_names)
    
    def _load_existing_data(
        self,
        images_dir: Path,
        features_dir: Path,
        model_names: List[str],
        num_images_per_model: int,
        device: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Load existing images and fingerprints from disk."""
        
        print(f"Loading existing data from disk (requesting {num_images_per_model} images per model)...")
        
        # Sort models by the same order used in training to ensure label consistency
        try:
            from src.utils.model_isolation import sort_models_by_load_order
            sorted_model_names = sort_models_by_load_order(model_names)
            print(f"Models sorted by load order: {sorted_model_names}")
        except ImportError:
            # Fallback to original order if sorting function not available
            sorted_model_names = model_names
            print(f"Using original model order: {sorted_model_names}")
        
        # Print explicit model-to-label mapping for explicit attack data loading
        print(f"\n📋 MODEL-TO-LABEL MAPPING (Explicit Attack Data Loading):")
        for i, model_name in enumerate(sorted_model_names):
            print(f"   Model {i}: {model_name}")
        print()
        
        all_features = []
        all_labels = []
        all_images = []
        
        # Import random and time for random selection with time-based seed
        # Use current time as seed to get different random selections in each run
        random.seed(int(time.time()))
        
        for i, model_name in enumerate(sorted_model_names):
            model_images_dir = images_dir / model_name
            model_features_dir = features_dir / model_name
            
            if not model_images_dir.exists() or not model_features_dir.exists():
                raise FileNotFoundError(f"Data for model {model_name} not found in {images_dir} or {features_dir}")
                
            # Load images
            image_files = sorted(list(model_images_dir.glob("*.png")))
            if len(image_files) == 0:
                raise FileNotFoundError(f"No images found for model {model_name} in {model_images_dir}")
                
            # Load fingerprints
            feature_files = sorted(list(model_features_dir.glob("*.npy")))
            if len(feature_files) == 0:
                raise FileNotFoundError(f"No fingerprints found for model {model_name} in {model_features_dir}")
                
            # Ensure image and feature files match
            if len(image_files) != len(feature_files):
                raise RuntimeError(f"Mismatch in number of images ({len(image_files)}) and fingerprints ({len(feature_files)}) for model {model_name}")
            
            # Determine how many images to load (either requested amount or available amount)
            available_images = len(image_files)
            images_to_load = min(num_images_per_model, available_images)
            
            print(f"  Model {model_name}: {available_images} images available, loading {images_to_load}")
            
            # Create list of indices and randomly sample from it
            indices = list(range(available_images))
            selected_indices = random.sample(indices, images_to_load)
            
            # Load only the randomly selected images
            for idx in selected_indices:
                img_path = image_files[idx]
                feature_path = feature_files[idx]
                
                # Load image using PIL and convert to tensor
                img_pil = Image.open(img_path)
                img_tensor = transforms.ToTensor()(img_pil)
                all_images.append(img_tensor)
                
                # Load feature
                feature_array = np.load(feature_path)
                feature_tensor = torch.tensor(feature_array, dtype=torch.float32)
                all_features.append(feature_tensor)
                
            all_labels.extend([i] * images_to_load)
            
            print(f"  ✅ Loaded {images_to_load} images and fingerprints for {model_name}")
        
        # Convert to tensors
        X_features = torch.stack(all_features, dim=0)
        X_images = torch.stack(all_images, dim=0)
        y = torch.tensor(all_labels, dtype=torch.long)
        
        print(f"✅ Total data loaded: {len(all_features)} samples")
        print(f"  Features shape: {X_features.shape}")
        print(f"  Images shape: {X_images.shape}")
        print(f"  Labels shape: {y.shape}")
        print(f"  Label range: {y.min().item()} to {y.max().item()}")
        print(f"  Label distribution: {torch.bincount(y).tolist()}")
        print(f"  Number of models: {len(model_names)}")
        
        return X_features, X_images, y, len(model_names)
    
    def _generate_test_data_on_the_fly(
        self,
        model_names: List[str],
        num_images_per_model: int,
        image_size: int,
        device: str,
        fingerprint_method: str,
        real_data_path: Optional[str] = None,
        cache_dir: str = "cache",
        data_dir: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Generate test images and extract fingerprints on-the-fly."""
        
        print(f"Generating {num_images_per_model} test images from each of {len(model_names)} models...")
        
        # Sort models by the same order used in training to ensure label consistency
        try:
            from src.utils.model_isolation import sort_models_by_load_order
            sorted_model_names = sort_models_by_load_order(model_names)
            print(f"Models sorted by load order: {sorted_model_names}")
        except ImportError:
            # Fallback to original order if sorting function not available
            sorted_model_names = model_names
            print(f"Using original model order: {sorted_model_names}")
        
        # Print explicit model-to-label mapping for attack test data generation
        print(f"\n📋 MODEL-TO-LABEL MAPPING (Attack Test Data Generation):")
        for i, model_name in enumerate(sorted_model_names):
            print(f"   Model {i}: {model_name}")
        print()
        
        try:
            from src.models import ModelSampler
        except ImportError as e:
            raise ImportError(f"Failed to import ModelSampler: {e}")
        
        all_images = []
        all_labels = []
        all_features = []
        
        # Use current time as seed to get different random selections in each run
        random.seed(int(time.time()))
        
        for i, model_name in enumerate(sorted_model_names):
            print(f"\n--- Generating test data for {model_name} ---")
            
            try:
                # Create sampler
                sampler = ModelSampler.create(
                    model_name=model_name,
                    dataset="ffhq",
                    image_size=image_size,
                    device=device,
                    batch_size=1
                )
                
                # Load model
                sampler.load_model()
                
                # Generate images in batches
                batch_size = min(10, num_images_per_model)
                total_generated = 0
                
                # Generate random seeds for each batch to ensure diversity
                num_batches = (num_images_per_model + batch_size - 1) // batch_size
                random_seeds = random.sample(range(1000000), num_batches)  # Large range for diversity
                
                for batch_idx, batch_start in enumerate(range(0, num_images_per_model, batch_size)):
                    current_batch_size = min(batch_size, num_images_per_model - batch_start)
                    print(f"    Generating batch {batch_start//batch_size + 1}/{(num_images_per_model + batch_size - 1)//batch_size} ({current_batch_size} images)...")
                    
                    # Set random seed for this batch based on current time
                    torch.manual_seed(int(time.time()) + batch_idx)
                    
                    try:
                        # Generate batch
                        batch_imgs = sampler.sample(num_images=current_batch_size, save_to_file=False)
                        
                        # Extract fingerprints
                        fingerprint_extractor = FingerprintExtractorFactory.create(
                            fingerprint_method,
                            device,
                            real_data_path=real_data_path,
                            cache_dir=cache_dir,
                            data_dir=data_dir
                        )
                        batch_fp = fingerprint_extractor.extract_fingerprint(batch_imgs)
                        
                        # Store for testing
                        all_images.extend([img.cpu() for img in batch_imgs])
                        all_features.extend([fp.cpu() for fp in batch_fp])
                        all_labels.extend([i] * current_batch_size)
                        
                        total_generated += current_batch_size
                        print(f"      ✅ Generated {current_batch_size} test images and fingerprints")
                        
                        # Clean up memory
                        del batch_imgs, batch_fp, fingerprint_extractor
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                            torch.mps.empty_cache()
                            
                    except Exception as e:
                        print(f"      ❌ Error in batch: {e}")
                        continue
                
                print(f"  ✅ Generated {total_generated} test images for {model_name}")
                
            except Exception as e:
                print(f"  ❌ Failed to generate test data for {model_name}: {e}")
                continue
        
        if len(all_features) == 0:
            raise RuntimeError("No test data was generated successfully")
        
        # Convert to tensors
        X_images = torch.stack(all_images, dim=0)
        X_features = torch.tensor(np.array(all_features), dtype=torch.float32)
        y = torch.tensor(all_labels, dtype=torch.long)
        
        print(f"✅ Total test data generated: {len(all_features)} samples")
        print(f"  Images shape: {X_images.shape}")
        print(f"  Features shape: {X_features.shape}")
        print(f"  Labels shape: {y.shape}")
        
        return X_images, X_features, y, len(model_names)

    def _extract_features_from_existing_images(
        self,
        images_dir: Path,
        model_names: List[str],
        num_images_per_model: int,
        image_size: int,
        device: str,
        fingerprint_method: str,
        real_data_path: Optional[str] = None,
        cache_dir: str = "cache",
        data_dir: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Load existing images and extract fingerprints (explicit methods)."""
        
        print(f"Extracting {fingerprint_method} fingerprints from existing images (requesting {num_images_per_model} images per model)...")
        
        # Sort models by the same order used in training to ensure label consistency
        try:
            from src.utils.model_isolation import sort_models_by_load_order
            sorted_model_names = sort_models_by_load_order(model_names)
            print(f"Models sorted by load order: {sorted_model_names}")
        except ImportError:
            sorted_model_names = model_names
            print(f"Using original model order: {sorted_model_names}")
        
        print(f"\n📋 MODEL-TO-LABEL MAPPING (Extract From Existing Images):")
        for i, model_name in enumerate(sorted_model_names):
            print(f"   Model {i}: {model_name}")
        print()
        
        all_images = []
        all_features = []
        all_labels = []
        
        # Use current time as seed to get different random selections in each run
        random.seed(int(time.time()))
        
        # Create the fingerprint extractor once and reuse
        fingerprint_extractor = FingerprintExtractorFactory.create(
            fingerprint_method,
            device,
            real_data_path=real_data_path,
            cache_dir=cache_dir,
            data_dir=data_dir
        )
        
        for i, model_name in enumerate(sorted_model_names):
            model_images_dir = images_dir / model_name
            if not model_images_dir.exists():
                raise FileNotFoundError(f"Images for model {model_name} not found in {images_dir}")
            
            image_files = sorted(list(model_images_dir.glob("*.png")))
            if len(image_files) == 0:
                raise FileNotFoundError(f"No images found for model {model_name} in {model_images_dir}")
            
            available_images = len(image_files)
            images_to_load = min(num_images_per_model, available_images)
            print(f"  Model {model_name}: {available_images} images available, loading {images_to_load}")
            
            indices = list(range(available_images))
            selected_indices = random.sample(indices, images_to_load)
            
            # Load only the randomly selected images
            model_images = []
            for idx in selected_indices:
                img_path = image_files[idx]
                img_pil = Image.open(img_path)
                img_tensor = transforms.ToTensor()(img_pil)
                model_images.append(img_tensor)
            
            if len(model_images) == 0:
                continue
            
            images_tensor = torch.stack(model_images, dim=0).to(device)
            
            # Process in batches
            batch_size = 10
            extracted_features = []
            for batch_start in range(0, images_tensor.shape[0], batch_size):
                batch_end = min(batch_start + batch_size, images_tensor.shape[0])
                batch_imgs = images_tensor[batch_start:batch_end]
                batch_fp = fingerprint_extractor.extract_fingerprint(batch_imgs)
                extracted_features.extend([fp.detach().cpu() for fp in batch_fp])
                
                # Clean up accelerator memory if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            
            all_images.extend([img.cpu() for img in model_images])
            all_features.extend(extracted_features)
            all_labels.extend([i] * images_to_load)
            print(f"  ✅ Extracted fingerprints for {images_to_load} images of {model_name}")
        
        if len(all_features) == 0:
            raise RuntimeError("No fingerprints were extracted from existing images")
        
        X_images = torch.stack(all_images, dim=0)
        X_features = torch.tensor(np.array(all_features), dtype=torch.float32)
        y = torch.tensor(all_labels, dtype=torch.long)
        
        print(f"✅ Total data prepared from existing images: {len(all_features)} samples")
        print(f"  Images shape: {X_images.shape}")
        print(f"  Features shape: {X_features.shape}")
        print(f"  Labels shape: {y.shape}")
        
        return X_features, X_images, y, len(model_names)
