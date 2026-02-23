"""
Song24 (ManiFPT) fingerprint extraction method implementation.

Based on "ManiFPT: Defining and Analyzing Fingerprints of Generative Models" by Song et al. (2024).

This method defines artifacts as the difference between generated images and their closest points
on the true data manifold, estimated using real samples in different embedding spaces.
"""

import torch
import torch.nn as nn
import torch.fft
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path
import pickle
import logging
import os
import subprocess
import sys

from .base import FingerprintExtractor


def download_ffhq_data(output_dir: str = "data/ffhq", num_images: int = 1000) -> str:
    """
    Download FFHQ-256 data for manifold estimation from HuggingFace.
    
    Uses the merkol/ffhq-256 dataset which contains ~70,000 high-quality face images
    at 256x256 resolution in parquet format.
    
    Args:
        output_dir: Directory to save the data
        num_images: Number of images to download (max ~70,000)
        
    Returns:
        Path to the downloaded data pickle file
    """
    import numpy as np  # Import numpy at the top of the function
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    pickle_file = output_path / "ffhq_real_data.pkl"
    
    # Check if data already exists and is valid
    if pickle_file.exists():
        try:
            # Try to load the data to verify it's not corrupted
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
            
            # Check if data has the expected structure
            if isinstance(data, dict) and 'images' in data:
                images = data['images']
                if isinstance(images, (np.ndarray, torch.Tensor)):
                    if len(images.shape) == 4 and images.shape[1] == 3:  # (N, C, H, W)
                        print(f"✅ FFHQ data already exists and is valid at {pickle_file}")
                        print(f"   Found {images.shape[0]} images of shape {images.shape[1:]}")
                        return str(pickle_file)
            
            print(f"⚠️  Found existing file but data format is invalid, re-downloading...")
        except Exception as e:
            print(f"⚠️  Found existing file but failed to load: {e}")
            print(f"   Re-downloading FFHQ data...")
    
    print(f"📥 Downloading {num_images} FFHQ-256 images from HuggingFace for Song24 manifold estimation...")
    print(f"   Dataset: merkol/ffhq-256 (70K high-quality face images at 256x256)")
    print(f"   Saving to: {pickle_file}")

    def _download_with_datasets():
        # Try to install datasets if not available
        try:
            # Some environments (e.g., with hf_xet) leave pyarrow extension types registered,
            # which causes datasets to fail on import. Unregister proactively.
            os.environ["HF_HUB_DISABLE_XET"] = "1"
            try:
                import pyarrow as pa
                # Purge any datasets-added extension types to avoid duplicate registration errors.
                for ext_name in list(pa.lib.get_extension_types().keys()):
                    if "datasets.features.features.Array" in ext_name:
                        try:
                            pa.unregister_extension_type(ext_name)
                        except Exception:
                            pass
            except Exception:
                pass

            from datasets import load_dataset
        except ImportError:
            print("📦 Installing datasets library...")
            try:
                result = subprocess.run([sys.executable, "-m", "pip", "install", "datasets"], 
                                      check=True, capture_output=True, text=True)
                print("✅ Successfully installed datasets library")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install datasets: {e}")
                print("Please install manually: pip install datasets")
                raise
            from datasets import load_dataset
        
        from torchvision import transforms
        import numpy as np
        
        print("🔄 Loading FFHQ-256 dataset from HuggingFace...")
        
        # Load FFHQ-256 dataset from HuggingFace
        # Using "merkol/ffhq-256" which is a proper FFHQ-256 dataset with parquet files
        dataset = load_dataset("merkol/ffhq-256", split="train")
        
        print(f"✅ Loaded {len(dataset)} images from HuggingFace")
        
        # Create transform to resize to 256x256
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        # Convert to tensors
        print(f"🔄 Processing {min(num_images, len(dataset))} images...")
        images = []
        
        # Check dataset structure
        first_example = dataset[0]
        print(f"   Dataset structure: {list(first_example.keys())}")
        
        for i, example in enumerate(dataset):
            if i >= num_images:
                break
            
            try:
                # Handle different possible image field names
                if 'image' in example:
                    img = example['image']
                elif 'img' in example:
                    img = example['img']
                elif 'data' in example:
                    img = example['data']
                else:
                    # Try to find any field that might contain image data
                    for key, value in example.items():
                        if hasattr(value, 'size') and len(value.size) >= 2:  # Likely an image
                            img = value
                            break
                    else:
                        raise ValueError(f"No image field found in example {i}")
                
                # Convert to PIL Image if needed
                if hasattr(img, 'convert'):
                    # Already a PIL Image
                    pass
                elif hasattr(img, 'numpy'):
                    # Tensor or numpy array
                    if hasattr(img, 'cpu'):
                        img = img.cpu().numpy()
                    img = transforms.ToPILImage()(torch.tensor(img))
                elif hasattr(img, 'shape'):
                    # Numpy array
                    img = transforms.ToPILImage()(torch.tensor(img))
                else:
                    # Assume it's already in the right format
                    pass
                
                img_tensor = transform(img)
                images.append(img_tensor)
                
                if (i + 1) % 100 == 0:
                    print(f"   Processed {i + 1}/{min(num_images, len(dataset))} images...")
                    
            except Exception as e:
                print(f"   Warning: Skipping image {i} due to error: {e}")
                continue
        
        if not images:
            raise RuntimeError("No images were successfully processed")
        
        # Stack images
        images_tensor = torch.stack(images)
        
        # Save to pickle
        data = {'images': images_tensor.cpu().numpy()}
        with open(pickle_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ Successfully downloaded and saved {len(images)} images to {pickle_file}")
        print(f"   Image shape: {images_tensor.shape}")
        print(f"   Data size: {pickle_file.stat().st_size / (1024*1024):.1f} MB")
        
        return str(pickle_file)

    def _download_in_subprocess() -> str:
        """Fallback: perform the download in a clean subprocess to avoid pyarrow extension conflicts."""
        env = os.environ.copy()
        env["HF_HUB_DISABLE_XET"] = "1"
        env["FFHQ_OUTPUT_DIR"] = str(output_path)
        env["FFHQ_NUM_IMAGES"] = str(num_images)
        script = r"""
import os, pickle
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
from datasets import load_dataset

output_dir = Path(os.environ["FFHQ_OUTPUT_DIR"])
num_images = int(os.environ.get("FFHQ_NUM_IMAGES", "1000"))
pickle_file = output_dir / "ffhq_real_data.pkl"
output_dir.mkdir(parents=True, exist_ok=True)

dataset = load_dataset("merkol/ffhq-256", split="train")
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
images = []
for i, example in enumerate(dataset):
    if i >= num_images:
        break
    img = example.get("image") or example.get("img") or example.get("data")
    if img is None:
        for v in example.values():
            if hasattr(v, "size") and len(getattr(v, "size")) >= 2:
                img = v
                break
    if img is None:
        continue
    if hasattr(img, "convert"):
        pil_img = img
    elif hasattr(img, "numpy"):
        pil_img = transforms.ToPILImage()(torch.tensor(img.cpu().numpy() if hasattr(img, "cpu") else img.numpy()))
    elif hasattr(img, "shape"):
        pil_img = transforms.ToPILImage()(torch.tensor(img))
    else:
        continue
    images.append(transform(pil_img))

if not images:
    raise RuntimeError("No images were successfully processed in subprocess")
images_tensor = torch.stack(images)
with open(pickle_file, "wb") as f:
    pickle.dump({"images": images_tensor.cpu().numpy()}, f)
print(f\"✅ Subprocess downloaded {len(images)} images to {pickle_file}\")
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Subprocess download failed: {result.stderr.strip() or result.stdout.strip()}")
        if not pickle_file.exists():
            raise RuntimeError("Subprocess reported success but pickle file not found")
        return str(pickle_file)

    try:
        return _download_with_datasets()
    except Exception as e:
        print(f"⚠️  Primary download path failed: {e}")
        print("🔄 Retrying FFHQ download in a clean subprocess...")
        return _download_in_subprocess()


class ManifoldEstimator:
    """
    Estimates the true data manifold using real samples in different embedding spaces.
    """
    
    def __init__(self, manifold_type: str = "rgb", device: str = "cpu"):
        """
        Initialize manifold estimator.
        
        Args:
            manifold_type: Type of embedding space ("rgb", "freq", "sl", "ssl")
            device: Device to use for computations
        """
        self.manifold_type = manifold_type
        self.device = device
        self.real_samples = None
        self.feature_extractor = None
        
        if manifold_type in ["sl", "ssl"]:
            self._setup_feature_extractor()
    
    def _setup_feature_extractor(self):
        """Setup neural network feature extractor for SL/SSL spaces."""
        if self.manifold_type == "sl":
            # Use pretrained ResNet50 for supervised learning features
            try:
                # Try newer torchvision API first
                import torchvision.models as models
                resnet = models.resnet50(pretrained=True)
                self.feature_extractor = nn.Sequential(
                    *list(resnet.children())[:-1]
                ).to(self.device)
            except:
                # Fallback to torch.hub
                self.feature_extractor = nn.Sequential(
                    *list(torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).children())[:-1]
                ).to(self.device)
            self.feature_extractor.eval()
        elif self.manifold_type == "ssl":
            # Use a self-supervised model (e.g., SimCLR, MoCo, etc.)
            # For now, using a simple autoencoder as placeholder
            # In practice, you'd use a proper SSL model
            self.feature_extractor = self._create_ssl_extractor()
    
    def _create_ssl_extractor(self):
        """Create a simple SSL feature extractor (placeholder)."""
        # This is a simplified version - in practice you'd use a proper SSL model
        extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 512)
        ).to(self.device)
        return extractor
    
    def fit(self, real_images: torch.Tensor):
        """
        Fit the manifold estimator using real images.
        
        Args:
            real_images: Real images tensor of shape (N, C, H, W)
        """
        with torch.no_grad():
            if self.manifold_type == "rgb":
                # Store RGB values directly
                self.real_samples = real_images.clone()
            
            elif self.manifold_type == "freq":
                # Convert to frequency domain
                self.real_samples = self._to_frequency_domain(real_images)
            
            elif self.manifold_type in ["sl", "ssl"]:
                # Extract features using neural network
                self.real_samples = self.feature_extractor(real_images)
    
    def _to_frequency_domain(self, images: torch.Tensor) -> torch.Tensor:
        """
        Convert images to frequency domain using FFT.
        
        Args:
            images: Input images tensor of shape (N, C, H, W)
            
        Returns:
            Frequency domain representation
        """
        # Convert to grayscale if needed
        if images.shape[1] == 3:
            rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=images.device).view(1, 3, 1, 1)
            images = torch.sum(images * rgb_weights, dim=1, keepdim=True)
        
        # Apply FFT
        fft = torch.fft.fft2(images)
        magnitude = torch.abs(fft)
        
        # Normalize by DC component
        dc = magnitude[:, :, 0, 0].unsqueeze(-1).unsqueeze(-1)
        normalized_magnitude = magnitude / (dc + 1e-8)
        
        return normalized_magnitude
    
    def find_nearest_neighbor(self, query: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Find the nearest neighbor of query in the real samples.
        
        Args:
            query: Query tensor in the same space as real_samples
            
        Returns:
            Tuple of (nearest_neighbor, index)
        """
        if self.real_samples is None:
            raise ValueError("Manifold estimator must be fitted with real samples first")
        
        # Ensure both query and real_samples are on the same device
        # CRITICAL FIX: Don't modify self.real_samples in place!
        # Instead, create a temporary copy for this computation
        query_device = query.device
        if self.real_samples.device != query_device:
            real_samples_on_device = self.real_samples.to(query_device)
        else:
            real_samples_on_device = self.real_samples
        
        # Compute distances
        distances = torch.cdist(query.flatten(1), real_samples_on_device.flatten(1))
        
        # Find nearest neighbor
        min_distances, indices = torch.min(distances, dim=1)
        
        # Get the nearest neighbors
        nearest_neighbors = real_samples_on_device[indices]
        
        return nearest_neighbors, indices
    
    def save(self, filepath: str):
        """Save the fitted manifold estimator."""
        data = {
            'manifold_type': self.manifold_type,
            'real_samples': self.real_samples.cpu() if self.real_samples is not None else None
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath: str, device: str = "cpu"):
        """Load a fitted manifold estimator."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        estimator = cls(manifold_type=data['manifold_type'], device=device)
        
        # Handle real_samples safely
        if data['real_samples'] is not None:
            estimator.real_samples = data['real_samples'].to(device)
        else:
            estimator.real_samples = None
        
        return estimator


@FingerprintExtractor.register("song24")
class Song24(FingerprintExtractor):
    """
    Song24 (ManiFPT) fingerprint extractor.
    
    Computes artifacts as deviations from the true data manifold in different embedding spaces.
    
    Features:
    - Auto-downloads FFHQ-256 data from HuggingFace (merkol/ffhq-256) if real_data_path not provided
    - Supports multiple manifold types: RGB, frequency, supervised learning, self-supervised learning
    - Caches manifold estimators for efficiency
    - Handles all data loading and preprocessing automatically
    """
    
    def __init__(self, manifold_type: str = "rgb", real_data_path: Optional[str] = None, 
                 device: str = "cpu", cache_dir: Optional[str] = None, data_dir: Optional[str] = None):
        """
        Initialize Song24 extractor.
        
        Args:
            manifold_type: Type of embedding space ("rgb", "freq", "sl", "ssl")
            real_data_path: Path to real data for manifold estimation
            device: Device to use for computations
            cache_dir: Directory to cache manifold estimators
            data_dir: Main data directory (will create ffhq_dataset subfolder here)
        """
        self.manifold_type = manifold_type
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.manifold_estimator = None
        
        super().__init__(
            method_name=f"song24_{manifold_type}",
            is_differentiable=False,  # Nearest neighbor search is not differentiable
            has_analytic_approx=False,  # No analytic approximation available
            feature_dim=self._get_feature_dim(manifold_type)
        )
        
        # Load or create manifold estimator
        if real_data_path:
            self._setup_manifold_estimator(real_data_path)
        else:
            # Auto-download real data if not provided
            print("🔄 No real data path provided. Auto-downloading FFHQ data from HuggingFace...")
            
            # Determine download directory
            if data_dir:
                # Use data_dir/ffhq_dataset subfolder
                download_dir = Path(data_dir) / "ffhq_dataset"
                print(f"   Using data directory: {download_dir}")
            else:
                # Fallback to default location
                download_dir = "data/ffhq"
                print(f"   Using default directory: {download_dir}")
            
            auto_real_data_path = download_ffhq_data(output_dir=str(download_dir), num_images=1000)
            self._setup_manifold_estimator(auto_real_data_path)
    
    def _get_feature_dim(self, manifold_type: str) -> int:
        """Get feature dimension based on manifold type."""
        if manifold_type == "rgb":
            return 3 * 256 * 256  # Assuming 256x256 RGB images
        elif manifold_type == "freq":
            return 256 * 256  # Frequency domain size
        elif manifold_type == "sl":
            return 2048  # ResNet50 feature dimension
        elif manifold_type == "ssl":
            return 512  # SSL feature dimension
        else:
            raise ValueError(f"Unknown manifold type: {manifold_type}")
    
    def _setup_manifold_estimator(self, real_data_path: str):
        """Setup manifold estimator using real data."""
        cache_path = None
        if self.cache_dir:
            cache_path = self.cache_dir / f"manifold_estimator_{self.manifold_type}.pkl"
        
        # CRITICAL FIX: Always create a fresh estimator to avoid shared state issues
        # This prevents order dependency by ensuring each attack gets a clean instance
        self.logger.info(f"Creating fresh manifold estimator for {self.manifold_type} (avoiding shared state)")
        
        # Create new estimator (never use cached one to avoid shared state)
        self.manifold_estimator = ManifoldEstimator(self.manifold_type, self.device)
        
        # Load real data and fit estimator
        real_images = self._load_real_data(real_data_path)
        self.manifold_estimator.fit(real_images)
        
        # Save to cache for future use (but don't load from cache to avoid shared state)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.manifold_estimator.save(str(cache_path))
            self.logger.info(f"Saved manifold estimator to cache: {cache_path}")
    
    def _load_real_data(self, real_data_path: str) -> torch.Tensor:
        """
        Load real data for manifold estimation.
        
        Args:
            real_data_path: Path to real data
            
        Returns:
            Real images tensor
        """
        if real_data_path.endswith('.pkl'):
            with open(real_data_path, 'rb') as f:
                data = pickle.load(f)
            
            # Handle different data formats
            if isinstance(data, dict):
                if 'images' in data:
                    images = data['images']
                elif 'data' in data:
                    images = data['data']
                else:
                    raise ValueError(f"Unknown data format in pickle file. Expected 'images' or 'data' key, got: {list(data.keys())}")
            else:
                # Assume data is directly the images array
                images = data
            
            # Convert to tensor if needed
            if not isinstance(images, torch.Tensor):
                images = torch.tensor(images, dtype=torch.float32)
            
            # Ensure correct shape and range
            if images.dim() == 4:  # (N, C, H, W)
                if images.max() > 1.0:
                    images = images / 255.0  # Normalize to [0, 1]
            else:
                raise ValueError(f"Expected 4D tensor (N, C, H, W), got shape: {images.shape}")
            
            return images.to(self.device)
            
        else:
            # Assume it's a directory of images
            from torchvision import transforms
            from PIL import Image
            import glob
            
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
            
            image_files = glob.glob(f"{real_data_path}/*.jpg") + glob.glob(f"{real_data_path}/*.png")
            if not image_files:
                raise ValueError(f"No image files found in {real_data_path}")
            
            images = []
            
            for img_file in image_files[:1000]:  # Limit to 1000 images for efficiency
                img = Image.open(img_file).convert('RGB')
                img_tensor = transform(img)
                images.append(img_tensor)
            
            images = torch.stack(images)
        
        return images.to(self.device)
    
    def extract_fingerprint(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract artifacts as fingerprints from images.
        
        Args:
            images: Input images tensor of shape (N, C, H, W)
            
        Returns:
            Artifact features tensor of shape (N, feature_dim)
        """
        if self.manifold_estimator is None:
            raise ValueError("Manifold estimator not initialized. Provide real_data_path during initialization.")
        
        # Preprocess images based on manifold type
        processed_images = self._preprocess_images(images)
        
        # Find nearest neighbors in the real data manifold
        nearest_neighbors, _ = self.manifold_estimator.find_nearest_neighbor(processed_images)
        
        # Compute artifacts as differences
        artifacts = processed_images - nearest_neighbors
        
        # Flatten artifacts to get fingerprint features
        fingerprints = artifacts.flatten(1)
        
        return fingerprints
    
    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images based on manifold type.
        
        Args:
            images: Input images tensor
            
        Returns:
            Preprocessed images tensor
        """
        if self.manifold_type == "rgb":
            # Ensure images are on the same device as the manifold estimator/model
            return images.to(self.device)
        
        elif self.manifold_type == "freq":
            # Move to target device before frequency conversion
            return self.manifold_estimator._to_frequency_domain(images.to(self.device))
        
        elif self.manifold_type in ["sl", "ssl"]:
            with torch.no_grad():
                # Align tensor device with feature extractor to avoid CPU/GPU mismatch
                return self.manifold_estimator.feature_extractor(images.to(self.device))
        
        else:
            raise ValueError(f"Unknown manifold type: {self.manifold_type}")
    
    def get_artifact_visualization(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get visualization of artifacts for debugging.
        
        Args:
            images: Input images tensor
            
        Returns:
            Dictionary containing original, nearest neighbor, and artifact images
        """
        if self.manifold_estimator is None:
            raise ValueError("Manifold estimator not initialized")
        
        processed_images = self._preprocess_images(images)
        nearest_neighbors, _ = self.manifold_estimator.find_nearest_neighbor(processed_images)
        artifacts = processed_images - nearest_neighbors
        
        return {
            'original': images,
            'nearest_neighbor': nearest_neighbors,
            'artifact': artifacts
        }


# Convenience classes for different manifold types
@FingerprintExtractor.register("song24rgb")
class Song24RGB(Song24):
    """Song24 method using RGB space manifold."""
    def __init__(self, **kwargs):
        super().__init__(manifold_type="rgb", **kwargs)


@FingerprintExtractor.register("song24freq")
class Song24Freq(Song24):
    """Song24 method using frequency space manifold."""
    def __init__(self, **kwargs):
        super().__init__(manifold_type="freq", **kwargs)


@FingerprintExtractor.register("song24sl")
class Song24SL(Song24):
    """Song24 method using supervised learning feature space manifold."""
    def __init__(self, **kwargs):
        super().__init__(manifold_type="sl", **kwargs)
