"""
GANFormer-FFHQ-256 model sampler implementation.

Based on GANFormer model from https://github.com/dorarad/gansformer
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
import urllib.request
import os
import sys
import subprocess

from .base import ModelSampler


@ModelSampler.register("ganformer-ffhq-256")
class ModelSampler_GANFormer_FFHQ(ModelSampler):
    """
    GANFormer-FFHQ-256 model sampler.
    
    Handles loading and sampling from the GANFormer model
    trained on FFHQ-256 dataset.
    """
    
    def __init__(self, model_name: str, dataset: str, image_size: int, 
                 device: str = "cuda", batch_size: int = 32):
        # Force correct parameters for this specific model
        super().__init__(
            model_name="ganformer-ffhq-256", 
            dataset="ffhq", 
            image_size=256, 
            device=device, 
            batch_size=batch_size
        )
        
        # Official GANformer FFHQ-256 PyTorch model URL
        self.model_url = "https://drive.google.com/uc?id=1-b0vwevUQs6LI_EybdO8XJ5uYSx63vEa"  # PyTorch pickle model
        self.generator = None
        self.latent_dim = 512  # GANFormer latent dimension
        self.components_num = 16  # FFHQ-256 uses 16 components (15 local + 1 global)
        
    def _setup_ganformer_repo(self) -> str:
        """Setup GANFormer repository and dependencies."""
        ganformer_dir = Path("libs/gansformer")
        
        if not ganformer_dir.exists():
            self.logger.info(f"Downloading GANFormer to {ganformer_dir}")
            ganformer_dir.mkdir(parents=True, exist_ok=True)
            
            # Clone the repository
            subprocess.run([
                "git", "clone", 
                "https://github.com/dorarad/gansformer.git",
                str(ganformer_dir)
            ], check=True)
            
            self.logger.info("✅ Downloaded GANFormer")
        else:
            self.logger.info(f"✅ GANFormer already exists at {ganformer_dir}")
        
        return str(ganformer_dir)
    
    def _install_ganformer_dependencies(self, ganformer_dir: str):
        """Install GANFormer dependencies."""
        try:
            # Add GANFormer directory to Python path
            if ganformer_dir not in sys.path:
                sys.path.insert(0, ganformer_dir)
                self.logger.info(f"Added GANFormer path to sys.path: {ganformer_dir}")
            
            # Install requirements
            try:
                subprocess.run([
                    "pip", "install", "torch>=1.6.0", "torchvision>=0.7.0", "einops", "termcolor"
                ], check=True)
                self.logger.info("✅ Installed GANFormer requirements")
            except Exception as e:
                self.logger.warning(f"Could not install requirements: {e}")
                
        except Exception as e:
            self.logger.error(f"Failed to setup GANFormer dependencies: {e}")
            raise
    
    def _download_model(self) -> str:
        """Download the GANFormer model file."""
        model_dir = Path("libs/ganformer-ffhq-256")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "ffhq256.pt"
        
        if not model_path.exists():
            self.logger.info(f"Downloading GANFormer model from {self.model_url}")
            
            try:
                # Download using gdown since it's a large file
                try:
                    import gdown
                except ImportError:
                    self.logger.info("Installing gdown for large file download...")
                    subprocess.run(["pip", "install", "gdown"], check=True)
                    import gdown
                
                # Download the model file using gdown
                gdown.download(self.model_url, str(model_path), quiet=False)
                self.logger.info("✅ Downloaded GANFormer model")
                
                # Validate the downloaded file
                if not model_path.exists() or model_path.stat().st_size < 1024 * 1024:
                    raise RuntimeError("Downloaded model file validation failed")
                    
            except Exception as e:
                self.logger.error(f"Failed to download GANFormer model: {e}")
                raise
        else:
            self.logger.info(f"✅ GANFormer model already exists at {model_path}")
        
        return str(model_path)
    
    def _load_model(self) -> None:
        """Load GANFormer-FFHQ-256 model."""
        try:
            # Setup GANFormer repository and dependencies
            ganformer_dir = self._setup_ganformer_repo()
            self._install_ganformer_dependencies(ganformer_dir)
            
            # Download model file
            model_path = self._download_model()
            
            self.logger.info(f"Loading GANFormer-FFHQ-256 from {model_path}")
            
            # Add GANFormer directory to Python path (at the beginning to take precedence)
            if ganformer_dir not in sys.path:
                sys.path.insert(0, ganformer_dir)
                self.logger.info(f"Added GANFormer path to sys.path: {ganformer_dir}")
            
            # Add pytorch_version to Python path
            pytorch_dir = os.path.join(ganformer_dir, "pytorch_version")
            if pytorch_dir not in sys.path:
                sys.path.insert(0, pytorch_dir)
            
            # Handle PyTorch import conflicts with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Import from training.networks
                    from training.networks import Generator
                    self.logger.info("✅ Successfully imported GANformer modules")
                    
                    # Import loader from pytorch_version
                    sys.path.insert(0, os.path.join(ganformer_dir, "pytorch_version"))
                    import loader
                    
                    # Load the model using GANFormer's custom loader
                    network_dict = loader.load_network(model_path)
                    
                    # Use Gs (generator moving-average) for higher quality images
                    self.generator = network_dict["Gs"]
                    self.generator.to(self.device)
                    self.generator.eval()
                    
                    # Set the model for the base class
                    self._model = self.generator
                    
                    self.logger.info("✅ Successfully loaded GANformer model!")
                    break  # Success, exit retry loop
                    
                except RuntimeError as e:
                    if "_InterpolationType" in str(e) and attempt < max_retries - 1:
                        self.logger.warning(f"PyTorch import conflict detected (attempt {attempt + 1}/{max_retries}), retrying...")
                        # Clear some PyTorch-related modules and retry
                        import gc
                        gc.collect()
                        # Remove some potentially conflicting modules
                        for module_name in list(sys.modules.keys()):
                            if any(prefix in module_name for prefix in ['torchvision', 'torch.nn.functional']):
                                try:
                                    del sys.modules[module_name]
                                except:
                                    pass
                        continue
                    else:
                        raise  # Re-raise if it's not the specific error or we've exhausted retries
            
        except Exception as e:
            self.logger.error(f"Failed to load GANformer-FFHQ-256 model: {e}")
            raise
    
    def _generate_batch(self, batch_size: int, truncation_psi: float = 0.7, **kwargs) -> torch.Tensor:
        """
        Generate a batch of images using GANFormer-FFHQ-256.
        
        Args:
            batch_size: Number of images to generate
            truncation_psi: Truncation parameter (0.7 is good for quality)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated images tensor of shape (batch_size, 3, 256, 256)
        """
        if self.generator is None:
            raise RuntimeError("GANFormer model not loaded. Call load_model() first.")
        
        try:
            with torch.no_grad():
                # Generate random latent codes matching GANformer's input shape
                z = torch.randn(batch_size, *self.generator.input_shape[1:], device=self.device)
                
                # Generate images using PyTorch GANformer's interface
                images = self.generator(
                    z,
                    truncation_psi=truncation_psi  # Truncation for better quality
                )[0]  # [0] to get images, discarding other outputs
                
                # Convert from [-1, 1] to [0, 1]
                images = (images + 1) / 2.0
                images = images.clamp(0, 1)
                
                return images
                
        except Exception as e:
            self.logger.error(f"Error generating images with GANFormer: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the GANFormer model."""
        return {
            "model_name": "ganformer-ffhq-256",
            "dataset": "ffhq",
            "image_size": 256,
            "model_type": "GANFormer",
            "latent_dim": self.latent_dim,
            "components_num": self.components_num,
            "device": self.device,
            "dtype": str(self.generator.dtype) if self.generator else None
        }
    
    @property
    def model(self) -> Optional[nn.Module]:
        """Get the underlying model."""
        return self.generator