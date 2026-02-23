"""
StyleGAN2-FFHQ-256 model sampler implementation.

Based on StyleGAN2 model from https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq140k-paper256-noaug.pkl
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
import urllib.request
import os
import sys

from .base import ModelSampler


@ModelSampler.register("stylegan2-ffhq-256")
class ModelSampler_StyleGAN2_FFHQ(ModelSampler):
    """
    StyleGAN2-FFHQ-256 model sampler.
    
    Handles loading and sampling from the StyleGAN2 model
    trained on FFHQ-256 dataset.
    """
    
    def __init__(self, model_name: str, dataset: str, image_size: int, 
                 device: str = "cuda", batch_size: int = 32):
        # Force correct parameters for this specific model
        super().__init__(
            model_name="stylegan2-ffhq-256", 
            dataset="ffhq", 
            image_size=256, 
            device=device, 
            batch_size=batch_size
        )
        
        self.model_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq140k-paper256-noaug.pkl"
        self.generator = None
        self.z_dim = 512  # StyleGAN2 latent dimension
        self.w_dim = 512  # StyleGAN2 W dimension
        self.c_dim = 0    # FFHQ is unconditional
        
    def _setup_stylegan2_repo(self) -> str:
        """Setup StyleGAN2 repository and dependencies."""
        stylegan2_dir = Path("libs/stylegan2-ada-pytorch")
        
        if not stylegan2_dir.exists():
            self.logger.info(f"Downloading StyleGAN2-ADA-PyTorch to {stylegan2_dir}")
            stylegan2_dir.mkdir(parents=True, exist_ok=True)
            
            # Clone the repository
            import subprocess
            subprocess.run([
                "git", "clone", 
                "https://github.com/NVlabs/stylegan2-ada-pytorch.git",
                str(stylegan2_dir)
            ], check=True)
            
            self.logger.info("✅ Downloaded StyleGAN2-ADA-PyTorch")
        else:
            self.logger.info(f"✅ StyleGAN2-ADA-PyTorch already exists at {stylegan2_dir}")
        
        return str(stylegan2_dir)
    
    def _install_stylegan2_dependencies(self, stylegan2_dir: str):
        """Install StyleGAN2 dependencies."""
        try:
            import subprocess
            
            # Add StyleGAN2 directory to Python path
            if stylegan2_dir not in sys.path:
                sys.path.insert(0, stylegan2_dir)
                self.logger.info(f"Added StyleGAN2 path to sys.path: {stylegan2_dir}")
            
            # Try to import dnnlib and legacy
            try:
                import dnnlib
                import legacy
                self.logger.info("✅ Successfully imported dnnlib and legacy from StyleGAN2")
            except ImportError as e:
                self.logger.warning(f"Could not import StyleGAN2 modules: {e}")
                # Try to install dependencies
                try:
                    subprocess.run([
                        "pip", "install", "-r", f"{stylegan2_dir}/requirements.txt"
                    ], check=True)
                    self.logger.info("✅ Installed StyleGAN2 requirements")
                except Exception as e:
                    self.logger.warning(f"Could not install requirements: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to setup StyleGAN2 dependencies: {e}")
            raise
    
    def _download_model(self) -> str:
        """Download the StyleGAN2 model file."""
        model_dir = Path("libs/stylegan2-ffhq-256")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "ffhq140k-paper256-noaug.pkl"
        
        if not model_path.exists():
            self.logger.info(f"Downloading StyleGAN2 model from {self.model_url}")
            
            try:
                # Download the model file
                urllib.request.urlretrieve(self.model_url, model_path)
                self.logger.info("✅ Downloaded StyleGAN2 model")
            except Exception as e:
                self.logger.error(f"Failed to download StyleGAN2 model: {e}")
                raise
        else:
            self.logger.info(f"✅ StyleGAN2 model already exists at {model_path}")
        
        return str(model_path)
    
    def _load_model(self) -> None:
        """Load StyleGAN2-FFHQ-256 model from pickle file."""
        try:
            # Setup StyleGAN2 repository and dependencies
            stylegan2_dir = self._setup_stylegan2_repo()
            self._install_stylegan2_dependencies(stylegan2_dir)
            
            # Download model file
            model_path = self._download_model()
            
            self.logger.info(f"Loading StyleGAN2-FFHQ-256 from {model_path}")
            
            # Import StyleGAN2 modules
            import dnnlib
            import legacy
            
            # Load the model
            with dnnlib.util.open_url(model_path) as f:
                network = legacy.load_network_pkl(f)
                self.generator = network['G_ema'].to(self.device)
            
            self.logger.info("✅ Successfully loaded StyleGAN2 model from pickle file!")
            
            # Get model info
            self.z_dim = self.generator.z_dim
            self.w_dim = self.generator.w_dim
            self.c_dim = self.generator.c_dim
            
            self.logger.info(f"Model z_dim: {self.z_dim}")
            self.logger.info(f"Model w_dim: {self.w_dim}")
            self.logger.info(f"Model c_dim: {self.c_dim}")
            
            # Set model to evaluation mode
            self.generator.eval()
            
            # Set the model for the base class
            self._model = self.generator
            
            self.logger.info(f"Successfully moved generator to {self.device}")
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load StyleGAN2-FFHQ-256 model: {e}")
            raise
    
    def _generate_batch(self, batch_size: int, truncation_psi: float = 0.7, 
                       noise_mode: str = 'const', **kwargs) -> torch.Tensor:
        """
        Generate a batch of images using StyleGAN2-FFHQ-256.
        
        Args:
            batch_size: Number of images to generate
            truncation_psi: Truncation parameter (0.7 is good for quality)
            noise_mode: Noise mode ('const', 'random', 'none')
            **kwargs: Additional generation parameters
            
        Returns:
            Generated images tensor of shape (batch_size, 3, 256, 256)
        """
        if self.generator is None:
            raise RuntimeError("StyleGAN2 model not loaded. Call load_model() first.")
        
        try:
            with torch.no_grad():
                # Generate random latent codes
                z = torch.randn(batch_size, self.z_dim, device=self.device)
                
                # Create labels (unconditional for FFHQ)
                if self.c_dim > 0:
                    labels = torch.zeros(batch_size, self.c_dim, device=self.device)
                else:
                    labels = None
                
                # Generate images
                images = self.generator(z, labels, truncation_psi=truncation_psi, noise_mode=noise_mode)
                
                # Convert to proper format [0, 1]
                # StyleGAN2 outputs are in [-1, 1], so we need to convert to [0, 1]
                images = (images + 1) / 2.0
                images = images.clamp(0, 1)
                
                # self.logger.info(f"✅ Generated {batch_size} images with StyleGAN2")
                return images
                
        except Exception as e:
            self.logger.error(f"Error generating images with StyleGAN2: {e}")
            raise
    
    def generate_with_seeds(self, seeds: list, truncation_psi: float = 0.7, 
                           noise_mode: str = 'const', **kwargs) -> torch.Tensor:
        """
        Generate images with specific random seeds.
        
        Args:
            seeds: List of random seeds
            truncation_psi: Truncation parameter
            noise_mode: Noise mode
            **kwargs: Additional generation parameters
            
        Returns:
            Generated images tensor
        """
        if self.generator is None:
            raise RuntimeError("StyleGAN2 model not loaded. Call load_model() first.")
        
        images = []
        for seed in seeds:
            # Set seed for reproducibility
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            
            # Generate single image
            with torch.no_grad():
                z = torch.from_numpy(np.random.RandomState(seed).randn(1, self.z_dim)).to(self.device)
                
                # Create labels (unconditional for FFHQ)
                if self.c_dim > 0:
                    labels = torch.zeros(1, self.c_dim, device=self.device)
                else:
                    labels = None
                
                # Generate image
                image = self.generator(z, labels, truncation_psi=truncation_psi, noise_mode=noise_mode)
                
                # Convert to proper format [0, 1]
                image = (image + 1) / 2.0
                image = image.clamp(0, 1)
                
                images.append(image)
        
        # Concatenate all images
        return torch.cat(images, dim=0)
    
    def interpolate_latents(self, z1: torch.Tensor, z2: torch.Tensor, 
                           num_steps: int = 10, truncation_psi: float = 0.7) -> torch.Tensor:
        """
        Interpolate between two latent codes.
        
        Args:
            z1: First latent code
            z2: Second latent code
            num_steps: Number of interpolation steps
            truncation_psi: Truncation parameter
            
        Returns:
            Interpolated images tensor
        """
        if self.generator is None:
            raise RuntimeError("StyleGAN2 model not loaded. Call load_model() first.")
        
        try:
            with torch.no_grad():
                # Create interpolation weights
                weights = torch.linspace(0, 1, num_steps, device=self.device)
                
                images = []
                for weight in weights:
                    # Interpolate latent codes
                    z_interp = z1 * (1 - weight) + z2 * weight
                    
                    # Create labels (unconditional for FFHQ)
                    if self.c_dim > 0:
                        labels = torch.zeros(1, self.c_dim, device=self.device)
                    else:
                        labels = None
                    
                    # Generate image
                    image = self.generator(z_interp, labels, truncation_psi=truncation_psi)
                    
                    # Convert to proper format [0, 1]
                    image = (image + 1) / 2.0
                    image = image.clamp(0, 1)
                    
                    images.append(image)
                
                return torch.cat(images, dim=0)
                
        except Exception as e:
            self.logger.error(f"Error interpolating latents: {e}")
            raise
    
    def sample(self, num_images: int, save_to_file: bool = True, 
               seed: Optional[int] = None, truncation_psi: float = 0.7, **kwargs) -> torch.Tensor:
        """
        Sample images from the StyleGAN2 model.
        
        Args:
            num_images: Number of images to generate
            save_to_file: Whether to save images to file
            seed: Random seed for reproducibility
            truncation_psi: Truncation parameter
            **kwargs: Additional sampling parameters
            
        Returns:
            Generated images tensor of shape (num_images, 3, 256, 256)
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        if self.generator is None:
            self.load_model()
        
        # Generate images in batches
        all_images = []
        for i in range(0, num_images, self.batch_size):
            batch_size = min(self.batch_size, num_images - i)
            batch_images = self._generate_batch(batch_size, truncation_psi=truncation_psi, **kwargs)
            all_images.append(batch_images)
        
        # Concatenate all batches
        images = torch.cat(all_images, dim=0)
        
        # Save images if requested
        if save_to_file:
            self._save_images(images, num_images)
        
        return images
    
    def _save_images(self, images: torch.Tensor, num_images: int) -> None:
        """Save generated images to file."""
        try:
            # Convert to numpy and save
            images_np = images.cpu().numpy()
            
            for i in range(min(num_images, len(images_np))):
                image = images_np[i]
                # Convert from [0, 1] to [0, 255] and transpose to (H, W, C)
                image = (image * 255).astype(np.uint8).transpose(1, 2, 0)
                
                # Save image using the base class method
                filepath = self.data_paths.get_images_path(
                    self.model_name, self.dataset, self.image_size, num_images
                )
                
                # Create directory if it doesn't exist
                filepath.parent.mkdir(parents=True, exist_ok=True)
                
                # Use PIL to save the image
                from PIL import Image
                pil_image = Image.fromarray(image)
                
                # Save with a more descriptive name
                sample_path = filepath.parent / f"{self.model_name}_sample_{i+1}.png"
                pil_image.save(sample_path)
                
                self.logger.info(f"✅ Saved image {i+1} to {sample_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to save images: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the StyleGAN2 model."""
        return {
            "model_name": "stylegan2-ffhq-256",
            "dataset": "ffhq",
            "image_size": 256,
            "model_type": "StyleGAN2",
            "latent_dim": self.z_dim,
            "w_dim": self.w_dim,
            "c_dim": self.c_dim,
            "device": self.device,
            "dtype": str(self.generator.dtype) if self.generator else None
        }
    
    @property
    def model(self) -> Optional[nn.Module]:
        """Get the underlying model."""
        return self.generator