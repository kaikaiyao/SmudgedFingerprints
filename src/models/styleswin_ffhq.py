"""
StyleSwin-FFHQ-256 model sampler implementation.

Based on StyleSwin model from https://github.com/microsoft/StyleSwin
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
import argparse
import math
from torch.serialization import add_safe_globals

from .base import ModelSampler


@ModelSampler.register("styleswin-ffhq-256")
class ModelSampler_StyleSwin_FFHQ(ModelSampler):
    """
    StyleSwin-FFHQ-256 model sampler.
    
    Handles loading and sampling from the StyleSwin model
    trained on FFHQ-256 dataset.
    """
    
    def __init__(self, model_name: str, dataset: str, image_size: int, 
                 device: str = "cuda", batch_size: int = 32):
        # Force correct parameters for this specific model
        super().__init__(
            model_name="styleswin-ffhq-256", 
            dataset="ffhq", 
            image_size=256, 
            device=device, 
            batch_size=batch_size
        )
        
        self.model_url = "https://drive.google.com/file/d/1OjYZ1zEWGNdiv0RFKv7KhXRmYko72LjO/view"
        self.generator = None
        self.style_dim = 512  # StyleSwin latent dimension
        
    def _setup_styleswin_repo(self) -> str:
        """Setup StyleSwin repository and dependencies."""
        styleswin_dir = Path("libs/StyleSwin")
        
        if not styleswin_dir.exists():
            self.logger.info(f"Downloading StyleSwin to {styleswin_dir}")
            styleswin_dir.mkdir(parents=True, exist_ok=True)
            
            # Clone the repository
            subprocess.run([
                "git", "clone", 
                "https://github.com/microsoft/StyleSwin.git",
                str(styleswin_dir)
            ], check=True)
            
            # Create a models package
            models_dir = os.path.join(styleswin_dir, "models")
            if not os.path.exists(os.path.join(models_dir, "__init__.py")):
                with open(os.path.join(models_dir, "__init__.py"), "w") as f:
                    f.write("")
            
            self.logger.info("✅ Downloaded StyleSwin")
        else:
            self.logger.info(f"✅ StyleSwin already exists at {styleswin_dir}")
        
        return str(styleswin_dir)
    
    def _install_styleswin_dependencies(self, styleswin_dir: str):
        """Install StyleSwin dependencies."""
        try:
            # Add StyleSwin directory to Python path
            if styleswin_dir not in sys.path:
                sys.path.insert(0, styleswin_dir)
                self.logger.info(f"Added StyleSwin path to sys.path: {styleswin_dir}")
            
            # Install requirements
            try:
                subprocess.run([
                    "pip", "install", "timm", "einops", "lmdb", "scipy", "scikit-learn"
                ], check=True)
                self.logger.info("✅ Installed StyleSwin requirements")
            except Exception as e:
                self.logger.warning(f"Could not install requirements: {e}")
                
        except Exception as e:
            self.logger.error(f"Failed to setup StyleSwin dependencies: {e}")
            raise
    
    def _download_model(self) -> str:
        """Download the StyleSwin model file."""
        model_dir = Path("libs/styleswin-ffhq-256")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "FFHQ_256.pt"
        
        if not model_path.exists():
            self.logger.info(f"Downloading StyleSwin model from {self.model_url}")
            
            try:
                # Install gdown if not already installed
                try:
                    import gdown
                except ImportError:
                    self.logger.info("Installing gdown for Google Drive download...")
                    subprocess.run(["pip", "install", "gdown"], check=True)
                    import gdown
                
                # Extract file ID from Google Drive URL
                file_id = "1OjYZ1zEWGNdiv0RFKv7KhXRmYko72LjO"
                
                # Download using gdown
                gdown.download(
                    f"https://drive.google.com/uc?id={file_id}",
                    str(model_path),
                    quiet=False
                )
                
                if not model_path.exists():
                    raise FileNotFoundError("Download appeared to succeed but file not found")
                
                # Validate the downloaded file
                if model_path.stat().st_size < 1024 * 1024:
                    raise RuntimeError("Downloaded model file validation failed")
                    
                self.logger.info("✅ Downloaded StyleSwin model")
                
            except Exception as e:
                self.logger.error(f"Failed to download StyleSwin model: {e}")
                raise
        else:
            self.logger.info(f"✅ StyleSwin model already exists at {model_path}")
        
        return str(model_path)
    
    def _load_model(self) -> None:
        """Load StyleSwin-FFHQ-256 model."""
        try:
            # Setup StyleSwin repository and dependencies
            styleswin_dir = self._setup_styleswin_repo()
            self._install_styleswin_dependencies(styleswin_dir)
            
            # Download model file
            model_path = self._download_model()
            
            self.logger.info(f"Loading StyleSwin-FFHQ-256 from {model_path}")
            
            # Add StyleSwin directory to Python path (at the beginning to take precedence)
            if styleswin_dir not in sys.path:
                sys.path.insert(0, styleswin_dir)
                self.logger.info(f"Added StyleSwin path to sys.path: {styleswin_dir}")
            
            # Add models directory to Python path
            models_dir = os.path.join(styleswin_dir, "models")
            if models_dir not in sys.path:
                sys.path.insert(0, models_dir)
            
            # Import directly from generator module
            sys.path.append(os.path.join(styleswin_dir, "models", "stylegan2"))  # For stylegan2 dependencies
            import generator
            from generator import Generator
            self.logger.info("✅ Successfully imported StyleSwin modules")
            
            # Initialize generator
            self.generator = Generator(
                size=256,
                style_dim=self.style_dim,
                n_mlp=8,
                channel_multiplier=2,
                enable_full_resolution=8,
                use_checkpoint=False
            ).to(self.device)
            
            # Add argparse.Namespace to safe globals for loading
            add_safe_globals([argparse.Namespace])
            
            # Load the model weights with weights_only=True for security
            try:
                ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
            except Exception as e:
                self.logger.warning(f"Failed to load with weights_only=True, attempting legacy load: {e}")
                # Fallback to legacy loading if needed - only for trusted checkpoints
                ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Try different checkpoint formats
            if 'g_ema' in ckpt:
                self.generator.load_state_dict(ckpt['g_ema'])
            elif 'generator' in ckpt:
                self.generator.load_state_dict(ckpt['generator'])
            elif 'model' in ckpt:
                self.generator.load_state_dict(ckpt['model'])
            else:
                self.generator.load_state_dict(ckpt)
            
            self.logger.info("✅ Successfully loaded StyleSwin model!")
            
            # Set model to evaluation mode
            self.generator.eval()
            
            # Set the model for the base class
            self._model = self.generator
            
        except Exception as e:
            self.logger.error(f"Failed to load StyleSwin-FFHQ-256 model: {e}")
            raise
    
    def _generate_batch(self, batch_size: int, truncation_psi: float = 0.7, **kwargs) -> torch.Tensor:
        """
        Generate a batch of images using StyleSwin-FFHQ-256.
        
        Args:
            batch_size: Number of images to generate
            truncation_psi: Truncation parameter (0.7 is good for quality)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated images tensor of shape (batch_size, 3, 256, 256)
        """
        if self.generator is None:
            raise RuntimeError("StyleSwin model not loaded. Call load_model() first.")
        
        try:
            with torch.no_grad():
                # Generate random latent vectors with correct batch size
                z = torch.randn(batch_size, self.style_dim, device=self.device)
                self.logger.info(f"Using batch size: {batch_size}")
                
                # Generate images with detailed logging
                self.logger.info(f"Generating with latent shape: {z.shape}, device: {z.device}")
                
                # Generate images using their original method
                # Looking at their train_styleswin.py, they use:
                # noise = torch.randn((args.batch, 512)).cuda()
                # fake_img, _ = generator(noise)
                try:
                    # First generate style vectors
                    styles = self.generator.style(z)
                    if styles is None:
                        raise ValueError("Style mapping network returned None")
                    
                    # Generate latent codes
                    latent = styles.unsqueeze(1).repeat(1, self.generator.n_latent, 1)
                    
                    # Generate constant input
                    x = self.generator.input(latent)
                    B, C, H, W = x.shape
                    x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
                    
                    # Pass through generator layers
                    skip = None
                    count = 0
                    for layer, to_rgb in zip(self.generator.layers, self.generator.to_rgbs):
                        x = layer(x, latent[:, count, :], latent[:, count + 1, :])
                        b, n, c = x.shape
                        h, w = int(math.sqrt(n)), int(math.sqrt(n))
                        skip = to_rgb(x.transpose(-1, -2).reshape(b, c, h, w), skip)
                        count = count + 2
                    
                    images = skip
                    
                    # Debug output type and shape
                    self.logger.info(f"Generated images shape: {images.shape}, device: {images.device}")
                except Exception as e:
                    self.logger.error(f"Detailed generation error: {str(e)}")
                    raise
                
                # Ensure we have a valid tensor before normalization
                if images is None:
                    raise ValueError("Generator returned None for images")
                if not torch.is_tensor(images):
                    raise ValueError(f"Generator returned non-tensor type: {type(images)}")
                
                # Convert from [-1, 1] to [0, 1] with bounds checking
                if torch.isnan(images).any():
                    raise ValueError("NaN values detected in generated images")
                    
                images = (images + 1) / 2.0
                images = images.clamp(0, 1)
                
                return images
                
        except Exception as e:
            self.logger.error(f"Error generating images with StyleSwin: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the StyleSwin model."""
        return {
            "model_name": "styleswin-ffhq-256",
            "dataset": "ffhq",
            "image_size": 256,
            "model_type": "StyleSwin",
            "latent_dim": self.style_dim,
            "device": self.device,
            "dtype": str(self.generator.dtype) if self.generator else None
        }
    
    @property
    def model(self) -> Optional[nn.Module]:
        """Get the underlying model."""
        return self.generator