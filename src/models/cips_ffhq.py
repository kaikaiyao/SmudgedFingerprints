"""
CIPS-FFHQ-256 model sampler implementation.

Based on CIPS model from https://github.com/advimman/CIPS
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


@ModelSampler.register("cips-ffhq-256")
class ModelSampler_CIPS_FFHQ(ModelSampler):
    """
    CIPS-FFHQ-256 model sampler.
    
    Handles loading and sampling from the CIPS model
    trained on FFHQ-256 dataset.
    """
    
    def __init__(self, model_name: str, dataset: str, image_size: int, 
                 device: str = "cuda", batch_size: int = 32):
        # Force correct parameters for this specific model
        super().__init__(
            model_name="cips-ffhq-256", 
            dataset="ffhq", 
            image_size=256, 
            device=device, 
            batch_size=batch_size
        )
        
        self.model_url = "https://drive.google.com/file/d/1JRd4ZpMDmlkbNlxnVvZx77Eyfac53KSq/view?usp=sharing"
        self.generator = None
        self.latent_dim = 512
        self.style_dim = 512
        self.truncation_latent = None
        
    def _setup_cips_repo(self) -> str:
        """Setup CIPS repository and dependencies."""
        cips_dir = Path("libs/CIPS")
        
        if not cips_dir.exists():
            self.logger.info(f"Downloading CIPS to {cips_dir}")
            cips_dir.mkdir(parents=True, exist_ok=True)
            
            # Clone the repository
            subprocess.run([
                "git", "clone", 
                "https://github.com/advimman/CIPS.git",
                str(cips_dir)
            ], check=True)
            
            self.logger.info("✅ Downloaded CIPS")
        else:
            self.logger.info(f"✅ CIPS already exists at {cips_dir}")
        
        return str(cips_dir)
    
    def _install_cips_dependencies(self, cips_dir: str):
        """Install CIPS dependencies."""
        try:
            # Add CIPS directory to Python path
            if cips_dir not in sys.path:
                sys.path.insert(0, cips_dir)
                self.logger.info(f"Added CIPS path to sys.path: {cips_dir}")
            
            # Install requirements
            try:
                subprocess.run([
                    "pip", "install", "-r", f"{cips_dir}/requirements.txt"
                ], check=True)
                self.logger.info("✅ Installed CIPS requirements")
            except Exception as e:
                self.logger.warning(f"Could not install requirements: {e}")
                
        except Exception as e:
            self.logger.error(f"Failed to setup CIPS dependencies: {e}")
            raise
    
    def _download_model(self) -> str:
        """Download the CIPS model file."""
        model_dir = Path("libs/cips-ffhq-256")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "ffhq256_g_ema.pt"
        
        if not model_path.exists():
            self.logger.info(f"Downloading CIPS model from {self.model_url}")
            
            try:
                # Install gdown if not already installed
                try:
                    import gdown
                except ImportError:
                    self.logger.info("Installing gdown for Google Drive download...")
                    import subprocess
                    subprocess.run(["pip", "install", "gdown"], check=True)
                    import gdown
                
                # Extract file ID from Google Drive URL
                file_id = "1JRd4ZpMDmlkbNlxnVvZx77Eyfac53KSq"
                
                # Download using gdown
                gdown.download(
                    f"https://drive.google.com/uc?id={file_id}",
                    str(model_path),
                    quiet=False
                )
                
                if not model_path.exists():
                    raise FileNotFoundError("Download appeared to succeed but file not found")
                    
                self.logger.info("✅ Downloaded CIPS model")
                
            except Exception as e:
                self.logger.error(f"Failed to download CIPS model: {e}")
                self.logger.warning(
                    "If automatic download fails, please download manually from:\n"
                    f"{self.model_url}\n"
                    f"and place it at: {model_path}"
                )
                raise
        else:
            self.logger.info(f"✅ CIPS model already exists at {model_path}")
        
        return str(model_path)
    
    def _load_model(self) -> None:
        """Load CIPS-FFHQ-256 model."""
        try:
            # Setup CIPS repository and dependencies
            cips_dir = self._setup_cips_repo()
            self._install_cips_dependencies(cips_dir)
            
            # Download model file
            model_path = self._download_model()
            
            self.logger.info(f"Loading CIPS-FFHQ-256 from {model_path}")
            
            # Import CIPS modules
            import model
            from tensor_transforms import convert_to_coord_format
            
            # Initialize generator with FFHQ-256 configuration
            self.generator = model.CIPSskip(
                size=256,
                hidden_size=512,
                style_dim=512,
                n_mlp=8,
                activation=None,
                channel_multiplier=2
            ).to(self.device)
            
            # Load the model weights
            ckpt = torch.load(model_path, map_location=self.device)
            self.generator.load_state_dict(ckpt)
            
            self.logger.info("✅ Successfully loaded CIPS model!")
            
            # Set model to evaluation mode
            self.generator.eval()
            
            # Compute truncation latent for better quality
            self.logger.info("Computing truncation latent...")
            self._compute_truncation_latent()
            
            # Set the model for the base class
            self._model = self.generator
            
            self.logger.info(f"Successfully moved generator to {self.device}")
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load CIPS-FFHQ-256 model: {e}")
            raise
    
    def _compute_truncation_latent(self):
        """Compute truncation latent for better image quality."""
        try:
            from tensor_transforms import convert_to_coord_format
            
            n_sample = 1
            latents = []
            
            # Generate coordinate format
            converted_full = convert_to_coord_format(
                n_sample, 256, 256, self.device, integer_values=False
            )
            
            with torch.no_grad():
                for _ in range(100):
                    sample_z = torch.randn(n_sample, self.latent_dim, device=self.device)
                    sample, latent = self.generator(
                        converted_full, [sample_z], return_latents=True
                    )
                    latents.append(latent.cpu())
            
            # Compute mean latent
            latents = torch.cat(latents, 0)
            self.truncation_latent = latents.mean(0).to(self.device)
            
            self.logger.info(f"✅ Computed truncation latent: {self.truncation_latent.shape}")
            
        except Exception as e:
            self.logger.warning(f"Could not compute truncation latent: {e}")
            self.truncation_latent = None
    
    def _generate_batch(self, batch_size: int, truncation: float = 0.6, **kwargs) -> torch.Tensor:
        """
        Generate a batch of images using CIPS-FFHQ-256.
        
        Args:
            batch_size: Number of images to generate
            truncation: Truncation parameter (default: 0.6)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated images tensor of shape (batch_size, 3, 256, 256)
        """
        if self.generator is None:
            raise RuntimeError("CIPS model not loaded. Call load_model() first.")
        
        try:
            from tensor_transforms import convert_to_coord_format
            
            with torch.no_grad():
                # Generate random latent codes
                sample_z = torch.randn(batch_size, self.latent_dim, device=self.device)
                
                # Generate coordinate format
                converted_full = convert_to_coord_format(
                    batch_size, 256, 256, self.device, integer_values=False
                )
                
                # Generate images with truncation trick
                if self.truncation_latent is not None:
                    style = self.generator.style(sample_z)
                    images, _ = self.generator(
                        converted_full, [style],
                        truncation=truncation,
                        truncation_latent=self.truncation_latent,
                        input_is_latent=True
                    )
                else:
                    # Fallback without truncation
                    images, _ = self.generator(converted_full, [sample_z], return_latents=True)
                
                # Convert from [-1, 1] to [0, 1]
                images = (images + 1) / 2.0
                images = images.clamp(0, 1)
                
                return images
                
        except Exception as e:
            self.logger.error(f"Error generating images with CIPS: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the CIPS model."""
        return {
            "model_name": "cips-ffhq-256",
            "dataset": "ffhq",
            "image_size": 256,
            "model_type": "CIPS",
            "latent_dim": self.latent_dim,
            "style_dim": self.style_dim,
            "device": self.device,
            "dtype": str(self.generator.dtype) if self.generator else None
        }
    
    @property
    def model(self) -> Optional[nn.Module]:
        """Get the underlying model."""
        return self.generator
