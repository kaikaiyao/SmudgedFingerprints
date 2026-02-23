"""
VDVAE-FFHQ-256 model sampler implementation.

Based on VDVAE model from https://github.com/openai/vdvae
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


@ModelSampler.register("vdvae-ffhq-256")
class ModelSampler_VDVAE_FFHQ(ModelSampler):
    """
    VDVAE-FFHQ-256 model sampler.
    
    Handles loading and sampling from the VDVAE model
    trained on FFHQ-256 dataset.
    """
    
    def __init__(self, model_name: str, dataset: str, image_size: int, 
                 device: str = "cuda", batch_size: int = 32):
        # Force correct parameters for this specific model
        super().__init__(
            model_name="vdvae-ffhq-256", 
            dataset="ffhq", 
            image_size=256, 
            device=device, 
            batch_size=batch_size
        )
        
        self.model_urls = {
            "model": "https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets/ffhq256-iter-1700000-model.th",
            "ema_model": "https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets/ffhq256-iter-1700000-model-ema.th"
        }
        self.vae = None
        self.ema_vae = None
        
    def _setup_vdvae_repo(self) -> str:
        """Setup VDVAE repository and dependencies."""
        vdvae_dir = Path("libs/vdvae")
        
        if not vdvae_dir.exists():
            self.logger.info(f"Downloading VDVAE to {vdvae_dir}")
            vdvae_dir.mkdir(parents=True, exist_ok=True)
            
            # Clone the repository
            subprocess.run([
                "git", "clone", 
                "https://github.com/openai/vdvae.git",
                str(vdvae_dir)
            ], check=True)
            
            self.logger.info("✅ Downloaded VDVAE")
        else:
            self.logger.info(f"✅ VDVAE already exists at {vdvae_dir}")
        
        return str(vdvae_dir)
    
    def _install_vdvae_dependencies(self, vdvae_dir: str):
        """Install VDVAE dependencies."""
        try:
            # Add VDVAE directory to Python path
            if vdvae_dir not in sys.path:
                sys.path.insert(0, vdvae_dir)
                self.logger.info(f"Added VDVAE path to sys.path: {vdvae_dir}")
            
            # Install requirements
            try:
                subprocess.run([
                    "pip", "install", "imageio", "mpi4py", "scikit-learn"
                ], check=True)
                self.logger.info("✅ Installed VDVAE requirements")
            except Exception as e:
                self.logger.warning(f"Could not install requirements: {e}")
                
        except Exception as e:
            self.logger.error(f"Failed to setup VDVAE dependencies: {e}")
            raise
    
    def _download_model(self) -> Dict[str, str]:
        """Download the VDVAE model files."""
        model_dir = Path("libs/vdvae-ffhq-256")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_files = {}
        for key, url in self.model_urls.items():
            filename = url.split('/')[-1]
            filepath = model_dir / filename
            
            if not filepath.exists():
                self.logger.info(f"Downloading {key} from {url}")
                try:
                    urllib.request.urlretrieve(url, filepath)
                    self.logger.info(f"✅ Downloaded {key}")
                except Exception as e:
                    self.logger.error(f"Failed to download {key}: {e}")
                    raise
            else:
                self.logger.info(f"✅ {key} already exists at {filepath}")
            
            model_files[key] = str(filepath)
        
        return model_files
    
    def _load_model(self) -> None:
        """Load VDVAE-FFHQ-256 model."""
        try:
            # Setup VDVAE repository and dependencies
            vdvae_dir = self._setup_vdvae_repo()
            self._install_vdvae_dependencies(vdvae_dir)
            
            # Download model files
            model_files = self._download_model()
            
            self.logger.info(f"Loading VDVAE-FFHQ-256 from {model_files}")
            
            # Add VDVAE directory to Python path (at the beginning to take precedence)
            if vdvae_dir not in sys.path:
                sys.path.insert(0, vdvae_dir)
                self.logger.info(f"Added VDVAE path to sys.path: {vdvae_dir}")
            
            # Import the generator module
            try:
                import vae
                VAE = vae.VAE
                self.logger.info("✅ Successfully imported VDVAE modules")
                
                # Import their hyperparameters
                from hps import Hyperparams
                H = Hyperparams()
                
                # FFHQ-256 hyperparameters from their training code
                H.dataset = 'ffhq_256'
                H.image_size = 256
                H.image_channels = 3
                H.n_batch = self.batch_size
                H.lr = 0.00015
                H.epochs_per_eval = 1
                H.epochs_per_eval_save = 1
                H.num_images_visualize = 2
                H.num_variables_visualize = 3
                H.num_temperatures_visualize = 1
                
                # Model architecture (FFHQ-256 configuration)
                H.width = 512
                H.zdim = 16
                H.bottleneck_multiple = 0.25
                
                # Block architecture
                H.no_bias_above = 64
                H.custom_width_str = ''
                
                # Block architecture strings - these must be strings for VDVAE code
                # Using the correct FFHQ-256 configuration from VDVAE repository
                H.enc_blocks = "256x3,256d2,128x8,128d2,64x12,64d2,32x17,32d2,16x7,16d2,8x5,8d2,4x5,4d4,1x4"
                H.dec_blocks = "1x2,4m1,4x3,8m4,8x4,16m8,16x9,32m16,32x21,64m32,64x13,128m64,128x7,256m128"
                
                # Import VDVAE's own parsing function
                from vae import parse_layer_string
                
                # Parse encoder and decoder blocks using VDVAE's own logic
                enc_blocks_parsed = parse_layer_string(H.enc_blocks)
                dec_blocks_parsed = parse_layer_string(H.dec_blocks)
                
                self.logger.info(f"Parsed encoder blocks: {enc_blocks_parsed}")
                self.logger.info(f"Parsed decoder blocks: {dec_blocks_parsed}")
                
                # Let VDVAE handle the block configuration automatically
                # Don't manually set num_channels_enc/dec or use_3x3_enc/dec
                
                # Training parameters (FFHQ-256 configuration)
                H.ema = True
                H.ema_rate = 0.999
                H.grad_clip = 130.0
                H.skip_threshold = 180.0
                H.num_mixtures = 10
                H.warmup_iters = 100
                H.wd = 0.01  # Weight decay from cifar10
                
                # Sampling parameters
                H.num_images_visualize = self.batch_size
                H.num_variables_visualize = 0
                H.num_temperatures_visualize = 1
                H.sample_batch_size = self.batch_size
                
                # Initialize VAE
                self.vae = VAE(H).to(self.device)
                self.ema_vae = VAE(H).to(self.device)
                
                # Load the model weights
                self.logger.info("Loading model weights...")
                
                # Load and process model weights
                model_state = torch.load(model_files["model"], map_location=self.device)
                ema_state = torch.load(model_files["ema_model"], map_location=self.device)
                
                # Remove 'module.' prefix if it exists (from DistributedDataParallel)
                model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
                ema_state = {k.replace('module.', ''): v for k, v in ema_state.items()}
                
                # Load the processed weights
                self.vae.load_state_dict(model_state)
                self.ema_vae.load_state_dict(ema_state)
                
                self.logger.info("✅ Successfully loaded VDVAE model!")
                
                # Set models to evaluation mode
                self.vae.eval()
                self.ema_vae.eval()
                
                # Set the model for the base class
                self._model = self.ema_vae
                
            except ImportError as e:
                self.logger.error(f"Failed to import VDVAE modules: {e}")
                raise
            
        except Exception as e:
            self.logger.error(f"Failed to load VDVAE-FFHQ-256 model: {e}")
            raise
    
    def _generate_batch(self, batch_size: int, temperature: float = 0.7, **kwargs) -> torch.Tensor:
        """
        Generate a batch of images using VDVAE-FFHQ-256.
        
        Args:
            batch_size: Number of images to generate
            temperature: Sampling temperature (default: 0.7)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated images tensor of shape (batch_size, 3, 256, 256)
        """
        if self.ema_vae is None:
            raise RuntimeError("VDVAE model not loaded. Call load_model() first.")
        
        try:
            with torch.no_grad():
                # Generate samples using the EMA model with their sampling method
                # VDVAE returns numpy arrays in [0, 255] uint8 format
                images = self.ema_vae.forward_uncond_samples(
                    batch_size,
                    t=temperature  # They use temperature between 0.7-1.0
                )
                
                # Convert numpy array to PyTorch tensor and normalize to [0, 1]
                if isinstance(images, np.ndarray):
                    images = torch.from_numpy(images).float()
                    # Convert from [0, 255] to [0, 1]
                    images = images / 255.0
                    # Convert from (B, H, W, C) to (B, C, H, W)
                    images = images.permute(0, 3, 1, 2)
                
                return images
                
        except Exception as e:
            self.logger.error(f"Error generating images with VDVAE: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the VDVAE model."""
        return {
            "model_name": "vdvae-ffhq-256",
            "dataset": "ffhq",
            "image_size": 256,
            "model_type": "VDVAE",
            "device": self.device,
            "dtype": str(self.ema_vae.dtype) if self.ema_vae else None
        }
    
    @property
    def model(self) -> Optional[nn.Module]:
        """Get the underlying model."""
        return self.ema_vae