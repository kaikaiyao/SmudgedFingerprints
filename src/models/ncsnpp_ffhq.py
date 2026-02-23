"""
NCSNPP-FFHQ-256 model sampler implementation.

Based on original PyTorch implementation from https://github.com/yang-song/score_sde_pytorch
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
import os
import sys
import subprocess
import urllib.request
import zipfile

from .base import ModelSampler


@ModelSampler.register("ncsnpp-ffhq-256")
class ModelSampler_NCSNPP_FFHQ(ModelSampler):
    """
    NCSNPP-FFHQ-256 model sampler using original PyTorch implementation.
    
    Handles loading and sampling from the Score-based SDE model
    trained on FFHQ-256 dataset using the original repository.
    """
    
    def __init__(self, model_name: str, dataset: str, image_size: int, 
                 device: str = "cuda", batch_size: int = 32, fast_sampling: bool = True):
        # Force correct parameters for this specific model
        super().__init__(
            model_name="ncsnpp-ffhq-256", 
            dataset="ffhq", 
            image_size=256, 
            device=device, 
            batch_size=batch_size
        )
        
        self.fast_sampling = fast_sampling
        self.repo_url = "https://github.com/yang-song/score_sde_pytorch.git"
        self.checkpoint_url = "https://drive.google.com/uc?id=1-mtdSwuefIZA0n85QWScQo2WRvJNWwUy"
        self.checkpoint_path = None
        self.score_model = None
        self.sde = None
        self.config = None
        self.sampling_fn = None
        
    def _setup_repository(self):
        """Setup the score_sde_pytorch repository in libs folder."""
        libs_dir = Path("libs")
        libs_dir.mkdir(exist_ok=True)
        
        repo_dir = libs_dir / "score_sde_pytorch"
        
        if not repo_dir.exists():
            self.logger.info(f"Cloning {self.repo_url} to {repo_dir}")
            try:
                subprocess.run([
                    "git", "clone", self.repo_url, str(repo_dir)
                ], check=True, capture_output=True)
                self.logger.info("Successfully cloned repository")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to clone repository: {e}")
                raise
        
        # Install dependencies
        self._install_dependencies(repo_dir)
        
        # Add repository to Python path
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))
        
        return repo_dir
    
    def _install_dependencies(self, repo_dir: Path):
        """Install dependencies for score_sde_pytorch."""
        try:
            self.logger.info("Installing score_sde_pytorch dependencies...")
            
            # Essential dependencies for score_sde_pytorch
            dependencies = [
                "tensorflow>=2.8.0",
                "tensorflow-datasets>=4.5.0",
                "tensorflow-gan>=0.3.0",
                "jax>=0.4.0",
                "jaxlib>=0.4.0",
                "matplotlib>=3.5.0",
                "seaborn>=0.11.0",
                "pandas>=1.3.0",
                "ml_collections>=0.1.0"
            ]
            
            # Install dependencies
            for dep in dependencies:
                try:
                    subprocess.run([
                        "pip", "install", dep
                    ], check=True, capture_output=True, timeout=120)
                    self.logger.debug(f"Installed {dep}")
                except subprocess.CalledProcessError as e:
                    self.logger.warning(f"Failed to install {dep}: {e}")
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"Timeout installing {dep}")
            
            self.logger.info("score_sde_pytorch dependencies installation completed")
            
        except Exception as e:
            self.logger.warning(f"Error installing score_sde_pytorch dependencies: {e}")
            self.logger.info("You may need to install these manually: tensorflow, tensorflow-datasets, tensorflow-gan, jax, jaxlib, matplotlib, seaborn, pandas, ml_collections")
    
    def _download_checkpoint(self):
        """Download the FFHQ-256 checkpoint."""
        libs_dir = Path("libs")
        exp_dir = libs_dir / "score_sde_pytorch" / "exp" / "ve" / "ffhq256_ncsnpp_continuous"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = exp_dir / "checkpoint_24.pth"
        
        if not checkpoint_path.exists():
            self.logger.info("Downloading FFHQ-256 checkpoint...")
            try:
                # Try to install gdown if not available
                try:
                    import gdown
                except ImportError:
                    self.logger.info("Installing gdown for Google Drive download...")
                    subprocess.run(["pip", "install", "gdown"], check=True)
                    import gdown
                
                gdown.download(
                    self.checkpoint_url,
                    str(checkpoint_path),
                    quiet=False
                )
                
                # Validate the downloaded file
                if not checkpoint_path.exists():
                    raise FileNotFoundError("Download appeared to succeed but file not found")
                
                if checkpoint_path.stat().st_size < 1024 * 1024:  # Less than 1MB
                    raise RuntimeError("Downloaded checkpoint file is too small, download may have failed")
                
                self.logger.info("Successfully downloaded checkpoint")
            except Exception as e:
                self.logger.error(f"Failed to download checkpoint: {e}")
                self.logger.warning(
                    "If automatic download fails, please download manually from:\n"
                    f"{self.checkpoint_url}\n"
                    f"and place it at: {checkpoint_path}"
                )
                raise
        else:
            self.logger.info(f"✅ Checkpoint already exists at {checkpoint_path}")
        
        return checkpoint_path
    
    def _load_model(self) -> None:
        """Load NCSNPP-FFHQ-256 model from original PyTorch implementation."""
        try:
            # Setup repository
            repo_dir = self._setup_repository()
            
            # Download checkpoint
            self.checkpoint_path = self._download_checkpoint()
            
            # Import required modules from the repository
            self.logger.info("Importing modules from score_sde_pytorch repository")
            
            # Import core modules
            from sde_lib import VESDE
            from models import utils as mutils
            from models import ncsnpp
            from losses import get_optimizer
            from models.ema import ExponentialMovingAverage
            from utils import restore_checkpoint
            import sampling
            import datasets
            
            # Import config - we need to create a config for FFHQ-256
            # Based on the notebook, we'll use VESDE configuration
            self.logger.info("Setting up VESDE configuration for FFHQ-256")
            
            # Import ml_collections for proper config structure
            import ml_collections
            
            # Create a proper config using ml_collections.ConfigDict
            config = ml_collections.ConfigDict()
            
            # Training parameters
            config.training = training = ml_collections.ConfigDict()
            training.batch_size = self.batch_size
            training.n_iters = 200000
            training.snapshot_freq = 5000
            training.log_freq = 50
            training.eval_freq = 100
            training.snapshot_freq_for_preemption = 5000
            training.snapshot_sampling = True
            training.likelihood_weighting = False
            training.continuous = True
            training.reduce_mean = True
            training.sde = 'vesde'
            
            # Sampling parameters
            config.sampling = sampling = ml_collections.ConfigDict()
            sampling.n_steps_each = 1
            sampling.noise_removal = True
            sampling.probability_flow = False
            sampling.snr = 0.075
            sampling.method = 'pc'
            sampling.predictor = 'reverse_diffusion'
            sampling.corrector = 'langevin'
            
            # Evaluation parameters
            config.eval = evaluate = ml_collections.ConfigDict()
            evaluate.begin_ckpt = 1
            evaluate.end_ckpt = 26
            evaluate.batch_size = self.batch_size
            evaluate.enable_sampling = True
            evaluate.num_samples = 50000
            evaluate.enable_loss = True
            evaluate.enable_bpd = False
            evaluate.bpd_dataset = 'test'
            
            # Data parameters
            config.data = data = ml_collections.ConfigDict()
            data.dataset = 'FFHQ'
            data.image_size = 256
            data.num_channels = 3
            data.random_flip = True
            data.uniform_dequantization = False
            data.centered = True
            data.tfrecords_path = '/home/yangsong/ncsc/ffhq/ffhq-r08.tfrecords'
            
            # Model parameters (based on ffhq_256_ncsnpp_continuous.py)
            config.model = model = ml_collections.ConfigDict()
            model.name = 'ncsnpp'
            model.sigma_max = 348
            model.sigma_min = 0.01
            # Use fewer steps for faster sampling
            model.num_scales = 200 if self.fast_sampling else 2000
            model.beta_min = 0.1
            model.beta_max = 20.0
            model.scale_by_sigma = True
            model.ema_rate = 0.999
            model.normalization = 'GroupNorm'
            model.nonlinearity = 'swish'
            model.nf = 128
            model.ch_mult = (1, 1, 2, 2, 2, 2, 2)
            model.num_res_blocks = 2
            model.attn_resolutions = (16,)
            model.resamp_with_conv = True
            model.conditional = True
            model.fir = True
            model.fir_kernel = [1, 3, 3, 1]
            model.skip_rescale = True
            model.resblock_type = 'biggan'
            model.progressive = 'output_skip'
            model.progressive_input = 'input_skip'
            model.progressive_combine = 'sum'
            model.attention_type = 'ddpm'
            model.init_scale = 0.
            model.fourier_scale = 16
            model.conv_size = 3
            model.dropout = 0.
            model.embedding_type = 'fourier'
            
            # Optimization parameters
            config.optim = optim = ml_collections.ConfigDict()
            optim.weight_decay = 0
            optim.optimizer = 'Adam'
            optim.lr = 2e-4
            optim.beta1 = 0.9
            optim.eps = 1e-8
            optim.warmup = 5000
            optim.grad_clip = 1.
            
            # Other parameters
            config.seed = 42
            config.device = self.device
            config.name = 'ffhq256_ncsnpp_continuous'
            
            self.config = config
            
            # Setup SDE
            self.sde = VESDE(
                sigma_min=self.config.model.sigma_min,
                sigma_max=self.config.model.sigma_max,
                N=self.config.model.num_scales
            )
            
            # Move SDE tensors to the correct device
            if hasattr(self.sde, 'discrete_sigmas'):
                self.sde.discrete_sigmas = self.sde.discrete_sigmas.to(self.device)
            if hasattr(self.sde, 'discrete_betas'):
                self.sde.discrete_betas = self.sde.discrete_betas.to(self.device)
            if hasattr(self.sde, 'alphas'):
                self.sde.alphas = self.sde.alphas.to(self.device)
            if hasattr(self.sde, 'alphas_cumprod'):
                self.sde.alphas_cumprod = self.sde.alphas_cumprod.to(self.device)
            if hasattr(self.sde, 'sqrt_alphas_cumprod'):
                self.sde.sqrt_alphas_cumprod = self.sde.sqrt_alphas_cumprod.to(self.device)
            if hasattr(self.sde, 'sqrt_1m_alphas_cumprod'):
                self.sde.sqrt_1m_alphas_cumprod = self.sde.sqrt_1m_alphas_cumprod.to(self.device)
            
            # Create model
            self.logger.info("Creating NCSN++ model")
            self.score_model = mutils.create_model(self.config)
            self.score_model = self.score_model.to(self.device)
            
            # Setup optimizer and EMA
            optimizer = get_optimizer(self.config, self.score_model.parameters())
            ema = ExponentialMovingAverage(
                self.score_model.parameters(),
                decay=self.config.model.ema_rate
            )
            
            # Create state dict
            state = dict(
                step=0,
                optimizer=optimizer,
                model=self.score_model,
                ema=ema
            )
            
            # Load checkpoint
            self.logger.info(f"Loading checkpoint from {self.checkpoint_path}")
            try:
                state = restore_checkpoint(str(self.checkpoint_path), state, self.device)
                ema.copy_to(self.score_model.parameters())
                self.logger.info("✅ Successfully loaded checkpoint")
            except Exception as e:
                self.logger.error(f"Failed to load checkpoint: {e}")
                self.logger.error("This might be due to:")
                self.logger.error("1. Corrupted checkpoint file")
                self.logger.error("2. Incompatible model architecture")
                self.logger.error("3. Missing dependencies")
                raise
            
            # Setup sampling function
            self.logger.info("Setting up sampling function")
            self._setup_sampling_function()
            
            self._model = self.score_model
            self.logger.info("Successfully loaded NCSN++ FFHQ-256 model")
            
        except Exception as e:
            self.logger.error(f"Failed to load NCSNPP-FFHQ-256 model: {e}")
            raise e
    
    def _setup_sampling_function(self):
        """Setup the sampling function for image generation."""
        from models import utils as mutils
        import sampling
        
        # Define scaler functions directly to avoid import conflicts
        def get_data_scaler(config):
            """Data scaler for FFHQ-256."""
            if config.data.centered:
                # For centered data, scale to [-1, 1]
                return lambda x: x
            else:
                # For non-centered data, scale to [0, 1]
                return lambda x: x
        
        def get_data_inverse_scaler(config):
            """Inverse data scaler for FFHQ-256."""
            if config.data.centered:
                # For centered data, scale from [-1, 1] to [0, 1]
                return lambda x: (x + 1.) / 2.
            else:
                # For non-centered data, no scaling needed
                return lambda x: x
        
        # Get sigmas and scalers
        sigmas = mutils.get_sigmas(self.config)
        scaler = get_data_scaler(self.config)
        inverse_scaler = get_data_inverse_scaler(self.config)
        
        # Setup shape for sampling
        img_size = self.config.data.image_size
        channels = self.config.data.num_channels
        shape = (self.batch_size, channels, img_size, img_size)
        
        # Use PC sampling with recommended settings for FFHQ-256
        # From notebook: VE + ReverseDiffusionPredictor + LangevinCorrector + snr=0.075 + n_steps=1
        from sampling import ReverseDiffusionPredictor, LangevinCorrector
        
        predictor = ReverseDiffusionPredictor
        corrector = LangevinCorrector
        
        # Adjust parameters for fast sampling
        if self.fast_sampling:
            snr = 0.15  # Higher SNR for faster convergence
            n_steps = 1
            sampling_eps = 1e-3  # Higher epsilon for faster convergence
        else:
            snr = 0.075  # Recommended for FFHQ-256
            n_steps = 1
            sampling_eps = 1e-5
        
        self.sampling_fn = sampling.get_pc_sampler(
            self.sde, shape, predictor, corrector,
            inverse_scaler, snr, n_steps=n_steps,
            probability_flow=False,
            continuous=self.config.training.continuous,
            eps=sampling_eps, device=self.device
        )
    
    def _setup_sampling_function_with_batch_size(self, batch_size: int):
        """Setup the sampling function with a specific batch size."""
        from models import utils as mutils
        import sampling
        
        # Define scaler functions directly to avoid import conflicts
        def get_data_scaler(config):
            """Data scaler for FFHQ-256."""
            if config.data.centered:
                # For centered data, scale to [-1, 1]
                return lambda x: x
            else:
                # For non-centered data, scale to [0, 1]
                return lambda x: x
        
        def get_data_inverse_scaler(config):
            """Inverse data scaler for FFHQ-256."""
            if config.data.centered:
                # For centered data, scale from [-1, 1] to [0, 1]
                return lambda x: (x + 1.) / 2.
            else:
                # For non-centered data, no scaling needed
                return lambda x: x
        
        # Get sigmas and scalers
        sigmas = mutils.get_sigmas(self.config)
        scaler = get_data_scaler(self.config)
        inverse_scaler = get_data_inverse_scaler(self.config)
        
        # Setup shape for sampling
        img_size = self.config.data.image_size
        channels = self.config.data.num_channels
        shape = (batch_size, channels, img_size, img_size)
        
        # Use PC sampling with recommended settings for FFHQ-256
        from sampling import ReverseDiffusionPredictor, LangevinCorrector
        
        predictor = ReverseDiffusionPredictor
        corrector = LangevinCorrector
        
        # Adjust parameters for fast sampling
        if self.fast_sampling:
            snr = 0.15  # Higher SNR for faster convergence
            n_steps = 1
            sampling_eps = 1e-3  # Higher epsilon for faster convergence
        else:
            snr = 0.075  # Recommended for FFHQ-256
            n_steps = 1
            sampling_eps = 1e-5
        
        return sampling.get_pc_sampler(
            self.sde, shape, predictor, corrector,
            inverse_scaler, snr, n_steps=n_steps,
            probability_flow=False,
            continuous=self.config.training.continuous,
            eps=sampling_eps, device=self.device
        )
    
    def _generate_batch(self, batch_size: int, num_inference_steps: int = None, 
                       **kwargs) -> torch.Tensor:
        """
        Generate a batch of images using NCSNPP-FFHQ-256.
        
        Args:
            batch_size: Number of images to generate
            num_inference_steps: Ignored for this model (uses fixed SDE steps)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated images tensor of shape (batch_size, 3, 256, 256)
        """
        if self.score_model is None or self.sampling_fn is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Generate images using the sampling function
        with torch.no_grad():
            x, n = self.sampling_fn(self.score_model)
        
        # Ensure images are in [0, 1] range and correct format
        x = torch.clamp(x, 0.0, 1.0)
        
        # Validate image format
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {x.dim()}D")
        
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got {x.shape[1]}")
        
        if x.shape[2] != 256 or x.shape[3] != 256:
            raise ValueError(f"Expected 256x256 images, got {x.shape[2]}x{x.shape[3]}")
        
        self.logger.debug(f"Generated images with shape: {x.shape}")
        
        # If we need more images than the batch size, generate multiple batches
        if batch_size > self.batch_size:
            all_images = [x]
            remaining = batch_size - self.batch_size
            
            while remaining > 0:
                current_batch = min(remaining, self.batch_size)
                # Create sampling function with specific batch size
                sampling_fn = self._setup_sampling_function_with_batch_size(current_batch)
                
                with torch.no_grad():
                    batch_x, _ = sampling_fn(self.score_model)
                
                batch_x = torch.clamp(batch_x, 0.0, 1.0)
                all_images.append(batch_x)
                
                remaining -= current_batch
            
            x = torch.cat(all_images, dim=0)
        
        return x[:batch_size]  # Ensure exact batch size
    
    def generate_with_seeds(self, seeds: list, num_inference_steps: int = None, 
                           **kwargs) -> torch.Tensor:
        """
        Generate images with specific seeds for reproducibility.
        
        Args:
            seeds: List of random seeds
            num_inference_steps: Ignored for this model
            **kwargs: Additional generation parameters
            
        Returns:
            Generated images tensor
        """
        self.load_model()
        
        all_images = []
        
        for seed in seeds:
            # Set random seed for reproducibility
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            
            # Generate single image
            with torch.no_grad():
                x, _ = self.sampling_fn(self.score_model)
            
            x = torch.clamp(x, 0.0, 1.0)
            all_images.append(x)
        
        images = torch.cat(all_images, dim=0)
        return images
    
    def generate_fast(self, batch_size: int, num_inference_steps: int = None, 
                     **kwargs) -> torch.Tensor:
        """
        Generate images with fast sampling settings (10x speedup).
        
        Args:
            batch_size: Number of images to generate
            num_inference_steps: Ignored for this model
            **kwargs: Additional parameters
            
        Returns:
            Generated images tensor
        """
        # Temporarily enable fast sampling if not already enabled
        original_fast_sampling = self.fast_sampling
        if not self.fast_sampling:
            self.fast_sampling = True
            # Reload model with fast sampling config
            self._load_model()
        
        try:
            result = self._generate_batch(batch_size, num_inference_steps, **kwargs)
        finally:
            # Restore original setting
            if not original_fast_sampling:
                self.fast_sampling = original_fast_sampling
                # Reload model with original config
                self._load_model()
        
        return result
    
    def generate_high_quality(self, batch_size: int, num_inference_steps: int = None,
                             **kwargs) -> torch.Tensor:
        """
        Generate high-quality images (same as fast for this model).
        
        Args:
            batch_size: Number of images to generate
            num_inference_steps: Ignored for this model
            **kwargs: Additional parameters
            
        Returns:
            Generated images tensor
        """
        return self._generate_batch(batch_size, num_inference_steps, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the NCSN++ model."""
        info = super().get_model_info()
        info.update({
            'model_type': 'Score-based SDE (NCSN++) - Original PyTorch',
            'repository': 'https://github.com/yang-song/score_sde_pytorch',
            'paper': 'https://arxiv.org/abs/2011.13456',
            'checkpoint_url': self.checkpoint_url,
            'supported_resolutions': [256],
            'default_steps': 'SDE-based (continuous)',
            'fast_steps': 'SDE-based (200 steps, 10x speedup)',
            'high_quality_steps': 'SDE-based (continuous)',
            'architecture': 'Noise Conditional Score Network++',
            'sampling_method': 'Predictor-Corrector (PC) sampling',
            'sde_type': 'VESDE',
            'predictor': 'ReverseDiffusionPredictor',
            'corrector': 'LangevinCorrector',
            'snr': 0.075 if not self.fast_sampling else 0.15,
            'n_steps': 1,
            'fast_sampling_enabled': self.fast_sampling,
            'sde_steps': 200 if self.fast_sampling else 2000
        })
        return info