"""
Latent Diffusion samplers for FFHQ using Hugging Face-hosted checkpoints.
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch

from .base import ModelSampler


class BaseHuggingFaceLDMSampler(ModelSampler):
    """
    Shared utilities for Hugging Face-hosted latent diffusion checkpoints that
    expose UNet2DModel + AutoencoderKL components.
    """

    def __init__(
        self,
        *,
        model_name: str,
        hf_repo_id: str,
        scheduler_class: str,
        image_size: int = 256,
        device: str = "cuda",
        batch_size: int = 32,
        num_inference_steps: int = 200,
        eta: float = 0.0,
    ):
        super().__init__(
            model_name=model_name,
            dataset="ffhq",
            image_size=image_size,
            device=device,
            batch_size=batch_size,
        )
        self.hf_repo_id = hf_repo_id
        self.scheduler_class_name = scheduler_class
        self.num_inference_steps = num_inference_steps
        self.eta = eta

        self.repo_root = Path("models") / "ldm" / model_name
        self._unet = None
        self._vae = None
        self._scheduler = None
        self._latent_scaling = 1.0
        self._scheduler_accepts_eta = False

    def _ensure_dependency(self, package: str) -> None:
        """Install a Python dependency if it is missing."""
        try:
            __import__(package)
            return
        except ImportError:
            pass

        self.logger.info("Installing missing dependency: %s", package)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            check=True,
            capture_output=True,
        )

    def _prepare_repo(self, snapshot_download) -> Path:
        """Ensure the Hugging Face weights are downloaded locally."""
        repo_dir = self.repo_root
        repo_dir.mkdir(parents=True, exist_ok=True)

        expected_subdirs = ["unet", "vae", "scheduler"]
        if all((repo_dir / subdir).exists() for subdir in expected_subdirs):
            return repo_dir

        self.logger.info(
            "Downloading %s from Hugging Face hub to %s", self.hf_repo_id, repo_dir
        )
        snapshot_download(
            repo_id=self.hf_repo_id,
            local_dir=str(repo_dir),
            local_dir_use_symlinks=False,
            repo_type="model",
            resume_download=True,
        )
        return repo_dir

    def _load_model(self) -> None:
        """Load UNet, VAE, and scheduler components from Hugging Face."""
        for pkg in ("diffusers", "huggingface_hub"):
            self._ensure_dependency(pkg)

        from diffusers import AutoencoderKL, UNet2DModel, DDIMScheduler, DDPMScheduler
        from huggingface_hub import snapshot_download

        scheduler_map = {
            "DDIMScheduler": DDIMScheduler,
            "DDPMScheduler": DDPMScheduler,
        }
        scheduler_cls = scheduler_map.get(self.scheduler_class_name)
        if scheduler_cls is None:
            raise ValueError(
                f"Unsupported scheduler '{self.scheduler_class_name}' for {self.model_name}"
            )

        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        repo_dir = self._prepare_repo(snapshot_download)

        try:
            vae = AutoencoderKL.from_pretrained(str(repo_dir / "vae")).to(self.device)
            unet = UNet2DModel.from_pretrained(str(repo_dir / "unet")).to(self.device)
            scheduler = scheduler_cls.from_pretrained(str(repo_dir / "scheduler"))
        except Exception as exc:
            raise RuntimeError(f"Failed to load Hugging Face LDM assets: {exc}") from exc

        vae.eval()
        unet.eval()
        self._vae = vae
        self._unet = unet
        self._scheduler = scheduler
        self._latent_scaling = getattr(vae.config, "scaling_factor", 1.0) or 1.0
        self._scheduler_accepts_eta = "eta" in inspect.signature(scheduler.step).parameters
        self._model = unet
        self.logger.info(
            "Loaded Hugging Face LDM components for %s from %s", self.model_name, repo_dir
        )

    def _postprocess(self, images: Union[torch.Tensor, np.ndarray, Sequence]) -> torch.Tensor:
        """Clamp images to [0, 1] and ensure float32 dtype."""
        if images is None:
            raise ValueError("Sampler returned no images")

        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        elif isinstance(images, (list, tuple)):
            images = [
                torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr for arr in images
            ]
            images = torch.stack(images, dim=0) if images else torch.empty(0)

        if not torch.is_tensor(images):
            raise TypeError(f"Unsupported image container type: {type(images)}")

        if images.ndim == 3:
            images = images.unsqueeze(0)

        images = images.to(torch.float32)
        return torch.clamp(images, 0.0, 1.0)

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to RGB images using the VAE."""
        if self._vae is None:
            raise RuntimeError("VAE not initialized")

        latents = latents / self._latent_scaling
        with torch.no_grad():
            images = self._vae.decode(latents).sample
        images = (images + 1.0) / 2.0
        return self._postprocess(images)

    def _generate_batch(self, batch_size: int, **kwargs) -> torch.Tensor:
        """Generate a batch of images via DDIM/DDPM sampling."""
        if self._unet is None or self._scheduler is None:
            raise RuntimeError("Model not initialized. Call load_model() first.")

        scheduler = self._scheduler
        unet = self._unet

        scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        latent_size = getattr(unet.config, "sample_size", 32)
        latent_channels = getattr(unet.config, "in_channels", 4)
        latents = torch.randn(
            (batch_size, latent_channels, latent_size, latent_size),
            device=self.device,
            dtype=unet.dtype,
        )
        if hasattr(scheduler, "init_noise_sigma"):
            latents = latents * scheduler.init_noise_sigma

        eta_kwargs = {"eta": self.eta} if self._scheduler_accepts_eta else {}

        with torch.no_grad():
            for t in scheduler.timesteps:
                latent_model_input = latents
                if hasattr(scheduler, "scale_model_input"):
                    latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                noise_pred = unet(latent_model_input, t).sample
                step_output = scheduler.step(noise_pred, t, latents, **eta_kwargs)
                latents = step_output.prev_sample

        return self._decode_latents(latents)


@ModelSampler.register("ldm-ffhq-256")
class ModelSampler_LDM_FFHQ(BaseHuggingFaceLDMSampler):
    """FFHQ LDM checkpoint from https://huggingface.co/kaayaanil/ldm-ffhq-256."""

    def __init__(
        self,
        model_name: str,
        dataset: str,
        image_size: int,
        device: str = "cuda",
        batch_size: int = 32,
        num_inference_steps: int = 200,
        eta: float = 0.0,
    ):
        super().__init__(
            model_name="ldm-ffhq-256",
            hf_repo_id="kaayaanil/ldm-ffhq-256",
            scheduler_class="DDPMScheduler",
            image_size=256,
            device=device,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            eta=eta,
        )
