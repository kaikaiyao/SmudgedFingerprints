"""
NVAE sampler for FFHQ-256.

Loads the official NVAE checkpoint released by NVIDIA and exposes a sampling
interface consistent with the rest of the fingerprint robustness suite.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional
import types

import torch
from torch.cuda.amp import autocast

from .base import ModelSampler


@ModelSampler.register("nvae-ffhq-256")
class ModelSampler_NVAE_FFHQ(ModelSampler):
    """
    Sampler for the NVAE FFHQ-256 checkpoint.

    This wrapper vendors the official NVAE repository, downloads the published
    FFHQ checkpoint from Google Drive, and provides a simple `sample` method that
    matches the rest of the codebase.
    """

    def __init__(
        self,
        model_name: str,
        dataset: str,
        image_size: int,
        device: str = "cuda",
        batch_size: int = 32,
        temperature: float = 0.7,
        bn_update_steps: int = 0,
        bn_update_batch_size: int = 16,
    ):
        if device != "cuda":
            raise ValueError(
                "NVAE FFHQ currently supports only CUDA devices. "
                "Please run with --device_preference=cuda."
            )

        super().__init__(
            model_name="nvae-ffhq-256",
            dataset="ffhq",
            image_size=256,
            device=device,
            batch_size=batch_size,
        )

        self.repo_url = "https://github.com/NVlabs/NVAE.git"
        self.repo_dir = Path("libs") / "nvae"
        self.checkpoint_id = "1lQzywY5O71Z5NqAUJUcWy2Q1K2hPFO6j"
        self.temperature = temperature
        self.bn_update_steps = bn_update_steps
        self.bn_update_batch_size = bn_update_batch_size

        self._model = None
        self._nvae_utils = None
        self._nvae_args = None

    # --------------------------------------------------------------------- #
    # Repository and dependency management
    # --------------------------------------------------------------------- #

    def _ensure_dependency(self, package: str) -> None:
        """Install a Python dependency if it is missing."""
        try:
            __import__(package)
        except ImportError:
            self.logger.info("Installing missing dependency: %s", package)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                check=True,
                capture_output=True,
            )

    def _setup_repository(self) -> Path:
        """Clone the NVAE repository locally if needed."""
        libs_dir = Path("libs")
        libs_dir.mkdir(exist_ok=True)

        if not self.repo_dir.exists():
            self.logger.info("Cloning NVAE repository to %s", self.repo_dir)
            subprocess.run(
                ["git", "clone", "--depth", "1", self.repo_url, str(self.repo_dir)],
                check=True,
                capture_output=True,
            )

        # Ensure repository is on sys.path for imports
        repo_path = str(self.repo_dir.resolve())
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        return self.repo_dir

    def _download_checkpoint(self, repo_dir: Path) -> Path:
        """Download the FFHQ NVAE checkpoint using gdown."""
        checkpoints_dir = repo_dir / "checkpoints" / "ffhq"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoints_dir / "checkpoint.pt"

        if checkpoint_path.exists():
            return checkpoint_path

        self._ensure_dependency("gdown")
        import gdown

        self.logger.info("Downloading NVAE FFHQ checkpoint (~1.8GB)...")
        gdown.download(
            id=self.checkpoint_id,
            output=str(checkpoint_path),
            quiet=False,
        )
        return checkpoint_path

    # --------------------------------------------------------------------- #
    # Model loading and preparation
    # --------------------------------------------------------------------- #

    def _load_model(self) -> None:
        """Load the NVAE model and prepare it for sampling."""
        if not torch.cuda.is_available():
            raise RuntimeError("NVAE FFHQ sampler requires CUDA for sampling.")

        self._ensure_dependency("ml_collections")

        repo_dir = self._setup_repository()
        checkpoint_path = self._download_checkpoint(repo_dir)

        # NVAE depends on tensorboardX for logging utilities even at import time.
        self._shim_tensorboardx()

        try:
            from model import AutoEncoder
            import utils as nvae_utils
        except Exception as exc:
            raise ImportError(
                f"Failed to import NVAE modules. Ensure the NVAE repository and its dependencies are available. "
                f"Original error: {exc}"
            ) from exc

        # Torch 2.6 defaults to weights_only=True which breaks older checkpoints.
        # These official NVAE weights are trusted, so load with weights_only=False.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        args = checkpoint["args"]
        # Ensure missing attributes from older checkpoints are populated.
        if not hasattr(args, "ada_groups"):
            args.ada_groups = False
        if not hasattr(args, "min_groups_per_scale"):
            args.min_groups_per_scale = 1
        if not hasattr(args, "num_mixture_dec"):
            args.num_mixture_dec = 10
        arch_instance = nvae_utils.get_arch_cells(args.arch_instance)

        self.logger.info("Creating NVAE model...")
        model = AutoEncoder(args, None, arch_instance)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model = model.to(self.device)
        model.eval()

        self._model = model
        self._nvae_utils = nvae_utils
        self._nvae_args = args

        if self.bn_update_steps > 0:
            self._readjust_batchnorm()

        self.logger.info("NVAE FFHQ model loaded successfully")

    def _readjust_batchnorm(self) -> None:
        """Optionally refresh running statistics for NVAE's BatchNorm layers."""
        if self._model is None or self.bn_update_steps <= 0:
            return

        self.logger.info(
            "Re-adjusting NVAE batchnorm stats (%d iterations)...",
            self.bn_update_steps,
        )

        self._model.train()
        batch_size = min(self.bn_update_batch_size, self.batch_size)

        for step in range(self.bn_update_steps):
            with autocast(enabled=torch.cuda.is_available()):
                with torch.no_grad():
                    _ = self._model.sample(batch_size, self.temperature)

            if (step + 1) % max(1, self.bn_update_steps // 5) == 0:
                self.logger.debug(
                    "  BN refresh progress: %d/%d",
                    step + 1,
                    self.bn_update_steps,
                )

        self._model.eval()

    # --------------------------------------------------------------------- #
    # Sampling utilities
    # --------------------------------------------------------------------- #

    def _postprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Convert NVAE outputs to [0, 1] float tensors."""
        if images.min() < 0:
            images = (images + 1.0) / 2.0
        return torch.clamp(images, 0.0, 1.0)

    def _generate_batch(self, batch_size: int, **kwargs) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("Model not initialized. Call load_model() first.")

        with torch.no_grad():
            logits = self._model.sample(batch_size, self.temperature)
            decoder = self._model.decoder_output(logits)
            if hasattr(decoder, "sample"):
                images = decoder.sample(self.temperature)
            elif hasattr(decoder, "mean"):
                images = decoder.mean
            else:
                raise RuntimeError("Unexpected NVAE decoder output.")

        return self._postprocess(images.to(torch.float32))

    # ------------------------------------------------------------------ #
    # Shims for optional deps
    # ------------------------------------------------------------------ #

    def _shim_tensorboardx(self) -> None:
        """Provide a lightweight tensorboardX shim for inference-only usage."""
        if "tensorboardX" in sys.modules:
            return

        tbx = types.ModuleType("tensorboardX")

        class _DummyWriter:
            def __init__(self, *args, **kwargs):
                pass

            def add_scalar(self, *args, **kwargs):
                return None

            def add_histogram(self, *args, **kwargs):
                return None

            def close(self):
                return None

        tbx.SummaryWriter = _DummyWriter
        sys.modules["tensorboardX"] = tbx
