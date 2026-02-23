"""
VQ-VAE sampler for FFHQ-256 images.

Integrates the publicly released VQ-VAE checkpoint from kohido/ffhq256_vqvae_mhvq
and reconstructs FFHQ images drawn from a local real-data cache.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
import types

from .base import ModelSampler
from ..fingerprints.song24 import download_ffhq_data


@ModelSampler.register("vqvae-ffhq-256")
class ModelSampler_VQVAE_FFHQ(ModelSampler):
    """
    Sampler for the FFHQ-256 VQ-VAE checkpoint published by Khoi Do (kohido).

    The sampler clones the vitvqganvae repository that defines the model, downloads
    the corresponding Hugging Face checkpoint, and decodes random latent codebook
    samples to produce face images.
    """

    def __init__(
        self,
        model_name: str,
        dataset: str,
        image_size: int,
        device: str = "cuda",
        batch_size: int = 32,
        latent_temperature: float = 1.0,
    ):
        super().__init__(
            model_name="vqvae-ffhq-256",
            dataset="ffhq",
            image_size=256,
            device=device,
            batch_size=batch_size,
        )

        self.repo_url = "https://github.com/KhoiDOO/vitvqganvae.git"
        self.repo_dir = Path("libs") / "vitvqganvae"
        self.hf_model_id = "kohido/ffhq256_vqvae_mhvq"
        self.latent_temperature = latent_temperature

        self._vqvae = None
        self._latent_tokens = None
        self._codebook_size = None
        self._num_heads = 1
        self._real_images = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _locate_ffhq_pickle(self) -> Optional[Path]:
        """
        Locate a local FFHQ pickle (ffhq_real_data.pkl) without downloading.

        We probe common data roots used by the pipeline:
        - $DATA_DIR/ffhq_dataset
        - /pvc-output/data[/ffhq_dataset]
        - /pvc-output/data_attack[/ffhq_dataset]
        - ./data/ffhq_dataset
        """
        candidates = []
        data_dir_env = os.environ.get("DATA_DIR")
        if data_dir_env:
            candidates.append(Path(data_dir_env))
        candidates.extend(
            [
                Path("/pvc-output/data"),
                Path("/pvc-output/data_attack"),
                Path("data"),
            ]
        )

        for base in candidates:
            pickle_path = base / "ffhq_dataset" / "ffhq_real_data.pkl"
            if pickle_path.exists():
                return pickle_path.resolve()
        return None

    def _download_ffhq_pickle(self) -> Optional[Path]:
        """Download FFHQ real data into the configured DATA_DIR."""
        data_dir = os.environ.get("DATA_DIR")
        if data_dir:
            target_dir = Path(data_dir)
        else:
            target_dir = Path("/pvc-output/data")
        download_root = target_dir / "ffhq_dataset"
        download_root.mkdir(parents=True, exist_ok=True)

        try:
            pickle_path = Path(download_ffhq_data(output_dir=str(download_root), num_images=1000))
            self.logger.info("Downloaded FFHQ real data to %s", pickle_path)
            return pickle_path
        except Exception as exc:
            self.logger.error("Failed to auto-download FFHQ data to %s: %s", download_root, exc)
            return None

    def _load_real_images(self) -> None:
        """Load cached FFHQ real images if available."""
        if self._real_images is not None:
            return

        pickle_path = self._locate_ffhq_pickle()
        if pickle_path is None:
            self.logger.warning(
                "FFHQ real data not found locally; attempting automatic download..."
            )
            pickle_path = self._download_ffhq_pickle()

        if pickle_path is None or not pickle_path.exists():
            raise FileNotFoundError(
                "FFHQ real data not found and automatic download failed. Expected "
                "ffhq_dataset/ffhq_real_data.pkl under DATA_DIR, /pvc-output/data, "
                "/pvc-output/data_attack, or ./data."
            )

        try:
            data = pickle.loads(pickle_path.read_bytes())
            images = data.get("images")
            if images is None:
                raise ValueError("Missing 'images' key in ffhq_real_data.pkl")
            images = torch.from_numpy(images).float()
            if images.ndim != 4 or images.shape[1] != 3:
                raise ValueError(f"Unexpected image tensor shape: {images.shape}")
            self._real_images = images
            self.logger.info("Loaded FFHQ real images from %s (N=%d)", pickle_path, images.shape[0])
        except Exception as exc:
            raise RuntimeError(f"Failed to load FFHQ real data from {pickle_path}: {exc}") from exc

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
        """Clone the vitvqganvae repository locally."""
        libs_dir = Path("libs")
        libs_dir.mkdir(exist_ok=True)

        if not self.repo_dir.exists():
            self.logger.info("Cloning vitvqganvae repository to %s", self.repo_dir)
            subprocess.run(
                ["git", "clone", "--depth", "1", self.repo_url, str(self.repo_dir)],
                check=True,
                capture_output=True,
            )

        repo_path = str(self.repo_dir.resolve())
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        return self.repo_dir

    def _download_checkpoint(self) -> Path:
        """Download the Hugging Face snapshot for the FFHQ VQ-VAE."""
        snapshot_path = Path(
            snapshot_download(
                repo_id=self.hf_model_id,
                allow_patterns=["config.json", "model.safetensors"],
            )
        )
        return snapshot_path

    # ------------------------------------------------------------------ #
    # Loading logic
    # ------------------------------------------------------------------ #

    def _load_model(self) -> None:
        """Load the VQ-VAE weights and prepare for sampling."""
        self._ensure_dependency("einops")
        self._ensure_dependency("vector_quantize_pytorch")
        self._ensure_dependency("beartype")
        self._ensure_dependency("safetensors")

        repo_dir = self._setup_repository()
        weights_dir = self._download_checkpoint()
        self._shim_pytorch_custom_utils()
        self._ensure_dependency("x_transformers")
        self._ensure_dependency("ema_pytorch")

        try:
            from vitvqganvae.model.cnn.vqvae import VQVAE
        except Exception as exc:
            raise ImportError(
                "Failed to import vitvqganvae VQVAE. Ensure repository cloning succeeded. "
                f"Original error: {exc}"
            ) from exc

        # The real decorator attaches total_parameters to the class; add a safe default if missing.
        if not hasattr(VQVAE, "total_parameters"):
            VQVAE.total_parameters = 0

        config_path = weights_dir / "config.json"
        state_path = weights_dir / "model.safetensors"

        if not config_path.exists() or not state_path.exists():
            raise FileNotFoundError(
                f"Missing config or weights in snapshot directory: {weights_dir}"
            )

        config = json.loads(config_path.read_text())

        self.logger.info("Initializing VQ-VAE with config: %s", config_path)
        model = VQVAE(**config)
        # Ensure attribute exists and populate it for downstream logging/debugging.
        if not hasattr(model, "total_parameters") or model.total_parameters == 0:
            self.logger.warning("VQ-VAE implementation missing total_parameters attribute; computing manually.")
            total_params = sum(p.numel() for p in model.parameters())
            model.total_parameters = total_params
            self.logger.info("VQ-VAE parameter count (approx): %d", total_params)
        state_dict = load_file(state_path)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()

        latent_hw = model.encoder.fmap_size(self.image_size)
        self._vqvae = model
        self._model = model
        self._latent_tokens = latent_hw * latent_hw
        self._codebook_size = config.get("codebook_size", 1024)
        self._num_heads = getattr(model._quantizer, "heads", 1)

        self.logger.info(
            "Loaded VQ-VAE (latent resolution %dx%d, codebook size %d)",
            latent_hw,
            latent_hw,
            self._codebook_size,
        )

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #

    def _decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Decode token IDs into RGB images."""
        if self._vqvae is None:
            raise RuntimeError("Model not initialized. Call load_model() first.")

        q = self._vqvae._quantizer  # type: ignore[attr-defined]
        fmap = q.get_output_from_indices(token_ids)

        # VectorQuantize with multiple heads returns (B, HW, H, C_per_head); flatten heads.
        if fmap.ndim == 4:
            b, hw, heads, c_per_head = fmap.shape
            fmap = fmap.view(b, hw, heads * c_per_head)

        expected_c = self._vqvae.encoder.encoded_dim  # type: ignore[attr-defined]
        if fmap.shape[-1] != expected_c:
            if hasattr(q, "project_out") and q.project_out is not None:
                b, hw, c = fmap.shape[0], fmap.shape[1], fmap.shape[-1]
                fmap_flat = fmap.view(b * hw, c)
                fmap_proj = q.project_out(fmap_flat)
                fmap = fmap_proj.view(b, hw, expected_c)
                self.logger.warning(
                    "Projected VQ-VAE latent from %d to %d channels for decoder compatibility.",
                    c,
                    expected_c,
                )
            else:
                self.logger.warning(
                    "VQ-VAE latent channels (%d) do not match decoder expected (%d); decoding may fail.",
                    fmap.shape[-1],
                    expected_c,
                )

        h = w = int(fmap.shape[1] ** 0.5)
        fmap = fmap.view(fmap.shape[0], h, w, fmap.shape[-1]).permute(0, 3, 1, 2).contiguous()

        images = self._vqvae.decoder(fmap)  # type: ignore[attr-defined]
        images = images.to(torch.float32)
        if images.min() < -0.1 or images.max() > 1.1:
            images = torch.tanh(images)
            images = (images + 1.0) / 2.0
        return torch.clamp(images, 0.0, 1.0)

    def _generate_batch(self, batch_size: int, **kwargs) -> torch.Tensor:
        if self._vqvae is None:
            raise RuntimeError("Model not initialized. Call load_model() first.")

        # Use real FFHQ images as inputs; do not download if already present.
        self._load_real_images()
        if self._real_images is None or len(self._real_images) == 0:
            raise RuntimeError("FFHQ real image cache is empty; cannot generate batch.")

        # Randomly sample real images and reconstruct them through the VQ-VAE.
        with torch.no_grad():
            idx = torch.randint(0, self._real_images.shape[0], (batch_size,))
            images_in = self._real_images[idx].to(self.device)
            fmap, _, _ = self._vqvae.encode(images_in)  # type: ignore[attr-defined]
            images = self._vqvae.decode(fmap)  # type: ignore[attr-defined]

        images = images.to(torch.float32)
        if images.min() < -0.1 or images.max() > 1.1:
            images = torch.tanh(images)
            images = (images + 1.0) / 2.0

        return images

    # ------------------------------------------------------------------ #
    # Shims for optional deps
    # ------------------------------------------------------------------ #

    def _shim_pytorch_custom_utils(self) -> None:
        """Provide minimal stand-ins for pytorch_custom_utils used by vitvqganvae."""
        if "pytorch_custom_utils" in sys.modules:
            return

        module = types.ModuleType("pytorch_custom_utils")

        def total_parameters(fn=None, *args, **kwargs):
            if callable(fn):
                return fn

            def decorator(inner):
                return inner

            return decorator

        def save_load(*args, **kwargs):
            def decorator(inner):
                return inner
            return decorator

        def add_wandb_tracker_contextmanager(*args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

        module.total_parameters = total_parameters
        module.add_wandb_tracker_contextmanager = add_wandb_tracker_contextmanager
        module.save_load = save_load
        sys.modules["pytorch_custom_utils"] = module
