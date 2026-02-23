"""
ADM-based DDPM sampler for FFHQ-256 images.

Weights sourced from https://huggingface.co/xutongda/adm_ffhq_256x256
which contains a diffusers-compatible conversion of OpenAI's ADM model.
"""

import json
import math
import os
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from .base import ModelSampler


def timestep_embedding_adm(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """ADM sinusoidal embedding variant."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=timesteps.device
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepsADM(nn.Module):
    """Time projection used by ADM (matches diffusers PR #6730)."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return timestep_embedding_adm(timesteps, self.num_channels)


def _reshape_legacy_qkv(attn, query, key, value, batch_size, head_dim):
    seq_len = query.shape[1]
    qkv = torch.cat([query, key, value], dim=2).transpose(1, 2)
    qkv = qkv.reshape(batch_size, attn.heads, head_dim * 3, seq_len)
    query, key, value = qkv.chunk(3, dim=2)
    query = query.transpose(-1, -2)
    key = key.transpose(-1, -2)
    value = value.transpose(-1, -2)
    return query, key, value


class ADMProcessor2_0:
    """Attention processor with legacy ADM ordering support."""

    def __init__(self):
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError("ADMProcessor2_0 requires PyTorch 2.0 or newer.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            pass  # diffusers handles deprecation elsewhere

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        if getattr(attn, "attention_legacy_order", False):
            query, key, value = _reshape_legacy_qkv(attn, query, key, value, batch_size, head_dim)
        else:
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class ADMFusedAttnProcessor2_0(ADMProcessor2_0):
    """Fused variant that also supports legacy ordering."""

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if encoder_hidden_states is None:
            qkv = attn.to_qkv(hidden_states)
            split_size = qkv.shape[-1] // 3
            query, key, value = torch.split(qkv, split_size, dim=-1)
        else:
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            query = attn.to_q(hidden_states)
            kv = attn.to_kv(encoder_hidden_states)
            split_size = kv.shape[-1] // 2
            key, value = torch.split(kv, split_size, dim=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        if getattr(attn, "attention_legacy_order", False):
            query, key, value = _reshape_legacy_qkv(attn, query, key, value, batch_size, head_dim)
        else:
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


@ModelSampler.register("adm-ffhq-256")
class ModelSampler_ADM_FFHQ(ModelSampler):
    """
    Sampler for the ADM (Augmented Diffusion Model) FFHQ checkpoint.

    Uses the diffusers `DDPMPipeline` interface to generate 256x256 RGB images.
    """

    def __init__(
        self,
        model_name: str,
        dataset: str,
        image_size: int,
        device: str = "cuda",
        batch_size: int = 32,
        num_inference_steps: int = 1000,
    ):
        # Force canonical parameters regardless of user input
        super().__init__(
            model_name="adm-ffhq-256",
            dataset="ffhq",
            image_size=256,
            device=device,
            batch_size=batch_size,
        )

        self.hf_model_id = "xutongda/adm_ffhq_256x256"
        self.num_inference_steps = num_inference_steps
        self._pipeline = None
        self._local_repo_path: Optional[Path] = None

    def _load_model(self) -> None:
        """Load the diffusers pipeline for ADM FFHQ."""
        try:
            from diffusers import DDPMPipeline, DDIMScheduler
        except ImportError as exc:
            raise ImportError(
                "The diffusers package is required to use the ADM FFHQ sampler. "
                "Install it with `pip install diffusers>=0.21.0`."
            ) from exc

        # Disable Hugging Face's optional Xet storage backend so that users
        # don't need the hf_xet dependency in standard environments.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        self.logger.info(
            f"Loading ADM FFHQ (256x256) pipeline from Hugging Face hub: {self.hf_model_id}"
        )
        model_root = self._prepare_local_repo()
        pipeline = DDPMPipeline.from_pretrained(model_root)
        # Swap to DDIM scheduler for potentially sharper samples.
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

        # Restore ADM-specific components (time embeddings + legacy attention).
        self._restore_adm_features(pipeline)

        # The converted ADM checkpoint outputs 6 channels (mean and variance head).
        # Reduce to RGB by taking only the mean prediction channels to satisfy the
        # DDPM/DDIM schedulers which expect 3 output channels.
        if hasattr(pipeline.unet, "conv_out") and pipeline.unet.conv_out.out_channels == 6:
            conv = pipeline.unet.conv_out
            new_conv = nn.Conv2d(
                in_channels=conv.in_channels,
                out_channels=3,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
            )
            with torch.no_grad():
                new_conv.weight.copy_(conv.weight[:3])
                new_conv.bias.copy_(conv.bias[:3])
            pipeline.unet.conv_out = new_conv
            pipeline.unet.config.out_channels = 3
            self.logger.info("Trimmed ADM conv_out to 3 channels for sampling compatibility.")
        pipeline = pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)

        self._pipeline = pipeline
        self._model = pipeline  # For consistency with base class expectations
        self.logger.info("ADM FFHQ pipeline loaded successfully")

    def _postprocess(self, images: Union[torch.Tensor, np.ndarray, Sequence]) -> torch.Tensor:
        """Clamp images to [0, 1] and ensure float32 dtype."""
        if images is None:
            raise ValueError("Pipeline returned no images")

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
            if images.shape[0] not in (1, 3) and images.shape[-1] in (1, 3):
                images = images.permute(2, 0, 1).contiguous()
        elif images.ndim == 4:
            if images.shape[1] not in (1, 3) and images.shape[-1] in (1, 3):
                images = images.permute(0, 3, 1, 2).contiguous()

        if images.ndim == 3:
            images = images.unsqueeze(0)

        images = images.to(torch.float32)

        # Some pipelines may output in [-1, 1]; rescale if needed.
        if images.min() < 0:
            images = (images + 1.0) / 2.0

        return torch.clamp(images, 0.0, 1.0)

    def _generate_batch(self, batch_size: int, **kwargs) -> torch.Tensor:
        """Generate a batch of images using the ADM diffusers pipeline."""
        if self._pipeline is None:
            raise RuntimeError("Pipeline not initialized. Call load_model() first.")

        try:
            output = self._pipeline(
                batch_size=batch_size,
                num_inference_steps=self.num_inference_steps,
                output_type="pil",
            )
            to_tensor = transforms.ToTensor()
            images = torch.stack([to_tensor(img) for img in output.images], dim=0)
        except Exception as exc:
            self.logger.error(f"ADM pipeline failed to generate images: {exc}")
            raise

        return self._postprocess(images)

    def _prepare_local_repo(self) -> Path:
        """
        Download the ADM weights locally and patch the config so it is compatible with the
        diffusers version that lacks ADM-specific time embeddings.
        """
        if self._local_repo_path and self._local_repo_path.exists():
            return self._local_repo_path

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required to download ADM checkpoints. "
                "Install it with `pip install huggingface_hub`."
            ) from exc

        snapshot_path = Path(snapshot_download(self.hf_model_id))
        self._patch_adm_config(snapshot_path)
        self._local_repo_path = snapshot_path
        return snapshot_path

    def _patch_adm_config(self, repo_path: Path) -> None:
        """Rewrite the UNet config to avoid unsupported ADM-specific settings."""
        config_path = repo_path / "config.json"
        if not config_path.exists():
            return

        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError as exc:
            self.logger.warning("Failed to parse %s: %s", config_path, exc)
            return

        changed = False
        if config.get("time_embedding_type") == "adm":
            config["_original_time_embedding_type"] = "adm"
            config["time_embedding_type"] = "positional"
            changed = True
        if "attention_legacy_order" in config:
            config["_attention_legacy_order"] = config["attention_legacy_order"]

        if changed:
            config_path.write_text(json.dumps(config, indent=2))
            self.logger.info(
                "Patched ADM UNet config at %s to use positional time embeddings for compatibility.",
                config_path,
            )

    def _restore_adm_features(self, pipeline) -> None:
        """Re-enable ADM-specific components that were patched out for compatibility."""
        unet = pipeline.unet
        config = unet.config

        original_time_type = getattr(config, "_original_time_embedding_type", None) or config.get(
            "_original_time_embedding_type"
        )
        if original_time_type == "adm":
            base_dim = config.block_out_channels[0]
            unet.time_proj = TimestepsADM(base_dim).to(self.device)
            unet.config.time_embedding_type = "adm"
            self.logger.info("Restored ADM time embedding projection.")

        legacy_flag = (
            bool(config.get("_attention_legacy_order"))
            or bool(config.get("attention_legacy_order"))
            or getattr(config, "_attention_legacy_order", False)
        )
        if legacy_flag:
            self._enable_attention_legacy_order(unet)

    def _enable_attention_legacy_order(self, unet) -> None:
        """Set all Attention modules to use legacy ADM ordering."""
        try:
            from diffusers.models.attention_processor import Attention
        except ImportError as exc:
            self.logger.warning("Could not import diffusers Attention to enable ADM legacy order: %s", exc)
            return

        processor = ADMProcessor2_0()
        fused_processor = ADMFusedAttnProcessor2_0()

        for module in unet.modules():
            if isinstance(module, Attention):
                module.attention_legacy_order = True
                if getattr(module, "to_qkv", None) is not None:
                    module.set_processor(fused_processor)
                else:
                    module.set_processor(processor)
        self.logger.info("Enabled ADM legacy attention ordering for UNet.")
