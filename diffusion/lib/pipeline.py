"""Inference pipeline for bbox- and pose-conditioned models."""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from diffusers import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils.torch_utils import randn_tensor


class ImageCompletionPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        clip_image_encoder: nn.Module,
        unet: UNet2DConditionModel,
        controlnet: ControlNetModel,
        scheduler: KarrasDiffusionSchedulers,
        refer_clip_proj: nn.Linear,
        torch_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            clip_image_encoder=clip_image_encoder,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            refer_clip_proj=refer_clip_proj,
        )
        self.weight_dtype = torch_dtype
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        if self.unet.config.in_channels != 12:
            raise ValueError(
                f"Selected U-Net must have 12 input channels, got {self.unet.config.in_channels}"
            )
        if self.controlnet.config.conditioning_channels not in (3, 6):
            raise ValueError(
                "ControlNet conditioning channels must be 3 or 6, got "
                f"{self.controlnet.config.conditioning_channels}"
            )

    def to(self, *args, **kwargs):
        """Move pipeline modules without requiring Diffusers ``ModelMixin``.

        ``clip_image_encoder`` and ``refer_clip_proj`` are ordinary
        :class:`torch.nn.Module` instances.  Some Diffusers releases assume
        every registered pipeline component has the ``.dtype`` property
        provided by ``ModelMixin`` and therefore fail before moving a plain
        ``nn.Linear`` (or the custom CLIP wrapper).  Moving the registered
        neural modules directly has the same intended effect for this
        pipeline and is compatible across those releases.
        """
        for module in (
            self.vae,
            self.clip_image_encoder,
            self.unet,
            self.controlnet,
            self.refer_clip_proj,
        ):
            module.to(*args, **kwargs)
        return self

    def _device(self) -> torch.device:
        return next(self.unet.parameters()).device

    def _decode_latents(self, latents: torch.Tensor) -> np.ndarray:
        latents = latents / self.vae.config.scaling_factor
        images = self.vae.decode(latents, return_dict=False)[0]
        return (
            images.div(2)
            .add(0.5)
            .clamp(0, 1)
            .cpu()
            .permute(0, 2, 3, 1)
            .float()
            .numpy()
        )

    def _prepare_noise_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator,
    ) -> torch.Tensor:
        shape = (
            batch_size,
            self.vae.config.latent_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"Expected {batch_size} generators, received {len(generator)}"
            )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents * self.scheduler.init_noise_sigma

    def _prepare_image_latents(
        self,
        images: torch.Tensor,
        effective_batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        use_guidance: bool,
    ) -> torch.Tensor:
        images = images.to(device=device, dtype=dtype)
        latents = self.vae.encode(images).latent_dist.mode()
        latents = latents * self.vae.config.scaling_factor
        if effective_batch_size % latents.shape[0] != 0:
            raise ValueError(
                "Effective inference batch is not divisible by the source batch: "
                f"{effective_batch_size} vs {latents.shape[0]}"
            )
        repeats = effective_batch_size // latents.shape[0]
        latents = latents.repeat_interleave(repeats, dim=0)
        if use_guidance:
            latents = torch.cat((torch.zeros_like(latents), latents), dim=0)
        return latents.to(device=device, dtype=dtype)

    def _encode_reference(
        self,
        images: torch.Tensor,
        num_images_per_prompt: int,
        device: torch.device,
        use_guidance: bool,
    ) -> torch.Tensor:
        encoder_dtype = next(self.clip_image_encoder.parameters()).dtype
        images = images.to(device=device, dtype=encoder_dtype)
        embeddings = self.clip_image_encoder.encode_image(images).float().unsqueeze(1)
        embeddings = self.refer_clip_proj(embeddings)
        embeddings = embeddings.repeat_interleave(num_images_per_prompt, dim=0)
        if use_guidance:
            embeddings = torch.cat((torch.zeros_like(embeddings), embeddings), dim=0)
        return embeddings.to(dtype=self.weight_dtype)

    @torch.no_grad()
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        generator=None,
        num_inference_steps: int = 50,
        guidance_scale: float = 2.0,
        num_images_per_prompt: int = 1,
        controlnet_conditioning_scale: float = 1.0,
    ) -> np.ndarray:
        device = self._device()
        use_guidance = guidance_scale > 1.0

        source_images = inputs["occ_src_img"].to(
            device=device, dtype=self.weight_dtype
        )
        source_masks = inputs["occ_src_mask"].to(
            device=device, dtype=self.weight_dtype
        )
        controls = inputs["occ_src_control"].to(
            device=device, dtype=self.weight_dtype
        )
        expected_channels = self.controlnet.config.conditioning_channels
        if controls.shape[1] != expected_channels:
            raise ValueError(
                f"ControlNet expects {expected_channels} control channels, got "
                f"{controls.shape}"
            )

        batch_size, _, height, width = controls.shape
        effective_batch_size = batch_size * num_images_per_prompt
        reference_embeddings = self._encode_reference(
            inputs["occ_src_clip_img"],
            num_images_per_prompt,
            device,
            use_guidance,
        )

        controls = controls.repeat_interleave(num_images_per_prompt, dim=0)
        if use_guidance:
            controls = torch.cat((controls, controls), dim=0)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        latents = self._prepare_noise_latents(
            effective_batch_size,
            height,
            width,
            reference_embeddings.dtype,
            device,
            generator,
        )
        source_latents = self._prepare_image_latents(
            source_images,
            effective_batch_size,
            reference_embeddings.dtype,
            device,
            use_guidance,
        )
        mask_latents = self._prepare_image_latents(
            source_masks.repeat(1, 3, 1, 1),
            effective_batch_size,
            reference_embeddings.dtype,
            device,
            use_guidance,
        )

        warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for index, timestep in enumerate(timesteps):
                latent_input = (
                    torch.cat((latents, latents), dim=0) if use_guidance else latents
                )
                latent_input = self.scheduler.scale_model_input(
                    latent_input, timestep
                )
                latent_input = torch.cat(
                    (latent_input, source_latents, mask_latents), dim=1
                )

                down_residuals, mid_residual = self.controlnet(
                    latent_input,
                    timestep,
                    encoder_hidden_states=reference_embeddings,
                    controlnet_cond=controls,
                    conditioning_scale=controlnet_conditioning_scale,
                    return_dict=False,
                )
                noise_prediction = self.unet(
                    latent_input,
                    timestep,
                    encoder_hidden_states=reference_embeddings,
                    down_block_additional_residuals=down_residuals,
                    mid_block_additional_residual=mid_residual,
                    return_dict=False,
                )[0]

                if use_guidance:
                    unconditioned, conditioned = noise_prediction.chunk(2)
                    noise_prediction = unconditioned + guidance_scale * (
                        conditioned - unconditioned
                    )

                latents = self.scheduler.step(
                    noise_prediction, timestep, latents, return_dict=False
                )[0]
                if index == len(timesteps) - 1 or (
                    index + 1 > warmup_steps
                    and (index + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        return self._decode_latents(latents)
