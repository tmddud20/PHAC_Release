"""Validation helpers for the selected image-completion model."""

import numpy as np
import torch
import wandb
from diffusers import UniPCMultistepScheduler

from .pipeline import ImageCompletionPipeline


CONTROL_CAPTIONS = {
    "ib": ("interested bbox",),
    "eb": ("entire bbox",),
    "po": ("pose",),
    "poib": ("pose", "interested bbox"),
    "poeb": ("pose", "entire bbox"),
}


def tensor_to_np(image: torch.Tensor) -> np.ndarray:
    image = (image / 2 + 0.5).clamp(0, 1)
    return image.detach().cpu().permute(0, 2, 3, 1).float().numpy()


def log_validation(
    logger,
    val_dataloader,
    vae,
    clip_image_encoder,
    unet,
    controlnet,
    scheduler,
    refer_clip_proj,
    args,
    accelerator,
    weight_dtype,
    step,
):
    logger.info("Running validation")
    validation_unet = accelerator.unwrap_model(unet)
    validation_controlnet = accelerator.unwrap_model(controlnet)
    validation_projection = accelerator.unwrap_model(refer_clip_proj)
    training_modes = {
        validation_unet: validation_unet.training,
        validation_controlnet: validation_controlnet.training,
        validation_projection: validation_projection.training,
    }
    for model in training_modes:
        model.eval()

    try:
        pipeline = ImageCompletionPipeline(
            vae=vae,
            clip_image_encoder=clip_image_encoder,
            unet=validation_unet,
            controlnet=validation_controlnet,
            scheduler=scheduler,
            refer_clip_proj=validation_projection,
            torch_dtype=weight_dtype,
        )
        pipeline.scheduler = UniPCMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)
        if args.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()

        generator = None
        if args.seed is not None:
            generator = torch.Generator(device=accelerator.device).manual_seed(
                args.seed
            )

        image_logs = []
        for index, data in enumerate(val_dataloader):
            if index >= args.num_validation_images:
                break
            with torch.autocast(
                device_type=accelerator.device.type,
                enabled=accelerator.device.type == "cuda",
            ):
                generated = pipeline.forward(
                    data,
                    num_inference_steps=50,
                    generator=generator,
                    guidance_scale=args.guidance_scale,
                    controlnet_conditioning_scale=args.conditioning_scale,
                    num_images_per_prompt=args.num_gen_images,
                )
            controls = data["occ_src_control"]
            control_images = [
                tensor_to_np(controls[:, offset : offset + 3])
                for offset in range(0, controls.shape[1], 3)
            ]
            image_logs.append(
                {
                    "source": tensor_to_np(data["occ_src_img"]),
                    "controls": control_images,
                    "target": tensor_to_np(data["deocc_tgt_img"]),
                    "generated": generated,
                }
            )

        for tracker in accelerator.trackers:
            if tracker.name != "wandb":
                logger.warning(
                    "Image logging is only implemented for W&B; skipping %s",
                    tracker.name,
                )
                continue
            formatted = []
            for log in image_logs:
                formatted.extend(
                    [
                        wandb.Image(log["target"][0], caption="ground truth"),
                        wandb.Image(log["source"][0], caption="occluded source"),
                    ]
                )
                formatted.extend(
                    wandb.Image(control[0], caption=caption)
                    for control, caption in zip(
                        log["controls"], CONTROL_CAPTIONS[args.control_type]
                    )
                )
                formatted.extend(
                    wandb.Image(image, caption=f"generated {index}")
                    for index, image in enumerate(log["generated"])
                )
            tracker.log({"validation": formatted}, step=step)
        return image_logs
    finally:
        for model, was_training in training_modes.items():
            model.train(was_training)
