"""
1) Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
2) KWU CVLAB, SY Noh
"""
import gc
import logging
import math
import os
import random
import re
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint


from datetime import datetime, timedelta
from packaging import version
from tqdm.auto import tqdm

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, InitProcessGroupKwargs

import diffusers
from diffusers import ControlNetModel, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

import transformers
from lib.config import parse_options, argparse_to_str
from lib.test_dataset import TestDataset
from lib.train_dataset import TrainDataset
from lib.pix2gestalt import (
    build_image_encoder,
    build_projection,
    build_unet,
    build_vae,
    load_state_dict as load_pix2gestalt_state_dict,
)
from lib.utils import log_validation

check_min_version("0.24.0")

logger = get_logger(__name__)

def main(args, args_str):


    project_dir =  os.path.join(
            args.output_dir,
            args.exp_name
        )

    accelerator_project_config = ProjectConfiguration(
        project_dir=project_dir,
        logging_dir=project_dir,
    )
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=18000))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs]
    )

    # Every rank must use exactly the same run directory. Computing this before
    # the process group is initialized can split a distributed checkpoint when
    # ranks cross a timestamp boundary.
    run_id = [
        datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        if accelerator.is_main_process
        else None
    ]
    if accelerator.num_processes > 1:
        if not dist.is_initialized():
            raise RuntimeError("Distributed process group was not initialized")
        dist.broadcast_object_list(run_id, src=0)
    logging_dir = os.path.join(project_dir, run_id[0])
    accelerator.project_configuration.logging_dir = logging_dir

    ##################  Prepare logger and set verbosity ##################

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(f"Info: \n{args_str}")

    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)


    # Handle the repository creation
    if accelerator.is_main_process:
        if logging_dir is not None:
            os.makedirs(logging_dir, exist_ok=True)

    ##################  Dataset preparation ##################

    train_dataset = TrainDataset(args)

    train_dataloader =  torch.utils.data.DataLoader(dataset=train_dataset,
                        batch_size=args.train_batch_size,
                        shuffle=True,
                        num_workers=args.dataloader_num_workers,
                        pin_memory=True)
    if args.validation:
        val_dataset = TestDataset(args)
        val_dataloader =  torch.utils.data.DataLoader(dataset=val_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True)

    ##################  Model preparation ##################

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        steps_offset=1,
        prediction_type="epsilon",
    )

    sd = load_pix2gestalt_state_dict(args.foundation_ckpt)
    vae = build_vae(sd)
    unet = build_unet(sd)
    if unet.config.in_channels != 12:
        raise ValueError(
            f"Selected U-Net must have 12 input channels, got {unet.config.in_channels}"
        )
    clip_image_encoder = build_image_encoder(sd)
    refer_clip_proj = build_projection(sd)
    del sd
    gc.collect()

    logger.info("Initializing controlnet weights from unet")
    logger.info(
        "Control type: %s (%d channels)",
        args.control_type,
        args.conditioning_channels,
    )

    controlnet = ControlNetModel.from_unet(
        unet,
        conditioning_channels=args.conditioning_channels,
    )

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            i = len(weights) - 1
            while len(weights) > 0:
                weights.pop()
                model = models[i]
                if accelerator.is_main_process:
                    if isinstance(model, UNet2DConditionModel):
                        model.save_pretrained(os.path.join(output_dir, "unet"))
                    elif isinstance(model, ControlNetModel):
                        model.save_pretrained(os.path.join(output_dir, "controlnet"))
                    else:
                        name = getattr(model, "__class__", type(model)).__name__
                        path = os.path.join(output_dir, f"{name}.pth")
                        torch.save(
                            model.state_dict() if hasattr(model, "state_dict") else model,
                            path,
                        )

                i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()
                if isinstance(model, UNet2DConditionModel):
                    load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    del load_model
                elif isinstance(model, ControlNetModel):
                    load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    del load_model
                else:
                    name = getattr(model, "__class__", type(model)).__name__
                    path = os.path.join(input_dir, f"{name}.pth")
                    state_dict = torch.load(path, map_location="cpu")
                    model.load_state_dict(state_dict, strict=True)

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)


    ## Freeze the weights of the vae, unet and text_encoder
    ## Only train the controlnet and the transformer blocks in the unet

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    clip_image_encoder.requires_grad_(False)

    controlnet.train()
    refer_clip_proj.train()

    if args.unet_all:
        print('**U-Net whole fine-tuning**')
        unet.requires_grad_(True)
    else:
        print('**only CA blocks fine-tuning**')
        for param_name, param in unet.named_parameters():
            if 'transformer_blocks' not in param_name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")


    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
        unet.enable_gradient_checkpointing()


    ##################  Optimizer creation ##################
    scale = 1.0
    if args.scale_lr:
        scale =  args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes


    params_to_optimize = []
    optimizer_class = torch.optim.AdamW

    params_to_optimize.append({'params': controlnet.parameters(),
                                'lr': args.lr_controlnet * scale,
                                })
    params_to_optimize.append({'params': refer_clip_proj.parameters(),
                                'lr': args.lr * scale,
                                'weight_decay': args.adam_weight_decay})
    params_to_optimize.append({'params': unet.parameters(),
                                'lr': args.lr * scale,
                                'weight_decay': args.adam_weight_decay})

    optimizer = optimizer_class(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )

    ##################  Scheduler creation ##################

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )


    ##################  Training preparation ##################

    # Prepare everything with our `accelerator`.
    unet, refer_clip_proj, controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, refer_clip_proj, controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    clip_image_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


    ##################  Trackers preparation ##################


    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        init_kwargs = {"wandb": {'name': os.path.basename(logging_dir), 'dir': logging_dir}}
        accelerator.init_trackers(args.exp_name, config=tracker_config, init_kwargs=init_kwargs)

        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                tracker.run.watch(unet)
                tracker.run.watch(refer_clip_proj)
                tracker.run.watch(controlnet)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    resume_micro_batches = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            candidates = list(Path(project_dir).glob("*/checkpoint-*"))
            if not candidates:
                raise FileNotFoundError(f"No checkpoint-* directories under {project_dir}")
            path = max(candidates, key=lambda candidate: candidate.stat().st_mtime)
        else:
            path = Path(args.resume_from_checkpoint).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match is None:
            raise ValueError(f"Resume directory must be named checkpoint-N: {path}")

        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(str(path))
        global_step = int(match.group(1))
        initial_global_step = global_step
        first_epoch = global_step // num_update_steps_per_epoch
        resume_updates = global_step % num_update_steps_per_epoch
        resume_micro_batches = min(
            len(train_dataloader),
            resume_updates * args.gradient_accumulation_steps,
        )
    else:
        initial_global_step = 0


    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )


    def param_counts(model: torch.nn.Module):
        """Return (#trainable_params, #total_params)."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return trainable, total

    def _human(n: int) -> str:
        """Human-friendly count: 1.23K / 4.56M / 7.89B ..."""
        units = ["", "K", "M", "B", "T", "P"]
        x = float(n)
        i = 0
        while x >= 1000 and i < len(units) - 1:
            x /= 1000.0
            i += 1
        return f"{x:.2f}{units[i]}"

    def print_param_totals(**named_counts):
        """
        named_counts: key=model_name, value=(trainable, total)
        e.g. print_param_totals(vae=vae_param, unet=unet_param, ...)
        """
        sum_trainable = sum(tr for tr, _ in named_counts.values())
        sum_total     = sum(tot for _, tot in named_counts.values())
        pct = (sum_trainable / sum_total * 100.0) if sum_total > 0 else 0.0

        print("=== Parameter Summary ===")
        for name, (tr, tot) in named_counts.items():
            p = (tr / tot * 100.0) if tot > 0 else 0.0
            print(f"- {name:18s} trainable {tr:>15,} ({_human(tr):>8s}) / "
                f"total {tot:>15,} ({_human(tot):>8s})  |  {p:6.2f}%")

        print("-" * 80)
        print(f"TOTAL              trainable {sum_trainable:>15,} ({_human(sum_trainable):>8s}) / "
            f"total {sum_total:>15,} ({_human(sum_total):>8s})  |  {pct:6.2f}% trainable")

    vae_param = param_counts(vae)
    unet_param = param_counts(unet)
    controlnet_param = param_counts(controlnet)
    refer_clip_proj_param = param_counts(refer_clip_proj)
    clip_image_encoder_param = param_counts(clip_image_encoder)

    print_param_totals(
        vae=vae_param,
        unet=unet_param,
        controlnet=controlnet_param,
        refer_clip_proj=refer_clip_proj_param,
        clip_image_encoder=clip_image_encoder_param,
    )

    trainable_parameters = [
        parameter
        for model in (controlnet, unet, refer_clip_proj)
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    for epoch in range(first_epoch, args.num_train_epochs):
        train_dataset.set_epoch(epoch)
        epoch_dataloader = train_dataloader
        if epoch == first_epoch and resume_micro_batches:
            epoch_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_micro_batches
            )
        if hasattr(epoch_dataloader, "set_epoch"):
            epoch_dataloader.set_epoch(epoch)
        for step, batch in enumerate(epoch_dataloader):
            with accelerator.accumulate(controlnet, unet, refer_clip_proj):
                # Convert images to latent space
                latents = vae.encode(batch["deocc_tgt_img"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                occ_src_clip_img = batch["occ_src_clip_img"]
                occ_src_img = batch["occ_src_img"]
                occ_src_mask = batch["occ_src_mask"].repeat(1, 3, 1, 1)

                if args.drop_prob > 0:
                    p = random.random()
                    if p <= args.drop_prob: # dropout ref image
                        occ_src_clip_img = torch.zeros_like(occ_src_clip_img)
                        occ_src_img = torch.zeros_like(occ_src_img)
                        occ_src_mask = torch.zeros_like(occ_src_mask)

                img_latents = vae.encode(occ_src_img.to(dtype=weight_dtype)).latent_dist.mode()
                img_latents = img_latents * vae.config.scaling_factor
                mask_latents = vae.encode(occ_src_mask.to(dtype=weight_dtype)).latent_dist.mode()
                mask_latents = mask_latents * vae.config.scaling_factor
                concat_latents = torch.cat((img_latents, mask_latents), dim=1)
                encoder_hidden_states = clip_image_encoder.encode_image(occ_src_clip_img).float().unsqueeze(1)
                encoder_hidden_states = refer_clip_proj(encoder_hidden_states)

                controlnet_image = batch["occ_src_control"]
                if controlnet_image.shape[1] != args.conditioning_channels:
                    raise ValueError(
                        f"control_type={args.control_type} expects "
                        f"{args.conditioning_channels} channels, got "
                        f"{controlnet_image.shape}"
                    )

                noisy_latents = torch.cat([noisy_latents, concat_latents], dim=1)

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    conditioning_scale=args.conditioning_scale,
                    return_dict=False,
                )

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_parameters, args.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(logging_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(logging_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        os.makedirs(
                            os.path.join(logging_dir, f"checkpoint-{global_step}"),
                            exist_ok=True,
                        )
                    accelerator.wait_for_everyone()
                    save_path = os.path.join(logging_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    if accelerator.is_main_process:
                        logger.info(f"Saved state to {save_path}")

                if accelerator.is_main_process:
                    if args.validation and global_step % args.validation_steps == 0:
                        torch.cuda.empty_cache()
                        _ = log_validation(
                            logger,
                            val_dataloader,
                            vae,
                            clip_image_encoder,
                            unet,
                            controlnet,
                            noise_scheduler,
                            refer_clip_proj,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )
                        torch.cuda.empty_cache()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break


    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(logging_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        unet = accelerator.unwrap_model(unet)
        controlnet = accelerator.unwrap_model(controlnet)
        refer_clip_proj = accelerator.unwrap_model(refer_clip_proj)
        unet.save_pretrained(os.path.join(final_dir, "unet"))
        controlnet.save_pretrained(os.path.join(final_dir, "controlnet"))
        torch.save(refer_clip_proj.state_dict(), os.path.join(final_dir, "Linear.pth"))
        logger.info(f"Saved final inference weights to {final_dir}")


    accelerator.end_training()


if __name__ == "__main__":
    parser = parse_options()
    args, args_str = argparse_to_str(parser)
    main(args, args_str)
