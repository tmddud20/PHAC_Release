import argparse
import gc
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from accelerate.utils import set_seed
from diffusers import ControlNetModel, DDPMScheduler, UNet2DConditionModel
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from diffusion.lib.test_dataset import build_test_dataset
from diffusion.lib.config import CONTROL_CHANNELS
from diffusion.lib.pipeline import ImageCompletionPipeline
from diffusion.lib.pix2gestalt import (
    FOUNDATION_CHECKPOINT_PATH,
    build_image_encoder,
    build_vae,
    load_state_dict as load_pix2gestalt_state_dict,
)


check_min_version("0.24.0")


def initialize_device():
    if not torch.cuda.is_available():
        raise RuntimeError("Coarse inference requires a CUDA GPU")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank}, but only {torch.cuda.device_count()} GPUs are visible"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        return device, dist.get_rank(), dist.get_world_size(), True
    return device, 0, 1, False


def load_projection(checkpoint_dir: Path) -> torch.nn.Linear:
    path = checkpoint_dir / "Linear.pth"
    if not path.is_file():
        raise FileNotFoundError(f"Projection checkpoint not found: {path}")
    state_dict = torch.load(path, map_location="cpu")
    if not isinstance(state_dict, dict) or "weight" not in state_dict:
        raise TypeError(f"Expected a Linear state_dict in {path}")
    projection = torch.nn.Linear(768, 768, bias="bias" in state_dict)
    projection.load_state_dict(state_dict, strict=True)
    return projection


def prepare_output(output_path: Path, rank: int, distributed: bool) -> Path:
    if output_path.exists() and any(output_path.rglob("*.png")):
        raise FileExistsError(
            f"Output already contains PNG files; refusing to mix runs: {output_path}"
        )
    if rank == 0:
        (output_path / "all_images").mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    rank_dir = output_path / "all_images" / f"rank{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    return rank_dir




def parse_final_output(filename: str):
    """
    Convert an intermediate rank filename such as
    0522_0_r0_000.png into:

        sample folder: 0
        final filename: 0522_000.png

    Rules:
    - remove the _r<rank> component
    - zero-pad a trailing numeric view component to 3 digits when present
    - use the final variant index as the numeric output subdirectory name
    - remove the variant index from the final filename
    """
    path = Path(filename)
    match = re.fullmatch(r"(.+)_r(\d+)_(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Unexpected intermediate filename format: {filename}")

    sample_name, _rank, variant = match.groups()
    final_stem = sample_name
    if "_" in sample_name:
        prefix, trailing = sample_name.rsplit("_", 1)
        if trailing.isdigit():
            final_stem = f"{prefix}_{trailing.zfill(3)}"
    variant_index = int(variant)
    final_filename = f"{final_stem}{path.suffix}"
    return variant_index, final_filename


def postprocess_rank_outputs(
    output_path: Path,
    world_size: int,
    num_validation_images: int,
) -> None:
    """
    Keep per-rank output directories during inference, then reorganize them as:

        all_images/
          0/<sample>.png
          1/<sample>.png
          ...
          (num_validation_images - 1)/<sample>.png

    Example:
        all_images/rank0/0522_0_r0_000.png
        -> all_images/0/0522_000.png

        all_images/rank0/0522_0_r0_015.png
        -> all_images/15/0522_000.png

    The temporary rank directories are removed after all moves succeed.
    """
    all_images_dir = output_path / "all_images"
    rank_dirs = [all_images_dir / f"rank{rank}" for rank in range(world_size)]

    # Create deterministic stochastic-sample folders (0, 1, ..., N-1).
    sample_dirs = {
        sample_id: all_images_dir / str(sample_id)
        for sample_id in range(num_validation_images)
    }
    for sample_dir in sample_dirs.values():
        sample_dir.mkdir(parents=True, exist_ok=True)

    move_pairs = []
    targets = set()

    for rank_dir in rank_dirs:
        if not rank_dir.is_dir():
            raise FileNotFoundError(f"Missing rank output directory: {rank_dir}")

        for src in sorted(rank_dir.glob("*.png")):
            variant_index, final_filename = parse_final_output(src.name)

            if variant_index not in sample_dirs:
                raise ValueError(
                    f"Variant index {variant_index} is outside the expected range "
                    f"0..{num_validation_images - 1}: {src.name}"
                )

            dst = sample_dirs[variant_index] / final_filename

            if dst in targets:
                raise FileExistsError(
                    f"Multiple intermediate files map to the same output: {dst}"
                )
            if dst.exists():
                raise FileExistsError(f"Final output already exists: {dst}")

            targets.add(dst)
            move_pairs.append((src, dst))

    # Only modify files after the complete collision/format validation succeeds.
    for src, dst in move_pairs:
        src.replace(dst)

    for rank_dir in rank_dirs:
        leftovers = list(rank_dir.iterdir())
        if leftovers:
            raise RuntimeError(
                f"Rank directory still contains unexpected files: {rank_dir}"
            )
        rank_dir.rmdir()

    print(
        f"Post-processing complete: moved {len(move_pairs)} images into "
        f"{num_validation_images} sample folders under {all_images_dir}"
    )


def build_original_size_map(full_dataset, rank_indices):
    """Collect source sizes through either the HDF5 or folder backend."""
    original_sizes = {}
    for global_index in rank_indices:
        name = full_dataset.sample_name(global_index)
        original_size = full_dataset.original_size(global_index)
        if name in original_sizes and original_sizes[name] != original_size:
            raise RuntimeError(
                f"Conflicting original sizes for sample {name}: "
                f"{original_sizes[name]} vs {original_size}"
            )
        original_sizes[name] = original_size

    return original_sizes


def resize_to_original(image: Image.Image, original_size):
    """Resize generated RGB output back to the source image's (W, H)."""
    original_size = tuple(map(int, original_size))
    if image.size == original_size:
        return image

    if hasattr(Image, "Resampling"):
        resample = Image.Resampling.BILINEAR
    else:
        resample = Image.BILINEAR
    return image.resize(original_size, resample=resample)

def main(args) -> None:
    device, rank, world_size, distributed = initialize_device()
    set_seed(args.seed + rank)

    output_path = Path(args.output_path).expanduser().resolve()
    rank_dir = prepare_output(output_path, rank, distributed)

    checkpoint_dir = Path(args.pretrained_model_name_or_path).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {checkpoint_dir}")

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        steps_offset=1,
        prediction_type="epsilon",
    )
    unet = UNet2DConditionModel.from_pretrained(checkpoint_dir, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(checkpoint_dir, subfolder="controlnet")
    expected_channels = CONTROL_CHANNELS[args.control_type]
    if controlnet.config.conditioning_channels != expected_channels:
        raise ValueError(
            f"control_type={args.control_type} expects {expected_channels} channels, "
            f"but the checkpoint ControlNet has "
            f"{controlnet.config.conditioning_channels}"
        )
    projection = load_projection(checkpoint_dir)

    foundation_state = load_pix2gestalt_state_dict(args.foundation_ckpt)
    vae = build_vae(foundation_state)
    clip_image_encoder = build_image_encoder(foundation_state)
    del foundation_state
    gc.collect()

    for model in (vae, unet, controlnet, projection, clip_image_encoder):
        model.requires_grad_(False)
        model.eval()

    pipeline = ImageCompletionPipeline(
        vae=vae,
        clip_image_encoder=clip_image_encoder,
        unet=unet,
        controlnet=controlnet,
        scheduler=scheduler,
        refer_clip_proj=projection,
        torch_dtype=torch.float32,
    ).to(device)
    from diffusers import UniPCMultistepScheduler

    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        if not is_xformers_available():
            raise RuntimeError("xformers is not installed")
        pipeline.enable_xformers_memory_efficient_attention()

    full_dataset = build_test_dataset(args)
    rank_indices = list(range(rank, len(full_dataset), world_size))
    original_sizes = build_original_size_map(full_dataset, rank_indices)
    dataset = Subset(full_dataset, rank_indices)
    loader_options = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        loader_options["multiprocessing_context"] = "spawn"
    data_loader = DataLoader(dataset, **loader_options)
    generator = torch.Generator(device=device).manual_seed(args.seed + rank)

    progress = tqdm(data_loader, disable=rank != 0, desc="coarse inference")
    for batch in progress:
        batch_size = batch["occ_src_img"].shape[0]
        with torch.autocast(device_type="cuda", enabled=True):
            outputs = pipeline.forward(
                batch,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                guidance_scale=args.guidance_scale,
                controlnet_conditioning_scale=args.conditioning_scale,
                num_images_per_prompt=args.num_validation_images,
            )
        expected = batch_size * args.num_validation_images
        if outputs.shape[0] != expected:
            raise RuntimeError(
                f"Pipeline returned {outputs.shape[0]} images; expected {expected}"
            )
        outputs = outputs.reshape(
            batch_size, args.num_validation_images, *outputs.shape[1:]
        )
        names = list(batch["name"])
        for batch_index, name in enumerate(names):
            for variant_index in range(args.num_validation_images):
                image = (outputs[batch_index, variant_index] * 255).clip(0, 255)
                filename = f"{name}_r{rank}_{variant_index:03d}.png"

                output_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
                if name not in original_sizes:
                    raise KeyError(f"Original size not found for sample: {name}")
                output_image = resize_to_original(output_image, original_sizes[name])
                output_image.save(rank_dir / filename)

    # Keep the current per-rank saving behavior during inference, then merge and
    # rename all outputs only after every rank has finished writing its files.
    if distributed:
        dist.barrier()

    if rank == 0:
        postprocess_rank_outputs(
            output_path,
            world_size,
            args.num_validation_images,
        )

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", required=True)
    parser.add_argument(
        "--validation_data_dir",
        required=True,
        help=(
            "HDF5 test file, or a rendered-data root containing occ_src_img/, "
            "modal_mask/, amodal_mask/, and occ_src_pose/"
        ),
    )
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--foundation_ckpt",
        default=str(FOUNDATION_CHECKPOINT_PATH),
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--control_type",
        choices=tuple(CONTROL_CHANNELS),
        default="po",
    )
    parser.add_argument(
        "--name_mode",
        choices=("subject_view", "subject"),
        default="subject_view",
        help="output stem: THuman uses subject_view; single-view AHP uses subject",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--conditioning_scale", type=float, default=1.0)
    parser.add_argument("--num_validation_images", type=int, default=16)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true"
    )
    options = parser.parse_args()

    if options.batch_size < 1 or options.num_validation_images < 1:
        parser.error("batch_size and num_validation_images must be positive")
    if not Path(options.validation_data_dir).expanduser().exists():
        parser.error(
            f"validation_data_dir does not exist: {options.validation_data_dir}"
        )
    return options


if __name__ == "__main__":
    main(parse_args())
