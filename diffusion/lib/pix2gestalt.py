from pathlib import Path
import torch
import torch.nn as nn
from clip.model import VisionTransformer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    convert_ldm_unet_checkpoint,
    convert_ldm_vae_checkpoint,
)


FOUNDATION_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[2]
    / "checkpoints"
    / "pix2gestalt.ckpt"
)


VAE_CONFIG = {
    "sample_size": 256,
    "in_channels": 3,
    "out_channels": 3,
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
    "block_out_channels": [128, 256, 512, 512],
    "latent_channels": 4,
    "layers_per_block": 2,
    "norm_num_groups": 32,
    "scaling_factor": 0.18215,
}

UNET_CONFIG = {
    "sample_size": 64,
    "in_channels": 12,
    "out_channels": 4,
    "layers_per_block": 2,
    "block_out_channels": [320, 640, 1280, 1280],
    "down_block_types": [
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    ],
    "up_block_types": [
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    ],
    "cross_attention_dim": 768,
    "attention_head_dim": [5, 10, 20, 20],
    "norm_num_groups": 32,
    "mid_block_type": "UNetMidBlock2DCrossAttn",
    "class_embed_type": None,
    "num_class_embeds": None,
    "addition_time_embed_dim": None,
    "addition_embed_type": None,
    "time_cond_proj_dim": None,
}


class Pix2GestaltImageEncoder(nn.Module):
    """The ViT-L/14 visual tower stored inside the Pix2Gestalt checkpoint."""

    def __init__(self, visual: VisionTransformer):
        super().__init__()
        self.visual = visual

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.visual(image.to(dtype=self.visual.conv1.weight.dtype))


def load_state_dict(checkpoint_path) -> dict:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Foundation checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unexpected foundation checkpoint format: {checkpoint_path}")
    return state_dict


def build_vae(state_dict: dict) -> AutoencoderKL:
    converted = convert_ldm_vae_checkpoint(state_dict, VAE_CONFIG)
    model = AutoencoderKL(**VAE_CONFIG)
    model.load_state_dict(converted, strict=True)
    return model


def build_unet(state_dict: dict) -> UNet2DConditionModel:
    converted = convert_ldm_unet_checkpoint(state_dict, UNET_CONFIG)
    model = UNet2DConditionModel(**UNET_CONFIG)
    model.load_state_dict(converted, strict=True)
    return model


def build_image_encoder(state_dict: dict) -> Pix2GestaltImageEncoder:
    prefix = "cond_stage_model.model.visual."
    visual_state = {
        key.removeprefix(prefix): value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    required = ("conv1.weight", "positional_embedding", "proj")
    missing = [key for key in required if key not in visual_state]
    if missing:
        raise KeyError(f"Pix2Gestalt CLIP visual state is missing: {missing}")

    width = visual_state["conv1.weight"].shape[0]
    patch_size = visual_state["conv1.weight"].shape[-1]
    layers = len(
        [
            key
            for key in visual_state
            if key.endswith("attn.in_proj_weight")
        ]
    )
    grid_size = round((visual_state["positional_embedding"].shape[0] - 1) ** 0.5)
    output_dim = visual_state["proj"].shape[1]
    visual = VisionTransformer(
        input_resolution=patch_size * grid_size,
        patch_size=patch_size,
        width=width,
        layers=layers,
        heads=width // 64,
        output_dim=output_dim,
    )
    visual.load_state_dict(visual_state, strict=True)
    return Pix2GestaltImageEncoder(visual)


def build_projection(state_dict: dict) -> nn.Linear:
    projection = nn.Linear(768, 768)
    projection.load_state_dict(
        {
            "weight": state_dict["cc_projection.weight"],
            "bias": state_dict["cc_projection.bias"],
        },
        strict=True,
    )
    return projection
