"""Checkpoint-compatible Lite U-Net for invisible-mask prediction."""

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def load_image(path: Union[str, Path], size: int, mode: str) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert(mode)
    if size > 0:
        interpolation = Image.BICUBIC if mode == "RGB" else Image.NEAREST
        image = image.resize((size, size), interpolation)
    return image


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = array[..., None]
    array = np.transpose(array / 255.0, (2, 0, 1))
    return torch.from_numpy(array.copy())


def save_mask(mask: torch.Tensor, path: Union[str, Path]) -> None:
    array = mask.detach().cpu().clamp(0, 1).numpy()
    if array.ndim == 3:
        array = array[0]
    array = np.rint(array * 255.0).astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = self.up(inputs)
        diff_y = skip.size(2) - inputs.size(2)
        diff_x = skip.size(3) - inputs.size(3)
        inputs = F.pad(
            inputs,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        return self.conv(torch.cat([skip, inputs], dim=1))


class OutConv(nn.Module):
    """Separate module keeps the original checkpoint keys: ``outc.conv.*``."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs)


class LightUNet(nn.Module):
    """7-channel input (occ, deocc, visible) to one invisible-mask logit."""

    def __init__(self, base_channels: int = 32):
        super().__init__()
        self.inc = DoubleConv(7, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)
        self.up1 = Up(base_channels * 16, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels)
        self.outc = OutConv(base_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        outputs = self.up1(x5, x4)
        outputs = self.up2(outputs, x3)
        outputs = self.up3(outputs, x2)
        outputs = self.up4(outputs, x1)
        return self.outc(outputs)


def load_checkpoint(
    checkpoint: Union[str, Path], base_channels: int, device: torch.device
) -> LightUNet:
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    payload = torch.load(str(checkpoint), map_location=device)
    state_dict = payload.get("model") if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise ValueError(f"Invalid checkpoint: {checkpoint}")

    model = LightUNet(base_channels=base_channels).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model
