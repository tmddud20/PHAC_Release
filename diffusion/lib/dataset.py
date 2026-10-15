import io
import re
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import CONTROL_CHANNELS


BASE_VIEW_FIELDS = ("occ_src_img", "deocc_tgt_img")
POSE_CONTROL_TYPES = {"po", "poib", "poeb"}
BBOX_CONTROL_TYPES = {"ib", "eb", "poib", "poeb"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _rgb_tensor(image: Image.Image, size: int) -> torch.Tensor:
    return transforms.Compose(
        [
            transforms.Resize(
                (size, size), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )(image)


def _mask_tensor(image: Image.Image, size: int) -> torch.Tensor:
    tensor = transforms.Compose(
        [
            transforms.Resize(
                (size, size), interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor(),
        ]
    )(image)
    return (tensor >= 0.5).float().mul(2).sub(1)


_CLIP_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(
            (224, 224), interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711],
        ),
    ]
)


def _decode_png(encoded, mode: str) -> Image.Image:
    payload = np.asarray(encoded, dtype=np.uint8).tobytes()
    with Image.open(io.BytesIO(payload)) as image:
        return image.convert(mode)


def _bbox_edge(mask: torch.Tensor, thickness: int = 4) -> torch.Tensor:
    """Return a three-channel bbox edge image in [-1, 1]."""
    mask = mask.squeeze(0) > 0
    height, width = mask.shape
    edge = torch.full((height, width), -1.0, dtype=torch.float32)
    if not torch.any(mask):
        return edge.unsqueeze(0).repeat(3, 1, 1)

    ys, xs = torch.where(mask)
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    thickness = max(1, int(thickness))
    edge[y1 : min(y1 + thickness, height), x1 : x2 + 1] = 1
    edge[max(y2 - thickness + 1, 0) : y2 + 1, x1 : x2 + 1] = 1
    edge[y1 : y2 + 1, x1 : min(x1 + thickness, width)] = 1
    edge[y1 : y2 + 1, max(x2 - thickness + 1, 0) : x2 + 1] = 1
    return edge.unsqueeze(0).repeat(3, 1, 1)


def _build_control(
    control_type: str,
    visible_mask: torch.Tensor,
    amodal_mask: Optional[torch.Tensor],
    pose_image: Optional[Image.Image],
    size: int,
) -> torch.Tensor:
    if control_type in BBOX_CONTROL_TYPES:
        if amodal_mask is None:
            raise ValueError(f"control_type={control_type} requires an amodal mask")
        invisible_mask = (
            (amodal_mask > 0) & (visible_mask <= 0)
        ).float().mul(2).sub(1)
        interested_bbox = _bbox_edge(invisible_mask)
        entire_bbox = _bbox_edge(amodal_mask)

        if control_type == "ib":
            return interested_bbox
        if control_type == "eb":
            return entire_bbox

    if pose_image is None:
        raise ValueError(f"control_type={control_type} requires a pose image")
    pose = _rgb_tensor(pose_image, size)
    if control_type == "po":
        return pose
    if control_type == "poib":
        return torch.cat((pose, interested_bbox), dim=0)
    if control_type == "poeb":
        return torch.cat((pose, entire_bbox), dim=0)
    raise ValueError(f"Unsupported control_type: {control_type}")


class ControlH5Dataset(Dataset):
    """Shared HDF5 reader for the supported control types."""

    def __init__(self, h5_path: str, resolution: int, control_type: str):
        self.h5_path = str(Path(h5_path).expanduser().resolve())
        self.img_size = resolution
        self.control_type = control_type
        if control_type not in CONTROL_CHANNELS:
            raise ValueError(
                f"control_type must be one of {tuple(CONTROL_CHANNELS)}, got {control_type}"
            )
        if not Path(self.h5_path).is_file():
            raise FileNotFoundError(f"HDF5 dataset not found: {self.h5_path}")

        required_fields = list(BASE_VIEW_FIELDS)
        if control_type in POSE_CONTROL_TYPES:
            required_fields.append("occ_src_pose")

        self.subject_names = []
        self.view_counts = {}
        with h5py.File(self.h5_path, "r") as dataset:
            self.subject_names = sorted(dataset.keys())
            if not self.subject_names:
                raise ValueError(f"HDF5 dataset has no subject groups: {self.h5_path}")
            for subject_name in self.subject_names:
                group = dataset[subject_name]
                missing = [field for field in required_fields if field not in group]
                if missing:
                    raise KeyError(
                        f"Subject '{subject_name}' is missing HDF5 fields: {missing}"
                    )
                view_count = len(group["occ_src_img"])
                mismatched = {
                    field: len(group[field])
                    for field in required_fields
                    if len(group[field]) != view_count
                }
                if view_count < 1 or mismatched:
                    raise ValueError(
                        f"Subject '{subject_name}' has invalid view counts: "
                        f"occ_src_img={view_count}, mismatched={mismatched}"
                    )
                self.view_counts[subject_name] = view_count

    def _load_view(self, subject_name: str, view_id: int):
        with h5py.File(self.h5_path, "r") as dataset:
            group = dataset[subject_name]
            source_rgba = _decode_png(group["occ_src_img"][view_id], "RGBA")
            target_rgba = _decode_png(group["deocc_tgt_img"][view_id], "RGBA")
            pose_image = None
            if self.control_type in POSE_CONTROL_TYPES:
                pose_image = _decode_png(group["occ_src_pose"][view_id], "RGB")

        source_rgb = source_rgba.convert("RGB")
        visible_mask = _mask_tensor(source_rgba.getchannel("A"), self.img_size)
        amodal_mask = _mask_tensor(target_rgba.getchannel("A"), self.img_size)
        control = _build_control(
            self.control_type,
            visible_mask,
            amodal_mask,
            pose_image,
            self.img_size,
        )
        expected_channels = CONTROL_CHANNELS[self.control_type]
        if control.shape[0] != expected_channels:
            raise RuntimeError(
                f"{self.control_type} produced {control.shape[0]} channels; "
                f"expected {expected_channels}"
            )

        return {
            "occ_src_img": _rgb_tensor(source_rgb, self.img_size),
            "deocc_tgt_img": _rgb_tensor(target_rgba.convert("RGB"), self.img_size),
            "occ_src_mask": visible_mask,
            "occ_src_control": control,
            "occ_src_clip_img": _CLIP_TRANSFORM(source_rgb),
            "name": f"{subject_name}_{view_id}",
        }


def _natural_sort_key(text: str):
    return [
        (0, int(token)) if token.isdigit() else (1, token.lower())
        for token in re.split(r"(\d+)", text)
    ]


def _index_image_folder(folder: Path) -> Dict[str, Path]:
    """Index images by basename and reject ambiguous duplicate basenames."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Image directory not found: {folder}")

    indexed = {}
    for path in folder.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in indexed:
            raise ValueError(
                f"Duplicate image basename '{path.stem}' under {folder}: "
                f"{indexed[path.stem]} and {path}"
            )
        indexed[path.stem] = path
    if not indexed:
        raise ValueError(f"No supported images found under: {folder}")
    return indexed


class ControlFolderDataset(Dataset):
    """Read the rendered folder equivalent of ``ControlH5Dataset``.

    The input root uses the same basenames in these subdirectories::

        occ_src_img/   modal_mask/   amodal_mask/   occ_src_pose/

    ``amodal_mask`` is required only for bbox-based controls, and
    ``occ_src_pose`` is required only for pose-based controls.
    """

    def __init__(self, folder_root: str, resolution: int, control_type: str):
        self.folder_root = Path(folder_root).expanduser().resolve()
        self.img_size = resolution
        self.control_type = control_type
        if control_type not in CONTROL_CHANNELS:
            raise ValueError(
                f"control_type must be one of {tuple(CONTROL_CHANNELS)}, got {control_type}"
            )
        if not self.folder_root.is_dir():
            raise FileNotFoundError(f"Rendered dataset root not found: {self.folder_root}")

        self.source_paths = _index_image_folder(self.folder_root / "occ_src_img")
        self.modal_paths = _index_image_folder(self.folder_root / "modal_mask")

        amodal_dir = self.folder_root / "amodal_mask"
        self.amodal_paths = (
            _index_image_folder(amodal_dir) if amodal_dir.is_dir() else {}
        )
        pose_dir = self.folder_root / "occ_src_pose"
        self.pose_paths = _index_image_folder(pose_dir) if pose_dir.is_dir() else {}

        if control_type in BBOX_CONTROL_TYPES and not self.amodal_paths:
            raise FileNotFoundError(
                f"control_type={control_type} requires: {amodal_dir}"
            )
        if control_type in POSE_CONTROL_TYPES and not self.pose_paths:
            raise FileNotFoundError(
                f"control_type={control_type} requires: {pose_dir}"
            )

        self.sample_names = sorted(self.source_paths, key=_natural_sort_key)
        required_maps = {"modal_mask": self.modal_paths}
        if control_type in BBOX_CONTROL_TYPES:
            required_maps["amodal_mask"] = self.amodal_paths
        if control_type in POSE_CONTROL_TYPES:
            required_maps["occ_src_pose"] = self.pose_paths

        for label, paths in required_maps.items():
            missing = [name for name in self.sample_names if name not in paths]
            if missing:
                preview = ", ".join(missing[:5])
                raise FileNotFoundError(
                    f"{label} is missing {len(missing)} matching files; first: {preview}"
                )

    def __len__(self) -> int:
        return len(self.sample_names)

    def original_size(self, index: int):
        with Image.open(self.source_paths[self.sample_names[index]]) as image:
            return image.size

    def __getitem__(self, index: int):
        name = self.sample_names[index]
        with Image.open(self.source_paths[name]) as image:
            source_rgb = image.convert("RGB")
        with Image.open(self.modal_paths[name]) as image:
            visible_mask = _mask_tensor(image.convert("L"), self.img_size)

        amodal_mask = None
        if name in self.amodal_paths:
            with Image.open(self.amodal_paths[name]) as image:
                amodal_mask = _mask_tensor(image.convert("L"), self.img_size)
            # Restrict the visible region to the supplied amodal silhouette.
            visible_mask = (
                (visible_mask > 0) & (amodal_mask > 0)
            ).float().mul(2).sub(1)

        pose_image = None
        if name in self.pose_paths:
            with Image.open(self.pose_paths[name]) as image:
                pose_image = image.convert("RGB")

        control = _build_control(
            self.control_type,
            visible_mask,
            amodal_mask,
            pose_image,
            self.img_size,
        )
        expected_channels = CONTROL_CHANNELS[self.control_type]
        if control.shape[0] != expected_channels:
            raise RuntimeError(
                f"{self.control_type} produced {control.shape[0]} channels; "
                f"expected {expected_channels}"
            )

        return {
            "occ_src_img": _rgb_tensor(source_rgb, self.img_size),
            "occ_src_mask": visible_mask,
            "occ_src_control": control,
            "occ_src_clip_img": _CLIP_TRANSFORM(source_rgb),
            "name": name,
        }
