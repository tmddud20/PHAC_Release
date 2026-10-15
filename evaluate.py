#!/usr/bin/env python3
import argparse
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import lpips
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

try:
    from transformers import CLIPImageProcessor, CLIPModel
except ImportError:
    CLIPImageProcessor = None
    CLIPModel = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
BODY_POSE_INDICES = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class RunningStats:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def add(self, value: float) -> None:
        value = float(value)
        if not np.isfinite(value):
            return
        self.count += 1
        self.total += value
        self.total_sq += value * value

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else float("nan")

    @property
    def var(self) -> float:
        if self.count == 0:
            return float("nan")
        mean = self.mean
        return max(self.total_sq / self.count - mean * mean, 0.0)

    @property
    def std(self) -> float:
        variance = self.var
        return math.sqrt(variance) if np.isfinite(variance) else float("nan")


def new_pair_stats() -> Dict[str, RunningStats]:
    return {
        "whole_l1": RunningStats(),
        "visible_l1": RunningStats(),
        "invisible_l1": RunningStats(),
        "lpips": RunningStats(),
    }


def new_pck_stats() -> Dict[str, RunningStats]:
    """Stats over per-GT PCK means (each GT has equal weight)."""
    return {
        "pck_005": RunningStats(),
        "pck_010": RunningStats(),
        "pck_020": RunningStats(),
    }


def is_image_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in IMAGE_EXTS


def natural_sort_key(text: str):
    return [int(token) if token.isdigit() else token.lower()
            for token in re.split(r"(\d+)", text)]


def find_same_basename(root: str, basename: str) -> Optional[str]:
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        path = os.path.join(root, basename + ext)
        if os.path.isfile(path):
            return path
    return None


def discover_sample_dirs(pred_image_root: str, expected_samples: int) -> List[Tuple[int, str]]:
    found = {}
    for name in os.listdir(pred_image_root):
        path = os.path.join(pred_image_root, name)
        if not os.path.isdir(path) or not re.fullmatch(r"\d+", name):
            continue
        sample_id = int(name)
        if sample_id in found:
            raise RuntimeError(f"Duplicate stochastic sample id: {sample_id}")
        found[sample_id] = path

    expected_ids = list(range(expected_samples))
    missing = [sid for sid in expected_ids if sid not in found]
    if missing:
        raise RuntimeError(
            "Missing stochastic sample folders: " + ", ".join(map(str, missing))
        )
    return [(sid, found[sid]) for sid in expected_ids]


def collect_predictions(
    pred_image_root: str,
    expected_samples: int,
) -> Tuple[Dict[str, List[Dict]], List[Tuple[int, str]]]:
    sample_dirs = discover_sample_dirs(pred_image_root, expected_samples)
    pred_by_base: Dict[str, List[Dict]] = defaultdict(list)

    for sample_id, sample_dir in sample_dirs:
        per_sample_seen = set()
        for dirpath, _, filenames in os.walk(sample_dir):
            for filename in sorted(filenames, key=natural_sort_key):
                if not is_image_file(filename):
                    continue
                stem = Path(filename).stem
                if stem.endswith("_all"):
                    continue
                if stem in per_sample_seen:
                    raise RuntimeError(
                        f"Duplicate prediction basename in sample {sample_id}: {stem}"
                    )
                per_sample_seen.add(stem)
                pred_by_base[stem].append(
                    {"sample": sample_id, "path": os.path.join(dirpath, filename)}
                )

    for base in pred_by_base:
        pred_by_base[base].sort(key=lambda item: item["sample"])
    return dict(pred_by_base), sample_dirs


def load_rgb_tensor(path: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().to(DEVICE)


def load_binary_mask(path: str, size_hw: Tuple[int, int], threshold: float) -> torch.Tensor:
    image = Image.open(path).convert("L")
    mask = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).unsqueeze(0)
    h, w = size_hw
    if tuple(mask.shape[-2:]) != (h, w):
        mask = F.interpolate(mask.unsqueeze(0), size=(h, w), mode="nearest").squeeze(0)
    return (mask > threshold).float().to(DEVICE)


def resize_pred_to_gt(
    pred: torch.Tensor,
    gt: torch.Tensor,
    basename: str,
    allow_resize: bool,
) -> torch.Tensor:
    if pred.shape[-2:] == gt.shape[-2:]:
        return pred
    if not allow_resize:
        ph, pw = pred.shape[-2:]
        gh, gw = gt.shape[-2:]
        raise ValueError(
            f"Resolution mismatch for {basename}: pred={pw}x{ph}, gt={gw}x{gh}. "
            "Use --resize to resize predictions to GT resolution."
        )
    return F.interpolate(
        pred.unsqueeze(0),
        size=gt.shape[-2:],
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)


def masked_l1(gt: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> float:
    mask3 = mask.expand_as(gt)
    denom = mask3.sum()
    if float(denom.item()) <= 0:
        return float("nan")
    return float((((gt - pred).abs() * mask3).sum() / denom).item())


def composite_white_tensor(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask3 = mask.expand_as(image)
    return image * mask3 + (1.0 - mask3)


@torch.no_grad()
def compute_lpips(lpips_model, gt: torch.Tensor, pred: torch.Tensor, amodal_mask: torch.Tensor) -> float:
    gt_comp = composite_white_tensor(gt, amodal_mask)
    pred_comp = composite_white_tensor(pred, amodal_mask)
    gt_lp = (gt_comp * 2.0 - 1.0).unsqueeze(0)
    pred_lp = (pred_comp * 2.0 - 1.0).unsqueeze(0)
    return float(lpips_model(gt_lp, pred_lp).item())


def tensor_to_rgb_uint8(image: torch.Tensor) -> np.ndarray:
    array = image.detach().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
    return np.round(array * 255.0).astype(np.uint8)


def get_pose_landmarks_rgb(image_rgb: np.ndarray, pose_estimator):
    result = pose_estimator.process(image_rgb)
    return result.pose_landmarks.landmark if result.pose_landmarks else []


def pck_bbox(landmarks_gt, landmarks_pred, width: int, height: int, alpha: float) -> float:
    if not landmarks_gt or not landmarks_pred:
        return float("nan")

    max_index = max(BODY_POSE_INDICES)
    if len(landmarks_gt) <= max_index or len(landmarks_pred) <= max_index:
        return float("nan")

    gt_xy = np.array(
        [(landmarks_gt[i].x * width, landmarks_gt[i].y * height) for i in BODY_POSE_INDICES],
        dtype=np.float32,
    )
    pred_xy = np.array(
        [(landmarks_pred[i].x * width, landmarks_pred[i].y * height) for i in BODY_POSE_INDICES],
        dtype=np.float32,
    )

    bbox_w = float(gt_xy[:, 0].max() - gt_xy[:, 0].min())
    bbox_h = float(gt_xy[:, 1].max() - gt_xy[:, 1].min())
    ref_len = max(bbox_w, bbox_h)
    if ref_len <= 1e-8:
        return float("nan")

    threshold = alpha * ref_len
    distances = np.linalg.norm(gt_xy - pred_xy, axis=1)
    return float(np.mean(distances <= threshold))


def pil_binary_mask(path: str, size_wh: Tuple[int, int], threshold: float) -> Image.Image:
    mask = Image.open(path).convert("L")
    if mask.size != size_wh:
        nearest = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
        mask = mask.resize(size_wh, resample=nearest)
    arr = np.asarray(mask, dtype=np.uint8)
    arr = (arr > int(round(threshold * 255.0))).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")


def composite_white_pil(image: Image.Image, mask: Image.Image) -> Image.Image:
    white = Image.new("RGB", image.size, (255, 255, 255))
    white.paste(image, (0, 0), mask)
    return white


def load_cmmd_pair(record: Dict, threshold: float, allow_resize: bool) -> Tuple[Image.Image, Image.Image]:
    pred = Image.open(record["pred_path"]).convert("RGB")
    gt = Image.open(record["gt_path"]).convert("RGB")

    if pred.size != gt.size:
        if not allow_resize:
            raise ValueError(
                f"Resolution mismatch for CMMD: pred={pred.size}, gt={gt.size}, "
                f"file={record['pred_path']}. Use --resize."
            )
        bilinear = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
        pred = pred.resize(gt.size, resample=bilinear)

    amodal = pil_binary_mask(record["amodal_path"], gt.size, threshold)
    return composite_white_pil(pred, amodal), composite_white_pil(gt, amodal)


def validate_features(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"Expected 2-D features, got {x.shape}, {y.shape}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"Feature dimensions differ: {x.shape[1]} vs {y.shape[1]}")
    return x, y


def rbf_kernel_sum(x: np.ndarray, y: np.ndarray, sigma: float, block_size: int) -> float:
    x, y = validate_features(x, y)
    gamma = 1.0 / (2.0 * sigma * sigma)
    total = 0.0
    for i in range(0, len(x), block_size):
        xb = x[i:i + block_size]
        xb_norm = np.sum(xb * xb, axis=1, keepdims=True)
        for j in range(0, len(y), block_size):
            yb = y[j:j + block_size]
            yb_norm = np.sum(yb * yb, axis=1, keepdims=True).T
            dist2 = np.maximum(xb_norm + yb_norm - 2.0 * (xb @ yb.T), 0.0)
            total += float(np.exp(-gamma * dist2).sum())
    return total


def gaussian_mmd2_biased(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float = 10.0,
    block_size: int = 4096,
) -> float:
    x, y = validate_features(x, y)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    k_xx = rbf_kernel_sum(x, x, sigma, block_size) / float(len(x) * len(x))
    k_yy = rbf_kernel_sum(y, y, sigma, block_size) / float(len(y) * len(y))
    k_xy = rbf_kernel_sum(x, y, sigma, block_size) / float(len(x) * len(y))
    return float(k_xx + k_yy - 2.0 * k_xy)


@torch.no_grad()
def extract_cmmd_features_for_sample(
    records: Sequence[Dict],
    processor,
    model,
    batch_size: int,
    threshold: float,
    allow_resize: bool,
    progress=None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    pred_features = []
    gt_features = []
    groups: List[str] = []

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        pred_images = []
        gt_images = []
        for record in batch:
            pred_pil, gt_pil = load_cmmd_pair(record, threshold, allow_resize)
            pred_images.append(pred_pil)
            gt_images.append(gt_pil)
            groups.append(record["group"])

        pred_inputs = processor(images=pred_images, return_tensors="pt")
        gt_inputs = processor(images=gt_images, return_tensors="pt")
        pred_pixels = pred_inputs["pixel_values"].to(DEVICE, non_blocking=True)
        gt_pixels = gt_inputs["pixel_values"].to(DEVICE, non_blocking=True)

        pred_feat = model.get_image_features(pixel_values=pred_pixels)
        gt_feat = model.get_image_features(pixel_values=gt_pixels)
        pred_features.append(pred_feat.detach().cpu().float().numpy())
        gt_features.append(gt_feat.detach().cpu().float().numpy())

        # Count prediction and GT CLIP extraction as two steps so that this
        # matches the H5 evaluator's single overall progress-bar accounting.
        if progress is not None:
            progress.update(2)

    if not pred_features:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 0), dtype=np.float32),
            groups,
        )

    return np.concatenate(pred_features, axis=0), np.concatenate(gt_features, axis=0), groups


def compute_cmmd_stats(
    records_by_sample: Dict[int, List[Dict]],
    clip_model_name: str,
    batch_size: int,
    threshold: float,
    allow_resize: bool,
    sigma: float,
    block_size: int,
    progress=None,
) -> Dict[str, RunningStats]:
    if CLIPModel is None or CLIPImageProcessor is None:
        raise ImportError("CMMD requires transformers. Install it with: pip install transformers")

    processor = CLIPImageProcessor.from_pretrained(clip_model_name)
    model = CLIPModel.from_pretrained(clip_model_name).to(DEVICE).eval()
    out = {"overall": RunningStats(), "high": RunningStats(), "low": RunningStats()}

    try:
        for sample_id in sorted(records_by_sample):
            records = records_by_sample[sample_id]
            if not records:
                if progress is not None:
                    progress.update(1)
                continue

            pred_feat, gt_feat, groups = extract_cmmd_features_for_sample(
                records=records,
                processor=processor,
                model=model,
                batch_size=batch_size,
                threshold=threshold,
                allow_resize=allow_resize,
                progress=progress,
            )
            if len(pred_feat) > 0:
                score = gaussian_mmd2_biased(pred_feat, gt_feat, sigma, block_size) * 1000.0
                out["overall"].add(score)

                groups_np = np.asarray(groups)
                for group in ("high", "low"):
                    idx = np.flatnonzero(groups_np == group)
                    if len(idx) == 0:
                        continue
                    score = gaussian_mmd2_biased(
                        pred_feat[idx], gt_feat[idx], sigma, block_size
                    ) * 1000.0
                    out[group].add(score)

            if progress is not None:
                progress.update(1)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return out


def format_mean(stats: RunningStats, scale: float = 1.0, decimals: int = 2) -> str:
    if stats.count == 0:
        return "nan"
    return f"{stats.mean * scale:.{decimals}f}"


def format_pck_mean_std(stats: RunningStats) -> str:
    if stats.count == 0:
        return "nan±nan"
    return f"{stats.mean:.3f}±{stats.std:.3f}"


def print_result_row(
    label: str,
    pair_stats: Dict[str, RunningStats],
    pck_stats: Dict[str, RunningStats],
    cmmd_stats: RunningStats,
) -> None:
    values = [
        f"whole_l1={format_mean(pair_stats['whole_l1'], scale=1000.0, decimals=2)}",
        f"vis_l1={format_mean(pair_stats['visible_l1'], scale=1000.0, decimals=2)}",
        f"invis_l1={format_mean(pair_stats['invisible_l1'], scale=1000.0, decimals=2)}",
        f"lpips={format_mean(pair_stats['lpips'], scale=1000.0, decimals=2)}",
        f"cmmd={format_mean(cmmd_stats, scale=1.0, decimals=2)}",
        f"pck@0.05={format_pck_mean_std(pck_stats['pck_005'])}",
        f"pck@0.10={format_pck_mean_std(pck_stats['pck_010'])}",
        f"pck@0.20={format_pck_mean_std(pck_stats['pck_020'])}",
    ]
    print(f"{label}: " + "  ".join(values))


def main(args) -> None:
    pred_root = os.path.abspath(args.pred_image_root)
    gt_root = os.path.abspath(args.gt_rgb_root)
    modal_root = os.path.abspath(args.modal_mask_root)
    amodal_root = os.path.abspath(args.amodal_mask_root)

    for path, label in (
        (pred_root, "pred_image_root"),
        (gt_root, "gt_rgb_root"),
        (modal_root, "modal_mask_root"),
        (amodal_root, "amodal_mask_root"),
    ):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Missing --{label}: {path}")

    pred_by_base, sample_dirs = collect_predictions(
        pred_root, expected_samples=args.expected_samples_per_gt
    )

    lpips_model = lpips.LPIPS(net="vgg").to(DEVICE).eval()

    pair_stats = {
        "overall": new_pair_stats(),
        "high": new_pair_stats(),
        "low": new_pair_stats(),
    }

    # Match the H5 evaluator: first average the stochastic PCK values within
    # each GT, then compute mean ± std across those per-GT means.
    pck_stats = {
        "overall": new_pck_stats(),
        "high": new_pck_stats(),
        "low": new_pck_stats(),
    }

    cmmd_records_by_sample: Dict[int, List[Dict]] = {
        sample_id: [] for sample_id, _ in sample_dirs
    }

    pose_estimator = mp.solutions.pose.Pose(
        static_image_mode=True,
        min_detection_confidence=args.pose_min_detection_confidence,
    )

    progress = None
    cmmd_stats = None
    try:
        # One progress bar for pairwise metrics + CLIP extraction + CMMD.
        pairwise_steps = sum(len(items) for items in pred_by_base.values())
        progress = tqdm(
            total=pairwise_steps + len(sample_dirs),
            desc="Evaluation",
            unit="step",
        )

        for base, pred_items in sorted(
            pred_by_base.items(), key=lambda kv: natural_sort_key(kv[0])
        ):
            gt_path = find_same_basename(gt_root, base)
            modal_path = find_same_basename(modal_root, base)
            amodal_path = find_same_basename(amodal_root, base)
            if gt_path is None or modal_path is None or amodal_path is None:
                progress.update(len(pred_items))
                continue

            gt = load_rgb_tensor(gt_path)
            h, w = gt.shape[-2:]
            amodal_mask = load_binary_mask(amodal_path, (h, w), args.mask_threshold)
            modal_mask = load_binary_mask(modal_path, (h, w), args.mask_threshold)

            visible_mask = ((amodal_mask * modal_mask) > 0.5).float()
            invisible_mask = ((amodal_mask - visible_mask).clamp(0.0, 1.0) > 0.5).float()

            amodal_area = float(amodal_mask.sum().item())
            if amodal_area <= 0:
                progress.update(len(pred_items))
                continue

            occ_ratio = float(invisible_mask.sum().item()) / amodal_area
            occ_group = "high" if occ_ratio >= 0.5 else "low"

            gt_pose = get_pose_landmarks_rgb(tensor_to_rgb_uint8(gt), pose_estimator)
            gt_pck_values = {
                "pck_005": [],
                "pck_010": [],
                "pck_020": [],
            }

            for item in pred_items:
                pred = load_rgb_tensor(item["path"])
                pred = resize_pred_to_gt(pred, gt, base, args.resize)

                metrics = {
                    "whole_l1": masked_l1(gt, pred, amodal_mask),
                    "visible_l1": masked_l1(gt, pred, visible_mask),
                    "invisible_l1": masked_l1(gt, pred, invisible_mask),
                    "lpips": compute_lpips(lpips_model, gt, pred, amodal_mask),
                }

                pred_pose = get_pose_landmarks_rgb(tensor_to_rgb_uint8(pred), pose_estimator)
                pck_values = {
                    "pck_005": pck_bbox(gt_pose, pred_pose, w, h, 0.05),
                    "pck_010": pck_bbox(gt_pose, pred_pose, w, h, 0.10),
                    "pck_020": pck_bbox(gt_pose, pred_pose, w, h, 0.20),
                }

                for key, value in metrics.items():
                    pair_stats["overall"][key].add(value)
                    pair_stats[occ_group][key].add(value)

                for key, value in pck_values.items():
                    if np.isfinite(value):
                        gt_pck_values[key].append(float(value))

                cmmd_records_by_sample[item["sample"]].append({
                    "pred_path": item["path"],
                    "gt_path": gt_path,
                    "amodal_path": amodal_path,
                    "group": occ_group,
                })
                progress.update(1)

            # Collapse stochastic predictions for this GT before dataset stats.
            for key, values in gt_pck_values.items():
                if not values:
                    continue
                gt_mean = float(np.mean(values))
                pck_stats["overall"][key].add(gt_mean)
                pck_stats[occ_group][key].add(gt_mean)

        # Add exact CLIP feature steps to the same progress bar.
        cmmd_feature_steps = sum(
            2 * math.ceil(len(records) / args.cmmd_batch_size)
            for records in cmmd_records_by_sample.values()
            if records
        )
        progress.total += cmmd_feature_steps
        progress.refresh()

        cmmd_stats = compute_cmmd_stats(
            records_by_sample=cmmd_records_by_sample,
            clip_model_name=args.cmmd_clip_model_name,
            batch_size=args.cmmd_batch_size,
            threshold=args.mask_threshold,
            allow_resize=args.resize,
            sigma=args.cmmd_sigma,
            block_size=args.cmmd_block_size,
            progress=progress,
        )
    finally:
        pose_estimator.close()
        if progress is not None:
            progress.close()

    print_result_row(
        "OVERALL", pair_stats["overall"], pck_stats["overall"], cmmd_stats["overall"]
    )
    print_result_row(
        "HIGH_OCC", pair_stats["high"], pck_stats["high"], cmmd_stats["high"]
    )
    print_result_row(
        "LOW_OCC", pair_stats["low"], pck_stats["low"], cmmd_stats["low"]
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate stochastic de-occlusion predictions with "
            "L1/LPIPS/CMMD/PCK and high-vs-low occlusion statistics."
        )
    )
    parser.add_argument("--pred_image_root", required=True)
    parser.add_argument("--gt_rgb_root", required=True)
    parser.add_argument("--modal_mask_root", required=True)
    parser.add_argument("--amodal_mask_root", required=True)

    parser.add_argument("--expected_samples_per_gt", type=int, default=16)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument(
        "--resize",
        action="store_true",
        help="Resize prediction RGB to GT resolution when resolutions differ.",
    )

    parser.add_argument(
        "--cmmd_clip_model_name",
        default="openai/clip-vit-large-patch14-336",
    )
    parser.add_argument("--cmmd_batch_size", type=int, default=32)
    parser.add_argument("--cmmd_sigma", type=float, default=10.0)
    parser.add_argument("--cmmd_block_size", type=int, default=4096)
    parser.add_argument("--pose_min_detection_confidence", type=float, default=0.0)

    args = parser.parse_args()
    if args.expected_samples_per_gt < 1:
        parser.error("--expected_samples_per_gt must be >= 1")
    if args.cmmd_batch_size < 1:
        parser.error("--cmmd_batch_size must be >= 1")
    if not (0.0 <= args.mask_threshold <= 1.0):
        parser.error("--mask_threshold must be in [0, 1]")
    if args.cmmd_sigma <= 0:
        parser.error("--cmmd_sigma must be > 0")
    if args.cmmd_block_size < 1:
        parser.error("--cmmd_block_size must be >= 1")
    return args


if __name__ == "__main__":
    main(parse_args())
