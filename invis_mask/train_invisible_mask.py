"""Train the Lite U-Net invisible-mask predictor.

Expected folders under ``--data_root``::

    occ_image/<stem>.*
    deocc_image/<stem>_000.*, <stem>_001.*, ...
    visible_mask/<stem>.*
    invisible_mask/<stem>.*        # preferred target
    amodal_mask/<stem>.*           # fallback target source

When ``invisible_mask`` is absent, the target is computed as
``amodal_mask AND NOT visible_mask``.
"""

import argparse
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from model import IMAGE_EXTENSIONS, LightUNet, load_image, pil_to_tensor, save_mask


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def files_by_stem(folder: Path, required: bool = True) -> Dict[str, Path]:
    if not folder.is_dir():
        if required:
            raise FileNotFoundError(f"Missing directory: {folder}")
        return {}
    result: Dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in result:
                raise ValueError(f"Duplicate stem '{path.stem}' in {folder}")
            result[path.stem] = path
    return result


class InvisibleMaskDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        img_size: int,
        deocc_variants: int,
        augment: bool,
        random_variant: bool,
    ):
        self.root = Path(data_root)
        self.img_size = img_size
        self.augment = augment
        self.random_variant = random_variant

        occ = files_by_stem(self.root / "occ_image")
        visible = files_by_stem(self.root / "visible_mask")
        deocc = files_by_stem(self.root / "deocc_image")
        invisible = files_by_stem(self.root / "invisible_mask", required=False)
        amodal = files_by_stem(self.root / "amodal_mask", required=False)

        records = []
        for stem in sorted(set(occ) & set(visible)):
            target_kind: Optional[str] = None
            target_path: Optional[Path] = None
            if stem in invisible:
                target_kind, target_path = "invisible", invisible[stem]
            elif stem in amodal:
                target_kind, target_path = "amodal", amodal[stem]
            if target_path is None:
                continue

            variants = [
                path
                for variant_stem, path in deocc.items()
                if variant_stem == stem
                or (
                    variant_stem.startswith(stem + "_")
                    and variant_stem[len(stem) + 1 :].isdigit()
                )
            ]
            variants.sort()
            if deocc_variants > 0:
                variants = variants[:deocc_variants]
            if variants:
                records.append(
                    {
                        "stem": stem,
                        "occ": occ[stem],
                        "visible": visible[stem],
                        "target": target_path,
                        "target_kind": target_kind,
                        "deocc": variants,
                    }
                )

        if not records:
            raise RuntimeError(
                "No valid samples found. Check occ_image, deocc_image, visible_mask, "
                "and invisible_mask (or amodal_mask)."
            )
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        variants: List[Path] = record["deocc"]
        deocc_path = random.choice(variants) if self.random_variant else variants[0]

        occ = pil_to_tensor(load_image(record["occ"], self.img_size, "RGB"))
        deocc = pil_to_tensor(load_image(deocc_path, self.img_size, "RGB"))
        visible = (
            pil_to_tensor(load_image(record["visible"], self.img_size, "L")) > 0.5
        ).float()
        target = (
            pil_to_tensor(load_image(record["target"], self.img_size, "L")) > 0.5
        ).float()
        if record["target_kind"] == "amodal":
            target = (target - visible).clamp(0, 1)

        if self.augment and random.random() < 0.5:
            occ = torch.flip(occ, dims=(2,))
            deocc = torch.flip(deocc, dims=(2,))
            visible = torch.flip(visible, dims=(2,))
            target = torch.flip(target, dims=(2,))

        return {
            "input": torch.cat((occ, deocc, visible), dim=0),
            "target": target,
            "stem": record["stem"],
        }


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    probability = torch.sigmoid(logits)
    numerator = 2 * (probability * target).sum(dim=(1, 2, 3)) + eps
    denominator = (
        probability.square().sum(dim=(1, 2, 3))
        + target.square().sum(dim=(1, 2, 3))
        + eps
    )
    return (1 - numerator / denominator).mean()


def binary_iou(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    prediction = torch.sigmoid(logits) > 0.5
    target = target > 0.5
    intersection = (prediction & target).sum(dim=(1, 2, 3)).float()
    union = (prediction | target).sum(dim=(1, 2, 3)).float()
    return torch.where(union > 0, intersection / (union + eps), torch.ones_like(union))


def save_validation_preview(
    output_dir: Path,
    epoch: int,
    stems,
    logits: torch.Tensor,
    targets: torch.Tensor,
    count: int,
) -> None:
    preview_dir = output_dir / f"epoch_{epoch:03d}"
    preview_dir.mkdir(parents=True, exist_ok=True)
    probabilities = torch.sigmoid(logits)
    for index in range(min(count, len(stems))):
        save_mask(probabilities[index], preview_dir / f"{stems[index]}_pred.png")
        save_mask(targets[index], preview_dir / f"{stems[index]}_gt.png")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument(
        "--deocc_variants",
        type=int,
        default=4,
        help="Use the first N variants per sample; 0 uses all variants.",
    )
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--ckpt_dir", default="checkpoints_im")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--val_vis_every", type=int, default=10)
    parser.add_argument("--vis_count", type=int, default=4)
    parser.add_argument("--vis_dir", default="vis_im")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.img_size < 16:
        parser.error("epochs/batch_size must be positive and img_size must be at least 16")
    if args.deocc_variants < 0 or args.base_ch < 1 or args.workers < 0:
        parser.error("deocc_variants/workers cannot be negative and base_ch must be positive")
    if not 0 < args.val_ratio < 1:
        parser.error("val_ratio must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")

    train_data = InvisibleMaskDataset(
        args.data_root, args.img_size, args.deocc_variants, True, True
    )
    val_data = InvisibleMaskDataset(
        args.data_root, args.img_size, args.deocc_variants, False, False
    )
    if len(train_data) < 2:
        raise RuntimeError("Training requires at least two valid samples")

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(train_data), generator=generator).tolist()
    num_val = min(len(indices) - 1, max(1, round(len(indices) * args.val_ratio)))
    val_indices, train_indices = indices[:num_val], indices[num_val:]

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        Subset(train_data, train_indices), shuffle=True, **loader_options
    )
    val_loader = DataLoader(Subset(val_data, val_indices), shuffle=False, **loader_options)

    model = LightUNet(base_channels=args.base_ch).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    bce = nn.BCEWithLogitsLoss()
    checkpoint_dir = Path(args.ckpt_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_score = float("inf")

    print(
        f"device={device} train={len(train_indices)} val={len(val_indices)} "
        f"amp={use_amp}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch in tqdm(train_loader, desc=f"train {epoch:03d}", dynamic_ncols=True):
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            amp_context = torch.cuda.amp.autocast() if use_amp else nullcontext()
            with amp_context:
                logits = model(inputs)
                loss = bce(logits, targets) + 0.5 * dice_loss(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * inputs.size(0)
            train_count += inputs.size(0)

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_count = 0
        preview = None
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"val   {epoch:03d}", dynamic_ncols=True):
                inputs = batch["input"].to(device, non_blocking=True)
                targets = batch["target"].to(device, non_blocking=True)
                amp_context = torch.cuda.amp.autocast() if use_amp else nullcontext()
                with amp_context:
                    logits = model(inputs)
                    loss = bce(logits, targets) + 0.5 * dice_loss(logits, targets)
                batch_size = inputs.size(0)
                val_loss += loss.item() * batch_size
                val_iou += binary_iou(logits, targets).sum().item()
                val_count += batch_size
                if preview is None:
                    preview = (batch["stem"], logits.cpu(), targets.cpu())

        train_average = train_loss / train_count
        val_average = val_loss / val_count
        iou_average = val_iou / val_count
        print(
            f"epoch={epoch:03d} train_loss={train_average:.6f} "
            f"val_loss={val_average:.6f} val_iou={iou_average:.6f}"
        )

        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "base_ch": args.base_ch,
            "val_loss": val_average,
        }
        torch.save(payload, checkpoint_dir / "last.pt")
        if val_average < best_score:
            best_score = val_average
            torch.save(payload, checkpoint_dir / "best.pt")
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(payload, checkpoint_dir / f"epoch_{epoch:03d}.pt")
        if (
            preview is not None
            and args.val_vis_every > 0
            and epoch % args.val_vis_every == 0
        ):
            save_validation_preview(
                Path(args.vis_dir), epoch, *preview, count=args.vis_count
            )

    print(f"Best checkpoint: {checkpoint_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
