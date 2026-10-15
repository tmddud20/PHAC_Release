"""Predict an invisible mask for every de-occluded image in a folder."""

import argparse
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from tqdm.auto import tqdm

from model import (
    IMAGE_EXTENSIONS,
    load_checkpoint,
    load_image,
    pil_to_tensor,
    save_mask,
)


def image_files(folder: Path) -> Iterable[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def files_by_stem(folder: Path) -> Dict[str, Path]:
    result = {}
    for path in image_files(folder):
        if path.stem in result:
            raise ValueError(f"Duplicate stem '{path.stem}' in {folder}")
        result[path.stem] = path
    return result


def resolve_directory(root: Path, explicit: Optional[str], candidates, label: str) -> Path:
    if explicit:
        path = Path(explicit)
        path = path if path.is_absolute() else root / path
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {label} directory: {path}")
        return path
    for name in candidates:
        path = root / name
        if path.is_dir():
            return path
    raise FileNotFoundError(
        f"Could not detect {label} directory under {root}; set --{label}_dir explicitly"
    )


def resolve_deocc_directory(root: Path, explicit: Optional[str], post_fix: Optional[str]) -> Path:
    if post_fix and explicit:
        raise ValueError("Use either --deocc_dir or --post_fix, not both")
    if post_fix:
        explicit = f"deocc_{post_fix}"
    if explicit:
        return resolve_directory(root, explicit, (), "deocc")
    standard = root / "deocc_image"
    if standard.is_dir():
        return standard
    candidates = sorted(path for path in root.glob("deocc_*") if path.is_dir())
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"Could not find deocc_image or deocc_* under {root}")
    names = ", ".join(path.name for path in candidates)
    raise ValueError(f"Multiple deocc directories found ({names}); set --deocc_dir")


def base_stem(deocc_stem: str, occ: Dict[str, Path], visible: Dict[str, Path]) -> str:
    if deocc_stem in occ and deocc_stem in visible:
        return deocc_stem
    stripped = re.sub(r"_\d{3,}$", "", deocc_stem)
    if stripped in occ and stripped in visible:
        return stripped
    return stripped


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--ckpt", default="checkpoints_im/best.pt")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument(
        "--out_folder",
        required=True,
        help="Exact directory in which output PNG masks are written.",
    )
    parser.add_argument("--occ_dir", help="Absolute path or name relative to data_root")
    parser.add_argument("--deocc_dir", help="Absolute path or name relative to data_root")
    parser.add_argument("--mask_dir", help="Absolute path or name relative to data_root")
    parser.add_argument(
        "--post_fix",
        help="Compatibility shortcut for --deocc_dir deocc_<post_fix>",
    )
    parser.add_argument(
        "--mask_thresh",
        type=float,
        help="Write a binary mask at this threshold; default writes probabilities.",
    )
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    if args.img_size < 16 or args.base_ch < 1:
        parser.error("img_size must be at least 16 and base_ch must be positive")
    if args.mask_thresh is not None and not 0 <= args.mask_thresh <= 1:
        parser.error("mask_thresh must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    root = Path(args.data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data root not found: {root}")

    occ_dir = resolve_directory(root, args.occ_dir, ("occ_image", "occ"), "occ")
    visible_dir = resolve_directory(
        root, args.mask_dir, ("visible_mask", "mask", "mask_ref"), "mask"
    )
    deocc_dir = resolve_deocc_directory(root, args.deocc_dir, args.post_fix)
    occ = files_by_stem(occ_dir)
    visible = files_by_stem(visible_dir)
    deocc = list(image_files(deocc_dir))
    if not deocc:
        raise FileNotFoundError(f"No images found in {deocc_dir}")

    jobs = []
    missing = []
    for deocc_path in deocc:
        stem = base_stem(deocc_path.stem, occ, visible)
        if stem not in occ or stem not in visible:
            missing.append(deocc_path.name)
            continue
        jobs.append((deocc_path, occ[stem], visible[stem]))
    if missing and not args.skip_missing:
        examples = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Missing matching occ/mask for {len(missing)} deocc images; examples: {examples}"
        )
    if not jobs:
        raise RuntimeError("No valid inference inputs")

    output_dir = Path(args.out_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / f"{job[0].stem}.png" for job in jobs]
    existing = [path for path in existing if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{len(existing)} output files already exist in {output_dir}; use --overwrite"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    model = load_checkpoint(args.ckpt, args.base_ch, device)
    print(
        f"device={device} inputs={len(jobs)} skipped={len(missing)} amp={use_amp}\n"
        f"occ={occ_dir}\ndeocc={deocc_dir}\nmask={visible_dir}\nout={output_dir}"
    )

    with torch.no_grad():
        for deocc_path, occ_path, visible_path in tqdm(
            jobs, desc="invisible mask", unit="img", dynamic_ncols=True
        ):
            occ_tensor = pil_to_tensor(load_image(occ_path, args.img_size, "RGB"))
            deocc_tensor = pil_to_tensor(load_image(deocc_path, args.img_size, "RGB"))
            visible_tensor = (
                pil_to_tensor(load_image(visible_path, args.img_size, "L")) > 0.5
            ).float()
            inputs = torch.cat(
                (occ_tensor, deocc_tensor, visible_tensor), dim=0
            ).unsqueeze(0).to(device)
            amp_context = torch.cuda.amp.autocast() if use_amp else nullcontext()
            with amp_context:
                probability = torch.sigmoid(model(inputs))[0, 0].cpu()
            if args.mask_thresh is not None:
                probability = (probability > args.mask_thresh).float()
            save_mask(probability, output_dir / f"{deocc_path.stem}.png")

    print(f"Saved {len(jobs)} masks to {output_dir}")


if __name__ == "__main__":
    main()
