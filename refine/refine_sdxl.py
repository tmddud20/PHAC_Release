import os, sys, argparse, glob, re
import multiprocessing as mp
from typing import List, Dict, Tuple
import torch
from PIL import Image
from tqdm.auto import tqdm
from diffusers import StableDiffusionXLInpaintPipeline, DPMSolverMultistepScheduler

# ----------------------------
# Utils
# ----------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

def is_img(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in IMG_EXTS

def load_rgb(path: str, side: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((side, side), Image.BICUBIC)

def load_mask(path: str, side: int) -> Image.Image:
    return Image.open(path).convert("L").resize((side, side), Image.NEAREST)

def stem_of(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def split_base_variant(stem: str) -> Tuple[str, str]:
    match = re.fullmatch(r"^(.*)_(\d{3})$", stem)
    if match is None:
        raise ValueError(f"Expected a three-digit variant suffix: {stem}")
    return match.group(1), match.group(2)

def index_by_stem(paths: List[str]) -> Dict[str, str]:
    d = {}
    for p in paths:
        stem = stem_of(p)
        if stem in d:
            raise ValueError(f"Duplicate image stem across extensions: {stem}")
        d[stem] = p
    return d

def collect_tasks_nk(input_dir: str, post_fix, num_variants: int):
    occ_dir  = os.path.join(input_dir, "occ")
    deocc_dir = os.path.join(input_dir, f"deocc_{post_fix}")
    mask_dir  = os.path.join(input_dir, "mask")
    invis_dir = os.path.join(input_dir, f"invisible_mask_{post_fix}")

    required_dirs = (occ_dir, deocc_dir, mask_dir, invis_dir)
    missing_dirs = [path for path in required_dirs if not os.path.isdir(path)]
    if missing_dirs:
        raise FileNotFoundError(
            "Missing refine input directories: " + ", ".join(missing_dirs)
        )

    occs   = [p for p in glob.glob(os.path.join(occ_dir,   "*")) if is_img(p)]
    deoccs = [p for p in glob.glob(os.path.join(deocc_dir, "*")) if is_img(p)]
    masks  = [p for p in glob.glob(os.path.join(mask_dir,  "*")) if is_img(p)]
    invs   = [p for p in glob.glob(os.path.join(invis_dir, "*")) if is_img(p)]

    D_occ  = index_by_stem(occs)
    D_mask = index_by_stem(masks)
    D_deocc = index_by_stem(deoccs)
    D_invis_full = index_by_stem(invs)

    if set(D_occ) != set(D_mask):
        raise ValueError(
            f"occ/mask stem sets differ: occ={len(D_occ)}, mask={len(D_mask)}"
        )
    expected_deocc = {
        f"{base_stem}_{variant:03d}"
        for base_stem in D_occ
        for variant in range(num_variants)
    }
    if set(D_deocc) != expected_deocc:
        raise ValueError(
            f"Expected {len(expected_deocc)} deocc variants, got {len(D_deocc)}"
        )

    tasks: List[Tuple[str, str, str, str, str]] = []
    missing: List[Tuple[str, bool, bool, bool]] = []

    for deocc_p in sorted(deoccs, key=lambda x: stem_of(x)):
        deocc_stem = stem_of(deocc_p)
        base_stem, _ = split_base_variant(deocc_stem)

        occ_p   = D_occ.get(base_stem)
        mask_p  = D_mask.get(base_stem)
        invis_p = D_invis_full.get(deocc_stem)

        if occ_p and mask_p and invis_p:
            tasks.append((occ_p, deocc_p, mask_p, invis_p, deocc_stem))
        else:
            missing.append((deocc_stem, bool(occ_p), bool(mask_p), bool(invis_p)))

    if not tasks:
        tqdm.write("[error] No matching inputs; check file names and directories.")
    else:
        tqdm.write(f"[info] de-occlusion variants: {len(deoccs)} | matched: {len(tasks)}")

    if missing:
        sample = ", ".join([f"{st}(occ={o},mask={m},invis={iv})" for st,o,m,iv in missing[:10]])
        raise FileNotFoundError(f"Missing {len(missing)} refinement inputs; examples: {sample}")

    deocc_stems = {stem_of(path) for path in deoccs}
    invisible_stems = set(D_invis_full)
    if deocc_stems != invisible_stems:
        raise ValueError(
            "deocc/invisible stem sets differ: "
            f"deocc={len(deocc_stems)}, invisible={len(invisible_stems)}"
        )

    return tasks

def build_pipe(model_id: str, device: str, dtype=torch.float16):
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        model_id, torch_dtype=dtype, variant="fp16"
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe

def refine_batch(pipe, items, side, save_side, out_dir, guidance, steps, strength, seed):
    As, Ms, save_stems = [], [], []
    for occ_p, deocc_p, mask_p, invis_mask_p, save_stem in items:
        occ = load_rgb(occ_p, side)
        deocc = load_rgb(deocc_p, side)
        vis = load_mask(mask_p, side)
        invis = load_mask(invis_mask_p, side)

        A = Image.composite(occ, deocc, vis)
        As.append(A); Ms.append(invis); save_stems.append(save_stem)

    B = len(As)
    if B == 0:
        return

    prompts = [""] * B

    gens = None
    if seed is not None:
        gens = []
        for j in range(B):
            g = torch.Generator(device=pipe._execution_device)
            g.manual_seed(seed + j)
            gens.append(g)
    out_imgs = pipe(
        prompt=prompts,
        image=As,
        mask_image=Ms,
        guidance_scale=guidance,
        num_inference_steps=steps,
        strength=strength,
        original_size=(side, side),
        target_size=(side, side),
        crops_coords_top_left=(0, 0),
        generator=gens,
    ).images

    os.makedirs(out_dir, exist_ok=True)
    for stem, img in zip(save_stems, out_imgs):
        if save_side is not None and save_side != img.size[0]:
            img.resize((save_side, save_side), Image.LANCZOS).save(
                os.path.join(out_dir, f"{stem}.png")
            )
        else:
            img.save(os.path.join(out_dir, f"{stem}.png"))

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def worker(rank, gpu_id, tasks, args):
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    pipe = build_pipe(args.model_id, device)

    def count_params(module):
        return sum(p.numel() for p in module.parameters())

    total_params = 0
    for name in ["unet", "vae", "text_encoder", "text_encoder_2"]:
        m = getattr(pipe, name, None)
        if m is not None:
            total_params += count_params(m)
    print("total_params:", total_params)

    os.makedirs(args.output_dir, exist_ok=True)

    batches = list(chunked(tasks, args.batch_size))
    desc = f"GPU{gpu_id} ({len(tasks)} imgs)"
    for i, batch in enumerate(tqdm(batches, total=len(batches), desc=desc, position=rank, leave=True)):
        refine_batch(
            pipe,
            batch,
            side=args.image_res,
            save_side=args.save_side,
            out_dir=args.output_dir,
            guidance=args.guidance_scale,
            steps=args.steps,
            strength=args.strength,
            seed=args.seed,
        )
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def auto_assign_devices(num_gpus: int):
    avail = torch.cuda.device_count() if torch.cuda.is_available() else 0
    num = min(num_gpus, avail) if avail else 0
    return list(range(num))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  type=str, required=True)
    parser.add_argument("--post_fix", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_res",  type=int, default=512)
    parser.add_argument(
        "--save_side", type=int, default=512,
        help="saved output resolution; retained experiments/evaluation use 512",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_gpus",   type=int, default=1)
    parser.add_argument("--num_variants", type=int, default=16)
    parser.add_argument("--model_id",   type=str, default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--steps",          type=int,   default=20)
    parser.add_argument("--strength",       type=float, default=0.5)
    parser.add_argument("--seed",           type=int,   default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("SDXL refinement requires a CUDA GPU")
    if min(args.batch_size, args.num_gpus, args.save_side, args.num_variants) < 1:
        parser.error("batch_size, num_gpus, save_side, and num_variants must be positive")

    output_path = os.path.abspath(args.output_dir)
    if os.path.isdir(output_path) and os.listdir(output_path):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    tasks = collect_tasks_nk(args.input_dir, args.post_fix, args.num_variants)
    if not tasks:
        tqdm.write("[error] No inputs to process; check file names and directories.")
        sys.exit(1)

    dev_ids = auto_assign_devices(args.num_gpus)
    if len(dev_ids) != args.num_gpus:
        raise RuntimeError(
            f"Requested {args.num_gpus} GPUs, but only {len(dev_ids)} are visible"
        )
    if len(dev_ids) <= 1:
        worker(0, dev_ids[0] if dev_ids else 0, tasks, args)
    else:
        mp.set_start_method("spawn", force=True)
        buckets = [[] for _ in dev_ids]
        for i, t in enumerate(tasks):
            buckets[i % len(dev_ids)].append(t)
        procs = []
        for rank, (gpu_id, bucket) in enumerate(zip(dev_ids, buckets)):
            p = mp.Process(target=worker, args=(rank, gpu_id, bucket, args), daemon=False)
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        failed = [p.exitcode for p in procs if p.exitcode != 0]
        if failed:
            raise RuntimeError(f"Refine worker processes failed: {failed}")

    expected_stems = {task[-1] for task in tasks}
    produced_stems = {
        stem_of(path)
        for path in glob.glob(os.path.join(args.output_dir, "*"))
        if is_img(path)
    }
    if produced_stems != expected_stems:
        raise RuntimeError(
            f"Refine output mismatch: expected {len(expected_stems)}, "
            f"produced {len(produced_stems)}"
        )

if __name__ == "__main__":
    main()
