# 2_build_occ_dataset.py
import os
import io
import glob
import argparse
import random
import hashlib
import numpy as np
from PIL import Image, ImageOps
import h5py
from tqdm import tqdm
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode as IM

EXPECTED_SUBJECTS = 526
EXPECTED_VIEWS = 10
IMG_SIZE = 512
TARGET_COUNTS = [105, 316, 736, 1473, 1473, 736, 316, 105, 0, 0]

def _ensure_rgba(img):
    return img.convert("RGBA") if img.mode != "RGBA" else img

def _resize_cover(img, size):
    w, h = img.size
    scale = max(size / w, size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    left, top = (new_w - size) // 2, (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))

def _pil_affine(img, angle, scale, interpolation):
    return TF.affine(img, angle=angle, translate=(0, 0), scale=scale, shear=[0.0, 0.0], interpolation=interpolation)

def _alpha_from_rgba(img):
    return img.split()[3] if img.mode == "RGBA" else Image.new("L", img.size, 255)

def _png_bytes_from_pil(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

def _h5_item_to_bytes(item):
    if isinstance(item, np.ndarray):
        return item.tobytes()
    if isinstance(item, (bytes, bytearray)):
        return bytes(item)
    return bytes(item)

def _read_h5_image(dataset, index, mode):
    raw = _h5_item_to_bytes(dataset[index])
    return Image.open(io.BytesIO(raw)).convert(mode)

def _rand(seed):
    r = random.Random()
    r.seed(seed)
    return r

class PlacesBackgrounds:
    def __init__(self, root, img_size):
        self.img_size = img_size
        if os.path.isdir(os.path.join(root, "data_256_standard")):
            root = os.path.join(root, "data_256_standard")
        pats = ["**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.bmp", "**/*.webp"]
        paths = []
        for p in pats:
            paths.extend(glob.glob(os.path.join(root, p), recursive=True))
        if not paths:
            raise FileNotFoundError(f"No background images in {root}")
        self.paths = sorted(paths)

    def pick(self, r):
        path = self.paths[r.randrange(len(self.paths))]
        img = Image.open(path).convert("RGB")
        return _resize_cover(img, self.img_size)

class CocoOccluder:
    def __init__(self, coco_root, include_person=False):
        from pycocotools.coco import COCO
        self.img_dir = os.path.join(coco_root, "train2017")
        ann_file = os.path.join(coco_root, "annotations_trainval2017", "annotations", "instances_train2017.json")
        if not os.path.isfile(ann_file):
            alt = os.path.join(coco_root, "annotations_trainval2017", "instances_train2017.json")
            ann_file = alt if os.path.isfile(alt) else ann_file
        if not os.path.isfile(ann_file):
            raise FileNotFoundError(f"COCO annotation not found: {ann_file}")
        self.coco = COCO(ann_file)
        person_id = None
        for cat in self.coco.loadCats(self.coco.getCatIds()):
            if cat.get("name") == "person":
                person_id = cat["id"]
                break
        anns = self.coco.loadAnns(self.coco.getAnnIds(iscrowd=False))
        keep = []
        for ann in anns:
            if ann.get("area", 0) < 500:
                continue
            if not include_person and person_id is not None and ann["category_id"] == person_id:
                continue
            keep.append(ann["id"])
        if not keep:
            raise RuntimeError("No COCO instances for occluders")
        self.ann_ids = sorted(keep)

    def _crop_rgba(self, ann):
        img_info = self.coco.loadImgs([ann["image_id"]])[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        if not os.path.isfile(img_path):
            return None
        img = Image.open(img_path).convert("RGB")
        mask = self.coco.annToMask(ann)
        if mask.sum() < 100:
            return None
        ys, xs = np.where(mask > 0)
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        img_c = img.crop((x0, y0, x1 + 1, y1 + 1))
        mask_c = Image.fromarray((mask[y0:y1 + 1, x0:x1 + 1] * 255).astype(np.uint8), mode="L")
        rgba = _ensure_rgba(img_c)
        rgba.putalpha(mask_c)
        return rgba

    def sample_instance_rgba(self, r):
        for _ in range(50):
            ann_id = self.ann_ids[r.randrange(len(self.ann_ids))]
            ann = self.coco.loadAnns([ann_id])[0]
            rgba = self._crop_rgba(ann)
            if rgba is not None and rgba.size[0] >= 2 and rgba.size[1] >= 2:
                return rgba
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

def _augment_person(pil_rgba, r, img_size, hflip_p=0.3, vflip_p=0.3, rotate_p=0.3, rotate_max_deg=180.0, scale_min=0.5, scale_max=1.5):
    s = r.uniform(scale_min, scale_max)
    angle = r.uniform(-rotate_max_deg, rotate_max_deg) if r.random() < rotate_p else 0.0
    do_hflip = r.random() < hflip_p
    do_vflip = r.random() < vflip_p
    rgba = _resize_cover(pil_rgba, img_size)
    if do_hflip:
        rgba = ImageOps.mirror(rgba)
    if do_vflip:
        rgba = ImageOps.flip(rgba)
    rgba = _pil_affine(_ensure_rgba(rgba), angle=angle, scale=s, interpolation=IM.BILINEAR)
    alpha = _alpha_from_rgba(rgba)
    return rgba, alpha, {"s": s, "angle": angle, "do_hflip": do_hflip, "do_vflip": do_vflip}

def _apply_same_aug(pil_rgb, aug, img_size):
    out = _resize_cover(pil_rgb, img_size)
    if aug["do_hflip"]:
        out = ImageOps.mirror(out)
    if aug["do_vflip"]:
        out = ImageOps.flip(out)
    return _pil_affine(out, angle=aug["angle"], scale=aug["s"], interpolation=IM.BILINEAR)

def _compose(bg_rgb, human_rgba):
    return Image.alpha_composite(_ensure_rgba(bg_rgb).copy(), _ensure_rgba(human_rgba))

def _ratio(human_mask_np, occ_layer):
    occ_np = (np.array(occ_layer.split()[3]) > 0).astype(np.uint8)
    inter = (occ_np & human_mask_np).sum()
    denom = max(1, human_mask_np.sum())
    return float(inter) / float(denom)

def _search_occluder_layer(r, coco, human_mask_np, img_size, lo, hi, tol, max_occluders=80, trials_per=300,
                                occ_scale_min=0.5, occ_scale_max=1.5, occ_rot_max=180.0):
    band_lo = max(0.0, lo - tol)
    band_hi = min(1.0, hi + tol)

    def in_band(x):
        return x >= lo and (x < hi if hi < 1.0 else x <= hi)

    best_gap, best_ratio, best_layer = 1e9, 0.0, None
    for _ in range(max_occluders):
        inst = coco.sample_instance_rgba(r)
        if inst.size[0] < 2 or inst.size[1] < 2:
            continue
        for _ in range(trials_per):
            occ_s = r.uniform(occ_scale_min, occ_scale_max)
            occ_angle = r.uniform(-occ_rot_max, occ_rot_max)
            inst_tf = _pil_affine(inst, angle=occ_angle, scale=occ_s, interpolation=IM.BILINEAR)
            ow, oh = inst_tf.size
            x = r.randint(-ow // 2, img_size - ow // 2)
            y = r.randint(-oh // 2, img_size - oh // 2)
            candidate = Image.new("RGBA", (img_size, img_size), (0, 0, 0, 0))
            candidate.alpha_composite(inst_tf, dest=(x, y))
            rr = _ratio(human_mask_np, candidate)
            if in_band(rr):
                return candidate, rr
            gap = band_lo - rr if rr < band_lo else (rr - band_hi if rr > band_hi else 0.0)
            if gap < best_gap:
                best_gap, best_ratio, best_layer = gap, rr, candidate

    if best_layer is None:
        best_layer = Image.new("RGBA", (img_size, img_size), (0, 0, 0, 0))
    return best_layer, best_ratio

def _create_dsets(g, num_views, vuint8):
    def mk_bytes(name):
        return g.create_dataset(name, shape=(num_views,), dtype=vuint8)

    mk_bytes("occ_src_img")
    mk_bytes("occ_src_smpl_uv")
    mk_bytes("occ_src_pose")
    g.create_dataset("occlusion_ratio", shape=(num_views,), dtype="float32")
    mk_bytes("deocc_tgt_img")

def validate_src_h5(src_h5):
    required_keys = ["cam_eva", "cam_azh", "cam_rad", "cam_pos", "rgb_img", "uv_img", "pose_img"]
    with h5py.File(src_h5, "r") as f:
        subj_names = sorted(list(f.keys()))
        if len(subj_names) != EXPECTED_SUBJECTS:
            raise RuntimeError(f"Expected {EXPECTED_SUBJECTS} subjects, but found {len(subj_names)}")

        for sname in tqdm(subj_names, desc="Validating source H5"):
            g = f[sname]
            missing = [k for k in required_keys if k not in g]
            if missing:
                raise RuntimeError(f"{sname}: missing keys {missing}")

            for k in ["cam_eva", "rgb_img", "uv_img", "pose_img"]:
                if g[k].shape[0] != EXPECTED_VIEWS:
                    raise RuntimeError(f"{sname}/{k}: found {g[k].shape[0]} views, expected {EXPECTED_VIEWS}")

    return subj_names

def copy_subset(src_path, dst_path, selected_names):
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for sname in tqdm(selected_names, desc=f"Writing {os.path.basename(dst_path)}"):
            src.copy(sname, dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_h5", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--places_dir", required=True)
    ap.add_argument("--coco_dir", required=True)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--tol", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.img_size != IMG_SIZE:
        raise ValueError("This version expects --img_size 512")

    os.makedirs(args.out_dir, exist_ok=True)
    subj_names = validate_src_h5(args.src_h5)
    num_subj, num_views = len(subj_names), EXPECTED_VIEWS

    if sum(TARGET_COUNTS) != EXPECTED_SUBJECTS * EXPECTED_VIEWS:
        raise RuntimeError("TARGET_COUNTS does not match 526 subjects x 10 views")

    stem = os.path.splitext(os.path.basename(args.src_h5))[0]
    prefix = f"occ_{stem}"
    all_path = os.path.join(args.out_dir, f"{prefix}_all.h5")
    train_path = os.path.join(args.out_dir, f"{prefix}_train.h5")
    test_path = os.path.join(args.out_dir, f"{prefix}_test.h5")

    print(f"Subjects: {num_subj}")
    print(f"Views: {num_views}")
    print(f"Image size: {args.img_size}")
    print(f"Total samples: {num_subj * num_views}")

    bg = PlacesBackgrounds(args.places_dir, args.img_size)
    coco = CocoOccluder(args.coco_dir)

    all_pairs = [(si, vi) for si in range(num_subj) for vi in range(num_views)]
    if len(all_pairs) != 5260:
        raise RuntimeError(f"Expected 5260 samples, got {len(all_pairs)}")

    bins = [(i / 10.0, (i + 1) / 10.0) for i in range(10)]
    assignment = []
    for bidx, cnt in enumerate(TARGET_COUNTS):
        assignment.extend([bidx] * cnt)

    r_global = _rand(args.seed)
    r_global.shuffle(assignment)
    vuint8 = h5py.special_dtype(vlen=np.uint8)

    with h5py.File(args.src_h5, "r") as fsrc, h5py.File(all_path, "w") as fall:
        for sname in tqdm(subj_names, desc="Creating output H5"):
            g_src = fsrc[sname]
            g_dst = fall.create_group(sname)
            for k in ["cam_eva", "cam_azh", "cam_rad", "cam_pos"]:
                g_dst.create_dataset(k, data=g_src[k][...])
            _create_dsets(g_dst, num_views, vuint8)

    with h5py.File(args.src_h5, "r") as fsrc, h5py.File(all_path, "a") as fall:
        for idx, (si, vi) in enumerate(tqdm(all_pairs, desc="Building occlusion dataset")):
            sname = subj_names[si]
            bidx = assignment[idx]
            lo, hi = bins[bidx]

            seed_i = (args.seed * 10_000_003) ^ int(hashlib.sha1(f"{sname}-{vi}".encode()).hexdigest(), 16)
            r = _rand(seed_i)
            _ = r.randint(1, num_views - 1)
            g = fsrc[sname]

            src_rgba = _read_h5_image(g["rgb_img"], vi, "RGBA")
            src_smpl_uv = _read_h5_image(g["uv_img"], vi, "RGB")
            src_pose = _read_h5_image(g["pose_img"], vi, "RGB")

            bg_img = bg.pick(r)

            src_rgba_aug, src_alpha, aug = _augment_person(src_rgba, r, args.img_size)
            src_smpl_uv_aug = _apply_same_aug(src_smpl_uv, aug, args.img_size)
            src_pose_aug = _apply_same_aug(src_pose, aug, args.img_size)
            _ = r.uniform(0.5, 1.5)
            if r.random() < 0.3:
                _ = r.uniform(-180.0, 180.0)
            _ = r.random()
            _ = r.random()
            base_src = _compose(bg_img, src_rgba_aug)

            white_bg = Image.new("RGB", (args.img_size, args.img_size), (255, 255, 255))
            deocc_rgb = Image.alpha_composite(_ensure_rgba(white_bg), src_rgba_aug).convert("RGB")

            human_mask_np = (np.array(src_alpha) > 0).astype(np.uint8)
            occ_layer, rr = _search_occluder_layer(r, coco, human_mask_np, args.img_size, lo, hi, args.tol)

            final_src_rgba = Image.alpha_composite(_ensure_rgba(base_src), occ_layer)

            occ_alpha_np = (np.array(occ_layer.split()[3]) > 0).astype(np.uint8)
            vis_mask_np = (human_mask_np * (1 - occ_alpha_np)).astype(np.uint8)
            src_vis_mask = Image.fromarray((vis_mask_np * 255).astype(np.uint8), mode="L")

            occ_src_rgba = final_src_rgba.convert("RGB").convert("RGBA")
            occ_src_rgba.putalpha(src_vis_mask)
            occ_src_png = _png_bytes_from_pil(occ_src_rgba)

            deocc_tgt_rgba = deocc_rgb.convert("RGBA")
            deocc_tgt_rgba.putalpha(src_alpha)
            deocc_tgt_png = _png_bytes_from_pil(deocc_tgt_rgba)

            occ_src_smpl_uv_png = _png_bytes_from_pil(src_smpl_uv_aug.convert("RGB"))
            occ_src_pose_png = _png_bytes_from_pil(src_pose_aug.convert("RGB"))

            gdst = fall[sname]
            gdst["occ_src_img"][vi] = np.frombuffer(occ_src_png, dtype=np.uint8)
            gdst["occ_src_smpl_uv"][vi] = np.frombuffer(occ_src_smpl_uv_png, dtype=np.uint8)
            gdst["occ_src_pose"][vi] = np.frombuffer(occ_src_pose_png, dtype=np.uint8)
            gdst["occlusion_ratio"][vi] = np.float32(rr)
            gdst["deocc_tgt_img"][vi] = np.frombuffer(deocc_tgt_png, dtype=np.uint8)

    train_names = subj_names[:522]
    test_names = subj_names[522:]

    copy_subset(all_path, train_path, train_names)
    copy_subset(all_path, test_path, test_names)

    with h5py.File(all_path, "r") as fall:
        ratios = []
        for sname in fall:
            ratios.extend(fall[sname]["occlusion_ratio"][...].astype(float).tolist())

    counts = []
    for lo, hi in bins:
        if hi < 1.0:
            counts.append(sum(lo <= r < hi for r in ratios))
        else:
            counts.append(sum(lo <= r <= hi for r in ratios))

    print("Target bin counts:", TARGET_COUNTS)
    print("Actual bin counts:", counts)
    os.remove(all_path)
    print("TRAIN:", train_path)
    print("TEST:", test_path)

if __name__ == "__main__":
    main()
