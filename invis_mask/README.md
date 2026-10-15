# Invisible-Mask Prediction

The Lite U-Net receives seven channels: occluded RGB (3), de-occluded RGB (3), and a visible mask (1). It predicts one invisible-mask logit channel and is trained with binary cross-entropy plus `0.5 * Dice` loss.

## Training Data

```text
DATA_ROOT/
├── occ_image/<stem>.png
├── deocc_image/<stem>_000.png, <stem>_001.png, ...
├── visible_mask/<stem>.png
├── invisible_mask/<stem>.png   # preferred target
└── amodal_mask/<stem>.png      # fallback target
```

When `invisible_mask/` is absent, the target is computed as `amodal_mask AND NOT visible_mask`. `--deocc_variants 4` uses the first four sorted variants and selects one at random during training.

```bash
bash invis_mask/run_train.sh /absolute/path/to/data
```

Checkpoints are written to `invis_mask/checkpoints_im/` by default. Additional arguments override the wrapper defaults:

```bash
bash invis_mask/run_train.sh /absolute/path/to/data --batch_size 8 --workers 2
```

## Inference

The standard `occ_image/`, `deocc_image/`, and `visible_mask/` directory names are detected automatically:

```bash
CKPT=/path/to/best.pt \
bash invis_mask/run_inference.sh \
    /absolute/path/to/data \
    output/invisible_masks
```

`--out_folder` is the exact directory in which PNG files are written. Outputs are 0–255 probability masks by default. Add `--mask_thresh 0.5` for binary masks and `--overwrite` to replace existing outputs.

For the refinement layout (`occ/`, `mask/`, and `deocc_<name>/`), select the de-occlusion directory explicitly:

```bash
CKPT=/path/to/best.pt \
bash invis_mask/run_inference.sh \
    /path/to/refine_input \
    /path/to/refine_input/invisible_mask_run_name \
    --deocc_dir deocc_run_name
```

Use `--occ_dir`, `--deocc_dir`, and `--mask_dir` when directory names differ. Each may be absolute or relative to `--data_root`. `--post_fix NAME` is shorthand for `--deocc_dir deocc_NAME`.

The Python entry points can also be used directly:

```bash
python invis_mask/train_invisible_mask.py \
    --data_root /path/to/data \
    --epochs 40 \
    --batch_size 16 \
    --img_size 512 \
    --deocc_variants 4 \
    --base_ch 32 \
    --amp \
    --val_vis_every 10

python invis_mask/infer_all_invisible.py \
    --data_root /path/to/data \
    --ckpt /path/to/best.pt \
    --img_size 512 \
    --base_ch 32 \
    --amp \
    --out_folder output/invisible_masks
```
