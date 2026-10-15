# Inference

The released pipeline contains coarse image completion, invisible-mask prediction, and optional SDXL refinement. Commands below are run from the repository root unless noted otherwise.

## 1. Coarse Image Completion

Prepare:

- a fine-tuned checkpoint directory containing `unet/`, `controlnet/`, and `Linear.pth`;
- the Pix2Gestalt foundation checkpoint;
- either an HDF5 test file or a rendered-folder dataset.

For folder input, images with matching basenames are expected under:

```text
DATA_ROOT/
├── occ_src_img/
├── modal_mask/
├── amodal_mask/     # required for ib, eb, poib, and poeb controls
└── occ_src_pose/    # required for po, poib, and poeb controls
```

Run the wrapper:

```bash
NUM_GPUS=1 CONTROL_TYPE=po \
bash inference_folder.sh \
    /path/to/fine_tuned_checkpoint \
    /path/to/test_data \
    output/coarse_run \
    /path/to/pix2gestalt.ckpt
```

`CONTROL_TYPE` accepts `ib`, `eb`, `po`, `poib`, or `poeb`. Optional environment variables are `BATCH_SIZE` and `NUM_VALIDATION_IMAGES`; extra CLI arguments are forwarded to `inference.py`.

Outputs are written to `OUTPUT_PATH/all_images/<variant>/`. The command refuses to mix a new run with existing PNG outputs.

## 2. Invisible-Mask Prediction

The default input layout is:

```text
DATA_ROOT/
├── occ_image/<stem>.png
├── deocc_image/<stem>_000.png
└── visible_mask/<stem>.png
```

Equivalent `occ/`, `mask/`, and `deocc_<name>/` directories are supported through explicit arguments. Run:

```bash
CKPT=/path/to/invisible_mask_checkpoint.pt \
bash invis_mask/run_inference.sh \
    /path/to/data \
    output/invisible_masks
```

See [the invisible-mask guide](../invis_mask/README.md) for custom directory names and thresholding options.

## 3. SDXL Refinement

The refinement input directory must contain matching `occ/`, `mask/`, `deocc_<post_fix>/`, and `invisible_mask_<post_fix>/` directories.

```bash
python refine/refine_sdxl.py \
    --input_dir /path/to/refine_input \
    --post_fix run_name \
    --output_dir output/refined_run \
    --num_gpus 1 \
    --num_variants 16
```

The default model is `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` and is downloaded by Diffusers when it is not already cached.

## Evaluation

Evaluate stochastic predictions with:

```bash
python evaluate.py \
    --pred_image_root output/coarse_run/all_images \
    --gt_rgb_root /path/to/gt_rgb \
    --modal_mask_root /path/to/modal_masks \
    --amodal_mask_root /path/to/amodal_masks
```

Run `python evaluate.py --help` for metric and resizing options.
