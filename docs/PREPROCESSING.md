# Preprocessing

The preprocessing scripts build HDF5 datasets from aligned THuman 2.0 scans and then synthesize occlusion using Places backgrounds and COCO object instances. Source datasets and body-model assets are not included in this repository; obtain them from their official providers and comply with their licenses.

## 1. Required Inputs

Prepare:

- aligned THuman 2.0 subject directories containing a scan mesh and an `_smpl.json` registration;
- matching pose-render directories for every subject;
- `data/smpl_uv.obj`;
- Places images;
- COCO 2017 `train2017/` images and instance annotations.

The pose loader expects numbered JPEG frames in each subject directory.

## 2. Render RGB, UV, and Pose Data

Run from the repository root:

```bash
python 1_build_pose_dataset.py \
    --input-path /path/to/aligned_thuman \
    --pose-path /path/to/pose_renders \
    --output-path data/thuman_pose.h5 \
    --size 512 \
    --nviews 10
```

`0_build_orig_dataset.py` is the equivalent renderer without pose images and is intended for reconstruction data preparation. The diffusion occlusion builder requires the pose-enabled output from `1_build_pose_dataset.py`.

## 3. Build the Synthetic Occlusion Dataset

```bash
python 2_build_occ_dataset.py \
    --src_h5 data/thuman_pose.h5 \
    --out_dir data/occ_thuman \
    --places_dir /path/to/places \
    --coco_dir /path/to/coco2017
```

The current builder validates the release split assumptions: 526 subjects, 10 views per subject, and 512-pixel images. It writes separate training and test HDF5 files and removes its temporary combined HDF5 file after both splits are created.

## 4. Folder Input for Inference

Coarse inference can also read rendered folders instead of HDF5. Use matching basenames:

```text
DATA_ROOT/
├── occ_src_img/
├── modal_mask/
├── amodal_mask/     # bbox-based controls
└── occ_src_pose/    # pose-based controls
```

See [Inference](INFERENCE.md) for the control-specific requirements.
