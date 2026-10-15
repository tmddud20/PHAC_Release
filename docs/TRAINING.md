# Training

Complete the [installation](INSTALLATION.md) and [preprocessing](PREPROCESSING.md) steps first. Dataset files, checkpoints, and experiment logs are excluded from Git.

## Coarse Completion Model

Edit `diffusion/config.yaml` or override its values on the command line. At minimum, set:

- `global.foundation_ckpt` to the Pix2Gestalt checkpoint;
- `dataset.train_data_dir` to the generated training HDF5 file;
- `dataset.validation_data_dir` when validation is enabled;
- `dataset.control_type` to one of `ib`, `eb`, `po`, `poib`, or `poeb`.

From the `diffusion/` directory, run:

```bash
cd diffusion
torchrun --nproc_per_node=1 train.py --config config.yaml
```

For multiple GPUs, set `--nproc_per_node` to the number of visible GPUs. To override configuration values directly:

```bash
torchrun --nproc_per_node=2 train.py \
    --config config.yaml \
    --foundation_ckpt /path/to/pix2gestalt.ckpt \
    --train_data_dir /path/to/train.h5 \
    --validation_data_dir /path/to/test.h5 \
    --control_type po
```

The final inference checkpoint contains `unet/`, `controlnet/`, and `Linear.pth`.

## Invisible-Mask Predictor

Prepare the layout described in [the invisible-mask guide](../invis_mask/README.md), then run:

```bash
bash invis_mask/run_train.sh /path/to/data
```

Additional arguments are forwarded to `train_invisible_mask.py`.

## Reconstruction Model

The optional reconstruction training code and its three-stage procedure are documented in [recon/README.md](../recon/README.md). The reconstruction module contains third-party-derived code with additional license restrictions; see the license note in the root README before redistribution.

## Optional xFormers Support

Install a PyTorch/CUDA-compatible xFormers build and pass:

```bash
--enable_xformers_memory_efficient_attention
```

to the diffusion training command when memory-efficient attention is desired.
