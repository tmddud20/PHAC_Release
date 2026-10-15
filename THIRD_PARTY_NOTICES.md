# Third-Party Notices

This repository includes or adapts code from third-party projects. Copyright and license notices embedded in individual source files must be retained.

## SiTH

Parts of the dataset-rendering and reconstruction code are derived from [SiTH](https://github.com/SiTH-Diffusion/SiTH), which is distributed under the MIT License. Relevant files retain the original ETH Zurich and Hsuan-I Ho notices.

## NVIDIA Kaolin Wisp

Files under `recon/models/ops/mesh/`, together with `recon/models/positional_encoding.py`, `recon/models/networks/layers.py`, and `recon/models/networks/mlps.py`, include or adapt Kaolin Wisp code. These files are governed by the [NVIDIA Source Code License for Kaolin Wisp](LICENSES/KAOLIN_WISP_LICENSE), including its non-commercial research/evaluation limitation.

## PIFuHD and Related Network Code

`recon/models/networks/encoder.py` and `recon/models/networks/normal_predictor.py` contain their own upstream attribution and license notices. Those notices and terms apply to the corresponding code.

## Other Dependencies

The root README acknowledges other projects used by this work. Dependencies installed separately remain governed by their respective licenses. Model checkpoints and datasets are not covered by the root MIT License unless their providers explicitly state otherwise.
