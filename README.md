# RobustHAC: Robust Human Amodal Completion under Imperfect Modal Masks

<p align="center">
  <span style="display:inline-block; vertical-align:middle; margin-right:10px;">
    <img src="assets/CVLAB_logo_KWU.png" alt="Lab Logo" height="24">
  </span>
  <span style="display:inline-block; vertical-align:middle;">
    <b>Seung Young Noh</b> &nbsp;&nbsp;
    <b>Ju Yong Chang</b>
  </span>
</p>

<p align="center">
  {kelvinnoh, jychang}@kw.ac.kr
  <br>
  Dept. of ECE, Kwangwoon University, Seoul, Korea
</p>

RobustHAC is a framework for human amodal completion that is robust to imperfect input modal masks. This repository contains the training, inference, evaluation, and refinement code. Datasets, body models, generated outputs, and pretrained checkpoints are distributed separately and are intentionally excluded from Git. This work extends our ACM Multimedia 2025 paper, [Stable Diffusion-Based Approach for Human De-Occlusion](https://arxiv.org/abs/2508.12663).

## Documentation

Please refer to the following documents for setup and usage instructions:

- [Installation](docs/INSTALLATION.md)
- [Preprocessing](docs/PREPROCESSING.md)
- [Training](docs/TRAINING.md)
- [Inference](docs/INFERENCE.md)
- [First GitHub upload](docs/GITHUB_UPLOAD.md)

## HITL-AHP Dataset

HITL-AHP provides 53,244 human-verified amodal masks obtained through SAM-assisted human-in-the-loop re-annotation and invalid-sample filtering of the [AHP training set](https://arxiv.org/abs/2103.11597). The annotations are available on Hugging Face.

Download [`train/amodal_mask.tar`](https://huggingface.co/datasets/tmddud20/HITL-AHP/blob/main/train/amodal_mask.tar) from the [HITL-AHP Hugging Face repository](https://huggingface.co/datasets/tmddud20/HITL-AHP/tree/main) and extract it before use.

The HITL-AHP annotations and corresponding source images remain subject to their original dataset licenses and terms of use. The repository's source-code license does not grant rights to those assets.

## Pretrained Checkpoints

The pretrained checkpoints are available in the [RobustHAC Hugging Face repository](https://huggingface.co/tmddud20/RobustHAC).

- **Diffusion model for coarse image completion:** [`26tmm_rgb_final`](https://huggingface.co/tmddud20/RobustHAC/tree/main/26tmm_rgb_final)

Checkpoints are intentionally not committed to Git. Pass local checkpoint paths through the documented command-line arguments.

## Acknowledgements

The coarse completion model builds on [pix2gestalt](https://github.com/cvlab-columbia/pix2gestalt) and [OpenAI CLIP](https://github.com/openai/CLIP). Dataset rendering and reconstruction code is derived in part from [SiTH](https://github.com/SiTH-Diffusion/SiTH), [Kaolin Wisp](https://github.com/NVIDIAGameWorks/kaolin-wisp), and [PIFuHD](https://github.com/facebookresearch/pifuhd). We thank the authors for making their work available.

## License

Original project code is released under the [MIT License](LICENSE). Files derived from third-party projects retain their own copyright and license notices; those terms take precedence for those files. In particular, Kaolin Wisp-derived files under `recon/` are restricted to non-commercial research or evaluation under the NVIDIA Source Code License. See [Third-Party Notices](THIRD_PARTY_NOTICES.md) and the bundled license texts for details.

## Citation

If you find this work useful for your research, please consider citing the following papers:

```bibtex
@inproceedings{noh2025stable,
  author={Noh, Seung Young and Chang, Ju Yong},
  title={Stable Diffusion-Based Approach for Human De-Occlusion},
  booktitle={Proceedings of the ACM International Conference on Multimedia},
  pages={10044--10053},
  year={2025},
  publisher={ACM},
  doi={10.1145/3746027.3755346}
}
```
