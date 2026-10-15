# Installation

## 1. Create and Activate the Conda Environment

```bash
conda create -n PHAC python=3.9.19 pip=24.2 setuptools=72.1.0 wheel=0.43.0
conda activate PHAC
```

## 2. Install PyTorch and torchvision

```bash
python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

## 3. Install the Python Dependencies

```bash
python -m pip install -r requirements.txt
```

## 4. Install Additional Dependencies

```bash
# Install Kaolin
python -m pip install --no-deps kaolin==0.13.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.1_cu113.html

# Install nvdiffrast
python -m pip install setuptools wheel ninja
python -m pip install --no-deps --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3"

# Install OpenAI CLIP
git clone https://github.com/openai/CLIP.git
python -m pip install -e CLIP/
```

## 5. Optional xFormers Support

Install xFormers only when you plan to pass `--enable_xformers_memory_efficient_attention`, and select a build compatible with the installed PyTorch and CUDA versions. For the tested PyTorch 1.12/CUDA 11.3 environment:

```bash
python -m pip install --no-deps --no-build-isolation "git+https://github.com/facebookresearch/xformers.git@v0.0.17"
```

## Tested Environments

The installation has been tested on the following configurations:

| OS | GPU | NVIDIA Driver | CUDA Version |
| --- | --- | --- | --- |
| Ubuntu 18.04 | NVIDIA GeForce RTX 3090 | 465.19.01 | 11.3 |
| Ubuntu 20.04 | NVIDIA GeForce RTX 3090 | 535.230.02 | 12.2 |
| Ubuntu 20.04 | NVIDIA GeForce RTX 4090 | 525.78.01 | 12.0 |
| Ubuntu 20.04 | NVIDIA RTX A6000 | 570.172.08 | 12.8 |
