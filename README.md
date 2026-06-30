# VQ-DAM

Official implementation for **Learning Vector-Quantised Degradation Representations with Multi-Scale Consistency for Blind Image Super-Resolution**.

VQ-DAM is a blind image super-resolution method that learns vector-quantised degradation representations and injects them into a degradation-aware restorer. The model is designed to reduce interference from image content and context variations, so that the learned latent codes better reflect the underlying degradation condition.

> Note: some training and evaluation scripts in this repository still use the internal name `VQDAE`. In this README, `VQ-DAM` refers to the paper method and `VQDAE` refers to the corresponding implementation classes/scripts.

## News

- Code release for VQ-DAM.
- Pretrained models and checkpoints will be released later.

## Method

VQ-DAM contains two main modules:

- **Degradation Representation Extractor (DRE)**: extracts degradation-aware latent features from the LR input.
- **Multi-scale vector quantisation**: constrains degradation embeddings across scales with patch scales `[1, 2, 4, 8, 16]`, a 4096-entry codebook, and 32-channel latent embeddings.
- **Degradation-Aware Restorer (DAR)**: injects the quantised degradation representation into the SR backbone through vector-quantised degradation-aware blocks.

The degradation process follows the standard blind SR setting:

```text
I_LR = downsample(I_HR * k) + n
```

where `k` is a blur kernel, `downsample` is bicubic downsampling, and `n` is optional Gaussian noise.

## Requirements

The project was developed with the following environment:

```bash
conda create -n vqdam python=3.9 -y
conda activate vqdam

pip install torch==1.12.1 torchvision==0.13.1
pip install opencv-python einops scikit-image numpy
```

`scikit-image` provides the Python package imported as `skimage`.

Recommended hardware:

- NVIDIA GPU with CUDA support.
- At least 12 GB GPU memory for training with the default batch size.
- 24 GB GPU memory is recommended for larger-scale experiments.

If your CUDA version differs, install the matching PyTorch 1.12.1 wheel from the official PyTorch website.

## Dataset Preparation

This repository uses its own lightweight dataset loader in `data/srdataset.py`. The code expects a dataset root specified by `--data_dir` in `options/*.py`. The default is:

```text
C:/Project/DCEDSR/dataset
```

### Training Data

Training only requires high-resolution images. Low-resolution inputs are generated online by the degradation modules in `utils/degradation.py`.

For the default `DF2K` training setting, place all HR images under:

```text
dataset/
  DF2K/
    HR/
      000001.png
      000002.png
      ...
```

The dataset name is controlled by `data_train` in the option file:

```text
data_train: DF2K
```

During the first run, the loader automatically creates a binary cache:

```text
dataset/
  DF2K/
    bin/
      HR/
        000001.pt
        000002.pt
        ...
```

The cache is generated from the original HR images and is reused in later runs.

### Synthetic Benchmark Testing

For synthetic benchmark evaluation, the test scripts can read HR images and generate LR images on the fly with fixed degradations. In this case, each test dataset only needs an `HR/` folder:

```text
dataset/
  Set5/
    HR/
  Set14/
    HR/
  B100/
    HR/
  Urban100/
    HR/
  DIV2KRK/
    HR/
```

Scripts such as `test/mytest_iso.py`, `test/mytest_iso_batch.py`, `test/mytest_aniso.py`, and `test/mytest_aniso_batch.py` use this mode.

## Degradation Settings

The main degradation settings are configured in `options/*.py`.

### Isotropic Gaussian degradation

Used by `options/option_vqdae_iso_x2.py`, `options/option_vqdae_iso_x3.py`, and `options/option_vqdae_iso_x4.py`.

```text
kernel_size: 21
blur_type: iso_gaussian
sigma range:
  x2: [0.2, 2.0]
  x3: [0.2, 3.0]
  x4: [0.2, 4.0]
downsampling: bicubic
```

### Anisotropic Gaussian degradation with noise

Used by `options/option_vqdae_aniso_x4.py`.

```text
kernel_size: 21
blur_type: aniso_gaussian
sigma range: [0.2, 4.0]
rotation: random in [0, pi]
noise range: [0, 10]
downsampling: bicubic
```

Evaluation scripts include fixed settings such as isotropic kernel widths `[0, 1.2, 2.4, 3.6]` and anisotropic kernels with noise levels `[0, 10]`.

## Training

Before training, edit the corresponding option file and set:

- `data_dir`: path to your dataset root.
- `project_path`: path to this repository.
- `resume`: set to `False` for training from scratch.
- `resume_path`: set only when resuming from a checkpoint.

The current option files use `argparse(type=bool)` for Boolean values, so editing the option file is safer than passing `--resume False` from the command line.

Train VQ-DAM for x4 isotropic Gaussian degradation:

```bash
python trains/train_vqdae.py
```

Train x2 and x3 models:

```bash
python trains/train_vqdae_x2.py
python trains/train_vqdae.py   # switch to option_vqdae_iso_x3.py if needed
```

Train x4 anisotropic Gaussian degradation with noise:

```bash
python trains/train_vqdae_aniso.py
```

Checkpoints and logs are saved under:

```text
experiment/<model_name>/
  model/
  logs/
```

## Testing

Pretrained checkpoints are not provided yet. To test your own checkpoint, either:

1. Put it at the path hard-coded in the selected test script, or
2. Edit `model_paths` in the test script.

Common evaluation scripts:

```bash
# Isotropic Gaussian evaluation
python test/mytest_iso.py

# Batch isotropic evaluation and optional visual result saving
python test/mytest_iso_batch.py

# Anisotropic Gaussian + noise evaluation
python test/mytest_aniso.py
```

For example, `test/mytest_iso.py` currently expects:

```text
experiment/VQDAE_iso_x4/model/model_0320.pth.tar
```

and `test/mytest_aniso.py` expects:

```text
experiment/VQDAE_aniso10_srmd/model/model_0360.pth.tar
```

Please update these paths before running the scripts.

## Official Results

The following results are reported in the paper.

### Bicubic and Gaussian8 degradation

PSNR/SSIM of VQ-DAM on classic benchmarks:

| Scale | Degradation | Set5 | Set14 | BSD100 | Urban100 |
| --- | --- | --- | --- | --- | --- |
| x2 | Bicubic | 38.36 / 0.9635 | 34.23 / 0.9234 | 32.47 / 0.9033 | 33.48 / 0.9400 |
| x2 | Gaussian8 | 37.73 / 0.9563 | 33.78 / 0.9106 | 32.16 / 0.8914 | 32.24 / 0.9255 |
| x3 | Bicubic | 34.90 / 0.9317 | 30.86 / 0.8524 | 29.44 / 0.8138 | 29.67 / 0.8805 |
| x3 | Gaussian8 | 34.38 / 0.9236 | 30.52 / 0.8359 | 29.23 / 0.7981 | 28.80 / 0.8592 |
| x4 | Bicubic | 32.86 / 0.9042 | 29.12 / 0.7941 | 27.92 / 0.7484 | 27.49 / 0.8253 |
| x4 | Gaussian8 | 32.43 / 0.8945 | 28.77 / 0.7788 | 27.75 / 0.7344 | 26.85 / 0.8020 |

### Isotropic Gaussian kernels on Urban100, x4

| Method | sigma=1.2 | sigma=2.4 | sigma=3.6 |
| --- | --- | --- | --- |
| DASR | 25.69 | 25.44 | 24.66 |
| DCLS | 26.50 | 26.24 | 25.34 |
| DAN-v2 | 26.34 | 26.04 | 25.07 |
| CDFormer | 27.06 | 26.66 | 25.72 |
| CdCL | 26.37 | 26.07 | 25.21 |
| DRAT | 26.79 | 26.48 | 25.47 |
| LightBSR | 26.50 | 26.20 | 25.31 |
| VQ-DAM | 27.36 | 26.96 | 25.95 |

## License

This repository is intended for academic research. For public release, the common practice is to include an open-source license such as MIT License together with the code.
