# Assisted Decoding in a JPEG-like Image Compression System

A data science and image compression project implementing a JPEG-like image coding system with CNN-assisted reconstruction.

GitHub repository: [Data-science-and-compression-project](https://github.com/MattiasHuen/Data-science-and-compression-project)

## Overview

This project implements and evaluates a simplified JPEG-like image compression pipeline. The codec is based on the main stages used in transform-based image coding:

1. Color conversion to YCbCr
2. 8x8 block Discrete Cosine Transform (DCT)
3. Uniform scalar quantization
4. Zigzag scanning
5. DPCM coding of DC coefficients
6. Run-length coding of AC coefficients
7. Huffman entropy coding
8. Decoding and image reconstruction

The project also extends the standard decoder with a Convolutional Neural Network (CNN), which is applied after normal decoding. The goal of the CNN is to reduce compression artifacts such as blocking and ringing and improve reconstruction quality.

## Project goal

The main goal is to evaluate the trade-off between compression performance and reconstruction quality.

Compression performance is evaluated using:

- Code length
- Bits per pixel
- Compression ratio
- Entropy estimates

Reconstruction quality is evaluated using:

- MSE
- PSNR
- SSIM
- Visual comparison

The CNN-enhanced reconstruction is compared against the normal decoded output to investigate whether learned post-processing can improve image quality after lossy compression.

## Repository structure

```txt
├── configs/                  # Configuration files
├── data/                     # Raw and processed data
│   ├── raw/
│   └── processed/
├── docs/                     # Documentation
├── models/                   # Trained model weights
├── notebooks/                # Experiments and exploratory notebooks
├── reports/                  # Report files and figures
│   └── figures/
├── src/                      # Source code
├── tests/                    # Unit tests
├── .github/                  # GitHub workflows and dependabot
├── .gitignore
├── .pre-commit-config.yaml
├── AGENTS.md                 # Development instructions
├── LICENSE
├── pyproject.toml
├── README.md
├── tasks.py
└── uv.lock
