

# NeuroJackSplat

**NeuroJackSplat** is an end-to-end engine for 3D reconstruction from single images, combining the power of visual transformers (**NeuroForge/DINOv2**) with the efficiency of **3D Gaussian Splatting (3DGS)**.

## Architecture
The project is built on three engineering pillars:
* **Feature Extraction:** Uses the `NeuroForge` architecture (DINOv2 backbone) to extract deep geometric representations.
* **Gaussian Regressor Head:** An "Anchor-based" regressor that predicts 3D parameters (means, scales, rotations, opacities) for millions of Gaussians.
* **CUDA Rasterization Engine:** Engine optimized for real-time visualization of Gaussian point clouds.

## Features
- [x] **Zero-Shot/Single-Image Reconstruction**: Convert a 2D image into a 3D "Splat".
- [x] **Differentiable Rendering**: Direct training using `L1 Loss` and `Backpropagation` on the CUDA engine.
- [x] **Customized Memory Management**: Optimized for Kaggle environments with `bitsandbytes`.
- [x] **Cyber-Ready Security**: Designed with `CyberJack` architecture standards.

## Installation
To compile the rendering engine (C++/CUDA), run:
```bash
git clone --recursive [https://github.com/graphdeco-inria/diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
pip install ./diff-gaussian-rasterization
