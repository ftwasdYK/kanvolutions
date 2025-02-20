# Kanvolution Layer 🚀

This repository contains an implementation of the **Kanvolution Layer**, a novel convolutional layer inspired by [**Kolmogorov-Arnold Networks (KANs)**](https://arxiv.org/abs/2404.19756). Unlike standard convolutional layers that use scalar parameters, the **Kanvolution Layer** utilizes **non-linear learnable functions** to enhance expressive power.

## 🔬 Key Features

- **Function-Based Parameters**: Instead of using traditional scalar parameters, the Kanvolution Layer employs **learnable rational functions** of the form $\frac{P(x)}{Q(x)}$, as inspired by [Kolmogorov–Arnold Transformer](https://arxiv.org/abs/2409.10594).
- **Improved Parallelism**: Unlike **B-splines**, which have interdependencies, **rational functions** allow for better parallelization.
- **PyTorch Compatible**: The layer is implemented in PyTorch and can be used as a replacement for `Conv2D`.
- **U-Net with Kanvolution**: A U-Net architecture using Kanvolution filters is available in `archs.py`.

## 📂 Repository Structure

- `custom_layers.py` - Contains the implementation of the **Kanvolution Layer**.
- `archs.py` - Implements a **U-Net** using Kanvolution filters.

## 🚧 Limitations

- **No CUDA Optimization (Yet)**: A CUDA-optimized version is not currently implemented.
- **No Groups Support**: Unlike PyTorch's `Conv2D`, the Kanvolution Layer does not yet support the `groups` argument.

## 🛠️ Usage

You can use the Kanvolution Layer as a drop-in replacement for `Conv2D` in PyTorch:

```python
from custom_layers import Kanvolution2d

layer = Kanvolution2d(in_channels=3, out_channels=64, kernel_size=(3,3))
out = layer(input_tensor)
```

## 📌 Future Work

- Implement CUDA optimization for better performance.
- Add support for the `groups` parameter.
